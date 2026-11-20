// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include <bits/stdc++.h>
#include <boost/program_options.hpp>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdfloat>
#include "utils/debug_utils.hpp"

#include "typedef.hpp"
#include "npu_utils/npu_utils.hpp"
#include "utils/utils.hpp"
#include "utils/vm_args.hpp"
#include "lm_config.hpp"
#include <math.h>
#include "metrices.hpp"
#include "seq_gen.hpp"
#include "cpu_ref_attention.hpp"

namespace po = boost::program_options;

// Qwen3.8-27B attention geometry
static constexpr int HEAD_DIM      = 256;
static constexpr int NUM_ATTN_HEADS = 24;
static constexpr int NUM_KV_HEADS   = 4;
static constexpr float ATTN_SCALE   = 0.0625f; // 1/sqrt(256)

int main(int argc, char **argv) {
    srand(0);
    po::options_description desc("Allowed options");
    po::variables_map vm;
    arg_utils::add_default_options(desc);

    desc.add_options()("Seed,s", po::value<unsigned int>()->default_value(32), "Random seed");
    desc.add_options()("Verify,v", po::value<bool>()->default_value(true), "Verify the result");
    desc.add_options()("L_wanted,l", po::value<int>()->default_value(1024), "L_wanted");
    desc.add_options()("L_begin,b", po::value<int>()->default_value(0), "L_begin");
    desc.add_options()("Iteration,i", po::value<int>()->default_value(1), "Iteration");
    desc.add_options()("Bfp16,p", po::value<bool>()->default_value(false),
                       "Pre-round q/k onto the bfp16 shared-exponent grid (diagnostic: "
                       "isolates mmul truncation from real errors)");
    arg_utils::parse_options(argc, argv, desc, vm);

    const int NT = vm["Iteration"].as<int>();
    const bool VERIFY = vm["Verify"].as<bool>();
    const unsigned int SEED = vm["Seed"].as<unsigned int>();

    const int d = HEAD_DIM;
    const int DQ = d * NUM_ATTN_HEADS;   // 6144
    const int DK = d * NUM_KV_HEADS;     // 1024
    const int DV = DK;

    const int L_wanted = vm["L_wanted"].as<int>();
    const int L_begin = vm["L_begin"].as<int>();

    const int L_wanted_chunked = ((L_wanted + 127) / 128) * 128;
    const int L_end = L_begin + L_wanted_chunked;
    const int L_padded = ((L_end + 127) / 128) * 128;

    std::cout << "L_begin: " << L_begin << " L_wanted: " << L_wanted
              << " L_end: " << L_end << " L_wanted_chunked: " << L_wanted_chunked
              << " L_padded: " << L_padded << std::endl;

    const int Q_VOLUME = L_padded * DQ;
    const int K_VOLUME = L_padded * DK;
    const int V_VOLUME = L_padded * DV;
    const int O_VOLUME = L_padded * DQ;

    bool pass = true;
    xrt::device npu_device = xrt::device(0);

    std::string xclbin_name = "build/xclbins/attn.xclbin";
    npu_xclbin_manager xclbin_mgr(npu_device::device_npu2, &npu_device, false);
    npu_app_manager* attn_app_mgr = xclbin_mgr.register_xclbin(xclbin_name);

    header_print("info", "Attention " << L_padded << " tokens, "
                 << NUM_ATTN_HEADS << " q heads / " << NUM_KV_HEADS
                 << " kv heads, head_dim " << d);

    npu_app app_0 = attn_app_mgr->create_app();

    buffer<bf16> o_0  = app_0.create_bo_buffer<bf16>(O_VOLUME);
    buffer<bf16> q_0  = app_0.create_bo_buffer<bf16>(Q_VOLUME);
    buffer<bf16> kv_0 = app_0.create_bo_buffer<bf16>(K_VOLUME + V_VOLUME);

    // -------------------------------------------------------------------------
    // Synthesize q / k / v.
    //
    // Independent N(0,1) q and k would put essentially all the softmax mass on
    // a single key (dot products of 256-dim unit-ish vectors spread far too
    // wide once multiplied by 1/sqrt(d)), which makes the test blind to almost
    // every arithmetic bug.  Instead each token gets a low-rank "topic" code
    // shared between q and k, so keys within a topic genuinely compete, and the
    // logits are scaled down to keep the distribution broad.
    // -------------------------------------------------------------------------
    const int RANK = 8;
    std::mt19937 rng(SEED);
    std::normal_distribution<float> gauss(0.0f, 1.0f);

    // per-token topic code, and per-head projections of it
    std::vector<float> topic((size_t)L_padded * RANK);
    for (auto &x : topic) x = gauss(rng);

    std::vector<float> q_head_basis((size_t)NUM_ATTN_HEADS * RANK * d);
    std::vector<float> k_head_basis((size_t)NUM_KV_HEADS * RANK * d);
    for (auto &x : q_head_basis) x = gauss(rng) * 0.25f;
    for (auto &x : k_head_basis) x = gauss(rng) * 0.25f;

    std::vector<bf16> q_host((size_t)L_padded * DQ);
    std::vector<bf16> k_host((size_t)L_padded * DK);
    std::vector<bf16> v_host((size_t)L_padded * DV);

    const float noise_amp = 0.15f;
    const float sig_amp = 0.06f; // keeps logits ~O(1) after the 1/sqrt(d) scale

    for (int t = 0; t < L_padded; t++) {
        const float *tc = topic.data() + (size_t)t * RANK;
        for (int h = 0; h < NUM_ATTN_HEADS; h++) {
            bf16 *dst = q_host.data() + (size_t)t * DQ + (size_t)h * d;
            const float *basis = q_head_basis.data() + (size_t)h * RANK * d;
            for (int i = 0; i < d; i++) {
                float acc = 0.0f;
                for (int r = 0; r < RANK; r++) acc += tc[r] * basis[(size_t)r * d + i];
                dst[i] = (bf16)(sig_amp * acc + noise_amp * gauss(rng));
            }
        }
        for (int h = 0; h < NUM_KV_HEADS; h++) {
            bf16 *kdst = k_host.data() + (size_t)t * DK + (size_t)h * d;
            bf16 *vdst = v_host.data() + (size_t)t * DV + (size_t)h * d;
            const float *basis = k_head_basis.data() + (size_t)h * RANK * d;
            for (int i = 0; i < d; i++) {
                float acc = 0.0f;
                for (int r = 0; r < RANK; r++) acc += tc[r] * basis[(size_t)r * d + i];
                kdst[i] = (bf16)(sig_amp * acc + noise_amp * gauss(rng));
                vdst[i] = (bf16)(0.5f * gauss(rng));
            }
        }
    }
    // Diagnostic: aie::mmul<8,8,8,bf16> on npu2 floors both operands onto a
    // shared-exponent-per-8 grid.  Pre-rounding q/k the same way makes that
    // truncation a no-op, which separates mmul quantisation from real bugs.
    if (vm["Bfp16"].as<bool>()) {
        auto to_bfp16 = [](bf16 *p, size_t n_groups) {
            for (size_t g = 0; g < n_groups; g++) {
                bf16 *b = p + g * 8;
                int max_exp = -1000;
                for (int i = 0; i < 8; i++) {
                    const float f = (float)b[i];
                    if (f != 0.0f) { int e; std::frexp(f, &e); max_exp = std::max(max_exp, e); }
                }
                if (max_exp == -1000) continue;
                const float q = std::ldexp(1.0f, max_exp - 8);
                for (int i = 0; i < 8; i++) b[i] = (bf16)(std::floor((float)b[i] / q) * q);
            }
        };
        to_bfp16(q_host.data(), q_host.size() / 8);
        to_bfp16(k_host.data(), k_host.size() / 8);
        header_print("info", "Pre-rounded q/k onto the bfp16 grid");
    }

    header_print("info", "Finished generating inputs");

    q_0.copy_from(q_host.data() + (size_t)L_begin * DQ, (size_t)L_padded * DQ - (size_t)L_begin * DQ);
    q_0.sync_to_device();

    // KV cache: two contiguous planes, [L_padded][DK] then [L_padded][DV],
    // token-major with the kv heads side by side inside a row -- so
    // k_at(t) = t*DK and v_at(t) = L_padded*DK + t*DV.  Same layout as
    // ../MTP_decoding, ../gated_attention_verify and the DLL's
    // q38npu::attn_layout; seq_gen.hpp derives every kv descriptor from it.
    const size_t k_at = 0;
    const size_t v_at = (size_t)L_padded * DK;
    memcpy(kv_0.data() + k_at, k_host.data(), (size_t)L_padded * DK * sizeof(bf16));
    memcpy(kv_0.data() + v_at, v_host.data(), (size_t)L_padded * DV * sizeof(bf16));
    kv_0.sync_to_device();

    time_utils::time_with_unit npu_time = {0.0, "us"};
    header_print("info", "Running kernel with bare call (direct based).");
    gen_seq_1_4(app_0.seq(), L_begin, L_wanted_chunked, L_padded, 512, 16,
                d, NUM_ATTN_HEADS, NUM_KV_HEADS, false);

    for (int i = 0; i < NT; i++) {
        time_utils::time_point start = time_utils::now();
        app_0(o_0, q_0, kv_0);
        time_utils::time_point stop = time_utils::now();
        npu_time.first += time_utils::duration_us(start, stop).first;
    }
    npu_time.first /= NT;
    MSG_BONDLINE(60);
    MSG_BOX_LINE(60, "NPU time with bare call: " << npu_time.first << " us");
    MSG_BONDLINE(60);

    o_0.sync_from_device();

    if (VERIFY) {
        header_print("info", "Computing CPU reference ...");
        cpu_ref::attn_geometry geo{d, NUM_ATTN_HEADS, NUM_KV_HEADS, ATTN_SCALE};

        std::vector<float> ref_f((size_t)L_wanted * DQ);
        double eff_keys = 0.0;
        time_utils::time_point cs = time_utils::now();
        cpu_ref::gqa_attention(q_host.data() + (size_t)L_begin * DQ,
                               k_host.data(), v_host.data(), ref_f.data(),
                               geo, L_begin, L_wanted, 0, &eff_keys);
        time_utils::time_point ce = time_utils::now();
        header_print("info", "CPU reference took "
                     << time_utils::duration_us(cs, ce).first << " us");
        header_print("info", "Mean effective keys per softmax: " << eff_keys
                     << "  (1.0 == degenerate, test would be meaningless)");
        if (eff_keys < 4.0) {
            header_print("warn", "Softmax is nearly collapsed -- widen the input "
                                 "distribution before trusting this comparison.");
        }

        buffer<bf16> o_npu((size_t)L_wanted * DQ);
        o_npu.copy_from(o_0.data(), (size_t)L_wanted * DQ);
        buffer<bf16> o_ref((size_t)L_wanted * DQ);
        for (size_t i = 0; i < ref_f.size(); i++) o_ref[i] = ref_f[i];

        header_print("info", "CPU reference");
        utils::print_matrix(o_ref, DQ);
        header_print("info", "NPU output");
        utils::print_matrix(o_npu, DQ);

        error_metrics metrics = get_error_metrics(o_npu, o_ref);
        print_error_metrics(metrics);
        // The flash-attention online softmax accumulates in bf16, so accuracy
        // degrades smoothly with the number of kv blocks -- a deep run is
        // legitimately worse than a shallow one.  Hence a loose aggregate
        // bound here; correctness is really established by the two structural
        // checks below (per-head uniformity and the cross-head control).
        pass = metrics.CosineSimilarity > 0.95f;

        // Per-token-segment cosine.  Output magnitude falls as the causal
        // horizon grows, so a whole-sequence cosine is dominated by the easy
        // early tokens; this shows the real accuracy at depth.
        {
            const int seg = 128;
            std::cout << "[info]  per-segment cosine (by absolute token):";
            for (int s0 = 0; s0 < L_wanted; s0 += seg) {
                const int s1 = std::min(s0 + seg, L_wanted);
                double dot = 0.0, na = 0.0, nb = 0.0;
                for (int t = s0; t < s1; t++)
                    for (int i = 0; i < DQ; i++) {
                        const double a = (float)o_npu[(size_t)t * DQ + i];
                        const double b = (float)o_ref[(size_t)t * DQ + i];
                        dot += a * b; na += a * a; nb += b * b;
                    }
                std::cout << "\n          [" << (L_begin + s0) << "," << (L_begin + s1)
                          << ") cos=" << std::fixed << std::setprecision(5)
                          << dot / (std::sqrt(na) * std::sqrt(nb) + 1e-30)
                          << "  |ref|rms=" << std::sqrt(nb / ((double)(s1 - s0) * DQ));
            }
            std::cout << std::defaultfloat << std::endl;
        }

        // Per-head cosine.  A single bad head means the head <-> CU mapping is
        // wrong; uniformly mediocre heads mean it is just accumulation error.
        {
            double worst = 1.0; int worst_h = -1;
            std::cout << "[info]  per-head cosine:";
            for (int h = 0; h < NUM_ATTN_HEADS; h++) {
                double dot = 0.0, na = 0.0, nb = 0.0;
                for (int t = 0; t < L_wanted; t++) {
                    const size_t base = (size_t)t * DQ + (size_t)h * d;
                    for (int i = 0; i < d; i++) {
                        const double a = (float)o_npu[base + i];
                        const double b = (float)o_ref[base + i];
                        dot += a * b; na += a * a; nb += b * b;
                    }
                }
                const double c = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-30);
                if (h % 8 == 0) std::cout << "\n        ";
                std::cout << " h" << h << "=" << std::fixed << std::setprecision(4) << c;
                if (c < worst) { worst = c; worst_h = h; }
            }
            std::cout << std::defaultfloat << std::endl;
            header_print("info", "worst head: h" << worst_h << " cosine " << worst);
            // A wrong head <-> CU <-> kv-stream mapping shows up as one or a few
            // heads collapsing while the rest stay good, not as a uniform sag.
            if (worst < metrics.CosineSimilarity - 0.05) {
                header_print("warn", "head h" << worst_h << " is an outlier -- "
                             "suspect the head/kv-stream mapping in seq_gen.hpp");
                pass = false;
            }
        }

        // ---------------------------------------------------------------------
        // Cross-head control.  Matching our own reference does not prove the
        // head <-> CU <-> kv-stream mapping is right, because a consistent
        // permutation would agree with itself.  Recompute the reference with a
        // deliberately rotated kv-head assignment; the NPU must agree *much*
        // less well with it than with the correct one.
        // ---------------------------------------------------------------------
        std::vector<bf16> k_rot((size_t)L_padded * DK);
        std::vector<bf16> v_rot((size_t)L_padded * DV);
        for (int t = 0; t < L_padded; t++) {
            for (int h = 0; h < NUM_KV_HEADS; h++) {
                const int src = (h + 1) % NUM_KV_HEADS;
                memcpy(k_rot.data() + (size_t)t * DK + (size_t)h * d,
                       k_host.data() + (size_t)t * DK + (size_t)src * d,
                       d * sizeof(bf16));
                memcpy(v_rot.data() + (size_t)t * DV + (size_t)h * d,
                       v_host.data() + (size_t)t * DV + (size_t)src * d,
                       d * sizeof(bf16));
            }
        }
        std::vector<float> ctl_f((size_t)L_wanted * DQ);
        cpu_ref::gqa_attention(q_host.data() + (size_t)L_begin * DQ,
                               k_rot.data(), v_rot.data(), ctl_f.data(),
                               geo, L_begin, L_wanted, 0, nullptr);
        buffer<bf16> o_ctl((size_t)L_wanted * DQ);
        for (size_t i = 0; i < ctl_f.size(); i++) o_ctl[i] = ctl_f[i];

        error_metrics ctl = get_error_metrics(o_npu, o_ctl);
        header_print("info", "Cross-head control (rotated kv heads) -- should be POOR");
        print_error_metrics(ctl);

        if (!(metrics.CosineSimilarity > ctl.CosineSimilarity + 0.05f)) {
            header_print("warn", "Control is as good as the real reference: the "
                                 "head mapping is not actually being tested.");
            pass = false;
        }
    }

    header_print("info", "NPU time: " << npu_time.first << " us");
    if (pass){
        header_print("info", "PASSED ");
    } else {
        header_print("info", "FAILED!");
    }
    return pass ? 0 : 1;
}
