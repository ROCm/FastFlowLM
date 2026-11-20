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
#include "tensor_2d.hpp"
#include "weights_desc.hpp"
#include "cpu_q4_1.hpp"
#include "cpu_attn.hpp"
#include "gemma4_weights.hpp"
#include "seq_gen.hpp"
#include "bf16_pack.hpp"

namespace po = boost::program_options;

// Fused Gemma4 prefill: one xclbin, three regions.
//
//     column 0     full attention,     head dim 512   (layer 4, 9, 14, ...)
//     columns 1-6  Q4_1 dequant + mm   (every projection)
//     column 7     sliding attention,  head dim 256, window 512
//
// Each region gets its own npu_app over the same xclbin, so the three are
// independent sequences that happen to share a bitstream.
//
// Everything is checked against a CPU reference computed from the real
// Gemma4-E2B-IT-NPU2 weights:
//
//   * the mm against an fp32 matmul of the weights the dequant core rebuilds
//     (q4_1_unpack), and separately against those weights pushed through
//     bfp16_floor_k, which is what aie::mmul actually multiplies on npu2;
//   * both attentions against a one-pass fp32 softmax over the same q/k/v the
//     NPU is given.  q is pre-scaled on the host because the kernel applies no
//     scaling of its own.
//
// The q/k/v fed to the attention columns are the real projections of a random
// activation, i.e. produced by the same weights the mm test uses.  Rope and the
// q/k norms are deliberately left out: they sit outside this xclbin and would
// only add a second unverified step between the reference and the hardware.

static const char* XCLBIN = "build/gemma4/xclbins/fused_prefill.xclbin";

/// \brief Put a weight matrix on the grid aie::mmul actually multiplies on.
///
/// mm_aie's `aie::mmul<8,8,8,bf16,bf16,accauto>` lowers to the bfp16 path on
/// npu2: every block of 8 consecutive k at a fixed n is given a shared exponent
/// and an 8-bit mantissa, truncated toward -infinity.  Multiplying by this
/// instead of by w_deq removes the kernel's largest remaining error term, so the
/// two references bracket it.
static void bfp16_floor_k(bf16* v, size_t rows, size_t k) {
#pragma omp parallel for schedule(static)
    for (size_t r = 0; r < rows; r++) {
        bf16* row = v + r * k;
        for (size_t g = 0; g < k; g += 8) {
            float mx = 0.0f;
            for (int e = 0; e < 8; e++) mx = std::max(mx, std::fabs((float)row[g + e]));
            if (mx == 0.0f) continue;
            const float st = std::exp2(std::floor(std::log2(mx)) + 1 - 7);
            for (int e = 0; e < 8; e++) row[g + e] = bf16(std::floor((float)row[g + e] / st) * st);
        }
    }
}

static void fill_normal(bf16* p, size_t n, unsigned seed, float sigma) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, sigma);
    for (size_t i = 0; i < n; i++) p[i] = bf16(dist(rng));
}

int main(int argc, char **argv) {
    srand(0);
    po::options_description desc("Allowed options");
    po::variables_map vm;
    arg_utils::add_default_options(desc);
    desc.add_options()("model", po::value<std::string>()->default_value("/path/to/models/Gemma4-E2B-IT-NPU2"),
                       "Gemma4 q4nx model directory");
    desc.add_options()("length", po::value<int>()->default_value(1024), "prefill length");
    arg_utils::parse_options(argc, argv, desc, vm);

    const std::string model_path = vm["model"].as<std::string>();
    const int L = vm["length"].as<int>();

    using cfg = gemma4_cfg;
    const int D  = cfg::hidden_size;                       // 1536
    const int H  = cfg::heads;                             // 8
    const int Nd = cfg::kv_heads;                          // 1, MQA

    const int DH_full = cfg::global_head_dim;              // 512
    const int DH_swa  = cfg::head_dim;                     // 256
    const int DQ_full = H * DH_full;                       // 4096
    const int DQ_swa  = H * DH_swa;                        // 2048

    // L must be a whole number of q rounds on both columns (4 * lq = 32 and 64)
    // and a whole number of mm token blocks (256), and a multiple of the larger
    // k block (128) so the last k stream stays inside the cache.
    assert(L % 128 == 0);

    const int full_layer = 4;   // first full_attention layer
    const int swa_layer  = 0;   // first sliding_attention layer

    header_print("info", "Gemma4 fused prefill, L = " << L
                 << ", hidden " << D << ", heads " << H << ", kv heads " << Nd);

    //////////////////////////// weights ////////////////////////////
    header_print("info", "Loading q4nx weights from " << model_path);
    Q4NX q4nx(model_path);

    // Layer 4: the full-attention projections, head dim 512.
    buffer<u8>   q4_packed_full(q4_1_size((size_t)DQ_full * D));
    buffer<bf16> wq_full((size_t)DQ_full * D);
    load_proj(q4nx, g4_name(full_layer, "q_proj"), DQ_full, D, q4_packed_full.data(), wq_full.data());

    buffer<bf16> wkv_full((size_t)2 * DH_full * D);   // k stacked on v, both 512 x 1536
    {
        buffer<u8> kp(q4_1_size((size_t)DH_full * D)), vp(q4_1_size((size_t)DH_full * D));
        load_proj(q4nx, g4_name(full_layer, "k_proj"), DH_full, D, kp.data(), wkv_full.data());
        load_proj(q4nx, g4_name(full_layer, "v_proj"), DH_full, D, vp.data(),
                  wkv_full.data() + (size_t)DH_full * D);
    }

    // Layer 0: the sliding projections, head dim 256.
    buffer<u8>   q4_packed_swa(q4_1_size((size_t)DQ_swa * D));
    buffer<bf16> wq_swa((size_t)DQ_swa * D);
    load_proj(q4nx, g4_name(swa_layer, "q_proj"), DQ_swa, D, q4_packed_swa.data(), wq_swa.data());

    buffer<bf16> wk_swa((size_t)DH_swa * D), wv_swa((size_t)DH_swa * D);
    {
        buffer<u8> kp(q4_1_size((size_t)DH_swa * D)), vp(q4_1_size((size_t)DH_swa * D));
        load_proj(q4nx, g4_name(swa_layer, "k_proj"), DH_swa, D, kp.data(), wk_swa.data());
        load_proj(q4nx, g4_name(swa_layer, "v_proj"), DH_swa, D, vp.data(), wv_swa.data());
    }

    //////////////////////////// activations ////////////////////////////
    // A hidden state with roughly the spread of a post-norm Gemma activation.
    buffer<bf16> x((size_t)L * D);
    fill_normal(x.data(), x.size(), 1, 1.0f);

    //////////////////////////// xclbin ////////////////////////////
    xrt::device npu_device = xrt::device(0);
    npu_xclbin_manager xclbin_mgr(npu_device::device_npu2, &npu_device, false);
    npu_app_manager* fused = xclbin_mgr.register_xclbin(XCLBIN);

    int failures = 0;

    // Case 1a's output, kept alive so 1b -- same weights, same x, different
    // path -- can be diffed straight against it.  That is a sharper check than
    // either CPU reference, since it cancels the bfp16 emulation exactly.
    buffer<bf16> y_q4_path((size_t)L * DQ_full);
    ////////////////////////////////////////////////////////////////////////
    // 1. dequant + matmul on columns 1-6: layer 4 q_proj, 1536 -> 4096.
    ////////////////////////////////////////////////////////////////////////
    {
        header_print("info", "[cols 1-6] dequant_mm  q_proj L" << full_layer
                     << "  " << D << " -> " << DQ_full);

        npu_app q_proj = fused->create_app();
        gen_dequant_mm_512(q_proj.seq(), L, D, DQ_full, 0, 0, 0);

        buffer<u8>   bo_w = q_proj.create_bo_buffer<u8>(q4_packed_full.size());
        buffer<bf16> bo_x = q_proj.create_bo_buffer<bf16>((size_t)L * D);
        buffer<bf16> bo_y = q_proj.create_bo_buffer<bf16>((size_t)L * DQ_full);

        bo_w.copy_from(q4_packed_full.data(), q4_packed_full.size());
        bo_w.sync_to_device();
        bo_x.copy_from(x.data(), x.size());
        bo_x.sync_to_device();
        bo_y.memset(bf16(-1.0f));
        bo_y.sync_to_device();

        header_print("info", "CPU reference");
        buffer<bf16> y_ref((size_t)L * DQ_full), y_ref_bfp((size_t)L * DQ_full);
        buffer<bf16> w_bfp((size_t)DQ_full * D);
        memcpy(w_bfp.data(), wq_full.data(), wq_full.size() * sizeof(bf16));
        bfp16_floor_k(w_bfp.data(), DQ_full, D);
        cpu_matmul_nt(y_ref.data(), x.data(), wq_full.data(), L, DQ_full, D);
        cpu_matmul_nt(y_ref_bfp.data(), x.data(), w_bfp.data(), L, DQ_full, D);

        time_utils::time_point start = time_utils::now();
        q_proj(bo_y, bo_x, bo_w);
        time_utils::time_point stop = time_utils::now();
        bo_y.sync_from_device();
        MSG_BOX_LINE(60, "NPU dequant_mm: " << time_utils::duration_us(start, stop).first << " us");

        buffer<bf16> y((size_t)L * DQ_full);
        y.copy_from(bo_y.data(), y.size());
        error_metrics m = get_error_metrics(y, y_ref);
        print_error_metrics(m, "dequant_mm vs fp32 matmul of the dequantized weights");
        print_error_metrics(get_error_metrics(y, y_ref_bfp),
                            "dequant_mm vs fp32 matmul of the bfp16-floored weights");
        if (m.CosineSimilarity < 0.99) { header_print("error", "dequant_mm mismatch"); failures++; }
        memcpy(y_q4_path.data(), y.data(), y.size() * sizeof(bf16));
    }

    ////////////////////////////////////////////////////////////////////////
    // 1b. the same matmul with plain bf16 weights, through the q_type RTP.
    //     This is what lets the per-layer-input projections -- which are not
    //     quantised -- stay on this overlay instead of forcing a context switch
    //     to the image encoder's mm overlay twice per layer.
    ////////////////////////////////////////////////////////////////////////
    {
        header_print("info", "[cols 1-6] bf16 mm  q_proj L" << full_layer
                     << "  " << D << " -> " << DQ_full);

        npu_app bf_proj = fused->create_app();
        gen_dequant_mm_512(bf_proj.seq(), L, D, DQ_full, 0, 0, 0, /*is_bf16=*/true);

        const size_t packed_n = bf16_pack::packed_bytes(DQ_full, D) / sizeof(bf16);
        buffer<bf16> w_packed(packed_n);
        bf16_pack::pack(w_packed.data(), wq_full.data(), DQ_full, D);

        buffer<bf16> bo_w = bf_proj.create_bo_buffer<bf16>(packed_n);
        buffer<bf16> bo_x = bf_proj.create_bo_buffer<bf16>((size_t)L * D);
        buffer<bf16> bo_y = bf_proj.create_bo_buffer<bf16>((size_t)L * DQ_full);

        bo_w.copy_from(w_packed.data(), w_packed.size());
        bo_w.sync_to_device();
        bo_x.copy_from(x.data(), x.size());
        bo_x.sync_to_device();
        bo_y.memset(bf16(-1.0f));
        bo_y.sync_to_device();

        // wq_full is already the dequantised q4_1 weight, so y_ref is the exact
        // reference here: same weights, same inputs, only the path differs.
        buffer<bf16> y_ref((size_t)L * DQ_full);
        cpu_matmul_nt(y_ref.data(), x.data(), wq_full.data(), L, DQ_full, D);

        time_utils::time_point start = time_utils::now();
        bf_proj(bo_y, bo_x, bo_w);
        time_utils::time_point stop = time_utils::now();
        bo_y.sync_from_device();
        MSG_BOX_LINE(60, "NPU bf16 mm: " << time_utils::duration_us(start, stop).first << " us");

        buffer<bf16> y((size_t)L * DQ_full);
        y.copy_from(bo_y.data(), y.size());
        error_metrics m = get_error_metrics(y, y_ref);
        print_error_metrics(m, "bf16 mm vs fp32 matmul of the same weights");
        print_error_metrics(get_error_metrics(y, y_q4_path),
                            "bf16 mm vs the q4_1 path's own output, same weights");
        if (m.CosineSimilarity < 0.999) { header_print("error", "bf16 mm mismatch"); failures++; }
    }

    ////////////////////////////////////////////////////////////////////////
    // 2 and 3. the two attention columns.
    ////////////////////////////////////////////////////////////////////////
    auto run_attention = [&](const char* tag, const attn_col_cfg& acfg, int DH, int window,
                             const bf16* wq, const bf16* wk, const bf16* wv) {
        header_print("info", "[col " << acfg.col << "] " << tag
                     << "  head dim " << DH << ", window " << window);

        const int DQ = H * DH;
        const size_t KV_CACHE_SIZE = (size_t)2 * L * DH * Nd;

        // Project on the CPU with the real weights, then pre-scale q: the kernel
        // does no scaling of its own, so 1/sqrt(head_dim) has to be folded in
        // before the data leaves the host.
        buffer<bf16> q((size_t)L * DQ), k((size_t)L * DH * Nd), v((size_t)L * DH * Nd);
        cpu_matmul_nt(q.data(), x.data(), wq, L, DQ, D);
        cpu_matmul_nt(k.data(), x.data(), wk, L, DH * Nd, D);
        cpu_matmul_nt(v.data(), x.data(), wv, L, DH * Nd, D);
        const float qscale = 1.0f / std::sqrt((float)DH);
        for (size_t i = 0; i < q.size(); i++) q[i] = bf16((float)q[i] * qscale);

        npu_app app = fused->create_app();
        gen_attn_column(app.seq(), acfg, 0, L, window, H, Nd, 0, KV_CACHE_SIZE);

        buffer<bf16> bo_q = app.create_bo_buffer<bf16>((size_t)L * DQ);
        buffer<bf16> bo_kv = app.create_bo_buffer<bf16>(KV_CACHE_SIZE);
        buffer<bf16> bo_o = app.create_bo_buffer<bf16>((size_t)L * DQ);

        bo_q.copy_from(q.data(), q.size());
        bo_q.sync_to_device();
        // k in the first half of the cache, v in the second -- the shim BD steps
        // between them by KV_CACHE_SIZE / 2.
        bo_kv.memset(bf16(0.0f));
        memcpy(bo_kv.data(), k.data(), k.size() * sizeof(bf16));
        memcpy(bo_kv.data() + KV_CACHE_SIZE / 2, v.data(), v.size() * sizeof(bf16));
        bo_kv.sync_to_device();
        bo_o.memset(bf16(-1.0f));
        bo_o.sync_to_device();

        header_print("info", "CPU reference");
        buffer<bf16> o_ref((size_t)L * DQ);
        cpu_attn_mha(o_ref.data(), q.data(), k.data(), v.data(), L, DH, H, window);

        time_utils::time_point start = time_utils::now();
        app(bo_o, bo_q, bo_kv);
        time_utils::time_point stop = time_utils::now();
        bo_o.sync_from_device();
        MSG_BOX_LINE(60, "NPU " << tag << ": " << time_utils::duration_us(start, stop).first << " us");

        buffer<bf16> o((size_t)L * DQ);
        o.copy_from(bo_o.data(), o.size());
        error_metrics m = get_error_metrics(o, o_ref);
        print_error_metrics(m, std::string(tag) + " vs CPU attention");
        if (m.CosineSimilarity < 0.99) { header_print("error", tag << " mismatch"); failures++; }
    };

    // Full attention: the window is made vacuous by setting it to the whole
    // sequence, leaving the causal mask alone.
    run_attention("full attention", attn_full_cfg, DH_full, L,
                  wq_full.data(), wkv_full.data(), wkv_full.data() + (size_t)DH_full * D);

    run_attention("sliding attention", attn_swa_cfg, DH_swa, cfg::sliding_window,
                  wq_swa.data(), wk_swa.data(), wv_swa.data());

    MSG_BONDLINE(60);
    if (failures) { header_print("error", failures << " region(s) failed"); return 1; }
    header_print("info", "all three regions match the CPU reference");
    return 0;
}
