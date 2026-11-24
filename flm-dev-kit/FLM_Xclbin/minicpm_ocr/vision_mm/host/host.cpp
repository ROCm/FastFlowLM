// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file host.cpp
/// \brief Bring-up for the Qwen3.8-27B vision-tower matmuls on the NPU.
///
/// The real vision weights are not converted yet, so every shape runs on seeded
/// random bf16 data. That is enough to prove the tiling, the array mapping and
/// the DDR addressing, which is what this stage is for.
///
/// Shapes (from vision_config in the Qwen3.8-27B config.json: hidden 1152,
/// intermediate 4304, out_hidden 5120, spatial_merge 2, act gelu_pytorch_tanh):
///
///   design A - 64 x 384 x 48 per core, 256 x 384 per array pass
///     attn.qkv          L   x 1152 x 3456
///     attn.proj         L   x 1152 x 1152
///     mlp.linear_fc1    L   x 1152 x 4304   (N padded to 4608, GELU)
///     mlp.linear_fc2    L   x 4304 x 1152   (K padded to 4608)
///     merger.fc1        L/4 x 4608 x 4608   (GELU)
///
///   design B - 32 x 384 x 128 per core, 128 x 1024 per array pass
///     merger.fc2        L/4 x 4608 x 5120
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
#include "cpu_ref_vision_mm.hpp"

namespace po = boost::program_options;
using vision_mm_ref::activation;

namespace {

/// One matmul to bring up.
struct mm_shape {
    std::string name;
    int M;              // logical rows (L, or L/4 for the merger)
    int K;              // logical inner dim
    int N;              // logical output dim
    int K_pad;          // K rounded up to the k tile
    int N_pad;          // N rounded up to the array N tile
    activation act;
    std::string design; // VISION_MM_A or VISION_MM_B
};

struct design_params {
    int m, k, n;
    static constexpr int total_rows = 4;
    static constexpr int total_cols = 8;
    int tile_M() const { return m * total_rows; }
    int tile_N() const { return n * total_cols; }
};

const design_params kDesignA{64, 384, 48};
const design_params kDesignB{32, 384, 128};

const design_params& params_for(const std::string& design) {
    return design == "VISION_MM_B" ? kDesignB : kDesignA;
}

int round_up(int v, int q) { return ((v + q - 1) / q) * q; }

/// Relative L2, ||y - ref|| / ||ref||. Reported alongside cosine because cosine
/// is blind to a uniform scale error and a worst-row cosine drifts down with row
/// count even at constant error.
double rel_l2(const bf16* y, const bf16* ref, size_t n) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < n; i++) {
        const double d = (double)(float)y[i] - (double)(float)ref[i];
        num += d * d;
        den += (double)(float)ref[i] * (double)(float)ref[i];
    }
    return den > 0.0 ? std::sqrt(num / den) : (num > 0.0 ? INFINITY : 0.0);
}

double cosine(const bf16* y, const bf16* ref, size_t n) {
    double dot = 0.0, ny = 0.0, nr = 0.0;
    for (size_t i = 0; i < n; i++) {
        const double a = (float)y[i], b = (float)ref[i];
        dot += a * b; ny += a * a; nr += b * b;
    }
    return (ny > 0.0 && nr > 0.0) ? dot / (std::sqrt(ny) * std::sqrt(nr)) : 0.0;
}

void fill_random(bf16* p, size_t n, uint32_t seed, float scale) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, scale);
    for (size_t i = 0; i < n; i++) p[i] = (bf16)dist(rng);
}

}  // namespace

int main(int argc, char **argv) {
    srand(0);
    po::options_description desc("Allowed options");
    po::variables_map vm;
    arg_utils::add_default_options(desc);

    desc.add_options()("Rows,L", po::value<int>()->default_value(1024), "Vision token count L");
    desc.add_options()("Iterations,i", po::value<int>()->default_value(20), "Timing iterations");
    desc.add_options()("CosBar", po::value<double>()->default_value(0.999), "Cosine pass bar");
    desc.add_options()("CtrlBar", po::value<double>()->default_value(0.5), "Control must score BELOW this");

    arg_utils::parse_options(argc, argv, desc, vm);

    const int L = vm["Rows"].as<int>();
    const int Iterations = vm["Iterations"].as<int>();
    const double cos_bar = vm["CosBar"].as<double>();
    const double ctrl_bar = vm["CtrlBar"].as<double>();

    const int H = 1152;     // vision hidden_size
    const int I = 4304;     // vision intermediate_size
    const int OUT = 5120;   // out_hidden_size
    const int MERGE = 4608; // hidden_size * spatial_merge_size^2

    // L/4 after the 2x2 spatial merge.
    const int Lm = L / 4;

    std::vector<mm_shape> shapes = {
        {"attn.qkv",       L,  H,     3 * H, 0, 0, activation::none, "VISION_MM_A"},
        {"attn.proj",      L,  H,     H,     0, 0, activation::none, "VISION_MM_A"},
        {"mlp.fc1",        L,  H,     I,     0, 0, activation::gelu, "VISION_MM_A"},
        {"mlp.fc2",        L,  I,     H,     0, 0, activation::none, "VISION_MM_A"},
        {"merger.fc1",     Lm, MERGE, MERGE, 0, 0, activation::gelu, "VISION_MM_A"},
        {"merger.fc2",     Lm, MERGE, OUT,   0, 0, activation::none, "VISION_MM_B"},
    };

    for (auto& s : shapes) {
        const design_params& dp = params_for(s.design);
        s.K_pad = round_up(s.K, dp.k);
        s.N_pad = round_up(s.N, dp.tile_N());
    }

    xrt::device npu_device = xrt::device(0);
    npu_xclbin_manager xclbin_mgr(npu_device::device_npu2, &npu_device, false);

    // Two designs means two bitstreams. A runlist is bound to one hw_context, so
    // these cannot be merged - each is registered and dispatched separately.
    std::map<std::string, npu_app_manager*> app_mgrs;
    std::map<std::string, uint32_t> rtp_addrs;
    for (const std::string& design : {std::string("VISION_MM_A"), std::string("VISION_MM_B")}) {
        // Each design deploys under its own name rather than a shared
        // mm.xclbin -- both land in one CI artifact, which is flattened. The
        // mapping is DEPLOY_NAME in the Makefile; this mirrors it.
        const std::string deploy_name =
            design == "VISION_MM_A" ? "vision_mm_a.xclbin" : "vision_mm_b.xclbin";
        const std::string xclbin = "build/" + design + "/xclbins/" + deploy_name;
        if (!std::filesystem::exists(xclbin)) {
            header_print("error", "missing " << xclbin << "; run: make bitstream MODEL_TYPE=" << design);
            return 1;
        }
        app_mgrs[design] = xclbin_mgr.register_xclbin(xclbin);

        // The RTP address is assigned per build. A stale one does not fault, it
        // hangs in wait(), so read it from the build rather than hardcoding.
        const std::string rtp_file = "build/" + design + "/rtp_addr.txt";
        std::ifstream f(rtp_file);
        if (!f) {
            header_print("error", "missing " << rtp_file << "; run: tools/extract_rtp_addr.sh " << design);
            return 1;
        }
        uint32_t addr = 0;
        f >> addr;
        rtp_addrs[design] = addr;
        header_print("info", design << " rtp_address = " << addr);
    }

    bool all_pass = true;

    for (const auto& s : shapes) {
        const design_params& dp = params_for(s.design);
        const int M_pad = round_up(s.M, dp.tile_M());

        MSG_BONDLINE(72);
        header_print("info", s.name << "  M=" << s.M << " K=" << s.K << " N=" << s.N
                     << "  ->  padded " << M_pad << " x " << s.K_pad << " x " << s.N_pad
                     << "  [" << s.design << "]");

        const size_t X_VOL = (size_t)M_pad * s.K_pad;
        const size_t W_VOL = (size_t)s.N_pad * s.K_pad + s.N_pad;  // bias then weights
        const size_t Y_VOL = (size_t)M_pad * s.N_pad;

        npu_app app = app_mgrs[s.design]->create_app();
        buffer<bf16> y_bo = app.create_bo_buffer<bf16>(Y_VOL);
        buffer<bf16> x_bo = app.create_bo_buffer<bf16>(X_VOL);
        buffer<bf16> w_bo = app.create_bo_buffer<bf16>(W_VOL);

        // Random inputs, zero in the pad. Scale ~1/sqrt(K) keeps the fp32
        // accumulator in a sane range so the bf16 output does not saturate.
        const float wscale = 1.0f / std::sqrt((float)s.K);
        std::vector<bf16> x_src((size_t)s.M * s.K);
        std::vector<bf16> w_src((size_t)s.N * s.K);
        std::vector<bf16> b_src((size_t)s.N);
        fill_random(x_src.data(), x_src.size(), 0x1234u, 1.0f);
        fill_random(w_src.data(), w_src.size(), 0x5678u, wscale);
        fill_random(b_src.data(), b_src.size(), 0x9abcu, 0.1f);

        x_bo.memset((bf16)0.0f);
        w_bo.memset((bf16)0.0f);
        y_bo.memset((bf16)(-1.0f));

        for (int i = 0; i < s.M; i++)
            memcpy(x_bo.data() + (size_t)i * s.K_pad, x_src.data() + (size_t)i * s.K, s.K * sizeof(bf16));
        memcpy(w_bo.data(), b_src.data(), s.N * sizeof(bf16));
        for (int j = 0; j < s.N; j++)
            memcpy(w_bo.data() + s.N_pad + (size_t)j * s.K_pad, w_src.data() + (size_t)j * s.K, s.K * sizeof(bf16));

        x_bo.sync_to_device();
        w_bo.sync_to_device();
        y_bo.sync_to_device();

        gen_mm_seq(app.seq(), M_pad, s.K_pad, s.N_pad, dp.m, dp.k, dp.n,
                   0, (int32_t)s.act, rtp_addrs[s.design]);

        // --- dispatch 1 ---
        time_utils::time_point t0 = time_utils::now();
        ert_cmd_state st = app(y_bo, x_bo, w_bo);
        double us1 = time_utils::duration_us(t0, time_utils::now()).first;
        if (st != ERT_CMD_STATE_COMPLETED) {
            header_print("error", s.name << ": dispatch 1 state " << (int)st << " (not COMPLETED)");
            all_pass = false;
            continue;
        }
        y_bo.sync_from_device();
        std::vector<bf16> y1(Y_VOL);
        memcpy(y1.data(), y_bo.data(), Y_VOL * sizeof(bf16));

        // --- dispatch 2: must be bit-identical ---
        // A design that is correct once and wrong on redispatch is a recurring
        // failure mode here (prologue BDs, un-drained rings). Check the state and
        // the wall clock too: a timeout leaves the previous contents in the BO,
        // which otherwise scores as a clean pass.
        y_bo.memset((bf16)(-1.0f));
        y_bo.sync_to_device();
        t0 = time_utils::now();
        st = app(y_bo, x_bo, w_bo);
        double us2 = time_utils::duration_us(t0, time_utils::now()).first;
        if (st != ERT_CMD_STATE_COMPLETED) {
            header_print("error", s.name << ": dispatch 2 state " << (int)st << " (not COMPLETED)");
            all_pass = false;
            continue;
        }
        y_bo.sync_from_device();
        const bool redispatch_ok = (memcmp(y1.data(), y_bo.data(), Y_VOL * sizeof(bf16)) == 0);

        // --- reference over the same bf16 bytes ---
        std::vector<bf16> ref((size_t)s.M * s.N);
        vision_mm_ref::mm_ref(x_src.data(), w_src.data(), b_src.data(), ref.data(),
                              s.M, s.K, s.N, s.act, 0);

        // Gather the logical (M, N) block out of the padded output.
        std::vector<bf16> got((size_t)s.M * s.N);
        for (int i = 0; i < s.M; i++)
            memcpy(got.data() + (size_t)i * s.N, y1.data() + (size_t)i * s.N_pad, s.N * sizeof(bf16));

        const double cos = cosine(got.data(), ref.data(), got.size());
        const double l2 = rel_l2(got.data(), ref.data(), got.size());

        // --- control: same weights, rows rotated ---
        // Rotating w's rows permutes the output columns. A correct result must
        // score badly against this; a transposed or permuted array mapping would
        // agree with the true reference *and* with itself, so without this a
        // pure cosine check cannot tell the difference.
        std::vector<bf16> ctrl((size_t)s.M * s.N);
        vision_mm_ref::mm_ref(x_src.data(), w_src.data(), b_src.data(), ctrl.data(),
                              s.M, s.K, s.N, s.act, s.N / 2);
        const double cos_ctrl = cosine(got.data(), ctrl.data(), got.size());

        // --- per-N-band relative L2 ---
        // One bad band can hide inside a good overall cosine, so break it out at
        // the array's N granularity.
        const int band = dp.tile_N();
        double worst_band = 0.0; int worst_idx = -1;
        for (int b0 = 0; b0 < s.N; b0 += band) {
            const int bw = std::min(band, s.N - b0);
            std::vector<bf16> gb((size_t)s.M * bw), rb((size_t)s.M * bw);
            for (int i = 0; i < s.M; i++) {
                memcpy(gb.data() + (size_t)i * bw, got.data() + (size_t)i * s.N + b0, bw * sizeof(bf16));
                memcpy(rb.data() + (size_t)i * bw, ref.data() + (size_t)i * s.N + b0, bw * sizeof(bf16));
            }
            const double e = rel_l2(gb.data(), rb.data(), gb.size());
            if (e > worst_band) { worst_band = e; worst_idx = b0 / band; }
        }

        // --- padding must be inert ---
        bool pad_zero = true;
        for (int i = 0; i < s.M && pad_zero; i++)
            for (int j = s.N; j < s.N_pad; j++)
                if ((float)y1[(size_t)i * s.N_pad + j] != 0.0f) { pad_zero = false; break; }
        for (int i = s.M; i < M_pad && pad_zero; i++)
            for (int j = 0; j < s.N; j++)
                if ((float)y1[(size_t)i * s.N_pad + j] != 0.0f) { pad_zero = false; break; }

        const bool ok = (cos >= cos_bar) && (cos_ctrl < ctrl_bar) && redispatch_ok;

        header_print("info", "  cosine        : " << std::setprecision(6) << cos
                     << (cos >= cos_bar ? "  PASS" : "  FAIL"));
        header_print("info", "  relative L2   : " << l2);
        header_print("info", "  worst N band  : " << worst_band << "  (band " << worst_idx << " of "
                     << ((s.N + band - 1) / band) << ")");
        header_print("info", "  control cos   : " << cos_ctrl
                     << (cos_ctrl < ctrl_bar ? "  PASS (correctly uncorrelated)" : "  FAIL (matches a rotated ref!)"));
        header_print("info", "  redispatch    : " << (redispatch_ok ? "bit-identical  PASS" : "DIFFERS  FAIL"));
        header_print("info", "  pad inert     : " << (pad_zero ? "yes" : "NO - pad is nonzero"));
        header_print("info", "  dispatch us   : " << us1 << " then " << us2);

        if (!ok) all_pass = false;

        if (Iterations > 0 && ok) {
            double acc = 0.0;
            for (int i = 0; i < Iterations; i++) {
                time_utils::time_point a = time_utils::now();
                app(y_bo, x_bo, w_bo);
                acc += time_utils::duration_us(a, time_utils::now()).first;
            }
            time_utils::time_with_unit t = {acc / Iterations, "us"};
            utils::print_npu_profile(t, 2.0 * (double)M_pad * (double)s.K_pad * (double)s.N_pad);
        }
    }

    MSG_BONDLINE(72);
    if (all_pass) {
        header_print("info", "PASSED");
        return 0;
    }
    header_print("info", "FAILED!");
    return 1;
}
