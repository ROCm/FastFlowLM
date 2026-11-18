// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file tools/bench_lm_head_rows.cpp
/// \brief How the host lm_head scales with row count, which is what decides
///        whether module 3 wants the L = 8 bitstream.
///
/// The question. Verify projects k+1 = 8 rows per speculation cycle through
/// `cpu_ref::lm_head::norm_and_forward(x, L, ..., keep = L, ...)`, one call.
/// FLM_Xclbin/Qwen3_8/lm_head_8 does the same 8 rows on the NPU in ~40 ms. So
/// the move is only a win if the host needs more than 40 ms for 8 rows.
///
/// Why it is not obvious. q4nx_matmul()'s `l` loop is INSIDE the (rb, cb) tile
/// loop -- each weight block is dequantised once and reused across all L rows.
/// So the 1.35 GB table is streamed once no matter what L is, and the marginal
/// cost of a row is a rank-1 update against resident tiles. A per-row loop
/// would cost 8x; this does not. Measure the slope rather than assume it.
///
/// Flags matter. Build at the ENGINE's flags -- at -O2 the host path is ~1.7x
/// slower than it ships and the verdict comes out with the wrong sign.
///
///   cd detail/qwen3_8mtp_npu
///   g++-13 -O3 -march=native -ffast-math -fopenmp -std=c++20 \
///       -mavx512f -mavx512vl -mavx512bw -mavx512dq -mfma \
///       -I. -I../../include -I../q4nx -I/opt/xilinx/xrt/include \
///       tools/bench_lm_head_rows.cpp \
///       -L../../../../src/lib/xrt -L/opt/xilinx/xrt/lib \
///       -lxrt_coreutil -luuid -lpthread -o /tmp/bench_lm_head_rows
///   /tmp/bench_lm_head_rows
#include "common/lm_head.hpp"
#include "common/cpu_ref_common.hpp"

#include <chrono>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    const std::string model_dir =
        argc > 1 ? argv[1] : "/path/to/models/Qwen3.8-27B-NPU2";
    const int REPS = argc > 2 ? std::atoi(argv[2]) : 3;

    std::printf("== host lm_head, cost vs row count ==\n   %s\n\n",
                model_dir.c_str());

    cpu_ref::safetensors q4nx(model_dir + "/model.q4nx");
    cpu_ref::lm_head head(q4nx);
    const int V = head.vocab();
    const int D = head.dim();
    std::printf("   table %d x %d\n\n", V, D);

    const std::vector<float> norm_w = q4nx.f32("model.norm.weight");

    // Real activations are not what is being measured -- the inner loop skips
    // exact zeros, so feed dense noise rather than a zeroed buffer, which
    // would time a branch instead of the matmul.
    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> x((size_t)8 * D);
    for (auto& v : x) v = nd(rng);

    std::vector<float> logits((size_t)8 * V);

    // warm: first touch faults in 1.35 GB of mapped weights, and that page-in
    // would otherwise be charged to L = 1.
    head.norm_and_forward(x.data(), 1, norm_w.data(), 1, logits.data(),
                          cpu_ref::q38::RMS_EPS);

    std::printf("      L    best ms   ms/row   vs L=1\n");
    std::printf("   ------------------------------------\n");
    double base = 0.0;
    for (int L : {1, 2, 4, 8}) {
        double best = 1e30;
        for (int r = 0; r < REPS; r++) {
            const auto t0 = std::chrono::steady_clock::now();
            head.norm_and_forward(x.data(), L, norm_w.data(), L,
                                  logits.data(), cpu_ref::q38::RMS_EPS);
            const auto t1 = std::chrono::steady_clock::now();
            const double ms =
                std::chrono::duration<double, std::milli>(t1 - t0).count();
            if (ms < best) best = ms;
        }
        if (L == 1) base = best;
        std::printf("   %4d   %8.2f   %6.2f   %6.2fx\n",
                    L, best, best / L, best / base);
    }

    std::printf("\n   lm_head_8 on the NPU does 8 rows in ~40 ms"
                " (FLM_Xclbin/Qwen3_8/lm_head_8/README.md).\n");
    return 0;
}
