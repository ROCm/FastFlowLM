// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file inkling2_lm_head_probe.cpp
/// \brief the NPU lm_head (inkling2_lm_head) against the host's on real weights
/// \note usage: i2lmprobe model.q4nx lm_head.xclbin [runs]
///       x: random final-normed tokens (bf16); per run the soft-capped logits are compared with the host (rel rms,
///       worst |difference|, argmax and top-8 agreement) and both are timed.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "../inkling2_host.hpp"
#include "../inkling2_lm_head.hpp"
#include "npu_utils/npu_utils.hpp"

using namespace inkling2_host;

namespace {
uint16_t f2bf(float f) { uint32_t u; memcpy(&u, &f, 4); u += 0x7FFF + ((u >> 16) & 1); return (uint16_t)(u >> 16); }
float bf2f(uint16_t v) { uint32_t u = (uint32_t)v << 16; float f; memcpy(&f, &u, 4); return f; }
std::vector<int> top(const std::vector<float>& l, int k) {
    std::vector<int> i(VOCAB);
    for (int j = 0; j < VOCAB; j++) i[j] = j;
    std::partial_sort(i.begin(), i.begin() + k, i.end(), [&](int a, int b) { return l[a] > l[b]; });
    i.resize(k);
    return i;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s model.q4nx lm_head.xclbin [runs]\n", argv[0]);
        return 2;
    }
    const int runs = argc > 3 ? atoi(argv[3]) : 8;
    model m;
    m.load(argv[1]);
    xrt::device dev(0);
    npu_xclbin_manager mgr(npu_device::device_npu2, &dev, false);
    using clk = std::chrono::steady_clock;
    auto t0 = clk::now();
    inkling2_lm_head L(&mgr, argv[2], m.w());
    printf("lm_head engine: %d rows, setup %.0f ms\n", inkling2_lm_head::ROWS,
           std::chrono::duration<double, std::milli>(clk::now() - t0).count());
    std::mt19937 g(1234);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<uint16_t> xb(D);
    std::vector<float> xf(D), lh(VOCAB), ln(VOCAB);
    double th = 0, worst_rel = 0;
    int am = 0, t8 = 0;
    for (int r = 0; r < runs; r++) {
        // rms 1 times a per-channel gain, like a final-normed token
        for (int i = 0; i < D; i++) { xb[i] = f2bf(nd(g) * (0.5f + (i % 7) * 0.25f)); xf[i] = bf2f(xb[i]); }
        t0 = clk::now();
        lm_head(m.w(), xf.data(), lh.data());
        th += std::chrono::duration<double, std::milli>(clk::now() - t0).count();
        if (!L.run(xb.data(), ln.data())) { printf("NPU run failed\n"); return 1; }
        double e = 0, s = 0, wd = 0;
        for (int i = 0; i < VOCAB; i++) {
            const double d = (double)ln[i] - lh[i];
            e += d * d; s += (double)lh[i] * lh[i]; wd = std::max(wd, std::fabs(d));
        }
        const auto a = top(lh, 8), b = top(ln, 8);
        int common = 0;
        for (int v : a) common += std::count(b.begin(), b.end(), v);
        am += a[0] == b[0]; t8 += common;
        worst_rel = std::max(worst_rel, std::sqrt(e / s));
        printf("run %d: rel rms %.2e, worst |d| %.3f, argmax host %d npu %d, top-8 common %d\n", r, std::sqrt(e / s), wd, a[0], b[0], common);
    }
    // padding rows past VOCAB_PAD are zero blocks
    double pad = 0;
    for (int i = VOCAB_PAD; i < inkling2_lm_head::ROWS; i++) pad = std::max(pad, (double)std::fabs(bf2f(L.y_out()[i])));
    printf("argmax %d/%d, top-8 %d/%d, worst rel rms %.2e, padding max %.3g\n", am, runs, t8, 8 * runs, worst_rel, pad);
    printf("host lm_head %.2f ms; NPU lm_head %.2f ms (x sync, run, y sync) + soft cap %.2f ms\n", th / runs, L.t_run / L.n_run,
           L.t_post / L.n_run);
    // timing alone
    const int nt = 50;
    const double r0 = L.t_run, p0 = L.t_post, d0 = L.t_dev, s0 = L.t_sync;
    for (int r = 0; r < nt; r++) L.run(xb.data(), ln.data());
    printf("NPU %d runs: %.3f ms (device %.3f, y sync %.3f) + soft cap %.3f ms\n", nt, (L.t_run - r0) / nt, (L.t_dev - d0) / nt,
           (L.t_sync - s0) / nt, (L.t_post - p0) / nt);
    return 0;
}
