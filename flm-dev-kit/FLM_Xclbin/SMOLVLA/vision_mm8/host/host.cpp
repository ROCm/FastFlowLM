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
#include "tensor_utils/q4_npu_eXpress.hpp"
#include "tensor_2d.hpp"

namespace po = boost::program_options;

void Silu(int dim, bf16* x, bf16* o) {
    for (int i = 0; i < dim; i++) {
        bf16 xi = x[i];
        o[i] = (bf16)(xi / (1.0f + std::exp(-xi)));
    }
}
/// \brief fp32 CPU reference of the kernel: y[r][n] = sum_k x[r][k] * w[n][k] (+ b[n]).
void mm_cpu_ref(const bf16* x, const bf16* w, const bf16* b, bf16* y, int rows, int K, int N) {
    for (int r = 0; r < rows; r++) {
        for (int n = 0; n < N; n++) {
            float acc = b ? (float)b[n] : 0.0f;
            for (int k = 0; k < K; k++) {
                acc += (float)x[(size_t)r * K + k] * (float)w[(size_t)n * K + k];
            }
            y[(size_t)r * N + n] = (bf16)acc;
        }
    }
}

int main(int argc, char **argv) {
    const uint32_t M = std::getenv("MM_M") ? atoi(std::getenv("MM_M")) : 3072;
    const uint32_t K = std::getenv("MM_K") ? atoi(std::getenv("MM_K")) : 768;
    const uint32_t N = std::getenv("MM_N") ? atoi(std::getenv("MM_N")) : 768;
    const int RTP_ADDR = std::getenv("MM_RTP") ? atoi(std::getenv("MM_RTP")) : 44480;
    std::mt19937 rng(32);
    std::normal_distribution<float> dx(0.f, 1.f), dw(0.f, 1.f / std::sqrt((float)K)), db(0.f, 1.f);
    std::vector<bf16> x(M * (size_t)K), w((size_t)N * K), b(N);
    for (auto& v : x) v = (bf16)dx(rng);
    for (auto& v : w) v = (bf16)dw(rng);
    for (auto& v : b) v = (bf16)db(rng);

    xrt::device npu_device = xrt::device(0);
    npu_xclbin_manager xclbin_mgr(npu_device::device_npu2, &npu_device, false);
    npu_app_manager* mgr = xclbin_mgr.register_xclbin("build/0/xclbins/vision_mm8.xclbin");
    npu_app app = mgr->create_app();
    buffer<bf16> y_0 = app.create_bo_buffer<bf16>((size_t)M * N);
    buffer<bf16> x_0 = app.create_bo_buffer<bf16>((size_t)M * K);
    buffer<bf16> w_0 = app.create_bo_buffer<bf16>((size_t)N * K + N);
    memcpy(w_0.data(), b.data(), N * sizeof(bf16));
    memcpy(w_0.data() + N, w.data(), (size_t)N * K * sizeof(bf16));
    memcpy(x_0.data(), x.data(), x.size() * sizeof(bf16));
    y_0.memset((bf16)(-1.0f));
    w_0.sync_to_device(); x_0.sync_to_device(); y_0.sync_to_device();

    gen_mm8_seq(app.seq(), M, K, N, N, 1, 0, RTP_ADDR);
    app(y_0, x_0, w_0);  // first call also writes the RTPs
    y_0.sync_from_device();
    // verify sampled rows (all N columns): every 8th row starting at 3 plus the last row
    std::vector<bf16> ref(N), got(N), refall, gotall;
    for (uint32_t r = 3; r < M; r += 8) {
        mm_cpu_ref(x.data() + (size_t)r * K, w.data(), b.data(), ref.data(), 1, K, N);
        refall.insert(refall.end(), ref.begin(), ref.end());
        gotall.insert(gotall.end(), y_0.data() + (size_t)r * N, y_0.data() + (size_t)(r + 1) * N);
    }
    buffer<bf16> wanted(gotall.size()), reference(refall.size());
    memcpy(wanted.data(), gotall.data(), gotall.size() * sizeof(bf16));
    memcpy(reference.data(), refall.data(), refall.size() * sizeof(bf16));
    error_metrics metrics = get_error_metrics(wanted, reference);
    print_error_metrics(metrics, "NPU 8-col vs CPU reference");
    bool pass = metrics.CosineSimilarity > 0.999f;

    xrt::runlist rl = mgr->create_runlist();
    for (int R : {1, 4, 16}) {
        double best = 1e30;
        for (int rep = 0; rep < 20; rep++) {
            rl.reset();
            for (int i = 0; i < R; i++) rl.add(app.create_run(y_0, x_0, w_0));
            time_utils::time_point t0 = time_utils::now();
            rl.execute(); rl.wait();
            time_utils::time_point t1 = time_utils::now();
            best = std::min(best, (double)(time_utils::duration_us(t0, t1).first / R));
        }
        header_print("info", "M=" << M << " K=" << K << " N=" << N << " runlist R=" << R << " per-op best " << best << " us ("
                                  << 2.0 * M * K * N / best / 1e6 << " TOPS)");
    }
    header_print("info", pass ? "PASSED " : "FAILED!");
    return pass ? 0 : 1;
}
