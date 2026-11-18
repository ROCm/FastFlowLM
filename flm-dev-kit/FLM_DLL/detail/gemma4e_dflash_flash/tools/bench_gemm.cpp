// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

// microbenchmark of the Q4 GEMM kernel of the host drafter (unity build to reach the internals): GMAC/s per thread, data in L2/L3.
#include "../dflash_draft.cpp"
#include <chrono>
#include <cstdio>
int main(int argc, char** argv) {
    const int N = 512, K = 2560, M = argc > 1 ? atoi(argv[1]) : 8;
    Mat m; m.N = N; m.K = K; m.bf = false;
    const int G = K / 32;
    m.q.alloc((size_t)N * G * 16); m.dm.alloc((size_t)N * G);
    for (size_t i = 0; i < m.q.n; i++) m.q[i] = (uint8_t)(i * 2654435761u >> 7);
    for (size_t i = 0; i < m.dm.n; i++) m.dm[i] = 0x3c233c23u;
    abuf<f32> X, Y; X.alloc((size_t)64 * K); Y.alloc((size_t)64 * N);
    for (size_t i = 0; i < X.n; i++) X[i] = 0.001f * (i % 97);
    double best = 1e30;
    for (int rep = 0; rep < 40; rep++) {
        auto t0 = std::chrono::steady_clock::now();
        for (int it = 0; it < 20; it++) {
            for (int b = 0; b < N / 16; b++) gemm_rows(m, X.data(), K, M, Y.data(), N, b * 16);
        }
        double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        best = std::min(best, s);
    }
    const double mac = 20.0 * N * K * M;
    std::printf("M=%d  best %.2f GMAC/s per thread (weights %.2f MB)  checksum %f\n", M, mac / best / 1e9, (double)(m.q.n + m.dm.n * 4) / 1e6, Y[0] + Y[N + 5]);
}
