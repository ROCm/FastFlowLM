// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

// Array GEMM with a large result buffer: d3-turbo's up/gate projection of a 32,768-token prompt is
// [32768, 2 * 17408] bf16 = 2.28 GB (d3-mini's 1.2 GB, Qwen3.5-9B's 1.6 GB were the largest before). Measured on
// this stack (XRT shim of the amdxdna driver): a buffer of 2^31 bytes or more does NOT allocate ("Failed to
// allocate xrt::ext::bo: mmap_range(len=-2013265920) failed (err=-12)": the length is a signed 32-bit int), one just
// under works and the GEMM into it is correct up to its last row (30,720 rows of d3-turbo's: 2.139 GB). d3_mlp_block
// therefore runs such a pass's up/gate GEMM as two column parts (D3_BO_LIMIT_MB / d3_upgate_pairs), which the
// engine run on a >= 31k-token prompt exercises. This tool runs gen_dequant_mm_512 for the shape on a synthetic
// q4_k weight in the device layout (zero-mean blocks with finite scales) and compares rows at the start, in the
// middle and at the end of the result with the host kernel (d3_cpu::dequant_mm). A result of 2^31 bytes or more
// aborts at the allocation: that is the finding, not a test failure of the engine.
//
//   d3_test_mm_big <mm.xclbin> [rows=30720] [K=5120] [N=34816]
//
// Exit code 0 = every checked row matches (cosine >= 0.999). Needs the NPU.
#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "d3_prefill.hpp"

static double ms_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s mm.xclbin [rows=30720] [K] [N]\n", argv[0]); return 2; }
    const int R = argc > 2 ? std::atoi(argv[2]) : 30720, K = argc > 3 ? std::atoi(argv[3]) : 5120, N = argc > 4 ? std::atoi(argv[4]) : 34816;
    if (R % 512 || K % 256 || N % 256) { std::fprintf(stderr, "rows %% 512, K %% 256, N %% 256 required\n"); return 2; }
    flm_rt::device device(0);
    npu_xclbin_manager npu(npu_device::device_npu2, &device, false);
    npu_app_manager* mgr = npu.register_xclbin(argv[1]);

    std::mt19937 rng(5);
    // weight: N/32 row blocks x K/256 column blocks of the device layout (d3_cpu::dequant_row_block's)
    const size_t bb = d3_cpu::block_bytes(flm_q4k);
    const size_t nblocks = (size_t)(N / 32) * (K / 256);
    buffer<u8> w = npu.create_bo_buffer<u8>(nblocks * bb);
    {
        std::uniform_int_distribution<int> byte(0, 255), small(0, 15);
        std::uniform_real_distribution<float> sd(0.002f, 0.02f);
        for (size_t b = 0; b < nblocks; b++) {
            u8* blk = w.data() + b * bb;
            // zero-mean weights like a trained layer's: w = S * s8 * (q - 7.5) = S * s8 * q + negM * m8 with m8 = 7.5 * s8
            for (int i = 0; i < 256; i++) blk[i] = (u8)(1 + small(rng) % 15);      // s8 in 1..15
            for (int i = 0; i < 256; i++) blk[256 + i] = (u8)((15 * blk[i] + 1) / 2);   // m8 ~ 7.5 * s8
            for (int i = 0; i < 4096; i++) blk[512 + i] = (u8)byte(rng);           // qs: two 4-bit values per byte
            bf16* S = reinterpret_cast<bf16*>(blk + 4608);
            for (int i = 0; i < 32; i++) { const float v = sd(rng); S[i] = bf16(v); S[32 + i] = bf16(-v); }   // S, negM
        }
        w.sync_to_device();
    }
    buffer<bf16> x = npu.create_bo_buffer<bf16>((size_t)R * K);
    {
        std::normal_distribution<float> nd(0.0f, 1.0f);
        for (size_t i = 0; i < x.size(); i++) x.data()[i] = bf16(nd(rng));
        x.sync_to_device();
    }
    buffer<bf16> y = npu.create_bo_buffer<bf16>((size_t)R * N);
    std::printf("y = %.3f GB (%zu bytes), last row starts at byte %zu\n", y.size() * 2 / 1e9, y.size() * 2, (size_t)(R - 1) * N * 2);
    memset(y.data(), 0, y.size() * sizeof(bf16));
    y.sync_to_device();

    d3_desc desc;   // the sequence generator needs none for the GEMM
    d3_npu_sequence seq_gen(desc);
    d3_mm mm;
    mm.build(mgr);
    mm.select(R);
    auto t0 = std::chrono::steady_clock::now();
    seq_gen.gen_dequant_mm_512(mm.seq(), R, K, N, 0, 0, 0);
    std::printf("sequence generated in %.0f ms\n", ms_since(t0));
    t0 = std::chrono::steady_clock::now();
    mm(y, x, w);
    std::printf("array GEMM %.0f ms\n", ms_since(t0));
    y.sync_from_device();

    // rows to check: first, last, and the first row whose end passes 2^31 bytes
    const long long row_2g = (long long)((1ull << 31) / ((size_t)N * 2));
    std::vector<int> rows = {0, 1, 511, 512, (int)row_2g - 1, (int)row_2g, (int)row_2g + 1, R / 2, R - 513, R - 512, R - 2, R - 1};
    int bad = 0;
    for (int r : rows) {
        if (r < 0 || r >= R) continue;
        std::vector<bf16> ref(N);
        d3_cpu::dequant_mm(ref.data(), N, w.data(), x.data() + (size_t)r * K, K, 1, K, N, d3_cpu::BODY_LAYOUT);
        double ab = 0, aa = 0, bb2 = 0;
        for (int n = 0; n < N; n++) {
            const double a = (float)y.data()[(size_t)r * N + n], b = (float)ref[n];
            ab += a * b; aa += a * a; bb2 += b * b;
        }
        const double c = ab / (std::sqrt(aa) * std::sqrt(bb2) + 1e-30);
        const bool ok = c >= 0.999;
        std::printf("row %6d (byte offset %.3f GB): cosine %.6f %s\n", r, (double)r * N * 2 / 1e9, c, ok ? "ok" : "FAIL");
        if (!ok) bad++;
    }
    std::printf("%s\n", bad ? "FAILED" : "ALL PASSED");
    return bad ? 1 : 0;
}
