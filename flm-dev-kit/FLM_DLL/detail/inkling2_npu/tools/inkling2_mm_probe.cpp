// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file inkling2_mm_probe.cpp
/// \brief check the prefill matmul layout (inkling2_npu_mm.hpp) on real weights against an fp32 reference
///
/// Usage: inkling2_mm_probe model.q4nx mm_rtm2_glu.xclbin
///   1. host read bandwidth of a device buffer vs a heap buffer (one copy of the weights for host and NPU?)
///   2. layer 0 q_proj (2304 -> 3072, Din 2304) at 1024 and 300 rows
///   3. layer 0 experts 5 and 127 of gate_up (2304 -> 2304) through a per-call offset
///   4. layer 0 expert 9 of down (1280 -> 2304), when down is Q4_K
///   With a third argument "sweep": time against rows (best of 10, dispatch overhead included), with the
///   weight bytes streamed per ms and the achieved TOPS.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "../inkling2_host.hpp"
#include "../inkling2_bfp16.hpp"
#include "../inkling2_npu_mm.hpp"
#include "typedef.hpp"

using namespace inkling2_host;
namespace mm = inkling2_npu_mm;
using clk = std::chrono::steady_clock;

static double ms_since(clk::time_point t) { return std::chrono::duration<double, std::milli>(clk::now() - t).count(); }

/// \brief block (32 rows x 256 columns) into w[r][c0 + c], row stride K
static void dequant_q4k(float* w, int K, const u8* blk) {
    const u8* s8 = blk;
    const u8* m8 = blk + 256;
    const u8* qs = blk + 512;
    const u16* S = reinterpret_cast<const u16*>(blk + 4608);
    const u16* M = reinterpret_cast<const u16*>(blk + 4672);
    auto bf = [](u16 v) { uint32_t u = (uint32_t)v << 16; float f; memcpy(&f, &u, 4); return f; };
    for (int c = 0; c < 256; c++) {
        const int g = c / 32;
        for (int r = 0; r < 32; r++) {
            const int q = (qs[c * 16 + r / 2] >> ((r % 2) * 4)) & 15;
            w[(size_t)r * K + c] = q * bf(S[r]) * s8[g * 32 + r] + bf(M[r]) * m8[g * 32 + r];
        }
    }
}

/// \brief rows [r0, r0 + n) of a [row block][column block] matrix as fp32 [n][K]
static std::vector<float> dequant(const qmat& w, int r0, int n) {
    std::vector<float> out((size_t)n * w.K);
    for (int rb = 0; rb < n / QROWS; rb++)
        for (int cb = 0; cb < w.col_blocks(); cb++)
            dequant_q4k(out.data() + (size_t)rb * QROWS * w.K + cb * QCOLS, w.K, w.block(r0 / QROWS + rb, cb));
    return out;
}

/// \brief the whole matrix into the grouped layout at dst (row blocks a multiple of NPU_GROUP)
static void group(u8* dst, const qmat& w) {
    const int nc = w.col_blocks();
    for (int p = 0; p < w.row_blocks() / NPU_GROUP; p++)
        for (int cb = 0; cb < nc; cb++)
            for (int i = 0; i < NPU_GROUP; i++)
                q4k_to_npu(dst + (((size_t)p * nc + cb) * NPU_GROUP + i) * Q4K_BYTES, w.block(NPU_GROUP * p + i, cb));
}

struct check {
    double rel = 0, maxabs = 0;
};

/// \brief y (bf16, row stride ldy) against x W^T (fp32) for rows x N
/// \brief rows of x (row stride ld, Kw columns) as one job's bfp16 x at dst
static void pack_x(u8* dst, const float* x, int ld, int rows, int Kw, int Din, const mm::geometry& g) {
    std::vector<const float*> src(g.covered, nullptr);
    for (int t = 0; t < rows; t++) src[t] = x + (size_t)t * ld;
    for (uint32_t R = 0; R < g.covered; R += 16)
        inkling2_bfp16::pack_group(dst + inkling2_bfp16::bytes(R / g.m * g.m, Din), g.m, (R % g.m) / 16, src.data() + R, Kw, Din);
}

static check compare(const buffer<bf16>& y, int ldy, const std::vector<float>& x, int ldx, const std::vector<float>& w, int K,
                     int rows, int N) {
    double num = 0, den = 0, mx = 0;
    #pragma omp parallel for reduction(+ : num, den) reduction(max : mx) schedule(dynamic)
    for (int t = 0; t < rows; t++)
        for (int n = 0; n < N; n++) {
            double ref = 0;
            for (int k = 0; k < K; k++) ref += (double)x[(size_t)t * ldx + k] * w[(size_t)n * K + k];
            const double d = (double)(float)y[(size_t)t * ldy + n] - ref;
            num += d * d;
            den += ref * ref;
            mx = std::max(mx, std::fabs(d));
        }
    return {std::sqrt(num / den), mx};
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s model.q4nx mm_rtm2_glu.xclbin\n", argv[0]);
        return 2;
    }
    model m;
    m.load(argv[1]);
    const layer_weights& L = m.w().layer[0];

    xrt::device dev(0);
    npu_xclbin_manager mgr(npu_device::device_npu2, &dev, false);
    npu_app_manager* mgr_mm = mgr.register_xclbin(argv[2]);

    // ---- 1. host read bandwidth
    {
        const size_t n = (size_t)1 << 30;
        buffer<u8> bo = mgr.create_bo_buffer<u8>(n);
        std::vector<u8> heap(n);
        for (size_t i = 0; i < n; i++) bo[i] = heap[i] = (u8)i;
        bo.sync_to_device();
        for (int rep = 0; rep < 3; rep++)
            for (int which = 0; which < 2; which++) {
                const uint64_t* p = reinterpret_cast<const uint64_t*>(which ? heap.data() : bo.data());
                auto t = clk::now();
                uint64_t s = 0;
                #pragma omp parallel for reduction(+ : s) schedule(static)
                for (size_t i = 0; i < n / 8; i++) s += p[i];
                const double ms = ms_since(t);
                printf("read %s: %.1f GB/s (sum %llu)\n", which ? "heap  " : "device", n / ms / 1e6, (unsigned long long)s);
            }
    }

    std::mt19937 rng(1);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    auto run_case = [&](const char* name, const qmat& w, int r0, int N, int Npad, int Din, int rows, int iters) {
        const int Kw = w.K;
        const size_t whole = (size_t)w.N * Kw;
        buffer<u8> wb = mgr.create_bo_buffer<u8>(mm::q4k_bytes(whole) + mm::tail_bytes(0, Npad, Din, Kw));
        memset(wb.data(), 0, wb.size());
        group(wb.data(), w);
        wb.sync_to_device();
        const mm::geometry g = mm::pick_geometry(rows, Din);
        buffer<u8> xb = mgr.create_bo_buffer<u8>(inkling2_bfp16::bytes(g.covered, Din));
        buffer<bf16> yb = mgr.create_bo_buffer<bf16>((size_t)g.covered * Npad);
        std::vector<float> x((size_t)rows * Din, 0.0f);
        for (int t = 0; t < rows; t++)
            for (int k = 0; k < Kw; k++) x[(size_t)t * Din + k] = (float)bf16(dist(rng));
        pack_x(xb.data(), x.data(), Din, rows, Kw, Din, g);
        xb.sync_to_device();
        npu_app app = mgr_mm->create_app();
        mm::gen(app.seq(), rows, Din, Kw, Npad);
        const size_t off = mm::q4k_bytes((size_t)r0 * Kw);
        double best = 1e9;
        for (int it = 0; it < iters; it++) {
            auto t = clk::now();
            auto run = app.create_run(yb, xb, bound_arg{&wb, off});
            run.start();
            run.wait();
            best = std::min(best, ms_since(t));
        }
        yb.sync_from_device();
        const std::vector<float> wf = dequant(w, r0, N);
        const check c = compare(yb, Npad, x, Din, wf, Kw, rows, N);
        const double wgb = (double)mm::q4k_bytes((size_t)Npad * Kw) * g.blocks / (best * 1e6);
        const double tops = 2.0 * rows * N * Kw / (best * 1e9);
        printf("%-22s rows %4d (covered %4d) Din %d Kw %d N %d/%d: rel err %.3e, max abs %.3e, %.3f ms, w %.1f GB/s, %.2f TOPS\n",
               name, rows, g.covered, Din, Kw, N, Npad, c.rel, c.maxabs, best, wgb, tops);
    };
    // all NE experts of one matrix in one dispatch, rows each (the engine's batched expert call): weights at their
    // layer offsets, x / y one covered block per expert; expert 3 checked against the host
    auto run_batch = [&](const char* name, const qmat& w, int N, int Npad, int Din, int rows, int iters) {
        const int Kw = w.K;
        const size_t per = mm::q4k_bytes((size_t)N * Kw);
        buffer<u8> wb = mgr.create_bo_buffer<u8>(per * NE + mm::tail_bytes(N, Npad, Din, Kw));
        memset(wb.data(), 0, wb.size());
        group(wb.data(), w);
        wb.sync_to_device();
        const mm::geometry g = mm::pick_geometry(rows, Din);
        const size_t xbytes = inkling2_bfp16::bytes(g.covered, Din);
        buffer<u8> xb = mgr.create_bo_buffer<u8>(xbytes * NE);
        buffer<bf16> yb = mgr.create_bo_buffer<bf16>((size_t)g.covered * Npad * NE);
        std::vector<float> x((size_t)rows * Din, 0.0f), xe((size_t)rows * Din, 0.0f);
        for (int e = 0; e < NE; e++) {
            for (int t = 0; t < rows; t++)
                for (int k = 0; k < Kw; k++) xe[(size_t)t * Din + k] = (float)bf16(dist(rng));
            pack_x(xb.data() + e * xbytes, xe.data(), Din, rows, Kw, Din, g);
            if (e == 3) x = xe;
        }
        xb.sync_to_device();
        std::vector<mm::job> jobs;
        for (int e = 0; e < NE; e++)
            jobs.push_back({(uint32_t)rows, (uint32_t)(e * xbytes), (uint32_t)(e * g.covered * Npad), e * per});
        npu_app app = mgr_mm->create_app();
        mm::gen_batch(app.seq(), Din, Kw, Npad, jobs);
        double best = 1e9;
        for (int it = 0; it < iters; it++) {
            auto t = clk::now();
            auto run = app.create_run(yb, xb, wb);
            run.start();
            run.wait();
            best = std::min(best, ms_since(t));
        }
        yb.sync_from_device();
        const std::vector<float> wf = dequant(w, 3 * N, N);
        buffer<bf16> y3 = mgr.create_bo_buffer<bf16>((size_t)rows * Npad);
        memcpy(y3.data(), yb.data() + (size_t)3 * g.covered * Npad, (size_t)rows * Npad * sizeof(bf16));
        const check c = compare(y3, Npad, x, Din, wf, Kw, rows, N);
        const double tops = 2.0 * rows * N * Kw * NE / (best * 1e9);
        printf("%-10s batch %d experts x %4d rows (covered %4d) Din %d Kw %d N %d/%d: rel err %.3e, %.3f ms, %.1f us/expert, %.2f TOPS\n",
               name, NE, rows, g.covered, Din, Kw, N, Npad, c.rel, best, best * 1e3 / NE, tops);
    };
    if (argc > 3 && !strcmp(argv[3], "batch")) {
        for (int rows : {64, 128, 256}) {
            run_batch("gate_up", L.gate_up, 2 * FF, 2304, 2304, rows, 5);
            if (L.down.fmt == qfmt::q4k) run_batch("down", L.down, D, 2304, 1280, rows, 5);
        }
        return 0;
    }
    // host cost of one expert group's dispatch (32 experts x 64 rows): sequence, ELF + kernel, run object
    if (argc > 3 && !strcmp(argv[3], "genbench")) {
        npu_app app = mgr_mm->create_app();
        buffer<u8> wb = mgr.create_bo_buffer<u8>(4096), xb = mgr.create_bo_buffer<u8>(4096);
        buffer<bf16> yb = mgr.create_bo_buffer<bf16>(4096);
        double tg = 0, tr = 0;
        const int n = 20;
        for (int it = 0; it < n; it++) {
            std::vector<mm::job> jobs;
            for (int e = 0; e < 32; e++) jobs.push_back({(uint32_t)(48 + (e * 7 + it) % 40), 0, 0, 0});
            auto t = clk::now();
            mm::gen_batch(app.seq(), 2304, 2304, 2304, jobs);
            tg += ms_since(t);
            t = clk::now();
            auto run = app.create_run(yb, xb, wb);
            tr += ms_since(t);
        }
        printf("gen_batch %.3f ms, ELF + kernel + run %.3f ms, %zu instruction words\n", tg / n, tr / n,
               app.seq()->dump().second);
        return 0;
    }
    // one q_proj case: one <rows> <Npad> (Npad / 768 passes)
    if (argc > 5 && !strcmp(argv[3], "one")) {
        const int Npad = atoi(argv[5]);
        run_case("q_proj", L.q, 0, Npad, Npad, 2304, atoi(argv[4]), 3);
        return 0;
    }
    if (argc > 3 && !strcmp(argv[3], "sweep")) {
        for (int rows : {16, 32, 64, 128, 256, 512, 1024, 2048}) run_case("gate_up expert 5", L.gate_up, 5 * 2 * FF, 2 * FF, 2304, 2304, rows, 10);
        for (int rows : {512, 1024, 2048}) run_case("q_proj", L.q, 0, QD, QD, 2304, rows, 10);
        return 0;
    }

    run_case("q_proj", L.q, 0, QD, QD, 2304, 1024, 10);
    run_case("q_proj", L.q, 0, QD, QD, 2304, 300, 10);
    run_case("gate_up expert 5", L.gate_up, 5 * 2 * FF, 2 * FF, 2304, 2304, 77, 10);
    run_case("gate_up expert 127", L.gate_up, 127 * 2 * FF, 2 * FF, 2304, 2304, 520, 10);
    if (L.down.fmt == qfmt::q4k) run_case("down expert 9", L.down, 9 * D, D, 2304, 1280, 64, 10);
    else printf("down is not Q4_K in this model: skipped\n");
    return 0;
}
