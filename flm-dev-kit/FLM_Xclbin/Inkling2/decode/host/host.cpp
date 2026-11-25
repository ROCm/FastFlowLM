// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.
//
// Stand-alone test of the decode overlay's stream cores (kernel/decode_core.cc).
//   run.exe sink [channels] [units] [tasks] [iters]   shim -> core stream bandwidth
//   run.exe gemv [channels] [groups] [nc] [tasks] [iters]   Q4_K GEMV vs an fp32 reference, and its speed
// Channel c is shim MM2S (c / 8) of column c % 8, feeding core (c % 8, 2 + c / 8).

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "typedef.hpp"
#include "npu_utils/npu_utils.hpp"

static constexpr int UNIT = 4736, X_OFF = 128, Y_UNIT = 96;
static constexpr npu_tiles IT[8] = {IT0, IT1, IT2, IT3, IT4, IT5, IT6, IT7};

static float bf2f(uint16_t v) { uint32_t u = (uint32_t)v << 16; float f; memcpy(&f, &u, 4); return f; }
static uint16_t f2bf(float f) { uint32_t u; memcpy(&u, &f, 4); u += 0x7FFF + ((u >> 16) & 1); return (uint16_t)(u >> 16); }

// w[row][col] of one Q4_K block in q4k_to_npu order: w = q S sc + M mn
static void ref_block(float* w, const uint8_t* b) {
    for (int H = 0; H < 2; H++)
        for (int par = 0; par < 2; par++) {
            const int hp = 2 * H + par;
            for (int r = 0; r < 8; r++) {
                const int row = 16 * H + 2 * r + par;
                uint16_t Sv, Mv;
                memcpy(&Sv, b + 4608 + hp * 16 + 2 * r, 2);
                memcpy(&Mv, b + 4672 + hp * 16 + 2 * r, 2);
                const float S = bf2f(Sv), M = bf2f(Mv);
                for (int kt = 0; kt < 32; kt++)
                    for (int k = 0; k < 8; k++) {
                        const int g = kt / 4, i = r * 8 + k;
                        const uint8_t byte = b[512 + 2048 * H + 1024 * par + 32 * kt + i / 2];
                        const int q = (i & 1) ? byte >> 4 : byte & 15;
                        w[row * 256 + 8 * kt + k] = q * (S * b[hp * 64 + g * 8 + r]) + M * b[256 + hp * 64 + g * 8 + r];
                    }
            }
        }
}

// hub core (kernel/hub.cc, iron/decode_aie.py)
static const npu_tiles HUB = get_tile(4, 0);
static constexpr uint32_t HUB_ARGS = 0x8200, HUB_START_LOCK = 0x1F000 + 16 * 10, SHIM_BD = 4, BASE_BD = 15;

// The hub picks the streams: every channel holds two sink tasks (A, B: different bytes); the hub points the 16
// channels at A or B (its extra offset) and pushes them. run.exe route [units] [iters]
static int route(int n, int iters, int dbg, int R) {
    const int C = getenv("DEC_ROWS") && atoi(getenv("DEC_ROWS")) == 1 ? 8 : 16;   // 8: the debug build, hub trace on
    const bool tr_on = C == 8;
    const int trw = getenv("TRW") ? atoi(getenv("TRW")) : 32;   // trace words; with TRW set, wait on the trace only
    const size_t task = (size_t)(1 + n) * UNIT, stride = 2 * task;
    xrt::device dev(0);
    npu_xclbin_manager mgr(npu_device::device_npu2, &dev, false);
    npu_app_manager* am = mgr.register_xclbin("build/xclbins/decode.xclbin");
    npu_app apps[2] = {am->create_app(), am->create_app()};
    buffer<uint8_t> in = apps[0].create_bo_buffer<uint8_t>(16 * stride);
    buffer<float> out = apps[0].create_bo_buffer<float>(16 * R * Y_UNIT);
    buffer<uint32_t> tr = apps[0].create_bo_buffer<uint32_t>(64);
    // ABS=1: the right data sits in another BO that is no run argument; the hub gets its device address
    const bool abs_mode = getenv("ABS") && atoi(getenv("ABS"));
    // CROSS=1: other is a 2040 MB BO (XRT sizes are int32) whose tasks sit across a 4 GB boundary of the device address
    const bool cross = getenv("CROSS") != nullptr;
    std::vector<buffer<uint8_t>> spare;
    buffer<uint8_t> other = apps[0].create_bo_buffer<uint8_t>(cross ? (size_t)2040 << 20 : 16 * stride);
    size_t shift = 0;
    for (int t = 0; cross; t++) {
        const uint64_t dev = other.bo().address() + 0x80000000ull, next = (dev + 0xFFFFFFFFull) & ~0xFFFFFFFFull;
        if (next - dev > 8 * stride && next - dev + 8 * stride < ((size_t)2040 << 20)) {
            shift = next - dev - 8 * stride + 4096;
            std::printf("other dev 0x%llx, shift %zu, boundary in task %zu\n", (unsigned long long)dev, shift, (size_t)(next - dev - shift) / stride);
            break;
        }
        if (t == 4) { std::printf("no crossing BO\n"); return 3; }
        spare.push_back(other);
        other = apps[0].create_bo_buffer<uint8_t>((size_t)2040 << 20);
    }
    for (int c = 0; c < 16; c++)
        for (int v = 0; v < 2; v++) {
            uint8_t* p = in.data() + c * stride + v * task;
            memset(p, 0, UNIT);
            ((int*)p)[0] = 1; ((int*)p)[1] = n;
            for (int i = 1; i <= n; i++) p[(size_t)i * UNIT] = (uint8_t)(i * (v + 1) + c);
        }
    const bool abs_in = getenv("ABSIN") != nullptr;   // ABS=1 ABSIN=1: absolute address of the run-argument BO itself
    if (abs_mode && !abs_in) {
        memcpy(other.data() + shift, in.data(), 16 * stride);
        memset(in.data(), 0, 16 * stride);
        other.sync_to_device();   // whole BO: slow for CROSS, fine for a probe
    }
    in.sync_to_device();
    const uint64_t oa = (abs_in ? in.bo().address() : other.bo().address()) + shift + (getenv("AOFF") ? strtoull(getenv("AOFF"), nullptr, 0) : 0);
    std::printf("in BO at 0x%llx, other at 0x%llx\n", (unsigned long long)in.bo().address(), (unsigned long long)other.bo().address());
    if (abs_mode) std::printf("other BO at 0x%llx\n", (unsigned long long)oa);
    for (int v = 0; v < 2; v++) {
        npu_sequence* seq = apps[v].seq();
        seq->clear_cmds();
        if (tr_on) seq->npu_dma_memcpy_nd(4, 2, S2MM, IT0, npu_bd_id(6), it_channel_1, {0, 0, 0, 0}, {1, 1, 1, (uint32_t)trw}, {0, 0, 0, 1});
        // the hub reads the buffer address from this BD once (never started)
        seq->npu_dma_memcpy_nd(1, 0, MM2S, IT0, npu_bd_id(BASE_BD), it_channel_0, {0, 0, 0, 0}, {1, 1, 1, 1}, {0, 0, 0, 1}, -1, 0, false, normal_cache, 15, false);
        for (int c = 0; c < C; c++) {
            const int col = c % 8, r = c / 8;
            const npu_it_channel ch = r ? it_channel_1 : it_channel_0;
            // the BD the hub patches: written here (address = the buffer), not started; in mode 0 it points at the
            // other variant, so only the hub's patch gives the right sums
            seq->npu_dma_memcpy_nd(1, 0, MM2S, IT[col], npu_bd_id(SHIM_BD + r), ch, {0, 0, 0, (uint32_t)(c * stride + (dbg ? v : 1 - v) * task)}, {1, 1, 1, (uint32_t)task}, {0, 0, 0, 1},
                                   -1, 0, false, normal_cache, 15, false);
            if (getenv("SEQABS")) {   // the sequence itself points the BD at an absolute address (raw register writes) and pushes it
                const uint64_t a = oa + c * stride + v * task;
                const uint32_t bda = 0x1D000u + (SHIM_BD + r) * 0x20u;
                seq->rtp_write(IT[col], bda + 4, (uint32_t)a);
                seq->npu_maskwrite(IT[col], bda + 8, (uint32_t)(a >> 32) & 0xFFFFu, 0xFFFFu);
                seq->rtp_write(IT[col], r ? 0x1D21Cu : 0x1D214u, SHIM_BD + r);
            }
            seq->npu_dma_memcpy_nd(4, 1, S2MM, IT[col], npu_bd_id(2 + r), ch, {0, 0, 0, (uint32_t)(c * R * Y_UNIT)}, {1, 1, (uint32_t)R, (uint32_t)Y_UNIT}, {0, 0, (uint32_t)Y_UNIT, 1});
        }
        seq->npu_maskwrite(HUB, 0x32038, 1, 1);
        seq->rtp_write(HUB, HUB_ARGS + 0, (uint32_t)task);
        seq->rtp_write(HUB, HUB_ARGS + 4, (uint32_t)stride);
        seq->rtp_write(HUB, HUB_ARGS + 8, (uint32_t)(v * task));
        seq->rtp_write(HUB, HUB_ARGS + 12, (uint32_t)dbg);
        seq->rtp_write(HUB, HUB_ARGS + 16, (uint32_t)tr_on);
        seq->rtp_write(HUB, HUB_ARGS + 20, getenv("SEQABS") ? 0u : (uint32_t)C);
        seq->rtp_write(HUB, HUB_ARGS + 24, (uint32_t)R);
        seq->rtp_write(HUB, HUB_ARGS + 28, (uint32_t)abs_mode);
        seq->rtp_write(HUB, HUB_ARGS + 32, (uint32_t)oa);
        seq->rtp_write(HUB, HUB_ARGS + 36, (uint32_t)(oa >> 32));
        seq->rtp_write(HUB, HUB_START_LOCK, 1);
        if (getenv("TRW")) seq->npu_dma_wait(IT0, S2MM, it_channel_1);
        else for (int c = 0; c < C; c++) seq->npu_dma_wait(IT[c % 8], S2MM, c / 8 ? it_channel_1 : it_channel_0);
        seq->cmds2seq();
    }
    bool ok = true;
    double us[2] = {};
    for (int it = 0; it < iters; it++)
        for (int v = 0; v < 2; v++) {
            for (int i = 0; i < 16 * R * Y_UNIT; i++) out.data()[i] = -1.0f;
            out.sync_to_device();
            const auto t0 = std::chrono::steady_clock::now();
            if (tr_on) {
                for (int i = 0; i < 64; i++) tr.data()[i] = 0xFFFFFFFF;
                tr.sync_to_device();
                xrt::run run = apps[v].create_run(in, out, tr);
                run.start();
                ert_cmd_state st = ERT_CMD_STATE_TIMEOUT;
                try { st = run.wait(std::chrono::milliseconds(3000)); } catch (const std::exception& e) { std::printf("wait: %s\n", e.what()); }
                tr.sync_from_device();
                std::printf("run state %d, trace:", (int)st);
                for (int i = 0; i < 32 && tr.data()[i] != 0xFFFFFFFF; i++) std::printf(" %x", tr.data()[i]);
                std::printf("\n");
                if (st != ERT_CMD_STATE_COMPLETED) return 2;
            } else if (getenv("ABSARG")) {
                apps[v](in, out, other);
            } else {
                apps[v](in, out);
            }
            us[v] += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
            out.sync_from_device();
            for (int c = 0; c < C; c++) {
                unsigned sum = 0;
                for (int i = 1; i <= n; i++) sum += (uint8_t)(i * (v + 1) + c);
                for (int q = 0; q < R; q++) {
                const float* y = out.data() + (c * R + q) * Y_UNIT;
                if (y[0] != n || y[1] != (float)sum) {
                    if (ok || getenv("ALLF")) std::printf("iter %d variant %d ch %d: %g %g (want %d %u)\n", it, v, c, y[0], y[1], n, sum);
                    ok = false;
                }
                }
            }
        }
    const double bytes = (double)C * R * task;
    std::printf("route n=%d: A %.1f us, B %.1f us per run (%.2f GB/s), check %s\n", n, us[0] / iters, us[1] / iters,
                bytes / (us[1] / iters) / 1e3, ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}

// A decode MoE call as the engine runs it: E of NX experts (each 24 groups of 96 rows x nc column blocks, the grouped
// layout) spread over the 16 channels in (expert, group) order. A channel's work is header tasks (x, from the io
// buffer) and weight tasks; arg 0 = io (headers, then y), args 1-4 = sub-buffers of the weight buffer at the chosen
// experts (per call), so the instruction sequence is fixed per E. shared_x: one x for all experts (gate_up), else
// one x per expert (down: h). run.exe experts [E] [nc] [shared_x] [iters]
static int experts(int E, int nc, bool shared_x, int iters) {
    constexpr int NX = 8, GPE = 24, C = 16;
    const size_t eb = (size_t)GPE * nc * 3 * UNIT;   // bytes per expert
    xrt::device dev(0);
    npu_xclbin_manager mgr(npu_device::device_npu2, &dev, false);
    npu_app_manager* am = mgr.register_xclbin("build/xclbins/decode.xclbin");
    npu_app app = am->create_app();
    buffer<uint8_t> wt = app.create_bo_buffer<uint8_t>(NX * eb);
    std::mt19937 rng(7);
    for (size_t b = 0; b < NX * eb / UNIT; b++) {
        uint8_t* blk = wt.data() + b * UNIT;
        for (int i = 0; i < UNIT; i++) blk[i] = (uint8_t)rng();
        for (int i = 0; i < 256; i++) { blk[i] &= 63; blk[256 + i] &= 63; }
        for (int i = 0; i < 32; i++) {
            uint16_t s = f2bf(0.0005f * (1 + (rng() % 1000) / 1000.0f)), m = f2bf(-0.004f * (rng() % 1000) / 1000.0f);
            memcpy(blk + 4608 + 2 * i, &s, 2);
            memcpy(blk + 4672 + 2 * i, &m, 2);
        }
    }
    wt.sync_to_device();
    // the split: channel c takes groups [c * total / C, (c + 1) * total / C) of the (expert slot, group) order
    struct seg { int slot, g0, n; };
    const int total = E * GPE;
    std::vector<std::vector<seg>> segs(C);
    for (int c = 0; c < C; c++) {
        int g = c * total / C;
        const int end = (c + 1) * total / C;
        while (g < end) {
            const int s = g / GPE, n = std::min(end, (s + 1) * GPE) - g;
            segs[c].push_back({s, g % GPE, n});
            g += n;
        }
    }
    // io: header units [C][2], then y [C][groups][96]
    const size_t hdr0 = 0, y0 = (size_t)C * 2 * UNIT;
    std::vector<size_t> ybase(C);
    size_t yo = y0;
    for (int c = 0; c < C; c++) {
        ybase[c] = yo;
        for (const seg& s : segs[c]) yo += (size_t)s.n * Y_UNIT * 4;
    }
    buffer<uint8_t> io = app.create_bo_buffer<uint8_t>(yo);
    npu_sequence* seq = app.seq();
    seq->clear_cmds();
    for (int c = 0; c < C; c++) {
        const int col = c % 8, r = c / 8;
        const npu_it_channel ch = r ? it_channel_1 : it_channel_0;
        int ny = 0;
        for (const seg& s : segs[c]) ny += s.n;
        seq->npu_dma_memcpy_nd(4, 0, S2MM, IT[col], npu_bd_id(14 + r), ch, {0, 0, 0, (uint32_t)(ybase[c] / 4)}, {1, 1, 1, (uint32_t)(ny * Y_UNIT)}, {0, 0, 0, 1});
        int bd = 6 * r;
        for (size_t i = 0; i < segs[c].size(); i++) {
            const seg& s = segs[c][i];
            if (i == 0 || !shared_x)
                seq->npu_dma_memcpy_nd(1, 0, MM2S, IT[col], npu_bd_id(bd++), ch, {0, 0, 0, (uint32_t)(hdr0 + (size_t)(c * 2 + i) * UNIT)}, {1, 1, 1, (uint32_t)UNIT}, {0, 0, 0, 1});
            seq->npu_dma_memcpy_nd(1, 1 + s.slot, MM2S, IT[col], npu_bd_id(bd++), ch, {0, 0, 0, (uint32_t)((size_t)s.g0 * nc * 3 * UNIT)}, {1, 1, 1, (uint32_t)((size_t)s.n * nc * 3 * UNIT)}, {0, 0, 0, 1});
        }
    }
    for (int c = 0; c < C; c++) seq->npu_dma_wait(IT[c % 8], S2MM, c / 8 ? it_channel_1 : it_channel_0);
    seq->cmds2seq();

    std::vector<int> chosen = {5, 2, 7, 0};
    std::vector<std::vector<uint16_t>> xs(E, std::vector<uint16_t>(nc * 256));
    for (int e = 0; e < E; e++)
        for (auto& v : xs[e]) v = f2bf(std::normal_distribution<float>(0, 1)(rng));
    if (shared_x) for (int e = 1; e < E; e++) xs[e] = xs[0];
    auto fill_headers = [&] {
        for (int c = 0; c < C; c++) {
            int all = 0;
            for (const seg& s : segs[c]) all += s.n;
            for (size_t i = 0; i < segs[c].size(); i++) {
                if (i > 0 && shared_x) break;
                uint8_t* h = io.data() + hdr0 + (size_t)(c * 2 + i) * UNIT;
                ((int*)h)[0] = 2; ((int*)h)[1] = shared_x ? all : segs[c][i].n; ((int*)h)[2] = nc;
                memcpy(h + X_OFF, xs[segs[c][i].slot].data(), nc * 512);
            }
        }
    };
    xrt::bo& wbo = wt.bo();
    double us = 0, part[7] = {};
    for (int it = 0; it <= iters; it++) {
        for (size_t i = y0; i < io.size(); i += 4) *(float*)(io.data() + i) = -1.0f;
        auto t0 = std::chrono::steady_clock::now();
        auto lap = [&](int k) { const auto t1 = std::chrono::steady_clock::now(); if (it > 0) part[k] += std::chrono::duration<double, std::micro>(t1 - t0).count(); t0 = t1; };
        fill_headers();
        lap(0);
        io.sync_to_device();
        lap(1);
        xrt::run run = app.create_run(io, wt, wt, wt, wt);
        lap(2);
        for (int s = 0; s < 4; s++) {
            const size_t off = (size_t)chosen[std::min(s, E - 1)] * eb;
            if (getenv("NOSUB")) continue;
            run.set_arg(4 + s, xrt::bo(wbo, eb, off));
        }
        lap(3);
        run.start();
        lap(4);
        run.wait();
        lap(5);
        io.sync_from_device();
        lap(6);
    }
    for (int k = 0; k < 7; k++) us += part[k];
    std::printf("parts us: hdr %.1f sync %.1f create %.1f subbo %.1f start %.1f wait %.1f syncback %.1f\n", part[0] / iters, part[1] / iters,
                part[2] / iters, part[3] / iters, part[4] / iters, part[5] / iters, part[6] / iters);
    // check
    double err2 = 0, ref2 = 0;
    std::vector<float> wb(32 * 256);
    for (int c = 0; c < C; c++) {
        const float* y = (const float*)(io.data() + ybase[c]);
        for (const seg& s : segs[c])
            for (int g = 0; g < s.n; g++, y += Y_UNIT) {
                double ref[96] = {};
                for (int cb = 0; cb < nc; cb++)
                    for (int b = 0; b < 3; b++) {
                        ref_block(wb.data(), wt.data() + chosen[s.slot] * eb + ((size_t)((s.g0 + g) * nc + cb) * 3 + b) * UNIT);
                        for (int row = 0; row < 32; row++)
                            for (int col = 0; col < 256; col++) ref[32 * b + row] += (double)wb[row * 256 + col] * bf2f(xs[s.slot][cb * 256 + col]);
                    }
                for (int i = 0; i < 96; i++) { const double e = y[i] - ref[i]; err2 += e * e; ref2 += ref[i] * ref[i]; }
            }
    }
    const double bytes = (double)E * eb;
    std::printf("experts E=%d nc=%d shared_x=%d: %.1f us/call (%.2f GB/s), rel rms err %.3e\n", E, nc, (int)shared_x, us / iters,
                bytes / (us / iters) / 1e3, std::sqrt(err2 / ref2));
    return std::sqrt(err2 / ref2) < 1e-2 ? 0 : 1;
}

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "sink";
    if (mode == "experts") return experts(argc > 2 ? std::atoi(argv[2]) : 3, argc > 3 ? std::atoi(argv[3]) : 9, argc > 4 ? std::atoi(argv[4]) : 1, argc > 5 ? std::atoi(argv[5]) : 20);
    if (mode == "route")return route(argc > 2 ? std::atoi(argv[2]) : 100, argc > 3 ? std::atoi(argv[3]) : 10, argc > 4 ? std::atoi(argv[4]) : 0, argc > 5 ? std::atoi(argv[5]) : 1);
    const bool gemv = mode == "gemv";
    auto arg = [&](int i, int d) { return argc > i ? std::atoi(argv[i]) : d; };
    const int C = arg(2, 16);
    const int G = gemv ? arg(3, 4) : 0, nc = gemv ? arg(4, 9) : 0;
    const int n = gemv ? 0 : arg(3, 2000);
    const int T = gemv ? arg(5, 1) : arg(4, 1);
    const int iters = gemv ? arg(6, 20) : arg(5, 20);

    const int x_units = gemv ? (X_OFF + nc * 512 + UNIT - 1) / UNIT : 1;
    const int task_units = gemv ? x_units + G * nc * 3 : 1 + n;
    const int y_units = gemv ? G : 1;
    const size_t ch_bytes = (size_t)T * task_units * UNIT;

    xrt::device dev(0);
    npu_xclbin_manager mgr(npu_device::device_npu2, &dev, false);
    npu_app_manager* am = mgr.register_xclbin("build/xclbins/decode.xclbin");
    npu_app app = am->create_app();
    buffer<uint8_t> in = app.create_bo_buffer<uint8_t>(ch_bytes * C);
    buffer<float> out = app.create_bo_buffer<float>((size_t)C * T * y_units * Y_UNIT);

    std::mt19937 rng(1);
    std::vector<float> xs(nc * 256);
    for (int c = 0; c < C; c++)
        for (int t = 0; t < T; t++) {
            uint8_t* p = in.data() + c * ch_bytes + (size_t)t * task_units * UNIT;
            memset(p, 0, (size_t)x_units * UNIT);
            int* h = (int*)p;
            if (!gemv) { h[0] = 1; h[1] = n; for (int i = 1; i <= n; i++) p[(size_t)i * UNIT] = (uint8_t)(i + c); continue; }
            h[0] = 2; h[1] = G; h[2] = nc;
            uint16_t* x = (uint16_t*)(p + X_OFF);
            for (int i = 0; i < nc * 256; i++) x[i] = f2bf(std::normal_distribution<float>(0, 1)(rng));
            uint8_t* w = p + (size_t)x_units * UNIT;
            for (size_t i = 0; i < (size_t)G * nc * 3 * UNIT; i++) w[i] = (uint8_t)rng();
            for (int b = 0; b < G * nc * 3; b++) {
                uint8_t* blk = w + (size_t)b * UNIT;
                for (int i = 0; i < 256; i++) { blk[i] &= 63; blk[256 + i] &= 63; }
                for (int i = 0; i < 32; i++) {
                    uint16_t s = f2bf(0.0005f * (1 + (rng() % 1000) / 1000.0f)), m = f2bf(-0.004f * (rng() % 1000) / 1000.0f);
                    memcpy(blk + 4608 + 2 * i, &s, 2);
                    memcpy(blk + 4672 + 2 * i, &m, 2);
                }
            }
        }
    in.sync_to_device();
    for (size_t i = 0; i < out.size(); i++) out.data()[i] = -1.0f;
    out.sync_to_device();

    const cache_flag_t cache = getenv("DEC_CACHE") ? (cache_flag_t)atoi(getenv("DEC_CACHE")) : normal_cache;
    npu_sequence* seq = app.seq();
    seq->clear_cmds();
    for (int t = 0; t < T; t++) {
        for (int c = 0; c < C; c++) {
            const int col = c % 8, r = c / 8;
            const npu_it_channel ch = r ? it_channel_1 : it_channel_0;
            seq->npu_dma_memcpy_nd(4, 1, S2MM, IT[col], npu_bd_id(2 + r), ch,
                {0, 0, 0, (uint32_t)(((size_t)c * T + t) * y_units * Y_UNIT)}, {1, 1, 1, (uint32_t)(y_units * Y_UNIT)}, {0, 0, 0, 1});
            seq->npu_dma_memcpy_nd(1, 0, MM2S, IT[col], npu_bd_id(r), ch,
                {0, 0, 0, (uint32_t)(c * ch_bytes + (size_t)t * task_units * UNIT)}, {1, 1, 1, (uint32_t)(task_units * UNIT)}, {0, 0, 0, 1}, -1, 0, false, cache);
        }
        for (int c = 0; c < C; c++) seq->npu_dma_wait(IT[c % 8], S2MM, c / 8 ? it_channel_1 : it_channel_0);
    }
    seq->cmds2seq();

    app(in, out);   // warm-up
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; i++) app(in, out);
    const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / iters;
    out.sync_from_device();
    const double bytes = (double)ch_bytes * C;
    std::printf("%s C=%d task_units=%d T=%d: %.1f us/run, %.2f GB/s (%.2f GB/s per channel), %.2f us/task\n", mode.c_str(), C,
                task_units, T, us, bytes / us / 1e3, bytes / us / 1e3 / C, us / T);

    if (!gemv) {
        bool ok = true;
        for (int c = 0; c < C; c++)
            for (int t = 0; t < T; t++) {
                const float* y = out.data() + ((size_t)c * T + t) * Y_UNIT;
                unsigned sum = 0;
                for (int i = 1; i <= n; i++) sum += (uint8_t)(i + c);
                if (y[0] != n || y[1] != (float)sum) { ok = false; std::printf("ch %d task %d: %g %g (want %d %u)\n", c, t, y[0], y[1], n, sum); }
            }
        std::printf("sink check %s\n", ok ? "OK" : "FAIL");
        return ok ? 0 : 1;
    }
    double max_rel = 0, err2 = 0, ref2 = 0;
    std::vector<float> wb(32 * 256);
    for (int c = 0; c < C; c++)
        for (int t = 0; t < T; t++) {
            const uint8_t* p = in.data() + c * ch_bytes + (size_t)t * task_units * UNIT;
            const uint16_t* x = (const uint16_t*)(p + X_OFF);
            const uint8_t* w = p + (size_t)x_units * UNIT;
            const float* y = out.data() + ((size_t)c * T + t) * y_units * Y_UNIT;
            for (int g = 0; g < G; g++) {
                double ref[96] = {};
                for (int cb = 0; cb < nc; cb++)
                    for (int b = 0; b < 3; b++) {
                        ref_block(wb.data(), w + ((size_t)(g * nc + cb) * 3 + b) * UNIT);
                        for (int row = 0; row < 32; row++)
                            for (int col = 0; col < 256; col++) ref[32 * b + row] += (double)wb[row * 256 + col] * bf2f(x[cb * 256 + col]);
                    }
                double scale = 0;
                for (int i = 0; i < 96; i++) scale = std::max(scale, std::fabs(ref[i]));
                for (int i = 0; i < 96; i++) {
                    const double e = y[g * 96 + i] - ref[i];
                    err2 += e * e; ref2 += ref[i] * ref[i];
                    max_rel = std::max(max_rel, std::fabs(e) / scale);
                    if (c == 0 && t == 0 && g == 0 && (i < 4 || getenv("DEC_DIAG"))) std::printf("y[%d] npu %.6f ref %.6f\n", i, y[i], ref[i]);
                }
            }
        }
    if (getenv("DEC_DIAG")) {   // first group's rows of block 0, cb 0: q x term and min term apart
        const uint8_t* p = in.data();
        const uint16_t* x = (const uint16_t*)(p + X_OFF);
        const uint8_t* b = p + (size_t)x_units * UNIT;
        for (int row = 0; row < 8; row++) {
            const int H = row / 16, par = row % 2, r = (row % 16) / 2, hp = 2 * H + par;
            uint16_t Sv, Mv;
            memcpy(&Sv, b + 4608 + hp * 16 + 2 * r, 2);
            memcpy(&Mv, b + 4672 + hp * 16 + 2 * r, 2);
            double qt = 0, mt = 0;
            for (int col = 0; col < 256; col++) {
                const int kt = col / 8, k = col % 8, g = kt / 4, i = r * 8 + k;
                const uint8_t byte = b[512 + 2048 * H + 1024 * par + 32 * kt + i / 2];
                const int q = (i & 1) ? byte >> 4 : byte & 15;
                qt += q * bf2f(Sv) * b[hp * 64 + g * 8 + r] * bf2f(x[col]);
                mt += bf2f(Mv) * b[256 + hp * 64 + g * 8 + r] * bf2f(x[col]);
            }
            std::printf("row %d: npu %.5f  q-term %.5f  min-term %.5f\n", row, out.data()[row], qt, mt);
        }
    }
    std::printf("gemv rel rms err %.3e, max err / max |y| %.3e\n", std::sqrt(err2 / ref2), max_rel);
    return 0;
}
