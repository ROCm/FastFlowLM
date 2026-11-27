// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.
//
// Stand-alone test of the decode overlay's stream cores (kernel/decode_core.cc).
//   run.exe sink [channels] [units] [tasks] [iters]   shim -> core stream bandwidth
//   run.exe gemv [channels] [groups] [nc] [tasks] [iters]   Q4_K GEMV vs an fp32 reference, and its speed
//   run.exe moe [iters]   router + top-k + the 8 chosen experts of one layer (synthetic weights) vs an fp64 reference
//   run.exe rms [iters]   glue cores (kernel/glue.hpp): RMS norm on every channel that has one
//   run.exe proj [iters] [nx] [steps]   the attention projections k v q gate rel of one layer on all 16 channels
//   run.exe attn [iters] [p] [global] [steps]   one token's attention of a layer on the 8 glue_up cores
//   run.exe chain [iters] [p] [global] [stages]   proj -> attention -> o proj of one layer in one run
//   run.exe layer [iters] [p] [global] [stages]   a whole layer application (mixes, attention half, MoE) in one run
// Channel c is shim MM2S (c / 8) of column c % 8, feeding core (c % 8, 3 + c / 8); its glue core is (c % 8, 2) or
// (c % 8, 5); channel 8 (core (0, 4), under the hub) has none.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "typedef.hpp"
#include "npu_utils/npu_utils.hpp"

static constexpr int UNIT = 4736, X_OFF = 128, Y_UNIT = 96;
static constexpr npu_tiles IT[8] = {IT0, IT1, IT2, IT3, IT4, IT5, IT6, IT7};

// A run that does not complete leaves cores stalled; tearing its hw context down right away can wedge amdxdna
// (processes in D state until a reboot). On a timeout: mark the device suspect, keep the context alive past the
// driver's 60 s job timeout so its own reset path runs first, then let the caller exit.
static ert_cmd_state wait_run(xrt::run& run, int ms) {
    ert_cmd_state st = ERT_CMD_STATE_TIMEOUT;
    try { st = run.wait(std::chrono::milliseconds(ms)); } catch (const std::exception& e) { std::printf("wait: %s\n", e.what()); }
    if (st != ERT_CMD_STATE_COMPLETED) {
        if (FILE* f = fopen((std::string(getenv("HOME") ? getenv("HOME") : "/tmp") + "/.npu_suspect").c_str(), "w")) {
            std::fprintf(f, "run state %d\n", (int)st);
            fclose(f);
        }
        std::printf("RUN DID NOT COMPLETE (state %d): holding the context 90 s for the driver timeout, then exiting\n", (int)st);
        std::fflush(stdout);
        std::this_thread::sleep_for(std::chrono::seconds(90));
        try { st = run.wait(std::chrono::milliseconds(1)); } catch (...) {}
        std::printf("after hold: state %d\n", (int)st);
    }
    return st;
}

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
static const npu_tiles HUB = get_tile(5, 0);
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
                st = wait_run(run, 3000);
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
                const int32_t* yi = (const int32_t*)y;
                if (yi[0] != n || (unsigned)yi[1] != sum) {
                    if (ok || getenv("ALLF")) std::printf("iter %d variant %d ch %d: %d %d (want %d %u)\n", it, v, c, yi[0], yi[1], n, sum);
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
        if (wait_run(run, 5000) != ERT_CMD_STATE_COMPLETED) return 2;
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

// The MoE half of a decode layer in one run: router (16 cores, bf16 hi + lo), top-k (core (0, 3) + hub), and the 8
// chosen experts (column e = slot e, its two cores = the two halves, situ on the cores), the experts' weights in the
// prefill's grouped layout (glu-interleaved gate_up), reached by the hub at absolute device addresses.
// run.exe moe [iters]: synthetic weights, checks ids, weights and every expert half against an fp64 reference.
static constexpr int OP_EXPERT = 3, OP_ROUTER = 4, OP_TOPK = 5;
static constexpr int NE = 128, D = 2304, FFH = 1152;
static constexpr size_t GU_E = (size_t)24 * 9 * 3 * UNIT, DN_E = (size_t)24 * 5 * 3 * UNIT;   // bytes per expert

static uint64_t sm_state = 0x9E3779B97F4A7C15ull;
static uint64_t splitmix() {
    uint64_t z = (sm_state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
static double urand() { return (splitmix() >> 11) * (1.0 / 9007199254740992.0); }
static double nrand() { return std::sqrt(-2 * std::log(urand() + 1e-300)) * std::cos(6.283185307179586 * urand()); }

// random Q4_K blocks: 6-bit scales / mins, S in [2e-4, 4e-4], M in [-1e-3, 0]
static void fill_q4k(uint8_t* p, size_t blocks) {
    for (size_t b = 0; b < blocks; b++) {
        uint8_t* blk = p + b * UNIT;
        for (int i = 0; i < UNIT; i += 8) { const uint64_t v = splitmix(); memcpy(blk + i, &v, 8); }
        for (int i = 0; i < 512; i++) blk[i] &= 63;
        for (int i = 0; i < 32; i++) {
            const uint16_t s = f2bf((float)(2e-4 * (1 + urand()))), m = f2bf((float)(-1e-3 * urand()));
            memcpy(blk + 4608 + 2 * i, &s, 2);
            memcpy(blk + 4672 + 2 * i, &m, 2);
        }
    }
}

// rows [g0 * 96, (g0 + n) * 96) of a grouped matrix (nc column blocks) times x: out[n * 96]
static void ref_rows(double* out, const uint8_t* w, int g0, int n, int nc, const float* x) {
    std::vector<float> wb(32 * 256);
    for (int g = 0; g < n; g++)
        for (int b = 0; b < 3; b++) {
            double* o = out + 96 * g + 32 * b;
            for (int i = 0; i < 32; i++) o[i] = 0;
            for (int cb = 0; cb < nc; cb++) {
                ref_block(wb.data(), w + ((size_t)((g0 + g) * nc + cb) * 3 + b) * UNIT);
                for (int row = 0; row < 32; row++) {
                    double s = 0;
                    for (int col = 0; col < 256; col++) s += (double)wb[row * 256 + col] * x[cb * 256 + col];
                    o[row] += s;
                }
            }
        }
}

static int moe_test(int iters) {
    xrt::device dev(0);
    npu_xclbin_manager mgr(npu_device::device_npu2, &dev, false);
    npu_app_manager* am = mgr.register_xclbin("build/xclbins/decode.xclbin");
    npu_app app = am->create_app();
    // io: headers, then y areas
    const size_t R_HDR = 0, T_HDR = 2 * UNIT, E_HDR = 3 * UNIT, RY = 4 * UNIT, TY = 6 * UNIT, EY = 7 * UNIT;
    const size_t IO = EY + (size_t)16 * 12 * Y_UNIT * 4;
    buffer<uint8_t> io = app.create_bo_buffer<uint8_t>(IO);
    buffer<uint8_t> rw = app.create_bo_buffer<uint8_t>((size_t)16 * 16 * UNIT);
    buffer<uint8_t> gu = app.create_bo_buffer<uint8_t>((size_t)NE * GU_E);
    buffer<uint8_t> dn = app.create_bo_buffer<uint8_t>((size_t)NE * DN_E);
    const auto t0 = std::chrono::steady_clock::now();
    fill_q4k(gu.data(), (size_t)NE * GU_E / UNIT);
    fill_q4k(dn.data(), (size_t)NE * DN_E / UNIT);
    gu.sync_to_device();
    dn.sync_to_device();
    std::printf("weights %.1f s\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());

    // x (fp32; the experts take bf16 x = x hi), router W fp32, bias
    std::vector<float> x(D), xh(D), W((size_t)NE * D), bias(NE);
    std::vector<uint16_t> xhi(D), xlo(D);
    for (int i = 0; i < D; i++) {
        x[i] = (float)nrand();
        xhi[i] = f2bf(x[i]);
        xlo[i] = f2bf(x[i] - bf2f(xhi[i]));
        xh[i] = bf2f(xhi[i]);
    }
    for (auto& v : W) v = (float)(0.02 * nrand());
    for (auto& v : bias) v = (float)(0.05 * nrand());
    // router weights: channel c = rows 8 c .. 8 c + 7, row r in units 2 r, 2 r + 1 of the channel: 18 granule pairs
    // [hi, lo] of 64 bf16 (column chunks kc = 18 h + k) per unit, 128 bytes padding
    memset(rw.data(), 0, rw.size());
    for (int c = 0; c < 16; c++)
        for (int kc = 0; kc < 36; kc++)
            for (int r = 0; r < 8; r++) {
                uint16_t* g = (uint16_t*)(rw.data() + (size_t)(c * 16 + 2 * r + kc / 18) * UNIT + (size_t)(kc % 18) * 256);
                for (int k = 0; k < 64; k++) {
                    const float w = W[(size_t)(8 * c + r) * D + 64 * kc + k];
                    g[k] = f2bf(w);
                    g[64 + k] = f2bf(w - bf2f(g[k]));
                }
            }
    rw.sync_to_device();
    memset(io.data(), 0, IO);
    ((int*)(io.data() + R_HDR))[0] = OP_ROUTER;
    memcpy(io.data() + R_HDR + X_OFF, xhi.data(), 2 * D);
    memcpy(io.data() + R_HDR + UNIT, xlo.data(), 2 * D);
    ((int*)(io.data() + T_HDR))[0] = OP_TOPK;
    memcpy(io.data() + T_HDR + X_OFF, bias.data(), 4 * NE);
    ((int*)(io.data() + E_HDR))[0] = OP_EXPERT;
    memcpy(io.data() + E_HDR + X_OFF, xhi.data(), 2 * D);

    const uint64_t gu_a = gu.bo().address() + 0x80000000ull, dn_a = dn.bo().address() + 0x80000000ull;
    npu_sequence* seq = app.seq();
    seq->clear_cmds();
    auto chn = [](int r) { return r ? it_channel_1 : it_channel_0; };
    // the BDs the hub points at the experts: linear, not started
    const bool nobd = getenv("MOE_NOBD") != nullptr;   // timing with MOE_ROUTER_ONLY: no hub BDs / args
    for (int col = 0; col < 8 && !nobd; col++)
        for (int b = 12; b < 16; b++)
            seq->npu_dma_memcpy_nd(1, 0, MM2S, IT[col], npu_bd_id(b), chn(b & 1), {0, 0, 0, 0}, {1, 1, 1, (uint32_t)UNIT}, {0, 0, 0, 1}, -1, 0, false, normal_cache, 15, false);
    seq->npu_maskwrite(HUB, 0x32038, 1, 1);
    seq->rtp_write(HUB, HUB_ARGS + 4 * 4, 0);
    seq->rtp_write(HUB, HUB_ARGS + 4 * 11, (uint32_t)gu_a);
    seq->rtp_write(HUB, HUB_ARGS + 4 * 12, (uint32_t)(gu_a >> 32));
    seq->rtp_write(HUB, HUB_ARGS + 4 * 13, (uint32_t)dn_a);
    seq->rtp_write(HUB, HUB_ARGS + 4 * 14, (uint32_t)(dn_a >> 32));
    seq->rtp_write(HUB, HUB_ARGS + 4 * 15, 1);
    // MOE_ROUTER_ONLY=1 (timing): the router step alone, the hub not started
    const bool ronly = getenv("MOE_ROUTER_ONLY") && atoi(getenv("MOE_ROUTER_ONLY"));
    if (!ronly) seq->rtp_write(HUB, HUB_START_LOCK, 1);
    // router
    for (int c = 0; c < 16; c++) {
        const int col = c % 8, r = c / 8;
        seq->npu_dma_memcpy_nd(4, 0, S2MM, IT[col], npu_bd_id(8 + r), chn(r), {0, 0, 0, (uint32_t)((RY + c * Y_UNIT * 4) / 4)}, {1, 1, 1, (uint32_t)Y_UNIT}, {0, 0, 0, 1});
        seq->npu_dma_memcpy_nd(1, 0, MM2S, IT[col], npu_bd_id(2 * r), chn(r), {0, 0, 0, (uint32_t)R_HDR}, {1, 1, 1, (uint32_t)(2 * UNIT)}, {0, 0, 0, 1});
        seq->npu_dma_memcpy_nd(1, 1, MM2S, IT[col], npu_bd_id(2 * r + 1), chn(r), {0, 0, 0, (uint32_t)(c * 16 * UNIT)}, {1, 1, 1, (uint32_t)(16 * UNIT)}, {0, 0, 0, 1});
    }
    for (int c = 0; c < 16; c++) seq->npu_dma_wait(IT[c % 8], S2MM, chn(c / 8));
    // top-k on core (0, 3), then the expert headers; the hub pushes the weights
    if (!ronly) {
        seq->npu_dma_memcpy_nd(4, 0, S2MM, IT[0], npu_bd_id(10), it_channel_1, {0, 0, 0, (uint32_t)(TY / 4)}, {1, 1, 1, (uint32_t)Y_UNIT}, {0, 0, 0, 1});
        seq->npu_dma_memcpy_nd(1, 0, MM2S, IT[0], npu_bd_id(6), it_channel_1, {0, 0, 0, (uint32_t)T_HDR}, {1, 1, 1, (uint32_t)UNIT}, {0, 0, 0, 1});
        seq->npu_dma_memcpy_nd(1, 0, MM2S, IT[0], npu_bd_id(7), it_channel_1, {0, 0, 0, (uint32_t)RY}, {1, 1, 1, (uint32_t)(2 * UNIT)}, {0, 0, 0, 1});
        for (int c = 0; c < 16; c++) {
            const int col = c % 8, r = c / 8;
            seq->npu_dma_memcpy_nd(4, 0, S2MM, IT[col], npu_bd_id(8 + r), chn(r), {0, 0, 0, (uint32_t)((EY + (size_t)c * 12 * Y_UNIT * 4) / 4)}, {1, 1, 1, (uint32_t)(12 * Y_UNIT)}, {0, 0, 0, 1});
            seq->npu_dma_memcpy_nd(1, 0, MM2S, IT[col], npu_bd_id(4 + r), chn(r), {0, 0, 0, (uint32_t)E_HDR}, {1, 1, 1, (uint32_t)UNIT}, {0, 0, 0, 1});
        }
        seq->npu_dma_wait(IT[0], S2MM, it_channel_1);
        for (int c = 0; c < 16; c++) seq->npu_dma_wait(IT[c % 8], S2MM, chn(c / 8));
    }
    seq->cmds2seq();

    // reference routing (fp64 logits of fp32 x and W, fp32 sigmoid as the host)
    std::vector<int> ref_id(8);
    std::vector<float> ref_w(8), p(NE), key(NE);
    for (int e = 0; e < NE; e++) {
        double s = 0;
        for (int i = 0; i < D; i++) s += (double)W[(size_t)e * D + i] * x[i];
        p[e] = 1.0f / (1.0f + std::exp(-(float)s));
        key[e] = p[e] + bias[e];
    }
    std::vector<int> idx(NE);
    for (int e = 0; e < NE; e++) idx[e] = e;
    std::partial_sort(idx.begin(), idx.begin() + 8, idx.end(), [&](int a, int b) { return key[a] > key[b]; });
    float sum = 0;
    for (int k = 0; k < 8; k++) sum += p[idx[k]];
    for (int k = 0; k < 8; k++) { ref_id[k] = idx[k]; ref_w[k] = p[idx[k]] / sum * 8; }

    double us = 0;
    for (int it = 0; it <= iters; it++) {
        for (size_t i = RY; i < IO; i += 4) *(float*)(io.data() + i) = -1.0f;
        io.sync_to_device();
        const auto t1 = std::chrono::steady_clock::now();
        xrt::run run = app.create_run(io, rw);
        run.start();
        ert_cmd_state st = ERT_CMD_STATE_TIMEOUT;
        st = wait_run(run, 5000);
        if (it > 0) us += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t1).count();
        io.sync_from_device();
        if (st != ERT_CMD_STATE_COMPLETED) {
            std::printf("run state %d\nlogit err:", (int)st);
            for (int c = 0; c < 16; c++) {
                double m = 0;
                for (int i = 0; i < 8; i++) {
                    double r = 0;
                    for (int k = 0; k < D; k++) r += (double)W[(size_t)(8 * c + i) * D + k] * x[k];
                    m = std::max(m, std::fabs(*(float*)(io.data() + RY + (c * Y_UNIT + i) * 4) - r));
                }
                std::printf(" %.1e", m);
            }
            std::printf("\ntopk unit:");
            for (int k = 0; k < 68; k++) std::printf("%s%08x", k % 16 ? " " : "\n  ", ((uint32_t*)(io.data() + TY))[k]);
            std::printf("\nref ids:");
            for (int k = 0; k < 8; k++) std::printf(" %d", ref_id[k]);
            std::printf("\nexpert y units written per channel:");
            for (int c = 0; c < 16; c++) {
                int n = 0;
                for (int u = 0; u < 12; u++) n += *(float*)(io.data() + EY + ((size_t)c * 12 + u) * Y_UNIT * 4) != -1.0f;
                std::printf(" %d", n);
            }
            std::printf("\n");
            return 2;
        }
    }
    if (ronly) { std::printf("router only: %.1f us per run (%d iters)\n", us / std::max(iters, 1), iters); return 0; }
    // router logits
    double lerr = 0, lref = 0;
    for (int c = 0; c < 16; c++)
        for (int i = 0; i < 8; i++) {
            const int e = 8 * c + i;
            double s = 0;
            for (int k = 0; k < D; k++) s += (double)W[(size_t)e * D + k] * x[k];
            const double d = *(float*)(io.data() + RY + (c * Y_UNIT + i) * 4) - s;
            lerr = std::max(lerr, std::fabs(d));
            lref = std::max(lref, std::fabs(s));
        }
    const int32_t* tid = (const int32_t*)(io.data() + TY);
    const float* tw = (const float*)(io.data() + TY) + 8;
    bool ids_ok = true;
    std::printf("logits max abs err %.3e (max |logit| %.3f)\nids  npu:", lerr, lref);
    for (int k = 0; k < 8; k++) { std::printf(" %d", tid[k]); ids_ok &= tid[k] == ref_id[k]; }
    std::printf("\n     ref:");
    for (int k = 0; k < 8; k++) std::printf(" %d", ref_id[k]);
    double werr = 0;
    for (int k = 0; k < 8; k++) werr = std::max(werr, (double)std::fabs(tw[k] - ref_w[k]));
    std::printf("\nweights max abs err %.3e (w0 npu %.6f ref %.6f)\n", werr, tw[0], ref_w[0]);
    // experts, from the NPU's ids
    double err2 = 0, ref2 = 0, out_err2 = 0, out_ref2 = 0;
    std::vector<double> out(D, 0.0), out_ref(D, 0.0);
    for (int e = 0; e < 8; e++) {
        const int id = tid[e];
        if (id < 0 || id >= NE) { std::printf("bad id %d\n", id); return 1; }
        std::vector<double> g(2 * FFH);
        ref_rows(g.data(), gu.data() + (size_t)id * GU_E, 0, 24, 9, xh.data());
        // physical row block pb: gate j = pb / 2 (even), up j (odd)
        std::vector<float> h(1280, 0.0f);
        for (int j = 0; j < 36; j++)
            for (int i = 0; i < 32; i++) {
                const double gv = g[64 * j + i], uv = g[64 * j + 32 + i];
                const double s = 4 * std::tanh(gv / 4) / (1 + std::exp(-gv)) * 25 * std::tanh(uv / 25);
                h[32 * j + i] = bf2f(f2bf((float)s));
            }
        std::vector<double> z(D);
        ref_rows(z.data(), dn.data() + (size_t)id * DN_E, 0, 24, 5, h.data());
        for (int r = 0; r < 2; r++) {
            const float* y = (const float*)(io.data() + EY) + (size_t)(e + 8 * r) * 12 * Y_UNIT;
            for (int i = 0; i < 1152; i++) {
                const double d = y[i] - z[1152 * r + i];
                err2 += d * d; ref2 += z[1152 * r + i] * z[1152 * r + i];
                out[1152 * r + i] += tw[e] * y[i];
                out_ref[1152 * r + i] += ref_w[e] * z[1152 * r + i];
            }
            if (e == 0) std::printf("slot 0 half %d: y[0..2] npu %.4f %.4f ref %.4f %.4f\n", r, y[0], y[1], z[1152 * r], z[1152 * r + 1]);
        }
    }
    for (int i = 0; i < D; i++) { const double d = out[i] - out_ref[i]; out_err2 += d * d; out_ref2 += out_ref[i] * out_ref[i]; }
    const double rel = std::sqrt(err2 / ref2), orel = std::sqrt(out_err2 / out_ref2);
    std::printf("experts rel rms err %.3e, moe out rel rms err %.3e; %.1f us per run (%d iters)\n", rel, orel, us / std::max(iters, 1), iters);
    return ids_ok && rel < 2e-2 ? 0 : 1;
}

// glue cores: y = x / rms(x) * g on every channel with a glue core; `steps` times in one run, each step waits for
// every channel (the per-step cost of a dependent step), run.exe rms [iters] [steps] [channels] [n]
static int rms_test(int iters, int steps, int nch, int D) {
    constexpr int OP_RMS = 16;
    const size_t TASK = 5 * (size_t)UNIT;   // header + x + g = 128 + 8 D bytes (D <= 2304) -> <= 4 units, padded to 5
    const uint32_t IN = (uint32_t)((X_OFF + 8 * D + UNIT - 1) / UNIT * UNIT);
    xrt::device dev(0);
    npu_xclbin_manager mgr(npu_device::device_npu2, &dev, false);
    npu_app_manager* am = mgr.register_xclbin("build/xclbins/decode.xclbin");
    npu_app app = am->create_app();
    buffer<uint8_t> in = app.create_bo_buffer<uint8_t>(16 * TASK);
    buffer<float> out = app.create_bo_buffer<float>((size_t)16 * 2304);
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0, 1);
    memset(in.data(), 0, in.size());
    std::vector<int> chans;
    for (int c = 0; c < 16 && (int)chans.size() < nch; c++) if (c != 8) chans.push_back(c);
    for (int c : chans) {
        uint8_t* p = in.data() + c * TASK;
        ((int*)p)[0] = OP_RMS;
        ((int*)p)[1] = D;
        float* x = (float*)(p + X_OFF);
        for (int i = 0; i < 2 * D; i++) x[i] = (i < D ? 3.0f : 1.0f) * nd(rng) + (i < D ? 0.5f * c : 0.0f);
    }
    in.sync_to_device();
    npu_sequence* seq = app.seq();
    seq->clear_cmds();
    for (int k = 0; k < steps; k++) {
        for (int c : chans) {
            const int col = c % 8, r = c / 8;
            const npu_it_channel ch = r ? it_channel_1 : it_channel_0;
            // SPLIT=w: the output as two S2MM BDs, w words then the rest (a BD edge inside a y unit)
            const uint32_t w1 = getenv("SPLIT") ? (uint32_t)atoi(getenv("SPLIT")) : 0;
            if (w1) {
                seq->npu_dma_memcpy_nd(4, 1, S2MM, IT[col], npu_bd_id(2 + r), ch, {0, 0, 0, (uint32_t)(c * D)}, {1, 1, 1, w1}, {0, 0, 0, 1});
                seq->npu_dma_memcpy_nd(4, 1, S2MM, IT[col], npu_bd_id(6 + r), ch, {0, 0, 0, (uint32_t)(c * D) + w1}, {1, 1, 1, (uint32_t)D - w1}, {0, 0, 0, 1});
            } else
            seq->npu_dma_memcpy_nd(4, 1, S2MM, IT[col], npu_bd_id(2 + r), ch, {0, 0, 0, (uint32_t)(c * D)}, {1, 1, 1, (uint32_t)D}, {0, 0, 0, 1});
            seq->npu_dma_memcpy_nd(1, 0, MM2S, IT[col], npu_bd_id(r), ch, {0, 0, 0, (uint32_t)(c * TASK)}, {1, 1, 1, IN}, {0, 0, 0, 1});
        }
        for (int c : chans) seq->npu_dma_wait(IT[c % 8], S2MM, c / 8 ? it_channel_1 : it_channel_0);
        if (getenv("SPLIT"))
            for (int c : chans) seq->npu_dma_wait(IT[c % 8], S2MM, c / 8 ? it_channel_1 : it_channel_0);
    }
    seq->cmds2seq();
    double us = 0;
    for (int it = 0; it <= iters; it++) {
        for (size_t i = 0; i < out.size(); i++) out.data()[i] = -1.0f;
        out.sync_to_device();
        const auto t1 = std::chrono::steady_clock::now();
        xrt::run run = app.create_run(in, out);
        run.start();
        ert_cmd_state st = ERT_CMD_STATE_TIMEOUT;
        st = wait_run(run, 3000);
        if (it > 0) us += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t1).count();
        if (st != ERT_CMD_STATE_COMPLETED) { std::printf("run state %d\n", (int)st); return 2; }
    }
    out.sync_from_device();
    double worst = 0;
    for (int c : chans) {
        const float* x = (const float*)(in.data() + c * TASK + X_OFF);
        double ss = 0;
        for (int i = 0; i < D; i++) ss += (double)x[i] * x[i];
        const double inv = 1.0 / std::sqrt(ss / D + 1e-6);
        double e = 0;
        for (int i = 0; i < D; i++) e = std::max(e, std::fabs(out.data()[(size_t)c * D + i] - x[i] * inv * x[D + i]) / (std::fabs(x[i] * inv * x[D + i]) + 1e-3));
        worst = std::max(worst, e);
        if (e > 1e-5) std::printf("ch %d: max rel err %.3e (y0 %.6f ref %.6f)\n", c, e, out.data()[(size_t)c * D], x[0] * inv * x[D]);
    }
    std::printf("rms on %zu glue cores, %d steps per run: max rel err %.3e; %.1f us per run\n", chans.size(), steps, worst, us / std::max(iters, 1));
    return worst < 1e-5 ? 0 : 1;
}

// The attention projections of a layer as one step: matrices k v q gate rel (11 + 11 + 32 + 32 + 4 groups of 96 rows,
// the prefill's grouped layout back to back), one shared OP_PROJ header with the channel table and the x vectors
// (nx 3: xq, xk, xv of a global layer; nx 1: one x), groups split evenly over the 16 channels; y = 90 groups x 96
// rows in matrix order, then MISC: 3 y units of dense bf16 rows (sink 24 rows on channel 15 with xq, k conv 32 on
// channel 0 with xk, v conv 32 on channel 2 with xv; nx 1: the sink only).
// run.exe proj [iters] [nx] [steps]: synthetic weights vs an fp64 reference
static int proj_test(int iters, int nx, int steps) {
    constexpr int OP_PROJ = 6, NG = 90, NC = 9;
    const int ends[5] = {11, 22, 54, 86, 90};
    const int xi3[5] = {1, 2, 0, 0, 0};   // k: xk, v: xv, q gate rel: xq
    const size_t GB = (size_t)NC * 3 * UNIT;   // bytes per group
    const size_t HDR = (size_t)(X_OFF + nx * D * 2 + UNIT - 1) / UNIT * UNIT;
    // dense rows: (channel, rows, x index, MISC unit)
    struct dense { int c, n, x, u; };
    std::vector<dense> dns = {{15, 24, 0, 0}};
    if (nx == 3) { dns.push_back({0, 32, 1, 1}); dns.push_back({2, 32, 2, 2}); }
    int nd_of[16] = {};
    for (const dense& d : dns) nd_of[d.c] = (int)(&d - dns.data()) + 1;
    const size_t DW = NG * GB;   // dense row units after the groups in wt, 32 per matrix
    xrt::device dev(0);
    npu_xclbin_manager mgr(npu_device::device_npu2, &dev, false);
    npu_app_manager* am = mgr.register_xclbin("build/xclbins/decode.xclbin");
    npu_app app = am->create_app();
    buffer<uint8_t> io = app.create_bo_buffer<uint8_t>(HDR + (size_t)(NG + 3) * Y_UNIT * 4);
    buffer<uint8_t> wt = app.create_bo_buffer<uint8_t>(DW + 3 * 32 * UNIT);
    fill_q4k(wt.data(), NG * GB / UNIT);
    std::vector<float> dwf(3 * 32 * D);   // the dense rows as fp32 (bf16 values)
    memset(wt.data() + DW, 0, 3 * 32 * UNIT);
    for (int i = 0; i < 3 * 32; i++)
        for (int k = 0; k < D; k++) {
            const uint16_t b = f2bf((float)(0.05 * nrand()));
            dwf[(size_t)i * D + k] = bf2f(b);
            memcpy(wt.data() + DW + (size_t)i * UNIT + 2 * k, &b, 2);
        }
    wt.sync_to_device();
    std::vector<std::vector<float>> xv(nx, std::vector<float>(D));
    memset(io.data(), 0, io.size());
    int* h = (int*)io.data();
    h[0] = OP_PROJ;
    h[1] = nx;
    // 6 groups per channel, 5 on the channels with dense rows and then on the last ones, until they add up to NG
    int g0[16], ng[16];
    for (int c = 0; c < 16; c++) ng[c] = 6;
    int over = 16 * 6 - NG;
    for (const dense& d : dns) if (over > 0) { ng[d.c]--; over--; }
    for (int c = 15; over > 0; c--) if (ng[c] == 6) { ng[c]--; over--; }
    for (int c = 0, g = 0; c < 16; c++) {
        g0[c] = g;
        g += ng[c];
        h[2 + c] = g0[c] | ng[c] << 16;
    }
    for (int m = 0; m < 5; m++) h[18 + m] = ends[m] | (nx == 3 ? xi3[m] : 0) << 16;
    for (const dense& d : dns) ((int16_t*)h)[46 + d.c] = (int16_t)(d.n | d.x << 8);
    for (int i = 0; i < nx; i++)
        for (int k = 0; k < D; k++) {
            const uint16_t b = f2bf((float)nrand());
            xv[i][k] = bf2f(b);
            memcpy(io.data() + X_OFF + (size_t)(i * D + k) * 2, &b, 2);
        }
    npu_sequence* seq = app.seq();
    seq->clear_cmds();
    auto chn = [](int r) { return r ? it_channel_1 : it_channel_0; };
    for (int k = 0; k < steps; k++) {
        for (int c = 0; c < 16; c++) {
            const int col = c % 8, r = c / 8;
            seq->npu_dma_memcpy_nd(4, 0, S2MM, IT[col], npu_bd_id(4 + r), chn(r), {0, 0, 0, (uint32_t)((HDR + (size_t)g0[c] * Y_UNIT * 4) / 4)}, {1, 1, 1, (uint32_t)(ng[c] * Y_UNIT)}, {0, 0, 0, 1});
            seq->npu_dma_memcpy_nd(1, 0, MM2S, IT[col], npu_bd_id(2 * r), chn(r), {0, 0, 0, 0}, {1, 1, 1, (uint32_t)HDR}, {0, 0, 0, 1});
            seq->npu_dma_memcpy_nd(1, 1, MM2S, IT[col], npu_bd_id(2 * r + 1), chn(r), {0, 0, 0, (uint32_t)(g0[c] * GB)}, {1, 1, 1, (uint32_t)(ng[c] * GB)}, {0, 0, 0, 1});
            if (nd_of[c]) {
                const dense& d = dns[nd_of[c] - 1];
                seq->npu_dma_memcpy_nd(1, 1, MM2S, IT[col], npu_bd_id(8 + r), chn(d.c / 8), {0, 0, 0, (uint32_t)(DW + (size_t)d.u * 32 * UNIT)}, {1, 1, 1, (uint32_t)(d.n * UNIT)}, {0, 0, 0, 1});
                seq->npu_dma_memcpy_nd(4, 0, S2MM, IT[col], npu_bd_id(10 + r), chn(r), {0, 0, 0, (uint32_t)((HDR + (size_t)(NG + d.u) * Y_UNIT * 4) / 4)}, {1, 1, 1, (uint32_t)Y_UNIT}, {0, 0, 0, 1});
            }
        }
        for (int c = 0; c < 16; c++) seq->npu_dma_wait(IT[c % 8], S2MM, chn(c / 8));
        for (const dense& d : dns) seq->npu_dma_wait(IT[d.c % 8], S2MM, chn(d.c / 8));
    }
    seq->cmds2seq();
    double us = 0;
    for (int it = 0; it <= iters; it++) {
        for (size_t i = HDR; i < io.size(); i += 4) *(float*)(io.data() + i) = -1.0f;
        io.sync_to_device();
        const auto t1 = std::chrono::steady_clock::now();
        xrt::run run = app.create_run(io, wt);
        run.start();
        ert_cmd_state st = ERT_CMD_STATE_TIMEOUT;
        st = wait_run(run, 5000);
        if (it > 0) us += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t1).count();
        io.sync_from_device();
        if (st != ERT_CMD_STATE_COMPLETED) {
            std::printf("run state %d; y units written per channel:", (int)st);
            for (int c = 0; c < 16; c++) {
                int n = 0;
                for (int g = g0[c]; g < g0[c] + ng[c]; g++) n += *(float*)(io.data() + HDR + (size_t)g * Y_UNIT * 4) != -1.0f;
                std::printf(" %d/%d", n, ng[c]);
            }
            std::printf("\n");
            return 2;
        }
    }
    double err2 = 0, ref2 = 0, worst = 0;
    std::vector<double> ref(96);
    for (int g = 0, m = 0; g < NG; g++) {
        while (g >= ends[m]) m++;
        ref_rows(ref.data(), wt.data(), g, 1, NC, xv[nx == 3 ? xi3[m] : 0].data());
        const float* y = (const float*)(io.data() + HDR) + (size_t)g * Y_UNIT;
        double e2 = 0, r2 = 0;
        for (int i = 0; i < 96; i++) { const double d = y[i] - ref[i]; e2 += d * d; r2 += ref[i] * ref[i]; }
        err2 += e2; ref2 += r2;
        worst = std::max(worst, std::sqrt(e2 / r2));
    }
    double dworst = 0;
    for (const dense& d : dns) {
        const float* y = (const float*)(io.data() + HDR) + (size_t)(NG + d.u) * Y_UNIT;
        for (int i = 0; i < d.n; i++) {
            double r = 0, a = 0;
            for (int k = 0; k < D; k++) { r += (double)dwf[((size_t)d.u * 32 + i) * D + k] * xv[d.x][k]; a += std::fabs(dwf[((size_t)d.u * 32 + i) * D + k] * xv[d.x][k]); }
            dworst = std::max(dworst, std::fabs(y[i] - r) / a);
        }
    }
    std::printf("dense rows: max err / sum |w x| %.3e\n", dworst);
    if (dworst > 1e-5) worst = 1;
    const double us1 = us / std::max(iters, 1);
    std::printf("proj nx=%d, %d steps per run: rel rms err %.3e (worst group %.3e); %.1f us per run, %.2f GB/s\n", nx, steps, std::sqrt(err2 / ref2), worst,
                us1, (double)steps * NG * GB / us1 / 1e3);
    return worst < 1e-2 ? 0 : 1;
}

// The attention half's layout (OP_ATTN, kernel/glue.hpp; KV head g on the glue_up core of column g). Per head a region of
// wt = [static 1456 fp32 (2 units)][rel_bias bf16 [2][1024 rows (distance 1023 - row)][8] (7 units)][conv ring 4 x 256
// fp32 (1 unit)][records, KPU per unit: K as [16 chunks][KPU][8] bf16, then V as [KPU][128] bf16]. The stream: header
// (128 B), q / gate of the head's 3 query heads, the A block (K 1056 | V 1056 | R 384 | MISC 288 fp32, its BD runs on
// to the unit boundary), the region; out: the ring row, the record (K, V) and O (384 bf16).
namespace attn {
constexpr int OP = 17, DH = 128, NREL = 16, NKV = 8, QD = 3072, KPU = 8, A_N = 2784;
constexpr size_t A_LEN = 4 * UNIT - 128 - 3072, RB = 2 * UNIT, RING = 9 * UNIT, REC0 = 10 * UNIT;

struct geom {
    int p, glob, nk, nrec, slot;
    bool ring;
    float tau;
    size_t RS;   // bytes per head region
    geom(int p_, int glob_) : p(p_), glob(glob_) {
        ring = !glob;
        nk = glob ? p : std::min(p, 1024);
        nrec = glob ? p + 1 : 1024;
        slot = glob ? p : p % 1024;
        RS = REC0 + (size_t)(nrec + KPU - 1) / KPU * UNIT;
        tau = glob ? (float)(1.0 + 0.1 * std::log(std::max(1.0, (p + 1) / 8192.0))) : 1.0f;
    }
};

// element c (K 0-127, V 128-255) of record s of a head region
static uint16_t* rec_at(const uint8_t* base, int s, int c) {
    uint8_t* u = const_cast<uint8_t*>(base) + REC0 + (size_t)(s / KPU) * UNIT;
    return c < DH ? (uint16_t*)(u + (c / 8) * 128 + (s % KPU) * 16) + c % 8 : (uint16_t*)(u + 2048 + (s % KPU) * 256) + c - DH;
}

// random head regions (static, rel bias, ring, records) and the headers
static void fill(uint8_t* wt, uint8_t* hdr, const geom& G) {
    for (int g = 0; g < NKV; g++) {
        float* st = (float*)(wt + g * G.RS);
        for (int i = 0; i < 384; i++) st[i] = (float)(1.5 + urand());
        for (int i = 0; i < 48; i++) st[384 + i] = (float)(1.0 + 2 * urand());
        for (int i = 0; i < 1024; i++) st[432 + i] = (float)(0.3 * nrand());
        uint16_t* rb = (uint16_t*)(wt + g * G.RS + RB);
        for (int d = 0; d < 1024; d++)
            for (int k = 0; k < NREL; k++) rb[((k / 8) * 1024 + 1023 - d) * 8 + k % 8] = f2bf((float)(0.3 * nrand()));
        float* rg = (float*)(wt + g * G.RS + RING);
        for (int r = 0; r < 4; r++)
            for (int c = 0; c < 256; c++) rg[r * 256 + c] = (float)(c < DH ? 0.5 * nrand() : nrand());
        for (int s = 0; s < G.nrec; s++)
            for (int c = 0; c < 256; c++) *rec_at(wt + g * G.RS, s, c) = f2bf((float)(c < DH ? 1.5 * nrand() : nrand()));
        int* h = (int*)(hdr + g * 128);
        h[0] = OP; h[1] = G.p; h[2] = G.glob; memcpy(h + 3, &G.tau, 4); h[4] = G.nk; h[5] = G.ring; h[6] = g; h[7] = getenv("ATTN_DBG") ? atoi(getenv("ATTN_DBG")) : 0;
    }
}

// fp64 reference of the host decode from the regions before the step: O (fp64) and the new records (bf16)
static void reference(const float* Q, const float* Gt, const float* A, const uint8_t* wt, const geom& G, std::vector<double>& o_ref,
                      std::vector<uint16_t>& rec_ref) {
    const int p = G.p, glob = G.glob;
    const float* misc = A + 2496;
    o_ref.assign(QD, 0.0);
    rec_ref.assign(NKV * 256, 0);
    for (int g = 0; g < NKV; g++) {
        const uint8_t* wg = wt + g * G.RS;
        const float* st = (const float*)wg;
        const uint16_t* rb = (const uint16_t*)(wg + RB);
        const float* rg = (const float*)(wg + RING);
        double kv[2][DH];
        for (int half = 0; half < 2; half++) {
            const float* w = st + 432 + 512 * half;
            const float* dyn = misc + 96 * (1 + half) + 4 * g;
            const float* x3 = A + 1056 * half + 128 * g;
            for (int c = 0; c < DH; c++) {
                const double x[4] = {rg[((p - 3) & 3) * 256 + DH * half + c], rg[((p - 2) & 3) * 256 + DH * half + c],
                                     rg[((p - 1) & 3) * 256 + DH * half + c], x3[c]};
                double acc = 0;
                for (int k = 0; k < 4; k++) acc += w[k * DH + c] * x[k] + (glob ? dyn[3 - k] * x[k] : 0.0);
                kv[half][c] = (x[3] + acc) / std::sqrt(glob ? 9.0 : 5.0);
            }
        }
        double n2 = 0, v2 = 0;
        for (int c = 0; c < DH; c++) { n2 += kv[0][c] * kv[0][c]; v2 += kv[1][c] * kv[1][c]; }
        const double fk = std::min(22.627416997969522 / (std::sqrt(n2) + 1e-6), 1.0), fv = 1 / std::sqrt(v2 / DH + 1e-6);
        float kb[DH], vb[DH];
        for (int c = 0; c < DH; c++) {
            rec_ref[g * 256 + c] = f2bf((float)(kv[0][c] * fk));
            rec_ref[g * 256 + DH + c] = f2bf((float)(kv[1][c] * fv));
            kb[c] = bf2f(rec_ref[g * 256 + c]);
            vb[c] = bf2f(rec_ref[g * 256 + DH + c]);
        }
        for (int i = 0; i < 3; i++) {
            const int hh = 3 * g + i;
            double q[DH], r[NREL], s2 = 0, r2 = 0;
            for (int c = 0; c < DH; c++) { q[c] = (double)Q[hh * DH + c] * st[i * DH + c]; s2 += q[c] * q[c]; }
            for (int c = 0; c < NREL; c++) { r[c] = (double)A[2112 + hh * NREL + c] * st[384 + i * NREL + c]; r2 += r[c] * r[c]; }
            const double fq = std::min(22.627416997969522 / (std::sqrt(s2) + 1e-6), 1.0) * G.tau, fr = std::min(8.0 / (std::sqrt(r2) + 1e-6), 1.0) * G.tau;
            auto tab = [&](int d) { if (d >= 1024) return 0.0; double a = 0; for (int k = 0; k < NREL; k++) a += bf2f(rb[((k / 8) * 1024 + 1023 - d) * 8 + k % 8]) * r[k] * fr; return a; };
            std::vector<double> sc;
            std::vector<std::vector<float>> vf;
            auto key = [&](const float* k, int d) {
                double a = 0;
                for (int c = 0; c < DH; c++) a += q[c] * fq * k[c];
                sc.push_back(a / std::sqrt(128.0) + tab(d));
            };
            key(kb, 0);
            vf.push_back(std::vector<float>(vb, vb + DH));
            for (int s = 0; s < G.nk; s++) {
                int d = p - s;
                if (G.ring) { d &= 1023; if (d == 0) continue; }
                std::vector<float> k(DH), v(DH);
                for (int c = 0; c < DH; c++) { k[c] = bf2f(*rec_at(wg, s, c)); v[c] = bf2f(*rec_at(wg, s, DH + c)); }
                key(k.data(), d);
                vf.push_back(v);
            }
            double m = misc[hh];
            for (double x : sc) m = std::max(m, x);
            double l = std::exp(misc[hh] - m);
            std::vector<double> o(DH, 0.0);
            for (size_t j = 0; j < sc.size(); j++) {
                const double e = std::exp(sc[j] - m);
                l += e;
                for (int c = 0; c < DH; c++) o[c] += e * vf[j][c];
            }
            for (int c = 0; c < DH; c++) o_ref[hh * DH + c] = o[c] / l / (1 + std::exp(-(double)Gt[hh * DH + c]));
        }
    }
}

// the regions after the step (ring row = the raw K / V of A, record) and O against the reference
static bool check(const uint8_t* wt, const float* A, const uint16_t* O, const geom& G, const std::vector<double>& o_ref,
                  const std::vector<uint16_t>& rec_ref) {
    int bad_ring = 0, bad_rec = 0;
    double e2 = 0, r2 = 0, emax = 0;
    for (int g = 0; g < NKV; g++) {
        const float* rr = (const float*)(wt + g * G.RS + RING + (G.p & 3) * 1024);
        for (int c = 0; c < 256; c++) bad_ring += memcmp(rr + c, A + (c < DH ? 0 : 1056) + 128 * g + c % DH, 4) != 0;
        for (int c = 0; c < 256; c++) {
            const float a = bf2f(*rec_at(wt + g * G.RS, G.slot, c)), b = bf2f(rec_ref[g * 256 + c]);
            if (std::fabs(a - b) > std::fabs(b) * (1.0 / 128) + 1e-6) { if (bad_rec < 4) std::printf("g %d rec %d: %g vs %g\n", g, c, a, b); bad_rec++; }
        }
    }
    for (int i = 0; i < QD; i++) {
        const double d = bf2f(O[i]) - o_ref[i];
        e2 += d * d; r2 += o_ref[i] * o_ref[i];
        emax = std::max(emax, std::fabs(d));
    }
    const double rms = std::sqrt(r2 / QD), rel = std::sqrt(e2 / r2);
    std::printf("attn p=%d %s (%d records): ring row mismatches %d, record mismatches %d, O rel rms err %.3e (max |err| / rms %.3e); "
                "o[0] %.5f ref %.5f\n", G.p, G.glob ? "global" : "swa", G.nk, bad_ring, bad_rec, rel, emax / rms, bf2f(O[0]), o_ref[0]);
    return bad_ring == 0 && bad_rec == 0 && rel < 5e-3;
}

// one step's BDs for the 8 heads (io = argument 0, wt = argument 1): headers at H0 + 128 g, q of head g at Q0 + 1536 g
// (gate QD floats after), A at A0, the regions at W0 + g RS, O of head g to o_at(g); BD set `set` (0 or 8), pushed or
// only written (push = false: pushed later by register writes, see attn_test)
template <class F>
static void issue(npu_sequence* seq, const geom& G, size_t W0, size_t H0, size_t Q0, size_t A0, F o_at, int set, bool push) {
    for (int g = 0; g < NKV; g++) {
        const uint32_t rs = (uint32_t)(W0 + g * G.RS);
        const uint32_t ru = (uint32_t)(rs + REC0 + (size_t)(G.slot / KPU) * UNIT);
        seq->npu_dma_memcpy_nd(4, 1, S2MM, IT[g], npu_bd_id(4 + set), it_channel_0, {0, 0, 0, (uint32_t)((rs + RING + (G.p & 3) * 1024) / 4)}, {1, 1, 1, 256}, {0, 0, 0, 1}, -1, 0, true, normal_cache, 15, push);
        seq->npu_dma_memcpy_nd(4, 1, S2MM, IT[g], npu_bd_id(5 + set), it_channel_0, {0, 0, 0, (ru + (G.slot % KPU) * 16) / 4}, {1, 1, 16, 4}, {0, 0, 32, 1}, -1, 0, true, normal_cache, 15, push);
        seq->npu_dma_memcpy_nd(4, 1, S2MM, IT[g], npu_bd_id(7 + set), it_channel_0, {0, 0, 0, (ru + 2048 + (G.slot % KPU) * 256) / 4}, {1, 1, 1, 64}, {0, 0, 0, 1}, -1, 0, true, normal_cache, 15, push);
        seq->npu_dma_memcpy_nd(4, 0, S2MM, IT[g], npu_bd_id(6 + set), it_channel_0, {0, 0, 0, (uint32_t)(o_at(g) / 4)}, {1, 1, 1, 192}, {0, 0, 0, 1}, -1, 0, true, normal_cache, 15, push);
        seq->npu_dma_memcpy_nd(1, 0, MM2S, IT[g], npu_bd_id(0 + set), it_channel_0, {0, 0, 0, (uint32_t)(H0 + g * 128)}, {1, 1, 1, 128}, {0, 0, 0, 1}, -1, 0, false, normal_cache, 15, push);
        seq->npu_dma_memcpy_nd(1, 0, MM2S, IT[g], npu_bd_id(1 + set), it_channel_0, {0, 0, 0, (uint32_t)(Q0 + g * 1536)}, {1, 1, 2, 1536}, {0, 0, QD * 4, 1}, -1, 0, false, normal_cache, 15, push);
        seq->npu_dma_memcpy_nd(1, 0, MM2S, IT[g], npu_bd_id(2 + set), it_channel_0, {0, 0, 0, (uint32_t)A0}, {1, 1, 1, (uint32_t)A_LEN}, {0, 0, 0, 1}, -1, 0, false, normal_cache, 15, push);
        seq->npu_dma_memcpy_nd(1, 1, MM2S, IT[g], npu_bd_id(3 + set), it_channel_0, {0, 0, 0, rs}, {1, 1, 1, (uint32_t)(REC0 + (size_t)(G.nk + KPU - 1) / KPU * UNIT)}, {0, 0, 0, 1}, -1, 0, false, normal_cache, 15, push);
    }
}
}   // namespace attn

// One decode token's attention of a layer alone: io = [8 headers of 128 B][Q 3072 | G 3072 fp32][A][O 3072 bf16].
// run.exe attn [iters] [p] [global] [steps]: random data vs an fp64 reference of the host decode
static int attn_test(int iters, int p, int glob, int steps) {
    using namespace attn;
    const geom G(p, glob);
    constexpr size_t QG = 1024, AB = QG + 2 * QD * 4, OB = AB + A_N * 4, IO = OB + QD * 2 + 8192;
    xrt::device dev(0);
    npu_xclbin_manager mgr(npu_device::device_npu2, &dev, false);
    npu_app_manager* am = mgr.register_xclbin("build/xclbins/decode.xclbin");
    npu_app app = am->create_app();
    buffer<uint8_t> io = app.create_bo_buffer<uint8_t>(IO);
    buffer<uint8_t> wt = app.create_bo_buffer<uint8_t>(NKV * G.RS);
    memset(io.data(), 0, IO);
    memset(wt.data(), 0, wt.size());
    float* Q = (float*)(io.data() + QG);
    float* Gt = Q + QD;
    float* A = (float*)(io.data() + AB);
    for (int i = 0; i < QD; i++) { Q[i] = (float)nrand(); Gt[i] = (float)(2 * nrand()); }
    for (int i = 0; i < 1024; i++) { A[i] = (float)(0.5 * nrand()); A[1056 + i] = (float)nrand(); }
    for (int i = 0; i < 384; i++) A[2112 + i] = (float)nrand();
    float* misc = A + 2496;
    for (int i = 0; i < 24; i++) misc[i] = (float)(2 * nrand());
    for (int i = 0; i < 32; i++) { misc[96 + i] = (float)(0.3 * nrand()); misc[192 + i] = (float)(0.3 * nrand()); }
    fill(wt.data(), io.data(), G);
    io.sync_to_device();
    wt.sync_to_device();
    std::vector<double> o_ref;
    std::vector<uint16_t> rec_ref;
    reference(Q, Gt, A, wt.data(), G, o_ref, rec_ref);   // before the device overwrites the ring row and the record

    npu_sequence* seq = app.seq();
    seq->clear_cmds();
    // ATTN_PRE=1: step k + 1's BDs (set 8-15) are written before step k's waits, then pushed (one register write each;
    // the shim task queues hold 4, so the pushes wait for step k)
    const bool pre = getenv("ATTN_PRE") && atoi(getenv("ATTN_PRE"));
    auto o_at = [&](int g) { return OB + g * 768; };
    if (pre) issue(seq, G, 0, 0, QG, AB, o_at, 0, true);
    for (int k = 0; k < steps; k++) {
        if (!pre) issue(seq, G, 0, 0, QG, AB, o_at, 0, true);
        else if (k + 1 < steps) issue(seq, G, 0, 0, QG, AB, o_at, 8 * ((k + 1) & 1), false);
        for (int g = 0; g < NKV; g++)
            for (int i = 0; i < 4; i++) seq->npu_dma_wait(IT[g], S2MM, it_channel_0);
        if (pre && k + 1 < steps)
            for (int g = 0; g < NKV; g++)
                for (int i = 0; i < 8; i++)   // S2MM BDs 4 5 7 6 with the token bit, MM2S BDs 0-3
                    seq->rtp_write(IT[g], i < 4 ? 0x1d204 : 0x1d214, (uint32_t)(i < 4 ? (1u << 31) | ("\4\5\7\6"[i] + 8 * ((k + 1) & 1)) : i - 4 + 8 * ((k + 1) & 1)));
    }
    seq->cmds2seq();
    double us = 0;
    for (int it = 0; it <= iters; it++) {
        const auto t1 = std::chrono::steady_clock::now();
        xrt::run run = app.create_run(io, wt);
        run.start();
        ert_cmd_state st = ERT_CMD_STATE_TIMEOUT;
        st = wait_run(run, 5000);
        if (it > 0) us += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t1).count();
        if (st != ERT_CMD_STATE_COMPLETED) { std::printf("run state %d\n", (int)st); return 2; }
    }
    io.sync_from_device();
    wt.sync_from_device();
    const bool ok = check(wt.data(), A, (const uint16_t*)(io.data() + OB), G, o_ref, rec_ref);
    std::printf("%.1f us per run of %d steps\n", us / std::max(iters, 1), steps);
    return ok ? 0 : 1;
}

// One layer's attention half as three dependent steps of one run:
//   proj    proj_test's step (nx 3 global, 1 SWA); the y groups land as [K 11 | V 11 | R 4 | MISC 3 | Q 32 | G 32] units
//           of 96 floats, so the attention reads its A block in one piece and q / gate where they are
//   attn    attn_test's step on the 8 glue_up cores; O of head g goes into the x of the o projection's header unit g / 4
//   o proj  3072 -> 2304 (24 groups x 12 column blocks, Q4_K), split over K: channel c runs groups 3 (c % 8) .. + 2
//           over column blocks 6 (c / 8) .. + 5, one 3D BD over the [group][cb][3] weights; the halves add on the host
// run.exe chain [iters] [p] [global] [stages: 1 proj | 2 attn | 4 o proj; checked when 7]: synthetic weights, each stage
// against an fp64 reference computed from the previous stage's NPU output
static int chain_test(int iters, int p, int glob, int stages) {
    using namespace attn;
    constexpr int OP_PROJ = 6, OP_GEMV = 2, NG = 90, NC = 9, OG = 24, ONC = 12;
    const int nx = glob ? 3 : 1;
    const int ends[5] = {11, 22, 54, 86, 90};
    const int xi3[5] = {1, 2, 0, 0, 0};
    auto slot_of = [](int g) { return g < 22 ? g : g < 86 ? g + 7 : g - 64; };   // y slot of group g; MISC unit u: 26 + u
    constexpr int QS = 29;   // slot of the first q group
    const size_t GB = (size_t)NC * 3 * UNIT;
    const size_t HDR = (size_t)(X_OFF + nx * D * 2 + UNIT - 1) / UNIT * UNIT;
    const size_t PY = HDR, AH = PY + (size_t)(NG + 3) * Y_UNIT * 4, OH = AH + NKV * 128, OY = OH + 2 * UNIT;
    const size_t IO = OY + (size_t)2 * OG * Y_UNIT * 4;
    const size_t DW = NG * GB, OW = DW + 3 * 32 * UNIT, AW = OW + (size_t)OG * ONC * 3 * UNIT;
    const geom G(p, glob);
    struct dense { int c, n, x, u; };
    std::vector<dense> dns = {{15, 24, 0, 0}};
    if (nx == 3) { dns.push_back({0, 32, 1, 1}); dns.push_back({2, 32, 2, 2}); }
    int nd_of[16] = {};
    for (const dense& d : dns) nd_of[d.c] = (int)(&d - dns.data()) + 1;

    xrt::device dev(0);
    npu_xclbin_manager mgr(npu_device::device_npu2, &dev, false);
    npu_app_manager* am = mgr.register_xclbin("build/xclbins/decode.xclbin");
    npu_app app = am->create_app();
    buffer<uint8_t> io = app.create_bo_buffer<uint8_t>(IO);
    buffer<uint8_t> wt = app.create_bo_buffer<uint8_t>(AW + NKV * G.RS);
    memset(io.data(), 0, IO);
    memset(wt.data(), 0, wt.size());
    fill_q4k(wt.data(), NG * GB / UNIT);
    fill_q4k(wt.data() + OW, (size_t)OG * ONC * 3);
    std::vector<float> dwf(3 * 32 * D);
    for (int i = 0; i < 3 * 32; i++)
        for (int k = 0; k < D; k++) {
            const uint16_t b = f2bf((float)(0.05 * nrand()));
            dwf[(size_t)i * D + k] = bf2f(b);
            memcpy(wt.data() + DW + (size_t)i * UNIT + 2 * k, &b, 2);
        }
    fill(wt.data() + AW, io.data() + AH, G);
    const std::vector<uint8_t> att0(wt.data() + AW, wt.data() + AW + NKV * G.RS);   // the regions before the step
    wt.sync_to_device();

    // proj header: as proj_test
    std::vector<std::vector<float>> xv(nx, std::vector<float>(D));
    int* h = (int*)io.data();
    h[0] = OP_PROJ;
    h[1] = nx;
    int g0[16], ng[16];
    for (int c = 0; c < 16; c++) ng[c] = 6;
    int over = 16 * 6 - NG;
    for (const dense& d : dns) if (over > 0) { ng[d.c]--; over--; }
    for (int c = 15; over > 0; c--) if (ng[c] == 6) { ng[c]--; over--; }
    for (int c = 0, g = 0; c < 16; c++) {
        g0[c] = g;
        g += ng[c];
        h[2 + c] = g0[c] | ng[c] << 16;
    }
    for (int m = 0; m < 5; m++) h[18 + m] = ends[m] | (nx == 3 ? xi3[m] : 0) << 16;
    for (const dense& d : dns) ((int16_t*)h)[46 + d.c] = (int16_t)(d.n | d.x << 8);
    for (int i = 0; i < nx; i++)
        for (int k = 0; k < D; k++) {
            const uint16_t b = f2bf((float)nrand());
            xv[i][k] = bf2f(b);
            memcpy(io.data() + X_OFF + (size_t)(i * D + k) * 2, &b, 2);
        }
    // each channel's groups as runs of consecutive y slots (one S2MM BD each)
    std::vector<std::pair<int, int>> runs[16];
    for (int c = 0; c < 16; c++)
        for (int g = g0[c]; g < g0[c] + ng[c]; g++) {
            if (!runs[c].empty() && runs[c].back().first + runs[c].back().second == slot_of(g)) runs[c].back().second++;
            else runs[c].push_back({slot_of(g), 1});
        }
    for (int c = 0; c < 16; c++) if (runs[c].size() > 2) { std::printf("channel %d: %zu runs\n", c, runs[c].size()); return 3; }
    // o proj headers: [OP_GEMV, 3 groups, 6 column blocks], x = O of heads 4 h .. 4 h + 3
    for (int hu = 0; hu < 2; hu++) {
        int* oh = (int*)(io.data() + OH + hu * UNIT);
        oh[0] = OP_GEMV; oh[1] = 3; oh[2] = 6;
    }
    for (size_t i = PY; i < AH; i += 4) *(float*)(io.data() + i) = -1.0f;
    for (size_t i = OY; i < IO; i += 4) *(float*)(io.data() + i) = -1.0f;
    io.sync_to_device();

    npu_sequence* seq = app.seq();
    seq->clear_cmds();
    auto chn = [](int r) { return r ? it_channel_1 : it_channel_0; };
    if (stages & 1) {
        for (int c = 0; c < 16; c++) {
            const int col = c % 8, r = c / 8;
            for (size_t i = 0; i < runs[c].size(); i++)
                seq->npu_dma_memcpy_nd(4, 0, S2MM, IT[col], npu_bd_id((i ? 12 : 4) + r), chn(r), {0, 0, 0, (uint32_t)((PY + (size_t)runs[c][i].first * Y_UNIT * 4) / 4)}, {1, 1, 1, (uint32_t)(runs[c][i].second * Y_UNIT)}, {0, 0, 0, 1});
            seq->npu_dma_memcpy_nd(1, 0, MM2S, IT[col], npu_bd_id(2 * r), chn(r), {0, 0, 0, 0}, {1, 1, 1, (uint32_t)HDR}, {0, 0, 0, 1});
            seq->npu_dma_memcpy_nd(1, 1, MM2S, IT[col], npu_bd_id(2 * r + 1), chn(r), {0, 0, 0, (uint32_t)(g0[c] * GB)}, {1, 1, 1, (uint32_t)(ng[c] * GB)}, {0, 0, 0, 1});
            if (nd_of[c]) {
                const dense& d = dns[nd_of[c] - 1];
                seq->npu_dma_memcpy_nd(1, 1, MM2S, IT[col], npu_bd_id(8 + r), chn(r), {0, 0, 0, (uint32_t)(DW + (size_t)d.u * 32 * UNIT)}, {1, 1, 1, (uint32_t)(d.n * UNIT)}, {0, 0, 0, 1});
                seq->npu_dma_memcpy_nd(4, 0, S2MM, IT[col], npu_bd_id(10 + r), chn(r), {0, 0, 0, (uint32_t)((PY + (size_t)(26 + d.u) * Y_UNIT * 4) / 4)}, {1, 1, 1, (uint32_t)Y_UNIT}, {0, 0, 0, 1});
            }
        }
        for (int c = 0; c < 16; c++)
            for (size_t i = 0; i < runs[c].size() + (nd_of[c] ? 1 : 0); i++) seq->npu_dma_wait(IT[c % 8], S2MM, chn(c / 8));
    }
    if (stages & 2) {
        issue(seq, G, AW, AH, PY + (size_t)QS * Y_UNIT * 4, PY, [&](int g) { return OH + (size_t)(g / 4) * UNIT + X_OFF + (g % 4) * 768; }, 0, true);
        for (int g = 0; g < NKV; g++)
            for (int i = 0; i < 4; i++) seq->npu_dma_wait(IT[g], S2MM, it_channel_0);
    }
    if (stages & 4) {
        for (int c = 0; c < 16; c++) {
            const int col = c % 8, r = c / 8, hu = c / 8, j = c % 8;
            seq->npu_dma_memcpy_nd(4, 0, S2MM, IT[col], npu_bd_id(4 + r), chn(r), {0, 0, 0, (uint32_t)((OY + (size_t)(hu * OG + 3 * j) * Y_UNIT * 4) / 4)}, {1, 1, 1, (uint32_t)(3 * Y_UNIT)}, {0, 0, 0, 1});
            seq->npu_dma_memcpy_nd(1, 0, MM2S, IT[col], npu_bd_id(2 * r), chn(r), {0, 0, 0, (uint32_t)(OH + hu * UNIT)}, {1, 1, 1, (uint32_t)UNIT}, {0, 0, 0, 1});
            seq->npu_dma_memcpy_nd(4, 1, MM2S, IT[col], npu_bd_id(2 * r + 1), chn(r), {0, 0, 0, (uint32_t)((OW + (size_t)(3 * j * ONC + 6 * hu) * 3 * UNIT) / 4)}, {1, 3, 333, 64},
                                   {0, (uint32_t)(ONC * 3 * UNIT / 4), 64, 1});
        }
        for (int c = 0; c < 16; c++) seq->npu_dma_wait(IT[c % 8], S2MM, chn(c / 8));
    }
    seq->cmds2seq();
    double us = 0;
    for (int it = 0; it <= iters; it++) {
        const auto t1 = std::chrono::steady_clock::now();
        xrt::run run = app.create_run(io, wt);
        run.start();
        ert_cmd_state st = ERT_CMD_STATE_TIMEOUT;
        st = wait_run(run, 5000);
        if (it > 0) us += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t1).count();
        if (st != ERT_CMD_STATE_COMPLETED) {
            io.sync_from_device();
            int ny = 0, no = 0, noy = 0;
            for (int s = 0; s < NG + 3; s++) ny += *(float*)(io.data() + PY + (size_t)s * Y_UNIT * 4) != -1.0f;
            for (int g = 0; g < NKV; g++) no += *(uint32_t*)(io.data() + OH + (size_t)(g / 4) * UNIT + X_OFF + (g % 4) * 768) != 0;
            for (int s = 0; s < 2 * OG; s++) noy += *(float*)(io.data() + OY + (size_t)s * Y_UNIT * 4) != -1.0f;
            std::printf("run state %d; proj y slots written %d / %d, attention heads %d / 8, o proj y %d / %d\n", (int)st, ny, NG + 3, no, noy, 2 * OG);
            return 2;
        }
    }
    const double us1 = us / std::max(iters, 1);
    std::printf("chain p=%d %s stages %d: %.1f us per run (%d iters)\n", p, glob ? "global" : "swa", stages, us1, iters);
    if (stages != 7) return 0;
    io.sync_from_device();
    wt.sync_from_device();
    // proj
    double err2 = 0, ref2 = 0, worst = 0;
    std::vector<double> ref(96 * OG);
    for (int g = 0, m = 0; g < NG; g++) {
        while (g >= ends[m]) m++;
        ref_rows(ref.data(), wt.data(), g, 1, NC, xv[nx == 3 ? xi3[m] : 0].data());
        const float* y = (const float*)(io.data() + PY) + (size_t)slot_of(g) * Y_UNIT;
        double e2 = 0, r2 = 0;
        for (int i = 0; i < 96; i++) { const double d = y[i] - ref[i]; e2 += d * d; r2 += ref[i] * ref[i]; }
        err2 += e2; ref2 += r2;
        worst = std::max(worst, std::sqrt(e2 / r2));
    }
    double dworst = 0;
    for (const dense& d : dns) {
        const float* y = (const float*)(io.data() + PY) + (size_t)(26 + d.u) * Y_UNIT;
        for (int i = 0; i < d.n; i++) {
            double r = 0, a = 0;
            for (int k = 0; k < D; k++) { r += (double)dwf[((size_t)d.u * 32 + i) * D + k] * xv[d.x][k]; a += std::fabs(dwf[((size_t)d.u * 32 + i) * D + k] * xv[d.x][k]); }
            dworst = std::max(dworst, std::fabs(y[i] - r) / a);
        }
    }
    std::printf("proj: rel rms err %.3e (worst group %.3e), dense rows max err / sum |w x| %.3e\n", std::sqrt(err2 / ref2), worst, dworst);
    bool ok = worst < 1e-2 && dworst < 1e-5;
    // attention, from the NPU's projections
    const float* Y = (const float*)(io.data() + PY);
    std::vector<double> o_ref;
    std::vector<uint16_t> rec_ref;
    reference(Y + QS * Y_UNIT, Y + QS * Y_UNIT + QD, Y, att0.data(), G, o_ref, rec_ref);
    std::vector<uint16_t> O(QD);
    for (int g = 0; g < NKV; g++) memcpy(O.data() + g * 384, io.data() + OH + (size_t)(g / 4) * UNIT + X_OFF + (g % 4) * 768, 768);
    ok &= check(wt.data() + AW, Y, O.data(), G, o_ref, rec_ref);
    // o proj, from the NPU's O
    std::vector<float> xo(QD);
    for (int i = 0; i < QD; i++) xo[i] = bf2f(O[i]);
    ref_rows(ref.data(), wt.data() + OW, 0, OG, ONC, xo.data());
    err2 = ref2 = 0;
    for (int g = 0; g < OG; g++)
        for (int i = 0; i < 96; i++) {
            const float* y0 = (const float*)(io.data() + OY) + (size_t)g * Y_UNIT;
            const double d = (double)y0[i] + y0[(size_t)OG * Y_UNIT + i] - ref[96 * g + i];
            err2 += d * d; ref2 += ref[96 * g + i] * ref[96 * g + i];
        }
    std::printf("o proj: rel rms err %.3e (y[0] %.5f ref %.5f)\n", std::sqrt(err2 / ref2),
                *(const float*)(io.data() + OY) + *(const float*)(io.data() + OY + (size_t)OG * Y_UNIT * 4), ref[0]);
    ok &= std::sqrt(err2 / ref2) < 1e-2;
    std::printf("chain %s\n", ok ? "PASSED" : "FAILED");
    return ok ? 0 : 1;
}

// OP_MIX on glue_dn (1, 5) (column 1, channel 1): a layer's glue vs an fp64 reference.
// run.exe mix [case] [iters]: case 0 "A" (experts, conv, residual, 4 states, 3 normed queries, bf16),
// 1 "F" (o halves, conv, residual, 3 states, 1 normed query, bf16 hi lo hi), 2 "M" (cur, 1 state, 48 fp32 queries),
// 3 "final" (experts, conv, residual, 4 states, 1 normed query, bf16)
struct mix_cfg { int pre, fl, S, nq, om, nn; };

// fp64 reference of OP_MIX from the bytes it read: states S [S][DM], prefix P, block B = [conv block | residual | queries]
// -> raw (the prefix, the new conv ring row), cur (after conv + residual), one output per query
static void mix_reference(const mix_cfg& C, int p, const float* S, const float* P, const float* B, std::vector<double>& raw,
                          std::vector<double>& cur, std::vector<std::vector<double>>& outs) {
    constexpr int DM = 2304, CH = 24;
    const size_t CB = (C.fl & 1) ? (size_t)CH * 8 * Y_UNIT : 0, RB = (C.fl & 2) ? (size_t)DM : 0;
    cur.assign(DM, 0.0);
    if (C.pre == 1) {
        for (int s = 0; s < 16; s++)
            for (int k = 0; k < 1152; k++) cur[(s >> 3) * 1152 + k] += (double)P[8 + (s & 7)] * P[Y_UNIT + (size_t)s * 1152 + k];
    } else
        for (int c = 0; c < DM; c++) cur[c] = P[c] + (C.pre == 2 ? P[DM + c] : 0.0);
    raw = cur;
    if (C.fl & 1)
        for (int c = 0; c < DM; c++) {
            const float* blk = B + (size_t)(c / Y_UNIT) * 8 * Y_UNIT + c % Y_UNIT;
            double v = raw[c] + blk[3 * Y_UNIT] * raw[c];
            for (int k = 0; k < 3; k++) v += (double)blk[k * Y_UNIT] * blk[(4 + ((p - 3 + k) & 3)) * Y_UNIT];
            cur[c] = v / std::sqrt(5.0);
        }
    if (C.fl & 2)
        for (int c = 0; c < DM; c++) cur[c] += B[CB + c];
    std::vector<const double*> xs;
    std::vector<std::vector<double>> st(C.S, std::vector<double>(DM));
    for (int i = 0; i < C.S; i++) { for (int c = 0; c < DM; c++) st[i][c] = S[(size_t)i * DM + c]; xs.push_back(st[i].data()); }
    xs.push_back(cur.data());
    outs.clear();
    for (int j = 0; j < C.nq; j++) {
        const float* q = B + CB + RB + (size_t)j * (1 + C.nn) * DM;
        std::vector<double> w(xs.size()), inv(xs.size());
        double mx = -1e300, sum = 0;
        for (size_t i = 0; i < xs.size(); i++) {
            double ss = 0, d = 0;
            for (int c = 0; c < DM; c++) { ss += xs[i][c] * xs[i][c]; d += q[c] * xs[i][c]; }
            inv[i] = 1.0 / std::sqrt(ss / DM + 1e-6);
            w[i] = d * inv[i] / 48.0;
            mx = std::max(mx, w[i]);
        }
        for (auto& v : w) sum += (v = std::exp(v - mx));
        std::vector<double> o(DM, 0.0);
        for (size_t i = 0; i < xs.size(); i++) for (int c = 0; c < DM; c++) o[c] += w[i] / sum * inv[i] * xs[i][c];
        if (C.nn) {
            double ss = 0;
            for (double v : o) ss += v * v;
            const double r = 1.0 / std::sqrt(ss / DM + 1e-6);
            for (int c = 0; c < DM; c++) o[c] *= r * q[DM + c];
        }
        outs.push_back(o);
    }
}

static double rel_rms(const double* ref, const std::function<double(int)>& get, int n) {
    double e2 = 0, r2 = 0;
    for (int i = 0; i < n; i++) { const double d = get(i) - ref[i]; e2 += d * d; r2 += ref[i] * ref[i]; }
    return std::sqrt(e2 / r2);
}

static int mix_test(int cs, int iters, const int* cust = nullptr) {
    constexpr int OP_MIX = 18, DM = 2304, CH = 24;
    using cfg = mix_cfg;
    const cfg cases[4] = {{1, 7, 4, 3, 1, 1}, {2, 7, 3, 1, 2, 1}, {0, 0, 1, 48, 0, 0}, {1, 3, 4, 1, 1, 1}};
    const cfg C = cust ? cfg{cust[0], cust[1], cust[2], cust[3], cust[4], cust[5]} : cases[cs];
    // only combinations whose host transfer sizes match what the kernel reads / writes (a mismatch can hang the NPU)
    const bool valid = C.pre >= 0 && C.pre <= 2 && C.fl >= 0 && C.fl <= 7 && C.S >= 0 && C.S <= 4 && C.nq >= 0 &&
                       C.om >= 0 && C.om <= 2 && (C.nn == 0 || C.nn == 1) && !(C.om == 0 && C.nn) &&
                       !(C.om == 2 && C.nq > 1) && ((C.fl & 4) || C.nq > 0);
    if (!valid) { std::printf("mix: invalid case pre %d fl %d S %d nq %d om %d nn %d\n", C.pre, C.fl, C.S, C.nq, C.om, C.nn); return 3; }
    const int p = 7;
    const size_t SB = (size_t)C.S * DM * 4;
    const size_t PB = C.pre == 1 ? (size_t)(1 + 192) * Y_UNIT * 4 : (size_t)(C.pre == 2 ? 2 : 1) * DM * 4;
    const size_t CB = (C.fl & 1) ? (size_t)CH * 8 * Y_UNIT * 4 : 0, RB = (C.fl & 2) ? (size_t)DM * 4 : 0;
    const size_t QB = (size_t)C.nq * (1 + C.nn) * DM * 4;
    const size_t total = X_OFF + SB + PB + CB + RB + QB, padded = (total + UNIT - 1) / UNIT * UNIT;
    const size_t BB = CB + RB + QB + (padded - total);   // the block task, padded
    const size_t H0 = 0, S0 = UNIT, P0 = S0 + SB, B0 = P0 + PB, O0 = (B0 + BB + 4095) / 4096 * 4096;
    const int n_out = (C.fl & 1 ? 24 : 0) + (C.fl & 4 ? 24 : 0) + C.nq * (C.om == 0 ? 24 : C.om == 1 ? 12 : 36);
    const size_t IO = O0 + (size_t)(n_out + 1) * Y_UNIT * 4;
    xrt::device dev(0);
    npu_xclbin_manager mgr(npu_device::device_npu2, &dev, false);
    npu_app_manager* am = mgr.register_xclbin("build/xclbins/decode.xclbin");
    npu_app app = am->create_app();
    buffer<uint8_t> io = app.create_bo_buffer<uint8_t>(IO);
    memset(io.data(), 0, IO);
    int* h = (int*)(io.data() + H0);
    h[0] = OP_MIX; h[1] = C.pre; h[2] = C.fl; h[3] = p; h[4] = C.S; h[5] = C.nq; h[6] = C.om; h[7] = C.nn;
    h[8] = getenv("MIX_SKIP") ? atoi(getenv("MIX_SKIP")) & 7 : 0;   // compute-only skips (timing), I/O unchanged
    auto F = [&](size_t off) { return (float*)(io.data() + off); };
    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0, 1);
    // states: different scales so the rms matters
    for (int i = 0; i < C.S; i++)
        for (int c = 0; c < DM; c++) F(S0)[(size_t)i * DM + c] = (0.5f + i) * nd(rng) + 0.2f * i;
    float* tw = F(P0) + 8;
    if (C.pre == 1) {
        for (int e = 0; e < 8; e++) { ((int*)F(P0))[e] = e * 3; tw[e] = 0.3f + 0.2f * e; }
        for (int k = 0; k < 192 * Y_UNIT; k++) F(P0)[Y_UNIT + k] = 0.5f * nd(rng);
    } else
        for (size_t k = 0; k < PB / 4; k++) F(P0)[k] = nd(rng);
    for (size_t k = 0; k < (CB + RB + QB) / 4; k++) F(B0)[k] = (k < CB / 4 ? 0.3f : 1.0f) * nd(rng);
    for (int j = 0; j < C.nq && C.nn; j++)   // norms around 1
        for (int c = 0; c < DM; c++) F(B0 + CB + RB)[(size_t)(2 * j + 1) * DM + c] = 1.0f + 0.1f * nd(rng);
    const std::vector<uint8_t> in0(io.data(), io.data() + O0);
    for (size_t i = O0; i < IO; i += 4) *(float*)(io.data() + i) = -1.0f;
    io.sync_to_device();

    npu_sequence* seq = app.seq();
    seq->clear_cmds();
    const int col = 1;
    const npu_it_channel ch = it_channel_1;
    if (C.fl & 1)   // raw cur into ring row p mod 4: [24 chunks][taps 4 | ring 4][96]
        seq->npu_dma_memcpy_nd(4, 0, S2MM, IT[col], npu_bd_id(4), ch, {0, 0, 0, (uint32_t)((B0 + (size_t)(4 + (p & 3)) * Y_UNIT * 4) / 4)}, {1, 1, CH, Y_UNIT}, {0, 0, 8 * Y_UNIT, 1});
    seq->npu_dma_memcpy_nd(4, 0, S2MM, IT[col], npu_bd_id(5), ch, {0, 0, 0, (uint32_t)(O0 / 4)}, {1, 1, 1, (uint32_t)((n_out - (C.fl & 1 ? 24 : 0)) * Y_UNIT)}, {0, 0, 0, 1});
    seq->npu_dma_memcpy_nd(1, 0, MM2S, IT[col], npu_bd_id(0), ch, {0, 0, 0, (uint32_t)H0}, {1, 1, 1, (uint32_t)X_OFF}, {0, 0, 0, 1});
    if (C.S) seq->npu_dma_memcpy_nd(1, 0, MM2S, IT[col], npu_bd_id(1), ch, {0, 0, 0, (uint32_t)S0}, {1, 1, 1, (uint32_t)SB}, {0, 0, 0, 1});
    seq->npu_dma_memcpy_nd(1, 0, MM2S, IT[col], npu_bd_id(2), ch, {0, 0, 0, (uint32_t)P0}, {1, 1, 1, (uint32_t)PB}, {0, 0, 0, 1});
    seq->npu_dma_memcpy_nd(1, 0, MM2S, IT[col], npu_bd_id(3), ch, {0, 0, 0, (uint32_t)B0}, {1, 1, 1, (uint32_t)BB}, {0, 0, 0, 1});
    if (C.fl & 1) seq->npu_dma_wait(IT[col], S2MM, ch);
    seq->npu_dma_wait(IT[col], S2MM, ch);
    seq->cmds2seq();
    double us = 0;
    for (int it = 0; it <= iters; it++) {
        if (it) { memcpy(io.data(), in0.data(), O0); io.sync_to_device(); }
        const auto t1 = std::chrono::steady_clock::now();
        xrt::run run = app.create_run(io);
        run.start();
        ert_cmd_state st = ERT_CMD_STATE_TIMEOUT;
        st = wait_run(run, 3000);
        if (it > 0) us += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t1).count();
        if (st != ERT_CMD_STATE_COMPLETED) {
            io.sync_from_device();
            int n = 0;
            for (int u = 0; u < n_out; u++) n += *(float*)(io.data() + O0 + (size_t)u * Y_UNIT * 4) != -1.0f;
            std::printf("run state %d, out units %d / %d\n", (int)st, n, n_out);
            return 2;
        }
    }
    io.sync_from_device();
    // reference
    const float* in = (const float*)in0.data();
    std::vector<double> raw, cur;
    std::vector<std::vector<double>> outs;
    mix_reference(C, p, in + S0 / 4, in + P0 / 4, in + B0 / 4, raw, cur, outs);
    // compare
    bool ok = true;
    const auto& rel = rel_rms;
    const float* O = (const float*)(io.data() + O0);
    if (C.fl & 1) {
        const double e = rel(raw.data(), [&](int c) { return (double)F(B0)[(size_t)(c / Y_UNIT) * 8 * Y_UNIT + (4 + (p & 3)) * Y_UNIT + c % Y_UNIT]; }, DM);
        std::printf("raw ring row: rel rms err %.3e\n", e);
        ok &= e < 1e-4;
    }
    if (C.fl & 4) {
        const double e = rel(cur.data(), [&](int c) { return (double)O[c]; }, DM);
        std::printf("cur: rel rms err %.3e\n", e);
        ok &= e < 1e-4;
        O += DM;
    }
    for (int j = 0; j < C.nq; j++) {
        double e;
        if (C.om == 0) { e = rel(outs[j].data(), [&](int c) { return (double)O[c]; }, DM); O += DM; }
        else if (C.om == 1) { const uint16_t* b = (const uint16_t*)O; e = rel(outs[j].data(), [&](int c) { return (double)bf2f(b[c]); }, DM); O += DM / 2; }
        else {
            const uint16_t* b = (const uint16_t*)O;
            e = rel(outs[j].data(), [&](int c) { return (double)bf2f(b[c]) + bf2f(b[DM + c]); }, DM);
            double e2 = 0;
            for (int c = 0; c < DM; c++) e2 += b[c] != b[2 * DM + c];
            std::printf("hi copy mismatches %g\n", e2);
            ok &= e2 == 0;
            O += 3 * DM / 2;
        }
        if (j < 3 || j == C.nq - 1) std::printf("query %d: rel rms err %.3e (o[0] %.5f)\n", j, e, outs[j][0]);
        ok &= e < (C.om == 1 ? 6e-3 : 1e-4);
    }
    if (cust) std::printf("mixc pre %d fl %d S %d nq %d om %d nn %d: ", C.pre, C.fl, C.S, C.nq, C.om, C.nn);
    std::printf("mix case %d: %.1f us per run (%d iters) %s\n", cs, us / std::max(iters, 1), iters, ok ? "PASSED" : "FAILED");
    return ok ? 0 : 1;
}

// One layer application in one run: A mix (the previous layer's experts, ffn conv, residual, block mix, norm) -> proj ->
// attention -> o proj -> F mix (o halves, attn conv, residual, block mix, norm) -> router -> top-k -> experts.
// run.exe layer [iters] [p] [global] [stages]: stage bits 32 A mix, 1 proj, 2 attention, 4 o proj, 8 F mix, 16 MoE.
// Synthetic weights; each step that ran is checked against an fp64 reference of the bytes that step read. The steps
// pass data through io only: A's queries land in the proj header, A's cur in F's residual slot, F's cur in A's (the
// next layer's) residual slot, F's bf16 hi / lo / hi after the expert header (the router reads its x from there), and
// the top-k unit + expert y units are A's prefix. Every step finishes (sequence waits) before the next one starts.
static int layer_test(int iters, int p, int glob, int stages) {
    using namespace attn;
    constexpr int OP_PROJ = 6, OP_GEMV = 2, OP_MIX = 18, NG = 90, NC = 9, OG = 24, ONC = 12, CH = 24, DM = 2304;
    const int nx = glob ? 3 : 1;
    const mix_cfg CA{1, 7, 4, nx, 1, 1}, CF{2, 7, 3, 1, 2, 1};
    const int ends[5] = {11, 22, 54, 86, 90};
    const int xi3[5] = {1, 2, 0, 0, 0};
    auto slot_of = [](int g) { return g < 22 ? g : g < 86 ? g + 7 : g - 64; };
    constexpr int QS = 29;
    const size_t GB = (size_t)NC * 3 * UNIT;
    const size_t HDR = (size_t)(X_OFF + nx * D * 2 + UNIT - 1) / UNIT * UNIT;
    const size_t PY = HDR, AH = PY + (size_t)(NG + 3) * Y_UNIT * 4, OH = AH + NKV * 128, OY = OH + 2 * UNIT;
    auto up = [](size_t x) { return (x + 4095) / 4096 * 4096; };
    // mix regions: block B = [conv block | residual | queries (+ norm g)], padded so header + S + P + B is whole units
    const size_t CB = (size_t)CH * 8 * Y_UNIT * 4, RES = (size_t)DM * 4;
    const size_t PBA = (size_t)193 * Y_UNIT * 4, PBF = (size_t)2 * DM * 4;
    auto bsize = [&](const mix_cfg& C, size_t PB) {
        const size_t QB = (size_t)C.nq * (1 + C.nn) * DM * 4, total = X_OFF + (size_t)C.S * DM * 4 + PB + CB + RES + QB;
        return CB + RES + QB + ((total + UNIT - 1) / UNIT * UNIT - total);
    };
    const size_t BBA = bsize(CA, PBA), BBF = bsize(CF, PBF);
    const size_t ST = up(OY + (size_t)2 * OG * Y_UNIT * 4), MHA = ST + (size_t)4 * DM * 4, MHF = MHA + 128;
    const size_t BA = up(MHF + 128), BF = up(BA + BBA);
    // MoE: router header (op only; x comes from after the expert header), expert header + F's hi lo hi, top-k
    const size_t R_HDR = up(BF + BBF), E_HDR = R_HDR + UNIT, T_HDR = E_HDR + 3 * UNIT, RY = T_HDR + UNIT;
    const size_t TY = RY + 2 * UNIT, EY = TY + Y_UNIT * 4, IO = EY + (size_t)16 * 12 * Y_UNIT * 4;
    const size_t DW = NG * GB, OW = DW + 3 * 32 * UNIT, AW = OW + (size_t)OG * ONC * 3 * UNIT;
    const geom G(p, glob);
    const size_t RW = AW + NKV * G.RS;
    struct dense { int c, n, x, u; };
    std::vector<dense> dns = {{15, 24, 0, 0}};
    if (nx == 3) { dns.push_back({0, 32, 1, 1}); dns.push_back({2, 32, 2, 2}); }
    int nd_of[16] = {};
    for (const dense& d : dns) nd_of[d.c] = (int)(&d - dns.data()) + 1;

    xrt::device dev(0);
    npu_xclbin_manager mgr(npu_device::device_npu2, &dev, false);
    npu_app_manager* am = mgr.register_xclbin("build/xclbins/decode.xclbin");
    npu_app app = am->create_app();
    buffer<uint8_t> io = app.create_bo_buffer<uint8_t>(IO);
    buffer<uint8_t> wt = app.create_bo_buffer<uint8_t>(RW + (size_t)16 * 16 * UNIT);
    buffer<uint8_t> gu = app.create_bo_buffer<uint8_t>((size_t)NE * GU_E);
    buffer<uint8_t> dn = app.create_bo_buffer<uint8_t>((size_t)NE * DN_E);
    memset(io.data(), 0, IO);
    memset(wt.data(), 0, wt.size());
    fill_q4k(wt.data(), NG * GB / UNIT);
    fill_q4k(wt.data() + OW, (size_t)OG * ONC * 3);
    std::vector<float> dwf(3 * 32 * D);
    for (int i = 0; i < 3 * 32; i++)
        for (int k = 0; k < D; k++) {
            const uint16_t b = f2bf((float)(0.05 * nrand()));
            dwf[(size_t)i * D + k] = bf2f(b);
            memcpy(wt.data() + DW + (size_t)i * UNIT + 2 * k, &b, 2);
        }
    fill(wt.data() + AW, io.data() + AH, G);
    const std::vector<uint8_t> att0(wt.data() + AW, wt.data() + AW + NKV * G.RS);
    // router W (fp32 as bf16 hi + lo, layout of moe_test), bias
    std::vector<float> W((size_t)NE * D), bias(NE);
    for (auto& v : W) v = (float)(0.02 * nrand());
    for (auto& v : bias) v = (float)(0.05 * nrand());
    for (int c = 0; c < 16; c++)
        for (int kc = 0; kc < 36; kc++)
            for (int r = 0; r < 8; r++) {
                uint16_t* g = (uint16_t*)(wt.data() + RW + (size_t)(c * 16 + 2 * r + kc / 18) * UNIT + (size_t)(kc % 18) * 256);
                for (int k = 0; k < 64; k++) {
                    const float w = W[(size_t)(8 * c + r) * D + 64 * kc + k];
                    g[k] = f2bf(w);
                    g[64 + k] = f2bf(w - bf2f(g[k]));
                }
            }
    wt.sync_to_device();
    fill_q4k(gu.data(), (size_t)NE * GU_E / UNIT);
    fill_q4k(dn.data(), (size_t)NE * DN_E / UNIT);
    gu.sync_to_device();
    dn.sync_to_device();

    // proj header (x: random, or A's queries)
    int* h = (int*)io.data();
    h[0] = OP_PROJ;
    h[1] = nx;
    int g0[16], ng[16];
    for (int c = 0; c < 16; c++) ng[c] = 6;
    int over = 16 * 6 - NG;
    for (const dense& d : dns) if (over > 0) { ng[d.c]--; over--; }
    for (int c = 15; over > 0; c--) if (ng[c] == 6) { ng[c]--; over--; }
    for (int c = 0, g = 0; c < 16; c++) {
        g0[c] = g;
        g += ng[c];
        h[2 + c] = g0[c] | ng[c] << 16;
    }
    for (int m = 0; m < 5; m++) h[18 + m] = ends[m] | (nx == 3 ? xi3[m] : 0) << 16;
    for (const dense& d : dns) ((int16_t*)h)[46 + d.c] = (int16_t)(d.n | d.x << 8);
    for (int k = 0; k < nx * D; k++) { const uint16_t b = f2bf((float)nrand()); memcpy(io.data() + X_OFF + (size_t)k * 2, &b, 2); }
    std::vector<std::pair<int, int>> runs[16];
    for (int c = 0; c < 16; c++)
        for (int g = g0[c]; g < g0[c] + ng[c]; g++) {
            if (!runs[c].empty() && runs[c].back().first + runs[c].back().second == slot_of(g)) runs[c].back().second++;
            else runs[c].push_back({slot_of(g), 1});
        }
    for (int c = 0; c < 16; c++) if (runs[c].size() > 2) { std::printf("channel %d: %zu runs\n", c, runs[c].size()); return 3; }
    for (int hu = 0; hu < 2; hu++) {
        int* oh = (int*)(io.data() + OH + hu * UNIT);
        oh[0] = OP_GEMV; oh[1] = 3; oh[2] = 6;
    }
    for (size_t i = PY; i < AH; i += 4) *(float*)(io.data() + i) = -1.0f;
    for (size_t i = OY; i < OY + (size_t)2 * OG * Y_UNIT * 4; i += 4) *(float*)(io.data() + i) = -1.0f;
    // mixes: headers, block states, A's prefix (top-k unit + expert y), blocks
    auto Fp = [&](size_t off) { return (float*)(io.data() + off); };
    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0, 1);
    for (int k = 0; k < 2; k++) {
        const mix_cfg& C = k ? CF : CA;
        int* mh = (int*)(io.data() + (k ? MHF : MHA));
        mh[0] = OP_MIX; mh[1] = C.pre; mh[2] = C.fl; mh[3] = p; mh[4] = C.S; mh[5] = C.nq; mh[6] = C.om; mh[7] = C.nn; mh[8] = 0;
        const size_t B0 = k ? BF : BA, QB = (size_t)C.nq * (1 + C.nn) * DM * 4;
        for (size_t i = 0; i < (CB + RES + QB) / 4; i++) Fp(B0)[i] = (i < CB / 4 ? 0.3f : 1.0f) * nd(rng);
        for (int j = 0; j < C.nq; j++)
            for (int c = 0; c < DM; c++) Fp(B0 + CB + RES)[(size_t)(2 * j + 1) * DM + c] = 1.0f + 0.1f * nd(rng);
    }
    for (int i = 0; i < 4; i++)
        for (int c = 0; c < DM; c++) Fp(ST)[(size_t)i * DM + c] = (0.5f + i) * nd(rng) + 0.2f * i;
    for (int e = 0; e < 8; e++) { ((int*)Fp(TY))[e] = e * 3; Fp(TY)[8 + e] = 0.3f + 0.2f * e; }
    for (int k = 0; k < 192 * Y_UNIT; k++) Fp(EY)[k] = 0.5f * nd(rng);
    // MoE headers; F's output region starts as random x hi / lo / hi
    ((int*)(io.data() + R_HDR))[0] = OP_ROUTER;
    ((int*)(io.data() + E_HDR))[0] = OP_EXPERT;
    ((int*)(io.data() + T_HDR))[0] = OP_TOPK;
    memcpy(io.data() + T_HDR + X_OFF, bias.data(), 4 * NE);
    for (int i = 0; i < D; i++) {
        const float x = (float)nrand();
        uint16_t* xo = (uint16_t*)(io.data() + E_HDR + X_OFF);
        xo[i] = xo[2 * D + i] = f2bf(x);
        xo[D + i] = f2bf(x - bf2f(xo[i]));
    }
    const std::vector<uint8_t> in0(io.data(), io.data() + IO);
    io.sync_to_device();

    const uint64_t gu_a = gu.bo().address() + 0x80000000ull, dn_a = dn.bo().address() + 0x80000000ull;
    npu_sequence* seq = app.seq();
    seq->clear_cmds();
    auto chn = [](int r) { return r ? it_channel_1 : it_channel_0; };
    // a mix on glue_dn (1, 5): S2MM ring row (raw prefix) into its conv block, cur, the outputs; then the inputs
    auto mix = [&](const mix_cfg& C, size_t MH, size_t P0, size_t PB, size_t B0, size_t BB, size_t cur_at, size_t out_at, int n_out) {
        const npu_it_channel ch = it_channel_1;
        seq->npu_dma_memcpy_nd(4, 0, S2MM, IT[1], npu_bd_id(4), ch, {0, 0, 0, (uint32_t)((B0 + (size_t)(4 + (p & 3)) * Y_UNIT * 4) / 4)}, {1, 1, CH, Y_UNIT}, {0, 0, 8 * Y_UNIT, 1});
        seq->npu_dma_memcpy_nd(4, 0, S2MM, IT[1], npu_bd_id(5), ch, {0, 0, 0, (uint32_t)(cur_at / 4)}, {1, 1, 1, (uint32_t)(24 * Y_UNIT)}, {0, 0, 0, 1});
        seq->npu_dma_memcpy_nd(4, 0, S2MM, IT[1], npu_bd_id(6), ch, {0, 0, 0, (uint32_t)(out_at / 4)}, {1, 1, 1, (uint32_t)(n_out * Y_UNIT)}, {0, 0, 0, 1});
        seq->npu_dma_memcpy_nd(1, 0, MM2S, IT[1], npu_bd_id(0), ch, {0, 0, 0, (uint32_t)MH}, {1, 1, 1, (uint32_t)X_OFF}, {0, 0, 0, 1});
        seq->npu_dma_memcpy_nd(1, 0, MM2S, IT[1], npu_bd_id(1), ch, {0, 0, 0, (uint32_t)ST}, {1, 1, 1, (uint32_t)((size_t)C.S * DM * 4)}, {0, 0, 0, 1});
        seq->npu_dma_memcpy_nd(1, 0, MM2S, IT[1], npu_bd_id(2), ch, {0, 0, 0, (uint32_t)P0}, {1, 1, 1, (uint32_t)PB}, {0, 0, 0, 1});
        seq->npu_dma_memcpy_nd(1, 0, MM2S, IT[1], npu_bd_id(3), ch, {0, 0, 0, (uint32_t)B0}, {1, 1, 1, (uint32_t)BB}, {0, 0, 0, 1});
        for (int i = 0; i < 3; i++) seq->npu_dma_wait(IT[1], S2MM, ch);
    };
    if (stages & 32) mix(CA, MHA, TY, PBA, BA, BBA, BF + CB, X_OFF, nx * 12);
    if (stages & 1) {
        for (int c = 0; c < 16; c++) {
            const int col = c % 8, r = c / 8;
            for (size_t i = 0; i < runs[c].size(); i++)
                seq->npu_dma_memcpy_nd(4, 0, S2MM, IT[col], npu_bd_id((i ? 12 : 4) + r), chn(r), {0, 0, 0, (uint32_t)((PY + (size_t)runs[c][i].first * Y_UNIT * 4) / 4)}, {1, 1, 1, (uint32_t)(runs[c][i].second * Y_UNIT)}, {0, 0, 0, 1});
            seq->npu_dma_memcpy_nd(1, 0, MM2S, IT[col], npu_bd_id(2 * r), chn(r), {0, 0, 0, 0}, {1, 1, 1, (uint32_t)HDR}, {0, 0, 0, 1});
            seq->npu_dma_memcpy_nd(1, 1, MM2S, IT[col], npu_bd_id(2 * r + 1), chn(r), {0, 0, 0, (uint32_t)(g0[c] * GB)}, {1, 1, 1, (uint32_t)(ng[c] * GB)}, {0, 0, 0, 1});
            if (nd_of[c]) {
                const dense& d = dns[nd_of[c] - 1];
                seq->npu_dma_memcpy_nd(1, 1, MM2S, IT[col], npu_bd_id(8 + r), chn(r), {0, 0, 0, (uint32_t)(DW + (size_t)d.u * 32 * UNIT)}, {1, 1, 1, (uint32_t)(d.n * UNIT)}, {0, 0, 0, 1});
                seq->npu_dma_memcpy_nd(4, 0, S2MM, IT[col], npu_bd_id(10 + r), chn(r), {0, 0, 0, (uint32_t)((PY + (size_t)(26 + d.u) * Y_UNIT * 4) / 4)}, {1, 1, 1, (uint32_t)Y_UNIT}, {0, 0, 0, 1});
            }
        }
        for (int c = 0; c < 16; c++)
            for (size_t i = 0; i < runs[c].size() + (nd_of[c] ? 1 : 0); i++) seq->npu_dma_wait(IT[c % 8], S2MM, chn(c / 8));
    }
    if (stages & 2) {
        issue(seq, G, AW, AH, PY + (size_t)QS * Y_UNIT * 4, PY, [&](int g) { return OH + (size_t)(g / 4) * UNIT + X_OFF + (g % 4) * 768; }, 0, true);
        for (int g = 0; g < NKV; g++)
            for (int i = 0; i < 4; i++) seq->npu_dma_wait(IT[g], S2MM, it_channel_0);
    }
    if (stages & 4) {
        for (int c = 0; c < 16; c++) {
            const int col = c % 8, r = c / 8, hu = c / 8, j = c % 8;
            seq->npu_dma_memcpy_nd(4, 0, S2MM, IT[col], npu_bd_id(4 + r), chn(r), {0, 0, 0, (uint32_t)((OY + (size_t)(hu * OG + 3 * j) * Y_UNIT * 4) / 4)}, {1, 1, 1, (uint32_t)(3 * Y_UNIT)}, {0, 0, 0, 1});
            seq->npu_dma_memcpy_nd(1, 0, MM2S, IT[col], npu_bd_id(2 * r), chn(r), {0, 0, 0, (uint32_t)(OH + hu * UNIT)}, {1, 1, 1, (uint32_t)UNIT}, {0, 0, 0, 1});
            seq->npu_dma_memcpy_nd(4, 1, MM2S, IT[col], npu_bd_id(2 * r + 1), chn(r), {0, 0, 0, (uint32_t)((OW + (size_t)(3 * j * ONC + 6 * hu) * 3 * UNIT) / 4)}, {1, 3, 333, 64},
                                   {0, (uint32_t)(ONC * 3 * UNIT / 4), 64, 1});
        }
        for (int c = 0; c < 16; c++) seq->npu_dma_wait(IT[c % 8], S2MM, chn(c / 8));
    }
    if (stages & 8) mix(CF, MHF, OY, PBF, BF, BBF, BA + CB, E_HDR + X_OFF, 36);
    if (stages & 16) {
        // the BDs the hub points at the experts (linear, not started), hub args, start; as moe_test
        for (int col = 0; col < 8; col++)
            for (int b = 12; b < 16; b++)
                seq->npu_dma_memcpy_nd(1, 0, MM2S, IT[col], npu_bd_id(b), chn(b & 1), {0, 0, 0, 0}, {1, 1, 1, (uint32_t)UNIT}, {0, 0, 0, 1}, -1, 0, false, normal_cache, 15, false);
        seq->npu_maskwrite(HUB, 0x32038, 1, 1);
        seq->rtp_write(HUB, HUB_ARGS + 4 * 4, 0);
        seq->rtp_write(HUB, HUB_ARGS + 4 * 11, (uint32_t)gu_a);
        seq->rtp_write(HUB, HUB_ARGS + 4 * 12, (uint32_t)(gu_a >> 32));
        seq->rtp_write(HUB, HUB_ARGS + 4 * 13, (uint32_t)dn_a);
        seq->rtp_write(HUB, HUB_ARGS + 4 * 14, (uint32_t)(dn_a >> 32));
        seq->rtp_write(HUB, HUB_ARGS + 4 * 15, 1);
        seq->rtp_write(HUB, HUB_START_LOCK, 1);
        // router: header unit start (op), then x hi / lo (+ 128 B pad) from after the expert header, then W rows
        for (int c = 0; c < 16; c++) {
            const int col = c % 8, r = c / 8;
            seq->npu_dma_memcpy_nd(4, 0, S2MM, IT[col], npu_bd_id(8 + r), chn(r), {0, 0, 0, (uint32_t)((RY + c * Y_UNIT * 4) / 4)}, {1, 1, 1, (uint32_t)Y_UNIT}, {0, 0, 0, 1});
            seq->npu_dma_memcpy_nd(1, 0, MM2S, IT[col], npu_bd_id(2 * r), chn(r), {0, 0, 0, (uint32_t)R_HDR}, {1, 1, 1, (uint32_t)X_OFF}, {0, 0, 0, 1});
            seq->npu_dma_memcpy_nd(1, 0, MM2S, IT[col], npu_bd_id(4 + r), chn(r), {0, 0, 0, (uint32_t)(E_HDR + X_OFF)}, {1, 1, 1, (uint32_t)(2 * UNIT - X_OFF)}, {0, 0, 0, 1});
            seq->npu_dma_memcpy_nd(1, 1, MM2S, IT[col], npu_bd_id(2 * r + 1), chn(r), {0, 0, 0, (uint32_t)(RW + c * 16 * UNIT)}, {1, 1, 1, (uint32_t)(16 * UNIT)}, {0, 0, 0, 1});
        }
        for (int c = 0; c < 16; c++) seq->npu_dma_wait(IT[c % 8], S2MM, chn(c / 8));
        seq->npu_dma_memcpy_nd(4, 0, S2MM, IT[0], npu_bd_id(10), it_channel_1, {0, 0, 0, (uint32_t)(TY / 4)}, {1, 1, 1, (uint32_t)Y_UNIT}, {0, 0, 0, 1});
        seq->npu_dma_memcpy_nd(1, 0, MM2S, IT[0], npu_bd_id(6), it_channel_1, {0, 0, 0, (uint32_t)T_HDR}, {1, 1, 1, (uint32_t)UNIT}, {0, 0, 0, 1});
        seq->npu_dma_memcpy_nd(1, 0, MM2S, IT[0], npu_bd_id(7), it_channel_1, {0, 0, 0, (uint32_t)RY}, {1, 1, 1, (uint32_t)(2 * UNIT)}, {0, 0, 0, 1});
        for (int c = 0; c < 16; c++) {
            const int col = c % 8, r = c / 8;
            seq->npu_dma_memcpy_nd(4, 0, S2MM, IT[col], npu_bd_id(8 + r), chn(r), {0, 0, 0, (uint32_t)((EY + (size_t)c * 12 * Y_UNIT * 4) / 4)}, {1, 1, 1, (uint32_t)(12 * Y_UNIT)}, {0, 0, 0, 1});
            seq->npu_dma_memcpy_nd(1, 0, MM2S, IT[col], npu_bd_id(4 + r), chn(r), {0, 0, 0, (uint32_t)E_HDR}, {1, 1, 1, (uint32_t)UNIT}, {0, 0, 0, 1});
        }
        seq->npu_dma_wait(IT[0], S2MM, it_channel_1);
        for (int c = 0; c < 16; c++) seq->npu_dma_wait(IT[c % 8], S2MM, chn(c / 8));
    }
    seq->cmds2seq();
    double us = 0;
    for (int it = 0; it <= iters; it++) {
        if (it) { memcpy(io.data(), in0.data(), IO); io.sync_to_device(); }
        const auto t1 = std::chrono::steady_clock::now();
        xrt::run run = app.create_run(io, wt);
        run.start();
        const ert_cmd_state st = wait_run(run, 5000);
        if (it > 0) us += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t1).count();
        if (st != ERT_CMD_STATE_COMPLETED) {
            io.sync_from_device();
            int ny = 0, no = 0, noy = 0;
            for (int s = 0; s < NG + 3; s++) ny += *(float*)(io.data() + PY + (size_t)s * Y_UNIT * 4) != -1.0f;
            for (int g = 0; g < NKV; g++) no += memcmp(io.data() + OH + (size_t)(g / 4) * UNIT + X_OFF + (g % 4) * 768, in0.data() + OH + (size_t)(g / 4) * UNIT + X_OFF + (g % 4) * 768, 4) != 0;
            for (int s = 0; s < 2 * OG; s++) noy += *(float*)(io.data() + OY + (size_t)s * Y_UNIT * 4) != -1.0f;
            std::printf("run state %d; A out changed %d, proj y slots %d / %d, attention heads %d / 8, o proj y %d / %d, F out changed %d, "
                        "top-k changed %d\n", (int)st, memcmp(io.data() + X_OFF, in0.data() + X_OFF, 4) != 0, ny, NG + 3, no, noy, 2 * OG,
                        memcmp(io.data() + E_HDR + X_OFF, in0.data() + E_HDR + X_OFF, 4) != 0, memcmp(io.data() + TY, in0.data() + TY, 4) != 0);
            return 2;
        }
    }
    std::printf("layer p=%d %s stages %d: %.1f us per run (%d iters)\n", p, glob ? "global" : "swa", stages, us / std::max(iters, 1), iters);
    io.sync_from_device();
    wt.sync_from_device();
    const float* in = (const float*)in0.data();
    auto IN = [&](size_t off) { return in + off / 4; };
    bool ok = true;
    // A mix: from the bytes before the run
    if (stages & 32) {
        std::vector<double> raw, cur;
        std::vector<std::vector<double>> outs;
        mix_reference(CA, p, IN(ST), IN(TY), IN(BA), raw, cur, outs);
        const double er = rel_rms(raw.data(), [&](int c) { return (double)Fp(BA)[(size_t)(c / Y_UNIT) * 8 * Y_UNIT + (4 + (p & 3)) * Y_UNIT + c % Y_UNIT]; }, DM);
        const double ec = rel_rms(cur.data(), [&](int c) { return (double)Fp(BF + CB)[c]; }, DM);
        double eq = 0;
        for (int j = 0; j < nx; j++) {
            const uint16_t* b = (const uint16_t*)(io.data() + X_OFF) + (size_t)j * DM;
            eq = std::max(eq, rel_rms(outs[j].data(), [&](int c) { return (double)bf2f(b[c]); }, DM));
        }
        std::printf("A mix: ring row %.3e, cur %.3e, queries (bf16) worst %.3e\n", er, ec, eq);
        ok &= er < 1e-4 && ec < 1e-4 && eq < 6e-3;
    }
    // proj, from the x in the header
    std::vector<std::vector<float>> xv(nx, std::vector<float>(D));
    for (int i = 0; i < nx; i++)
        for (int k = 0; k < D; k++) xv[i][k] = bf2f(((const uint16_t*)(io.data() + X_OFF))[(size_t)i * D + k]);
    std::vector<double> ref(96 * OG);
    if (stages & 1) {
        double err2 = 0, ref2 = 0, worst = 0;
        for (int g = 0, m = 0; g < NG; g++) {
            while (g >= ends[m]) m++;
            ref_rows(ref.data(), wt.data(), g, 1, NC, xv[nx == 3 ? xi3[m] : 0].data());
            const float* y = (const float*)(io.data() + PY) + (size_t)slot_of(g) * Y_UNIT;
            double e2 = 0, r2 = 0;
            for (int i = 0; i < 96; i++) { const double d = y[i] - ref[i]; e2 += d * d; r2 += ref[i] * ref[i]; }
            err2 += e2; ref2 += r2;
            worst = std::max(worst, std::sqrt(e2 / r2));
        }
        double dworst = 0;
        for (const dense& d : dns) {
            const float* y = (const float*)(io.data() + PY) + (size_t)(26 + d.u) * Y_UNIT;
            for (int i = 0; i < d.n; i++) {
                double r = 0, a = 0;
                for (int k = 0; k < D; k++) { r += (double)dwf[((size_t)d.u * 32 + i) * D + k] * xv[d.x][k]; a += std::fabs(dwf[((size_t)d.u * 32 + i) * D + k] * xv[d.x][k]); }
                dworst = std::max(dworst, std::fabs(y[i] - r) / a);
            }
        }
        std::printf("proj: rel rms err %.3e (worst group %.3e), dense rows max err / sum |w x| %.3e\n", std::sqrt(err2 / ref2), worst, dworst);
        ok &= worst < 1e-2 && dworst < 1e-5;
    }
    std::vector<uint16_t> O(QD);
    for (int g = 0; g < NKV; g++) memcpy(O.data() + g * 384, io.data() + OH + (size_t)(g / 4) * UNIT + X_OFF + (g % 4) * 768, 768);
    if (stages & 2) {
        const float* Y = (const float*)(io.data() + PY);
        std::vector<double> o_ref;
        std::vector<uint16_t> rec_ref;
        reference(Y + QS * Y_UNIT, Y + QS * Y_UNIT + QD, Y, att0.data(), G, o_ref, rec_ref);
        ok &= check(wt.data() + AW, Y, O.data(), G, o_ref, rec_ref);
    }
    if (stages & 4) {
        std::vector<float> xo(QD);
        for (int i = 0; i < QD; i++) xo[i] = bf2f(O[i]);
        ref_rows(ref.data(), wt.data() + OW, 0, OG, ONC, xo.data());
        double err2 = 0, ref2 = 0;
        for (int g = 0; g < OG; g++)
            for (int i = 0; i < 96; i++) {
                const float* y0 = (const float*)(io.data() + OY) + (size_t)g * Y_UNIT;
                const double d = (double)y0[i] + y0[(size_t)OG * Y_UNIT + i] - ref[96 * g + i];
                err2 += d * d; ref2 += ref[96 * g + i] * ref[96 * g + i];
            }
        std::printf("o proj: rel rms err %.3e\n", std::sqrt(err2 / ref2));
        ok &= std::sqrt(err2 / ref2) < 1e-2;
    }
    // F mix: o halves as written, the block as before the run with the residual A wrote
    if (stages & 8) {
        std::vector<float> Bf(IN(BF), IN(BF) + BBF / 4);
        memcpy(Bf.data() + CB / 4, io.data() + BF + CB, RES);
        std::vector<double> raw, cur;
        std::vector<std::vector<double>> outs;
        mix_reference(CF, p, IN(ST), Fp(OY), Bf.data(), raw, cur, outs);
        const double er = rel_rms(raw.data(), [&](int c) { return (double)Fp(BF)[(size_t)(c / Y_UNIT) * 8 * Y_UNIT + (4 + (p & 3)) * Y_UNIT + c % Y_UNIT]; }, DM);
        const double ec = rel_rms(cur.data(), [&](int c) { return (double)Fp(BA + CB)[c]; }, DM);
        const uint16_t* b = (const uint16_t*)(io.data() + E_HDR + X_OFF);
        const double eo = rel_rms(outs[0].data(), [&](int c) { return (double)bf2f(b[c]) + bf2f(b[DM + c]); }, DM);
        int cm = 0;
        for (int c = 0; c < DM; c++) cm += b[c] != b[2 * DM + c];
        std::printf("F mix: ring row %.3e, cur %.3e, out (hi + lo) %.3e, hi copy mismatches %d\n", er, ec, eo, cm);
        ok &= er < 1e-4 && ec < 1e-4 && eo < 1e-4 && cm == 0;
    }
    // MoE, from the x after the expert header
    if (stages & 16) {
        const uint16_t* xb = (const uint16_t*)(io.data() + E_HDR + X_OFF);
        std::vector<float> x(D), xh(D);
        for (int i = 0; i < D; i++) { xh[i] = bf2f(xb[i]); x[i] = xh[i] + bf2f(xb[D + i]); }
        std::vector<float> pr(NE), key(NE);
        double lerr = 0, lref = 0;
        for (int e = 0; e < NE; e++) {
            double s = 0;
            for (int i = 0; i < D; i++) s += (double)W[(size_t)e * D + i] * x[i];
            pr[e] = 1.0f / (1.0f + std::exp(-(float)s));
            key[e] = pr[e] + bias[e];
            lerr = std::max(lerr, std::fabs(*(float*)(io.data() + RY + ((e / 8) * Y_UNIT + e % 8) * 4) - s));
            lref = std::max(lref, std::fabs(s));
        }
        std::vector<int> idx(NE);
        for (int e = 0; e < NE; e++) idx[e] = e;
        std::partial_sort(idx.begin(), idx.begin() + 8, idx.end(), [&](int a, int b) { return key[a] > key[b]; });
        float sum = 0;
        for (int k = 0; k < 8; k++) sum += pr[idx[k]];
        const int32_t* tid = (const int32_t*)(io.data() + TY);
        const float* tw = (const float*)(io.data() + TY) + 8;
        bool ids_ok = true;
        double werr = 0;
        for (int k = 0; k < 8; k++) { ids_ok &= tid[k] == idx[k]; werr = std::max(werr, (double)std::fabs(tw[k] - pr[idx[k]] / sum * 8)); }
        double err2 = 0, ref2 = 0;
        for (int e = 0; e < 8 && ids_ok; e++) {
            const int id = tid[e];
            std::vector<double> g(2 * FFH);
            ref_rows(g.data(), gu.data() + (size_t)id * GU_E, 0, 24, 9, xh.data());
            std::vector<float> hh(1280, 0.0f);
            for (int j = 0; j < 36; j++)
                for (int i = 0; i < 32; i++) {
                    const double gv = g[64 * j + i], uv = g[64 * j + 32 + i];
                    hh[32 * j + i] = bf2f(f2bf((float)(4 * std::tanh(gv / 4) / (1 + std::exp(-gv)) * 25 * std::tanh(uv / 25))));
                }
            std::vector<double> z(D);
            ref_rows(z.data(), dn.data() + (size_t)id * DN_E, 0, 24, 5, hh.data());
            for (int r = 0; r < 2; r++) {
                const float* y = (const float*)(io.data() + EY) + (size_t)(e + 8 * r) * 12 * Y_UNIT;
                for (int i = 0; i < 1152; i++) { const double d = y[i] - z[1152 * r + i]; err2 += d * d; ref2 += z[1152 * r + i] * z[1152 * r + i]; }
            }
        }
        const double rel = ids_ok ? std::sqrt(err2 / ref2) : 1.0;
        std::printf("MoE: logits max abs err %.3e (max |logit| %.3f), ids %s, weights max err %.3e, experts rel rms err %.3e\n",
                    lerr, lref, ids_ok ? "equal" : "DIFFER", werr, rel);
        ok &= ids_ok && werr < 1e-4 && rel < 2e-2;
    }
    std::printf("layer %s\n", ok ? "PASSED" : "FAILED");
    return ok ? 0 : 1;
}

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "sink";
    if (mode == "moe") return moe_test(argc > 2 ? std::atoi(argv[2]) : 20);
    if (mode == "proj") return proj_test(argc > 2 ? std::atoi(argv[2]) : 20, argc > 3 ? std::atoi(argv[3]) : 3, argc > 4 ? std::atoi(argv[4]) : 1);
    if (mode == "mixc") {
        if (argc < 8) { std::printf("usage: run.exe mixc pre fl S nq om nn [iters]\n"); return 3; }
        int c[6];
        for (int i = 0; i < 6; i++) c[i] = atoi(argv[2 + i]);
        return mix_test(-1, argc > 8 ? atoi(argv[8]) : 20, c);
    }
    if (mode == "mix") return mix_test(argc > 2 ? std::atoi(argv[2]) : 0, argc > 3 ? std::atoi(argv[3]) : 20);
    if (mode == "layer") return layer_test(argc > 2 ? std::atoi(argv[2]) : 20, argc > 3 ? std::atoi(argv[3]) : 100, argc > 4 ? std::atoi(argv[4]) : 0, argc > 5 ? std::atoi(argv[5]) : 63);
    if (mode == "chain") return chain_test(argc > 2 ? std::atoi(argv[2]) : 20, argc > 3 ? std::atoi(argv[3]) : 100, argc > 4 ? std::atoi(argv[4]) : 0, argc > 5 ? std::atoi(argv[5]) : 7);
    if (mode == "attn") return attn_test(argc > 2 ? std::atoi(argv[2]) : 20, argc > 3 ? std::atoi(argv[3]) : 100, argc > 4 ? std::atoi(argv[4]) : 0, argc > 5 ? std::atoi(argv[5]) : 1);
    if (mode == "rms") return rms_test(argc > 2 ? std::atoi(argv[2]) : 20, argc > 3 ? std::atoi(argv[3]) : 1, argc > 4 ? std::atoi(argv[4]) : 15, argc > 5 ? std::atoi(argv[5]) : 2304);
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
    // SINK_PRE=1: step t + 1's S2MM tasks and MM2S BDs (the other BD set) go out before step t's waits; after them
    // only the MM2S pushes (one register write per channel) remain on the step boundary
    const bool pre = getenv("SINK_PRE") && atoi(getenv("SINK_PRE"));
    auto s2mm = [&](int t, int c) {
        const int col = c % 8, r = c / 8, set = pre ? 4 * (t & 1) : 0;
        seq->npu_dma_memcpy_nd(4, 1, S2MM, IT[col], npu_bd_id(2 + r + set), r ? it_channel_1 : it_channel_0,
            {0, 0, 0, (uint32_t)(((size_t)c * T + t) * y_units * Y_UNIT)}, {1, 1, 1, (uint32_t)(y_units * Y_UNIT)}, {0, 0, 0, 1});
    };
    auto mm2s = [&](int t, int c, bool push) {
        const int col = c % 8, r = c / 8, set = pre ? 4 * (t & 1) : 0;
        seq->npu_dma_memcpy_nd(1, 0, MM2S, IT[col], npu_bd_id(r + set), r ? it_channel_1 : it_channel_0,
            {0, 0, 0, (uint32_t)(c * ch_bytes + (size_t)t * task_units * UNIT)}, {1, 1, 1, (uint32_t)(task_units * UNIT)}, {0, 0, 0, 1}, -1, 0, false, cache, 15, push);
    };
    if (pre)
        for (int c = 0; c < C; c++) { s2mm(0, c); mm2s(0, c, true); }
    for (int t = 0; t < T; t++) {
        if (!pre)
            for (int c = 0; c < C; c++) { s2mm(t, c); mm2s(t, c, true); }
        else if (t + 1 < T)
            for (int c = 0; c < C; c++) { s2mm(t + 1, c); mm2s(t + 1, c, false); }
        for (int c = 0; c < C; c++) seq->npu_dma_wait(IT[c % 8], S2MM, c / 8 ? it_channel_1 : it_channel_0);
        if (pre && t + 1 < T)
            for (int c = 0; c < C; c++) seq->rtp_write(IT[c % 8], c / 8 ? 0x1d21c : 0x1d214, (uint32_t)(c / 8 + 4 * ((t + 1) & 1)));
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
                const int32_t* yi = (const int32_t*)y;
                if (yi[0] != n || (unsigned)yi[1] != sum) { ok = false; std::printf("ch %d task %d: %d %d (want %d %u)\n", c, t, yi[0], yi[1], n, sum); }
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
