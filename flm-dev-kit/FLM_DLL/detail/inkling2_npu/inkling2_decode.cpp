// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file inkling2_decode.cpp
/// \brief inkling2_decode: a decode token's routed experts on the NPU
#include "inkling2_decode.hpp"

#include <immintrin.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <stdexcept>

#include "inkling2_host.hpp"

using namespace inkling2_host;

namespace {

constexpr int UNIT = 4736;     // one Q4_K block: a task is whole units
constexpr int X_OFF = 128;     // x in a header unit
constexpr int Y_UNIT = 96;     // outputs per group
constexpr int GPE = D / (NPU_GROUP * QROWS);   // groups per expert: 24 (gate_up 2 FF = D rows, down D rows)
constexpr int C = 16;          // channels
constexpr int OP_GEMV_Q4K = 2;
constexpr npu_tiles IT[8] = {IT0, IT1, IT2, IT3, IT4, IT5, IT6, IT7};
constexpr int NC[2] = {D / 256, FF_PAD / 256};   // column blocks: gate_up 9, down 5

static_assert(D == 2 * FF && GPE * NPU_GROUP * QROWS == D, "gate_up and down are 24 groups per expert");
static_assert(X_OFF + D * 2 <= UNIT, "x fits a header unit");

using clk = std::chrono::steady_clock;
double since(clk::time_point& t0) {
    const clk::time_point t1 = clk::now();
    const double s = std::chrono::duration<double>(t1 - t0).count();
    t0 = t1;
    return s;
}

void to_bf16(u16* dst, const float* src, int n) {
    const __m512i one = _mm512_set1_epi32(1), bias = _mm512_set1_epi32(0x7FFF);
    for (int i = 0; i < n; i += 16) {
        __m512i v = _mm512_castps_si512(_mm512_loadu_ps(src + i));
        v = _mm512_add_epi32(v, _mm512_add_epi32(_mm512_and_si512(_mm512_srli_epi32(v, 16), one), bias));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), _mm512_cvtepi32_epi16(_mm512_srli_epi32(v, 16)));
    }
}

}  // namespace

inkling2_decode::inkling2_decode(npu_xclbin_manager* npu, const std::string& xclbin) : mgr(npu->register_xclbin(xclbin)) {}

inkling2_decode::plan& inkling2_decode::get(int kind, int n) {
    auto it = this->plans.find(kind * 16 + n);
    if (it != this->plans.end()) return it->second;
    plan& p = this->plans[kind * 16 + n];
    p.app = this->mgr->create_app();
    const int nc = NC[kind];
    const bool shared_x = kind == 0;
    // channel c takes groups [c total / C, (c + 1) total / C) of the (expert slot, group) order: at most two experts
    const int total = n * GPE;
    p.segs.resize(C);
    for (int c = 0; c < C; c++) {
        int g = c * total / C;
        const int end = (c + 1) * total / C;
        while (g < end) {
            const int s = g / GPE, m = std::min(end, (s + 1) * GPE) - g;
            p.segs[c].push_back({s, g % GPE, m});
            g += m;
        }
        if (p.segs[c].size() > 2) throw std::runtime_error("inkling2 decode: a channel spans more than two experts");
    }
    // io: header units [C][2], then y [C][groups][96] fp32
    p.ybase.resize(C);
    size_t yo = (size_t)C * 2 * UNIT;
    for (int c = 0; c < C; c++) {
        p.ybase[c] = yo;
        for (const seg& s : p.segs[c]) yo += (size_t)s.n * Y_UNIT * 4;
    }
    p.io_bytes = yo;
    if (this->io[kind].size() < yo) {
        this->io[kind] = p.app.create_bo_buffer<uint8_t>(yo);
        memset(this->io[kind].data(), 0, yo);
    }
    // per channel: y (S2MM BD 14 + r), then header and weight tasks (MM2S BDs 6 r ..): at most 4 outstanding
    npu_sequence* seq = p.app.seq();
    seq->clear_cmds();
    for (int c = 0; c < C; c++) {
        const int col = c % 8, r = c / 8;
        const npu_it_channel ch = r ? it_channel_1 : it_channel_0;
        int ny = 0;
        for (const seg& s : p.segs[c]) ny += s.n;
        seq->npu_dma_memcpy_nd(4, 0, S2MM, IT[col], npu_bd_id(14 + r), ch, {0, 0, 0, (uint32_t)(p.ybase[c] / 4)},
                               {1, 1, 1, (uint32_t)(ny * Y_UNIT)}, {0, 0, 0, 1});
        int bd = 6 * r;
        for (size_t i = 0; i < p.segs[c].size(); i++) {
            const seg& s = p.segs[c][i];
            if (i == 0 || !shared_x)
                seq->npu_dma_memcpy_nd(1, 0, MM2S, IT[col], npu_bd_id(bd++), ch, {0, 0, 0, (uint32_t)((c * 2 + i) * UNIT)},
                                       {1, 1, 1, (uint32_t)UNIT}, {0, 0, 0, 1});
            seq->npu_dma_memcpy_nd(1, 1 + s.slot, MM2S, IT[col], npu_bd_id(bd++), ch,
                                   {0, 0, 0, (uint32_t)((size_t)s.g0 * nc * NPU_GROUP * UNIT)},
                                   {1, 1, 1, (uint32_t)((size_t)s.n * nc * NPU_GROUP * UNIT)}, {0, 0, 0, 1});
        }
    }
    for (int c = 0; c < C; c++) seq->npu_dma_wait(IT[c % 8], S2MM, c / 8 ? it_channel_1 : it_channel_0);
    seq->cmds2seq();
    return p;
}

void inkling2_decode::start(int kind, bytes& w, size_t base, const int* ex, int n, const float* in) {
    clk::time_point t0 = clk::now();
    if (n < 1 || n > MAX_N) throw std::runtime_error("inkling2 decode: 1 to 4 experts per dispatch");
    plan& p = this->get(kind, n);
    const int nc = NC[kind];
    const size_t eb = (size_t)GPE * nc * NPU_GROUP * UNIT;   // bytes per expert
    u8* io = this->io[kind].data();
    // headers: shared x (gate_up) once per channel over all its groups, else one per segment with its expert's input
    u16 xb[MAX_N][D];
    const int xn = kind == 0 ? 1 : n, xlen = nc * 256;
    for (int s = 0; s < xn; s++) to_bf16(xb[s], in + (size_t)s * (kind == 0 ? D : FF_PAD), xlen);
    for (int c = 0; c < C; c++) {
        int all = 0;
        for (const seg& s : p.segs[c]) all += s.n;
        for (size_t i = 0; i < p.segs[c].size(); i++) {
            if (i > 0 && kind == 0) break;
            u8* h = io + (size_t)(c * 2 + i) * UNIT;
            int32_t hw[3] = {OP_GEMV_Q4K, kind == 0 ? all : p.segs[c][i].n, nc};
            memcpy(h, hw, sizeof(hw));
            memcpy(h + X_OFF, xb[kind == 0 ? 0 : p.segs[c][i].slot], (size_t)xlen * 2);
        }
    }
    this->io[kind].sync_to_device();
    bound_arg a[MAX_N];
    for (int s = 0; s < MAX_N; s++) a[s] = {&w, base + (size_t)ex[std::min(s, n - 1)] * eb};
    this->run = p.app.create_run(this->io[kind], a[0], a[1], a[2], a[3]);
    this->run.start();
    this->kind = kind;
    this->n = n;
    this->t_start += since(t0);
    this->calls++;
}

void inkling2_decode::wait(float* out) {
    clk::time_point t0 = clk::now();
    this->run.wait();
    plan& p = this->get(this->kind, this->n);
    this->io[this->kind].sync_from_device();
    const u8* io = this->io[this->kind].data();
    for (int c = 0; c < C; c++) {
        const float* y = reinterpret_cast<const float*>(io + p.ybase[c]);
        for (const seg& s : p.segs[c])
            for (int g = s.g0; g < s.g0 + s.n; g++, y += Y_UNIT) {
                if (this->kind == 1) {
                    memcpy(out + (size_t)s.slot * D + g * Y_UNIT, y, Y_UNIT * 4);
                    continue;
                }
                // gate_up storage block j = 3 g + b: gate block j / 2 when even, up block j / 2 when odd
                float* o = out + (size_t)s.slot * 2 * FF;
                for (int b = 0; b < NPU_GROUP; b++) {
                    const int j = NPU_GROUP * g + b;
                    memcpy(o + (j % 2 ? FF : 0) + (j / 2) * QROWS, y + b * QROWS, QROWS * 4);
                }
            }
    }
    this->t_wait += since(t0);
}
