// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file inkling2_fused.cpp
/// \brief decode on the NPU: the runs of a token's layer applications (see inkling2_fused.hpp). The transfers of each
///        step are the ones FLM_Xclbin/Inkling2/decode_layer/host/host.cpp checks (layer_test): the cores count on
///        exactly these sizes, a mismatch stalls the array.

#include "inkling2_fused.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>

#if !defined(FLM_USE_HRX)
#include "launch/txn/elf.hpp"
#endif

#if !defined(FLM_USE_HRX)   // see inkling2_fused.hpp

using namespace inkling2_host;

namespace {
constexpr int UNIT = 4736, X_OFF = 128, YU = 96, CH = 24;
constexpr npu_tiles IT[8] = {IT0, IT1, IT2, IT3, IT4, IT5, IT6, IT7};
constexpr uint32_t HUB_ARGS = 0x8200, HUB_START_LOCK = 0x1F000 + 16 * 10;
constexpr int OP_GEMV = 2, OP_EXPERT = 3, OP_ROUTER = 4, OP_TOPK = 5, OP_PROJ = 6, OP_UP = 7, OP_ATTN = 17, OP_MIX = 18, OP_CTOUT = 19;
constexpr int NG = 90, NC = 9, OG = 24, ONC = 12, QS = 29, KPU = 8;
constexpr size_t GB = (size_t)NC * 3 * UNIT;   // one group of 96 rows x 2304
constexpr size_t A_LEN = 4 * UNIT - 128 - 3072, RB = 2 * UNIT, RING = 9 * UNIT, REC0 = 10 * UNIT;
constexpr size_t PBA = (size_t)193 * YU * 4, PBF = (size_t)2 * D * 4;
// per layer in argument 1: dense rows (sinks, K / V conv weights; 96 units), router (256), top-k header (1)
constexpr size_t LW = (size_t)353 * UNIT, L_RW = (size_t)96 * UNIT, L_TH = (size_t)352 * UNIT;
constexpr int ENDS[5] = {11, 22, 54, 86, 90}, XI3[5] = {1, 2, 0, 0, 0};
constexpr size_t BO_MAX = (size_t)2040 << 20;

int slot_of(int g) { return g < 22 ? g : g < 86 ? g + 7 : g - 64; }
size_t up(size_t x) { return (x + 4095) / 4096 * 4096; }
float bf2f(u16 v) { uint32_t u = (uint32_t)v << 16; float f; memcpy(&f, &u, 4); return f; }
u16 f2bf(float f) { uint32_t u; memcpy(&u, &f, 4); u += 0x7FFF + ((u >> 16) & 1); return (u16)(u >> 16); }
npu_it_channel chn(int r) { return r ? it_channel_1 : it_channel_0; }

// A run that does not complete leaves cores stalled; tearing its hw context down right away can wedge amdxdna. On a
// timeout: mark the device suspect (a test harness's INKLING2_SUSPECT_FILE), keep the context alive past the driver's
// 60 s job timeout, then report.
ert_cmd_state wait_run(xrt::run& run, int ms) {
    ert_cmd_state st = ERT_CMD_STATE_TIMEOUT;
    try { st = run.wait(std::chrono::milliseconds(ms)); } catch (const std::exception& e) { std::fprintf(stderr, "inkling2 fused: wait: %s\n", e.what()); }
    if (st != ERT_CMD_STATE_COMPLETED) {
        if (FILE* f = getenv("INKLING2_SUSPECT_FILE") ? fopen(getenv("INKLING2_SUSPECT_FILE"), "w") : nullptr) {
            std::fprintf(f, "inkling2 fused run state %d\n", (int)st);
            fclose(f);
        }
        std::fprintf(stderr, "inkling2 fused: RUN DID NOT COMPLETE (state %d): holding the context 90 s\n", (int)st);
        std::this_thread::sleep_for(std::chrono::seconds(90));
        try { st = run.wait(std::chrono::milliseconds(1)); } catch (...) {}
    }
    return st;
}
}   // namespace

inkling2_fused::app_info inkling2_fused::info(int j) {
    app_info I;
    if (j < CORE0) { I.il = j; I.pass = 0; }
    else if (j < CORE0 + NPASS * (CORE1 - CORE0)) { I.pass = (j - CORE0) / (CORE1 - CORE0); I.il = CORE0 + (j - CORE0) % (CORE1 - CORE0); }
    else { I.il = CORE1 + j - CORE0 - NPASS * (CORE1 - CORE0); I.pass = 0; }   // the last layers: outside the loop
    I.slot = layer_slot(I.il, I.pass);
    I.glob = is_global(I.il);
    I.starts = I.il % BLOCK == 0;
    I.three = I.glob || I.pass > 0;
    I.S = I.il == 0 ? 0 : (I.il - 1) / BLOCK + 1;
    return I;
}

// A: layer 0 mixes h0 alone (no query: a zero one), later ones the previous experts + ffn conv + residual
inkling2_fused::mix_cfg inkling2_fused::cfg_A(int j) {
    const app_info I = info(j);
    if (j == 0) return {0, 0, 0, 1, 1, 1};
    if (I.pass > 0) return I.il == CORE0 ? cfg_P(j) : mix_cfg{1, 7, I.S, I.glob ? 3 : 1, 3, 0};
    return {1, 7, I.S, I.glob ? 3 : 1, 1, 1};
}

inkling2_fused::mix_cfg inkling2_fused::cfg_C() { return {1, 7, 0, 0, 0, 0}; }
inkling2_fused::mix_cfg inkling2_fused::cfg_M() { return {0, 0, 1, 4 * (CORE1 - CORE0), 3, 0}; }
inkling2_fused::mix_cfg inkling2_fused::cfg_P(int j) { return {0, 0, 1, info(j).glob ? 3 : 1, 3, 0}; }
inkling2_fused::mix_cfg inkling2_fused::cfg_Z() { return {1, 7, (NL - 1) / BLOCK + 1, 1, 1, 1}; }

// output units of 96 words: ring row, cur, then per query fp32 (24) / bf16 (12) / bf16 hi lo hi (36) / fp32, bf16 / rms (36)
int inkling2_fused::mix_nout(const mix_cfg& C) { return C.nq * (C.om == 0 ? 24 : C.om == 1 ? 12 : 36); }

// F: a block's first layer has no residual (its attention output starts the stream) and one more block state
inkling2_fused::mix_cfg inkling2_fused::cfg_F(int j) {
    const app_info I = info(j);
    return {2, I.starts ? 5 : 7, I.starts ? I.S + 1 : I.S, 1, I.pass > 0 ? 3 : 2, I.pass > 0 ? 0 : 1};
}

size_t inkling2_fused::mix_pb(const mix_cfg& C) { return C.pre == 1 ? PBA : (size_t)(C.pre == 2 ? 2 : 1) * D * 4; }

// the block task [conv block | residual | queries, norms], padded so header + states + prefix + block is whole units
size_t inkling2_fused::mix_bb(const mix_cfg& C) {
    const size_t cb = (C.fl & 1) ? CB : 0, rb = (C.fl & 2) ? RES : 0, qb = (size_t)C.nq * (1 + C.nn) * D * 4;
    const size_t total = X_OFF + (size_t)C.S * D * 4 + mix_pb(C) + cb + rb + qb;
    return cb + rb + qb + ((total + UNIT - 1) / UNIT * UNIT - total);
}

void inkling2_fused::plan_proj(proj_plan& P, size_t at) {
    P.dns = {{15, 24, 0, 0}};
    if (P.nx == 3) { P.dns.push_back({0, 32, 1, 1}); P.dns.push_back({2, 32, 2, 2}); }
    for (size_t i = 0; i < P.dns.size(); i++) P.nd_of[P.dns[i].c] = (int)i + 1;
    P.hdr = (size_t)(X_OFF + P.nx * D * 2 + UNIT - 1) / UNIT * UNIT;
    int* h = (int*)(this->io.data() + at);
    h[0] = OP_PROJ;
    h[1] = P.nx;
    for (int c = 0; c < 16; c++) P.ng[c] = 6;
    int over = 16 * 6 - NG;
    for (const auto& d : P.dns) if (over > 0) { P.ng[d.c]--; over--; }
    for (int c = 15; over > 0; c--) if (P.ng[c] == 6) { P.ng[c]--; over--; }
    for (int c = 0, g = 0; c < 16; c++) {
        P.g0[c] = g;
        g += P.ng[c];
        h[2 + c] = P.g0[c] | P.ng[c] << 16;
    }
    for (int m = 0; m < 5; m++) h[18 + m] = ENDS[m] | (P.nx == 3 ? XI3[m] : 0) << 16;
    for (const auto& d : P.dns) ((int16_t*)h)[46 + d.c] = (int16_t)(d.n | d.x << 8);
    for (int c = 0; c < 16; c++) {
        for (int g = P.g0[c]; g < P.g0[c] + P.ng[c]; g++) {
            if (!P.runs[c].empty() && P.runs[c].back().first + P.runs[c].back().second == slot_of(g)) P.runs[c].back().second++;
            else P.runs[c].push_back({slot_of(g), 1});
        }
        if (P.runs[c].size() > 2) throw std::runtime_error("inkling2 fused: projection channel with more than 2 runs");
    }
}

void inkling2_fused::region(int slot, int& arg, size_t& off, size_t& rs) const {
    if (this->gidx[slot] >= 0) {
        arg = 2 + this->gidx[slot] / 4;
        off = (size_t)(this->gidx[slot] % 4) * NKV * this->RSG;
        rs = this->RSG;
    } else {
        arg = 1;
        off = this->SW0 + (size_t)this->sidx[slot] * NKV * this->RSS;
        rs = this->RSS;
    }
}

inkling2_fused::inkling2_fused(npu_xclbin_manager* npu, const std::string& xclbin, const weights& w_,
                               const std::vector<layer_src>& src_, int max_len_)
    : w(w_), src(src_), max_len(max_len_) {
    if (const char* e = getenv("INKLING2_FUSED_CHAIN")) this->chain = std::max(1, atoi(e));
    for (const char* e : {"INKLING2_HOST_CTRL", "INKLING2_CTRL_CHECK"})
        if (getenv(e) && atoi(getenv(e))) this->dev_ctrl = false;
    if ((int)this->src.size() != NL) throw std::runtime_error("inkling2 fused: need every layer's weights");
    this->mgr = npu->register_xclbin(xclbin);
    if (const char* e = getenv("INKLING2_FUSED_PIPE")) this->pipe = atoi(e) != 0;
    for (auto& a : this->apps) {
        a = this->mgr->create_app();
#if !defined(FLM_USE_HRX)
        // each run is a new instruction stream: write its ELF directly (~7x faster than aiebu), INKLING2_AIEBU=1 keeps
        // aiebu
        if (!getenv("INKLING2_AIEBU") || atoi(getenv("INKLING2_AIEBU")) == 0)
            a.set_elf_writer([](const uint32_t* w, size_t n) {
                return launch::txn::to_elf(std::vector<uint32_t>(w, w + n));
            });
#endif
    }
    // io
    this->H1 = 0;
    this->H3 = UNIT;
    this->PY = 4 * UNIT;
    this->AH = this->PY + (size_t)(NG + 3) * YU * 4;
    this->OH = up(this->AH + (size_t)NAPP * NKV * 128);
    this->OY = this->OH + 2 * UNIT;
    this->ST = up(this->OY + (size_t)2 * OG * YU * 4);
    this->R_HDR = up(this->ST + (size_t)4 * D * 4);
    this->E_HDR = this->R_HDR + UNIT;
    this->RY = this->E_HDR + 3 * UNIT;
    this->TY = this->RY + 2 * UNIT;
    this->EY = this->TY + YU * 4;   // right after TY: the next A mix reads both as its prefix
    this->MH = up(this->EY + (size_t)16 * 12 * YU * 4);
    this->MHX = this->MH + (size_t)NAPP * 256;
    this->MZ = this->MHX + (size_t)(NPASS - 1) * 256;
    this->XC = up(this->MZ + 256);
    this->XF = this->XC + 3 * CQ;
    this->BANK = this->XF + CQ;
    this->XZ = up(this->BANK + (size_t)D * 4);
    this->MSG = up(this->XZ + (size_t)D * 2);
    size_t at = up(this->MSG + (size_t)cfg_M().nq * CQ);
    this->BA.resize(NAPP + 1);
    this->BF.resize(NAPP);
    for (int j = 0; j < NAPP; j++) {
        // a later pass's first application: its A block is the pass completion's (the previous one's ffn conv + res)
        const app_info I = info(j);
        this->BA[j] = at;
        at = up(at + mix_bb(I.pass > 0 && I.il == CORE0 ? cfg_C() : cfg_A(j)));
        this->BF[j] = at;
        at = up(at + mix_bb(cfg_F(j)));
        if (I.pass > 0 && I.il == CORE0) {
            this->BP[I.pass - 1] = at;
            at = up(at + mix_bb(cfg_A(j)));
        }
    }
    // the token's output block: its residual (the last F mix's cur) is FIN
    this->BA[NAPP] = at;
    this->FIN = at + CB;
    at = up(at + mix_bb(cfg_Z()));
    this->BM = at;
    at = up(at + mix_bb(cfg_M()));
    this->H_DN = at;
    this->H_UP = at + 256;
    this->H_CT = at + 512;
    this->GD = at + 768;
    this->FA = up(this->GD + (size_t)16 * YU * 4);
    this->CK = this->FA + (size_t)4 * (CORE1 - CORE0) * FA_LEN;
    at = up(this->CK + (size_t)36 * YU * 4);
    for (int il = 0; il < NL; il++) {
        const layer_weights& L = this->w.layer[il];
        if ((il > 0 && !L.attn_query) || !L.ffn_query || (il >= CORE0 && il < CORE1 && !L.msg_query))
            throw std::runtime_error("inkling2 fused: layer " + std::to_string(il) + " without a mix query");
    }
    this->io = npu->create_bo_buffer<uint8_t>(at);
    memset(this->io.data(), 0, at);
    this->pp1.nx = 1;
    this->pp3.nx = 3;
    plan_proj(this->pp1, this->H1);
    plan_proj(this->pp3, this->H3);
    for (int hu = 0; hu < 2; hu++) {
        int* oh = (int*)(this->io.data() + this->OH + hu * UNIT);
        oh[0] = OP_GEMV; oh[1] = 3; oh[2] = 6;
    }
    ((int*)(this->io.data() + this->R_HDR))[0] = OP_ROUTER;
    // the controllers' headers (attention: 3 controllers, ffn: 1), down as dense projection rows (12 channels of 32
    // rows, 16 of 8) of one x vector; the norms after each controller
    for (int a = 0; a < 2; a++) {
        int* hd = (int*)(this->io.data() + this->H_DN + a * 128);
        hd[0] = OP_PROJ; hd[1] = 1;
        for (int c = 0; c < (a ? 16 : 12); c++) ((int16_t*)hd)[46 + c] = (int16_t)(a ? 8 : 32);
        int* hu = (int*)(this->io.data() + this->H_UP + a * 128);
        hu[0] = OP_UP; hu[1] = a ? 1 : 3;
        int* hc = (int*)(this->io.data() + this->H_CT + a * 128);
        hc[0] = OP_CTOUT; hc[1] = a ? 2 : 1;
    }
    for (int il = CORE0; il < CORE1; il++) {
        const layer_weights& L = this->w.layer[il];
        for (int k = 0; k < 4; k++)
            memcpy(this->io.data() + this->FA + ((size_t)(il - CORE0) * 4 + k) * FA_LEN + (size_t)48 * YU * 4,
                   k == 3 ? L.ffn_norm : is_global(il) ? L.qkv_norm[k] : L.attn_norm, (size_t)D * 4);
    }
    ((int*)(this->io.data() + this->E_HDR))[0] = OP_EXPERT;
    // the attention regions: a sliding-window slot keeps SWA records, a global one max_len
    int ng = 0, ns = 0;
    for (int s = 0; s < NSLOT; s++) {
        const bool g = is_global(slot_layer(s));
        this->gidx[s] = g ? ng++ : -1;
        this->sidx[s] = g ? -1 : ns++;
    }
    this->RSS = REC0 + (size_t)SWA / KPU * UNIT;
    this->RSG = REC0 + (size_t)(this->max_len + KPU - 1) / KPU * UNIT;
    this->SW0 = (size_t)NL * LW;
    const size_t wt_bytes = this->SW0 + (size_t)ns * NKV * this->RSS, gl_bytes = (size_t)4 * NKV * this->RSG;
    if (ng != 8 || wt_bytes > BO_MAX || gl_bytes > BO_MAX)
        throw std::runtime_error("inkling2 fused: context " + std::to_string(this->max_len) + " too long for the decode buffers");
    this->wt = npu->create_bo_buffer<uint8_t>(wt_bytes);
    for (int i = 0; i < 2; i++) this->gl[i] = npu->create_bo_buffer<uint8_t>(gl_bytes);
    // the controllers' weights per looped layer: down unit 128 k + 64 h + r (ffn: 384 + 64 h + r) = row r's half h
    // (x / message); up unit 512 + 144 k + v (ffn: 944 + v) = rows 16 v .. 16 v + 15 as [64 ranks][16 forget | 16 gain]
    const size_t cw_bytes = (size_t)(CORE1 - CORE0) * CW_UNITS * UNIT;
    this->cw = npu->create_bo_buffer<uint8_t>(cw_bytes);
    memset(this->cw.data(), 0, cw_bytes);
    for (int il = CORE0; il < CORE1; il++) {
        uint8_t* b = this->cw.data() + (size_t)(il - CORE0) * CW_UNITS * UNIT;
        for (int k = 0; k < 4; k++) {
            const loop_ctrl& C = this->w.layer[il].ctrl[k];
            const int db = k < 3 ? 128 * k : 384, ub = k < 3 ? 512 + 144 * k : 944;
            for (int h = 0; h < 2; h++)
                for (int r = 0; r < RANK; r++) {
                    u16* d = (u16*)(b + (size_t)(db + 64 * h + r) * UNIT);
                    for (int c = 0; c < D; c++) d[c] = f2bf(C.down[(size_t)r * 2 * D + (size_t)h * D + c]);
                }
            for (int v = 0; v < 144; v++) {
                u16* d = (u16*)(b + (size_t)(ub + v) * UNIT);
                for (int kk = 0; kk < RANK; kk++)
                    for (int i = 0; i < 16; i++) {
                        d[32 * kk + i] = f2bf(C.forget[(size_t)(16 * v + i) * RANK + kk]);
                        d[32 * kk + 16 + i] = f2bf(C.gain[(size_t)(16 * v + i) * RANK + kk]);
                    }
            }
        }
    }
    this->cw.sync_to_device();
    memset(this->wt.data(), 0, this->SW0);
    // per layer: dense rows (bf16), router (fp32 as bf16 hi + lo), top-k header + expert bias
    for (int il = 0; il < NL; il++) {
        const layer_weights& L = this->w.layer[il];
        uint8_t* b = this->wt.data() + (size_t)il * LW;
        auto row = [&](int unit, const u16* r) { memcpy(b + (size_t)unit * UNIT, r, (size_t)D * 2); };
        for (int i = 0; i < NH; i++) row(i, L.sink_proj + (size_t)i * D);
        if (is_global(il))
            for (int i = 0; i < NKV * NCONV; i++) {
                row(32 + i, L.k_conv_proj + (size_t)i * D);
                row(64 + i, L.v_conv_proj + (size_t)i * D);
            }
        for (int c = 0; c < 16; c++)
            for (int kc = 0; kc < 36; kc++)
                for (int r = 0; r < 8; r++) {
                    u16* g = (u16*)(b + L_RW + (size_t)(c * 16 + 2 * r + kc / 18) * UNIT + (size_t)(kc % 18) * 256);
                    for (int k = 0; k < 64; k++) {
                        const float x = L.router[(size_t)(8 * c + r) * D + 64 * kc + k];
                        g[k] = f2bf(x);
                        g[64 + k] = f2bf(x - bf2f(g[k]));
                    }
                }
        ((int*)(b + L_TH))[0] = OP_TOPK;
        memcpy(b + L_TH + X_OFF, L.expert_bias, 4 * NE);
    }
}

void inkling2_fused::load_state(const model& m) {
    const int p = this->pos = this->pos0 = m.position();
    std::vector<float> cs((size_t)(NCONV - 1) * CONV_CH), cp((size_t)(NCONV - 1) * CONV_CH);
    std::vector<u16> kr(KVD), vr(KVD);
    // a mix's conv block: taps of channel c, then the ring rows 4 + (q & 3) of the inputs at q = p - 3 .. p - 1
    auto conv_block = [&](uint8_t* B, const float* taps, const float* state) {
        float* f = (float*)B;
        for (int c = 0; c < D; c++) {
            float* blk = f + (size_t)(c / YU) * 8 * YU + c % YU;
            for (int k = 0; k < NCONV; k++) blk[k * YU] = taps[c * NCONV + k];
            for (int k = 0; k < NCONV - 1; k++) blk[(4 + ((p - 3 + k) & 3)) * YU] = state[(size_t)k * CONV_CH + c];
        }
    };
    for (int j = 0; j < NAPP; j++) {
        const app_info I = info(j);
        const layer_weights& L = this->w.layer[I.il];
        m.slot_conv(I.slot, cs.data());
        // the attention region of each KV head
        int arg;
        size_t off, rs;
        region(I.slot, arg, off, rs);
        uint8_t* R = (arg == 1 ? this->wt : this->gl[arg - 2]).data() + off;
        memset(R, 0, NKV * rs);
        for (int g = 0; g < NKV; g++) {
            uint8_t* Rg = R + g * rs;
            float* st = (float*)Rg;
            for (int i = 0; i < 3; i++) {
                for (int c = 0; c < DH; c++) st[i * DH + c] = L.q_clip[(3 * g + i) * DH + c];
                for (int c = 0; c < NREL; c++) st[384 + i * NREL + c] = L.rel_clip[(3 * g + i) * NREL + c];
            }
            for (int half = 0; half < 2; half++) {
                const float* cw = half ? L.v_conv : L.k_conv;
                for (int k = 0; k < NCONV; k++)
                    for (int c = 0; c < DH; c++) st[432 + 512 * half + k * DH + c] = cw[(g * DH + c) * NCONV + k];
            }
            u16* rb = (u16*)(Rg + RB);
            for (int d = 0; d < REL_EXT; d++)
                for (int k = 0; k < NREL; k++) rb[((k / 8) * 1024 + 1023 - d) * 8 + k % 8] = f2bf(L.rel_bias[d * NREL + k]);
            float* rg = (float*)(Rg + RING);
            for (int k = 0; k < NCONV - 1; k++)
                for (int c = 0; c < DH; c++) {
                    rg[((p - 3 + k) & 3) * 256 + c] = cs[(size_t)k * CONV_CH + CONV_K + g * DH + c];
                    rg[((p - 3 + k) & 3) * 256 + DH + c] = cs[(size_t)k * CONV_CH + CONV_V + g * DH + c];
                }
        }
        // the records: global slots one per position, sliding-window slots position q at q mod SWA
        const int nrec = I.glob ? p : std::min(p, SWA);
        for (int r = 0; r < nrec; r++) {
            const int q = I.glob ? r : r + (p - 1 - r) / SWA * SWA;
            m.slot_row(I.slot, q, false, kr.data());
            m.slot_row(I.slot, q, true, vr.data());
            for (int g = 0; g < NKV; g++) {
                uint8_t* u = R + g * rs + REC0 + (size_t)(r / KPU) * UNIT;
                for (int c = 0; c < DH; c++) {
                    ((u16*)(u + (c / 8) * 128 + (r % KPU) * 16))[c % 8] = kr[g * DH + c];
                    ((u16*)(u + 2048 + (r % KPU) * 256))[c] = vr[g * DH + c];
                }
            }
        }
        // F block: attention conv, (residual: the A mix's), the FFN query and norm
        {
            const mix_cfg C = cfg_F(j);
            uint8_t* B = this->io.data() + this->BF[j];
            memset(B, 0, mix_bb(C));
            conv_block(B, L.attn_conv, cs.data() + CONV_ATTN);
            float* q = (float*)(B + CB + ((C.fl & 2) ? RES : 0));
            memcpy(q, L.ffn_query, (size_t)D * 4);
            if (C.nn) memcpy(q + D, L.ffn_norm, (size_t)D * 4);
        }
        // A block: the previous application's FFN conv, (residual: its F mix's), the attention queries (and norms);
        // a later pass's first application: the completion block here, the queries in BP
        auto attn_block = [&](size_t at_, const mix_cfg& C) {
            uint8_t* B = this->io.data() + at_;
            memset(B, 0, mix_bb(C));
            if (C.fl & 1) {
                const app_info P = info(j - 1);
                m.slot_conv(P.slot, cp.data());
                conv_block(B, this->w.layer[P.il].ffn_conv, cp.data() + CONV_FFN);
            }
            float* q = (float*)(B + ((C.fl & 1) ? CB : 0) + ((C.fl & 2) ? RES : 0));
            for (int i = 0; i < C.nq; i++) {
                if (L.attn_query) memcpy(q + (size_t)(1 + C.nn) * i * D, L.attn_query + (size_t)i * D, (size_t)D * 4);
                if (C.nn) memcpy(q + (size_t)(2 * i + 1) * D, I.glob ? L.qkv_norm[i] : L.attn_norm, (size_t)D * 4);
            }
        };
        if (I.pass > 0 && I.il == CORE0) {
            attn_block(this->BA[j], cfg_C());
            attn_block(this->BP[I.pass - 1], cfg_A(j));
        } else attn_block(this->BA[j], cfg_A(j));
    }
    // the output block: the last application's FFN conv, the final query and norm
    {
        const mix_cfg C = cfg_Z();
        uint8_t* B = this->io.data() + this->BA[NAPP];
        memset(B, 0, mix_bb(C));
        const app_info P = info(NAPP - 1);
        m.slot_conv(P.slot, cp.data());
        conv_block(B, this->w.layer[P.il].ffn_conv, cp.data() + CONV_FFN);
        float* q = (float*)(B + CB + RES);
        memcpy(q, this->w.final_query, (size_t)D * 4);
        memcpy(q + D, this->w.norm, (size_t)D * 4);
    }
    // the message queries of the looped layers, 4 each
    {
        uint8_t* B = this->io.data() + this->BM;
        memset(B, 0, mix_bb(cfg_M()));
        for (int il = CORE0; il < CORE1; il++) memcpy(B + (size_t)(il - CORE0) * 4 * D * 4, this->w.layer[il].msg_query, (size_t)4 * D * 4);
    }
    this->io.sync_to_device();
    this->wt.sync_to_device();
    for (auto& g : this->gl) g.sync_to_device();
}

void inkling2_fused::store_state(model& m) {
    const int p = this->pos, p0 = this->pos0;
    if (p == p0) return;
    if (!this->sync_all) this->io.sync_from_device();
    std::vector<float> cs((size_t)(NCONV - 1) * CONV_CH);
    std::vector<u16> kr(KVD), vr(KVD);
    // a mix's conv block ring rows 4 + (q & 3) -> the state rows of q = p - 3 .. p - 1
    auto conv_rows = [&](const uint8_t* B, float* state) {
        const float* f = (const float*)B;
        for (int c = 0; c < D; c++) {
            const float* blk = f + (size_t)(c / YU) * 8 * YU + c % YU;
            for (int k = 0; k < NCONV - 1; k++) state[(size_t)k * CONV_CH + c] = blk[(4 + ((p - 3 + k) & 3)) * YU];
        }
    };
    for (int j = 0; j < NAPP; j++) {
        const app_info I = info(j);
        int arg;
        size_t off, rs;
        region(I.slot, arg, off, rs);
        buffer<uint8_t>& bo = arg == 1 ? this->wt : this->gl[arg - 2];
        // the new records: global slots at q, sliding-window ones at q mod SWA (the last SWA positions only)
        const int q0 = I.glob ? p0 : std::max(p0, p - SWA);
        auto rec = [&](int q) { return I.glob ? q : q % SWA; };
        for (int g = 0; g < NKV; g++) {
            const size_t Rg = off + g * rs;
            bo.bo().sync(XCL_BO_SYNC_BO_FROM_DEVICE, (size_t)4 * 1024, Rg + RING);
            for (int q = q0; q < p;) {
                // one sync per run of consecutive records (a sliding-window slot wraps once at most)
                const int r0 = rec(q), n = I.glob ? p - q : std::min(p - q, SWA - r0);
                const size_t u0 = (size_t)r0 / KPU, u1 = (size_t)(r0 + n - 1) / KPU + 1;
                bo.bo().sync(XCL_BO_SYNC_BO_FROM_DEVICE, (u1 - u0) * UNIT, Rg + REC0 + u0 * UNIT);
                q += n;
            }
        }
        for (int q = q0; q < p; q++) {
            const int r = rec(q);
            for (int g = 0; g < NKV; g++) {
                const uint8_t* u = bo.data() + off + g * rs + REC0 + (size_t)(r / KPU) * UNIT;
                for (int c = 0; c < DH; c++) {
                    kr[g * DH + c] = ((const u16*)(u + (c / 8) * 128 + (r % KPU) * 16))[c % 8];
                    vr[g * DH + c] = ((const u16*)(u + 2048 + (r % KPU) * 256))[c];
                }
            }
            m.set_slot_row(I.slot, q, false, kr.data());
            m.set_slot_row(I.slot, q, true, vr.data());
        }
        // the convolution inputs: K / V rings, the attention conv (F block), the FFN conv (the next A block)
        for (int g = 0; g < NKV; g++) {
            const float* rg = (const float*)(bo.data() + off + g * rs + RING);
            for (int k = 0; k < NCONV - 1; k++)
                for (int c = 0; c < DH; c++) {
                    cs[(size_t)k * CONV_CH + CONV_K + g * DH + c] = rg[((p - 3 + k) & 3) * 256 + c];
                    cs[(size_t)k * CONV_CH + CONV_V + g * DH + c] = rg[((p - 3 + k) & 3) * 256 + DH + c];
                }
        }
        conv_rows(this->io.data() + this->BF[j], cs.data() + CONV_ATTN);
        conv_rows(this->io.data() + this->BA[j + 1], cs.data() + CONV_FFN);
        m.set_slot_conv(I.slot, cs.data());
    }
    m.set_position(p);
    this->pos0 = p;
}

void inkling2_fused::begin_token(int id) {
    const int p = this->pos;
    if (p >= this->max_len) throw std::runtime_error("inkling2 fused: context full");
    uint8_t* io = this->io.data();
    // h0 = rmsnorm(embed[id]) * embed_norm
    float* h0 = (float*)(io + this->ST);
    double ss = 0;
    for (int c = 0; c < D; c++) { h0[c] = bf2f(this->w.embed[(size_t)id * D + c]); ss += (double)h0[c] * h0[c]; }
    const float inv = (float)(1.0 / std::sqrt(ss / D + EPS));
    for (int c = 0; c < D; c++) h0[c] = h0[c] * inv * this->w.embed_norm[c];
    auto header = [&](size_t at_, const mix_cfg& C) {
        int* mh = (int*)(io + at_);
        mh[0] = OP_MIX; mh[1] = C.pre; mh[2] = C.fl; mh[3] = p; mh[4] = C.S; mh[5] = C.nq; mh[6] = C.om; mh[7] = C.nn; mh[8] = 0;
    };
    for (int k = 0; k < NPASS - 1; k++) {
        header(this->MHX + (size_t)k * 256, cfg_C());
        header(this->MHX + (size_t)k * 256 + 128, cfg_M());
    }
    header(this->MZ, cfg_Z());
    for (int j = 0; j < NAPP; j++) {
        const app_info I = info(j);
        header(this->MH + (size_t)j * 256, cfg_A(j));
        header(this->MH + (size_t)j * 256 + 128, cfg_F(j));
        const float tau = I.glob ? (float)(1.0 + LOG_SCALE_ALPHA * std::log(std::max(1.0, (p + 1) / (double)LOG_SCALE_FLOOR))) : 1.0f;
        for (int g = 0; g < NKV; g++) {
            int* h = (int*)(io + this->AH + (size_t)j * NKV * 128 + g * 128);
            h[0] = OP_ATTN; h[1] = p; h[2] = I.glob; memcpy(h + 3, &tau, 4); h[4] = I.glob ? p : std::min(p, SWA); h[5] = !I.glob; h[6] = g; h[7] = 0;
        }
    }
    this->io.sync_to_device();
}

void inkling2_fused::sync_range(size_t at, size_t n) { this->io.bo().sync(XCL_BO_SYNC_BO_TO_DEVICE, n, at); }

// INKLING2_CTRL_CHECK (test only): an out mode 3 query's bf16 x / rms(x) vs the host's from its fp32 x
void inkling2_fused::ctrl_check(const uint8_t* q) {
    static const bool on = getenv("INKLING2_CTRL_CHECK") && atoi(getenv("INKLING2_CTRL_CHECK"));
    if (!on) return;
    const float* x = (const float*)q;
    const u16* b = (const u16*)(q + (size_t)D * 4);
    double ss = 0;
    for (int c = 0; c < D; c++) ss += (double)x[c] * x[c];
    const float ix = (float)(1.0 / std::sqrt(ss / D + EPS));
    int off = 0;
    float worst = 0;
    for (int c = 0; c < D; c++) {
        const u16 h = f2bf(x[c] * ix);
        off += b[c] != h;
        worst = std::max(worst, std::fabs(bf2f(b[c]) - bf2f(h)));
    }
    this->ck_n++;
    this->ck_off += off;
    this->ck_worst = std::max(this->ck_worst, worst);
}

// a loop pass's attention inputs: the A mix's raw output(s) through the q / k / v controllers, then the norms -> the
// projections' x (bf16)
void inkling2_fused::host_attn_ctrl(int j) {
    const app_info I = info(j);
    const layer_weights& L = this->w.layer[I.il];
    const uint8_t* x = this->io.data() + this->XC;
    const uint8_t* ms = this->io.data() + this->MSG + (size_t)(I.il - CORE0) * 4 * CQ;
    u16* xo = (u16*)(this->io.data() + this->H3 + X_OFF);
    float y[D];
    for (int k = 0; k < 3; k++) {
        const uint8_t* xk = x + (I.glob ? k * CQ : 0);
        ctrl_check(xk);
        ctrl_check(ms + k * CQ);
        control_norm((const float*)xk, (const float*)(ms + k * CQ), L.ctrl[k], I.glob ? L.qkv_norm[k] : L.attn_norm, y);
        for (int c = 0; c < D; c++) xo[(size_t)k * D + c] = f2bf(y[c]);
    }
    sync_range(this->H3 + X_OFF, (size_t)3 * D * 2);
}

// a loop pass's FFN input: the F mix's raw output through the ffn controller, the norm -> router / expert x (hi lo hi)
void inkling2_fused::host_ffn_ctrl(int j) {
    const app_info I = info(j);
    const layer_weights& L = this->w.layer[I.il];
    const uint8_t* x = this->io.data() + this->XF;
    const uint8_t* ms = this->io.data() + this->MSG + ((size_t)(I.il - CORE0) * 4 + 3) * CQ;
    u16* xo = (u16*)(this->io.data() + this->E_HDR + X_OFF);
    float y[D];
    ctrl_check(x);
    ctrl_check(ms);
    control_norm((const float*)x, (const float*)ms, L.ctrl[3], L.ffn_norm, y);
    for (int c = 0; c < D; c++) {
        xo[c] = xo[2 * D + c] = f2bf(y[c]);
        xo[D + c] = f2bf(y[c] - bf2f(xo[c]));
    }
    sync_range(this->E_HDR + X_OFF, (size_t)3 * D * 2);
}

size_t inkling2_fused::ctrl_q(int j, bool attn, int k, int h) const {
    const app_info I = info(j);
    if (h) return this->MSG + ((size_t)(I.il - CORE0) * 4 + (attn ? k : 3)) * CQ;
    return attn ? this->XC + (I.glob ? k * CQ : 0) : this->XF;
}

void inkling2_fused::add_ctrl(int j, bool attn, int upto, size_t ct_at) {
    const app_info I = info(j);
    const int li = I.il - CORE0, nk = attn ? 3 : 1, nch = attn ? 12 : 16, nd = attn ? 32 : 8;
    const size_t CWL = (size_t)li * CW_UNITS * UNIT, hs = attn ? 0 : 128;
    auto fa = [&](int k) { return this->FA + ((size_t)li * 4 + (attn ? k : 3)) * FA_LEN; };
    npu_sequence* seq = this->app().seq();
    // down: channel c = rows nd c .. of [controller][half] (32 c: k = c / 4, h = c / 2 % 2; ffn 8 c: h = c / 8), the
    // half's bf16 x / rms -> one y unit of partials at GD + c
    for (int c = 0; c < nch; c++) {
        const int col = c % 8, r = c / 8, k = attn ? c / 4 : 0, h = attn ? c / 2 % 2 : c / 8;
        seq->npu_dma_memcpy_nd(4, 0, S2MM, IT[col], npu_bd_id(4 + r), chn(r), {0, 0, 0, (uint32_t)((this->GD + (size_t)c * YU * 4) / 4)}, {1, 1, 1, (uint32_t)YU}, {0, 0, 0, 1});
        seq->npu_dma_memcpy_nd(1, 0, MM2S, IT[col], npu_bd_id(2 * r), chn(r), {0, 0, 0, (uint32_t)(this->H_DN + hs)}, {1, 1, 1, (uint32_t)X_OFF}, {0, 0, 0, 1});
        seq->npu_dma_memcpy_nd(1, 0, MM2S, IT[col], npu_bd_id(2 * r + 1), chn(r), {0, 0, 0, (uint32_t)(ctrl_q(j, attn, k, h) + (size_t)D * 4)}, {1, 1, 1, (uint32_t)(D * 2)}, {0, 0, 0, 1});
        seq->npu_dma_memcpy_nd(1, 4, MM2S, IT[col], npu_bd_id(8 + r), chn(r), {0, 0, 0, (uint32_t)(CWL + (size_t)(attn ? 32 * c : 384 + 8 * c) * UNIT)}, {1, 1, 1, (uint32_t)(nd * UNIT)}, {0, 0, 0, 1});
    }
    for (int c = 0; c < nch; c++) seq->npu_dma_wait(IT[c % 8], S2MM, chn(c / 8));
    if (upto < 2) return;
    // up: every channel gathers the partials P[k][h][64], then its 9 nk units of [64 ranks][16 forget | 16 gain]
    // -> 3 nk y units of tanh f / a, into controller (y unit / 48)'s FA region (two pieces where a channel straddles)
    int nw[16];
    for (int c = 0; c < 16; c++) {
        const int col = c % 8, r = c / 8, y0 = 3 * nk * c, ny = 3 * nk, k0 = y0 / 48, n0 = std::min(ny, 48 * (k0 + 1) - y0);
        seq->npu_dma_memcpy_nd(4, 0, S2MM, IT[col], npu_bd_id(4 + r), chn(r), {0, 0, 0, (uint32_t)((fa(k0) + (size_t)(y0 - 48 * k0) * YU * 4) / 4)}, {1, 1, 1, (uint32_t)(n0 * YU)}, {0, 0, 0, 1});
        nw[c] = 1;
        if (n0 < ny) {
            seq->npu_dma_memcpy_nd(4, 0, S2MM, IT[col], npu_bd_id(6 + r), chn(r), {0, 0, 0, (uint32_t)(fa(k0 + 1) / 4)}, {1, 1, 1, (uint32_t)((ny - n0) * YU)}, {0, 0, 0, 1});
            nw[c]++;
        }
        seq->npu_dma_memcpy_nd(1, 0, MM2S, IT[col], npu_bd_id(2 * r), chn(r), {0, 0, 0, (uint32_t)(this->H_UP + hs)}, {1, 1, 1, (uint32_t)X_OFF}, {0, 0, 0, 1});
        seq->npu_dma_memcpy_nd(4, 0, MM2S, IT[col], npu_bd_id(2 * r + 1), chn(r), {0, 0, 0, (uint32_t)(this->GD / 4)}, {1, 1, (uint32_t)nch, (uint32_t)nd}, {0, 0, (uint32_t)YU, 1});
        seq->npu_dma_memcpy_nd(1, 0, MM2S, IT[col], npu_bd_id(8 + r), chn(r), {0, 0, 0, 0}, {1, 1, 1, (uint32_t)(UNIT - X_OFF - 512 * nk)}, {0, 0, 0, 1});
        seq->npu_dma_memcpy_nd(1, 4, MM2S, IT[col], npu_bd_id(10 + r), chn(r), {0, 0, 0, (uint32_t)(CWL + (size_t)(attn ? 512 + 27 * c : 944 + 9 * c) * UNIT)}, {1, 1, 1, (uint32_t)(9 * nk * UNIT)}, {0, 0, 0, 1});
    }
    for (int c = 0; c < 16; c++)
        for (int i = 0; i < nw[c]; i++) seq->npu_dma_wait(IT[c % 8], S2MM, chn(c / 8));
    if (upto < 3) return;
    // out: controller k on glue_dn (1 + k, 5): x, message (fp32), its FA region -> the projections' x k (bf16), or
    // the router / experts' x (hi, lo, hi)
    for (int k = 0; k < nk; k++) {
        const size_t out = ct_at ? ct_at + (size_t)k * D * 2 : attn ? this->H3 + X_OFF + (size_t)k * D * 2 : this->E_HDR + X_OFF;
        seq->npu_dma_memcpy_nd(4, 0, S2MM, IT[1 + k], npu_bd_id(4), it_channel_1, {0, 0, 0, (uint32_t)(out / 4)}, {1, 1, 1, (uint32_t)((attn ? 12 : 36) * YU)}, {0, 0, 0, 1});
        seq->npu_dma_memcpy_nd(1, 0, MM2S, IT[1 + k], npu_bd_id(0), it_channel_1, {0, 0, 0, (uint32_t)(this->H_CT + hs)}, {1, 1, 1, (uint32_t)X_OFF}, {0, 0, 0, 1});
        seq->npu_dma_memcpy_nd(1, 0, MM2S, IT[1 + k], npu_bd_id(1), it_channel_1, {0, 0, 0, (uint32_t)ctrl_q(j, attn, k, 0)}, {1, 1, 1, (uint32_t)(D * 4)}, {0, 0, 0, 1});
        seq->npu_dma_memcpy_nd(1, 0, MM2S, IT[1 + k], npu_bd_id(2), it_channel_1, {0, 0, 0, (uint32_t)ctrl_q(j, attn, k, 1)}, {1, 1, 1, (uint32_t)(D * 4)}, {0, 0, 0, 1});
        seq->npu_dma_memcpy_nd(1, 0, MM2S, IT[1 + k], npu_bd_id(3), it_channel_1, {0, 0, 0, (uint32_t)fa(k)}, {1, 1, 1, (uint32_t)FA_LEN}, {0, 0, 0, 1});
    }
    for (int k = 0; k < nk; k++) seq->npu_dma_wait(IT[1 + k], S2MM, it_channel_1);
}

// INKLING2_CTRL_CHECK >= 2 (test only): after the host controllers, the device's in a run of their own (out to CK),
// each step against the host's from the device's own inputs
void inkling2_fused::dev_check(int j, bool attn, int level) {
    const app_info I = info(j);
    const int li = I.il - CORE0, nk = attn ? 3 : 1;
    this->app().seq()->clear_cmds();
    this->t_build = std::chrono::steady_clock::now();
    add_ctrl(j, attn, level, this->CK);
    this->want.push_back({this->GD, (size_t)16 * YU * 4});
    this->want.push_back({this->FA + (size_t)li * 4 * FA_LEN, 4 * FA_LEN});
    this->want.push_back({this->CK, (size_t)36 * YU * 4});
    if (!submit(0)) throw std::runtime_error("inkling2 fused: controller check run did not complete");
    const uint8_t* io = this->io.data();
    const float* gd = (const float*)(io + this->GD);
    for (int k = 0; k < nk; k++) {
        const loop_ctrl& C = this->w.layer[I.il].ctrl[attn ? k : 3];
        float P[2][RANK], gmax = 0, derr = 0;
        for (int h = 0; h < 2; h++) {
            const u16* xv = (const u16*)(io + ctrl_q(j, attn, k, h) + (size_t)D * 4);
            for (int r = 0; r < RANK; r++) {
                P[h][r] = attn ? gd[(4 * k + 2 * h + r / 32) * YU + r % 32] : gd[(8 * h + r / 8) * YU + r % 8];
                double s = 0;
                for (int c = 0; c < D; c++) s += (double)C.down[(size_t)r * 2 * D + (size_t)h * D + c] * bf2f(xv[c]);
                gmax = std::max(gmax, (float)std::fabs(s));
                derr = std::max(derr, (float)std::fabs(s - P[h][r]));
            }
        }
        this->dk_down = std::max(this->dk_down, derr / std::max(gmax, 1e-20f));
        if (level < 2) continue;
        u16 g[RANK];
        for (int r = 0; r < RANK; r++) g[r] = f2bf(P[0][r] + P[1][r]);
        const float* fy = (const float*)(io + this->FA + ((size_t)li * 4 + (attn ? k : 3)) * FA_LEN);
        for (int row = 0; row < D; row++) {
            double f = 0, a = 0;
            for (int kk = 0; kk < RANK; kk++) {
                f += (double)C.forget[(size_t)row * RANK + kk] * bf2f(g[kk]);
                a += (double)C.gain[(size_t)row * RANK + kk] * bf2f(g[kk]);
            }
            const float* yb = fy + 96 * (row / 48) + 32 * (row % 48 / 16) + row % 16;
            this->dk_up = std::max(this->dk_up, (float)std::max(std::fabs(std::tanh(f) - yb[0]), std::fabs(std::tanh(a) - yb[16])));
        }
        if (level < 3) continue;
        const u16* ho = attn ? (const u16*)(io + this->H3 + X_OFF) + (size_t)k * D : (const u16*)(io + this->E_HDR + X_OFF);
        const u16* dv = (const u16*)(io + this->CK) + (size_t)k * D;
        // the bf16 outputs (ffn: hi, not lo): elements that differ, more than 1 ulp
        for (int c = 0; c < (attn ? D : 3 * D); c++) {
            if (c >= D && c < 2 * D) continue;
            this->dk_el++;
            this->dk_off += ho[c] != dv[c];
            this->dk_far += std::abs((int)(int16_t)ho[c] - (int)(int16_t)dv[c]) > 1;
            this->dk_out = std::max(this->dk_out, std::fabs(bf2f(ho[c]) - bf2f(dv[c])));
        }
    }
    this->dk_n++;
}

bool inkling2_fused::run_app(int j, int stages) {
    if (info(j).pass == 0 || this->dev_ctrl) return run_steps(j, stages);
    // host controllers (INKLING2_HOST_CTRL): between the mixes and the steps that read their outputs
    if (stages & ST_A) {
        if (!run_steps(j, ST_A)) return false;
        const auto t0 = std::chrono::steady_clock::now();
        host_attn_ctrl(j);
        this->t_ctrl += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    }
    if (stages & (ST_PROJ | ST_ATTN | ST_OPROJ | ST_F)) {
        if (!run_steps(j, stages & (ST_PROJ | ST_ATTN | ST_OPROJ | ST_F))) return false;
        const auto t0 = std::chrono::steady_clock::now();
        if (stages & ST_F) host_ffn_ctrl(j);
        this->t_ctrl += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    }
    return !(stages & ST_MOE) || run_steps(j, ST_MOE);
}

// the token's 48 applications, up to `chain` applications per run: pass-0 applications whole, a loop application's MoE
// with the following steps; with host controllers runs end where they need the A / F mixes' outputs
bool inkling2_fused::run_token() {
    if (this->dev_ctrl && this->chain >= NAPP && this->pipe) return run_token_pipe();
    this->next_run.reset();
    this->next_pos = -1;
    npu_sequence* seq = this->app().seq();
    static const int dev_ck = getenv("INKLING2_CTRL_CHECK") ? atoi(getenv("INKLING2_CTRL_CHECK")) : 0;
    int pending = 0, last = 0;
    auto start = [&]() {
        if (pending) return;
        seq->clear_cmds();
        this->t_build = std::chrono::steady_clock::now();
    };
    auto flush = [&]() {
        if (!pending) return true;
        pending = 0;
        return submit(last);
    };
    auto add = [&](int j, int st, bool count) {
        start();
        add_steps(j, st);
        pending += count ? 1 : 0;
        if (!count) pending = std::max(pending, 1);
        last = st;
    };
    for (int j = 0; j < NAPP; j++) {
        const app_info I = info(j);
        if (I.pass == 0 || this->dev_ctrl) {
            add(j, ST_ALL | ST_Z, true);
        } else {
            add(j, ST_A, false);
            if (!flush()) return false;
            auto t0 = std::chrono::steady_clock::now();
            host_attn_ctrl(j);
            this->t_ctrl += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            if (dev_ck > 1) dev_check(j, true, dev_ck - 1);
            add(j, ST_PROJ | ST_ATTN | ST_OPROJ | ST_F, false);
            if (!flush()) return false;
            t0 = std::chrono::steady_clock::now();
            host_ffn_ctrl(j);
            this->t_ctrl += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            if (dev_ck > 1) dev_check(j, false, dev_ck - 1);
            add(j, ST_MOE, true);
        }
        if (pending >= this->chain && !flush()) return false;
    }
    return flush();
}

// one run per token: start this position's run (created during the previous run if it was for this position), then
// build and create the next position's in the other app while this one is on the device
bool inkling2_fused::run_token_pipe() {
    const int p = this->pos;
    auto build = [&](int at) {
        this->cur = at & 1;
        this->pos = at;
        npu_sequence* seq = this->app().seq();
        seq->clear_cmds();
        for (int j = 0; j < NAPP; j++) add_steps(j, ST_ALL | ST_Z);
        seq->cmds2seq();
        xrt::run r = this->app().create_run(this->io, this->wt, this->gl[0], this->gl[1], this->cw);
        this->pos = p;
        this->want.clear();
        return r;
    };
    const auto t0 = std::chrono::steady_clock::now();
    std::optional<xrt::run> run;
    if (this->next_run && this->next_pos == p) run = std::move(this->next_run);
    else run = build(p);
    this->next_run.reset();
    this->next_pos = -1;
    const auto t1 = std::chrono::steady_clock::now();
#if !defined(FLM_USE_HRX)
    if (this->lmh) {
        // the token and its lm_head as one window; the next position's run is built while it is on the device
        std::unique_ptr<window> win = start_window(*run, true);
        if (p + 1 < this->max_len) {
            this->next_run = build(p + 1);
            this->next_pos = p + 1;
        }
        const auto t2 = std::chrono::steady_clock::now();
        if (!wait_window(std::move(win))) {
            this->next_run.reset();
            this->next_pos = -1;
            return false;
        }
        const auto t3 = std::chrono::steady_clock::now();
        this->io.bo().sync(XCL_BO_SYNC_BO_FROM_DEVICE, (size_t)D * 2, this->XZ);
        const auto t4 = std::chrono::steady_clock::now();
        auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
        this->t_seq += ms(t0, t1);
        this->t_pre += ms(t1, t2);
        this->t_run += ms(t1, t3);
        this->t_sync += ms(t3, t4);
        this->n_run++;
        return true;
    }
#endif
    run->start();
    if (p + 1 < this->max_len) {
        this->next_run = build(p + 1);
        this->next_pos = p + 1;
    }
    const auto t2 = std::chrono::steady_clock::now();
    const ert_cmd_state st = wait_run(*run, 5000);
    if (st != ERT_CMD_STATE_COMPLETED) {
        this->next_run.reset();
        this->next_pos = -1;
        return false;
    }
    const auto t3 = std::chrono::steady_clock::now();
    this->io.bo().sync(XCL_BO_SYNC_BO_FROM_DEVICE, (size_t)D * 2, this->XZ);
    const auto t4 = std::chrono::steady_clock::now();
    auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    this->t_seq += ms(t0, t1);
    this->t_pre += ms(t1, t2);
    this->t_run += ms(t1, t3);
    this->t_sync += ms(t3, t4);
    this->n_run++;
    this->t_by[(ST_ALL | ST_Z) & 127] += ms(t1, t3);
    this->n_by[(ST_ALL | ST_Z) & 127]++;
    return true;
}

bool inkling2_fused::enable_lm_head(npu_xclbin_manager* npu, const std::string& switches) {
#if defined(FLM_USE_HRX)
    (void)npu;
    (void)switches;
    return false;
#else
    try {
        this->ll.names = {"cold_dec", "dec_then_lmh", "lmh_then_dec"};
        this->ll.load_file(switches, this->mgr->get_device(), this->mgr->get_context(), this->mgr->get_kernel_name());
        this->lmh = std::make_unique<inkling2_lm_head>(npu, this->mgr, this->w, &this->io, this->XZ);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "inkling2 fused: lm_head on the host (%s)\n", e.what());
        this->lmh.reset();
        this->ll.sw.clear();
        this->ll.ready = false;
        return false;
    }
    return true;
#endif
}

#if !defined(FLM_USE_HRX)
// one submission: cold_dec (the array as the decode's whatever ran on it since), the decode run, and with lm the switch
// to the lm_head and its run. Another process's context can take the array between submissions: hence cold_dec first
// and the system-wide lock (inkling2_array_lock) around it. A window that does not complete: the device is suspect.
std::unique_ptr<inkling2_fused::window> inkling2_fused::start_window(xrt::run& run, bool lm) {
    auto w = std::make_unique<window>();
    w->lm = lm;
    w->rl = this->mgr->create_runlist();
    w->rl.add(this->ll.make_run("cold_dec"));
    w->rl.add(run);
    if (lm) {
        w->guest = this->lmh->guest_run();
        w->rl.add(this->ll.make_run("dec_then_lmh"));
        w->rl.add(w->guest);
    }
    w->rl.execute();
    return w;
}

bool inkling2_fused::wait_window(std::unique_ptr<window> w) {
    const bool lm = w->lm;
    std::cv_status cs = std::cv_status::timeout;
    bool failed = false;
    try { cs = w->rl.wait(std::chrono::milliseconds(5000)); } catch (const std::exception& e) { failed = true; std::fprintf(stderr, "inkling2 fused: window: %s\n", e.what()); }
    if (failed || cs != std::cv_status::no_timeout) {
        if (FILE* f = getenv("INKLING2_SUSPECT_FILE") ? fopen(getenv("INKLING2_SUSPECT_FILE"), "w") : nullptr) {
            std::fprintf(f, "inkling2 fused window (lm %d) did not complete\n", (int)lm);
            fclose(f);
        }
        std::fprintf(stderr, "inkling2 fused: WINDOW DID NOT COMPLETE: holding the context 90 s\n");
        std::this_thread::sleep_for(std::chrono::seconds(90));
        return false;
    }
    this->ll.launches += lm ? 2 : 1;
    this->lmh_state = lm;
    return true;
}
#endif

// a mix on glue_dn (1, 5): S2MM ring row (raw prefix) into its conv block, cur, the outputs; then the inputs
void inkling2_fused::add_mix(const mix_cfg& C, size_t MH_, size_t S0, size_t P0, size_t B0, size_t cur_at, size_t out_at) {
    npu_sequence* seq = this->app().seq();
    const npu_it_channel ch = it_channel_1;
    const int p = this->pos, n_out = mix_nout(C);
    int nw = 0;
    if (C.fl & 1) {
        seq->npu_dma_memcpy_nd(4, 0, S2MM, IT[1], npu_bd_id(4), ch, {0, 0, 0, (uint32_t)((B0 + (size_t)(4 + (p & 3)) * YU * 4) / 4)}, {1, 1, CH, YU}, {0, 0, 8 * YU, 1});
        nw++;
    }
    if (C.fl & 4) {
        seq->npu_dma_memcpy_nd(4, 0, S2MM, IT[1], npu_bd_id(5), ch, {0, 0, 0, (uint32_t)(cur_at / 4)}, {1, 1, 1, (uint32_t)(24 * YU)}, {0, 0, 0, 1});
        nw++;
    }
    if (n_out) {
        seq->npu_dma_memcpy_nd(4, 0, S2MM, IT[1], npu_bd_id(6), ch, {0, 0, 0, (uint32_t)(out_at / 4)}, {1, 1, 1, (uint32_t)(n_out * YU)}, {0, 0, 0, 1});
        nw++;
    }
    seq->npu_dma_memcpy_nd(1, 0, MM2S, IT[1], npu_bd_id(0), ch, {0, 0, 0, (uint32_t)MH_}, {1, 1, 1, (uint32_t)X_OFF}, {0, 0, 0, 1});
    if (C.S) seq->npu_dma_memcpy_nd(1, 0, MM2S, IT[1], npu_bd_id(1), ch, {0, 0, 0, (uint32_t)S0}, {1, 1, 1, (uint32_t)((size_t)C.S * D * 4)}, {0, 0, 0, 1});
    seq->npu_dma_memcpy_nd(1, 0, MM2S, IT[1], npu_bd_id(2), ch, {0, 0, 0, (uint32_t)P0}, {1, 1, 1, (uint32_t)mix_pb(C)}, {0, 0, 0, 1});
    seq->npu_dma_memcpy_nd(1, 0, MM2S, IT[1], npu_bd_id(3), ch, {0, 0, 0, (uint32_t)B0}, {1, 1, 1, (uint32_t)mix_bb(C)}, {0, 0, 0, 1});
    for (int i = 0; i < nw; i++) seq->npu_dma_wait(IT[1], S2MM, ch);
}

bool inkling2_fused::run_steps(int j, int stages) {
    this->next_run.reset();
    this->next_pos = -1;
    this->app().seq()->clear_cmds();
    this->t_build = std::chrono::steady_clock::now();
    add_steps(j, stages);
    return submit(stages);
}

void inkling2_fused::add_steps(int j, int stages) {
    const int p = this->pos;
    const app_info I = info(j);
    const proj_plan& PP = I.three ? this->pp3 : this->pp1;
    const size_t H = I.three ? this->H3 : this->H1;
    const layer_src& Ls = this->src[I.il];
    const uint64_t wa = Ls.w->bo().address() + 0x80000000ull;
    npu_sequence* seq = this->app().seq();
    // an MM2S BD at an absolute device address (any buffer): written unpushed, its address patched, then pushed
    auto abs_mm2s = [&](int col, int r, int bd, uint64_t a, int elem, const std::vector<uint32_t>& sizes, const std::vector<uint32_t>& strides) {
        seq->npu_dma_memcpy_nd(elem, 0, MM2S, IT[col], npu_bd_id(bd), chn(r), {0, 0, 0, 0}, sizes, strides, -1, 0, false, normal_cache, 15, false);
        const uint32_t bda = 0x1D000u + (uint32_t)bd * 0x20u;
        seq->rtp_write(IT[col], bda + 4, (uint32_t)a);
        seq->npu_maskwrite(IT[col], bda + 8, (uint32_t)(a >> 32) & 0xFFFFu, 0xFFFFu);
        seq->rtp_write(IT[col], r ? 0x1D21Cu : 0x1D214u, (uint32_t)bd);
    };
    if (stages & ST_A) {
        const mix_cfg C = cfg_A(j);
        if (I.pass > 0 && I.il == CORE0) {
            // the previous pass's output o17 (-> BANK), its messages from [o11, o17], then this pass's first mix
            // over [h0] and the prelude output o5 (both already in ST)
            const size_t MX = this->MHX + (size_t)(I.pass - 1) * 256;
            add_mix(cfg_C(), MX, this->ST, this->TY, this->BA[j], this->BANK, 0);
            add_mix(cfg_M(), MX + 128, this->ST + (size_t)2 * D * 4, this->BANK, this->BM, 0, this->MSG);
            add_mix(C, this->MH + (size_t)j * 256, this->ST, this->ST + (size_t)D * 4, this->BP[I.pass - 1], 0, this->XC);
        } else {
            const size_t cur_at = I.starts ? this->ST + (size_t)I.S * D * 4 : this->BF[j] + CB;
            add_mix(C, this->MH + (size_t)j * 256, this->ST, j == 0 ? this->ST : this->TY, this->BA[j], cur_at, I.pass > 0 ? this->XC : H + X_OFF);
        }
        if (I.pass > 0 && this->dev_ctrl) add_ctrl(j, true, 3, 0);
    }
    if (stages & ST_PROJ) {
        for (int c = 0; c < 16; c++) {
            const int col = c % 8, r = c / 8;
            for (size_t i = 0; i < PP.runs[c].size(); i++)
                seq->npu_dma_memcpy_nd(4, 0, S2MM, IT[col], npu_bd_id((i ? 12 : 4) + r), chn(r), {0, 0, 0, (uint32_t)((this->PY + (size_t)PP.runs[c][i].first * YU * 4) / 4)}, {1, 1, 1, (uint32_t)(PP.runs[c][i].second * YU)}, {0, 0, 0, 1});
            seq->npu_dma_memcpy_nd(1, 0, MM2S, IT[col], npu_bd_id(2 * r), chn(r), {0, 0, 0, (uint32_t)H}, {1, 1, 1, (uint32_t)PP.hdr}, {0, 0, 0, 1});
            abs_mm2s(col, r, 2 * r + 1, wa + (size_t)PP.g0[c] * GB, 1, {1, 1, 1, (uint32_t)(PP.ng[c] * GB)}, {0, 0, 0, 1});
            if (PP.nd_of[c]) {
                const auto& d = PP.dns[PP.nd_of[c] - 1];
                seq->npu_dma_memcpy_nd(1, 1, MM2S, IT[col], npu_bd_id(8 + r), chn(r), {0, 0, 0, (uint32_t)((size_t)I.il * LW + (size_t)d.u * 32 * UNIT)}, {1, 1, 1, (uint32_t)(d.n * UNIT)}, {0, 0, 0, 1});
                seq->npu_dma_memcpy_nd(4, 0, S2MM, IT[col], npu_bd_id(10 + r), chn(r), {0, 0, 0, (uint32_t)((this->PY + (size_t)(26 + d.u) * YU * 4) / 4)}, {1, 1, 1, (uint32_t)YU}, {0, 0, 0, 1});
            }
        }
        for (int c = 0; c < 16; c++)
            for (size_t i = 0; i < PP.runs[c].size() + (PP.nd_of[c] ? 1 : 0); i++) seq->npu_dma_wait(IT[c % 8], S2MM, chn(c / 8));
    }
    if (stages & ST_ATTN) {
        int arg;
        size_t W0, rs;
        region(I.slot, arg, W0, rs);
        const int nk = I.glob ? p : std::min(p, SWA), rec = I.glob ? p : p % SWA;
        const size_t AHj = this->AH + (size_t)j * NKV * 128, Q0 = this->PY + (size_t)QS * YU * 4;
        for (int g = 0; g < NKV; g++) {
            const size_t rsg = W0 + g * rs, ru = rsg + REC0 + (size_t)(rec / KPU) * UNIT;
            const size_t o_at = this->OH + (size_t)(g / 4) * UNIT + X_OFF + (g % 4) * 768;
            seq->npu_dma_memcpy_nd(4, arg, S2MM, IT[g], npu_bd_id(4), it_channel_0, {0, 0, 0, (uint32_t)((rsg + RING + (p & 3) * 1024) / 4)}, {1, 1, 1, 256}, {0, 0, 0, 1});
            seq->npu_dma_memcpy_nd(4, arg, S2MM, IT[g], npu_bd_id(5), it_channel_0, {0, 0, 0, (uint32_t)((ru + (rec % KPU) * 16) / 4)}, {1, 1, 16, 4}, {0, 0, 32, 1});
            seq->npu_dma_memcpy_nd(4, arg, S2MM, IT[g], npu_bd_id(7), it_channel_0, {0, 0, 0, (uint32_t)((ru + 2048 + (rec % KPU) * 256) / 4)}, {1, 1, 1, 64}, {0, 0, 0, 1});
            seq->npu_dma_memcpy_nd(4, 0, S2MM, IT[g], npu_bd_id(6), it_channel_0, {0, 0, 0, (uint32_t)(o_at / 4)}, {1, 1, 1, 192}, {0, 0, 0, 1});
            seq->npu_dma_memcpy_nd(1, 0, MM2S, IT[g], npu_bd_id(0), it_channel_0, {0, 0, 0, (uint32_t)(AHj + g * 128)}, {1, 1, 1, 128}, {0, 0, 0, 1});
            seq->npu_dma_memcpy_nd(1, 0, MM2S, IT[g], npu_bd_id(1), it_channel_0, {0, 0, 0, (uint32_t)(Q0 + g * 1536)}, {1, 1, 2, 1536}, {0, 0, QD * 4, 1});
            seq->npu_dma_memcpy_nd(1, 0, MM2S, IT[g], npu_bd_id(2), it_channel_0, {0, 0, 0, (uint32_t)this->PY}, {1, 1, 1, (uint32_t)A_LEN}, {0, 0, 0, 1});
            seq->npu_dma_memcpy_nd(1, arg, MM2S, IT[g], npu_bd_id(3), it_channel_0, {0, 0, 0, (uint32_t)rsg}, {1, 1, 1, (uint32_t)(REC0 + (size_t)(nk + KPU - 1) / KPU * UNIT)}, {0, 0, 0, 1});
        }
        for (int g = 0; g < NKV; g++)
            for (int i = 0; i < 4; i++) seq->npu_dma_wait(IT[g], S2MM, it_channel_0);
    }
    if (stages & ST_OPROJ) {
        for (int c = 0; c < 16; c++) {
            const int col = c % 8, r = c / 8, hu = c / 8, jj = c % 8;
            seq->npu_dma_memcpy_nd(4, 0, S2MM, IT[col], npu_bd_id(4 + r), chn(r), {0, 0, 0, (uint32_t)((this->OY + (size_t)(hu * OG + 3 * jj) * YU * 4) / 4)}, {1, 1, 1, (uint32_t)(3 * YU)}, {0, 0, 0, 1});
            seq->npu_dma_memcpy_nd(1, 0, MM2S, IT[col], npu_bd_id(2 * r), chn(r), {0, 0, 0, (uint32_t)(this->OH + hu * UNIT)}, {1, 1, 1, (uint32_t)UNIT}, {0, 0, 0, 1});
            abs_mm2s(col, r, 2 * r + 1, wa + Ls.o + (size_t)(3 * jj * ONC + 6 * hu) * 3 * UNIT, 4, {1, 3, 333, 64}, {0, (uint32_t)(ONC * 3 * UNIT / 4), 64, 1});
        }
        for (int c = 0; c < 16; c++) seq->npu_dma_wait(IT[c % 8], S2MM, chn(c / 8));
    }
    if (stages & ST_F)
        add_mix(cfg_F(j), this->MH + (size_t)j * 256 + 128, this->ST, this->OY, this->BF[j], j + 1 < NAPP ? this->BA[j + 1] + CB : this->FIN,
                I.pass > 0 ? this->XF : this->E_HDR + X_OFF);
    if ((stages & ST_F) && I.pass > 0 && this->dev_ctrl) add_ctrl(j, false, 3, 0);
    if (stages & ST_MOE) {
        const uint64_t gu_a = wa + Ls.gate_up, dn_a = wa + Ls.down;
        const size_t LB = (size_t)I.il * LW;
        const npu_tiles HUB = get_tile(5, 0);
        // the BDs the hub points at the experts (linear, not started), hub args, start
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
            seq->npu_dma_memcpy_nd(4, 0, S2MM, IT[col], npu_bd_id(8 + r), chn(r), {0, 0, 0, (uint32_t)((this->RY + c * YU * 4) / 4)}, {1, 1, 1, (uint32_t)YU}, {0, 0, 0, 1});
            seq->npu_dma_memcpy_nd(1, 0, MM2S, IT[col], npu_bd_id(2 * r), chn(r), {0, 0, 0, (uint32_t)this->R_HDR}, {1, 1, 1, (uint32_t)X_OFF}, {0, 0, 0, 1});
            seq->npu_dma_memcpy_nd(1, 0, MM2S, IT[col], npu_bd_id(4 + r), chn(r), {0, 0, 0, (uint32_t)(this->E_HDR + X_OFF)}, {1, 1, 1, (uint32_t)(2 * UNIT - X_OFF)}, {0, 0, 0, 1});
            seq->npu_dma_memcpy_nd(1, 1, MM2S, IT[col], npu_bd_id(2 * r + 1), chn(r), {0, 0, 0, (uint32_t)(LB + L_RW + (size_t)c * 16 * UNIT)}, {1, 1, 1, (uint32_t)(16 * UNIT)}, {0, 0, 0, 1});
        }
        for (int c = 0; c < 16; c++) seq->npu_dma_wait(IT[c % 8], S2MM, chn(c / 8));
        seq->npu_dma_memcpy_nd(4, 0, S2MM, IT[0], npu_bd_id(10), it_channel_1, {0, 0, 0, (uint32_t)(this->TY / 4)}, {1, 1, 1, (uint32_t)YU}, {0, 0, 0, 1});
        seq->npu_dma_memcpy_nd(1, 1, MM2S, IT[0], npu_bd_id(6), it_channel_1, {0, 0, 0, (uint32_t)(LB + L_TH)}, {1, 1, 1, (uint32_t)UNIT}, {0, 0, 0, 1});
        seq->npu_dma_memcpy_nd(1, 0, MM2S, IT[0], npu_bd_id(7), it_channel_1, {0, 0, 0, (uint32_t)this->RY}, {1, 1, 1, (uint32_t)(2 * UNIT)}, {0, 0, 0, 1});
        for (int c = 0; c < 16; c++) {
            const int col = c % 8, r = c / 8;
            seq->npu_dma_memcpy_nd(4, 0, S2MM, IT[col], npu_bd_id(8 + r), chn(r), {0, 0, 0, (uint32_t)((this->EY + (size_t)c * 12 * YU * 4) / 4)}, {1, 1, 1, (uint32_t)(12 * YU)}, {0, 0, 0, 1});
            seq->npu_dma_memcpy_nd(1, 0, MM2S, IT[col], npu_bd_id(4 + r), chn(r), {0, 0, 0, (uint32_t)this->E_HDR}, {1, 1, 1, (uint32_t)UNIT}, {0, 0, 0, 1});
        }
        seq->npu_dma_wait(IT[0], S2MM, it_channel_1);
        for (int c = 0; c < 16; c++) seq->npu_dma_wait(IT[c % 8], S2MM, chn(c / 8));
    }
    if ((stages & ST_Z) && j == NAPP - 1) add_mix(cfg_Z(), this->MZ, this->ST, this->TY, this->BA[NAPP], this->XF, this->XZ);
    // what the host reads after the run (sync_all off): the loop passes' raw mix outputs and messages, the token's x
    if (I.pass > 0 && (stages & ST_A) && !this->dev_ctrl) {
        this->want.push_back({this->XC, 3 * CQ});
        if (I.il == CORE0) this->want.push_back({this->MSG, (size_t)cfg_M().nq * CQ});
    }
    if (I.pass > 0 && (stages & ST_F) && !this->dev_ctrl) this->want.push_back({this->XF, CQ});
    if ((stages & ST_Z) && j == NAPP - 1) this->want.push_back({this->XZ, (size_t)D * 2});
}

// run the steps added since the sequence was cleared; then the io host copy (all of it, or the ranges the host reads)
bool inkling2_fused::submit(int stages) {
    npu_sequence* seq = this->app().seq();
    const auto t0 = this->t_build;
    seq->cmds2seq();
    const auto t1 = std::chrono::steady_clock::now();
    xrt::run run = this->app().create_run(this->io, this->wt, this->gl[0], this->gl[1], this->cw);
    this->t_create += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count();
#if !defined(FLM_USE_HRX)
    if (this->lmh_state) {
        if (!wait_window(start_window(run, false))) return false;
    } else
#endif
    {
        run.start();
        const ert_cmd_state st = wait_run(run, 5000);
        if (st != ERT_CMD_STATE_COMPLETED) return false;
    }
    const auto t2 = std::chrono::steady_clock::now();
    if (this->sync_all) this->io.sync_from_device();
    else
        for (const auto& r : this->want) this->io.bo().sync(XCL_BO_SYNC_BO_FROM_DEVICE, r.second, r.first);
    this->want.clear();
    const auto t3 = std::chrono::steady_clock::now();
    this->t_seq += std::chrono::duration<double, std::milli>(t1 - t0).count();
    this->t_run += std::chrono::duration<double, std::milli>(t2 - t1).count();
    this->t_sync += std::chrono::duration<double, std::milli>(t3 - t2).count();
    this->n_run++;
    this->t_by[stages & 127] += std::chrono::duration<double, std::milli>(t2 - t1).count();
    this->n_by[stages & 127]++;
    return true;
}

#endif   // !FLM_USE_HRX
