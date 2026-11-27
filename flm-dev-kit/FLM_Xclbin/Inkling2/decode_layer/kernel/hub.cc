// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.
//
// Hub core of the Inkling-2-Mini decode overlay: drives the 16 weight streams itself. Per call it waits for the
// host's start lock, reads the weight buffer's address from shim 0 BD BASE_BD (once), then points BD SHIM_BD + ch
// of every shim at the stream it chooses and pushes it.
//
// Probe call (args): [0] bytes per channel task, [1] channel stride in bytes, [2] extra offset in bytes, [3] debug: 1 push only, 2 read + push, [4] trace on, [5] channels, [6] rounds (timing), [7] absolute base in [8] lo, [9] hi.
// MoE call (args[15] = 1): [11] [12] device address (lo, hi) of the layer's gate_up experts, [13] [14] of its down
// experts. Waits for the router logits from core (0, 3) (tk), routes (fp32 sigmoid, top 8 on p + bias,
// w = p / sum * 8), gives the ids and weights back, then points shim BDs GU_BD + ch / DN_BD + ch of column e at
// expert e's half ch (gate_up groups 12 ch .. 12 ch + 11, then the same down groups) and pushes them; the header
// unit before them is the sequence's.

#include "hub_ctrl.hpp"
#include "router_math.hpp"

static constexpr int START = 10, TR_PROD = 11, TR_CONS = 12, TK_PROD = 13, TK_CONS = 14, TK_DONE = 15;
static constexpr uint32_t GU_BD = 12, DN_BD = 14;
static constexpr uint32_t UNIT = 4736, GU_HALF = 12 * 9 * 3 * UNIT, DN_HALF = 12 * 5 * 3 * UNIT;   // bytes
static constexpr uint32_t SHIM_BD = 4, BASE_BD = 15;   // BASE_BD of shim 0: the host points it at the buffer (never started)
// control packet ids to shim j and of the read responses: must match CTRL_IDS / RESP_ID in iron/decode_aie.py
static constexpr uint32_t CTRL_ID[8] = {16, 17, 0, 8, 12, 14, 6, 7}, RESP_ID = 28;

// debug progress words on MM2S 0 (args[4] != 0, the 8-channel debug build only)
static void trace(volatile uint32_t* tr, const int32_t* args, uint32_t v) {
    if (!args[4]) return;
    _lock_acquire(TR_PROD);
    tr[0] = v;
    _lock_release(TR_CONS);
}

static volatile uint32_t* g_tr;
static void tp(uint32_t v) { _lock_acquire(TR_PROD); g_tr[0] = v; _lock_release(TR_CONS); }

static void moe(uint32_t* out, const int32_t* args, float* tk) {
    _lock_acquire(TK_CONS);
    __builtin_aie2p_sched_barrier();
    alignas(aie::vector_decl_align) int32_t idx[128];   // partial_select_largest writes all 128
    alignas(aie::vector_decl_align) float w[8];
    router_math::route_sigmoid_topk<128, 8>(w, idx, tk, tk + 128, 8.0f);
    for (int k = 0; k < 8; k++) { ((int32_t*)tk)[256 + k] = idx[k]; tk[264 + k] = w[k]; }
    __builtin_aie2p_sched_barrier();
    _lock_release(TK_DONE);
    const uint32_t gu_lo = (uint32_t)args[11], gu_hi = (uint32_t)args[12], dn_lo = (uint32_t)args[13], dn_hi = (uint32_t)args[14];
    for (uint32_t c = 0; c < 16; c++) {
        const uint32_t e = c & 7, ch = c >> 3;
        hub_ctrl::shim_bd(out, CTRL_ID[e], GU_BD + ch, gu_lo, gu_hi, (uint32_t)idx[e] * (2 * GU_HALF) + ch * GU_HALF, GU_HALF / 4);
        hub_ctrl::shim_push(out, CTRL_ID[e], ch, GU_BD + ch);
    }
    for (uint32_t c = 0; c < 16; c++) {
        const uint32_t e = c & 7, ch = c >> 3;
        hub_ctrl::shim_bd(out, CTRL_ID[e], DN_BD + ch, dn_lo, dn_hi, (uint32_t)idx[e] * (2 * DN_HALF) + ch * DN_HALF, DN_HALF / 4);
        hub_ctrl::shim_push(out, CTRL_ID[e], ch, DN_BD + ch);
    }
}

extern "C" void hub(uint32_t* out, uint32_t* in, int32_t* args, uint32_t* tr, float* tk) {
    _lock_acquire(START);
    if (args[15] == 1) { moe(out, args, tk); return; }
    trace(tr, args, 1);
    static uint32_t base_lo, base_hi;
    static bool known = false;
    const int dbg = args[3];
    g_tr = tr;
    if (args[7]) {   // probe: the base is a device address the host passes (any BO, not a run argument)
        base_lo = (uint32_t)args[8];
        base_hi = (uint32_t)args[9];
    } else if (!known && dbg != 1) {
        base_lo = hub_ctrl::shim_read(out, in, CTRL_ID[0], RESP_ID, hub_ctrl::bd_addr(BASE_BD) + 4, args[4] ? tp : nullptr);
        trace(tr, args, 2);
        base_hi = hub_ctrl::shim_read(out, in, CTRL_ID[0], RESP_ID, hub_ctrl::bd_addr(BASE_BD) + 8);
        known = true;
        trace(tr, args, 3);
        trace(tr, args, base_lo);
        trace(tr, args, base_hi);
    }
    const uint32_t bytes = (uint32_t)args[0], stride = (uint32_t)args[1], extra = (uint32_t)args[2];
    const uint32_t C = (uint32_t)args[5], R = args[6] > 0 ? (uint32_t)args[6] : 1;
    for (uint32_t r = 0; r < R; r++)
    for (uint32_t c = 0; c < C; c++) {
        const uint32_t col = c % 8, ch = c / 8;
        trace(tr, args, 0x100 + c);
        if (dbg == 0) hub_ctrl::shim_bd(out, CTRL_ID[col], SHIM_BD + ch, base_lo, base_hi, c * stride + extra, bytes / 4);
        if (c == 0) trace(tr, args, 4);
        hub_ctrl::shim_push(out, CTRL_ID[col], ch, SHIM_BD + ch);
    }
    trace(tr, args, 0xD0E);
}
