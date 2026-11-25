// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.
//
// Hub core of the Inkling-2-Mini decode overlay: drives the 16 weight streams itself. Per call it waits for the
// host's start lock, reads the weight buffer's address from shim 0 BD BASE_BD (once), then points BD SHIM_BD + ch
// of every shim at the stream it chooses and pushes it.
//
// Probe call (args): [0] bytes per channel task, [1] channel stride in bytes, [2] extra offset in bytes, [3] debug: 1 push only, 2 read + push, [4] trace on, [5] channels, [6] rounds (timing), [7] absolute base in [8] lo, [9] hi.

#include "hub_ctrl.hpp"

static constexpr int START = 10, TR_PROD = 11, TR_CONS = 12;
static constexpr uint32_t SHIM_BD = 4, BASE_BD = 15;   // BASE_BD of shim 0: the host points it at the buffer (never started)
// control packet ids to shim j and of the read responses: must match CTRL_IDS / RESP_ID in iron/decode_aie.py
static constexpr uint32_t CTRL_ID[8] = {5, 8, 17, 31, 22, 30, 21, 23}, RESP_ID = 9;

// debug progress words on MM2S 0 (args[4] != 0, the 8-channel debug build only)
static void trace(volatile uint32_t* tr, const int32_t* args, uint32_t v) {
    if (!args[4]) return;
    _lock_acquire(TR_PROD);
    tr[0] = v;
    _lock_release(TR_CONS);
}

static volatile uint32_t* g_tr;
static void tp(uint32_t v) { _lock_acquire(TR_PROD); g_tr[0] = v; _lock_release(TR_CONS); }

extern "C" void hub(uint32_t* out, uint32_t* in, int32_t* args, uint32_t* tr) {
    _lock_acquire(START);
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
