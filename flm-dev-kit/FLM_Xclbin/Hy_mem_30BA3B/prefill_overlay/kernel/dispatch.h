// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

// The overlay's dispatcher, shared by fused_e.cc (rows 2, 4) and fused_o.cc (rows 3, 5).
//
// The host configures the array for one op (see host/overlay_host.hpp), writes
// this core's RTP block, then sets OP_LOCK to 1. The core then runs `calls`
// calls of that op -- exactly what the op's standalone core loop would have run
// in that run -- and comes back here to wait for the next one.
//
// Buffers: every op lays its buffers into the same arena (layout.py); `other` is
// the neighbour's arena of the dequant_mm pair (north for even rows, south for odd).
#pragma once

static inline void ovl_lock_acquire(int lock_id){
    acquire_greater_equal(lock_id + 48, 1);   // the core sees its own locks at id + 48
}

template <typename T>
static inline T* at_byte(bfloat16* arena, int offset){
    return (T*)((char*)arena + offset);
}

extern "C" {
    void fused(bfloat16* arena, bfloat16* other, int32* rtp, int row, int col){
        volatile int32* rtp_v = rtp;
        ovl_lock_acquire(OP_LOCK);
        const int op = rtp_v[RTP_OP];
        const int calls = rtp_v[RTP_CALLS];
        // After a switch the op's DMA chains start again at their ping BD,
        // so the op's own ping / pong flags must too.
        const bool switched = rtp_v[RTP_SWITCHED] != 0;
        int* rtp_i = (int*)rtp;

        if (op == OP_DQ) {
            if (switched) {
                dq::is_x_ping = false;
                dq::is_qw_ping = false;
                dq::is_y_ping = false;
            }
            for (int c = 0; c < calls; c++) {
#if DQ_EVEN
                dq::dequant_mm_e(
                    at_byte<bfloat16>(arena, CT_DQ_E_Y0), at_byte<bfloat16>(arena, CT_DQ_E_Y1),
                    at_byte<bfloat16>(arena, CT_DQ_E_X0), at_byte<bfloat16>(arena, CT_DQ_E_X1),
                    at_byte<uint8>(arena, CT_DQ_E_QW0), at_byte<uint8>(arena, CT_DQ_E_QW1),
                    at_byte<bfloat16>(other, CT_DQ_O_W),          // w tile lives in the odd core
                    at_byte<float>(arena, CT_DQ_E_Y_ACC),
                    rtp_i + RTP_DQ_K_DIV_K, rtp_i + RTP_DQ_ACT);
#else
                dq::dequant_mm_o(
                    at_byte<bfloat16>(arena, CT_DQ_O_Y0), at_byte<bfloat16>(arena, CT_DQ_O_Y1),
                    at_byte<bfloat16>(arena, CT_DQ_O_X0), at_byte<bfloat16>(arena, CT_DQ_O_X1),
                    at_byte<uint8>(other, CT_DQ_E_QW0), at_byte<uint8>(other, CT_DQ_E_QW1),   // q4_k blocks live in the even core
                    at_byte<bfloat16>(arena, CT_DQ_O_W),
                    at_byte<float>(arena, CT_DQ_O_Y_ACC),
                    rtp_i + RTP_DQ_K_DIV_K, rtp_i + RTP_DQ_ACT);
#endif
            }
        } else if (op == OP_ATTN) {
            if (switched) {
                at::is_in_ping = false;
                at::is_o_ping = false;
            }
            for (int c = 0; c < calls; c++) {
                at::attn(
                    at_byte<bfloat16>(arena, CT_AT_O0), at_byte<bfloat16>(arena, CT_AT_O1),
                    at_byte<bfloat16>(arena, CT_AT_Q),
                    at_byte<bfloat16>(arena, CT_AT_IN0), at_byte<bfloat16>(arena, CT_AT_IN1),
                    at_byte<bfloat16>(arena, CT_AT_S), at_byte<bfloat16>(arena, CT_AT_M),
                    at_byte<float>(arena, CT_AT_Y),
                    rtp_i + RTP_AT_L_BEGIN, rtp_i + RTP_AT_L_CHUNKS, rtp_i + RTP_AT_WINDOW,
                    row, col & 1);
            }
        }
    }
}
