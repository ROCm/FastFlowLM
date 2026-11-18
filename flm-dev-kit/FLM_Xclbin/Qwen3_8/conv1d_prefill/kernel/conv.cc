// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include <aie_api/aie.hpp>
#include "aie_kernel_utils.h"
#include <math.h>
#include "conv_kernel.h"

// Lock ids, mirroring iron/conv_tile.py.
constexpr int x_prod_lock = 0;
constexpr int x_cons_lock = 1;
constexpr int w_prod_lock = 2;
constexpr int w_cons_lock = 3;
constexpr int o_prod_lock = 4;
constexpr int o_cons_lock = 5;
// Released by the host's rtp_write once L_buffer holds this pass's values.
// Without it the core could read L_buffer[0] from the previous pass.
constexpr int rtp_sync_lock = 6;

// What this core's channels are, written to L_buffer[1] by the host.  The
// core cannot work this out from its own (row, col): which slice of
// q | k | v a column carries changes between the two passes it takes to
// cover CONV_DIM = 10240 channels with 32 cores x 256.
constexpr int NORM_NONE  = 0;  // v: conv + silu only
constexpr int NORM_K     = 1;  // k: l2norm per 128-head
constexpr int NORM_Q     = 2;  // q: l2norm per 128-head, times 1/sqrt(head_dim)

extern "C" {
    void conv(
        bf16 *__restrict o_ping,
        bf16 *__restrict o_pong,
        bf16 *__restrict x_ping,
        bf16 *__restrict x_pong,
        bf16 *__restrict x,
        bf16 *__restrict w,
        int * L_buffer,
        int row,
        int col
    ){
        // Peano's CDO loader does not zero .bss; keep ping/pong selectors in .data.
#if !defined(__chess__)
        __attribute__((section(".data")))
#endif
        static bool is_x_ping = false;
#if !defined(__chess__)
        __attribute__((section(".data")))
#endif
        static bool is_o_ping = false;

        // Wait for this pass's RTPs to land before reading them.
        _lock_acquire(rtp_sync_lock);
        const int L         = L_buffer[0];
        const int norm_mode = L_buffer[1];

        // Prime the 4-tap window with the 3 history rows DDR puts in front of
        // the real tokens, so output token t sees input rows t .. t+3.  Slot 0
        // is left for the first real token, written at the top of the loop.
        for (int j = 0; j < 3; j++){
            _lock_acquire(x_cons_lock);
            is_x_ping = !is_x_ping;
            bf16* x_in = is_x_ping ? x_ping : x_pong;
            bf16* x_write = x + (j + 1) * CORE_CHANNELS;
            passThrough_aie<bf16, bf16, CORE_CHANNELS>(x_write, x_in);
            _lock_release(x_prod_lock);
        }
        _lock_acquire(w_cons_lock);
        // This core owns CORE_CHANNELS of the column's 4 * CORE_CHANNELS, one
        // row each; taps are CONV_K rows of the column's full channel span.
        bf16* w_0 = w + row * CORE_CHANNELS;
        bf16* w_1 = w + row * CORE_CHANNELS + 4 * CORE_CHANNELS;
        bf16* w_2 = w + row * CORE_CHANNELS + 4 * CORE_CHANNELS * 2;
        bf16* w_3 = w + row * CORE_CHANNELS + 4 * CORE_CHANNELS * 3;

        for (AIE_LOCK_LOOP_CTR i = 0; i < L; i++) {
            int p = i & 0x3;
            _lock_acquire(x_cons_lock);
            _lock_acquire(o_prod_lock);
            is_x_ping = !is_x_ping;
            bf16* x_in = is_x_ping ? x_ping : x_pong;
            bf16* x_write = x + p * CORE_CHANNELS;
            passThrough_aie<bf16, bf16, CORE_CHANNELS>(x_write, x_in);
            // Oldest tap first: slot p holds the newest token, so the window
            // reads p+1, p+2, p+3, p modulo 4.
            bf16* x_0 = x + ((p + 1) & 0x3) * CORE_CHANNELS;
            bf16* x_1 = x + ((p + 2) & 0x3) * CORE_CHANNELS;
            bf16* x_2 = x + ((p + 3) & 0x3) * CORE_CHANNELS;
            bf16* x_3 = x + ((p + 0) & 0x3) * CORE_CHANNELS;

            is_o_ping = !is_o_ping;
            bf16* o_out = is_o_ping ? o_ping : o_pong;
            _conv_kernel(w_0, w_1, w_2, w_3, x_0, x_1, x_2, x_3, o_out);
            _lock_release(x_prod_lock);

            // CORE_CHANNELS is exactly two heads, and a core's slice never
            // straddles the q/k or k/v boundary (DQ and DQ+DK are both
            // multiples of CORE_CHANNELS), so one mode covers both heads.
            if (norm_mode == NORM_Q){
                l2_norm(o_out, LINEAR_ATTN_SCALE);
                l2_norm(o_out + HEAD_DIM, LINEAR_ATTN_SCALE);
            } else if (norm_mode == NORM_K){
                l2_norm(o_out, 1.0f);
                l2_norm(o_out + HEAD_DIM, 1.0f);
            }
            _lock_release(o_cons_lock);
        }
        _lock_release(w_prod_lock);
    }
}
