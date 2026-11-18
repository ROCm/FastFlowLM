// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#include <aie_api/aie.hpp>
#include "aie_kernel_utils.h"
#include <math.h>
#include "conv_kernel.h"

constexpr int x_prod_lock = 0;
constexpr int x_cons_lock = 1;
constexpr int o_prod_lock = 4;
constexpr int o_cons_lock = 5;

constexpr int D = 192;          // channels per core: one q/k head (128) + 64 v
constexpr int KS = 4;           // conv taps

extern "C" {
    // One run: every row arrives as one D-wide message on x_ping / x_pong.
    // The first KS rows are this core's conv weights (tap 0..3), then KS - 1
    // conv-state rows, then L token rows; each token row produces one output row.
    void conv(
        bf16 *__restrict o_ping,
        bf16 *__restrict o_pong,
        bf16 *__restrict x_ping,
        bf16 *__restrict x_pong,
        bf16 *__restrict x,          // ring of the last KS rows
        bf16 *__restrict w,          // KS weight rows, copied from the stream
        int * L_buffer
    ){
        // bf16 roundings are round-to-nearest-even, like the host reference
        aie::set_rounding(aie::rounding_mode::conv_even);

        static AIE_PING_FLAG is_x_ping = false;
        static AIE_PING_FLAG is_o_ping = false;

        // the weight rows
        for (int j = 0; j < KS; j++){
            _lock_acquire(x_cons_lock);
            is_x_ping = !is_x_ping;
            bf16* x_in = is_x_ping ? x_ping : x_pong;
            passThrough_aie<bf16, bf16, D>(w + j * D, x_in);
            _lock_release(x_prod_lock);
        }
        // the conv-state rows go to ring slots 1..3; slot 0 gets the first token
        for (int j = 0; j < KS - 1; j++){
            _lock_acquire(x_cons_lock);
            is_x_ping = !is_x_ping;
            bf16* x_in = is_x_ping ? x_ping : x_pong;
            passThrough_aie<bf16, bf16, D>(x + j * D + D, x_in);
            _lock_release(x_prod_lock);
        }
        zero_vectorized<bf16, D>(x);
        const bf16* w_0 = w;
        const bf16* w_1 = w + D;
        const bf16* w_2 = w + D * 2;
        const bf16* w_3 = w + D * 3;

        int i = 0;
        int L;
        do {
            int p = i & 0x3;
            _lock_acquire(x_cons_lock);
            _lock_acquire(o_prod_lock);
            is_x_ping = !is_x_ping;
            bf16* x_in = is_x_ping ? x_ping : x_pong;
            bf16* x_write = x + p * D;
            passThrough_aie<bf16, bf16, D>(x_write, x_in);
            bf16* x_0 = x + ((p + 1) & 0x3) * D;     // oldest row, pairs with w_0
            bf16* x_1 = x + ((p + 2) & 0x3) * D;
            bf16* x_2 = x + ((p + 3) & 0x3) * D;
            bf16* x_3 = x + ((p + 0) & 0x3) * D;     // the current token, pairs with w_3

            is_o_ping = !is_o_ping;
            bf16* o_out = is_o_ping ? o_ping : o_pong;
#ifndef DMA_ONLY
            conv_silu_norm_192(x_0, x_1, x_2, x_3, w_0, w_1, w_2, w_3, o_out);
#else
            // DMA_ONLY: no math. Every lock and buffer swap still happens,
            // so the run time is the data-movement floor.
            (void)x_0; (void)x_1; (void)x_2; (void)x_3; (void)w_0; (void)w_1; (void)w_2; (void)w_3;
#endif
            _lock_release(x_prod_lock);
            _lock_release(o_cons_lock);
            i++;
            L = L_buffer[0];
        } while(i < L);
    }
}
