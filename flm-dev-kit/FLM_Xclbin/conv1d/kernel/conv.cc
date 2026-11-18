// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include <aie_api/aie.hpp>
#include "aie_kernel_utils.h"
#include <math.h>
#include "conv_kernel.h"

constexpr int x_prod_lock = 0;
constexpr int x_cons_lock = 1;
constexpr int w_prod_lock = 2;
constexpr int w_cons_lock = 3;
constexpr int o_prod_lock = 4;
constexpr int o_cons_lock = 5;

extern "C" {
    void conv(
        bf16 *__restrict o_ping,
        bf16 *__restrict o_pong,
        bf16 *__restrict x_ping,
        bf16 *__restrict x_pong,
        bf16 *__restrict x,
        bf16 *__restrict w,
        int * L_buffer,
        int row
    ){
        AIE_PERSIST_ZERO static bool is_x_ping = false;
        AIE_PERSIST_ZERO static bool is_o_ping = false;

        int i = 0;
        int L;
        for (int j = 0; j < 3; j++){
            _lock_acquire(x_cons_lock);
            is_x_ping = !is_x_ping;
            bf16* x_in = is_x_ping ? x_ping : x_pong;
            bf16* x_write = x + j * 128 + 128;
            passThrough_aie<bf16, bf16, 128>(x_write, x_in);
            _lock_release(x_prod_lock);
        }
        zero_vectorized<bf16, 128>(x);
        _lock_acquire(w_cons_lock);
        bf16* w_0 = w + row * 128;
        bf16* w_1 = w + row * 128 + 512;
        bf16* w_2 = w + row * 128 + 512 * 2;
        bf16* w_3 = w + row * 128 + 512 * 3;
        do {
            int p = i & 0x3;
            _lock_acquire(x_cons_lock);
            _lock_acquire(o_prod_lock);
            is_x_ping = !is_x_ping;
            bf16* x_in = is_x_ping ? x_ping : x_pong;
            bf16* x_write = x + p * 128;
            passThrough_aie<bf16, bf16, 128>(x_write, x_in);
            bf16* x_0 = x + ((p + 1) & 0x3) * 128;
            bf16* x_1 = x + ((p + 2) & 0x3) * 128;
            bf16* x_2 = x + ((p + 3) & 0x3) * 128;
            bf16* x_3 = x + ((p + 0) & 0x3) * 128;

            is_o_ping = !is_o_ping;
            bf16* o_out = is_o_ping ? o_ping : o_pong;
            _conv_kernel(w_0, w_1, w_2, w_3, x_0, x_1, x_2, x_3, o_out);
            _lock_release(x_prod_lock);
            _lock_release(o_cons_lock);
            i++;
            L = L_buffer[0];
        } while(i < L);
        _lock_release(w_prod_lock);
    }
}