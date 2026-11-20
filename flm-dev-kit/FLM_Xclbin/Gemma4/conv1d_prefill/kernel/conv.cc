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
constexpr int l_prod_lock = 6;

extern "C" {
    void conv(
        bf16 *__restrict o_ping,
        bf16 *__restrict o_pong,
        bf16 *__restrict x_ping,
        bf16 *__restrict x_pong,
        bf16 *__restrict x,
        bf16 *__restrict w,
        float * min_buffer,
        float * max_buffer,
        int * L_buffer,
        int * round_buffer,
        int row
    ){
        // Force into .data (file-backed) rather than .bss: under Peano (llvm-aie)
        // the aie-rt CDO loader does not reliably zero a .bss tail, leaving these
        // persistent ping-pong toggles as garbage on load. (chess is fine.)
        AIE_PERSIST_ZERO static bool is_x_ping = false;
        AIE_PERSIST_ZERO static bool is_o_ping = false;

        _lock_acquire(w_cons_lock);
        bf16* w_0 = w + row * 256;
        bf16* w_1 = w + row * 256 + 1024;
        bf16* w_2 = w + row * 256 + 1024 * 2;
        bf16* w_3 = w + row * 256 + 1024 * 3;
        bf16* w_4 = w + row * 256 + 1024 * 4;

        _lock_acquire(l_prod_lock);
        bf16 min_val = (bf16)min_buffer[0];
        bf16 max_val = (bf16)max_buffer[0];
        int round = 0;
        do{
            int i = 0;
            for (int j = 0; j < 4; j++){
                _lock_acquire(x_cons_lock);
                is_x_ping = !is_x_ping;
                bf16* x_in = is_x_ping ? x_ping : x_pong;
                bf16* x_write = x + j * 256 + 256;
                passThrough_aie<bf16, bf16, 256>(x_write, x_in);
                _lock_release(x_prod_lock);
            }
            do {
                int p = i % 5;
                _lock_acquire(x_cons_lock);
                _lock_acquire(o_prod_lock);
                is_x_ping = !is_x_ping;
                bf16* x_in = is_x_ping ? x_ping : x_pong;
                bf16* x_write = x + p * 256;
                passThrough_aie<bf16, bf16, 256>(x_write, x_in);
                bf16* x_0 = x + ((p + 1) % 5) * 256;
                bf16* x_1 = x + ((p + 2) % 5) * 256;
                bf16* x_2 = x + ((p + 3) % 5) * 256;
                bf16* x_3 = x + ((p + 4) % 5) * 256;
                bf16* x_4 = x + ((p + 0) % 5) * 256;
                is_o_ping = !is_o_ping;
                bf16* o_out = is_o_ping ? o_ping : o_pong;
                _conv_kernel(w_0, w_1, w_2, w_3, w_4, x_0, x_1, x_2, x_3, x_4, min_val, max_val, o_out);
                _lock_release(x_prod_lock);
                _lock_release(o_cons_lock);
                i++;
            } while(i < L_buffer[0]);
            round++;
        } while(round < round_buffer[0]);
        _lock_release(w_prod_lock);
    }
}