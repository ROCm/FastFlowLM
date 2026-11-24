// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <aie_api/aie.hpp>
#include "common.h"

// Peano runs out of registers passing six 64-lane vectors by value; inline it there.
#if !defined(__chess__)
__attribute__((always_inline)) inline
#endif
void conv1d_core(
    bf16 *__restrict y, 
    aie::vector<bf16, 64> w0, 
    aie::vector<bf16, 64> w1, 
    aie::vector<bf16, 64> w2, 
    aie::vector<bf16, 64> x0, 
    aie::vector<bf16, 64> x1, 
    aie::vector<bf16, 64> x2
) {
    event0();
    bf16 *__restrict y_ptr = y;

    aie::accum<accfloat, d> y_acc_0;

    y_acc_0 = aie::mul(x0, w2);
    y_acc_0 = aie::mac(y_acc_0, x1, w1);
    y_acc_0 = aie::mac(y_acc_0, x2, w0);

    aie::store_v(y_ptr, y_acc_0.template to_vector<bf16>());
    event1();
}

constexpr int w_prod_lock = 0;
constexpr int w_cons_lock = 1;
constexpr int x_prod_lock = 2;
constexpr int x_cons_lock = 3;
constexpr int y_prod_lock = 4;
constexpr int y_cons_lock = 5;

// Under Peano these flags live in .data, since its CDO loader does not zero
// .bss. They are volatile there too: Peano does not treat a lock acquire as a
// memory barrier, so if it can prove two picks give the same buffer it reuses
// the earlier load, even though the DMA has refilled the buffer in between.
#if defined(__chess__)
#define AIE_PING_FLAG static bool
#else
#define AIE_PING_FLAG __attribute__((section(".data"))) static volatile bool
#endif
AIE_PING_FLAG is_x_ping = false;
AIE_PING_FLAG is_y_ping = false;

extern "C" {

void conv1d(
    bf16 *restrict y_ping,
    bf16 *restrict y_pong,
    bf16 *restrict x_ping,
    bf16 *restrict x_pong,
    bf16 *restrict w,
    int * L,
    int row
){

    aie::vector<bf16, 64> w0 = aie::zeros<bf16, 64>();
    aie::vector<bf16, 64> w1 = aie::zeros<bf16, 64>();
    aie::vector<bf16, 64> w2 = aie::zeros<bf16, 64>();
    aie::vector<bf16, 64> x0 = aie::zeros<bf16, 64>();
    aie::vector<bf16, 64> x1 = aie::zeros<bf16, 64>();
    aie::vector<bf16, 64> x2 = aie::zeros<bf16, 64>();


    //NOTE: TODO:
    // THIS implemenation assume kernel=3 with stride=1 with padding done externally already!!!

    int L_OUT_SIZE = *(L) - 3 + 1;


    // load weights
    _lock_acquire(w_cons_lock);
    bf16 *w_ptr = w + row * d;
    w0 = aie::load_v<64>(w_ptr);
    w1 = aie::load_v<64>(w_ptr + d * 4);
    w2 = aie::load_v<64>(w_ptr + d * 4 * 2);
    _lock_release(w_prod_lock);

    // i == 0, load x
    _lock_acquire(x_cons_lock);
    is_x_ping = !is_x_ping;
    bf16 *x = is_x_ping ? x_ping : x_pong;
    x2 = aie::load_v<64>(x);
    _lock_release(x_prod_lock);
    
    _lock_acquire(x_cons_lock);
    is_x_ping = !is_x_ping;
    x = is_x_ping ? x_ping : x_pong;
    x1 = aie::load_v<64>(x);
    _lock_release(x_prod_lock);

    _lock_acquire(x_cons_lock);
    is_x_ping = !is_x_ping;
    x = is_x_ping ? x_ping : x_pong;
    x0 = aie::load_v<64>(x);
    _lock_release(x_prod_lock);

    _lock_acquire(y_prod_lock);
    is_y_ping = !is_y_ping;
    bf16 *y = is_y_ping ? y_ping : y_pong;
    conv1d_core(y, w0, w1, w2, x0, x1, x2);
    _lock_release(y_cons_lock);

    int i = 1;
    do {
        x2 = x1;
        x1 = x0;
        _lock_acquire(x_cons_lock);
        _lock_acquire(y_prod_lock);
        is_y_ping = !is_y_ping;
        is_x_ping = !is_x_ping;
        bf16 *x = is_x_ping ? x_ping : x_pong;
        bf16 *y = is_y_ping ? y_ping : y_pong;
        x0 = aie::load_v<64>(x);
        conv1d_core(y, w0, w1, w2, x0, x1, x2);
        _lock_release(y_cons_lock);
        _lock_release(x_prod_lock);
        i++;
    } while (i <L_OUT_SIZE);
}

}