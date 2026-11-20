// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <aie_api/aie.hpp>
#include "common.h"


template <typename T, typename T_out, typename T_acc, unsigned m, unsigned k, unsigned r>
void mv_core(T_out *__restrict y, T *__restrict w, T *__restrict x) {
    event0();
    T *__restrict w_ptr = w;
    T *__restrict x_ptr = x;
    T_out *__restrict y_ptr = y;

    for (int row = 0; row < m; row += r)
    {
        aie::accum<T_acc, r> y_acc_in;
        y_acc_in.from_vector(aie::load_v<r>(y_ptr));

        w_ptr = w + 2 * row;
        x_ptr = x;
        for (int col = 0; col < k; col += 8) 
        {
            aie::vector<T, 8> x_vec = aie::load_v<8>(x_ptr);

            const aie::vector<T, 2 * r> w_vec_0 = aie::load_v<2 * r>(w_ptr);
            const aie::vector<T, 2 * r> w_vec_1 = aie::load_v<2 * r>(w_ptr + 2 * m);
            const aie::vector<T, 2 * r> w_vec_2 = aie::load_v<2 * r>(w_ptr + 4 * m);
            const aie::vector<T, 2 * r> w_vec_3 = aie::load_v<2 * r>(w_ptr + 6 * m);

            const aie::vector<T, r> w_vec_00 = aie::filter_even(w_vec_0, 1);
            const aie::vector<T, r> w_vec_01 = aie::filter_odd(w_vec_0, 1);
            const aie::vector<T, r> w_vec_10 = aie::filter_even(w_vec_1, 1);
            const aie::vector<T, r> w_vec_11 = aie::filter_odd(w_vec_1, 1);
            const aie::vector<T, r> w_vec_20 = aie::filter_even(w_vec_2, 1);
            const aie::vector<T, r> w_vec_21 = aie::filter_odd(w_vec_2, 1);
            const aie::vector<T, r> w_vec_30 = aie::filter_even(w_vec_3, 1);
            const aie::vector<T, r> w_vec_31 = aie::filter_odd(w_vec_3, 1);

            // y_acc_in = aie::accumulate<r>(y_acc_in, x_vec, 0, 
            //     w_vec_00, w_vec_01, 
            //     w_vec_10, w_vec_11, 
            //     w_vec_20, w_vec_21, 
            //     w_vec_30, w_vec_31 
            // );
            y_acc_in = aie::mac(y_acc_in, x_vec.get(0), w_vec_00);
            y_acc_in = aie::mac(y_acc_in, x_vec.get(1), w_vec_01);
            y_acc_in = aie::mac(y_acc_in, x_vec.get(2), w_vec_10);
            y_acc_in = aie::mac(y_acc_in, x_vec.get(3), w_vec_11);
            y_acc_in = aie::mac(y_acc_in, x_vec.get(4), w_vec_20);
            y_acc_in = aie::mac(y_acc_in, x_vec.get(5), w_vec_21);
            y_acc_in = aie::mac(y_acc_in, x_vec.get(6), w_vec_30);
            y_acc_in = aie::mac(y_acc_in, x_vec.get(7), w_vec_31);

            w_ptr += 8 * m; // Move to next 8 columns of W.
            x_ptr += 8;     // Move to next s (==8) rows of X.
        }
        aie::store_v(y_ptr, y_acc_in.template to_vector<T_out>());
        y_ptr += r;
    }
    event1();
}

constexpr int w_prod_lock = 0;
constexpr int w_cons_lock = 1;
constexpr int x_prod_lock = 2;
constexpr int x_cons_lock = 3;
constexpr int y_prod_lock = 4;
constexpr int y_cons_lock = 5;
constexpr int rtp_lock = 6;


AIE_PERSIST_ZERO static bool is_wx_ping = false;
AIE_PERSIST_ZERO static bool is_y_ping = false;
    

extern "C" {

void mv(
    bf16 *restrict y_ping,
    bf16 *restrict y_pong,
    bf16 *restrict w_ping, // matrix
    bf16 *restrict w_pong,
    bf16 *restrict x_ping,
    bf16 *restrict x_pong,
    float * y_acc,
    int * M_div_m, // m div 16
    int * K_div_k
){

    _lock_acquire(rtp_lock);
    
    int i = 0;
    do {
        zero_vectorized<float, m>(y_acc);
        int j = 0;
        do {
            _lock_acquire(x_cons_lock);
            _lock_acquire(w_cons_lock);
            is_wx_ping = !is_wx_ping;
            bf16 *x = is_wx_ping ? x_ping : x_pong;
            bf16 *w = is_wx_ping ? w_ping : w_pong;
            mv_core<bf16, float, accfloat, m, k, r>(y_acc, w, x);
            _lock_release(x_prod_lock);
            _lock_release(w_prod_lock);
            j++;
        } while (j < K_div_k[0]);
        _lock_acquire(y_prod_lock);
        is_y_ping = !is_y_ping;
        bf16 *y = is_y_ping ? y_ping : y_pong;
        passThrough_aie<float, bf16, m>(y_acc, y);
        _lock_release(y_cons_lock);
        i++;
    } while (i < M_div_m[0]);
}

} // extern "C"