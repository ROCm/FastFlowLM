// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include "common.h"
#include "flm_rsqrt.h"

///@brief pseduo rms norm: y = x * w
///@param y: output
///@param x_buf: copy of x
///@param x: input
///@param w: weight
///@param overwrite_x_buf: if true, overwrite x_buf with x
// Peano spills the scalar magic fast-inv-sqrt values when this is inlined into
// the pressure-heavy rms core and wedges the core; noinline cures the deadlock.
#if !defined(__chess__)
__attribute__((noinline))
#endif
void pseduo_rms_norm(
    bf16 *restrict y,
    const bf16 *restrict x_buf,
    const bf16 *restrict x,
    const bf16 *restrict w,
    bool overwrite_x_buf = true
){
    bf16 *it_y = const_cast<bf16*>(y);
    bf16 *it_x_buf = const_cast<bf16*>(x_buf);
    bf16 *it_x = const_cast<bf16*>(x);
    bf16 *it_w = const_cast<bf16*>(w);
    const float epsilon = 1e-5;

    constexpr int vector_size = 16;
    constexpr float one_over_D = 1.0f / (float)D;

    aie::accum<accfloat, vector_size> sum_squares = aie::zeros<accfloat>();
    aie::vector<float, vector_size> mx_vec;

    const int F = D / vector_size;
    for (int i = 0; i < F; i++)
    chess_prepare_for_pipelining chess_loop_range(128, ) {
        auto x_vec = aie::load_v<vector_size>(it_x);
        sum_squares = aie::mac_square(sum_squares, x_vec);
        aie::store_v(it_x_buf, x_vec);
        it_x += vector_size;
        it_x_buf += vector_size;
    }

    it_x = const_cast<bf16*>(x);

    mx_vec = sum_squares.template to_vector<float>();
    float sum = aie::reduce_add(mx_vec);
    sum = sum * one_over_D;
    sum = sum + epsilon;
    const float divrms = flm_rsqrt(sum);

    for (int i = 0; i < F; i++)
    chess_prepare_for_pipelining chess_loop_range(128, ) {
        aie::vector<bf16, vector_size> x_vec = aie::load_v<vector_size>(it_x);
        aie::vector<bf16, vector_size> w_vec = aie::load_v<vector_size>(it_w);
        aie::vector<float, vector_size> wx_vec = aie::mul(x_vec, w_vec);
        aie::vector<bf16, vector_size> o_vec = aie::mul(wx_vec, divrms);
        aie::store_v(it_y, o_vec);
        it_x += vector_size;
        it_y += vector_size;
        it_w += vector_size;
    }
    event1();
}

///@brief pseduo residual add: y = x + x_buf, and x_buf = y
///@param y: output
///@param x: input
///@param x_buf: copy of x
void pseduo_residual_add(
    bf16 *restrict y,
    const bf16 *restrict x_buf,
    const bf16 *restrict x
){
    bf16 *it_x = const_cast<bf16*>(x);
    bf16 *it_y = const_cast<bf16*>(y);
    bf16 *it_x_buf = const_cast<bf16*>(x_buf);

    for (int i = 0; i < D / 16; i++) {
        aie::vector<bf16, 16> x_vec = aie::load_v<16>(it_x);
        aie::vector<bf16, 16> x_buf_vec = aie::load_v<16>(it_x_buf);
        auto out_vec = aie::add(x_vec, x_buf_vec);
        aie::store_v(it_y, out_vec);
        aie::store_v(it_x_buf, out_vec);
        it_x += 16;
        it_y += 16;
        it_x_buf += 16;
    }
}



extern "C" {

void rms_residual(
    bf16 *restrict y,
    const bf16 *restrict x,
    const bf16 *restrict w
){
    constexpr int w_prod_lock = 0;
    constexpr int w_cons_lock = 1;
    constexpr int y_prod_lock = 2;
    constexpr int y_cons_lock = 3;
    constexpr int x_prod_lock = 4;
    constexpr int x_cons_lock = 5;

    static bf16 x_buf[D];
    // Peano may reorder the pkt_id store past the triggering lock release, which
    // misroutes the packet-switched output -> consumer waits forever (deadlock).
    // volatile forces the store to keep its program-order position.
    volatile uint32_t* pkt_id_ptr = reinterpret_cast<volatile uint32_t*>(y + 14);

    _lock_acquire(x_cons_lock);
    _lock_acquire(w_cons_lock);
    _lock_acquire(y_prod_lock, 1);
    *pkt_id_ptr = 0;
    pseduo_rms_norm(y + 16, x_buf, x, w, true);
    _lock_release(x_prod_lock);
    _lock_release(w_prod_lock);
    _lock_release(y_cons_lock, qkv_repeats);


    _lock_acquire(x_cons_lock);
    _lock_acquire(w_cons_lock);
    _lock_acquire(y_prod_lock, qkv_repeats);
    *pkt_id_ptr = 0;
    pseduo_residual_add(y + 16, x_buf, x);
    pseduo_rms_norm(y + 16, x_buf, y + 16, w, false);
    _lock_release(x_prod_lock);
    _lock_release(w_prod_lock);
    _lock_release(y_cons_lock, up_gate_repeats);

    _lock_acquire(x_cons_lock);
    _lock_acquire(y_prod_lock, up_gate_repeats);
    *pkt_id_ptr = 1;
    pseduo_residual_add(y + 16, x_buf, x);
    _lock_release(x_prod_lock);
    _lock_release(y_cons_lock, 1);
}

}