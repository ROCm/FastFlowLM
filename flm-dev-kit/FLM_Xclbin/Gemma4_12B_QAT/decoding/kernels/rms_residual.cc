// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include "aie_array_layout.h"
#include "aie_kernel_utils.h"
#include "typedef.hpp"
#include "flm_rsqrt.h"

///@brief  rms norm: y = nrom(x) * w;
///@brief                    x_buf = x
///@param y: output
///@param x_buf: copy of x
///@param x: input
///@param w: weight
void rms_norm(
    bf16 *restrict y,
    const bf16 *restrict x,
    const bf16 *restrict w
){
    const float epsilon = 1e-6;

    constexpr int vector_size = 16;
    constexpr float one_over_D = 1.0f / (float)MODEL_DIM;

    // Two interleaved streams: a single accumulator serialises back-to-back
    // mac_square on its own dependency, and a 16-wide vector is too narrow to
    // cover the latency. Two of them halve the recurrence.
    static_assert(MODEL_DIM % (2 * vector_size) == 0,
                  "MODEL_DIM must cover a whole number of interleaved pairs");
    const int F = MODEL_DIM / (2 * vector_size);

    bf16 *it_y_0 = const_cast<bf16*>(y);
    bf16 *it_y_1 = const_cast<bf16*>(y) + vector_size;
    bf16 *it_x_0 = const_cast<bf16*>(x);
    bf16 *it_x_1 = const_cast<bf16*>(x) + vector_size;
    bf16 *it_w_0 = const_cast<bf16*>(w);
    bf16 *it_w_1 = const_cast<bf16*>(w) + vector_size;

    aie::accum<accfloat, vector_size> sum_squares_0 = aie::zeros<accfloat>();
    aie::accum<accfloat, vector_size> sum_squares_1 = aie::zeros<accfloat>();
    aie::vector<float, vector_size> mx_vec;

    for (int i = 0; i < F; i++){
        auto x_vec_0 = aie::load_v<vector_size>(it_x_0);
        auto x_vec_1 = aie::load_v<vector_size>(it_x_1);
        sum_squares_0 = aie::mac_square(sum_squares_0, x_vec_0);
        sum_squares_1 = aie::mac_square(sum_squares_1, x_vec_1);
        it_x_0 += 2 * vector_size;
        it_x_1 += 2 * vector_size;
    }

    it_x_0 = const_cast<bf16*>(x);
    it_x_1 = const_cast<bf16*>(x) + vector_size;

    // the two partials meet once, outside the loop
    mx_vec = aie::add(sum_squares_0.template to_vector<float>(),
                      sum_squares_1.template to_vector<float>());
    float sum = aie::reduce_add(mx_vec);
#if defined(__chess__)
    sum = sum * one_over_D;
    sum = sum + epsilon;
#else
    // Same math as the chess branch, rounded like chess (see AIE_FADD).
    sum = AIE_FMUL(sum, one_over_D);
    sum = AIE_FADD(sum, epsilon);
#endif
    const float divrms = flm_rsqrt(sum);

    for (int i = 0; i < F; i++){
        aie::vector<bf16, vector_size> x_vec_0 = aie::load_v<vector_size>(it_x_0);
        aie::vector<bf16, vector_size> x_vec_1 = aie::load_v<vector_size>(it_x_1);
        aie::vector<bf16, vector_size> w_vec_0 = aie::load_v<vector_size>(it_w_0);
        aie::vector<bf16, vector_size> w_vec_1 = aie::load_v<vector_size>(it_w_1);
        aie::vector<float, vector_size> wx_vec_0 = aie::mul(x_vec_0, w_vec_0);
        aie::vector<float, vector_size> wx_vec_1 = aie::mul(x_vec_1, w_vec_1);
        aie::vector<bf16, vector_size> o_vec_0 = aie::mul(wx_vec_0, divrms);
        aie::vector<bf16, vector_size> o_vec_1 = aie::mul(wx_vec_1, divrms);
        aie::store_v(it_y_0, o_vec_0);
        aie::store_v(it_y_1, o_vec_1);
        it_x_0 += 2 * vector_size;
        it_x_1 += 2 * vector_size;
        it_y_0 += 2 * vector_size;
        it_y_1 += 2 * vector_size;
        it_w_0 += 2 * vector_size;
        it_w_1 += 2 * vector_size;
    }
}

///@brief residual add: y = x + x_buf, and x_buf = y
///@note  y and x may alias (in-place accumulate), so neither is marked restrict.
///@param y: output
///@param x: input
///@param x_buf: copy of x
void residual_add(
    bf16 *y,
    const bf16 *restrict x_buf,
    const bf16 *x
){
    constexpr int vector_size = 16;
    static_assert(MODEL_DIM % (2 * vector_size) == 0,
                  "MODEL_DIM must cover a whole number of interleaved pairs");

    // Two interleaved streams, same reason as rms_norm above. y may alias x,
    // but the two streams never touch the same offset, so an in-place
    // accumulate stays correct.
    bf16 *it_x_0 = const_cast<bf16*>(x);
    bf16 *it_x_1 = const_cast<bf16*>(x) + vector_size;
    bf16 *it_y_0 = const_cast<bf16*>(y);
    bf16 *it_y_1 = const_cast<bf16*>(y) + vector_size;
    bf16 *it_x_buf_0 = const_cast<bf16*>(x_buf);
    bf16 *it_x_buf_1 = const_cast<bf16*>(x_buf) + vector_size;

    for (int i = 0; i < MODEL_DIM / (2 * vector_size); i++) {
        aie::vector<bf16, vector_size> x_vec_0 = aie::load_v<vector_size>(it_x_0);
        aie::vector<bf16, vector_size> x_vec_1 = aie::load_v<vector_size>(it_x_1);
        aie::vector<bf16, vector_size> x_buf_vec_0 = aie::load_v<vector_size>(it_x_buf_0);
        aie::vector<bf16, vector_size> x_buf_vec_1 = aie::load_v<vector_size>(it_x_buf_1);
        auto out_vec_0 = aie::add(x_vec_0, x_buf_vec_0);
        auto out_vec_1 = aie::add(x_vec_1, x_buf_vec_1);
        aie::store_v(it_y_0, out_vec_0);
        aie::store_v(it_y_1, out_vec_1);
        aie::store_v(it_x_buf_0, out_vec_0);
        aie::store_v(it_x_buf_1, out_vec_1);
        it_x_0 += 2 * vector_size;
        it_x_1 += 2 * vector_size;
        it_y_0 += 2 * vector_size;
        it_y_1 += 2 * vector_size;
        it_x_buf_0 += 2 * vector_size;
        it_x_buf_1 += 2 * vector_size;
    }
}


///@brief


extern "C" {

void rms_residual(
    bf16 *restrict y,
    bf16 *restrict x_ping,
    bf16 *restrict x_pong,
    bf16 *restrict y_final,
    bf16 *restrict w,
    bf16 *restrict x_temp_buf,
    int* IS_SWA
){
    constexpr int w_prod_lock = 0;
    constexpr int w_cons_lock = 1;
    constexpr int y_prod_lock = 2;
    constexpr int y_cons_lock = 3;
    constexpr int x_prod_lock = 4;
    constexpr int x_cons_lock = 5;
    // lock ids 6..8 (rtp_available, ab_out prod/cons) are allocated on the tile
    // but unused by this kernel: there is no MM2S channel 1 in mem_body.
    (void)y_final;
#if defined(__chess__)
    uint32_t* pkt_id_ptr = reinterpret_cast<uint32_t*>(y + 14);
#else
    // y is restrict, so Peano assumes the lock calls cannot read the packet id and
    // drops a store that a later one overwrites. The packet then goes to the
    // wrong consumer and the layer hangs. volatile keeps every store.
    volatile uint32_t* pkt_id_ptr = reinterpret_cast<volatile uint32_t*>(y + 14);
#endif
    bf16* x;
    bf16* x_buf = x_temp_buf;
    bf16* temp_buf = x_temp_buf + MODEL_DIM;

    AIE_PERSIST_ZERO static bool is_x_ping = false;

    _lock_acquire(x_cons_lock);
    // global attention, skip v proj
    if (IS_SWA[0] == 0) {
        // input later norm + residul
        *pkt_id_ptr = 0;
        is_x_ping = !is_x_ping;
        x = is_x_ping ? x_ping : x_pong;
        _memcpy<MODEL_DIM * sizeof(bf16)>((uint8*)x_buf, (uint8*)x);
        _lock_acquire(w_cons_lock);
        rms_norm(y + 16, x_buf, w);
        _lock_release(w_prod_lock);
        _lock_release(x_prod_lock);
        _lock_release(y_cons_lock, QK_REPEATS);


        _lock_acquire(x_cons_lock);

        // drain the sends issued above, not another batch of them
        _lock_acquire(y_prod_lock, QK_REPEATS);
    }
    // sliding attention
    else {
        // input later norm + residul
        *pkt_id_ptr = 0;
        is_x_ping = !is_x_ping;
        x = is_x_ping ? x_ping : x_pong;

        _memcpy<MODEL_DIM * sizeof(bf16)>((uint8*)x_buf, (uint8*)x);
        _lock_acquire(w_cons_lock);
        rms_norm(y + 16, x_buf, w);
        _lock_release(w_prod_lock);
        _lock_release(x_prod_lock);
        _lock_release(y_cons_lock, SWA_QKV_REPEATS);

        _lock_acquire(x_cons_lock);
    
        _lock_acquire(y_prod_lock, SWA_QKV_REPEATS);
 
    }
    
    *pkt_id_ptr = 0;
    is_x_ping = !is_x_ping;
    x = is_x_ping ? x_ping : x_pong;
    // chunk 2: post_attention_layernorm
    _lock_acquire(w_cons_lock);
    rms_norm(y + 16, x, w);
    _lock_release(w_prod_lock);
    // residual_add already mirrors the new residual into x_buf, so it is
    // the live residual for the pre_feed_forward stage below.
    residual_add(temp_buf, x_buf, y + 16);
    // chunk 3: pre_feedforward_layernorm
    _lock_acquire(w_cons_lock);
    rms_norm(y + 16, temp_buf, w);
    _lock_release(w_prod_lock);
    _lock_release(x_prod_lock);

 
    _lock_release(y_cons_lock, UP_GATE_REPEATS / 4);
    _lock_acquire(y_prod_lock, UP_GATE_REPEATS / 4);
    _lock_release(y_cons_lock, UP_GATE_REPEATS / 4);
    _lock_acquire(y_prod_lock, UP_GATE_REPEATS / 4);
    _lock_release(y_cons_lock, UP_GATE_REPEATS / 4);
    _lock_acquire(y_prod_lock, UP_GATE_REPEATS / 4);
    _lock_release(y_cons_lock, UP_GATE_REPEATS / 4);

    _lock_acquire(y_prod_lock, UP_GATE_REPEATS / 4);
    _lock_acquire(x_cons_lock);
    *pkt_id_ptr = 5;
    is_x_ping = !is_x_ping;
    x = is_x_ping ? x_ping : x_pong;
    _lock_acquire(w_cons_lock);
    rms_norm(temp_buf, x, w);
    _lock_release(w_prod_lock);
    residual_add(temp_buf, x_buf, temp_buf);
    bf16 scale = *((bf16*)(IS_SWA + 1));
    aie::vector<bf16, 16> ss = aie::broadcast<bf16, 16>(scale);
    // two interleaved streams, same reason as rms_norm
    bf16* it_in_0 = temp_buf;
    bf16* it_in_1 = temp_buf + 16;
    bf16* it_out_0 = y + 16;
    bf16* it_out_1 = y + 16 + 16;
    for (int i = 0; i < MODEL_DIM / 32; i++){
        aie::vector<bf16, 16> in_vec_0 = aie::load_v<16>(it_in_0);
        aie::vector<bf16, 16> in_vec_1 = aie::load_v<16>(it_in_1);
        aie::vector<bf16, 16> out_vec_0 = aie::mul(in_vec_0, ss);
        aie::vector<bf16, 16> out_vec_1 = aie::mul(in_vec_1, ss);
        aie::store_v(it_out_0, out_vec_0);
        aie::store_v(it_out_1, out_vec_1);
        it_in_0 += 32;
        it_in_1 += 32;
        it_out_0 += 32;
        it_out_1 += 32;
    }
    _lock_release(x_prod_lock);
    _lock_release(y_cons_lock, 1);
    _lock_acquire(y_prod_lock, 1);
}

}