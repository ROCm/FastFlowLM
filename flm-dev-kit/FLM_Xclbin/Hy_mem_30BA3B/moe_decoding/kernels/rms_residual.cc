// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#include "aie_array_layout.h"
#include "aie_kernel_utils.h"
#include "typedef.hpp"

#include "passThroughCommon.hpp"


///@brief  rms norm: y = nrom(x) * w;
///@param y: output
///@param x: input
///@param w: weight
void rms_norm(
    bf16 *restrict y,
    const bf16 *restrict x,
    const bf16 *restrict w
){
    const float epsilon = RMS_EPS;

    constexpr int vector_size = 16;
    constexpr float one_over_D = 1.0f / (float)MODEL_DIM;

    bf16 *it_y_0 = const_cast<bf16*>(y);
    bf16 *it_y_1 = const_cast<bf16*>(y) + vector_size;
    bf16 *it_x_0 = const_cast<bf16*>(x);
    bf16 *it_x_1 = const_cast<bf16*>(x) + vector_size;
    bf16 *it_w_0 = const_cast<bf16*>(w);
    bf16 *it_w_1 = const_cast<bf16*>(w) + vector_size;

    aie::accum<accfloat, vector_size> sum_squares_0 = aie::zeros<accfloat>();
    aie::accum<accfloat, vector_size> sum_squares_1 = aie::zeros<accfloat>();
    aie::vector<float, vector_size> mx_vec;

    const int F = MODEL_DIM / (2 * vector_size);
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

    mx_vec = aie::add(sum_squares_0.template to_vector<float>(),
                      sum_squares_1.template to_vector<float>());
    float sum = aie::reduce_add(mx_vec);
    sum = AIE_FADD(AIE_FMUL(sum, one_over_D), epsilon);
    aie::vector<float, vector_size> sum_vec = aie::broadcast<float, vector_size>(sum);
    float divrms = aie::invsqrt(sum_vec)[0];

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
    event1();
}


///@brief residual add: y = x + x_buf, and x_buf = y
///@param y: output
///@param x: input
///@param x_buf: copy of x
void residual_add(
    bf16 *restrict y,
    const bf16 *restrict x_buf,
    const bf16 *restrict x
){
    constexpr int vector_size = 16;
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


extern "C" {

// One decoder layer per call. y is the outgoing packet buffer: y + 14 holds the
// packet id, y + 16 the data, so every send below first picks a destination.
void rms_residual(
    bf16 * AIE_DMA_RESTRICT y,
    bf16 * AIE_DMA_RESTRICT x_ping,
    bf16 * AIE_DMA_RESTRICT x_pong,
    bf16 * AIE_DMA_RESTRICT w,
    bf16 * AIE_DMA_RESTRICT x_buf
){
    constexpr int w_prod_lock = 0;
    constexpr int w_cons_lock = 1;
    constexpr int y_prod_lock = 2;
    constexpr int y_cons_lock = 3;
    constexpr int x_prod_lock = 4;
    constexpr int x_cons_lock = 5;
    uint32_t* pkt_id_ptr = reinterpret_cast<uint32_t*>(y + 14);
    bf16* x;

    AIE_PERSIST_ZERO static bool is_x_ping = false;

    bf16* w_input_norm = w;
    bf16* w_post_attention_norm = w_input_norm + MODEL_DIM;

    // 1. layer input: keep a copy for the residual, send input_layernorm(x)
    //    to the proj engine once per q/k/v row-block round.
    _lock_acquire(x_cons_lock);
    _lock_acquire(w_cons_lock);
    *pkt_id_ptr = pkt_id_rms_to_proj;
    is_x_ping = !is_x_ping;
    x = is_x_ping ? x_ping : x_pong;
    _memcpy<MODEL_DIM * 2>((uint8*)x_buf, (uint8*)x);
    rms_norm(y + 16, x_buf, w_input_norm);
    _lock_release(x_prod_lock);
    _lock_release(y_cons_lock, QKV_REPEATS);

    // 2. o_proj result: residual add, then post_attention_layernorm.
    _lock_acquire(x_cons_lock);
    _lock_acquire(y_prod_lock, QKV_REPEATS);
    is_x_ping = !is_x_ping;
    x = is_x_ping ? x_ping : x_pong;
    residual_add(x_buf, x_buf, x);
    rms_norm(y + 16, x_buf, w_post_attention_norm);
    _lock_release(x_prod_lock);

    // 3. send the normed x to GLU (the router reads it from there) ...
    *pkt_id_ptr = pkt_id_rms_to_glu;
    _lock_release(y_cons_lock, 1);
    _lock_acquire(y_prod_lock, 1);

    // ... and to the proj engine for the shared expert's up/gate.
    *pkt_id_ptr = pkt_id_rms_to_proj;
    _lock_release(y_cons_lock, UP_GATE_REPEATS);

    // 4. shared expert down result: pass it on to GLU, which adds the routed
    //    experts to it.
    _lock_acquire(x_cons_lock);
    _lock_acquire(y_prod_lock, UP_GATE_REPEATS);
    *pkt_id_ptr = pkt_id_rms_to_glu;
    is_x_ping = !is_x_ping;
    x = is_x_ping ? x_ping : x_pong;
    copy_bf16_to_bf16<MODEL_DIM>(y + 16, x);
    _lock_release(w_prod_lock);
    _lock_release(x_prod_lock);
    _lock_release(y_cons_lock, 1);

    // 5. MoE output from GLU: final residual add, send the layer output out.
    _lock_acquire(x_cons_lock);
    _lock_acquire(y_prod_lock, 1);
    *pkt_id_ptr = pkt_id_rms_to_it;
    is_x_ping = !is_x_ping;
    x = is_x_ping ? x_ping : x_pong;
    residual_add(y + 16, x_buf, x);
    _lock_release(x_prod_lock);
    _lock_release(y_cons_lock, 1);
    _lock_acquire(y_prod_lock, 1);
}

}
