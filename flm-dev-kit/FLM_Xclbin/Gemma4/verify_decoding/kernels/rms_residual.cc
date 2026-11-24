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
    bf16 *it_y = const_cast<bf16*>(y);
    bf16 *it_x = const_cast<bf16*>(x);
    bf16 *it_w = const_cast<bf16*>(w);
    const float epsilon = 1e-6;

    constexpr int vector_size = 16;
    constexpr float one_over_D = 1.0f / (float)MODEL_DIM;

    aie::accum<accfloat, vector_size> sum_squares = aie::zeros<accfloat>();
    aie::vector<float, vector_size> mx_vec;

    const int F = MODEL_DIM / vector_size;
    for (int i = 0; i < F; i++){
        auto x_vec = aie::load_v<vector_size>(it_x);
        sum_squares = aie::mac_square(sum_squares, x_vec);
        it_x += vector_size;
    }

    it_x = const_cast<bf16*>(x);

    mx_vec = sum_squares.template to_vector<float>();
    float sum = aie::reduce_add(mx_vec);
    #if defined(__chess__)
    sum = sum * one_over_D;
    #else
    sum = vmulf(sum, one_over_D);
    #endif
    sum = sum + epsilon;
    const float divrms = flm_rsqrt(sum);

    for (int i = 0; i < F; i++){
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

///@brief residual add: y = x + x_buf, and x_buf = y
///@param y: output
///@param x: input
///@param x_buf: copy of x
void residual_add(
    bf16 *restrict y,
    const bf16 *restrict x_buf,
    const bf16 *restrict x
){
    bf16 *it_x = const_cast<bf16*>(x);
    bf16 *it_y = const_cast<bf16*>(y);
    bf16 *it_x_buf = const_cast<bf16*>(x_buf);

    for (int i = 0; i < MODEL_DIM / 16; i++) {
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


///@brief


extern "C" {

void rms_residual(
    bf16 *restrict y,
    bf16 *restrict x_ping,
    bf16 *restrict x_pong,
    bf16 *restrict y_final,
    bf16 *restrict w,
    bf16 *restrict x_temp_buf,
    int* IS_SWA,
    int* SKIP_KV
){
    constexpr int w_prod_lock = 0;
    constexpr int w_cons_lock = 1;
    constexpr int y_prod_lock = 2;
    constexpr int y_cons_lock = 3;
    constexpr int x_prod_lock = 4;
    constexpr int x_cons_lock = 5;
    constexpr int rtp_available_lock = 6;
    constexpr int lm_head_out_prod_lock = 7;
    constexpr int lm_head_out_cons_lock = 8;
    uint32_t* pkt_id_ptr = reinterpret_cast<uint32_t*>(y + 14);
    bf16* x;
    bf16* w_input_layernorm = w;
    bf16* w_post_attn_layernorm = w + MODEL_DIM;
    bf16* w_pre_feedforward_layernorm = w + MODEL_DIM * 2;
    bf16* w_post_feedforward_layer_norm = w + MODEL_DIM * 3;
    bf16* x_buf = x_temp_buf;
    bf16* temp_buf = x_temp_buf + MODEL_DIM;

    AIE_PERSIST_ZERO static bool is_x_ping = false;

    _lock_acquire(rtp_available_lock);
    _lock_acquire(x_cons_lock);
    _lock_acquire(w_cons_lock);
    if (IS_SWA[0] == 0) {
        // input later norm + residul
        *pkt_id_ptr = 0;
        is_x_ping = !is_x_ping;
        x = is_x_ping ? x_ping : x_pong;
        #if defined(__chess__)
        memcpy(x_buf, x, MODEL_DIM * sizeof(bf16));
        #else
        copy_vectorized<bf16, MODEL_DIM>(x_buf, x);
        #endif
        rms_norm(y + 16, x_buf, w_input_layernorm);
        _lock_release(x_prod_lock);
        if (SKIP_KV[0] == 0) {
            _lock_release(y_cons_lock, QKV_REPEATS);
        }
        else {
            _lock_release(y_cons_lock, Q_REPEATS);
        }


        _lock_acquire(x_cons_lock);

        if (SKIP_KV[0] == 0) {
            _lock_acquire(y_prod_lock, QKV_REPEATS);
        }
        else {
            _lock_acquire(y_prod_lock, Q_REPEATS);
        }
    }
    else {
        // input later norm + residul
        *pkt_id_ptr = 0;
        is_x_ping = !is_x_ping;
        x = is_x_ping ? x_ping : x_pong;

        #if defined(__chess__)
        memcpy(x_buf, x, MODEL_DIM * sizeof(bf16));
        #else
        copy_vectorized<bf16, MODEL_DIM>(x_buf, x);
        #endif
        rms_norm(y + 16, x_buf, w_input_layernorm);
        _lock_release(x_prod_lock);
        if (SKIP_KV[0] == 0) {
            _lock_release(y_cons_lock, SWA_QKV_REPEATS);
        }
        else {
            _lock_release(y_cons_lock, SWA_Q_REPEATS);
        }

        _lock_acquire(x_cons_lock);

        if (SKIP_KV[0] == 0) {
            _lock_acquire(y_prod_lock, SWA_QKV_REPEATS);
        }
        else {
            _lock_acquire(y_prod_lock, SWA_Q_REPEATS);
        }
    }
    
    *pkt_id_ptr = 0;
    is_x_ping = !is_x_ping;
    x = is_x_ping ? x_ping : x_pong;
    rms_norm(y + 16, x, w_post_attn_layernorm);
    residual_add(temp_buf, x_buf, y + 16);
    // pre_feed_forward
    #if defined(__chess__)
    memcpy(x_buf, temp_buf, MODEL_DIM * sizeof(bf16));
    #else
    copy_vectorized<bf16, MODEL_DIM>(x_buf, temp_buf);
    #endif
    rms_norm(y + 16, temp_buf, w_pre_feedforward_layernorm);
    _lock_release(x_prod_lock);
    #if MODEL_TYPE == GEMMA4_E2B
    if (SKIP_KV[0] == 0) {
        _lock_release(y_cons_lock, UP_GATE_REPEATS);
    }
    else {
        _lock_release(y_cons_lock, UP_GATE_REPEATS * 2);
    }
    #else
    _lock_release(y_cons_lock, UP_GATE_REPEATS);
    #endif

    _lock_acquire(x_cons_lock);

    #if MODEL_TYPE == GEMMA4_E2B
    if (SKIP_KV[0] == 0) {
        _lock_acquire(y_prod_lock, UP_GATE_REPEATS);
    }
    else {
        _lock_acquire(y_prod_lock, UP_GATE_REPEATS * 2);
    }
    #else
    _lock_acquire(y_prod_lock, UP_GATE_REPEATS);
    #endif
    is_x_ping = !is_x_ping;
    x = is_x_ping ? x_ping : x_pong;
    _lock_acquire(lm_head_out_prod_lock, 1);
    rms_norm(temp_buf, x, w_post_feedforward_layer_norm);
    residual_add(y_final, x_buf, temp_buf);
    _lock_release(w_prod_lock);
    _lock_release(x_prod_lock);
    _lock_release(lm_head_out_cons_lock, 1);
}

}