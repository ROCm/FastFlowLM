// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#include "common.h"
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
    constexpr float one_over_D = 1.0f / (float)D;

    aie::accum<accfloat, vector_size> sum_squares = aie::zeros<accfloat>();
    aie::vector<float, vector_size> mx_vec;

    const int F = D / vector_size;
    for (int i = 0; i < F; i++){
        auto x_vec = aie::load_v<vector_size>(it_x);
        sum_squares = aie::mac_square(sum_squares, x_vec);
        it_x += vector_size;
    }

    it_x = const_cast<bf16*>(x);

    mx_vec = sum_squares.template to_vector<float>();
    float sum = aie::reduce_add(mx_vec);
    sum = sum * one_over_D;
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


///@brief


extern "C" {

void rms_residual(
    bf16 *restrict y,
    bf16 *restrict x_ping,
    bf16 *restrict x_pong,
    bf16 *restrict w_ping,
    bf16 *restrict w_pong,
    bf16 *restrict y_out_0,
    bf16 *restrict y_out_1,
    int* IS_ATTN
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
    alignas(aie::vector_decl_align) static bf16 x_buf[D];
    alignas(aie::vector_decl_align) static bf16 temp_buf[D];
    uint32_t* pkt_id_ptr = reinterpret_cast<uint32_t*>(y + 14);
    bf16* x;
    bf16* w;

    // Peano's CDO loader does not zero .bss; keep ping/pong selectors in .data.
#if !defined(__chess__)
    __attribute__((section(".data")))
#endif
    static bool is_x_ping = false;
#if !defined(__chess__)
    __attribute__((section(".data")))
#endif
    static bool is_lm_head_ping = false;
#if !defined(__chess__)
    __attribute__((section(".data")))
#endif
    static bool is_w_ping = false;

    _lock_acquire(rtp_available_lock);
    if (IS_ATTN[0] == 1) {
        // input later norm + residul
        _lock_acquire(x_cons_lock);
        _lock_acquire(w_cons_lock);
        *pkt_id_ptr = 0;
        is_x_ping = !is_x_ping;
        x = is_x_ping ? x_ping : x_pong;
        is_w_ping = !is_w_ping;
        w = is_w_ping ? w_ping : w_pong;
        memcpy(x_buf, x, D * sizeof(bf16));
        rms_norm(y + 16, x_buf, w);
        _lock_release(x_prod_lock);
        _lock_release(w_prod_lock);
        _lock_release(y_cons_lock, qkv_repeats);


        _lock_acquire(x_cons_lock);

        _lock_acquire(y_prod_lock, qkv_repeats);
        _lock_acquire(w_cons_lock);
        *pkt_id_ptr = 0;
        is_x_ping = !is_x_ping;
        x = is_x_ping ? x_ping : x_pong;
        is_w_ping = !is_w_ping;
        w = is_w_ping ? w_ping : w_pong;
        rms_norm(y + 16, x, w);
        residual_add(temp_buf, x_buf, y + 16);
        _lock_release(w_prod_lock);
        // pre_feed_forward
        memcpy(x_buf, temp_buf, D * sizeof(bf16));
        _lock_acquire(w_cons_lock);
        is_w_ping = !is_w_ping;
        w = is_w_ping ? w_ping : w_pong;
        rms_norm(y + 16, temp_buf, w);
        _lock_release(w_prod_lock);

        _lock_release(x_prod_lock);
        _lock_release(y_cons_lock, up_gate_repeats);

        _lock_acquire(x_cons_lock);
        _lock_acquire(y_prod_lock, up_gate_repeats);
        _lock_acquire(w_cons_lock);
        *pkt_id_ptr = 1;
        is_x_ping = !is_x_ping;
        x = is_x_ping ? x_ping : x_pong;
        is_w_ping = !is_w_ping;
        w = is_w_ping ? w_ping : w_pong;
        rms_norm(temp_buf, x, w);
        residual_add(y + 16, x_buf, temp_buf);
        _lock_release(w_prod_lock);
        _lock_release(x_prod_lock);
        _lock_release(y_cons_lock, 1);
        _lock_acquire(y_prod_lock, 1);
    }
    else {
        _lock_acquire(x_cons_lock);
        _lock_acquire(w_cons_lock);
        *pkt_id_ptr = 0;
        is_x_ping = !is_x_ping;
        x = is_x_ping ? x_ping : x_pong;
        is_w_ping = !is_w_ping;
        w = is_w_ping ? w_ping : w_pong;
        if (IS_ATTN[0] == 2) {
            memcpy(y + 16, x, D * sizeof(bf16));
        } else {
            rms_norm(y + 16, x, w);
        }
        _lock_release(x_prod_lock);
        _lock_release(w_prod_lock);

        constexpr int vocab_size_per_round = vocab_size_padded / D;
        constexpr int y_repeats_per_round = D / m_per_round;
        for (int i = 0; i < vocab_size_per_round; i++) {
            _lock_release(y_cons_lock, y_repeats_per_round);
            _lock_acquire(lm_head_out_prod_lock, 1);
            _lock_acquire(x_cons_lock);

            is_lm_head_ping = !is_lm_head_ping;
            bf16* lm_head_dst = is_lm_head_ping ? y_out_0 : y_out_1;
            is_x_ping = !is_x_ping;
            x = is_x_ping ? x_ping : x_pong;
            memcpy(lm_head_dst, x, D * sizeof(bf16));

            _lock_release(x_prod_lock);
            _lock_release(lm_head_out_cons_lock, 1);
            _lock_acquire(y_prod_lock, y_repeats_per_round);
        }
    }
}

}