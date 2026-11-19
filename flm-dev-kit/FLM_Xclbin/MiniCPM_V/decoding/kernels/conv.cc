// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
#include "model_spec.h"
#include "aie_kernel_utils.h"
#include "aie_array_layout.h"
#include "flm_rsqrt.h"
// #include "lut_based_ops.h"

void l2_norm(
    bf16 *restrict y
){
    const float epsilon = 1e-6;

    constexpr int vector_size = 16;

    bf16 *it_y_0 = const_cast<bf16*>(y);
    bf16 *it_y_1 = const_cast<bf16*>(y) + vector_size;
    bf16 *it_x_0 = const_cast<bf16*>(y);
    bf16 *it_x_1 = const_cast<bf16*>(y) + vector_size;

    aie::accum<accfloat, vector_size> sum_squares_0 = aie::zeros<accfloat>();
    aie::accum<accfloat, vector_size> sum_squares_1 = aie::zeros<accfloat>();
    aie::vector<float, vector_size> mx_vec;

    const int F = L_V_HEAD_DIM / (2 * vector_size);
    for (int i = 0; i < F; i++){
        auto x_vec_0 = aie::load_v<vector_size>(it_x_0);
        auto x_vec_1 = aie::load_v<vector_size>(it_x_1);
        sum_squares_0 = aie::mac_square(sum_squares_0, x_vec_0);
        sum_squares_1 = aie::mac_square(sum_squares_1, x_vec_1);
        it_x_0 += 2 * vector_size;
        it_x_1 += 2 * vector_size;
    }

    it_x_0 = const_cast<bf16*>(y);
    it_x_1 = const_cast<bf16*>(y) + vector_size;

    mx_vec = aie::add(sum_squares_0.template to_vector<float>(),
                      sum_squares_1.template to_vector<float>());
    float sum = aie::reduce_add(mx_vec);
#if defined(__chess__)
    sum = sum + epsilon;
#else
    // Same math as the chess branch, rounded like chess (see AIE_FADD).
    sum = AIE_FADD(sum, epsilon);
#endif
    const float divrms = flm_rsqrt(sum);

    for (int i = 0; i < F; i++){
        aie::vector<bf16, vector_size> x_vec_0 = aie::load_v<vector_size>(it_x_0);
        aie::vector<bf16, vector_size> x_vec_1 = aie::load_v<vector_size>(it_x_1);
        aie::accum<accfloat, vector_size> x_acc_0;
        aie::accum<accfloat, vector_size> x_acc_1;
        x_acc_0.from_vector(x_vec_0);
        x_acc_1.from_vector(x_vec_1);
        aie::vector<bf16, vector_size> o_vec_0 = aie::mul(x_acc_0.template to_vector<float>(), divrms);
        aie::vector<bf16, vector_size> o_vec_1 = aie::mul(x_acc_1.template to_vector<float>(), divrms);
        aie::store_v(it_y_0, o_vec_0);
        aie::store_v(it_y_1, o_vec_1);
        it_x_0 += 2 * vector_size;
        it_x_1 += 2 * vector_size;
        it_y_0 += 2 * vector_size;
        it_y_1 += 2 * vector_size;
    }
}

void scale_q(
    bf16 *restrict y
){
    const float epsilon = 1e-6;

    constexpr int vector_size = 16;
    const int F = L_V_HEAD_DIM / (2 * vector_size);

    bf16 *it_y_0 = const_cast<bf16*>(y);
    bf16 *it_y_1 = const_cast<bf16*>(y) + vector_size;
    bf16 *it_x_0 = const_cast<bf16*>(y);
    bf16 *it_x_1 = const_cast<bf16*>(y) + vector_size;

    for (int i = 0; i < F; i++){
        aie::vector<bf16, vector_size> x_vec_0 = aie::load_v<vector_size>(it_x_0);
        aie::vector<bf16, vector_size> x_vec_1 = aie::load_v<vector_size>(it_x_1);
        aie::vector<bf16, vector_size> o_vec_0 = aie::mul(x_vec_0, aie::broadcast<bf16, vector_size>(LINEAR_ATTN_SCALE));
        aie::vector<bf16, vector_size> o_vec_1 = aie::mul(x_vec_1, aie::broadcast<bf16, vector_size>(LINEAR_ATTN_SCALE));
        aie::store_v(it_y_0, o_vec_0);
        aie::store_v(it_y_1, o_vec_1);
        it_x_0 += 2 * vector_size;
        it_x_1 += 2 * vector_size;
        it_y_0 += 2 * vector_size;
        it_y_1 += 2 * vector_size;
    }
}

constexpr int wx_prod_lock = 0;
constexpr int wx_cons_lock = 1;
constexpr int y_prod_lock = 2;
constexpr int y_cons_lock = 3;
constexpr int ab_prod_lock = 4;
constexpr int ab_cons_lock = 5;
constexpr int tb_prod_lock = 7;
constexpr int tb_cons_lock = 8;

AIE_PERSIST_ZERO bool is_wx_ping = false;

void _conv(
    bf16* wx,
    bf16* y
){
    constexpr int vec_size = 16;

    bf16* x_0_a = wx;
    bf16* x_0_b = wx + vec_size;
    bf16* w_0_a = wx + L_V_HEAD_DIM;
    bf16* w_0_b = wx + L_V_HEAD_DIM + vec_size;
    bf16* x_1_a = wx + 2 * L_V_HEAD_DIM;
    bf16* x_1_b = wx + 2 * L_V_HEAD_DIM + vec_size;
    bf16* w_1_a = wx + 3 * L_V_HEAD_DIM;
    bf16* w_1_b = wx + 3 * L_V_HEAD_DIM + vec_size;
    bf16* x_2_a = wx + 4 * L_V_HEAD_DIM;
    bf16* x_2_b = wx + 4 * L_V_HEAD_DIM + vec_size;
    bf16* w_2_a = wx + 5 * L_V_HEAD_DIM;
    bf16* w_2_b = wx + 5 * L_V_HEAD_DIM + vec_size;
    bf16* x_3_a = wx + 6 * L_V_HEAD_DIM;
    bf16* x_3_b = wx + 6 * L_V_HEAD_DIM + vec_size;
    bf16* w_3_a = wx + 7 * L_V_HEAD_DIM;
    bf16* w_3_b = wx + 7 * L_V_HEAD_DIM + vec_size;

    bf16* y_a = y;
    bf16* y_b = y + vec_size;

    for (int i = 0; i < L_V_HEAD_DIM / (2 * vec_size); i++){
        aie::vector<bf16, vec_size> x_vec_0_a = aie::load_v<vec_size>(x_0_a);
        aie::vector<bf16, vec_size> x_vec_0_b = aie::load_v<vec_size>(x_0_b);
        aie::vector<bf16, vec_size> w_vec_0_a = aie::load_v<vec_size>(w_0_a);
        aie::vector<bf16, vec_size> w_vec_0_b = aie::load_v<vec_size>(w_0_b);
        aie::vector<bf16, vec_size> x_vec_1_a = aie::load_v<vec_size>(x_1_a);
        aie::vector<bf16, vec_size> x_vec_1_b = aie::load_v<vec_size>(x_1_b);
        aie::vector<bf16, vec_size> w_vec_1_a = aie::load_v<vec_size>(w_1_a);
        aie::vector<bf16, vec_size> w_vec_1_b = aie::load_v<vec_size>(w_1_b);
        aie::vector<bf16, vec_size> x_vec_2_a = aie::load_v<vec_size>(x_2_a);
        aie::vector<bf16, vec_size> x_vec_2_b = aie::load_v<vec_size>(x_2_b);
        aie::vector<bf16, vec_size> w_vec_2_a = aie::load_v<vec_size>(w_2_a);
        aie::vector<bf16, vec_size> w_vec_2_b = aie::load_v<vec_size>(w_2_b);
        aie::vector<bf16, vec_size> x_vec_3_a = aie::load_v<vec_size>(x_3_a);
        aie::vector<bf16, vec_size> x_vec_3_b = aie::load_v<vec_size>(x_3_b);
        aie::vector<bf16, vec_size> w_vec_3_a = aie::load_v<vec_size>(w_3_a);
        aie::vector<bf16, vec_size> w_vec_3_b = aie::load_v<vec_size>(w_3_b);

        aie::accum<accfloat, vec_size> acc_a = aie::mul(x_vec_0_a, w_vec_0_a);
        aie::accum<accfloat, vec_size> acc_b = aie::mul(x_vec_0_b, w_vec_0_b);
        acc_a = aie::mac(acc_a, x_vec_1_a, w_vec_1_a);
        acc_b = aie::mac(acc_b, x_vec_1_b, w_vec_1_b);
        acc_a = aie::mac(acc_a, x_vec_2_a, w_vec_2_a);
        acc_b = aie::mac(acc_b, x_vec_2_b, w_vec_2_b);
        acc_a = aie::mac(acc_a, x_vec_3_a, w_vec_3_a);
        acc_b = aie::mac(acc_b, x_vec_3_b, w_vec_3_b);

        constexpr bfloat16 one = 1.0f;

        aie::vector<float, vec_size> y_vec_f32_a = acc_a.template to_vector<float>();
        aie::vector<float, vec_size> y_vec_f32_b = acc_b.template to_vector<float>();
        aie::vector<bf16, vec_size> y_vec_a      = acc_a.template to_vector<bf16>();
        aie::vector<bf16, vec_size> y_vec_b      = acc_b.template to_vector<bf16>();
        aie::vector<float, vec_size> y_scaled_a = aie::mul(y_vec_a, aie::broadcast<bf16, vec_size>(-exp_scale));
        aie::vector<float, vec_size> y_scaled_b = aie::mul(y_vec_b, aie::broadcast<bf16, vec_size>(-exp_scale));
        aie::vector<bf16, vec_size> y_exp_a = aie::exp2(y_scaled_a);
        aie::vector<bf16, vec_size> y_exp_b = aie::exp2(y_scaled_b);
        aie::vector<bf16, vec_size> vec_one = aie::broadcast<bf16, vec_size>(one);
        aie::accum<accfloat, vec_size> y_exp_acc_a;
        aie::accum<accfloat, vec_size> y_exp_acc_b;
        y_exp_acc_a.from_vector(y_exp_a);
        y_exp_acc_b.from_vector(y_exp_b);
        y_exp_acc_a = aie::add(y_exp_acc_a, vec_one);
        y_exp_acc_b = aie::add(y_exp_acc_b, vec_one);
        aie::vector<float, vec_size> inv_y_a = aie::inv(y_exp_acc_a.template to_vector<float>());
        aie::vector<float, vec_size> inv_y_b = aie::inv(y_exp_acc_b.template to_vector<float>());
        aie::accum<accfloat, vec_size> silu_y_f32_a = aie::mul(inv_y_a, y_vec_f32_a);
        aie::accum<accfloat, vec_size> silu_y_f32_b = aie::mul(inv_y_b, y_vec_f32_b);

        aie::store_v(y_a, silu_y_f32_a.template to_vector<bf16>());
        aie::store_v(y_b, silu_y_f32_b.template to_vector<bf16>());

        x_0_a += 2 * vec_size;
        x_0_b += 2 * vec_size;
        w_0_a += 2 * vec_size;
        w_0_b += 2 * vec_size;
        x_1_a += 2 * vec_size;
        x_1_b += 2 * vec_size;
        w_1_a += 2 * vec_size;
        w_1_b += 2 * vec_size;
        x_2_a += 2 * vec_size;
        x_2_b += 2 * vec_size;
        w_2_a += 2 * vec_size;
        w_2_b += 2 * vec_size;
        x_3_a += 2 * vec_size;
        x_3_b += 2 * vec_size;
        w_3_a += 2 * vec_size;
        w_3_b += 2 * vec_size;
        y_a += 2 * vec_size;
        y_b += 2 * vec_size;
    }
}


extern "C" {
AIE_PERSIST_ZERO static bool is_y_ping = false;
void conv(
    bf16 * w_ping,
    bf16 * w_pong,
    bf16 * y_buffer_0,
    bf16 * y_buffer_1,
    bf16 * conv_buffer,
    float* alpha_beta,
    int tb_flag
)
{
    FLM_SET_FLOOR_ROUNDING();
    bf16* q_ptr = conv_buffer;
    bf16* k_ptr = conv_buffer + Linear_DQ;
    bf16* v_ptr = conv_buffer + Linear_DK + Linear_DQ;
    if (tb_flag != 0) {
        _lock_acquire(tb_prod_lock);
    }
    for (int i = 0; i < (Linear_DQ) / L_V_HEAD_DIM; i++)
    {
        _lock_acquire(wx_cons_lock);
        is_wx_ping = !is_wx_ping;
        bf16* w_using = is_wx_ping ? w_ping : w_pong;
        _conv(w_using, q_ptr + i * L_V_HEAD_DIM);
        l2_norm(q_ptr + i * L_V_HEAD_DIM);
        scale_q(q_ptr + i * L_V_HEAD_DIM);
        _lock_release(wx_prod_lock);
    }
    for (int i = 0; i < (Linear_DK) / L_V_HEAD_DIM; i++)
    {
        _lock_acquire(wx_cons_lock);
        is_wx_ping = !is_wx_ping;
        bf16* w_using = is_wx_ping ? w_ping : w_pong;
        _conv(w_using, k_ptr + i * L_V_HEAD_DIM);
        l2_norm(k_ptr + i * L_V_HEAD_DIM);
        _lock_release(wx_prod_lock);
    }
    for (int i = 0; i < (Linear_DV) / L_V_HEAD_DIM; i++)
    {
        _lock_acquire(wx_cons_lock);
        is_wx_ping = !is_wx_ping;
        bf16* w_using = is_wx_ping ? w_ping : w_pong;
        _conv(w_using, v_ptr + i * L_V_HEAD_DIM);
        _lock_release(wx_prod_lock);
    }
    _lock_acquire(ab_cons_lock);
    // for (int i = 0; i < 4096; i++){
    //     q_ptr[i] = (bf16)LINEAR_ATTN_SCALE;
    //     // v_ptr[i] = (bf16)1.0f;
    // }
    
    constexpr int G = L_V_HEADS / L_QK_HEADS;
    constexpr int stride = L_V_HEAD_DIM + 32;
    float * alpha_ptr = alpha_beta;
    float * beta_ptr = alpha_beta + L_V_HEADS_PADDED;
    for (int h = 0; h < L_QK_HEADS; h++){
        for (int g = 0; g < G; g++){
            _lock_acquire(y_prod_lock, 1);
            is_y_ping = !is_y_ping;
            bf16* y_using = is_y_ping ? y_buffer_0 : y_buffer_1;
            // k alpha padding | v alpha padding | q padding
            memcpy(y_using, k_ptr, L_V_HEAD_DIM * sizeof(bf16));
            memcpy(y_using + L_V_HEAD_DIM, alpha_ptr, sizeof(float));
            alpha_ptr++;
            y_using += stride;
            // *beta_ptr = 1.0f;
            memcpy(y_using, v_ptr, L_V_HEAD_DIM * sizeof(bf16));
            memcpy(y_using + L_V_HEAD_DIM, beta_ptr, sizeof(float));
            v_ptr += L_V_HEAD_DIM;
            beta_ptr++;
            y_using += stride;
            memcpy(y_using, q_ptr, L_V_HEAD_DIM * sizeof(bf16));
            _lock_release(y_cons_lock, 1);
        }
        q_ptr += L_V_HEAD_DIM;
        k_ptr += L_V_HEAD_DIM;
    }

    _lock_release(ab_prod_lock);

    if (tb_flag != 0) {
        for (int i = 0; i < L_V_HEADS_PADDED * 2; i++){
            float v = alpha_beta[i];
            bf16 v_bf16 = (bf16)v;
            conv_buffer[i] = v_bf16;
        }
        _lock_release(tb_cons_lock);
    }

}


}