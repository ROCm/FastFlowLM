// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include "aie_array_layout.h"
#include "aie_kernel_utils.h"
#include "typedef.hpp"
#include "flm_rsqrt.h"

#ifdef HAS_QK_NORM
///@brief pseduo rms norm: y = x * w
///@param y: output
///@param x_buf: copy of x
///@param x: input
///@param w: weight
///@param overwrite_x_buf: if true, overwrite x_buf with x
void qk_norm(
    const bf16 *restrict x,
    const bf16 *restrict w
){
    const float epsilon = 1e-6;

    constexpr int vector_size = 16;
    constexpr float one_over_D = 1.0f / (float)DH;

    // Two interleaved streams: a single accumulator serialises back-to-back
    // mac_square on its own dependency, and a 16-wide vector is too narrow to
    // cover the latency. Two of them halve the recurrence.
    static_assert(DH % (2 * vector_size) == 0,
                  "DH must cover a whole number of interleaved pairs");
    const int F = DH / (2 * vector_size);

    bf16 *it_x_0 = const_cast<bf16*>(x);
    bf16 *it_x_1 = const_cast<bf16*>(x) + vector_size;
    bf16 *it_w_0 = const_cast<bf16*>(w);
    bf16 *it_w_1 = const_cast<bf16*>(w) + vector_size;

    aie::accum<accfloat, vector_size> sum_squares_0 = aie::zeros<accfloat>();
    aie::accum<accfloat, vector_size> sum_squares_1 = aie::zeros<accfloat>();
    aie::vector<float, vector_size> mx_vec;

    for (int i = 0; i < F; i++)
    chess_prepare_for_pipelining chess_loop_range(8, ) {
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

    for (int i = 0; i < F; i++)
    chess_prepare_for_pipelining chess_loop_range(8, ) {
        aie::vector<bf16, vector_size> x_vec_0 = aie::load_v<vector_size>(it_x_0);
        aie::vector<bf16, vector_size> x_vec_1 = aie::load_v<vector_size>(it_x_1);
        aie::vector<bf16, vector_size> w_vec_0 = aie::load_v<vector_size>(it_w_0);
        aie::vector<bf16, vector_size> w_vec_1 = aie::load_v<vector_size>(it_w_1);
        aie::vector<float, vector_size> wx_vec_0 = aie::mul(x_vec_0, w_vec_0);
        aie::vector<float, vector_size> wx_vec_1 = aie::mul(x_vec_1, w_vec_1);
        aie::vector<bf16, vector_size> o_vec_0 = aie::mul(wx_vec_0, divrms);
        aie::vector<bf16, vector_size> o_vec_1 = aie::mul(wx_vec_1, divrms);
        aie::store_v(it_x_0, o_vec_0);
        aie::store_v(it_x_1, o_vec_1);
        it_x_0 += 2 * vector_size;
        it_x_1 += 2 * vector_size;
        it_w_0 += 2 * vector_size;
        it_w_1 += 2 * vector_size;
    }
}


void v_norm(
    const bf16 *restrict y,
    const bf16 *restrict x
){
    const float epsilon = 1e-6;

    constexpr int vector_size = 16;
    constexpr float one_over_D = 1.0f / (float)DH;

    // Two interleaved streams: a single accumulator serialises back-to-back
    // mac_square on its own dependency, and a 16-wide vector is too narrow to
    // cover the latency. Two of them halve the recurrence.
    static_assert(DH % (2 * vector_size) == 0,
                  "DH must cover a whole number of interleaved pairs");
    const int F = DH / (2 * vector_size);

    bf16 *it_x_0 = const_cast<bf16*>(x);
    bf16 *it_x_1 = const_cast<bf16*>(x) + vector_size;
    bf16 *it_y_0 = const_cast<bf16*>(y);
    bf16 *it_y_1 = const_cast<bf16*>(y) + vector_size;

    aie::accum<accfloat, vector_size> sum_squares_0 = aie::zeros<accfloat>();
    aie::accum<accfloat, vector_size> sum_squares_1 = aie::zeros<accfloat>();
    aie::vector<float, vector_size> mx_vec;

    for (int i = 0; i < F; i++)
    chess_prepare_for_pipelining chess_loop_range(8, ) {
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

    for (int i = 0; i < F; i++)
    chess_prepare_for_pipelining chess_loop_range(8, ) {
        aie::vector<bf16, vector_size> x_vec_0 = aie::load_v<vector_size>(it_x_0);
        aie::vector<bf16, vector_size> x_vec_1 = aie::load_v<vector_size>(it_x_1);
        aie::accum<accfloat, vector_size> x_acc_0;
        aie::accum<accfloat, vector_size> x_acc_1;
        x_acc_0.from_vector(x_vec_0);
        x_acc_1.from_vector(x_vec_1);
        aie::vector<float, vector_size> x_float_0 = x_acc_0.template to_vector<float>();
        aie::vector<float, vector_size> x_float_1 = x_acc_1.template to_vector<float>();
        aie::vector<bf16, vector_size> o_vec_0 = aie::mul(x_float_0, divrms);
        aie::vector<bf16, vector_size> o_vec_1 = aie::mul(x_float_1, divrms);
        aie::store_v(it_y_0, o_vec_0);
        aie::store_v(it_y_1, o_vec_1);
        it_x_0 += 2 * vector_size;
        it_x_1 += 2 * vector_size;
        it_y_0 += 2 * vector_size;
        it_y_1 += 2 * vector_size;
    }
}
#endif

void apply_rope(
    bf16 *restrict y,
    bf16 *restrict x,
    bf16 *restrict cos_val,
    bf16 *restrict sin_val
) {
    constexpr int vector_size = 16;
    const int DH_2 = DH / 2;
    constexpr int F = DH_2 / vector_size;
    bf16 *it_y_p1 = y;
    bf16 *it_y_p2 = y + DH_2;
    bf16 *it_x_p1 = x;
    bf16 *it_x_p2 = x + DH_2;
    bf16 *it_cos = cos_val;
    bf16 *it_sin = sin_val;
    
    for (int i = 0; i < F; i++)
    chess_prepare_for_pipelining chess_loop_range(8, ) {
        aie::vector<bf16, vector_size> x1_vec = aie::load_v<vector_size>(it_x_p1);
        aie::vector<bf16, vector_size> x2_vec = aie::load_v<vector_size>(it_x_p2);
        aie::vector<bf16, vector_size> cos_vec = aie::load_v<vector_size>(it_cos);
        aie::vector<bf16, vector_size> sin_vec = aie::load_v<vector_size>(it_sin);
        aie::accum<accfloat, vector_size> C = aie::mul(x1_vec, cos_vec);
#if defined(__chess__)
        aie::vector<bf16, vector_size> neg_sin_vec = aie::neg(sin_vec);
        C = aie::mac(C, x2_vec, neg_sin_vec);
#else
        // Same math as the chess branch, without the bf16 vector negate, which
        // hangs the rope core under Peano.
        C = aie::msc(C, x2_vec, sin_vec);
#endif
        aie::accum<accfloat, vector_size> D = aie::mul(x1_vec, sin_vec);
        D = aie::mac(D, x2_vec, cos_vec);
        aie::store_v(it_y_p1, C.template to_vector<bf16>());
        aie::store_v(it_y_p2, D.template to_vector<bf16>());
        it_y_p1 += vector_size;
        it_y_p2 += vector_size;
        it_x_p1 += vector_size;
        it_x_p2 += vector_size;
        it_cos += vector_size;
        it_sin += vector_size;
    }
}

///@brief pseduo RoPE: q = q, k = k + 1, v = v - 1
///@param y: output
///@param x: input
///@param w: weight
///@note: C = A cos - B sin, D = A sin + B cos
void _rotate(
    bf16 * restrict q,
    bf16 * restrict qkv,
    bf16 * restrict rope_w,
    bf16 * restrict norm_weight
) {
    bf16 *it_rope = rope_w;
    bf16* cos_val = it_rope;
    bf16* sin_val = it_rope + DH / 2;
    const int DH_2 = DH / 2;
    qk_norm(qkv, norm_weight);
    apply_rope(q, qkv, cos_val, sin_val);
}

extern "C" {

void rope(
    bf16* restrict q,
    bf16* restrict k,
    bf16* restrict v,
    bf16* restrict qkv_ping,
    bf16* restrict qkv_pong,
    bf16* restrict rope_w
)
{
    constexpr int qkv_prod_lock = 0;
    constexpr int qkv_cons_lock = 1;
    constexpr int q_prod_lock = 2;
    constexpr int q_cons_lock = 3;
    constexpr int k_prod_lock = 4;
    constexpr int k_cons_lock = 5;
    constexpr int v_prod_lock = 6;
    constexpr int v_cons_lock = 7;
    constexpr int rope_prod_lock = 8;
    constexpr int rope_cons_lock = 9;
    AIE_PERSIST_ZERO static bool is_qkv_ping = false;
    _lock_acquire(rope_cons_lock);
    _lock_acquire(q_prod_lock);

    for (int i = 0; i < DQ / DH; i++) {
        is_qkv_ping = !is_qkv_ping;
        bf16* qkv_using = is_qkv_ping ? qkv_ping : qkv_pong;
        _lock_acquire(qkv_cons_lock);
        _rotate(q + i * DH, qkv_using, rope_w, rope_w + DH);
        _lock_release(qkv_prod_lock);
    }

    _lock_release(q_cons_lock);

    bf16* k_ptr = k;
    bf16* v_ptr = v;

    _lock_acquire(k_prod_lock, 1);
    _lock_acquire(v_prod_lock, 1);
    is_qkv_ping = !is_qkv_ping;
    bf16* qkv_using = is_qkv_ping ? qkv_ping : qkv_pong;
    _lock_acquire(qkv_cons_lock);
    v_norm(v_ptr, qkv_using);
    v_ptr += DH;
    _rotate(k_ptr, qkv_using, rope_w, rope_w + DH + DH);
    k_ptr += DH;
    _lock_release(qkv_prod_lock);
    _lock_release(k_cons_lock, 1);
    _lock_release(v_cons_lock, 1);

    _lock_release(rope_prod_lock);
}


}