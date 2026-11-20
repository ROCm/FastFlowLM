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
    bf16 *it_x = const_cast<bf16*>(x);
    bf16 *it_w = const_cast<bf16*>(w);
    const float epsilon = 1e-6;

    constexpr int vector_size = 16;
    constexpr float one_over_D = 1.0f / (float)DH;

    aie::accum<accfloat, vector_size> sum_squares = aie::zeros<accfloat>();
    aie::vector<float, vector_size> mx_vec;

    const int F = DH / vector_size;
    for (int i = 0; i < F; i++)
    chess_prepare_for_pipelining chess_loop_range(16, ) {
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

    for (int i = 0; i < F; i++)
    chess_prepare_for_pipelining chess_loop_range(16, ) {
        aie::vector<bf16, vector_size> x_vec = aie::load_v<vector_size>(it_x);
        aie::vector<bf16, vector_size> w_vec = aie::load_v<vector_size>(it_w);
        aie::vector<float, vector_size> wx_vec = aie::mul(x_vec, w_vec);
        aie::vector<bf16, vector_size> o_vec = aie::mul(wx_vec, divrms);
        aie::store_v(it_x, o_vec);
        it_x += vector_size;
        it_w += vector_size;
    }
}


void v_norm(
    const bf16 *restrict y,
    const bf16 *restrict x
){
    bf16 *it_x = const_cast<bf16*>(x);
    bf16 *it_y = const_cast<bf16*>(y);
    const float epsilon = 1e-6;

    constexpr int vector_size = 16;
    constexpr float one_over_D = 1.0f / (float)DH;

    aie::accum<accfloat, vector_size> sum_squares = aie::zeros<accfloat>();
    aie::vector<float, vector_size> mx_vec;

    const int F = DH / vector_size;
    for (int i = 0; i < F; i++)
    chess_prepare_for_pipelining chess_loop_range(16, ) {
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

    for (int i = 0; i < F; i++)
    chess_prepare_for_pipelining chess_loop_range(16, ) {
        aie::vector<bf16, vector_size> x_vec = aie::load_v<vector_size>(it_x);
        aie::accum<accfloat, vector_size> x_acc;
        x_acc.from_vector(x_vec);
        aie::vector<float, vector_size> x_float = x_acc.template to_vector<float>();
        aie::vector<bf16, vector_size> o_vec = aie::mul(x_float, divrms);
        aie::store_v(it_y, o_vec);
        it_x += vector_size;
        it_y += vector_size;
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
        aie::vector<bf16, vector_size> neg_sin_vec = aie::neg(sin_vec);
        aie::accum<accfloat, vector_size> C = aie::mul(x1_vec, cos_vec);
        C = aie::mac(C, x2_vec, neg_sin_vec);
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
    bf16* restrict rope_w,
    int* SKIP_KV
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

    if (SKIP_KV[0] == 0) {
        _lock_acquire(k_prod_lock, 1);
        for (int i = 0; i < DK / DH; i++) {
            is_qkv_ping = !is_qkv_ping;
            bf16* qkv_using = is_qkv_ping ? qkv_ping : qkv_pong;
            _lock_acquire(qkv_cons_lock);
            _rotate(k_ptr, qkv_using, rope_w, rope_w + DH + DH);
            k_ptr += DH;
            _lock_release(qkv_prod_lock);
        }
        _lock_release(k_cons_lock, 1);

        _lock_acquire(v_prod_lock, 1);
        for (int i = 0; i < DV / DH; i++) {
            is_qkv_ping = !is_qkv_ping;
            bf16* qkv_using = is_qkv_ping ? qkv_ping : qkv_pong;
            _lock_acquire(qkv_cons_lock);
            v_norm(v + i * DH, qkv_using);
            v_ptr += DH;
            _lock_release(qkv_prod_lock);
        }
        _lock_release(v_cons_lock, 1);
    }

    _lock_release(rope_prod_lock);
}


}