// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include "aie_array_layout.h"
#include "aie_kernel_utils.h"
#include "typedef.hpp"

#ifdef HAS_QK_NORM
///@brief pseduo rms norm: y = x * w
///@param y: output
///@param x_buf: copy of x
///@param x: input
///@param w: weight
///@param overwrite_x_buf: if true, overwrite x_buf with x
///@note Body in aie_kernel_utils.h, shared with rms_residual.cc's rms_norm.
///      A head vector is small enough to stay in registers across both passes.
void qk_norm(
    const bf16 *restrict x,
    const bf16 *restrict w
){
    _rms_norm_inplace<DH>(const_cast<bf16*>(x), w, 1e-6f);
}
#endif

void apply_rope(
    bf16 *restrict y,
    bf16 *restrict x,
    bf16 *restrict cos_val,
    bf16 *restrict sin_val
) {
    constexpr int vector_size = _fit_lanes<DH / 2, AIE_BF16_LANES>();
    const int DH_2 = DH / 2;
    constexpr int F = DH_2 / vector_size;
    bf16 *it_y_p1 = y;
    bf16 *it_y_p2 = y + DH_2;
    bf16 *it_x_p1 = x;
    bf16 *it_x_p2 = x + DH_2;
    bf16 *it_cos = cos_val;
    bf16 *it_sin = sin_val;
    
    AIE_PREPARE_FOR_PIPELINING AIE_LOOP_RANGE(8)
    for (int i = 0; i < F; i++) {
        aie::vector<bf16, vector_size> x1_vec = aie::load_v<vector_size>(it_x_p1);
        aie::vector<bf16, vector_size> x2_vec = aie::load_v<vector_size>(it_x_p2);
        aie::vector<bf16, vector_size> cos_vec = aie::load_v<vector_size>(it_cos);
        aie::vector<bf16, vector_size> sin_vec = aie::load_v<vector_size>(it_sin);
        aie::accum<accfloat, vector_size> C = aie::mul(x1_vec, cos_vec);
        // Under Peano, aie::neg on a bf16 vector (G_FNEG) produces code that
        // stalls the rope core, and decode hangs. msc computes C - x2*sin
        // directly: the same math without the negate.
#if defined(__chess__)
        aie::vector<bf16, vector_size> neg_sin_vec = aie::neg(sin_vec);
        C = aie::mac(C, x2_vec, neg_sin_vec);
#else
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

///@brief Split the K|V half of the fused QKV row into K and V, RoPE-ing (and
///       optionally QK-norming) K on the way.
///@note: C = A cos - B sin, D = A sin + B cos
void pseduo_rope_kv(
    bf16 *restrict k,
    bf16 *restrict v,
    bf16 * restrict kv,
    bf16 * restrict rope_w
) {
    bf16 *it_k = k;
    bf16 *it_v = v;
    bf16 *it_kv = kv;
    bf16 *it_rope = rope_w;
    #ifdef HAS_QK_NORM
    bf16 *k_norm_weight = rope_w + DH + DH;
    #endif
    const int DH_2 = DH / 2;

    bf16* cos_val = it_rope;
    bf16* sin_val = it_rope + DH_2;

    for (int h = 0; h < NUM_KV_HEADS; h++){
        #ifdef HAS_QK_NORM
        qk_norm(it_kv, k_norm_weight);
        #endif
        apply_rope(it_k, it_kv, cos_val, sin_val);
        it_k += DH;
        it_kv += DH;
    }

    // move v from kv to v, N
    for (int i = 0; i < DV / 32; i++) {
        aie::vector<bf16, 32> v_vec = aie::load_v<32>(it_kv);
        aie::store_v(it_v, v_vec);
        it_v += 32;
        it_kv += 32;
    }
}

///@brief RoPE (and optionally QK-norm) the Q half of the fused QKV row, zero
///       filling the padded Q heads.
///@note: C = A cos - B sin, D = A sin + B cos
void pseduo_rope_q(
    bf16 *restrict q,
    bf16 * restrict q_in,
    bf16 * restrict rope_w
) {
    bf16 *it_q = q;
    bf16 *it_q_in = q_in;
    bf16 *it_rope = rope_w;
    #ifdef HAS_QK_NORM
    bf16 *q_norm_weight = rope_w + DH;
    #endif
    const int DH_2 = DH / 2;

    bf16* cos_val = it_rope;
    bf16* sin_val = it_rope + DH_2;

    for (int h = 0; h < NUM_KV_HEADS; h++){
        for (int g = 0; g < Q_HEADS_PER_GROUP; g++){
            #ifdef HAS_QK_NORM
            qk_norm(it_q_in, q_norm_weight);
            #endif
            apply_rope(it_q, it_q_in, cos_val, sin_val);
            it_q += DH;
            it_q_in += DH;
        }
        aie::vector<bf16, DH_2> zero = aie::zeros<bf16, DH_2>();
        for (int i = 0; i < ATTN_GROUPS_PADDING; i++){
            aie::store_v(it_q, zero);
            it_q += DH_2;
            aie::store_v(it_q, zero);
            it_q += DH_2;
        }
    }
}

extern "C" {

/// K|V arrives on the QKV input first, then Q -- the two chunks alternate
/// between kv_in and q_in on the same S2MM channel.  Releasing K and V before
/// touching Q lets the KV cache write overlap the Q rotation.
void rope(
    bf16* restrict q,
    bf16* restrict k,
    bf16* restrict v,
    bf16* restrict q_in,
    bf16* restrict kv_in,
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
    _lock_acquire(qkv_cons_lock);
    _lock_acquire(rope_cons_lock);
    _lock_acquire(k_prod_lock, 2);
    _lock_acquire(v_prod_lock, 2);
    pseduo_rope_kv(k, v, kv_in, rope_w);
    _lock_release(qkv_prod_lock);
    _lock_release(v_cons_lock, 2);
    _lock_release(k_cons_lock, 2);

    _lock_acquire(qkv_cons_lock);
    _lock_acquire(q_prod_lock);
    pseduo_rope_q(q, q_in, rope_w);
    _lock_release(q_cons_lock);
    _lock_release(qkv_prod_lock);
    _lock_release(rope_prod_lock);
}


}
