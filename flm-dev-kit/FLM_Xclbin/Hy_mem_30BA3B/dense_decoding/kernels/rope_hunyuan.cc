// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include "aie_array_layout.h"
#include "aie_kernel_utils.h"
#include "typedef.hpp"

/// Hunyuan applies the per-head Q/K RMS norms *after* RoPE, not before.
///
/// Every other QK-norm model in the tree (Qwen3) normalizes first, which is what
/// rope.cc implements.  RMSNorm is scale-only and RoPE is a rotation, so the two
/// orders are only equivalent when the norm weight is all ones -- they are not
/// interchangeable here (HunYuan applies the per-head Q/K RMS norm after RoPE).
///
/// The consequence for the cache is unchanged from rope.cc: the K this kernel
/// hands to IT3 is post-RoPE *and* post-norm, so nothing is re-applied to
/// replayed keys.
///
/// Everything else -- the half-rotation, the zero fill of the padded Q heads,
/// the V passthrough, the lock protocol -- is identical to rope.cc.

// kernels/*.cc is a wildcard rule, so this file is compiled for every model.
// It only makes sense for a model that has the norms at all; for anything else
// it collapses to an empty translation unit and is never linked.
#ifdef HAS_QK_NORM

///@brief RMS norm over one head vector, in place: x = w * x / rms(x)
///@param x  head vector of DH elements, overwritten
///@param w  per-head-dim norm weight, DH elements
///@note Body in aie_kernel_utils.h, shared with rms_residual.cc's rms_norm.
///      A head vector is small enough to stay in registers across both passes.
void qk_norm_hunyuan(
    const bf16 *restrict x,
    const bf16 *restrict w
){
    _rms_norm_inplace<DH>(const_cast<bf16*>(x), w, 1e-5f);
}

///@brief Half-rotation RoPE (GPT-NeoX / Llama style), y = rotate(x)
///@note: C = A cos - B sin, D = A sin + B cos
void apply_rope_hunyuan(
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
        // Peano: use msc instead of aie::neg + mac, see rope.cc.
#if defined(__chess__)
        aie::vector<bf16, vector_size> neg_sin_vec = aie::neg(sin_vec);
        aie::accum<accfloat, vector_size> C = aie::mul(x1_vec, cos_vec);
        C = aie::mac(C, x2_vec, neg_sin_vec);
#else
        aie::accum<accfloat, vector_size> C = aie::mul(x1_vec, cos_vec);
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

///@brief Split the K|V half of the fused QKV row into K and V, RoPE-ing then
///       QK-norming K on the way.
void pseduo_rope_hunyuan_kv(
    bf16 *restrict k,
    bf16 *restrict v,
    bf16 * restrict kv,
    bf16 * restrict rope_w
) {
    bf16 *it_k = k;
    bf16 *it_v = v;
    bf16 *it_kv = kv;
    bf16 *it_rope = rope_w;
    bf16 *k_norm_weight = rope_w + DH + DH;
    const int DH_2 = DH / 2;

    bf16* cos_val = it_rope;
    bf16* sin_val = it_rope + DH_2;

    for (int h = 0; h < NUM_KV_HEADS; h++){
        // RoPE first, then norm the rotated head in place -- the deviation.
        apply_rope_hunyuan(it_k, it_kv, cos_val, sin_val);
        qk_norm_hunyuan(it_k, k_norm_weight);
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

///@brief RoPE then QK-norm the Q half of the fused QKV row, zero filling the
///       padded Q heads.
void pseduo_rope_hunyuan_q(
    bf16 *restrict q,
    bf16 * restrict q_in,
    bf16 * restrict rope_w
) {
    bf16 *it_q = q;
    bf16 *it_q_in = q_in;
    bf16 *it_rope = rope_w;
    bf16 *q_norm_weight = rope_w + DH;
    const int DH_2 = DH / 2;

    bf16* cos_val = it_rope;
    bf16* sin_val = it_rope + DH_2;

    for (int h = 0; h < NUM_KV_HEADS; h++){
        for (int g = 0; g < Q_HEADS_PER_GROUP; g++){
            // RoPE first, then norm the rotated head in place -- the deviation.
            apply_rope_hunyuan(it_q, it_q_in, cos_val, sin_val);
            qk_norm_hunyuan(it_q, q_norm_weight);
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

/// K|V arrives on the QKV input first, then Q -- see rope.cc.
void rope_hunyuan(
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
    pseduo_rope_hunyuan_kv(k, v, kv_in, rope_w);
    _lock_release(qkv_prod_lock);
    _lock_release(v_cons_lock, 2);
    _lock_release(k_cons_lock, 2);

    _lock_acquire(qkv_cons_lock);
    _lock_acquire(q_prod_lock);
    pseduo_rope_hunyuan_q(q, q_in, rope_w);
    _lock_release(q_cons_lock);
    _lock_release(qkv_prod_lock);
    _lock_release(rope_prod_lock);
}


}

#endif // HAS_QK_NORM
