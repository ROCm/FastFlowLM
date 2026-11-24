// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include "aie_array_layout.h"
#include "aie_kernel_utils.h"
#include "typedef.hpp"

/// Phi 4 is special RoPE implementation
/// In Phi4, the rope is only applied to 3/4 part of q, k heads
///  DH = 128, then ROPE is applied to 96 dim of q, k
///  the rest 32 dim is passed through directly

///@brief Split the K|V half of the fused QKV row into K and V, RoPE-ing the
///       leading 3/4 of every K head and passing the tail through.
///@note: C = A cos - B sin, D = A sin + B cos
void pseduo_rope_phi4_kv(
    bf16 *restrict k,
    bf16 *restrict v,
    bf16 * restrict kv,
    bf16 * restrict rope_w
) {
    static constexpr int DH_2 = DH / 2;
    static constexpr int vec_factor = DH / 4 / 2; // 2 is for making sure even number of rope segments
    static constexpr int segments = DH / vec_factor;
    static constexpr int rope_segments = segments * 3 / 4; // 3/4 of DH
    static constexpr int pass_segments = segments - rope_segments; // 1/4 of DH
    static constexpr int rope_offset = rope_segments * vec_factor / 2;
    bf16 *it_k = k;
    bf16 *it_v = v;
    bf16 *it_kv = kv;
    bf16 *cos = rope_w;
    bf16 *sin = rope_w + DH_2;

    for (int h = 0; h < NUM_KV_HEADS; h++){
        for (int s = 0; s < rope_segments / 2; s++) {
            aie::vector<bf16, vec_factor> k_p0 = aie::load_v<vec_factor>(it_kv);
            aie::vector<bf16, vec_factor> k_p1 = aie::load_v<vec_factor>(it_kv + rope_offset);
            it_kv += vec_factor;
            aie::vector<bf16, vec_factor> cos_vec = aie::load_v<vec_factor>(cos + s * vec_factor);
            aie::vector<bf16, vec_factor> sin_vec = aie::load_v<vec_factor>(sin + s * vec_factor);
            aie::vector<bf16, vec_factor> neg_sin_vec = aie::neg(sin_vec);


            aie::accum<accfloat, vec_factor> C = aie::mul(k_p0, cos_vec);
            C = aie::mac(C, k_p1, neg_sin_vec);
            aie::accum<accfloat, vec_factor> D = aie::mul(k_p0, sin_vec);
            D = aie::mac(D, k_p1, cos_vec);
            aie::store_v(it_k, C.template to_vector<bf16>());
            aie::store_v(it_k + rope_offset, D.template to_vector<bf16>());
            it_k += vec_factor;
        }
        it_kv += rope_offset;
        it_k += rope_offset;
        for (int s = 0; s < pass_segments; s++) {
            aie::vector<bf16, vec_factor> k_p = aie::load_v<vec_factor>(it_kv);
            it_kv += vec_factor;
            aie::store_v(it_k, k_p);
            it_k += vec_factor;
        }
    }

    // move v from kv to v, N
    for (int i = 0; i < DV / 32; i++) {
        aie::vector<bf16, 32> v_vec = aie::load_v<32>(it_kv);
        aie::store_v(it_v, v_vec);
        it_v += 32;
        it_kv += 32;
    }
}

///@brief RoPE the leading 3/4 of every Q head, pass the tail through and zero
///       fill the padded Q heads.
///@note: C = A cos - B sin, D = A sin + B cos
void pseduo_rope_phi4_q(
    bf16 *restrict q,
    bf16 * restrict q_in,
    bf16 * restrict rope_w
) {
    static constexpr int DH_2 = DH / 2;
    static constexpr int vec_factor = DH / 4 / 2; // 2 is for making sure even number of rope segments
    static constexpr int segments = DH / vec_factor;
    static constexpr int rope_segments = segments * 3 / 4; // 3/4 of DH
    static constexpr int pass_segments = segments - rope_segments; // 1/4 of DH
    static constexpr int rope_offset = rope_segments * vec_factor / 2;
    bf16 *it_q = q;
    bf16 *it_q_in = q_in;
    bf16 *cos = rope_w;
    bf16 *sin = rope_w + DH_2;

    for (int h = 0; h < NUM_KV_HEADS; h++){
        for (int g = 0; g < Q_HEADS_PER_GROUP; g++){
            for (int s = 0; s < rope_segments / 2; s++) {
                aie::vector<bf16, vec_factor> q_p0 = aie::load_v<vec_factor>(it_q_in);
                aie::vector<bf16, vec_factor> q_p1 = aie::load_v<vec_factor>(it_q_in + rope_offset);
                it_q_in += vec_factor;
                aie::vector<bf16, vec_factor> cos_vec = aie::load_v<vec_factor>(cos + s * vec_factor);
                aie::vector<bf16, vec_factor> sin_vec = aie::load_v<vec_factor>(sin + s * vec_factor);
                aie::vector<bf16, vec_factor> neg_sin_vec = aie::neg(sin_vec);

                aie::accum<accfloat, vec_factor> C = aie::mul(q_p0, cos_vec);
                C = aie::mac(C, q_p1, neg_sin_vec);
                aie::accum<accfloat, vec_factor> D = aie::mul(q_p0, sin_vec);
                D = aie::mac(D, q_p1, cos_vec);
                aie::store_v(it_q, C.template to_vector<bf16>());
                aie::store_v(it_q + rope_offset, D.template to_vector<bf16>());
                it_q += vec_factor;
            }
            it_q_in += rope_offset;
            it_q += rope_offset;
            for (int s = 0; s < pass_segments; s++) {
                aie::vector<bf16, vec_factor> q_p = aie::load_v<vec_factor>(it_q_in);
                it_q_in += vec_factor;
                aie::store_v(it_q, q_p);
                it_q += vec_factor;
            }
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
void rope_phi4(
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
    pseduo_rope_phi4_kv(k, v, kv_in, rope_w);
    _lock_release(qkv_prod_lock);
    _lock_release(v_cons_lock, 2);
    _lock_release(k_cons_lock, 2);

    _lock_acquire(qkv_cons_lock);
    _lock_acquire(q_prod_lock);
    pseduo_rope_phi4_q(q, q_in, rope_w);
    _lock_release(q_cons_lock);
    _lock_release(qkv_prod_lock);
    _lock_release(rope_prod_lock);
}


}
