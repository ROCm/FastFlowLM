// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include "common.h"
#include "gptossParam.hpp"
#include "passThroughCommon.hpp"
#include "aie_kernel_utils.h"

///@brief pseduo RoPE: q = q, k = k + 1, v = v - 1
///@param y: output
///@param x: input
///@param w: weight
///@note: C = A cos - B sin, D = A sin + B cos
void pseduo_rope(
    bf16 *restrict q,
    bf16 *restrict k,
    bf16 *restrict v,
    bf16 * restrict qkv,
    bf16 * restrict rope_w
) {
    bf16 *it_q = q;
    bf16 *it_k = k;
    bf16 *it_v = v;
    bf16 *it_qkv = qkv;
    bf16 *it_rope = rope_w;
    const int DH_2 = DH / 2;
    // move q from qkv to q, N
    aie::vector<bf16, DH_2> cos_vec = aie::load_v<DH_2>(it_rope);
    it_rope += DH_2;
    aie::vector<bf16, DH_2> sin_vec = aie::load_v<DH_2>(it_rope);
    // Peano cannot legalize a vector FNEG on <DH_2 x bf16>; instead of forming
    // neg_sin_vec and using mac, use msc (acc - a*b) directly below (identical
    // numerics). chess keeps the neg + mac form.
#if defined(__chess__)
    aie::vector<bf16, DH_2> neg_sin_vec = aie::neg(sin_vec);
#endif
    it_rope += DH_2;

    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_RANGE(NUM_KV_HEADS, NUM_KV_HEADS)
    for (int h = 0; h < NUM_KV_HEADS; h++){
        AIE_PREPARE_FOR_PIPELINING
        AIE_LOOP_RANGE(ATTN_GROUPS,ATTN_GROUPS)
        for (int g = 0; g < ATTN_GROUPS; g++){
            aie::vector<bf16, DH_2> q_p0 = aie::load_v<DH_2>(it_qkv);
            it_qkv += DH_2;
            aie::vector<bf16, DH_2> q_p1 = aie::load_v<DH_2>(it_qkv);
            it_qkv += DH_2;

            aie::accum<accfloat, DH_2> C = aie::mul(q_p0, cos_vec);
#if defined(__chess__)
            C = aie::mac(C, q_p1, neg_sin_vec);
#else
            C = aie::msc(C, q_p1, sin_vec);
#endif
            aie::accum<accfloat, DH_2> D = aie::mul(q_p0, sin_vec);
            D = aie::mac(D, q_p1, cos_vec);
            aie::store_v(it_q, C.template to_vector<bf16>());
            it_q += DH_2;
            aie::store_v(it_q, D.template to_vector<bf16>());
            it_q += DH_2;
        }

    }

    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_RANGE(NUM_KV_HEADS, NUM_KV_HEADS)
    for (int h = 0; h < NUM_KV_HEADS; h++){
        aie::vector<bf16, DH_2> k_p0 = aie::load_v<DH_2>(it_qkv);
        it_qkv += DH_2;
        aie::vector<bf16, DH_2> k_p1 = aie::load_v<DH_2>(it_qkv);
        it_qkv += DH_2;

        aie::accum<accfloat, DH_2> C = aie::mul(k_p0, cos_vec);
#if defined(__chess__)
        C = aie::mac(C, k_p1, neg_sin_vec);
#else
        C = aie::msc(C, k_p1, sin_vec);
#endif
        aie::accum<accfloat, DH_2> D = aie::mul(k_p0, sin_vec);
        D = aie::mac(D, k_p1, cos_vec);
        aie::store_v(it_k, C.template to_vector<bf16>());
        it_k += DH_2;
        aie::store_v(it_k, D.template to_vector<bf16>());
        it_k += DH_2;
    }

    // move v from qkv to v, N
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_RANGE(GPTOSS_DV / 32, GPTOSS_DV / 32)
    for (int i = 0; i < GPTOSS_DV / 32; i++) {
        aie::vector<bf16, 32> v_vec = aie::load_v<32>(it_qkv);
        aie::store_v(it_v, v_vec);
        it_v += 32;
        it_qkv += 32;
    }
}


void add_q_k_v_bias( 
    const bf16*  q_k_v_bias_in,
    bf16*  qkv
){
    constexpr uint32_t vec_len = 16;

    static_assert( (GPTOSS_DQ + GPTOSS_DK +GPTOSS_DV)%vec_len == 0  );
    const bfloat16* bias_ptr = q_k_v_bias_in;

    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_RANGE((GPTOSS_DQ + GPTOSS_DK +GPTOSS_DV)/vec_len,   (GPTOSS_DQ + GPTOSS_DK +GPTOSS_DV)/vec_len )
    for(int i = 0; i < GPTOSS_DQ + GPTOSS_DK +GPTOSS_DV; i+= vec_len){
        
        aie::vector<bfloat16, vec_len> bias_vec = aie::load_v<vec_len>(bias_ptr );
        aie::vector<bfloat16, vec_len> val_vec = aie::load_v<vec_len>( qkv);
        aie::vector<bfloat16, vec_len> res = aie::add(bias_vec, val_vec);
        aie::store_v(qkv, res);

        bias_ptr+= vec_len;
        qkv+= vec_len;

    }
}

extern "C" {

void rope(
    bf16* restrict q,
    bf16* restrict k,
    bf16* restrict v,
    bf16* restrict qkv,
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
    _lock_acquire(q_prod_lock);
    _lock_acquire(k_prod_lock, 2);
    _lock_acquire(v_prod_lock, 2);

    // first add bias to q, k, v
    add_q_k_v_bias( rope_w,qkv );
    // testing purpose, just flush out q, k, v

    pseduo_rope(q, k, v, qkv, rope_w+GPTOSS_DQ+GPTOSS_DK+GPTOSS_DV+DH);
    //memcpy(q+GPTOSS_DQ,  rope_w+GPTOSS_DQ+GPTOSS_DK+GPTOSS_DV, DH*sizeof(bf16)); // append the sink values directly to the end of q buffer
    copy_bf16_to_bf16<DH>(q+GPTOSS_DQ,  rope_w+GPTOSS_DQ+GPTOSS_DK+GPTOSS_DV );
    _lock_release(rope_prod_lock);
    _lock_release(qkv_prod_lock);
    _lock_release(q_cons_lock);
    _lock_release(v_cons_lock, 2);
    _lock_release(k_cons_lock, 2);

}


}