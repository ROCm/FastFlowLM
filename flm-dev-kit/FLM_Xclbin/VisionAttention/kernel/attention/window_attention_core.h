// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __WINDOW_ATTENTION_CORE_H__
#define __WINDOW_ATTENTION_CORE_H__

#include "attention_common.h"
#include "window_attention_common.h"

#include <stdint.h>
#include <stdlib.h>

#include <aie_api/aie.hpp>


#include "../passThroughCommon.hpp"

#include "../common_external.hpp"
#include "../lock_util.hpp"



constexpr float csqrt(float x)
{
    float r = x;
    for (int i = 0; i < 10; ++i)
        r = 0.5f * (r + x / r);
    return r;
}

// TODO: sync the lock defintion with the python implementation
constexpr int lq_internal_prod_lock = 0;
constexpr int lq_internal_con_lock = 1;

constexpr int lk_internal_prod_lock = 2;
constexpr int lk_internal_con_lock = 3;

constexpr int lv_internal_prod_lock = 4;
constexpr int lv_internal_con_lock = 5;

constexpr int o_prod_lock = 6;
constexpr int o_con_lock = 7;

constexpr int S_DEBUG_prod_lock = 11;
constexpr int S_DEBUG_con_lock = 12;




// no flash attention, just pure attention
template <int DH, int LQ, int LK, int r, int s, int t, bool use_low_precision, bool use_casual_mask>
void window_attention_bf16(
    bfloat16 *q_buffer_ping, bfloat16* q_buffer_pong,// size of LQxDH,
    bfloat16 *k_buffer_ping,  // size of LKxDH,
    bfloat16 *v_buffer_ping, // size of LVxDH
    bfloat16 *S_pij_buffer,    // size of LQ * LK,
    bfloat16 *output_buffer,       // size of LQ * DH
    int32_t *ping_pong_buffer
){

    constexpr bfloat16 attention_scale = 1.0f / csqrt(static_cast<float>(DH)); // 1/sqrt(DH)

    alignas(aie::vector_decl_align) bfloat16 m_i[LQ];
    alignas(aie::vector_decl_align) bfloat16 m_i_new[LQ];

    alignas(aie::vector_decl_align) bfloat16 l_i[LQ];

    constexpr int vec_size = 32;
    aie::vector<bfloat16, vec_size> neg_inf_vec = aie::broadcast<bfloat16, vec_size>(NEG_INF_FP32);

    event0();

    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_RANGE(LQ / vec_size, LQ / vec_size)
    for(int l = 0; l < LQ; l +=vec_size){
        aie::store_v(m_i + l, neg_inf_vec);
        aie::store_v(l_i + l, aie::zeros<bfloat16, vec_size>());
    }


    bfloat16* cur_q_buffer = (bfloat16*)retrive_ping_pong_buffer(
        q_buffer_ping,
        q_buffer_pong,
        ping_pong_buffer
    );


    // // first, acquire q buffer
    _lock_acquire(lq_internal_con_lock, 1);
    _lock_acquire(lk_internal_con_lock, 1);        
    // do the q*scale first
    event0();
    mul_scalar<LQ, DH>(
        cur_q_buffer,
        attention_scale);
    event0();


    // do the matmul q*k^T to get S_O buffer
    event1();
    window_attention_matmul_q_k_transpose_mmul<bfloat16, bfloat16, LQ, DH, LK,
    r,s,t>(
        cur_q_buffer,
        k_buffer_ping,
        S_pij_buffer
    );
    event1();
    
    // release q, k buffer locks
    _lock_release(lq_internal_prod_lock, 1);            
    _lock_release(lk_internal_prod_lock, 1);

    event0();
    // find the max per row and do the softmax normalization
    find_m_new_using_gt_and_reduce_max<LQ, LK, r, t, use_casual_mask>(
        S_pij_buffer,
        m_i,
        m_i_new
    );
    event0();

    

    
    event1();
    if constexpr(use_low_precision){
        // now do the softmax normalization 
        calculate_Pij_unnormalize_and_sum_pij_low_precision<LQ, LK, r, t>(
                    S_pij_buffer,
                    m_i_new,
                    l_i);
    }else{

        // now do the softmax normalization 
        calculate_Pij_unnormalize_and_sum_pij<LQ, LK, r, t>(
                    S_pij_buffer,
                    m_i_new,
                    l_i);

    }

    event1();
    
    event0();
    pij_normalize<LQ, LK, r, t>(
        S_pij_buffer,
        l_i
    );       
    event0();         

    _lock_acquire(o_prod_lock, 1);
    _lock_acquire(lv_internal_con_lock, 1);
    // now do the matmul pij * v to get O_buffer
    event1();
    window_attention_matmul_pij_v<bfloat16, bfloat16, 
    LQ, DH, LK, r, s, t>(
        S_pij_buffer,
        v_buffer_ping,
        output_buffer
    );
    event1();
    
    _lock_release(o_con_lock, 1);
    _lock_release(lv_internal_prod_lock, 1);

    
    event1();
    

}








#endif