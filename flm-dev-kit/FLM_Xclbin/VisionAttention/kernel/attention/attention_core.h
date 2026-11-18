// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __ATTENTION_8_8_8_CORE_H__
#define __ATTENTION_8_8_8_CORE_H__

#include "attention_common.h"

// TODO: sync the lock defintion with the python implementation
constexpr int lq_internal_prod_lock = 0;
constexpr int lq_internal_con_lock = 1;

constexpr int lk_lv_internal_prod_lock = 2;
constexpr int lk_lv_internal_con_lock = 3;

constexpr int o_prod_lock = 4;
constexpr int o_con_lock = 5;

constexpr int S_DEBUG_prod_lock = 11;
constexpr int S_DEBUG_con_lock = 12;

constexpr float csqrt(float x)
{
    float r = x;
    for (int i = 0; i < 10; ++i)
        r = 0.5f * (r + x / r);
    return r;
}

constexpr int ceil_div_int(int a, int b)
{
    return (a + b - 1) / b;
}
constexpr int min(int a, int b)
{
    return (a < b) ? a : b;
}

// version that has q_repeat for attention
template <
    unsigned LQ_internal,
    unsigned LK_V_internal,
    unsigned DH,
    unsigned LQ,
    unsigned LK,
    unsigned r,
    unsigned s,
    unsigned t,
    bool LOW_PRECISION>
void core_attention_flow_q_repeat(

    bfloat16 *q_buffer,        // size of LQ_internal * DH
    bfloat16 *k_v_buffer_ping, // size of LK_V_internal * DH
    bfloat16 *k_v_buffer_pong, // size of LK_V_internal * DH
    bfloat16 *S_pij_buffer,    // size of LQ_internal * LK,
    float *O_buffer,           // size of LQ * DH
    int32_t *rtp_buffer,
    int32_t *ping_pong_buffer

)
{

    static_assert(LQ % LQ_internal == 0);
    static_assert(LQ / LQ_internal > 1); // This is the version with q repeats, so


    alignas(aie::vector_decl_align) bfloat16 m_i[LQ];
    alignas(aie::vector_decl_align) bfloat16 m_i_new[LQ];
    alignas(aie::vector_decl_align) bfloat16 exp_old_vec[LQ_internal];

    alignas(aie::vector_decl_align) float l_i[LQ];

    constexpr int vec_size =  min(32, LQ_internal);

    int32_t S_SEQ = *(rtp_buffer );  // shimtile S_SEQ
    int32_t L_SEQ_PER_CT = *(rtp_buffer + 1);  // shimtile L_SEQ
    
    float* attention_scalar_ptr = (float*)(rtp_buffer + 2); // pre-computed attention scalar, to avoid repeated computation of 1/sqrt(DH) in the case of q repeat
    float attention_scale_fp32 = *attention_scalar_ptr; // ideally is 1/DH, but model such as gemma4 is 1.0, so use rtp for flexibility.
    bfloat16 attention_scale = (bfloat16)attention_scale_fp32;



    int32_t *k_ping_pong_flag_ptr = ping_pong_buffer; // ping pong flag for q buffer

    for (int i = 0; i < L_SEQ_PER_CT; i += LQ)
    //while(true)
    {

        // NOTE: initialize m_i, l_i every LQ_internal_block
        // for now, assume LQ_internal == LQ

        static_assert(LQ_internal % vec_size == 0); // for better load/store
        aie::vector<bfloat16, vec_size> neg_inf_vec = aie::broadcast<bfloat16, vec_size>(NEG_INF_FP32);
        for (int l = 0; l < LQ; l += vec_size)
        {
            aie::store_v(m_i + l, neg_inf_vec);
            aie::store_v(l_i+l, aie::zeros<float, vec_size>());
        }




        int S_seq_ceil_div_LK = ceil_div_int(S_SEQ, LK);
        int S_seq_mod_LK = S_SEQ % LK;

        _lock_acquire(o_prod_lock, 1); // NOTE: order matter
        zero_vectorized_512<float, LQ, DH>(O_buffer);

        for (int j = 0; j < S_seq_ceil_div_LK * LK; j += LK)
        {
            for (int lq_idx = 0; lq_idx < LQ; lq_idx += LQ_internal)
            {
                _lock_acquire(lq_internal_con_lock, 1);
                event1();
                mul_scalar<LQ_internal, DH>(
                    q_buffer,
                    attention_scale);
                event1();
                event0();
                // for performance reason
                matmul_q_k_transpose_mmul_modify_buffer<bfloat16, bfloat16,
                                                        LQ_internal, DH, LK, r, s, t, LK_V_internal>(
                    q_buffer,
                    k_v_buffer_ping,
                    k_v_buffer_pong,
                    S_pij_buffer,
                    k_ping_pong_flag_ptr,
                    lk_lv_internal_prod_lock,
                    lk_lv_internal_con_lock);
                event1();

                // NOTE: release early here on purpose
                _lock_release(lq_internal_prod_lock, 1);



                if(S_seq_mod_LK > 0 && j+LK >= S_SEQ){
                    //last block with padding
                    apply_padding_mask<LQ_internal, LK, r, t>(
                        S_pij_buffer,
                        S_seq_mod_LK);
                }


                // now find m_i_new
                event0();

                find_m_new_using_gt_and_reduce_max<LQ_internal, LK, r, t>(
                    S_pij_buffer,
                    m_i + lq_idx,
                    m_i_new + lq_idx);
                event1();
                event0();
                calculate_exp_difference<LQ_internal>(
                    m_i + lq_idx,
                    m_i_new + lq_idx,
                    exp_old_vec);
                event1();

                // update l_i
                // li = li * exp_old
                event0();
                vec_dot_fp32_intrinsic<LQ_internal, min(LQ_internal, 32)>(l_i + lq_idx, exp_old_vec, l_i + lq_idx);

                // li +=  sum_j S_ij_unnormalized
                if constexpr (LOW_PRECISION) {
                    calculate_Pij_unnormalize_and_sum_pij_low_precision<LQ_internal, LK, r, t>(
                        S_pij_buffer,
                        m_i_new + lq_idx,
                        l_i + lq_idx);
                } else {
                    calculate_Pij_unnormalize_and_sum_pij<LQ_internal, LK, r, t>(
                        S_pij_buffer,
                        m_i_new + lq_idx,
                        l_i + lq_idx);
                }
                event1();

                static_assert(s == t);

                // NOTE: order mater, simiar for l_nw
                event0();
                calculate_Oij_mul_exp_old<LQ_internal, DH, r, t, min(LQ_internal, 32)>(
                    O_buffer + lq_idx * DH,
                    exp_old_vec);
                event1();

                event0();
                static_assert(LK_V_internal %t ==0);
                matmul_pij_vmmul_generic<bfloat16, float, LQ_internal, LK, DH, r, s, t, LK_V_internal/t>(
                    S_pij_buffer,
                    k_v_buffer_ping,
                    k_v_buffer_pong,
                    O_buffer + lq_idx * DH,
                    k_ping_pong_flag_ptr,
                    lk_lv_internal_prod_lock,
                    lk_lv_internal_con_lock);
            
                event1();

                copy_bf16_to_bf16<LQ_internal>(
                    m_i + lq_idx,
                    m_i_new + lq_idx);
            }
        }

        event1();
        O_normalize<LQ, DH, r, t>(
            O_buffer,
            l_i);
        event1();

        _lock_release(o_con_lock, 1);
    }
}

template <
    unsigned LQ_internal,
    unsigned LK_V_internal,
    unsigned DH,
    unsigned LQ,
    unsigned LK,
    unsigned r,
    unsigned s,
    unsigned t,
    bool LOW_PRECISION  >
void core_attention_flow_no_q_repeat(

    bfloat16 *q_buffer,        // size of LQ_internal * DH
    bfloat16 *k_v_buffer_ping, // size of LK_V_internal * DH
    bfloat16 *k_v_buffer_pong, // size of LK_V_internal * DH
    bfloat16 *S_pij_buffer,    // size of LQ_internal * LK,
    float *O_buffer,           // size of LQ * DH
    int32_t *rtp_buffer,
    int32_t *ping_pong_buffer

)
{
    static_assert(LQ % LQ_internal == 0);
    static_assert(LQ / LQ_internal == 1); // This is the version with no q repeats, so
    int32_t S_SEQ = *rtp_buffer;      // shimtile S_SEQ
    int32_t L_SEQ_PER_CT = *(rtp_buffer + 1);      // shimtile L_SEQ

    float* attention_scalar_ptr = (float*)(rtp_buffer + 2); // pre-computed attention scalar, to avoid repeated computation of 1/sqrt(DH) in the case of q repeat
    float attention_scale_fp32 = *attention_scalar_ptr; // ideally is 1/DH, but model such as gemma4 is 1.0, so use rtp for flexibility.
    bfloat16 attention_scale = (bfloat16)attention_scale_fp32;



    int32_t *k_ping_pong_flag_ptr = ping_pong_buffer; // ping pong flag for q buffer



    alignas(aie::vector_decl_align) bfloat16 m_i[LQ_internal];
    alignas(aie::vector_decl_align) bfloat16 m_i_new[LQ_internal];
    alignas(aie::vector_decl_align) bfloat16 exp_old_vec[LQ_internal];

    alignas(aie::vector_decl_align) float l_i[LQ_internal];

    constexpr int vec_size = min(32, LQ_internal);

    for (int i = 0; i < L_SEQ_PER_CT; i += LQ)
    // while(true)
    {

        // NOTE: initialize m_i, l_i every LQ_internal_block
        // for now, assume LQ_internal == LQ
        static_assert(LQ == LQ_internal); // for this implementation

        static_assert(LQ_internal % vec_size == 0); // for better load/store
        aie::vector<bfloat16, vec_size> neg_inf_vec = aie::broadcast<bfloat16, vec_size>(NEG_INF_FP32);
        for (int l = 0; l < LQ_internal; l += vec_size)
        {
            aie::store_v(m_i + l, neg_inf_vec);
            aie::store_v(l_i + l, aie::zeros<float, vec_size>());
        }

        _lock_acquire(lq_internal_con_lock, 1);

        // NOTE: s is zero out auto inside the matmul function
        //  note: for now


        event0();

        mul_scalar<LQ_internal, DH>(
            q_buffer,
            attention_scale);
        event1();
        
        int S_seq_ceil_div_LK = ceil_div_int(S_SEQ, LK);
        int S_seq_mod_LK = S_SEQ % LK;

        _lock_acquire(o_prod_lock, 1); // NOTE: order matter
        zero_vectorized_512<float, LQ_internal, DH>(O_buffer);

        for (int j = 0; j < S_seq_ceil_div_LK * LK; j += LK)
        {
            event0();

            // for performance reaso
            matmul_q_k_transpose_mmul_modify_buffer<bfloat16, bfloat16,
                                                    LQ_internal, DH, LK, r, s, t, LK_V_internal>(
                q_buffer,
                k_v_buffer_ping,
                k_v_buffer_pong,
                S_pij_buffer,
                k_ping_pong_flag_ptr,
                lk_lv_internal_prod_lock,
                lk_lv_internal_con_lock);
            event1();

            if(S_seq_mod_LK > 0 && j+LK >= S_SEQ){
                //last block with padding
                apply_padding_mask<LQ_internal, LK, r, t>(
                    S_pij_buffer,
                    S_seq_mod_LK);
            }


            // now find m_i_new
            event0();

            find_m_new_using_gt_and_reduce_max<LQ_internal, LK, r, t>(
                S_pij_buffer,
                m_i,
                m_i_new);
            event1();
            event0();
            calculate_exp_difference<LQ_internal>(
                m_i,
                m_i_new,
                exp_old_vec);
            event1();

            // update l_i
            // li = li * exp_old
            event0();
            vec_dot_fp32_intrinsic<LQ_internal, min(32, LQ_internal)>(l_i, exp_old_vec, l_i);

            // li +=  sum_j S_ij_unnormalized
            if constexpr(LOW_PRECISION) {
                calculate_Pij_unnormalize_and_sum_pij_low_precision<LQ_internal, LK, r, t>(
                    S_pij_buffer,
                    m_i_new,
                    l_i);
            } else {
                calculate_Pij_unnormalize_and_sum_pij<LQ_internal, LK, r, t>(
                    S_pij_buffer,
                    m_i_new,
                    l_i);
            }
            event1();

            static_assert(s == t);

            // NOTE: order mater, simiar for l_nw
            event0();
            calculate_Oij_mul_exp_old<LQ_internal, DH, r, t, min(LQ_internal, 32)>(
                O_buffer,
                exp_old_vec);
            event1();

            event0();
    
            static_assert(LK_V_internal %t ==0);
            matmul_pij_vmmul_generic<bfloat16, float, LQ_internal, LK, DH, r, s, t, LK_V_internal/t>(
                S_pij_buffer,
                k_v_buffer_ping,
                k_v_buffer_pong,
                O_buffer,
                k_ping_pong_flag_ptr,
                lk_lv_internal_prod_lock,
                lk_lv_internal_con_lock);
    
            event1();

            // now, update for next iteration

            // o_buffer is already updated
            // l_i buffer is also alread updated
            // update m_i = m_i_new
            copy_bf16_to_bf16<LQ_internal>(
                m_i,
                m_i_new);

        }
        _lock_release(lq_internal_prod_lock, 1); // release early for next iteration
        event1();
        O_normalize<LQ, DH, r, t>(
            O_buffer,
            l_i);
        event1();

        _lock_release(o_con_lock, 1);
    }
}
#endif