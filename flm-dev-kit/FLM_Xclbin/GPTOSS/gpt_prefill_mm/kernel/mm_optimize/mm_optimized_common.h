// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __MM_OPTIMIZE_COMMON_H__
#define __MM_OPTIMIZE_COMMON_H__
#define NOCPP
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>
#include "../zero.h"
#include <aie_api/aie.hpp>
#include "../aie_kernel_utils.h" //TODO: a better method?
#include "../NonLinear/nonlut_based_ops.h"
// // note: also work for 64x64, not 64x128
// #define CT_k_max_m_128_n_64 64 // maximum k size of 128 to store on CT for m=128,n=64

constexpr int CT_k_max_n_64 = 64;
constexpr int CT_k_max_n_128 = 32;
constexpr int CT_k_max_n_256 = 16;
constexpr int CT_k_max_n_32 = 32;
constexpr int CT_k_max_n_16 = 16;





template<int N>
constexpr int compute_CT_k_max_n() {
    if constexpr (N == 64) {
        return CT_k_max_n_64;
    } else if constexpr (N == 128) {
        return CT_k_max_n_128;
    } else if constexpr (N == 256) {
        return CT_k_max_n_256;
    } else if constexpr (N == 32) {
        return CT_k_max_n_32;
    }else if constexpr (N == 16){
        return CT_k_max_n_16;
    }
     else {
        return -1;
    }
}



template <int m>
void copy_float_to_bfloat16_lock_aware(bfloat16 *y_pin, bfloat16 *y_pon, const float *y_acc, int y_prod_lock, int y_con_lock)
{
    // acquire vs release for every 16 value
    // thus assume y size is 16 or greater
    static_assert(m % 16 == 0);
    constexpr int ping_pong_size = 512;    
    static_assert(m % (ping_pong_size * 2) == 0); // for ping pong assertion with each ping-pong write 32
    
    constexpr int lenPerLoop = ping_pong_size * 2;

    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MAX_ITERATION_COUNT(m / lenPerLoop)
    for (int i = 0; i < m / lenPerLoop; i++) {
        // Process ping buffer (first 64 elements)
        {
            acquire_greater_equal(y_prod_lock, 1);
            
            // Load, convert and store  vectors of 16 elements each
            for (int j = 0; j < (ping_pong_size)/16; j++) {
                aie::accum<accfloat, 16> acc;
                aie::vector<float, 16> y_vec = aie::load_v<16>(y_acc + j * 16);
                acc.from_vector(y_vec);
                aie::vector<bfloat16, 16> acc_b16 = to_v16bfloat16(acc);
                aie::store_v(y_pin + j * 16, acc_b16);
            }
            
            release(y_con_lock, 1);
        }
        
        // Process pong buffer (next 64 elements)
        {
            acquire_greater_equal(y_prod_lock, 1);
            
            // Load, convert and store  vectors of 16 elements each
            for (int j = 0; j < (ping_pong_size)/16; j++) {
                aie::accum<accfloat, 16> acc;
                aie::vector<float, 16> y_vec = aie::load_v<16>(y_acc + ping_pong_size + j * 16);
                acc.from_vector(y_vec);
                aie::vector<bfloat16, 16> acc_b16 = to_v16bfloat16(acc);
                aie::store_v(y_pon + j * 16, acc_b16);
            }
            
            release(y_con_lock, 1);
        }
        
        y_acc += lenPerLoop;
    }
}






template <int m>
void copy_float_to_bfloat16_lock_aware_with_nonlinear(bfloat16 *y_pin, bfloat16 *y_pon, const float *y_acc, int y_prod_lock, int y_con_lock, int MODE,

    int CLAMP_OUTPUT, // 0 means no clamp, 1 means clamp
    float clamp_min, float clamp_max

)
{
    // acquire vs release for every 16 value
    // thus assume y size is 16 or greater
    static_assert(m % 16 == 0);
    constexpr int ping_pong_size = 512;    
    static_assert(m % (ping_pong_size * 2) == 0); // for ping pong assertion with each ping-pong write 32
    
    constexpr int lenPerLoop = ping_pong_size * 2;
    aie::vector<bfloat16, 16> min_value = aie::broadcast<bfloat16, 16>(static_cast<bfloat16>(clamp_min));
    aie::vector<bfloat16, 16> max_value = aie::broadcast<bfloat16, 16>(static_cast<bfloat16>(clamp_max));
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MAX_ITERATION_COUNT(m / lenPerLoop)
    for (int i = 0; i < m / lenPerLoop; i++) {
        // Process ping buffer (first 64 elements)
        {
            acquire_greater_equal(y_prod_lock, 1);
            
            // Load, convert and store  vectors of 16 elements each
            for (int j = 0; j < (ping_pong_size)/16; j++) {
                aie::accum<accfloat, 16> acc;
                aie::vector<float, 16> y_vec = aie::load_v<16>(y_acc + j * 16);
                acc.from_vector(y_vec);
                aie::vector<bfloat16, 16> acc_b16 = to_v16bfloat16(acc);
                if (MODE== 0){
                    if(CLAMP_OUTPUT == 1){
                        acc_b16 = aie::clamp( acc_b16, min_value, max_value);
                    }
                    aie::store_v(y_pin + j * 16, acc_b16);
                }
                else if (MODE == 1){
                    aie::vector<bfloat16, 16> gelu_res = getGeluBf16_nonLUT<16>(acc_b16);
                    if(CLAMP_OUTPUT == 1){
                        gelu_res = aie::clamp( gelu_res, min_value, max_value);
                    }
                    aie::store_v(y_pin + j * 16, gelu_res);
                }else if(MODE==2){
                    aie::vector<bfloat16, 16> gelu_res = getSiluBf16_nonLUT<16>(acc_b16);
                    if(CLAMP_OUTPUT == 1){
                        gelu_res = aie::clamp( gelu_res, min_value, max_value);
                    }
                    aie::store_v(y_pin + j * 16, gelu_res);
                }else{
                    aie::vector<bfloat16, 16> sigmoid_res = getSigmoidBf16_nonLUT<16>(acc_b16);
                    if(CLAMP_OUTPUT == 1){
                        sigmoid_res = aie::clamp( sigmoid_res, min_value, max_value);
                    }
                    aie::store_v(y_pin + j * 16, sigmoid_res);
                }

            }
            
            release(y_con_lock, 1);
        }
        
        // Process pong buffer (next 64 elements)
        {
            acquire_greater_equal(y_prod_lock, 1);
            
            // Load, convert and store  vectors of 16 elements each
            for (int j = 0; j < (ping_pong_size)/16; j++) {
                aie::accum<accfloat, 16> acc;
                aie::vector<float, 16> y_vec = aie::load_v<16>(y_acc + ping_pong_size + j * 16);
                acc.from_vector(y_vec);
                aie::vector<bfloat16, 16> acc_b16 = to_v16bfloat16(acc);
                if(MODE == 0){
                    if(CLAMP_OUTPUT == 1){
                        acc_b16 = aie::clamp( acc_b16, min_value, max_value);
                    }
                    aie::store_v(y_pon + j * 16, acc_b16);
                }
                else if(MODE == 1){
                    aie::vector<bfloat16, 16> gelu_res = getGeluBf16_nonLUT<16>(acc_b16);
                    if(CLAMP_OUTPUT == 1){
                        gelu_res = aie::clamp( gelu_res, min_value, max_value);
                    }
                    aie::store_v(y_pon + j * 16, gelu_res);
                }else if (MODE == 2){
                    aie::vector<bfloat16, 16> gelu_res = getSiluBf16_nonLUT<16>(acc_b16);
                    if(CLAMP_OUTPUT == 1){
                        gelu_res = aie::clamp( gelu_res, min_value, max_value);
                    }
                    aie::store_v(y_pon + j * 16, gelu_res);
                }else{
                    aie::vector<bfloat16, 16> sigmoid_res = getSigmoidBf16_nonLUT<16>(acc_b16);
                    if(CLAMP_OUTPUT == 1){
                        sigmoid_res = aie::clamp( sigmoid_res, min_value, max_value);
                    }
                    aie::store_v(y_pon + j * 16, sigmoid_res);
                }

            }
            
            release(y_con_lock, 1);
        }
        
        y_acc += lenPerLoop;
    }
}


template <int m>
void copy_float_to_float_lock_aware(float *y_pin, float *y_pon, const float *y_acc, int y_prod_lock, int y_con_lock)
{
    // acquire vs release for every 16 value
    // thus assume y size is 16 or greater
    static_assert(m % 16 == 0);
    constexpr int ping_pong_size = 512;    
    static_assert(m % (ping_pong_size * 2) == 0); // for ping pong assertion with each ping-pong write 32
    
    constexpr int lenPerLoop = ping_pong_size * 2;

    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MAX_ITERATION_COUNT(m / lenPerLoop)
    for (int i = 0; i < m / lenPerLoop; i++) {
        // Process ping buffer (first 64 elements)
        {
            acquire_greater_equal(y_prod_lock, 1);
            
            // Load, convert and store  vectors of 16 elements each
            for (int j = 0; j < (ping_pong_size)/16; j++) {
                const aie::vector<float, 16> y_vec = aie::load_v<16>(y_acc + j * 16);
                aie::store_v(y_pin + j * 16, y_vec);
            }
            
            release(y_con_lock, 1);
        }
        
        // Process pong buffer (next 64 elements)
        {
            acquire_greater_equal(y_prod_lock, 1);
            
            // Load, convert and store  vectors of 16 elements each
            for (int j = 0; j < (ping_pong_size)/16; j++) {
                const   aie::vector<float, 16> y_vec = aie::load_v<16>(y_acc + ping_pong_size + j * 16);
                aie::store_v(y_pon + j * 16, y_vec);
            }
            
            release(y_con_lock, 1);
        }
        
        y_acc += lenPerLoop;
    }
}

template <typename T_in>
inline const T_in *retrieve_correct_A_buffer(
    const T_in *PA_ping, const T_in *PA_pong,
    int *a_ping_pong_flag)
{
    const T_in *return_ptr = nullptr;

    if (*(a_ping_pong_flag) == 0)
    {

        *a_ping_pong_flag = 1;
        return_ptr = PA_ping;
    }
    else
    {
        *a_ping_pong_flag = 0;

        return_ptr = PA_pong;
    }

    return return_ptr;
}


//This function convert the small sxt blocks of B matrix from col major to row major
// However, the arrangement of sxt blocks is not changed, thus the B matrix is still in col major on bigger view
template <typename T_in, unsigned rowB,
          unsigned colB, unsigned r, unsigned s, unsigned t>
void convert_vectorized_B_s_t_bloc_from_col_major_to_row_major(
    T_in *__restrict pB 
){



    using MMUL = aie::mmul<r, s, t, T_in, T_in, accauto>;
    static_assert( s *t== MMUL::size_B);
    // event0();

    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_RANGE(colB / 2, colB / 2)
    for (unsigned j = 0; j < colB; j += 2)
        AIE_LOOP_FLATTEN
        {
            T_in * pB1;
            T_in * pB2;
            pB1 = pB + (j * rowB) * MMUL::size_B;
            pB2 = pB + ((j + 1) * rowB) * MMUL::size_B;
        
            aie::vector<T_in, MMUL::size_B> B0;
            aie::vector<T_in, MMUL::size_B> B1;

            AIE_PREPARE_FOR_PIPELINING
            AIE_LOOP_RANGE(rowB,rowB) // TODO: faster if unroll, but only should do it if necessary
            for (unsigned i = 0; i < rowB; i++)
                AIE_LOOP_FLATTEN
                {

                    B0 = aie::transpose(aie::load_v<MMUL::size_B>(pB1), t, s);
                    B1 = aie::transpose(aie::load_v<MMUL::size_B>(pB2), t, s);
                    aie::store_v(pB1, B0);
                    aie::store_v(pB2, B1);

                    pB1 += MMUL::size_B;
                    pB2 += MMUL::size_B;
                }
        }


    // event1();
}



template <typename T_in, typename T_out, unsigned rowA, unsigned colA,
          unsigned colB, unsigned r, unsigned s, unsigned t,
          bool b_row_maj = true, bool is_b_s_t_in_row_major=false>
void matmul_vectorized_2x2_mmul_modify_buffer(
    const T_in *__restrict PA_ping, const T_in *__restrict PA_pong, // each buffer is  (2*r) x K(colA=rowB)
    const T_in *__restrict pB,
    int *a_ping_pong_flag, // ping if == 0, else pong
    int a_prod_lock, int a_con_lock,
    T_out *__restrict pC)
{
    // for now

    using MMUL = aie::mmul<r, s, t, T_in, T_in, accauto>;
#if !defined(__chess__) && defined(AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16) && defined(MM_PEANO_BFP16_STEP)
    constexpr bool bfp16_step = r == 8 && s == 8 && t == 8 && std::is_same_v<T_in, bfloat16> &&
                                !b_row_maj && is_b_s_t_in_row_major;
    using MMUL_C = std::conditional_t<bfp16_step, aie_mmul_bf16_888, MMUL>;
#else
    constexpr bool bfp16_step = false;
    using MMUL_C = MMUL;
#endif
    static_assert(r * s == MMUL::size_A);
    event0();
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MAX_ITERATION_COUNT(rowA / 2)
    for (unsigned z = 0; z < rowA; z += 2)
    {
        T_out *__restrict pC1 = pC + (z * colB) * MMUL::size_C;
        T_out *__restrict pC2 = pC + ((z + 1) * colB) * MMUL::size_C;

        const T_in *__restrict pA_cur_buf =  retrieve_correct_A_buffer(PA_ping, PA_pong, a_ping_pong_flag);


        aie::vector<T_in, MMUL::size_A> A0;
        aie::vector<T_in, MMUL::size_A> A1;
        aie::vector<T_in, MMUL::size_B> B0;
        aie::vector<T_in, MMUL::size_B> B1;

        acquire_greater_equal(a_con_lock, 1);
        AIE_PREPARE_FOR_PIPELINING
        AIE_LOOP_MAX_ITERATION_COUNT(colB / 2)
        // Unrolled fully on the BFP16 step path, so the C stores of one j can overlap
        // the loads and conversions of the next.
        AIE_LOOP_UNROLL(bfp16_step ? colB / 2 : 1)
        for (unsigned j = 0; j < colB; j += 2)
            AIE_LOOP_FLATTEN
            {
                const T_in *__restrict pA1 = pA_cur_buf;
                const T_in *__restrict pA2 = pA_cur_buf + colA * MMUL::size_A;
                const T_in *__restrict pB1;
                const T_in *__restrict pB2;
                if constexpr (b_row_maj)
                {
                    pB1 = pB + (j)*MMUL::size_B;
                    pB2 = pB + (j + 1) * MMUL::size_B;
                }
                else
                {
                    pB1 = pB + (j * colA) * MMUL::size_B;
                    pB2 = pB + ((j + 1) * colA) * MMUL::size_B;
                }


                // // Load partial results from C buffer for accumulation in-place. The
                // // zero.cc function handles the zeroing of data when a new
                // // accumulation is needed (after the 'K' reduction dimension)
                // aie::vector<T_out, MMUL::size_C> acc_C00 =
                //     aie::load_v<MMUL::size_C>(pC1);
                // aie::vector<T_out, MMUL::size_C> acc_C01 =
                //     aie::load_v<MMUL::size_C>(pC1 + MMUL::size_C);
                // aie::vector<T_out, MMUL::size_C> acc_C10 =
                //     aie::load_v<MMUL::size_C>(pC2);
                // aie::vector<T_out, MMUL::size_C> acc_C11 =
                //     aie::load_v<MMUL::size_C>(pC2 + MMUL::size_C);

                MMUL_C C00(aie::load_v<MMUL::size_C>(pC1));
                MMUL_C C01(aie::load_v<MMUL::size_C>(pC1 + MMUL::size_C));
                MMUL_C C10(aie::load_v<MMUL::size_C>(pC2));
                MMUL_C C11(aie::load_v<MMUL::size_C>(pC2 + MMUL::size_C));
                
                static_assert(colA%2 == 0);
                #if defined(__chess__)
                    AIE_PREPARE_FOR_PIPELINING
                    AIE_LOOP_MAX_ITERATION_COUNT(colA) // TODO: faster if unroll, but only should do it if necessary
                    for (unsigned i = 0; i < colA; i++)
                        AIE_LOOP_FLATTEN
                        {
                            // First iteration
                            A0 = aie::load_v<MMUL::size_A>(pA1);
                            pA1 += MMUL::size_A;
                            A1 = aie::load_v<MMUL::size_A>(pA2);
                            pA2 += MMUL::size_A;

                            if constexpr (b_row_maj)
                            {
                                B0 = aie::load_v<MMUL::size_B>(pB1);
                                pB1 += MMUL::size_B * colB;
                                B1 = aie::load_v<MMUL::size_B>(pB2);
                                pB2 += MMUL::size_B * colB;
                            }
                            else
                            {
                                if constexpr(is_b_s_t_in_row_major == false){
                                    B0 = aie::transpose(aie::load_v<MMUL::size_B>(pB1), t, s);
                                    pB1 += MMUL::size_B;
                                    B1 = aie::transpose(aie::load_v<MMUL::size_B>(pB2), t, s);
                                    pB2 += MMUL::size_B;
                                }else{
                                    B0 = aie::load_v<MMUL::size_B>(pB1);
                                    pB1 += MMUL::size_B;
                                    B1 = aie::load_v<MMUL::size_B>(pB2);
                                    pB2 += MMUL::size_B;
                                }

                            }

                            C00.mac(A0, B0);
                            C01.mac(A0, B1);
                            C10.mac(A1, B0);
                            C11.mac(A1, B1);


                        }
                #else
                    AIE_PREPARE_FOR_PIPELINING
                    AIE_LOOP_MAX_ITERATION_COUNT(colA/2)
                    // chess flattens this loop into the j loop. Under Peano, unroll
                    // only short loops; unrolling longer ones miscompiles.
                    AIE_LOOP_UNROLL((colA / 2 <= 2 || bfp16_step) ? 2 : 1)
                    for (unsigned i = 0; i < colA; i+=2)
                        AIE_LOOP_FLATTEN
                        {
                            if constexpr (bfp16_step)
                            {
                                // Load each operand where it is converted: one conversion at a
                                // time leaves the four C blocks in registers.
                                AIE_LOOP_UNROLL_FULL
                                for (unsigned h = 0; h < 2; h++) {
                                    auto a0 = MMUL_C::cvt_a(aie::load_v<MMUL::size_A>(pA1));
                                    auto b0 = MMUL_C::cvt_b(aie::load_v<MMUL::size_B>(pB1));
                                    C00.mac(a0, b0);
                                    auto b1 = MMUL_C::cvt_b(aie::load_v<MMUL::size_B>(pB2));
                                    C01.mac(a0, b1);
                                    auto a1 = MMUL_C::cvt_a(aie::load_v<MMUL::size_A>(pA2));
                                    C10.mac(a1, b0);
                                    C11.mac(a1, b1);
                                    pA1 += MMUL::size_A;
                                    pA2 += MMUL::size_A;
                                    pB1 += MMUL::size_B;
                                    pB2 += MMUL::size_B;
                                }
                                continue;
                            }
                            // First iteration
                            A0 = aie::load_v<MMUL::size_A>(pA1);
                            pA1 += MMUL::size_A;
                            A1 = aie::load_v<MMUL::size_A>(pA2);
                            pA2 += MMUL::size_A;

                            if constexpr (b_row_maj)
                            {
                                B0 = aie::load_v<MMUL::size_B>(pB1);
                                pB1 += MMUL::size_B * colB;
                                B1 = aie::load_v<MMUL::size_B>(pB2);
                                pB2 += MMUL::size_B * colB;
                            }
                            else
                            {
                                if constexpr(is_b_s_t_in_row_major == false){
                                    B0 = aie::transpose(aie::load_v<MMUL::size_B>(pB1), t, s);
                                    pB1 += MMUL::size_B;
                                    B1 = aie::transpose(aie::load_v<MMUL::size_B>(pB2), t, s);
                                    pB2 += MMUL::size_B;
                                }else{
                                    B0 = aie::load_v<MMUL::size_B>(pB1);
                                    pB1 += MMUL::size_B;
                                    B1 = aie::load_v<MMUL::size_B>(pB2);
                                    pB2 += MMUL::size_B;
                                }

                            }

                            C00.mac(A0, B0);
                            C01.mac(A0, B1);
                            C10.mac(A1, B0);
                            C11.mac(A1, B1);

                            // Second iteration
                            A0 = aie::load_v<MMUL::size_A>(pA1);
                            pA1 += MMUL::size_A;
                            A1 = aie::load_v<MMUL::size_A>(pA2);
                            pA2 += MMUL::size_A;

                            if constexpr (b_row_maj)
                            {
                                B0 = aie::load_v<MMUL::size_B>(pB1);
                                pB1 += MMUL::size_B * colB;
                                B1 = aie::load_v<MMUL::size_B>(pB2);
                                pB2 += MMUL::size_B * colB;
                            }
                            else
                            {
                                if constexpr(is_b_s_t_in_row_major == false){
                                    B0 = aie::transpose(aie::load_v<MMUL::size_B>(pB1), t, s);
                                    pB1 += MMUL::size_B;
                                    B1 = aie::transpose(aie::load_v<MMUL::size_B>(pB2), t, s);
                                    pB2 += MMUL::size_B;
                                }else{
                                    B0 = aie::load_v<MMUL::size_B>(pB1);
                                    pB1 += MMUL::size_B;
                                    B1 = aie::load_v<MMUL::size_B>(pB2);
                                    pB2 += MMUL::size_B;
                                }
                            }

                            C00.mac(A0, B0);
                            C01.mac(A0, B1);
                            C10.mac(A1, B0);
                            C11.mac(A1, B1);
                        }
                #endif
                aie::store_v(pC1, C00.template to_vector<T_out>());
                pC1 += MMUL::size_C;
                aie::store_v(pC1, C01.template to_vector<T_out>());
                pC1 += MMUL::size_C;
                aie::store_v(pC2, C10.template to_vector<T_out>());
                pC2 += MMUL::size_C;
                aie::store_v(pC2, C11.template to_vector<T_out>());
                pC2 += MMUL::size_C;
            }
        release(a_prod_lock, 1);
    }

    event1();
}


template <typename T_in, unsigned rowA>
void matmul_vectorized_2x2_mmul_consume_a_lock(
    const T_in *__restrict PA_ping, const T_in *__restrict PA_pong, // each buffer is  (2*r) x K(colA=rowB)
    int *a_ping_pong_flag, // ping if == 0, else pong
    int a_prod_lock, int a_con_lock)
{
    // for now


    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MAX_ITERATION_COUNT(rowA / 2)
    for (unsigned z = 0; z < rowA; z += 2)
    {


        const T_in *__restrict pA_cur_buf =  retrieve_correct_A_buffer(PA_ping, PA_pong, a_ping_pong_flag);

        // purely consume a lock here
        acquire_greater_equal(a_con_lock, 1);

        release(a_prod_lock, 1);
    }


}




#endif