// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <aie_api/aie.hpp>
#include "mm_888_function_mega_down.h"
#include "../../common_external.hpp"

extern "C"
{


    void matmul_64_256_64_8_8_8_ping_pong_float_acc_bfloat16_out_mega_down(
        bfloat16 *a_pin, bfloat16 *a_pon,
        bfloat16 *b_pin, bfloat16 *b_pon,
        bfloat16* bias_buf,        
        float *y_acc,
        bfloat16 *y_out_pin, bfloat16 *y_out_pon,
        int32_t *a_ping_pong_turn, int32_t *b_ping_pong_turn,

        int32_t CT_row_idx, int32_t CT_col_idx,
        int32_t TOTAL_CT_ROW, int32_t TOTAL_CT_COL,
        int32_t * rtp_buffer,
        int32_t a_prod_lock,
        int32_t a_con_lock,
        int32_t b_prod_lock,
        int32_t b_con_lock,
        int32_t y_prod_lock,
        int32_t y_con_lock)
    {
        // M,K,N is 64
        int32_t K_div_k = rtp_buffer[0];
        int32_t Matrix_M = rtp_buffer[1];
        int32_t Matrix_N = rtp_buffer[2];
        int32_t ENABLE_BIAS = rtp_buffer[3];
        int32_t OUTPUT_MODE = rtp_buffer[4];
        int32_t CLAMP_OUTPUT = rtp_buffer[5];
        float *clamp_output_min_ptr = (float*)(rtp_buffer + 6);
        float OUTPUT_CLAMP_MIN = *clamp_output_min_ptr;
        float OUTPUT_CLAMP_MAX = *(clamp_output_min_ptr + 1);             
        constexpr int M = 64;
        constexpr int K = 256;
        constexpr int N = 64;
        matmul_M_K_N_8_8_8_ping_pong_float_acc_bfloat16_out_mega_down<M, K, N>(
            a_pin, a_pon,
            b_pin, b_pon,
            bias_buf,
            y_acc,
            y_out_pin, y_out_pon,
            a_ping_pong_turn, b_ping_pong_turn,
            K_div_k, Matrix_M, Matrix_N,
            CT_row_idx, CT_col_idx,
            TOTAL_CT_ROW, TOTAL_CT_COL,
            ENABLE_BIAS, OUTPUT_MODE,
            CLAMP_OUTPUT, OUTPUT_CLAMP_MIN, OUTPUT_CLAMP_MAX,
            a_prod_lock,
            a_con_lock,
            b_prod_lock,
            b_con_lock,
            y_prod_lock,
            y_con_lock);
    }

    // void matmul_64_512_128_8_8_8_ping_pong_float_acc_float_out(
    //     bfloat16 *a_pin, bfloat16 *a_pon,
    //     bfloat16 *b_pin, bfloat16 *b_pon,
    //     float *y_acc,
    //     float *y_out_pin, float *y_out_pon,
    //     int32_t *a_ping_pong_turn, int32_t *b_ping_pong_turn,
    //     int32_t K_div_k, int32_t Matrix_M, int32_t Matrix_N,
    //     int32_t CT_row_idx, int32_t CT_col_idx,
    //     int32_t TOTAL_CT_ROW, int32_t TOTAL_CT_COL,
    //     int32_t a_prod_lock,
    //     int32_t a_con_lock,
    //     int32_t b_prod_lock,
    //     int32_t b_con_lock,
    //     int32_t y_prod_lock,
    //     int32_t y_con_lock)
    // {
    //     // M,K,N is 64

    //     constexpr int M = 64;
    //     constexpr int K = 512;
    //     constexpr int N = 128;

    //     // first, zero out the y_acc value
    //     // becaused called for 8x8x8, so
    //     const int M_div_num_row_m = Matrix_M / (M*TOTAL_CT_ROW);
    //     const int N_div_num_col_n = Matrix_N / (N*TOTAL_CT_COL);
    //     const int N_div_num_col_n_remainder_blocks = (Matrix_N % (N*TOTAL_CT_COL))/N;        
    //     for(int mega_block_row_idx = 0; mega_block_row_idx < M_div_num_row_m; mega_block_row_idx++){
    //         for(int mega_block_col_idx = 0; mega_block_col_idx < N_div_num_col_n; mega_block_col_idx++){
    //             zero_vectorized_512<float, M, N>(y_acc);
    //             for(int k = 0; k < K_div_k;k++){
    //                 matmul_vectorized_8x8x8_bf16_f32_ping_pong_a<M, K, N, true, true,  false, CT_k_max_n_64>(a_pin, a_pon, 
    //                                                                             b_pin, b_pon,
    //                                                                             a_ping_pong_turn, b_ping_pong_turn, 
    //                                                                             a_prod_lock, a_con_lock, y_acc,
    //                                                                             b_prod_lock, b_con_lock);
                    
    //             }
    //             copy_float_to_float_lock_aware<M * N>(y_out_pin, y_out_pon, y_acc, y_prod_lock, y_con_lock);
    //         }
    //         if(N_div_num_col_n_remainder_blocks != 0){

    //             if(  (CT_col_idx+1)<= N_div_num_col_n_remainder_blocks  ){
    //                 zero_vectorized_512<float, M, N>(y_acc);
    //                 for(int k = 0; k < K_div_k;k++){
    //                     matmul_vectorized_8x8x8_bf16_f32_ping_pong_a<M, K, N, true, true,  false, CT_k_max_n_64>(a_pin, a_pon, 
    //                                                                                 b_pin, b_pon,
    //                                                                                 a_ping_pong_turn, b_ping_pong_turn, 
    //                                                                                 a_prod_lock, a_con_lock, y_acc,
    //                                                                                 b_prod_lock, b_con_lock);
                        
    //                 }
    //                 copy_float_to_float_lock_aware<M * N>(y_out_pin, y_out_pon, y_acc, y_prod_lock, y_con_lock);
    //             }else{
    //                 // simply consume A
    //                 for(int k = 0; k < K_div_k;k++){
    //                     matmul_vectorized_8x8x8_bf16_f32_consume_a<M, K, CT_k_max_n_64>(a_pin, a_pon, 
    //                                                                                 a_ping_pong_turn, b_ping_pong_turn,
    //                                                                                 a_prod_lock, a_con_lock);
    //                 }
    //             }

    //         }             
    //     }
    // }

}