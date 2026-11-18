// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <aie_api/aie.hpp>


#include "mm_888_function.h"

extern "C"
{

    void matmul_128_128_64_8_8_8_ping_pong_float_acc_bfloat16_out(
        bfloat16 *a_pin, bfloat16 *a_pon,
        bfloat16 *b_pin, bfloat16 *b_pon,
        float *y_acc,
        bfloat16 *y_out_pin, bfloat16 *y_out_pon,
        int32_t *a_ping_pong_turn, int32_t *b_ping_pong_turn,
        int32_t K_div_k, int32_t Matrix_M, int32_t Matrix_N,
        int32_t CT_row_idx, int32_t CT_col_idx,
        int32_t TOTAL_CT_ROW, int32_t TOTAL_CT_COL,
        int32_t a_prod_lock,
        int32_t a_con_lock,
        int32_t b_prod_lock,
        int32_t b_con_lock,
        int32_t y_prod_lock,
        int32_t y_con_lock)
    {
        // M,K,N is 64

        constexpr int M = 128;
        constexpr int K = 128;
        constexpr int N = 64;

        matmul_m_k_n_8_8_8_ping_pong_float_acc_bfloat16_out<M, K, N>(
            a_pin, a_pon,
            b_pin, b_pon,
            y_acc,
            y_out_pin, y_out_pon,
            a_ping_pong_turn, b_ping_pong_turn,
            K_div_k, Matrix_M, Matrix_N,
            CT_row_idx, CT_col_idx,
            TOTAL_CT_ROW, TOTAL_CT_COL,
            a_prod_lock,
            a_con_lock,
            b_prod_lock,
            b_con_lock,
            y_prod_lock,
            y_con_lock);
    }

    void matmul_128_128_64_8_8_8_ping_pong_float_acc_float_out(
        bfloat16 *a_pin, bfloat16 *a_pon,
        bfloat16 *b_pin, bfloat16 *b_pon,
        float *y_acc,
        float *y_out_pin, float *y_out_pon,
        int32_t *a_ping_pong_turn, int32_t *b_ping_pong_turn,
        int32_t K_div_k, int32_t Matrix_M, int32_t Matrix_N,
        int32_t CT_row_idx, int32_t CT_col_idx,
        int32_t TOTAL_CT_ROW, int32_t TOTAL_CT_COL,
        int32_t a_prod_lock,
        int32_t a_con_lock,
        int32_t b_prod_lock,
        int32_t b_con_lock,
        int32_t y_prod_lock,
        int32_t y_con_lock)
    {
        // M,K,N is 64

        constexpr int M = 128;
        constexpr int K = 128;
        constexpr int N = 64;
        matmul_M_K_N_8_8_8_ping_pong_float_acc_float_out<M, K, N>(
            a_pin, a_pon,
            b_pin, b_pon,
            y_acc,
            y_out_pin, y_out_pon,
            a_ping_pong_turn, b_ping_pong_turn,
            K_div_k, Matrix_M, Matrix_N,
            CT_row_idx, CT_col_idx,
            TOTAL_CT_ROW, TOTAL_CT_COL,
            a_prod_lock,
            a_con_lock,
            b_prod_lock,
            b_con_lock,
            y_prod_lock,
            y_con_lock);
    }

}