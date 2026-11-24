// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <aie_api/aie.hpp>
#include "mm_8_8_8_bf16_f32_bf16.h"
#include "../../common_external.hpp"

template <unsigned M, unsigned K, unsigned N>
void matmul_m_k_n_8_8_8_ping_pong_float_acc_bfloat16_out(
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

    constexpr int CT_k_max_n = compute_CT_k_max_n<N>();

    // first, zero out the y_acc value
    // becaused called for 8x8x8, so
    const int M_div_num_row_m = Matrix_M / (M * TOTAL_CT_ROW);
    const int N_div_num_col_n = Matrix_N / (N * TOTAL_CT_COL);
    const int N_div_num_col_n_remainder_blocks = (Matrix_N % (N * TOTAL_CT_COL)) / N;
    for (int mega_block_row_idx = 0; mega_block_row_idx < M_div_num_row_m; mega_block_row_idx++)
    {
        for (int mega_block_col_idx = 0; mega_block_col_idx < N_div_num_col_n; mega_block_col_idx++)
        {
            zero_vectorized_512<float, M, N>(y_acc);
            for (int k = 0; k < K_div_k; k++)
            {
                matmul_vectorized_8x8x8_bf16_f32_ping_pong_a<M, K, N, true, true, false, CT_k_max_n>(a_pin, a_pon,
                                                                                                     b_pin, b_pon,
                                                                                                     a_ping_pong_turn, b_ping_pong_turn,
                                                                                                     a_prod_lock, a_con_lock, y_acc,
                                                                                                     b_prod_lock, b_con_lock);
            }
            copy_float_to_bfloat16_lock_aware<M * N>(y_out_pin, y_out_pon, y_acc, y_prod_lock, y_con_lock);
        }
        if (N_div_num_col_n_remainder_blocks != 0)
        {

            if ((CT_col_idx + 1) <= N_div_num_col_n_remainder_blocks)
            {
                zero_vectorized_512<float, M, N>(y_acc);
                for (int k = 0; k < K_div_k; k++)
                {
                    matmul_vectorized_8x8x8_bf16_f32_ping_pong_a<M, K, N, true, true, false, CT_k_max_n>(a_pin, a_pon,
                                                                                                         b_pin, b_pon,
                                                                                                         a_ping_pong_turn, b_ping_pong_turn,
                                                                                                         a_prod_lock, a_con_lock, y_acc,
                                                                                                         b_prod_lock, b_con_lock);
                }
                copy_float_to_bfloat16_lock_aware<M * N>(y_out_pin, y_out_pon, y_acc, y_prod_lock, y_con_lock);
            }
            else
            {
                // simply consume A
                for (int k = 0; k < K_div_k; k++)
                {
                    matmul_vectorized_8x8x8_bf16_f32_consume_a<M, K, CT_k_max_n>(a_pin, a_pon,
                                                                                 a_ping_pong_turn, b_ping_pong_turn,
                                                                                 a_prod_lock, a_con_lock);
                }
            }
        }
    }
}

template <unsigned M, unsigned K, unsigned N>
void matmul_M_K_N_8_8_8_ping_pong_float_acc_float_out(
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

    constexpr int CT_k_max_n = compute_CT_k_max_n<N>();

    // first, zero out the y_acc value
    // becaused called for 8x8x8, so
    const int M_div_num_row_m = Matrix_M / (M * TOTAL_CT_ROW);
    const int N_div_num_col_n = Matrix_N / (N * TOTAL_CT_COL);
    const int N_div_num_col_n_remainder_blocks = (Matrix_N % (N * TOTAL_CT_COL)) / N;
    for (int mega_block_row_idx = 0; mega_block_row_idx < M_div_num_row_m; mega_block_row_idx++)
    {
        for (int mega_block_col_idx = 0; mega_block_col_idx < N_div_num_col_n; mega_block_col_idx++)
        {
            zero_vectorized_512<float, M, N>(y_acc);
            for (int k = 0; k < K_div_k; k++)
            {
                matmul_vectorized_8x8x8_bf16_f32_ping_pong_a<M, K, N, true, true, false, CT_k_max_n>(a_pin, a_pon,
                                                                                                     b_pin, b_pon,
                                                                                                     a_ping_pong_turn, b_ping_pong_turn,
                                                                                                     a_prod_lock, a_con_lock, y_acc,
                                                                                                     b_prod_lock, b_con_lock);
            }
            copy_float_to_float_lock_aware<M * N>(y_out_pin, y_out_pon, y_acc, y_prod_lock, y_con_lock);
        }
        if (N_div_num_col_n_remainder_blocks != 0)
        {

            if ((CT_col_idx + 1) <= N_div_num_col_n_remainder_blocks)
            {
                zero_vectorized_512<float, M, N>(y_acc);
                for (int k = 0; k < K_div_k; k++)
                {
                    matmul_vectorized_8x8x8_bf16_f32_ping_pong_a<M, K, N, true, true, false, CT_k_max_n>(a_pin, a_pon,
                                                                                                         b_pin, b_pon,
                                                                                                         a_ping_pong_turn, b_ping_pong_turn,
                                                                                                         a_prod_lock, a_con_lock, y_acc,
                                                                                                         b_prod_lock, b_con_lock);
                }
                copy_float_to_float_lock_aware<M * N>(y_out_pin, y_out_pon, y_acc, y_prod_lock, y_con_lock);
            }
            else
            {
                // simply consume A
                for (int k = 0; k < K_div_k; k++)
                {
                    matmul_vectorized_8x8x8_bf16_f32_consume_a<M, K, CT_k_max_n>(a_pin, a_pon,
                                                                                 a_ping_pong_turn, b_ping_pong_turn,
                                                                                 a_prod_lock, a_con_lock);
                }
            }
        }
    }
}

template <unsigned M, unsigned K, unsigned N>
void matmul_M_K_N_8_8_8_ping_pong_float_acc_col_major_bfloat16_out(
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
    constexpr int CT_k_max_n = compute_CT_k_max_n<N>();

    // first, zero out the y_acc value
    // becaused called for 8x8x8, so
    const int M_div_num_row_m = Matrix_M / (M * TOTAL_CT_ROW);
    const int N_div_num_col_n = Matrix_N / (N * TOTAL_CT_COL);
    const int N_div_num_col_n_remainder_blocks = (Matrix_N % (N * TOTAL_CT_COL)) / N;
    for (int mega_block_row_idx = 0; mega_block_row_idx < M_div_num_row_m; mega_block_row_idx++)
    {
        for (int mega_block_col_idx = 0; mega_block_col_idx < N_div_num_col_n; mega_block_col_idx++)
        {
            zero_vectorized_512<float, M, N>(y_acc);
            for (int k = 0; k < K_div_k; k++)
            {
                matmul_vectorized_8x8x8_bf16_f32_ping_pong_a<M, K, N, false, false, false, CT_k_max_n>(a_pin, a_pon,
                                                                                                       b_pin, b_pon,
                                                                                                       a_ping_pong_turn, b_ping_pong_turn,
                                                                                                       a_prod_lock, a_con_lock, y_acc,
                                                                                                       b_prod_lock, b_con_lock);
            }

            copy_float_to_bfloat16_lock_aware<M * N>(y_out_pin, y_out_pon, y_acc, y_prod_lock, y_con_lock);
        }
        if (N_div_num_col_n_remainder_blocks != 0)
        {

            if ((CT_col_idx + 1) <= N_div_num_col_n_remainder_blocks)
            {
                zero_vectorized_512<float, M, N>(y_acc);
                for (int k = 0; k < K_div_k; k++)
                {
                    matmul_vectorized_8x8x8_bf16_f32_ping_pong_a<M, K, N, false, false, false, CT_k_max_n>(a_pin, a_pon,
                                                                                                           b_pin, b_pon,
                                                                                                           a_ping_pong_turn, b_ping_pong_turn,
                                                                                                           a_prod_lock, a_con_lock, y_acc,
                                                                                                           b_prod_lock, b_con_lock);
                }

                copy_float_to_bfloat16_lock_aware<M * N>(y_out_pin, y_out_pon, y_acc, y_prod_lock, y_con_lock);
            }
            else
            {
                // simply consume A
                for (int k = 0; k < K_div_k; k++)
                {
                    matmul_vectorized_8x8x8_bf16_f32_consume_a<M, K, CT_k_max_n>(a_pin, a_pon,
                                                                                 a_ping_pong_turn, b_ping_pong_turn,
                                                                                 a_prod_lock, a_con_lock);
                }
            }
        }
    }
}

template <unsigned M, unsigned K, unsigned N>
void matmul_M_K_N_8_8_8_ping_pong_float_acc_col_major_float_out(
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

    constexpr int CT_k_max_n = compute_CT_k_max_n<N>();
    // first, zero out the y_acc value
    // becaused called for 8x8x8, so
    const int M_div_num_row_m = Matrix_M / (M * TOTAL_CT_ROW);
    const int N_div_num_col_n = Matrix_N / (N * TOTAL_CT_COL);
    const int N_div_num_col_n_remainder_blocks = (Matrix_N % (N * TOTAL_CT_COL)) / N;
    for (int mega_block_row_idx = 0; mega_block_row_idx < M_div_num_row_m; mega_block_row_idx++)
    {
        for (int mega_block_col_idx = 0; mega_block_col_idx < N_div_num_col_n; mega_block_col_idx++)
        {
            zero_vectorized_512<float, M, N>(y_acc);
            for (int k = 0; k < K_div_k; k++)
            {
                matmul_vectorized_8x8x8_bf16_f32_ping_pong_a<M, K, N, false, false, false, CT_k_max_n>(a_pin, a_pon,
                                                                                                       b_pin, b_pon,
                                                                                                       a_ping_pong_turn, b_ping_pong_turn,
                                                                                                       a_prod_lock, a_con_lock, y_acc,
                                                                                                       b_prod_lock, b_con_lock);
            }

            copy_float_to_float_lock_aware<M * N>(y_out_pin, y_out_pon, y_acc, y_prod_lock, y_con_lock);
        }
        if (N_div_num_col_n_remainder_blocks != 0)
        {

            if ((CT_col_idx + 1) <= N_div_num_col_n_remainder_blocks)
            {
                zero_vectorized_512<float, M, N>(y_acc);
                for (int k = 0; k < K_div_k; k++)
                {
                    matmul_vectorized_8x8x8_bf16_f32_ping_pong_a<M, K, N, false, false, false, CT_k_max_n>(a_pin, a_pon,
                                                                                                           b_pin, b_pon,
                                                                                                           a_ping_pong_turn, b_ping_pong_turn,
                                                                                                           a_prod_lock, a_con_lock, y_acc,
                                                                                                           b_prod_lock, b_con_lock);
                }

                copy_float_to_float_lock_aware<M * N>(y_out_pin, y_out_pon, y_acc, y_prod_lock, y_con_lock);
            }
            else
            {
                // simply consume A
                for (int k = 0; k < K_div_k; k++)
                {
                    matmul_vectorized_8x8x8_bf16_f32_consume_a<M, K, CT_k_max_n>(a_pin, a_pon,
                                                                                 a_ping_pong_turn, b_ping_pong_turn,
                                                                                 a_prod_lock, a_con_lock);
                }
            }
        }
    }
}
template <unsigned M, unsigned K, unsigned N>
void matmul_M_K_N_8_8_8_ping_pong_float_acc_col_major_s_t_row_major_bfloat16_out(
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

    constexpr int CT_k_max_n = compute_CT_k_max_n<N>();

    // first, zero out the y_acc value
    // becaused called for 8x8x8, so
    const int M_div_num_row_m = Matrix_M / (M * TOTAL_CT_ROW);
    const int N_div_num_col_n = Matrix_N / (N * TOTAL_CT_COL);
    const int N_div_num_col_n_remainder_blocks = (Matrix_N % (N * TOTAL_CT_COL)) / N;
    for (int mega_block_row_idx = 0; mega_block_row_idx < M_div_num_row_m; mega_block_row_idx++)
    {
        for (int mega_block_col_idx = 0; mega_block_col_idx < N_div_num_col_n; mega_block_col_idx++)
        {
            zero_vectorized_512<float, M, N>(y_acc);
            for (int k = 0; k < K_div_k; k++)
            {
                matmul_vectorized_8x8x8_bf16_f32_ping_pong_a<M, K, N, false, true, false, CT_k_max_n>(a_pin, a_pon,
                                                                                                      b_pin, b_pon,
                                                                                                      a_ping_pong_turn, b_ping_pong_turn,
                                                                                                      a_prod_lock, a_con_lock, y_acc,
                                                                                                      b_prod_lock, b_con_lock);
            }

            copy_float_to_bfloat16_lock_aware<M * N>(y_out_pin, y_out_pon, y_acc, y_prod_lock, y_con_lock);
        }
        if (N_div_num_col_n_remainder_blocks != 0)
        {

            if ((CT_col_idx + 1) <= N_div_num_col_n_remainder_blocks)
            {
                zero_vectorized_512<float, M, N>(y_acc);
                for (int k = 0; k < K_div_k; k++)
                {
                    matmul_vectorized_8x8x8_bf16_f32_ping_pong_a<M, K, N, false, true, false, CT_k_max_n>(a_pin, a_pon,
                                                                                                          b_pin, b_pon,
                                                                                                          a_ping_pong_turn, b_ping_pong_turn,
                                                                                                          a_prod_lock, a_con_lock, y_acc,
                                                                                                          b_prod_lock, b_con_lock);
                }

                copy_float_to_bfloat16_lock_aware<M * N>(y_out_pin, y_out_pon, y_acc, y_prod_lock, y_con_lock);
            }
            else
            {
                // simply consume A
                for (int k = 0; k < K_div_k; k++)
                {
                    matmul_vectorized_8x8x8_bf16_f32_consume_a<M, K, CT_k_max_n>(a_pin, a_pon,
                                                                                 a_ping_pong_turn, b_ping_pong_turn,
                                                                                 a_prod_lock, a_con_lock);
                }
            }
        }
    }
}
