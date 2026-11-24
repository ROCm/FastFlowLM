// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#pragma once
#include "gpt_oss_params.hpp"
#include "npu_utils/npu_instr_utils.hpp"

void generate_expert_mm_sequence(npu_sequence* seq, uint32_t M, uint32_t K, uint32_t N){
    constexpr int m = 64;
    constexpr int k = 128;
    constexpr int n = 4;
    constexpr int total_rows = 4;
    constexpr int total_cols = 8;
    constexpr npu_tiles IT[8] = {IT0, IT1, IT2, IT3, IT4, IT5, IT6, IT7};
    uint32_t K_div_k = K / k;
    constexpr uint32_t CT_rtp_address = 50176;
    seq->clear_cmds();
    seq->npu_preemption(0);
    // set K_DIV_K
    for (int row = 2; row < 6; row++){
        for (int col = 0; col < 8; col++){
            npu_tiles tile = get_tile(row, col);
            seq->rtp_write(tile, CT_rtp_address, K_div_k);
        }
    }
    int M_div_m = M / m / total_rows;
    int N_div_n = N / n / total_cols;
    for (int j = 0; j < N_div_n; j++){
        for (int i = 0; i < M_div_m; i++){
            int round = j * M_div_m + i;
            int is_pong = round % 2;
            int bd_offset = is_pong * 8;
            int y_offset = i * m * N * total_rows + j * n * total_cols;
            int x_offset = i * m * K * total_rows;
            npu_bd_id y_bd = npu_bd_id(0 + bd_offset);
            npu_bd_id x_bd = npu_bd_id(1 + bd_offset);
            npu_bd_id w_bd = npu_bd_id(2 + bd_offset);
            for (int r = 0; r < total_rows; r++){
                // receive y
                seq->npu_dma_memcpy_nd(
                    2, 0,
                    S2MM, IT[r],
                    y_bd, it_channel_0,
                    {0, 0, 0, y_offset + r * m * N},
                    {1, total_rows, m, n},
                    {0, n, N, 1},
                    -1, 0, true
                );

                seq->npu_dma_memcpy_nd(
                    2, 0,
                    S2MM, IT[r + 4],
                    y_bd, it_channel_0,
                    {0, 0, 0, y_offset + r * m * N + 4 * n},
                    {1, total_rows, m, n},
                    {0, n, N, 1},
                    -1, 0, true
                );

                // send x
                seq->npu_dma_memcpy_nd(
                    2, 1,
                    MM2S, IT[r + 2],
                    x_bd, it_channel_0,
                    {0, 0, 0, x_offset + r * m * K},
                    {1, (uint32_t)K_div_k, m, k},
                    {0, k, K, 1},
                    -1, 0, false, aggressive_cache
                );
            } // end r
            for (int c = 0; c < total_cols; c++){
                uint32_t w_offset = j * n * K * total_cols + c * n * K;
                seq->npu_dma_memcpy_nd(
                    2, 2,
                    MM2S, IT[c],
                    w_bd, it_channel_1,
                    {0, 0, 0, w_offset},
                    {1, K_div_k, n, k},
                    {0, k, K, 1},
                    -1, 0, false, aggressive_cache
                );
            }
            if (round > 0){
                for (int c = 0; c < total_cols; c++){
                    seq->npu_dma_wait(IT[c], S2MM, it_channel_0);
                }
            }
        }
    }
    for (int c = 0; c < total_cols; c++){
        seq->npu_dma_wait(IT[c], S2MM, it_channel_0);
    }
    seq->cmds2seq();
}
