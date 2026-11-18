// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __SEQ_GEN_HPP__
#define __SEQ_GEN_HPP__
#pragma once
#include "npu_utils/npu_utils.hpp"

void gen_mm_seq(npu_sequence* seq, const uint32_t M, const uint32_t K, const uint32_t N, const uint32_t m, const uint32_t k, const uint32_t n, const uint32_t weight_offset, int32_t rtp_act){

    constexpr npu_tiles IT[8] = {IT0, IT1, IT2, IT3, IT4, IT5, IT6, IT7};
    constexpr int total_cols = 8;
    constexpr int total_rows = 4;
    constexpr int CT_lock_address_base = 0x000001F000;
    constexpr int CT_rtp_sync_lock_id = 6;
    assert(N % 256 == 0);
    seq->clear_cmds();

    constexpr int rtp_address = 56832;
    int K_div_k = K / k;
    int M_div_m = M / m / total_cols;
    int N_div_n = N / n / total_rows;
    for (int row = 2; row < 6; row++){
        for (int col = 0; col < total_cols; col++){
            npu_tiles tile = get_tile(row, col);
            seq->rtp_write(tile, rtp_address, M_div_m);
            seq->rtp_write(tile, rtp_address + 4, K_div_k);
            seq->rtp_write(tile, rtp_address + 8, rtp_act);
        }
    }

    for (int j = 0; j < N_div_n; j++){
        uint32_t b_offset = j * n * total_rows;
        for (int r = 0; r < total_rows; r++){
            seq->npu_dma_memcpy_nd(
                2, 2,
                MM2S, IT[r],
                npu_bd_id(3), it_channel_1,
                {0, 0, 0, b_offset + r * n},
                {1, 1, 1, (uint32_t)n * k},
                {0, 0, 0, 1},
                -1, 0, false
            );
        }
        for (int i = 0; i < M_div_m; i++){
            int round = j * M_div_m + i;
            int bd_offset = (round % 2) * 8;
            int y_offset = i * m * N * total_cols + j * n * total_rows;
            int x_offset = i * m * K * total_cols;  
            uint32_t w_offset = j * n * K * total_rows + N;
            npu_bd_id y_bd = npu_bd_id(0 + bd_offset);
            npu_bd_id y_bd_1 = npu_bd_id(3 + bd_offset);
            npu_bd_id x_bd = npu_bd_id(1 + bd_offset);
            npu_bd_id w_bd = npu_bd_id(2 + bd_offset);
            // send w
            for (int r = 0; r < total_rows; r++){
                seq->npu_dma_memcpy_nd(
                    2, 2,
                    MM2S, IT[r],
                    w_bd, it_channel_1,
                    {0, 0, 0, w_offset + r * n * K},
                    {1, K_div_k, n, (uint32_t)k},
                    {0, k, K, 1},
                    -1, 0, false
                );
            }
            for (int c = 0; c < total_cols; c++){
                // receive y
                seq->npu_dma_memcpy_nd(
                    2, 0,
                    S2MM, IT[c],
                    y_bd, it_channel_0,
                    {0, 0, 0, y_offset + c * m * N},
                    {1, 1, m, n * 4},
                    {0, 0, N, 1},
                    -1, 0, true
                );
                // send x
                seq->npu_dma_memcpy_nd(
                    2, 1,
                    MM2S, IT[c],
                    x_bd, it_channel_0,
                    {0, 0, 0, x_offset + c * m * K},
                    {1, (uint32_t)K_div_k, m, k},
                    {0, k, K, 1},
                    -1, 0, false
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
#endif
