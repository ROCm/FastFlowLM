// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __SEQ_GEN_HPP__
#define __SEQ_GEN_HPP__

#include "npu_utils/npu_utils.hpp"

constexpr int total_cols = 8;
constexpr int total_rows = 4;
void gen_mm_seq(npu_sequence* seq, const uint32_t M, const uint32_t K, const uint32_t N, const uint32_t weight_offset){
    static constexpr int m = 16;
    static constexpr int k = 128;
    static constexpr int n = 64;
    
    npu_sequence& seq_ptr = *seq;
    constexpr npu_tiles IT[8] = {IT0, IT1, IT2, IT3, IT4, IT5, IT6, IT7};
    int K_div_k = K / k;
    seq_ptr.clear_cmds();
    // set K_DIV_K
    for (int row = 2; row < 6; row++){
        for (int col = 0; col < 8; col++){
            npu_tiles tile = get_tile(row, col);
            seq_ptr.rtp_write(tile, 61440, K_div_k); // K_DIV_K_RTP_ADDRESS in iron/mm_aie.py
        }
    }
    int M_div_m = M / m / total_cols;
    int N_div_n = N / n / total_rows;
    for (int j = 0; j < N_div_n; j++){
        for (int i = 0; i < M_div_m; i++){
            int round = j * M_div_m + i;
            int is_pong = round % 2;
            int bd_offset = is_pong * 8;
            int y_offset = i * m * N * total_cols + j * n * total_rows;
            int x_offset = i * m * K * total_cols;  
            npu_bd_id y_bd = npu_bd_id(0 + bd_offset);
            npu_bd_id x_bd = npu_bd_id(1 + bd_offset);
            npu_bd_id w_bd = npu_bd_id(2 + bd_offset);
            for (int c = 0; c < total_cols; c++){
                // receive y
                seq_ptr.npu_dma_memcpy_nd(
                    2, 0,
                    S2MM, IT[c],
                    y_bd, it_channel_0,
                    {0, 0, 0, y_offset + c * m * N},
                    {1, 4, m, n},
                    {0, n, N, 1},
                    -1, 0, true
                );

                // send x
                seq_ptr.npu_dma_memcpy_nd(
                    2, 1,
                    MM2S, IT[c],
                    x_bd, it_channel_0,
                    {0, 0, 0, x_offset + c * m * K},
                    {1, (uint32_t)K_div_k, m, k},
                    {0, k, K, 1},
                    -1, 0, false
                );
                // seq_ptr.npu_maskwrite(
                //     IT[c],  DMA_BD_x_5 + 32*(uint32_t)(x_bd), AXI_FLAG, 0xF000000
                // );
            } // end c
            for (int r = 0; r < total_rows; r++){
                uint32_t w_offset = j * n * K * total_rows + r * n * K + weight_offset;
                seq_ptr.npu_dma_memcpy_nd(
                    2, 2,
                    MM2S, IT[r + 2],
                    w_bd, it_channel_1,
                    {0, 0, 0, w_offset},
                    {1, (uint32_t)K_div_k, n, k},
                    {0, k, K, 1},
                    -1, 0, false
                );
                // seq_ptr.npu_maskwrite(
                //     IT[r + 2],  DMA_BD_x_5 + 32*(uint32_t)(w_bd), AXI_FLAG, 0xF000000
                // );
            }
            if (round > 0){
                for (int c = 0; c < total_cols; c++){
                    seq_ptr.npu_dma_wait(IT[c], S2MM, it_channel_0);
                }
            }
        }
    }
    for (int c = 0; c < total_cols; c++){
        seq_ptr.npu_dma_wait(IT[c], S2MM, it_channel_0);
    }
    seq_ptr.cmds2seq();
}
#endif