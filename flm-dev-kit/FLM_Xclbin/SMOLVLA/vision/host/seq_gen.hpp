// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __SEQ_GEN_HPP__
#define __SEQ_GEN_HPP__
#pragma once
#include "npu_utils/npu_utils.hpp"

void gen_attn_seq(
    npu_sequence* seq,
    const uint32_t L,
    const int lq,
    const int DH,
    const int Heads
){

    npu_tiles IT[2][2] = {{IT0, IT1}, {IT6, IT7}};
    assert(L % (lq * 8) == 0);

    int l_begin_mha_address = 10496;

    seq->clear_cmds();
    for (int row = 2; row < 6; row++){
        for (int col = 0; col < 8; col++){
            npu_tiles tile = get_tile(row, col);
            seq->rtp_write(tile, l_begin_mha_address, L);
        }
    }

    // each round, each CU works on 1 head and lc * 8 of l, total_cols / 2 is corresponding to the number of CUs
    const int data_per_round = lq * 8; 
    // zero padding the last round if necessary
    const int down_rounds = (L + data_per_round - 1) / data_per_round;

    for (int head = 0; head < Heads / 2; head++){
        for (int round = 0; round < down_rounds; round++){
            std::cout << "round: " << round << std::endl;

            int bd_offset = (round % 2) * 8;
            std::cout << "bd_offset: " << bd_offset << std::endl;

            // cu 0,
            for (int cu = 0; cu < 2; cu++){
                int head_offset = head * 2 + cu;
                std::cout << "head_offset: " << head_offset << std::endl;
                for (int col = 0; col < 2; col++){    
                    // receive y
                    size_t y_offset = head_offset * DH + round * data_per_round * DH * Heads + col * lq * 4 * DH * Heads; 
                    seq->npu_dma_memcpy_nd(
                        2, 0,
                        S2MM, IT[cu][col],
                        (npu_bd_id)(bd_offset + 0), it_channel_0,
                        {0, 0, 0, (uint32_t)y_offset},
                        {1, 1, (uint32_t)4 * lq, (uint32_t)(DH)},
                        {0, 0, (uint32_t)DH * Heads, 1},
                        -1, 0, true
                    );
                }
                // send q
                size_t q_offset = head_offset * DH + round * data_per_round * DH * Heads;
                seq->npu_dma_memcpy_nd(
                    2, 1,
                    MM2S, IT[cu][0],
                    (npu_bd_id)(bd_offset + 1), it_channel_0,
                    {0, 0, 0, (uint32_t)q_offset},
                    {1, 1, (uint32_t)lq * 4, (uint32_t)(DH)},
                    {0, 0, (uint32_t)DH * Heads, 1},
                    -1, 0, false
                );
                seq->npu_dma_memcpy_nd(
                    2, 1,
                    MM2S, IT[cu][0],
                    (npu_bd_id)(bd_offset + 2), it_channel_1,
                    {0, 0, 0, (uint32_t)(q_offset + (size_t)lq * 4 * DH * Heads)},
                    {1, 1, (uint32_t)lq * 4, (uint32_t)(DH)},
                    {0, 0, (uint32_t)DH * Heads, 1},
                    -1, 0, false
                );
                // send k
                size_t k_offset = (size_t)head_offset * DH;
                seq->npu_dma_memcpy_nd(
                    2, 2,
                    MM2S, IT[cu][1],
                    (npu_bd_id)(bd_offset + 3), it_channel_0,
                    {0, 0, 0, (uint32_t)k_offset},
                    {1, (L / 512), 512, DH},
                    {0, (uint32_t)512 * DH * Heads, (uint32_t)DH * Heads, 1},
                    -1, 0, false
                );
                // send v
                size_t v_offset = head_offset * DH + L * DH * Heads;
                seq->npu_dma_memcpy_nd(
                    2, 2,
                    MM2S, IT[cu][1],
                    (npu_bd_id)(bd_offset + 4), it_channel_1,
                    {0, 0, 0, (uint32_t)v_offset},
                    {1, (L / 512), 512, DH},
                    {0, (uint32_t)512 * DH * Heads, (uint32_t)DH * Heads, 1},
                    -1, 0, false
                );
            } // cu
            if (round > 0){
                for (int cu = 0; cu < 2; cu++){
                    for (int col = 0; col < 2; col++){
                        seq->npu_dma_wait(
                            IT[cu][col],
                            S2MM,
                            it_channel_0
                        );
                    } // col
                } // cu
            }// round > 0
        }// round
        for (int cu = 0; cu < 2; cu++){
            for (int col = 0; col < 2; col++){
                seq->npu_dma_wait(
                    IT[cu][col],
                    S2MM,
                    it_channel_0
                );
            } // col
        } // cu
    } // head
    seq->cmds2seq();
}

void gen_mm_seq(npu_sequence* seq, const uint32_t M, const uint32_t K, const uint32_t N, const uint32_t m, const uint32_t k, const uint32_t n, const uint32_t weight_offset, int32_t rtp_bias, int32_t rtp_act){

    constexpr npu_tiles IT[4] = {IT2, IT3, IT4, IT5};
    constexpr int total_cols = 4;
    constexpr int total_rows = 4;
    constexpr int CT_lock_address_base = 0x000001F000;
    constexpr int CT_rtp_sync_lock_id = 6;
    assert(N % 256 == 0);
    seq->clear_cmds();

    constexpr int rtp_address = 56832;
    int K_div_k = K / k;
    const int M_div_m = M / m / total_rows;
    const int N_div_n  = N / n / total_cols; 
    for (int row = 2; row < 6; row++){
        for (int col = 2; col < total_cols + 2; col++){
            npu_tiles tile = get_tile(row, col);
            seq->rtp_write(tile, rtp_address, M_div_m);
            seq->rtp_write(tile, rtp_address + 4, K_div_k);
        }
    }

    for (int j = 0; j < N_div_n; j++){
        if(rtp_bias == 1){
            // send b
            uint32_t b_offset = j * n * total_cols;
            for (int c = 0; c < total_cols; c++){
                seq->npu_dma_memcpy_nd(
                    2, 2,
                    MM2S, IT[c],
                    npu_bd_id(3), it_channel_1,
                    {0, 0, 0, b_offset + c * n},
                    {1, 1, 1, (uint32_t)n * k},
                    {0, 0, 0, 1},
                    -1, 0, false
                );
            }
        }
        int bias_offset = (rtp_bias == 1)? N : 0;
        for (int i = 0; i < M_div_m; i++){
            int round = j * M_div_m + i;
            int bd_offset = (round % 2) * 8;
            int y_offset = i * m * N * total_rows + j * n * total_cols;
            int x_offset = i * m * K * total_rows;
            uint32_t w_offset = j * n * K * total_cols + bias_offset;
            npu_bd_id y_bd = npu_bd_id(0 + bd_offset);
            npu_bd_id x_bd = npu_bd_id(1 + bd_offset);
            npu_bd_id w_bd = npu_bd_id(2 + bd_offset);
            // send w
            for (int c = 0; c < total_cols; c++){
                seq->npu_dma_memcpy_nd(
                    2, 2,
                    MM2S, IT[c],
                    w_bd, it_channel_1,
                    {0, 0, 0, w_offset + c * n * K},
                    {1, K_div_k, n, (uint32_t)k},
                    {0, k, K, 1},
                    -1, 0, false
                );
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
                for (int r = 0; r < total_rows; r++){
                    seq->npu_dma_wait(IT[r], S2MM, it_channel_0);
                }
            }
        }
    }
    for (int r = 0; r < total_rows; r++){
        seq->npu_dma_wait(IT[r], S2MM, it_channel_0);
    }
    seq->cmds2seq();
}
#endif