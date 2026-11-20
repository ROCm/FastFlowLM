// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#pragma once
#include "npu_utils/npu_utils.hpp"

/// \param L        sequence length padded up to a multiple of 512
/// \param L_valid  the real token count; keys at or past it are masked off
/// \param l_rtp_address  byte address of the L_attn_core_* buffer. This is a
///        per-build value and must be read out of the generated
///        input_with_addresses.mlir - a stale one hangs in wait().
void gen_attn_seq(
    npu_sequence* seq,
    const uint32_t L,
    const uint32_t L_valid,
    const int DH,
    const int Heads,
    const int l_rtp_address
){

    npu_tiles IT[4][2] = {{IT0, IT1}, {IT2, IT3}, {IT4, IT5}, {IT6, IT7}};

    int lq = 64;

    seq->clear_cmds();
    for (int row = 2; row < 6; row++){
        for (int col = 0; col < 8; col++){
            npu_tiles tile = get_tile(row, col);
            seq->rtp_write(tile, l_rtp_address, L);
            seq->rtp_write(tile, l_rtp_address + 4, L_valid);
        }
    }

    // each round, each CU works on 1 head and lc * 8 of l, total_cols / 2 is corresponding to the number of CUs
    const int data_per_round = lq * 8; 
    // zero padding the last round if necessary
    const int down_rounds = (L + data_per_round - 1) / data_per_round;

    for (int head = 0; head < Heads / 4; head++){
        for (int round = 0; round < down_rounds; round++){

            int bd_offset = (round % 2) * 8;

            // cu 0,
            for (int cu = 0; cu < 4; cu++){
                int head_offset = head * 4 + cu;
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
                size_t v_offset = (size_t)head_offset * DH;
                seq->npu_dma_memcpy_nd(
                    2, 3,
                    MM2S, IT[cu][1],
                    (npu_bd_id)(bd_offset + 4), it_channel_1,
                    {0, 0, 0, (uint32_t)v_offset},
                    {1, (L / 512), 512, DH},
                    {0, (uint32_t)512 * DH * Heads, (uint32_t)DH * Heads, 1},
                    -1, 0, false
                );
            } // cu
            if (round > 0){
                for (int cu = 0; cu < 4; cu++){
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
        for (int cu = 0; cu < 4; cu++){
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