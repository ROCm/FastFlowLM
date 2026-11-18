// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#pragma once
#include "npu_utils/npu_utils.hpp"

void gen_seq_1_3(
    npu_sequence* seq,
    const uint32_t L_begin,
    const uint32_t L_end,
    const int MAX_L,
    const int WINDOW_SIZE,
    const int lc,
    const int DH,
    const int Heads,
    const int Num_of_d_in_KV_cache, // the column size of KV_cache
    const int KV_CACHE_SIZE,
    bool is_sliding_window = false
){

    const int l_begin_mha_address = 15360;
    const int l_end_mha_address = 25088;
    const int window_size_address = 36992;
    npu_tiles IT[4][2] = {{IT0, IT1}, {IT2, IT3}, {IT4, IT5}, {IT6, IT7}};
    assert(L_begin % (lc * 8) == 0);
    assert(L_end % (lc * 8) == 0);
    int using_window_size = is_sliding_window ? WINDOW_SIZE : (L_end);
    seq->clear_cmds();
    for (int row = 2; row < 6; row++){
        for (int col = 0; col < 8; col++){
            npu_tiles tile = get_tile(row, col);
            seq->rtp_write(tile, l_begin_mha_address, L_begin);
            seq->rtp_write(tile, l_end_mha_address, L_end);
            seq->rtp_write(tile, window_size_address, using_window_size);
        }
    }
    // each CU has 8 CTs and works on 1 head. 
    int all_data_size = L_end - L_begin;
    // each round, each CU works on 1 head and lc * 8 of l, total_cols / 2 is corresponding to the number of CUs
    const int data_per_round = lc * 8; 
    // zero padding the last round if necessary
    const int down_rounds = (all_data_size + data_per_round - 1) / data_per_round;
    // std::cout << "down_rounds: " << down_rounds << std::endl;
    // std::cout << "IT_0: " << IT[0] << " IT_1: " << IT[1] << std::endl;
    for (int head = 0; head < Heads / 4; head++){
        for (int round = 0; round < down_rounds; round++){
            int Lq_current = L_begin + round * data_per_round;
            int kv_begin = ((Lq_current - using_window_size) > 0) ? (Lq_current - using_window_size) : 0;
            int kv_length = Lq_current - kv_begin + data_per_round;
            std::cout << "Round " << round << " Lq_current " << Lq_current << " kv_begin " << kv_begin << " kv_length " << kv_length << std::endl;
            
            int bd_offset = (round % 2) * 8;
            std::cout << "bd_offset: " << bd_offset << std::endl;

            // cu 0, 1, 2
            for (int cu = 0; cu < 3; cu++){
                int head_offset = head * 3 + cu;
                std::cout << "head_offset: " << head_offset << std::endl;
                for (int col = 0; col < 2; col++){    
                    // receive y
                    size_t y_offset = head_offset * DH + round * data_per_round * DH * Heads + col * lc * 4 * DH * Heads; 
                    seq->npu_dma_memcpy_nd(
                        2, 0,
                        S2MM, IT[cu][col],
                        (npu_bd_id)(bd_offset + 0), it_channel_0,
                        {0, 0, 0, (uint32_t)y_offset},
                        {1, 1, (uint32_t)4 * lc, (uint32_t)(DH)},
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
                    {1, 1, (uint32_t)lc * 4, (uint32_t)(DH)},
                    {0, 0, (uint32_t)DH * Heads, 1},
                    -1, 0, false
                );
                seq->npu_dma_memcpy_nd(
                    2, 1,
                    MM2S, IT[cu][0],
                    (npu_bd_id)(bd_offset + 2), it_channel_1,
                    {0, 0, 0, (uint32_t)(q_offset + (size_t)lc * 4 * DH * Heads)},
                    {1, 1, (uint32_t)lc * 4, (uint32_t)(DH)},
                    {0, 0, (uint32_t)DH * Heads, 1},
                    -1, 0, false
                );
            } // cu
            int kv_chunk_offset = head / Num_of_d_in_KV_cache;
            int kv_head_offset = head % Num_of_d_in_KV_cache;
            std::cout << "kv_chunk_offset: " << kv_chunk_offset << " kv_head_offset: " << kv_head_offset << std::endl;
            // send k
            size_t k_offset = kv_chunk_offset * MAX_L * Num_of_d_in_KV_cache * DH + kv_head_offset * DH + kv_begin * DH * Num_of_d_in_KV_cache;
            seq->npu_dma_memcpy_nd(
                2, 2,
                MM2S, IT[0][1],
                (npu_bd_id)(bd_offset + 3), it_channel_0,
                {0, 0, 0, (uint32_t)k_offset},
                {1, (uint32_t)kv_length / 128, (uint32_t)128, (uint32_t)(DH)},
                {0, (uint32_t)128 * DH * Num_of_d_in_KV_cache, (uint32_t)DH * Num_of_d_in_KV_cache, 1},
                -1, 0, false
            );
            // send v
            size_t v_offset = kv_chunk_offset * MAX_L * Num_of_d_in_KV_cache * DH + kv_head_offset * DH + kv_begin * DH * Num_of_d_in_KV_cache + KV_CACHE_SIZE / 2;
            seq->npu_dma_memcpy_nd(
                2, 2,
                MM2S, IT[0][1],
                (npu_bd_id)(bd_offset + 4), it_channel_1,
                {0, 0, 0, (uint32_t)v_offset},
                {1, (uint32_t)kv_length / 128, (uint32_t)128, (uint32_t)(DH)},
                {0, (uint32_t)128 * DH * Num_of_d_in_KV_cache, (uint32_t)DH * Num_of_d_in_KV_cache, 1},
                -1, 0, false
            );
            // cu 3
            int head_offset = head + 18;
            std::cout << "head_offset: " << head_offset << std::endl;
            for (int col = 0; col < 2; col++){    
                // receive y
                size_t y_offset = head_offset * DH + round * data_per_round * DH * Heads + col * lc * 4 * DH * Heads; 
                seq->npu_dma_memcpy_nd(
                    2, 0,
                    S2MM, IT[3][col],
                    (npu_bd_id)(bd_offset + 0), it_channel_0,
                    {0, 0, 0, (uint32_t)y_offset},
                    {1, 1, (uint32_t)4 * lc, (uint32_t)(DH)},
                    {0, 0, (uint32_t)DH * Heads, 1},
                    -1, 0, true
                );
            }
            // send q
            size_t q_offset = head_offset * DH + round * data_per_round * DH * Heads;
            seq->npu_dma_memcpy_nd(
                2, 1,
                MM2S, IT[3][0],
                (npu_bd_id)(bd_offset + 1), it_channel_0,
                {0, 0, 0, (uint32_t)q_offset},
                {1, 1, (uint32_t)lc * 4, (uint32_t)(DH)},
                {0, 0, (uint32_t)DH * Heads, 1},
                -1, 0, false
            );
            seq->npu_dma_memcpy_nd(
                2, 1,
                MM2S, IT[3][0],
                (npu_bd_id)(bd_offset + 2), it_channel_1,
                {0, 0, 0, (uint32_t)(q_offset + (size_t)lc * 4 * DH * Heads)},
                {1, 1, (uint32_t)lc * 4, (uint32_t)(DH)},
                {0, 0, (uint32_t)DH * Heads, 1},
                -1, 0, false
            );
            kv_head_offset = (head / 3 + 6) % Num_of_d_in_KV_cache;
            kv_chunk_offset = 1;
            std::cout << "kv_chunk_offset: " << 1 << " kv_head_offset: " << kv_head_offset << std::endl;
            // send k
            k_offset = kv_chunk_offset * MAX_L * Num_of_d_in_KV_cache * DH + kv_head_offset * DH + kv_begin * DH * Num_of_d_in_KV_cache;
            seq->npu_dma_memcpy_nd(
                2, 2,
                MM2S, IT[3][1],
                (npu_bd_id)(bd_offset + 3), it_channel_0,
                {0, 0, 0, (uint32_t)k_offset},
                {1, (uint32_t)kv_length / 128, (uint32_t)128, (uint32_t)(DH)},
                {0, (uint32_t)128 * DH * Num_of_d_in_KV_cache, (uint32_t)DH * Num_of_d_in_KV_cache, 1},
                -1, 0, false
            );
            // send v
            v_offset = kv_chunk_offset * MAX_L * Num_of_d_in_KV_cache * DH + kv_head_offset * DH + kv_begin * DH * Num_of_d_in_KV_cache + KV_CACHE_SIZE / 2;
            seq->npu_dma_memcpy_nd(
                2, 2,
                MM2S, IT[3][1],
                (npu_bd_id)(bd_offset + 4), it_channel_1,
                {0, 0, 0, (uint32_t)v_offset},
                {1, (uint32_t)kv_length / 128, (uint32_t)128, (uint32_t)(DH)},
                {0, (uint32_t)128 * DH * Num_of_d_in_KV_cache, (uint32_t)DH * Num_of_d_in_KV_cache, 1},
                -1, 0, false
            );
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
