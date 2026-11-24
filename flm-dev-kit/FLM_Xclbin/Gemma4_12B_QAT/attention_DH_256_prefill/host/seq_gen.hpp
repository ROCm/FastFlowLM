// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#pragma once
#include "npu_utils/npu_utils.hpp"

void gen_seq_1_4(
    npu_sequence* seq,
    const uint32_t L_begin,
    const uint32_t L_wanted,
    const int MAX_L,
    const int WINDOW_SIZE,
    const int lc,
    const int DH,
    const int Heads,
    const int Num_of_d_in_KV_cache, // the column size of KV_cache
    const int KV_CACHE_SIZE,
    bool is_sliding_window = false
){

    npu_tiles IT[4][2] = {{IT0, IT1}, {IT2, IT3}, {IT4, IT5}, {IT6, IT7}};
    assert((L_wanted) % (lc * 8) == 0);
    int using_window_size = is_sliding_window ? WINDOW_SIZE : MAX_L;
    const int Q_chunks = L_wanted / (lc * 8);
    seq->clear_cmds();

    const int l_begin_mha_address = 61568;
    const int l_end_mha_address = 11904;
    const int window_size_address = 61600;
    for (int row = 2; row < 6; row++){
        for (int col = 0; col < 8; col++){
            npu_tiles tile = get_tile(row, col);
            seq->rtp_write(tile, l_begin_mha_address, L_begin);
            seq->rtp_write(tile, l_end_mha_address, Q_chunks);
            seq->rtp_write(tile, window_size_address, using_window_size);
        }
    }

    // each CU has 8 CTs and works on 1 head. 
    const int data_per_round = lc * 8; 
    const int down_rounds = Q_chunks;

    int num_cu = 4;
    for (int head = 0; head < Heads / num_cu; head++){
        for (int round = 0; round < down_rounds; round++){
            int Lq_current = L_begin + round * data_per_round;
            int kv_begin = ((Lq_current - using_window_size) > 0) ? (Lq_current - using_window_size) : 0;
            int kv_end = (Lq_current + 2 * data_per_round - 1) / data_per_round * data_per_round;
            int kv_length = kv_end - kv_begin;

            int bd_offset = (round % 2) * 8;
            for (int cu = 0; cu < num_cu; cu++){
                int head_offset = head * num_cu + cu;
                for (int col = 0; col < 2; col++){    
                    // receive y
                    size_t y_offset = head_offset * DH + (round * data_per_round + col * lc * 4) * DH * Heads; 
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
            
            if (kv_length <= 128 * 1024){ 
                // 2 is beacuse 1: 2, 4 CUs has 2 shared CUs
                for (int i = 0; i < 2; i++){
                    int kv_head_offset = head * 2 + i;  // 2 is beacuse 1: 2, 2 q shared one k and v
                    int kv_chunk_offset = kv_head_offset / Num_of_d_in_KV_cache;
                    int kv_head_offset_in_chunk = kv_head_offset % Num_of_d_in_KV_cache;

                    size_t k_offset = kv_chunk_offset * MAX_L * DH * Num_of_d_in_KV_cache + kv_head_offset_in_chunk * DH + kv_begin * DH * Num_of_d_in_KV_cache;
                    seq->npu_dma_memcpy_nd(
                        2, 2,
                        MM2S, IT[i * 2 + 1][1],
                        (npu_bd_id)(bd_offset + 3), it_channel_0,
                        {0, 0, 0, (uint32_t)k_offset},
                        {1, (uint32_t)kv_length / 128, (uint32_t)128, (uint32_t)(DH)},
                        {0, (uint32_t)128 * DH * Num_of_d_in_KV_cache, (uint32_t)DH * Num_of_d_in_KV_cache, 1},
                        -1, 0, false
                    );
                    size_t v_offset = kv_chunk_offset * MAX_L * DH * Num_of_d_in_KV_cache + kv_head_offset_in_chunk * DH + kv_begin * DH * Num_of_d_in_KV_cache + KV_CACHE_SIZE / 2;
                    seq->npu_dma_memcpy_nd(
                        2, 2,
                        MM2S, IT[i * 2 + 1][1],
                        (npu_bd_id)(bd_offset + 4), it_channel_1,
                        {0, 0, 0, (uint32_t)v_offset},
                        {1, (uint32_t)kv_length / 128, (uint32_t)128, (uint32_t)(DH)},
                        {0, (uint32_t)128 * DH * Num_of_d_in_KV_cache, (uint32_t)DH * Num_of_d_in_KV_cache, 1},
                        -1, 0, false
                    );
                }
            }else{
                // 2 is beacuse 1: 2, 4 CUs has 2 shared CUs
                for (int i = 0; i < 2; i++){
                    int kv_head_offset = head * 2 + i;  // 2 is beacuse 1: 2, 2 q shared one k and v
                    int kv_chunk_offset = kv_head_offset / Num_of_d_in_KV_cache;
                    int kv_head_offset_in_chunk = kv_head_offset % Num_of_d_in_KV_cache;
                    
                    size_t k_offset = kv_chunk_offset * MAX_L * DH * Num_of_d_in_KV_cache + kv_head_offset_in_chunk * DH + kv_begin * DH * Num_of_d_in_KV_cache;
                    seq->npu_dma_memcpy_nd(
                        2, 2,
                        MM2S, IT[i * 2 + 1][1],
                        (npu_bd_id)(bd_offset + 3), it_channel_0,
                        {0, 0, 0, (uint32_t)k_offset},
                        {1, (uint32_t)1024, (uint32_t)128, (uint32_t)(DH)},
                        {0, (uint32_t)128 * DH * Num_of_d_in_KV_cache, (uint32_t)DH * Num_of_d_in_KV_cache, 1},
                        -1, 0, false
                    );
                    size_t v_offset = kv_chunk_offset * MAX_L * DH * Num_of_d_in_KV_cache + kv_head_offset_in_chunk * DH + kv_begin * DH * Num_of_d_in_KV_cache + KV_CACHE_SIZE / 2;
                    seq->npu_dma_memcpy_nd(
                        2, 2,
                        MM2S, IT[i * 2 + 1][1],
                        (npu_bd_id)(bd_offset + 4), it_channel_1,
                        {0, 0, 0, (uint32_t)v_offset},
                        {1, (uint32_t)1024, (uint32_t)128, (uint32_t)(DH)},
                        {0, (uint32_t)128 * DH * Num_of_d_in_KV_cache, (uint32_t)DH * Num_of_d_in_KV_cache, 1},
                        -1, 0, false
                    );
                    int remaining_kv_length = kv_length - 1024 * 128;
                    k_offset = kv_chunk_offset * MAX_L * DH * Num_of_d_in_KV_cache + kv_head_offset_in_chunk * DH + kv_begin * DH * Num_of_d_in_KV_cache + 1024 * 128 * DH * Num_of_d_in_KV_cache;
                    seq->npu_dma_memcpy_nd(
                        2, 2,
                        MM2S, IT[i * 2 + 1][1],
                        (npu_bd_id)(bd_offset + 5), it_channel_0,
                        {0, 0, 0, (uint32_t)k_offset},
                        {1, (uint32_t)remaining_kv_length / 128, (uint32_t)128, (uint32_t)(DH)},
                        {0, (uint32_t)128 * DH * Num_of_d_in_KV_cache, (uint32_t)DH * Num_of_d_in_KV_cache, 1},
                        -1, 0, false
                    );
                    v_offset = kv_chunk_offset * MAX_L * DH * Num_of_d_in_KV_cache + kv_head_offset_in_chunk * DH + kv_begin * DH * Num_of_d_in_KV_cache + KV_CACHE_SIZE / 2 + 1024 * 128 * DH * Num_of_d_in_KV_cache;
                    seq->npu_dma_memcpy_nd(
                        2, 2,
                        MM2S, IT[i * 2 + 1][1],
                        (npu_bd_id)(bd_offset + 6), it_channel_1,
                        {0, 0, 0, (uint32_t)v_offset},
                        {1, (uint32_t)remaining_kv_length / 128, (uint32_t)128, (uint32_t)(DH)},
                        {0, (uint32_t)128 * DH * Num_of_d_in_KV_cache, (uint32_t)DH * Num_of_d_in_KV_cache, 1},
                        -1, 0, false
                    );
                }
            }
            if (round > 0){
                for (int cu = 0; cu < num_cu; cu++){
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
        for (int cu = 0; cu < num_cu; cu++){
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
