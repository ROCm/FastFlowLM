// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#pragma once
#include "npu_utils/npu_utils.hpp"

/// Instruction sequence of the bidirectional prefill attention (one prompt, no KV cache).
/// q [L_padded][Heads * DH], kv = K [L_padded][Num_of_d_in_KV_cache * DH] then V (same shape)
/// at KV_CACHE_SIZE / 2, o like q. Every 128-row Q chunk of [0, L_valid) streams K and V rows
/// [0, ceil128(L_valid)); the kernel masks keys >= L_valid. L_valid goes in the window_size slot.
/// Heads: 4 CUs run 4 q heads at a time; q head h reads kv head h / (Heads / Num_of_d_in_KV_cache).
void gen_seq_bidirectional(
    npu_sequence* seq,
    const uint32_t L_valid,
    const int lc,
    const int DH,
    const int Heads,
    const int Num_of_d_in_KV_cache,
    const int KV_CACHE_SIZE
){
    npu_tiles IT[4][2] = {{IT0, IT1}, {IT2, IT3}, {IT4, IT5}, {IT6, IT7}};
    const int data_per_round = lc * 8;
    const int Q_chunks = (L_valid + data_per_round - 1) / data_per_round;
    const int kv_length = Q_chunks * data_per_round;
    const int num_cu = 4;
    const int group = Heads / Num_of_d_in_KV_cache;   // q heads per kv head
    seq->clear_cmds();

    const int l_begin_mha_address = 61568;
    const int l_end_mha_address = 11904;
    const int window_size_address = 61600;
    for (int row = 2; row < 6; row++){
        for (int col = 0; col < 8; col++){
            npu_tiles tile = get_tile(row, col);
            seq->rtp_write(tile, l_begin_mha_address, 0);
            seq->rtp_write(tile, l_end_mha_address, Q_chunks);
            seq->rtp_write(tile, window_size_address, L_valid);
        }
    }
    for (int head = 0; head < Heads / num_cu; head++){
        // the 4 q heads of this pass share one kv head (group == 4 here)
        const int kv_head = head * num_cu / group;
        for (int round = 0; round < Q_chunks; round++){
            int bd_offset = (round % 2) * 8;
            for (int cu = 0; cu < num_cu; cu++){
                int head_offset = head * num_cu + cu;
                for (int col = 0; col < 2; col++){
                    // receive y
                    size_t y_offset = (size_t)head_offset * DH + (size_t)(round * data_per_round + col * lc * 4) * DH * Heads;
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
                size_t q_offset = (size_t)head_offset * DH + (size_t)round * data_per_round * DH * Heads;
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

            // all of K, then all of V, from row 0
            size_t k_offset = (size_t)kv_head * DH;
            seq->npu_dma_memcpy_nd(
                2, 2,
                MM2S, IT[1][1],
                (npu_bd_id)(bd_offset + 3), it_channel_0,
                {0, 0, 0, (uint32_t)k_offset},
                {1, (uint32_t)kv_length / 128, (uint32_t)128, (uint32_t)(DH)},
                {0, (uint32_t)128 * DH * Num_of_d_in_KV_cache, (uint32_t)DH * Num_of_d_in_KV_cache, 1},
                -1, 0, false
            );
            size_t v_offset = k_offset + KV_CACHE_SIZE / 2;
            seq->npu_dma_memcpy_nd(
                2, 2,
                MM2S, IT[1][1],
                (npu_bd_id)(bd_offset + 4), it_channel_1,
                {0, 0, 0, (uint32_t)v_offset},
                {1, (uint32_t)kv_length / 128, (uint32_t)128, (uint32_t)(DH)},
                {0, (uint32_t)128 * DH * Num_of_d_in_KV_cache, (uint32_t)DH * Num_of_d_in_KV_cache, 1},
                -1, 0, false
            );
            if (round > 0){
                for (int cu = 0; cu < num_cu; cu++)
                    for (int col = 0; col < 2; col++)
                        seq->npu_dma_wait(IT[cu][col], S2MM, it_channel_0);
            }
        } // round
        for (int cu = 0; cu < num_cu; cu++)
            for (int col = 0; col < 2; col++)
                seq->npu_dma_wait(IT[cu][col], S2MM, it_channel_0);
    } // head
    seq->cmds2seq();
}
