// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#pragma once
#include "npu_utils/npu_utils.hpp"

void gen_sliding_mha_engine_seq(npu_sequence* seq, const uint32_t L){
    constexpr int DQ = 768;
    constexpr int DK = 256;
    constexpr int DV = DK;
    constexpr int MAX_L = 2048;
    constexpr int W = 257;
    constexpr int KV_CACHE_SIZE = MAX_L * (DK + DV);
    constexpr uint32_t K_OFFSET = 0;
    constexpr uint32_t V_OFFSET = KV_CACHE_SIZE / 2;
    constexpr int CHUNK_SIZE = 128;
    constexpr int total_cols = 8;
    constexpr int total_rows = 4;
    constexpr npu_tiles IT[8] = {IT0, IT1, IT2, IT3, IT4, IT5, IT6, IT7};
    constexpr int lc = 16;
    const int G = DQ / DK;
    const int L_buffer_address = 14848;
    const int window_size_address = 14880;
    seq->clear_cmds();
    for (int row = 2; row < 5; row++){
        for (int col = 0; col < total_cols; col++){
            npu_tiles tile = get_tile(row, col);
            seq->rtp_write(tile, L_buffer_address, L);
            seq->rtp_write(tile, window_size_address, W);
        }
    }
    int L_PADDED = (L + CHUNK_SIZE - 1) / CHUNK_SIZE * CHUNK_SIZE;
    int total_chunks = L_PADDED / CHUNK_SIZE;
   

    for (int chunk_idx = 0; chunk_idx < total_chunks; chunk_idx++){
        int chunk_q_begin = chunk_idx * CHUNK_SIZE;
        int chunk_q_end = chunk_q_begin + CHUNK_SIZE - 1;
        int chunk_k_begin = chunk_q_begin - W + 1;
        int chunk_k_end = chunk_q_end + W;
        chunk_k_begin = chunk_k_begin < 0 ? 0 : chunk_k_begin;
        chunk_k_end = chunk_k_end > L ? L : chunk_k_end;
        int k_begin = chunk_k_begin / CHUNK_SIZE * CHUNK_SIZE;
        int k_end = (chunk_k_end + CHUNK_SIZE - 1) / CHUNK_SIZE * CHUNK_SIZE;
        int k_length = k_end - k_begin;
        int k_chunks = k_length / CHUNK_SIZE;
        int bd_offset = (chunk_idx % 2) * 8;
       
        std::cout << "Chunk " << chunk_idx << " chunk_q_begin " << chunk_q_begin << " chunk_q_end " << chunk_q_end << " chunk_k_begin " << chunk_k_begin << " chunk_k_end " << chunk_k_end << " k_begin " << k_begin << " k_end " << k_end << " k_length " << k_length << std::endl;
        for (int col = 0; col < total_cols; col++){
            // receive y
            size_t y_offset = (chunk_idx * lc * total_cols + col * lc) * DQ; // y does not have history
            seq->npu_dma_memcpy_nd(
                2, 0,
                S2MM, IT[col],
                (npu_bd_id)(bd_offset + 0), it_channel_0,
                {0, 0, 0, (uint32_t)y_offset},
                {1, 1, (uint32_t)1, (uint32_t)(lc * DQ)},
                {0, 0, (uint32_t)0, 1},
                -1, 0, true
            );
            // send q
            size_t q_offset = (chunk_idx * lc * total_cols + col * lc) * DQ; // q does not have history
            seq->npu_dma_memcpy_nd(
                2, 1,
                MM2S, IT[col],
                (npu_bd_id)(bd_offset + 1), it_channel_0,
                {0, 0, 0, (uint32_t)q_offset},
                {1, 1, (uint32_t)1, (uint32_t)(lc * DQ)},
                {0, 0, (uint32_t)0, 1},
                -1, 0, false
            );

        }
        // send k
        seq->npu_dma_memcpy_nd(
            2, 2,
            MM2S, IT[0],
            (npu_bd_id)(bd_offset + 2), it_channel_1,
            {0, 0, 0, K_OFFSET + k_begin * DK},
            {1, 1, 1, (uint32_t)k_length * DK},
            {0, 0, 0, 1},
            -1, 0, false
        );

        // send v
        seq->npu_dma_memcpy_nd(
            2, 2,
            MM2S, IT[4],
            (npu_bd_id)(bd_offset + 3), it_channel_1,
            {0, 0, 0, (uint32_t)V_OFFSET + k_begin * DV},
            {1, 1, 1, (uint32_t)k_length * DV},
            {0, 0, 0, 1},
            -1, 0, false
        );

        
        if (chunk_idx > 0){
            for (int col = 0; col < total_cols; col++){
                seq->npu_dma_wait(
                    IT[col],
                    S2MM,
                    it_channel_0
                );
            }
        }
    }
    for (int col = 0; col < total_cols; col++){
        seq->npu_dma_wait(
            IT[col],
            S2MM,
            it_channel_0
        );
    }
    seq->cmds2seq();
}