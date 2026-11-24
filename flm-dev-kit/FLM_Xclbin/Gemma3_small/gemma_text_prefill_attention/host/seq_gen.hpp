// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#pragma once
#include "npu_utils/npu_utils.hpp"

void gen_mha_engine_seq(npu_sequence* seq, const uint32_t L_begin, const uint32_t L_end, bool is_sliding_window = false){
    constexpr int DQ = 1024;
    constexpr int DK = 256;
    constexpr int DV = DK;
    constexpr int MAX_L = 2048;
    constexpr int WINDOW_SIZE = 512;
    constexpr int KV_CACHE_SIZE = MAX_L * (DK + DV);
    constexpr int K_OFFSET = 0;
    constexpr int V_OFFSET = KV_CACHE_SIZE / 2;
    constexpr int total_cols = 8;
    constexpr int total_rows = 4;
    constexpr npu_tiles IT[8] = {IT0, IT1, IT2, IT3, IT4, IT5, IT6, IT7};
    constexpr int lc = 16;
    const int G = DQ / DK;
    const int l_begin_mha_address = 15872;
    const int l_end_mha_address = 15904;
    const int window_size_address = 15936;
    int using_sliding_window = is_sliding_window ? WINDOW_SIZE : L_end;
    assert(L_begin % (16 * 8) == 0);
    assert(L_end % (16 * 8) == 0);
    seq->clear_cmds();
    for (int row = 2; row < 6; row++){
        for (int col = 0; col < total_cols; col++){
            npu_tiles tile = get_tile(row, col);
            seq->rtp_write(tile, l_begin_mha_address, L_begin);
            seq->rtp_write(tile, l_end_mha_address, L_end);
            seq->rtp_write(tile, window_size_address, using_sliding_window);
        }
    }
    int new_L = L_end - L_begin;

    for (int round = 0; round < new_L / 16 / 8; round++){
        int bd_offset = (round % 2) * 8;
        int L_current = L_begin + round * 16 * 8;
        int kv_begin = ((L_current - using_sliding_window) > 0) ? (L_current - using_sliding_window) : 0;
        int kv_length = (L_current - kv_begin + 16 * 8) * DK;
        std::cout << "Round " << round << " L_current " << L_current << " kv_begin " << kv_begin << " kv_length " << kv_length / DK << std::endl;
        for (int col = 0; col < total_cols; col++){
            // receive y
            size_t y_offset = (round * lc * total_cols + col * lc) * DQ; // y does not have history
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
            size_t q_offset = (round * lc * total_cols + col * lc) * DQ; // q does not have history
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
            {0, 0, 0, K_OFFSET + kv_begin * DK},
            {1, 1, 1, kv_length},
            {0, 0, 0, 1},
            -1, 0, false
        );

        // send v
        seq->npu_dma_memcpy_nd(
            2, 2,
            MM2S, IT[1],
            (npu_bd_id)(bd_offset + 3), it_channel_1,
            {0, 0, 0, (uint32_t)V_OFFSET + kv_begin * DV},
            {1, 1, 1, kv_length},
            {0, 0, 0, 1},
            -1, 0, false
        );

        
        if (round > 0){
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