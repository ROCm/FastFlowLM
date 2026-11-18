// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#pragma once
#include "npu_utils/npu_utils.hpp"

void gen_mha_engine_seq(npu_sequence* seq, const uint32_t LK_end, const uint32_t LQ_end, int L_size){
    constexpr int DQ = 768;
    constexpr int DK = 256;
    constexpr int DV = DK;
    constexpr int DH = 256;
    constexpr int MAX_L = 2048;
    constexpr int KV_CACHE_SIZE = MAX_L * (DK + DV);
    constexpr int K_OFFSET = 0;
    constexpr int V_OFFSET = KV_CACHE_SIZE / 2;
    constexpr int total_cols = 8;
    constexpr int total_rows = 3;
    constexpr npu_tiles IT[8] = {IT0, IT1, IT2, IT3, IT4, IT5, IT6, IT7};
    constexpr int lc = 16;
    const int G = DQ / DK;
    const int lk_end_mha_address = 11776;
    assert(LQ_end % (16 * 8) == 0);
    seq->clear_cmds();
    for (int row = 2; row < total_rows + 2; row++){
        for (int col = 0; col < total_cols; col++){
            npu_tiles tile = get_tile(row, col);
            seq->rtp_write(tile, lk_end_mha_address, L_size); // LK
        }
    }
    int new_L = LQ_end;

    for (int round = 0; round < new_L / 16 / 8; round++){
        int bd_offset = (round % 2) * 8;
        for (int col = 0; col < total_cols; col++){ // col is bind to l
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
            {0, 0, 0, K_OFFSET},
            {1, 1, 1, LK_end * DK},
            {0, 0, 0, 1},
            -1, 0, false
        );

        // send v
        seq->npu_dma_memcpy_nd(
            2, 2,
            MM2S, IT[4],
            (npu_bd_id)(bd_offset + 3), it_channel_1,
            {0, 0, 0, (uint32_t)V_OFFSET},
            {1, 1, 1, LK_end * DK},
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