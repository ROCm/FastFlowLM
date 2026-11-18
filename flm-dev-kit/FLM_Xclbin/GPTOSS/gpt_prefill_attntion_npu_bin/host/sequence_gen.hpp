// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#pragma once
#include "npu_utils/npu_utils.hpp"
#include "typedef.hpp"

void gen_mha_engine_seq(npu_sequence* seq, const uint32_t L_begin, const uint32_t L_end, const int MAX_L, bf16* sinks, bool is_sliding_window = true){
    constexpr int DQ = 4096;
    constexpr int DK = 512;
    constexpr int DV = 512;
    constexpr int attn_heads = 64;
    constexpr int kv_heads = 8;
    constexpr int lc = 32;
    constexpr int DH = 64;
    constexpr int total_cols = 8;
    constexpr npu_tiles IT[8] = {IT0, IT1, IT2, IT3, IT4, IT5, IT6, IT7};
    constexpr int L_begin_addr = 55296;
    constexpr int L_end_addr = 25088;
    constexpr int sink_addr = 41472;
    constexpr int window_size_addr = 55328;
    constexpr int WINDOW_SIZE = 128;
    constexpr int heads_per_group = 32;
    const int KV_CACHE_SIZE = MAX_L * (DK + DV);
    const int PARTIAL_SIZE = KV_CACHE_SIZE / 4;
    const int V_OFFSET = PARTIAL_SIZE * 2;
    int using_sliding_window = is_sliding_window ? WINDOW_SIZE : L_end;
    seq->clear_cmds();
    int new_L = L_end - L_begin;
    // in this case, there are two rounds, round 0 is for heads 0~31, round 1 is for heads 32~63
    for (int head_group = 0; head_group < attn_heads / heads_per_group; head_group++){
        uint32_t HEAD_OFFSET = head_group * heads_per_group * DH;
        
        for (int col = 0; col < 8; col++){
            for (int row = 2; row < 6; row++){
                npu_tiles tile = get_tile(row, col);
                seq->rtp_write(tile, L_begin_addr, L_begin);
                seq->rtp_write(tile, L_end_addr, L_end);
                float float_sink = (float)(sinks[col * 4 + row - 2 + head_group * heads_per_group]);
                // std::cout << "float_sink: " << float_sink << std::endl;
                // seq->rtp_write(tile, sink_addr, *(uint32_t*)(&float_sink));
                // float_sink = -80;
                seq->rtp_write(tile, sink_addr, *(uint32_t*)(&float_sink));
                seq->rtp_write(tile, window_size_addr, using_sliding_window);
            }
        }
        for (int round = 0; round < new_L / 32; round++){
            int bd_offset = (round % 2) * 8;
            int L_CURRENT = L_begin + round * 32;
            int kv_begin = ((L_CURRENT - using_sliding_window) > 0) ? (L_CURRENT - using_sliding_window) : 0;
            uint32_t kv_length = (L_CURRENT - kv_begin + 32) * DK / 2;
            uint32_t kv_offset = (uint32_t)(kv_begin * DK / 2) + PARTIAL_SIZE * head_group;
            for (int col = 0; col < total_cols; col++){
                // receive y
                size_t y_offset = round * lc * DQ + col * 4 * DH + HEAD_OFFSET; // y does not have history
                seq->npu_dma_memcpy_nd(
                    2, 0,
                    S2MM, IT[col],
                    (npu_bd_id)(bd_offset + 0), it_channel_0,
                    {0, 0, 0, (uint32_t)y_offset},
                    {1, 1, (uint32_t)lc, (uint32_t)DH * 4},
                    {0, 0, (uint32_t)DQ, 1},
                    -1, 0, true
                );
                // send q
                size_t q_offset = round * lc * DQ + col * 4 * DH + HEAD_OFFSET; // q does not have history
                seq->npu_dma_memcpy_nd(
                    2, 1,
                    MM2S, IT[col],
                    (npu_bd_id)(bd_offset + 1), it_channel_0,
                    {0, 0, 0, (uint32_t)q_offset},
                    {1, 1, (uint32_t)lc, (uint32_t)DH * 4},
                    {0, 0, (uint32_t)DQ, 1},
                    -1, 0, false
                );
            }
            // send k
            seq->npu_dma_memcpy_nd(
                2, 2,
                MM2S, IT[0],
                (npu_bd_id)(bd_offset + 2), it_channel_1,
                {0, 0, 0, (uint32_t)(kv_offset)},
                {1, 1, 1, kv_length},
                {0, 0, 0, 1},
                -1, 0, false
            );

            // send v
            seq->npu_dma_memcpy_nd(
                2, 2,
                MM2S, IT[4],
                (npu_bd_id)(bd_offset + 3), it_channel_1,
                {0, 0, 0, (uint32_t)(V_OFFSET + kv_offset)},
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
        } // end in round for time index
        for (int col = 0; col < total_cols; col++){
            seq->npu_dma_wait(
                IT[col],
                S2MM,
                it_channel_0
            );
        }
    } // end of head_group
    seq->cmds2seq();
}