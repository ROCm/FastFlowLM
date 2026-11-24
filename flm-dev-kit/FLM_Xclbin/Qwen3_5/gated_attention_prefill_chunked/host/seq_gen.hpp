// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#pragma once
#include "npu_utils/npu_utils.hpp"

// Copy of qwen3_5vl_npu_sequence::gen_mha_engine_seq (kernels_2b path, FLM_DLL/detail/qwen3_5vl_npu), with the
// engine's members turned into parameters. Keep the two in sync.
// RTP addresses: after a rebuild, re-read them from build/<MODEL_TYPE>/bitstream/from_iron/*.prj/input_with_addresses.mlir
static constexpr uint32_t l_begin_mha_address = 64256;
static constexpr uint32_t l_end_mha_address = 64288;
static constexpr uint32_t window_size_address = 64320;

// q, K and V^T are bfp16 tiles of 16 rows (288 bytes per row of 256): q [head][L_wanted / 16][...] from offset 0 of
// its buffer, K [kv head][MAX_L / 16][...], then V^T [kv head][MAX_L / 16][...] (gen_mha_bfp16_layout in host.cpp).
void gen_mha_engine_seq(
    npu_sequence* seq,
    const uint32_t L_begin,
    const uint32_t L_wanted,
    const int MAX_L,
    const int Heads,
    const int KV_heads,
    const int DH
){
    const uint32_t row_bytes = DH * 9 / 8;
    npu_tiles IT[4][2] = {{IT0, IT1}, {IT2, IT3}, {IT4, IT5}, {IT6, IT7}};

    // a round: 8 cores per q head x 32 query rows
    const int data_per_round = 32 * 8;
    const int rows_per_memtile = data_per_round / 2;
    assert((L_wanted) % data_per_round == 0);
    int using_window_size = MAX_L;
    const int Q_chunks = L_wanted / data_per_round;
    seq->clear_cmds();

    for (int row = 2; row < 6; row++){
        for (int col = 0; col < 8; col++){
            npu_tiles tile = get_tile(row, col);
            seq->rtp_write(tile, l_begin_mha_address, L_begin);
            seq->rtp_write(tile, l_end_mha_address, Q_chunks);
            seq->rtp_write(tile, window_size_address, using_window_size);
        }
    }

    int num_cu = 4;
    for (int head = 0; head < Heads / num_cu; head++){
        for (int round = 0; round < Q_chunks; round++){
            int Lq_current = L_begin + round * data_per_round;
            int kv_begin = ((Lq_current - using_window_size) > 0) ? (Lq_current - using_window_size) : 0;
            int kv_end = (Lq_current + data_per_round + 128 - 1) / 128 * 128;
            int kv_length = kv_end - kv_begin;

            int bd_offset = (round % 2) * 8;
            for (int cu = 0; cu < num_cu; cu++){
                int head_offset = head * num_cu + cu;
                for (int col = 0; col < 2; col++){
                    size_t y_offset = head_offset * DH + (round * data_per_round + col * rows_per_memtile) * DH * Heads;
                    seq->npu_dma_memcpy_nd(
                        2, 0,
                        S2MM, IT[cu][col],
                        (npu_bd_id)(bd_offset + 0), it_channel_0,
                        {0, 0, 0, (uint32_t)y_offset},
                        {1, 1, (uint32_t)rows_per_memtile, (uint32_t)(DH)},
                        {0, 0, (uint32_t)DH * Heads, 1},
                        -1, 0, true
                    );
                }
                const uint32_t q_bytes = (head_offset * L_wanted + round * data_per_round) * row_bytes;
                for (int half = 0; half < 2; half++) {
                    seq->npu_dma_memcpy_nd(
                        1, 1,
                        MM2S, IT[cu][0],
                        (npu_bd_id)(bd_offset + 1 + half), half == 0 ? it_channel_0 : it_channel_1,
                        {0, 0, 0, q_bytes + half * (uint32_t)rows_per_memtile * row_bytes},
                        {1, 1, (uint32_t)rows_per_memtile * row_bytes / 512, 512},
                        {0, 0, 512, 1},
                        -1, 0, false
                    );
                }
            }

            for (int part = 0; part < 2; part++) {
                const uint32_t kv_bytes = ((part * KV_heads + head) * MAX_L + kv_begin) * row_bytes;
                seq->npu_dma_memcpy_nd(
                    1, 2,
                    MM2S, IT[1][1],
                    (npu_bd_id)(bd_offset + 3 + part), part == 0 ? it_channel_0 : it_channel_1,
                    {0, 0, 0, kv_bytes},
                    {1, (uint32_t)kv_length / 128, 128 * row_bytes / 512, 512},
                    {0, 128 * row_bytes, 512, 1},
                    -1, 0, false
                );
            }
            if (round > 0){
                for (int cu = 0; cu < num_cu; cu++){
                    for (int col = 0; col < 2; col++){
                        seq->npu_dma_wait(IT[cu][col], S2MM, it_channel_0);
                    }
                }
            }
        }
        for (int cu = 0; cu < num_cu; cu++){
            for (int col = 0; col < 2; col++){
                seq->npu_dma_wait(IT[cu][col], S2MM, it_channel_0);
            }
        }
    }
    seq->cmds2seq();
}
