// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#pragma once
#include "npu_utils/npu_utils.hpp"

void gen_attn_seq(
    npu_sequence* seq,
    const int DH,
    const int Heads,
    const int kv_heads,
    const int number_of_blocks,
    const int text_chunk_id,
    const int effective_number,
    const int is_self
){
    npu_tiles IT[4] = {IT0, IT1, IT7, IT6};
    seq->clear_cmds();

    // rtp_buffer is int[16]: consecutive slots are 4 bytes apart.
    //   [0]=blocks  [1]=text_chunk_id  [2]=effective_number  [3]=is_self
    const int rtp_address_1 = 12544;
    const int rtp_address_0 = 57472;

    for (int row = 2; row < 6; row++){
        for (int col = 0; col < 2; col++){
            npu_tiles tile = get_tile(row, col);
            seq->rtp_write(tile, rtp_address_1 + 0,  number_of_blocks);
            seq->rtp_write(tile, rtp_address_1 + 4,  text_chunk_id);
            seq->rtp_write(tile, rtp_address_1 + 8,  effective_number);
            seq->rtp_write(tile, rtp_address_1 + 12, is_self);
        }
        npu_tiles tile = get_tile(row, 7);
        seq->rtp_write(tile, rtp_address_1 + 0,  number_of_blocks);
        seq->rtp_write(tile, rtp_address_1 + 4,  text_chunk_id);
        seq->rtp_write(tile, rtp_address_1 + 8,  effective_number);
        seq->rtp_write(tile, rtp_address_1 + 12, is_self);
    }

    for (int row = 2; row < 5; row++){
        npu_tiles tile = get_tile(row, 6);
        seq->rtp_write(tile, rtp_address_0 + 0,  number_of_blocks);
        seq->rtp_write(tile, rtp_address_0 + 4,  text_chunk_id);
        seq->rtp_write(tile, rtp_address_0 + 8,  effective_number);
        seq->rtp_write(tile, rtp_address_0 + 12, is_self);
    }

    // send k / v, one transfer per chunk from its chunk offset in the packed buffer
    const int total_blocks = number_of_blocks + (is_self == 1 ? 1 : 0);

    for (int round = 0; round < total_blocks; round++){
        int bd_offset = (round % 2) * 8;
        for (int col = 0; col < 4; col++){
            if (round == 0 && col < 3){
                // send q
                size_t q_offset = (size_t)col * 4 * DH;
                seq->npu_dma_memcpy_nd(
                    2, 1,
                    MM2S, IT[col],
                    (npu_bd_id)(1), it_channel_0,
                    {0, 0, 0, (uint32_t)q_offset},
                    {1, 1, (uint32_t)64, (uint32_t)(DH * 4)},
                    {0, 0, (uint32_t)DH * Heads, 1},
                    -1, 0, false
                );
                // receive y
                size_t y_offset = (size_t)col * 4 * DH;
                seq->npu_dma_memcpy_nd(
                    2, 0,
                    S2MM, IT[col],
                    (npu_bd_id)(0), it_channel_0,
                    {0, 0, 0, (uint32_t)y_offset},
                    {1, 1, (uint32_t)64, (uint32_t)(DH * 4)},
                    {0, 0, (uint32_t)DH * Heads, 1},
                    -1, 0, true
                );
            }
            else if(round == 0 && col == 3){
                // send q
                seq->npu_dma_memcpy_nd(
                    2, 1,
                    MM2S, IT[col],
                    (npu_bd_id)(1), it_channel_0,
                    {0, 0, 0, (uint32_t)12 * DH},
                    {1, 1, (uint32_t)64, (uint32_t)(DH * 3)},
                    {0, 0, (uint32_t)DH * Heads, 1},
                    -1, 0, false
                );
                // receive y
                seq->npu_dma_memcpy_nd(
                    2, 0,
                    S2MM, IT[col],
                    (npu_bd_id)(0), it_channel_0,
                    {0, 0, 0, (uint32_t)12 * DH},
                    {1, 1, (uint32_t)64, (uint32_t)(DH * 3)},
                    {0, 0, (uint32_t)DH * Heads, 1},
                    -1, 0, true
                );
            }

            if (col < 3){
                size_t kv_offset = (size_t)round * 64 * DH * kv_heads + (size_t)col * DH;
                seq->npu_dma_memcpy_nd(
                    2, 2,
                    MM2S, IT[col],
                    (npu_bd_id)(bd_offset + 2), it_channel_1,
                    {0, 0, 0, (uint32_t)kv_offset},
                    {1, 1, (uint32_t)64, (uint32_t)DH * 2},
                    {0, 0, (uint32_t)DH * kv_heads, 1},
                    -1, 0, true
                );
                seq->npu_dma_memcpy_nd(
                    2, 3,
                    MM2S, IT[col],
                    (npu_bd_id)(bd_offset + 3), it_channel_1,
                    {0, 0, 0, (uint32_t)kv_offset},
                    {1, 1, (uint32_t)64, (uint32_t)DH * 2},
                    {0, 0, (uint32_t)DH * kv_heads, 1},
                    -1, 0, true
                );
            }
            else if (col == 3){
                size_t kv_offset = (size_t)round * 64 * DH * kv_heads + 4 * DH;
                seq->npu_dma_memcpy_nd(
                    2, 2,
                    MM2S, IT[col],
                    (npu_bd_id)(bd_offset + 2), it_channel_1,
                    {0, 0, 0, (uint32_t)kv_offset},
                    {1, 1, (uint32_t)64, (uint32_t)DH},
                    {0, 0, (uint32_t)DH * kv_heads, 1},
                    -1, 0, true
                );
                seq->npu_dma_memcpy_nd(
                    2, 3,
                    MM2S, IT[col],
                    (npu_bd_id)(bd_offset + 3), it_channel_1,
                    {0, 0, 0, (uint32_t)kv_offset},
                    {1, 1, (uint32_t)64, (uint32_t)DH},
                    {0, 0, (uint32_t)DH * kv_heads, 1},
                    -1, 0, true
                );
            }
        }
        if (round > 0){
            for (int col = 0; col < 4; col++){
                seq->npu_dma_wait(
                    IT[col],
                    MM2S,
                    it_channel_1
                );
                seq->npu_dma_wait(
                    IT[col],
                    MM2S,
                    it_channel_1
                );
            } // end col
        } // end round > 0
    }
    for (int col = 0; col < 4; col++){
        seq->npu_dma_wait(
            IT[col],
            S2MM,
            it_channel_0
        );
        seq->npu_dma_wait(
            IT[col],
            MM2S,
            it_channel_1
        );
        seq->npu_dma_wait(
            IT[col],
            MM2S,
            it_channel_1
        );
    }
    seq->cmds2seq();
}

void gen_mm_seq(npu_sequence* seq, const uint32_t M, const uint32_t K, const uint32_t N, const uint32_t m, const uint32_t k, const uint32_t n, const uint32_t weight_offset, int32_t rtp_act){

    constexpr npu_tiles IT[4] = {IT2, IT3, IT4, IT5};
    constexpr int total_cols = 4;
    constexpr int total_rows = 4;
    assert(N % 256 == 0);
    seq->clear_cmds();

    constexpr int rtp_address = 9728;  // mm tiles (input_with_addresses.mlir)
    int K_div_k = K / k;
    const int M_div_m = M / m / total_rows;
    const int N_div_n  = N / n / total_cols; 
    for (int row = 2; row < 6; row++){
        for (int col = 2; col < total_cols + 2; col++){
            npu_tiles tile = get_tile(row, col);
            seq->rtp_write(tile, rtp_address + 0, K_div_k);
            seq->rtp_write(tile, rtp_address + 4, rtp_act);
        }
    }

    for (int j = 0; j < N_div_n; j++){
        for (int i = 0; i < M_div_m; i++){
            int round = j * M_div_m + i;
            int bd_offset = (round % 2) * 8;
            int y_offset = i * m * N * total_rows + j * n * total_cols;
            int x_offset = i * m * K * total_rows;
            // W is bfp16ebs8 bytes: strip s = j*total_cols + c (64 n rows) is contiguous, 72*K bytes
            const uint32_t strip_bytes = (uint32_t)n * K / 8 * 9;
            uint32_t w_offset = j * total_cols * strip_bytes;
            npu_bd_id y_bd = npu_bd_id(0 + bd_offset);
            npu_bd_id x_bd = npu_bd_id(1 + bd_offset);
            npu_bd_id w_bd = npu_bd_id(2 + bd_offset);
            // send w
            for (int c = 0; c < total_cols; c++){
                seq->npu_dma_memcpy_nd(
                    1, 2,
                    MM2S, IT[c],
                    w_bd, it_channel_1,
                    {0, 0, 0, w_offset + c * strip_bytes},
                    {1, 1, strip_bytes / 2304, 2304},
                    {0, 0, 2304, 1},
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