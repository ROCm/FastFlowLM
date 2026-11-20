// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __SEQ_GEN_HPP__
#define __SEQ_GEN_HPP__
#pragma once
#include "npu_utils/npu_utils.hpp"
#include "weights_desc.hpp"

// rtp_value tells the kernel which quantized weight format to dequantize:
//   flm_q41 -> 1, flm_q40 -> 2, flm_q81 -> 3, flm_q80 -> 4
void gen_dequant_seq(npu_sequence* seq, const uint32_t Dout, const uint32_t Din, const uint32_t desired_m, const uint32_t columns, const size_t w_offset, const int rtp_transpose_value, flm_dtype_t dtype){
    constexpr npu_tiles IT[8] = {IT0, IT1, IT2, IT3, IT4, IT5, IT6, IT7};
    constexpr int CT_lock_address_base = 0x000001F000;
    constexpr int CT_rtp_sync_lock_id = 6;

    int rtp_value;
    switch (dtype){
        case flm_q41: rtp_value = 1; break;
        case flm_q40: rtp_value = 2; break;
        case flm_q81: rtp_value = 3; break;
        case flm_q80: rtp_value = 4; break;
        default: assert(false && "unsupported dtype for gen_dequant_seq");
    }

    static constexpr int m = QXNX_ROW_BLOCK_SIZE;
    static constexpr int k = QXNX_COL_BLOCK_SIZE;
    const int desired_k = 512;

    // byte size of one quantized block expressed in bf16 units (the DMA transfer unit)
    uint32_t a_block_size = get_quantization_byte_size(m * k, dtype) / sizeof(bf16);
    const uint32_t blocks_per_row = Din / k;

    const uint32_t cores = columns * 4;
    assert(Dout / m / cores > 0);
    assert(Dout % (m * cores) == 0);

    seq->clear_cmds();

    for (int row = 2; row < 6; row++){
        for (int col = 0; col < columns; col++){
            npu_tiles tile = get_tile(row, col);
            seq->rtp_write(tile, 9216, rtp_value);
            seq->rtp_write(tile, 54272, rtp_transpose_value);
            seq->rtp_write(tile, 9280, Din);
            seq->rtp_write(tile, 54336, Dout / cores);
            seq->rtp_write(tile, CT_lock_address_base + 16 * CT_rtp_sync_lock_id, 1); // set lock to 1
        }
    }

    uint32_t y_base;
    if(desired_m == 64){
        y_base = 64 * Din;
    }
    else if(desired_m == 128){
        y_base = 64 * desired_k;
    }
    for (size_t round = 0; round < Dout / m / cores; round++){
        uint32_t bd_offset = (round % 2) * 8;
        for (uint32_t col = 0; col < columns; col++){
            seq->npu_dma_memcpy_nd(
                sizeof(bf16),
                1,
                MM2S,
                IT[col],
                npu_bd_id(bd_1 + bd_offset),
                it_channel_0,
                {0, 0, 0, (uint32_t)((round * cores + col * 4) * a_block_size * blocks_per_row + (uint32_t)w_offset / sizeof(bf16))},
                {1, 1, 1, 2 * blocks_per_row * a_block_size},
                {0, 0, 0, 1},
                -1, 0, false, aggressive_cache
            );
            seq->npu_dma_memcpy_nd(
                sizeof(bf16),
                1,
                MM2S,
                IT[col],
                npu_bd_id(bd_2 + bd_offset),
                it_channel_1,
                {0, 0, 0, (uint32_t)((round * cores + col * 4 + 2) * a_block_size * blocks_per_row + (uint32_t)w_offset / sizeof(bf16))},
                {1, 1, 1, 2 * blocks_per_row * a_block_size},
                {0, 0, 0, 1},
                -1, 0, false, aggressive_cache
            );
            seq->npu_dma_memcpy_nd(
                sizeof(bf16),
                0,
                S2MM,
                IT[col],
                npu_bd_id(bd_0 + bd_offset),
                it_channel_0,
                {0, 0, 0, (uint32_t)((round * cores + col * 4) * m * Din)},
                {Din / desired_k, desired_k / k, 64, k},
                {desired_m * desired_k, k, desired_k, 1},
                -1, 0, true, aggressive_cache
            );
            seq->npu_dma_memcpy_nd(
                sizeof(bf16),
                0,
                S2MM,
                IT[col],
                npu_bd_id(bd_3 + bd_offset),
                it_channel_1,
                {0, 0, 0, (uint32_t)((round * cores + col * 4) * m * Din + y_base)},
                {Din / desired_k, desired_k / k, 64, k},
                {desired_m * desired_k, k, desired_k, 1},
                -1, 0, true, aggressive_cache
            );
        }
        if (round > 0){
            for (uint32_t col = 0; col < columns; col++){
                seq->npu_dma_wait(
                    IT[col],
                    S2MM,
                    it_channel_0
                );
                seq->npu_dma_wait(
                    IT[col],
                    S2MM,
                    it_channel_1
                );
            }
        }
    }
    for (uint32_t col = 0; col < columns; col++){
        seq->npu_dma_wait(
            IT[col],
            S2MM,
            it_channel_0
        );
        seq->npu_dma_wait(
            IT[col],
            S2MM,
            it_channel_1
        );
    }
    seq->cmds2seq();
}

void gen_mm_seq(npu_sequence* seq, const uint32_t M, const uint32_t K, const uint32_t N, const uint32_t n, const uint32_t weight_offset, int32_t rtp_act){

    constexpr npu_tiles IT[8] = {IT0, IT1, IT2, IT3, IT4, IT5, IT6, IT7};
    constexpr int CT_lock_address_base = 0x000001F000;
    constexpr int CT_rtp_sync_lock_id = 6;
    constexpr int total_cols = 8;
    constexpr int total_rows = 4;

    constexpr int m = 64;
    constexpr int k = 512;
    int K_div_k = K / k;
    assert(N % (n * 4) == 0);
    
    seq->clear_cmds();

    int K_div_k_address;
    int rtp_act_address;

    K_div_k_address = (n * 4 ==256) ? 43008 : 56832;
    rtp_act_address = (n * 4 ==256) ? 57472 : 56896;

    for (int row = 2; row < 6; row++){
        for (int col = 0; col < total_cols; col++){
            npu_tiles tile = get_tile(row, col);
            seq->rtp_write(tile, K_div_k_address, K_div_k);
            seq->rtp_write(tile, rtp_act_address, rtp_act);
            seq->rtp_write(tile, CT_lock_address_base + 16 * CT_rtp_sync_lock_id, 1); // set lock to 1
        }
    }

    // One tiled MM pass. `split` selects between the two layouts:
    //   split == true : 4 active columns, the weights are broadcast to both
    //                   column groups (BDs 2 and 5) and the second group reads
    //                   x/y from the M*K/2 / M*N/2 half of DDR.
    //   split == false: 8 active columns, a single weight BD per column and the
    //                   second group is the adjacent 4*n slice of the same row.
    auto run_tiles = [&](bool split, int M_div_m, int N_div_n, int y_base, uint32_t w_base){
        const int n_cols_used = split ? 4 : total_cols;
        const int i_stride = split ? 2 : 1;
        const int y_bd1_extra = split ? (int)(m * N * total_rows) : (int)(4 * n);
        const int x_bd1_extra = split ? (int)(m * K * total_rows) : 0;
        for (int j = 0; j < N_div_n; j++){
            for (int i = 0; i < M_div_m; i++){
                int round = j * M_div_m + i;
                int bd_offset = (round % 2) * 8;
                int y_offset = i * i_stride * m * N * total_rows + j * n * n_cols_used + y_base;
                int x_offset = i * i_stride * m * K * total_rows;
                uint32_t w_offset = j * n * K * n_cols_used + w_base;
                npu_bd_id y_bd_0 = npu_bd_id(0 + bd_offset);
                npu_bd_id y_bd_1 = npu_bd_id(3 + bd_offset);
                npu_bd_id x_bd_0 = npu_bd_id(1 + bd_offset);
                npu_bd_id x_bd_1 = npu_bd_id(4 + bd_offset);
                npu_bd_id w_bd_0 = npu_bd_id(2 + bd_offset);
                npu_bd_id w_bd_1 = npu_bd_id(5 + bd_offset);
                // send w
                for (int c = 0; c < n_cols_used; c++){
                    seq->npu_dma_memcpy_nd(
                        2, 2,
                        MM2S, IT[c],
                        w_bd_0, it_channel_1,
                        {0, 0, 0, w_offset + c * n * K},
                        {1, 1, 1, (uint32_t)K_div_k * n * k},
                        {0, 0, 0, 1},
                        -1, 0, false
                    );
                    if (split){
                        seq->npu_dma_memcpy_nd(
                            2, 2,
                            MM2S, IT[c + 4],
                            w_bd_1, it_channel_1,
                            {0, 0, 0, w_offset + c * n * K},
                            {1, 1, 1, (uint32_t)K_div_k * n * k},
                            {0, 0, 0, 1},
                            -1, 0, false
                        );
                    }
                }
                for (int r = 0; r < total_rows; r++){
                    // receive y
                    seq->npu_dma_memcpy_nd(
                        2, 0,
                        S2MM, IT[r],
                        y_bd_0, it_channel_0,
                        {0, 0, 0, y_offset + r * m * N},
                        {1, 1, m, n * 4},
                        {0, 0, N, 1},
                        -1, 0, true
                    );
                    seq->npu_dma_memcpy_nd(
                        2, 0,
                        S2MM, IT[r + 4],
                        y_bd_1, it_channel_0,
                        {0, 0, 0, y_offset + r * m * N + y_bd1_extra},
                        {1, 1, m, n * 4},
                        {0, 0, N, 1},
                        -1, 0, true
                    );
                    // send x
                    seq->npu_dma_memcpy_nd(
                        2, 1,
                        MM2S, IT[r],
                        x_bd_0, it_channel_0,
                        {0, 0, 0, x_offset + r * m * K},
                        {1, (uint32_t)K_div_k, m, k},
                        {0, k, K, 1},
                        -1, 0, false
                    );
                    seq->npu_dma_memcpy_nd(
                        2, 1,
                        MM2S, IT[r + 4],
                        x_bd_1, it_channel_0,
                        {0, 0, 0, x_offset + r * m * K + x_bd1_extra},
                        {1, (uint32_t)K_div_k, m, k},
                        {0, k, K, 1},
                        -1, 0, false
                    );
                }
                if (round > 0){
                    for (int r = 0; r < total_rows; r++){
                        seq->npu_dma_wait(IT[r], S2MM, it_channel_0);
                        seq->npu_dma_wait(IT[r + 4], S2MM, it_channel_0);
                    }
                }
            }
        }
        for (int r = 0; r < total_rows; r++){
            seq->npu_dma_wait(IT[r], S2MM, it_channel_0);
            seq->npu_dma_wait(IT[r + 4], S2MM, it_channel_0);
        }
    };

    // there will be three cases:
    // 1) N < (n * 8), in this case we only use 4 columns
    // 2) N % (n * 8) == 0, in this case we use 8 columns
    // 3) N % (n * 8) != 0 and N > (n * 8), in this case we use 8 columns but only part of the second 4 columns
    // need some bool parameters to indicate which case it is, but for now we can just check N at compile time since N is a constant in this test
    const int M_div_m = M / m / total_rows;
    const int D_main  = (N / (n * 8)) * (n * 8);  // 8-column region (multiple of (n * 8))
    const int D_rem   = N - D_main;       // 0 or (n * 4)
    if (D_main > 0){
        run_tiles(false, M_div_m, D_main / n / total_cols, 0, weight_offset);
    }
    if (D_rem > 0){
        run_tiles(true, M_div_m / 2, D_rem / n / 4, D_main, weight_offset + D_main * K);
    }
    seq->cmds2seq();
}
#endif
