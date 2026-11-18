// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __SEQ_GEN_HPP__
#define __SEQ_GEN_HPP__
#pragma once
#include "npu_utils/npu_utils.hpp"

void gen_mm_seq(npu_sequence* seq, const uint32_t M, const uint32_t K, const uint32_t N, const uint32_t m, const uint32_t k, const uint32_t n, const uint32_t weight_offset, int32_t rtp_act){

    constexpr npu_tiles IT[8] = {IT0, IT1, IT2, IT3, IT4, IT5, IT6, IT7};
    constexpr int total_cols = 8;
    constexpr int total_rows = 4;
    int K_div_k = K / k;
    constexpr int CT_lock_address_base = 0x000001F000;
    constexpr int CT_rtp_sync_lock_id = 6;
    assert(N % 512 == 0);
    seq->clear_cmds();

    for (int row = 2; row < 6; row++){
        for (int col = 0; col < total_cols; col++){
            npu_tiles tile = get_tile(row, col);
            seq->rtp_write(tile, 56832, K_div_k);
            seq->rtp_write(tile, 56896, rtp_act);
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
        // When split, the two column groups interleave the M dimension block by
        // block (block = m * total_rows rows): group 0 takes even blocks, group 1
        // the odd block immediately after. So the i stride doubles and the bd1
        // offset is a single block gap rather than the contiguous M/2 half.
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
                        {1, (uint32_t)K_div_k, n, k},
                        {0, k, K, 1},
                        -1, 0, false
                    );
                    if (split){
                        seq->npu_dma_memcpy_nd(
                            2, 2,
                            MM2S, IT[c + 4],
                            w_bd_1, it_channel_1,
                            {0, 0, 0, w_offset + c * n * K},
                            {1, (uint32_t)K_div_k, n, k},
                            {0, k, K, 1},
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

    // N is always a multiple of 512, so N % 1024 is either 0 or 512. The general
    // layout is an 8-column pass over the 1024-aligned region followed by a
    // 4-column split pass over the trailing 256 (if any). The two historical
    // "only 4 columns" (N < 1024) and "only 8 columns" (N % 1024 == 0) cases are
    // just the degenerate ends where one of the two passes has zero width.
    const int M_div_m = M / m / total_rows;
    const int D_main  = (N / 1024) * 1024;  // 8-column region (multiple of 512)
    const int D_rem   = N - D_main;       // 0 or 256
    if (D_main > 0){
        run_tiles(false, M_div_m, D_main / n / total_cols, 0, weight_offset);
    }
    if (D_rem > 0){
        run_tiles(true, M_div_m / 2, D_rem / n / 4, D_main, weight_offset + D_main * K);
    }
    seq->cmds2seq();
}
#endif
