// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __SEQ_GEN_HPP__
#define __SEQ_GEN_HPP__
#pragma once
#include "npu_utils/npu_utils.hpp"

/// Build the command sequence for one (M, K, N) matmul on the vision MM array.
///
/// Array mapping: 4 rows split M, 8 cols split N. One round therefore covers
/// (m * total_rows) x (n * total_cols) of the output, and the host walks the
/// output in those tiles: j over N, i over M.
///
///   x -> IT[r] ch0  for r in [0, total_rows)   rows  i*m*total_rows + r*m
///   w -> IT[c] ch1  for c in [0, total_cols)   cols  j*n*total_cols + c*n
///   y <- IT[c] ch0  for c in [0, total_cols)   a (m*total_rows, n) column chunk
///
/// This is the transpose of the SmolVLA scaffold, which split M across the 8
/// columns. Rows-for-M gives an M granularity of m*total_rows rather than
/// m*total_cols, which matters for the merger layers that run on L/4 rows.
///
/// DDR layout: x is (M, K) row-major; w is N bias elements followed by (N, K)
/// row-major weights; y is (M, N) row-major.
///
/// @param rtp_address byte address of the RTP buffer in the compute tile. This is
///        assigned by the compiler and changes whenever the design changes, so it
///        must be read from the build (tools/extract_rtp_addr.sh) rather than
///        hardcoded - a stale address does not fault, it hangs in wait().
/// @param rtp_act 0 = none, 1 = SiLU, 2 = GELU.
inline void gen_mm_seq(npu_sequence* seq, const uint32_t M, const uint32_t K, const uint32_t N,
                       const uint32_t m, const uint32_t k, const uint32_t n,
                       const uint32_t weight_offset, int32_t rtp_act,
                       uint32_t rtp_address){

    constexpr npu_tiles IT[8] = {IT0, IT1, IT2, IT3, IT4, IT5, IT6, IT7};
    constexpr uint32_t total_cols = 8;   // split N
    constexpr uint32_t total_rows = 4;   // split M

    const uint32_t tile_M = m * total_rows;
    const uint32_t tile_N = n * total_cols;

    assert(M % tile_M == 0 && "M must be padded up to m * total_rows");
    assert(N % tile_N == 0 && "N must be padded up to n * total_cols");
    assert(K % k == 0 && "K must be padded up to the k tile");

    seq->clear_cmds();

    const uint32_t K_div_k = K / k;
    const uint32_t M_div_m = M / tile_M;
    const uint32_t N_div_n = N / tile_N;

    // Each mm() invocation on the core runs M_div_m rounds; the core body calls
    // mm() in an endless loop, so N_div_n invocations cover the whole output.
    for (int row = 2; row < 2 + (int)total_rows; row++){
        for (int col = 0; col < (int)total_cols; col++){
            npu_tiles tile = get_tile(row, col);
            seq->rtp_write(tile, rtp_address, M_div_m);
            seq->rtp_write(tile, rtp_address + 4, K_div_k);
            seq->rtp_write(tile, rtp_address + 8, rtp_act);
        }
    }

    for (uint32_t j = 0; j < N_div_n; j++){
        // Bias prologue for this N strip. The kernel reads the bias out of the
        // leading n elements of the first w chunk, then drains the rest of the
        // k chunks, so this ships n*k elements even though only n are used.
        const npu_bd_id bias_bd = npu_bd_id(6 + (j % 2));
        for (uint32_t c = 0; c < total_cols; c++){
            uint32_t b_offset = weight_offset + j * tile_N + c * n;
            seq->npu_dma_memcpy_nd(
                2, 2,
                MM2S, IT[c],
                bias_bd, it_channel_1,
                {0, 0, 0, b_offset},
                {1, 1, 1, n * k},
                {0, 0, 0, 1},
                -1, 0, false
            );
        }

        for (uint32_t i = 0; i < M_div_m; i++){
            const uint32_t round = j * M_div_m + i;
            const uint32_t p = (round % 2) * 8;
            const npu_bd_id y_bd = npu_bd_id(0 + p);
            const npu_bd_id x_bd = npu_bd_id(1 + p);
            const npu_bd_id w_bd = npu_bd_id(2 + p);

            // w: column c owns cols [j*tile_N + c*n, +n). Sent k-tile major so the
            // memtile receives one natural (n, k) block per buffer fill.
            for (uint32_t c = 0; c < total_cols; c++){
                uint32_t w_offset = weight_offset + N + (j * tile_N + c * n) * K;
                seq->npu_dma_memcpy_nd(
                    2, 2,
                    MM2S, IT[c],
                    w_bd, it_channel_1,
                    {0, 0, 0, w_offset},
                    {1, K_div_k, n, k},
                    {0, k, K, 1},
                    -1, 0, false
                );
            }

            // y: column c returns a (tile_M, n) chunk, contiguous rows of n that
            // land at DDR row stride N.
            for (uint32_t c = 0; c < total_cols; c++){
                uint32_t y_offset = i * tile_M * N + j * tile_N + c * n;
                seq->npu_dma_memcpy_nd(
                    2, 0,
                    S2MM, IT[c],
                    y_bd, it_channel_0,
                    {0, 0, 0, y_offset},
                    {1, 1, tile_M, n},
                    {0, 0, N, 1},
                    -1, 0, true
                );
            }

            // x: array row r owns rows [i*tile_M + r*m, +m). Only total_rows shims
            // carry x, unlike w and y which use all total_cols.
            for (uint32_t r = 0; r < total_rows; r++){
                uint32_t x_offset = (i * tile_M + r * m) * K;
                seq->npu_dma_memcpy_nd(
                    2, 1,
                    MM2S, IT[r],
                    x_bd, it_channel_0,
                    {0, 0, 0, x_offset},
                    {1, K_div_k, m, k},
                    {0, k, K, 1},
                    -1, 0, false
                );
            }

            if (round > 0){
                for (uint32_t c = 0; c < total_cols; c++){
                    seq->npu_dma_wait(IT[c], S2MM, it_channel_0);
                }
            }
        }
    }
    for (uint32_t c = 0; c < total_cols; c++){
        seq->npu_dma_wait(IT[c], S2MM, it_channel_0);
    }
    seq->cmds2seq();
}
#endif
