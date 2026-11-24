// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#ifndef __SEQ_GEN_P_HPP__
#define __SEQ_GEN_P_HPP__
#pragma once
#include <cassert>
#include <cstdint>
#include "npu_utils/npu_utils.hpp"

/// Build the command sequence for one (M, K, N) matmul on the vision MM array, design P.
///
/// Design C's array mapping and y path (see vision_mm_c/host/seq_gen.hpp) with x and w
/// as packed bfp16 tiles. A tile pair is 16 rows x 8 k in 144 B: two 72 B tiles (rows
/// 0-7, rows 8-15), each 8 blocks of [exponent byte][8 int8 mantissas], one block per
/// row's 8 consecutive k (pack_bfp16.hpp).
///
/// DDR layouts, all offsets in bytes:
///   x: [M/m tile][K/k tile][m/16 groups][k/8 tile pairs][144 B]  -- 72*K B per m tile
///   w: [N bias, bf16][N/n strip][K/k tile][n/16 groups][k/8][144 B]  -- 108*K B per strip
///   y: (M, N) bf16 row-major, as design C.
/// So a column's x for one round and a row's w for one strip are each one contiguous
/// run. A shim BD dimension wraps at 1024 words, so a run is issued as pieces of
/// kPiece bytes; m tile and k tile blocks are whole multiples of it.
///
/// The bias prologue sends one memtile w buffer (n/16 groups x one k tile) read
/// linearly from the strip's bias: its first 2n bytes are the bias, the rest is
/// whatever follows and is drained unread.
inline void gen_mm_p_seq(npu_sequence* seq, const uint32_t M, const uint32_t K, const uint32_t N,
                         const uint32_t m, const uint32_t k, const uint32_t n,
                         int32_t rtp_act, uint32_t rtp_address){
    constexpr npu_tiles IT[8] = {IT0, IT1, IT2, IT3, IT4, IT5, IT6, IT7};
    constexpr uint32_t total_cols = 8;   // split M
    constexpr uint32_t total_rows = 4;   // split N
    constexpr uint32_t kPiece = 3456;    // bytes, 864 words: divides 72*k and 108*k for k = 384
    const uint32_t tile_M = m * total_cols, tile_N = n * total_rows;
    assert(M % tile_M == 0 && "M must be padded up to m * total_cols");
    assert(N % tile_N == 0 && "N must be padded up to n * total_rows");
    assert(K % k == 0 && "K must be padded up to the k tile");
    const uint32_t x_tile_bytes = m * K / 8 * 9;        // one column's m rows, all K
    const uint32_t w_strip_bytes = n * K / 8 * 9;       // one row's n channels, all K
    const uint32_t bias_bytes = N * 2;
    const uint32_t w_mem_bytes = n * k / 8 * 9;         // one memtile w buffer
    assert(x_tile_bytes % kPiece == 0 && w_strip_bytes % kPiece == 0);
    seq->clear_cmds();
    const uint32_t K_div_k = K / k, M_div_m = M / tile_M, N_div_n = N / tile_N;
    for (int row = 2; row < 2 + (int)total_rows; row++)
        for (int col = 0; col < (int)total_cols; col++){
            npu_tiles tile = get_tile(row, col);
            seq->rtp_write(tile, rtp_address, M_div_m);
            seq->rtp_write(tile, rtp_address + 4, K_div_k);
            seq->rtp_write(tile, rtp_address + 8, rtp_act);
        }
    // Repeat strides must fit a BD step field (2^20 words); a larger one wraps silently.
    constexpr uint64_t kMaxStrideWords = (1ull << 20) - 1;
    const uint32_t kRepeat = ((uint64_t)total_cols * x_tile_bytes / 4 > kMaxStrideWords ||
                              (uint64_t)tile_M * N / 2 > kMaxStrideWords) ? 1 : 32;
    uint32_t group = 0;
    for (uint32_t j = 0; j < N_div_n; j++){
        const npu_bd_id bias_bd = npu_bd_id(6 + (j % 2));
        for (uint32_t r = 0; r < total_rows; r++){
            seq->npu_dma_memcpy_nd(1, 2, MM2S, IT[r], bias_bd, it_channel_1,
                {0, 0, 0, (j * tile_N + r * n) * 2}, {1, 1, 1, w_mem_bytes}, {0, 0, 0, 1}, -1, 0, false);
        }
        for (uint32_t i0 = 0; i0 < M_div_m; i0 += kRepeat, group++){
            const uint32_t reps = (M_div_m - i0 < kRepeat) ? (M_div_m - i0) : kRepeat;
            const uint32_t p = (group % 2) * 8;
            for (uint32_t r = 0; r < total_rows; r++){
                const uint32_t w_offset = bias_bytes + (j * total_rows + r) * w_strip_bytes;
                seq->npu_dma_memcpy_nd(1, 2, MM2S, IT[r], (npu_bd_id)(2 + p), it_channel_1,
                    {0, 0, 0, w_offset}, {reps, 1, w_strip_bytes / kPiece, kPiece}, {0, 0, kPiece, 1}, -1, 0, false);
            }
            for (uint32_t c = 0; c < total_cols; c++){
                uint32_t y_offset = (i0 * tile_M + c * m) * N + j * tile_N;
                seq->npu_dma_memcpy_nd(2, 0, S2MM, IT[c], (npu_bd_id)(0 + p), it_channel_0,
                    {0, 0, 0, y_offset}, {1, reps, m, tile_N}, {0, tile_M * N, N, 1}, -1, 0, true);
            }
            for (uint32_t c = 0; c < total_cols; c++){
                const uint32_t x_offset = (i0 * total_cols + c) * x_tile_bytes;
                seq->npu_dma_memcpy_nd(1, 1, MM2S, IT[c], (npu_bd_id)(1 + p), it_channel_0,
                    {0, 0, 0, x_offset}, {reps, 1, x_tile_bytes / kPiece, kPiece},
                    {total_cols * x_tile_bytes, 0, kPiece, 1}, -1, 0, false);
            }
            // The wait lags a group: this group's descriptors are written while the previous
            // group is still in flight, which is what the two BD banks (p) are for.
            if (group > 0)
                for (uint32_t c = 0; c < total_cols; c++) seq->npu_dma_wait(IT[c], S2MM, it_channel_0);
        }
    }
    for (uint32_t c = 0; c < total_cols; c++) seq->npu_dma_wait(IT[c], S2MM, it_channel_0);
    seq->cmds2seq();
}

#endif
