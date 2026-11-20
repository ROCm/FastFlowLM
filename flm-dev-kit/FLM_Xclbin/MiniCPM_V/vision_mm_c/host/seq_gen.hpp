// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#ifndef __SEQ_GEN_HPP__
#define __SEQ_GEN_HPP__
#pragma once
#include <cassert>
#include <cstdint>
#include "npu_utils/npu_utils.hpp"

/// Build the command sequence for one (M, K, N) matmul on the vision MM array, design C.
///
/// Array mapping: 8 columns split M, 4 rows split N. One round covers
/// (m * 8) x (n * 4) of the output (512 x 384) and the host walks the output in
/// those tiles: j over N strips, i over M tiles.
///
///   x -> IT[c] ch0  for c in [0, 8)   tokens   i*tile_M + c*m, +m          (all K)
///   w -> IT[r] ch1  for r in [0, 4)   channels j*tile_N + r*n, +n          (all K)
///   y <- IT[c] ch0  for c in [0, 8)   a (m, tile_N) block of FULL token rows
///
/// y rows are tile_N = 384 contiguous elements (768 B) in DDR, so the write-back is
/// long row segments, not the 96 B pieces of the old mapping.
///
/// One BD per (strip, operand) covers up to 32 consecutive rounds: the first (d3)
/// dimension of a shim BD is a repeat count (w: stride 0; x: stride tile_M rows),
/// y adds a dimension for the round. This takes the per-round command stream out of
/// the critical path (~28 commands per round before). The y / x repeat strides must
/// fit a BD step field (20 bits of 32-bit words = 2^21 bf16 elements); a larger stride
/// does NOT error, it wraps and the data lands in the wrong rows (measured), so such
/// shapes are issued one round per group.
///
/// DDR layout: x is (M, K) row-major; w is N bias elements followed by (N, K)
/// row-major weights; y is (M, N) row-major. Same as every vision_mm design.
///
/// @param rtp_address byte address of the RTP buffer in the compute tile. This is
///        assigned by the compiler and changes whenever the design changes, so it
///        must be read from the build (tools/extract_rtp_addr.sh) rather than
///        hardcoded - a stale address does not fault, it hangs in wait().
/// @param rtp_act 0 = none, 1 = SiLU, 2 = GELU.
inline void gen_mm_seq(npu_sequence* seq, const uint32_t M, const uint32_t K, const uint32_t N,
                       const uint32_t m, const uint32_t k, const uint32_t n,
                       const uint32_t weight_offset, int32_t rtp_act, uint32_t rtp_address){
    constexpr npu_tiles IT[8] = {IT0, IT1, IT2, IT3, IT4, IT5, IT6, IT7};
    constexpr uint32_t total_cols = 8;   // split M
    constexpr uint32_t total_rows = 4;   // split N
    const uint32_t tile_M = m * total_cols, tile_N = n * total_rows;
    assert(M % tile_M == 0 && "M must be padded up to m * total_cols");
    assert(N % tile_N == 0 && "N must be padded up to n * total_rows");
    assert(K % k == 0 && "K must be padded up to the k tile");
    seq->clear_cmds();
    const uint32_t K_div_k = K / k, M_div_m = M / tile_M, N_div_n = N / tile_N;
    for (int row = 2; row < 2 + (int)total_rows; row++)
        for (int col = 0; col < (int)total_cols; col++){
            npu_tiles tile = get_tile(row, col);
            seq->rtp_write(tile, rtp_address, M_div_m);
            seq->rtp_write(tile, rtp_address + 4, K_div_k);
            seq->rtp_write(tile, rtp_address + 8, rtp_act);
        }
    constexpr uint64_t kMaxStrideElems = (1ull << 21) - 1;
    const uint32_t kRepeat = ((uint64_t)tile_M * K > kMaxStrideElems || (uint64_t)tile_M * N > kMaxStrideElems) ? 1 : 32;
    uint32_t group = 0;
    for (uint32_t j = 0; j < N_div_n; j++){
        // Bias prologue for this N strip (row r's n channels). The kernel reads the bias out of the
        // leading n elements of the first w chunk, then drains the rest of the k chunks, so this ships
        // n*k elements even though only n are used.
        const npu_bd_id bias_bd = npu_bd_id(6 + (j % 2));
        for (uint32_t r = 0; r < total_rows; r++){
            seq->npu_dma_memcpy_nd(2, 2, MM2S, IT[r], bias_bd, it_channel_1,
                {0, 0, 0, weight_offset + j * tile_N + r * n}, {1, 1, 1, n * k}, {0, 0, 0, 1}, -1, 0, false);
        }
        for (uint32_t i0 = 0; i0 < M_div_m; i0 += kRepeat, group++){
            const uint32_t reps = (M_div_m - i0 < kRepeat) ? (M_div_m - i0) : kRepeat;
            const uint32_t p = (group % 2) * 8;
            for (uint32_t r = 0; r < total_rows; r++){
                uint32_t w_offset = weight_offset + N + (j * tile_N + r * n) * K;
                seq->npu_dma_memcpy_nd(2, 2, MM2S, IT[r], (npu_bd_id)(2 + p), it_channel_1,
                    {0, 0, 0, w_offset}, {reps, K_div_k, n, k}, {0, k, K, 1}, -1, 0, false);
            }
            for (uint32_t c = 0; c < total_cols; c++){
                uint32_t y_offset = (i0 * tile_M + c * m) * N + j * tile_N;
                seq->npu_dma_memcpy_nd(2, 0, S2MM, IT[c], (npu_bd_id)(0 + p), it_channel_0,
                    {0, 0, 0, y_offset}, {1, reps, m, tile_N}, {0, tile_M * N, N, 1}, -1, 0, true);
            }
            for (uint32_t c = 0; c < total_cols; c++){
                uint32_t x_offset = (i0 * tile_M + c * m) * K;
                seq->npu_dma_memcpy_nd(2, 1, MM2S, IT[c], (npu_bd_id)(1 + p), it_channel_0,
                    {0, 0, 0, x_offset}, {reps, K_div_k, m, k}, {tile_M * K, k, K, 1}, -1, 0, false);
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
