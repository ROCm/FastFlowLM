// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __SEQ_GEN_HPP__
#define __SEQ_GEN_HPP__
#pragma once
#include "npu_utils/npu_utils.hpp"
#include "weights_desc.hpp"

// Six-column variant of dequant_mm_Q4_1_256x512x512's sequence.  The two columns
// it gives up are for the attention half of the fused prefill overlay.
//
// Geometry.  A core still owns one (desired_m x desired_n) output tile; the array
// is 4 core rows (token-blocks) x 6 columns (N-slabs), so a round covers
// 4 * 64 = 256 tokens and 6 * 64 = 384 output columns.
//
// The drain is the part that is not just "the 8-column build with two columns
// deleted".  24 core lanes have to land in 6 memtiles, and a memtile can only
// take 4 y lanes (its other two s2mm channels are x and qw).  So a memtile no
// longer holds one token-block's worth of N; it holds a 2x2 tile of
// (token-block, N-slab) -- memtile g = 3a + b owns token-blocks {2a, 2a+1} and
// N-slabs {2b, 2b+1} -- and emits [2 token-blocks][desired_m rows][2 * desired_n].
// See `y_pair_2x2` in iron/dequant_mm_tile.py.
//
// Tail rounds.  Dout / 128 is not generally a multiple of 3, so the last round's
// window is slid back to end on the last slab-pair instead of running past it.
// That recomputes up to two slab-pairs, writing byte-identical results over the
// previous round's output.
//
// These are core-local buffer addresses assigned by aie-assign-buffer-addresses,
// so they move whenever the core's data layout changes.  After any bitstream
// rebuild, re-read them from
//   build/bitstream/from_iron/dequant_mm.mlir.prj/input_with_addresses.mlir
// (grep for K_div_k_2_0 / rtp_act_2_0 and K_div_k_3_0 / rtp_act_3_0).
static constexpr uint32_t CT_rtp_address_even_row  = 41984;
static constexpr uint32_t act_rtp_address_even_row = 54400;
static constexpr uint32_t CT_rtp_address_odd_row   = 53376;
static constexpr uint32_t act_rtp_address_odd_row  = 8320;

void gen_dequant_mm_512(npu_sequence* seq, const uint32_t L, const uint32_t Din, const uint32_t Dout, const size_t weight_offset, const size_t out_offset, const int rtp_act){
    static constexpr npu_tiles IT[8] = {IT0, IT1, IT2, IT3, IT4, IT5, IT6, IT7};

    static constexpr uint32_t desired_m = 64;
    static constexpr uint32_t desired_k = 512;
    static constexpr uint32_t desired_n = 64;

    static constexpr uint32_t columns = 6;
    static constexpr uint32_t pairs_per_round = columns / 2; // 3 slab-pairs

    const int K_div_k = Din / desired_k;

    assert(Din % desired_k == 0);
    assert(L % (desired_m * 4) == 0);
    assert(Dout % (desired_n * 2) == 0);

    const uint32_t pairs = Dout / (desired_n * 2);
    assert(pairs >= pairs_per_round && "Dout must be at least 384");

    seq->clear_cmds();

    const int M_div_m = L / (desired_m * 4);
    const int rounds_n = (pairs + pairs_per_round - 1) / pairs_per_round;

    for (int row = 2; row < 6; row += 2){
        for (uint32_t col = 0; col < columns; col++){
            npu_tiles tile = get_tile(row, col);
            seq->rtp_write(tile, CT_rtp_address_even_row, K_div_k);
            seq->rtp_write(tile, act_rtp_address_even_row, rtp_act);
        }
    }
    for (int row = 3; row < 6; row += 2){
        for (uint32_t col = 0; col < columns; col++){
            npu_tiles tile = get_tile(row, col);
            seq->rtp_write(tile, CT_rtp_address_odd_row, K_div_k);
            seq->rtp_write(tile, act_rtp_address_odd_row, rtp_act);
        }
    }

    int round = 0;
    for (int i = 0; i < M_div_m; i++){
        const uint32_t x_offset = i * desired_m * 4 * Din;
        const uint32_t m_offset = i * desired_m * 4 * Dout;
        for (int r = 0; r < rounds_n; r++){
            // slide the last window back rather than running off the end
            const uint32_t pair_base = (uint32_t)std::min<int>(r * pairs_per_round, (int)pairs - (int)pairs_per_round);
            const int bd_offset = (round % 2) * 8;

            const npu_bd_id y_bd = npu_bd_id(0 + bd_offset);
            const npu_bd_id x_bd = npu_bd_id(1 + bd_offset);
            const npu_bd_id w_bd = npu_bd_id(2 + bd_offset);

            // weights: column c takes N-slab pair_base * 2 + c
            for (uint32_t c = 0; c < columns; c++){
                const uint32_t slab = pair_base * 2 + c;
                seq->npu_dma_memcpy_nd(
                    sizeof(char), 2,
                    MM2S, IT[c],
                    w_bd, it_channel_1,
                    {0, 0, 0, (uint32_t)(weight_offset + get_quantization_byte_size((size_t)slab * desired_n * Din, flm_q41))},
                    {1, 1, 1, (uint32_t)get_quantization_byte_size(desired_n * Din, flm_q41)},
                    {0, 0, 0, 1},
                    -1, 0, false, aggressive_cache
                );
            }

            // x: one broadcast per core row, out of the first four mm shims
            for (uint32_t rb = 0; rb < 4; rb++){
                seq->npu_dma_memcpy_nd(
                    2, 1,
                    MM2S, IT[rb],
                    x_bd, it_channel_0,
                    {0, 0, 0, x_offset + rb * desired_m * Din},
                    {1, (uint32_t)K_div_k, desired_m, desired_k},
                    {0, desired_k, Din, 1},
                    -1, 0, false
                );
            }

            // y: memtile g = 3a + b drains token-blocks {2a, 2a+1} x slab-pair
            // (pair_base + b) out of its own shim, as 2 blocks of desired_m rows.
            for (uint32_t a = 0; a < 2; a++){
                for (uint32_t b = 0; b < pairs_per_round; b++){
                    const uint32_t g = a * pairs_per_round + b;
                    const uint32_t base = out_offset + m_offset
                                        + (size_t)2 * a * desired_m * Dout
                                        + (pair_base + b) * 2 * desired_n;
                    seq->npu_dma_memcpy_nd(
                        2, 0,
                        S2MM, IT[g],
                        y_bd, it_channel_0,
                        {0, 0, 0, base},
                        {1, 2, desired_m, 2 * desired_n},
                        {0, desired_m * Dout, Dout, 1},
                        -1, 0, true
                    );
                }
            }

            if (round > 0){
                for (uint32_t g = 0; g < columns; g++){
                    seq->npu_dma_wait(IT[g], S2MM, it_channel_0);
                }
            }
            round++;
        }
    }
    for (uint32_t g = 0; g < columns; g++){
        seq->npu_dma_wait(IT[g], S2MM, it_channel_0);
    }
    seq->cmds2seq();
}

#endif
