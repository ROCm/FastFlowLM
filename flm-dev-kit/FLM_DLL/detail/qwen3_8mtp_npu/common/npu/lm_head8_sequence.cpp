// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file common/npu/lm_head8_sequence.cpp
/// \brief The L = 8 lm_head pass, descriptor for descriptor.
///
/// Every descriptor below is `generate_seq()`'s from
/// FLM_Xclbin/Qwen3_8/lm_head_8/host/host.cpp, with the same two deliberate
/// divergences the L = 1 port makes:
///
///   * upstream's `assert(VOCAB_SIZE % M_PER_ROUND == 0)` is promoted to a
///     throw. An assert is compiled out of the release .so, and this one is a
///     shape that would emit a short sequence and hang the dispatch rather
///     than fail it.
///   * the weight offsets are `size_t` through to the cast. `w_col_offset`
///     reaches 676 M bf16 elements on this checkpoint, close enough to the
///     32-bit ceiling that the intermediate type is worth pinning down.
///
/// The `FLM_LM8_LOOKAHEAD` override is deliberately *not* ported. It exists in
/// the design's harness to bisect stalls; in the engine it would be a way to
/// hang a shipping dispatch from the environment.
#include "common/npu/lm_head8_sequence.hpp"

#include <stdexcept>
#include <string>

namespace q38npu {

using namespace lm_head8_tiles;

namespace {

/// The eight shims, indexable. Upstream's `ITs[8]`.
constexpr npu_tiles IT[COLUMNS] = {IT0, IT1, IT2, IT3, IT4, IT5, IT6, IT7};

/// Kernel argument order, matching `app(y, w, x)`.
constexpr int y_arg_id = 0;
constexpr int w_arg_id = 1;
constexpr int x_arg_id = 2;

/// Column 0's MM2S 0 is packet-switched; must match `W_PKT_ID` in the design's
/// iron/q4k_mm.py. Unlike L = 1 there is no X_PKT_ID -- x no longer shares a
/// port with a broadcast, it lands in each column's own memtile.
constexpr int W_PKT_ID = 1;
constexpr int NO_PKT   = -1;

} // namespace

void gen_lm_head8_seq(npu_sequence* seq, uint32_t vocab_padded, uint32_t D) {
    if (!seq) throw std::runtime_error("gen_lm_head8_seq: no sequence");
    if (vocab_padded == 0 || vocab_padded % (uint32_t)M_PER_ROUND != 0)
        throw std::runtime_error(
            "gen_lm_head8_seq: vocab_padded is " + std::to_string(vocab_padded) +
            ", not a positive multiple of " + std::to_string(M_PER_ROUND) +
            " -- use lm_head_vocab_padded()");
    if (D == 0 || D % (uint32_t)K_CHUNK != 0)
        throw std::runtime_error(
            "gen_lm_head8_seq: D is " + std::to_string(D) +
            ", not a multiple of the " + std::to_string(K_CHUNK) +
            "-wide Q8_0 chunk");

    const int rounds = (int)(vocab_padded / (uint32_t)M_PER_ROUND);

    // In bf16 ELEMENTS: the weights are bytes, but the sequencer only ever
    // addresses them through this one element size.
    const size_t W_PER_COL    = (size_t)M_PER_COL * (size_t)D *
                                lm_head_tiles::Q80_NUM / lm_head_tiles::Q80_DEN /
                                sizeof(bf16);
    const size_t W_PER_ROUND  = W_PER_COL * (size_t)COLUMNS;
    const size_t W_PER_STREAM = W_PER_COL / 2;

    // The whole activation plus the norm weight lane, every round, to every
    // column.
    const uint32_t X_ELEMS = (uint32_t)X_LANES * D;

    seq->clear_cmds();

    for (int r = 0; r < rounds; r++) {
        // Four BDs per round now that x is per-round, times LOOKAHEAD + 1 = 3
        // rounds in flight, is 12 of the shim's 16.
        const int bd_offset = (r % (LOOKAHEAD + 1)) * 4;
        const npu_bd_id bd_x  = npu_bd_id(bd_0 + bd_offset);
        const npu_bd_id bd_y  = npu_bd_id(bd_1 + bd_offset);
        const npu_bd_id bd_wa = npu_bd_id(bd_2 + bd_offset);
        const npu_bd_id bd_wb = npu_bd_id(bd_3 + bd_offset);

        for (int col = 0; col < COLUMNS; col++) {
            // x goes out ahead of this round's group-a weights, on the same
            // channel. The memtile's S2MM ring is [x, wa, x, wa, ...] and this
            // alternation is the only thing holding the two in phase: one
            // extra or missing x transfer and a weight block is dequantised as
            // an activation for the rest of the dispatch.
            seq->npu_dma_memcpy_nd(
                sizeof(bf16),
                x_arg_id,
                MM2S,
                IT[col],
                bd_x,
                it_channel_1,
                {0, 0, 0, 0},
                {1, 1, 1, X_ELEMS},
                {0, 0, 0, 1},
                NO_PKT, 0, false, aggressive_cache
            );

            const size_t w_col_offset = (size_t)r * W_PER_ROUND +
                                        (size_t)col * W_PER_COL;
            // Each core emits L * 32 values per round instead of 32, so a
            // column's slice is L times wider and lands L times further apart.
            // The DDR image is [round][col][core][token][32]; the caller
            // un-interleaves it.
            const uint32_t y_offset =
                (uint32_t)((size_t)L_TOKENS *
                           ((size_t)r * (size_t)M_PER_ROUND +
                            (size_t)col * (size_t)M_PER_COL));

            seq->npu_dma_memcpy_nd(
                sizeof(bf16),
                y_arg_id,
                S2MM,
                IT[col],
                bd_y,
                it_channel_0,
                {0, 0, 0, y_offset},
                {1, 1, 1, (uint32_t)((size_t)L_TOKENS * (size_t)M_PER_COL)},
                {0, 0, 0, 1},
                NO_PKT, 0, true, aggressive_cache
            );

            // Group a (cores 0/1) on channel 1, group b (cores 2/3) on
            // channel 0. Only column 0's channel 0 is packet-switched.
            seq->npu_dma_memcpy_nd(
                sizeof(bf16),
                w_arg_id,
                MM2S,
                IT[col],
                bd_wa,
                it_channel_1,
                {0, 0, 0, (uint32_t)w_col_offset},
                {1, 1, 1, (uint32_t)W_PER_STREAM},
                {0, 0, 0, 1},
                NO_PKT, 0, false, aggressive_cache
            );
            seq->npu_dma_memcpy_nd(
                sizeof(bf16),
                w_arg_id,
                MM2S,
                IT[col],
                bd_wb,
                it_channel_0,
                {0, 0, 0, (uint32_t)(w_col_offset + W_PER_STREAM)},
                {1, 1, 1, (uint32_t)W_PER_STREAM},
                {0, 0, 0, 1},
                col == 0 ? W_PKT_ID : NO_PKT, 0, false, aggressive_cache
            );

            // A round waits for the y of the round LOOKAHEAD back rather than
            // the one immediately before, which keeps two more rounds of
            // weight transfers queued ahead of the sequencer. On a design
            // whose whole cost is the weight stream, that is the difference
            // between saturating DDR and not.
            if (r > LOOKAHEAD - 1)
                seq->npu_dma_wait(IT[col], S2MM, it_channel_0);
        }
    }

    // Drain the LOOKAHEAD rounds still outstanding.
    for (int col = 0; col < COLUMNS; col++)
        for (int i = 0; i < LOOKAHEAD; i++)
            seq->npu_dma_wait(IT[col], S2MM, it_channel_0);

    seq->cmds2seq();
}

} // namespace q38npu
