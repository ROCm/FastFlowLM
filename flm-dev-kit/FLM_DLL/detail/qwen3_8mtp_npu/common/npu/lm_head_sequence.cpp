// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file common/npu/lm_head_sequence.cpp
/// \brief The lm_head pass, descriptor for descriptor.
///
/// Every descriptor below is `generate_seq()`'s from
/// FLM_Xclbin/Qwen3_8/lm_head_npu_bin/host/host.cpp, with two deliberate
/// divergences:
///
///   * the upstream `assert(VOCAB_SIZE % M_PER_ROUND == 0)` is promoted to a
///     throw. An assert is compiled out of the release .so, and this one is a
///     shape that would emit a short sequence and hang the dispatch rather
///     than fail it.
///   * the weight offsets are `size_t` through to the cast. Upstream's
///     `W_PER_ROUND` is a `size_t` already but `w_col_offset` reaches 676 M
///     bf16 elements on this checkpoint, which is close enough to the 32-bit
///     ceiling that the intermediate type is worth pinning down.
///
/// Nothing else moves. The packet ids, the channel assignment and the
/// LOOKAHEAD depth are all properties of the bitstream.
#include "common/npu/lm_head_sequence.hpp"

#include <stdexcept>
#include <string>

namespace q38npu {

using namespace lm_head_tiles;

namespace {

/// The eight shims, indexable. Upstream's `ITs[8]`.
constexpr npu_tiles IT[COLUMNS] = {IT0, IT1, IT2, IT3, IT4, IT5, IT6, IT7};

/// Kernel argument order, matching `app(y, w, x)`.
constexpr int y_arg_id = 0;
constexpr int w_arg_id = 1;
constexpr int x_arg_id = 2;

/// Packet ids for the two flows that share IT0's MM2S 0. Must match
/// X_PKT_ID / W_PKT_ID in the design's iron/q4k_mm.py. Every other shim MM2S
/// port is circuit-switched.
constexpr int X_PKT_ID = 0;
constexpr int W_PKT_ID = 1;
constexpr int NO_PKT   = -1;

} // namespace

void gen_lm_head_seq(npu_sequence* seq, uint32_t vocab_padded, uint32_t D) {
    if (!seq) throw std::runtime_error("gen_lm_head_seq: no sequence");
    if (vocab_padded == 0 || vocab_padded % (uint32_t)M_PER_ROUND != 0)
        throw std::runtime_error(
            "gen_lm_head_seq: vocab_padded is " + std::to_string(vocab_padded) +
            ", not a positive multiple of " + std::to_string(M_PER_ROUND) +
            " -- use lm_head_vocab_padded()");
    if (D == 0 || D % 256 != 0)
        throw std::runtime_error(
            "gen_lm_head_seq: D is " + std::to_string(D) +
            ", not a multiple of the 256-wide Q8_0 chunk");

    const int rounds = (int)(vocab_padded / (uint32_t)M_PER_ROUND);

    // In bf16 ELEMENTS, because every descriptor below is emitted at
    // sizeof(bf16) granularity -- the weights are bytes, but the sequencer
    // only ever addresses them through this one element size.
    const size_t W_PER_COL    = (size_t)M_PER_COL * (size_t)D * Q80_NUM /
                                Q80_DEN / sizeof(bf16);
    const size_t W_PER_ROUND  = W_PER_COL * (size_t)COLUMNS;
    const size_t W_PER_STREAM = W_PER_COL / 2;

    seq->clear_cmds();

    // x in, once for the whole dispatch: D hidden-state elements followed by D
    // norm-weight elements. The kernel norms the first half in place against
    // the second, so both have to be rewritten every dispatch -- the first
    // because the device destroyed it, the second because it is cheap enough
    // (10 KB against 1.35 GB) that tracking which norm is loaded would only
    // add a way to get it wrong.
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        x_arg_id,
        MM2S,
        IT[0],
        npu_bd_id(bd_0),
        it_channel_0,
        {0, 0, 0, 0},
        {1, 1, 1, D * 2},
        {0, 0, 0, 1},
        X_PKT_ID, 0, false, aggressive_cache
    );

    for (int r = 0; r < rounds; r++) {
        // Four sets of shim BDs in flight, so the BDs a round uses are picked
        // modulo LOOKAHEAD + 1.
        const int bd_offset = (r % (LOOKAHEAD + 1)) * 3;
        const npu_bd_id bd_y  = npu_bd_id(bd_1 + bd_offset);
        const npu_bd_id bd_wa = npu_bd_id(bd_2 + bd_offset);
        const npu_bd_id bd_wb = npu_bd_id(bd_3 + bd_offset);

        for (int col = 0; col < COLUMNS; col++) {
            const size_t   w_col_offset = (size_t)r * W_PER_ROUND +
                                          (size_t)col * W_PER_COL;
            const uint32_t y_offset     = (uint32_t)r * (uint32_t)M_PER_ROUND +
                                          (uint32_t)col * (uint32_t)M_PER_COL;

            seq->npu_dma_memcpy_nd(
                sizeof(bf16),
                y_arg_id,
                S2MM,
                IT[col],
                bd_y,
                it_channel_0,
                {0, 0, 0, y_offset},
                {1, 1, 1, (uint32_t)M_PER_COL},
                {0, 0, 0, 1},
                NO_PKT, 0, true, aggressive_cache
            );

            // Group a (cores 0/1) on channel 1, group b (cores 2/3) on
            // channel 0. Channel 0 of IT0 is the one shared with the x
            // broadcast, so only that transfer needs a packet id.
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

            // A round only has to wait for the y of the round three back
            // rather than the one immediately before, which keeps three rounds
            // of weight transfers queued ahead of the sequencer instead of
            // one. On a design whose whole cost is the weight stream, that is
            // the difference between saturating DDR and not.
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
