// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#ifndef __SEQ_GEN_HPP__
#define __SEQ_GEN_HPP__
#pragma once
#include <vector>
#include "npu_utils/npu_utils.hpp"

// Mirror of the engine's FLM_DLL/detail/qwen3_asr_npu/prefill/qwen3_asr_prefill_seq.hpp. Keep the two in sync.
// These are core-local buffer addresses assigned by aie-assign-buffer-addresses; a bitstream build lists them in
// build/xclbins/rtp_addresses.txt (K_div_k_* and rtp_act_*), locked in FLM_Xclbin/rtp_addresses.lock.txt.
static constexpr uint32_t CT_rtp_address_even_row  = 53760;
static constexpr uint32_t act_rtp_address_even_row = 53824;
static constexpr uint32_t CT_rtp_address_odd_row   = 61952;
static constexpr uint32_t act_rtp_address_odd_row  = 62016;

// the weight formats the cores decode (K_div_k[1]); see kernel/common.hpp
static constexpr int MIX_FMT_Q4_1 = 0;
static constexpr int MIX_FMT_Q8_0 = 1;
// one 32-row block row of the mixed layout is Din / 512 (Q4_1) or Din / 256 (Q8_0) units of this many bytes
static constexpr uint32_t MIX_UNIT_BYTES = 10240;
static constexpr uint32_t MIX_SLAB_ROWS = 64;
static constexpr uint32_t MIX_COLUMNS = 8;
static constexpr uint32_t MIX_CORE_ROWS = 4;
// bytes of one bfp16ebs8 packet of X a core multiplies: 16 rows x 128 k
static constexpr uint32_t MIX_X_PACKET_BYTES = 16 * 128 * 9 / 8;

/// @brief One weight of a call: `rows` output rows of format `fmt`, at byte `offset` of the weight buffer.  A call may
///        stack several (the K, V and Q projections are one call); their rows must be whole multiples of a column's share.
struct mix_band {
    size_t offset;
    uint32_t rows;
    int fmt;
};

/// @brief bytes of one 64-row slab (a pair of 32-row blocks) of a Din-wide weight
inline uint32_t mix_slab_bytes(int fmt, uint32_t Din) {
    return 2 * Din / (fmt == MIX_FMT_Q4_1 ? 512 : 256) * MIX_UNIT_BYTES;
}

/// @brief Slabs each column multiplies per pass.  A core always computes 128 x 64 outputs per pass: nb slabs for 128 / nb rows
///        of X.  The largest nb whose rows still hold L (up to 8) reads every weight once and X the fewest times.
inline uint32_t mix_nb(uint32_t L, uint32_t Dout) {
    uint32_t nb = 8;
    while (nb > 1 && (Dout % (MIX_COLUMNS * MIX_SLAB_ROWS * nb) != 0 || L > MIX_CORE_ROWS * 128 / nb)) nb /= 2;
    return nb;
}

/// @brief rows of X (and Y) of one pass
inline uint32_t mix_pass_rows(uint32_t nb) { return MIX_CORE_ROWS * 128 / nb; }

/// @brief L rounded up to whole passes: the rows the packed X and the output buffer must hold
inline uint32_t mix_rows_padded(uint32_t L, uint32_t Dout) {
    const uint32_t m = mix_pass_rows(mix_nb(L, Dout));
    return (L + m - 1) / m * m;
}

/// @brief Y = X W^T for the mixed-precision weights of the decode layer, read in place.
///
/// X is bfp16ebs8, packed by pack_x_mix for this (L, Dout); Y is bf16 [rows_padded][Dout] at out_offset (elements); the
/// bands are stacked in order along Dout.  Passes go over N (512 nb columns), then M (512 / nb rows).  Column c of the
/// array takes slabs c nb .. c nb + nb - 1 of a pass, core row r rows r 128 / nb ..; every X packet goes to all eight
/// columns, every weight chunk to both row pairs of its column.
inline void gen_dequant_mm_mix(npu_sequence* seq, const uint32_t L, const uint32_t Din, const std::vector<mix_band>& bands,
                               const size_t weight_offset, const size_t out_offset) {
    static constexpr npu_tiles IT[8] = {IT0, IT1, IT2, IT3, IT4, IT5, IT6, IT7};
    uint32_t Dout = 0;
    for (const mix_band& b : bands) Dout += b.rows;
    assert(Din % 512 == 0);

    const uint32_t nb = mix_nb(L, Dout);
    const uint32_t ms = 8 / nb;                       // 16-row groups of a core row
    const uint32_t m_pass = mix_pass_rows(nb);
    const uint32_t n_pass = MIX_COLUMNS * MIX_SLAB_ROWS * nb;
    assert(Dout % n_pass == 0);
    const uint32_t M_passes = mix_rows_padded(L, Dout) / m_pass;
    const uint32_t N_passes = Dout / n_pass;
    const uint32_t k_rounds = Din / 512;
    // one core row's X of one pass: per 512 k, 2 k slices x ms groups x 2 halves of 128
    const uint32_t x_block_bytes = k_rounds * 4 * ms * MIX_X_PACKET_BYTES;

    seq->clear_cmds();

    // the slabs of pass np that column c reads, as (byte offset, format)
    auto column_weights = [&](uint32_t np, uint32_t c, size_t& off, int& fmt) {
        uint32_t row0 = np * n_pass + c * MIX_SLAB_ROWS * nb;
        uint32_t band_row = 0;
        for (const mix_band& b : bands) {
            if (row0 < band_row + b.rows) {
                assert(row0 + MIX_SLAB_ROWS * nb <= band_row + b.rows);
                fmt = b.fmt;
                off = weight_offset + b.offset + (size_t)(row0 - band_row) / MIX_SLAB_ROWS * mix_slab_bytes(b.fmt, Din);
                return;
            }
            band_row += b.rows;
        }
        assert(false);
    };
    auto wait_pass = [&]() {
        for (uint32_t r = 0; r < MIX_CORE_ROWS; r++) {
            seq->npu_dma_wait(IT[r], S2MM, it_channel_0);
            seq->npu_dma_wait(IT[r + 4], S2MM, it_channel_0);
        }
    };

    int fmt_set[MIX_COLUMNS];
    for (uint32_t c = 0; c < MIX_COLUMNS; c++) fmt_set[c] = -1;
    int round = 0;
    for (uint32_t np = 0; np < N_passes; np++) {
        size_t w_off[MIX_COLUMNS];
        int fmt[MIX_COLUMNS];
        bool fmt_changed = false;
        for (uint32_t c = 0; c < MIX_COLUMNS; c++) {
            column_weights(np, c, w_off[c], fmt[c]);
            fmt_changed |= fmt[c] != fmt_set[c];
        }
        for (uint32_t mp = 0; mp < M_passes; mp++, round++) {
            // A core reads its RTPs once a call's first weight chunk is in, so they may only change once every core has
            // finished the pass before (its whole output is back): drain it first, then the pipelining starts over.
            const bool rewrite = mp == 0 && fmt_changed;
            if (rewrite && round > 0) wait_pass();
            if (rewrite) {
                for (uint32_t c = 0; c < MIX_COLUMNS; c++) {
                    for (int row = 2; row < 6; row++) {
                        npu_tiles tile = get_tile(row, c);
                        const uint32_t base = (row % 2 == 0) ? CT_rtp_address_even_row : CT_rtp_address_odd_row;
                        const uint32_t act = (row % 2 == 0) ? act_rtp_address_even_row : act_rtp_address_odd_row;
                        seq->rtp_write(tile, base, k_rounds);
                        seq->rtp_write(tile, base + 4, fmt[c]);
                        seq->rtp_write(tile, base + 8, nb);
                        seq->rtp_write(tile, act, 0);
                    }
                    fmt_set[c] = fmt[c];
                }
            }

            const int bd_offset = (round % 2) * 8;
            const npu_bd_id y_bd = npu_bd_id(0 + bd_offset);
            const npu_bd_id x_bd = npu_bd_id(1 + bd_offset);
            const npu_bd_id w_bd = npu_bd_id(2 + bd_offset);
            for (uint32_t c = 0; c < MIX_COLUMNS; c++) {
                seq->npu_dma_memcpy_nd(
                    sizeof(char), 2, MM2S, IT[c], w_bd, it_channel_1,
                    {0, 0, 0, (uint32_t)w_off[c]},
                    {1, 1, 1, nb * mix_slab_bytes(fmt[c], Din)},
                    {0, 0, 0, 1},
                    -1, 0, false, aggressive_cache
                );
            }
            for (uint32_t r = 0; r < MIX_CORE_ROWS; r++) {
                // Y of core row r: columns 0..3 come back through IT[r], 4..7 through IT[r + 4], as rows [slab nb][row of
                // the slab's group 16 ms][column 4][64]
                const size_t y0 = out_offset + (size_t)(mp * m_pass + r * 16 * ms) * Dout + np * n_pass;
                for (uint32_t h = 0; h < 2; h++) {
                    seq->npu_dma_memcpy_nd(
                        2, 0, S2MM, IT[r + 4 * h], y_bd, it_channel_0,
                        {0, 0, 0, (uint32_t)(y0 + h * 4 * nb * MIX_SLAB_ROWS)},
                        {nb, 16 * ms, 4, MIX_SLAB_ROWS},
                        {MIX_SLAB_ROWS, Dout, nb * MIX_SLAB_ROWS, 1},
                        -1, 0, true
                    );
                }
                // X of core row r, once per slab: the cores multiply every slab against the same rows
                seq->npu_dma_memcpy_nd(
                    1, 1, MM2S, IT[2 * r], x_bd, it_channel_0,
                    {0, 0, 0, (uint32_t)((size_t)(mp * MIX_CORE_ROWS + r) * x_block_bytes)},
                    {nb, 1, 1, x_block_bytes},
                    {0, 0, 0, 1},
                    -1, 0, false
                );
            }
            // keep one pass in flight: wait for the one before (unless it was drained above)
            if (round > 0 && !rewrite) wait_pass();
        }
    }
    wait_pass();
    seq->cmds2seq();
}

#endif
