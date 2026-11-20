// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __SEQ_GEN_HPP__
#define __SEQ_GEN_HPP__
#pragma once
#include "npu_utils/npu_utils.hpp"
#include "weights_desc.hpp"

// Sequences for the fused Gemma4 prefill overlay.  Three independent generators,
// one per region of the array:
//
//   column 0    gen_attn_full  -- head dim 512, causal
//   columns 1-6 gen_dequant_mm -- Q4_1 dequant + matmul
//   column 7    gen_attn_swa   -- head dim 256, sliding window
//
// They are independent because each is driven by its own npu_app: the regions
// share no tile, no lock and no DDR channel pair, so the runtime is free to
// interleave them (or, later, to fold them into one sequence) without any change
// here.
//
// ---------------------------------------------------------------------------
// RTP addresses.
//
// These are core-local buffer addresses assigned by aie-assign-buffer-addresses.
// They move whenever a core's data layout changes, and a stale one writes the
// runtime parameter into the middle of some other buffer, which shows up as a
// hang rather than as wrong numbers.  They are regenerated on every bitstream
// build into
//   build/$(MODEL_TYPE)/xclbins/rtp_addresses.txt
// and the values below must match that file.
static constexpr uint32_t CT_rtp_address_even_row  = 53248;
static constexpr uint32_t act_rtp_address_even_row = 21632;
static constexpr uint32_t CT_rtp_address_odd_row   = 8192;
static constexpr uint32_t act_rtp_address_odd_row  = 41088;
// 0 = q4_1, non-zero = the weights are already bf16 and the core copies them into
// w_buffer instead of unpacking.  Uninitialised at load, so always written.
static constexpr uint32_t qtype_rtp_address_even_row = 38016;
static constexpr uint32_t qtype_rtp_address_odd_row  = 53376;

/// @brief bytes one w tile costs on the wire.  bf16 rides in BF16_PACKETS 5120 B
///        packets of which only 4096 B are used, which is exactly 4x q4_1.
inline size_t w_stream_bytes(size_t elems, bool is_bf16) {
    const size_t q41 = get_quantization_byte_size(elems, flm_q41);
    return is_bf16 ? 4 * q41 : q41;
}

static constexpr uint32_t attn_full_l_begin_address    = 57472;
static constexpr uint32_t attn_full_l_end_address      = 11392;
static constexpr uint32_t attn_full_window_address     = 42112;
static constexpr uint32_t attn_swa_l_begin_address     = 57856;
static constexpr uint32_t attn_swa_l_end_address       = 11392;
static constexpr uint32_t attn_swa_window_address      = 45184;

// Physical columns.  Attention takes the two outermost so that the six mm
// columns split evenly across the two DDR channels (columns 0-3 / 4-7).
static constexpr uint32_t ATTN_FULL_COL = 0;
static constexpr uint32_t MM_COL_BASE   = 1;
static constexpr uint32_t ATTN_SWA_COL  = 7;

static constexpr npu_tiles ALL_IT[8] = {IT0, IT1, IT2, IT3, IT4, IT5, IT6, IT7};

// ---------------------------------------------------------------------------
// dequant_mm, columns 1-6.
//
// Geometry.  A core owns one (desired_m x desired_n) output tile; the region is
// 4 core rows (token-blocks) x 6 columns (N-slabs), so a round covers
// 4 * 64 = 256 tokens and 6 * 64 = 384 output columns.
//
// The drain is the part that is not the 8-column build with two columns deleted:
// 24 core lanes have to land in 6 memtiles, and a memtile can only take 4 y lanes
// (its other two s2mm channels are x and qw).  So memtile g = 3a + b holds a 2x2
// tile of (token-block, N-slab) -- token-blocks {2a, 2a+1}, N-slabs {2b, 2b+1} --
// and emits [2 token-blocks][desired_m rows][2 * desired_n].  See `y_pair_2x2` in
// iron/dequant_mm_tile.py.
//
// Tail rounds.  Dout / 128 is not generally a multiple of 3, so the last round's
// window slides back to end on the last slab-pair rather than running past it,
// recomputing up to two slab-pairs over byte-identical results.
void gen_dequant_mm_512(npu_sequence* seq, const uint32_t L, const uint32_t Din, const uint32_t Dout,
                        const size_t weight_offset, const size_t out_offset, const int rtp_act,
                        const bool is_bf16 = false){
    static constexpr uint32_t desired_m = 32;   // 4 core rows -> a 128-row step
    static constexpr uint32_t desired_k = 256;   // one k_slice per memtile load
    static constexpr uint32_t desired_n = 64;

    static constexpr uint32_t columns = 6;
    static constexpr uint32_t pairs_per_round = columns / 2; // 3 slab-pairs

    // Column c of the mm region is physical column c + 1.
    auto IT = [](uint32_t c) { return ALL_IT[MM_COL_BASE + c]; };
    // ... except the x broadcasters, which do not sit on the first four mm
    // columns: a memtile that hosts a broadcast spends an s2mm channel on it, so
    // the overlay spreads them over columns 1, 2, 4, 5 (X_COLS in
    // iron/fused_prefill_aie.py) rather than stacking two of them on the same
    // DDR-channel half.  The host has to feed the shims the flows actually start
    // from, so x goes out on these and not on IT(0..3).
    static const uint32_t x_cols[4] = {1, 2, 4, 5};

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
            npu_tiles tile = get_tile(row, MM_COL_BASE + col);
            seq->rtp_write(tile, CT_rtp_address_even_row, K_div_k);
            seq->rtp_write(tile, act_rtp_address_even_row, rtp_act);
            seq->rtp_write(tile, qtype_rtp_address_even_row, is_bf16 ? 1 : 0);
        }
    }
    for (int row = 3; row < 6; row += 2){
        for (uint32_t col = 0; col < columns; col++){
            npu_tiles tile = get_tile(row, MM_COL_BASE + col);
            seq->rtp_write(tile, CT_rtp_address_odd_row, K_div_k);
            seq->rtp_write(tile, act_rtp_address_odd_row, rtp_act);
            seq->rtp_write(tile, qtype_rtp_address_odd_row, is_bf16 ? 1 : 0);
        }
    }

    int round = 0;
    for (int i = 0; i < M_div_m; i++){
        const uint32_t x_offset = i * desired_m * 4 * Din;
        const uint32_t m_offset = i * desired_m * 4 * Dout;
        for (int r = 0; r < rounds_n; r++){
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
                    MM2S, IT(c),
                    w_bd, it_channel_1,
                    {0, 0, 0, (uint32_t)(weight_offset + w_stream_bytes((size_t)slab * desired_n * Din, is_bf16))},
                    {1, 1, 1, (uint32_t)w_stream_bytes((size_t)desired_n * Din, is_bf16)},
                    {0, 0, 0, 1},
                    -1, 0, false, aggressive_cache
                );
            }

            // x: one broadcast per core row, out of the four x_cols shims
            for (uint32_t rb = 0; rb < 4; rb++){
                seq->npu_dma_memcpy_nd(
                    2, 1,
                    MM2S, ALL_IT[x_cols[rb]],
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
                        S2MM, IT(g),
                        y_bd, it_channel_0,
                        {0, 0, 0, base},
                        {1, 2, desired_m, 2 * desired_n},
                        {0, desired_m * Dout, Dout, 1},
                        -1, 0, true
                    );
                }
            }

            if (round > 0){
                for (uint32_t g = 0; g < columns; g++) seq->npu_dma_wait(IT(g), S2MM, it_channel_0);
            }
            round++;
        }
    }
    for (uint32_t g = 0; g < columns; g++) seq->npu_dma_wait(IT(g), S2MM, it_channel_0);
    seq->cmds2seq();
}

// ---------------------------------------------------------------------------
// Attention, one solo column.
//
// The standalone Gemma4 prefill designs spend four columns on a CU and drive q
// and k/v from separate shims.  A fused column has one shim, so the three
// streams are packed as
//
//     mm2s 0 = q      mm2s 1 = k and v interleaved      s2mm 0 = o
//
// Interleaving k and v on one channel is what makes the solo column fit.  The
// memtile stages them with a four-BD chain that alternates halves (see
// attn_memtile in iron/attn_tile.py), so the host emits k and v of a block back
// to back -- a single four-dimensional BD walking
// {block, k-or-v, token, head-dim}, with the k-or-v step being the fixed
// KV_CACHE_SIZE / 2 gap between the two halves of the cache.
//
// Block accounting has to agree with the core exactly, or the two run out of
// step and the column deadlocks.  The core computes
//     pbk      = round_down(max(pbq - W, 0), lk_mt)
//     n_blocks = ceil((pbq + q_block - pbk) / lk_mt)
// and `attn_kv_blocks` below is the same arithmetic, kept in one place so the
// two cannot drift.  Rounding pbk down to lk_mt is what lets q_block (4 * lq,
// so 32 or 64 tokens) be smaller than the k block the host streams; the extra
// keys that pulls in are masked off by the core.
struct attn_col_cfg {
    uint32_t col;          // physical column
    int      lq;           // tokens per core
    int      lk_mt;        // k block the memtile stages
    int      DH;           // head dim
    uint32_t l_begin_addr;
    uint32_t l_end_addr;
    uint32_t window_addr;
};

static constexpr attn_col_cfg attn_full_cfg = {
    ATTN_FULL_COL, 8, 64, 512,
    attn_full_l_begin_address, attn_full_l_end_address, attn_full_window_address,
};
static constexpr attn_col_cfg attn_swa_cfg = {
    ATTN_SWA_COL, 16, 128, 256,
    attn_swa_l_begin_address, attn_swa_l_end_address, attn_swa_window_address,
};

/// \brief Number of lk_mt-sized k/v blocks the core will consume for one q round.
inline int attn_kv_blocks(const attn_col_cfg& c, int Lq_current, int window){
    const int q_block = 4 * c.lq;
    int kv_begin = Lq_current - window;
    if (kv_begin < 0) kv_begin = 0;
    kv_begin = (kv_begin / c.lk_mt) * c.lk_mt;
    return (Lq_current + q_block - kv_begin + c.lk_mt - 1) / c.lk_mt;
}

inline int attn_kv_begin(const attn_col_cfg& c, int Lq_current, int window){
    int kv_begin = Lq_current - window;
    if (kv_begin < 0) kv_begin = 0;
    return (kv_begin / c.lk_mt) * c.lk_mt;
}

/// \brief Build one attention column's sequence.
///
/// Buffer arguments, in the order the app is called with: 0 = o, 1 = q, 2 = kv.
///
/// \param L_begin,L_end   token range, both multiples of 4 * lq
/// \param WINDOW_SIZE     sliding window; pass >= L_end for full causal
/// \param Heads           q heads (this column runs them one after another)
/// \param Nd              kv heads packed side by side in one cache row
/// \param kv_head         which of them this run reads
/// \param KV_CACHE_SIZE   elements in the whole k+v buffer; v starts halfway
void gen_attn_column(npu_sequence* seq, const attn_col_cfg& c,
                     const uint32_t L_begin, const uint32_t L_end,
                     const int WINDOW_SIZE, const int Heads,
                     const int Nd, const int kv_head, const size_t KV_CACHE_SIZE){
    const npu_tiles IT = ALL_IT[c.col];
    const int q_block = 4 * c.lq;
    const int DH = c.DH;

    assert(L_begin % q_block == 0);
    assert(L_end % q_block == 0);

    const int window = WINDOW_SIZE;

    seq->clear_cmds();

    for (int row = 2; row < 6; row++){
        npu_tiles tile = get_tile(row, c.col);
        seq->rtp_write(tile, c.l_begin_addr, L_begin);
        seq->rtp_write(tile, c.l_end_addr, L_end);
        seq->rtp_write(tile, c.window_addr, window);
    }

    const int rounds = (L_end - L_begin) / q_block;

    for (int head = 0; head < Heads; head++){
        for (int round = 0; round < rounds; round++){
            const int Lq_current = L_begin + round * q_block;
            const int kv_begin = attn_kv_begin(c, Lq_current, window);
            const int n_blocks = attn_kv_blocks(c, Lq_current, window);
            const int bd_offset = (round % 2) * 8;

            // o back out.  Written first so its BD is already armed when the
            // column starts producing.
            const size_t o_offset = (size_t)head * DH + (size_t)round * q_block * DH * Heads;
            seq->npu_dma_memcpy_nd(
                2, 0,
                S2MM, IT,
                (npu_bd_id)(bd_offset + 0), it_channel_0,
                {0, 0, 0, (uint32_t)o_offset},
                {1, 1, (uint32_t)q_block, (uint32_t)DH},
                {0, 0, (uint32_t)(DH * Heads), 1},
                -1, 0, true
            );

            // q in: this column's whole round, one lq slice per core row.
            const size_t q_offset = (size_t)head * DH + (size_t)round * q_block * DH * Heads;
            seq->npu_dma_memcpy_nd(
                2, 1,
                MM2S, IT,
                (npu_bd_id)(bd_offset + 1), it_channel_0,
                {0, 0, 0, (uint32_t)q_offset},
                {1, 1, (uint32_t)q_block, (uint32_t)DH},
                {0, 0, (uint32_t)(DH * Heads), 1},
                -1, 0, false
            );

            // k and v in, interleaved per block: k0 v0 k1 v1 ...
            // Gemma4 is MQA, so every q head reads the one kv head; a grouped
            // model would pick head / (Heads / Nd) here instead.
            const size_t k_offset = (size_t)kv_head * DH + (size_t)kv_begin * DH * Nd;
            seq->npu_dma_memcpy_nd(
                2, 2,
                MM2S, IT,
                (npu_bd_id)(bd_offset + 2), it_channel_1,
                {0, 0, 0, (uint32_t)k_offset},
                {(uint32_t)n_blocks, 2, (uint32_t)c.lk_mt, (uint32_t)DH},
                {(uint32_t)(c.lk_mt * DH * Nd), (uint32_t)(KV_CACHE_SIZE / 2), (uint32_t)(DH * Nd), 1},
                -1, 0, false
            );

            if (round > 0) seq->npu_dma_wait(IT, S2MM, it_channel_0);
        }
        seq->npu_dma_wait(IT, S2MM, it_channel_0);
    }
    seq->cmds2seq();
}

#endif
