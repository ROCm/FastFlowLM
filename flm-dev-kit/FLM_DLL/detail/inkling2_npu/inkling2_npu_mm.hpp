// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file inkling2_npu_mm.hpp
/// \brief the prefill matmuls on the NPU: mm_rtm2_glu.xclbin (FLM_Xclbin/dequant_mm_bfp16_rtm2_glu_512x512x512),
///        Q4_K weights dequantized in the cores, bf16 x and y, plain dispatches only (no fused GLU)
/// \note The kernel multiplies x[rows][Din] by 768-row passes of weights, Din a multiple of 256. Column j of
///       the array takes one triple of 32-row blocks per pass (triple 8 p + j of pass p), the three blocks
///       interleaved along the column blocks:
///         triple t = [block(3t, 0) block(3t + 1, 0) block(3t + 2, 0) block(3t, 1) ...]
///       Core rows 2, 3 (pair A) produce output rows 0-47 of the triple, rows 4, 5 (pair B) 48-95; inside a
///       pair the even core multiplies the first m rows of x of a block of 2m, the odd core the next m.
///       The model's widths are not multiples of 256 x 3 (2304, 1280), and padding them in memory would cost
///       ~2 GB, so the weights stay at their real width Kw (triples Kw / 256 x 3 blocks apart) and the
///       sequence reads Din >= Kw columns per triple anyway: the surplus is the next triple's first blocks,
///       finite numbers that meet the zero columns [Kw, Din) of x. The same goes for the outputs: a matrix
///       of N rows runs ceil(N / 768) passes and the surplus columns of y come from the rows after it
///       (the next expert, or the zero tail of the buffer); they are ignored.
///       The host reads the same buffers (qmat::group 3), so there is one copy of the weights.
#pragma once
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include "inkling2_bfp16.hpp"
#include "npu_utils/npu_utils.hpp"

namespace inkling2_npu_mm {

// core-local RTP addresses of mm_rtm2_glu.xclbin (K_div_k, rtp_act of even / odd core rows)
constexpr uint32_t CT_RTP_EVEN = 54784, ACT_RTP_EVEN = 54848, CT_RTP_ODD = 61696, ACT_RTP_ODD = 61760;
constexpr uint32_t BLOCK_BYTES = 4736;      // one Q4_K block, 32 rows x 256 columns
constexpr uint32_t PASS_ROWS = 768;         // weight rows per pass (8 columns x 96)
constexpr uint32_t COL_ROWS = 96;           // weight rows per column and pass (a block triple)

inline uint64_t q4k_bytes(uint64_t elems) { return elems / (32 * 256) * BLOCK_BYTES; }

// bytes of a block's x a memtile holds (m x Din bfp16, dequant_mm_tile.py X_RES_ELEMS)
constexpr uint32_t MT_X_BYTES = 331776;

/// \brief the most rows per core row of a block whose x fits a memtile: 128, fewer for Din > 2304
inline uint32_t max_m(uint32_t Din) { return std::min<uint32_t>(128, MT_X_BYTES / (uint32_t)inkling2_bfp16::bytes(16, Din) * 16); }

/// \brief rows per core of a dispatch: 2 x m rows per block (the two cores of a pair), 16 <= m <= max_m(Din)
struct geometry {
    uint32_t blocks, m, covered;
};
inline geometry pick_geometry(uint32_t rows, uint32_t Din) {
    assert(rows > 0);
    const uint32_t mm = max_m(Din);
    geometry g;
    g.blocks = (rows + 2 * mm - 1) / (2 * mm);
    g.m = 16 * ((rows + g.blocks * 32 - 1) / (g.blocks * 32));
    g.covered = g.blocks * 2 * g.m;
    // no rounding up to max_m to save memtile reconfigurations (~50 us per m change): a batch sorted by m pays one
    // per m value, and rounding up to save them measured 2% slower at 2k tokens
    return g;
}

// memtile DMA registers of the x / y paths
constexpr uint32_t MT_BD_BASE = 0xA0000, MT_BD_STRIDE = 0x20, MT_LOCK_BASE = 0xC0000, MT_LOCK_STRIDE = 0x10;
struct channel {
    uint32_t ctrl, start_bd;
};
constexpr channel MT_X_CH[] = {{0xA0600, 0}, {0xA0630, 2}};
constexpr channel MT_Y_CH[] = {{0xA0608, 24}, {0xA0610, 4}, {0xA0618, 26}, {0xA0620, 6}, {0xA0638, 28}};
// x: one block in two halves (a, b), each with a producer / consumer lock pair: S2MM BD 0 / 1, MM2S BD 2 / 3
constexpr uint32_t MT_X_S2MM[] = {0, 1};
constexpr uint32_t MT_X_MM2S[] = {2, 3};
constexpr uint32_t MT_X_LOCKS[] = {0, 1, 2, 3};   // a producer, a consumer, b producer, b consumer
constexpr uint32_t MT_Y_S2MM[] = {4, 5, 6, 7, 24, 25, 26, 27};
constexpr uint32_t MT_Y_MM2S[] = {28, 29};
constexpr uint32_t MT_Y_LOCKS[] = {5, 6, 7, 8};
constexpr uint32_t MT_Y_LOCK_INIT[] = {4, 4, 0, 0};
// y BD m wrap word (MM2S D2: m rows of 24 words); the other words do not depend on m
constexpr uint32_t MT_Y_MM2S_WRAP_WORD = 4, MT_Y_MM2S_WRAP_LOW = 0x17;
// memtiles holding x: 0 (the even cores of every pair) and 2 (the odd cores)
inline bool has_x(uint32_t col) { return col == 0 || col == 2; }

/// \brief put the memtile x / y paths on m rows per core, x of width Din read P times (passes)
inline void rearm(npu_sequence* seq, uint32_t m, uint32_t Din, uint32_t P) {
    // the halves split a block at a 16-row group of one slice, so each holds whole core chunks
    const uint32_t total = (uint32_t)inkling2_bfp16::bytes(m, Din), gran = (uint32_t)inkling2_bfp16::SLICE_BYTES;
    const uint32_t La = (total / gran + 1) / 2 * gran, Lb = total - La;
    assert(total <= MT_X_BYTES && Lb > 0 && P < 64);
    // S2MM of a half: acquire >= P on its producer lock, release P on its consumer lock (memtile locks: 0x40 + id)
    auto x_word7 = [P](uint32_t acq, uint32_t rel) {
        return 0x80000000u | (P << 24) | ((0x40 + rel) << 16) | ((0x80 | (-P & 0x7F)) << 8) | (0x40 + acq);
    };
    const uint32_t x_init[] = {P, 0, P, 0};
    auto bd_word = [](uint32_t bd, uint32_t word) { return MT_BD_BASE + MT_BD_STRIDE * bd + 4 * word; };
    for (uint32_t col = 0; col < 8; col++) {
        npu_tiles mt = get_tile(1, col);
        const bool x = has_x(col);
        if (x)
            for (auto& c : MT_X_CH) seq->rtp_write(mt, c.ctrl, 2);
        for (auto& c : MT_Y_CH) seq->rtp_write(mt, c.ctrl, 2);
        if (x)
            for (uint32_t i = 0; i < 4; i++) {
                seq->rtp_write(mt, MT_LOCK_BASE + MT_LOCK_STRIDE * MT_X_LOCKS[i], 63);
                seq->rtp_write(mt, MT_LOCK_BASE + MT_LOCK_STRIDE * MT_X_LOCKS[i], x_init[i]);
            }
        for (uint32_t i = 0; i < 4; i++) {
            seq->rtp_write(mt, MT_LOCK_BASE + MT_LOCK_STRIDE * MT_Y_LOCKS[i], 63);
            seq->rtp_write(mt, MT_LOCK_BASE + MT_LOCK_STRIDE * MT_Y_LOCKS[i], MT_Y_LOCK_INIT[i]);
        }
        if (x) {
            for (uint32_t h = 0; h < 2; h++) {
                seq->rtp_write(mt, bd_word(MT_X_S2MM[h], 0), (h ? Lb : La) / 4);
                seq->rtp_write(mt, bd_word(MT_X_S2MM[h], 7), x_word7(MT_X_LOCKS[2 * h], MT_X_LOCKS[2 * h + 1]));
                seq->rtp_write(mt, bd_word(MT_X_MM2S[h], 0), (h ? Lb : La) / 4);
            }
        }
        // a core's m rows x 48 n land in its region; the MM2S sends [2m rows][96 n]
        for (uint32_t bd : MT_Y_S2MM) seq->rtp_write(mt, bd_word(bd, 0), m * 24);
        for (uint32_t bd : MT_Y_MM2S) {
            seq->rtp_write(mt, bd_word(bd, 0), m * 96);
            seq->rtp_write(mt, bd_word(bd, MT_Y_MM2S_WRAP_WORD), (m << 17) | MT_Y_MM2S_WRAP_LOW);
        }
        if (x)
            for (auto& c : MT_X_CH) {
                seq->rtp_write(mt, c.ctrl, 0);
                seq->rtp_write(mt, c.ctrl + 4, c.start_bd);
            }
        for (auto& c : MT_Y_CH) {
            seq->rtp_write(mt, c.ctrl, 0);
            seq->rtp_write(mt, c.ctrl + 4, c.start_bd);
        }
    }
}

/// \brief RTP word 1 of every core: rows per core row / 16
inline void set_core_rows(npu_sequence* seq, uint32_t m) {
    for (int row = 2; row < 6; row++) {
        const uint32_t ctp = (row % 2 == 0) ? CT_RTP_EVEN : CT_RTP_ODD;
        for (uint32_t col = 0; col < 8; col++) seq->rtp_write(get_tile(row, col), ctp + 4, m / 16);
    }
}

/// \brief one matmul of a batch: y[rows][Npad] at element y_off of the y argument = x[rows][Din] (bfp16, packed by
///        inkling2_bfp16::pack_job) at byte x_off of the x argument times the weights at byte w_off of the weight argument
struct job {
    uint32_t rows;
    uint32_t x_off, y_off;
    uint64_t w_off;
};

/// \brief the jobs of one dispatch, all of the same shape: y = x W^T, W in block triples at width Kw (Kw <= Din,
///        both multiples of 256), Npad a multiple of 768. Arguments of the run: (y, x, weights). x and y of a job
///        must span pick_geometry(rows).covered rows; x columns [Kw, Din) must be zero (they meet real weights).
///        The weight argument must hold Npad x Kw + 96 x (Din - Kw) elements from a job's w_off (use
///        tail_bytes). Jobs run in order of rows per core (m): the memtiles are set up for the dispatch, and
///        again when m changes, after the work before it has drained.
inline void gen_batch(npu_sequence* seq, uint32_t Din, uint32_t Kw, uint32_t Npad, std::vector<job> jobs) {
    static constexpr npu_tiles IT[8] = {IT0, IT1, IT2, IT3, IT4, IT5, IT6, IT7};
    assert(Din % 256 == 0 && Kw % 256 == 0 && Kw <= Din && Npad % PASS_ROWS == 0 && !jobs.empty());
    const uint32_t K_div_k = Din / 256;
    const uint32_t passes = Npad / PASS_ROWS;
    // what a column consumes per pass, q4k_bytes(96 Din) = 6 K_div_k x W_ROW bytes (a BD row is <= 1024 words)
    constexpr uint32_t W_ROW = 2368;
    constexpr uint32_t X_ROW = 1152;   // bytes of x per BD row: m / 1024 rows per column
    const uint32_t col_rows = (uint32_t)(q4k_bytes((uint64_t)COL_ROWS * Din) / W_ROW);
    const uint32_t pass_w = (uint32_t)q4k_bytes((uint64_t)PASS_ROWS * Kw);   // weight bytes per pass
    assert(passes < 64);
    std::stable_sort(jobs.begin(), jobs.end(),
                     [Din](const job& a, const job& b) { return pick_geometry(a.rows, Din).m > pick_geometry(b.rows, Din).m; });

    seq->clear_cmds();
    // RTP word 0: k steps; word 2: the pair (rows 4, 5 take the second 48 rows of a triple); rtp_act: plain
    for (int row = 2; row < 6; row++) {
        const bool even = (row % 2) == 0;
        const uint32_t ctp = even ? CT_RTP_EVEN : CT_RTP_ODD;
        for (uint32_t col = 0; col < 8; col++) {
            npu_tiles tile = get_tile(row, col);
            seq->rtp_write(tile, ctp, K_div_k);
            seq->rtp_write(tile, ctp + 8, row >= 4 ? 1 : 0);
            seq->rtp_write(tile, even ? ACT_RTP_EVEN : ACT_RTP_ODD, 0);
        }
    }
    auto drain = [&] {
        for (int c = 0; c < 8; c++) seq->npu_dma_wait(IT[c], S2MM, it_channel_0);
    };
    uint32_t cur_m = 0, round = 0;
    for (const job& jb : jobs) {
        const geometry g = pick_geometry(jb.rows, Din);
        const uint32_t m = g.m;
        if (m != cur_m) {
            if (round > 0) drain();
            round = 0;
            set_core_rows(seq, m);
            rearm(seq, m, Din, passes);
            cur_m = m;
        }
        // one round per block of 2m rows: w and y advance by a pass, x is sent once and read every pass by its memtile
        for (uint32_t i = 0; i < g.blocks; i++, round++) {
            const int bd_offset = (round % 2) * 8;
            const uint32_t y_offset = jb.y_off + i * 2 * m * Npad;
            const uint32_t x_offset = jb.x_off + (uint32_t)inkling2_bfp16::bytes((size_t)i * 2 * m, Din);
            const npu_bd_id y_bd = npu_bd_id(0 + bd_offset), x_bd = npu_bd_id(1 + bd_offset), w_bd = npu_bd_id(2 + bd_offset);
            for (uint32_t col = 0; col < 8; col++) {
                const uint64_t w0 = jb.w_off + q4k_bytes((uint64_t)col * COL_ROWS * Kw);
                seq->npu_dma_memcpy_nd(sizeof(char), 2, MM2S, IT[col], w_bd, it_channel_1, {0, 0, 0, (uint32_t)w0},
                                       {passes, 1, col_rows, W_ROW}, {pass_w, 0, W_ROW, 1}, -1, 0, false, aggressive_cache);
                seq->npu_dma_memcpy_nd(2, 0, S2MM, IT[col], y_bd, it_channel_0, {0, 0, 0, y_offset + col * COL_ROWS},
                                       {passes, 1, 2 * m, COL_ROWS}, {PASS_ROWS, 0, Npad, 1}, -1, 0, true);
            }
            // the even cores' x (rows 0..m-1 of the block) through memtile 0, the odd cores' (m..2m-1) through 2;
            // a core's x is contiguous, in the order it is consumed (inkling2_bfp16.hpp)
            for (uint32_t h = 0; h < 2; h++)
                seq->npu_dma_memcpy_nd(sizeof(char), 1, MM2S, IT[2 * h], x_bd, it_channel_0,
                                       {0, 0, 0, x_offset + (uint32_t)inkling2_bfp16::bytes((size_t)h * m, Din)},
                                       {1, 1, m * Din / 1024, X_ROW}, {0, 0, X_ROW, 1}, -1, 0, false);
            if (round > 0) drain();
        }
    }
    drain();
    seq->cmds2seq();
}

/// \brief one job at offset 0 of every argument
inline void gen(npu_sequence* seq, uint32_t rows, uint32_t Din, uint32_t Kw, uint32_t Npad) {
    gen_batch(seq, Din, Kw, Npad, {{rows, 0, 0, 0}});
}

/// \brief bytes a weight buffer needs past the end of its last matrix (N rows of width Kw, read as Npad x Din)
inline uint64_t tail_bytes(uint32_t N, uint32_t Npad, uint32_t Din, uint32_t Kw) {
    return q4k_bytes((uint64_t)(Npad - N) * Kw + (uint64_t)COL_ROWS * (Din - Kw));
}

}  // namespace inkling2_npu_mm
