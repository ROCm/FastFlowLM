// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file qwen3_asr_audio_gemm_seq.hpp
/// \brief Run-time sequence of the bfp16 GEMM (iron/gemm_aie.py): Y = act(X W^T + b), bf16 out.
///
/// Copy of FLM_Xclbin/bfp16_gemm/host/seq_gen.hpp and pack_bfp16.hpp in this engine's namespace (engines do not share
/// headers with inline functions); keep the two in sync with the kernel.
///
/// One call runs R rounds of 256 rows against NB blocks of 1024 output columns, K = 128 KC.
/// Weights go through the memtiles in one of two modes:
///   panel   (KC <= 24): a column's whole block of weights stays in its memtile and is sent R times;
///                       read from DDR once per call. With KC <= 12 (hybrid) round 1 starts on the first
///                       chunk instead of waiting for the whole panel.
///   stream  (KC > 24, or forced): the memtile is a ring of chunk slots refilled from DDR; the weights
///                       are read once per round. KC multiple of 4, KC R <= 256 per call.
/// DDR layouts (all bfp16ebs8, 9 B per 8 k; see pack_bfp16.hpp for the block encoding):
///   x  [M / 16 row blocks][KC][16 k tiles][2 row halves][8 rows][9 B]          (2304 B per block and kc)
///   w  [NB][8 columns][KC][lo | hi][4 jp][16 k tiles][2][8 columns][9 B] + 64 bf16 bias  (9344 B per half)
///   y  bf16 row-major, row pitch ldy elements
/// The memtile weight path is not in the design: each call writes its descriptors, panel locks and
/// start queues (check_memtile_bds.py verifies at build time that the design leaves them free).
#pragma once
#include <cassert>
#include <cstring>
#include "npu_utils/npu_utils.hpp"

namespace qwen3_asr_audio_gemm_seq {

/// 8 bf16 -> one 9 B block: the largest fp32 exponent field of the 8, then each
/// value's signed 24-bit mantissa shifted right by (17 + max_exp - exp), capped at
/// 31, rounded to nearest even (what the core's own bf16 -> bfp16 conversion does
/// under conv_even). A value is mantissa x 2^(exponent - 133).
inline void encode_block(uint8_t* out, const uint16_t* v) {
    int ex[8], mx = 0;
    for (int i = 0; i < 8; i++) { ex[i] = (v[i] >> 7) & 0xFF; if (ex[i] > mx) mx = ex[i]; }
    out[0] = (uint8_t)mx;
    for (int i = 0; i < 8; i++) {
        int32_t mag = (int32_t)(((uint32_t)v[i] & 0x7Fu) << 16);
        if (ex[i]) mag |= 0x800000;
        const int32_t s = (v[i] & 0x8000u) ? -mag : mag;
        int sh = 17 + mx - ex[i]; if (sh > 31) sh = 31;
        int32_t q = s >> sh;
        const int32_t rem = s - (int32_t)((uint32_t)q << sh), half = 1 << (sh - 1);
        if (rem > half || (rem == half && (q & 1))) q++;
        if (q > 127) q = 127;
        out[1 + i] = (uint8_t)(int8_t)q;
    }
}

// core RTP (iron/gemm_tile.py RTP_ADDR): [0] KC, [1] rounds of 256 rows (all N blocks), [2] act
static constexpr uint32_t CT_RTP_ADDR = 58112;

static constexpr uint32_t X_BLOCK_WORDS = 576;    // 16 rows x 128 k
static constexpr uint32_t W_HALF_WORDS = 2336;    // 64 columns x 128 k + 64 bf16 bias
static constexpr uint32_t W_CHUNK_WORDS = 2 * W_HALF_WORDS;
static constexpr uint32_t ROUND_ROWS = 256, BLOCK_COLS = 1024, COLUMNS = 8, CORE_ROWS = 4;
static constexpr uint32_t MAX_ROUNDS = 31;        // panel locks count 2 credits per round, 7-bit lock values
static constexpr uint32_t KC_DOUBLE = 12, KC_SINGLE = 24;   // panel fits twice / once in the memtile

// memtile (row 1); keep in sync with iron/gemm_tile.py and check_memtile_bds.py
static constexpr uint32_t MT_BD_BASE = 0xA0000, MT_BD_STRIDE = 0x20, MT_LOCK_BASE = 0xC0000, MT_LOCK_STRIDE = 0x10;
static constexpr uint32_t MT_OWN_WORDS = 0x80000 / 4, MT_OWN_LOCK = 64;
static constexpr uint32_t MT_PANEL_ADDR[2] = {65536, 294912};
static constexpr uint32_t MT_S2MM0_BD[2] = {20, 21}, MT_LO_BD[2] = {22, 23}, MT_HI_BD[2] = {44, 45};
static constexpr uint32_t MT_PANEL_PROD[2] = {8, 9}, MT_PANEL_CONS[2] = {10, 11};
static constexpr uint32_t MT_S2MM0_QUEUE = 0xA0604, MT_MM2S0_QUEUE = 0xA0634, MT_MM2S1_QUEUE = 0xA063C;
// stream mode: ring at panel 0's address; locks: free slots, lo half ready, hi half ready
static constexpr uint32_t MT_RING_FREE = 8, MT_RING_LO = 10, MT_RING_HI = 11;
// hybrid mode: per panel "chunk lo ready" / "chunk hi ready" locks; rounds 2..R send the panel with these BDs
static constexpr uint32_t MT_CHUNK_LO[2] = {8, 9}, MT_CHUNK_HI[2] = {10, 11};
static constexpr uint32_t MT_LO_PANEL_BD[2] = {18, 19}, MT_HI_PANEL_BD[2] = {46, 47};
static constexpr uint32_t MAX_QUEUE_REPEAT = 256;   // start queue repeat field: 8 bits

inline uint32_t mt_lock_word(uint32_t acq_id, int acq_ge, uint32_t rel_id, int rel_val) {
    return (1u << 31) | ((uint32_t)(rel_val & 0x7F) << 24) | ((MT_OWN_LOCK + rel_id) << 16) | (1u << 15) |
           ((uint32_t)(-acq_ge & 0x7F) << 8) | (MT_OWN_LOCK + acq_id);
}
inline uint32_t mt_release_word(uint32_t rel_id, int rel_val) {
    return (1u << 31) | ((uint32_t)(rel_val & 0x7F) << 24) | ((MT_OWN_LOCK + rel_id) << 16);
}
inline uint32_t mt_acquire_word(uint32_t acq_id, int acq_ge) {
    return (1u << 31) | (1u << 15) | ((uint32_t)(-acq_ge & 0x7F) << 8) | (MT_OWN_LOCK + acq_id);
}
inline uint32_t mt_dim(uint32_t wrap, uint32_t stride) { return (wrap << 17) | (stride - 1); }
inline uint32_t mt_iter(uint32_t wrap, uint32_t stride) { return ((wrap - 1) << 17) | (stride - 1); }

/// One call's shape. x_words / y_elems / w_bytes: offsets into the three buffers.
struct call_t {
    uint32_t rounds;      //!< R: 256-row rounds, <= MAX_ROUNDS
    uint32_t kc;          //!< K / 128, <= KC_SINGLE
    uint32_t nb;          //!< N blocks of 1024 columns
    uint32_t ldy;         //!< row pitch of y in elements (>= 1024 nb)
    uint32_t act;         //!< 0 none, 1 GELU (tanh form)
    size_t x_words = 0;   //!< first 16-row block of x
    size_t w_words = 0;   //!< first N block of w
    size_t y_elems = 0;   //!< y[0][0] of this call
    cache_flag_t x_cache = normal_cache;   //!< x is read once per N block
    bool stream = false;  //!< stream mode even if the panel fits (KC > KC_SINGLE always streams)
    bool hybrid = true;   //!< KC <= KC_DOUBLE: round 1 sends each chunk as it lands (false: after the whole panel)
};

inline bool streams(const call_t& c) { return c.stream || c.kc > KC_SINGLE; }
inline bool hybrids(const call_t& c) { return c.hybrid && !streams(c) && c.kc <= KC_DOUBLE; }

inline size_t x_bytes(uint32_t m_pad, uint32_t K) { return (size_t)m_pad / 16 * (K / 128) * X_BLOCK_WORDS * 4; }
inline size_t w_bytes(uint32_t N, uint32_t K) { return (size_t)N / 128 * (K / 128) * W_CHUNK_WORDS * 4; }

/// Arguments: 0 = y (bf16), 1 = x (bfp16), 2 = w (bfp16 + bias).
inline void gen_gemm(npu_sequence* seq, const call_t& c) {
    static constexpr npu_tiles IT[8] = {IT0, IT1, IT2, IT3, IT4, IT5, IT6, IT7};
    assert(c.rounds >= 1 && c.rounds <= MAX_ROUNDS);
    const bool stream = streams(c);
    assert(c.kc >= 1 && c.nb >= 1);
    assert(!stream || (c.kc % 4 == 0 && c.kc * c.rounds <= MAX_QUEUE_REPEAT));
    assert((uint64_t)16 * c.kc * X_BLOCK_WORDS <= (1u << 20));   // x round stride, in words
    assert(c.ldy % 2 == 0 && c.ldy >= BLOCK_COLS * c.nb);
    assert((uint64_t)128 * c.ldy <= (1u << 20));   // y round stride, in words
    const uint32_t R = c.rounds, KC = c.kc;
    const bool two_panels = !stream && KC <= KC_DOUBLE, hybrid = hybrids(c);
    const uint32_t panel_words = KC * W_CHUNK_WORDS;
    // ring slots divide the chunks of one N block, so a slot sequence restarted per task lands on
    // the same addresses as one carried over
    const uint32_t slots = (KC * R) % 8 == 0 ? 8 : 4;

    seq->clear_cmds();
    for (int row = 2; row < 6; row++)
        for (uint32_t col = 0; col < COLUMNS; col++) {
            const npu_tiles t = get_tile(row, col);
            seq->rtp_write(t, CT_RTP_ADDR, KC);
            seq->rtp_write(t, CT_RTP_ADDR + 4, R * c.nb);
            seq->rtp_write(t, CT_RTP_ADDR + 8, c.act);
        }

    // memtile weight path. Panel b: S2MM0 fills it once per N block, then MM2S0 (lo halves) and MM2S1
    // (hi halves) send it R times each (start-queue repeats; the descriptor's locks run on every repeat).
    // The fill takes 2R producer credits and gives 2R consumer credits; each repeat of either channel takes
    // one consumer credit and returns one producer credit. A channel's task has exactly R repeats, so
    // neither channel can use the other's credits.
    // Stream mode: S2MM0 fills one slot per chunk (takes a free slot, gives "lo ready"), MM2S0 sends its
    // lo half (gives "hi ready"), MM2S1 its hi half (frees the slot). One descriptor per channel, its
    // iteration walks the ring, the start-queue repeat runs it once per chunk.
    // Hybrid mode (two panels): S2MM0 fills the panel chunk by chunk (gives "lo ready"); round 1 sends each
    // chunk as it lands (MM2S0: takes "lo ready", gives "hi ready"; MM2S1 takes "hi ready"), rounds 2..R
    // send the resident panel without locks. No lock guards the refill: the host has waited for y of
    // N block j - 2, the panel's previous user, before it pushes block j.
    auto bd_word = [](uint32_t bd, uint32_t w) { return MT_BD_BASE + MT_BD_STRIDE * bd + 4 * w; };
    for (uint32_t col = 0; col < COLUMNS; col++) {
        const npu_tiles mt = get_tile(1, col);
        if (stream) {
            seq->rtp_write(mt, MT_LOCK_BASE + MT_LOCK_STRIDE * MT_RING_FREE, slots);
            seq->rtp_write(mt, MT_LOCK_BASE + MT_LOCK_STRIDE * MT_RING_LO, 0);
            seq->rtp_write(mt, MT_LOCK_BASE + MT_LOCK_STRIDE * MT_RING_HI, 0);
            const uint32_t base = MT_OWN_WORDS + MT_PANEL_ADDR[0] / 4, it = mt_iter(slots, W_CHUNK_WORDS);
            const uint32_t fill[8] = {W_CHUNK_WORDS, base, 0, 0, 0, 0, it, mt_lock_word(MT_RING_FREE, 1, MT_RING_LO, 1)};
            const uint32_t lo[8] = {W_HALF_WORDS, base, 0, 0, 0, 0, it, mt_lock_word(MT_RING_LO, 1, MT_RING_HI, 1)};
            const uint32_t hi[8] = {W_HALF_WORDS, base + W_HALF_WORDS, 0, 0, 0, 0, it, mt_lock_word(MT_RING_HI, 1, MT_RING_FREE, 1)};
            for (uint32_t w = 0; w < 8; w++) {
                seq->rtp_write(mt, bd_word(MT_S2MM0_BD[0], w), fill[w]);
                seq->rtp_write(mt, bd_word(MT_LO_BD[0], w), lo[w]);
                seq->rtp_write(mt, bd_word(MT_HI_BD[0], w), hi[w]);
            }
            continue;
        }
        if (hybrid) {
            for (uint32_t b = 0; b < 2; b++) {
                seq->rtp_write(mt, MT_LOCK_BASE + MT_LOCK_STRIDE * MT_CHUNK_LO[b], 0);
                seq->rtp_write(mt, MT_LOCK_BASE + MT_LOCK_STRIDE * MT_CHUNK_HI[b], 0);
                const uint32_t base = MT_OWN_WORDS + MT_PANEL_ADDR[b] / 4, it = mt_iter(KC, W_CHUNK_WORDS);
                const uint32_t fill[8] = {W_CHUNK_WORDS, base, 0, 0, 0, 0, it, mt_release_word(MT_CHUNK_LO[b], 1)};
                const uint32_t lo[8] = {W_HALF_WORDS, base, 0, 0, 0, 0, it, mt_lock_word(MT_CHUNK_LO[b], 1, MT_CHUNK_HI[b], 1)};
                const uint32_t hi[8] = {W_HALF_WORDS, base + W_HALF_WORDS, 0, 0, 0, 0, it, mt_acquire_word(MT_CHUNK_HI[b], 1)};
                for (uint32_t w = 0; w < 8; w++) {
                    seq->rtp_write(mt, bd_word(MT_S2MM0_BD[b], w), fill[w]);
                    seq->rtp_write(mt, bd_word(MT_LO_BD[b], w), lo[w]);
                    seq->rtp_write(mt, bd_word(MT_HI_BD[b], w), hi[w]);
                }
                for (uint32_t h = 0; h < 2 && R > 1; h++) {
                    const uint32_t send[8] = {KC * W_HALF_WORDS, base + h * W_HALF_WORDS,
                                              mt_dim(584, 1), mt_dim(4, 584), mt_dim(KC, W_CHUNK_WORDS), 0, 0, 1u << 31};
                    const uint32_t bd = h ? MT_HI_PANEL_BD[b] : MT_LO_PANEL_BD[b];
                    for (uint32_t w = 0; w < 8; w++) seq->rtp_write(mt, bd_word(bd, w), send[w]);
                }
            }
            continue;
        }
        for (uint32_t b = 0; b < (two_panels ? 2u : 1u); b++) {
            seq->rtp_write(mt, MT_LOCK_BASE + MT_LOCK_STRIDE * MT_PANEL_PROD[b], 2 * R);
            seq->rtp_write(mt, MT_LOCK_BASE + MT_LOCK_STRIDE * MT_PANEL_CONS[b], 0);
            const uint32_t base = MT_OWN_WORDS + MT_PANEL_ADDR[b] / 4;
            const uint32_t fill[8] = {panel_words, base, 0, 0, 0, 0, 0,
                                      mt_lock_word(MT_PANEL_PROD[b], 2 * (int)R, MT_PANEL_CONS[b], 2 * (int)R)};
            for (uint32_t w = 0; w < 8; w++) seq->rtp_write(mt, bd_word(MT_S2MM0_BD[b], w), fill[w]);
            for (uint32_t h = 0; h < 2; h++) {
                // [kc][lo | hi] -> one half of every chunk: 584 x 4 contiguous words, then the next chunk
                const uint32_t send[8] = {KC * W_HALF_WORDS, base + h * W_HALF_WORDS,
                                          mt_dim(584, 1), mt_dim(4, 584), mt_dim(KC, W_CHUNK_WORDS), 0, 0,
                                          mt_lock_word(MT_PANEL_CONS[b], 1, MT_PANEL_PROD[b], 1)};
                const uint32_t bd = h ? MT_HI_BD[b] : MT_LO_BD[b];
                for (uint32_t w = 0; w < 8; w++) seq->rtp_write(mt, bd_word(bd, w), send[w]);
            }
        }
    }

    for (uint32_t j = 0; j < c.nb; j++) {
        const uint32_t b = two_panels ? (j & 1) : 0;
        const uint32_t o = (j & 1) * 4;
        const uint32_t fills = stream ? KC * R : hybrid ? KC : 1, sends = stream ? KC * R : hybrid ? KC : R;
        for (uint32_t col = 0; col < COLUMNS; col++) {
            const npu_tiles mt = get_tile(1, col);
            seq->rtp_write(mt, MT_S2MM0_QUEUE, ((fills - 1) << 16) | MT_S2MM0_BD[b]);
            seq->rtp_write(mt, MT_MM2S0_QUEUE, ((sends - 1) << 16) | MT_LO_BD[b]);
            seq->rtp_write(mt, MT_MM2S1_QUEUE, ((sends - 1) << 16) | MT_HI_BD[b]);
            if (hybrid && R > 1) {
                seq->rtp_write(mt, MT_MM2S0_QUEUE, ((R - 2) << 16) | MT_LO_PANEL_BD[b]);
                seq->rtp_write(mt, MT_MM2S1_QUEUE, ((R - 2) << 16) | MT_HI_PANEL_BD[b]);
            }
        }
        for (uint32_t col = 0; col < COLUMNS; col++) {
            // weights: the column's whole panel of this N block (stream mode: once per round)
            seq->npu_dma_memcpy_nd(4, 2, MM2S, IT[col], npu_bd_id(0 + o), it_channel_0,
                {0, 0, 0, (uint32_t)(c.w_words + ((size_t)j * COLUMNS + col) * panel_words)},
                {stream ? R : 1, 1, 1, panel_words}, {0, 0, 0, 1}, -1, 0, false, aggressive_cache);
            // y: R rounds of 256 rows x the column's 128 columns
            seq->npu_dma_memcpy_nd(4, 0, S2MM, IT[col], npu_bd_id(2 + o), it_channel_0,
                {0, 0, 0, (uint32_t)((c.y_elems + (size_t)j * BLOCK_COLS + 128 * col) / 2)},
                {R, 1, ROUND_ROWS, 64}, {128 * c.ldy, 0, c.ldy / 2, 1}, -1, 0, true);
        }
        for (uint32_t r = 0; r < CORE_ROWS; r++) {
            // x of core row r: per round, per kc, its 4 strips (16-row blocks 16 t + 4 r + s)
            seq->npu_dma_memcpy_nd(4, 1, MM2S, IT[r], npu_bd_id(1 + o), it_channel_1,
                {0, 0, 0, (uint32_t)(c.x_words + (size_t)4 * r * KC * X_BLOCK_WORDS)},
                {R, KC, 4, X_BLOCK_WORDS}, {16 * KC * X_BLOCK_WORDS, X_BLOCK_WORDS, KC * X_BLOCK_WORDS, 1},
                -1, 0, false, c.x_cache);
        }
        if (j > 0)
            for (uint32_t col = 0; col < COLUMNS; col++) seq->npu_dma_wait(IT[col], S2MM, it_channel_0);
    }
    for (uint32_t col = 0; col < COLUMNS; col++) seq->npu_dma_wait(IT[col], S2MM, it_channel_0);
    seq->cmds2seq();
}

}  // namespace qwen3_asr_audio_gemm_seq
