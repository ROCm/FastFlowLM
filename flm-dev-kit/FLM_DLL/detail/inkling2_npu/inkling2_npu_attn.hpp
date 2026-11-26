// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file inkling2_npu_attn.hpp
/// \brief prompt attention on the NPU (Inkling2/prefill_attention): query packs, the bias table and the sequence
/// \note Kept in step with FLM_Xclbin/Inkling2/prefill_attention/host/seq_gen.hpp (RTP addresses included).
#pragma once
#include <cassert>

#include "npu_utils/npu_utils.hpp"
#include "typedef.hpp"

// Inkling2 prefill attention: 24 query heads, 8 KV heads (3 query heads per KV head), DH 128.
// Arguments: 0 o [L][24][128], 1 query packs (pack_q), 2 kv (head-major [8][MAX_L / 16] bfp16 K tiles,
// pack_kv_tile, then the V tiles at KV_CACHE_SIZE / 2 bf16 units),
// 3 bias table (pack_table).
namespace inkling2_attn {

constexpr int DH = 128;
constexpr int HEADS = 24;
constexpr int NR = 16;                                // rel features per head
constexpr int QP = 32 * DH + 32 * NR + 32;            // pack of one core row: q tiles, rel tiles, sinks
constexpr int ROUND = 128;                            // queries per round
constexpr int TAB_ROWS = 1280;
constexpr int TAB_SIZE = 2 * TAB_ROWS * 8;
constexpr int REL_EXT = 1024;                         // the bias covers distances 0 .. REL_EXT - 1
constexpr int TAB_D0 = 1151;                          // table row x holds distance TAB_D0 - x

constexpr int KV_TILE = 16 * DH / 8 * 9;              // bytes of the K (or V) tile of 16 keys: 32 bfp16 8x8 tiles
constexpr int KV_ROW = KV_TILE / 16 / 2;              // bf16 units per key row

/// \brief byte offset, in the K tile of 16 keys, of the bfp16 block of key k (0 .. 15) and dims 8 i .. 8 i + 7
inline size_t kv_k_block(int k, int i) { return (size_t)(i * 2 + k / 8) * 72 + (k % 8) * 9; }
/// \brief byte offset, in the V tile of 16 keys, of the bfp16 block of dim c and keys 8 kb .. 8 kb + 7 (V^T)
inline size_t kv_v_block(int c, int kb) { const int j = c / 8; return (size_t)(j / 2 * 4 + kb * 2 + j % 2) * 72 + (c % 8) * 9; }

/// \brief the K and V tiles of 16 keys from k / v rows of DH floats (row stride ld); pack2(p, b0, b1) packs
///        p[0 .. 7] into the bfp16 block b0 and p[8 .. 15] into b1
template <typename Pack2>
inline void pack_kv_tile(uint8_t* kt, uint8_t* vt, const float* k, const float* v, size_t ld, Pack2 pack2) {
    for (int r = 0; r < 16; r++)
        for (int i = 0; i < DH / 8; i += 2) pack2(k + r * ld + i * 8, kt + kv_k_block(r, i), kt + kv_k_block(r, i + 1));
    alignas(64) float col[16];
    for (int c = 0; c < DH; c++) {
        for (int r = 0; r < 16; r++) col[r] = v[r * ld + c];
        pack2(col, vt + kv_v_block(c, 0), vt + kv_v_block(c, 1));
    }
}

constexpr uint32_t L_BEGIN_ADDR = 15680;
constexpr uint32_t L_END_ADDR = 30208;
constexpr uint32_t WINDOW_ADDR = 42112;

/// \brief elements of the query packs for rounds rounds
inline size_t q_pack_size(int rounds) { return (size_t)HEADS * rounds * 4 * QP; }

/// \brief the query pack of query row t (0 .. rounds * 128 - 1 from L_begin) of head h
/// \param q 128 query values, r NR rel features, sink the sink logit
template <typename T>
inline void pack_q(bf16* pack, int rounds, int h, int t, const T* q, const T* r, float sink) {
    const int round = t / ROUND, row = (t % ROUND) / 32, i = t % 32;
    bf16* p = pack + (((size_t)h * rounds + round) * 4 + row) * QP;
    const int rb = i / 8, ri = i % 8;
    for (int c = 0; c < DH; c++)
        p[((rb * (DH / 8) + c / 8) * 8 + ri) * 8 + c % 8] = (bf16)(float)q[c];
    bf16* pr = p + 32 * DH;
    for (int c = 0; c < NR; c++)
        pr[((rb * (NR / 8) + c / 8) * 8 + ri) * 8 + c % 8] = (bf16)(float)r[c];
    p[32 * DH + 32 * NR + i] = (bf16)sink;
}

/// \brief the bias table from the layer's rel_bias [REL_EXT][NR]
template <typename T>
inline void pack_table(bf16* tab, const T* rel_bias) {
    for (int half = 0; half < 2; half++)
        for (int x = 0; x < TAB_ROWS; x++) {
            const int d = TAB_D0 - x;
            for (int f = 0; f < 8; f++)
                tab[((size_t)half * TAB_ROWS + x) * 8 + f] =
                    (d >= 0 && d < REL_EXT) ? (bf16)(float)rel_bias[d * NR + half * 8 + f] : (bf16)0.0f;
        }
}

/// \param L_begin, L_end query rows [L_begin, L_end), multiples of 128; keys are rows 0 .. L_end - 1
/// \param MAX_L rows of the kv buffer per chunk; window 0: global attention, else keys with q - k < window
inline void gen_seq(npu_sequence* seq, uint32_t L_begin, uint32_t L_end, int MAX_L, int window, int KV_CACHE_SIZE) {
    npu_tiles IT[4][2] = {{IT0, IT1}, {IT2, IT3}, {IT4, IT5}, {IT6, IT7}};
    assert(L_begin % ROUND == 0 && L_end % ROUND == 0);
    const int using_window = window > 0 ? window : (int)L_end;
    seq->clear_cmds();
    for (int row = 2; row < 6; row++)
        for (int col = 0; col < 8; col++) {
            npu_tiles tile = get_tile(row, col);
            seq->rtp_write(tile, L_BEGIN_ADDR, L_begin);
            seq->rtp_write(tile, L_END_ADDR, L_end);
            seq->rtp_write(tile, WINDOW_ADDR, using_window);
        }
    const int rounds = (L_end - L_begin) / ROUND;
    for (int head = 0; head < HEADS / 4; head++) {
        for (int round = 0; round < rounds; round++) {
            const int Lq = L_begin + round * ROUND;
            const int kv_begin = Lq - using_window > 0 ? Lq - using_window : 0;
            const int kv_length = Lq - kv_begin + ROUND;
            const int bd = (round % 2) * 8;
            auto cu_io = [&](int cu, int hq, int kv_head) {
                for (int col = 0; col < 2; col++) {
                    size_t y_off = (size_t)hq * DH + (size_t)round * ROUND * DH * HEADS + (size_t)col * 64 * DH * HEADS;
                    seq->npu_dma_memcpy_nd(2, 0, S2MM, IT[cu][col], (npu_bd_id)(bd + 0), it_channel_0,
                                           {0, 0, 0, (uint32_t)y_off}, {1, 1, 64, (uint32_t)DH},
                                           {0, 0, (uint32_t)DH * HEADS, 1}, -1, 0, true);
                }
                size_t q_off = (((size_t)hq * rounds + round) * 4) * QP;
                seq->npu_dma_memcpy_nd(2, 1, MM2S, IT[cu][0], (npu_bd_id)(bd + 1), it_channel_0,
                                       {0, 0, 0, (uint32_t)q_off}, {1, 1, 4 * QP / 128, 128}, {0, 0, 128, 1},
                                       -1, 0, false);
                if (kv_head < 0) return;
                for (int v = 0; v < 2; v++) {
                    // head-major tiles: one contiguous run of kv_length rows, as kv_length / 128 x 16 runs of 8 rows
                    // (a dimension wraps at 1024: one of kv_length / 8 stops at 8192 keys)
                    size_t off = ((size_t)kv_head * MAX_L + kv_begin) * KV_ROW + (v ? (size_t)KV_CACHE_SIZE / 2 : 0);
                    seq->npu_dma_memcpy_nd(2, 2, MM2S, IT[cu][1], (npu_bd_id)(bd + 3 + v), v ? it_channel_1 : it_channel_0,
                                           {0, 0, 0, (uint32_t)off},
                                           {1, (uint32_t)kv_length / 128, 16, 8 * (uint32_t)KV_ROW},
                                           {0, 128 * (uint32_t)KV_ROW, 8 * (uint32_t)KV_ROW, 1}, -1, 0, false);
                }
            };
            // CUs 0-2 share the kv head of query heads 3 head .. 3 head + 2 (sent once, to memtiles 1, 3, 5)
            cu_io(0, head * 3 + 0, head);
            cu_io(1, head * 3 + 1, -1);
            cu_io(2, head * 3 + 2, -1);
            // bias table, broadcast to the 4 even memtiles
            seq->npu_dma_memcpy_nd(2, 3, MM2S, IT[1][1], (npu_bd_id)(bd + 5), it_channel_0,
                                   {0, 0, 0, 0}, {1, 1, TAB_SIZE / 128, 128}, {0, 0, 128, 1}, -1, 0, false);
            cu_io(3, head + 18, 6 + head / 3);
            if (round > 0)
                for (int cu = 0; cu < 4; cu++)
                    for (int col = 0; col < 2; col++) seq->npu_dma_wait(IT[cu][col], S2MM, it_channel_0);
        }
        for (int cu = 0; cu < 4; cu++)
            for (int col = 0; col < 2; col++) seq->npu_dma_wait(IT[cu][col], S2MM, it_channel_0);
    }
    seq->cmds2seq();
}

}  // namespace inkling2_attn
