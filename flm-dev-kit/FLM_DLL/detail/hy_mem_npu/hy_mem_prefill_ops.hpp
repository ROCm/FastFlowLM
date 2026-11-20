// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file hy_mem_prefill_ops.hpp
/// \brief The two heavy prefill operations, on the host: dequant_mm and attention
/// \note These are the stages a prefill overlay will take over later, so their
///       signatures are what such an overlay would take: whole tensors in, whole
///       tensors out, bf16 activations, the weights read in place from the decode
///       buffers (no second copy), the K/V read in place from the decode KV cache.
///       hy_mem_prefill.cpp calls them through HY_MEM_MM_ON_NPU / HY_MEM_ATTN_ON_NPU,
///       which are 0 for now.
///
///       The small per-row steps (norms, RoPE, GLU, router, residual) are in
///       hy_mem_cpu_functions.hpp.
#pragma once
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <immintrin.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "typedef.hpp"
#include "utils/avx512_util.hpp"
#include "hy_mem_model_def.hpp"

namespace hy_mem_cpu {

/// \brief threads for the host prefill: the whole machine (these kernels are the
///        whole cost of a CPU prefill). Override with FLM_PREFILL_THREADS.
inline int prefill_threads() {
    static const int nt = [] {
#ifdef _OPENMP
        int n = omp_get_max_threads();
#else
        int n = 1;
#endif
        if (const char* env = std::getenv("FLM_PREFILL_THREADS")) {
            const int v = std::atoi(env);
            if (v > 0) n = v;
        }
        return n < 1 ? 1 : n;
    }();
    return nt;
}

// =============================================================================
// Q4_K blocks
// =============================================================================
// One block = 32 rows x 256 columns, 4736 bytes, as in model.q4nx
// (FLM_Q4NX_Converter/verify_hy_mem_q4nx.py::decode_q4k):
//   uint8 s8  [8 groups][32 rows]   per-group scale
//   uint8 m8  [8 groups][32 rows]   per-group min
//   uint4 q   [256 cols][16 bytes]  byte b of a column = rows 2b (low nibble), 2b + 1 (high)
//   bf16  S   [32 rows]             super-block scale
//   bf16  M   [32 rows]             super-block min, stored negated
//   w[r][c] = S[r] * s8[g][r] * q[r][c] + M[r] * m8[g][r],   g = c / 32
constexpr int Q4K_ROWS = 32;
constexpr int Q4K_COLS = 256;
constexpr int Q4K_GROUPS = 8;
constexpr size_t Q4K_BLOCK = hy_mem_desc::Q4K_BLOCK_BYTES;
constexpr size_t Q4K_S8 = 0, Q4K_M8 = 256, Q4K_QS = 512, Q4K_S = 4608, Q4K_M = 4672;

/// \brief undo hy_mem_model_def.cpp's q4k_relayout_for_mmul (the MoE overlay's
///        int8-mmul order) back into the plain q4nx order above
/// \note relaid qs:     [k step 16][half t 2][16 columns][8 B], column = 16k + c
///       relaid scales: [h 2][t 2][a 4][16 rows], group = 4h + a, row = 16t + i
///       mins, S, M were not moved
inline void q4k_unrelayout(u8* plain, const u8* relaid) {
    memcpy(plain + Q4K_M8, relaid + Q4K_M8, Q4K_QS - Q4K_M8);           // mins
    memcpy(plain + Q4K_S, relaid + Q4K_S, Q4K_BLOCK - Q4K_S);          // S, M
    for (int k = 0; k < 16; k++)
        for (int t = 0; t < 2; t++)
            for (int c = 0; c < 16; c++)
                memcpy(plain + Q4K_QS + (k * 16 + c) * 16 + t * 8,
                       relaid + Q4K_QS + (k * 2 + t) * 128 + c * 8, 8);
    for (int h = 0; h < 2; h++)
        for (int t = 0; t < 2; t++)
            for (int a = 0; a < 4; a++)
                memcpy(plain + Q4K_S8 + (4 * h + a) * 32 + 16 * t,
                       relaid + Q4K_S8 + ((h * 2 + t) * 4 + a) * 16, 16);
}

/// \brief one plain block into a fp32 tile: tile[r * ld + col0 + c]
inline void q4k_decode_block(float* tile, int ld, int col0, const u8* blk) {
    const u8* s8 = blk + Q4K_S8;
    const u8* m8 = blk + Q4K_M8;
    const u8* qs = blk + Q4K_QS;
    const bf16* S = reinterpret_cast<const bf16*>(blk + Q4K_S);
    const bf16* M = reinterpret_cast<const bf16*>(blk + Q4K_M);
    for (int r = 0; r < Q4K_ROWS; r++) {
        float* row = tile + (size_t)r * ld + col0;
        const float S_r = (float)S[r], M_r = (float)M[r];
        const int byte = r / 2, shift = (r & 1) * 4;   // this row's nibble in a column
        for (int g = 0; g < Q4K_GROUPS; g++) {
            const float scale = S_r * (float)s8[g * Q4K_ROWS + r];
            const float min = M_r * (float)m8[g * Q4K_ROWS + r];
            for (int c = g * 32; c < g * 32 + 32; c++) {
                const int q = (qs[c * 16 + byte] >> shift) & 0x0F;
                row[c] = scale * (float)q + min;
            }
        }
    }
}

// =============================================================================
// Where a weight's blocks are in the decode buffers
// =============================================================================
/// \brief one weight matrix [N rows][K cols] inside a layer's proj buffer, as the
///        decode overlays laid it out (hy_mem_model_def.cpp). block() is the
///        executable spec of those layouts: hy_mem_desc::check_layer_layout tests
///        it byte for byte against model.q4nx.
struct q4k_matrix {
    enum kind_t {
        pairs,             // reorder_cpy(..., 2): row blocks in pairs, interleaved block by block
        expert_channels,   // one routed expert's [channel 0 | channel 1] streams (reorder_cpy(..., 4) + split)
    };
    const u8* base = nullptr;   // pairs: the matrix; expert_channels: the expert's start
    int K = 0, N = 0;           // columns (input), rows (output)
    kind_t kind = pairs;
    bool relaid = false;        // blocks in the int8-mmul order (every MoE-layer block)
    size_t channel_bytes = 0;   // expert_channels: one channel stream
    size_t part_offset = 0;     // expert_channels: where this matrix starts in each stream

    int blocks_per_row() const { return K / Q4K_COLS; }
    int row_blocks() const { return N / Q4K_ROWS; }

    /// \brief the block of row block rb (rows 32 rb ..), column block cb
    const u8* block(int rb, int cb) const {
        const size_t bpr = (size_t)blocks_per_row();
        if (kind == pairs) {
            // [pair rb / 2][cb][rb % 2]
            return base + (((size_t)(rb / 2) * bpr + cb) * 2 + rb % 2) * Q4K_BLOCK;
        }
        // groups of 4 row blocks, [group rb / 4][cb][rb % 4]; of each 4, the first
        // 2 went to channel 0 and the last 2 to channel 1
        const size_t group = (size_t)(rb / 4) * bpr + cb;
        const int i = rb % 4;
        return base + (size_t)(i / 2) * channel_bytes + part_offset + (group * 2 + i % 2) * Q4K_BLOCK;
    }

    /// \brief rows 32 rb .. 32 rb + 31, all K columns, as fp32: tile[r * K + col]
    void dequant_row_block(float* tile, int rb) const {
        alignas(64) u8 plain[Q4K_BLOCK];
        for (int cb = 0; cb < blocks_per_row(); cb++) {
            const u8* blk = block(rb, cb);
            if (relaid) {
                q4k_unrelayout(plain, blk);
                blk = plain;
            }
            q4k_decode_block(tile, K, cb * Q4K_COLS, blk);
        }
    }
};

// =============================================================================
// dequant_mm: y[L][N] = x[L][K] W^T
// =============================================================================
/// \brief y[l][r0 .. r0 + 31] = x[l] . tile[r], for one 32-row tile and all L rows
/// \note Register blocking: 4 rows of x by 4 rows of the tile at a time, so each
///       pair of loads feeds 16 FMAs and the tile (in L2) is read L / 4 times
///       instead of L times. A tail of fewer than 4 x rows points the missing
///       ones at the last real row and does not store them.
inline void mm_tile(bf16* y, int ldy, const float* tile, const float* xf, int L, int K) {
    for (int l = 0; l < L; l += 4) {
        const float* xr[4];
        for (int i = 0; i < 4; i++) xr[i] = xf + (size_t)std::min(l + i, L - 1) * K;
        for (int r = 0; r < Q4K_ROWS; r += 4) {
            __m512 acc[4][4];
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++) acc[i][j] = _mm512_setzero_ps();
            const float* wr = tile + (size_t)r * K;
            for (int k = 0; k < K; k += 16) {
                const __m512 w0 = _mm512_loadu_ps(wr + k);
                const __m512 w1 = _mm512_loadu_ps(wr + K + k);
                const __m512 w2 = _mm512_loadu_ps(wr + 2 * (size_t)K + k);
                const __m512 w3 = _mm512_loadu_ps(wr + 3 * (size_t)K + k);
                for (int i = 0; i < 4; i++) {
                    const __m512 xv = _mm512_loadu_ps(xr[i] + k);
                    acc[i][0] = _mm512_fmadd_ps(xv, w0, acc[i][0]);
                    acc[i][1] = _mm512_fmadd_ps(xv, w1, acc[i][1]);
                    acc[i][2] = _mm512_fmadd_ps(xv, w2, acc[i][2]);
                    acc[i][3] = _mm512_fmadd_ps(xv, w3, acc[i][3]);
                }
            }
            for (int i = 0; i < 4 && l + i < L; i++)
                for (int j = 0; j < 4; j++)
                    y[(size_t)(l + i) * ldy + r + j] = (bf16)_mm512_reduce_add_ps(acc[i][j]);
        }
    }
}

/// \brief y[L][N] = x[L][K] W^T, W read in place from the decode buffer
/// \param y [L][N] bf16, row major
/// \param x [L][K] bf16, row major
/// \note Each 32-row block of W is dequantized once and used for all L rows, so
///       the weight traffic is the weight size whatever L is. fp32 inside, bf16
///       out (what a dequant_mm overlay would return).
inline void dequant_mm(bf16* y, const q4k_matrix& w, const bf16* x, int L) {
    if (L <= 0) return;
    const int K = w.K, N = w.N;
    assert(K % Q4K_COLS == 0 && N % Q4K_ROWS == 0);

    // x to fp32 once, not once per tile
    std::vector<float> xf((size_t)L * K);
    #pragma omp parallel for num_threads(prefill_threads()) schedule(static)
    for (int l = 0; l < L; l++) {
        for (int k = 0; k < K; k += 16) {
            _mm512_storeu_ps(xf.data() + (size_t)l * K + k, load_bfloat16_to_m512(x + (size_t)l * K + k));
        }
    }

    #pragma omp parallel num_threads(prefill_threads())
    {
        std::vector<float> tile((size_t)Q4K_ROWS * K);
        #pragma omp for schedule(dynamic)
        for (int rb = 0; rb < w.row_blocks(); rb++) {
            w.dequant_row_block(tile.data(), rb);
            mm_tile(y + (size_t)rb * Q4K_ROWS, N, tile.data(), xf.data(), L, K);
        }
    }
}

// =============================================================================
// attention: causal GQA over the decode KV cache
// =============================================================================
/// \brief one layer's KV cache as the decode overlays keep it:
///        [K col 0][K col 1][V col 0][V col 1], each MAX_L x (2 heads x DH),
///        i.e. all K then all V (same rule as hy_mem_desc::kv_block_offset)
struct kv_cache_view {
    bf16* base = nullptr;
    uint32_t MAX_L = 0;
    int DH = 0;
    int heads_per_col = 0;

    size_t col_row() const { return (size_t)heads_per_col * DH; }
    /// \brief K (kv = 0) or V (kv = 1) of KV head `head` at token j
    bf16* row(int kv, int head, int j) const {
        const int col = head / heads_per_col;
        const size_t block = (size_t)(2 * kv + col) * MAX_L * col_row();
        return base + block + (size_t)j * col_row() + (size_t)(head % heads_per_col) * DH;
    }
};

/// \brief out[l] = softmax(q[l] K^T / sqrt(DH)) V for every q head, causal
/// \param out [L][num_heads * DH] bf16, q-head order
/// \param q   [L] rows of num_heads * DH, row stride q_stride (normed and roped)
/// \param kv  the cache; rows 0 .. pos0 + L - 1 are filled (this call's rows too)
/// \param pos0 position of q row 0: row l attends to cache rows 0 .. pos0 + l
/// \note q head h reads KV head h / (num_heads / num_kv_heads), as transformers'
///       repeat_kv. fp32 scores and softmax.
inline void attention(bf16* out, const bf16* q, int q_stride, const kv_cache_view& kv,
                      int L, int pos0, int num_heads, int num_kv_heads) {
    const int DH = kv.DH;
    const int group = num_heads / num_kv_heads;
    const float scale = 1.0f / std::sqrt((float)DH);
    const int L_kv = pos0 + L;

    #pragma omp parallel num_threads(prefill_threads())
    {
        std::vector<float> scores(L_kv);
        std::vector<float> qf(DH), acc(DH);
        #pragma omp for collapse(2) schedule(dynamic, 16)
        for (int l = 0; l < L; l++) {
            for (int h = 0; h < num_heads; h++) {
                const int hk = h / group;
                const int last = pos0 + l;                  // causal: keys 0 .. last
                const bf16* qh = q + (size_t)l * q_stride + (size_t)h * DH;
                for (int d = 0; d < DH; d++) qf[d] = (float)qh[d] * scale;

                float m = -INFINITY;
                for (int j = 0; j <= last; j++) {
                    const bf16* kj = kv.row(0, hk, j);
                    __m512 s = _mm512_setzero_ps();
                    for (int d = 0; d < DH; d += 16)
                        s = _mm512_fmadd_ps(_mm512_loadu_ps(qf.data() + d), load_bfloat16_to_m512(kj + d), s);
                    scores[j] = _mm512_reduce_add_ps(s);
                    m = std::max(m, scores[j]);
                }
                float sum = 0.0f;
                for (int j = 0; j <= last; j++) {
                    scores[j] = std::exp(scores[j] - m);
                    sum += scores[j];
                }
                std::fill(acc.begin(), acc.end(), 0.0f);
                for (int j = 0; j <= last; j++) {
                    const bf16* vj = kv.row(1, hk, j);
                    const __m512 p = _mm512_set1_ps(scores[j]);
                    for (int d = 0; d < DH; d += 16)
                        _mm512_storeu_ps(acc.data() + d, _mm512_fmadd_ps(p, load_bfloat16_to_m512(vj + d),
                                                                          _mm512_loadu_ps(acc.data() + d)));
                }
                bf16* o = out + (size_t)l * num_heads * DH + (size_t)h * DH;
                const float inv = 1.0f / sum;
                for (int d = 0; d < DH; d++) o[d] = (bf16)(acc[d] * inv);
            }
        }
    }
}

}  // namespace hy_mem_cpu
