// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#include "qwen3_asr_cpu_decoder.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include <omp.h>
#include <stdexcept>

#include "bf16_util.hpp"
#include "../omp_threads.hpp"
#include "mapped_safetensors.hpp"

namespace qwen3_asr_cpu {

namespace {
struct prof_t { double mm = 0, att = 0, total = 0; } g_prof;
inline double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// ---- Q8_0 q4nx block geometry (see the header) ------------------------------------------------------------------
constexpr int ROWS_PER_BLOCK = 32;
constexpr int COLS_PER_BLOCK = 256;
constexpr int GROUP = 32;                                            // columns that share one scale
constexpr int SCALE_BYTES = (COLS_PER_BLOCK / GROUP) * ROWS_PER_BLOCK * 2;   // 512
constexpr int BLOCK_BYTES = SCALE_BYTES + ROWS_PER_BLOCK * COLS_PER_BLOCK;   // 8704
constexpr int TOKENS_PER_TILE = 8;                                   // 8 tokens x 2 zmm = 16 accumulator registers

struct q8_matrix {
    int rows = 0;                 // output features (multiple of 32)
    int cols = 0;                 // input features (multiple of 256)
    const uint8_t* blocks = nullptr;
};

// round 16 floats to the nearest bf16 value (kept as floats), ties to even; same as bf16r() per lane
inline __m512 bf16_round_ps(__m512 v) {
    __m512i u = _mm512_castps_si512(v);
    __m512i lsb = _mm512_and_si512(_mm512_srli_epi32(u, 16), _mm512_set1_epi32(1));
    u = _mm512_add_epi32(u, _mm512_add_epi32(lsb, _mm512_set1_epi32(0x7fff)));
    u = _mm512_and_si512(u, _mm512_set1_epi32((int)0xffff0000u));
    return _mm512_castsi512_ps(u);
}

// 16 bf16 (as uint16) -> 16 floats
inline __m512 load_bf16x16(const uint16_t* p) {
    __m512i w = _mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i*)p));
    return _mm512_castsi512_ps(_mm512_slli_epi32(w, 16));
}

/// Y[t][row0 .. row0+31] = bf16( sum_c X[t][c] * W[row][c] ) for T tokens, one 32-row block row of a Q8_0 matrix.
/// Per column: dequantize the 32 int8 of that column (2 zmm, exact in fp32: bf16 scale x int8), then one broadcast
/// FMA per token. The dequantized column is reused by all T tokens, so the weight stream is read once per tile.
template <int T>
inline void block_row_tile(const uint8_t* row_blocks, int col_blocks, const float* X, int ldx, float* Y, int ldy) {
    __m512 acc[T][2];
    for (int t = 0; t < T; ++t) acc[t][0] = acc[t][1] = _mm512_setzero_ps();

    for (int cb = 0; cb < col_blocks; ++cb) {
        const uint8_t* blk = row_blocks + (size_t)cb * BLOCK_BYTES;
        const uint16_t* scales = (const uint16_t*)blk;                       // [8 groups][32 rows] bf16
        const int8_t* q = (const int8_t*)(blk + SCALE_BYTES);                // [256 cols][32 rows]
        const float* x = X + (size_t)cb * COLS_PER_BLOCK;
        for (int g = 0; g < COLS_PER_BLOCK / GROUP; ++g) {
            const __m512 s0 = load_bf16x16(scales + g * ROWS_PER_BLOCK);
            const __m512 s1 = load_bf16x16(scales + g * ROWS_PER_BLOCK + 16);
            for (int cc = 0; cc < GROUP; ++cc) {
                const int c = g * GROUP + cc;
                const int8_t* col = q + (size_t)c * ROWS_PER_BLOCK;
                const __m512 w0 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*)col))), s0);
                const __m512 w1 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*)(col + 16)))), s1);
                for (int t = 0; t < T; ++t) {
                    const __m512 bx = _mm512_set1_ps(x[(size_t)t * ldx + c]);
                    acc[t][0] = _mm512_fmadd_ps(w0, bx, acc[t][0]);
                    acc[t][1] = _mm512_fmadd_ps(w1, bx, acc[t][1]);
                }
            }
        }
    }
    for (int t = 0; t < T; ++t) {
        _mm512_storeu_ps(Y + (size_t)t * ldy, bf16_round_ps(acc[t][0]));
        _mm512_storeu_ps(Y + (size_t)t * ldy + 16, bf16_round_ps(acc[t][1]));
    }
}


// ---- int8 VNNI path --------------------------------------------------------------------------------------------------------
// Activations are quantized to int8 per 32-column group (scale = max|x| / 127, as Q8_0 does). vpdpbusd multiplies unsigned bytes with
// signed bytes, so the activation is stored with +128 (u8) and the error term 128 * sum(w) of the group is subtracted afterwards;
// sum(w) is one more vpdpbusd against a vector of ones per 4 columns, shared by all tokens of the tile.
constexpr int VNNI_TOKENS = 6;     // 6 tokens x (fp32 acc + int32 acc) x 2 zmm = 24 registers + weights/scales

struct quant_act {
    std::vector<uint8_t> q;        // [n][cols], int8 + 128
    std::vector<float> s;          // [n][cols / 32]
};

void quantize_activations(const float* X, int n, int cols, quant_act& out) {
    const int groups = cols / GROUP;
    out.q.resize((size_t)n * cols);
    out.s.resize((size_t)n * groups);
#pragma omp parallel for num_threads(qwen3_asr_cpu::asr_threads()) schedule(static)
    for (int t = 0; t < n; ++t) {
        const float* x = X + (size_t)t * cols;
        for (int g = 0; g < groups; ++g) {
            const float* xg = x + g * GROUP;
            __m512 a = _mm512_max_ps(_mm512_abs_ps(_mm512_loadu_ps(xg)), _mm512_abs_ps(_mm512_loadu_ps(xg + 16)));
            const float amax = _mm512_reduce_max_ps(a);
            const float scale = amax / 127.0f;
            const float inv = scale > 0.f ? 127.0f / amax : 0.f;
            out.s[(size_t)t * groups + g] = scale;
            uint8_t* qg = out.q.data() + (size_t)t * cols + g * GROUP;
            for (int i = 0; i < GROUP; ++i) {
                const float v = xg[i] * inv;
                const int r = (int)(v >= 0.f ? v + 0.5f : v - 0.5f);          // round half away from zero
                qg[i] = (uint8_t)(r + 128);
            }
        }
    }
}

/// Byte indices for vpermi2b: out lane l (4 bytes) = rows (row0 + l) of the 4 consecutive columns k0..k3, which sit 32 bytes apart in the block
/// ([col][32 rows] int8): index = kj * 32 + row. Two zmm loads (4 columns = 128 B) are the two sources.
inline __m512i vnni_index(int row0) {
    alignas(64) uint8_t idx[64];
    for (int i = 0; i < 64; ++i) idx[i] = (uint8_t)((i % 4) * 32 + row0 + i / 4);
    return _mm512_load_si512((const void*)idx);
}

/// One 32-row block row of a matrix, interleaved for vpdpbusd once and reused by every token tile: per 32-column group, the 8 x 2
/// weight vectors (4 columns x 16 rows each), the error terms 128 * sum(w) of rows 0-15 / 16-31 and the row scales as fp32.
struct packed_row_block {
    static constexpr int VECS = 8 * 2 + 2 + 2;   // weights, 128 * sum(w), scales
    std::vector<__m512i> v;                      // [groups][VECS]
};

void pack_row_block(const uint8_t* row_blocks, int col_blocks, packed_row_block& out) {
    const __m512i ones = _mm512_set1_epi8(1);
    const __m512i idx_lo = vnni_index(0), idx_hi = vnni_index(16);
    const int groups = col_blocks * (COLS_PER_BLOCK / GROUP);
    out.v.resize((size_t)groups * packed_row_block::VECS);
    __m512i* o = out.v.data();
    for (int cb = 0; cb < col_blocks; ++cb) {
        const uint8_t* blk = row_blocks + (size_t)cb * BLOCK_BYTES;
        const uint16_t* scales = (const uint16_t*)blk;
        const int8_t* q = (const int8_t*)(blk + SCALE_BYTES);
        for (int g = 0; g < COLS_PER_BLOCK / GROUP; ++g, o += packed_row_block::VECS) {
            __m512i wsum0 = _mm512_setzero_si512(), wsum1 = _mm512_setzero_si512();
            for (int j = 0; j < GROUP / 4; ++j) {
                const int8_t* col = q + (size_t)(g * GROUP + j * 4) * ROWS_PER_BLOCK;
                const __m512i z0 = _mm512_loadu_si512((const void*)col), z1 = _mm512_loadu_si512((const void*)(col + 64));
                const __m512i w0 = _mm512_permutex2var_epi8(z0, idx_lo, z1);                                // rows 0-15, k0..k3 per lane
                const __m512i w1 = _mm512_permutex2var_epi8(z0, idx_hi, z1);                                // rows 16-31
                wsum0 = _mm512_dpbusd_epi32(wsum0, ones, w0);
                wsum1 = _mm512_dpbusd_epi32(wsum1, ones, w1);
                o[2 * j] = w0;
                o[2 * j + 1] = w1;
            }
            o[16] = _mm512_slli_epi32(wsum0, 7);                                                               // 128 * sum(w)
            o[17] = _mm512_slli_epi32(wsum1, 7);
            o[18] = _mm512_castps_si512(load_bf16x16(scales + g * ROWS_PER_BLOCK));
            o[19] = _mm512_castps_si512(load_bf16x16(scales + g * ROWS_PER_BLOCK + 16));
        }
    }
}

template <int T>
inline void block_row_tile_vnni(const packed_row_block& P, const quant_act& A, int t0, int cols, float* Y, int ldy) {
    const int groups = cols / GROUP;
    __m512 acc[T][2];
    for (int t = 0; t < T; ++t) acc[t][0] = acc[t][1] = _mm512_setzero_ps();
    const __m512i* pw = P.v.data();
    for (int gg = 0; gg < groups; ++gg, pw += packed_row_block::VECS) {
        __m512i ai[T][2];
        for (int t = 0; t < T; ++t) ai[t][0] = ai[t][1] = _mm512_setzero_si512();
        const size_t xoff = (size_t)gg * GROUP;
        for (int j = 0; j < GROUP / 4; ++j) {
            const __m512i w0 = _mm512_load_si512((const void*)(pw + 2 * j)), w1 = _mm512_load_si512((const void*)(pw + 2 * j + 1));
            for (int t = 0; t < T; ++t) {
                int32_t four;
                std::memcpy(&four, A.q.data() + (size_t)(t0 + t) * cols + xoff + 4 * j, 4);
                const __m512i bx = _mm512_set1_epi32(four);
                ai[t][0] = _mm512_dpbusd_epi32(ai[t][0], bx, w0);
                ai[t][1] = _mm512_dpbusd_epi32(ai[t][1], bx, w1);
            }
        }
        const __m512i off0 = _mm512_load_si512((const void*)(pw + 16)), off1 = _mm512_load_si512((const void*)(pw + 17));
        const __m512 s0 = _mm512_castsi512_ps(_mm512_load_si512((const void*)(pw + 18)));
        const __m512 s1 = _mm512_castsi512_ps(_mm512_load_si512((const void*)(pw + 19)));
        for (int t = 0; t < T; ++t) {
            const __m512 sx = _mm512_set1_ps(A.s[(size_t)(t0 + t) * groups + gg]);
            acc[t][0] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(ai[t][0], off0)), _mm512_mul_ps(s0, sx), acc[t][0]);
            acc[t][1] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(ai[t][1], off1)), _mm512_mul_ps(s1, sx), acc[t][1]);
        }
    }
    for (int t = 0; t < T; ++t) {
        _mm512_storeu_ps(Y + (size_t)t * ldy, bf16_round_ps(acc[t][0]));
        _mm512_storeu_ps(Y + (size_t)t * ldy + 16, bf16_round_ps(acc[t][1]));
    }
}

void matmul_q8_vnni(const q8_matrix& W, const float* X, int n, float* Y) {
    static thread_local quant_act A;       // the caller thread owns it; the OpenMP team only reads it
    quantize_activations(X, n, W.cols, A);
    const int col_blocks = W.cols / COLS_PER_BLOCK;
    const int row_blocks = W.rows / ROWS_PER_BLOCK;
    const size_t row_block_bytes = (size_t)col_blocks * BLOCK_BYTES;
    const quant_act& Ar = A;
#pragma omp parallel for num_threads(qwen3_asr_cpu::asr_threads()) schedule(dynamic)
    for (int rb = 0; rb < row_blocks; ++rb) {
        static thread_local packed_row_block P;   // one block row, interleaved once for all token tiles (<= 240 KB: stays in L2)
        pack_row_block(W.blocks + (size_t)rb * row_block_bytes, col_blocks, P);
        for (int t0 = 0; t0 < n; t0 += VNNI_TOKENS) {
            const int T = std::min(VNNI_TOKENS, n - t0);
            float* y = Y + (size_t)t0 * W.rows + (size_t)rb * ROWS_PER_BLOCK;
            switch (T) {
                case 1: block_row_tile_vnni<1>(P, Ar, t0, W.cols, y, W.rows); break;
                case 2: block_row_tile_vnni<2>(P, Ar, t0, W.cols, y, W.rows); break;
                case 3: block_row_tile_vnni<3>(P, Ar, t0, W.cols, y, W.rows); break;
                case 4: block_row_tile_vnni<4>(P, Ar, t0, W.cols, y, W.rows); break;
                case 5: block_row_tile_vnni<5>(P, Ar, t0, W.cols, y, W.rows); break;
                default: block_row_tile_vnni<6>(P, Ar, t0, W.cols, y, W.rows); break;
            }
        }
    }
}

/// Y[n][rows] = bf16( X[n][cols] * W^T ), fp32 path, threads over the 32-row block rows.
void matmul_q8_fp32(const q8_matrix& W, const float* X, int n, float* Y) {
    const int col_blocks = W.cols / COLS_PER_BLOCK;
    const int row_blocks = W.rows / ROWS_PER_BLOCK;
    const size_t row_block_bytes = (size_t)col_blocks * BLOCK_BYTES;
#pragma omp parallel for num_threads(qwen3_asr_cpu::asr_threads()) schedule(static)
    for (int rb = 0; rb < row_blocks; ++rb) {
        const uint8_t* rbp = W.blocks + (size_t)rb * row_block_bytes;
        for (int t0 = 0; t0 < n; t0 += TOKENS_PER_TILE) {
            const int T = std::min(TOKENS_PER_TILE, n - t0);
            const float* x = X + (size_t)t0 * W.cols;
            float* y = Y + (size_t)t0 * W.rows + (size_t)rb * ROWS_PER_BLOCK;
            switch (T) {
                case 1: block_row_tile<1>(rbp, col_blocks, x, W.cols, y, W.rows); break;
                case 2: block_row_tile<2>(rbp, col_blocks, x, W.cols, y, W.rows); break;
                case 3: block_row_tile<3>(rbp, col_blocks, x, W.cols, y, W.rows); break;
                case 4: block_row_tile<4>(rbp, col_blocks, x, W.cols, y, W.rows); break;
                case 5: block_row_tile<5>(rbp, col_blocks, x, W.cols, y, W.rows); break;
                case 6: block_row_tile<6>(rbp, col_blocks, x, W.cols, y, W.rows); break;
                case 7: block_row_tile<7>(rbp, col_blocks, x, W.cols, y, W.rows); break;
                default: block_row_tile<8>(rbp, col_blocks, x, W.cols, y, W.rows); break;
            }
        }
    }
}

/// Y[n][rows] = bf16( X[n][cols] * W^T ): int8 VNNI for several tokens (prefill), the exact fp32 path otherwise or with QWEN3_ASR_PREFILL_FP32=1
void matmul_q8(const q8_matrix& W, const float* X, int n, float* Y) {
    const double t_start = now_ms();
    struct stop { double s; ~stop() { g_prof.mm += now_ms() - s; } } stop_guard{t_start};
    static const bool force_fp32 = std::getenv("QWEN3_ASR_PREFILL_FP32") != nullptr;
    if (n >= 4 && !force_fp32) matmul_q8_vnni(W, X, n, Y);
    else matmul_q8_fp32(W, X, n, Y);
}

/// HF RMSNorm: fp32 statistics, x*rsqrt in fp32, round to bf16, then weight * that rounded to bf16 again.
void rmsnorm_row(const float* x, const float* w, float* y, int d, float eps) {
    float ss = 0.f;
    for (int i = 0; i < d; ++i) ss += x[i] * x[i];
    const float r = 1.0f / std::sqrt(ss / (float)d + eps);
    for (int i = 0; i < d; ++i) y[i] = bf16r(w[i] * bf16r(x[i] * r));
}

std::vector<float> load_bf16_vector(const mapped_tensor& t) {
    std::vector<float> v(t.bytes / 2);
    const uint16_t* p = (const uint16_t*)t.data;
    for (size_t i = 0; i < v.size(); ++i) v[i] = bf2f(p[i]);
    return v;
}

q8_matrix load_q8(const mapped_safetensors& f, const std::string& name, int rows, int cols) {
    const mapped_tensor& t = f.get(name);
    if (t.dtype != "I8" || t.shape.size() != 3 || t.shape[2] != (size_t)BLOCK_BYTES ||
        (int)t.shape[0] * ROWS_PER_BLOCK != rows || (int)t.shape[1] * COLS_PER_BLOCK != cols)
        throw std::runtime_error("unexpected Q8_0 tensor layout for " + name);
    return q8_matrix{rows, cols, t.data};
}

/// e^x for x in [-87, 88] (clamped outside): 2^n * p(r), r = x - n ln2 in [-ln2/2, ln2/2], p the degree-7 Taylor polynomial
/// (relative error < 1e-8, below fp32 rounding)
inline __m512 exp_ps(__m512 x) {
    x = _mm512_min_ps(_mm512_max_ps(x, _mm512_set1_ps(-87.0f)), _mm512_set1_ps(88.0f));
    const __m512 n = _mm512_roundscale_ps(_mm512_mul_ps(x, _mm512_set1_ps(1.44269504088896341f)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m512 r = _mm512_fnmadd_ps(n, _mm512_set1_ps(0.693359375f), x);          // ln2 split in a high part exact in fp32 and the rest
    r = _mm512_fnmadd_ps(n, _mm512_set1_ps(-2.12194440e-4f), r);
    __m512 y = _mm512_set1_ps(1.0f / 5040);
    y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(1.0f / 720));
    y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(1.0f / 120));
    y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(1.0f / 24));
    y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(1.0f / 6));
    y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(0.5f));
    y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(1.0f));
    y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(1.0f));
    return _mm512_scalef_ps(y, n);
}

/// Causal attention of n new tokens (positions pos0 ..) over the cache rows [0, pos0 + n); head_dim 128, at most 2 query heads per kv
/// head. A work item is one kv head x ATT_TOKENS tokens (2 * ATT_TOKENS queries). Keys and values are read in chunks of ATT_KEYS rows
/// (128 KB of keys, 128 KB of values: they stay in L2) and every chunk serves all queries of the item: scores 16 keys x 8 queries at a
/// time from a transposed copy of the keys, then a softmax per query, then the outputs per token with both heads sharing the value
/// loads. fp32 inside, the output rounded to bf16 (as the per-(token, head) loop).
constexpr int ATT_TOKENS = 32;
constexpr int ATT_KEYS = 256;
void attention_blocked(const float* q, const std::vector<uint16_t>& Kc, const std::vector<uint16_t>& Vc, int pos0, int n, int H, int KV, int DH,
                       float* att, std::vector<float>& Kt) {
    const int DQ = H * DH, DK = KV * DH, group = H / KV, L = pos0 + n, Lp = (L + 15) & ~15;
    const float scale = 1.0f / std::sqrt((float)DH);
    // keys transposed per kv head: Kt[kvh][d][j], zero beyond L
    Kt.assign((size_t)KV * DH * Lp, 0.f);
#pragma omp parallel for num_threads(qwen3_asr_cpu::asr_threads())
    for (int hd = 0; hd < KV * DH; ++hd) {
        const int kvh = hd / DH, d = hd % DH;
        float* dst = &Kt[(size_t)hd * Lp];
        for (int j = 0; j < L; ++j) dst[j] = bf2f(Kc[(size_t)j * DK + (size_t)kvh * DH + d]);
    }
    const int tblocks = (n + ATT_TOKENS - 1) / ATT_TOKENS;
#pragma omp parallel for num_threads(qwen3_asr_cpu::asr_threads()) schedule(dynamic)
    for (int item = 0; item < KV * tblocks; ++item) {
        const int kvh = item % KV, tb = tblocks - 1 - item / KV;   // longest rows first
        const int t0 = tb * ATT_TOKENS, tn = std::min(ATT_TOKENS, n - t0), QN = tn * group;
        const int Lmax = pos0 + t0 + tn, Lmax_p = (Lmax + 15) & ~15;
        static thread_local std::vector<float> S;
        S.resize((size_t)ATT_TOKENS * 2 * Lmax_p);
        const float* qp[ATT_TOKENS * 2];
        for (int i = 0; i < QN; ++i) qp[i] = q + (size_t)(t0 + i / group) * DQ + (size_t)(kvh * group + i % group) * DH;
        // scores, key chunk by key chunk, 8 queries x 16 keys per register block
        const float* kt = &Kt[(size_t)kvh * DH * Lp];
        for (int c0 = 0; c0 < Lmax; c0 += ATT_KEYS) {
            const int c1 = std::min(Lmax, c0 + ATT_KEYS);
            for (int i0 = 0; i0 < QN; i0 += 8) {
                const int qn = std::min(8, QN - i0);
                const int qlen = pos0 + t0 + (i0 + qn - 1) / group + 1;   // the longest causal length in this group of queries
                for (int j0 = c0; j0 < std::min(c1, qlen); j0 += 16) {
                    __m512 acc[8];
                    for (int i = 0; i < qn; ++i) acc[i] = _mm512_setzero_ps();
                    for (int d = 0; d < DH; ++d) {
                        const __m512 kv = _mm512_loadu_ps(kt + (size_t)d * Lp + j0);
                        for (int i = 0; i < qn; ++i) acc[i] = _mm512_fmadd_ps(_mm512_set1_ps(qp[i0 + i][d]), kv, acc[i]);
                    }
                    for (int i = 0; i < qn; ++i) _mm512_storeu_ps(&S[(size_t)(i0 + i) * Lmax_p + j0], _mm512_mul_ps(acc[i], _mm512_set1_ps(scale)));
                }
            }
        }
        // softmax per query over its own causal length; probabilities = e / sum (division, as before)
        for (int i = 0; i < QN; ++i) {
            const int len = pos0 + t0 + i / group + 1;
            float* s = &S[(size_t)i * Lmax_p];
            __m512 mx = _mm512_set1_ps(-INFINITY);
            for (int j = 0; j < len; j += 16) mx = _mm512_max_ps(mx, _mm512_mask_loadu_ps(_mm512_set1_ps(-INFINITY), (__mmask16)(len - j >= 16 ? 0xFFFF : (1u << (len - j)) - 1), s + j));
            const __m512 m = _mm512_set1_ps(_mm512_reduce_max_ps(mx));
            __m512 sum = _mm512_setzero_ps();
            for (int j = 0; j < len; j += 16) {
                const __mmask16 k = (__mmask16)(len - j >= 16 ? 0xFFFF : (1u << (len - j)) - 1);
                const __m512 e = _mm512_maskz_mov_ps(k, exp_ps(_mm512_sub_ps(_mm512_loadu_ps(s + j), m)));
                sum = _mm512_add_ps(sum, e);
                _mm512_storeu_ps(s + j, e);
            }
            const __m512 den = _mm512_set1_ps(_mm512_reduce_add_ps(sum));
            for (int j = 0; j < len; j += 16) _mm512_storeu_ps(s + j, _mm512_div_ps(_mm512_loadu_ps(s + j), den));
        }
        // outputs, value chunk by value chunk; per token its heads share the value loads, partial sums kept in O between chunks
        static thread_local std::vector<float> O;
        O.assign((size_t)ATT_TOKENS * 2 * DH, 0.f);
        for (int c0 = 0; c0 < Lmax; c0 += ATT_KEYS) {
            const int c1 = std::min(Lmax, c0 + ATT_KEYS);
            for (int tt = 0; tt < tn; ++tt) {
                const int len = std::min(c1, pos0 + t0 + tt + 1);
                if (len <= c0) continue;
                float* o0 = &O[(size_t)(tt * 2) * DH];
                float* o1 = o0 + DH;
                __m512 ov[2][8];
                for (int c = 0; c < 8; ++c) {
                    ov[0][c] = _mm512_loadu_ps(o0 + 16 * c);
                    ov[1][c] = _mm512_loadu_ps(o1 + 16 * c);
                }
                const float* s0 = &S[(size_t)(tt * group) * Lmax_p];
                const float* s1 = group > 1 ? &S[(size_t)(tt * group + 1) * Lmax_p] : s0;
                for (int j = c0; j < len; ++j) {
                    const uint16_t* vj = &Vc[(size_t)j * DK + (size_t)kvh * DH];
                    const __m512 p0 = _mm512_set1_ps(s0[j]), p1 = _mm512_set1_ps(s1[j]);
                    for (int c = 0; c < 8; ++c) {
                        const __m512 v = load_bf16x16(vj + 16 * c);
                        ov[0][c] = _mm512_fmadd_ps(p0, v, ov[0][c]);
                        ov[1][c] = _mm512_fmadd_ps(p1, v, ov[1][c]);
                    }
                }
                for (int c = 0; c < 8; ++c) {
                    _mm512_storeu_ps(o0 + 16 * c, ov[0][c]);
                    _mm512_storeu_ps(o1 + 16 * c, ov[1][c]);
                }
            }
        }
        for (int tt = 0; tt < tn; ++tt)
            for (int g = 0; g < group; ++g) {
                float* oh = att + (size_t)(t0 + tt) * DQ + (size_t)(kvh * group + g) * DH;
                const float* src = &O[(size_t)(tt * 2 + g) * DH];
                for (int c = 0; c < 8; ++c) _mm512_storeu_ps(oh + 16 * c, bf16_round_ps(_mm512_loadu_ps(src + 16 * c)));
            }
    }
}

struct layer_weights {
    q8_matrix q, k, v, o, gate, up, down;
    std::vector<float> input_norm, post_norm, q_norm, k_norm;
};

}  // namespace

struct decoder::impl {
    decoder_config cfg;
    mapped_safetensors file;
    std::vector<layer_weights> layers;
    q8_matrix lm_head;
    const uint16_t* embed_table = nullptr;     // bf16 [vocab][hidden]
    std::vector<float> final_norm;
    std::vector<float> inv_freq;               // [head_dim/2]
    // KV cache per layer: [context][kv_heads * head_dim], bf16 (the values are bf16-rounded, so this is exact), K after norm + rope
    std::vector<std::vector<uint16_t>> k_cache, v_cache;
    std::vector<float> kt_scratch;                 // attention_blocked's transposed keys
    int ctx = 0;
    int peak_ctx = 0;                              // the longest context since release()
    projections* proj = nullptr;                   // the prompt's matrix products elsewhere (set_projections)

    impl(const std::string& path, decoder_config c) : cfg(c), file(path) {
        const int D = cfg.hidden, DH = cfg.head_dim;
        const int DQ = cfg.heads * DH, DK = cfg.kv_heads * DH;
        layers.resize(cfg.layers);
        for (int i = 0; i < cfg.layers; ++i) {
            const std::string p = "model.layers." + std::to_string(i) + ".";
            layer_weights& L = layers[i];
            L.q = load_q8(file, p + "self_attn.q_proj.weight", DQ, D);
            L.k = load_q8(file, p + "self_attn.k_proj.weight", DK, D);
            L.v = load_q8(file, p + "self_attn.v_proj.weight", DK, D);
            L.o = load_q8(file, p + "self_attn.o_proj.weight", D, DQ);
            L.gate = load_q8(file, p + "mlp.gate_proj.weight", cfg.inter, D);
            L.up = load_q8(file, p + "mlp.up_proj.weight", cfg.inter, D);
            L.down = load_q8(file, p + "mlp.down_proj.weight", D, cfg.inter);
            L.input_norm = load_bf16_vector(file.get(p + "input_layernorm.weight"));
            L.post_norm = load_bf16_vector(file.get(p + "post_attention_layernorm.weight"));
            L.q_norm = load_bf16_vector(file.get(p + "self_attn.q_norm.weight"));
            L.k_norm = load_bf16_vector(file.get(p + "self_attn.k_norm.weight"));
        }
        lm_head = load_q8(file, "lm_head.weight", cfg.vocab, D);
        embed_table = (const uint16_t*)file.get("model.embed_tokens.weight").data;
        final_norm = load_bf16_vector(file.get("model.norm.weight"));
        // inv_freq = 1 / theta^(2i/head_dim), in fp32 like the reference
        inv_freq.resize(DH / 2);
        for (int i = 0; i < DH / 2; ++i) inv_freq[i] = 1.0f / std::pow(cfg.rope_theta, (float)(2 * i) / (float)DH);
        k_cache.resize(cfg.layers);
        v_cache.resize(cfg.layers);
    }

    /// bf16 rope tables for one absolute position: cos/sin of pos*inv_freq in fp32, then rounded to bf16
    void rope_tables(int pos, float* cos_t, float* sin_t) const {
        const int half = cfg.head_dim / 2;
        for (int i = 0; i < half; ++i) {
            const float a = (float)pos * inv_freq[i];
            cos_t[i] = cos_t[i + half] = bf16r(std::cos(a));
            sin_t[i] = sin_t[i + half] = bf16r(std::sin(a));
        }
    }

    /// q_embed = q*cos + rotate_half(q)*sin, with bf16 rounding after each of the three ops (as in the reference)
    void apply_rope(float* v, const float* cos_t, const float* sin_t) const {
        const int DH = cfg.head_dim, half = DH / 2;
        float out[256];
        for (int i = 0; i < half; ++i) {
            out[i] = bf16r(bf16r(v[i] * cos_t[i]) + bf16r(-v[i + half] * sin_t[i]));
            out[i + half] = bf16r(bf16r(v[i + half] * cos_t[i + half]) + bf16r(v[i] * sin_t[i + half]));
        }
        std::memcpy(v, out, sizeof(float) * DH);
    }

    void forward(const float* embeds, int n, std::vector<float>& last_logits) {
        const double fwd_start = now_ms();
        struct report { double s; int n; ~report() { if (std::getenv("QWEN3_ASR_PROF") && n > 8) std::fprintf(stderr, "[prof] n=%d total %.0f ms: matmul %.0f, attention %.0f, rest %.0f\n", n, now_ms() - s, g_prof.mm, g_prof.att, now_ms() - s - g_prof.mm - g_prof.att); } } rep{fwd_start, n};
        g_prof = prof_t();
        const int D = cfg.hidden, DH = cfg.head_dim, H = cfg.heads, KV = cfg.kv_heads;
        const int DQ = H * DH, DK = KV * DH, I = cfg.inter;
        const int pos0 = ctx;
        projections* P = (proj && n >= proj->min_rows()) ? proj : nullptr;
        auto timed = [](auto&& f) { const double s = now_ms(); f(); g_prof.mm += now_ms() - s; };

        std::vector<float> h(embeds, embeds + (size_t)n * D);          // residual stream, bf16-valued
        std::vector<float> x((size_t)n * D), q((size_t)n * DQ), k((size_t)n * DK), v((size_t)n * DK);
        std::vector<float> att((size_t)n * DQ), o((size_t)n * D);
        std::vector<float> g(P ? 0 : (size_t)n * I), u(P ? 0 : (size_t)n * I), dn((size_t)n * D);
        std::vector<float> cos_t((size_t)n * DH), sin_t((size_t)n * DH);
        for (int t = 0; t < n; ++t) rope_tables(pos0 + t, &cos_t[(size_t)t * DH], &sin_t[(size_t)t * DH]);

        for (int li = 0; li < cfg.layers; ++li) {
            layer_weights& L = layers[li];
            // ---- attention block ----
#pragma omp parallel for num_threads(qwen3_asr_cpu::asr_threads())
            for (int t = 0; t < n; ++t) rmsnorm_row(&h[(size_t)t * D], L.input_norm.data(), &x[(size_t)t * D], D, cfg.rms_eps);
            if (P) {
                timed([&] { P->qkv(li, x.data(), n, q.data(), k.data(), v.data()); });
            } else {
                matmul_q8(L.q, x.data(), n, q.data());
                matmul_q8(L.k, x.data(), n, k.data());
                matmul_q8(L.v, x.data(), n, v.data());
            }

            std::vector<uint16_t>& Kc = k_cache[li];
            std::vector<uint16_t>& Vc = v_cache[li];
            Kc.resize((size_t)(pos0 + n) * DK);
            Vc.resize((size_t)(pos0 + n) * DK);
#pragma omp parallel for num_threads(qwen3_asr_cpu::asr_threads())
            for (int t = 0; t < n; ++t) {
                float* qt = &q[(size_t)t * DQ];
                float* kt = &k[(size_t)t * DK];
                for (int hd = 0; hd < H; ++hd) {          // per-head q norm (over head_dim), then rope
                    rmsnorm_row(qt + hd * DH, L.q_norm.data(), qt + hd * DH, DH, cfg.rms_eps);
                    apply_rope(qt + hd * DH, &cos_t[(size_t)t * DH], &sin_t[(size_t)t * DH]);
                }
                for (int hd = 0; hd < KV; ++hd) {
                    rmsnorm_row(kt + hd * DH, L.k_norm.data(), kt + hd * DH, DH, cfg.rms_eps);
                    apply_rope(kt + hd * DH, &cos_t[(size_t)t * DH], &sin_t[(size_t)t * DH]);
                }
                const float* vt = &v[(size_t)t * DK];
                for (int i = 0; i < DK; ++i) {
                    Kc[(size_t)(pos0 + t) * DK + i] = f2bf(kt[i]);
                    Vc[(size_t)(pos0 + t) * DK + i] = f2bf(vt[i]);
                }
            }

            const double att_start = now_ms();
            static const bool att_scalar = std::getenv("QWEN3_ASR_ATT_SCALAR") != nullptr;   // the per-(token, head) loop below, for A/B
            if (DH == 128 && H / KV <= 2 && n >= 4 && !att_scalar) {   // one-token decode: the transpose would cost more than it saves
                attention_blocked(q.data(), Kc, Vc, pos0, n, H, KV, DH, att.data(), kt_scratch);
            } else {
                // causal attention; 16 query heads share 8 kv heads (2 per group); fp32 inside, bf16 out
                const float scale = 1.0f / std::sqrt((float)DH);
                const int group = H / KV;
    #pragma omp parallel for num_threads(qwen3_asr_cpu::asr_threads())
                for (int th = 0; th < n * H; ++th) {      // one (token, head) pair per iteration (OpenMP 2.0: no collapse)
                    {
                        const int t = th / H, hd = th % H;
                        const float* qh = &q[(size_t)t * DQ + (size_t)hd * DH];
                        const int kvh = hd / group;
                        const int len = pos0 + t + 1;
                        std::vector<float> sc(len);
                        float mx = -INFINITY;
                        // DH == 128: q (8 zmm) stays in registers for the whole key loop
                        __m512 qv[8];
                        for (int c = 0; c < 8; ++c) qv[c] = _mm512_loadu_ps(qh + 16 * c);
                        for (int j = 0; j < len; ++j) {
                            const uint16_t* kj = &Kc[(size_t)j * DK + (size_t)kvh * DH];
                            __m512 a0 = _mm512_mul_ps(qv[0], load_bf16x16(kj)), a1 = _mm512_mul_ps(qv[1], load_bf16x16(kj + 16));
                            for (int c = 2; c < 8; c += 2) {
                                a0 = _mm512_fmadd_ps(qv[c], load_bf16x16(kj + 16 * c), a0);
                                a1 = _mm512_fmadd_ps(qv[c + 1], load_bf16x16(kj + 16 * (c + 1)), a1);
                            }
                            sc[j] = _mm512_reduce_add_ps(_mm512_add_ps(a0, a1)) * scale;
                            mx = std::max(mx, sc[j]);
                        }
                        float den = 0.f;
                        for (int j = 0; j < len; ++j) { sc[j] = std::exp(sc[j] - mx); den += sc[j]; }
                        float* oh = &att[(size_t)t * DQ + (size_t)hd * DH];
                        __m512 ov[8];
                        for (int c = 0; c < 8; ++c) ov[c] = _mm512_setzero_ps();
                        for (int j = 0; j < len; ++j) {
                            const uint16_t* vj = &Vc[(size_t)j * DK + (size_t)kvh * DH];
                            const __m512 pj = _mm512_set1_ps(sc[j] / den);
                            for (int c = 0; c < 8; ++c) ov[c] = _mm512_fmadd_ps(pj, load_bf16x16(vj + 16 * c), ov[c]);
                        }
                        for (int c = 0; c < 8; ++c) _mm512_storeu_ps(oh + 16 * c, ov[c]);
                        for (int d = 0; d < DH; ++d) oh[d] = bf16r(oh[d]);
                    }
                }
            }
            g_prof.att += now_ms() - att_start;
            if (P) timed([&] { P->o(li, att.data(), n, o.data()); });
            else matmul_q8(L.o, att.data(), n, o.data());
#pragma omp parallel for num_threads(qwen3_asr_cpu::asr_threads())
            for (long long i = 0; i < (long long)n * D; ++i) h[i] = bf16r(h[i] + o[i]);   // residual

            // ---- MLP block (SwiGLU) ----
#pragma omp parallel for num_threads(qwen3_asr_cpu::asr_threads())
            for (int t = 0; t < n; ++t) rmsnorm_row(&h[(size_t)t * D], L.post_norm.data(), &x[(size_t)t * D], D, cfg.rms_eps);
            if (P) {
                timed([&] { P->mlp(li, x.data(), n, dn.data()); });
#pragma omp parallel for num_threads(qwen3_asr_cpu::asr_threads())
                for (long long i = 0; i < (long long)n * D; ++i) h[i] = bf16r(h[i] + dn[i]);   // residual
                continue;
            }
            matmul_q8(L.gate, x.data(), n, g.data());
            matmul_q8(L.up, x.data(), n, u.data());
#pragma omp parallel for num_threads(qwen3_asr_cpu::asr_threads())
            for (long long i = 0; i < (long long)n * I; i += 16) {          // n * I is a multiple of 16 (I = 6144)
                const __m512 gv = _mm512_loadu_ps(&g[i]);
                const __m512 s = bf16_round_ps(_mm512_div_ps(gv, _mm512_add_ps(_mm512_set1_ps(1.0f), exp_ps(_mm512_sub_ps(_mm512_setzero_ps(), gv)))));   // silu in bf16
                _mm512_storeu_ps(&g[i], bf16_round_ps(_mm512_mul_ps(s, _mm512_loadu_ps(&u[i]))));
            }
            matmul_q8(L.down, g.data(), n, dn.data());
#pragma omp parallel for num_threads(qwen3_asr_cpu::asr_threads())
            for (long long i = 0; i < (long long)n * D; ++i) h[i] = bf16r(h[i] + dn[i]);   // residual
        }
        ctx = pos0 + n;
        peak_ctx = std::max(peak_ctx, ctx);

        // final norm + tied lm_head on the last position only
        std::vector<float> last(D), logits(cfg.vocab);
        rmsnorm_row(&h[(size_t)(n - 1) * D], final_norm.data(), last.data(), D, cfg.rms_eps);
        matmul_q8(lm_head, last.data(), 1, logits.data());
        last_logits = std::move(logits);
    }
};

decoder::decoder(const std::string& q4nx_path, decoder_config cfg) : p_(new impl(q4nx_path, cfg)) {}
decoder::~decoder() = default;
const decoder_config& decoder::cfg() const { return p_->cfg; }
void decoder::set_projections(projections* p) { p_->proj = p; }
void decoder::embed(int token, float* out) const {
    const uint16_t* row = p_->embed_table + (size_t)token * p_->cfg.hidden;
    for (int i = 0; i < p_->cfg.hidden; ++i) out[i] = bf2f(row[i]);
}
const uint16_t* decoder::embed_row(int token) const { return p_->embed_table + (size_t)token * p_->cfg.hidden; }
void decoder::reset() {
    p_->ctx = 0;
    for (auto& k : p_->k_cache) k.clear();
    for (auto& v : p_->v_cache) v.clear();
}
void decoder::release() {
    reset();
    for (auto& k : p_->k_cache) k.shrink_to_fit();
    for (auto& v : p_->v_cache) v.shrink_to_fit();
    std::vector<float>().swap(p_->kt_scratch);
    p_->peak_ctx = 0;
}
int decoder::context_length() const { return p_->ctx; }
void decoder::set_context_length(int rows) {
    p_->ctx = rows;
    p_->peak_ctx = std::max(p_->peak_ctx, rows);
}
int decoder::peak_context_length() const { return p_->peak_ctx; }
void decoder::export_kv(int layer, int from, int rows, uint16_t* k, uint16_t* v) const {
    const size_t DK = (size_t)p_->cfg.kv_heads * p_->cfg.head_dim;
    std::memcpy(k, &p_->k_cache[layer][(size_t)from * DK], sizeof(uint16_t) * rows * DK);
    std::memcpy(v, &p_->v_cache[layer][(size_t)from * DK], sizeof(uint16_t) * rows * DK);
}
void decoder::import_kv(int layer, int rows, const uint16_t* k, const uint16_t* v) {
    const size_t DK = (size_t)p_->cfg.kv_heads * p_->cfg.head_dim;
    p_->k_cache[layer].assign(k, k + (size_t)rows * DK);
    p_->v_cache[layer].assign(v, v + (size_t)rows * DK);
}
void decoder::forward(const float* embeds, int n, std::vector<float>& last_logits) { p_->forward(embeds, n, last_logits); }

}  // namespace qwen3_asr_cpu
