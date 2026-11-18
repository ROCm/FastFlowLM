// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#ifndef __LFM3_CPU_HPP__
#define __LFM3_CPU_HPP__
/// \file lfm3_cpu.hpp
/// \brief Host (CPU) implementation of every stage of the LFM3 model.
///
/// This is the "complete host model": it reads model.q4nx blocks as the converter wrote them and computes the same
/// thing as models_simple/simple/lfm3_simple.py, with the same rounding points (activations are bf16 between ops,
/// every matmul accumulates in fp32 and rounds to bf16 once). It never touches the NPU, so it is safe to run anywhere
/// and is the reference every NPU stage is checked against.
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <immintrin.h>
#include "typedef.hpp"
#include "buffer.hpp"

namespace lfm3_cpu {

// ---------------------------------------------------------------------------------------------------------------
// bf16 helpers. Activations live in u16 storage (same bits as bf16); conversions are explicit so the rounding points
// are visible in the code.
// ---------------------------------------------------------------------------------------------------------------
inline float bf2f(uint16_t b) {
    uint32_t u = (uint32_t)b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

/// round to nearest even, like torch's float -> bfloat16
inline uint16_t f2bf(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    if ((u & 0x7FFFFFFFu) > 0x7F800000u) return (uint16_t)((u >> 16) | 0x40);   // NaN stays NaN
    u += 0x7FFFu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

inline uint16_t* u16p(bf16* p) { return reinterpret_cast<uint16_t*>(p); }
inline const uint16_t* u16p(const bf16* p) { return reinterpret_cast<const uint16_t*>(p); }

/// Threads for the host kernels: one per physical core (SMT siblings add no FMA throughput). FLM_PREFILL_THREADS overrides.
inline int host_threads() {
    static const int nt = [] {
#ifdef _OPENMP
        int n = omp_get_max_threads();
#else
        int n = 1;
#endif
        const int cores = (int)std::thread::hardware_concurrency() / 2;
        if (cores > 0 && n > cores) n = cores;
        if (const char* env = std::getenv("FLM_PREFILL_THREADS")) {
            int v = std::atoi(env);
            if (v > 0) n = v;
        }
        return n < 1 ? 1 : n;
    }();
    return nt;
}

// ---------------------------------------------------------------------------------------------------------------
// The q4nx block, as FLM_Q4NX_Converter writes it for the LFM configs (row_block 32, col_block 256, parallel_size 16).
// A weight of `rows` x `cols` is a grid of 32 x 256 blocks, stored block row by block row (block (rb, cb) at index
// rb * (cols / 256) + cb), each block 5120 bytes:
//
//   bf16 scale[8][32]      scale of column group g (32 columns), row r
//   bf16 min[8][32]        zero for the legacy Q4_0 form; real min for Q4_1 (the lm_head)
//   u8   q[2][256][8]      plane p (rows 16p..16p+15), column c, byte j: rows 16p+2j (low nibble), 16p+2j+1 (high)
//
//   w[r][c] = scale[c / 32][r] * q + min[c / 32][r]     q signed (two's complement nibble) for Q4_0, unsigned for Q4_1
// ---------------------------------------------------------------------------------------------------------------
constexpr int QB_ROWS = 32;
constexpr int QB_COLS = 256;
constexpr int QB_GROUP = 32;
constexpr int QB_BYTES = 5120;

/// \brief A quantized weight matrix as a read-only view of blocks in the device layout.
///
/// The NPU dequant kernels read the blocks of consecutive row blocks interleaved (`vblocks` row blocks per column block),
/// so the host reads the same bytes the same way and one copy of the weights serves both. Block (rb, cb) of an `rows` x
/// `cols` weight sits at
///     slice(rb / slice_rb) * slice_stride + ((rb % slice_rb) / vblocks * vblocks * cbs + cb * vblocks + rb % vblocks) * 5120
/// with cbs = cols / 256. A plain weight is one slice; the up/gate band is cut in slices of 512 rows (16 row blocks)
/// laid out up, gate, up, gate, ... so up and gate have slice_stride = 2 * slice bytes and gate starts one slice later.
struct qmat {
    const uint8_t* base = nullptr;
    int rows = 0;         ///< output features N
    int cols = 0;         ///< input features K
    bool is_signed = true;///< nibbles are signed (legacy Q4_0) or unsigned (Q4_1)
    int vblocks = 2;      ///< row blocks interleaved per column block (1 = plain q4nx order)
    int slice_rb = 0;     ///< row blocks per slice (0 = the whole matrix is one slice)
    size_t slice_stride = 0;

    int row_blocks() const { return rows / QB_ROWS; }
    int col_blocks() const { return cols / QB_COLS; }
    const uint8_t* block(int rb, int cb) const {
        const int per_slice = slice_rb ? slice_rb : row_blocks();
        const int s = rb / per_slice, r = rb % per_slice;
        const size_t idx = (size_t)(r / vblocks) * vblocks * col_blocks() + (size_t)cb * vblocks + (size_t)(r % vblocks);
        return base + (size_t)s * slice_stride + idx * QB_BYTES;
    }
    /// bytes of one whole matrix (slices of an interleaved band not included)
    size_t bytes() const { return (size_t)row_blocks() * col_blocks() * QB_BYTES; }
};

/// \brief Dequantizes block row `rb` of `m` to fp32 into dst[cols][32] (the 32 row values of one input column are
///        contiguous, which is the order the matmul wants).
inline void dequant_block_row(float* dst, const qmat& m, int rb) {
    const int cbs = m.cols / QB_COLS;
    const __m128i nib = _mm_set1_epi8(0x0F);
    for (int cb = 0; cb < cbs; cb++) {
        const uint8_t* blk = m.block(rb, cb);
        const uint16_t* scale = reinterpret_cast<const uint16_t*>(blk);          // [8][32]
        const uint16_t* mins = scale + 8 * 32;                                    // [8][32]
        const uint8_t* qs = blk + 1024;                                           // [2][256][8]
        for (int g = 0; g < 8; g++) {
            __m512 s[2], z[2];
            for (int p = 0; p < 2; p++) {
                __m256i sb = _mm256_loadu_si256((const __m256i*)(scale + g * 32 + p * 16));
                __m256i zb = _mm256_loadu_si256((const __m256i*)(mins + g * 32 + p * 16));
                s[p] = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(sb), 16));
                z[p] = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(zb), 16));
            }
            for (int c = g * QB_GROUP; c < (g + 1) * QB_GROUP; c++) {
                float* out = dst + (size_t)(cb * QB_COLS + c) * QB_ROWS;
                for (int p = 0; p < 2; p++) {
                    // 8 bytes -> 16 nibbles in row order: interleave low and high nibbles
                    __m128i b8 = _mm_loadl_epi64((const __m128i*)(qs + ((size_t)p * QB_COLS + c) * 8));
                    __m128i lo = _mm_and_si128(b8, nib);
                    __m128i hi = _mm_and_si128(_mm_srli_epi16(b8, 4), nib);
                    __m128i rows16 = _mm_unpacklo_epi8(lo, hi);
                    __m512i q = _mm512_cvtepu8_epi32(rows16);
                    if (m.is_signed) {
                        // two's complement 4-bit: (v ^ 8) - 8
                        q = _mm512_sub_epi32(_mm512_xor_si512(q, _mm512_set1_epi32(8)), _mm512_set1_epi32(8));
                    }
                    __m512 w = _mm512_fmadd_ps(_mm512_cvtepi32_ps(q), s[p], z[p]);
                    _mm512_storeu_ps(out + p * 16, w);
                }
            }
        }
    }
}

/// \brief y[l][n] = bf16( sum_k x[l][k] * W[n][k] ) for l < L, n < W.rows. x and y are bf16 in u16 storage.
/// \note Each 32-row band is dequantized once and reused for every one of the L rows, so the weight traffic is one pass
///       over the weights however large L is.
inline void matmul(uint16_t* y, int ldy, const qmat& W, const uint16_t* x, int ldx, int L) {
    const int K = W.cols;
    const int row_blocks = W.rows / QB_ROWS;
    std::vector<float> xf((size_t)L * K);
    for (int l = 0; l < L; l++)
        for (int k = 0; k < K; k++) xf[(size_t)l * K + k] = bf2f(x[(size_t)l * ldx + k]);

    #pragma omp parallel num_threads(host_threads())
    {
        std::vector<float> dst((size_t)K * QB_ROWS + 16);
        #pragma omp for schedule(static)
        for (int rb = 0; rb < row_blocks; rb++) {
            dequant_block_row(dst.data(), W, rb);
            int l = 0;
            for (; l + 4 <= L; l += 4) {
                __m512 a0[4], a1[4];
                for (int j = 0; j < 4; j++) { a0[j] = _mm512_setzero_ps(); a1[j] = _mm512_setzero_ps(); }
                const float* xr[4];
                for (int j = 0; j < 4; j++) xr[j] = &xf[(size_t)(l + j) * K];
                for (int k = 0; k < K; k++) {
                    const __m512 w0 = _mm512_loadu_ps(&dst[(size_t)k * QB_ROWS]);
                    const __m512 w1 = _mm512_loadu_ps(&dst[(size_t)k * QB_ROWS + 16]);
                    for (int j = 0; j < 4; j++) {
                        const __m512 xv = _mm512_set1_ps(xr[j][k]);
                        a0[j] = _mm512_fmadd_ps(xv, w0, a0[j]);
                        a1[j] = _mm512_fmadd_ps(xv, w1, a1[j]);
                    }
                }
                for (int j = 0; j < 4; j++) {
                    alignas(64) float o[32];
                    _mm512_store_ps(o, a0[j]);
                    _mm512_store_ps(o + 16, a1[j]);
                    for (int r = 0; r < 32; r++) y[(size_t)(l + j) * ldy + rb * 32 + r] = f2bf(o[r]);
                }
            }
            for (; l < L; l++) {
                __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
                const float* xr = &xf[(size_t)l * K];
                for (int k = 0; k < K; k++) {
                    const __m512 xv = _mm512_set1_ps(xr[k]);
                    a0 = _mm512_fmadd_ps(xv, _mm512_loadu_ps(&dst[(size_t)k * QB_ROWS]), a0);
                    a1 = _mm512_fmadd_ps(xv, _mm512_loadu_ps(&dst[(size_t)k * QB_ROWS + 16]), a1);
                }
                alignas(64) float o[32];
                _mm512_store_ps(o, a0);
                _mm512_store_ps(o + 16, a1);
                for (int r = 0; r < 32; r++) y[(size_t)l * ldy + rb * 32 + r] = f2bf(o[r]);
            }
        }
    }
}

/// y[l][n] = bf16(sum_k x[l][k] * W[n][k]) for a small dense bf16 matrix W[N][K] (the 16-row attention gate)
inline void matmul_dense(uint16_t* y, int ldy, const uint16_t* W, int N, int K, const uint16_t* x, int ldx, int L) {
    #pragma omp parallel for num_threads(host_threads()) schedule(static) collapse(2)
    for (int l = 0; l < L; l++) {
        for (int n = 0; n < N; n++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++) acc += bf2f(x[(size_t)l * ldx + k]) * bf2f(W[(size_t)n * K + k]);
            y[(size_t)l * ldy + n] = f2bf(acc);
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Element-wise stages. Rounding follows the upstream bf16 model: RMSNorm normalizes in fp32, rounds to bf16, then
// multiplies by the bf16 weight (another rounding).
// ---------------------------------------------------------------------------------------------------------------

/// y = weight * rmsnorm(x) over one vector of n elements (y may alias x)
inline void rms_norm_vec(uint16_t* y, const uint16_t* x, const uint16_t* w, int n, float eps) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++) { const float v = bf2f(x[i]); ss += v * v; }
    const float r = 1.0f / std::sqrt(ss / (float)n + eps);
    for (int i = 0; i < n; i++) {
        const uint16_t normed = f2bf(bf2f(x[i]) * r);
        y[i] = f2bf(bf2f(w[i]) * bf2f(normed));
    }
}

/// x[l] = bf16(x[l] + y[l]) over L rows of n
inline void residual_add(uint16_t* x, const uint16_t* y, size_t count) {
    for (size_t i = 0; i < count; i++) x[i] = f2bf(bf2f(x[i]) + bf2f(y[i]));
}

inline float sigmoidf(float v) { return 1.0f / (1.0f + std::exp(-v)); }
inline float siluf(float v) { return v * sigmoidf(v); }

/// RoPE on one head vector of DH (already q/k-normed): cos and sin hold DH/2 bf16-rounded values (the angle of
/// element i and of element i + DH/2 is the same; a zero angle leaves the pair untouched).
inline void rope_head(uint16_t* v, const float* cosv, const float* sinv, int DH) {
    const int H = DH / 2;
    for (int i = 0; i < H; i++) {
        const float a = bf2f(v[i]), b = bf2f(v[i + H]);
        v[i] = f2bf(a * cosv[i] - b * sinv[i]);
        v[i + H] = f2bf(b * cosv[i] + a * sinv[i]);
    }
}

} // namespace lfm3_cpu

#endif
