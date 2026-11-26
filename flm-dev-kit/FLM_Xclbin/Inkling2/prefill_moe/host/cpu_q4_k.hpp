// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __CPU_Q4_K_HPP__
#define __CPU_Q4_K_HPP__
#pragma once

// CPU model of the FLM q4_k block, used to build a golden reference for this
// xclbin without a converted model on disk.
//
// This mirrors FLM_Q4NX_Converter/q4nx/model_converter.py::_pack_q4k (and the
// (BF16 P', uint8 p'_j) re-fit in q4nx/gguf_tensor.py::_refit_one_side), i.e.
// exactly the bytes kernel/q4_core.h decodes.  Per 32x256 chunk, everything
// column major over the chunk -- q4_k is never parallel-strided, PARALLEL_ROWS
// is pinned to QXNX_ROW_BLOCK_SIZE:
//
//     uint8 scales[COLS / GROUP][ROWS]                 256 B
//     uint8 mins  [COLS / GROUP][ROWS]                 256 B
//     uint4 qs    [COLS][ROWS / 2]                    4096 B
//     bf16  S     [COLS / SUPER][ROWS]                  64 B
//     bf16  M     [COLS / SUPER][ROWS]                  64 B
//
// 4736 B total, 4.625 bits per weight.  GGUF subtracts the min while the kernel
// adds both accumulators, so M is stored negated and w = S' s'_j q + M' m'_j.
//
// Chunks are emitted row-block major, matching what Q4NX::load_weights hands
// back; the vertical interleave the cores expect is applied by reorder_cpy.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

#include "typedef.hpp"
#include "weights_desc.hpp"

struct q4_k_block {
    static constexpr int ROWS        = QXNX_ROW_BLOCK_SIZE;              // 32
    static constexpr int COLS        = QXNX_COL_BLOCK_SIZE;              // 256
    static constexpr int GROUP       = GGML_GROUP_SIZE;                  // 32
    static constexpr int SUPER       = Q4K_SUPER_BLOCK_SIZE;             // 256
    static constexpr int GROUPS      = COLS / GROUP;                     // 8
    static constexpr int SUPERS      = COLS / SUPER;                     // 1
    static constexpr int SCALE_BYTES = GROUPS * ROWS;                    // 256
    static constexpr int MIN_BYTES   = SCALE_BYTES;                      // 256
    static constexpr int QS_BYTES    = COLS * ROWS / 2;                  // 4096
    static constexpr int S_BYTES     = SUPERS * ROWS * (int)sizeof(bf16);// 64
    static constexpr int M_BYTES     = S_BYTES;                          // 64
    static constexpr int BYTES       = SCALE_BYTES + MIN_BYTES + QS_BYTES + S_BYTES + M_BYTES;

    static_assert(BYTES == 4736, "q4_k is 4.625 bits per weight");
};

/// \brief Round the magnitude of a non-negative float UP to a bf16 value.
inline float q4k_to_bf16_up(float x) {
    uint32_t u;
    std::memcpy(&u, &x, 4);
    if (u & 0xFFFFu) u += 0x10000u;
    u &= 0xFFFF0000u;
    float r;
    std::memcpy(&r, &u, 4);
    return r;
}

/// \brief The n-th bf16 value above a bf16-representable, non-negative x.
inline float q4k_bf16_next_up(float x, int n) {
    if (n == 0) return x;
    uint32_t u;
    std::memcpy(&u, &x, 4);
    u += (uint32_t)n * 0x10000u;
    float r;
    std::memcpy(&r, &u, 4);
    return r;
}

/// \brief Fit one metadata side of a super-block: exact t_j -> (bf16 P', uint8 p'_j).
///
/// Port of _refit_one_side. P' is fixed first, rounded away from zero so p'_j can
/// never overflow the cap, then p'_j is re-derived against it; `search` extra bf16
/// candidates above the base are scored and the best sum of squares wins.
///
/// \param t      exact per-group values, \p n of them (one super-block, one row)
/// \param n      groups per super-block
/// \param P_out  the bf16-representable super-block value (carries the sign)
/// \param p_out  the per-group uint8 values, in [0, cap]
inline void q4k_refit_one_side(const float* t, int n, float& P_out, float* p_out,
                               int search = 3, float cap = 255.0f) {
    float tmax = 0.0f;
    int lead = 0;
    for (int j = 0; j < n; j++) {
        float a = std::fabs(t[j]);
        if (a > tmax) { tmax = a; lead = j; }   // np.argmax: first occurrence wins
    }
    if (!(tmax > 0.0f)) {                        // dead super-block -> all zero
        P_out = 0.0f;
        for (int j = 0; j < n; j++) p_out[j] = 0.0f;
        return;
    }
    // p'_j is unsigned, so the sign rides on P'
    const float sigma = t[lead] < 0.0f ? -1.0f : 1.0f;
    float ts[q4_k_block::GROUPS];
    for (int j = 0; j < n; j++) ts[j] = sigma * t[j];

    const float base = q4k_to_bf16_up(tmax / cap);
    double best_e = std::numeric_limits<double>::infinity();
    float best_P = 0.0f;
    float best_p[q4_k_block::GROUPS] = {};
    for (int c = 0; c <= search; c++) {
        const float Pc = q4k_bf16_next_up(base, c);
        const float inv = 1.0f / Pc;
        float pc[q4_k_block::GROUPS];
        double err = 0.0;
        for (int j = 0; j < n; j++) {
            pc[j] = std::min(std::max(std::rint(ts[j] * inv), 0.0f), cap);
            const double d = (double)(Pc * pc[j] - ts[j]);
            err += d * d;
        }
        if (err < best_e) {
            best_e = err;
            best_P = Pc;
            for (int j = 0; j < n; j++) best_p[j] = pc[j];
        }
    }
    P_out = sigma * best_P;
    for (int j = 0; j < n; j++) p_out[j] = best_p[j];
}

/// \brief Quantize a bf16 weight matrix to the packed FLM q4_k layout.
///
/// The values the kernel reconstructs are written back to \p w_deq -- including
/// the kernel's own bf16 rounding of the S' s'_j and M' m'_j products -- so the
/// caller can build a reference that carries the same quantization error.
///
/// \param dst   packed output, get_quantization_byte_size(rows * cols, flm_q4k) bytes
/// \param w_deq rows x cols, the values the kernel reconstructs (may be null)
/// \param w     rows x cols row major source weights (out_features x in_features)
/// \param rows  out features, must be a multiple of 32
/// \param cols  in features, must be a multiple of 256
/// \brief fp32 -> bf16 truncating toward -infinity.
///
/// This is what the AIE srs does on `aie::accum::to_vector<bf16>()`, so the
/// dequant core applies it to both the S' s'_j / M' m'_j products and the final
/// mac.  Rounding to nearest instead leaves ~56% of the reconstructed weights
/// 1 ulp off hardware; with the floor the CPU model is bit-exact.
inline float q4k_bf16_floor(float x) {
    uint32_t u; std::memcpy(&u, &x, 4);
    if (u & 0xFFFFu) {
        if (u & 0x80000000u) u += 0x10000u;        // negative: away from zero
        u &= 0xFFFF0000u;
    }
    float o; std::memcpy(&o, &u, 4); return o;
}


inline void q4_k_quantize_pack(u8* dst, bf16* w_deq, const bf16* w, int rows, int cols) {
    using B = q4_k_block;
    assert(rows % B::ROWS == 0);
    assert(cols % B::COLS == 0);

    const int row_blocks = rows / B::ROWS;
    const int col_blocks = cols / B::COLS;

#pragma omp parallel for collapse(2) schedule(static)
    for (int br = 0; br < row_blocks; br++) {
        for (int bc = 0; bc < col_blocks; bc++) {
            u8* blk = dst + ((size_t)br * col_blocks + bc) * B::BYTES;
            u8* scales = blk;
            u8* mins = blk + B::SCALE_BYTES;
            u8* quants = blk + B::SCALE_BYTES + B::MIN_BYTES;
            bf16* S = (bf16*)(quants + B::QS_BYTES);
            bf16* M = S + B::ROWS;
            memset(quants, 0, B::QS_BYTES);

            for (int r = 0; r < B::ROWS; r++) {
                const int row = br * B::ROWS + r;
                const bf16* src = w + (size_t)row * cols + bc * B::COLS;
                bf16* deq = w_deq ? w_deq + (size_t)row * cols + bc * B::COLS : nullptr;

                // Per-group min/max fit: w = t_j q - u_j, both exact in fp32.
                float t[B::GROUPS], u[B::GROUPS];
                u8 q[B::COLS];
                for (int g = 0; g < B::GROUPS; g++) {
                    float lo = (float)src[g * B::GROUP];
                    float hi = lo;
                    for (int e = 1; e < B::GROUP; e++) {
                        float v = (float)src[g * B::GROUP + e];
                        lo = std::min(lo, v);
                        hi = std::max(hi, v);
                    }
                    t[g] = (hi - lo) / 15.0f;
                    u[g] = -lo;
                    const float inv = t[g] > 0.0f ? 1.0f / t[g] : 0.0f;
                    for (int e = 0; e < B::GROUP; e++) {
                        const int c = g * B::GROUP + e;
                        q[c] = (u8)std::clamp((int)lrintf(((float)src[c] - lo) * inv), 0, 15);
                    }
                }

                // Re-fit both metadata sides onto (bf16 super-block, uint8 group).
                float S_fit, M_fit, s8[B::GROUPS], m8[B::GROUPS];
                q4k_refit_one_side(t, B::GROUPS, S_fit, s8);
                q4k_refit_one_side(u, B::GROUPS, M_fit, m8);
                const bf16 S16 = bf16(S_fit);
                const bf16 M16 = bf16(-M_fit);   // the kernel adds the min path
                S[r] = S16;
                M[r] = M16;

                for (int g = 0; g < B::GROUPS; g++) {
                    // group major over the full row block, same as the q4nx scales
                    scales[g * B::ROWS + r] = (u8)s8[g];
                    mins[g * B::ROWS + r] = (u8)m8[g];
                    if (!deq) continue;
                    // The kernel forms S' s'_j and M' m'_j in bf16 before the mac,
                    // so convert here too, with the srs's floor.
                    const float t_eff = q4k_bf16_floor(s8[g] * (float)S16);
                    const float u_eff = q4k_bf16_floor(m8[g] * (float)M16);
                    for (int e = 0; e < B::GROUP; e++) {
                        const int c = g * B::GROUP + e;
                        deq[c] = bf16(q4k_bf16_floor(std::fma((float)q[c], t_eff, u_eff)));
                    }
                }

                // quants: one column is ROWS nibbles, rows paired into a byte with
                // the even row in the low nibble, columns ascending.
                for (int c = 0; c < B::COLS; c++)
                    quants[c * (B::ROWS / 2) + r / 2] |= (u8)(q[c] << ((r % 2) * 4));
            }
        }
    }
}

/// \brief y = x * w^T, accumulated in fp32.
/// \param y    L x N row major output
/// \param x    L x K row major activations
/// \param w    N x K row major (already dequantized) weights
inline void cpu_matmul_nt(bf16* y, const bf16* x, const bf16* w, int L, int N, int K) {
#pragma omp parallel for schedule(static)
    for (int l = 0; l < L; l++) {
        const bf16* x_row = x + (size_t)l * K;
        for (int n = 0; n < N; n++) {
            const bf16* w_row = w + (size_t)n * K;
            float acc = 0.0f;
            for (int k = 0; k < K; k++) acc = std::fma((float)x_row[k], (float)w_row[k], acc);
            y[(size_t)l * N + n] = bf16(acc);
        }
    }
}

#endif
