// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __CPU_Q4_1_HPP__
#define __CPU_Q4_1_HPP__
#pragma once

// CPU model of the q4nx Q4_1 block, used to build a golden reference for this
// xclbin without a converted model on disk.
//
// This mirrors FLM_Q4NX_Converter/q4nx/model_converter.py::_pack_q4nx with
// `m` present and `parallel_size == 16`, i.e. exactly the bytes the NPU kernel
// in kernel/q4_core.h decodes.  Per 32x256 chunk:
//
//     bf16 scales[COLS / GROUP][ROWS]                  512 B, group major
//     bf16 mins  [COLS / GROUP][ROWS]                  512 B, group major
//     u4   qs    [ROWS / PARALLEL][COLS][PARALLEL/2]  4096 B, plane / column major
//
// 5120 B total, 5 bits per weight.  Only the quant plane is parallel-strided:
// the block's 32 rows are two 16-row planes, each laid out column major on its
// own, which is why kernel/common.hpp sets PARALLEL_ROWS = 16.  Scales and mins
// are group major over the whole 32-row block regardless.
//
// Chunks are emitted row-block major (all column blocks of row block 0, then of
// row block 1, ...), matching what Q4NX::load_weights hands back; the vertical
// interleave the cores expect is applied afterwards by reorder_cpy.

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "typedef.hpp"
#include "weights_desc.hpp"

struct q4_1_block {
    static constexpr int ROWS        = QXNX_ROW_BLOCK_SIZE;              // 32
    static constexpr int COLS        = QXNX_COL_BLOCK_SIZE;              // 256
    static constexpr int GROUP       = GGML_GROUP_SIZE;                  // 32
    static constexpr int PARALLEL    = 16;                               // converter parallel_size
    static constexpr int GROUPS      = COLS / GROUP;                     // 8
    static constexpr int SCALE_BYTES = GROUPS * ROWS * (int)sizeof(bf16);// 512
    static constexpr int MIN_BYTES   = SCALE_BYTES;                      // 512
    static constexpr int PLANE_BYTES = COLS * (PARALLEL / 2);            // 2048
    static constexpr int PLANES      = ROWS / PARALLEL;                  // 2
    static constexpr int BYTES       = SCALE_BYTES + MIN_BYTES + PLANES * PLANE_BYTES; // 5120

    static_assert(BYTES == ROWS * COLS * 5 / 8, "q4_1 is 5 bits per weight");
};

/// \brief Quantize a bf16 weight matrix to the packed q4nx Q4_1 layout.
///
/// The dequantized values are written back to \p w_deq so the caller can build a
/// reference that carries the same quantization error the NPU sees; the only
/// remaining difference is the NPU's bf16/bfp16 arithmetic.
///
/// \param dst   packed output, get_quantization_byte_size(rows * cols, flm_q41) bytes
/// \param w_deq rows x cols, the values the kernel reconstructs (may be null)
/// \param w     rows x cols row major source weights (out_features x in_features)
/// \param rows  out features, must be a multiple of 32
/// \param cols  in features, must be a multiple of 256
/// \brief fp32 -> bf16 truncating toward -infinity.
///
/// This is what the AIE srs does on `aie::accum::to_vector<bf16>()`, and the
/// dequant core's only conversion is the `aie::mac(...).to_vector<bf16>()` in
/// kernel/q4_core.h.  Rounding to nearest here instead costs ~8% of the
/// reconstructed weights a 1 ulp mismatch against hardware.
inline float q41_bf16_floor(float x) {
    uint32_t u; std::memcpy(&u, &x, 4);
    if (u & 0xFFFFu) {
        if (u & 0x80000000u) u += 0x10000u;        // negative: away from zero
        u &= 0xFFFF0000u;
    }
    float o; std::memcpy(&o, &u, 4); return o;
}

inline void q4_1_quantize_pack(u8* dst, bf16* w_deq, const bf16* w, int rows, int cols) {
    using B = q4_1_block;
    assert(rows % B::ROWS == 0);
    assert(cols % B::COLS == 0);

    const int row_blocks = rows / B::ROWS;
    const int col_blocks = cols / B::COLS;

#pragma omp parallel for collapse(2) schedule(static)
    for (int br = 0; br < row_blocks; br++) {
        for (int bc = 0; bc < col_blocks; bc++) {
            u8* blk = dst + ((size_t)br * col_blocks + bc) * B::BYTES;
            bf16* scales = (bf16*)blk;
            bf16* mins = (bf16*)(blk + B::SCALE_BYTES);
            u8* quants = blk + B::SCALE_BYTES + B::MIN_BYTES;
            memset(quants, 0, B::PLANES * B::PLANE_BYTES);

            for (int r = 0; r < B::ROWS; r++) {
                const int row = br * B::ROWS + r;
                const bf16* src = w + (size_t)row * cols + bc * B::COLS;
                bf16* deq = w_deq ? w_deq + (size_t)row * cols + bc * B::COLS : nullptr;
                // The quant plane is strided: row r lives in plane r / PARALLEL at
                // sub-row r % PARALLEL, two sub-rows to a byte, low nibble first.
                const int plane = r / B::PARALLEL;
                const int sub = r % B::PARALLEL;
                u8* plane_base = quants + plane * B::PLANE_BYTES + sub / 2;
                const int shift = (sub % 2) * 4;

                for (int g = 0; g < B::GROUPS; g++) {
                    float lo = (float)src[g * B::GROUP];
                    float hi = lo;
                    for (int t = 1; t < B::GROUP; t++) {
                        float v = (float)src[g * B::GROUP + t];
                        lo = std::min(lo, v);
                        hi = std::max(hi, v);
                    }
                    // ggml q4_1: d = (max - min) / 15, m = min, q = round((x - m) / d).
                    // Round d and m through bf16 *before* quantizing so the values
                    // recorded in w_deq are bit-exact with what the kernel rebuilds.
                    bf16 d16 = bf16((hi - lo) / 15.0f);
                    bf16 m16 = bf16(lo);
                    const float d = (float)d16;
                    const float m = (float)m16;
                    const float inv_d = d > 0.0f ? 1.0f / d : 0.0f;
                    // group major over the full row block: scales + g * ROWS + r
                    scales[g * B::ROWS + r] = d16;
                    mins[g * B::ROWS + r] = m16;

                    for (int t = 0; t < B::GROUP; t++) {
                        const int c = g * B::GROUP + t;
                        int q = (int)lrintf(((float)src[c] - m) * inv_d);
                        q = std::clamp(q, 0, 15);
                        plane_base[c * (B::PARALLEL / 2)] |= (u8)(q << shift);
                        if (deq) deq[c] = bf16(q41_bf16_floor(std::fma((float)q, d, m)));
                    }
                }
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

/// \brief Unpack a packed q4nx Q4_1 tensor back to bf16.
///
/// The inverse of q4_1_quantize_pack, for the case this test actually needs: the
/// weights come off disk already quantized, so the golden reference has to be
/// built from whatever the kernel will reconstruct rather than from an original
/// the host never sees.  Same bf16-floor on the reconstructed value, so this
/// agrees with the dequant core bit for bit.
///
/// \param w_deq rows x cols row major output
/// \param src   packed input, get_quantization_byte_size(rows * cols, flm_q41) bytes
inline void q4_1_unpack(bf16* w_deq, const u8* src, int rows, int cols) {
    using B = q4_1_block;
    assert(rows % B::ROWS == 0);
    assert(cols % B::COLS == 0);

    const int row_blocks = rows / B::ROWS;
    const int col_blocks = cols / B::COLS;

#pragma omp parallel for collapse(2) schedule(static)
    for (int br = 0; br < row_blocks; br++) {
        for (int bc = 0; bc < col_blocks; bc++) {
            const u8* blk = src + ((size_t)br * col_blocks + bc) * B::BYTES;
            const bf16* scales = (const bf16*)blk;
            const bf16* mins = (const bf16*)(blk + B::SCALE_BYTES);
            const u8* quants = blk + B::SCALE_BYTES + B::MIN_BYTES;

            for (int r = 0; r < B::ROWS; r++) {
                const int row = br * B::ROWS + r;
                bf16* deq = w_deq + (size_t)row * cols + bc * B::COLS;
                const int plane = r / B::PARALLEL;
                const int sub = r % B::PARALLEL;
                const u8* plane_base = quants + plane * B::PLANE_BYTES + sub / 2;
                const int shift = (sub % 2) * 4;

                for (int g = 0; g < B::GROUPS; g++) {
                    const float d = (float)scales[g * B::ROWS + r];
                    const float m = (float)mins[g * B::ROWS + r];
                    for (int t = 0; t < B::GROUP; t++) {
                        const int c = g * B::GROUP + t;
                        const int q = (plane_base[c * (B::PARALLEL / 2)] >> shift) & 0xF;
                        deq[c] = bf16(q41_bf16_floor(std::fma((float)q, d, m)));
                    }
                }
            }
        }
    }
}

#endif
