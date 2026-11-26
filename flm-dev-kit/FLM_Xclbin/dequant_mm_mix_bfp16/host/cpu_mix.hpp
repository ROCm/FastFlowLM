// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#ifndef __CPU_MIX_HPP__
#define __CPU_MIX_HPP__
#pragma once

// CPU model of the mixed-precision decode layout (FLM_DLL/detail/qwen3_asr_npu, MIX_FILE), used to build the weights and the
// golden reference of this xclbin without a model on disk.  Per 32 x 256 block, w = d q + m (Q4_1) or w = d q (Q8_0):
//
//     Q4_1  bf16 scales[8][32], bf16 mins[8][32], uint4 qs[256 cols][32 rows]   5120 B
//     Q8_0  bf16 scales[8][32],                   int8  qs[256 cols][32 rows]   8704 B
//
// qs is plain column major over the whole block (a Q4_1 column is 16 B, byte j holding rows 2j and 2j + 1, low nibble
// first).  On the NPU a weight is a sequence of 10240 B units, two Q4_1 blocks or one Q8_0 block and padding, ordered per
// pair of 32-row blocks and per unit: the unit of the first block row, then that of the second (reorder_units).

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

#include "cpu_q4_1.hpp"

static constexpr int MIX_Q41_BLOCK = 5120;
static constexpr int MIX_Q80_BLOCK = 8704;

/// @brief Quantize rows x cols weights to Q4_1 (fmt 0) or Q8_0 (fmt 1) blocks, block-row major; w_deq gets what the core
///        rebuilds (its bf16 conversion truncates toward -infinity, see q41_bf16_floor)
inline void mix_quantize(u8* dst, bf16* w_deq, const bf16* w, int rows, int cols, int fmt) {
    const int block = fmt == 0 ? MIX_Q41_BLOCK : MIX_Q80_BLOCK;
    const int row_blocks = rows / 32;
    const int col_blocks = cols / 256;
#pragma omp parallel for collapse(2) schedule(static)
    for (int br = 0; br < row_blocks; br++) {
        for (int bc = 0; bc < col_blocks; bc++) {
            u8* blk = dst + ((size_t)br * col_blocks + bc) * block;
            bf16* scales = (bf16*)blk;
            bf16* mins = (bf16*)(blk + 512);
            u8* qs = blk + (fmt == 0 ? 1024 : 512);
            memset(qs, 0, fmt == 0 ? 4096 : 8192);
            for (int r = 0; r < 32; r++) {
                const int row = br * 32 + r;
                const bf16* src = w + (size_t)row * cols + bc * 256;
                bf16* deq = w_deq + (size_t)row * cols + bc * 256;
                for (int g = 0; g < 8; g++) {
                    float lo = (float)src[g * 32], hi = lo, amax = 0.0f;
                    for (int t = 0; t < 32; t++) {
                        const float v = (float)src[g * 32 + t];
                        lo = std::min(lo, v);
                        hi = std::max(hi, v);
                        amax = std::max(amax, std::fabs(v));
                    }
                    if (fmt == 0) {
                        const bf16 d16 = bf16((hi - lo) / 15.0f);
                        const bf16 m16 = bf16(lo);
                        const float d = (float)d16, m = (float)m16;
                        const float inv_d = d > 0.0f ? 1.0f / d : 0.0f;
                        scales[g * 32 + r] = d16;
                        mins[g * 32 + r] = m16;
                        for (int t = 0; t < 32; t++) {
                            const int c = g * 32 + t;
                            const int q = std::clamp((int)lrintf(((float)src[c] - m) * inv_d), 0, 15);
                            qs[c * 16 + r / 2] |= (u8)(q << ((r % 2) * 4));
                            deq[c] = bf16(q41_bf16_floor(std::fma((float)q, d, m)));
                        }
                    } else {
                        const bf16 d16 = bf16(amax / 127.0f);
                        const float d = (float)d16;
                        const float inv_d = d > 0.0f ? 1.0f / d : 0.0f;
                        scales[g * 32 + r] = d16;
                        for (int t = 0; t < 32; t++) {
                            const int c = g * 32 + t;
                            const int q = std::clamp((int)lrintf((float)src[c] * inv_d), -127, 127);
                            qs[c * 32 + r] = (u8)(int8_t)q;
                            deq[c] = bf16(q41_bf16_floor((float)q * d));
                        }
                    }
                }
            }
        }
    }
}

/// @brief The NPU units of a quantized weight (see the file comment); the same as qwen3_asr_desc::reorder_units
inline void mix_reorder_units(u8* dst, const u8* src, int rows, int cols, int fmt) {
    const size_t block = fmt == 0 ? MIX_Q41_BLOCK : MIX_Q80_BLOCK;
    const int blocks_per_row = cols / 256;
    const int per_unit = fmt == 0 ? 2 : 1;
    const int units_per_row = blocks_per_row / per_unit;
    u8* d = dst;
    for (int r = 0; r < rows / 32; r += 2)
        for (int u = 0; u < units_per_row; u++)
            for (int i = 0; i < 2; i++, d += 10240) {
                const u8* s = src + ((size_t)(r + i) * blocks_per_row + (size_t)u * per_unit) * block;
                memcpy(d, s, per_unit * block);
                memset(d + per_unit * block, 0, 10240 - per_unit * block);
            }
}

#endif
