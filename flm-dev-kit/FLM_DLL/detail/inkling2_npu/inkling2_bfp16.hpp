// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file inkling2_bfp16.hpp
/// \brief fp32 rows -> the bfp16 x operand of the prompt matmuls (inkling2_npu_mm.hpp), in the order the cores
///        consume it, so the memtiles and core DMAs move it unchanged
/// \note bfp16 block: one shared exponent byte e and 8 int8 mantissas q, value q x 2^(e - 133) (the 8 k of one row).
///       Tile: 8 rows x 8 k = 8 blocks, 72 bytes. A 16-row group of one 256-column slice is [k8 32][rows 0-7, 8-15]
///       tiles = 4608 bytes. A core row of a dispatch block (m rows) is [slice Din / 256][m / 16 groups],
///       and the blocks of a job follow each other core row by core row: row R of a job is in core row
///       (R / m) % 4 of block R / 4m, the same rows as the y output.
#pragma once

#include <immintrin.h>

#include <cstddef>
#include <cstdint>

namespace inkling2_bfp16 {

constexpr size_t SLICE_BYTES = 4608;   // one 16-row group x 256 columns

/// \brief bytes of rows x Din
inline size_t bytes(size_t rows, uint32_t Din) { return rows * Din / 8 * 9; }

/// \brief 16 floats -> two bfp16 blocks (p[0 .. 7] -> b0, p[8 .. 15] -> b1). Rounds to nearest even, saturating;
///        a block whose largest exponent is below 6 is zero.
inline void pack2(const float* p, uint8_t* b0, uint8_t* b1) {
    const __m512i xor1 = _mm512_set_epi32(14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1);
    const __m512i xor2 = _mm512_set_epi32(13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2);
    const __m512i xor4 = _mm512_set_epi32(11, 10, 9, 8, 15, 14, 13, 12, 3, 2, 1, 0, 7, 6, 5, 4);
    const __m512i exp_mask = _mm512_set1_epi32(0xFF);
    const __m512i scale_bias = _mm512_set1_epi32(260);   // 2^(133 - e) has the biased exponent 260 - e
    const __m512i min_exp = _mm512_set1_epi32(6);         // below that a block is too small to matter: zero
    const __m512 v = _mm512_loadu_ps(p);
    const __m512i exp = _mm512_and_si512(_mm512_srli_epi32(_mm512_castps_si512(v), 23), exp_mask);
    __m512i e = _mm512_max_epu32(exp, _mm512_permutexvar_epi32(xor1, exp));
    e = _mm512_max_epu32(e, _mm512_permutexvar_epi32(xor2, e));
    e = _mm512_max_epu32(e, _mm512_permutexvar_epi32(xor4, e));
    const __mmask16 live = _mm512_cmpge_epu32_mask(e, min_exp);
    const __m512 scale = _mm512_maskz_mov_ps(live, _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_sub_epi32(scale_bias, e), 23)));
    const __m512i qi = _mm512_cvt_roundps_epi32(_mm512_mul_ps(v, scale), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    const __m128i q8 = _mm512_cvtsepi32_epi8(qi);
    const __m512i eo = _mm512_maskz_mov_epi32(live, e);
    b0[0] = (uint8_t)_mm_cvtsi128_si32(_mm512_castsi512_si128(eo));
    b1[0] = (uint8_t)_mm_cvtsi128_si32(_mm512_extracti32x4_epi32(eo, 2));
    _mm_storel_epi64((__m128i*)(b0 + 1), q8);
    _mm_storel_epi64((__m128i*)(b1 + 1), _mm_srli_si128(q8, 8));
}

/// \brief 16 rows (src[r] -> Kw floats, nullptr = a zero row) of a group into its core row region (m rows):
///        columns [Kw, Din) are zero
inline void pack_group(uint8_t* region, uint32_t m, uint32_t g, const float* const* src, uint32_t Kw, uint32_t Din) {
    const size_t slice_stride = (size_t)m / 16 * SLICE_BYTES;
    for (uint32_t r = 0; r < 16; r++) {
        const float* p = src[r];
        for (uint32_t kb = 0; kb < Din / 8; kb += 2) {
            const uint32_t q = kb / 32;   // 256-column slice
            uint8_t* tile = region + q * slice_stride + g * SLICE_BYTES + (kb % 32) * 144 +
                            (r / 8) * 72 + (r % 8) * 9;
            if (!p || kb * 8 >= Kw) {
                for (int b = 0; b < 2; b++) {
                    uint8_t* blk = tile + b * 144;
                    for (int i = 0; i < 9; i++) blk[i] = 0;
                }
                continue;
            }
            pack2(p + kb * 8, tile, tile + 144);
        }
    }
}

/// \brief the rows of one job (covered rows, src[R] as in pack_group) at dst: m rows per core row
inline void pack_job(uint8_t* dst, uint32_t m, uint32_t covered, const float* const* src, uint32_t Kw, uint32_t Din) {
    for (uint32_t R = 0; R < covered; R += 16)
        pack_group(dst + bytes(R / m * m, Din), m, (R % m) / 16, src + R, Kw, Din);
}

}  // namespace inkling2_bfp16
