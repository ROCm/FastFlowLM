// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file gemma_bfp16_pack.hpp
/// \brief bf16 -> bfp16 packers for the gemma3 decode KV copy. tile_group / vt_tile_group are copies of
///        FLM_DLL/detail/minicpm_ocr_npu/bfp16_pack.hpp (own namespace: engines must not share inline code,
///        AGENTS.md section 7.1). No engine dependencies, so the kernel test can include it. See README.md
///        "bfp16 KV cache".
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <immintrin.h>
#include "typedef.hpp"

namespace gemma_npu_bfp16 {

constexpr int chunk_rows = 16;
constexpr int head_dim = 256;
constexpr int heads_per_stream = 2;
constexpr int head_bytes = chunk_rows * head_dim / 8 * 9;    // one head of one chunk: 4,608 B
constexpr int unit_bytes = heads_per_stream * head_bytes;     // one stream (K01, K23, V01 or V23) of one chunk
constexpr int slice_bytes = head_dim * 2;                     // 512 B: one head's part of one bf16 cache row
constexpr int slices = head_bytes / slice_bytes;              // 9

// 16 bf16 rows (row stride row_stride) -> one 16-row group of bfp16 tiles [Din / 8][2 row blocks][72 B].
inline void tile_group(uint8_t* group_tiles, const bf16* rows, int Din, int row_stride) {
    const size_t stride = row_stride > 0 ? row_stride : Din;
    const __m512i xor1 = _mm512_set_epi32(14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1);
    const __m512i xor2 = _mm512_set_epi32(13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2);
    const __m512i xor4 = _mm512_set_epi32(11, 10, 9, 8, 15, 14, 13, 12, 3, 2, 1, 0, 7, 6, 5, 4);
    const __m512i exp_mask = _mm512_set1_epi32(0xFF);
    const __m512i mantissa_mask = _mm512_set1_epi32(0x7FFFFF);
    const __m512i implicit_one = _mm512_set1_epi32(0x800000);
    const __m512i zero = _mm512_setzero_si512();

    for (int kb = 0; kb < Din / 8; kb += 2) {
        uint8_t* tile_pair = group_tiles + (size_t)kb * 144;
        for (int r = 0; r < 16; r++) {
            __m512i bits = _mm512_castps_si512(bf16o_fp32_512(_mm256_loadu_si256((const __m256i*)(rows + r * stride + kb * 8))));
            __m512i exp = _mm512_and_si512(_mm512_srli_epi32(bits, 23), exp_mask);
            __m512i max_exp = _mm512_max_epu32(exp, _mm512_permutexvar_epi32(xor1, exp));
            max_exp = _mm512_max_epu32(max_exp, _mm512_permutexvar_epi32(xor2, max_exp));
            max_exp = _mm512_max_epu32(max_exp, _mm512_permutexvar_epi32(xor4, max_exp));

            __m512i magnitude = _mm512_and_si512(bits, mantissa_mask);
            __mmask16 normal = _mm512_cmpneq_epi32_mask(exp, zero);
            magnitude = _mm512_mask_or_epi32(magnitude, normal, magnitude, implicit_one);
            __mmask16 negative = _mm512_cmplt_epi32_mask(bits, zero);
            __m512i mantissa = _mm512_mask_sub_epi32(magnitude, negative, zero, magnitude);
            __m512i shift = _mm512_min_epi32(_mm512_add_epi32(_mm512_sub_epi32(max_exp, exp), _mm512_set1_epi32(17)), _mm512_set1_epi32(31));
            __m128i q8 = _mm512_cvtepi32_epi8(_mm512_srav_epi32(mantissa, shift));

            uint8_t* block_0 = tile_pair + (r / 8) * 72 + (r % 8) * 9;
            uint8_t* block_1 = block_0 + 144;
            block_0[0] = (uint8_t)_mm_cvtsi128_si32(_mm512_castsi512_si128(max_exp));
            block_1[0] = (uint8_t)_mm_cvtsi128_si32(_mm512_extracti32x4_epi32(max_exp, 2));
            _mm_storel_epi64((__m128i*)(block_0 + 1), q8);
            _mm_storel_epi64((__m128i*)(block_1 + 1), _mm_srli_si128(q8, 8));
        }
    }
}

// 16 keys (row_stride apart) x Din of V -> V^T bfp16 tiles [Din / 8][2 key blocks][72 B]: a tile is 8 dh x 8 keys,
// each dh one block of its 8 keys.
inline void vt_tile_group(uint8_t* group_tiles, const bf16* rows, int Din, int row_stride) {
    const __m256i exp_mask = _mm256_set1_epi32(0xFF);
    const __m256i mantissa_mask = _mm256_set1_epi32(0x7FFFFF);
    const __m256i implicit_one = _mm256_set1_epi32(0x800000);
    const __m256i zero = _mm256_setzero_si256();

    for (int kg = 0; kg < 2; kg++) {
        for (int j = 0; j < Din / 8; j++) {
            __m256i bits[8];
            __m256i max_exp = zero;
            for (int k = 0; k < 8; k++) {
                const __m128i v = _mm_loadu_si128((const __m128i*)(rows + (size_t)(kg * 8 + k) * row_stride + j * 8));
                bits[k] = _mm256_slli_epi32(_mm256_cvtepu16_epi32(v), 16);
                max_exp = _mm256_max_epu32(max_exp, _mm256_and_si256(_mm256_srli_epi32(bits[k], 23), exp_mask));
            }
            alignas(32) uint32_t exps[8];
            _mm256_store_si256((__m256i*)exps, max_exp);
            uint8_t* tile = group_tiles + ((size_t)j * 2 + kg) * 72;
            for (int r = 0; r < 8; r++) {
                tile[r * 9] = (uint8_t)exps[r];
            }
            for (int k = 0; k < 8; k++) {
                __m256i exp = _mm256_and_si256(_mm256_srli_epi32(bits[k], 23), exp_mask);
                __m256i magnitude = _mm256_and_si256(bits[k], mantissa_mask);
                magnitude = _mm256_mask_or_epi32(magnitude, _mm256_cmpneq_epi32_mask(exp, zero), magnitude, implicit_one);
                __m256i mantissa = _mm256_mask_sub_epi32(magnitude, _mm256_cmplt_epi32_mask(bits[k], zero), zero, magnitude);
                __m256i shift = _mm256_min_epi32(_mm256_add_epi32(_mm256_sub_epi32(max_exp, exp), _mm256_set1_epi32(17)), _mm256_set1_epi32(31));
                alignas(16) uint8_t q8[16];
                _mm_store_si128((__m128i*)q8, _mm256_cvtepi32_epi8(_mm256_srav_epi32(mantissa, shift)));
                for (int r = 0; r < 8; r++) {
                    tile[r * 9 + 1 + k] = q8[r];
                }
            }
        }
    }
}

// One stream's 16 cache rows (row = 2 heads x 256 bf16) -> its 9,216 B unit. Each head is packed to 4,608 B
// (K: tile_group, blocks along dh; V: vt_tile_group, blocks along keys), then the two heads are interleaved in
// 512 B slices, [9][head 0 slice | head 1 slice], the shape of 9 bf16 cache rows. So the mem tile splits packed
// units and bf16 rows into the two heads with the same fixed pattern.
inline void pack_stream_chunk(uint8_t* unit, const bf16* rows, bool is_v) {
    alignas(64) uint8_t head[heads_per_stream][head_bytes];
    const int row_stride = heads_per_stream * head_dim;
    for (int h = 0; h < heads_per_stream; h++) {
        if (is_v) {
            vt_tile_group(head[h], rows + h * head_dim, head_dim, row_stride);
        } else {
            tile_group(head[h], rows + h * head_dim, head_dim, row_stride);
        }
    }
    for (int s = 0; s < slices; s++) {
        for (int h = 0; h < heads_per_stream; h++) {
            memcpy(unit + (s * heads_per_stream + h) * slice_bytes, head[h] + s * slice_bytes, slice_bytes);
        }
    }
}

} // namespace gemma_npu_bfp16
