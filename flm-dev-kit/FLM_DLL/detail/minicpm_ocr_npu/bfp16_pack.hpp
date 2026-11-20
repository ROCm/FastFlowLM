// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file bfp16_pack.hpp
/// \brief bf16 -> bfp16 packers shared by the prefill attention (attn_bfp16.hpp), the decode KV copy
///        (decode_kv_bfp16.hpp) and the decode kernel test (FLM_Xclbin/minicpm_ocr/decoding/host). No engine
///        dependencies, so the kernel test can include it. See README.md "Decode attention on bfp16 K and V".
#pragma once

#include <cstddef>
#include <cstdint>
#include <immintrin.h>
#include "typedef.hpp"

namespace minicpm_ocr_bfp16 {

// 16 bf16 rows (row stride row_stride) -> one 16-row group of bfp16 tiles [Din / 8][2 row blocks][72 B]. Copy of
// cpu_func::_bfp16_tile_group (origin/main, FLM_DLL/detail/qwen3_5vl_flash/cpu_functions.hpp).
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
// each dh one block of its 8 keys. Copy of cpu_func::_bfp16_vt_tile_group (same file as above).
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

// Decode copy of one attention layer (decode_kv_bfp16.hpp): every finished 16-key chunk as one K unit and one V unit,
// each [KV head][DH / 8][2 key blocks][72 B] = 9,216 B at the 0.8B shape. A K head part is tile_group's 16-row group
// (blocks along dh, one key per block), a V head part vt_tile_group's (blocks along keys): the blocks the decode
// kernel's emulated bf16 mmul makes, so the kernel can feed them to the native bfp16 mmul unchanged.
constexpr int decode_chunk_rows = 16;

inline size_t decode_head_bytes(int DH) { return (size_t)decode_chunk_rows * DH / 8 * 9; }
inline size_t decode_unit_bytes(int kv_heads, int DH) { return (size_t)kv_heads * decode_head_bytes(DH); }

// rows of one chunk (row = kv_heads x DH bf16, K and V each) -> its K unit and V unit
inline void pack_decode_chunk(uint8_t* k_unit, uint8_t* v_unit, const bf16* k_rows, const bf16* v_rows, int kv_heads, int DH) {
    const int row = kv_heads * DH;
    for (int h = 0; h < kv_heads; h++) {
        tile_group(k_unit + h * decode_head_bytes(DH), k_rows + h * DH, DH, row);
        vt_tile_group(v_unit + h * decode_head_bytes(DH), v_rows + h * DH, DH, row);
    }
}

} // namespace minicpm_ocr_bfp16
