// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file hunyuan_bfp16.hpp
/// \brief bfp16 on the host: the KV cache copy for decode attention (two packers, "bfp16 KV
///        copy" in README.md), the prefill matmul input x ("x as bfp16 tiles" in README.md), and
///        the ranged BO syncs they need.
#ifndef __HUNYUAN_BFP16_HPP__
#define __HUNYUAN_BFP16_HPP__
#include <immintrin.h>
#include <cassert>
#include <cstdint>
#include "buffer.hpp"
#include "typedef.hpp"

namespace hunyuan_bfp16 {

/// bytes of one 16-key chunk of one head (128 dh): 16 d-blocks x 2 key blocks x 72 B
constexpr size_t HEAD_CHUNK_BYTES = 2304;
constexpr int CHUNK_KEYS = 16;

// The packers below are copied verbatim from FLM_DLL/detail/qwen3_5vl_flash/cpu_functions.hpp
// (their rounding is the AIE2P bf16 -> bfp16 truncation; checked bit-identical against the
// emulated mmul for Hy-MT2's decode attention, decoding/README.md).
void _bfp16_tile_group(uint8_t* group_tiles, const bf16* rows, int Din, int row_stride = 0){
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

// 16 rows (keys, row_stride apart) x Din of V -> V^T bfp16 tiles [Din / 8][2 key blocks][72 B] for the 2B attention
// kernel: a tile is 8 dh x 8 keys, each dh one block of its 8 keys (same rounding as _bfp16_tile_group).
void _bfp16_vt_tile_group(uint8_t* group_tiles, const bf16* rows, int Din, int row_stride){
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
                // key k of every dh row r: the 8 x 8 transpose done with plain stores (no AVX-512 VBMI needed)
                for (int r = 0; r < 8; r++) {
                    tile[r * 9 + 1 + k] = q8[r];
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// GEMM A side (the prefill overlay's dequant_mm reads x as bfp16, WP2)
// ---------------------------------------------------------------------------

/// bytes of `rows` (a multiple of 16) rows of Din activations as bfp16 tiles
constexpr size_t x_tiles_bytes(size_t rows, size_t Din) { return rows * Din * 9 / 8; }

/// columns the host makes and packs at a time (hunyuan_write_x_tiles in hunyuan_prefill.hpp)
constexpr int X_TILE_PIECE = 256;

/// 16 rows of Din bf16 (row_stride apart) -> bfp16 A tiles [Din / 8][2 row blocks][72 B], the
/// blocks the prefill overlay's emulated bf16 mmul makes from x (8 k of one row share an exponent).
/// Its rounding is not _bfp16_tile_group's truncation: the overlay kernel sets conv_even, and then
/// to_v64bfp16ebs8 rounds to nearest even, bumps the block exponent when a value rounds up to +128
/// (and rounds the 8 again from the inputs) and reads subnormal inputs as 0. Measured with
/// run.exe --identity; see "x as bfp16 tiles" in prefill_overlay/README.md. Din % 32 == 0.
inline void _bfp16_rne_tile_group(uint8_t* group_tiles, const bf16* rows, int Din, int row_stride = 0) {
    const size_t stride = row_stride > 0 ? row_stride : Din;
    const __m512i exp_mask = _mm512_set1_epi16(0xFF);
    const __m512i mantissa_mask = _mm512_set1_epi16(0x7F);
    const __m512i implicit_one = _mm512_set1_epi16(0x80);
    const __m512i one = _mm512_set1_epi16(1);
    const __m512i eight = _mm512_set1_epi16(8);
    const __m512i top = _mm512_set1_epi16(128);
    const __m512i zero = _mm512_setzero_si512();
    // 16-bit lanes: one 128-bit lane = one block of 8, so the block max needs in-lane shuffles only
    auto block_max = [](__m512i v) {
        v = _mm512_max_epu16(v, _mm512_shuffle_epi32(v, _MM_PERM_BADC));
        v = _mm512_max_epu16(v, _mm512_shuffle_epi32(v, _MM_PERM_CDAB));
        return _mm512_max_epu16(v, _mm512_rol_epi32(v, 16));
    };

    assert(Din % 32 == 0);
    for (int kb = 0; kb < Din / 8; kb += 4) {
        uint8_t* tiles = group_tiles + (size_t)kb * 144;
        for (int r = 0; r < 16; r++) {
            // 32 values = the 4 blocks of k blocks kb .. kb + 3 of row r
            const __m512i bits = _mm512_loadu_si512((const void*)(rows + r * stride + kb * 8));
            const __m512i exp = _mm512_and_si512(_mm512_srli_epi16(bits, 7), exp_mask);
            const __mmask32 normal = _mm512_cmpneq_epi16_mask(exp, zero);
            const __mmask32 negative = _mm512_movepi16_mask(bits);
            // 8-bit mantissa with the implicit one; subnormals read as 0
            const __m512i m8 = _mm512_maskz_mov_epi16(normal, _mm512_or_si512(_mm512_and_si512(bits, mantissa_mask), implicit_one));
            __m512i max_exp = block_max(exp);

            // the 24-bit mantissa >> (max_exp - exp + 17) is m8 >> (d + 1), d = max_exp - exp, rounded
            // to nearest even: (m8 + 2^d - 1 + odd bit) >> (d + 1). From d = 8 on it is 0.
            auto round_shift = [&](__m512i d) {
                d = _mm512_min_epu16(d, eight);
                const __m512i shift = _mm512_add_epi16(d, one);
                const __m512i half_minus_one = _mm512_sub_epi16(_mm512_sllv_epi16(one, d), one);
                const __m512i odd = _mm512_and_si512(_mm512_srlv_epi16(m8, shift), one);
                return _mm512_srlv_epi16(_mm512_add_epi16(_mm512_add_epi16(m8, half_minus_one), odd), shift);
            };
            const __m512i d = _mm512_sub_epi16(max_exp, exp);
            __m512i q = round_shift(d);
            // +128 does not fit int8 (-128 does): that block takes the next exponent
            const __mmask32 overflow = _mm512_mask_cmpge_epu16_mask(~negative, q, top);
            if (overflow) {
                const __m512i bump = block_max(_mm512_maskz_mov_epi16(overflow, one));
                max_exp = _mm512_add_epi16(max_exp, bump);
                q = round_shift(_mm512_add_epi16(d, bump));
            }
            const __m512i mantissa = _mm512_mask_sub_epi16(q, negative, zero, q);

            // block j: exponent byte, then its 8 mantissas; the 4 blocks are 144 B apart
            const __m256i q8 = _mm512_cvtepi16_epi8(mantissa);
            const __m256i e8 = _mm512_cvtepi16_epi8(max_exp);
            const __m128i q_lo = _mm256_castsi256_si128(q8);
            const __m128i q_hi = _mm256_extracti128_si256(q8, 1);
            const __m128i e_lo = _mm256_castsi256_si128(e8);
            const __m128i e_hi = _mm256_extracti128_si256(e8, 1);
            uint8_t* block_0 = tiles + (r / 8) * 72 + (r % 8) * 9;
            uint8_t* block_1 = block_0 + 144;
            uint8_t* block_2 = block_0 + 288;
            uint8_t* block_3 = block_0 + 432;
            block_0[0] = (uint8_t)_mm_extract_epi8(e_lo, 0);
            block_1[0] = (uint8_t)_mm_extract_epi8(e_lo, 8);
            block_2[0] = (uint8_t)_mm_extract_epi8(e_hi, 0);
            block_3[0] = (uint8_t)_mm_extract_epi8(e_hi, 8);
            _mm_storel_epi64((__m128i*)(block_0 + 1), q_lo);
            _mm_storeh_pd((double*)(block_1 + 1), _mm_castsi128_pd(q_lo));
            _mm_storel_epi64((__m128i*)(block_2 + 1), q_hi);
            _mm_storeh_pd((double*)(block_3 + 1), _mm_castsi128_pd(q_hi));
        }
    }
}

/// push host writes of bytes [off, off + len) of `b` to the device
template <typename T>
inline void flush_bytes(buffer<T>& b, size_t off, size_t len) {
    if (len == 0) return;
#if defined(FLM_USE_HRX)
    hrx::hrx_report(hrx_buffer_flush_range(b.bo().handle(), off, len), "hunyuan_bfp16 flush");
#else
    b.bo().sync(XCL_BO_SYNC_BO_TO_DEVICE, len, off);
#endif
}

/// make device writes to bytes [off, off + len) of `b` visible to the host
template <typename T>
inline void invalidate_bytes(buffer<T>& b, size_t off, size_t len) {
    if (len == 0) return;
#if defined(FLM_USE_HRX)
    hrx::hrx_report(hrx_buffer_invalidate_range(b.bo().handle(), off, len), "hunyuan_bfp16 invalidate");
#else
    b.bo().sync(XCL_BO_SYNC_BO_FROM_DEVICE, len, off);
#endif
}

} // namespace hunyuan_bfp16

#endif // __HUNYUAN_BFP16_HPP__
