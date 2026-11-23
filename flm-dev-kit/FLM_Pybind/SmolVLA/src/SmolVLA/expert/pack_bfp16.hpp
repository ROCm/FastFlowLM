// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file pack_bfp16.hpp
/// \brief Scalar reference of design P's operand formats: the bfp16 pre-round the
///        engine applies (vision_bfp16.hpp) and the packed tile layout.
#ifndef __PACK_BFP16_HPP__
#define __PACK_BFP16_HPP__
#pragma once
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace pack_bfp16 {

inline float bf16_to_f32(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; std::memcpy(&f, &u, 4); return f; }
inline uint16_t f32_to_bf16(float f) {   // round to nearest even
    uint32_t u; std::memcpy(&u, &f, 4);
    return (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

/// The engine's pre-round of one group of 8 (bfp16_preround_group_scalar).
inline void preround8(uint16_t* blk) {
    uint16_t mxb = 0;
    for (int i = 0; i < 8; i++) { const uint16_t m = (uint16_t)(blk[i] & 0x7FFFu); if (m > mxb) mxb = m; }
    const uint32_t be = (uint32_t)(mxb >> 7);
    if (be < 7u || be == 0xFFu) return;
    const uint32_t sb = (be - 6u) << 23, ib = (260u - be) << 23;
    float step, inv; std::memcpy(&step, &sb, 4); std::memcpy(&inv, &ib, 4);
    const float lim = std::truncf(bf16_to_f32(mxb) * inv) * step;
    for (int i = 0; i < 8; i++) {
        float q = std::nearbyintf(bf16_to_f32(blk[i]) * inv) * step;
        if (q > lim) q = lim;
        if (q < -lim - step) q = -lim - step;
        blk[i] = f32_to_bf16(q);
    }
}

/// 8 bf16 -> one 9 B block: the largest fp32 exponent field of the 8, then each
/// value's signed 24-bit mantissa shifted right by (17 + max_exp - exp), capped at
/// 31, rounded to nearest even -- to_v64bfp16ebs8 under conv_even, which is what
/// design C's cores run. On pre-rounded input the shift is exact except when a
/// block's -128 * step value (= -2^(E+1)) lifts the shared exponent by one; the
/// other values then lose their LSB, and that LSB must round as the core does.
inline void encode_block(uint8_t* out, const uint16_t* v) {
    int ex[8], mx = 0;
    for (int i = 0; i < 8; i++) { ex[i] = (v[i] >> 7) & 0xFF; if (ex[i] > mx) mx = ex[i]; }
    out[0] = (uint8_t)mx;
    for (int i = 0; i < 8; i++) {
        int32_t mag = (int32_t)(((uint32_t)v[i] & 0x7Fu) << 16);
        if (ex[i]) mag |= 0x800000;
        const int32_t s = (v[i] & 0x8000u) ? -mag : mag;
        int sh = 17 + mx - ex[i]; if (sh > 31) sh = 31;
        int32_t q = s >> sh;
        const int32_t rem = s - (int32_t)((uint32_t)q << sh), half = 1 << (sh - 1);
        if (rem > half || (rem == half && (q & 1))) q++;
        if (q > 127) q = 127;
        out[1 + i] = (uint8_t)(int8_t)q;
    }
}

/// Byte offset of the block holding (row, k8) in a blocked operand:
/// [rows/R block][K/kt tile][R/16 groups][kt/8 tile pairs][2 halves][8 rows][9 B].
/// x uses R = m (64), w R = n (96), both with kt = 384.
inline size_t block_offset(size_t row, size_t k8, size_t K, size_t R, size_t kt = 384) {
    const size_t KT = K / kt, pairs = kt / 8;
    const size_t T = row / R, rr = row % R, g = rr / 16, half = (rr % 16) / 8, r8 = rr % 8;
    const size_t t = k8 / pairs, k8l = k8 % pairs;
    return (((T * KT + t) * (R / 16) + g) * pairs + k8l) * 144 + half * 72 + r8 * 9;
}

/// rows x K bf16 (row-major, stride K) -> blocked bfp16. Rows [rows, rows_pad) are zero blocks.
inline void pack_rows(uint8_t* dst, const uint16_t* src, size_t rows, size_t rows_pad, size_t K, size_t R) {
    for (size_t r = 0; r < rows_pad; r++)
        for (size_t k8 = 0; k8 < K / 8; k8++) {
            uint8_t* o = dst + block_offset(r, k8, K, R);
            if (r < rows) encode_block(o, src + r * K + k8 * 8);
            else std::memset(o, 0, 9);
        }
}

}  // namespace pack_bfp16

#endif
