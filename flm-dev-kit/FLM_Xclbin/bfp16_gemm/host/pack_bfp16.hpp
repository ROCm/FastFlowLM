// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file pack_bfp16.hpp
/// \brief bf16 <-> fp32 and the bf16 -> bfp16ebs8 block encoder the GEMM's operands use.
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

/// 8 bf16 -> one 9 B block: the largest fp32 exponent field of the 8, then each
/// value's signed 24-bit mantissa shifted right by (17 + max_exp - exp), capped at
/// 31, rounded to nearest even (what the core's own bf16 -> bfp16 conversion does
/// under conv_even). A value is mantissa x 2^(exponent - 133).
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

}  // namespace pack_bfp16

#endif
