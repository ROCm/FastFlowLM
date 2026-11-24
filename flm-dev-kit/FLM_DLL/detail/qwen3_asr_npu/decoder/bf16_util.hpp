// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file bf16_util.hpp
/// \brief bf16 <-> float helpers for the CPU reference decoder.
///
/// The PyTorch reference runs in bfloat16: every elementwise op rounds its result to bf16. To land on the same
/// tokens the CPU decoder keeps activations as floats that always hold a bf16-representable value, and calls
/// bf16r() wherever torch would round.
#pragma once
#include <cstdint>
#include <cstring>
#include <cmath>

namespace qwen3_asr_cpu {

inline float bf2f(uint16_t b) {
    uint32_t u = (uint32_t)b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

/// float -> bf16 bits, round to nearest even (what torch does for .to(bfloat16)); NaN stays NaN.
inline uint16_t f2bf(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x40);
    u += 0x7fffu + ((u >> 16) & 1u);          // add half an ulp, ties go to even
    return (uint16_t)(u >> 16);
}

/// round a float to the nearest bf16 value and give it back as a float
inline float bf16r(float f) { return bf2f(f2bf(f)); }

}  // namespace qwen3_asr_cpu
