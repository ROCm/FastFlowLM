//===- bf16.hpp - bf16 conversions and the result check of the host programs -*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
// Portions of this file consist of AI-generated content.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace ew {

/// bf16 is the top half of an fp32; round to nearest even.
inline uint16_t to_bf16(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  if ((u & 0x7fffffff) > 0x7f800000)
    return (uint16_t)((u >> 16) | 0x40); // a NaN stays a (quiet) NaN
  return (uint16_t)((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}
inline float from_bf16(uint16_t h) {
  uint32_t u = (uint32_t)h << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

/// The NPU's result `got` for a op b (bf16 inputs) is right if it is the CPU's
/// value, or one ulp away (the NPU rounds the bf16 narrowing its own way), or
/// both are NaN, or +0 / -0 (0 times a negative number). A wrong program gives
/// a different number (T1 + C for T1 * C).
inline bool bf16_ok(uint16_t a, uint16_t b, uint16_t got, bool mul) {
  const float fa = from_bf16(a), fb = from_bf16(b);
  const uint16_t want = to_bf16(mul ? fa * fb : fa + fb);
  const float g = from_bf16(got), w = from_bf16(want);
  if (std::isnan(g) || std::isnan(w))
    return std::isnan(g) && std::isnan(w);
  if (g == w)
    return true;
  return std::abs((int)got - (int)want) <= 1;
}

} // namespace ew
