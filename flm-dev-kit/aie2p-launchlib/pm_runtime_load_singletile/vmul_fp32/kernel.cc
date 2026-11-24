//===- kernel.cc - vmul_fp32: C = A * B, from the column's pair block -*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
// Portions of this file consist of AI-generated content.
//
//===----------------------------------------------------------------------===//
//
// One vector of FP_N fp32 per core and launch. The core gets the block of its whole
// column, [A of every core | B of every core] (FP_TILES * FP_N each), and computes its
// own slice: c = a[slice] * b[slice].
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef FP_N
#define FP_N 256
#endif
#ifndef FP_TILES
#define FP_TILES 4
#endif

extern "C" {
void vmul_fp32(float *__restrict pair, float *__restrict c, int32_t slice);
}

void vmul_fp32(float *__restrict pair, float *__restrict c, int32_t slice) {
  constexpr int L = 16;
  const float *a = pair + slice * FP_N, *b = pair + FP_TILES * FP_N + slice * FP_N;
  for (int i = 0; i < FP_N; i += L) {
    aie::vector<float, L> va = aie::load_v<L>(a + i), vb = aie::load_v<L>(b + i);
    aie::store_v(c + i, aie::mul(va, vb).to_vector<float>());
  }
}
