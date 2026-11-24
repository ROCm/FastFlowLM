//===- kernel.cc - vadd_fp32: C = A + B, A the same for every core -*- C++ -*-===//
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
// One vector of FP_N fp32 per core and launch: c = a + b. `a` is the same vector
// on every core (broadcast from column 0); `b` and `c` are the core's slices.
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>

#ifndef FP_N
#define FP_N 256
#endif

extern "C" {
void vadd_fp32(float *__restrict a, float *__restrict b, float *__restrict c);
}

void vadd_fp32(float *__restrict a, float *__restrict b, float *__restrict c) {
  constexpr int L = 16;
  for (int i = 0; i < FP_N; i += L) {
    aie::vector<float, L> va = aie::load_v<L>(a + i), vb = aie::load_v<L>(b + i);
    aie::store_v(c + i, aie::add(va, vb));
  }
}
