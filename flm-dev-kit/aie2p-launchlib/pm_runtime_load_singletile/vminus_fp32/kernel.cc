//===- kernel.cc - vminus_fp32: C = A - B, A from the core's stream port -*- C++ -*-===//
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
// A does not go through a DMA into data memory: the shim streams the column's
// A block (FP_TILES * count fp32) to the stream port of every core of the column,
// and each core reads all of it and keeps its own slice. `count` comes from the
// core's RTP buffer (the runtime sequence writes it). B and C are the core's
// slices in data memory.
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef FP_TILES
#define FP_TILES 4
#endif
#ifndef FP_N
#define FP_N 256
#endif

extern "C" {
void vminus_fp32(float *__restrict b, float *__restrict c, int32_t count, int32_t slice);
}

void vminus_fp32(float *__restrict b, float *__restrict c, int32_t count, int32_t slice) {
  // Every core reads the whole column block from its stream port (the stream is
  // broadcast); the words of other cores are dropped.
  for (int32_t s = 0; s < FP_TILES; ++s) {
    for (int32_t i = 0; i < count; ++i) {
      const int32_t w = get_ss_int(); // one fp32, as its bits
      if (s == slice) {
        float a;
        __builtin_memcpy(&a, &w, 4);
        c[i] = a - b[i];
      }
    }
  }
}
