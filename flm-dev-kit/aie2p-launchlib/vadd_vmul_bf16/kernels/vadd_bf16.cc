//===- vadd_bf16.cc - elementwise bf16 vector add -------------*- C++ -*-===//
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
// C = A + B over EW_N bf16 elements, 16 lanes at a time. Its twin
// vmul_bf16.cc differs in one operator, so a wrong answer reads as "the wrong
// program ran". No state: a core restarted at PC 0 needs no data-memory fix.
//
//===----------------------------------------------------------------------===//

#include "elementwise_config.h"

#include <aie_api/aie.hpp>
#include <stdint.h>

extern "C" {
void vadd_bf16(bfloat16 *__restrict a, bfloat16 *__restrict b,
               bfloat16 *__restrict c);
void vadd_pair_bf16(bfloat16 *__restrict pair, bfloat16 *__restrict c,
                    int32_t slice);
}

// The "pair" layout (iron/, CORES 3-4): the core gets the block of its whole
// column, [lhs of every core | rhs of every core], and computes its own slice.
void vadd_pair_bf16(bfloat16 *__restrict pair, bfloat16 *__restrict c,
                    int32_t slice) {
  bfloat16 *a = pair + slice * EW_N;
  bfloat16 *b = pair + EW_CORES * EW_N + slice * EW_N;
  for (int i = 0; i < EW_N; i += EW_VECTOR_LEN) {
    aie::vector<bfloat16, EW_VECTOR_LEN> va =
        aie::load_v<EW_VECTOR_LEN>(a + i);
    aie::vector<bfloat16, EW_VECTOR_LEN> vb =
        aie::load_v<EW_VECTOR_LEN>(b + i);
    aie::store_v(c + i, aie::add(va, vb));
  }
}

void vadd_bf16(bfloat16 *__restrict a, bfloat16 *__restrict b,
               bfloat16 *__restrict c) {
  for (int i = 0; i < EW_N; i += EW_VECTOR_LEN) {
    aie::vector<bfloat16, EW_VECTOR_LEN> va =
        aie::load_v<EW_VECTOR_LEN>(a + i);
    aie::vector<bfloat16, EW_VECTOR_LEN> vb =
        aie::load_v<EW_VECTOR_LEN>(b + i);
    aie::store_v(c + i, aie::add(va, vb));
  }
}
