//===---  exp_lut.h - get exponential values from loopup tables ---===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.
//
//
//===----------------------------------------------------------------------===//
// This is the implementation of getting exponential values for a bfloat16
// vector from exponential lookup tables.
//===----------------------------------------------------------------------===//
#ifndef __LUT_BASED_OPS_H__
#define __LUT_BASED_OPS_H__

#include "aie_api/aie.hpp"


// The tables _ab and _cd are copies of each other
// Also, each table has a copy of data seperated by 256-bit
// This effectively creates 4 copies of the table (located in 4 (256-bit) banks)
// This allows the user to use the "gather" read feature and read 4 values from
// 4 different banks at once Tables entries are in BF16

#if defined(__chess__)
alignas(aie::vector_decl_align) unsigned char m_inv_lut[128] = {
    0,   126, 124, 122, 120, 118, 117, 115, 113, 111, 109, 108, 106, 104, 103,
    101, 100, 98,  96,  95,  93,  92,  90,  89,  88,  86,  85,  83,  82,  81,
    79,  78,  77,  76,  74,  73,  72,  71,  69,  68,  67,  66,  65,  64,  63,
    61,  60,  59,  58,  57,  56,  55,  54,  53,  52,  51,  50,  49,  48,  47,
    46,  45,  44,  44,  43,  42,  41,  40,  39,  38,  37,  37,  36,  35,  34,
    33,  33,  32,  31,  30,  30,  29,  28,  27,  27,  26,  25,  24,  24,  23,
    22,  22,  21,  20,  20,  19,  18,  18,  17,  16,  16,  15,  14,  14,  13,
    13,  12,  11,  11,  10,  10,  9,   9,   8,   7,   7,   6,   6,   5,   5,
    4,   4,   3,   3,   2,   2,   1,   1};
#endif

extern "C" {

__attribute__((always_inline)) bfloat16 getInvBf16(float x) {
  unsigned int *B_x;
  unsigned int exp_mask = 0x7F800000;
  unsigned int mantissa_mask = 0x007FFFFF;
  unsigned int mantissa_Q = 0x00008000;
  unsigned char exponent, mantissa;
  unsigned inv_exponent;
  unsigned short inv_x_val;
  unsigned int B_Q;
  bfloat16 *inv_x;
  B_x = (unsigned int *)&x;
  B_Q = *B_x + mantissa_Q;
  exponent = (B_Q & exp_mask) >> 23;
  mantissa = (B_Q & mantissa_mask) >> 16;
  inv_exponent = (mantissa == 0) + (253 - exponent);
#if defined(__chess__)
  inv_x_val = (inv_exponent << 7) + m_inv_lut[mantissa];
#else
  // Same values as m_inv_lut, computed: the table would need .data, and this
  // design's buffers leave no room for it under Peano.
  unsigned inv_m = mantissa ? (32768u + (128u + mantissa) / 2) / (128u + mantissa) - 128u : 0u;
  inv_x_val = (inv_exponent << 7) + inv_m;
#endif
  inv_x = (bfloat16 *)&inv_x_val;
  return *inv_x;
}

}
#endif //__LUT_BASED_OPS_H__