//===---  exp_lut.h - get exponential values from loopup tables ---===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
//
//
//===----------------------------------------------------------------------===//
// This is the implementation of getting exponential values for a bfloat16
// vector from exponential lookup tables.
//===----------------------------------------------------------------------===//
#ifndef __LUT_BASED_OPS_H__
#define __LUT_BASED_OPS_H__

#include "aie_api/aie.hpp"
#include "tables.h"

extern "C" {
__attribute__((always_inline)) v16accfloat getExpBf16(v16bfloat16 x) {
    bfloat16 __aie_dm_resource_a *ilut_ab = (bfloat16 __aie_dm_resource_a *)exp_ilut_ab;
    bfloat16 __aie_dm_resource_b *ilut_cd = (bfloat16 __aie_dm_resource_b *)exp_ilut_cd;
    bfloat16 __aie_dm_resource_a *flut_ab = (bfloat16 __aie_dm_resource_a *)exp_flut_ab;
    bfloat16 __aie_dm_resource_b *flut_cd = (bfloat16 __aie_dm_resource_b *)exp_flut_cd;

    using lut_type = aie::lut<4, bfloat16, bfloat16>;
    const int LUT_elems = 256;
    const int step_i = 8;
    const int step_f = 0;

    lut_type lut_i(LUT_elems, ilut_ab, ilut_cd);
    lut_type lut_f(LUT_elems, flut_ab, flut_cd);
    aie::parallel_lookup<uint16, lut_type, aie::lut_oor_policy::truncate> lookup_i(lut_i, step_i);
    aie::parallel_lookup<uint16, lut_type, aie::lut_oor_policy::truncate> lookup_f(lut_f, step_f);

    aie::vector<bfloat16, 16> I_val_vec, F_val_vec;
    aie::accum<accfloat, 16> exp_val;
    aie::vector<bfloat16, 16> input_bf16 = x;

    // position of output decimal point = 8, making input become 8 bits, and for
    // LUT_elems = 256 lookup. aie::vector<int16, 16>
    // input=aie::to_fixed<int16>(input_bf16,8);
    aie::vector<int16, 32> input0 = v32int16(bfloat16_to_int(input_bf16, 8));
    aie::vector<int16, 16> input = aie::filter_even(input0);

    I_val_vec = lookup_i.fetch(input.cast_to<uint16>());
    F_val_vec = lookup_f.fetch(input.cast_to<uint16>());
    exp_val = aie::mul(I_val_vec, F_val_vec);
    return v16accfloat(exp_val);
}



__attribute__((always_inline)) v16accfloat getExpBf16_clamped(v16bfloat16 x) {
    bfloat16 __aie_dm_resource_a *ilut_ab = (bfloat16 __aie_dm_resource_a *)exp_ilut_ab;
    bfloat16 __aie_dm_resource_b *ilut_cd = (bfloat16 __aie_dm_resource_b *)exp_ilut_cd;
    bfloat16 __aie_dm_resource_a *flut_ab = (bfloat16 __aie_dm_resource_a *)exp_flut_ab;
    bfloat16 __aie_dm_resource_b *flut_cd = (bfloat16 __aie_dm_resource_b *)exp_flut_cd;

    using lut_type = aie::lut<4, bfloat16, bfloat16>;
    const int LUT_elems = 256;
    const int step_i = 8;
    const int step_f = 0;

    lut_type lut_i(LUT_elems, ilut_ab, ilut_cd);
    lut_type lut_f(LUT_elems, flut_ab, flut_cd);
    aie::parallel_lookup<uint16, lut_type, aie::lut_oor_policy::truncate> lookup_i(lut_i, step_i);
    aie::parallel_lookup<uint16, lut_type, aie::lut_oor_policy::truncate> lookup_f(lut_f, step_f);

    aie::vector<bfloat16, 16> I_val_vec, F_val_vec;
    aie::accum<accfloat, 16> exp_val;
    aie::vector<bfloat16, 16> input_bf16 = x;

    // position of output decimal point = 8, making input become 8 bits, and for
    // LUT_elems = 256 lookup. aie::vector<int16, 16>
    // input=aie::to_fixed<int16>(input_bf16,8);
    aie::vector<int16, 32> input0 = v32int16(bfloat16_to_int(input_bf16, 8));
    aie::vector<int16, 16> input = aie::filter_even(input0);

    I_val_vec = lookup_i.fetch(input.cast_to<uint16>());
    F_val_vec = lookup_f.fetch(input.cast_to<uint16>());
    exp_val = aie::mul(I_val_vec, F_val_vec);

    // need to do a clamp on the smallest value, so
      constexpr float min_clamp =1.64386e-38f;
      constexpr float max_clamp = 3.40282e+38f;
    aie::vector<float, 16> min_value = aie::broadcast<float, 16>(min_clamp);
    aie::vector<float, 16> max_value = aie::broadcast<float, 16>(max_clamp); // max of float


    aie::vector<float, 16> exp_val_float = exp_val.template to_vector<float>();
    aie::vector<float, 16> clamp_val =  aie::clamp( exp_val_float, min_value, max_value);
    

    exp_val.from_vector(clamp_val);
    return v16accfloat(exp_val);
}



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
  inv_x_val = (inv_exponent << 7) + m_inv_lut[mantissa];
  inv_x = (bfloat16 *)&inv_x_val;
  return *inv_x;
}

__attribute__((always_inline)) v16bfloat16 getCustomSILU(v16bfloat16 vInput) {
  aie::vector<bfloat16, 16> input = vInput;

  int step_bits = -3;
  int bias = 32;
  int data_size = 16;
  int LUT_elems = 64;
  int shift_offset = 0; // unused

  using lut_type = aie::lut<4, float, bfloat16>;

  lut_type test_lut(LUT_elems, (bfloat16 *)custom_sil_lut_ab, (bfloat16 *)custom_sil_lut_cd);

  aie::linear_approx<bfloat16, lut_type> lin_aprox(test_lut, step_bits, bias, shift_offset);

  aie::vector<bfloat16, 16> output = lin_aprox.compute(input).to_vector<bfloat16>();

  return (v16bfloat16)output;
}

}
#endif //__LUT_BASED_OPS_H__