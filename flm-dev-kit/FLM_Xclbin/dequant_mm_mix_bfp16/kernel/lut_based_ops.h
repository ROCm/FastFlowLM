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
//
template <unsigned N>
__attribute__((always_inline)) aie::vector<bfloat16, N> getSiluBf16_nonLUT(aie::vector<bfloat16, N> x){
    const bfloat16 half = 0.5f;
    const bfloat16 one = 1.0f;
    aie::vector<bfloat16, N> v_half = aie::broadcast<bfloat16, N>(half);
    aie::vector<bfloat16, N> v_one = aie::broadcast<bfloat16, N>(one);
    aie::accum<accfloat, N> x_mul_half = aie::mul(x, v_half);
    aie::vector<bfloat16, N> tanh_x_half = aie::tanh<bfloat16>(x_mul_half.template to_vector<float>());

    aie::vector<bfloat16, N> tanh_x_half_plus_one = aie::add(tanh_x_half, v_one);
    aie::vector<bfloat16, N> tanh_x_half_plus_one_div_2 = aie::mul(tanh_x_half_plus_one, v_half).template to_vector<bfloat16>();

    aie::vector<bfloat16, N> res = aie::mul(x, tanh_x_half_plus_one_div_2).template to_vector<bfloat16>();
    return res;
}
/// gelu with the tanh approximation:  0.5 x (1 + tanh(c (x + b x^3))),  c = sqrt(2/pi), b = 0.044715
/// Written to round to bf16 as rarely as possible: the polynomial is  c x (1 + b x^2)  = x (c + c b x^2)  and the outer factor
/// 0.5 (1 + tanh) is one multiply-accumulate, so only x^2, the polynomial factor, the outer factor and the result are rounded.
template <unsigned N>
__attribute__((always_inline)) aie::vector<bfloat16, N> getGeluBf16_nonLUT(aie::vector<bfloat16, N> x) {
    const aie::vector<bfloat16, N> v_c  = aie::broadcast<bfloat16, N>(bfloat16(0.79788456f));
    const aie::vector<bfloat16, N> v_cb = aie::broadcast<bfloat16, N>(bfloat16(0.79788456f * 0.044715f));
    const aie::vector<bfloat16, N> v_half = aie::broadcast<bfloat16, N>(bfloat16(0.5f));

    const aie::vector<bfloat16, N> x2 = aie::mul(x, x).template to_vector<bfloat16>();

    // c + c b x^2
    aie::accum<accfloat, N> poly;
    poly.from_vector(v_c);
    poly = aie::mac(poly, x2, v_cb);
    const aie::vector<bfloat16, N> factor = poly.template to_vector<bfloat16>();

    // tanh(x * factor), the argument stays in float
    const aie::vector<float, N> arg = aie::mul(x, factor).template to_vector<float>();
    const aie::vector<bfloat16, N> t = aie::tanh<bfloat16>(arg);

    // 0.5 + 0.5 tanh
    aie::accum<accfloat, N> outer;
    outer.from_vector(v_half);
    outer = aie::mac(outer, t, v_half);
    const aie::vector<bfloat16, N> gate = outer.template to_vector<bfloat16>();

    return aie::mul(x, gate).template to_vector<bfloat16>();
}

#endif //__LUT_BASED_OPS_H__