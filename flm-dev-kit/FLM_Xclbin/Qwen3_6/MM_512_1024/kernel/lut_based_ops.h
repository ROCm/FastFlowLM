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

// .bss is not zeroed when a Peano-built core is loaded. AIE_PERSIST_ZERO puts a
// zero-initialized static that must keep its value across calls in .data.
// AIE_TRANSIENT drops `static` where the value does not need to persist, so no
// init guard in .bss is involved. Both do nothing under chess.
#ifndef AIE_PERSIST_ZERO
#if defined(__chess__)
#define AIE_PERSIST_ZERO
#define AIE_TRANSIENT static
#else
#define AIE_PERSIST_ZERO __attribute__((section(".data")))
#define AIE_TRANSIENT
#endif
#endif


// The tables _ab and _cd are copies of each other
// Also, each table has a copy of data seperated by 256-bit
// This effectively creates 4 copies of the table (located in 4 (256-bit) banks)
// This allows the user to use the "gather" read feature and read 4 values from
// 4 different banks at once Tables entries are in BF16
//
extern "C" {
    __attribute__((always_inline)) v16bfloat16 getSiluBf16_nonLUT(v16bfloat16 vInput){
        aie::vector<bfloat16, 16> x = vInput;
        const bfloat16 half = 0.5f;
        const bfloat16 one = 1.0f;
        aie::vector<bfloat16, 16> v_half = aie::broadcast<bfloat16, 16>(half);
        aie::vector<bfloat16, 16> v_one = aie::broadcast<bfloat16, 16>(one);
        aie::accum<accfloat, 16> x_mul_half =aie::mul(x, v_half);
        aie::vector<bfloat16, 16> tanh_x_half = aie::tanh<bfloat16>(x_mul_half.template to_vector<float>());

        aie::vector<bfloat16, 16> tanh_x_half_plus_one = aie::add(tanh_x_half, v_one);
        aie::vector<bfloat16, 16> tanh_x_half_plus_one_div_2 = aie::mul(tanh_x_half_plus_one, v_half);

        aie::vector<bfloat16, 16> res = aie::mul(x, tanh_x_half_plus_one_div_2);
        return res;
    }
    __attribute__((always_inline)) v16bfloat16 getGeluBf16_nonLUT(v16bfloat16 vInput) {
        aie::vector<bfloat16, 16> x = vInput;

        // Constants
        const bfloat16 k0_5 = 0.5f;
        const bfloat16 k1 = 1.0f;
        const bfloat16 sqrt_2_over_pi = 0.79788456f; // ≈ sqrt(2/π)
        const bfloat16 kBeta = 0.044715f;

        auto v05 = aie::broadcast<bfloat16, 16>(k0_5);
        auto v1 = aie::broadcast<bfloat16, 16>(k1);
        auto vs2opi = aie::broadcast<bfloat16, 16>(sqrt_2_over_pi);
        auto vBeta = aie::broadcast<bfloat16, 16>(kBeta);


        // Compute x^3
        aie::vector<bfloat16, 16> x2 = aie::mul(x, x);  // x^2
        aie::vector<bfloat16, 16> x3 = aie::mul(x, x2); // x^3

        // inner = sqrt(2/pi) * (x + 0.044715 * x^3)
        aie::vector<bfloat16, 16> x3_beta = aie::mul(x3, vBeta);
        aie::vector<bfloat16, 16> inner = aie::add(x, x3_beta);
        auto inner1 = aie::mul(inner, vs2opi);

        // tanh_out = tanh(inner)
        auto tanh_out = aie::tanh<bfloat16>(inner1.to_vector<float>());

        // result = 0.5 * x * (1 + tanh_out)
        aie::vector<bfloat16, 16> one_plus_tanh = aie::add(tanh_out, v1);
        // Multiply by x and 0.5
        aie::vector<bfloat16, 16> mul_v05 = aie::mul(v05, one_plus_tanh);
        auto result = aie::mul(x, mul_v05);    
        return result.template to_vector<bfloat16>();
    }
}

#endif //__LUT_BASED_OPS_H__