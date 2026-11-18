// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __NON_LUT_BASED_OPS_H__
#define __NON_LUT_BASED_OPS_H__
#include "aie_api/aie.hpp"

__attribute__((always_inline)) v16float getGelufloat16_nonLUT(v16float vInput) {
  aie::vector<float, 16> x = vInput;

    const float k0_5 = 0.5f;
    const float k1 = 1.0f;
    const float sqrt_2_over_pi = 0.79788456f; // ≈ sqrt(2/π)
    const float kBeta = 0.044715f;

    aie::vector<float, 16> v05 = aie::broadcast<float, 16>(k0_5);
    aie::vector<float, 16> v1 = aie::broadcast<float, 16>(k1);
    aie::vector<float, 16> vs2opi = aie::broadcast<float, 16>(sqrt_2_over_pi);
    aie::vector<float, 16> vBeta = aie::broadcast<float, 16>(kBeta);

    aie::accum<accfloat, 16> v1_acc; //TODO: optimize here
    v1_acc.from_vector(v1);


    aie::accum<accfloat, 16> x_acc;
    x_acc.from_vector(x);
    // Compute x^3
    aie::accum<accfloat, 16> x3_acc = mul_elem_16(x, x  );
    x3_acc = mul_elem_16(x3_acc.template to_vector<float>(), x);
    

    // // inner = sqrt(2/pi) * (x + 0.044715 * x^3)
    aie::accum<accfloat,16> x3_beta_acc = mul_elem_16(x3_acc.template to_vector<float>(), vBeta);
    aie::accum<accfloat, 16> inner = add( x_acc,  x3_beta_acc );
    aie::accum<accfloat, 16>  inner1 = mul_elem_16(inner.template to_vector<float>(), vs2opi);

    // // // tanh_out = tanh(inner)
    aie::vector<bfloat16, 16> tanh_out = aie::tanh<bfloat16>(inner1.to_vector<float>());
    aie::accum<accfloat, 16> tanh_out_acc;
    tanh_out_acc.from_vector(tanh_out);

    // // result = 0.5 * x * (1 + tanh_out)
    aie::accum<accfloat, 16> one_plus_tanh = add(tanh_out_acc, v1_acc);
    // // Multiply by x and 0.5
    aie::accum<accfloat, 16> mul_v05 = mul_elem_16(v05, one_plus_tanh.template to_vector<float>());
    // auto result = aie::mul(x, mul_v05);
    aie::accum<accfloat, 16> res = mul_elem_16(x, mul_v05.template to_vector<float>() );


  return res.template to_vector<float>();
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


#endif