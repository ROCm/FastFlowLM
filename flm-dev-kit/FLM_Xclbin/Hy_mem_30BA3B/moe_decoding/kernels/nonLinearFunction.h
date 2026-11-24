// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __NONLINEAR_HPP__
#define __NONLINEAR_HPP__

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "zero.h"
#include <aie_api/aie.hpp>
#include "passThroughCommon.hpp"
#include "aie_kernel_utils.h"

#include "lut_based_ops.h"

template<int vec_len=16>
__attribute__((always_inline)) aie::vector<bfloat16, vec_len> exp2_bf16_noLUT(aie::vector<bfloat16, vec_len>&x){

  constexpr bfloat16 min_clamp = -87.0f;
  constexpr bfloat16 max_clamp = 88.0f;
  aie::vector<bfloat16, vec_len> min_value = aie::broadcast<bfloat16, vec_len>(min_clamp);
  aie::vector<bfloat16, vec_len> max_value = aie::broadcast<bfloat16, vec_len>(max_clamp); // max of float

  x = aie::clamp(x, min_value, max_value);
  aie::accum<accfloat, vec_len> Vec_acc = aie::mul(x, exp_scale);
  aie::vector<bfloat16, vec_len> result = aie::exp2<bfloat16>(Vec_acc.template to_vector<float>());
  return result;

}





template<int vec_size, typename output = bfloat16>
aie::vector<output, vec_size> getSigmoidBf16_nonLUT(aie::vector<bfloat16, vec_size> x){
  

  const bfloat16 half = 0.5f;
  const bfloat16 one = 1.0f;
  aie::vector<bfloat16, vec_size> v_half = aie::broadcast<bfloat16, vec_size>(half);
  aie::vector<bfloat16, vec_size> v_one = aie::broadcast<bfloat16, vec_size>(one);
  aie::accum<accfloat, vec_size> x_mul_half =aie::mul(x, v_half);
  aie::vector<bfloat16, vec_size> tanh_x_half = aie::tanh<bfloat16>(x_mul_half.template to_vector<float>());

  aie::vector<bfloat16, vec_size> tanh_x_half_plus_one = aie::add(tanh_x_half, v_one);
  aie::vector<output, vec_size> tanh_x_half_plus_one_div_2 = aie::mul(tanh_x_half_plus_one, v_half);

  return tanh_x_half_plus_one_div_2;


}


template<int vec_size>
aie::vector<bfloat16, vec_size> getSiluBf16_nonLUT(aie::vector<bfloat16, vec_size> x){
  

  aie::vector<bfloat16, vec_size> sigmoid_x = getSigmoidBf16_nonLUT<vec_size>(x);
  aie::vector<bfloat16, vec_size> res = aie::mul(x, sigmoid_x);
  return res;

}

template<unsigned int size>
void softmax(
    bfloat16* vec_in
){
    // apply softmax function on this input vector, assume vec_in is size 



    constexpr int vecLen = 32;
    static_assert(size % vecLen == 0);
    float exp_sum =0.0f;


    constexpr bf16 min_clamp = -87.0f;
    constexpr bf16 max_clamp = 88.0f;
    aie::vector<bf16, vecLen> min_value = aie::broadcast<bf16, vecLen>(min_clamp);
    aie::vector<bf16, vecLen> max_value = aie::broadcast<bf16, vecLen>(max_clamp);

    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_RANGE(size/vecLen,size/vecLen)
    for(int i = 0; i < size; i += vecLen){
        aie::vector<bf16, vecLen> v = aie::load_v<vecLen>(vec_in + i);
        // aie::vector<bf16, vecLen> v_clamped = aie::clamp(v, min_value, max_value);
        // aie::accum<accfloat, vecLen> exp_res_acc = getExpBf16(v_clamped);
        // aie::vector<bf16, vecLen> exp_res_bf16 = exp_res_acc.template to_vector<bfloat16>();
        aie::vector<bf16, vecLen>  exp_res_bf16= exp2_bf16_noLUT<vecLen>(v);
        aie::store_v(vec_in + i, exp_res_bf16);
        exp_sum += aie::reduce_add(exp_res_bf16);
    }
    bf16 inv_exp_sum = 1.0f / exp_sum;

    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_RANGE(size/vecLen,size/vecLen)
    for(int i = 0; i < size; i += vecLen){
        aie::vector<bf16, vecLen> v = aie::load_v<vecLen>(vec_in + i);
        aie::accum<accfloat, vecLen> mul_res = aie::mul(inv_exp_sum, v);
        aie::store_v(vec_in + i, mul_res.template to_vector<bfloat16>());
    }



}

/// @brief Softmax over the K selected router logits, kept in float32.
///
/// Running the full NUM_EXPERTS-wide softmax before the top-K selection is both
/// wasted work and the router's largest avoidable error term.  Its denominator
/// is cancelled by the renormalization that follows the selection, so
///
///     softmax_all(i) / sum_{j in topK} softmax_all(j)
///         == exp(z_i - zmax) / sum_{j in topK} exp(z_j - zmax)
///
/// and evaluating the right-hand side over K values instead of 256 lets the
/// whole thing stay in float32.  Two roundings disappear with it.  The logits
/// no longer round to bfloat16 before the exponential -- a bfloat16 ulp at
/// |z| = 8 is 0.0625, which is a 6% relative error on that expert's weight and,
/// worse, collapses near-ties so the selection itself picks the wrong expert.
/// And subtracting zmax puts every argument in (-inf, 0], which removes the
/// [-87, 88] clamp: it was a hard clip, not a max subtraction, so a logit past
/// the top of that range used to saturate silently.
///
/// The one rounding left is aie::exp2's bfloat16 result -- the hardware op only
/// returns float32 on AIE_ARCH 22 with FP32 emulation -- and it lands on a
/// routing weight as a smooth 0.4% perturbation, not on the discrete choice.
///
/// @param out  K normalized routing weights, float32
/// @param z    the K selected logits, descending, so z[0] is the max
template<int K, int VLEN = 16>
void softmax_topk_f32(float* out, const float* z)
{
    static_assert(K <= VLEN);
    constexpr float log2_e = 1.44269504088896341f;

    // 2^-127 flushes to zero, so the VLEN-K padding lanes drop out of the sum
    aie::vector<float, VLEN> t = aie::broadcast<float, VLEN>(-127.0f);

    const float zmax = z[0];
    AIE_LOOP_UNROLL_FULL
    for (int i = 0; i < K; i++) {
        t.set((z[i] - zmax) * log2_e, i);
    }
    aie::vector<bfloat16, VLEN> e = aie::exp2<bfloat16>(t);

    float exp_sum = 0.0f;
    AIE_LOOP_UNROLL_FULL
    for (int i = 0; i < K; i++) {
        exp_sum += (float)e.get(i);
    }
    const float inv_exp_sum = 1.0f / exp_sum;

    AIE_LOOP_UNROLL_FULL
    for (int i = 0; i < K; i++) {
        out[i] = (float)e.get(i) * inv_exp_sum;
    }
}

#endif