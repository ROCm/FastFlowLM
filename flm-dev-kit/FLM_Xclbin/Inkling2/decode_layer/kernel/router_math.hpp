// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#ifndef __ROUTER_MATH_HPP__
#define __ROUTER_MATH_HPP__

// fp32 sigmoid routing for Hy-Memory-30B-A3B:
//     scores  = sigmoid(logits)                      all experts
//     choose  = top K of (scores + bias)             the bias only picks experts
//     weights = picked scores / their sum * scaling
//
// AIE2P's exp2 returns bf16 (0.4% error), enough to swap two close experts, so
// the exponential here is built from fp32 vector ops (aie::mul on fp32 is exact,
// see bf16_limb_math.hpp). All 128 experts take a few hundred vector ops.
// The loops stay rolled (AIE_LOOP_NO_UNROLL): every fp32 multiply is 9 bf16
// MACs of code, and unrolled they overflowed the router core's 16 KB program memory.

#include <aie_api/aie.hpp>
#include "sort.hpp"
#include "aie_kernel_utils.h"

namespace router_math {

// 2^t for t <= 0, fp32, relative error ~1e-7.
// Trick: split t = n + f with n an integer. 2^f (|f| < 1) is a short polynomial,
// and 2^n * 2^f needs no multiply: add n to the float's exponent field.
template<int N>
aie::vector<float, N> exp2_f32(aie::vector<float, N> t)
{
    // keep the result a normal float (2^f >= 0.5, so 2^t >= 2^-126);
    // sigmoid is 0 or 1 to fp32 there anyway
    t = aie::max(t, -125.0f);

    // n = t rounded to an integer (nearest or down; either way |f| < 1)
    aie::vector<int32, N> n = aie::to_fixed<int32>(t, 0);
    aie::vector<float, N> f = aie::sub(t, aie::to_float<float>(n, 0));

    // 2^f = e^(f ln2): Taylor series to degree 8, c_k = ln2^k / k!.
    // Error <= ln2^9 / 9! = 1e-7 for |f| < 1. Horner form: p = (..(c8 f + c7) f ..) f + c0
    constexpr float c[9] = {1.0f,          0.69314718f,   0.24022651f,
                            0.055504109f,  0.0096181291f, 0.0013333558f,
                            1.5403530e-4f, 1.5252734e-5f, 1.3215487e-6f};
    aie::vector<float, N> p = aie::broadcast<float, N>(c[8]);
    AIE_LOOP_NO_UNROLL
    for (int k = 7; k >= 0; k--) {
        p = aie::add(aie::mul(p, f).template to_vector<float>(), c[k]);
    }

    // p * 2^n: add n to the exponent field (bits 23..30) of p, as integers.
    // Exact while the result stays normal (the clamp above). This saves an fp32
    // multiply, which is 9 bf16 MACs of program memory.
    aie::vector<int32, N> p_bits = p.template cast_to<int32>();
    return aie::add(p_bits, aie::upshift(n, 23)).template cast_to<float>();
}

// 1 / d for d in [1, 2], fp32.
// Start from the hardware bf16 reciprocal (error ~2^-8), then Newton:
// r <- r + r (1 - d r) squares the error each step: 2^-8 -> 2^-16 -> 2^-32.
template<int N>
aie::vector<float, N> inv_1_to_2(aie::vector<float, N> d)
{
    aie::accum<accfloat, N> d_acc(d);
    aie::vector<bfloat16, N> r_bf16 = aie::inv(d_acc.template to_vector<bfloat16>());
    aie::accum<accfloat, N> r_acc(r_bf16);
    aie::vector<float, N> r = r_acc.template to_vector<float>();
    AIE_LOOP_NO_UNROLL
    for (int step = 0; step < 2; step++) {
        aie::vector<float, N> err = aie::sub(aie::broadcast<float, N>(1.0f),
                                             aie::mul(d, r).template to_vector<float>());
        r = aie::add(r, aie::mul(r, err).template to_vector<float>());
    }
    return r;
}

// out[i] = sigmoid(in[i]), fp32, E a multiple of VLEN.
// Trick: only ever take exp of -|x|, so e = e^-|x| is in (0, 1] and cannot overflow:
//   sigmoid(|x|) = 1 / (1 + e),   sigmoid(-|x|) = e / (1 + e)
template<int E, int VLEN = 16>
void sigmoid_f32(float* out, const float* in)
{
    static_assert(E % VLEN == 0);
    constexpr float log2_e = 1.44269504088896341f;
    AIE_LOOP_NO_UNROLL
    for (int i = 0; i < E; i += VLEN) {
        aie::vector<float, VLEN> x = aie::load_v<VLEN>(in + i);
        aie::vector<float, VLEN> t = aie::mul(aie::abs(x), -log2_e).template to_vector<float>();
        aie::vector<float, VLEN> e = exp2_f32<VLEN>(t);                       // e^-|x|
        aie::vector<float, VLEN> r = inv_1_to_2<VLEN>(aie::add(e, 1.0f));    // 1 / (1 + e)
        aie::vector<float, VLEN> s_neg = aie::mul(e, r).template to_vector<float>();
        // select(a, b, m) takes b where m is set
        aie::store_v(out + i, aie::select(r, s_neg, aie::lt(x, 0.0f)));
    }
}

// The whole routing step.
//   logits:  E router logits; overwritten (used as scratch for scores + bias)
//   bias:    E fp32 expert biases
//   idx:     out, the K chosen experts, highest score + bias first
//   weights: out, their routing weights
template<int E, int K, int VLEN = 16>
void route_sigmoid_topk(float* weights, int32_t* idx, float* logits, const float* bias, float scaling)
{
    alignas(aie::vector_decl_align) float scores[E];
    sigmoid_f32<E, VLEN>(scores, logits);

    // choice = scores + bias, over the logits (not needed any more)
    for (int i = 0; i < E; i += VLEN) {
        aie::store_v(logits + i, aie::add(aie::load_v<VLEN>(scores + i), aie::load_v<VLEN>(bias + i)));
    }
    partial_select_largest<float, E, K>(logits, idx);

    // weights come from the plain scores, not score + bias
    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        sum = AIE_FADD(sum, scores[idx[k]]);
    }
    const float factor = scaling / sum;
    for (int k = 0; k < K; k++) {
        weights[k] = AIE_FMUL(scores[idx[k]], factor);
    }
}

} // namespace router_math

#endif // __ROUTER_MATH_HPP__
