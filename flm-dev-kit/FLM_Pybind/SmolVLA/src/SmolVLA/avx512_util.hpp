// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __AVX512__UTIL__HPP__
#define __AVX512__UTIL__HPP__

#include <immintrin.h>
#include "typedef.hpp"
#include <cmath>
#include <algorithm>
#include <cstdint>

// Fast, corrected AVX-512 exp approximation (single-precision).
// Notes:
//  - Input x is clamped to [-88, 88] to avoid overflow/underflow.
//  - Uses range reduction x = n*ln2 + r, where n is rounded to nearest int.
//  - Uses a degree-5 polynomial for exp(r) evaluated with Horner + FMAs.
//  - Constructs 2^n by writing the biased exponent field; the biased exponent
//    is clamped to [0,255] as a safety measure.
//
// This is an approximation (not fully IEEE-754 accurate for all cases).
inline __m512 _mm512_exp_ps_corrected(__m512 x) {
    // clamp x to a reasonable range to avoid overflow/underflow
    const __m512 max_val = _mm512_set1_ps(88.0f);
    const __m512 min_val = _mm512_set1_ps(-88.0f);
    x = _mm512_min_ps(x, max_val);
    x = _mm512_max_ps(x, min_val);

    // constants: 1/ln2 and split ln2 = ln2_hi + ln2_lo for extra precision
    const __m512 ln2_inv = _mm512_set1_ps(1.44269504088896341f);  // 1/ln(2)
    const __m512 ln2_hi  = _mm512_set1_ps(0.6931471824645996f);   // hi part
    const __m512 ln2_lo  = _mm512_set1_ps(1.9082149292705877e-10f);// lo part

    // compute fx = x * (1/ln2)
    __m512 fx = _mm512_mul_ps(x, ln2_inv);

    // round to nearest integer (using rounding intrinsic), storing integer-valued floats
    fx = _mm512_roundscale_ps(fx, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

    // convert to int32 (safe since fx holds integer values after rounding)
    __m512i emm0 = _mm512_cvttps_epi32(fx);

    // convert back to float for range-reduction arithmetic
    __m512 n_ps = _mm512_cvtepi32_ps(emm0);

    // r = x - n * ln2  (use fnmadd to compute c - a*b robustly)
    // first r1 = x - n*ln2_hi
    __m512 r = _mm512_fnmadd_ps(n_ps, ln2_hi, x);  // r = x - n*ln2_hi
    // then r = r - n*ln2_lo
    r = _mm512_fnmadd_ps(n_ps, ln2_lo, r);         // r = x - n*(ln2_hi + ln2_lo)

    // polynomial coefficients for exp(r) ~ 1 + r + r^2/2 + r^3/6 + r^4/24 + r^5/120
    const __m512 c5 = _mm512_set1_ps(0.008333333333333333f);  // 1/120
    const __m512 c4 = _mm512_set1_ps(0.041666666666666664f);  // 1/24
    const __m512 c3 = _mm512_set1_ps(0.16666666666666666f);   // 1/6
    const __m512 c2 = _mm512_set1_ps(0.5f);                   // 1/2
    const __m512 c1 = _mm512_set1_ps(1.0f);
    const __m512 one = _mm512_set1_ps(1.0f);

    // Horner evaluation using FMA: (((c5*r + c4)*r + c3)*r + c2)*r + c1 ; then final *r + 1
    __m512 y = _mm512_fmadd_ps(c5, r, c4);
    y = _mm512_fmadd_ps(y, r, c3);
    y = _mm512_fmadd_ps(y, r, c2);
    y = _mm512_fmadd_ps(y, r, c1);
    y = _mm512_fmadd_ps(y, r, one); // y now approximates exp(r)

    // Build 2^n by inserting biased exponent into float bits:
    // biased = n + 127
    __m512i biased = _mm512_add_epi32(emm0, _mm512_set1_epi32(127));

    // clamp biased exponent to [0,255] to avoid invalid bit patterns
    biased = _mm512_max_epi32(biased, _mm512_set1_epi32(0));
    biased = _mm512_min_epi32(biased, _mm512_set1_epi32(255));

    // shift into exponent position (bits 23..30) and reinterpret as float
    biased = _mm512_slli_epi32(biased, 23);
    __m512 pow2n = _mm512_castsi512_ps(biased);

    // final result: exp(x) ≈ exp(r) * 2^n
    return _mm512_mul_ps(y, pow2n);
}

#endif // __AVX512__UTIL__HPP__
