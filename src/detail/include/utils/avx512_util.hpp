#pragma once
#include <immintrin.h>
#include "typedef.hpp"
#include <cmath>
#include <algorithm>
#include <cstdint>


/**
 * @brief Helper function to load 16 bfloat16 values and convert to __m512 (float).
 */
inline __m512 load_bfloat16_to_m512(const bf16* ptr) {
    // Load 16 bfloat16 values (32 bytes) into a __m256i
    __m256i bf16_data = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr));
    
    // Convert bfloat16 to float by shifting left 16 bits (bfloat16 is upper 16 bits of float)
    __m512i shifted = _mm512_cvtepu16_epi32(bf16_data);
    shifted = _mm512_slli_epi32(shifted, 16);
    
    return _mm512_castsi512_ps(shifted);
}

/**
 * @brief Helper function to store __m512 (float) as 16 bfloat16 values.
 * Uses truncation (no rounding).
 */
inline void store_m512_to_bfloat16(bf16* ptr, __m512 data) {
    // Convert float to bfloat16 by extracting upper 16 bits
    __m512i int_data = _mm512_castps_si512(data);
    __m512i shifted = _mm512_srli_epi32(int_data, 16);
    __m256i bf16_data = _mm512_cvtepi32_epi16(shifted);
    
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), bf16_data);
}

/**
 * @brief Helper function to store __m512 (float) as 16 bfloat16 values with rounding.
 * Uses round-to-nearest-even for better accuracy.
 */
inline void store_m512_to_bfloat16_rne(bf16* ptr, __m512 data) {
    // Convert float to bfloat16 with rounding to nearest even
    __m512i int_data = _mm512_castps_si512(data);
    
    // Add 0x7FFF for round-to-nearest-even
    __m512i rounding = _mm512_set1_epi32(0x7FFF);
    __m512i rounded = _mm512_add_epi32(int_data, rounding);
    
    // Shift right by 16 to get bf16 in lower 16 bits
    __m512i shifted = _mm512_srli_epi32(rounded, 16);
    
    // Pack to 16-bit values
    __m256i bf16_data = _mm512_cvtepi32_epi16(shifted);
    
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), bf16_data);
}



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

// Fast AVX-512 log approximation (single-precision).
// Input x must be strictly positive.
inline __m512 _mm512_log_ps_approx(__m512 x) {
    const __m512i inv_mant_mask = _mm512_set1_epi32(~0x7f800000);
    const __m512i min_norm_pos = _mm512_set1_epi32(0x00800000);
    const __m512i exponent_mask = _mm512_set1_epi32(0x7f800000);
    const __m512 one = _mm512_set1_ps(1.0f);
    
    // Extract exponent
    __m512i vx = _mm512_castps_si512(x);
    __m512i emm0 = _mm512_srli_epi32(vx, 23);
    emm0 = _mm512_sub_epi32(emm0, _mm512_set1_epi32(127));
    __m512 e = _mm512_cvtepi32_ps(emm0);
    
    // Extract mantissa and force exponent to 0 (which means range [1.0, 2.0))
    __m512i m_bits = _mm512_and_si512(vx, inv_mant_mask);
    m_bits = _mm512_or_si512(m_bits, _mm512_set1_epi32(0x3f800000));
    __m512 m = _mm512_castsi512_ps(m_bits);
    
    // Map m from [1, 2) to a symmetric range using p = (m - 1) / (m + 1)
    __m512 p1 = _mm512_sub_ps(m, one);
    __m512 p2 = _mm512_add_ps(m, one);
    __m512 p = _mm512_div_ps(p1, p2);
    __m512 p_sq = _mm512_mul_ps(p, p);
    
    // Evaluate Taylor series for log((1+p)/(1-p)) = 2 * (p + p^3/3 + p^5/5 + p^7/7)
    const __m512 c7 = _mm512_set1_ps(2.0f / 7.0f);
    const __m512 c5 = _mm512_set1_ps(2.0f / 5.0f);
    const __m512 c3 = _mm512_set1_ps(2.0f / 3.0f);
    const __m512 c1 = _mm512_set1_ps(2.0f);
    
    __m512 res = _mm512_fmadd_ps(c7, p_sq, c5);
    res = _mm512_fmadd_ps(res, p_sq, c3);
    res = _mm512_fmadd_ps(res, p_sq, c1);
    res = _mm512_mul_ps(res, p);
    
    // log(x) = res + e * ln(2)
    const __m512 ln2 = _mm512_set1_ps(0.6931471805599453f);
    return _mm512_fmadd_ps(e, ln2, res);
}


// AVX-512 GELU tanh-based, now using the corrected exp function
inline __m512 gelu_tanh_avx512_simd(__m512 gate_vec_fp32) {
   // ---- GELU(gate) with tanh approximation: 0.5 * gate * (1 + tanh(sqrt(2/pi) * (gate + 0.044715 * gate^3)))  ----
    const __m512 half = _mm512_set1_ps(0.5f);
    const __m512 one = _mm512_set1_ps(1.0f);
    const __m512 sqrt_2_pi = _mm512_set1_ps(0.7978845608f); // sqrt(2/pi)
    const __m512 coeff = _mm512_set1_ps(0.044715f);
    
    // Compute gate^3
    __m512 gate_squared = _mm512_mul_ps(gate_vec_fp32, gate_vec_fp32);
    __m512 gate_cubed = _mm512_mul_ps(gate_squared, gate_vec_fp32);
    
    // Compute gate + 0.044715 * gate^3
    __m512 inner_term = _mm512_fmadd_ps(coeff, gate_cubed, gate_vec_fp32);
    
    // Compute sqrt(2/pi) * (gate + 0.044715 * gate^3)
    __m512 scaled_term = _mm512_mul_ps(sqrt_2_pi, inner_term);
    
    // Compute tanh using the corrected exp function: tanh(x) ≈ (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    __m512 exp_pos = _mm512_exp_ps_corrected(scaled_term);
    __m512 exp_neg = _mm512_exp_ps_corrected(_mm512_sub_ps(_mm512_setzero_ps(), scaled_term));
    
    __m512 numerator = _mm512_sub_ps(exp_pos, exp_neg);
    __m512 denominator = _mm512_add_ps(exp_pos, exp_neg);
    __m512 tanh_approx = _mm512_div_ps(numerator, denominator);
                
    // Compute 1 + tanh(...)
    __m512 one_plus_tanh = _mm512_add_ps(one, tanh_approx);
    
    // Compute 0.5 * gate * (1 + tanh(...))
    __m512 gelu = _mm512_mul_ps(half, _mm512_mul_ps(gate_vec_fp32, one_plus_tanh));
    return gelu;
}




// Vectorized gaussian function for 16 floats using AVX-512 exp approximation
inline __m512 gaussian_avx512(__m512 x, __m512 sigma) {
    const __m512 one = _mm512_set1_ps(1.0f);
    const __m512 two = _mm512_set1_ps(2.0f);
    const __m512 zero = _mm512_setzero_ps();

    // Check if sigma <= 0
    __mmask16 mask_zero_sigma = _mm512_cmp_ps_mask(sigma, zero, _CMP_LE_OQ);
    
    // Compute exp(-(x*x)/(2*sigma*sigma)) using fast AVX-512 approximation
    __m512 x_sq = _mm512_mul_ps(x, x);
    __m512 sigma_sq = _mm512_mul_ps(sigma, sigma);
    __m512 two_sigma_sq = _mm512_mul_ps(two, sigma_sq);
    
    // Compute -(x*x)/(2*sigma*sigma)
    __m512 neg_x_sq_over_2sigma_sq = _mm512_div_ps(_mm512_sub_ps(zero, x_sq), two_sigma_sq);
    
    // Apply fast exponential
    __m512 exp_result = _mm512_exp_ps_corrected(neg_x_sq_over_2sigma_sq);

    // Return 1.0 if sigma <= 0, otherwise exp result
    return _mm512_mask_blend_ps(mask_zero_sigma, exp_result, one);
}

// Fast conversion from uint8 to float with normalization
inline void convert_uint8_to_float_avx512(const uint8_t* src, float* dst, size_t count) {
    const size_t simd_count = count & ~15; // Process in chunks of 16
    
    for (size_t i = 0; i < simd_count; i += 16) {
        // Load 16 uint8 values
        __m128i u8_vec = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i));
        
        // Convert to 32-bit integers
        __m512i i32_vec = _mm512_cvtepu8_epi32(u8_vec);
        
        // Convert to float
        __m512 f32_vec = _mm512_cvtepi32_ps(i32_vec);
        
        // Store result
        _mm512_storeu_ps(dst + i, f32_vec);
    }
    
    // Handle remaining elements
    for (size_t i = simd_count; i < count; ++i) {
        dst[i] = static_cast<float>(src[i]);
    }
}

// Fast conversion from float to uint8 with clamping
inline void convert_float_to_uint8_avx512(const float* src, uint8_t* dst, size_t count) {
    const __m512 zero = _mm512_setzero_ps();
    const __m512 max_val = _mm512_set1_ps(255.0f);
    const size_t simd_count = count & ~15; // Process in chunks of 16
    
    for (size_t i = 0; i < simd_count; i += 16) {
        // Load 16 float values
        __m512 f32_vec = _mm512_loadu_ps(src + i);
        
        // Round to nearest integer
        f32_vec = _mm512_roundscale_ps(f32_vec, _MM_FROUND_TO_NEAREST_INT);
        
        // Clamp to [0, 255]
        f32_vec = _mm512_max_ps(f32_vec, zero);
        f32_vec = _mm512_min_ps(f32_vec, max_val);
        
        // Convert to 32-bit integers
        __m512i i32_vec = _mm512_cvtps_epi32(f32_vec);
        
        // Pack to uint8 (with saturation)
        __m128i u8_vec = _mm512_cvtusepi32_epi8(i32_vec);
        
        // Store result
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i), u8_vec);
    }
    
    // Handle remaining elements
    for (size_t i = simd_count; i < count; ++i) {
        dst[i] = static_cast<uint8_t>(std::clamp(std::round(src[i]), 0.0f, 255.0f));
    }
}
