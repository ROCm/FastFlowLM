// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file qwen3_asr_audio_simd.hpp
/// \brief AVX-512 helpers of the audio encoder, 16 lanes: bf16 rounding, exp and the exact (erf) GELU.
#pragma once
#include <immintrin.h>

namespace qwen3_asr_audio_simd {

/// round to bf16 (nearest even), kept as fp32
inline __m512 round_bf16(__m512 v) {
    __m512i u = _mm512_castps_si512(v);
    u = _mm512_add_epi32(u, _mm512_add_epi32(_mm512_set1_epi32(0x7FFF), _mm512_and_si512(_mm512_srli_epi32(u, 16), _mm512_set1_epi32(1))));
    return _mm512_castsi512_ps(_mm512_and_si512(u, _mm512_set1_epi32((int)0xFFFF0000u)));
}

/// 16 bf16 bit patterns -> fp32
inline __m512 load_bf16(const uint16_t* p) {
    return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i*)p)), 16));
}

/// e^x for x in [-87, 88] (clamped outside), within 2 ulp of std::exp: Cody-Waite reduction, then a degree-7 polynomial
inline __m512 exp(__m512 x) {
    x = _mm512_min_ps(_mm512_max_ps(x, _mm512_set1_ps(-87.f)), _mm512_set1_ps(88.f));
    const __m512 n = _mm512_roundscale_ps(_mm512_mul_ps(x, _mm512_set1_ps(1.44269504088896341f)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m512 r = _mm512_fnmadd_ps(n, _mm512_set1_ps(0.693359375f), x);
    r = _mm512_fnmadd_ps(n, _mm512_set1_ps(-2.12194440e-4f), r);
    __m512 p = _mm512_fmadd_ps(r, _mm512_set1_ps(1.9875691500e-4f), _mm512_set1_ps(1.3981999507e-3f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(8.3334519073e-3f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(4.1665795894e-2f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.6666665459e-1f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(5.0000001201e-1f));
    p = _mm512_fmadd_ps(p, _mm512_mul_ps(r, r), _mm512_add_ps(r, _mm512_set1_ps(1.f)));
    return _mm512_scalef_ps(p, n);
}

/// 0.5 x (1 + erf(x / sqrt 2)); erf from a rational approximation (fp32 accuracy, its argument clamped to [-4, 4]). Rounded to
/// bf16 it equals the std::erf form but for a few inputs below -3.9, where the result is under 3e-4 and off by one bf16 step.
inline __m512 gelu_erf(__m512 x) {
    const __m512 z = _mm512_max_ps(_mm512_min_ps(_mm512_mul_ps(x, _mm512_set1_ps(0.70710678118654752440f)), _mm512_set1_ps(4.f)),
                                   _mm512_set1_ps(-4.f));
    const __m512 z2 = _mm512_mul_ps(z, z);
    __m512 p = _mm512_fmadd_ps(z2, _mm512_set1_ps(-2.72614225801306e-10f), _mm512_set1_ps(2.77068142495902e-08f));
    p = _mm512_fmadd_ps(z2, p, _mm512_set1_ps(-2.10102402082508e-06f));
    p = _mm512_fmadd_ps(z2, p, _mm512_set1_ps(-5.69250639462346e-05f));
    p = _mm512_fmadd_ps(z2, p, _mm512_set1_ps(-7.34990630326855e-04f));
    p = _mm512_fmadd_ps(z2, p, _mm512_set1_ps(-2.95459980854025e-03f));
    p = _mm512_fmadd_ps(z2, p, _mm512_set1_ps(-1.60960333262415e-02f));
    p = _mm512_mul_ps(z, p);
    __m512 q = _mm512_fmadd_ps(z2, _mm512_set1_ps(-1.45660718464996e-05f), _mm512_set1_ps(-2.13374055278905e-04f));
    q = _mm512_fmadd_ps(z2, q, _mm512_set1_ps(-1.68282697438203e-03f));
    q = _mm512_fmadd_ps(z2, q, _mm512_set1_ps(-7.37332916720468e-03f));
    q = _mm512_fmadd_ps(z2, q, _mm512_set1_ps(-1.42647390514189e-02f));
    return _mm512_mul_ps(_mm512_mul_ps(_mm512_set1_ps(0.5f), x), _mm512_add_ps(_mm512_set1_ps(1.f), _mm512_div_ps(p, q)));
}

}  // namespace qwen3_asr_audio_simd
