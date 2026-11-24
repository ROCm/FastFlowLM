// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file common/cpu_simd.hpp
/// \brief The fp32 <-> bf16 marshalling around every dispatch, vectorised.
///
/// `qwen3_5vl_npu/cpu_functions.hpp`'s idiom, narrowed to the five loop shapes
/// module 1 actually runs at the host/device boundary. Everything here is an
/// elementwise pass over a contiguous range: load 16 lanes, convert, store.
///
/// ## Why this exists
///
/// The host half of module 1 speaks fp32 and the device speaks bf16, so every
/// hand-off narrows or widens. Those loops were ~2.9 s of an 11.5 s L = 1024
/// window, and they were slow for one reason: `biovault::bfloat16_t(float)` is
/// a `std::fpclassify` switch (biovault_bfloat16.h:124). It is branchy, so gcc
/// cannot vectorise any loop containing it at any `-O`, and the whole loop runs
/// one element at a time at ~0.6 ns each. Measured on one `[512][17408]` tile:
///
/// | | scalar x1 | scalar x32 | **simd x1** |
/// |---|---:|---:|---:|
/// | f32 -> bf16 (`f32_to_bf16`)        | 5.58 ms | 6.39 ms | **1.47 ms** |
/// | bf16 -> f32 + resid (`bf16_to_f32_add`) | 3.04 ms | 6.18 ms | **1.99 ms** |
/// | bf16 * bf16 -> bf16 (`bf16_mul_to_bf16`) | 8.78 ms | 6.20 ms | **1.06 ms** |
///
/// Note the third column against the second: **one vectorised core beats all
/// 32 scalar ones**, by 4-8x. Threading was the free part of this; vectorising
/// is the rest of it, and it is the part that does not depend on the OpenMP
/// wait policy the library cannot set (see `q38npu::host_par_min()`).
///
/// ## The conversion is bit-identical, and that is checked
///
/// `_cvt_f32_bf16` is round-to-nearest-even -- the same expression as
/// `cpu_ref::f32_to_bf16` (cpu_ref_common.hpp:80) and as
/// `torch.Tensor.bfloat16()` -- plus one masked fixup that reproduces the
/// constructor's `FP_ZERO`/`FP_SUBNORMAL` arm, which flushes to a signed zero
/// where plain RNE would round up to the smallest normal.
///
/// With that fixup it agrees with `biovault::bfloat16_t(float)` on **all 2^32
/// float bit patterns**, verified exhaustively. The NaN and infinity arms of
/// the constructor need no fixup here because this module builds with
/// `-ffast-math`: `-ffinite-math-only` folds those two `fpclassify` cases away,
/// so the scalar code takes the RNE path for them too. That is a dependency on
/// a build flag, so `tools/verify_cpu_simd.cpp` re-runs the exhaustive
/// comparison under whatever flags the module was actually built with.
///
/// Being bit-identical is the point: swapping these in changes no output, so
/// the hardware driver's cosines and the M9 argmax must not move, and if they
/// do it is a bug in this file and not a numerics question to be adjudicated.
///
/// ## Scope
///
/// Marshalling only. The arithmetic stages (`rms_norm`, `dense_matmul`, the
/// gates) are fp32 in and fp32 out with no bf16 constructor in them, so gcc
/// already vectorises those; they are not slow for this reason and are not
/// addressed here.
///
/// Source and destination must not overlap in any of these. Every one is safe
/// to call from inside a `#pragma omp parallel for` over rows -- they hold no
/// state.
#ifndef __QWEN3_8MTP_NPU_CPU_SIMD_HPP__
#define __QWEN3_8MTP_NPU_CPU_SIMD_HPP__

#include "typedef.hpp"          // bf16

#include <cstddef>

#if defined(__AVX512F__) && defined(__AVX512DQ__)
#include <immintrin.h>
#define QWEN3_8MTP_CPU_SIMD 1
#endif

namespace q38simd {

#ifdef QWEN3_8MTP_CPU_SIMD

/// 16 x fp32 -> 16 x bf16, bit-identical to `biovault::bfloat16_t(float)`.
static inline __m256i _cvt_f32_bf16(__m512 v) {
    const __m512i u     = _mm512_castps_si512(v);
    const __m512i trunc = _mm512_srli_epi32(u, 16);
    // round to nearest even: bias by 0x7FFF + (the bit we are about to keep)
    const __m512i lsb   = _mm512_and_si512(trunc, _mm512_set1_epi32(1));
    __m512i r = _mm512_srli_epi32(
        _mm512_add_epi32(_mm512_add_epi32(u, _mm512_set1_epi32(0x7FFF)), lsb), 16);
    // +0, -0 and denormals keep their sign bit and nothing else -- the ctor's
    // "denormal go to zero" arm. Without this, the largest denormal would round
    // up to the smallest normal bf16 and 2^24 inputs would disagree.
    const __mmask16 zero_or_denorm = _mm512_fpclass_ps_mask(v, 0x02 | 0x04 | 0x20);
    r = _mm512_mask_and_epi32(r, zero_or_denorm, trunc, _mm512_set1_epi32(0x8000));
    return _mm512_cvtepi32_epi16(r);
}

/// 16 x bf16 -> 16 x fp32. Exact: bf16 is the top half of the fp32 word.
static inline __m512 _cvt_bf16_f32(__m256i b) {
    return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(b), 16));
}

#define Q38_LOAD_BF16(p) _mm256_loadu_si256((const __m256i*)(p))
#define Q38_STORE_BF16(p, v) _mm256_storeu_si256((__m256i*)(p), (v))

#endif  // QWEN3_8MTP_CPU_SIMD

/// `dst[i] = (bf16)src[i]` -- x staging, `dequant_mm_op::run_tile`.
inline void f32_to_bf16(bf16* dst, const float* src, size_t n) {
    size_t i = 0;
#ifdef QWEN3_8MTP_CPU_SIMD
    for (; i + 16 <= n; i += 16)
        Q38_STORE_BF16(dst + i, _cvt_f32_bf16(_mm512_loadu_ps(src + i)));
#endif
    for (; i < n; i++) dst[i] = (bf16)src[i];
}

/// `dst[i] = (float)src[i]` -- the y widen and the q|k|v|z scatter.
inline void bf16_to_f32(float* dst, const bf16* src, size_t n) {
    size_t i = 0;
#ifdef QWEN3_8MTP_CPU_SIMD
    for (; i + 16 <= n; i += 16)
        _mm512_storeu_ps(dst + i, _cvt_bf16_f32(Q38_LOAD_BF16(src + i)));
#endif
    for (; i < n; i++) dst[i] = (float)src[i];
}

/// `dst[i] = (float)src[i] + resid[i]` -- the y widen with the residual folded
/// in. The sum is fp32, as in the scalar loop it replaces.
inline void bf16_to_f32_add(float* dst, const bf16* src, const float* resid,
                            size_t n) {
    size_t i = 0;
#ifdef QWEN3_8MTP_CPU_SIMD
    for (; i + 16 <= n; i += 16)
        _mm512_storeu_ps(dst + i, _mm512_add_ps(_cvt_bf16_f32(Q38_LOAD_BF16(src + i)),
                                                _mm512_loadu_ps(resid + i)));
#endif
    for (; i < n; i++) dst[i] = (float)src[i] + resid[i];
}

/// `dst[i] = (bf16)((float)a[i] * (float)b[i])` -- the SwiGLU un-weave writing
/// straight into down's x BO. One fp32 multiply, one rounding, as before.
inline void bf16_mul_to_bf16(bf16* dst, const bf16* a, const bf16* b, size_t n) {
    size_t i = 0;
#ifdef QWEN3_8MTP_CPU_SIMD
    for (; i + 16 <= n; i += 16)
        Q38_STORE_BF16(dst + i, _cvt_f32_bf16(
            _mm512_mul_ps(_cvt_bf16_f32(Q38_LOAD_BF16(a + i)),
                          _cvt_bf16_f32(Q38_LOAD_BF16(b + i)))));
#endif
    for (; i < n; i++) dst[i] = (bf16)((float)a[i] * (float)b[i]);
}

/// `dst[i] = (float)a[i] * (float)b[i]` -- the same un-weave for the fp32
/// boundary `up_gate_swiglu` keeps for the hardware driver to score.
inline void bf16_mul_to_f32(float* dst, const bf16* a, const bf16* b, size_t n) {
    size_t i = 0;
#ifdef QWEN3_8MTP_CPU_SIMD
    for (; i + 16 <= n; i += 16)
        _mm512_storeu_ps(dst + i,
            _mm512_mul_ps(_cvt_bf16_f32(Q38_LOAD_BF16(a + i)),
                          _cvt_bf16_f32(Q38_LOAD_BF16(b + i))));
#endif
    for (; i < n; i++) dst[i] = (float)a[i] * (float)b[i];
}

/// Index of the largest element, ties to the lowest index -- the rule
/// `cpu_ref::argmax` and `npu_lm_head::argmax` both state, and the rule greedy
/// decoding needs, since MTP acceptance is an integer compare against what the
/// draft head picked.
///
/// The scalar loop this replaces is not slow because of the compare; it is slow
/// because `best` feeds the *address* of the next load, so each iteration waits
/// a full load-to-use latency. Keeping 16 running maxima in a register breaks
/// that chain: 4x on one core, measured over the real 250k-token vocabulary.
///
/// The horizontal reduce has to break lane ties by index, not just by value --
/// two columns can hold the same maximum and only the earlier one is the
/// scalar's answer. `_CMP_GT_OQ` is false for NaN exactly as `>` is, so the two
/// also agree on a vocabulary that has one, which this module's `-ffast-math`
/// build says cannot happen anyway.
inline int argmax_f32(const float* v, int n) {
    if (n <= 0) return 0;
    int i = 0;
    float bestf = v[0];
    int   bestx = 0;
#ifdef QWEN3_8MTP_CPU_SIMD
    __m512  bestv = _mm512_set1_ps(v[0]);
    __m512i besti = _mm512_setzero_si512();
    __m512i idx   = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7,
                                      8, 9, 10, 11, 12, 13, 14, 15);
    const __m512i step = _mm512_set1_epi32(16);
    for (; i + 16 <= n; i += 16) {
        const __m512    x  = _mm512_loadu_ps(v + i);
        const __mmask16 gt = _mm512_cmp_ps_mask(x, bestv, _CMP_GT_OQ);
        bestv = _mm512_mask_blend_ps(gt, bestv, x);
        besti = _mm512_mask_blend_epi32(gt, besti, idx);
        idx   = _mm512_add_epi32(idx, step);
    }
    alignas(64) float bv[16];
    alignas(64) int   bi[16];
    _mm512_store_ps(bv, bestv);
    _mm512_store_si512((__m512i*)bi, besti);
    bestf = bv[0];
    bestx = bi[0];
    for (int l = 1; l < 16; l++)
        if (bv[l] > bestf || (bv[l] == bestf && bi[l] < bestx)) {
            bestf = bv[l];
            bestx = bi[l];
        }
#endif
    for (; i < n; i++)
        if (v[i] > bestf) { bestf = v[i]; bestx = i; }
    return bestx;
}

/// The same over a bf16 logit row -- `npu_lm_head`'s shape. Widening is exact
/// (bf16 is the top half of the fp32 word), so this is the fp32 answer on the
/// widened row, which is what the scalar loop computed one `(float)` cast at a
/// time.
inline int argmax_bf16(const bf16* v, int n) {
    if (n <= 0) return 0;
    int i = 0;
    float bestf = (float)v[0];
    int   bestx = 0;
#ifdef QWEN3_8MTP_CPU_SIMD
    __m512  bestv = _mm512_set1_ps(bestf);
    __m512i besti = _mm512_setzero_si512();
    __m512i idx   = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7,
                                      8, 9, 10, 11, 12, 13, 14, 15);
    const __m512i step = _mm512_set1_epi32(16);
    for (; i + 16 <= n; i += 16) {
        const __m512    x  = _cvt_bf16_f32(Q38_LOAD_BF16(v + i));
        const __mmask16 gt = _mm512_cmp_ps_mask(x, bestv, _CMP_GT_OQ);
        bestv = _mm512_mask_blend_ps(gt, bestv, x);
        besti = _mm512_mask_blend_epi32(gt, besti, idx);
        idx   = _mm512_add_epi32(idx, step);
    }
    alignas(64) float bv[16];
    alignas(64) int   bi[16];
    _mm512_store_ps(bv, bestv);
    _mm512_store_si512((__m512i*)bi, besti);
    bestf = bv[0];
    bestx = bi[0];
    for (int l = 1; l < 16; l++)
        if (bv[l] > bestf || (bv[l] == bestf && bi[l] < bestx)) {
            bestf = bv[l];
            bestx = bi[l];
        }
#endif
    for (; i < n; i++) {
        const float x = (float)v[i];
        if (x > bestf) { bestf = x; bestx = i; }
    }
    return bestx;
}

#ifdef QWEN3_8MTP_CPU_SIMD
#undef Q38_LOAD_BF16
#undef Q38_STORE_BF16
#endif

} // namespace q38simd

#endif  // __QWEN3_8MTP_NPU_CPU_SIMD_HPP__
