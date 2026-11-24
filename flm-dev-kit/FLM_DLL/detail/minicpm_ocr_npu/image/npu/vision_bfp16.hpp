// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file image/npu/vision_bfp16.hpp
/// \brief The bfp16 pre-round of the `vision_mm` x operand, vectorised, and
///        fused with the fp32 -> bf16 narrow that always precedes it.
///
/// Ported verbatim from Qwen3.8-27B's tower (namespaces only); the numbers
/// below are that tower's measurements.
///
/// ## Why
///
/// `aie::mmul<8,8,8,bf16>` on npu2 floors both operands onto a shared-exponent
/// grid, one exponent per 8 contiguous elements, and it floors toward -inf
/// rather than rounding. A floor is
/// a biased error, so over a K-long dot product it accumulates linearly instead
/// of as sqrt(K); pre-rounding onto the same grid on the host converts the
/// hardware's floor into a no-op and buys the bias back. `vision_mm_op::bind`
/// does it once per weight, and `run` does it to x on every call.
///
/// Doing it to x is what makes it expensive. At the tower's operating point of
/// 1200 patches that is ~265M elements per image across 111 dispatches, and
/// the version this replaces was scalar inside the group: `VISION_PROFILE`
/// charged it **318 ms of a 1662 ms encode**, the single largest stage, with
/// the fp32 -> bf16 narrow that feeds it second at 278 ms.
///
/// Those two are one pass here. They were two full traversals of the same
/// half-gigabyte buffer -- narrow fp32 into the BO, then read the BO back,
/// round, and write it again -- and the pass is memory bound, so folding them
/// removes a read and a write as well as vectorising the arithmetic.
///
/// ## Bit-identity
///
/// Both halves reproduce the scalar expressions exactly rather than
/// approximating them, so swapping this in changes no output and the verify
/// drivers' cosines must not move:
///
///   * the narrow is `mco::simd::_cvt_f32_bf16`, which agrees with
///     `biovault::bfloat16_t(float)` on all 2^32 bit patterns (see
///     Qwen3.8's `common/cpu_simd.hpp`, where that is checked);
///   * the pre-round's own store is plain round-to-nearest-even, which is what
///     `mco::ref::f32_to_bf16` is. The two differ only on inputs that are zero
///     or subnormal, and a pre-rounded value is `k * 2^(E-6)` for an integer k
///     with `E >= -120`, so it is either exactly zero -- where the two agree,
///     both yielding the input's sign bit -- or normal. The distinction cannot
///     be reached, and one helper serves both halves.
///
/// Qwen3.8's `tools/verify_vision_bfp16.cpp` is the gate: it runs the scalar
/// and the vector paths over random and adversarial inputs and `memcmp`s them.
///
/// ## Group alignment
///
/// A group of 8 must not straddle a row, because each row is a separate dot
/// product and the hardware's groups start at the row. Every K the device sees
/// is a multiple of 384 (MiniCPM's patch K of 588 is zero-padded to 768 first), so rows are multiples of 16 and the vector body's 16-lane
/// step covers exactly two groups. The callers assert this rather than assume
/// it; an `n` that is not a multiple of 16 falls to the scalar tail, which is
/// correct but group-aligned only if `n` is a multiple of 8.
#ifndef __MINICPM_OCR_VISION_BFP16_HPP__
#define __MINICPM_OCR_VISION_BFP16_HPP__
#pragma once

#include "image/vision_common.hpp"
#include "image/vision_ops.hpp"
#include "image/vision_par.hpp"
#include "typedef.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace mco_npu {

// ==========================================================================
// scalar -- the reference, and the tail
// ==========================================================================

/// One group of `group` bf16 values, pre-rounded in place.
///
/// The block max is an integer max: |bf16| is monotonic in its bit pattern once
/// the sign bit is cleared, so there is no fabs and no float compare. The step
/// and its reciprocal are built by hand out of the exponent field rather than
/// by `ldexp`, and the limit by `truncf` rather than `log2` -- a first version
/// used libm and cost 4.0 ms of `patch_embed`'s 10 ms on its own.
inline void bfp16_preround_group_scalar(uint16_t* blk, int group) {
    uint16_t mxb = 0;
    for (int i = 0; i < group; i++) {
        const uint16_t m = (uint16_t)(blk[i] & 0x7FFFu);
        if (m > mxb) mxb = m;
    }
    const uint32_t be = (uint32_t)(mxb >> 7);   // the max's biased exponent
    // be < 7 covers an all-zero block and one small enough that the step would
    // be subnormal; 0xFF is inf/nan, which is not ours to reinterpret.
    if (be < 7u || be == 0xFFu) return;

    // step = 2^(E-6), E = be - 127, and its exact reciprocal.
    const uint32_t sb = (be -   6u) << 23;
    const uint32_t ib = (260u - be) << 23;
    float step, inv;
    std::memcpy(&step, &sb, 4);
    std::memcpy(&inv,  &ib, 4);

    // The clamp lands on the block's FLOORED max. Let rounding push a value
    // past it and the hardware derives a LARGER block exponent, which coarsens
    // the step for all 8 and loses more than the bias we came to remove.
    // mxb*inv is in [64,128), so truncf is the floor.
    const float lim = std::truncf(mco::ref::bf16_to_f32(mxb) * inv) * step;

    for (int i = 0; i < group; i++) {
        const float v = mco::ref::bf16_to_f32(blk[i]);
        float q = std::nearbyintf(v * inv) * step;
        if (q >  lim)        q =  lim;
        if (q < -lim - step) q = -lim - step;   // the grid is signed
        blk[i] = mco::ref::f32_to_bf16(q);
    }
}

/// `[i0, i1)` of a bf16 range, scalar, group-aligned at both ends.
inline void bfp16_preround_range_scalar(uint16_t* w, size_t n, int group) {
    const size_t blocks = n / (size_t)group;
    for (size_t b = 0; b < blocks; b++)
        bfp16_preround_group_scalar(w + b * (size_t)group, group);
}

// ==========================================================================
// vector
// ==========================================================================
#ifdef MINICPM_OCR_CPU_SIMD

/// 16 x fp32 -> the same 16 bf16 bit patterns, one per 32-bit lane.
///
/// `mco::simd::_cvt_f32_bf16` with the final pack removed: the pre-round needs to
/// keep working in 32-bit lanes, and packing to 16 and back costs two shuffles
/// per group for nothing.
static inline __m512i _bfp_bf16_bits(__m512 v) {
    const __m512i u     = _mm512_castps_si512(v);
    const __m512i trunc = _mm512_srli_epi32(u, 16);
    const __m512i lsb   = _mm512_and_si512(trunc, _mm512_set1_epi32(1));
    __m512i r = _mm512_srli_epi32(
        _mm512_add_epi32(_mm512_add_epi32(u, _mm512_set1_epi32(0x7FFF)), lsb),
        16);
    const __mmask16 zero_or_denorm =
        _mm512_fpclass_ps_mask(v, 0x02 | 0x04 | 0x20);
    r = _mm512_mask_and_epi32(r, zero_or_denorm, trunc,
                              _mm512_set1_epi32(0x8000));
    return r;
}

/// Two groups of 8, pre-rounded. In and out are bf16 bit patterns held one per
/// 32-bit lane.
///
/// The block max is a three-step butterfly inside each 8-lane half, so every
/// lane ends holding its own group's max and the rest of the body is branch
/// free and lane-wise. The two groups never interact: each permute index stays
/// within its half.
static inline __m512i _bfp_preround16(__m512i b32) {
    const __m512i IDX_1 = _mm512_setr_epi32(1, 0, 3, 2, 5, 4, 7, 6,
                                            9, 8, 11, 10, 13, 12, 15, 14);
    const __m512i IDX_2 = _mm512_setr_epi32(2, 3, 0, 1, 6, 7, 4, 5,
                                            10, 11, 8, 9, 14, 15, 12, 13);
    const __m512i IDX_4 = _mm512_setr_epi32(4, 5, 6, 7, 0, 1, 2, 3,
                                            12, 13, 14, 15, 8, 9, 10, 11);

    __m512i mx = _mm512_and_si512(b32, _mm512_set1_epi32(0x7FFF));
    mx = _mm512_max_epu32(mx, _mm512_permutexvar_epi32(IDX_1, mx));
    mx = _mm512_max_epu32(mx, _mm512_permutexvar_epi32(IDX_2, mx));
    mx = _mm512_max_epu32(mx, _mm512_permutexvar_epi32(IDX_4, mx));

    const __m512i be = _mm512_srli_epi32(mx, 7);
    // The scalar's two `continue` conditions, as a mask: an all-zero or tiny
    // block, and inf/nan. Those lanes keep their input untouched at the blend.
    const __mmask16 skip =
        _mm512_cmplt_epu32_mask(be, _mm512_set1_epi32(7)) |
        _mm512_cmpeq_epu32_mask(be, _mm512_set1_epi32(0xFF));

    const __m512 step = _mm512_castsi512_ps(_mm512_slli_epi32(
        _mm512_sub_epi32(be, _mm512_set1_epi32(6)), 23));
    const __m512 inv  = _mm512_castsi512_ps(_mm512_slli_epi32(
        _mm512_sub_epi32(_mm512_set1_epi32(260), be), 23));

    // 0x0B = truncate, suppress precision; 0x08 = nearest-even, ditto. Those
    // are `truncf` and `nearbyintf` under the default rounding mode.
    const __m512 fmax = _mm512_castsi512_ps(_mm512_slli_epi32(mx, 16));
    const __m512 lim  = _mm512_mul_ps(
        _mm512_roundscale_ps(_mm512_mul_ps(fmax, inv), 0x0B), step);

    const __m512 v = _mm512_castsi512_ps(_mm512_slli_epi32(b32, 16));
    __m512 q = _mm512_mul_ps(
        _mm512_roundscale_ps(_mm512_mul_ps(v, inv), 0x08), step);
    q = _mm512_min_ps(q, lim);
    q = _mm512_max_ps(q, _mm512_sub_ps(_mm512_sub_ps(_mm512_setzero_ps(), lim),
                                       step));

    return _mm512_mask_blend_epi32(skip, _bfp_bf16_bits(q), b32);
}

#endif  // MINICPM_OCR_CPU_SIMD

// ==========================================================================
// public
// ==========================================================================

/// `[0, n)` of a bf16 range, pre-rounded in place, serial. `group` other than 8
/// takes the scalar path -- the vector body's butterfly is 8-specific and the
/// hardware has no other group size.
inline void bfp16_preround_range(uint16_t* w, size_t n, int group = 8) {
    size_t i = 0;
#ifdef MINICPM_OCR_CPU_SIMD
    if (group == 8) {
        for (; i + 16 <= n; i += 16) {
            const __m512i b = _mm512_cvtepu16_epi32(
                _mm256_loadu_si256((const __m256i*)(w + i)));
            _mm256_storeu_si256((__m256i*)(w + i),
                                _mm512_cvtepi32_epi16(_bfp_preround16(b)));
        }
    }
#endif
    bfp16_preround_range_scalar(w + i, n - i, group);
}

/// The same, threaded. Every group is independent, so this is a plain split;
/// the chunk is a multiple of 16 so no split lands inside a group.
inline void preround_bfp16_inplace(uint16_t* w, size_t n, int group = 8) {
    constexpr size_t CHUNK  = 1u << 16;
    const size_t     stride = CHUNK - (CHUNK % (size_t)group);
    const int64_t    nchunk = (int64_t)((n + stride - 1) / stride);
    // The chunk index is signed on purpose: MSVC's /openmp is OpenMP 2.0, which
    // requires a signed integral loop variable (C3016). Same for the three
    // loops below, lm_head8_op's row loop and vision_glue's add_inplace().
#pragma omp parallel for schedule(static) num_threads(mco::vision::par_threads())
    for (int64_t c = 0; c < nchunk; c++) {
        const size_t i0 = (size_t)c * stride;
        bfp16_preround_range(w + i0, std::min(n - i0, stride), group);
    }
}

/// `dst[i] = (bf16)src[i]`, threaded. The plain narrow, for the operands that
/// are not an mmul's -- `vision_attn`'s q, k and v go to a kernel that does its
/// own thing with them, so they get no pre-round.
inline void narrow_bf16_par(bf16* dst, const float* src, size_t n) {
    constexpr size_t CHUNK = 1u << 16;
    const int64_t nchunk = (int64_t)((n + CHUNK - 1) / CHUNK);
#pragma omp parallel for schedule(static) num_threads(mco::vision::par_threads())
    for (int64_t c = 0; c < nchunk; c++) {
        const size_t i0 = (size_t)c * CHUNK;
        mco::simd::f32_to_bf16(dst + i0, src + i0, std::min(n - i0, CHUNK));
    }
}

/// `dst[i] = (float)src[i]`, threaded -- every dispatch's output.
inline void widen_f32_par(float* dst, const bf16* src, size_t n) {
    constexpr size_t CHUNK = 1u << 16;
    const int64_t nchunk = (int64_t)((n + CHUNK - 1) / CHUNK);
#pragma omp parallel for schedule(static) num_threads(mco::vision::par_threads())
    for (int64_t c = 0; c < nchunk; c++) {
        const size_t i0 = (size_t)c * CHUNK;
        mco::simd::bf16_to_f32(dst + i0, src + i0, std::min(n - i0, CHUNK));
    }
}

/// `dst[i] = gelu_tanh((float)src[i])`, threaded and vectorised -- the widen
/// of a `vision_mm` output with the tanh gelu applied on the way, which is how
/// the tower's fc1s run (see `vision_mm_op.hpp`, "Why the tanh gelu is on the
/// host"). ~250M elements per 720p image, so it cannot be a scalar `tanhf`.
///
/// gelu_tanh(x) = 0.5x(1 + tanh(z)) = x * sigmoid(2z) = x / (1 + exp(-2z)),
/// z = sqrt(2/pi)(x + 0.044715x^3): one exp per element. The exp is the usual
/// Cody-Waite split (2^n * e^r, |r| <= ln2/2) with a degree-6 polynomial,
/// relative error ~2e-7 -- three orders below the bf16 output it feeds. The
/// argument is clamped to [-87, 87] so a large |x| saturates to 0 or x rather
/// than producing an inf the -ffast-math build is allowed to mishandle.
inline void widen_gelu_tanh_par(float* dst, const bf16* src, size_t n) {
    constexpr size_t CHUNK = 1u << 16;
    const int64_t nchunk = (int64_t)((n + CHUNK - 1) / CHUNK);
#pragma omp parallel for schedule(static) num_threads(mco::vision::par_threads())
    for (int64_t c = 0; c < nchunk; c++) {
        const size_t i0 = (size_t)c * CHUNK;
        const size_t len = std::min(n - i0, CHUNK);
        size_t i = 0;
#ifdef MINICPM_OCR_CPU_SIMD
        const __m512 k0   = _mm512_set1_ps(-2.0f * 0.79788456080286535588f);
        const __m512 k1   = _mm512_set1_ps(0.044715f);
        const __m512 lo   = _mm512_set1_ps(-87.0f), hi = _mm512_set1_ps(87.0f);
        const __m512 log2e = _mm512_set1_ps(1.44269504088896341f);
        const __m512 ln2hi = _mm512_set1_ps(0.693359375f), ln2lo = _mm512_set1_ps(-2.12194440e-4f);
        const __m512 one  = _mm512_set1_ps(1.0f);
        for (; i + 16 <= len; i += 16) {
            const __m512 x = mco::simd::_cvt_bf16_f32(
                _mm256_loadu_si256((const __m256i*)(src + i0 + i)));
            // a = -2z
            const __m512 x2 = _mm512_mul_ps(x, x);
            __m512 a = _mm512_mul_ps(k0, _mm512_fmadd_ps(_mm512_mul_ps(k1, x2), x, x));
            a = _mm512_min_ps(_mm512_max_ps(a, lo), hi);
            // e^a = 2^n e^r
            const __m512 nf = _mm512_roundscale_ps(_mm512_mul_ps(a, log2e), 0x08);
            __m512 r = _mm512_fnmadd_ps(nf, ln2hi, a);
            r = _mm512_fnmadd_ps(nf, ln2lo, r);
            __m512 p = _mm512_set1_ps(1.9875691500e-4f);
            p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.3981999507e-3f));
            p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(8.3334519073e-3f));
            p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(4.1665795894e-2f));
            p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.6666665459e-1f));
            p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(5.0000001201e-1f));
            p = _mm512_fmadd_ps(_mm512_mul_ps(p, r), r, _mm512_add_ps(r, one));
            const __m512 e = _mm512_scalef_ps(p, nf);
            _mm512_storeu_ps(dst + i0 + i, _mm512_div_ps(x, _mm512_add_ps(one, e)));
        }
#endif
        for (; i < len; i++) dst[i0 + i] = mco::vision::gelu_tanh((float)src[i0 + i]);
    }
}

/// `dst[i] = preround((bf16)src[i])` -- the narrow and the pre-round in one
/// traversal. `n` should be a multiple of 8; anything past the last whole group
/// is narrowed but not rounded, exactly as the two separate passes did.
inline void narrow_preround_bfp16(bf16* dst, const float* src, size_t n) {
    uint16_t* d = reinterpret_cast<uint16_t*>(dst);
    constexpr size_t CHUNK = 1u << 16;          // a multiple of 16
    const int64_t nchunk = (int64_t)((n + CHUNK - 1) / CHUNK);
#pragma omp parallel for schedule(static) num_threads(mco::vision::par_threads())
    for (int64_t c = 0; c < nchunk; c++) {
        const size_t i0 = (size_t)c * CHUNK;
        const size_t len = std::min(n - i0, CHUNK);
        size_t i = 0;
#ifdef MINICPM_OCR_CPU_SIMD
        for (; i + 16 <= len; i += 16) {
            const __m512i b =
                _bfp_bf16_bits(_mm512_loadu_ps(src + i0 + i));
            _mm256_storeu_si256((__m256i*)(d + i0 + i),
                                _mm512_cvtepi32_epi16(_bfp_preround16(b)));
        }
#endif
        for (; i < len; i++) dst[i0 + i] = (bf16)src[i0 + i];
        // The scalar tail narrowed whole elements; round whatever whole groups
        // of 8 it left. Only reachable when n is not a multiple of 16.
        const size_t done = len - (len % 16);
        if (done < len)
            bfp16_preround_range_scalar(d + i0 + done, len - done, 8);
    }
}

}  // namespace mco_npu

#endif  // __MINICPM_OCR_VISION_BFP16_HPP__
