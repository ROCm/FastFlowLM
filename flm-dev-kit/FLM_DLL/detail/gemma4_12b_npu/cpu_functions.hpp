// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __GEMMA4_12B_CPU_FUNCTIONS_HPP__
#define __GEMMA4_12B_CPU_FUNCTIONS_HPP__

/// \file cpu_functions.hpp
/// \brief AVX-512 host kernels used by the gemma4-12b prefill graph: sub-vector
///        (per head) rms norm with an optional weight, the scalar-position rope,
///        the residual add, the per layer output scale and the two GLU flavours.
/// \note Every function is `inline` so that this copy can coexist with the other
///       models' cpu_functions.hpp in a single link.

#include <vector>
#include <algorithm>
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "avx512_util.hpp"

static constexpr int max_prefill_thread = 4;

namespace cpu_func {

    /// \brief RMS norm over sub-vectors of a row, with an optional weight.
    /// \param y      destination, L x D
    /// \param x      source, L x D (may alias y)
    /// \param w_ptr  norm_D weights, or nullptr for the unweighted variant
    ///               (SimpleRMSNorm(with_scale=False), used by value_norm)
    /// \param norm_D length of one normalised sub-vector; norm_D == D
    ///               normalises whole rows (the four layer norms), norm_D == DH
    ///               normalises each head separately (q/k/v norms)
    /// \param D      row stride
    /// \param L      number of rows to process
    /// \note eps is 1e-6, matching config.json rms_norm_eps of gemma4-12b, and
    ///       the gemma norms are plain `x * rsqrt(mean(x^2) + eps) * w`; there
    ///       is no (1 + w) as in gemma-2/3.
    inline void _rms_norm_batch(bf16* y, bf16* x, const bf16* w_ptr, int norm_D, int D, int L,
                                int L_offset_dest, int L_offset_input){
        bf16* x_base = x + (size_t)L_offset_input * D;
        bf16* y_base = y + (size_t)L_offset_dest  * D;
        const int simd_width = 16;

        #pragma omp parallel for num_threads(max_prefill_thread) schedule(static)
        for (int l = 0; l < L; l++) {
            bf16* x_row = x_base + (size_t)l * D;
            bf16* y_row = y_base + (size_t)l * D;
            for (int d = 0; d < D / norm_D; d++){
                bf16* x_row_inner = x_row + (size_t)d * norm_D;
                bf16* y_row_inner = y_row + (size_t)d * norm_D;

                // Step 1: sum of squares over norm_D elements
                __m512 sum_xx_vec = _mm512_setzero_ps();
                int i = 0;
                for (; i + simd_width <= norm_D; i += simd_width) {
                    __m256i bf16_vals = _mm256_loadu_si256((const __m256i*)(x_row_inner + i));
                    __m512 fp32_vals  = bf16o_fp32_512(bf16_vals);
                    sum_xx_vec = _mm512_fmadd_ps(fp32_vals, fp32_vals, sum_xx_vec);
                }
                f32 sum_xx = _mm512_reduce_add_ps(sum_xx_vec);
                for (; i < norm_D; i++) { f32 v = static_cast<f32>(x_row_inner[i]); sum_xx += v * v; }

                f32 inv_rms_x = 1.0f / sqrtf(sum_xx / (f32)norm_D + 1e-6f);

                // Step 2: y[d] = w[d] * x[d] * inv_rms_x  (w == 1 when w_ptr is null)
                __m512 inv_rms_vec = _mm512_set1_ps(inv_rms_x);
                i = 0;
                for (; i + simd_width <= norm_D; i += simd_width) {
                    __m512 fp32_x = bf16o_fp32_512(_mm256_loadu_si256((const __m256i*)(x_row_inner + i)));
                    __m512 y_vec;
                    if (w_ptr != nullptr){
                        __m512 fp32_w = bf16o_fp32_512(_mm256_loadu_si256((const __m256i*)(w_ptr + i)));
                        y_vec = _mm512_mul_ps(_mm512_mul_ps(fp32_w, fp32_x), inv_rms_vec);
                    }
                    else {
                        y_vec = _mm512_mul_ps(fp32_x, inv_rms_vec);
                    }
                    _mm256_storeu_si256((__m256i*)(y_row_inner + i), f32o_bf16_512(y_vec));
                }
                for (; i < norm_D; i++) {
                    f32 w = (w_ptr != nullptr) ? static_cast<f32>(w_ptr[i]) : 1.0f;
                    y_row_inner[i] = static_cast<bf16>(w * static_cast<f32>(x_row_inner[i]) * inv_rms_x);
                }
            }
        }
    }

    inline void _rms_norm_batch(buffer<bf16>& y, buffer<bf16>& x, const bf16* w_ptr, int norm_D, int D, int L,
                                int L_offset_dest, int L_offset_input){
        assert((size_t)(L_offset_input + L) * D <= x.size() && "x buffer overflow in _rms_norm_batch");
        assert((size_t)(L_offset_dest  + L) * D <= y.size() && "y buffer overflow in _rms_norm_batch");
        assert(D % norm_D == 0 && "D must be a multiple of norm_D in _rms_norm_batch");
        _rms_norm_batch(y.data(), x.data(), w_ptr, norm_D, D, L, L_offset_dest, L_offset_input);
    }

    /// \brief dest = x + residual, over L rows of D elements, with per operand row offsets.
    inline void _residual_add_batch(bf16* dest, bf16* x, bf16* residual, int D, int L,
                                    int L_offset_dest, int L_offset_x, int L_offset_r){
        bf16*       dest_base = dest     + (size_t)L_offset_dest * D;
        const bf16* src_base  = x        + (size_t)L_offset_x    * D;
        const bf16* res_base  = residual + (size_t)L_offset_r    * D;

        const int simd_width = 16;

        #pragma omp parallel for num_threads(max_prefill_thread) schedule(static)
        for (int l = 0; l < L; l++) {
            bf16*       dest_row = dest_base + (size_t)l * D;
            const bf16* src_row  = src_base  + (size_t)l * D;
            const bf16* res_row  = res_base  + (size_t)l * D;

            int i = 0;
            for (; i + simd_width <= D; i += simd_width) {
                __m256i bf16_src = _mm256_loadu_si256((const __m256i*)(src_row + i));
                __m256i bf16_res = _mm256_loadu_si256((const __m256i*)(res_row + i));
                __m512 result    = _mm512_add_ps(bf16o_fp32_512(bf16_src), bf16o_fp32_512(bf16_res));
                _mm256_storeu_si256((__m256i*)(dest_row + i), f32o_bf16_512(result));
            }
            for (; i < D; i++) {
                dest_row[i] = static_cast<bf16>(static_cast<float>(src_row[i]) + static_cast<float>(res_row[i]));
            }
        }
    }

    /// \brief x *= scale, in place, over L rows of D elements starting at L_offset.
    /// \note this is the per layer output scale of the gemma4 decoder layer
    ///       (`output = residual * layer_scalar`).
    inline void _elementwise_scale_batch(bf16* x, float scale, int D, int L, int L_offset){
        bf16* dest_base      = x + (size_t)L_offset * D;
        const int total      = L * D;
        const __m512 scale_v = _mm512_set1_ps(scale);

        int i = 0;
        for (; i + 15 < total; i += 16) {
            __m256i bf16_vals = _mm256_loadu_si256((const __m256i*)(dest_base + i));
            __m512 result     = _mm512_mul_ps(bf16o_fp32_512(bf16_vals), scale_v);
            _mm256_storeu_si256((__m256i*)(dest_base + i), f32o_bf16_512(result));
        }
        for (; i < total; i++){
            dest_base[i] = static_cast<bf16>(static_cast<f32>(dest_base[i]) * scale);
        }
    }

    /// \brief Per head rms norm followed by rope, scalar positions.
    /// \param x           L x D activations (q or k of one layer), modified in place
    /// \param D           row stride, _DQ or _DK
    /// \param L_offset    first row to touch
    /// \param L_begin     absolute position of row L_offset
    /// \param L_effective number of rows to touch
    /// \param rms_weight  _DH q_norm / k_norm weights
    /// \param DH          head dim of this layer kind (DH or SWA_DH)
    /// \param inv_freq    DH/2 frequencies of this layer kind, see
    ///                    gemma4_12b_desc::inv_freq(); entries past the rotated
    ///                    part are zero, which makes those lanes an identity
    ///                    rotation, so the full head can be rotated blindly.
    /// \note This is q/k only. v takes the unweighted per head norm
    ///       (_rms_norm_batch with w_ptr == nullptr) and no rope.
    inline void _rope_rms_batch(bf16* x, int D, int L_offset, int L_begin, int L_effective,
                                const bf16* rms_weight, int DH, const f32* inv_freq){
        const int heads = D / DH;
        const int half  = DH / 2;
        bf16* x_base = x + (size_t)L_offset * D;

        #pragma omp parallel for num_threads(max_prefill_thread) schedule(static)
        for (int ll = 0; ll < L_effective; ll++){
            // Compute thread-local sin/cos values
            std::vector<float> local_cos(half);
            std::vector<float> local_sin(half);
            for (int j = 0; j < half; j++){
                float angle = inv_freq[j] * (float)(L_begin + ll);
                local_cos[j] = cosf(angle);
                local_sin[j] = sinf(angle);
            }

            bf16* x_token = x_base + (size_t)ll * D;
            for (int h = 0; h < heads; h++){
                bf16* x_head = x_token + (size_t)h * DH;
                // inline rms_norm on single head
                {
                    const int simd_width = 16;
                    __m512 sum_xx_vec = _mm512_setzero_ps();
                    int i = 0;
                    for (; i + simd_width <= DH; i += simd_width) {
                        __m256i bf16_vals = _mm256_loadu_si256((const __m256i*)(x_head + i));
                        __m512 fp32_vals  = bf16o_fp32_512(bf16_vals);
                        sum_xx_vec = _mm512_fmadd_ps(fp32_vals, fp32_vals, sum_xx_vec);
                    }
                    f32 sum_xx = _mm512_reduce_add_ps(sum_xx_vec);
                    for (; i < DH; i++) { f32 v = static_cast<f32>(x_head[i]); sum_xx += v * v; }
                    f32 inv_rms = 1.0f / sqrtf(sum_xx / (f32)DH + 1e-6f);
                    __m512 inv_rms_vec = _mm512_set1_ps(inv_rms);
                    i = 0;
                    for (; i + simd_width <= DH; i += simd_width) {
                        __m256i bf16_x = _mm256_loadu_si256((const __m256i*)(x_head + i));
                        __m256i bf16_w = _mm256_loadu_si256((const __m256i*)(rms_weight + i));
                        __m512 y_vec = _mm512_mul_ps(_mm512_mul_ps(bf16o_fp32_512(bf16_x), bf16o_fp32_512(bf16_w)), inv_rms_vec);
                        _mm256_storeu_si256((__m256i*)(x_head + i), f32o_bf16_512(y_vec));
                    }
                    for (; i < DH; i++) {
                        x_head[i] = static_cast<bf16>(static_cast<f32>(rms_weight[i]) * static_cast<f32>(x_head[i]) * inv_rms);
                    }
                }
                // apply rope rotation: (x1, x2) -> (x1 cos - x2 sin, x1 sin + x2 cos)
                bf16* x_left  = x_head;
                bf16* x_right = x_head + half;
                int i = 0;
                const int simd = 16;
                for (; i + simd <= half; i += simd) {
                    __m512  Lv = bf16o_fp32_512(_mm256_loadu_si256((__m256i*)(x_left  + i)));
                    __m512  Rv = bf16o_fp32_512(_mm256_loadu_si256((__m256i*)(x_right + i)));
                    __m512 C = _mm512_loadu_ps(local_cos.data() + i);
                    __m512 S = _mm512_loadu_ps(local_sin.data() + i);
                    __m512 newL = _mm512_sub_ps(_mm512_mul_ps(Lv, C), _mm512_mul_ps(Rv, S));
                    __m512 newR = _mm512_add_ps(_mm512_mul_ps(Lv, S), _mm512_mul_ps(Rv, C));
                    _mm256_storeu_si256((__m256i*)(x_left  + i), f32o_bf16_512(newL));
                    _mm256_storeu_si256((__m256i*)(x_right + i), f32o_bf16_512(newR));
                }
                for (; i < half; i++) {
                    float lv = static_cast<float>(x_left[i]);
                    float rv = static_cast<float>(x_right[i]);
                    x_left[i]  = static_cast<bf16>(lv * local_cos[i] - rv * local_sin[i]);
                    x_right[i] = static_cast<bf16>(lv * local_sin[i] + rv * local_cos[i]);
                }
            }
        }
    }

    /// \brief Fused GeLU-tanh GLU where up and gate are interleaved along an inner axis.
    ///
    /// up_gate layout: (L, intermediate_size / slice_size, 2, slice_size), the
    /// "2" axis is 0 = up, 1 = gate -- which is the order
    /// gemma4_12b_desc::load_layer_weights() writes the up/gate weights in, with
    /// slice_size == gemma4_12b_desc::UP_GATE_OUTDIM_REORDER.
    /// dst layout: (L, intermediate_size), dst = gelu_tanh(gate) * up.
    inline void _gelu_glu(bf16* dst, bf16* up_gate, int L, int intermediate_size, int slice_size) {
        const int simd_width = 16;
        const int num_slices = intermediate_size / slice_size;
        const int slice_stride = 2 * slice_size; // one (up, gate) pair per slice

        #pragma omp parallel for num_threads(max_prefill_thread) schedule(static)
        for (int l = 0; l < L; l++) {
            const bf16* ug_row  = up_gate + (size_t)l * intermediate_size * 2;
            bf16*       dst_row = dst     + (size_t)l * intermediate_size;

            for (int s = 0; s < num_slices; s++) {
                const bf16* up_slice   = ug_row + (size_t)s * slice_stride;
                const bf16* gate_slice = up_slice + slice_size;
                bf16*       out_slice  = dst_row + (size_t)s * slice_size;

                int i = 0;
                for (; i + simd_width <= slice_size; i += simd_width) {
                    __m512 up_v   = bf16o_fp32_512(_mm256_loadu_si256((const __m256i*)(up_slice   + i)));
                    __m512 gate_v = bf16o_fp32_512(_mm256_loadu_si256((const __m256i*)(gate_slice + i)));
                    __m512 result = _mm512_mul_ps(up_v, gelu_tanh_avx512_simd(gate_v));
                    _mm256_storeu_si256((__m256i*)(out_slice + i), f32o_bf16_512(result));
                }
                for (; i < slice_size; i++) {
                    float up_val   = static_cast<float>(up_slice[i]);
                    float gate_val = static_cast<float>(gate_slice[i]);
                    // 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 x^3)))
                    float inner    = 0.7978845608f * (gate_val + 0.044715f * gate_val * gate_val * gate_val);
                    float gelu     = 0.5f * gate_val * (1.0f + tanhf(inner));
                    out_slice[i]   = static_cast<bf16>(up_val * gelu);
                }
            }
        }
    }

    /// \brief The multiply half of the GLU, for a gate that already carries its GeLU.
    ///
    /// up_gate layout: (L, 2 * intermediate_size), de-interleaved as [ up | gate ] --
    /// what gen_dequant_mm_512 scatters the fused up/gate weight region into when it is
    /// run once per slab parity, the gate run carrying DEQUANT_MM_ACT_GELU.
    /// dst layout: (L, intermediate_size), dst = gate * up.
    inline void _glu_mul(bf16* dst, bf16* up_gate, int L, int intermediate_size) {
        const int simd_width = 16;

        #pragma omp parallel for num_threads(max_prefill_thread) schedule(static)
        for (int l = 0; l < L; l++) {
            const bf16* up_row   = up_gate + (size_t)l * intermediate_size * 2;
            const bf16* gate_row = up_row + intermediate_size;
            bf16*       dst_row  = dst + (size_t)l * intermediate_size;

            int i = 0;
            for (; i + simd_width <= intermediate_size; i += simd_width) {
                __m512 up_v   = bf16o_fp32_512(_mm256_loadu_si256((const __m256i*)(up_row   + i)));
                __m512 gate_v = bf16o_fp32_512(_mm256_loadu_si256((const __m256i*)(gate_row + i)));
                _mm256_storeu_si256((__m256i*)(dst_row + i),
                                    f32o_bf16_512(_mm512_mul_ps(up_v, gate_v)));
            }
            for (; i < intermediate_size; i++) {
                dst_row[i] = static_cast<bf16>(static_cast<float>(up_row[i])
                                             * static_cast<float>(gate_row[i]));
            }
        }
    }
}

#endif
