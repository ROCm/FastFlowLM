// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#ifndef __CPU_FUNCTIONS_HPP__
#define __CPU_FUNCTIONS_HPP__

#include <type_traits>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <thread>
#ifdef _OPENMP
#include <omp.h>
#endif

/// \brief The one OpenMP team size every host kernel of the prefill uses.
/// \note  Capped at one thread per physical core (omp_get_max_threads(), at most hardware_concurrency()/2;
///        measured: SMT siblings add no FMA throughput, and with every hardware thread claimed any other
///        load stalls each region's barrier for a scheduler tick; FLM_PREFILL_THREADS overrides).
///        It is ONE size on purpose. libgomp keeps a pool of threads sized to the last team it ran, and a
///        smaller team makes the surplus threads exit: the elementwise stages used to ask for 4 and the
///        delta rule / conv for 12, so the pool was torn down and rebuilt (new, unbound threads, cold
///        caches) around every other region -- the 1032-token scan ran at its 4-thread speed.
inline int host_team_threads() {
    static const int nt = [] {
#ifdef _OPENMP
        int n = omp_get_max_threads();
#else
        int n = 1;
#endif
        const int cores = (int)std::thread::hardware_concurrency() / 2;
        if (cores > 0 && n > cores) n = cores;
        if (const char* env = std::getenv("FLM_PREFILL_THREADS")) {
            int v = std::atoi(env);
            if (v > 0) n = v;
        }
        return n < 1 ? 1 : n;
    }();
    return nt;
}

namespace cpu_func {

    /// \brief fp32 -> bf16, round to nearest even, 16 lanes.
    /// \note  typedef.hpp's f32o_bf16_512 *truncates*. Truncation is biased toward zero
    ///        by 2^-9 on average, and every residual add and norm here stores through it, so
    ///        the bias compounds down the stack: through 24 layers it cost the host path
    ///        cosine 0.9977 against an fp32 reference on the same weights, against 0.9999+
    ///        rounded. Matches the scalar static_cast<bf16> (biovault_bfloat16.h), which
    ///        already rounds. Inputs here are finite, so there is no NaN special case.
    static inline __m256i f32_bf16_rne_512(__m512 fp32_vals) {
        const __m512i bits = _mm512_castps_si512(fp32_vals);
        const __m512i lsb  = _mm512_and_si512(_mm512_srli_epi32(bits, 16), _mm512_set1_epi32(1));
        const __m512i rnd  = _mm512_add_epi32(bits, _mm512_add_epi32(lsb, _mm512_set1_epi32(0x7FFF)));
        return _mm512_cvtepi32_epi16(_mm512_srli_epi32(rnd, 16));
    }

    void _residual_add(bf16* dest_base, const bf16* src_base, const bf16* res_base, size_t length){

        const size_t simd_width = 16;
        const size_t simd_end = length - (length % simd_width);

        #pragma omp parallel for num_threads(host_team_threads()) schedule(static)
        for (int i = 0; i < simd_end; i += simd_width) {
            __m256i bf16_src = _mm256_loadu_si256((const __m256i*)(src_base + i));
            __m256i bf16_res = _mm256_loadu_si256((const __m256i*)(res_base + i));
            __m512 result    = _mm512_add_ps(bf16o_fp32_512(bf16_src), bf16o_fp32_512(bf16_res));
            _mm256_storeu_si256((__m256i*)(dest_base + i), f32_bf16_rne_512(result));
        }

        for (size_t i = simd_end; i < length; i++) {
            dest_base[i] = static_cast<bf16>(static_cast<float>(src_base[i]) + static_cast<float>(res_base[i]));
        }
    }

    void _residual_add(buffer<bf16>& dest, buffer<bf16>& src, buffer<bf16>& residual, size_t length){
        _residual_add(dest.data(), src.data(), residual.data(), length);
    }

    void _rms_norm(bf16* y, bf16*  x, const bf16*w_ptr, int D, int L, int L_offset_dest, int L_offset_input){

        const bf16* x_base = x         + (size_t)L_offset_input * D;
        bf16*       y_base = y         + (size_t)L_offset_dest  * D;
        const int simd_width = 16;

        #pragma omp parallel for num_threads(host_team_threads()) schedule(static)
        for (int l = 0; l < L; l++) {
            const bf16* x_row = x_base + (size_t)l * D;
            bf16*       y_row = y_base + (size_t)l * D;

            // Step 1: sum of squares over D elements
            __m512 sum_xx_vec = _mm512_setzero_ps();
            int i = 0;
            for (; i + simd_width <= D; i += simd_width) {
                __m256i bf16_vals = _mm256_loadu_si256((const __m256i*)(x_row + i));
                __m512 fp32_vals  = bf16o_fp32_512(bf16_vals);
                sum_xx_vec = _mm512_fmadd_ps(fp32_vals, fp32_vals, sum_xx_vec);
            }
            f32 sum_xx = _mm512_reduce_add_ps(sum_xx_vec);
            for (; i < D; i++) { f32 v = static_cast<f32>(x_row[i]); sum_xx += v * v; }

            f32 inv_rms_x = 1.0f / sqrtf(sum_xx / (f32)D + 1e-6f);

            // Step 2: y[d] = w[d] * x[d] * inv_rms_x
            __m512 inv_rms_vec = _mm512_set1_ps(inv_rms_x);
            i = 0;
            for (; i + simd_width <= D; i += simd_width) {
                __m256i bf16_x = _mm256_loadu_si256((const __m256i*)(x_row + i));
                __m256i bf16_w = _mm256_loadu_si256((const __m256i*)(w_ptr + i));
                __m512 fp32_x  = bf16o_fp32_512(bf16_x);
                __m512 fp32_w  = bf16o_fp32_512(bf16_w);
                __m512 y_vec   = _mm512_mul_ps(_mm512_mul_ps(fp32_w, fp32_x), inv_rms_vec);
                _mm256_storeu_si256((__m256i*)(y_row + i), f32_bf16_rne_512(y_vec));
            }
            for (; i < D; i++) {
                y_row[i] = static_cast<bf16>(static_cast<f32>(w_ptr[i]) * static_cast<f32>(x_row[i]) * inv_rms_x);
            }
        }
    }
    void _rms_norm(buffer<bf16>& y, buffer<bf16>& x, const bf16*w_ptr, int D, int L, int L_offset_dest, int L_offset_input){
        assert((size_t)(L_offset_input + L) * D <= x.size()         && "x buffer overflow in _rms_norm");
        assert((size_t)(L_offset_dest  + L) * D <= y.size()         && "y buffer overflow in _rms_norm");
        // assert((size_t)D                        <= rms_weight.size() && "rms_weight buffer too small in _rms_norm");

        _rms_norm( y.data(), x.data(), w_ptr, D, L, L_offset_dest, L_offset_input);
    }

    // Load 16 weights starting at wp into a __m512, converting from bf16 if needed.
    template <typename WT>
    static inline __m512 _load_w16(const WT* wp) {
        if constexpr (std::is_same_v<WT, float>) {
            return _mm512_loadu_ps(wp);
        } else {
            return bf16o_fp32_512(_mm256_loadu_si256((const __m256i*)wp));
        }
    }

    template <typename WT>
    void _cpu_mm(float* y, WT* w_ptr, bf16* x, int M, int K, int N){
        // y[m][n] = dot(x[m,:], weight[n,:])
        // x      : M x K, row-major, bf16
        // weight : N x K, row-major, WT (bf16 or float)
        // y      : M x N, row-major, float

        // Fast path: N == 32
        // Process 4 weight rows at a time → 8 groups cover all 32 outputs.
        if (N == 32) {
            #pragma omp parallel for num_threads(host_team_threads()) schedule(static)
            for (int m = 0; m < M; m++) {
                const bf16* x_row = x + (size_t)m * K;
                float*       y_row = y + (size_t)m * 32;

                for (int ng = 0; ng < 8; ng++) {
                    const int    n  = ng * 4;
                    const WT* w0 = w_ptr + (size_t)(n + 0) * K;
                    const WT* w1 = w_ptr + (size_t)(n + 1) * K;
                    const WT* w2 = w_ptr + (size_t)(n + 2) * K;
                    const WT* w3 = w_ptr + (size_t)(n + 3) * K;

                    __m512 acc0 = _mm512_setzero_ps();
                    __m512 acc1 = _mm512_setzero_ps();
                    __m512 acc2 = _mm512_setzero_ps();
                    __m512 acc3 = _mm512_setzero_ps();

                    int k = 0;
                    for (; k <= K - 16; k += 16) {
                        __m256i bf16_x = _mm256_loadu_si256((const __m256i*)(x_row + k));
                        __m512 xv = bf16o_fp32_512(bf16_x);
                        acc0 = _mm512_fmadd_ps(xv, _load_w16(w0 + k), acc0);
                        acc1 = _mm512_fmadd_ps(xv, _load_w16(w1 + k), acc1);
                        acc2 = _mm512_fmadd_ps(xv, _load_w16(w2 + k), acc2);
                        acc3 = _mm512_fmadd_ps(xv, _load_w16(w3 + k), acc3);
                    }
                    float s0 = _mm512_reduce_add_ps(acc0);
                    float s1 = _mm512_reduce_add_ps(acc1);
                    float s2 = _mm512_reduce_add_ps(acc2);
                    float s3 = _mm512_reduce_add_ps(acc3);
                    for (; k < K; k++) {
                        float xk = static_cast<float>(x_row[k]);
                        s0 += xk * static_cast<float>(w0[k]);
                        s1 += xk * static_cast<float>(w1[k]);
                        s2 += xk * static_cast<float>(w2[k]);
                        s3 += xk * static_cast<float>(w3[k]);
                    }
                    y_row[n + 0] = s0;
                    y_row[n + 1] = s1;
                    y_row[n + 2] = s2;
                    y_row[n + 3] = s3;
                }
            }
        } else {
            // General fallback: one dot product per (m, n), vectorised over K
            #pragma omp parallel for num_threads(host_team_threads()) schedule(static)
            for (int m = 0; m < M; m++) {
                const bf16* x_row = x + (size_t)m * K;
                float*       y_row = y + (size_t)m * N;

                for (int n = 0; n < N; n++) {
                    const WT* w_row = w_ptr + (size_t)n * K;
                    __m512 acc = _mm512_setzero_ps();
                    int k = 0;
                    for (; k <= K - 16; k += 16) {
                        __m256i bf16_x = _mm256_loadu_si256((const __m256i*)(x_row + k));
                        __m512 xv = bf16o_fp32_512(bf16_x);
                        __m512 wv = _load_w16(w_row + k);
                        acc = _mm512_fmadd_ps(xv, wv, acc);
                    }
                    float s = _mm512_reduce_add_ps(acc);
                    for (; k < K; k++)
                        s += static_cast<float>(x_row[k]) * static_cast<float>(w_row[k]);
                    y_row[n] = s;
                }
            }
        }
    }

    void _activate_a(float* y, float* A, float* dt_bias, int L, int D){
        const int simd_width = 16;
        #pragma omp parallel for num_threads(host_team_threads()) schedule(static)
        for (int l = 0; l < L; l++) {
            float* y_row = y + (size_t)l * D;
            int i = 0;
            for (; i + simd_width <= D; i += simd_width) {
                __m512 y_vec = _mm512_loadu_ps(y_row + i);
                __m512 dt_bias_vec = _mm512_loadu_ps(dt_bias + i);
                __m512 A_vec = _mm512_loadu_ps(A + i);

                __m512 sum_vec = _mm512_add_ps(y_vec, dt_bias_vec);
                __m512 exp_sum = _mm512_exp_ps_corrected(sum_vec);

                // softplus = log(1 + exp(x))
                __m512 one_vec = _mm512_set1_ps(1.0f);
                __m512 softplus_arg = _mm512_add_ps(one_vec, exp_sum);
                __m512 softplus_vec = _mm512_log_ps_approx(softplus_arg);

                __m512 mul_vec = _mm512_mul_ps(A_vec, softplus_vec);
                __m512 res_vec = _mm512_exp_ps_corrected(mul_vec);
                _mm512_storeu_ps(y_row + i, res_vec);
            }
            for (; i < D; i++) {
                float sum = y_row[i] + dt_bias[i];
                float softplus = logf(1.0f + expf(sum));
                y_row[i] = expf(A[i] * softplus);
            }
        }
    }

    void _activate_b(float* y, int L, int D){
        const int simd_width = 16;
        #pragma omp parallel for num_threads(host_team_threads()) schedule(static)
        for (int l = 0; l < L; l++) {
            float* y_row = y + (size_t)l * D;
            int i = 0;
            __m512 one_vec = _mm512_set1_ps(1.0f);
            __m512 zero_vec = _mm512_set1_ps(0.0f);
            for (; i + simd_width <= D; i += simd_width) {
                __m512 y_vec = _mm512_loadu_ps(y_row + i);
                __m512 neg_y = _mm512_sub_ps(zero_vec, y_vec);
                __m512 exp_neg_y = _mm512_exp_ps_corrected(neg_y);
                __m512 den_vec = _mm512_add_ps(one_vec, exp_neg_y);
                __m512 res_vec = _mm512_div_ps(one_vec, den_vec);
                _mm512_storeu_ps(y_row + i, res_vec);
            }
            for (; i < D; i++) {
                y_row[i] = 1.0f / (1.0f + expf(-y_row[i]));
            }
        }
    }

    void _gated_norm(bf16* x, bf16* gate, const bf16* w_ptr, int D, int L) {
        bf16* dest_ptr = x;
        bf16* x_base = x;
        bf16* gate_base = gate;
        const int simd_width = 16;

        // Iterate over L (rows)
        for (int l = 0; l < L; l++) {
            bf16* x_row = x_base + (size_t)l * D;
            bf16* dest_row = dest_ptr + (size_t)l * D;
            bf16* gate_row = gate_base + (size_t)l * D;

            // Step 1: RMS Norm calculation (sum of squares)
            __m512 sum_xx_vec = _mm512_setzero_ps();
            int i = 0;
            for (; i + simd_width <= D; i += simd_width) {
                __m256i bf16_vals = _mm256_loadu_si256((const __m256i*)(x_row + i));
                __m512 fp32_vals  = bf16o_fp32_512(bf16_vals);
                sum_xx_vec = _mm512_fmadd_ps(fp32_vals, fp32_vals, sum_xx_vec);
            }
            f32 sum_xx = _mm512_reduce_add_ps(sum_xx_vec);
            for (; i < D; i++) { f32 v = static_cast<f32>(x_row[i]); sum_xx += v * v; }

            f32 rms_x = sqrtf(sum_xx / (f32)D + 1e-6f);
            __m512 rms_x_vec = _mm512_set1_ps(rms_x);

            // Step 2: Normalize and gate
            i = 0;
            for (; i + simd_width <= D; i += simd_width) {
                // Load x
                __m256i bf16_x = _mm256_loadu_si256((const __m256i*)(x_row + i));
                __m512 fp32_x  = bf16o_fp32_512(bf16_x);

                // Load w
                __m256i bf16_w = _mm256_loadu_si256((const __m256i*)(w_ptr + i));
                __m512 fp32_w  = bf16o_fp32_512(bf16_w);

                // RMS Norm: y = x * w / rms
                __m512 y_vec = _mm512_div_ps(_mm512_mul_ps(fp32_w, fp32_x), rms_x_vec);

                // Store y back to x
                _mm256_storeu_si256((__m256i*)(x_row + i), f32_bf16_rne_512(y_vec));

                // Load gate
                __m256i bf16_gate = _mm256_loadu_si256((const __m256i*)(gate_row + i));
                __m512 fp32_gate = bf16o_fp32_512(bf16_gate);

                // Multiply: dest = y * gate
                __m512 dest_vec = _mm512_mul_ps(y_vec, fp32_gate);

                // Store dest
                _mm256_storeu_si256((__m256i*)(dest_row + i), f32_bf16_rne_512(dest_vec));
            }

            // Scalar fallback
            for (; i < D; i++) {
                float x_val = static_cast<float>(x_row[i]);
                float w_val = static_cast<float>(w_ptr[i]);
                float y_val = x_val * w_val / rms_x;
                x_row[i] = static_cast<bf16>(y_val);

                float gate_val = static_cast<float>(gate_row[i]);
                dest_row[i] = static_cast<bf16>(y_val * gate_val);
            }
        }
    }

    void _elementwise_mul(bf16* dst, bf16* src_0, bf16* src_1, int length){
        const int simd_width = 16;
        const int unroll = simd_width * 4;

        int i = 0;
        for (; i + unroll <= length; i += unroll){
            __m256i a_vals_0 = _mm256_loadu_si256((const __m256i*)(src_0 + i));
            __m256i a_vals_1 = _mm256_loadu_si256((const __m256i*)(src_0 + i + simd_width));
            __m256i a_vals_2 = _mm256_loadu_si256((const __m256i*)(src_0 + i + simd_width * 2));
            __m256i a_vals_3 = _mm256_loadu_si256((const __m256i*)(src_0 + i + simd_width * 3));

            __m256i b_vals_0 = _mm256_loadu_si256((const __m256i*)(src_1 + i));
            __m256i b_vals_1 = _mm256_loadu_si256((const __m256i*)(src_1 + i + simd_width));
            __m256i b_vals_2 = _mm256_loadu_si256((const __m256i*)(src_1 + i + simd_width * 2));
            __m256i b_vals_3 = _mm256_loadu_si256((const __m256i*)(src_1 + i + simd_width * 3));

            __m512 result_vec0 = _mm512_mul_ps(bf16o_fp32_512(a_vals_0), bf16o_fp32_512(b_vals_0));
            __m512 result_vec1 = _mm512_mul_ps(bf16o_fp32_512(a_vals_1), bf16o_fp32_512(b_vals_1));
            __m512 result_vec2 = _mm512_mul_ps(bf16o_fp32_512(a_vals_2), bf16o_fp32_512(b_vals_2));
            __m512 result_vec3 = _mm512_mul_ps(bf16o_fp32_512(a_vals_3), bf16o_fp32_512(b_vals_3));

            _mm256_storeu_si256((__m256i*)(dst + i), f32_bf16_rne_512(result_vec0));
            _mm256_storeu_si256((__m256i*)(dst + i + simd_width), f32_bf16_rne_512(result_vec1));
            _mm256_storeu_si256((__m256i*)(dst + i + simd_width * 2), f32_bf16_rne_512(result_vec2));
            _mm256_storeu_si256((__m256i*)(dst + i + simd_width * 3), f32_bf16_rne_512(result_vec3));
        }

        for (; i + simd_width <= length; i += simd_width){
            __m256i a_vals = _mm256_loadu_si256((const __m256i*)(src_0 + i));
            __m256i b_vals = _mm256_loadu_si256((const __m256i*)(src_1 + i));
            __m512 result_vec = _mm512_mul_ps(bf16o_fp32_512(a_vals), bf16o_fp32_512(b_vals));
            _mm256_storeu_si256((__m256i*)(dst + i), f32_bf16_rne_512(result_vec));
        }

        for (; i < length; i++){
            dst[i] = static_cast<bf16>(static_cast<float>(src_0[i]) * static_cast<float>(src_1[i]));
        }
    }

    // Fused SwiGLU where up and gate are interleaved along an inner axis.
    // up_gate layout: (L, intermediate_size / slice_size, 2, slice_size), the
    // "2" axis is 0 = up, 1 = gate. dst layout: (L, intermediate_size).
    // dst = up * silu(gate) = up * gate * sigmoid(gate).
    void _exp_glu(bf16* dst, bf16* up_gate, int L, int intermediate_size, int slice_size, int nthreads = host_team_threads()) {
        const int simd_width = 16;
        const int num_slices = intermediate_size / slice_size;
        const int slice_stride = 2 * slice_size; // one (up, gate) pair per slice

        const __m512 v_one = _mm512_set1_ps(1.0f);

        #pragma omp parallel for num_threads(nthreads) schedule(static)
        for (int l = 0; l < L; l++) {
            const bf16* ug_row = up_gate + (size_t)l * intermediate_size * 2;
            bf16*       dst_row = dst    + (size_t)l * intermediate_size;

            for (int s = 0; s < num_slices; s++) {
                const bf16* up_slice   = ug_row + (size_t)s * slice_stride;
                const bf16* gate_slice = up_slice + slice_size;
                bf16*       out_slice  = dst_row + (size_t)s * slice_size;

                int i = 0;
                for (; i + simd_width <= slice_size; i += simd_width) {
                    __m512 up_v   = bf16o_fp32_512(_mm256_loadu_si256((const __m256i*)(up_slice   + i)));
                    __m512 gate_v = bf16o_fp32_512(_mm256_loadu_si256((const __m256i*)(gate_slice + i)));

                    // silu(gate) = gate / (1 + exp(-gate))
                    __m512 neg_gate = _mm512_sub_ps(_mm512_setzero_ps(), gate_v);
                    __m512 denom    = _mm512_add_ps(v_one, _mm512_exp_ps_corrected(neg_gate));
                    __m512 silu     = _mm512_div_ps(gate_v, denom);

                    __m512 result = _mm512_mul_ps(up_v, silu);
                    _mm256_storeu_si256((__m256i*)(out_slice + i), f32_bf16_rne_512(result));
                }
                for (; i < slice_size; i++) {
                    float up_val   = static_cast<float>(up_slice[i]);
                    float gate_val = static_cast<float>(gate_slice[i]);
                    float silu     = gate_val / (1.0f + expf(-gate_val));
                    out_slice[i]   = static_cast<bf16>(up_val * silu);
                }
            }
        }
    }

    void _elementwise_mul(float* dst, bf16* src_0, float src_1, int length){
        const int simd_width = 16;
        const int unroll = simd_width * 4;
        const __m512 b_vals = _mm512_set1_ps(src_1);

        int i = 0;
        for (; i + unroll <= length; i += unroll){
            __m256i a_vals_0 = _mm256_loadu_si256((const __m256i*)(src_0 + i));
            __m256i a_vals_1 = _mm256_loadu_si256((const __m256i*)(src_0 + i + simd_width));
            __m256i a_vals_2 = _mm256_loadu_si256((const __m256i*)(src_0 + i + simd_width * 2));
            __m256i a_vals_3 = _mm256_loadu_si256((const __m256i*)(src_0 + i + simd_width * 3));

            __m512 result_vec0 = _mm512_mul_ps(bf16o_fp32_512(a_vals_0), b_vals);
            __m512 result_vec1 = _mm512_mul_ps(bf16o_fp32_512(a_vals_1), b_vals);
            __m512 result_vec2 = _mm512_mul_ps(bf16o_fp32_512(a_vals_2), b_vals);
            __m512 result_vec3 = _mm512_mul_ps(bf16o_fp32_512(a_vals_3), b_vals);

            _mm512_storeu_ps(dst + i, result_vec0);
            _mm512_storeu_ps(dst + i + simd_width, result_vec1);
            _mm512_storeu_ps(dst + i + simd_width * 2, result_vec2);
            _mm512_storeu_ps(dst + i + simd_width * 3, result_vec3);
        }

        for (; i + simd_width <= length; i += simd_width){
            __m256i a_vals = _mm256_loadu_si256((const __m256i*)(src_0 + i));
            __m512 result_vec = _mm512_mul_ps(bf16o_fp32_512(a_vals), b_vals);
            _mm512_storeu_ps(dst + i, result_vec);
        }

        for (; i < length; i++){
            dst[i] = static_cast<float>(src_0[i]) * src_1;
        }
    }

    void _sigmoid(bf16* y, const bf16* x, int length) {
        const int simd_width = 16;
        const __m512 v_one     = _mm512_set1_ps(1.0f);
        const __m512 v_neg_one = _mm512_set1_ps(-1.0f);

        int i = 0;
        for (; i + simd_width <= length; i += simd_width) {
            __m256i bf16_vals = _mm256_loadu_si256((const __m256i*)(x + i));
            __m512  fp32_x    = bf16o_fp32_512(bf16_vals);

            __m512 neg_x      = _mm512_mul_ps(fp32_x, v_neg_one);
            __m512 exp_neg_x  = _mm512_exp_ps_corrected(neg_x);
            __m512 denom      = _mm512_add_ps(v_one, exp_neg_x);
            __m512 result     = _mm512_div_ps(v_one, denom);

            _mm256_storeu_si256((__m256i*)(y + i), f32_bf16_rne_512(result));
        }
        for (; i < length; i++) {
            float val = static_cast<float>(x[i]);
            y[i]  = static_cast<bf16>(1.0f / (1.0f + expf(-val)));
        }
    }

    void _sigmoid(float* y, const bf16* x, int length) {
        const int simd_width = 16;
        const __m512 v_one     = _mm512_set1_ps(1.0f);
        const __m512 v_neg_one = _mm512_set1_ps(-1.0f);

        int i = 0;
        for (; i + simd_width <= length; i += simd_width) {
            __m256i bf16_vals = _mm256_loadu_si256((const __m256i*)(x + i));
            __m512  fp32_x    = bf16o_fp32_512(bf16_vals);

            __m512 neg_x      = _mm512_mul_ps(fp32_x, v_neg_one);
            __m512 exp_neg_x  = _mm512_exp_ps_corrected(neg_x);
            __m512 denom      = _mm512_add_ps(v_one, exp_neg_x);
            __m512 result     = _mm512_div_ps(v_one, denom);

            _mm512_storeu_ps(y + i, result);
        }
        for (; i < length; i++) {
            float val = static_cast<float>(x[i]);
            y[i] = 1.0f / (1.0f + expf(-val));
        }
    }

    void _sigmoid(float* y, const float* x, int length) {
        const int simd_width = 16;
        const __m512 v_one     = _mm512_set1_ps(1.0f);
        const __m512 v_neg_one = _mm512_set1_ps(-1.0f);

        int i = 0;
        for (; i + simd_width <= length; i += simd_width) {
            __m512 fp32_x    = _mm512_loadu_ps(x + i);

            __m512 neg_x     = _mm512_mul_ps(fp32_x, v_neg_one);
            __m512 exp_neg_x = _mm512_exp_ps_corrected(neg_x);
            __m512 denom     = _mm512_add_ps(v_one, exp_neg_x);
            __m512 result    = _mm512_div_ps(v_one, denom);

            _mm512_storeu_ps(y + i, result);
        }
        for (; i < length; i++)
            y[i] = 1.0f / (1.0f + expf(-x[i]));
    }

    // 16 bf16 rows (row stride Din) -> one 16-row group of the bfp16 tiles the 2B mm kernel reads:
    // [Din / 8][2 row blocks][72 B], written in order, two tiles (288 B) per step. A tile is 8 rows x 8 k, each row
    // one block of an exponent byte (largest exponent of the 8) and 8 two's complement mantissas, truncated like the
    // AIE2P conversion: q = signed 24-bit mantissa >> (17 + max_exp - exp), the shift capped at 31.
    void _bfp16_tile_group(uint8_t* group_tiles, const bf16* rows, int Din, int row_stride = 0){
        const size_t stride = row_stride > 0 ? row_stride : Din;
        const __m512i xor1 = _mm512_set_epi32(14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1);
        const __m512i xor2 = _mm512_set_epi32(13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2);
        const __m512i xor4 = _mm512_set_epi32(11, 10, 9, 8, 15, 14, 13, 12, 3, 2, 1, 0, 7, 6, 5, 4);
        const __m512i exp_mask = _mm512_set1_epi32(0xFF);
        const __m512i mantissa_mask = _mm512_set1_epi32(0x7FFFFF);
        const __m512i implicit_one = _mm512_set1_epi32(0x800000);
        const __m512i zero = _mm512_setzero_si512();

        for (int kb = 0; kb < Din / 8; kb += 2) {
            uint8_t* tile_pair = group_tiles + (size_t)kb * 144;
            for (int r = 0; r < 16; r++) {
                __m512i bits = _mm512_castps_si512(bf16o_fp32_512(_mm256_loadu_si256((const __m256i*)(rows + r * stride + kb * 8))));
                __m512i exp = _mm512_and_si512(_mm512_srli_epi32(bits, 23), exp_mask);
                __m512i max_exp = _mm512_max_epu32(exp, _mm512_permutexvar_epi32(xor1, exp));
                max_exp = _mm512_max_epu32(max_exp, _mm512_permutexvar_epi32(xor2, max_exp));
                max_exp = _mm512_max_epu32(max_exp, _mm512_permutexvar_epi32(xor4, max_exp));

                __m512i magnitude = _mm512_and_si512(bits, mantissa_mask);
                __mmask16 normal = _mm512_cmpneq_epi32_mask(exp, zero);
                magnitude = _mm512_mask_or_epi32(magnitude, normal, magnitude, implicit_one);
                __mmask16 negative = _mm512_cmplt_epi32_mask(bits, zero);
                __m512i mantissa = _mm512_mask_sub_epi32(magnitude, negative, zero, magnitude);
                __m512i shift = _mm512_min_epi32(_mm512_add_epi32(_mm512_sub_epi32(max_exp, exp), _mm512_set1_epi32(17)), _mm512_set1_epi32(31));
                __m128i q8 = _mm512_cvtepi32_epi8(_mm512_srav_epi32(mantissa, shift));

                uint8_t* block_0 = tile_pair + (r / 8) * 72 + (r % 8) * 9;
                uint8_t* block_1 = block_0 + 144;
                block_0[0] = (uint8_t)_mm_cvtsi128_si32(_mm512_castsi512_si128(max_exp));
                block_1[0] = (uint8_t)_mm_cvtsi128_si32(_mm512_extracti32x4_epi32(max_exp, 2));
                _mm_storel_epi64((__m128i*)(block_0 + 1), q8);
                _mm_storel_epi64((__m128i*)(block_1 + 1), _mm_srli_si128(q8, 8));
            }
        }
    }

    // 16 rows (keys, row_stride apart) x Din of V -> V^T bfp16 tiles [Din / 8][2 key blocks][72 B] for the 2B attention
    // kernel: a tile is 8 dh x 8 keys, each dh one block of its 8 keys (same rounding as _bfp16_tile_group).
    void _bfp16_vt_tile_group(uint8_t* group_tiles, const bf16* rows, int Din, int row_stride){
        const __m256i exp_mask = _mm256_set1_epi32(0xFF);
        const __m256i mantissa_mask = _mm256_set1_epi32(0x7FFFFF);
        const __m256i implicit_one = _mm256_set1_epi32(0x800000);
        const __m256i zero = _mm256_setzero_si256();

        for (int kg = 0; kg < 2; kg++) {
            for (int j = 0; j < Din / 8; j++) {
                __m256i bits[8];
                __m256i max_exp = zero;
                for (int k = 0; k < 8; k++) {
                    const __m128i v = _mm_loadu_si128((const __m128i*)(rows + (size_t)(kg * 8 + k) * row_stride + j * 8));
                    bits[k] = _mm256_slli_epi32(_mm256_cvtepu16_epi32(v), 16);
                    max_exp = _mm256_max_epu32(max_exp, _mm256_and_si256(_mm256_srli_epi32(bits[k], 23), exp_mask));
                }
                alignas(32) uint32_t exps[8];
                _mm256_store_si256((__m256i*)exps, max_exp);
                uint8_t* tile = group_tiles + ((size_t)j * 2 + kg) * 72;
                for (int r = 0; r < 8; r++) {
                    tile[r * 9] = (uint8_t)exps[r];
                }
                for (int k = 0; k < 8; k++) {
                    __m256i exp = _mm256_and_si256(_mm256_srli_epi32(bits[k], 23), exp_mask);
                    __m256i magnitude = _mm256_and_si256(bits[k], mantissa_mask);
                    magnitude = _mm256_mask_or_epi32(magnitude, _mm256_cmpneq_epi32_mask(exp, zero), magnitude, implicit_one);
                    __m256i mantissa = _mm256_mask_sub_epi32(magnitude, _mm256_cmplt_epi32_mask(bits[k], zero), zero, magnitude);
                    __m256i shift = _mm256_min_epi32(_mm256_add_epi32(_mm256_sub_epi32(max_exp, exp), _mm256_set1_epi32(17)), _mm256_set1_epi32(31));
                    alignas(16) uint8_t q8[16];
                    _mm_store_si128((__m128i*)q8, _mm256_cvtepi32_epi8(_mm256_srav_epi32(mantissa, shift)));
                    // key k of every dh row r: the 8 x 8 transpose done with plain stores (no AVX-512 VBMI needed)
                    for (int r = 0; r < 8; r++) {
                        tile[r * 9 + 1 + k] = q8[r];
                    }
                }
            }
        }
    }

    void _rope_rms_batch(buffer<bf16>& tensor, int D, std::vector<int>& input_idx, int L_effective, int num_heads, bf16* rms_weights, int RoPE_DIM, int DH, const f32* inv_freq){
        const int heads = num_heads;
        const int half_rope = RoPE_DIM / 2;

        #pragma omp parallel for num_threads(host_team_threads()) schedule(static)
        for (int ll = 0; ll < L_effective; ll++){
            // Compute thread-local sin/cos values
            std::vector<float> local_cos(RoPE_DIM / 2);
            std::vector<float> local_sin(RoPE_DIM / 2);
            int idxs[3] = { input_idx[ll * 3 + 0], input_idx[ll * 3 + 1], input_idx[ll * 3 + 2] };
            int idx_round_3 = 0;
            for (int j = 0; j < half_rope; j++){
                float angle = inv_freq[j] * idxs[idx_round_3];
                local_cos[j] = cosf(angle);
                local_sin[j] = sinf(angle);
                idx_round_3 = (idx_round_3 + 1) % 3;
            }

            bf16* x_token = tensor.data() + (size_t)ll * D;
            for (int h = 0; h < heads; h++){
                bf16* x_head = x_token + h * DH;
                // inline rms_norm on single head
                {
                    const int simd_width = 16;
                    __m512 sum_xx_vec = _mm512_setzero_ps();
                    int i = 0;
                    for (; i + simd_width <= (int)DH; i += simd_width) {
                        __m256i bf16_vals = _mm256_loadu_si256((const __m256i*)(x_head + i));
                        __m512 fp32_vals  = bf16o_fp32_512(bf16_vals);
                        sum_xx_vec = _mm512_fmadd_ps(fp32_vals, fp32_vals, sum_xx_vec);
                    }
                    f32 sum_xx = _mm512_reduce_add_ps(sum_xx_vec);
                    for (; i < (int)DH; i++) { f32 v = static_cast<f32>(x_head[i]); sum_xx += v * v; }
                    f32 inv_rms = 1.0f / sqrtf(sum_xx / (f32)DH + 1e-6f);
                    __m512 inv_rms_vec = _mm512_set1_ps(inv_rms);
                    i = 0;
                    for (; i + simd_width <= (int)DH; i += simd_width) {
                        __m256i bf16_x = _mm256_loadu_si256((const __m256i*)(x_head + i));
                        __m256i bf16_w = _mm256_loadu_si256((const __m256i*)(rms_weights + i));
                        __m512 y_vec = _mm512_mul_ps(_mm512_mul_ps(bf16o_fp32_512(bf16_x), bf16o_fp32_512(bf16_w)), inv_rms_vec);
                        _mm256_storeu_si256((__m256i*)(x_head + i), f32_bf16_rne_512(y_vec));
                    }
                    for (; i < (int)DH; i++) {
                        x_head[i] = static_cast<bf16>(static_cast<f32>(rms_weights[i]) * static_cast<f32>(x_head[i]) * inv_rms);
                    }
                }
                // apply rope rotation
                bf16* x_left  = x_head;
                bf16* x_right = x_head + half_rope;
                int i = 0, simd = 16;
                for (; i + simd <= half_rope; i += simd) {
                    __m256i left_bf16  = _mm256_loadu_si256((__m256i*)(x_left  + i));
                    __m256i right_bf16 = _mm256_loadu_si256((__m256i*)(x_right + i));
                    __m512  Lv = bf16o_fp32_512(left_bf16);
                    __m512  Rv = bf16o_fp32_512(right_bf16);
                    __m512 C = _mm512_loadu_ps(local_cos.data() + i);
                    __m512 S = _mm512_loadu_ps(local_sin.data() + i);
                    __m512 newL = _mm512_sub_ps(_mm512_mul_ps(Lv, C), _mm512_mul_ps(Rv, S));
                    __m512 newR = _mm512_add_ps(_mm512_mul_ps(Lv, S), _mm512_mul_ps(Rv, C));
                    _mm256_storeu_si256((__m256i*)(x_left  + i), f32_bf16_rne_512(newL));
                    _mm256_storeu_si256((__m256i*)(x_right + i), f32_bf16_rne_512(newR));
                }
            }
        }
    }
}

#endif
