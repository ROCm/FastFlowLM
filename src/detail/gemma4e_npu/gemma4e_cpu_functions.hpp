#ifndef __GEMMA4E_CPU_FUNCTIONS_HPP__
#define __GEMMA4E_CPU_FUNCTIONS_HPP__
#include <cmath>
#include <cstring>
#include <omp.h>
#include "typedef.hpp"
#include "buffer.hpp"
#include "models/gemma4e/gemma4e_npu.hpp"
#include "avx512_util.hpp"

/// @brief Host-side batched kernels used by the gemma4e prefill path.
/// @note  These used to be private members of gemma4e_npu::Impl; they are free
///        functions here so the prefill blocks can call them without a handle
///        on the model. Every one of them walks whole rows of a padded batch,
///        hence the L_offset_* arguments: row 0 of the buffer is the first
///        chunk-padding row, and the first live token sits at L_offset.
namespace gemma4e_cpu_func {

/// @brief Threads the batched kernels fan out over.
static constexpr int MAX_PREFILL_THREAD = 4;

/// @brief Rope frequency tables, one per attention flavour.
/// @note  Both are zero-padded up to the head dim the kernels stride over.
inline constexpr f32 inv_freq_global[] = {
    1.0000e+00, 9.4746e-01, 8.9769e-01, 8.5053e-01, 8.0584e-01, 7.6351e-01, 7.2339e-01, 6.8539e-01,
    6.4938e-01, 6.1527e-01, 5.8294e-01, 5.5232e-01, 5.2330e-01, 4.9581e-01, 4.6976e-01, 4.4508e-01,
    4.2170e-01, 3.9954e-01, 3.7855e-01, 3.5866e-01, 3.3982e-01, 3.2197e-01, 3.0505e-01, 2.8903e-01,
    2.7384e-01, 2.5946e-01, 2.4582e-01, 2.3291e-01, 2.2067e-01, 2.0908e-01, 1.9810e-01, 1.8769e-01,
    1.7783e-01, 1.6849e-01, 1.5963e-01, 1.5125e-01, 1.4330e-01, 1.3577e-01, 1.2864e-01, 1.2188e-01,
    1.1548e-01, 1.0941e-01, 1.0366e-01, 9.8217e-02, 9.3057e-02, 8.8168e-02, 8.3536e-02, 7.9148e-02,
    7.4989e-02, 7.1050e-02, 6.7317e-02, 6.3780e-02, 6.0430e-02, 5.7255e-02, 5.4247e-02, 5.1397e-02,
    4.8697e-02, 4.6138e-02, 4.3714e-02, 4.1418e-02, 3.9242e-02, 3.7180e-02, 3.5227e-02, 3.3376e-02,
    0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
    0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
    0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
    0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
    0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
    0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
    0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
    0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
    0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
    0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
    0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
    0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
    0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
    0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
    0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
    0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
    0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
    0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
    0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
    0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
    0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
    0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
    0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
    0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00
};
inline constexpr f32 inv_freq_swa[] = {
    1.0000e+00, 9.3057e-01, 8.6596e-01, 8.0584e-01, 7.4989e-01, 6.9783e-01, 6.4938e-01, 6.0430e-01,
    5.6234e-01, 5.2330e-01, 4.8697e-01, 4.5316e-01, 4.2170e-01, 3.9242e-01, 3.6517e-01, 3.3982e-01,
    3.1623e-01, 2.9427e-01, 2.7384e-01, 2.5483e-01, 2.3714e-01, 2.2067e-01, 2.0535e-01, 1.9110e-01,
    1.7783e-01, 1.6548e-01, 1.5399e-01, 1.4330e-01, 1.3335e-01, 1.2409e-01, 1.1548e-01, 1.0746e-01,
    1.0000e-01, 9.3057e-02, 8.6596e-02, 8.0584e-02, 7.4989e-02, 6.9783e-02, 6.4938e-02, 6.0430e-02,
    5.6234e-02, 5.2330e-02, 4.8697e-02, 4.5316e-02, 4.2170e-02, 3.9242e-02, 3.6517e-02, 3.3982e-02,
    3.1623e-02, 2.9427e-02, 2.7384e-02, 2.5483e-02, 2.3714e-02, 2.2067e-02, 2.0535e-02, 1.9110e-02,
    1.7783e-02, 1.6548e-02, 1.5399e-02, 1.4330e-02, 1.3335e-02, 1.2409e-02, 1.1548e-02, 1.0746e-02,
    1.0000e-02, 9.3057e-03, 8.6596e-03, 8.0584e-03, 7.4989e-03, 6.9783e-03, 6.4938e-03, 6.0430e-03,
    5.6234e-03, 5.2330e-03, 4.8697e-03, 4.5316e-03, 4.2170e-03, 3.9242e-03, 3.6517e-03, 3.3982e-03,
    3.1623e-03, 2.9427e-03, 2.7384e-03, 2.5483e-03, 2.3714e-03, 2.2067e-03, 2.0535e-03, 1.9110e-03,
    1.7783e-03, 1.6548e-03, 1.5399e-03, 1.4330e-03, 1.3335e-03, 1.2409e-03, 1.1548e-03, 1.0746e-03,
    1.0000e-03, 9.3057e-04, 8.6596e-04, 8.0584e-04, 7.4989e-04, 6.9783e-04, 6.4938e-04, 6.0430e-04,
    5.6234e-04, 5.2330e-04, 4.8697e-04, 4.5316e-04, 4.2170e-04, 3.9242e-04, 3.6517e-04, 3.3982e-04,
    3.1623e-04, 2.9427e-04, 2.7384e-04, 2.5483e-04, 2.3714e-04, 2.2067e-04, 2.0535e-04, 1.9110e-04,
    1.7783e-04, 1.6548e-04, 1.5399e-04, 1.4330e-04, 1.3335e-04, 1.2409e-04, 1.1548e-04, 1.0746e-04
};

/// @brief y[l] = rms_norm(x[l]) * w, over L rows of a D-wide batch.
/// @param w_ptr may be nullptr, which normalizes without a learned weight.
inline void _rms_norm_batch(bf16* y, bf16*  x, const bf16* w_ptr, int norm_D, int D, int L, int L_offset_dest, int L_offset_input){

    bf16* x_base = x + (size_t)L_offset_input * D;
    bf16* y_base = y + (size_t)L_offset_dest  * D;
    const int simd_width = 16;

    #pragma omp parallel for num_threads(MAX_PREFILL_THREAD) schedule(static)
    for (int l = 0; l < L; l++) {
        bf16* x_row = x_base + (size_t)l * D;
        bf16*       y_row = y_base + (size_t)l * D;
        for (int d = 0; d < D / norm_D; d++){
            // Step 1: sum of squares over norm_D elements
            bf16* x_row_inner = x_row + d * norm_D;
            bf16* y_row_inner = y_row + d * norm_D;
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

            // Step 2: y[d] = w[d] * x[d] * inv_rms_x
            __m512 inv_rms_vec = _mm512_set1_ps(inv_rms_x);
            i = 0;
            for (; i + simd_width <= norm_D; i += simd_width) {
                __m256i bf16_x = _mm256_loadu_si256((const __m256i*)(x_row_inner + i));
                if (w_ptr != nullptr){
                    __m256i bf16_w = _mm256_loadu_si256((const __m256i*)(w_ptr + i));
                    __m512 fp32_x  = bf16o_fp32_512(bf16_x);
                    __m512 fp32_w  = bf16o_fp32_512(bf16_w);
                    __m512 y_vec   = _mm512_mul_ps(_mm512_mul_ps(fp32_w, fp32_x), inv_rms_vec);
                    _mm256_storeu_si256((__m256i*)(y_row_inner + i), f32o_bf16_512(y_vec));
                }
                else{
                    __m512 fp32_x  = bf16o_fp32_512(bf16_x);
                    __m512 y_vec   = _mm512_mul_ps(fp32_x, inv_rms_vec);
                    _mm256_storeu_si256((__m256i*)(y_row_inner + i), f32o_bf16_512(y_vec));
                }
            }
            for (; i < norm_D; i++) {
                y_row_inner[i] = static_cast<bf16>(static_cast<f32>(w_ptr[i]) * static_cast<f32>(x_row_inner[i]) * inv_rms_x);
            }
        }
    }
}

/// @brief x[l] *= scale, over L rows of a D-wide batch.
inline void _elementwise_scale_batch(bf16* x, float scale, int D, int L, int L_offset){
    bf16*       dest_base = x + (size_t)L_offset * D;
    int         total_elements = L * D;
    int         i = 0;

    __m512 scale_vec = _mm512_set1_ps(scale);
    for (; i + 15 < total_elements; i += 16) {
        __m256i bf16_vals = _mm256_loadu_si256((const __m256i*)(dest_base + i));
        __m512 fp32_vals  = bf16o_fp32_512(bf16_vals);
        __m512 result     = _mm512_mul_ps(fp32_vals, scale_vec);
        _mm256_storeu_si256((__m256i*)(dest_base + i), f32o_bf16_512(result));
    }
    for (; i < total_elements; i++){
        dest_base[i] = static_cast<bf16>(static_cast<f32>(dest_base[i]) * scale);
    }
}

/// @brief Applies rope to q or k and the per-head rms norm, over L_effective rows.
/// @note  The frequency table is chosen by layer type: swa layers use a shorter
///        wavelength set than global ones. _DH is the head dimension of this
///        layer type, which the caller reads off the model descriptor.
inline void _rope_rms_batch(bf16* x, int D, int L_offset, int L_begin, int L_effective, bf16* rms_weight, gemma4e_layer_type_t layer_type, int _DH){
    const float* inv_freq = is_swa_layer(layer_type) ? inv_freq_swa : inv_freq_global;
    const int heads = D / _DH;
    bf16* x_base = x + (size_t)L_offset * D;
    #pragma omp parallel for num_threads(MAX_PREFILL_THREAD) schedule(static)
    for (int ll = 0; ll < L_effective; ll++){
        // Compute thread-local sin/cos values
        std::vector<float> local_cos(_DH / 2);
        std::vector<float> local_sin(_DH / 2);
        for (int j = 0; j < _DH / 2; j++){
            float angle = inv_freq[j] * (L_begin + ll);
            local_cos[j] = cosf(angle);
            local_sin[j] = sinf(angle);
        }

        bf16* x_token = x_base + ll * D;
        for (int h = 0; h < heads; h++){
            bf16* x_head = x_token + h * _DH;
            // inline rms_norm on single head
            {
                const int simd_width = 16;
                __m512 sum_xx_vec = _mm512_setzero_ps();
                int i = 0;
                for (; i + simd_width <= (int)_DH; i += simd_width) {
                    __m256i bf16_vals = _mm256_loadu_si256((const __m256i*)(x_head + i));
                    __m512 fp32_vals  = bf16o_fp32_512(bf16_vals);
                    sum_xx_vec = _mm512_fmadd_ps(fp32_vals, fp32_vals, sum_xx_vec);
                }
                f32 sum_xx = _mm512_reduce_add_ps(sum_xx_vec);
                for (; i < (int)_DH; i++) { f32 v = static_cast<f32>(x_head[i]); sum_xx += v * v; }
                f32 inv_rms = 1.0f / sqrtf(sum_xx / (f32)_DH + 1e-6f);
                __m512 inv_rms_vec = _mm512_set1_ps(inv_rms);
                i = 0;
                for (; i + simd_width <= (int)_DH; i += simd_width) {
                    __m256i bf16_x = _mm256_loadu_si256((const __m256i*)(x_head + i));
                    __m256i bf16_w = _mm256_loadu_si256((const __m256i*)(rms_weight + i));
                    __m512 y_vec = _mm512_mul_ps(_mm512_mul_ps(bf16o_fp32_512(bf16_x), bf16o_fp32_512(bf16_w)), inv_rms_vec);
                    _mm256_storeu_si256((__m256i*)(x_head + i), f32o_bf16_512(y_vec));
                }
                for (; i < (int)_DH; i++) {
                    x_head[i] = static_cast<bf16>(static_cast<f32>(rms_weight[i]) * static_cast<f32>(x_head[i]) * inv_rms);
                }
            }
            // apply rope rotation
            bf16* x_left  = x_head;
            bf16* x_right = x_head + _DH / 2;
            int i = 0, simd = 16;
            for (; i + simd <= _DH / 2; i += simd) {
                __m256i left_bf16  = _mm256_loadu_si256((__m256i*)(x_left  + i));
                __m256i right_bf16 = _mm256_loadu_si256((__m256i*)(x_right + i));
                __m512  Lv = bf16o_fp32_512(left_bf16);
                __m512  Rv = bf16o_fp32_512(right_bf16);
                __m512 C = _mm512_loadu_ps(local_cos.data() + i);
                __m512 S = _mm512_loadu_ps(local_sin.data() + i);
                __m512 newL = _mm512_sub_ps(_mm512_mul_ps(Lv, C), _mm512_mul_ps(Rv, S));
                __m512 newR = _mm512_add_ps(_mm512_mul_ps(Lv, S), _mm512_mul_ps(Rv, C));
                _mm256_storeu_si256((__m256i*)(x_left  + i), f32o_bf16_512(newL));
                _mm256_storeu_si256((__m256i*)(x_right + i), f32o_bf16_512(newR));
            }
        }
    }
}

/// @brief dest = x + residual, over L rows of a D-wide batch.
inline void _residual_add_batch(bf16* dest, bf16* x, bf16* residual, int D, int L, int L_offset_dest, int L_offset_x, int L_offset_r){
    bf16*       dest_base = dest     + (size_t)L_offset_dest * D;
    const bf16* src_base  = x        + (size_t)L_offset_x    * D;
    const bf16* res_base  = residual + (size_t)L_offset_r    * D;

    const int simd_width = 16;

    #pragma omp parallel for num_threads(MAX_PREFILL_THREAD) schedule(static)
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

/// @brief dest = a * b, over L rows of a D-wide batch.
inline void _elementwise_mul_batch(bf16* dest, bf16* a,bf16* b, int D, int L, int L_offset_dest, int L_offset_a, int L_offset_b){

    bf16*       dest_base = dest + (size_t)L_offset_dest * D;
    const bf16* a_base    = a    + (size_t)L_offset_a    * D;
    const bf16* b_base    = b    + (size_t)L_offset_b    * D;

    const int simd_width = 16;
    const int unroll = simd_width * 4;

    #pragma omp parallel for num_threads(MAX_PREFILL_THREAD) schedule(static)
    for (int l = 0; l < L; l++) {
        bf16*       dest_row = dest_base + (size_t)l * D;
        const bf16* a_row    = a_base    + (size_t)l * D;
        const bf16* b_row    = b_base    + (size_t)l * D;

        int i = 0;
        // 4-way unrolled loop for maximum throughput
        for (; i + unroll <= D; i += unroll){
            __m256i bf16_vals_a0 = _mm256_loadu_si256((const __m256i*)(a_row + i));
            __m256i bf16_vals_a1 = _mm256_loadu_si256((const __m256i*)(a_row + i + simd_width));
            __m256i bf16_vals_a2 = _mm256_loadu_si256((const __m256i*)(a_row + i + simd_width * 2));
            __m256i bf16_vals_a3 = _mm256_loadu_si256((const __m256i*)(a_row + i + simd_width * 3));

            __m256i bf16_vals_b0 = _mm256_loadu_si256((const __m256i*)(b_row + i));
            __m256i bf16_vals_b1 = _mm256_loadu_si256((const __m256i*)(b_row + i + simd_width));
            __m256i bf16_vals_b2 = _mm256_loadu_si256((const __m256i*)(b_row + i + simd_width * 2));
            __m256i bf16_vals_b3 = _mm256_loadu_si256((const __m256i*)(b_row + i + simd_width * 3));

            __m512 fp32_vals_a0 = bf16o_fp32_512(bf16_vals_a0);
            __m512 fp32_vals_a1 = bf16o_fp32_512(bf16_vals_a1);
            __m512 fp32_vals_a2 = bf16o_fp32_512(bf16_vals_a2);
            __m512 fp32_vals_a3 = bf16o_fp32_512(bf16_vals_a3);

            __m512 fp32_vals_b0 = bf16o_fp32_512(bf16_vals_b0);
            __m512 fp32_vals_b1 = bf16o_fp32_512(bf16_vals_b1);
            __m512 fp32_vals_b2 = bf16o_fp32_512(bf16_vals_b2);
            __m512 fp32_vals_b3 = bf16o_fp32_512(bf16_vals_b3);

            __m512 result_vec0 = _mm512_mul_ps(fp32_vals_a0, fp32_vals_b0);
            __m512 result_vec1 = _mm512_mul_ps(fp32_vals_a1, fp32_vals_b1);
            __m512 result_vec2 = _mm512_mul_ps(fp32_vals_a2, fp32_vals_b2);
            __m512 result_vec3 = _mm512_mul_ps(fp32_vals_a3, fp32_vals_b3);

            _mm256_storeu_si256((__m256i*)(dest_row + i), f32o_bf16_512(result_vec0));
            _mm256_storeu_si256((__m256i*)(dest_row + i + simd_width), f32o_bf16_512(result_vec1));
            _mm256_storeu_si256((__m256i*)(dest_row + i + simd_width * 2), f32o_bf16_512(result_vec2));
            _mm256_storeu_si256((__m256i*)(dest_row + i + simd_width * 3), f32o_bf16_512(result_vec3));
        }

        // Handle remaining SIMD-width chunks
        for (; i + simd_width <= D; i += simd_width){
            __m256i bf16_vals_a = _mm256_loadu_si256((const __m256i*)(a_row + i));
            __m256i bf16_vals_b = _mm256_loadu_si256((const __m256i*)(b_row + i));
            __m512 result_vec = _mm512_mul_ps(bf16o_fp32_512(bf16_vals_a), bf16o_fp32_512(bf16_vals_b));
            _mm256_storeu_si256((__m256i*)(dest_row + i), f32o_bf16_512(result_vec));
        }

        // Scalar tail
        for (; i < D; i++){
            dest_row[i] = static_cast<bf16>(static_cast<float>(a_row[i]) * static_cast<float>(b_row[i]));
        }
    }
}

/// @brief dest = gate * b, where b is a PLI_D-wide slice of a D-wide row.
inline void _elementwise_mul_batch(bf16* dest, bf16* gate,bf16* b, int PLI_D, int D, int L, int L_offset_dest, int L_offset_a, int L_offset_b){

    bf16*       dest_base = dest + (size_t)L_offset_dest * PLI_D;
    const bf16* a_base    = gate    + (size_t)L_offset_a * PLI_D;
    const bf16* b_base    = b    + (size_t)L_offset_b    * D;

    const int simd_width = 16;
    const int unroll = simd_width * 4;

    #pragma omp parallel for num_threads(MAX_PREFILL_THREAD) schedule(static)
    for (int l = 0; l < L; l++) {
        bf16*       dest_row = dest_base + (size_t)l * PLI_D;
        const bf16* a_row    = a_base    + (size_t)l * PLI_D;
        const bf16* b_row    = b_base    + (size_t)l * D;

        int i = 0;
        // 4-way unrolled loop for maximum throughput
        for (; i + unroll <= PLI_D; i += unroll){
            __m256i bf16_vals_a0 = _mm256_loadu_si256((const __m256i*)(a_row + i));
            __m256i bf16_vals_a1 = _mm256_loadu_si256((const __m256i*)(a_row + i + simd_width));
            __m256i bf16_vals_a2 = _mm256_loadu_si256((const __m256i*)(a_row + i + simd_width * 2));
            __m256i bf16_vals_a3 = _mm256_loadu_si256((const __m256i*)(a_row + i + simd_width * 3));

            __m256i bf16_vals_b0 = _mm256_loadu_si256((const __m256i*)(b_row + i));
            __m256i bf16_vals_b1 = _mm256_loadu_si256((const __m256i*)(b_row + i + simd_width));
            __m256i bf16_vals_b2 = _mm256_loadu_si256((const __m256i*)(b_row + i + simd_width * 2));
            __m256i bf16_vals_b3 = _mm256_loadu_si256((const __m256i*)(b_row + i + simd_width * 3));

            __m512 fp32_vals_a0 = bf16o_fp32_512(bf16_vals_a0);
            __m512 fp32_vals_a1 = bf16o_fp32_512(bf16_vals_a1);
            __m512 fp32_vals_a2 = bf16o_fp32_512(bf16_vals_a2);
            __m512 fp32_vals_a3 = bf16o_fp32_512(bf16_vals_a3);

            __m512 fp32_vals_b0 = bf16o_fp32_512(bf16_vals_b0);
            __m512 fp32_vals_b1 = bf16o_fp32_512(bf16_vals_b1);
            __m512 fp32_vals_b2 = bf16o_fp32_512(bf16_vals_b2);
            __m512 fp32_vals_b3 = bf16o_fp32_512(bf16_vals_b3);

            __m512 result_vec0 = _mm512_mul_ps(fp32_vals_a0, fp32_vals_b0);
            __m512 result_vec1 = _mm512_mul_ps(fp32_vals_a1, fp32_vals_b1);
            __m512 result_vec2 = _mm512_mul_ps(fp32_vals_a2, fp32_vals_b2);
            __m512 result_vec3 = _mm512_mul_ps(fp32_vals_a3, fp32_vals_b3);

            _mm256_storeu_si256((__m256i*)(dest_row + i), f32o_bf16_512(result_vec0));
            _mm256_storeu_si256((__m256i*)(dest_row + i + simd_width), f32o_bf16_512(result_vec1));
            _mm256_storeu_si256((__m256i*)(dest_row + i + simd_width * 2), f32o_bf16_512(result_vec2));
            _mm256_storeu_si256((__m256i*)(dest_row + i + simd_width * 3), f32o_bf16_512(result_vec3));
        }

        // Handle remaining SIMD-width chunks
        for (; i + simd_width <= PLI_D; i += simd_width){
            __m256i bf16_vals_a = _mm256_loadu_si256((const __m256i*)(a_row + i));
            __m256i bf16_vals_b = _mm256_loadu_si256((const __m256i*)(b_row + i));
            __m512 result_vec = _mm512_mul_ps(bf16o_fp32_512(bf16_vals_a), bf16o_fp32_512(bf16_vals_b));
            _mm256_storeu_si256((__m256i*)(dest_row + i), f32o_bf16_512(result_vec));
        }

        // Scalar tail
        for (; i < PLI_D; i++){
            dest_row[i] = static_cast<bf16>(static_cast<float>(a_row[i]) * static_cast<float>(b_row[i]));
        }
    }
}

} // namespace gemma4e_cpu_func

#endif // __GEMMA4E_CPU_FUNCTIONS_HPP__
