// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __NANBEIGE_CPU_FUNCTIONS_HPP__
#define __NANBEIGE_CPU_FUNCTIONS_HPP__
#include <cmath>
#include "typedef.hpp"
#include "buffer.hpp"

/// @brief Host-side elementwise kernels used by the nanbeige prefill path.
/// @note  These used to be private members of nanbeige_npu::Impl; they are free
///        functions here so the prefill blocks can call them without a handle
///        on the model.
namespace nanbeige_cpu_func {

/// @brief dest = a * b, elementwise over the whole of a.
inline void _elementwise_mul(buffer<bf16>& dest, buffer<bf16>& a, buffer<bf16>& b) {
    int n = a.size();
#ifdef __AVX512F__
    const int simd_width = 16;
    for (int i = 0; i < n; i += simd_width) {
        __m256i bf16_vals_a = _mm256_loadu_si256((__m256i*)&a[i]);
        __m256i bf16_vals_b = _mm256_loadu_si256((__m256i*)&b[i]);
        __m512 fp32_vals_a = bf16o_fp32_512(bf16_vals_a);
        __m512 fp32_vals_b = bf16o_fp32_512(bf16_vals_b);
        __m512 result_vec = _mm512_mul_ps(fp32_vals_a, fp32_vals_b);
        __m256i result_bf16 = f32o_bf16_512(result_vec);
        _mm256_storeu_si256((__m256i*)&dest[i], result_bf16);
    }
#else
    const int simd_width = 8;
    for (int i = 0; i < n; i += simd_width) {
        __m128i bf16_vals_a = _mm_loadu_si128((__m128i*)&a[i]);
        __m128i bf16_vals_b = _mm_loadu_si128((__m128i*)&b[i]);
        __m256 fp32_vals_a = bf16o_fp32(bf16_vals_a);
        __m256 fp32_vals_b = bf16o_fp32(bf16_vals_b);
        __m256 result_vec = _mm256_mul_ps(fp32_vals_a, fp32_vals_b);
        __m128i result_bf16 = f32o_bf16(result_vec);
        _mm_storeu_si128((__m128i*)&dest[i], result_bf16);
    }
#endif
}

/// @brief y = rms_weight * x / rms(x), over one row.
inline void _rms_norm(buffer<bf16>& y, buffer<bf16>& x, buffer<bf16>& rms_weight) {
    int n = x.size();
    const int simd_width = 16;

    __m512 sum_xx_vec = _mm512_setzero_ps();

    // Step 1: Compute sum of squares using AVX-512
    int i = 0;
    for (; i <= n - simd_width; i += simd_width) {
        __m256i bf16_vals = _mm256_loadu_si256((__m256i*)&x[i]);

        // Convert BF16 to FP32
        __m512 fp32_vals = bf16o_fp32_512(bf16_vals);
        sum_xx_vec = _mm512_fmadd_ps(fp32_vals, fp32_vals, sum_xx_vec);  // sum_xx += x[i] * x[i];
    }

    // Horizontal sum of the AVX-512 register
    f32 sum_xx = _mm512_reduce_add_ps(sum_xx_vec);

    sum_xx /= (f32)n;
    f32 rms_x = sqrtf(sum_xx + 1e-5f);

    // Step 2: Compute element-wise operation y[i] = w[i] * x[i] / rms_x
    __m512 rms_x_vec = _mm512_set1_ps(rms_x);
    for (i = 0; i <= n - simd_width; i += simd_width) {
        __m256i bf16_vals = _mm256_loadu_si256((__m256i*)&x[i]);
        __m512 fp32_vals = bf16o_fp32_512(bf16_vals);
        __m256i bf16_vals_w = _mm256_loadu_si256((__m256i*)&rms_weight[i]);
        __m512 fp32_vals_w = bf16o_fp32_512(bf16_vals_w);
        __m512 y_vec = _mm512_div_ps(_mm512_mul_ps(fp32_vals_w, fp32_vals), rms_x_vec);
        __m256i result_bf16 = f32o_bf16_512(y_vec);
        _mm256_storeu_si256((__m256i*)&y[i], result_bf16);
    }
}

/// @brief In-place rotary embedding of every head of one row at position idx.
inline void _rope(buffer<bf16>& x, int idx, uint32_t DH, const f32* inv_freq) {
    int heads = x.size() / DH;
    buffer<f32> cos_buf(DH / 2);
    buffer<f32> sin_buf(DH / 2);
    for (uint32_t i = 0; i < DH / 2; i++) {
        cos_buf[i] = std::cos(inv_freq[i] * idx);
        sin_buf[i] = std::sin(inv_freq[i] * idx);
    }

    for (int h = 0; h < heads; h++) {
        buffer<bf16> x_left(x.data() + h * DH, DH / 2);
        buffer<bf16> x_right(x.data() + h * DH + DH / 2, DH / 2);
        int i = 0, simd = 16;
        // vectorized main loop
        for (; i + simd <= (int)(DH / 2); i += simd) {
            // load 16 BF16 -> float32
            __m256i left_bf16  = _mm256_loadu_si256((__m256i*)(x_left.data()  + i));
            __m256i right_bf16 = _mm256_loadu_si256((__m256i*)(x_right.data() + i));
            __m512  L = bf16o_fp32_512(left_bf16);
            __m512  R = bf16o_fp32_512(right_bf16);

            // load sin/cos
            __m512 C = _mm512_loadu_ps(cos_buf.data() + i);
            __m512 S = _mm512_loadu_ps(sin_buf.data() + i);

            // new_left  = L*C - R*S
            __m512 newL = _mm512_sub_ps(_mm512_mul_ps(L, C),
                                        _mm512_mul_ps(R, S));
            // new_right = L*S + R*C
            __m512 newR = _mm512_add_ps(_mm512_mul_ps(L, S),
                                        _mm512_mul_ps(R, C));

            // store back
            __m256i newL_bf16 = f32o_bf16_512(newL);
            __m256i newR_bf16 = f32o_bf16_512(newR);
            _mm256_storeu_si256((__m256i*)(x_left.data()  + i), newL_bf16);
            _mm256_storeu_si256((__m256i*)(x_right.data() + i), newR_bf16);
        }
    }
}

/// @brief dest = src + residual, over the whole of src.
inline void _residual_add(buffer<bf16>& dest, buffer<bf16>& src, buffer<bf16>& residual) {
    int n = src.size();
    const int simd_width = 16;

    for (int i = 0; i < n; i += simd_width) {
        __m256i bf16_vals = _mm256_loadu_si256((__m256i*)&src[i]);
        __m256i bf16_vals_r = _mm256_loadu_si256((__m256i*)&residual[i]);
        __m512 fp32_vals = bf16o_fp32_512(bf16_vals);
        __m512 fp32_vals_r = bf16o_fp32_512(bf16_vals_r);
        __m512 result_vec = _mm512_add_ps(fp32_vals, fp32_vals_r);
        __m256i result_bf16 = f32o_bf16_512(result_vec);
        _mm256_storeu_si256((__m256i*)&dest[i], result_bf16);
    }
}

} // namespace nanbeige_cpu_func

#endif
