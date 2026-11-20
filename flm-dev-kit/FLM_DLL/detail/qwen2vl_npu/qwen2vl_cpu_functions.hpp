// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __QWEN2VL_CPU_FUNCTIONS_HPP__
#define __QWEN2VL_CPU_FUNCTIONS_HPP__
#include <vector>
#include <cmath>
#include "typedef.hpp"
#include "buffer.hpp"

/// @brief Host-side elementwise kernels used by the qwen2vl prefill path.
/// @note  These used to be private members of qwen2vl_npu::Impl; they are free
///        functions here so the prefill blocks can call them without a handle
///        on the model.
namespace qwen2vl_cpu_func {

/// @brief dest = a * b, elementwise over the whole of a.
inline void _elementwise_mul(buffer<bf16>& dest, buffer<bf16>& a, buffer<bf16>& b) {
    int n = a.size();
    const int simd_width = 16;
    const int unroll = simd_width * 4;

    int i = 0;
    // 4-way unrolled loop for maximum throughput
    for (; i <= n - unroll; i += unroll) {
        __m256i bf16_vals_a0 = _mm256_loadu_si256((__m256i *)&a[i]);
        __m256i bf16_vals_a1 = _mm256_loadu_si256((__m256i *)&a[i + simd_width]);
        __m256i bf16_vals_a2 = _mm256_loadu_si256((__m256i *)&a[i + simd_width * 2]);
        __m256i bf16_vals_a3 = _mm256_loadu_si256((__m256i *)&a[i + simd_width * 3]);

        __m256i bf16_vals_b0 = _mm256_loadu_si256((__m256i *)&b[i]);
        __m256i bf16_vals_b1 = _mm256_loadu_si256((__m256i *)&b[i + simd_width]);
        __m256i bf16_vals_b2 = _mm256_loadu_si256((__m256i *)&b[i + simd_width * 2]);
        __m256i bf16_vals_b3 = _mm256_loadu_si256((__m256i *)&b[i + simd_width * 3]);

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

        __m256i result_bf16_0 = f32o_bf16_512(result_vec0);
        __m256i result_bf16_1 = f32o_bf16_512(result_vec1);
        __m256i result_bf16_2 = f32o_bf16_512(result_vec2);
        __m256i result_bf16_3 = f32o_bf16_512(result_vec3);

        _mm256_storeu_si256((__m256i *)&dest[i], result_bf16_0);
        _mm256_storeu_si256((__m256i *)&dest[i + simd_width], result_bf16_1);
        _mm256_storeu_si256((__m256i *)&dest[i + simd_width * 2], result_bf16_2);
        _mm256_storeu_si256((__m256i *)&dest[i + simd_width * 3], result_bf16_3);
    }

    // Handle remaining elements
    for (; i <= n - simd_width; i += simd_width) {
        __m256i bf16_vals_a = _mm256_loadu_si256((__m256i *)&a[i]);
        __m256i bf16_vals_b = _mm256_loadu_si256((__m256i *)&b[i]);
        __m512 fp32_vals_a = bf16o_fp32_512(bf16_vals_a);
        __m512 fp32_vals_b = bf16o_fp32_512(bf16_vals_b);
        __m512 result_vec = _mm512_mul_ps(fp32_vals_a, fp32_vals_b);
        __m256i result_bf16 = f32o_bf16_512(result_vec);
        _mm256_storeu_si256((__m256i *)&dest[i], result_bf16);
    }
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
    f32 rms_x = sqrtf(sum_xx + 1e-6f);

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
/// @brief Fills the cos | sin table (DH/2 entries each) for one token.
///
/// Qwen2-VL uses 3-axis mrope: the frequency pairs are split into
/// mrope_section = [16, 24, 24], following the temporal, height and width
/// index respectively. The split is fixed by the checkpoint.
inline void _update_rope_values(
    std::vector<float>& rope_values,
    int idx_0, int idx_1, int idx_2,
    uint32_t DH,
    const f32* inv_freq
) {
    buffer<float> cos_buf(DH / 2);
    buffer<float> sin_buf(DH / 2);
    for (int j = 0; j < 16; j++) {
        cos_buf[j] = std::cos(inv_freq[j] * idx_0);
        sin_buf[j] = std::sin(inv_freq[j] * idx_0);
    }
    for (int j = 0; j < 24; j++) {
        cos_buf[j + 16] = std::cos(inv_freq[j + 16] * idx_1);
        sin_buf[j + 16] = std::sin(inv_freq[j + 16] * idx_1);
    }
    for (int j = 0; j < 24; j++) {
        cos_buf[j + 40] = std::cos(inv_freq[j + 40] * idx_2);
        sin_buf[j + 40] = std::sin(inv_freq[j + 40] * idx_2);
    }
    for (uint32_t j = 0; j < DH / 2; j++) {
        rope_values[j] = cos_buf[j];
        rope_values[j + DH / 2] = sin_buf[j];
    }
}

/// @brief Applies 3-axis mrope to every head of `x`, in place.
/// @param rope_values scratch of DH entries, laid out cos | sin; overwritten.
inline void _rope(
    buffer<bf16>& x,
    int idx_0, int idx_1, int idx_2,
    uint32_t DH,
    const f32* inv_freq,
    std::vector<float>& rope_values
) {
    int heads = x.size() / DH;
    _update_rope_values(rope_values, idx_0, idx_1, idx_2, DH, inv_freq);

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
            __m512 C = _mm512_loadu_ps(rope_values.data() + i);
            __m512 S = _mm512_loadu_ps(rope_values.data() + i + DH / 2);

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

} // namespace qwen2vl_cpu_func

#endif
