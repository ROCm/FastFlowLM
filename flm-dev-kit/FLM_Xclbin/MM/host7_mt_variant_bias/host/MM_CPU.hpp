// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __MM_CPU__
#define __MM_CPU__
#include <cassert>
#include <cinttypes>
#include <immintrin.h> // For AVX-512 intrinsics
#include <stdfloat>
#include <vector>
#include "avx512_util.hpp"
template<typename T_out>
void linear_row_major_w_template(T_out* y,
                                 const std::bfloat16_t* x,
                                 const std::bfloat16_t* w,
                                 int M, int K, int N) {
    static_assert(std::is_same<T_out, float>::value || std::is_same<T_out, std::bfloat16_t>::value,
                  "Output type must be float or std::bfloat16_t");

    constexpr int VEC_WIDTH_PS = 16;

    for (int row = 0; row < M; ++row) {
        int col = 0;
        for (; col <= N - VEC_WIDTH_PS; col += VEC_WIDTH_PS) {
            __m512 y_vec_ps = _mm512_setzero_ps();

            for (int k = 0; k < K; ++k) {
                __m512 x_broadcast_ps = _mm512_set1_ps(static_cast<float>(x[row * K + k]));

                __m256i w_vec_bf16_raw = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(&w[k * N + col])
                );

                __m512 w_vec_ps = _mm512_cvtpbh_ps(reinterpret_cast<__m256bh>(w_vec_bf16_raw));
                y_vec_ps = _mm512_fmadd_ps(x_broadcast_ps, w_vec_ps, y_vec_ps);
            }

            if constexpr (std::is_same<T_out, std::bfloat16_t>::value) {
                __m256bh y_vec_bf16 = _mm512_cvtneps_pbh(y_vec_ps);
                _mm256_storeu_si256(
                    reinterpret_cast<__m256i*>(&y[row * N + col]),
                    reinterpret_cast<__m256i>(y_vec_bf16)
                );
            } else {
                _mm512_storeu_ps(&y[row * N + col], y_vec_ps);
            }
        }

        for (; col < N; ++col) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += static_cast<float>(x[row * K + k]) * static_cast<float>(w[k * N + col]);
            }
            y[row * N + col] = static_cast<T_out>(sum);
        }
    }
}

template<typename T_out>
void linear_col_major_w_template(T_out* y,
                                 const std::bfloat16_t* x,
                                 const std::bfloat16_t* w,
                                 int M, int K, int N) {
    static_assert(std::is_same<T_out, float>::value || std::is_same<T_out, std::bfloat16_t>::value,
                  "Output type must be float or std::bfloat16_t");

    constexpr int VEC_WIDTH_BF16 = 32;

    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            const auto* x_row = &x[row * K];
            const auto* w_col = &w[col * K];

            __m512 sum_vec_ps = _mm512_setzero_ps();

            int k = 0;
            for (; k <= K - VEC_WIDTH_BF16; k += VEC_WIDTH_BF16) {
                __m512bh x_vec_bf16 = reinterpret_cast<__m512bh>(_mm512_loadu_si512(&x_row[k]));
                __m512bh w_vec_bf16 = reinterpret_cast<__m512bh>(_mm512_loadu_si512(&w_col[k]));

                sum_vec_ps = _mm512_dpbf16_ps(sum_vec_ps, x_vec_bf16, w_vec_bf16);
            }

            float sum = _mm512_reduce_add_ps(sum_vec_ps);

            for (; k < K; ++k) {
                sum += static_cast<float>(x_row[k]) * static_cast<float>(w_col[k]);
            }

            y[row * N + col] = static_cast<T_out>(sum);
        }
    }
}



/**
 * @brief Performs y = x * w + b (row-major w) with AVX-512.
 * Assumes w_and_bias points to [bias[0...N-1], weights[0...K*N-1]].
 * Weights are stored row-major (KxN).
 * * @tparam T_out Output type (float or std::bfloat16_t).
 * @param y Output buffer (M x N).
 * @param x Input buffer (M x K).
 * @param w_and_bias Bias (N) followed by Weights (K x N, row-major).
 * @param M Dimension M.
 * @param K Dimension K.
 * @param N Dimension N.
 */
template<typename T_out>
void linear_row_major_w_with_bias_template(T_out* y,
                                           const std::bfloat16_t* x,
                                           const std::bfloat16_t* w_and_bias,
                                           int M, int K, int N) {
    static_assert(std::is_same<T_out, float>::value || std::is_same<T_out, std::bfloat16_t>::value,
                  "Output type must be float or std::bfloat16_t");

    // Bias vector is the first N elements
    const std::bfloat16_t* b = w_and_bias;
    // Weight matrix starts after the bias vector
    const std::bfloat16_t* w = w_and_bias + N;

    constexpr int VEC_WIDTH_PS = 16; // 16 floats in __m512

    for (int row = 0; row < M; ++row) {
        int col = 0;
        // Vectorized loop (processes 16 output columns at a time)
        for (; col <= N - VEC_WIDTH_PS; col += VEC_WIDTH_PS) {
            
            // 1. Initialize accumulator with bias
            // Load 16 bf16 bias values
            __m256i b_vec_bf16_raw = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(&b[col])
            );
            // Convert bias to 16 float values
            __m512 y_vec_ps = _mm512_cvtpbh_ps(reinterpret_cast<__m256bh>(b_vec_bf16_raw));

            // 2. Perform matrix multiplication (dot product)
            for (int k = 0; k < K; ++k) {
                // Broadcast the single x value
                __m512 x_broadcast_ps = _mm512_set1_ps(static_cast<float>(x[row * K + k]));

                // Load 16 weights from the k-th row of w
                __m256i w_vec_bf16_raw = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(&w[k * N + col])
                );
                // Convert weights to 16 float values
                __m512 w_vec_ps = _mm512_cvtpbh_ps(reinterpret_cast<__m256bh>(w_vec_bf16_raw));
                
                // Fused multiply-add: y_vec_ps += x_broadcast_ps * w_vec_ps
                y_vec_ps = _mm512_fmadd_ps(x_broadcast_ps, w_vec_ps, y_vec_ps);
            }

            // 3. Store the result
            if constexpr (std::is_same<T_out, std::bfloat16_t>::value) {
                // Convert 16 floats back to 16 bf16
                __m256bh y_vec_bf16 = _mm512_cvtneps_pbh(y_vec_ps);
                // Store 16 bf16 values
                _mm256_storeu_si256(
                    reinterpret_cast<__m256i*>(&y[row * N + col]),
                    reinterpret_cast<__m256i>(y_vec_bf16)
                );
            } else {
                // Store 16 float values
                _mm512_storeu_ps(&y[row * N + col], y_vec_ps);
            }
        }

        // Remainder loop (processes 1 output column at a time)
        for (; col < N; ++col) {
            // Initialize sum with bias
            float sum = static_cast<float>(b[col]);
            for (int k = 0; k < K; ++k) {
                sum += static_cast<float>(x[row * K + k]) * static_cast<float>(w[k * N + col]);
            }
            y[row * N + col] = static_cast<T_out>(sum);
        }
    }
}

/**
 * @brief Performs y = x * w + b (col-major w) with AVX-512 (BF16).
 * Assumes w_and_bias points to [bias[0...N-1], weights[0...K*N-1]].
 * Weights are stored col-major (KxN).
 * * @tparam T_out Output type (float or std::bfloat16_t).
 * @param y Output buffer (M x N).
 * @param x Input buffer (M x K).
 * @param w_and_bias Bias (N) followed by Weights (K x N, col-major).
 * @param M Dimension M.
 * @param K Dimension K.
 * @param N Dimension N.
 */
template<typename T_out>
void linear_col_major_w_with_bias_template(T_out* y,
                                           const std::bfloat16_t* x,
                                           const std::bfloat16_t* w_and_bias,
                                           int M, int K, int N) {
    static_assert(std::is_same<T_out, float>::value || std::is_same<T_out, std::bfloat16_t>::value,
                  "Output type must be float or std::bfloat16_t");

    // Bias vector is the first N elements
    const std::bfloat16_t* b = w_and_bias;
    // Weight matrix starts after the bias vector
    const std::bfloat16_t* w = w_and_bias + N;

    // 32 bf16 values in __m512i (__m512bh)
    constexpr int VEC_WIDTH_BF16 = 32;

    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            const auto* x_row = &x[row * K];
            // w is col-major, so w[col * K] is the start of the col-th column
            const auto* w_col = &w[col * K]; 

            // Initialize float accumulator vector
            __m512 sum_vec_ps = _mm512_setzero_ps();

            int k = 0;
            // Vectorized loop: dot product of two bf16 vectors
            for (; k <= K - VEC_WIDTH_BF16; k += VEC_WIDTH_BF16) {
                // Load 32 bf16 values from x
                __m512bh x_vec_bf16 = reinterpret_cast<__m512bh>(_mm512_loadu_si512(&x_row[k]));
                // Load 32 bf16 values from w
                __m512bh w_vec_bf16 = reinterpret_cast<__m512bh>(_mm512_loadu_si512(&w_col[k]));

                // Dot product of bf16 vectors, accumulating in float vector
                sum_vec_ps = _mm512_dpbf16_ps(sum_vec_ps, x_vec_bf16, w_vec_bf16);
            }

            // Horizontally sum the 16 float values in the accumulator
            float sum = _mm512_reduce_add_ps(sum_vec_ps);

            // Remainder loop for dot product
            for (; k < K; ++k) {
                sum += static_cast<float>(x_row[k]) * static_cast<float>(w_col[k]);
            }

            // Add the bias element for this column
            sum += static_cast<float>(b[col]);

            // Store the final result
            y[row * N + col] = static_cast<T_out>(sum);
        }
    }
}
















inline void gelu_bfloat16_ref_scalar(
    const std::bfloat16_t* input,
    std::bfloat16_t* output,
    size_t size
) {
    constexpr float sqrt_2_over_pi = 0.7978845608028654f; // sqrt(2/pi)
    constexpr float coeff = 0.044715f;

    for (size_t i = 0; i < size; ++i) {
        float x = static_cast<float>(input[i]);
        float x_cubed = x * x * x;
        float inner = sqrt_2_over_pi * (x + coeff * x_cubed);
        float tanh_val = std::tanh(inner);
        float gelu = 0.5f * x * (1.0f + tanh_val);
        output[i] = static_cast<std::bfloat16_t>(gelu);
    }
}

inline void gelu_bfloat16_ref(
    const std::bfloat16_t* input,
    std::bfloat16_t* output,
    size_t size
) {
    size_t i = 0;
    for (; i + 15 < size; i += 16) {
        __m512 input_vec = load_bfloat16_to_m512(input + i);
        __m512 gelu_vec = gelu_tanh_avx512_simd(input_vec);
        store_m512_to_bfloat16_rne(output + i, gelu_vec);
    }

    if (i < size) {
        gelu_bfloat16_ref_scalar(input + i, output + i, size - i);
    }
}

inline void silu_bfloat16_ref_scalar(
    const std::bfloat16_t* input,
    std::bfloat16_t* output,
    size_t size
) {
    for (size_t i = 0; i < size; ++i) {
        float x = static_cast<float>(input[i]);
        float sigmoid_x = 1.0f / (1.0f + std::exp(-x));
        float silu = x * sigmoid_x;
        output[i] = static_cast<std::bfloat16_t>(silu);
    }
}



inline void sigmoid_bfloat16_ref_scalar(
    const std::bfloat16_t* input,
    std::bfloat16_t* output,
    size_t size
) {
    for (size_t i = 0; i < size; ++i) {
        float x = static_cast<float>(input[i]);
        float sigmoid_x = 1.0f / (1.0f + std::exp(-x));
        output[i] = static_cast<std::bfloat16_t>(sigmoid_x);
    }
}

inline void silu_bfloat16_ref(
    const std::bfloat16_t* input,
    std::bfloat16_t* output,
    size_t size
) {
    size_t i = 0;
    for (; i + 15 < size; i += 16) {
        __m512 input_vec = load_bfloat16_to_m512(input + i);
        __m512 silu_vec = silu_avx512_simd(input_vec);
        store_m512_to_bfloat16_rne(output + i, silu_vec);
    }

    if (i < size) {
        silu_bfloat16_ref_scalar(input + i, output + i, size - i);
    }
}

#endif