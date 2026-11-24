// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#pragma once

#include <immintrin.h>
#include <cstdint>
#include <limits>
#include <type_traits>
// #include <stdfloat> // For bf16 in C++23
#include "gpt_router.hpp"
namespace moeMLP{

/**
 * @brief Scalar reference implementation for the operation.
 *
 * @param out Pointer to the output array.
 * @param up Pointer to the 'up' input array.
 * @param gate Pointer to the 'gate' input array.
 * @param alpha Scalar alpha value.
 * @param n Number of elements in the arrays.
 */
void custom_op_scalar(
    bf16* out,
    const bf16* up,
    const bf16* gate,
    float alpha,
    size_t n
) {
    for (size_t i = 0; i < n; ++i) {
        float up_f = static_cast<float>(up[i]);
        float gate_f = static_cast<float>(gate[i]);

        float sigmoid_arg = gate_f * alpha;
        float sigmoid_val = 1.0f / (1.0f + std::exp(-sigmoid_arg));

        float result_f = (up_f + 1.0f) * (gate_f * sigmoid_val);

        out[i] = static_cast<bf16>(result_f);
    }
}

/**
 * @brief AVX512 implementation of the operation.
 *
 * Processes 32 bfloat16 elements per iteration.
 *
 * @param out Pointer to the output array.
 * @param up Pointer to the 'up' input array.
 * @param gate Pointer to the 'gate' input array.
 * @param alpha Scalar alpha value.
 * @param n Number of elements in the arrays.
 */
// Fused function that combines bias addition, clamping, and custom operations
template<float GATE_LOWER_CLAMP, float GATE_UPPER_CLAMP, float UP_LOWER_CLAMP, float UP_UPPER_CLAMP>
void fused_gate_up_ops_avx512(
    bf16* out,
    bf16* up,
    bf16* gate,
    const bf16* up_bias,
    const bf16* gate_bias,
    float alpha,
    size_t n
) {
    // Pre-broadcast scalar constants into 512-bit vectors
    const __m512 v_one = _mm512_set1_ps(1.0f);
    const __m512 v_alpha = _mm512_set1_ps(alpha);
    const __m512 v_neg_one = _mm512_set1_ps(-1.0f);
    const __m512 v_log2e = _mm512_set1_ps(1.44269504089f);

    // Clamp bounds for gate and up
    constexpr bool gate_needs_lower_clamp = (GATE_LOWER_CLAMP > -std::numeric_limits<float>::infinity());
    constexpr bool gate_needs_upper_clamp = (GATE_UPPER_CLAMP < std::numeric_limits<float>::infinity());
    constexpr bool up_needs_lower_clamp = (UP_LOWER_CLAMP > -std::numeric_limits<float>::infinity());
    constexpr bool up_needs_upper_clamp = (UP_UPPER_CLAMP < std::numeric_limits<float>::infinity());

    const __m512 gate_lower_bound_vec_ps = _mm512_set1_ps(GATE_LOWER_CLAMP);
    const __m512 gate_upper_bound_vec_ps = _mm512_set1_ps(GATE_UPPER_CLAMP);
    const __m512 up_lower_bound_vec_ps = _mm512_set1_ps(UP_LOWER_CLAMP);
    const __m512 up_upper_bound_vec_ps = _mm512_set1_ps(UP_UPPER_CLAMP);

    size_t i = 0;
    // Main loop processes 32 bfloat16 elements per iteration
    for (; i + 31 < n; i += 32) {
        // Load 32 bf16 elements (512 bits) for up, gate, and their biases
        __m512i v_up_bf16 = _mm512_loadu_si512(up + i);
        __m512i v_gate_bf16 = _mm512_loadu_si512(gate + i);
        __m512i v_up_bias_bf16 = _mm512_loadu_si512(up_bias + i);
        __m512i v_gate_bias_bf16 = _mm512_loadu_si512(gate_bias + i);

        // Split the 32-element bf16 vectors into two 16-element chunks
        __m256i v_up_lo_16 = _mm512_castsi512_si256(v_up_bf16);
        __m256i v_gate_lo_16 = _mm512_castsi512_si256(v_gate_bf16);
        __m256i v_up_hi_16 = _mm512_extracti64x4_epi64(v_up_bf16, 1);
        __m256i v_gate_hi_16 = _mm512_extracti64x4_epi64(v_gate_bf16, 1);

        __m256i v_up_bias_lo_16 = _mm512_castsi512_si256(v_up_bias_bf16);
        __m256i v_gate_bias_lo_16 = _mm512_castsi512_si256(v_gate_bias_bf16);
        __m256i v_up_bias_hi_16 = _mm512_extracti64x4_epi64(v_up_bias_bf16, 1);
        __m256i v_gate_bias_hi_16 = _mm512_extracti64x4_epi64(v_gate_bias_bf16, 1);

        // Convert 16-element bf16 chunks to 16-element float vectors
        __m512 v_up_lo_ps = _mm512_cvtpbh_ps((__m256bh)v_up_lo_16);
        __m512 v_up_hi_ps = _mm512_cvtpbh_ps((__m256bh)v_up_hi_16);
        __m512 v_gate_lo_ps = _mm512_cvtpbh_ps((__m256bh)v_gate_lo_16);
        __m512 v_gate_hi_ps = _mm512_cvtpbh_ps((__m256bh)v_gate_hi_16);

        __m512 v_up_bias_lo_ps = _mm512_cvtpbh_ps((__m256bh)v_up_bias_lo_16);
        __m512 v_up_bias_hi_ps = _mm512_cvtpbh_ps((__m256bh)v_up_bias_hi_16);
        __m512 v_gate_bias_lo_ps = _mm512_cvtpbh_ps((__m256bh)v_gate_bias_lo_16);
        __m512 v_gate_bias_hi_ps = _mm512_cvtpbh_ps((__m256bh)v_gate_bias_hi_16);

        // Add bias to up and gate
        v_up_lo_ps = _mm512_add_ps(v_up_lo_ps, v_up_bias_lo_ps);
        v_up_hi_ps = _mm512_add_ps(v_up_hi_ps, v_up_bias_hi_ps);
        v_gate_lo_ps = _mm512_add_ps(v_gate_lo_ps, v_gate_bias_lo_ps);
        v_gate_hi_ps = _mm512_add_ps(v_gate_hi_ps, v_gate_bias_hi_ps);

        // Apply clamping to gate
        if constexpr (gate_needs_lower_clamp && gate_needs_upper_clamp) {
            v_gate_lo_ps = _mm512_max_ps(v_gate_lo_ps, gate_lower_bound_vec_ps);
            v_gate_lo_ps = _mm512_min_ps(v_gate_lo_ps, gate_upper_bound_vec_ps);
            v_gate_hi_ps = _mm512_max_ps(v_gate_hi_ps, gate_lower_bound_vec_ps);
            v_gate_hi_ps = _mm512_min_ps(v_gate_hi_ps, gate_upper_bound_vec_ps);
        } else if constexpr (gate_needs_lower_clamp) {
            v_gate_lo_ps = _mm512_max_ps(v_gate_lo_ps, gate_lower_bound_vec_ps);
            v_gate_hi_ps = _mm512_max_ps(v_gate_hi_ps, gate_lower_bound_vec_ps);
        } else if constexpr (gate_needs_upper_clamp) {
            v_gate_lo_ps = _mm512_min_ps(v_gate_lo_ps, gate_upper_bound_vec_ps);
            v_gate_hi_ps = _mm512_min_ps(v_gate_hi_ps, gate_upper_bound_vec_ps);
        }

        // Store the updated gate values back
        __m256bh v_gate_lo_bf16 = _mm512_cvtneps_pbh(v_gate_lo_ps);
        __m256bh v_gate_hi_bf16 = _mm512_cvtneps_pbh(v_gate_hi_ps);
        __m512i v_gate_updated_bf16 = _mm512_castsi256_si512((__m256i)v_gate_lo_bf16);
        v_gate_updated_bf16 = _mm512_inserti64x4(v_gate_updated_bf16, (__m256i)v_gate_hi_bf16, 1);
        _mm512_storeu_si512(gate + i, v_gate_updated_bf16);

        // Apply clamping to up
        if constexpr (up_needs_lower_clamp && up_needs_upper_clamp) {
            v_up_lo_ps = _mm512_max_ps(v_up_lo_ps, up_lower_bound_vec_ps);
            v_up_lo_ps = _mm512_min_ps(v_up_lo_ps, up_upper_bound_vec_ps);
            v_up_hi_ps = _mm512_max_ps(v_up_hi_ps, up_lower_bound_vec_ps);
            v_up_hi_ps = _mm512_min_ps(v_up_hi_ps, up_upper_bound_vec_ps);
        } else if constexpr (up_needs_lower_clamp) {
            v_up_lo_ps = _mm512_max_ps(v_up_lo_ps, up_lower_bound_vec_ps);
            v_up_hi_ps = _mm512_max_ps(v_up_hi_ps, up_lower_bound_vec_ps);
        } else if constexpr (up_needs_upper_clamp) {
            v_up_lo_ps = _mm512_min_ps(v_up_lo_ps, up_upper_bound_vec_ps);
            v_up_hi_ps = _mm512_min_ps(v_up_hi_ps, up_upper_bound_vec_ps);
        }

        // Store the updated up values back
        __m256bh v_up_lo_bf16 = _mm512_cvtneps_pbh(v_up_lo_ps);
        __m256bh v_up_hi_bf16 = _mm512_cvtneps_pbh(v_up_hi_ps);
        __m512i v_up_updated_bf16 = _mm512_castsi256_si512((__m256i)v_up_lo_bf16);
        v_up_updated_bf16 = _mm512_inserti64x4(v_up_updated_bf16, (__m256i)v_up_hi_bf16, 1);
        _mm512_storeu_si512(up + i, v_up_updated_bf16);

        // Now perform the custom operation using the updated values
        // --- Process low 16 elements ---
        // term1 = (up + 1)
        __m512 v_up1_lo = _mm512_add_ps(v_up_lo_ps, v_one);
        // x = gate * alpha
        __m512 v_x_lo = _mm512_mul_ps(v_gate_lo_ps, v_alpha);
        // Fast approximate sigmoid(x) = 1 / (1 + exp(-x))
        __m512 v_neg_x_lo = _mm512_mul_ps(v_x_lo, v_neg_one);
        __m512 v_exp_lo = exp512_approx(v_neg_x_lo);
        __m512 v_denom_lo = _mm512_add_ps(v_one, v_exp_lo);
        __m512 v_sigmoid_lo = _mm512_rcp14_ps(v_denom_lo);
        // term2 = gate * sigmoid(x)
        __m512 v_gated_sigmoid_lo = _mm512_mul_ps(v_gate_lo_ps, v_sigmoid_lo);
        // result = term1 * term2
        __m512 v_result_lo_ps = _mm512_mul_ps(v_up1_lo, v_gated_sigmoid_lo);

        // --- Process high 16 elements ---
        __m512 v_up1_hi = _mm512_add_ps(v_up_hi_ps, v_one);
        __m512 v_x_hi = _mm512_mul_ps(v_gate_hi_ps, v_alpha);
        __m512 v_neg_x_hi = _mm512_mul_ps(v_x_hi, v_neg_one);
        __m512 v_exp_hi = exp512_approx(v_neg_x_hi);
        __m512 v_denom_hi = _mm512_add_ps(v_one, v_exp_hi);
        __m512 v_sigmoid_hi = _mm512_rcp14_ps(v_denom_hi);
        __m512 v_gated_sigmoid_hi = _mm512_mul_ps(v_gate_hi_ps, v_sigmoid_hi);
        __m512 v_result_hi_ps = _mm512_mul_ps(v_up1_hi, v_gated_sigmoid_hi);

        // Convert float results back to bf16 chunks
        __m256bh v_result_lo_bf16 = _mm512_cvtneps_pbh(v_result_lo_ps);
        __m256bh v_result_hi_bf16 = _mm512_cvtneps_pbh(v_result_hi_ps);

        __m512i v_result_bf16 = _mm512_castsi256_si512((__m256i)v_result_lo_bf16);
        v_result_bf16 = _mm512_inserti64x4(v_result_bf16, (__m256i)v_result_hi_bf16, 1);

        // Store 32 bf16 results
        _mm512_storeu_si512(out + i, v_result_bf16);
    }

    // Remainder loop for n that is not a multiple of 32
    if (i < n) {
        // For remainder, we need to do the operations sequentially
        for (size_t j = i; j < n; ++j) {
            // Add bias
            float up_val = static_cast<float>(up[j]) + static_cast<float>(up_bias[j]);
            float gate_val = static_cast<float>(gate[j]) + static_cast<float>(gate_bias[j]);

            // Apply clamping
            if constexpr (gate_needs_lower_clamp) {
                gate_val = std::max(gate_val, GATE_LOWER_CLAMP);
            }
            if constexpr (gate_needs_upper_clamp) {
                gate_val = std::min(gate_val, GATE_UPPER_CLAMP);
            }
            if constexpr (up_needs_lower_clamp) {
                up_val = std::max(up_val, UP_LOWER_CLAMP);
            }
            if constexpr (up_needs_upper_clamp) {
                up_val = std::min(up_val, UP_UPPER_CLAMP);
            }

            // Store updated values
            up[j] = static_cast<bf16>(up_val);
            gate[j] = static_cast<bf16>(gate_val);

            // Perform custom operation
            float x = gate_val * alpha;
            float sigmoid = 1.0f / (1.0f + std::exp(-x));
            out[j] = static_cast<bf16>((up_val + 1.0f) * gate_val * sigmoid);
        }
    }
}

void custom_op_avx512(
    bf16* out,
    const bf16* up,
    const bf16* gate,
    float alpha,
    size_t n
) {
    // Pre-broadcast scalar constants into 512-bit vectors
    const __m512 v_one = _mm512_set1_ps(1.0f);
    const __m512 v_alpha = _mm512_set1_ps(alpha);
    const __m512 v_neg_one = _mm512_set1_ps(-1.0f);
    // log2(e) is used for the identity e^x = 2^(x * log2(e))
    const __m512 v_log2e = _mm512_set1_ps(1.44269504089f);

    size_t i = 0;
    // Main loop processes 32 bfloat16 elements per iteration
    for (; i + 31 < n; i += 32) {
        // Load 32 bf16 elements (512 bits). Assumes `up` and `gate` are pointers to bf16.
        __m512i v_up_bf16 = _mm512_loadu_si512(up + i);
        __m512i v_gate_bf16 = _mm512_loadu_si512(gate + i);

        // Split the 32-element bf16 vectors into two 16-element chunks
        __m256i v_up_lo_16 = _mm512_castsi512_si256(v_up_bf16);
        __m256i v_gate_lo_16 = _mm512_castsi512_si256(v_gate_bf16);
        __m256i v_up_hi_16 = _mm512_extracti64x4_epi64(v_up_bf16, 1);
        __m256i v_gate_hi_16 = _mm512_extracti64x4_epi64(v_gate_bf16, 1);

        // Convert 16-element bf16 chunks to 16-element float vectors
        __m512 v_up_lo_ps = _mm512_cvtpbh_ps((__m256bh)v_up_lo_16);
        __m512 v_up_hi_ps = _mm512_cvtpbh_ps((__m256bh)v_up_hi_16);
        __m512 v_gate_lo_ps = _mm512_cvtpbh_ps((__m256bh)v_gate_lo_16);
        __m512 v_gate_hi_ps = _mm512_cvtpbh_ps((__m256bh)v_gate_hi_16);

        // --- Process low 16 elements ---
        // term1 = (up + 1)
        __m512 v_up1_lo = _mm512_add_ps(v_up_lo_ps, v_one);
        // x = gate * alpha
        __m512 v_x_lo = _mm512_mul_ps(v_gate_lo_ps, v_alpha);
        // Fast approximate sigmoid(x) = 1 / (1 + exp(-x))
        __m512 v_neg_x_lo = _mm512_mul_ps(v_x_lo, v_neg_one);
        __m512 v_exp_arg_lo = _mm512_mul_ps(v_neg_x_lo, v_log2e);
    __m512 v_exp_lo = exp512_approx(v_neg_x_lo);
        __m512 v_denom_lo = _mm512_add_ps(v_one, v_exp_lo);
        __m512 v_sigmoid_lo = _mm512_rcp14_ps(v_denom_lo);
        // term2 = gate * sigmoid(x)
        __m512 v_gated_sigmoid_lo = _mm512_mul_ps(v_gate_lo_ps, v_sigmoid_lo);
        // result = term1 * term2
        __m512 v_result_lo_ps = _mm512_mul_ps(v_up1_lo, v_gated_sigmoid_lo);

        // --- Process high 16 elements ---
        __m512 v_up1_hi = _mm512_add_ps(v_up_hi_ps, v_one);
        __m512 v_x_hi = _mm512_mul_ps(v_gate_hi_ps, v_alpha);
        __m512 v_neg_x_hi = _mm512_mul_ps(v_x_hi, v_neg_one);
        __m512 v_exp_arg_hi = _mm512_mul_ps(v_neg_x_hi, v_log2e);
    __m512 v_exp_hi = exp512_approx(v_neg_x_hi);
        __m512 v_denom_hi = _mm512_add_ps(v_one, v_exp_hi);
        __m512 v_sigmoid_hi = _mm512_rcp14_ps(v_denom_hi);
        __m512 v_gated_sigmoid_hi = _mm512_mul_ps(v_gate_hi_ps, v_sigmoid_hi);
        __m512 v_result_hi_ps = _mm512_mul_ps(v_up1_hi, v_gated_sigmoid_hi);

        // Convert float results back to bf16 chunks
    __m256bh v_result_lo_bf16 = _mm512_cvtneps_pbh(v_result_lo_ps);
    __m256bh v_result_hi_bf16 = _mm512_cvtneps_pbh(v_result_hi_ps);

    __m512i v_result_bf16 = _mm512_castsi256_si512((__m256i)v_result_lo_bf16);
    v_result_bf16 = _mm512_inserti64x4(v_result_bf16, (__m256i)v_result_hi_bf16, 1);

        // Store 32 bf16 results
        _mm512_storeu_si512(out + i, v_result_bf16);
    }

    // Remainder loop for n that is not a multiple of 32
    // Calls the scalar function for the last few elements.
    if (i < n) {
        custom_op_scalar(out + i, up + i, gate + i, alpha, n - i);
    }
}
}
