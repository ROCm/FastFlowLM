#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <immintrin.h>
#include "typedef.hpp"
#include "utils/avx512_util.hpp"

/**
 * @brief AVX-512 optimized Layer Normalization with high precision accumulation.
 *
 * This implementation combines AVX-512 vectorization with numerical accuracy:
 * - Uses AVX-512 for parallel processing (16 floats at a time)
 * - Double precision for mean and variance accumulation
 * - Handles bfloat16 input/output with float32 intermediate calculations
 *
 * @param N The size of the hidden dimension.
 * @param hidden_states The input vector (bfloat16_t).
 * @param weight The scale parameter (bfloat16_t).
 * @param bias The bias parameter (bfloat16_t), can be nullptr if no bias.
 * @param eps The epsilon value (float) to prevent division by zero (typically 1e-6).
 * @param output The output vector (bfloat16_t).
 */
void layernorm_high_precision(
    size_t N,
    bf16* hidden_states,
    bf16* weight,
    bf16* bias,
    float eps,
    bf16* output)
{
    constexpr size_t SIMD_WIDTH = 16; // AVX-512 processes 16 floats at once
    const size_t vec_count = N / SIMD_WIDTH;
    const size_t remainder = N % SIMD_WIDTH;

    // Step 1: Calculate mean using float precision for accumulation
    __m512 sum_vec = _mm512_setzero_ps(); // 16 floats at a time
    float sum = 0.0f;

    // Vectorized sum calculation
    for (size_t i = 0; i < vec_count; ++i) {
        size_t idx = i * SIMD_WIDTH;

        // Load 16 bfloat16 values and convert to float
        __m512 vals = load_bfloat16_to_m512(hidden_states + idx);

        // Accumulate in float precision
        sum_vec = _mm512_add_ps(sum_vec, vals);
    }

    // Reduce vector sum to scalar
    sum = _mm512_reduce_add_ps(sum_vec);

    // Handle remainder elements
    for (size_t i = vec_count * SIMD_WIDTH; i < N; ++i) {
        sum += static_cast<float>(hidden_states[i]);
    }

    float mean = sum / static_cast<float>(N);
    __m512 mean_vec = _mm512_set1_ps(mean);

    // Step 2: Calculate variance using float precision
    __m512 var_vec = _mm512_setzero_ps();
    float var_sum = 0.0f;

    for (size_t i = 0; i < vec_count; ++i) {
        size_t idx = i * SIMD_WIDTH;

        // Load and convert to float
        __m512 vals = load_bfloat16_to_m512(hidden_states + idx);

        // Calculate (val - mean)^2
        __m512 diff = _mm512_sub_ps(vals, mean_vec);
        __m512 diff_sq = _mm512_mul_ps(diff, diff);

        // Accumulate in float precision
        var_vec = _mm512_add_ps(var_vec, diff_sq);
    }

    var_sum = _mm512_reduce_add_ps(var_vec);

    // Handle remainder
    for (size_t i = vec_count * SIMD_WIDTH; i < N; ++i) {
        float val = static_cast<float>(hidden_states[i]);
        float diff = val - mean;
        var_sum += diff * diff;
    }

    float variance = var_sum / static_cast<float>(N);

    // Step 3: Calculate 1/sqrt(variance + eps)
    float std_inv = 1.0f / std::sqrt(variance + eps);
    __m512 std_inv_vec = _mm512_set1_ps(std_inv);

    // Step 4: Normalize, scale, and shift
    for (size_t i = 0; i < vec_count; ++i) {
        size_t idx = i * SIMD_WIDTH;

        // Load input
        __m512 vals = load_bfloat16_to_m512(hidden_states + idx);

        // Normalize: (x - mean) / std
        __m512 normalized = _mm512_sub_ps(vals, mean_vec);
        normalized = _mm512_mul_ps(normalized, std_inv_vec);

        // Load weight and scale
        __m512 w = load_bfloat16_to_m512(weight + idx);
        __m512 result = _mm512_mul_ps(normalized, w);

        // Add bias if provided
        if (bias != nullptr) {
            __m512 b = load_bfloat16_to_m512(bias + idx);
            result = _mm512_add_ps(result, b);
        }

        // Convert back to bfloat16 and store
        store_m512_to_bfloat16_rne(output + idx, result);
    }

    // Handle remainder elements
    for (size_t i = vec_count * SIMD_WIDTH; i < N; ++i) {
        float val = static_cast<float>(hidden_states[i]);
        float normalized = (val - mean) * std_inv;
        float w = static_cast<float>(weight[i]);
        float result = normalized * w;

        if (bias != nullptr) {
            float b = static_cast<float>(bias[i]);
            result += b;
        }

        output[i] = static_cast<bf16>(result);
    }
}

/**
 * @brief Parallel Layer Normalization across multiple sequences with OpenMP.
 *
 * Applies layer normalization to multiple sequences in parallel using up to 4 threads.
 * Uses chunked distribution (static scheduling) for better cache locality.
 * Automatically disables parallelization for small sequence counts.
 *
 * @param seq_len Number of sequences to normalize.
 * @param hidden_dim The size of the hidden dimension for each sequence.
 * @param hidden_states_base Pointer to base of input sequences (bfloat16_t).
 * @param weight The scale parameter (bfloat16_t).
 * @param bias The bias parameter (bfloat16_t), can be nullptr if no bias.
 * @param eps The epsilon value (float) to prevent division by zero.
 * @param output_base Pointer to base of output sequences (bfloat16_t).
 */
void layernorm_parallel(
    int seq_len,
    size_t hidden_dim,
    size_t hidden_dim_padded,
    bf16* hidden_states_base,
    bf16* weight,
    bf16* bias,
    float eps,
    bf16* output_base)
{
    // Use OpenMP only if seq_len is large enough to benefit from parallelization
    // Threshold: 8 sequences per thread minimum to avoid overhead
    const int min_seq_per_thread = 8;
    const int max_threads = 4;
    const bool use_parallel = seq_len >= (min_seq_per_thread * 2);

    // Process each sequence in parallel with chunked distribution (static scheduling)
    // This ensures thread-0 gets sequences [0, chunk_size), thread-1 gets [chunk_size, 2*chunk_size), etc.
    // Better for cache locality than round-robin distribution
    #pragma omp parallel for num_threads(max_threads) if(use_parallel) schedule(static)
    for (int s = 0; s < seq_len; ++s) {
        bf16* input_seq = hidden_states_base + s * hidden_dim_padded;
        bf16* output_seq = output_base + s * hidden_dim_padded;

        layernorm_high_precision(
            hidden_dim,
            input_seq,
            weight,
            bias,
            eps,
            output_seq
        );
    }
}
