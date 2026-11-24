// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include "vision/conv3dPatchEmbedding.hpp"
#include <immintrin.h>

std::vector<bf16> conv3d_patch_embed(
    const bf16* hidden_states,
    const bf16* weights,
    const bf16* bias,
    size_t seq_len,
    size_t in_channels,
    size_t embed_dim,
    size_t temporal_patch_size,
    size_t patch_size)
{
    // Total number of elements in a single input patch / single convolution kernel
    const size_t kernel_volume = in_channels * temporal_patch_size * patch_size * patch_size;

    // --- Sanity Checks ---
    // NOTE: Size checks are disabled as inputs are raw pointers (*), which do not carry size information.
    // The caller is responsible for ensuring the arrays pointed to are of the correct length:
    // hidden_states: seq_len * kernel_volume
    // weights: embed_dim * kernel_volume
    // bias: embed_dim
    // --- End Sanity Checks ---

    std::vector<bf16> output(seq_len * embed_dim);

    // Iterate through each batch element (B)
    for (size_t b = 0; b < seq_len; ++b) {
        // Iterate through each output embedding dimension (Ed)
        for (size_t e = 0; e < embed_dim; ++e) {
            // Use 'float' for accumulation (sum) to preserve precision before casting back to bfloat16_t (if applicable)
            float sum = 0.0f;

            // Calculate the dot product between the input patch and the kernel
            const size_t input_start_idx = b * kernel_volume;
            const size_t weight_start_idx = e * kernel_volume;

            for (size_t i = 0; i < kernel_volume; ++i) {
                // This is the core dot product operation (sum(Input * Weight))
                // Array indexing [] syntax works with raw pointers
                bf16 input_val = hidden_states[input_start_idx + i];
                bf16 weight_val = weights[weight_start_idx + i];
                sum += input_val * weight_val;
            }

            // Add bias and store the final result
            // Note: The assignment performs the cast/conversion back to bfloat16_t (float in this demo)
            output[b * embed_dim + e] = sum + (bias ? (float)bias[e] : 0.0f);
        }
    }

    return output;
}

/**
 * @brief Performs a batched 3D convolution patch embedding using AVX-512 intrinsics.
 *
 * This function computes: output[b][e] = sum(hidden_states[b][i] * weights[e][i]) + bias[e]
 * where the sum runs over the 'kernel_volume'.
 *
 * @param hidden_states Input tensor (seq_len * kernel_volume)
 * @param weights Convolution kernels (embed_dim * kernel_volume)
 * @param bias Bias vector (embed_dim), or nullptr if the conv has no bias
 * @param seq_len Batch size (B)
 * @param in_channels Input channels (C_in)
 * @param embed_dim Output embedding dimension (Ed) / Output channels (C_out)
 * @param temporal_patch_size Time dimension of the patch (Kt)
 * @param patch_size Spatial dimension of the patch (Ks)
 * @return std::vector<bf16> The resulting embeddings (seq_len * embed_dim)
 */
std::vector<bf16> conv3d_patch_embed_avx512(
    const bf16* hidden_states,
    const bf16* weights,
    const bf16* bias,
    size_t seq_len,
    size_t in_channels,
    size_t embed_dim,
    size_t temporal_patch_size,
    size_t patch_size)
{
    // Total number of elements in a single input patch / single convolution kernel
    const size_t kernel_volume = in_channels * temporal_patch_size * patch_size * patch_size;
    const size_t VEC_WIDTH = 16; // AVX-512 processes 16 32-bit floats at once

    // NOTE: Size checks are disabled as inputs are raw pointers (*), which do not carry size information.
    std::vector<bf16> output(seq_len * embed_dim);

    // Iterate through each batch element (B)
    for (size_t b = 0; b < seq_len; ++b) {
        // Iterate through each output embedding dimension (Ed)
        for (size_t e = 0; e < embed_dim; ++e) {

            // Initialize the 16-element vector accumulator for the dot product
            __m512 sum_vec = _mm512_setzero_ps();

            // Pointers to the start of the current input patch and weight kernel
            const bf16* input_ptr = hidden_states + b * kernel_volume;
            const bf16* weight_ptr = weights + e * kernel_volume;

            // 1. Vectorized loop (16 elements per iteration)
            size_t i = 0;
            for (; i + VEC_WIDTH <= kernel_volume; i += VEC_WIDTH) {
                // Load 16 bfloat16 elements (256 bits) from input and weights
                // Note: bfloat16 is 16-bit, so we load 256 bits (not 512)
                __m256i input_bf16 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input_ptr + i));
                __m256i weight_bf16 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(weight_ptr + i));

                // Convert bfloat16 to float32 by shifting left 16 bits (bfloat16 is just the upper 16 bits of float32)
                __m512i input_i32 = _mm512_cvtepu16_epi32(input_bf16);
                __m512i weight_i32 = _mm512_cvtepu16_epi32(weight_bf16);

                __m512 input_v = _mm512_castsi512_ps(_mm512_slli_epi32(input_i32, 16));
                __m512 weight_v = _mm512_castsi512_ps(_mm512_slli_epi32(weight_i32, 16));

                // Use Fused Multiply-Add (FMA) for higher efficiency and precision than separate MUL and ADD
                // sum_vec = (input_v * weight_v) + sum_vec
                sum_vec = _mm512_fmadd_ps(input_v, weight_v, sum_vec);
            }

            // 2. Horizontal sum of the vector accumulator to get the final float sum
            // _mm512_reduce_add_ps sums all 16 elements in the vector into a single scalar float.
            float final_sum = _mm512_reduce_add_ps(sum_vec);

            // 3. Handle the tail (remaining elements, less than 16)
            for (; i < kernel_volume; ++i) {
                final_sum += input_ptr[i] * weight_ptr[i];
            }

            // 4. Add bias and store the final result
            output[b * embed_dim + e] = final_sum + (bias ? (float)bias[e] : 0.0f);
        }
    }

    return output;
}

/**
 * @brief Optimized multi-threaded version with loop unrolling and prefetching.
 *
 * Optimizations applied:
 * - Multi-threading with OpenMP (up to 4 threads)
 * - Loop unrolling (2x) to reduce loop overhead
 * - Software prefetching for better cache utilization
 * - Better memory access patterns
 *
 * @param hidden_states Input tensor (seq_len * kernel_volume)
 * @param weights Convolution kernels (embed_dim * kernel_volume)
 * @param bias Bias vector (embed_dim), or nullptr if the conv has no bias
 * @param seq_len Batch size (B)
 * @param in_channels Input channels (C_in)
 * @param embed_dim Output embedding dimension (Ed) / Output channels (C_out)
 * @param temporal_patch_size Time dimension of the patch (Kt)
 * @param patch_size Spatial dimension of the patch (Ks)
 * @return std::vector<bf16> The resulting embeddings (seq_len * embed_dim)
 */
std::vector<bf16> conv3d_patch_embed_avx512_optimized(
    const bf16* hidden_states,
    const bf16* weights,
    const bf16* bias,
    size_t seq_len,
    size_t in_channels,
    size_t embed_dim,
    size_t temporal_patch_size,
    size_t patch_size)
{
    const size_t kernel_volume = in_channels * temporal_patch_size * patch_size * patch_size;
    const size_t VEC_WIDTH = 16;
    const int PREFETCH_DISTANCE = 256; // Prefetch 256 bytes ahead

    std::vector<bf16> output(seq_len * embed_dim);

    // Parallelize over batch dimension with OpenMP, max 4 threads
    // Note: OpenMP requires signed integral type for loop index
    const int seq_len_signed = static_cast<int>(seq_len);
    #pragma omp parallel for num_threads(4) schedule(dynamic)
    for (int b = 0; b < seq_len_signed; ++b) {
        const bf16* input_base = hidden_states + b * kernel_volume;
        bf16* output_base = output.data() + b * embed_dim;

        for (size_t e = 0; e < embed_dim; ++e) {
            __m512 sum_vec1 = _mm512_setzero_ps();
            __m512 sum_vec2 = _mm512_setzero_ps();

            const bf16* input_ptr = input_base;
            const bf16* weight_ptr = weights + e * kernel_volume;

            // Prefetch next weight row if processing multiple embeddings
            if (e + 1 < embed_dim) {
                _mm_prefetch(reinterpret_cast<const char*>(weights + (e + 1) * kernel_volume), _MM_HINT_T0);
            }

            // Unrolled vectorized loop (2x unroll, processing 32 elements per iteration)
            size_t i = 0;
            for (; i + 2 * VEC_WIDTH <= kernel_volume; i += 2 * VEC_WIDTH) {
                // Prefetch ahead
                _mm_prefetch(reinterpret_cast<const char*>(input_ptr + i + PREFETCH_DISTANCE), _MM_HINT_T0);
                _mm_prefetch(reinterpret_cast<const char*>(weight_ptr + i + PREFETCH_DISTANCE), _MM_HINT_T0);

                // First iteration (elements i to i+15)
                __m256i input_bf16_1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input_ptr + i));
                __m256i weight_bf16_1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(weight_ptr + i));

                __m512i input_i32_1 = _mm512_cvtepu16_epi32(input_bf16_1);
                __m512i weight_i32_1 = _mm512_cvtepu16_epi32(weight_bf16_1);

                __m512 input_v1 = _mm512_castsi512_ps(_mm512_slli_epi32(input_i32_1, 16));
                __m512 weight_v1 = _mm512_castsi512_ps(_mm512_slli_epi32(weight_i32_1, 16));

                sum_vec1 = _mm512_fmadd_ps(input_v1, weight_v1, sum_vec1);

                // Second iteration (elements i+16 to i+31)
                __m256i input_bf16_2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input_ptr + i + VEC_WIDTH));
                __m256i weight_bf16_2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(weight_ptr + i + VEC_WIDTH));

                __m512i input_i32_2 = _mm512_cvtepu16_epi32(input_bf16_2);
                __m512i weight_i32_2 = _mm512_cvtepu16_epi32(weight_bf16_2);

                __m512 input_v2 = _mm512_castsi512_ps(_mm512_slli_epi32(input_i32_2, 16));
                __m512 weight_v2 = _mm512_castsi512_ps(_mm512_slli_epi32(weight_i32_2, 16));

                sum_vec2 = _mm512_fmadd_ps(input_v2, weight_v2, sum_vec2);
            }

            // Handle remaining full vectors (16 elements)
            for (; i + VEC_WIDTH <= kernel_volume; i += VEC_WIDTH) {
                __m256i input_bf16 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input_ptr + i));
                __m256i weight_bf16 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(weight_ptr + i));

                __m512i input_i32 = _mm512_cvtepu16_epi32(input_bf16);
                __m512i weight_i32 = _mm512_cvtepu16_epi32(weight_bf16);

                __m512 input_v = _mm512_castsi512_ps(_mm512_slli_epi32(input_i32, 16));
                __m512 weight_v = _mm512_castsi512_ps(_mm512_slli_epi32(weight_i32, 16));

                sum_vec1 = _mm512_fmadd_ps(input_v, weight_v, sum_vec1);
            }

            // Combine the two accumulators
            sum_vec1 = _mm512_add_ps(sum_vec1, sum_vec2);
            float final_sum = _mm512_reduce_add_ps(sum_vec1);

            // Handle tail elements
            for (; i < kernel_volume; ++i) {
                final_sum += input_ptr[i] * weight_ptr[i];
            }

            // Add bias and store result
            output_base[e] = final_sum + (bias ? (float)bias[e] : 0.0f);
        }
    }

    return output;
}

/**
 * @brief Maximum performance version with 4x loop unrolling and aggressive optimizations.
 *
 * Optimizations applied:
 * - Multi-threading with OpenMP (up to 4 threads)
 * - 4x loop unrolling (64 elements per iteration) for maximum throughput
 * - Four independent accumulator chains to maximize instruction-level parallelism
 * - Aggressive software prefetching
 * - Optimized for modern CPUs with deep pipelines
 *
 * @param hidden_states Input tensor (seq_len * kernel_volume)
 * @param weights Convolution kernels (embed_dim * kernel_volume)
 * @param bias Bias vector (embed_dim), or nullptr if the conv has no bias
 * @param seq_len Batch size (B)
 * @param in_channels Input channels (C_in)
 * @param embed_dim Output embedding dimension (Ed) / Output channels (C_out)
 * @param temporal_patch_size Time dimension of the patch (Kt)
 * @param patch_size Spatial dimension of the patch (Ks)
 * @return std::vector<bf16> The resulting embeddings (seq_len * embed_dim)
 */
std::vector<bf16> conv3d_patch_embed_avx512_max_performance(
    const bf16* hidden_states,
    const bf16* weights,
    const bf16* bias,
    size_t seq_len,
    size_t in_channels,
    size_t embed_dim,
    size_t temporal_patch_size,
    size_t patch_size)
{
    const size_t kernel_volume = in_channels * temporal_patch_size * patch_size * patch_size;
    const size_t VEC_WIDTH = 16;
    const int PREFETCH_DISTANCE = 512; // Larger prefetch distance for aggressive pipelining

    std::vector<bf16> output(seq_len * embed_dim);

    // Parallelize over batch dimension with OpenMP, max 4 threads
    const int seq_len_signed = static_cast<int>(seq_len);
    #pragma omp parallel for num_threads(4) schedule(dynamic)
    for (int b = 0; b < seq_len_signed; ++b) {
        const bf16* input_base = hidden_states + b * kernel_volume;
        bf16* output_base = output.data() + b * embed_dim;

        for (size_t e = 0; e < embed_dim; ++e) {
            // Four independent accumulator chains for maximum ILP
            __m512 sum_vec1 = _mm512_setzero_ps();
            __m512 sum_vec2 = _mm512_setzero_ps();
            __m512 sum_vec3 = _mm512_setzero_ps();
            __m512 sum_vec4 = _mm512_setzero_ps();

            const bf16* input_ptr = input_base;
            const bf16* weight_ptr = weights + e * kernel_volume;

            // Prefetch next 2 weight rows for better cache utilization
            if (e + 1 < embed_dim) {
                _mm_prefetch(reinterpret_cast<const char*>(weights + (e + 1) * kernel_volume), _MM_HINT_T0);
            }
            if (e + 2 < embed_dim) {
                _mm_prefetch(reinterpret_cast<const char*>(weights + (e + 2) * kernel_volume), _MM_HINT_T1);
            }

            // 4x unrolled vectorized loop (processing 64 elements per iteration)
            size_t i = 0;
            for (; i + 4 * VEC_WIDTH <= kernel_volume; i += 4 * VEC_WIDTH) {
                // Aggressive prefetching
                _mm_prefetch(reinterpret_cast<const char*>(input_ptr + i + PREFETCH_DISTANCE), _MM_HINT_T0);
                _mm_prefetch(reinterpret_cast<const char*>(weight_ptr + i + PREFETCH_DISTANCE), _MM_HINT_T0);

                // Process 4 vectors (64 elements total)
                __m256i input_bf16_1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input_ptr + i));
                __m256i weight_bf16_1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(weight_ptr + i));
                __m512i input_i32_1 = _mm512_cvtepu16_epi32(input_bf16_1);
                __m512i weight_i32_1 = _mm512_cvtepu16_epi32(weight_bf16_1);
                __m512 input_v1 = _mm512_castsi512_ps(_mm512_slli_epi32(input_i32_1, 16));
                __m512 weight_v1 = _mm512_castsi512_ps(_mm512_slli_epi32(weight_i32_1, 16));
                sum_vec1 = _mm512_fmadd_ps(input_v1, weight_v1, sum_vec1);

                __m256i input_bf16_2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input_ptr + i + VEC_WIDTH));
                __m256i weight_bf16_2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(weight_ptr + i + VEC_WIDTH));
                __m512i input_i32_2 = _mm512_cvtepu16_epi32(input_bf16_2);
                __m512i weight_i32_2 = _mm512_cvtepu16_epi32(weight_bf16_2);
                __m512 input_v2 = _mm512_castsi512_ps(_mm512_slli_epi32(input_i32_2, 16));
                __m512 weight_v2 = _mm512_castsi512_ps(_mm512_slli_epi32(weight_i32_2, 16));
                sum_vec2 = _mm512_fmadd_ps(input_v2, weight_v2, sum_vec2);

                __m256i input_bf16_3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input_ptr + i + 2 * VEC_WIDTH));
                __m256i weight_bf16_3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(weight_ptr + i + 2 * VEC_WIDTH));
                __m512i input_i32_3 = _mm512_cvtepu16_epi32(input_bf16_3);
                __m512i weight_i32_3 = _mm512_cvtepu16_epi32(weight_bf16_3);
                __m512 input_v3 = _mm512_castsi512_ps(_mm512_slli_epi32(input_i32_3, 16));
                __m512 weight_v3 = _mm512_castsi512_ps(_mm512_slli_epi32(weight_i32_3, 16));
                sum_vec3 = _mm512_fmadd_ps(input_v3, weight_v3, sum_vec3);

                __m256i input_bf16_4 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input_ptr + i + 3 * VEC_WIDTH));
                __m256i weight_bf16_4 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(weight_ptr + i + 3 * VEC_WIDTH));
                __m512i input_i32_4 = _mm512_cvtepu16_epi32(input_bf16_4);
                __m512i weight_i32_4 = _mm512_cvtepu16_epi32(weight_bf16_4);
                __m512 input_v4 = _mm512_castsi512_ps(_mm512_slli_epi32(input_i32_4, 16));
                __m512 weight_v4 = _mm512_castsi512_ps(_mm512_slli_epi32(weight_i32_4, 16));
                sum_vec4 = _mm512_fmadd_ps(input_v4, weight_v4, sum_vec4);
            }

            // Handle remaining 2x unrolled iterations
            for (; i + 2 * VEC_WIDTH <= kernel_volume; i += 2 * VEC_WIDTH) {
                __m256i input_bf16_1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input_ptr + i));
                __m256i weight_bf16_1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(weight_ptr + i));
                __m512i input_i32_1 = _mm512_cvtepu16_epi32(input_bf16_1);
                __m512i weight_i32_1 = _mm512_cvtepu16_epi32(weight_bf16_1);
                __m512 input_v1 = _mm512_castsi512_ps(_mm512_slli_epi32(input_i32_1, 16));
                __m512 weight_v1 = _mm512_castsi512_ps(_mm512_slli_epi32(weight_i32_1, 16));
                sum_vec1 = _mm512_fmadd_ps(input_v1, weight_v1, sum_vec1);

                __m256i input_bf16_2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input_ptr + i + VEC_WIDTH));
                __m256i weight_bf16_2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(weight_ptr + i + VEC_WIDTH));
                __m512i input_i32_2 = _mm512_cvtepu16_epi32(input_bf16_2);
                __m512i weight_i32_2 = _mm512_cvtepu16_epi32(weight_bf16_2);
                __m512 input_v2 = _mm512_castsi512_ps(_mm512_slli_epi32(input_i32_2, 16));
                __m512 weight_v2 = _mm512_castsi512_ps(_mm512_slli_epi32(weight_i32_2, 16));
                sum_vec2 = _mm512_fmadd_ps(input_v2, weight_v2, sum_vec2);
            }

            // Handle remaining single vectors
            for (; i + VEC_WIDTH <= kernel_volume; i += VEC_WIDTH) {
                __m256i input_bf16 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input_ptr + i));
                __m256i weight_bf16 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(weight_ptr + i));
                __m512i input_i32 = _mm512_cvtepu16_epi32(input_bf16);
                __m512i weight_i32 = _mm512_cvtepu16_epi32(weight_bf16);
                __m512 input_v = _mm512_castsi512_ps(_mm512_slli_epi32(input_i32, 16));
                __m512 weight_v = _mm512_castsi512_ps(_mm512_slli_epi32(weight_i32, 16));
                sum_vec1 = _mm512_fmadd_ps(input_v, weight_v, sum_vec1);
            }

            // Combine all four accumulators
            sum_vec1 = _mm512_add_ps(sum_vec1, sum_vec2);
            sum_vec3 = _mm512_add_ps(sum_vec3, sum_vec4);
            sum_vec1 = _mm512_add_ps(sum_vec1, sum_vec3);
            float final_sum = _mm512_reduce_add_ps(sum_vec1);

            // Handle tail elements
            for (; i < kernel_volume; ++i) {
                final_sum += input_ptr[i] * weight_ptr[i];
            }

            // Add bias and store result
            output_base[e] = final_sum + (bias ? (float)bias[e] : 0.0f);
        }
    }

    return output;
}
