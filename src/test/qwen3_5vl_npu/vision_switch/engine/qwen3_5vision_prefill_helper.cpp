#include "vision/vision_prefill_helper.hpp"

#include "utils/avx512_util.hpp"
#include <immintrin.h>
#include <stdfloat>
#include <vector>
#include <bit> // std::bit_cast (C++20+)
#include <cstdint>
#include <omp.h> // OpenMP for multi-threading

void simd_add(
    bf16* input1,
    bf16* input2,
    bf16* output,
    size_t size
){
    constexpr size_t SIMD_WIDTH = 16;
    constexpr size_t UNROLL_FACTOR = 4; // Process 64 elements per iteration
    constexpr size_t CHUNK_SIZE = SIMD_WIDTH * UNROLL_FACTOR; // 64 elements

    // Use OpenMP only if size is large enough to benefit from parallelization
    // Threshold: 8 chunks (512 elements) per thread minimum to avoid overhead
    constexpr size_t min_elements_per_thread = CHUNK_SIZE * 8;
    constexpr int max_threads = 4;
    const bool use_parallel = size >= (min_elements_per_thread * 2);

    // Calculate number of full chunks for parallel processing
    const size_t num_chunks = size / CHUNK_SIZE;
    const int signed_num_chunks = static_cast<int>(num_chunks);

    // Process full chunks with OpenMP (chunked distribution for cache locality)
    #pragma omp parallel for num_threads(max_threads) if(use_parallel) schedule(static)
    for (int chunk_idx = 0; chunk_idx < signed_num_chunks; ++chunk_idx) {
        size_t i = static_cast<size_t>(chunk_idx) * CHUNK_SIZE;

        // Prefetch next cache lines
        _mm_prefetch(reinterpret_cast<const char*>(input1 + i + 64), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(input2 + i + 64), _MM_HINT_T0);

        // Load all inputs first (better for out-of-order execution)
        __m256i a_bh_vec0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input1 + i));
        __m256i b_bh_vec0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input2 + i));

        __m256i a_bh_vec1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input1 + i + 16));
        __m256i b_bh_vec1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input2 + i + 16));

        __m256i a_bh_vec2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input1 + i + 32));
        __m256i b_bh_vec2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input2 + i + 32));

        __m256i a_bh_vec3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input1 + i + 48));
        __m256i b_bh_vec3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input2 + i + 48));

        // Convert to 32-bit and shift (interleaved for better pipeline usage)
        __m512i a_shifted0 = _mm512_slli_epi32(_mm512_cvtepu16_epi32(a_bh_vec0), 16);
        __m512i b_shifted0 = _mm512_slli_epi32(_mm512_cvtepu16_epi32(b_bh_vec0), 16);

        __m512i a_shifted1 = _mm512_slli_epi32(_mm512_cvtepu16_epi32(a_bh_vec1), 16);
        __m512i b_shifted1 = _mm512_slli_epi32(_mm512_cvtepu16_epi32(b_bh_vec1), 16);

        __m512i a_shifted2 = _mm512_slli_epi32(_mm512_cvtepu16_epi32(a_bh_vec2), 16);
        __m512i b_shifted2 = _mm512_slli_epi32(_mm512_cvtepu16_epi32(b_bh_vec2), 16);

        __m512i a_shifted3 = _mm512_slli_epi32(_mm512_cvtepu16_epi32(a_bh_vec3), 16);
        __m512i b_shifted3 = _mm512_slli_epi32(_mm512_cvtepu16_epi32(b_bh_vec3), 16);

        // Add as floats
        __m512 sum0 = _mm512_add_ps(_mm512_castsi512_ps(a_shifted0), _mm512_castsi512_ps(b_shifted0));
        __m512 sum1 = _mm512_add_ps(_mm512_castsi512_ps(a_shifted1), _mm512_castsi512_ps(b_shifted1));
        __m512 sum2 = _mm512_add_ps(_mm512_castsi512_ps(a_shifted2), _mm512_castsi512_ps(b_shifted2));
        __m512 sum3 = _mm512_add_ps(_mm512_castsi512_ps(a_shifted3), _mm512_castsi512_ps(b_shifted3));

        // Convert back to bfloat16 and store
        __m256i bf16_0 = _mm512_cvtepi32_epi16(_mm512_srli_epi32(_mm512_castps_si512(sum0), 16));
        __m256i bf16_1 = _mm512_cvtepi32_epi16(_mm512_srli_epi32(_mm512_castps_si512(sum1), 16));
        __m256i bf16_2 = _mm512_cvtepi32_epi16(_mm512_srli_epi32(_mm512_castps_si512(sum2), 16));
        __m256i bf16_3 = _mm512_cvtepi32_epi16(_mm512_srli_epi32(_mm512_castps_si512(sum3), 16));

        _mm256_storeu_si256(reinterpret_cast<__m256i*>(output + i), bf16_0);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(output + i + 16), bf16_1);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(output + i + 32), bf16_2);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(output + i + 48), bf16_3);
    }

    // Process remaining elements (remainder after full chunks)
    size_t i = num_chunks * CHUNK_SIZE;

    // Process remaining 16-element chunks
    for (; i + SIMD_WIDTH <= size; i += SIMD_WIDTH) {
        __m256i a_bh_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input1 + i));
        __m256i b_bh_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input2 + i));

        __m512i a_shifted = _mm512_slli_epi32(_mm512_cvtepu16_epi32(a_bh_vec), 16);
        __m512i b_shifted = _mm512_slli_epi32(_mm512_cvtepu16_epi32(b_bh_vec), 16);

        __m512 sum = _mm512_add_ps(_mm512_castsi512_ps(a_shifted), _mm512_castsi512_ps(b_shifted));

        __m256i bf16_data = _mm512_cvtepi32_epi16(_mm512_srli_epi32(_mm512_castps_si512(sum), 16));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(output + i), bf16_data);
    }

    // Scalar tail
    for (; i < size; ++i) {
        output[i] = static_cast<bf16>(static_cast<float>(input1[i]) + static_cast<float>(input2[i]));
    }
}

void simd_add(
    const float* input1,
    const bf16* input2,
    float* output,
    size_t size
) {
    size_t i = 0;

#if defined(__AVX512BF16__)
    for (; i + 15 < size; i += 16) {
        // Load 16 bf16 values (256 bits) from input2
        __m256i b_bh_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input2 + i));

        // Convert bf16 → float32
        __m512 b_ps_vec = _mm512_cvtpbh_ps(reinterpret_cast<__m256bh>(b_bh_vec));

        // Load 16 float32 values from input1
        __m512 a_ps_vec = _mm512_loadu_ps(input1 + i);

        // Add them
        __m512 sum_ps_vec = _mm512_add_ps(a_ps_vec, b_ps_vec);

        // Store result as float32
        _mm512_storeu_ps(output + i, sum_ps_vec);
    }
#else
    // Fallback: emulate conversion manually if AVX-512BF16 not available
    for (; i + 15 < size; i += 16) {
        // Load 16 bf16s as 16-bit ints
        __m256i b_bh_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input2 + i));
        // Zero-extend to 32 bits
        __m512i b_int_vec = _mm512_cvtepu16_epi32(b_bh_vec);
        // Shift left to form valid float32 representation
        __m512i b_shifted = _mm512_slli_epi32(b_int_vec, 16);
        __m512 b_ps_vec = _mm512_castsi512_ps(b_shifted);

        // Load 16 floats
        __m512 a_ps_vec = _mm512_loadu_ps(input1 + i);
        // Add
        __m512 sum_ps_vec = _mm512_add_ps(a_ps_vec, b_ps_vec);
        // Store
        _mm512_storeu_ps(output + i, sum_ps_vec);
    }
#endif

    // Scalar tail
    for (; i < size; ++i) {
        output[i] = input1[i] + static_cast<float>(input2[i]);
    }
}

void simd_add(
    const float* input1,
    const bf16* input2,
    bf16* output,
    size_t size
) {
    size_t i = 0;
    constexpr size_t SIMD_WIDTH = 16;
    constexpr size_t UNROLL_FACTOR = 4; // Process 64 elements per iteration

    // Process 64 elements at a time (4x unrolled)
    for (; i + SIMD_WIDTH * UNROLL_FACTOR <= size; i += SIMD_WIDTH * UNROLL_FACTOR) {
        // Prefetch next cache lines
        _mm_prefetch(reinterpret_cast<const char*>(input1 + i + 64), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(input2 + i + 64), _MM_HINT_T0);

        // Load all data
        __m512 a_vec0 = _mm512_loadu_ps(input1 + i);
        __m512 b_vec0 = load_bfloat16_to_m512(input2 + i);

        __m512 a_vec1 = _mm512_loadu_ps(input1 + i + 16);
        __m512 b_vec1 = load_bfloat16_to_m512(input2 + i + 16);

        __m512 a_vec2 = _mm512_loadu_ps(input1 + i + 32);
        __m512 b_vec2 = load_bfloat16_to_m512(input2 + i + 32);

        __m512 a_vec3 = _mm512_loadu_ps(input1 + i + 48);
        __m512 b_vec3 = load_bfloat16_to_m512(input2 + i + 48);

        // Add and store
        store_m512_to_bfloat16_rne(output + i, _mm512_add_ps(a_vec0, b_vec0));
        store_m512_to_bfloat16_rne(output + i + 16, _mm512_add_ps(a_vec1, b_vec1));
        store_m512_to_bfloat16_rne(output + i + 32, _mm512_add_ps(a_vec2, b_vec2));
        store_m512_to_bfloat16_rne(output + i + 48, _mm512_add_ps(a_vec3, b_vec3));
    }

    // Process remaining 16-element chunks
    for (; i + SIMD_WIDTH <= size; i += SIMD_WIDTH) {
        __m512 a_vec = _mm512_loadu_ps(input1 + i);
        __m512 b_vec = load_bfloat16_to_m512(input2 + i);
        __m512 sum_vec = _mm512_add_ps(a_vec, b_vec);
        store_m512_to_bfloat16_rne(output + i, sum_vec);
    }

    // Scalar tail
    for (; i < size; ++i) {
        float sum = input1[i] + static_cast<float>(input2[i]);
        output[i] = static_cast<bf16>(sum);
    }
}

void simd_add(
    const bf16* input1,
    const bf16* input2,
    float* output,
    size_t size
) {
    size_t i = 0;
    constexpr size_t SIMD_WIDTH = 16;
    constexpr size_t UNROLL_FACTOR = 4; // Process 64 elements per iteration

    // Process 64 elements at a time (4x unrolled)
    for (; i + SIMD_WIDTH * UNROLL_FACTOR <= size; i += SIMD_WIDTH * UNROLL_FACTOR) {
        // Prefetch next cache lines
        _mm_prefetch(reinterpret_cast<const char*>(input1 + i + 64), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(input2 + i + 64), _MM_HINT_T0);

        // Load and compute in parallel
        __m512 a_vec0 = load_bfloat16_to_m512(input1 + i);
        __m512 b_vec0 = load_bfloat16_to_m512(input2 + i);

        __m512 a_vec1 = load_bfloat16_to_m512(input1 + i + 16);
        __m512 b_vec1 = load_bfloat16_to_m512(input2 + i + 16);

        __m512 a_vec2 = load_bfloat16_to_m512(input1 + i + 32);
        __m512 b_vec2 = load_bfloat16_to_m512(input2 + i + 32);

        __m512 a_vec3 = load_bfloat16_to_m512(input1 + i + 48);
        __m512 b_vec3 = load_bfloat16_to_m512(input2 + i + 48);

        // Add and store as float32
        _mm512_storeu_ps(output + i, _mm512_add_ps(a_vec0, b_vec0));
        _mm512_storeu_ps(output + i + 16, _mm512_add_ps(a_vec1, b_vec1));
        _mm512_storeu_ps(output + i + 32, _mm512_add_ps(a_vec2, b_vec2));
        _mm512_storeu_ps(output + i + 48, _mm512_add_ps(a_vec3, b_vec3));
    }

    // Process remaining 16-element chunks
    for (; i + SIMD_WIDTH <= size; i += SIMD_WIDTH) {
        __m512 a_vec = load_bfloat16_to_m512(input1 + i);
        __m512 b_vec = load_bfloat16_to_m512(input2 + i);
        __m512 sum_vec = _mm512_add_ps(a_vec, b_vec);
        _mm512_storeu_ps(output + i, sum_vec);
    }

    // Scalar tail
    for (; i < size; ++i) {
        float sum = static_cast<float>(input1[i]) + static_cast<float>(input2[i]);
        output[i] = sum;
    }
}

// Optimized AVX-512 version that combines bias addition and GELU activation
// This version processes the entire array in one pass, reducing memory traffic
// Note: bias is broadcast across all sequence positions (hidden_dim sized, repeated for seq_len)
// Optimizations:
// - OpenMP parallelization (4 threads) for sequence positions
// - Prefetching for cache optimization
// - 2x loop unrolling for better ILP (instruction-level parallelism)
void simd_bias_add_gelu(
    bf16* input,
    const bf16* bias,
    bf16* output,
    size_t total_size,
    size_t hidden_dim
){
    size_t seq_len = total_size / hidden_dim;
    constexpr size_t SIMD_WIDTH = 16;
    constexpr size_t UNROLL_FACTOR = 2;

    // Use signed integer for OpenMP compatibility
    int signed_seq_len = static_cast<int>(seq_len);

    // Parallelize over sequence positions with OpenMP (max 4 threads)
    // Each thread processes different sequence positions independently
    #pragma omp parallel for num_threads(4) schedule(static)
    for (int seq = 0; seq < signed_seq_len; ++seq) {
        size_t seq_offset = static_cast<size_t>(seq) * hidden_dim;
        size_t i = 0;

        // Process 32 elements at a time (2x unrolled for better ILP)
        for (; i + SIMD_WIDTH * UNROLL_FACTOR <= hidden_dim; i += SIMD_WIDTH * UNROLL_FACTOR) {
            // Prefetch next cache lines for input, bias, and output
            _mm_prefetch(reinterpret_cast<const char*>(input + seq_offset + i + 64), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(bias + i + 64), _MM_HINT_T0);

            // First iteration (elements i to i+15)
            __m512 input_vec1 = load_bfloat16_to_m512(input + seq_offset + i);
            __m512 bias_vec1 = load_bfloat16_to_m512(bias + i);

            // Second iteration (elements i+16 to i+31) - load in parallel to hide latency
            __m512 input_vec2 = load_bfloat16_to_m512(input + seq_offset + i + SIMD_WIDTH);
            __m512 bias_vec2 = load_bfloat16_to_m512(bias + i + SIMD_WIDTH);

            // Compute first iteration
            __m512 sum_vec1 = _mm512_add_ps(input_vec1, bias_vec1);

            // Compute second iteration (parallel to first GELU computation)
            __m512 sum_vec2 = _mm512_add_ps(input_vec2, bias_vec2);

            // Apply GELU activation (computationally expensive, so interleave)
            __m512 gelu_vec1 = gelu_tanh_avx512_simd(sum_vec1);
            __m512 gelu_vec2 = gelu_tanh_avx512_simd(sum_vec2);

            // Store results
            store_m512_to_bfloat16_rne(output + seq_offset + i, gelu_vec1);
            store_m512_to_bfloat16_rne(output + seq_offset + i + SIMD_WIDTH, gelu_vec2);
        }

        // Process remaining 16-element chunks
        for (; i + SIMD_WIDTH <= hidden_dim; i += SIMD_WIDTH) {
            __m512 input_vec = load_bfloat16_to_m512(input + seq_offset + i);
            __m512 bias_vec = load_bfloat16_to_m512(bias + i);

            __m512 sum_vec = _mm512_add_ps(input_vec, bias_vec);
            __m512 gelu_vec = gelu_tanh_avx512_simd(sum_vec);

            store_m512_to_bfloat16_rne(output + seq_offset + i, gelu_vec);
        }

        // Handle remaining elements with scalar loop
        constexpr float sqrt_2_over_pi = 0.7978845608f;  // √(2/π)
        constexpr float coeff = 0.044715f;

        for (; i < hidden_dim; ++i) {
            float x = static_cast<float>(input[seq_offset + i]);
            float b = static_cast<float>(bias[i]);
            float sum = x + b;

            // GELU approximation (tanh-based)
            float x_cubed = sum * sum * sum;
            float inner = sqrt_2_over_pi * (sum + coeff * x_cubed);
            float tanh_val = std::tanh(inner);
            float gelu = 0.5f * sum * (1.0f + tanh_val);

            output[seq_offset + i] = static_cast<bf16>(gelu);
        }
    }
}

void gelu_bfloat16_ref(
    const bf16* input,
    bf16* output,
    size_t size
) {
    size_t i = 0;
    constexpr size_t SIMD_WIDTH = 16;
    constexpr size_t UNROLL_FACTOR = 2; // Process 32 elements per iteration

    // Process 32 elements at a time (2x unrolled for GELU's computational intensity)
    for (; i + SIMD_WIDTH * UNROLL_FACTOR <= size; i += SIMD_WIDTH * UNROLL_FACTOR) {
        // Prefetch next cache line
        _mm_prefetch(reinterpret_cast<const char*>(input + i + 64), _MM_HINT_T0);

        // Load both sets of data
        __m512 input_vec1 = load_bfloat16_to_m512(input + i);
        __m512 input_vec2 = load_bfloat16_to_m512(input + i + SIMD_WIDTH);

        // Apply GELU activation to both
        __m512 gelu_vec1 = gelu_tanh_avx512_simd(input_vec1);
        __m512 gelu_vec2 = gelu_tanh_avx512_simd(input_vec2);

        // Store results
        store_m512_to_bfloat16_rne(output + i, gelu_vec1);
        store_m512_to_bfloat16_rne(output + i + SIMD_WIDTH, gelu_vec2);
    }

    // Process remaining 16-element chunks
    for (; i + SIMD_WIDTH <= size; i += SIMD_WIDTH) {
        __m512 input_vec = load_bfloat16_to_m512(input + i);
        __m512 gelu_vec = gelu_tanh_avx512_simd(input_vec);
        store_m512_to_bfloat16_rne(output + i, gelu_vec);
    }

    // Constants for the GELU approximation (tanh-based)
    constexpr double sqrt_2_over_pi = 0.7978845608028654;  // √(2/π)
    constexpr double coeff = 0.044715;

    // Scalar tail
    for (; i < size; ++i) {
        double x = static_cast<double>(input[i]);
        double x_cubed = x * x * x;
        double inner = sqrt_2_over_pi * (x + coeff * x_cubed);
        double tanh_val = std::tanh(inner);
        double gelu = 0.5 * x * (1.0 + tanh_val);
        output[i] = static_cast<bf16>(static_cast<float>(gelu));
    }
}
