#include "gemma4e_vision_prefill_helper.hpp"

#include "avx512_util.hpp"
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

    const bool use_parallel = size >= (min_elements_per_thread * 2);

    // Calculate number of full chunks for parallel processing
    const size_t num_chunks = size / CHUNK_SIZE;
    const int signed_num_chunks = static_cast<int>(num_chunks);

    // Process full chunks with OpenMP (chunked distribution for cache locality)
    #pragma omp parallel for num_threads(max_prefill_threads) if(use_parallel) schedule(static)
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

        // Convert back to bfloat16 and store with round-to-nearest-even
        store_m512_to_bfloat16_rne(output + i,      sum0);
        store_m512_to_bfloat16_rne(output + i + 16, sum1);
        store_m512_to_bfloat16_rne(output + i + 32, sum2);
        store_m512_to_bfloat16_rne(output + i + 48, sum3);
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

        store_m512_to_bfloat16_rne(output + i, sum);
    }

    // Scalar tail
    for (; i < size; ++i) {
        output[i] = static_cast<bf16>(static_cast<float>(input1[i]) + static_cast<float>(input2[i]));
    }
}

void transpose_2d(
    const bf16* input,
    bf16* output,
    size_t rows,
    size_t cols
){
    int signed_rows = static_cast<int>(rows);
    #pragma omp parallel for num_threads(max_prefill_threads) schedule(static) if(rows >= 8)
    for(int r = 0; r < signed_rows; r++){
        for(int c = 0; c < static_cast<int>(cols); c++){
            output[c * rows + r] = input[r * cols + c];
        }
    }
}

void simd_add(
    const float* input1,
    const bf16* input2,
    float* output,
    size_t size
) {
    constexpr size_t SIMD_WIDTH = 16;
    constexpr size_t UNROLL_FACTOR = 4;
    constexpr size_t CHUNK_SIZE = SIMD_WIDTH * UNROLL_FACTOR;
    constexpr size_t min_elements_per_thread = CHUNK_SIZE * 8;

    const bool use_parallel = size >= (min_elements_per_thread * 2);
    const size_t num_chunks = size / CHUNK_SIZE;
    const int signed_num_chunks = static_cast<int>(num_chunks);

    #pragma omp parallel for num_threads(max_prefill_threads) if(use_parallel) schedule(static)
    for (int chunk_idx = 0; chunk_idx < signed_num_chunks; ++chunk_idx) {
        size_t base = static_cast<size_t>(chunk_idx) * CHUNK_SIZE;

        _mm_prefetch(reinterpret_cast<const char*>(input1 + base + 64), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(input2 + base + 64), _MM_HINT_T0);

        for (size_t j = 0; j < CHUNK_SIZE; j += SIMD_WIDTH) {
            size_t i = base + j;
            __m512 a_ps_vec = _mm512_loadu_ps(input1 + i);
            __m512 b_ps_vec = load_bfloat16_to_m512(input2 + i);
            _mm512_storeu_ps(output + i, _mm512_add_ps(a_ps_vec, b_ps_vec));
        }
    }

    // Remainder after full chunks
    size_t i = num_chunks * CHUNK_SIZE;
    for (; i + SIMD_WIDTH <= size; i += SIMD_WIDTH) {
        __m512 a_ps_vec = _mm512_loadu_ps(input1 + i);
        __m512 b_ps_vec = load_bfloat16_to_m512(input2 + i);
        _mm512_storeu_ps(output + i, _mm512_add_ps(a_ps_vec, b_ps_vec));
    }

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
    constexpr size_t SIMD_WIDTH = 16;
    constexpr size_t UNROLL_FACTOR = 4;
    constexpr size_t CHUNK_SIZE = SIMD_WIDTH * UNROLL_FACTOR;
    constexpr size_t min_elements_per_thread = CHUNK_SIZE * 8;

    const bool use_parallel = size >= (min_elements_per_thread * 2);
    const size_t num_chunks = size / CHUNK_SIZE;
    const int signed_num_chunks = static_cast<int>(num_chunks);

    #pragma omp parallel for num_threads(max_prefill_threads) if(use_parallel) schedule(static)
    for (int chunk_idx = 0; chunk_idx < signed_num_chunks; ++chunk_idx) {
        size_t base = static_cast<size_t>(chunk_idx) * CHUNK_SIZE;

        _mm_prefetch(reinterpret_cast<const char*>(input1 + base + 64), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(input2 + base + 64), _MM_HINT_T0);

        __m512 a_vec0 = _mm512_loadu_ps(input1 + base);
        __m512 b_vec0 = load_bfloat16_to_m512(input2 + base);
        __m512 a_vec1 = _mm512_loadu_ps(input1 + base + 16);
        __m512 b_vec1 = load_bfloat16_to_m512(input2 + base + 16);
        __m512 a_vec2 = _mm512_loadu_ps(input1 + base + 32);
        __m512 b_vec2 = load_bfloat16_to_m512(input2 + base + 32);
        __m512 a_vec3 = _mm512_loadu_ps(input1 + base + 48);
        __m512 b_vec3 = load_bfloat16_to_m512(input2 + base + 48);

        store_m512_to_bfloat16_rne(output + base,      _mm512_add_ps(a_vec0, b_vec0));
        store_m512_to_bfloat16_rne(output + base + 16, _mm512_add_ps(a_vec1, b_vec1));
        store_m512_to_bfloat16_rne(output + base + 32, _mm512_add_ps(a_vec2, b_vec2));
        store_m512_to_bfloat16_rne(output + base + 48, _mm512_add_ps(a_vec3, b_vec3));
    }

    size_t i = num_chunks * CHUNK_SIZE;
    for (; i + SIMD_WIDTH <= size; i += SIMD_WIDTH) {
        __m512 a_vec = _mm512_loadu_ps(input1 + i);
        __m512 b_vec = load_bfloat16_to_m512(input2 + i);
        store_m512_to_bfloat16_rne(output + i, _mm512_add_ps(a_vec, b_vec));
    }

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
    constexpr size_t SIMD_WIDTH = 16;
    constexpr size_t UNROLL_FACTOR = 4;
    constexpr size_t CHUNK_SIZE = SIMD_WIDTH * UNROLL_FACTOR;
    constexpr size_t min_elements_per_thread = CHUNK_SIZE * 8;

    const bool use_parallel = size >= (min_elements_per_thread * 2);
    const size_t num_chunks = size / CHUNK_SIZE;
    const int signed_num_chunks = static_cast<int>(num_chunks);

    #pragma omp parallel for num_threads(max_prefill_threads) if(use_parallel) schedule(static)
    for (int chunk_idx = 0; chunk_idx < signed_num_chunks; ++chunk_idx) {
        size_t base = static_cast<size_t>(chunk_idx) * CHUNK_SIZE;

        _mm_prefetch(reinterpret_cast<const char*>(input1 + base + 64), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(input2 + base + 64), _MM_HINT_T0);

        __m512 a_vec0 = load_bfloat16_to_m512(input1 + base);
        __m512 b_vec0 = load_bfloat16_to_m512(input2 + base);
        __m512 a_vec1 = load_bfloat16_to_m512(input1 + base + 16);
        __m512 b_vec1 = load_bfloat16_to_m512(input2 + base + 16);
        __m512 a_vec2 = load_bfloat16_to_m512(input1 + base + 32);
        __m512 b_vec2 = load_bfloat16_to_m512(input2 + base + 32);
        __m512 a_vec3 = load_bfloat16_to_m512(input1 + base + 48);
        __m512 b_vec3 = load_bfloat16_to_m512(input2 + base + 48);

        _mm512_storeu_ps(output + base,      _mm512_add_ps(a_vec0, b_vec0));
        _mm512_storeu_ps(output + base + 16, _mm512_add_ps(a_vec1, b_vec1));
        _mm512_storeu_ps(output + base + 32, _mm512_add_ps(a_vec2, b_vec2));
        _mm512_storeu_ps(output + base + 48, _mm512_add_ps(a_vec3, b_vec3));
    }

    size_t i = num_chunks * CHUNK_SIZE;
    for (; i + SIMD_WIDTH <= size; i += SIMD_WIDTH) {
        __m512 a_vec = load_bfloat16_to_m512(input1 + i);
        __m512 b_vec = load_bfloat16_to_m512(input2 + i);
        _mm512_storeu_ps(output + i, _mm512_add_ps(a_vec, b_vec));
    }

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
    #pragma omp parallel for num_threads(max_prefill_threads) schedule(static)
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

// RMS Norm without scale weights
// Input layout: [seq_len_padded x X_padded], processes seq_len rows, X elements per row
// Stride between rows is X_padded
void simd_rms_norm(
    const bf16* input,
    bf16* output,
    size_t seq_len,
    size_t X,
    size_t seq_len_padded,
    size_t X_padded,
    float eps
) {
    constexpr size_t SIMD_WIDTH = 16;
    const float inv_X = 1.0f / static_cast<float>(X);

    int signed_seq_len = static_cast<int>(seq_len);

    #pragma omp parallel for num_threads(max_prefill_threads) schedule(static) if(seq_len >= 8)
    for (int row = 0; row < signed_seq_len; ++row) {
        const bf16* row_in  = input  + static_cast<size_t>(row) * X_padded;
        bf16*       row_out = output + static_cast<size_t>(row) * X_padded;

        // === Pass 1: compute sum of squares ===
        __m512 acc0 = _mm512_setzero_ps();
        __m512 acc1 = _mm512_setzero_ps();
        size_t i = 0;

        // 2x unrolled SIMD loop
        for (; i + SIMD_WIDTH * 2 <= X; i += SIMD_WIDTH * 2) {
            __m512 v0 = load_bfloat16_to_m512(row_in + i);
            __m512 v1 = load_bfloat16_to_m512(row_in + i + SIMD_WIDTH);
            acc0 = _mm512_fmadd_ps(v0, v0, acc0);
            acc1 = _mm512_fmadd_ps(v1, v1, acc1);
        }
        for (; i + SIMD_WIDTH <= X; i += SIMD_WIDTH) {
            __m512 v = load_bfloat16_to_m512(row_in + i);
            acc0 = _mm512_fmadd_ps(v, v, acc0);
        }

        float sum_sq = _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));

        // Scalar tail for sum of squares
        for (; i < X; ++i) {
            float val = static_cast<float>(row_in[i]);
            sum_sq += val * val;
        }

        // mean_sq = sum_sq / X + eps, then rsqrt
        float mean_sq = sum_sq * inv_X + eps;
        float scale = 1.0f / std::sqrt(mean_sq);
        __m512 scale_vec = _mm512_set1_ps(scale);

        // === Pass 2: multiply input by scale ===
        i = 0;
        for (; i + SIMD_WIDTH * 2 <= X; i += SIMD_WIDTH * 2) {
            __m512 v0 = load_bfloat16_to_m512(row_in + i);
            __m512 v1 = load_bfloat16_to_m512(row_in + i + SIMD_WIDTH);
            store_m512_to_bfloat16_rne(row_out + i, _mm512_mul_ps(v0, scale_vec));
            store_m512_to_bfloat16_rne(row_out + i + SIMD_WIDTH, _mm512_mul_ps(v1, scale_vec));
        }
        for (; i + SIMD_WIDTH <= X; i += SIMD_WIDTH) {
            __m512 v = load_bfloat16_to_m512(row_in + i);
            store_m512_to_bfloat16_rne(row_out + i, _mm512_mul_ps(v, scale_vec));
        }

        // Scalar tail
        for (; i < X; ++i) {
            float val = static_cast<float>(row_in[i]);
            row_out[i] = static_cast<bf16>(val * scale);
        }
    }
}

// RMS Norm with scale weights (Gemma4RMSNorm with with_scale=True)
// Input layout: [seq_len_padded x X_padded], processes seq_len rows, X elements per row
// norm_weight: bf16 pointer of size [X], broadcast across all rows
void simd_rms_norm(
    const bf16* input,
    const bf16* norm_weight,
    bf16* output,
    size_t seq_len,
    size_t X,
    size_t seq_len_padded,
    size_t X_padded,
    float eps
) {
    constexpr size_t SIMD_WIDTH = 16;
    const float inv_X = 1.0f / static_cast<float>(X);

    int signed_seq_len = static_cast<int>(seq_len);

    #pragma omp parallel for num_threads(max_prefill_threads) schedule(static) if(seq_len >= 8)
    for (int row = 0; row < signed_seq_len; ++row) {
        const bf16* row_in  = input  + static_cast<size_t>(row) * X_padded;
        bf16*       row_out = output + static_cast<size_t>(row) * X_padded;

        // === Pass 1: compute sum of squares ===
        __m512 acc0 = _mm512_setzero_ps();
        __m512 acc1 = _mm512_setzero_ps();
        size_t i = 0;

        // 2x unrolled SIMD loop
        for (; i + SIMD_WIDTH * 2 <= X; i += SIMD_WIDTH * 2) {
            __m512 v0 = load_bfloat16_to_m512(row_in + i);
            __m512 v1 = load_bfloat16_to_m512(row_in + i + SIMD_WIDTH);
            acc0 = _mm512_fmadd_ps(v0, v0, acc0);
            acc1 = _mm512_fmadd_ps(v1, v1, acc1);
        }
        for (; i + SIMD_WIDTH <= X; i += SIMD_WIDTH) {
            __m512 v = load_bfloat16_to_m512(row_in + i);
            acc0 = _mm512_fmadd_ps(v, v, acc0);
        }

        float sum_sq = _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));

        // Scalar tail for sum of squares
        for (; i < X; ++i) {
            float val = static_cast<float>(row_in[i]);
            sum_sq += val * val;
        }

        // mean_sq = sum_sq / X + eps, then rsqrt
        float mean_sq = sum_sq * inv_X + eps;
        float scale = 1.0f / std::sqrt(mean_sq);
        __m512 scale_vec = _mm512_set1_ps(scale);

        // === Pass 2: multiply input by scale and norm_weight ===
        i = 0;
        for (; i + SIMD_WIDTH * 2 <= X; i += SIMD_WIDTH * 2) {
            __m512 v0 = load_bfloat16_to_m512(row_in + i);
            __m512 w0 = load_bfloat16_to_m512(norm_weight + i);
            __m512 v1 = load_bfloat16_to_m512(row_in + i + SIMD_WIDTH);
            __m512 w1 = load_bfloat16_to_m512(norm_weight + i + SIMD_WIDTH);
            store_m512_to_bfloat16_rne(row_out + i,             _mm512_mul_ps(_mm512_mul_ps(v0, scale_vec), w0));
            store_m512_to_bfloat16_rne(row_out + i + SIMD_WIDTH, _mm512_mul_ps(_mm512_mul_ps(v1, scale_vec), w1));
        }
        for (; i + SIMD_WIDTH <= X; i += SIMD_WIDTH) {
            __m512 v = load_bfloat16_to_m512(row_in + i);
            __m512 w = load_bfloat16_to_m512(norm_weight + i);
            store_m512_to_bfloat16_rne(row_out + i, _mm512_mul_ps(_mm512_mul_ps(v, scale_vec), w));
        }

        // Scalar tail
        for (; i < X; ++i) {
            float val = static_cast<float>(row_in[i]);
            row_out[i] = static_cast<bf16>(val * scale * static_cast<float>(norm_weight[i]));
        }
    }
}

// AVX-512 clamp for bfloat16: output[i] = clamp(input[i], min_val, max_val)
// Uses 4x unrolling for better ILP and OpenMP parallelization for large arrays.
void simd_clamp(
    const bf16* input,
    bf16* output,
    bf16 min_val,
    bf16 max_val,
    size_t size
) {
    constexpr size_t SIMD_WIDTH = 16;
    constexpr size_t UNROLL_FACTOR = 4;
    constexpr size_t CHUNK_SIZE = SIMD_WIDTH * UNROLL_FACTOR; // 64 elements

    const __m512 vmin = _mm512_set1_ps(static_cast<float>(min_val));
    const __m512 vmax = _mm512_set1_ps(static_cast<float>(max_val));

    constexpr size_t min_elements_per_thread = CHUNK_SIZE * 8;
    const bool use_parallel = size >= (min_elements_per_thread * 2);
    const size_t num_chunks = size / CHUNK_SIZE;
    const int signed_num_chunks = static_cast<int>(num_chunks);

    #pragma omp parallel for num_threads(max_prefill_threads) if(use_parallel) schedule(static)
    for (int chunk_idx = 0; chunk_idx < signed_num_chunks; ++chunk_idx) {
        size_t i = static_cast<size_t>(chunk_idx) * CHUNK_SIZE;

        _mm_prefetch(reinterpret_cast<const char*>(input + i + 64), _MM_HINT_T0);

        __m512 v0 = load_bfloat16_to_m512(input + i);
        __m512 v1 = load_bfloat16_to_m512(input + i + 16);
        __m512 v2 = load_bfloat16_to_m512(input + i + 32);
        __m512 v3 = load_bfloat16_to_m512(input + i + 48);

        v0 = _mm512_min_ps(_mm512_max_ps(v0, vmin), vmax);
        v1 = _mm512_min_ps(_mm512_max_ps(v1, vmin), vmax);
        v2 = _mm512_min_ps(_mm512_max_ps(v2, vmin), vmax);
        v3 = _mm512_min_ps(_mm512_max_ps(v3, vmin), vmax);

        store_m512_to_bfloat16_rne(output + i,      v0);
        store_m512_to_bfloat16_rne(output + i + 16, v1);
        store_m512_to_bfloat16_rne(output + i + 32, v2);
        store_m512_to_bfloat16_rne(output + i + 48, v3);
    }

    // Remaining 16-element chunks
    size_t i = num_chunks * CHUNK_SIZE;
    for (; i + SIMD_WIDTH <= size; i += SIMD_WIDTH) {
        __m512 v = load_bfloat16_to_m512(input + i);
        v = _mm512_min_ps(_mm512_max_ps(v, vmin), vmax);
        store_m512_to_bfloat16_rne(output + i, v);
    }

    // Scalar tail
    for (; i < size; ++i) {
        float val = static_cast<float>(input[i]);
        float fmin = static_cast<float>(min_val);
        float fmax = static_cast<float>(max_val);
        if (val < fmin) val = fmin;
        if (val > fmax) val = fmax;
        output[i] = static_cast<bf16>(val);
    }
}

void simd_mul(
    const bf16* input1,
    const bf16* input2,
    bf16* output,
    size_t size
) {
    constexpr size_t SIMD_WIDTH = 16;
    constexpr size_t UNROLL_FACTOR = 4;
    constexpr size_t CHUNK_SIZE = SIMD_WIDTH * UNROLL_FACTOR;
    constexpr size_t min_elements_per_thread = CHUNK_SIZE * 8;
    constexpr int max_threads = max_prefill_threads;

    const bool use_parallel = size >= (min_elements_per_thread * 2);
    const size_t num_chunks = size / CHUNK_SIZE;
    const int signed_num_chunks = static_cast<int>(num_chunks);

    #pragma omp parallel for num_threads(max_threads) if(use_parallel) schedule(static)
    for (int chunk_idx = 0; chunk_idx < signed_num_chunks; ++chunk_idx) {
        const size_t i = static_cast<size_t>(chunk_idx) * CHUNK_SIZE;

        _mm_prefetch(reinterpret_cast<const char*>(input1 + i + 64), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(input2 + i + 64), _MM_HINT_T0);

        const __m512 a_vec0 = load_bfloat16_to_m512(input1 + i);
        const __m512 b_vec0 = load_bfloat16_to_m512(input2 + i);
        const __m512 a_vec1 = load_bfloat16_to_m512(input1 + i + 16);
        const __m512 b_vec1 = load_bfloat16_to_m512(input2 + i + 16);
        const __m512 a_vec2 = load_bfloat16_to_m512(input1 + i + 32);
        const __m512 b_vec2 = load_bfloat16_to_m512(input2 + i + 32);
        const __m512 a_vec3 = load_bfloat16_to_m512(input1 + i + 48);
        const __m512 b_vec3 = load_bfloat16_to_m512(input2 + i + 48);

        store_m512_to_bfloat16_rne(output + i,      _mm512_mul_ps(a_vec0, b_vec0));
        store_m512_to_bfloat16_rne(output + i + 16, _mm512_mul_ps(a_vec1, b_vec1));
        store_m512_to_bfloat16_rne(output + i + 32, _mm512_mul_ps(a_vec2, b_vec2));
        store_m512_to_bfloat16_rne(output + i + 48, _mm512_mul_ps(a_vec3, b_vec3));
    }

    size_t i = num_chunks * CHUNK_SIZE;

    for (; i + SIMD_WIDTH <= size; i += SIMD_WIDTH) {
        const __m512 a_vec = load_bfloat16_to_m512(input1 + i);
        const __m512 b_vec = load_bfloat16_to_m512(input2 + i);
        store_m512_to_bfloat16_rne(output + i, _mm512_mul_ps(a_vec, b_vec));
    }

    for (; i < size; ++i) {
        output[i] = static_cast<bf16>(static_cast<float>(input1[i]) * static_cast<float>(input2[i]));
    }
}

void simd_mul(
    const bf16* input1,
    bf16 input2_scalar,
    bf16* output,
    size_t size
) {
    constexpr size_t SIMD_WIDTH = 16;
    constexpr size_t UNROLL_FACTOR = 4;
    constexpr size_t CHUNK_SIZE = SIMD_WIDTH * UNROLL_FACTOR;
    constexpr size_t min_elements_per_thread = CHUNK_SIZE * 8;
    constexpr int max_threads = max_prefill_threads;

    // Broadcast scalar to all 16 lanes once
    const __m512 scalar_vec = _mm512_set1_ps(static_cast<float>(input2_scalar));

    const bool use_parallel = size >= (min_elements_per_thread * 2);
    const size_t num_chunks = size / CHUNK_SIZE;
    const int signed_num_chunks = static_cast<int>(num_chunks);

    #pragma omp parallel for num_threads(max_threads) if(use_parallel) schedule(static)
    for (int chunk_idx = 0; chunk_idx < signed_num_chunks; ++chunk_idx) {
        const size_t i = static_cast<size_t>(chunk_idx) * CHUNK_SIZE;

        _mm_prefetch(reinterpret_cast<const char*>(input1 + i + 64), _MM_HINT_T0);

        const __m512 a_vec0 = load_bfloat16_to_m512(input1 + i);
        const __m512 a_vec1 = load_bfloat16_to_m512(input1 + i + 16);
        const __m512 a_vec2 = load_bfloat16_to_m512(input1 + i + 32);
        const __m512 a_vec3 = load_bfloat16_to_m512(input1 + i + 48);

        store_m512_to_bfloat16_rne(output + i,      _mm512_mul_ps(a_vec0, scalar_vec));
        store_m512_to_bfloat16_rne(output + i + 16, _mm512_mul_ps(a_vec1, scalar_vec));
        store_m512_to_bfloat16_rne(output + i + 32, _mm512_mul_ps(a_vec2, scalar_vec));
        store_m512_to_bfloat16_rne(output + i + 48, _mm512_mul_ps(a_vec3, scalar_vec));
    }

    size_t i = num_chunks * CHUNK_SIZE;

    for (; i + SIMD_WIDTH <= size; i += SIMD_WIDTH) {
        const __m512 a_vec = load_bfloat16_to_m512(input1 + i);
        store_m512_to_bfloat16_rne(output + i, _mm512_mul_ps(a_vec, scalar_vec));
    }

    const float scalar_f = static_cast<float>(input2_scalar);
    for (; i < size; ++i) {
        output[i] = static_cast<bf16>(static_cast<float>(input1[i]) * scalar_f);
    }
}

// ============================================================================
// simd_relu: AVX-512 ReLU for bfloat16
// output[i] = max(input[i], 0)
// ============================================================================
void simd_relu(
    const bf16* input,
    bf16* output,
    size_t size
) {
    constexpr size_t SIMD_WIDTH = 16;
    constexpr size_t UNROLL_FACTOR = 4;
    constexpr size_t CHUNK_SIZE = SIMD_WIDTH * UNROLL_FACTOR; // 64 elements

    const __m512 zero = _mm512_setzero_ps();

    constexpr size_t min_elements_per_thread = CHUNK_SIZE * 8;
    const bool use_parallel = size >= (min_elements_per_thread * 2);
    const size_t num_chunks = size / CHUNK_SIZE;
    const int signed_num_chunks = static_cast<int>(num_chunks);

    #pragma omp parallel for num_threads(max_prefill_threads) if(use_parallel) schedule(static)
    for (int chunk_idx = 0; chunk_idx < signed_num_chunks; ++chunk_idx) {
        size_t i = static_cast<size_t>(chunk_idx) * CHUNK_SIZE;

        _mm_prefetch(reinterpret_cast<const char*>(input + i + 64), _MM_HINT_T0);

        __m512 v0 = load_bfloat16_to_m512(input + i);
        __m512 v1 = load_bfloat16_to_m512(input + i + 16);
        __m512 v2 = load_bfloat16_to_m512(input + i + 32);
        __m512 v3 = load_bfloat16_to_m512(input + i + 48);

        store_m512_to_bfloat16_rne(output + i,      _mm512_max_ps(v0, zero));
        store_m512_to_bfloat16_rne(output + i + 16, _mm512_max_ps(v1, zero));
        store_m512_to_bfloat16_rne(output + i + 32, _mm512_max_ps(v2, zero));
        store_m512_to_bfloat16_rne(output + i + 48, _mm512_max_ps(v3, zero));
    }

    size_t i = num_chunks * CHUNK_SIZE;
    for (; i + SIMD_WIDTH <= size; i += SIMD_WIDTH) {
        __m512 v = load_bfloat16_to_m512(input + i);
        store_m512_to_bfloat16_rne(output + i, _mm512_max_ps(v, zero));
    }

    // Scalar tail
    for (; i < size; ++i) {
        float val = static_cast<float>(input[i]);
        output[i] = static_cast<bf16>(val > 0.0f ? val : 0.0f);
    }
}

// ============================================================================
// simd_silu: AVX-512 SiLU (Swish) for bfloat16
// output[i] = input[i] * sigmoid(input[i])
// ============================================================================
void simd_silu(
    const bf16* input,
    bf16* output,
    size_t size
) {
    constexpr size_t SIMD_WIDTH = 16;
    constexpr size_t UNROLL_FACTOR = 4;
    constexpr size_t CHUNK_SIZE = SIMD_WIDTH * UNROLL_FACTOR; // 64 elements

    constexpr size_t min_elements_per_thread = CHUNK_SIZE * 8;
    const bool use_parallel = size >= (min_elements_per_thread * 2);
    const size_t num_chunks = size / CHUNK_SIZE;
    const int signed_num_chunks = static_cast<int>(num_chunks);

    #pragma omp parallel for num_threads(max_prefill_threads) if(use_parallel) schedule(static)
    for (int chunk_idx = 0; chunk_idx < signed_num_chunks; ++chunk_idx) {
        size_t i = static_cast<size_t>(chunk_idx) * CHUNK_SIZE;

        _mm_prefetch(reinterpret_cast<const char*>(input + i + 64), _MM_HINT_T0);

        __m512 v0 = load_bfloat16_to_m512(input + i);
        __m512 v1 = load_bfloat16_to_m512(input + i + 16);
        __m512 v2 = load_bfloat16_to_m512(input + i + 32);
        __m512 v3 = load_bfloat16_to_m512(input + i + 48);

        store_m512_to_bfloat16_rne(output + i,      silu_avx512(v0));
        store_m512_to_bfloat16_rne(output + i + 16, silu_avx512(v1));
        store_m512_to_bfloat16_rne(output + i + 32, silu_avx512(v2));
        store_m512_to_bfloat16_rne(output + i + 48, silu_avx512(v3));
    }

    size_t i = num_chunks * CHUNK_SIZE;
    for (; i + SIMD_WIDTH <= size; i += SIMD_WIDTH) {
        __m512 v = load_bfloat16_to_m512(input + i);
        store_m512_to_bfloat16_rne(output + i, silu_avx512(v));
    }

    // Scalar tail
    for (; i < size; ++i) {
        float val = static_cast<float>(input[i]);
        float sig = 1.0f / (1.0f + std::exp(-val));
        output[i] = static_cast<bf16>(val * sig);
    }
}

// ============================================================================
// simd_layernorm: AVX-512 Layer Normalization for bfloat16
// input shape:  [seq_len, D_padded] in row major, only D columns valid
// output shape: [seq_len, D_padded]
// weights: [D] (per-element scale, applied after normalization)
// Formula per row: output = ((input - mean) / sqrt(var + eps)) * weights
// ============================================================================
void simd_layernorm(
    bf16* input,
    bf16* output,
    const bf16* weights,
    int D,
    int D_padded,
    int seq_len,
    float eps
) {
    constexpr size_t SIMD_WIDTH = 16;
    const float inv_D = 1.0f / static_cast<float>(D);

    #pragma omp parallel for num_threads(max_prefill_threads) schedule(static) if(seq_len >= 8)
    for (int row = 0; row < seq_len; ++row) {
        bf16* row_in  = input  + static_cast<size_t>(row) * D_padded;
        bf16* row_out = output + static_cast<size_t>(row) * D_padded;

        // === Pass 1: compute mean ===
        __m512 sum_acc0 = _mm512_setzero_ps();
        __m512 sum_acc1 = _mm512_setzero_ps();
        size_t i = 0;

        for (; i + SIMD_WIDTH * 2 <= static_cast<size_t>(D); i += SIMD_WIDTH * 2) {
            sum_acc0 = _mm512_add_ps(sum_acc0, load_bfloat16_to_m512(row_in + i));
            sum_acc1 = _mm512_add_ps(sum_acc1, load_bfloat16_to_m512(row_in + i + SIMD_WIDTH));
        }
        for (; i + SIMD_WIDTH <= static_cast<size_t>(D); i += SIMD_WIDTH) {
            sum_acc0 = _mm512_add_ps(sum_acc0, load_bfloat16_to_m512(row_in + i));
        }

        float sum = _mm512_reduce_add_ps(_mm512_add_ps(sum_acc0, sum_acc1));
        for (; i < static_cast<size_t>(D); ++i) {
            sum += static_cast<float>(row_in[i]);
        }

        float mean = sum * inv_D;
        __m512 mean_vec = _mm512_set1_ps(mean);

        // === Pass 2: compute variance ===
        __m512 var_acc0 = _mm512_setzero_ps();
        __m512 var_acc1 = _mm512_setzero_ps();
        i = 0;

        for (; i + SIMD_WIDTH * 2 <= static_cast<size_t>(D); i += SIMD_WIDTH * 2) {
            __m512 diff0 = _mm512_sub_ps(load_bfloat16_to_m512(row_in + i), mean_vec);
            __m512 diff1 = _mm512_sub_ps(load_bfloat16_to_m512(row_in + i + SIMD_WIDTH), mean_vec);
            var_acc0 = _mm512_fmadd_ps(diff0, diff0, var_acc0);
            var_acc1 = _mm512_fmadd_ps(diff1, diff1, var_acc1);
        }
        for (; i + SIMD_WIDTH <= static_cast<size_t>(D); i += SIMD_WIDTH) {
            __m512 diff = _mm512_sub_ps(load_bfloat16_to_m512(row_in + i), mean_vec);
            var_acc0 = _mm512_fmadd_ps(diff, diff, var_acc0);
        }

        float var_sum = _mm512_reduce_add_ps(_mm512_add_ps(var_acc0, var_acc1));
        for (; i < static_cast<size_t>(D); ++i) {
            float diff = static_cast<float>(row_in[i]) - mean;
            var_sum += diff * diff;
        }

        float variance = var_sum * inv_D;
        float inv_std = 1.0f / std::sqrt(variance + eps);
        __m512 inv_std_vec = _mm512_set1_ps(inv_std);

        // === Pass 3: normalize and apply weights ===
        i = 0;
        for (; i + SIMD_WIDTH * 2 <= static_cast<size_t>(D); i += SIMD_WIDTH * 2) {
            __m512 x0 = load_bfloat16_to_m512(row_in + i);
            __m512 x1 = load_bfloat16_to_m512(row_in + i + SIMD_WIDTH);
            __m512 w0 = load_bfloat16_to_m512(weights + i);
            __m512 w1 = load_bfloat16_to_m512(weights + i + SIMD_WIDTH);

            __m512 norm0 = _mm512_mul_ps(_mm512_sub_ps(x0, mean_vec), inv_std_vec);
            __m512 norm1 = _mm512_mul_ps(_mm512_sub_ps(x1, mean_vec), inv_std_vec);

            store_m512_to_bfloat16_rne(row_out + i,             _mm512_mul_ps(norm0, w0));
            store_m512_to_bfloat16_rne(row_out + i + SIMD_WIDTH, _mm512_mul_ps(norm1, w1));
        }
        for (; i + SIMD_WIDTH <= static_cast<size_t>(D); i += SIMD_WIDTH) {
            __m512 x = load_bfloat16_to_m512(row_in + i);
            __m512 w = load_bfloat16_to_m512(weights + i);
            __m512 norm = _mm512_mul_ps(_mm512_sub_ps(x, mean_vec), inv_std_vec);
            store_m512_to_bfloat16_rne(row_out + i, _mm512_mul_ps(norm, w));
        }

        // Scalar tail
        for (; i < static_cast<size_t>(D); ++i) {
            float val = (static_cast<float>(row_in[i]) - mean) * inv_std;
            row_out[i] = static_cast<bf16>(val * static_cast<float>(weights[i]));
        }
    }
}

// ============================================================================
// scalar_conv2d: Reference scalar 2D convolution for bfloat16 (verification)
// Same interface and semantics as simd_conv2d but purely scalar.
// Accumulates in float32 for precision.
// ============================================================================
void scalar_conv2d(
    const bf16* input,
    const bf16* kernel,
    bf16* output,
    int C_in, int H_in, int W_in,
    int C_out, int K, int stride, int padding
) {
    const int H_out = (H_in + 2 * padding - K) / stride + 1;
    const int W_out = (W_in + 2 * padding - K) / stride + 1;
    const int kernel_size = C_in * K * K;

    for (int oc = 0; oc < C_out; ++oc) {
        const bf16* oc_kernel = kernel + static_cast<size_t>(oc) * kernel_size;
        bf16* oc_output = output + static_cast<size_t>(oc) * H_out * W_out;

        for (int oh = 0; oh < H_out; ++oh) {
            for (int ow = 0; ow < W_out; ++ow) {
                float acc = 0.0f;
                int k_idx = 0;

                for (int ic = 0; ic < C_in; ++ic) {
                    const bf16* ic_input = input + static_cast<size_t>(ic) * H_in * W_in;

                    for (int kh = 0; kh < K; ++kh) {
                        int ih = oh * stride - padding + kh;

                        if (ih < 0 || ih >= H_in) {
                            k_idx += K;
                            continue;
                        }

                        const bf16* input_row = ic_input + ih * W_in;

                        for (int kw = 0; kw < K; ++kw, ++k_idx) {
                            int iw = ow * stride - padding + kw;

                            if (iw < 0 || iw >= W_in) {
                                continue;
                            }

                            acc += static_cast<float>(input_row[iw]) * static_cast<float>(oc_kernel[k_idx]);
                        }
                    }
                }

                oc_output[oh * W_out + ow] = static_cast<bf16>(acc);
            }
        }
    }
}

// ============================================================================
// simd_conv2d: AVX-512 2D convolution for bfloat16 with OpenMP
// input:  [C_in, H_in, W_in]  in CHW layout
// kernel: [C_out, C_in, K, K]
// output: [C_out, H_out, W_out]
// H_out = (H_in + 2*padding - K) / stride + 1
// W_out = (W_in + 2*padding - K) / stride + 1
//
// Strategy: im2col-style accumulation with cache-friendly access patterns.
// - Outer loop over output channels (parallelized with OpenMP)
// - For each output position, accumulate over C_in * K * K with SIMD
// ============================================================================
void simd_conv2d(
    bf16* input,
    const bf16* kernel,
    bf16* output,
    int C_in, int H_in, int W_in,
    int C_out, int K, int stride, int padding
) {
    const int H_out = (H_in + 2 * padding - K) / stride + 1;
    const int W_out = (W_in + 2 * padding - K) / stride + 1;
    const int kernel_size = C_in * K * K;   // elements per output channel kernel

    constexpr size_t SIMD_WIDTH = 16;

    // Parallelize over output channels — each thread works on independent output channels
    #pragma omp parallel for num_threads(max_prefill_threads) schedule(static) if(C_out >= max_prefill_threads)
    for (int oc = 0; oc < C_out; ++oc) {
        const bf16* oc_kernel = kernel + static_cast<size_t>(oc) * kernel_size;
        bf16* oc_output = output + static_cast<size_t>(oc) * H_out * W_out;

        for (int oh = 0; oh < H_out; ++oh) {
            // --- SIMD path: vectorize across output width (ow) ---
            // For each (ic, kh, kw), broadcast the kernel weight and accumulate
            // over SIMD_WIDTH output positions simultaneously.
            int ow = 0;
            for (; ow + static_cast<int>(SIMD_WIDTH) <= W_out; ow += static_cast<int>(SIMD_WIDTH)) {
                __m512 acc_vec = _mm512_setzero_ps();
                int k_idx = 0;

                for (int ic = 0; ic < C_in; ++ic) {
                    const bf16* ic_input = input + static_cast<size_t>(ic) * H_in * W_in;

                    for (int kh = 0; kh < K; ++kh) {
                        int ih = oh * stride - padding + kh;

                        if (ih < 0 || ih >= H_in) {
                            k_idx += K;
                            continue;
                        }

                        const bf16* input_row = ic_input + ih * W_in;

                        for (int kw = 0; kw < K; ++kw, ++k_idx) {
                            __m512 k_vec = _mm512_set1_ps(static_cast<float>(oc_kernel[k_idx]));

                            // Gather SIMD_WIDTH input values for consecutive ow positions
                            // iw = (ow + lane) * stride - padding + kw
                            int iw_base = ow * stride - padding + kw;

                            if (stride == 1) {
                                // Contiguous access: load directly from input_row
                                if (iw_base >= 0 && iw_base + static_cast<int>(SIMD_WIDTH) <= W_in) {
                                    // All elements in bounds — fast path
                                    __m512 in_vec = load_bfloat16_to_m512(input_row + iw_base);
                                    acc_vec = _mm512_fmadd_ps(in_vec, k_vec, acc_vec);
                                } else {
                                    // Some elements may be out of bounds — zero-padded
                                    alignas(64) float tmp[16] = {0};
                                    for (int lane = 0; lane < static_cast<int>(SIMD_WIDTH); ++lane) {
                                        int iw = iw_base + lane;
                                        if (iw >= 0 && iw < W_in) {
                                            tmp[lane] = static_cast<float>(input_row[iw]);
                                        }
                                    }
                                    __m512 in_vec = _mm512_load_ps(tmp);
                                    acc_vec = _mm512_fmadd_ps(in_vec, k_vec, acc_vec);
                                }
                            } else {
                                // Strided access — gather element by element
                                alignas(64) float tmp[16] = {0};
                                for (int lane = 0; lane < static_cast<int>(SIMD_WIDTH); ++lane) {
                                    int iw = (ow + lane) * stride - padding + kw;
                                    if (iw >= 0 && iw < W_in) {
                                        tmp[lane] = static_cast<float>(input_row[iw]);
                                    }
                                }
                                __m512 in_vec = _mm512_load_ps(tmp);
                                acc_vec = _mm512_fmadd_ps(in_vec, k_vec, acc_vec);
                            }
                        }
                    }
                }

                // Store SIMD_WIDTH output values
                store_m512_to_bfloat16_rne(oc_output + oh * W_out + ow, acc_vec);
            }

            // --- Scalar tail for remaining ow positions ---
            for (; ow < W_out; ++ow) {
                float acc_scalar = 0.0f;
                int k_idx = 0;

                for (int ic = 0; ic < C_in; ++ic) {
                    const bf16* ic_input = input + static_cast<size_t>(ic) * H_in * W_in;

                    for (int kh = 0; kh < K; ++kh) {
                        int ih = oh * stride - padding + kh;

                        if (ih < 0 || ih >= H_in) {
                            k_idx += K;
                            continue;
                        }

                        const bf16* input_row = ic_input + ih * W_in;

                        for (int kw = 0; kw < K; ++kw, ++k_idx) {
                            int iw = ow * stride - padding + kw;

                            if (iw < 0 || iw >= W_in) {
                                continue;
                            }

                            float in_val = static_cast<float>(input_row[iw]);
                            float k_val  = static_cast<float>(oc_kernel[k_idx]);
                            acc_scalar += in_val * k_val;
                        }
                    }
                }

                oc_output[oh * W_out + ow] = static_cast<bf16>(acc_scalar);
            }
        }
    }
}

// ============================================================================
// layernorm_relu_nchw: Applies LayerNorm over C dimension + ReLU activation
// in an NCHW layout. Strides over channels by HW.
// Uses float32 accumulation to match PyTorch's LayerNorm internal precision.
// ============================================================================
void layernorm_relu_nchw(bf16* data, const bf16* norm_weight, int C, int HW, float eps) {
    #pragma omp parallel for num_threads(max_prefill_threads) schedule(static)
    for(int hw = 0; hw < HW; hw++){
        float mean = 0.0f;
        for(int c = 0; c < C; c++){
            mean += static_cast<float>(data[c * HW + hw]);
        }
        mean /= C;

        float var = 0.0f;
        for(int c = 0; c < C; c++){
            float diff = static_cast<float>(data[c * HW + hw]) - mean;
            var += diff * diff;
        }
        var /= C;

        float inv_std = 1.0f / std::sqrt(var + eps);

        for(int c = 0; c < C; c++){
            float val = static_cast<float>(data[c * HW + hw]);
            float normed = (val - mean) * inv_std * static_cast<float>(norm_weight[c]);
            data[c * HW + hw] = static_cast<bf16>(std::max(0.0f, normed));
        }
    }
}

// ============================================================================
// layernorm_gelu_nchw: Applies LayerNorm over C dimension + GELU (tanh approx)
// ============================================================================
void layernorm_gelu_nchw(bf16* data, const bf16* norm_weight, int C, int HW, float eps) {
    #pragma omp parallel for num_threads(max_prefill_threads) schedule(static)
    for(int hw = 0; hw < HW; hw++){
        double mean = 0.0;
        for(int c = 0; c < C; c++){
            mean += static_cast<float>(data[c * HW + hw]);
        }
        mean /= C;

        double var = 0.0;
        for(int c = 0; c < C; c++){
            double diff = static_cast<float>(data[c * HW + hw]) - mean;
            var += diff * diff;
        }
        var /= C;

        float inv_std = 1.0f / std::sqrt(static_cast<float>(var) + eps);

        for(int c = 0; c < C; c++){
            float val = static_cast<float>(data[c * HW + hw]);
            float normed = (val - static_cast<float>(mean)) * inv_std * static_cast<float>(norm_weight[c]);

            // GELU tanh approx
            float x_cubed = normed * normed * normed;
            float inner = 0.7978845608f * (normed + 0.044715f * x_cubed);
            float gelu_val = 0.5f * normed * (1.0f + std::tanh(inner));

            data[c * HW + hw] = static_cast<bf16>(gelu_val);
        }
    }
}

// ============================================================================
// rmsnorm_gelu_nchw: Applies RMSNorm over C dimension + GELU (tanh approx)
// ============================================================================
void rmsnorm_gelu_nchw(bf16* data, const bf16* norm_weight, int C, int HW, float eps) {
    #pragma omp parallel for num_threads(max_prefill_threads) schedule(static)
    for(int hw = 0; hw < HW; hw++){
        double sq_sum = 0.0;
        for(int c = 0; c < C; c++){
            float val = static_cast<float>(data[c * HW + hw]);
            sq_sum += val * val;
        }
        float var = sq_sum / C;
        float inv_std = 1.0f / std::sqrt(var + eps);

        for(int c = 0; c < C; c++){
            float val = static_cast<float>(data[c * HW + hw]);
            float normed = val * inv_std * static_cast<float>(norm_weight[c]);

            // GELU tanh approx
            float x_cubed = normed * normed * normed;
            float inner = 0.7978845608f * (normed + 0.044715f * x_cubed);
            float gelu_val = 0.5f * normed * (1.0f + std::tanh(inner));

            data[c * HW + hw] = static_cast<bf16>(gelu_val);
        }
    }
}

void simd_gemm_abt_bf16(
    const bf16* A, const bf16* B, bf16* C,
    int M, int N, int K,
    int lda, int ldb, int ldc
) {
    // C[i][j] = sum_k A[i*lda + k] * B[j*ldb + k]
    constexpr size_t SIMD_WIDTH = 16;

    #pragma omp parallel for num_threads(max_prefill_threads) schedule(static) if(M * N > 64)
    for (int i = 0; i < M; i++) {
        const bf16* a_row = A + (size_t)i * lda;
        for (int j = 0; j < N; j++) {
            const bf16* b_row = B + (size_t)j * ldb;

            __m512 acc0 = _mm512_setzero_ps();
            __m512 acc1 = _mm512_setzero_ps();
            int k = 0;
            for (; k + 2 * (int)SIMD_WIDTH <= K; k += 2 * SIMD_WIDTH) {
                acc0 = _mm512_fmadd_ps(load_bfloat16_to_m512(a_row + k),
                                        load_bfloat16_to_m512(b_row + k), acc0);
                acc1 = _mm512_fmadd_ps(load_bfloat16_to_m512(a_row + k + SIMD_WIDTH),
                                        load_bfloat16_to_m512(b_row + k + SIMD_WIDTH), acc1);
            }
            __m512 acc = _mm512_add_ps(acc0, acc1);
            for (; k + (int)SIMD_WIDTH <= K; k += SIMD_WIDTH) {
                acc = _mm512_fmadd_ps(load_bfloat16_to_m512(a_row + k),
                                       load_bfloat16_to_m512(b_row + k), acc);
            }
            float sum = _mm512_reduce_add_ps(acc);
            for (; k < K; k++) {
                sum += (float)a_row[k] * (float)b_row[k];
            }
            C[(size_t)i * ldc + j] = (bf16)sum;
        }
    }
}

// ============================================================================
// simd_glu: Gated Linear Unit for bfloat16
// input:  [seq_len, 2, hidden_dim] — first half is value, second half is gate
// output: [seq_len, hidden_dim]
// Formula: output[s, :] = input[s, 0, :] * sigmoid(input[s, 1, :])
// Fused single-pass: load both halves, sigmoid the gate, multiply, store.
// ============================================================================
void simd_glu(
    const bf16* input,
    bf16* output,
    size_t seq_len,
    size_t hidden_dim
) {
    constexpr size_t SIMD_WIDTH = 16;
    constexpr size_t UNROLL_FACTOR = 4;
    constexpr size_t CHUNK_SIZE = SIMD_WIDTH * UNROLL_FACTOR; // 64 elements
    const size_t stride = 2 * hidden_dim; // row stride in input

    int signed_seq_len = static_cast<int>(seq_len);

    #pragma omp parallel for num_threads(max_prefill_threads) schedule(static) if(seq_len >= 4)
    for (int s = 0; s < signed_seq_len; ++s) {
        const bf16* val_ptr  = input + (size_t)s * stride;               // first half
        const bf16* gate_ptr = input + (size_t)s * stride + hidden_dim;  // second half
        bf16* out_ptr = output + (size_t)s * hidden_dim;

        size_t d = 0;

        // Main loop: 4x unrolled SIMD
        for (; d + CHUNK_SIZE <= hidden_dim; d += CHUNK_SIZE) {
            _mm_prefetch(reinterpret_cast<const char*>(val_ptr + d + 64), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(gate_ptr + d + 64), _MM_HINT_T0);

            __m512 v0 = load_bfloat16_to_m512(val_ptr + d);
            __m512 g0 = sigmoid_avx512(load_bfloat16_to_m512(gate_ptr + d));

            __m512 v1 = load_bfloat16_to_m512(val_ptr + d + 16);
            __m512 g1 = sigmoid_avx512(load_bfloat16_to_m512(gate_ptr + d + 16));

            __m512 v2 = load_bfloat16_to_m512(val_ptr + d + 32);
            __m512 g2 = sigmoid_avx512(load_bfloat16_to_m512(gate_ptr + d + 32));

            __m512 v3 = load_bfloat16_to_m512(val_ptr + d + 48);
            __m512 g3 = sigmoid_avx512(load_bfloat16_to_m512(gate_ptr + d + 48));

            store_m512_to_bfloat16_rne(out_ptr + d,      _mm512_mul_ps(v0, g0));
            store_m512_to_bfloat16_rne(out_ptr + d + 16, _mm512_mul_ps(v1, g1));
            store_m512_to_bfloat16_rne(out_ptr + d + 32, _mm512_mul_ps(v2, g2));
            store_m512_to_bfloat16_rne(out_ptr + d + 48, _mm512_mul_ps(v3, g3));
        }

        // Remaining 16-element chunks
        for (; d + SIMD_WIDTH <= hidden_dim; d += SIMD_WIDTH) {
            __m512 v = load_bfloat16_to_m512(val_ptr + d);
            __m512 g = sigmoid_avx512(load_bfloat16_to_m512(gate_ptr + d));
            store_m512_to_bfloat16_rne(out_ptr + d, _mm512_mul_ps(v, g));
        }

        // Scalar tail
        for (; d < hidden_dim; ++d) {
            float v = static_cast<float>(val_ptr[d]);
            float g = static_cast<float>(gate_ptr[d]);
            float sig = 1.0f / (1.0f + std::exp(-g));
            out_ptr[d] = static_cast<bf16>(v * sig);
        }
    }
}

void scalar_conv1d(
    int conv_kernel_size,
    int conv_stride,
    const bf16* input,   // [seq_len + (conv_kernel_size - conv_stride), hidden_dim]
    const bf16* kernel_weights, // [conv_kernel_size, hidden_dim]
    bf16* output, // [seq_len, hidden_dim]
    int seq_len,
    int hidden_dim
){
    for (int o = 0; o < seq_len; o++) {
        bf16* out_ptr = output + o * hidden_dim;
        for (int d = 0; d < hidden_dim; d++) {
            float acc = 0.0f;
            for (int k = 0; k < conv_kernel_size; k++) {
                int in_idx = (o * conv_stride + k) * hidden_dim + d;
                int w_idx  = k * hidden_dim + d;
                acc += static_cast<float>(input[in_idx]) * static_cast<float>(kernel_weights[w_idx]);
            }
            out_ptr[d] = static_cast<bf16>(acc);
        }
    }
}
