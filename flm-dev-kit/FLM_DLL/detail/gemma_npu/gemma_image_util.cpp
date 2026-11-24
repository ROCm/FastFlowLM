// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include "gemma_image_util.hpp"
#include "gemma_image_param.hpp"
#include <thread>
#include <omp.h>

/**
 * @brief Performs 2D convolution using AVX512 with a bfloat16 input, adds a bias, and stores the output as bfloat16.
 *
 * @param image Pointer to the input image data as bfloat16 (layout: C_in, H, W).
 * @param kernel Pointer to the kernel/filter data as bfloat16 (layout: C_out, C_in, K, K).
 * @param bias Pointer to the bias data as bfloat16 (layout: C_out).
 * @param output Pointer to the output buffer as bfloat16 (layout: H_out*W_out, C_out).
 * @param C_in Number of input channels.
 * @param H Input image height.
 * @param W Input image width.
 * @param C_out Number of output channels (filters).
 * @param K Kernel size (KxK).
 * @param stride Convolution stride.
 */
void conv2d_avx512_bf16_ptr_bias_hwc_out(
    const  biovault::bfloat16_t* image,
    const biovault::bfloat16_t* kernel,
    const biovault::bfloat16_t* bias,
    biovault::bfloat16_t* output,
    int C_in, int H, int W,
    int C_out, int K,
    int stride) {

    // Calculate output dimensions
    int H_out = (H - K) / stride + 1;
    int W_out = (W - K) / stride + 1;

    assert(W_out % 16 == 0 && "Output width must be divisible by 16 for this AVX-512 implementation.");
    // assert(C_out % 14 == 0 && "Output channels must be divisible by 14 for this implementation.");

    // Use __restrict to indicate that pointers do not alias.
    const  biovault::bfloat16_t* __restrict img = image;
    const biovault::bfloat16_t* __restrict ker = kernel;
    const biovault::bfloat16_t* __restrict b = bias;
    biovault::bfloat16_t* __restrict out = output;

    const int C_OUT_BLOCK_SIZE = 14;

    if (stride == 1) {
        // OPTIMIZED PATH FOR STRIDE = 1
        #pragma omp parallel for schedule(static) num_threads(TOTAL_OPENMP_THREAD)
        for (int h_out = 0; h_out < H_out; ++h_out) {
            for (int w_out = 0; w_out < W_out; w_out += 16) {
                for (int c_out_b = 0; c_out_b < C_out; c_out_b += C_OUT_BLOCK_SIZE) {

                    int current_block_size = std::min(C_OUT_BLOCK_SIZE, C_out - c_out_b);
                    __m512 acc[C_OUT_BLOCK_SIZE];
                    for (int i = 0; i < current_block_size; ++i) {
                        acc[i] = _mm512_setzero_ps();
                    }

                    for (int c_in = 0; c_in < C_in; ++c_in) {
                        for (int kh = 0; kh < K; ++kh) {
                            for (int kw = 0; kw < K; ++kw) {
                                int h_in = h_out + kh;
                                int w_in = w_out + kw;
                                const  biovault::bfloat16_t* img_ptr = img + c_in * H * W + h_in * W + w_in;

                                #ifdef __WINDOWS__
                                const __m256bh bf16_vals = _mm256_loadu_si256(reinterpret_cast<const __m256bh*>(img_ptr));
                                #else
                                const __m256bh bf16_vals = reinterpret_cast<__m256bh>(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(img_ptr)));
                                #endif

                                __m512 img_vals = _mm512_cvtpbh_ps(bf16_vals);
                                for (int i = 0; i < current_block_size; ++i) {
                                    int c_out = c_out_b + i;
                                    int kernel_idx = c_out * C_in * K * K + c_in * K * K + kh * K + kw;
                                    // Convert bfloat16 kernel value to float
                                    __m512 ker_val = _mm512_set1_ps(static_cast<float>(ker[kernel_idx]));
                                    acc[i] = _mm512_fmadd_ps(img_vals, ker_val, acc[i]);
                                }
                            }
                        }
                    }

                    // Add bias and store results with new [H_out*W_out, C_out] layout.
                    float temp_results[C_OUT_BLOCK_SIZE][16];

                    // Step 1: Add bias and store accumulator results into a temporary FP32 buffer.
                    for (int i = 0; i < current_block_size; ++i) {
                        int c_out = c_out_b + i;
                        // Convert bfloat16 bias to float
                        __m512 bias_vec = _mm512_set1_ps(static_cast<float>(b[c_out]));
                        __m512 result_fp32 = _mm512_add_ps(acc[i], bias_vec);
                        _mm512_storeu_ps(temp_results[i], result_fp32);
                    }

                    // Step 2: Transpose data from the temporary buffer to the final output buffer,
                    // converting to bfloat16. This reorders the data for the channels-last layout.
                    for (int j = 0; j < 16; ++j) { // Iterate over the 16 horizontal pixel positions.
                        // Calculate the base index for the current output pixel (h_out, w_out + j).
                        int pixel_base_idx = (h_out * W_out + w_out + j) * C_out;
                        // Get a pointer to the start of the channel block for this pixel.
                         biovault::bfloat16_t* out_ptr = out + pixel_base_idx + c_out_b;

                        // Write the channel data for this single pixel.
                        for (int i = 0; i < current_block_size; ++i) {
                            out_ptr[i] = static_cast< biovault::bfloat16_t>(temp_results[i][j]);
                        }
                    }
                }
            }
        }
    } else {
        // GENERAL PATH FOR STRIDE > 1
        #pragma omp parallel for schedule(static) num_threads(TOTAL_OPENMP_THREAD)
        for (int h_out = 0; h_out < H_out; ++h_out) {
            for (int w_out = 0; w_out < W_out; w_out += 16) {
                for (int c_out_b = 0; c_out_b < C_out; c_out_b += C_OUT_BLOCK_SIZE) {

                    int current_block_size = std::min(C_OUT_BLOCK_SIZE, C_out - c_out_b);
                    __m512 acc[C_OUT_BLOCK_SIZE];
                    for (int i = 0; i < current_block_size; ++i) {
                        acc[i] = _mm512_setzero_ps();
                    }

                    for (int c_in = 0; c_in < C_in; ++c_in) {
                        for (int kh = 0; kh < K; ++kh) {
                            for (int kw = 0; kw < K; ++kw) {
                                int h_in = h_out * stride + kh;
                                int w_in = w_out * stride + kw;
                                const  biovault::bfloat16_t* img_ptr = img + c_in * H * W + h_in * W + w_in;

                                // Load 16 strided bfloat16 values, convert to fp32
                                __m256i bf16_vals[16];
                                for(int j = 0; j < 16; ++j) {
                                    bf16_vals[j] = _mm256_set1_epi16(*reinterpret_cast<const uint16_t*>(&img_ptr[j * stride]));
                                }
                                // Pack and convert
                                alignas(32) uint16_t packed[16];
                                for(int j = 0; j < 16; ++j) {
                                    packed[j] = *reinterpret_cast<const uint16_t*>(&img_ptr[j * stride]);
                                }
                                __m256i bf16_packed = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(packed));
                                #ifdef __WINDOWS__
                                __m512 img_vals = _mm512_cvtpbh_ps(bf16_packed);
                                #else
                                __m512 img_vals = _mm512_cvtpbh_ps(reinterpret_cast<__m256bh>(bf16_packed));
                                #endif

                                for (int i = 0; i < current_block_size; ++i) {
                                    int c_out = c_out_b + i;
                                    int kernel_idx = c_out * C_in * K * K + c_in * K * K + kh * K + kw;
                                    // Convert bfloat16 kernel value to float
                                    __m512 ker_val = _mm512_set1_ps(static_cast<float>(ker[kernel_idx]));
                                    acc[i] = _mm512_fmadd_ps(img_vals, ker_val, acc[i]);
                                }
                            }
                        }
                    }

                    // Add bias and store results with new [H_out*W_out, C_out] layout.
                    float temp_results[C_OUT_BLOCK_SIZE][16];

                    // Step 1: Add bias and store accumulator results into a temporary FP32 buffer.
                    for (int i = 0; i < current_block_size; ++i) {
                        int c_out = c_out_b + i;
                        // Convert bfloat16 bias to float
                        __m512 bias_vec = _mm512_set1_ps(static_cast<float>(b[c_out]));
                        __m512 result_fp32 = _mm512_add_ps(acc[i], bias_vec);
                        _mm512_storeu_ps(temp_results[i], result_fp32);
                    }

                    // Step 2: Transpose data from the temporary buffer to the final output buffer,
                    // converting to bfloat16. This reorders the data for the channels-last layout.
                    for (int j = 0; j < 16; ++j) { // Iterate over the 16 horizontal pixel positions.
                        // Calculate the base index for the current output pixel (h_out, w_out + j).
                        int pixel_base_idx = (h_out * W_out + w_out + j) * C_out;
                        // Get a pointer to the start of the channel block for this pixel.
                         biovault::bfloat16_t* out_ptr = out + pixel_base_idx + c_out_b;

                        // Write the channel data for this single pixel.
                        for (int i = 0; i < current_block_size; ++i) {
                            out_ptr[i] = static_cast< biovault::bfloat16_t>(temp_results[i][j]);
                        }
                    }
                }
            }
        }
    }
}

/**
 * @brief Performs an in-place bfloat16 addition of positional embeddings.
 *
 * Adds a bfloat16 embedding matrix to a bfloat16 data tensor.
 * The addition is performed in float32 for precision.
 * This function requires a CPU with AVX512_BF16 instruction set support.
 *
 * @param data Pointer to the bfloat16 input/output tensor of shape [batch_size, sequence_length, padded_hidden_size].
 * @param positional_embedding Pointer to the bfloat16 embedding matrix of shape [sequence_length, hidden_size].
 * @param batch_size The number of sequences in the batch.
 * @param sequence_length The number of tokens in each sequence (4096).
 * @param hidden_size The size of the embedding vector for each token (1152).
 * @param padded_hidden_size: The size of embedding vector being padded to 1536
 */
void add_embedding(
     biovault::bfloat16_t* __restrict data,
     const biovault::bfloat16_t* __restrict positional_embedding, // <-- MODIFIED
     const int batch_size,
     const int sequence_length,
     const int hidden_size,
     const int padded_hidden_size) {

     const long long sequence_mem_size = (long long)sequence_length * padded_hidden_size;

     #pragma omp parallel for schedule(static) num_threads(TOTAL_OPENMP_THREAD)
     for (int i = 0; i < batch_size; ++i) {
          biovault::bfloat16_t* data_sequence = data + i * sequence_mem_size;

          for (int j = 0; j < sequence_length; ++j) {
               const long long data_token_offset = (long long)j * padded_hidden_size;
               const long long embedding_token_offset = (long long)j * hidden_size;
                 biovault::bfloat16_t* data_token_vec = data_sequence + data_token_offset;
               // Pointer type is now const bfloat16_t*
               const biovault::bfloat16_t* embedding_vec = positional_embedding + embedding_token_offset; // <-- MODIFIED

               for (int k = 0; k < hidden_size; k += 16) {
                    // 1. Load 16 bfloat16 values from the data tensor.
                    __m256i data_bf16_raw = _mm256_loadu_si256(
                        reinterpret_cast<const __m256i*>(data_token_vec + k)
                    );

                    // 2. Convert data from bf16 to fp32.
                    #ifdef __WINDOWS__
                    __m512 data_fp32 = _mm512_cvtpbh_ps(data_bf16_raw);
                    #else
                    __m512 data_fp32 = _mm512_cvtpbh_ps(reinterpret_cast<__m256bh>(data_bf16_raw));
                    #endif

                    // 3. Load 16 bfloat16 values from the embedding matrix. (MODIFIED)
                    __m256i embedding_bf16_raw = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(embedding_vec + k)
                    );

                    // 4. Convert embedding from bf16 to fp32. (NEW STEP)
                    #ifdef __WINDOWS__
                    __m512 embedding_fp32 = _mm512_cvtpbh_ps(embedding_bf16_raw);
                    #else
                    __m512 embedding_fp32 = _mm512_cvtpbh_ps(reinterpret_cast<__m256bh>(embedding_bf16_raw));
                    #endif

                    // 5. Add the two float32 vectors.
                    __m512 result_fp32 = _mm512_add_ps(data_fp32, embedding_fp32);

                    // 6. Convert the float32 result back to a bfloat16 vector (__m256i).
                    #ifdef __WINDOWS__
                   __m256i result_bf16 = _mm512_cvtneps_pbh(result_fp32);
                    #else
                    __m256i result_bf16 = reinterpret_cast<__m256i>(_mm512_cvtneps_pbh(result_fp32));
                    #endif

                    // 7. Store the 16 bfloat16 results back into the data tensor.
                    _mm256_storeu_si256((__m256i*)(data_token_vec + k), result_bf16);
               }
          }
     }
}

/**
 * @brief Performs an out-of-place bfloat16 Layer Normalization on a single core.
 *
 * Normalizes the input tensor, stores it in the output tensor,
 * and applies bfloat16 scale (weight) and bias.
 * All intermediate calculations (mean, variance, normalization) are done in float32.
 * This function requires a CPU with AVX512_BF16 instruction set support.
 *
 * @param input Pointer to the bfloat16 input tensor [batch_size, seq_len_padded, hidden_size_padded].
 * @param output Pointer to the bfloat16 output tensor [batch_size, seq_len_padded, hidden_size_padded].
 * @param weight Pointer to the bfloat16 weight vector (gamma) [hidden_size].
 * @param bias Pointer to the bfloat16 bias vector (beta) [hidden_size].
 * @param batch_size The number of sequences in the batch.
 * @param seq_len The actual sequence length (non-padded).
 * @param seq_len_padded The padded sequence length.
 * @param hidden_size The actual hidden size (non-padded).
 * @param hidden_size_padded The padded hidden size (must be a multiple of 16).
 * @param eps A small float value to prevent division by zero in variance calculation.
 */
void layernorm_avx512_bf16_single_core(
     const biovault::bfloat16_t* input,   // <-- MODIFIED
     biovault::bfloat16_t* output,        // <-- MODIFIED
     const biovault::bfloat16_t* weight,
     const biovault::bfloat16_t* bias,
     const int batch_size,
     const int seq_len,
     const int seq_len_padded,
     const int hidden_size,
     const int hidden_size_padded,
     const float eps ) {

     // AVX-512 lane width assertions
     // For bfloat16: AVX-512 processes 16 bfloat16 values (32 bytes) per instruction
     // For float32: AVX-512 processes 16 float values (64 bytes) per instruction
     constexpr int AVX512_BF16_LANE_WIDTH = 16;
     constexpr int AVX512_F32_LANE_WIDTH = 16;

     assert(hidden_size_padded % AVX512_BF16_LANE_WIDTH == 0 &&
            "hidden_size_padded must be aligned to AVX-512 bfloat16 lane width (16 elements)");
     assert(seq_len_padded >= seq_len &&
            "seq_len_padded must be greater than or equal to seq_len");
     assert(hidden_size_padded >= hidden_size &&
            "hidden_size_padded must be greater than or equal to hidden_size");

     const float inv_hidden_size = 1.0f / static_cast<float>(hidden_size);

     // Sequentially iterate over each row in the batch.
     #pragma omp parallel for schedule(static) num_threads(TOTAL_OPENMP_THREAD)
     for (int b = 0; b < batch_size; ++b) {
         for (int s = 0; s < seq_len; ++s) { // Only process actual sequence length, not padded
              // <-- MODIFIED: Create separate input and output row pointers
              const biovault::bfloat16_t* input_row_ptr = input + (b * seq_len_padded + s) * hidden_size_padded;
              biovault::bfloat16_t* output_row_ptr = output + (b * seq_len_padded + s) * hidden_size_padded;

             // === PASS 1: Calculate Mean and Variance ===
             __m512 sum_vec = _mm512_setzero_ps();
             __m512 sum_sq_vec = _mm512_setzero_ps();

             // Only process actual hidden_size elements, not padded ones
             for (int i = 0; i < hidden_size; i += 16) {
                 const __mmask16 mask = (i + 16 <= hidden_size) ? 0xFFFF : (1 << (hidden_size - i)) - 1;

                     // MSVC: use __m256i for bfloat16
                     // <-- MODIFIED: Read from input_row_ptr
                     __m256i bf16_data_i = _mm256_maskz_loadu_epi16(mask, input_row_ptr + i);
                     #ifdef __WINDOWS__
                     __m512 f_vec = _mm512_cvtpbh_ps(bf16_data_i);
                     #else
                     __m512 f_vec = _mm512_cvtpbh_ps(reinterpret_cast<__m256bh>(bf16_data_i));
                     #endif

                 sum_vec = _mm512_add_ps(sum_vec, f_vec);
                 // sum_sq_vec = _mm512_fmadd_ps(f_vec, f_vec, sum_sq_vec); // <-- This is one-pass (Welford)
             }

             const float total_sum = _mm512_reduce_add_ps(sum_vec);
             const float mean = total_sum * inv_hidden_size;
             __m512 mean_vec = _mm512_set1_ps(mean);

             // This is the second pass for a more stable variance calculation
             for (int i = 0; i < hidden_size; i += 16) {
                 const __mmask16 mask = (i + 16 <= hidden_size) ? 0xFFFF : (1 << (hidden_size - i)) - 1;

                     // MSVC: use __m256i for bfloat16
                     // <-- MODIFIED: Read from input_row_ptr
                     __m256i bf16_data_i = _mm256_maskz_loadu_epi16(mask, input_row_ptr + i);
                     #ifdef __WINDOWS__
                     __m512 f_vec = _mm512_cvtpbh_ps(bf16_data_i);
                     #else
                     __m512 f_vec = _mm512_cvtpbh_ps(reinterpret_cast<__m256bh>(bf16_data_i));
                     #endif
                 __m512 diff_vec = _mm512_sub_ps(f_vec, mean_vec);
                 sum_sq_vec = _mm512_fmadd_ps(diff_vec, diff_vec, sum_sq_vec);
             }

             const float total_sum_sq = _mm512_reduce_add_ps(sum_sq_vec);

             const float variance = total_sum_sq * inv_hidden_size;
             const float inv_std_dev = 1.0f / std::sqrt(std::max(variance, 0.0f) + eps);

             const __m512 inv_std_dev_vec = _mm512_set1_ps(inv_std_dev);

             // === PASS 2: Normalize, Scale, Shift, and Store ===
             // Process all elements including padded ones to maintain memory layout
             for (int i = 0; i < hidden_size_padded; i += 16) {
                 // For actual data elements
                 if (i < hidden_size) {
                     const __mmask16 mask = (i + 16 <= hidden_size) ? 0xFFFF : (1 << (hidden_size - i)) - 1;

                     // 1. Load data (BF16) and convert to FP32
                     // <-- MODIFIED: Read from input_row_ptr
                     __m256i bf16_data_i = _mm256_maskz_loadu_epi16(mask, input_row_ptr + i);
                     #ifdef __WINDOWS__
                     __m512 f_vec = _mm512_cvtpbh_ps(bf16_data_i);
                     #else
                     __m512 f_vec = _mm512_cvtpbh_ps(reinterpret_cast<__m256bh>(bf16_data_i));
                     #endif

                     // 2. Load weight (BF16) and convert to FP32
                     __m256i bf16_w_vec = _mm256_maskz_loadu_epi16(mask, weight + i);
                     #ifdef __WINDOWS__
                     __m512 w_vec = _mm512_cvtpbh_ps(bf16_w_vec);
                     #else
                    __m512 w_vec = _mm512_cvtpbh_ps(reinterpret_cast<__m256bh>(bf16_w_vec));
                    #endif

                     // 3. Load bias (BF16) and convert to FP32
                     __m256i bf16_b_vec = _mm256_maskz_loadu_epi16(mask, bias + i);
                     #ifdef __WINDOWS__
                     __m512 b_vec = _mm512_cvtpbh_ps(bf16_b_vec);
                     #else
                    __m512 b_vec = _mm512_cvtpbh_ps(reinterpret_cast<__m256bh>(bf16_b_vec));
                    #endif

                     // 4. Normalize
                     __m512 norm_vec = _mm512_sub_ps(f_vec, mean_vec);
                     norm_vec = _mm512_mul_ps(norm_vec, inv_std_dev_vec);

                     // 5. Scale and Shift (FMA)
                     __m512 result_vec = _mm512_fmadd_ps(norm_vec, w_vec, b_vec);

                     // 6. Convert back to BF16 and store
                     #ifdef __WINDOWS__
                     __m256i result_bf16 = _mm512_cvtneps_pbh(result_vec);
                     #else
                     __m256i result_bf16 = reinterpret_cast<__m256i>(_mm512_cvtneps_pbh(result_vec));
                     #endif
                     // <-- MODIFIED: Store in output_row_ptr
                     _mm256_mask_storeu_epi16(output_row_ptr + i, mask, result_bf16);
                 } else {
                     // For padded elements, set them to zero to maintain clean memory layout
                     const __mmask16 mask = 0xFFFF; // All 16 elements for padded region
                     __m256i zero_data = _mm256_setzero_si256();
                     // <-- MODIFIED: Store in output_row_ptr
                     _mm256_mask_storeu_epi16(output_row_ptr + i, mask, zero_data);
                 }
             }
         }
     }
}
void rmsnorm_avx512_bf16_single_core(
    biovault::bfloat16_t* output,           // MODIFIED: Separate output buffer
    const biovault::bfloat16_t* input,      // MODIFIED: Separate const input buffer
    const biovault::bfloat16_t* weight,
    const int batch_size,
    const int seq_len,
    const int seq_len_padded,
    const int hidden_size,
    const int hidden_size_padded,
    const float eps) {

    // AVX-512 lane width assertions
    constexpr int AVX512_BF16_LANE_WIDTH = 16;
    constexpr int AVX512_F32_LANE_WIDTH = 16;

    assert(hidden_size_padded % AVX512_BF16_LANE_WIDTH == 0 &&
           "hidden_size_padded must be aligned to AVX-512 bfloat16 lane width (16 elements)");
    assert(seq_len_padded >= seq_len &&
           "seq_len_padded must be greater than or equal to seq_len");
    assert(hidden_size_padded >= hidden_size &&
           "hidden_size_padded must be greater than or equal to hidden_size");

    const float inv_hidden_size = 1.0f / static_cast<float>(hidden_size);

    // Sequentially iterate over each row in the batch.
    #pragma omp parallel for schedule(static) num_threads(TOTAL_OPENMP_THREAD)
    for (int b = 0; b < batch_size; ++b) {
        for (int s = 0; s < seq_len; ++s) { // Only process actual sequence length, not padded

            // MODIFIED: Create separate input and output row pointers
            const int row_offset = (b * seq_len_padded + s) * hidden_size_padded;
            const biovault::bfloat16_t* input_row_ptr = input + row_offset;
            biovault::bfloat16_t* output_row_ptr = output + row_offset;

            // === PASS 1: Calculate Mean of Squares ===
            __m512 sum_sq_vec = _mm512_setzero_ps();

            // Only process actual hidden_size elements, not padded ones
            for (int i = 0; i < hidden_size; i += AVX512_F32_LANE_WIDTH) {
                // Create a mask for the last chunk if hidden_size is not a multiple of 16
                const int remaining = hidden_size - i;
                const __mmask16 mask = (remaining >= AVX512_F32_LANE_WIDTH) ? 0xFFFF : (1U << remaining) - 1;

                // Load 16 bfloat16 values from INPUT, convert to float32
                __m256i bf16_data_i = _mm256_maskz_loadu_epi16(mask, input_row_ptr + i); // MODIFIED
                #ifdef __WINDOWS__
                __m512 f_vec = _mm512_cvtpbh_ps(bf16_data_i);
                #else
                __m512 f_vec = _mm512_cvtpbh_ps(reinterpret_cast<__m256bh>(bf16_data_i));
                #endif

                // Accumulate the sum of squares (x*x)
                sum_sq_vec = _mm512_fmadd_ps(f_vec, f_vec, sum_sq_vec);
            }

            // Horizontally sum the squared values across the AVX-512 vector
            const float total_sum_sq = _mm512_reduce_add_ps(sum_sq_vec);

            // Calculate the inverse of the RMS value
            const float mean_sq = total_sum_sq * inv_hidden_size;
            const float inv_rms = 1.0f / std::sqrt(mean_sq + eps);

            const __m512 inv_rms_vec = _mm512_set1_ps(inv_rms);

            // === PASS 2: Normalize, Scale, and Store ===
            // Process all elements including padded ones to maintain memory layout
            for (int i = 0; i < hidden_size_padded; i += AVX512_F32_LANE_WIDTH) {
                // Process actual data elements within the hidden size
                if (i < hidden_size) {
                    const int remaining = hidden_size - i;
                    const __mmask16 mask = (remaining >= AVX512_F32_LANE_WIDTH) ? 0xFFFF : (1U << remaining) - 1;

                    // Load bfloat16 data from INPUT and convert to float32
                    __m256i bf16_data_i = _mm256_maskz_loadu_epi16(mask, input_row_ptr + i); // MODIFIED
                    #ifdef __WINDOWS__
                    __m512 f_vec = _mm512_cvtpbh_ps(bf16_data_i);
                    #else
                    __m512 f_vec = _mm512_cvtpbh_ps(reinterpret_cast<__m256bh>(bf16_data_i));
                    #endif

                    // Load bfloat16 weights and convert to float32
                    __m256i w_bf16_data_i = _mm256_maskz_loadu_epi16(mask, weight + i);
                    #ifdef __WINDOWS__
                    __m512 w_vec = _mm512_cvtpbh_ps(w_bf16_data_i);
                    #else
                    __m512 w_vec = _mm512_cvtpbh_ps(reinterpret_cast<__m256bh>(w_bf16_data_i));
                    #endif

                    // Normalize: x * (1 / sqrt(mean(x^2) + eps))
                    __m512 norm_vec = _mm512_mul_ps(f_vec, inv_rms_vec);
                    // Scale: normalized_x * weight
                    __m512 result_vec = _mm512_mul_ps(norm_vec, w_vec);

                    // Convert float32 result back to bfloat16
                    #ifdef __WINDOWS__
                    __m256i result_bf16 = _mm512_cvtneps_pbh(result_vec);
                    #else
                    __m256i result_bf16 = reinterpret_cast<__m256i>(_mm512_cvtneps_pbh(result_vec));
                    #endif

                    // Store the result back into the OUTPUT buffer
                    _mm256_mask_storeu_epi16(output_row_ptr + i, mask, result_bf16); // MODIFIED
                } else {
                    // For padded elements, set them to zero in the OUTPUT buffer
                    const int remaining_elements = hidden_size_padded - i;
                    const __mmask16 mask = (remaining_elements >= AVX512_F32_LANE_WIDTH) ? 0xFFFF : (1 << remaining_elements) - 1;
                    __m256i zero_data = _mm256_setzero_si256();
                    _mm256_mask_storeu_epi16(output_row_ptr + i, mask, zero_data); // MODIFIED
                }
            }
        }
    }
}

// a = a+b
void vector_add_bfloat16(biovault::bfloat16_t *dest,biovault::bfloat16_t *a,  biovault::bfloat16_t *b, size_t size) {
    size_t i = 0;

    // Process 16 elements at a time using AVX512 instructions.
    for (; i + 15 < size; i += 16) {
        // --- UP-CONVERSION from bfloat16 to float ---
        __m256i a_bh_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
        __m256i b_bh_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));

        __m512i a_int_vec = _mm512_cvtepu16_epi32(a_bh_vec);
        __m512i b_int_vec = _mm512_cvtepu16_epi32(b_bh_vec);

        __m512i a_shifted = _mm512_slli_epi32(a_int_vec, 16);
        __m512i b_shifted = _mm512_slli_epi32(b_int_vec, 16);

        __m512 a_ps_vec = _mm512_castsi512_ps(a_shifted);
        __m512 b_ps_vec = _mm512_castsi512_ps(b_shifted);

        // --- FLOATING POINT ADDITION ---
        __m512 sum_ps_vec = _mm512_add_ps(a_ps_vec, b_ps_vec);

        // --- DOWN-CONVERSION from float to bfloat16 ---
        __m256bh result_bh_vec = _mm512_cvtneps_pbh(sum_ps_vec);

        // Store (MSVC-compatible cast)
        #ifdef __WINDOWS__
        _mm256_storeu_si256(
            reinterpret_cast<__m256i*>(dest + i),
            _mm256_castph_si256(result_bh_vec)
        );
        #else
        _mm256_storeu_si256(
            reinterpret_cast<__m256i*>(dest + i),
            reinterpret_cast<__m256i>(result_bh_vec)
        );
        #endif
    }

    // Scalar fallback for tail
    for (; i < size; ++i) {
        dest[i] = static_cast<biovault::bfloat16_t>(
            static_cast<float>(a[i]) + static_cast<float>(b[i])
        );
    }
}

// from average pooling.hpp
/**
 * @brief Performs in-place 2D average pooling using AVX-512 intrinsics.
 *
 * This function treats each sequence in a batch as a 2D grid and applies average pooling.
 * The operation is performed in-place and is optimized by vectorizing across the hidden dimension.
 *
 * PREREQUISITE: For this AVX-512 version, `hidden_size_padded` must be a multiple of 16.
 *
 * @param input_output Pointer to the data buffer. Data is read from and written to this buffer.
 * @param batch_size The number of sequences in the batch.
 * @param seq_len The original logical sequence length (not used in calculation, for context).
 * @param seq_len_padded The padded sequence length, must be a perfect square.
 * @param hidden_size The original logical hidden size (not used in calculation, for context).
 * @param hidden_size_padded The padded hidden size. Must be a multiple of 16.
 * @param kernel The size of the pooling window (e.g., 2 for a 2x2 kernel).
 * @param stride The step size of the pooling window.
 * @param padding The padding size. Currently, only padding=0 is supported.
 * @return The new sequence length after pooling, or -1 on error.
 */
int average_pooling_avx512(
    biovault::bfloat16_t* __restrict input_output, // [batch_size, seq_len_padded, hidden_size_padded]
    int batch_size,
    int seq_len,
    int seq_len_padded,
    int hidden_size,
    int hidden_size_padded,
    int kernel,
    int stride,
    int padding
) {
    if (padding != 0) {
        return -1; // Only padding=0 is supported.
    }
    if (hidden_size_padded % 16 != 0) {
        return -1;
    }

    const double side_len_double = std::sqrt(static_cast<double>(seq_len_padded));
    if (side_len_double != static_cast<int>(side_len_double)) {
        return -1;
    }
    const int side_len = static_cast<int>(side_len_double);

    if (kernel <= 0 || stride <= 0 || side_len < kernel || (side_len - kernel) % stride != 0) {
        return -1;
    }

    const int new_side_len = (side_len - kernel) / stride + 1;
    const int new_seq_len  = new_side_len * new_side_len;

    const float pool_area = static_cast<float>(kernel * kernel);
    const __m512 divisor_vec = _mm512_set1_ps(pool_area);

    #pragma omp parallel for schedule(static) num_threads(TOTAL_OPENMP_THREAD)
    for (int b = 0; b < batch_size; ++b) {
        for (int out_y = 0; out_y < new_side_len; ++out_y) {
            for (int out_x = 0; out_x < new_side_len; ++out_x) {
                for (int h = 0; h < hidden_size_padded; h += 16) {
                    __m512 sum_vec = _mm512_setzero_ps();

                    const int in_y_start = out_y * stride;
                    const int in_x_start = out_x * stride;

                    for (int ky = 0; ky < kernel; ++ky) {
                        for (int kx = 0; kx < kernel; ++kx) {
                            const int in_y = in_y_start + ky;
                            const int in_x = in_x_start + kx;
                            const int s = in_y * side_len + in_x;

                            const size_t read_idx =
                                static_cast<size_t>(b) * seq_len_padded * hidden_size_padded +
                                static_cast<size_t>(s) * hidden_size_padded +
                                h;

                            __m256i bf16_vec_int =
                                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&input_output[read_idx]));
                            __m512i extended_vec = _mm512_cvtepu16_epi32(bf16_vec_int);
                            __m512i shifted_vec  = _mm512_slli_epi32(extended_vec, 16);
                            __m512 float_vec     = _mm512_castsi512_ps(shifted_vec);

                            sum_vec = _mm512_add_ps(sum_vec, float_vec);
                        }
                    }

                    __m512 average_vec = _mm512_div_ps(sum_vec, divisor_vec);

                    const int new_s = out_y * new_side_len + out_x;
                    const size_t write_idx =
                        static_cast<size_t>(b) * new_seq_len * hidden_size_padded +
                        static_cast<size_t>(new_s) * hidden_size_padded +
                        h;

                    __m256bh result_bf16_vec = _mm512_cvtneps_pbh(average_vec);

                    #ifdef __WINDOWS__
                    _mm256_storeu_si256(
                        reinterpret_cast<__m256i*>(&input_output[write_idx]),
                        _mm256_castph_si256(result_bf16_vec)
                    );
                    #else
                    _mm256_storeu_si256(
                        reinterpret_cast<__m256i*>(&input_output[write_idx]),
                        reinterpret_cast<__m256i>(result_bf16_vec)
                    );
                    #endif
                }
            }
        }
    }

    return new_seq_len;
}
