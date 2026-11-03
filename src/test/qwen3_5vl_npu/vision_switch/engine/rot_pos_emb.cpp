#include "vision/rot_pos_emb.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <stdfloat>
#include <cstring>
#include "qwen3_5vl_image.hpp"
#include "utils/avx512_util.hpp"
#include <immintrin.h>

std::vector<float> generate_rotary_freq_table(
    int32_t seqlen,
    int32_t dim,
    float theta
) {
    // Generate inverse frequencies: inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
    // torch.arange(0, dim, 2) creates [0, 2, 4, ..., dim-2] with length dim/2
    int32_t inv_freq_len = dim / 2;
    std::vector<float> inv_freq(inv_freq_len);

    // AVX-512 optimized version for computing inverse frequencies
    __m512 v_dim = _mm512_set1_ps(static_cast<float>(dim));
    float log_theta = std::log(theta);
    __m512 v_log_theta = _mm512_set1_ps(log_theta);

    int32_t i = 0;
    // Process 16 elements at a time with AVX-512
    for (; i + 15 < inv_freq_len; i += 16) {
        // Create indices: [i*2, (i+1)*2, (i+2)*2, ..., (i+15)*2]
        __m512 indices = _mm512_setr_ps(
            (i+0)*2.0f, (i+1)*2.0f, (i+2)*2.0f, (i+3)*2.0f,
            (i+4)*2.0f, (i+5)*2.0f, (i+6)*2.0f, (i+7)*2.0f,
            (i+8)*2.0f, (i+9)*2.0f, (i+10)*2.0f, (i+11)*2.0f,
            (i+12)*2.0f, (i+13)*2.0f, (i+14)*2.0f, (i+15)*2.0f
        );

        // exponent = indices / dim
        __m512 exponent = _mm512_div_ps(indices, v_dim);

        // inv_freq[i:i+16] = 1.0 / (theta ** exponent)
        // Using exp(exponent * log(theta)) for pow
        __m512 exp_arg = _mm512_mul_ps(exponent, v_log_theta);

        // Compute exp using scalar math (no native AVX-512 exp)
        float temp[16];
        _mm512_storeu_ps(temp, exp_arg);
        for (int j = 0; j < 16; ++j) {
            inv_freq[i + j] = 1.0f / std::exp(temp[j]);
        }
    }

    // Handle remaining elements
    for (; i < inv_freq_len; ++i) {
        float exponent = static_cast<float>(i * 2) / static_cast<float>(dim);
        inv_freq[i] = 1.0f / std::pow(theta, exponent);
    }

    // Generate sequence positions: seq = torch.arange(seqlen)
    // Compute outer product: freqs = torch.outer(seq, inv_freq)
    // Result shape: [seqlen, inv_freq_len] = [seqlen, dim/2]
    std::vector<float> freqs(seqlen * inv_freq_len);

    // AVX-512 optimized outer product
    for (int32_t pos = 0; pos < seqlen; ++pos) {
        __m512 v_pos = _mm512_set1_ps(static_cast<float>(pos));
        float* freq_row = &freqs[pos * inv_freq_len];

        int32_t d = 0;
        // Process 16 elements at a time
        for (; d + 15 < inv_freq_len; d += 16) {
            __m512 v_inv_freq = _mm512_loadu_ps(&inv_freq[d]);
            __m512 result = _mm512_mul_ps(v_pos, v_inv_freq);
            _mm512_storeu_ps(&freq_row[d], result);
        }

        // Handle remaining elements (should be 0 for inv_freq_len=16)
        for (; d < inv_freq_len; ++d) {
            freq_row[d] = static_cast<float>(pos) * inv_freq[d];
        }
    }

    return freqs;
}

std::vector<float> rot_pos_emb_float(
    const std::vector<std::vector<int32_t>>& grid_thw,
    int32_t spatial_merge_size,
    int32_t rot_dim,
    float theta
) {
    // Find max_hw from grid_thw (max of height and width across all images)
    int32_t max_hw = 0;
    for (const auto& thw : grid_thw) {
        max_hw = std::max(max_hw, std::max(thw[1], thw[2]));
    }

    // Generate frequency table: freq_table = self.rotary_pos_emb(max_hw)
    // The rot_dim passed here is head_dim/2, so freq table should be [max_hw, rot_dim/2]
    std::vector<float> freq_table = generate_rotary_freq_table(max_hw, rot_dim, theta);
    int32_t freq_dim = rot_dim / 2;  // Actual dimension of freq_table per position

    // Calculate total number of tokens
    size_t total_tokens = 0;
    for (const auto& thw : grid_thw) {
        int32_t t = thw[0];
        int32_t h = thw[1];
        int32_t w = thw[2];
        total_tokens += static_cast<size_t>(t) * h * w;
    }

    // Allocate output embeddings directly
    std::vector<float> embeddings(total_tokens * rot_dim);

    // Process each image/video in grid_thw
    size_t offset = 0;
    for (const auto& thw : grid_thw) {
        int32_t num_frames = thw[0];
        int32_t height = thw[1];
        int32_t width = thw[2];

        int32_t merged_h = height / spatial_merge_size;
        int32_t merged_w = width / spatial_merge_size;

        // Pre-compute coordinates for one frame (avoid recomputing for each frame)
        size_t hw_size = static_cast<size_t>(height) * width;
        std::vector<int32_t> row_coords(hw_size);
        std::vector<int32_t> col_coords(hw_size);

        // Generate coordinates with optimized loop order
        size_t coord_idx = 0;
        for (int32_t block_h = 0; block_h < merged_h; ++block_h) {
            for (int32_t block_w = 0; block_w < merged_w; ++block_w) {
                for (int32_t intra_h = 0; intra_h < spatial_merge_size; ++intra_h) {
                    int32_t row_idx = block_h * spatial_merge_size + intra_h;
                    for (int32_t intra_w = 0; intra_w < spatial_merge_size; ++intra_w) {
                        int32_t col_idx = block_w * spatial_merge_size + intra_w;
                        row_coords[coord_idx] = row_idx;
                        col_coords[coord_idx] = col_idx;
                        ++coord_idx;
                    }
                }
            }
        }

        // Process all frames with AVX-512 optimized embedding lookup
        for (int32_t frame = 0; frame < num_frames; ++frame) {
            for (size_t i = 0; i < hw_size; ++i) {
                int32_t row_pos = row_coords[i];
                int32_t col_pos = col_coords[i];

                float* emb_ptr = &embeddings[offset * rot_dim];
                const float* row_freq = &freq_table[row_pos * freq_dim];
                const float* col_freq = &freq_table[col_pos * freq_dim];

                // AVX-512 optimized copy: Process 16 elements at a time
                int32_t d = 0;
                for (; d + 15 < freq_dim; d += 16) {
                    // Load and store row embeddings
                    __m512 row_emb = _mm512_loadu_ps(&row_freq[d]);
                    _mm512_storeu_ps(&emb_ptr[d], row_emb);

                    // Load and store col embeddings
                    __m512 col_emb = _mm512_loadu_ps(&col_freq[d]);
                    _mm512_storeu_ps(&emb_ptr[freq_dim + d], col_emb);
                }

                // Handle remaining elements (should be 0 for freq_dim=16)
                for (; d < freq_dim; ++d) {
                    emb_ptr[d] = row_freq[d];
                    emb_ptr[freq_dim + d] = col_freq[d];
                }

                ++offset;
            }
        }
    }

    return embeddings;
}

void cos_sine_pose_emb(
    const std::vector<float> &rotary_emb,
    std::vector<float> &cos_emb,
    std::vector<float> &sin_emb,
    int QWEN3_5_VISION_HEAD_DIM
) {
    // rotary_emb is [seq_len, rot_dim] where rot_dim = head_dim / 2 = 32
    // stored as flattened 1D array: total_elements = seq_len * rot_dim
    // We need to concatenate each row with itself: [seq_len, rot_dim * 2] = [seq_len, 64]
    // Then compute cos and sin

    int32_t rot_dim = QWEN3_5_VISION_HEAD_DIM/2;  // QWEN3_VISION_HEAD_DIM / 2
    int32_t output_dim = QWEN3_5_VISION_HEAD_DIM;  // QWEN3_VISION_HEAD_DIM

    size_t total_elements = rotary_emb.size();
    int32_t seq_len = total_elements / rot_dim;

    // Ensure output vectors are properly sized
    cos_emb.resize(seq_len * output_dim);
    sin_emb.resize(seq_len * output_dim);

    // Process each sequence position (row) with AVX-512
    for (int32_t seq_idx = 0; seq_idx < seq_len; ++seq_idx) {
        const float* input_row = &rotary_emb[seq_idx * rot_dim];
        float* cos_output_row = &cos_emb[seq_idx * output_dim];
        float* sin_output_row = &sin_emb[seq_idx * output_dim];

        int32_t i = 0;

        // Process 16 elements at a time with AVX-512
        for (; i + 15 < rot_dim; i += 16) {
            // Load 16 float values from input row
            __m512 float_vals = _mm512_loadu_ps(&input_row[i]);

            // Compute cosine and sine using scalar fallback
            // (no native AVX-512 cos/sin without SVML)
            float temp_val[16];
            float cos_temp[16], sin_temp[16];
            _mm512_storeu_ps(temp_val, float_vals);
            for (int j = 0; j < 16; ++j) {
                cos_temp[j] = std::cos(temp_val[j]);
                sin_temp[j] = std::sin(temp_val[j]);
            }
            __m512 cos_vals = _mm512_loadu_ps(cos_temp);
            __m512 sin_vals = _mm512_loadu_ps(sin_temp);

            // Store to BOTH first and second half of output (as float)
            // Store to first half [0:32]
            _mm512_storeu_ps(&cos_output_row[i], cos_vals);
            _mm512_storeu_ps(&sin_output_row[i], sin_vals);

            // Store to second half [32:64] - duplicate
            _mm512_storeu_ps(&cos_output_row[rot_dim + i], cos_vals);
            _mm512_storeu_ps(&sin_output_row[rot_dim + i], sin_vals);
        }

        // Handle remaining elements (should be 0 for rot_dim=32)
        for (; i < rot_dim; ++i) {
            float val = input_row[i];
            float cos_val = std::cos(val);
            float sin_val = std::sin(val);

            // Store to first half
            cos_output_row[i] = cos_val;
            sin_output_row[i] = sin_val;

            // Store to second half (duplicate)
            cos_output_row[rot_dim + i] = cos_val;
            sin_output_row[rot_dim + i] = sin_val;
        }
    }
}

/**
 * Apply rotary position embeddings to Q and K in-place within mm_res buffer.
 * New Layout: [3 * num_heads, seq_len, head_dim]
 * * @param mm_res_ptr Pointer to buffer of shape [3 * num_heads, seq_len, head_dim]
 * Structure: [Block Q (Heads 0..H-1)] | [Block K (Heads 0..H-1)] | [Block V...]
 * @param cos_emb_ptr Pointer to cosine embeddings of shape [seq_len, head_dim] (float)
 * @param sin_emb_ptr Pointer to sine embeddings of shape [seq_len, head_dim] (float)
 * @param seq_len The L_Seq dimension size
 * @param hidden_size Hidden dimension size (total across all heads)
 * @param num_heads Number of attention heads
 */
void apply_rotary_pos_emb_vision_inplace_reordered(
    bf16* mm_res_ptr,
    const float* cos_emb_ptr,
    const float* sin_emb_ptr,
    int seq_len,
    int seq_len_padded,
    int hidden_size,
    int hidden_size_padded,
    int num_heads
) {
    const int head_dim = hidden_size / num_heads;
    const int head_dim_padded = hidden_size_padded / num_heads;
    const int half_head_dim = head_dim / 2;

    // Stride calculation based on [3 * num_heads, seq_len_padded, head_dim]
    // The distance in elements between one head and the next is (seq_len_padded * head_dim)
    const size_t head_stride = (size_t)seq_len_padded * head_dim_padded;

    // Parallelize over heads (Q and K pairs).
    // We have 'num_heads' pairs of Q/K to process.
    #pragma omp parallel for schedule(static)
    for (int h = 0; h < num_heads; ++h) {

        // Calculate base pointers for Head 'h' of Q and Head 'h' of K.
        // Q block starts at 0. K block starts at num_heads offset.
        bf16* q_head_base = mm_res_ptr + (h * head_stride);
        bf16* k_head_base = mm_res_ptr + ((h + num_heads) * head_stride);

        // Iterate over sequence length for this specific head
        for (int s = 0; s < seq_len; ++s) {

            // Pointers to the specific vector [head_dim] for this sequence position
            bf16* q_ptr = q_head_base + s * head_dim_padded;
            bf16* k_ptr = k_head_base + s * head_dim_padded;

            // Pointers to cos/sin for current sequence position
            // cos/sin shape is [seq_len, head_dim], so stride is head_dim
            const float* cos_ptr = cos_emb_ptr + s * head_dim;
            const float* sin_ptr = sin_emb_ptr + s * head_dim;

            // Apply Rotary Embedding using AVX-512
            for (int d = 0; d < half_head_dim; d += 16) {
                int remaining = half_head_dim - d;
                __mmask16 mask = (remaining >= 16) ? (__mmask16)0xFFFF : (__mmask16)((1u << remaining) - 1);

                // Masked bf16 load -> float
                __m256i q1_raw = _mm256_maskz_loadu_epi16(mask, &q_ptr[d]);
                __m512 q1 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(q1_raw), 16));
                __m256i q2_raw = _mm256_maskz_loadu_epi16(mask, &q_ptr[d + half_head_dim]);
                __m512 q2 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(q2_raw), 16));

                __m256i k1_raw = _mm256_maskz_loadu_epi16(mask, &k_ptr[d]);
                __m512 k1 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(k1_raw), 16));
                __m256i k2_raw = _mm256_maskz_loadu_epi16(mask, &k_ptr[d + half_head_dim]);
                __m512 k2 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(k2_raw), 16));

                // Load cos/sin values (shared for Q and K)
                __m512 cos_val = _mm512_maskz_loadu_ps(mask, &cos_ptr[d]);
                __m512 sin_val = _mm512_maskz_loadu_ps(mask, &sin_ptr[d]);

                // Apply rotary formulas:
                // q_new_1 = q1 * cos - q2 * sin
                // q_new_2 = q2 * cos + q1 * sin
                __m512 q_res_1 = _mm512_fmsub_ps(q1, cos_val, _mm512_mul_ps(q2, sin_val));
                __m512 q_res_2 = _mm512_fmadd_ps(q2, cos_val, _mm512_mul_ps(q1, sin_val));

                __m512 k_res_1 = _mm512_fmsub_ps(k1, cos_val, _mm512_mul_ps(k2, sin_val));
                __m512 k_res_2 = _mm512_fmadd_ps(k2, cos_val, _mm512_mul_ps(k1, sin_val));

                // Masked store back to bf16 (round-to-nearest-even)
                __m512i q_res_1_rnd = _mm512_srli_epi32(_mm512_add_epi32(_mm512_castps_si512(q_res_1), _mm512_set1_epi32(0x7FFF)), 16);
                _mm256_mask_storeu_epi16(&q_ptr[d], mask, _mm512_cvtepi32_epi16(q_res_1_rnd));
                __m512i q_res_2_rnd = _mm512_srli_epi32(_mm512_add_epi32(_mm512_castps_si512(q_res_2), _mm512_set1_epi32(0x7FFF)), 16);
                _mm256_mask_storeu_epi16(&q_ptr[d + half_head_dim], mask, _mm512_cvtepi32_epi16(q_res_2_rnd));

                __m512i k_res_1_rnd = _mm512_srli_epi32(_mm512_add_epi32(_mm512_castps_si512(k_res_1), _mm512_set1_epi32(0x7FFF)), 16);
                _mm256_mask_storeu_epi16(&k_ptr[d], mask, _mm512_cvtepi32_epi16(k_res_1_rnd));
                __m512i k_res_2_rnd = _mm512_srli_epi32(_mm512_add_epi32(_mm512_castps_si512(k_res_2), _mm512_set1_epi32(0x7FFF)), 16);
                _mm256_mask_storeu_epi16(&k_ptr[d + half_head_dim], mask, _mm512_cvtepi32_epi16(k_res_2_rnd));
            }
        }
    }
}

/**
 * Apply rotary position embeddings to Q and K in-place within mm_res buffer (float cos/sin version).
 *
 * @param mm_res_ptr Pointer to row-major matrix of shape [seq_len_padded, 3*hidden_size]
 *                   containing concatenated Q, K, V where each row is [Q|K|V]
 * @param cos_emb_ptr Pointer to cosine embeddings of shape [seq_len, head_dim] (float)
 * @param sin_emb_ptr Pointer to sine embeddings of shape [seq_len, head_dim] (float)
 * @param seq_len Actual sequence length (without padding)
 * @param seq_len_padded Padded sequence length (for memory layout)
 * @param hidden_size Hidden dimension size (1024 for Qwen3-VL)
 * @param num_heads Number of attention heads (16 for Qwen3-VL)
 */
void apply_rotary_pos_emb_vision_inplace(
    bf16* mm_res_ptr,
    const float* cos_emb_ptr,
    const float* sin_emb_ptr,
    int seq_len,
    int seq_len_padded,
    int hidden_size,
    int per_seqlen_row_stride,
    int num_heads
) {
    const int head_dim = hidden_size / num_heads;
    const int half_head_dim = head_dim / 2;

    // Use OpenMP only if seq_len is large enough to benefit from parallelization
    // Threshold: 8 sequences per thread minimum to avoid overhead
    const int min_seq_per_thread = 8;
    const int max_threads = 4;
    const bool use_parallel = seq_len >= (min_seq_per_thread * 2);

    // Process each sequence position
    #pragma omp parallel for num_threads(max_threads) if(use_parallel) schedule(static)
    for (int s = 0; s < seq_len; ++s) {
        // Pointer to current row in mm_res: [Q|K|V]
        bf16* row_ptr = mm_res_ptr + s * (per_seqlen_row_stride);

        // Pointers to Q, K, V within the row
        bf16* q_ptr = row_ptr;                    // First hidden_size elements
        bf16* k_ptr = row_ptr + hidden_size;      // Next hidden_size elements

        // Pointer to cos/sin for current sequence position (float)
        const float* cos_ptr = cos_emb_ptr + s * head_dim;
        const float* sin_ptr = sin_emb_ptr + s * head_dim;

        // Apply rotary embeddings to each head in Q and K
        for (int h = 0; h < num_heads; ++h) {
            bf16* q_head = q_ptr + h * head_dim;
            bf16* k_head = k_ptr + h * head_dim;

            // Process with AVX-512: work on up to 16 elements at a time, masked for tail
            for (int d = 0; d < half_head_dim; d += 16) {
                int remaining = half_head_dim - d;
                __mmask16 mask = (remaining >= 16) ? (__mmask16)0xFFFF : (__mmask16)((1u << remaining) - 1);

                // Masked bf16 load -> float
                __m256i q1_raw = _mm256_maskz_loadu_epi16(mask, &q_head[d]);
                __m512 q1 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(q1_raw), 16));
                __m256i q2_raw = _mm256_maskz_loadu_epi16(mask, &q_head[d + half_head_dim]);
                __m512 q2 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(q2_raw), 16));

                __m256i k1_raw = _mm256_maskz_loadu_epi16(mask, &k_head[d]);
                __m512 k1 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(k1_raw), 16));
                __m256i k2_raw = _mm256_maskz_loadu_epi16(mask, &k_head[d + half_head_dim]);
                __m512 k2 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(k2_raw), 16));

                // Load cos/sin values
                __m512 cos_val = _mm512_maskz_loadu_ps(mask, &cos_ptr[d]);
                __m512 sin_val = _mm512_maskz_loadu_ps(mask, &sin_ptr[d]);

                // Apply rotary embedding formulas:
                // q_new_1 = q1 * cos - q2 * sin
                // q_new_2 = q2 * cos + q1 * sin
                __m512 q_res_1 = _mm512_fmsub_ps(q1, cos_val, _mm512_mul_ps(q2, sin_val));
                __m512 q_res_2 = _mm512_fmadd_ps(q2, cos_val, _mm512_mul_ps(q1, sin_val));

                __m512 k_res_1 = _mm512_fmsub_ps(k1, cos_val, _mm512_mul_ps(k2, sin_val));
                __m512 k_res_2 = _mm512_fmadd_ps(k2, cos_val, _mm512_mul_ps(k1, sin_val));

                // Masked store back to bf16 (round-to-nearest-even)
                __m512i q_res_1_rnd = _mm512_srli_epi32(_mm512_add_epi32(_mm512_castps_si512(q_res_1), _mm512_set1_epi32(0x7FFF)), 16);
                _mm256_mask_storeu_epi16(&q_head[d], mask, _mm512_cvtepi32_epi16(q_res_1_rnd));
                __m512i q_res_2_rnd = _mm512_srli_epi32(_mm512_add_epi32(_mm512_castps_si512(q_res_2), _mm512_set1_epi32(0x7FFF)), 16);
                _mm256_mask_storeu_epi16(&q_head[d + half_head_dim], mask, _mm512_cvtepi32_epi16(q_res_2_rnd));

                __m512i k_res_1_rnd = _mm512_srli_epi32(_mm512_add_epi32(_mm512_castps_si512(k_res_1), _mm512_set1_epi32(0x7FFF)), 16);
                _mm256_mask_storeu_epi16(&k_head[d], mask, _mm512_cvtepi32_epi16(k_res_1_rnd));
                __m512i k_res_2_rnd = _mm512_srli_epi32(_mm512_add_epi32(_mm512_castps_si512(k_res_2), _mm512_set1_epi32(0x7FFF)), 16);
                _mm256_mask_storeu_epi16(&k_head[d + half_head_dim], mask, _mm512_cvtepi32_epi16(k_res_2_rnd));
            }
        }
    }
}
