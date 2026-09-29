#include "gemma4e_audio_attention.hpp"
#include "gemma4e_vision_prefill_helper.hpp"
#include "avx512_util.hpp"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <vector>
#include <immintrin.h>
#include <omp.h>

void create_sliding_window_attention_mask(
    int seq_len,
    int sliding_window_left,
    int sliding_window_right,
    std::vector<int> &mask

){
    mask.assign(seq_len * seq_len, 0);
    for (int q_idx = 0; q_idx < seq_len; ++q_idx) {
        for (int kv_idx = 0; kv_idx < seq_len; ++kv_idx) {
            int dist = q_idx - kv_idx;
            bool left_mask = (dist >= 0) && (dist < sliding_window_left);
            bool right_mask = (dist < 0) && (-dist < sliding_window_right);

            if (left_mask || right_mask) {
                mask[q_idx * seq_len + kv_idx] = 1;
            } else {
                mask[q_idx * seq_len + kv_idx] = 0;
            }
        }
    }
}
void convert_mask_to_blocked(
    std::vector<int> &input_mask, // [seq_len, seq_len]
    int chunk_size,
    int attention_context_left,
    int attention_context_right,
    std::vector<int> &blocked_mask, // [num_chunks, chunk_size, context_size],
    int &num_blocks,
    int &context_size
) {
    int seq_len = std::round(std::sqrt(input_mask.size()));
    int max_past_horizon = attention_context_left - 1;
    int max_future_horizon = attention_context_right;

    num_blocks = (seq_len + chunk_size - 1) / chunk_size;
    context_size = chunk_size + max_past_horizon + max_future_horizon;

    blocked_mask.assign(num_blocks * chunk_size * context_size, 0);

    for (int b = 0; b < num_blocks; ++b) {
        for (int c = 0; c < chunk_size; ++c) {
            int q_idx = b * chunk_size + c;
            for (int ctx = 0; ctx < context_size; ++ctx) {
                int kv_idx = b * chunk_size + ctx - max_past_horizon;
                int out_idx = b * chunk_size * context_size + c * context_size + ctx;

                if (q_idx >= 0 && q_idx < seq_len && kv_idx >= 0 && kv_idx < seq_len) {
                    blocked_mask[out_idx] = input_mask[q_idx * seq_len + kv_idx];
                } else {
                    blocked_mask[out_idx] = 0;
                }
            }
        }
    }
}

void convert_to_block(
    const bf16* input,
    bf16* output,
    int seq_len,
    int chunk_size,
    int row_stride,
    int num_blocks
) {
    // Output is expected to be zero-initialized by caller.
    // The reshape from [num_blocks*chunk_size, row_stride] to [num_blocks, chunk_size, row_stride]
    // is a no-op in row-major memory. We just copy valid rows and leave padding as zeros.
    int total_output_rows = num_blocks * chunk_size;
    int rows_to_copy = std::min(seq_len, total_output_rows);
    memcpy(output, input, (size_t)rows_to_copy * row_stride * sizeof(bf16));
}

void extract_block_context(
    const bf16* input,
    bf16* output,
    int seq_len,
    int chunk_size,
    int max_past_horizon,
    int max_future_horizon,
    int row_stride,
    int num_blocks,
    int context_size
) {
    // 1. Create padded buffer: [max_past_horizon + seq_len + max_future_horizon + chunk_size - 1, row_stride]
    //    Left padding: max_past_horizon zeros, Right padding: max_future_horizon + chunk_size - 1 zeros
    int padded_len = max_past_horizon + seq_len + max_future_horizon + chunk_size - 1;
    std::vector<bf16> padded((size_t)padded_len * row_stride, bf16(0));

    // Copy input into padded buffer at offset max_past_horizon
    memcpy(
        padded.data() + (size_t)max_past_horizon * row_stride,
        input,
        (size_t)seq_len * row_stride * sizeof(bf16)
    );

    // 2. Extract overlapping windows (unfold): for block b, copy context_size rows starting at b*chunk_size
    for (int b = 0; b < num_blocks; b++) {
        int src_start = b * chunk_size;
        memcpy(
            output + (size_t)(b * context_size) * row_stride,
            padded.data() + (size_t)src_start * row_stride,
            (size_t)context_size * row_stride * sizeof(bf16)
        );
    }
}

// ============================================================================
// AVX512 dot product of bf16 vectors (returns float32)
// ============================================================================
static inline float avx512_dot_bf16(const bf16* a, const bf16* b, int len) {
    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();
    int k = 0;
    for (; k + 32 <= len; k += 32) {
        acc0 = _mm512_fmadd_ps(load_bfloat16_to_m512(a + k),
                                load_bfloat16_to_m512(b + k), acc0);
        acc1 = _mm512_fmadd_ps(load_bfloat16_to_m512(a + k + 16),
                                load_bfloat16_to_m512(b + k + 16), acc1);
    }
    __m512 acc = _mm512_add_ps(acc0, acc1);
    for (; k + 16 <= len; k += 16) {
        acc = _mm512_fmadd_ps(load_bfloat16_to_m512(a + k),
                               load_bfloat16_to_m512(b + k), acc);
    }
    float sum = _mm512_reduce_add_ps(acc);
    for (; k < len; k++) {
        sum += (float)a[k] * (float)b[k];
    }
    return sum;
}

// ============================================================================
// _rel_shift for one block: [chunk_size, position_length] -> [chunk_size, context_size]
// Python equivalent:
//   x = F.pad(x, (0, context_size + 1 - position_length))
//   x = x.view(..., block_size * (context_size + 1))
//   x = x[..., : block_size * context_size]
//   x = x.view(..., block_size, context_size)
// ============================================================================
static void rel_shift_block(
    const float* input,   // [chunk_size, position_length]
    float* output,        // [chunk_size, context_size]
    int chunk_size,
    int position_length,
    int context_size
) {
    int padded_col = context_size + 1;
    int padded_size = chunk_size * padded_col;
    // Use stack buffer for small sizes (typical: 12*25=300 floats = 1.2KB), heap fallback otherwise
    float stack_buf[512];
    float* padded = (padded_size <= 512) ? stack_buf : new float[padded_size];
    memset(padded, 0, padded_size * sizeof(float));
    for (int c = 0; c < chunk_size; c++) {
        memcpy(&padded[c * padded_col], &input[c * position_length],
               position_length * sizeof(float));
    }
    memcpy(output, padded, chunk_size * context_size * sizeof(float));
    if (padded_size > 512) delete[] padded;
}

void compute_audio_blocked_attention(
    const bf16* q_blocked,
    const bf16* k_blocked,
    const bf16* v_blocked,
    const bf16* relative_key_states,
    const int* block_attention_mask,
    bf16* output,
    int seq_len,
    int num_blocks,
    int chunk_size,
    int context_size,
    int num_positions,
    int num_heads,
    int head_dim,
    int hidden_size,
    int padded_hidden_size,
    float softcap,
    float invalid_logits_value
) {
    memset(output, 0, (size_t)seq_len * padded_hidden_size * sizeof(bf16));

    int total_q_rows = num_blocks * chunk_size;
    float inv_softcap = 1.0f / softcap;

    // Pre-allocate per-thread buffers to avoid repeated heap allocation inside OMP loop
    std::vector<std::vector<float>> thread_matrix_bd(num_heads, std::vector<float>(total_q_rows * num_positions));
    std::vector<std::vector<float>> thread_shifted_bd(num_heads, std::vector<float>(total_q_rows * context_size));
    std::vector<std::vector<float>> thread_attn_weights(num_heads, std::vector<float>(chunk_size * context_size));

    // Parallelize across heads — each head writes to non-overlapping columns
    #pragma omp parallel for num_threads(max_prefill_threads) schedule(static)
    for (int h = 0; h < num_heads; h++) {
        int h_offset = h * head_dim;
        int hd_vecs = head_dim / 16;

        float* matrix_bd = thread_matrix_bd[h].data();
        float* shifted_bd = thread_shifted_bd[h].data();
        float* attn_weights = thread_attn_weights[h].data();

        // ---- Step 1: matrix_bd = queries_flat_h @ rel_K_h^T ----
        // queries_flat_h: [total_q_rows, head_dim] (head slice of q_blocked)
        // rel_K_h:        [num_positions, head_dim] (head slice of relative_key_states)
        // matrix_bd:      [total_q_rows, num_positions]
        for (int i = 0; i < total_q_rows; i++) {
            const bf16* q_row = q_blocked + (size_t)i * padded_hidden_size + h_offset;
            for (int p = 0; p < num_positions; p++) {
                const bf16* rk_row = relative_key_states + (size_t)p * hidden_size + h_offset;
                matrix_bd[i * num_positions + p] = avx512_dot_bf16(q_row, rk_row, head_dim);
            }
        }

        // ---- Step 2: Reshape to [num_blocks, chunk_size, num_positions] then _rel_shift per block ----
        // _rel_shift: [chunk_size, num_positions] -> [chunk_size, context_size]
        for (int blk = 0; blk < num_blocks; blk++) {
            rel_shift_block(
                &matrix_bd[blk * chunk_size * num_positions],
                &shifted_bd[blk * chunk_size * context_size],
                chunk_size,
                num_positions,
                context_size
            );
        }

        // ---- Step 3: Per-block attention (matrix_ac + bd, softcap, mask, softmax, attn@V) ----
        for (int blk = 0; blk < num_blocks; blk++) {
            // 3a. Compute combined logits: matrix_ac + shifted_bd, apply softcap + mask
            for (int c = 0; c < chunk_size; c++) {
                const bf16* q_row = q_blocked +
                    (size_t)(blk * chunk_size + c) * padded_hidden_size + h_offset;

                for (int ctx = 0; ctx < context_size; ctx++) {
                    const bf16* k_row = k_blocked +
                        (size_t)(blk * context_size + ctx) * padded_hidden_size + h_offset;

                    // matrix_ac = Q_row · K_row
                    float w = avx512_dot_bf16(q_row, k_row, head_dim);

                    // + shifted_bd
                    w += shifted_bd[(blk * chunk_size + c) * context_size + ctx];

                    // Softcap: tanh(w / softcap) * softcap
                    w = std::tanh(w * inv_softcap) * softcap;

                    // Mask
                    int mask_idx = blk * chunk_size * context_size + c * context_size + ctx;
                    if (!block_attention_mask[mask_idx]) {
                        w = invalid_logits_value;
                    }

                    attn_weights[c * context_size + ctx] = w;
                }
            }

            // 3b. Softmax over context_size dimension (per chunk row)
            for (int c = 0; c < chunk_size; c++) {
                float* row = &attn_weights[c * context_size];

                // Find max for numerical stability
                __m512 max_vec = _mm512_set1_ps(-1e30f);
                int ctx = 0;
                for (; ctx + 16 <= context_size; ctx += 16) {
                    max_vec = _mm512_max_ps(max_vec, _mm512_loadu_ps(row + ctx));
                }
                float max_val = _mm512_reduce_max_ps(max_vec);
                for (; ctx < context_size; ctx++) {
                    max_val = std::max(max_val, row[ctx]);
                }

                // Exp and sum
                __m512 sum_vec = _mm512_setzero_ps();
                __m512 max_broadcast = _mm512_set1_ps(max_val);
                ctx = 0;
                for (; ctx + 16 <= context_size; ctx += 16) {
                    __m512 val = _mm512_loadu_ps(row + ctx);
                    __m512 exp_val = _mm512_exp_ps_corrected(_mm512_sub_ps(val, max_broadcast));
                    _mm512_storeu_ps(row + ctx, exp_val);
                    sum_vec = _mm512_add_ps(sum_vec, exp_val);
                }
                float sum_exp = _mm512_reduce_add_ps(sum_vec);
                for (; ctx < context_size; ctx++) {
                    row[ctx] = std::exp(row[ctx] - max_val);
                    sum_exp += row[ctx];
                }

                // Normalize (guard against division by zero for fully-masked rows)
                float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
                __m512 inv_sum_vec = _mm512_set1_ps(inv_sum);
                ctx = 0;
                for (; ctx + 16 <= context_size; ctx += 16) {
                    _mm512_storeu_ps(row + ctx,
                        _mm512_mul_ps(_mm512_loadu_ps(row + ctx), inv_sum_vec));
                }
                for (; ctx < context_size; ctx++) {
                    row[ctx] *= inv_sum;
                }
            }

            // 3c. attn_output = attn_weights @ V_block (write to output)
            for (int c = 0; c < chunk_size; c++) {
                int out_seq_idx = blk * chunk_size + c;
                if (out_seq_idx >= seq_len) break;

                bf16* out_ptr = output + (size_t)out_seq_idx * padded_hidden_size + h_offset;

                // Accumulate over context_size using AVX512 across head_dim
                __m512 acc[16]; // supports up to head_dim=256
                for (int v = 0; v < hd_vecs; v++) acc[v] = _mm512_setzero_ps();

                for (int ctx = 0; ctx < context_size; ctx++) {
                    __m512 w_broadcast = _mm512_set1_ps(attn_weights[c * context_size + ctx]);
                    const bf16* v_row = v_blocked +
                        (size_t)(blk * context_size + ctx) * padded_hidden_size + h_offset;

                    for (int v = 0; v < hd_vecs; v++) {
                        __m512 v_vec = load_bfloat16_to_m512(v_row + v * 16);
                        acc[v] = _mm512_fmadd_ps(w_broadcast, v_vec, acc[v]);
                    }
                }

                // Store bf16 result
                for (int v = 0; v < hd_vecs; v++) {
                    store_m512_to_bfloat16(out_ptr + v * 16, acc[v]);
                }

                // Scalar tail (head_dim not multiple of 16)
                int tail_start = hd_vecs * 16;
                for (int d = tail_start; d < head_dim; d++) {
                    float sum = 0.0f;
                    for (int ctx = 0; ctx < context_size; ctx++) {
                        float v_val = (float)v_blocked[
                            (size_t)(blk * context_size + ctx) * padded_hidden_size + h_offset + d];
                        sum += attn_weights[c * context_size + ctx] * v_val;
                    }
                    out_ptr[d] = (bf16)sum;
                }
            }
        }
    }
}

void compute_audio_self_attention(
    const bf16* q_proj_output,
    const bf16* k_proj_output,
    const bf16* v_proj_output,
    const bf16* k_rel_weight,
    const bf16* position_embedding,
    const std::vector<std::vector<int>>& block_attention_mask_per_audio,
    const std::vector<int>& num_blocks_per_audio,
    const std::vector<int>& context_size_per_audio,
    const std::vector<int>& seq_len_per_audio,
    const std::vector<int>& start_seq_len_index_per_audio,
    bf16* output,
    int num_audios,
    int chunk_size,
    int context_left,
    int context_right,
    int num_heads,
    int head_dim,
    int hidden_size,
    int padded_hidden_size,
    int num_positions,
    float softcap,
    float invalid_logits_value
) {
    int max_past_horizon = context_left - 1;
    int max_future_horizon = context_right;

    // ---- Block Q/K/V per audio ----
    std::vector<std::vector<bf16>> blocked_q_per_audio(num_audios);
    std::vector<std::vector<bf16>> blocked_k_per_audio(num_audios);
    std::vector<std::vector<bf16>> blocked_v_per_audio(num_audios);

    for (int i = 0; i < num_audios; i++) {
        int nb = num_blocks_per_audio[i];
        int cs = context_size_per_audio[i];

        blocked_q_per_audio[i].assign((size_t)nb * chunk_size * padded_hidden_size, bf16(0));
        convert_to_block(
            q_proj_output + start_seq_len_index_per_audio[i] * padded_hidden_size,
            blocked_q_per_audio[i].data(),
            seq_len_per_audio[i], chunk_size, padded_hidden_size, nb
        );

        blocked_k_per_audio[i].assign((size_t)nb * cs * padded_hidden_size, bf16(0));
        extract_block_context(
            k_proj_output + start_seq_len_index_per_audio[i] * padded_hidden_size,
            blocked_k_per_audio[i].data(),
            seq_len_per_audio[i], chunk_size, max_past_horizon, max_future_horizon,
            padded_hidden_size, nb, cs
        );

        blocked_v_per_audio[i].assign((size_t)nb * cs * padded_hidden_size, bf16(0));
        extract_block_context(
            v_proj_output + start_seq_len_index_per_audio[i] * padded_hidden_size,
            blocked_v_per_audio[i].data(),
            seq_len_per_audio[i], chunk_size, max_past_horizon, max_future_horizon,
            padded_hidden_size, nb, cs
        );
    }

    // ---- Compute relative_key_states = position_embedding @ k_rel_weight^T ----
    std::vector<bf16> relative_key_states(num_positions * hidden_size, bf16(0));
    simd_gemm_abt_bf16(
        position_embedding,
        k_rel_weight,
        relative_key_states.data(),
        num_positions, hidden_size, hidden_size,
        hidden_size, padded_hidden_size, hidden_size
    );

    // ---- Compute blocked attention per audio ----
    for (int i = 0; i < num_audios; i++) {
        int nb = num_blocks_per_audio[i];
        int cs = context_size_per_audio[i];

        compute_audio_blocked_attention(
            blocked_q_per_audio[i].data(),
            blocked_k_per_audio[i].data(),
            blocked_v_per_audio[i].data(),
            relative_key_states.data(),
            block_attention_mask_per_audio[i].data(),
            output + start_seq_len_index_per_audio[i] * padded_hidden_size,
            seq_len_per_audio[i],
            nb, chunk_size, cs, num_positions,
            num_heads, head_dim, hidden_size, padded_hidden_size,
            softcap, invalid_logits_value
        );
    }
}
