#pragma once

#include <vector>
#include "typedef.hpp"
#include "buffer.hpp"

void create_sliding_window_attention_mask(
    int seq_len,
    int sliding_window_left,
    int sliding_window_right,
    std::vector<int> &mask

);

void convert_mask_to_blocked(
    std::vector<int> &input_mask, // [seq_len, seq_len]
    int chunk_size,
    int attention_context_left,
    int attention_context_right,
    std::vector<int> &blocked_mask, // [num_chunks, chunk_size, context_size],
    int &num_chunks,
    int &context_size
);

// Splits [seq_len, row_stride] into [num_blocks * chunk_size, row_stride] with zero-padding
// Equivalent to Python: F.pad(hidden_states, (0,0,0,0,0,pad)).reshape(batch, num_blocks, chunk_size, num_heads, head_dim)
void convert_to_block(
    const bf16* input,      // [seq_len, row_stride]
    bf16* output,           // [num_blocks * chunk_size, row_stride], caller must pre-allocate & zero-init
    int seq_len,
    int chunk_size,
    int row_stride,         // Padded hidden size (num_heads_padded * head_dim)
    int num_blocks          // = ceil(seq_len / chunk_size)
);

// Extracts overlapping context windows for blocked attention
// Equivalent to Python: F.pad(...).unfold(1, context_size, chunk_size).movedim(-1, 2)
// Output shape per audio: [num_blocks, context_size, num_heads_padded, head_dim]
void extract_block_context(
    const bf16* input,      // [seq_len, row_stride]
    bf16* output,           // [num_blocks * context_size, row_stride], caller must pre-allocate & zero-init
    int seq_len,
    int chunk_size,
    int max_past_horizon,   // attention_context_left - 1
    int max_future_horizon, // attention_context_right
    int row_stride,         // Padded hidden size
    int num_blocks,
    int context_size        // = chunk_size + max_past_horizon + max_future_horizon
);

// Full blocked audio attention computation for a single audio sample.
// Implements the Python forward() from after Q/K/V scaling through attn_output (before post projection).
//
// q_blocked:            [num_blocks * chunk_size, padded_hidden_size]  (from convert_to_block)
// k_blocked:            [num_blocks * context_size, padded_hidden_size] (from extract_block_context)
// v_blocked:            [num_blocks * context_size, padded_hidden_size] (from extract_block_context)
// relative_key_states:  [num_positions, hidden_size] dense row-major (= position_embedding @ k_rel_weight^T)
// block_attention_mask: [num_blocks * chunk_size * context_size] (1=attend, 0=mask)
// output:               [seq_len, padded_hidden_size] row-major (caller pre-allocates)
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
);

// Top-level audio self-attention: blocking + relative key computation + blocked attention + output assembly.
// Replaces the inline code in encode() that does convert_to_block, extract_block_context,
// simd_gemm_abt_bf16 for relative_key_states, and compute_audio_blocked_attention per audio.
//
// q_proj_output / k_proj_output / v_proj_output: [seq_len_padded, padded_hidden_size] packed for all audios
// k_rel_weight:       [padded_hidden_size, padded_hidden_size] row-major
// position_embedding: [num_positions, hidden_size] dense
// output:             [seq_len_padded, padded_hidden_size] pre-zeroed by caller
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
);
