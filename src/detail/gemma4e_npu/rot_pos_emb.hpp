#pragma once
#include <vector>
#include <cstdint>
#include "typedef.hpp"
#include <cmath>
#include <algorithm>

void generate_gemma4_audio_rotary_pos_emb(
    int hidden_size,
    int attention_chunk_size,
    int attention_context_left,
    int attention_context_right,

    std::vector<bf16>& position_embedding  // shape of [13, hidden_size]
);

/**
 * Generate Gemma4 vision rotary position embeddings (cos and sin).
 *
 * Python equivalent: Gemma4VisionRotaryEmbedding.forward(hidden_states, pixel_position_ids)
 *
 * For each patch s with (x_val, y_val):
 *   spatial_dim = head_dim / 2
 *   inv_freq[j] = 1 / (theta ^ (j*2 / spatial_dim))  for j in [0, spatial_dim/2)
 *   emb_x[j] = x_val * inv_freq[j]  (duplicated: positions [0..spatial_dim-1])
 *   emb_y[j] = y_val * inv_freq[j]  (duplicated: positions [spatial_dim..head_dim-1])
 *   cos_row = [cos(emb_x)*scale, cos(emb_y)*scale]  of length head_dim
 *
 * Output shape: [seq_len_padded, head_dim]  (padded rows are zero)
 *
 * @param grid_pairs_per_image  per-image flat (x,y) pairs  [img][s*2]
 * @param seq_len_per_image     unpadded patch count per image
 * @param start_seq_len_index_per_image  cumulative start offset per image
 * @param seq_len_padded        padded total sequence length
 * @param head_dim              GEMMA4E_VISION_HEAD_DIM (e.g. 64)
 * @param theta                 GEMMA4E_ROPE_THETA (e.g. 100.0)
 * @param scale                 attention_scaling (typically 1.0)
 * @param cos_emb               output [seq_len_padded * head_dim] (resized, bf16)
 * @param sin_emb               output [seq_len_padded * head_dim] (resized, bf16)
 */
void generate_gemma4_vision_rotary_pos_emb(
    const std::vector<std::vector<int>>& grid_pairs_per_image,
    const std::vector<int>& seq_len_per_image,
    const std::vector<int>& start_seq_len_index_per_image,
    int seq_len_padded,
    int head_dim,
    float theta,
    float scale,
    std::vector<bf16>& cos_emb,
    std::vector<bf16>& sin_emb
);

/**
 * Apply multidimensional rotary position embeddings (RoPE) in-place.
 *
 * C++ equivalent of Gemma4's apply_multidimensional_rope with ndim=2, followed
 * by apply_rotary_pos_emb (rotate_half variant).
 *
 * The head_dim is split into two spatial halves:
 *   x-spatial: channels [0 .. spatial_dim-1]         (spatial_dim = head_dim/2)
 *   y-spatial: channels [spatial_dim .. head_dim-1]
 * Standard RoPE is applied independently to each half:
 *   out_lo = x_lo * cos - x_hi * sin
 *   out_hi = x_hi * cos + x_lo * sin
 * where _lo/_hi refer to the lower and upper quarter of each spatial half.
 *
 * @param qkv      Pointer to buffer of shape [seq_len, num_head, head_dim] (bf16, in-place)
 * @param cos_emb  Pointer to cosine embeddings of shape [seq_len_padded, head_dim] (bf16)
 * @param sin_emb  Pointer to sine embeddings of shape [seq_len_padded, head_dim] (bf16)
 * @param seq_len  Number of valid (non-padded) sequence positions to process
 * @param num_head Number of attention heads
 * @param head_dim Head dimension; must be divisible by 4 (64 for Gemma4e vision)
 */
void apply_multidimensional_rope(
    bf16* qkv,
    const bf16* cos_emb,
    const bf16* sin_emb,
    int seq_len,
    int num_head,
    int head_dim,
    int ndim = 2
);
