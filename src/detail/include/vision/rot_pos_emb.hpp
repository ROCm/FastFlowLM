#pragma once
#include <vector>
#include <cstdint>
#include "typedef.hpp"
#include <cmath>
#include <algorithm>

/**
 * Generate the frequency table for rotary embeddings
 * Equivalent to Qwen3VLVisionRotaryEmbedding.forward(seqlen)
 * 
 * @param seqlen: Maximum sequence length (max_hw)
 * @param dim: Rotary dimension (head_dim / 2)
 * @param theta: Base value for inverse frequency (default 10000.0)
 * @return: Frequency table [seqlen, dim/2] in row-major format
 */
std::vector<float> generate_rotary_freq_table(
    int32_t seqlen,
    int32_t dim,
    float theta = 10000.0f
);


/**
 * Generate rotary position embeddings for vision tokens (float version)
 * 
 * This function creates 2D position IDs for each token based on spatial merge patterns,
 * generates the frequency table, and looks up the corresponding rotary embeddings.
 * 
 * @param grid_thw: Vector of [t, h, w] for each image/video
 * @param spatial_merge_size: Spatial merge size (typically 2)
 * @param rot_dim: Rotary dimension (head_dim / 2)
 * @param theta: Base value for inverse frequency (default 10000.0)
 * @return: Flattened vector of rotary embeddings [total_tokens, rot_dim] in float format
 */
std::vector<float> rot_pos_emb_float(
    const std::vector<std::vector<int32_t>>& grid_thw,
    int32_t spatial_merge_size,
    int32_t rot_dim,
    float theta = 10000.0f
);


/**
 * Compute cosine and sine embeddings from rotary position embeddings (float input/output)
 * 
 * This function duplicates the rotary embeddings (concatenates with itself)
 * and then computes element-wise cosine and sine. This matches the PyTorch code:
 *   emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
 *   position_embeddings = (emb.cos(), emb.sin())
 * 
 * @param rotary_emb: Input rotary embeddings [seq_len, rot_dim] where rot_dim = head_dim / 2 (float)
 * @param cos_emb: Output cosine embeddings [seq_len, head_dim] (will be resized)
 * @param sin_emb: Output sine embeddings [seq_len, head_dim] (will be resized)
 */
void cos_sine_pose_emb(
    const std::vector<float> &rotary_emb,
    std::vector<float> &cos_emb,
    std::vector<float> &sin_emb,
    int QWEN3_5_VISION_HEAD_DIM    
);


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
);

// void apply_rotary_pos_emb_vision_inplace_reordered(
//     bf16* mm_res_ptr,
//     const float* cos_emb_ptr,
//     const float* sin_emb_ptr,
//     int seq_len,
//     int seq_len_padded,
//     int hidden_size,
//     int hidden_size_padded,
//     int num_heads
// );
