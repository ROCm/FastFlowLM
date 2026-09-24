#pragma once
#include <vector>
#include <cstdint>
#include "typedef.hpp"
#include <cmath>
#include <algorithm>

/**
 * Fast positional embedding interpolation for Qwen3 Vision Model
 * 
 * @param grid_thw: Vector of [t, h, w] for each image/video
 * @param pos_embed_weight: Pointer to position embedding weight [num_position_embeddings, hidden_size]
 * @param num_grid_per_side: Square root of num_position_embeddings (e.g., sqrt(14*14) = 14)
 * @param hidden_size: Hidden dimension size
 * @param spatial_merge_size: Spatial merge size (typically 2)
 * @return: Vector of interpolated position embeddings
 */
std::vector<bf16> fast_pos_embed_interpolate(
    const std::vector<std::vector<int32_t>>& grid_thw,
    const bf16* pos_embed_weight,
    int num_grid_per_side,
    int hidden_size,
    int spatial_merge_size
);
