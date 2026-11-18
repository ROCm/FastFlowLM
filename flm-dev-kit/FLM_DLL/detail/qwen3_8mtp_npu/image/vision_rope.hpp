// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file image/vision_rope.hpp
/// \brief 2-D rotary position embedding for the vision tower.
///
/// Not the language model's RoPE and not built from the same constants: the
/// vision tower rotates over a (row, column) pair rather than a linear
/// position, its base is a hardcoded 10000 rather than the model's rope_theta,
/// and it rotates the FULL head_dim rather than a partial_rotary_factor slice.
///
/// Layout, for head_dim = 72:
///   rot_dim  = head_dim / 2 = 36   frequencies per token
///   rot_half = rot_dim  / 2 = 18   of them for the row, 18 for the column
///   the [36] vector is then duplicated to [72] so rotate_half() lines the two
///   halves up -- which is why head_dim must be a multiple of 4.
#ifndef __QWEN3_8MTP_VISION_ROPE_HPP__
#define __QWEN3_8MTP_VISION_ROPE_HPP__
#pragma once

#include <cstdint>
#include <vector>

namespace qwen38 {
namespace vision {

/// \brief cos/sin tables for one image's patches, [tokens][head_dim] each.
struct rope_tables {
    std::vector<float> cos, sin;
    int tokens   = 0;
    int head_dim = 0;
};

/// \brief Build the tables for a t x h x w grid.
///
/// The (row, column) pair for each token follows the merge-block walk:
///   reshape(h/m, m, w/m, m).permute(0, 2, 1, 3)
/// so token s belongs to block (s / (m*m)) in raster order and holds sub-cell
/// (s % (m*m)) within it. pixel_values arrives in that same order, so this
/// table and the patch rows are already aligned -- do not reorder either.
///
/// \param grid_t frames; the (h, w) coordinates repeat for each one.
rope_tables build_rope(int grid_t, int grid_h, int grid_w,
                       int spatial_merge_size, int head_dim, float theta);

/// \brief q' = q*cos + rotate_half(q)*sin, in place, for every head of a row.
///
/// \param x       [L][heads*head_dim] fp32, head-major within a row
/// \param tables  built by build_rope() for the same L
///
/// Applied to Q and to K, never to V. The fused qkv projection emits one
/// [L][3D] row, so the caller passes the Q third and the K third separately
/// rather than letting this function guess where the thirds are.
void apply_rope(float* x, int L, int heads, const rope_tables& tables);

}  // namespace vision
}  // namespace qwen38

#endif  // __QWEN3_8MTP_VISION_ROPE_HPP__
