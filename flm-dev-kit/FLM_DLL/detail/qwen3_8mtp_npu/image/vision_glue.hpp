// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file image/vision_glue.hpp
/// \brief The O(L*D) host math around the two device-bound operators.
///
/// Layer normalisation, the bilinear resample of the learned position grid, and
/// the merger's regroup. None of these are worth a dispatch: together they are
/// a few percent of the tower, and each one is a place where an off-by-one in
/// the token walk would be invisible to a cosine. They stay on the host in
/// Phase 2 as well, which is why they live outside vision_ops.hpp.
#ifndef __QWEN3_8MTP_VISION_GLUE_HPP__
#define __QWEN3_8MTP_VISION_GLUE_HPP__
#pragma once

#include <cstdint>
#include <vector>

namespace qwen38 {
namespace vision {

/// \brief LayerNorm with learned gain AND bias, over the last axis.
///
/// The vision tower uses nn.LayerNorm, not the RMSNorm the language model uses:
/// the mean is subtracted and there is a bias. Substituting RMSNorm here is
/// cosine-invisible on well-centred activations (see rmsnorm-is-cosine-
/// invisible), so it is worth being explicit.
///
/// Accumulates the mean and variance in double: 1152 elements is small enough
/// that the cost is nil and the fp32 catastrophic-cancellation risk is real.
void layer_norm(const float* x, int L, int D,
                const float* gain, const float* bias, float eps, float* y);

/// \brief Resample the learned square position grid onto this image's grid.
///
/// The table is num_grid_per_side x num_grid_per_side (48 x 48 here) and the
/// image is h x w patches, so each output position is a bilinear blend of four
/// table entries at linspace(0, N-1, h) x linspace(0, N-1, w).
///
/// \param out receives [t*h*w][D] fp32 IN MERGE-BLOCK ORDER -- the same walk
///        pixel_values uses, so it can be added row-for-row to the patch
///        embedding with no reordering.
void pos_embed_interpolate(const uint16_t* table, int num_grid_per_side, int D,
                           int grid_t, int grid_h, int grid_w,
                           int spatial_merge_size, float* out);

// The merger's regroup -- [L][D] -> [L/4][4D] -- has no function here on
// purpose. Because the whole pipeline walks in merge-block order, patches
// 4s..4s+3 ARE the 2x2 cell that becomes language token s, laid out
// consecutively, so the regroup is a reinterpretation of the same bytes and
// costs nothing. It is only free while that walk holds: feed the tower
// raster-order rows and each merged token silently mixes four cells from
// different parts of the image, which still normalises, still attends, and
// still produces fluent text.

/// \brief x += y, elementwise.
void add_inplace(float* x, const float* y, size_t n);

}  // namespace vision
}  // namespace qwen38

#endif  // __QWEN3_8MTP_VISION_GLUE_HPP__
