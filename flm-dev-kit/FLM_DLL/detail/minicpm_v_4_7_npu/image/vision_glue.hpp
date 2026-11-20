// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file image/vision_glue.hpp
/// \brief The O(L*D) host math around the two device-bound operators.
///
/// Layer normalisation, the position lookup, residual adds, the two 2x2
/// regroups and the vit_merger's windowed attention. None of these are worth a
/// dispatch, and each is a place where an off-by-one in the token walk would be
/// invisible to a cosine, so they stay on the host.
///
/// **Token order.** Unlike Qwen3.8's tower, rows here are in RASTER order within
/// each view -- that is what the NaViT processor emits and what the HF taps are
/// in. So the 2x2 regroups are real gathers, not free reinterpretations.
#ifndef __MINICPM_V_VISION_GLUE_HPP__
#define __MINICPM_V_VISION_GLUE_HPP__
#pragma once

#include <cstddef>
#include <cstdint>

namespace mcv {
namespace vision {

/// LayerNorm with learned gain and bias over the last axis (nn.LayerNorm, not
/// RMSNorm). Mean and variance in double.
void layer_norm(const float* x, int L, int D,
                const float* gain, const float* bias, float eps, float* y);

/// \brief The learned position table, looked up the way
///        MiniCPMV4_7VisionEmbeddings does it -- NOT a bilinear resample.
///
/// The table is side x side (70 x 70). A view of h x w patches gets, per axis,
/// `torch.bucketize(arange(0, 1-1e-6, 1/n), arange(1/side, 1, 1/side),
/// right=True)` -- each patch takes the table cell whose bucket its fractional
/// coordinate falls in. Nearest-cell, so a 28 x 36 view reads 28 x 36 distinct
/// rows of the table with uneven strides.
///
/// The bucket of a coordinate that lands exactly ON a boundary (70*i/n an
/// integer) depends on float rounding, and torch's arange rounds differently
/// from `i/n`: the integer formula `floor(70*i/n)` disagrees with torch at n =
/// 25, 49, 50, 55, 100, ... This reproduces torch's float32 arithmetic instead
/// (see the .cpp), which matches it bit for bit on every n < 600.
///
/// \param out [h*w][D] fp32, raster order.
void pos_embed_lookup(const uint16_t* table, int side, int D, int h, int w, float* out);

/// x += y, elementwise.
void add_inplace(float* x, const float* y, size_t n);

/// \brief The 2x2 regroup both merges do: `view(h/2, 2, w/2, 2, D)
///        .permute(0, 2, 1, 3, 4).reshape(h/2 * w/2, 4*D)`.
///
/// \param x    [h*w][D] one view, raster
/// \param out  [(h/2)*(w/2)][4D]: merged row (r, c) is patches (2r, 2c),
///             (2r, 2c+1), (2r+1, 2c), (2r+1, 2c+1), in that order
/// \param mean if non-null, [(h/2)*(w/2)][D] receives the mean of the four --
///             the vit_merger's second residual
void regroup_2x2(const float* x, int h, int w, int D, float* out, float* mean);

/// \brief The vit_merger's attention core: softmax(q k^T * scale) v within each
///        2x2 window of one view, all heads.
///
/// HF regroups rows into window order, runs attention with cu_seqlens of 4, and
/// argsorts back. The projections are per-token, so this indexes the four
/// raster rows of each window directly and writes o in raster order. 4x4 scores
/// per head per window -- negligible work. It stays off the array because
/// vision_attn works in 512-row rounds: each 4-token window would be its own
/// segment padded to 512, ~1800 of them at 720p.
///
/// \param q,k,v [h*w][ld], head-major in the first heads*dh columns (ld is
///              3*heads*dh when they point into the fused qkv output)
/// \param o     [h*w][heads*dh]
void window_attention(const float* q, const float* k, const float* v, int ld,
                      int h, int w, int heads, int dh, float scale, float* o);

}  // namespace vision
}  // namespace mcv

#endif  // __MINICPM_V_VISION_GLUE_HPP__
