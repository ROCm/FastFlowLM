// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file image/mrope.hpp
/// \brief m-rope position ids for a window that contains images.
///
/// A multimodal window's positions are NOT 0..L-1. An image occupies one prompt
/// row per merged patch but advances position by the merged grid's EXTENT, so
/// the 16x16 grid in the captured reference fills 64 rows while moving position
/// by 8. Getting this wrong is not a crash: every downstream stage still runs,
/// the text simply drifts.
///
/// The rule, for a visual block starting at position st whose merged grid is
/// mt frames of mh x mw (so frame = mh*mw blocks per frame):
///
///   block s          ->   (st + s / frame,              <- frame index
///                          st + (s % frame) / mw,       <- row in that frame
///                          st + (s % frame) % mw)       <- column
///   the next text token -> st + max(mt, mh, mw)
///
/// which is HF get_rope_index's `st + stack(t,h,w).max() + 1` written out. An
/// image has mt == 1, so the frame term vanishes and this reduces to the form
/// read off ref.safetensors (prefill.position_ids against prefill.input_ids and
/// prefill.image_grid_thw): (st, st + s/mw, st + s%mw), advancing by
/// max(mh, mw). That image case is the only one covered by the dump; the frame
/// term is the general law, not a measured result.
///
/// The max matters: 5VL's walker advances by the merged WIDTH, which agrees
/// with this only for landscape images. A portrait image would silently
/// overlap the text that follows it.
///
/// VIDEO IS NOT SUPPORTED, and build_mrope throws on a video token rather than
/// letting it through as text. The position law above is general, but the
/// vision tower is not: its attention mask is one full block over the whole
/// sequence, which is only correct WITHIN a frame -- across frames the mask
/// must be block-diagonal. At mt == 1 the two are the same mask, which is why
/// nothing has ever caught it. Pass the checkpoint's video_token_id so the
/// failure is a throw at the seam instead of drift a thousand tokens later.
///
/// Output is [3][L] CHANNEL-MAJOR, matching qwen3_8mtp_npu's own convention --
/// qwen3_5vl_npu emits [L][3] row-major instead, and the two are the same bytes
/// in a different order, so a copied loop transposes silently.
#ifndef __QWEN3_8MTP_MROPE_HPP__
#define __QWEN3_8MTP_MROPE_HPP__
#pragma once

#include "image/image_module.hpp"

#include <cstdint>
#include <vector>

namespace qwen38 {

/// Where an image's rows sit inside one window, and which image they are.
struct image_span {
    int begin       = 0;   ///< first row in the window
    int count       = 0;   ///< rows, == tokens of that image in THIS window
    int image_index = 0;   ///< index into the payload's image array
    int token_offset = 0;  ///< how many of that image's tokens precede `begin`
};

/// Carried across prefill() calls so a prompt that arrives in chunks -- which
/// is what _chunked_insert does -- can split an image across windows without
/// restarting its position walk.
struct mrope_cursor {
    int32_t pos             = 0;   ///< next position for a text token
    int     image           = 0;   ///< next (or in-progress) image
    int     consumed        = 0;   ///< tokens of that image already emitted
    int32_t image_start_pos = 0;   ///< st of the in-progress image
};

/// \brief Walk one window, emitting [3][L] positions and the image spans.
///
/// \param grids   one entry per image in the payload, in prompt order
/// \param cursor  advanced in place; seed it with the engine's running position
/// \param spans   appended to, may be null
/// \param video_token_id  the checkpoint's video token, or -1 to disable the
///         check. Never silently treated as text -- see the file comment.
/// \throws if the window holds more image rows than the remaining images
///         account for -- that means the prompt and the payload disagree, and
///         continuing would splice the wrong rows.
/// \throws if the window holds a video token, or an image whose grid has more
///         than one frame: the position law handles both, the tower does not.
std::vector<int32_t> build_mrope(const int* tokens, int L,
                                 int32_t image_token_id,
                                 const image_grid* grids, int num_images,
                                 int spatial_merge_size,
                                 mrope_cursor& cursor,
                                 std::vector<image_span>* spans,
                                 int32_t video_token_id = -1);

}  // namespace qwen38

#endif  // __QWEN3_8MTP_MROPE_HPP__
