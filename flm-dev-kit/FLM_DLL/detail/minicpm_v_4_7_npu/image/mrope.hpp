// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file image/mrope.hpp
/// \brief MiniCPM-V-4.7's "canvas" m-rope: the (t, h, w) position of every
///        prompt token when the prompt carries images.
///
/// A port of `_compute_canvas_packed()` in the checkpoint's own
/// `mrope_minicpmv4_7.py`, single-image-group path. It is NOT Qwen3.5-VL's rule
/// (`st + w/2` per image, text positions untouched by structure), and the
/// difference is not cosmetic: the language model was trained on these
/// positions.
///
/// One image is a GROUP: a thumbnail segment `<image> pad.. </image>` followed
/// by slice segments `<slice> pad.. </slice>`, slice rows separated by `\n`.
/// The group is laid out on a canvas of `rows*sh x cols*sw` LLM cells, where
/// (sh, sw) is one slice's LLM grid. Inside the group every token's t is the
/// group's base position; h and w are base + the cell it covers:
///
///   * slice k's tokens sit at its own cell block (row k/cols, col k%cols);
///   * the thumbnail is stretched over the WHOLE canvas: its rows and columns
///     are `round(linspace(0, canvas-1, n))`, not 0..n-1;
///   * the structural tokens get halo positions -- `<image>` at (base-1,
///     base-1), `</image>` at (base+H, base+W), each `<slice>`/`</slice>` at its
///     block's first/last cell, a `\n` between slice rows one column past the
///     right edge on the row that just ended.
///
/// Text before and after advances one position per token, and the position
/// after a group is base + max(H, W) + 1.
///
/// Consecutive groups separated by nothing but structural tokens and `\n` are
/// merged into a VIDEO group in the HF code, with a different layout. The
/// prompt this engine is given always carries `<image_id>N</image_id>` before
/// each image (the processor's `use_image_id=True` default), so that never
/// triggers for images, and video is rejected upstream. It is detected and
/// refused here rather than silently laid out the wrong way.
#pragma once

#include "image/image_module.hpp"

#include <cstdint>
#include <vector>

namespace mcv {

struct canvas_token_ids {
    int image_pad   = 248056;   ///< <|image_pad|>
    int im_start    = 248078;   ///< <image>
    int im_end      = 248079;   ///< </image>
    int slice_start = 248088;   ///< <slice>
    int slice_end   = 248089;   ///< </slice>
    int newline     = 198;      ///< "\n"
};

/// `compute_llm_grid()`: the LLM grid of a segment of `n_tokens` rows produced
/// from a `vh x vw` patch grid. (0, 0) when it cannot be factored, in which
/// case HF falls back to 1-D positions for that segment.
void llm_grid(int n_tokens, int vh, int vw, int& h, int& w);

/// \brief Positions for `L` tokens, the first at text position `pos0`.
///
/// \param tokens  the prompt (or the not-yet-cached tail of it). Every image
///                group must be whole inside it.
/// \param views   one patch grid per visual segment, in prompt order -- the
///                processor's `target_sizes`. Must match the number of segments.
/// \param out     [L][3] (t, h, w)
/// \return        the position the next token takes.
int canvas_mrope(const int* tokens, int L, const canvas_token_ids& ids,
                 const std::vector<view_grid>& views, int pos0, int* out);

}  // namespace mcv
