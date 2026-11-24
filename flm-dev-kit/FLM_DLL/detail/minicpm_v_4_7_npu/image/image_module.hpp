// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file image/image_module.hpp
/// \brief The vision tower's surface: views of patches in, language-model rows
///        out.
///
/// Same shape as Qwen3.8's module 4, with one change forced by MiniCPM: the unit
/// is the VIEW, not the image. The processor slices an image into a source view
/// plus up to nine slices, each with its own patch grid, and attention is full
/// within a view and blocked across views. `encode` takes any number of views
/// (all of one image, or of several) because every matmul batches across them
/// for free; only attention needs the boundaries, and it gets them from `views`.
#pragma once

#include <cstdint>
#include <functional>
#include <vector>

namespace mcv {

/// One view's patch grid -- `target_sizes` row. h and w count PATCHES.
struct view_grid {
    int h = 0;
    int w = 0;
};

class image_module {
public:
    virtual ~image_module() = default;

    /// Language-model rows one view produces: h*w/16 (h*w/4 in "4x" mode). The
    /// prompt must carry exactly this many image_token_id rows for the view.
    virtual int tokens_for(view_grid v) const = 0;

    /// \param pixel_values [sum h*w][3*14*14] fp32: view after view, RASTER order
    ///        within a view, each row a patch flattened (c, kh, kw) -- the NaViT
    ///        `[3, 14, T*14]` tensor the HF processor emits, transposed per patch.
    /// \param out [sum tokens_for][out_hidden_size] fp32, in the same view order,
    ///        ready to be spliced over the prompt's image_token_id rows.
    virtual void encode(const float* pixel_values, const std::vector<view_grid>& views,
                        float* out) = 0;

    /// Intermediate tap: a stage name and an [rows][cols] fp32 view whose
    /// lifetime ends when the call returns. Null by default; free when unset.
    using tap_fn = std::function<void(const char* name, const float* data, int rows, int cols)>;
    virtual void set_tap(tap_fn fn) = 0;

    /// e.g. "mm=npu attn=npu bf16".
    virtual const char* backend() const = 0;
};

}  // namespace mcv
