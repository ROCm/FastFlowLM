// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file image/image_module.hpp
/// \brief Module 4: the vision tower. One image of patches in, language-model
///        rows out.
///
/// Same shape as the engine's other three module headers
/// (prefill/prefill_module.hpp, speculate/speculate_module.hpp,
/// verify/verify_module.hpp): an abstract surface the engine holds by
/// unique_ptr, with the backend chosen in exactly one place.
///
/// The module boundary is per-IMAGE, not per-window. Attention inside the tower
/// is bidirectional within an image and blocked across images, so a per-window
/// interface would have invited exactly the mask bug that a single-image test
/// cannot see.
#pragma once

#include <cstdint>
#include <functional>

namespace qwen38 {

/// Patch grid of one image, as the preprocessor measured it. h and w count
/// PATCHES (pixels / patch_size), not pixels, and both must be multiples of
/// spatial_merge_size.
struct image_grid {
    int t = 1;
    int h = 0;
    int w = 0;
};

class image_module {
public:
    virtual ~image_module() = default;

    /// Language tokens this grid produces: t*h*w / spatial_merge_size^2. The
    /// prompt must carry exactly this many image_token_id rows for the image.
    virtual int tokens_for(image_grid g) const = 0;

    /// \param pixel_values [t*h*w][patch_dim] fp32, normalised patches in
    ///        merge-block order -- byte-for-byte what the HF image processor
    ///        emits as `pixel_values`.
    /// \param out [tokens_for(g)][out_hidden_size] fp32, ready to be spliced
    ///        into the prefill window's hidden states.
    virtual void encode(const float* pixel_values, image_grid g, float* out) = 0;

    /// Intermediate tap, for the milestone driver. Called with a stage name and
    /// an [rows][cols] fp32 view whose lifetime ends when the call returns.
    /// Null by default, and the tower must not pay for it when unset.
    using tap_fn = std::function<void(const char* name, const float* data,
                                      int rows, int cols)>;
    virtual void set_tap(tap_fn fn) = 0;

    /// e.g. "cpu", or "npu:mm+attn" once the device backends land.
    virtual const char* backend() const = 0;
};

}  // namespace qwen38
