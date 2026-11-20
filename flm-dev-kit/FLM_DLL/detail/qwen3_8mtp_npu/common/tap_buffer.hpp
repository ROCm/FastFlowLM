// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file common/tap_buffer.hpp
/// \brief bf16 copies of the hidden state after chosen layers: the DFlash2
///        drafter's view of the target.
///
/// The drafter conditions on the target's hidden states at five layers
/// (`dflash_config.target_layer_ids`, 5 19 33 47 61 for Qwen3.8-27B). Those are
/// the OUTPUTS of the named layers -- HF's `hidden_states[id + 1]` -- so they
/// are read from the residual stream right after the sweep's layer `id`, not
/// before it and not after the final norm.
///
/// Both executors over the layer stack (prefill and verify) own a pointer to
/// one of these and call `capture()` after every layer; the call is a no-op
/// for the 59 layers that are not tapped, and the pointer is null when there
/// is no drafter, so a plain MTP run pays nothing.
#pragma once

#include "common/cpu_simd.hpp"
#include "typedef.hpp"

#include <algorithm>
#include <cstddef>
#include <vector>

namespace qwen38 {

struct tap_buffer {
    /// Ascending layer indices to tap. Empty = disabled.
    std::vector<int> layers;
    int D = 0;
    /// Only the last `keep_rows` rows of a window are kept (0 = all). The drafter's
    /// attention reaches back `sliding_window` rows at most, so older prompt rows are
    /// never read and a long prefill window need not be copied whole.
    int keep_rows = 0;

    /// The rows the last window left: `rows` of them, the first being window row `first_row`.
    int rows      = 0;
    int first_row = 0;
    /// [rows][layers.size()][D]
    std::vector<bf16> data;

    bool enabled() const { return !layers.empty(); }
    int  taps() const { return (int)layers.size(); }

    /// Size for a window of `L` rows. Call before the sweep.
    void begin(int L) {
        first_row = (keep_rows > 0 && L > keep_rows) ? L - keep_rows : 0;
        rows      = L - first_row;
        data.resize((size_t)rows * layers.size() * (size_t)D);
    }

    int tap_index(int layer) const {
        for (size_t i = 0; i < layers.size(); i++)
            if (layers[i] == layer) return (int)i;
        return -1;
    }

    /// x is the layer's output, [L][D] fp32.
    void capture(int layer, const float* x, int L) {
        const int ti = tap_index(layer);
        if (ti < 0) return;
        const size_t T = layers.size();
        for (int r = 0; r < rows; r++)
            q38simd::f32_to_bf16(data.data() + ((size_t)r * T + (size_t)ti) * (size_t)D,
                                 x + (size_t)(first_row + r) * (size_t)D, (size_t)D);
        (void)L;
    }

    /// One row's `taps() * D` values: [taps][D].
    const bf16* row(int r) const { return data.data() + (size_t)r * layers.size() * (size_t)D; }
};

} // namespace qwen38
