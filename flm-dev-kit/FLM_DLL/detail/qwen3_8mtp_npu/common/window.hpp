// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file common/window.hpp
/// \brief Per-window scratch, shared by the prefill and verify executors.
#pragma once

#include <vector>

namespace qwen38 {

/// Sized on first use, grown only when a longer window arrives. Everything is
/// fp32; rounding between layers would add error the NPU port has not
/// committed to yet.
struct window_buffers {
    std::vector<float> x;     ///< [L][D] hidden state, carried across layers
    std::vector<float> y;     ///< [L][D] layer output
    std::vector<float> cos;   ///< [L][ROPE_DIM]
    std::vector<float> sin;   ///< [L][ROPE_DIM]

    void ensure(int L, int D, int rope_dim) {
        const size_t need = (size_t)L * D;
        if (x.size() < need) { x.assign(need, 0.0f); y.assign(need, 0.0f); }
        const size_t rneed = (size_t)L * rope_dim;
        if (cos.size() < rneed) { cos.assign(rneed, 0.0f); sin.assign(rneed, 0.0f); }
    }
};

} // namespace qwen38
