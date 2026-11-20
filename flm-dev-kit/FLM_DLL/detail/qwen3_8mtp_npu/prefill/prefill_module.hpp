// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file prefill/prefill_module.hpp
/// \brief Module 1: batched prefill of a long prompt over the 64-layer stack.

#pragma once
#include <cstdint>

#include "common/tap_buffer.hpp"

namespace qwen38 {

class prefill_module {
public:
    virtual ~prefill_module() = default;

    /// Size the scratch and build the m-rope tables. `pos_thw` is [3][L]; for
    /// pure text all three channels are equal and this degenerates to 1-D RoPE.
    virtual void begin_window(int L, const int32_t* pos_thw) = 0;

    /// [L][D] scratch: the caller writes the embeddings here, run() leaves the
    /// layer-63 output here.
    virtual float* hidden() = 0;

    /// Sweep all 64 layers in place, appending L rows to the caches.
    virtual void run(int L) = 0;

    /// Where run() copies the tapped layers' outputs (DFlash2's target taps), or
    /// null for none. Borrowed; the caller sizes it per window with begin().
    virtual void set_taps(tap_buffer* /*taps*/) {}
};

} // namespace qwen38
