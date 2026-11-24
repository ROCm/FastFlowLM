// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file speculate/speculate_module.hpp
/// \brief Module 2: the MTP draft head proposes k tokens.
///
/// NPU backend: one small fused layer.xclbin -- the head is a single gated
/// full-attention decoder layer plus a fusion projection and a norm. The draft
/// loop stays on the host; the xclbin replaces the layer, not the loop.
#pragma once

#include <cstdint>
#include <vector>

namespace qwen38 {

class speculate_module {
public:
    virtual ~speculate_module() = default;

    virtual int  max_steps() const = 0;
    virtual int  cache_len() const = 0;
    virtual void reset() = 0;
    /// Zero the rows an earlier context left in the head's cache above its cursor (reset() moves only the cursor).
    virtual void zero_stale() {}
    virtual void crop(int len) = 0;
    virtual void set_max_length(int n) = 0;

    /// k drafts from a T-token catch-up window and the normed states that
    /// predicted it. `pos0` is the absolute position of ids[0]; the head's
    /// cache trails the base stack by exactly one, so it is not the engine's
    /// context length.
    virtual std::vector<int> draft(const std::vector<int>& ids,
                                   const float* prev_hidden,
                                   int T, int pos0, int k) = 0;

    /// \brief wall time of the most recent draft()'s step 0, in microseconds
    /// \note Step 0 is the only step whose width is not 1: it re-absorbs the
    ///       rows the head has not seen yet. On the first cycle after a prompt
    ///       prefill that window is the whole last prefill chunk, so the pass
    ///       is prefill-shaped work wearing a decode-loop costume -- which is
    ///       the phase boundary the engine needs and cannot see from outside
    ///       draft(). Every later cycle's step 0 is one or two rows and is
    ///       ordinary decode; it is the caller, not this number, that decides
    ///       which. Set by draft(), so it is only meaningful right after one.
    uint64_t last_step0_us() const { return last_step0_us_; }

protected:
    uint64_t last_step0_us_ = 0;
};

} // namespace qwen38
