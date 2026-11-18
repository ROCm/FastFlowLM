// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file verify/verify_module.hpp
/// \brief Module 3: push the k+1 verify rows through the 64-layer stack and
///        accept the longest prefix the base model agrees with.
///
/// NPU backend: two highly fused layer.xclbins, one for the full-attention
/// layers and one for the gated-delta-net layers, over the same caches
/// module 1 filled.
#pragma once

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "common/tap_buffer.hpp"

namespace qwen38 {

struct accept_result {
    /// argmax at each of the k+1 positions; position i predicts position i+1.
    std::vector<int> argmax;
    int              accepted = 0;
    /// The accepted drafts plus the free base-model token at the first
    /// mismatch: always `accepted + 1` entries.
    std::vector<int> committed;
};

inline accept_result accept_prefix(const std::vector<int>& vargmax,
                                   const std::vector<int>& drafts) {
    if (vargmax.size() != drafts.size() + 1)
        throw std::runtime_error("accept_prefix: expected k+1 argmax for k drafts");
    accept_result r;
    r.argmax = vargmax;
    while (r.accepted < (int)drafts.size() &&
           vargmax[r.accepted] == drafts[r.accepted])
        r.accepted++;
    r.committed.assign(vargmax.begin(), vargmax.begin() + r.accepted + 1);
    return r;
}

class verify_module {
public:
    virtual ~verify_module() = default;

    /// Take the snapshot a rejected draft rolls back to. Must precede the
    /// batch: the delta-net fold is destructive and this copy is its only
    /// inverse.
    virtual void snapshot() = 0;
    virtual void rollback() = 0;
    /// The batch committed in full; the snapshot is stale.
    virtual void discard() = 0;

    /// Whether `commit_prefix()` is available on this backend's stack. False
    /// means a partial accept has to roll back and re-run the accepted tokens
    /// through the model, which is the original and much slower recovery.
    virtual bool can_commit_prefix() const { return false; }

    /// Keep the first `accept_len` of the last `run()`'s rows and drop the
    /// rest, in place, without a second forward pass. Consumes the snapshot.
    ///
    /// The caller still owes the engine-level bookkeeping `_prefill_with_mm()`
    /// would have done for it: `committed_ids`, the context length, and the
    /// hidden-state window the draft head reads.
    virtual void commit_prefix(int accept_len) = 0;

    /// Embed, sweep, and return the per-row argmax over all L rows.
    virtual std::vector<int> run(const std::vector<int>& ids,
                                 const int32_t* pos_thw) = 0;

    /// [L][D] pre-norm hidden states the last run() produced.
    virtual const float* hidden() const = 0;

    /// Where run() copies the tapped layers' outputs (DFlash2's target taps), or
    /// null for none. Borrowed; sized per pass by run().
    virtual void set_taps(tap_buffer* /*taps*/) {}

    /// [L][vocab] logits the last run() produced: row i is the distribution
    /// the argmax at position i was taken from. nullptr when the backend keeps
    /// none on the host. Valid until the next run().
    virtual const float* logits(int& /*rows*/, int& /*vocab*/) const { return nullptr; }

    /// Assert the two cache-alignment invariants a cycle must leave behind:
    /// one base KV row per committed token, and a head cache strictly behind
    /// the base stack. Throws on a violation; a backend with no caches to check
    /// does nothing.
    ///
    /// This is what covers the attention half of `commit_prefix()`: that path
    /// moves a cursor rather than re-running the layer, and a cursor arithmetic
    /// error emits perfectly fluent text while acceptance decays to zero. The
    /// call site is under `DEBUG_BLOCK(1)`, so it costs nothing shipped.
    virtual void check_invariant(int /*committed_tokens*/,
                                 int /*mtp_cache_len*/) const {}
};

} // namespace qwen38
