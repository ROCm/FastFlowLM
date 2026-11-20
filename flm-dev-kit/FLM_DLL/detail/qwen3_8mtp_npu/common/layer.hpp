// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file common/layer.hpp
/// \brief One interface over the two kinds of decoder layer.
///
/// Both implementations live in verify/npu/ -- `q38npu::npu_attn_layer` and
/// `q38npu::npu_delta_layer`. The host adapters that used to sit here are gone:
/// module 1 has refused a stack containing one since it lost its host arms, so
/// they were unreachable. `cpu_ref::attention_layer` and
/// `cpu_ref::delta_net_layer` are still very much alive -- they are the stage
/// helpers the hybrid path calls and the reference the tools score against --
/// they are just no longer a *layer*.
#pragma once

#include "common/cpu_ref_attention_layer.hpp"
#include "common/cpu_ref_delta_net_layer.hpp"

#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace qwen38 {

/// Which of a layer's shadow copies a save/restore names. Two, because a
/// `turn` shadow must survive every speculative cycle taken before its restore.
enum class state_slot : int {
    speculative = 0,
    turn        = 1,
    count       = 2
};

inline constexpr size_t NUM_STATE_SLOTS = static_cast<size_t>(state_slot::count);

inline size_t slot_index(state_slot s) { return static_cast<size_t>(s); }

class decoder_layer {
public:
    virtual ~decoder_layer() = default;

    /// x, y: [L][MODEL_DIM]. cos/sin: [L][ROPE_DIM], ignored by delta-net.
    virtual void forward(const float* x, int L,
                         const float* cos, const float* sin, float* y) = 0;
    virtual void reset() = 0;
    virtual int  tokens_seen() const = 0;
    virtual bool is_full_attention() const = 0;

    /// Grow this layer's own cache in place, keeping what it holds. A no-op on
    /// delta-net: its state is the same size at 128 tokens and at 32768.
    virtual void set_max_length(int max_len) = 0;

    /// Copy into / out of the layer's OWN pre-allocated shadow. Attention
    /// stores a cursor (4 B); delta-net stores ~3.3 MB, because its fold has
    /// no inverse.
    virtual void save_state(state_slot slot) = 0;
    virtual void restore_state(state_slot slot) = 0;
    virtual size_t state_bytes() const = 0;

    /// A copy of this layer's state that lives outside the layer (the multi-slot prefix cache keeps many):
    /// attention exports its cursor (an int), delta-net its ~3.3 MB recurrent state and conv window.
    virtual size_t blob_bytes() const { return 0; }
    virtual void export_blob(void* dst) { (void)dst; }
    virtual void import_blob(const void* src) { (void)src; }

    /// Attention only: K/V rows [from, to) out of / into the cache, dense [n][kv_row_width()] bf16 each,
    /// and zero_rows(from, to) puts the zeros of a fresh cache back in rows [from, to)
    /// (the kernels stream rows above the cursor and a stale one moves the result slightly).
    virtual int  kv_row_width() const { return 0; }
    virtual void read_rows(int from, int to, void* k, void* v) { (void)from; (void)to; (void)k; (void)v; }
    virtual void write_rows(int from, int to, const void* k, const void* v) { (void)from; (void)to; (void)k; (void)v; }
    virtual void zero_rows(int from, int to) { (void)from; (void)to; }

    /// Land on the state that `accept_len` of the last `forward()` window's
    /// tokens would have produced -- **without re-running the layer**.
    ///
    /// This is the fast half of a rejected speculative batch. The pass had to
    /// run all `k+1` rows before anyone could know how many were good, so by
    /// the time `accept_len` is known the layer has already folded the rejected
    /// ones in. The two kinds of carried state are wrong in different ways:
    ///
    ///  - An attention cache **is a list**. Rows above the cursor were
    ///    appended but are never read, so the whole correction is the cursor.
    ///  - A delta-net block's conv window and recurrent state have no length
    ///    and no inverse: `S_t = a_t S_{t-1} + k_t (x) u_t` sums every token it
    ///    ever saw into one tensor. The correction is to reload `slot` and
    ///    re-derive the accepted prefix from what the pass left behind -- the
    ///    device's per-accept-length conv windows and the triplet arena.
    ///
    /// Before this existed the only way to advance those two by n tokens was to
    /// run the model on n tokens, which is what `prefill_verify()` did: a full
    /// 64-layer weight stream paid to recover a tensor, and 46% of the cycle.
    ///
    /// Consumes `slot` exactly as `restore_state()` does, and owes the same one
    /// `advance()`. `accept_len` is in `[0, L_TOKENS]`; 0 is `restore_state()`.
    virtual void commit_prefix(state_slot slot, int accept_len) = 0;

    /// Whether `commit_prefix()` is implemented here. The stack is
    /// all-or-nothing -- a delta-net block left on the host has no arena to
    /// re-derive from, and one of those anywhere sends *every* layer down the
    /// re-prefill path, because the stack has to agree on how far it advanced.
    virtual bool can_commit_prefix() const { return false; }
};

/// A slot may not be restored before it has been saved: a zeroed delta-net
/// shadow folds on silently and the output stays fluent.
class layer_state_slots {
protected:
    void _mark_saved(state_slot s) { saved_[slot_index(s)] = true; }
    void _check_saved(state_slot s, const char* who) const {
        if (!saved_[slot_index(s)])
            throw std::runtime_error(std::string(who) + ": restore of slot " +
                                     std::to_string(slot_index(s)) +
                                     " before any save");
    }

private:
    std::array<bool, NUM_STATE_SLOTS> saved_{};
};

// --------------------------------------------------------------------------
/// Check one entry of config.json's layer_types against (i+1)%4==0. The array
/// is what ships, so parse it and assert the two agree. True = full attention.
inline bool check_layer_type(const std::string& type, size_t i) {
    const bool full = type == "full_attention";
    if (!full && type != "linear_attention")
        throw std::runtime_error("unknown layer_type '" + type + "' at index " +
                                 std::to_string(i));
    if (full != cpu_ref::q38::is_full_attention(static_cast<int>(i)))
        throw std::runtime_error(
            "layer_types[" + std::to_string(i) + "] = " + type +
            " disagrees with (i+1)%4==0");
    return full;
}

} // namespace qwen38
