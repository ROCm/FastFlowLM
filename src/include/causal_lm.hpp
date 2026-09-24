/// \file causal_lm.hpp
/// \brief causal_lm class
/// \author FastFlowLM Team
/// \date 2025-06-24
/// \version 0.9.10
/// \note This class is a virtual class for causal language models
/// \note All other models should inherit from this class so that they can be used in the same way.
#pragma once
#include <cstdint>

#include "tensor_utils/q4_npu_eXpress.hpp"
#include "tensor_2d.hpp"
#include "utils/utils.hpp"
#include "buffer.hpp"

/// \brief causal_lm class
class causal_lm {
public:
    causal_lm(){}
    virtual ~causal_lm(){}

    /// \brief forward the causal_lm
    /// \param ids the ids
    /// \return the output
    virtual buffer<bf16> forward(int ids) = 0;

    /// \brief prefill the causal_lm
    /// \param ids the ids
    /// \return the output
    virtual buffer<bf16> prefill(std::vector<int>& ids, void* payload = nullptr) = 0;

    /// \brief set the context length
    /// \param L the context length
    virtual void set_context_length(int L) = 0;

    /// \brief load the weights
    /// \param q4nx the q4nx
    virtual void load_weights(Q4NX& q4nx) = 0;

    /// \brief update the max length
    /// \param MAX_L the max length
    virtual void update_max_length(uint32_t MAX_L) = 0;

    /// \brief clear the context
    virtual void clear_context() = 0;

    /// \brief get the k cache
    /// \param layer_idx the layer index
    /// \param idx the index
    /// \return the k cache
    virtual buffer<bf16> get_k_cache(int layer_idx, int idx) = 0;

    /// \brief get the v cache
    /// \param layer_idx the layer index
    /// \param idx the index
    /// \return the v cache
    virtual buffer<bf16> get_v_cache(int layer_idx, int idx) = 0;

    /// \brief get the current context length
    /// \return the current context length
    virtual int get_current_context_length() = 0;

    virtual int checkpoint() = 0;

    virtual int restore() = 0;

    // ---- speculative decoding ---------------------------------------------
    // Defaulted, not pure: only engines with a draft head override these, and
    // a pure virtual here would fail to compile all 16 other subclasses.
    //
    // ABI WARNING -- adding a virtual to this class is a lockstep change.
    // Every engine .so emits its own vtable for its causal_lm subclass and flm
    // loads those .so files prebuilt; flm emits no vtable of its own. Appending
    // here renumbers the vtable, so any .so not rebuilt keeps one that is short
    // by the number of methods added, and calling a new method reads past its
    // end. Measured: these two took the vtable from 120 to 136 bytes, and every
    // stale .so segfaulted on its first decode step -- after a clean prefill,
    // with no diagnostic. A link check cannot catch it; the vtable symbol still
    // resolves, it is merely the wrong size.
    //
    // So: every .so in src/lib/xrt must be rebuilt and re-copied in the same
    // change as any edit to this class. Verify with
    //   readelf -sW <lib>.so | grep _ZTV   # all engine vtables must agree
    // not with "it compiled" and not with "it linked".

    /// \brief whether speculate() can currently run
    /// \return false unless the engine has a draft head and loaded weights for it
    /// \note Default false: an engine that does not override this is never
    ///       asked to speculate and keeps the single-token path exactly.
    virtual bool supports_speculation() const { return false; }

    /// \brief propose several tokens at once and return the ones the model agrees with
    /// \param last_token the most recently committed token -- the one the caller
    ///        would otherwise pass to forward()
    /// \param max_draft how many tokens to draft; the engine may draft fewer
    /// \return the committed tokens, in order, at least one on success;
    ///         empty means "could not speculate this step"
    /// \note An empty return is not an error and not end-of-stream. It is how
    ///       the engine declines -- no draft head, not enough context, no room
    ///       under MAX_L -- and the caller must fall back to forward() +
    ///       sample() for that step.
    /// \note PRECONDITION: greedy sampling. Acceptance is an exact integer
    ///       compare against the base model's argmax, so the tokens returned
    ///       are argmax tokens. Calling this with top_k != 1, a temperature, or
    ///       repetition penalties active silently substitutes greedy decoding
    ///       for the sampler the user configured -- the output stays fluent, so
    ///       nothing downstream would ever flag it. The caller must gate on its
    ///       own sampler; the engine cannot see it.
    /// \note The engine has already committed these tokens to its caches. The
    ///       caller must not re-feed them through forward().
    virtual std::vector<int> speculate(int last_token, int max_draft) { return {}; }

    /// \brief how much of the last speculate() was prompt-phase, in microseconds
    /// \return 0 by default, and 0 on any cycle that did no prompt-phase work
    /// \note A draft head keeps its own KV cache, and on the first cycle after
    ///       a prefill that cache is empty: the cycle opens by re-absorbing the
    ///       prompt window before it drafts anything. That pass costs what a
    ///       prefill costs and scales with the prompt, but it happens inside a
    ///       speculate() call, where the caller's decode timer is running.
    ///       Engines that can tell the two apart report the split here so the
    ///       caller can move it to its prefill bucket; engines that cannot
    ///       return 0 and lose nothing, since the time is already counted.
    /// \note Valid only until the next speculate().
    virtual uint64_t last_speculation_prime_us() const { return 0; }
};