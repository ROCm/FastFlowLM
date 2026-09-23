/// \file qwen3_8mtp_npu.hpp
/// \brief qwen3_8mtp_npu class
/// \author FastFlowLM Team
/// \date 2026-09-16
/// \version 0.9.28
/// \note Qwen3.8-27B: 64 layers, 48 GatedDeltaNet + 16 full attention in the
///       pattern [lin,lin,lin,full] x 16, plus a 1-layer MTP draft head.
/// \note Phase 1 is CPU-only -- no xclbin, no NPU sequence header. The
///       constructor still takes an npu_xclbin_manager* so the signature
///       matches every other engine and Phase 2 needs no call-site change.
/// \note Nothing from detail/ may be included here: the runtime's
///       automodel.hpp pulls this header in.
#pragma once
#include "lm_config.hpp"
#include "npu_utils/npu_utils.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"
#include "tensor_2d.hpp"
#include "utils/utils.hpp"
#include "causal_lm.hpp"
#if USEAVX2
#include <immintrin.h>  // For AVX intrinsics
#endif
#include <functional>


/// \brief one image's patches, as the HF image processor emits them
/// \note `pixel_values` is [grid_t*grid_h*grid_w][in_channels * temporal_patch *
///       patch^2] fp32, already in MERGE-BLOCK order (the processor's
///       reshape/permute), which is what makes the merger's 2x2 regroup a free
///       reinterpretation downstream. Feeding raster order is not a crash: the
///       tower runs and the image is silently scrambled.
/// \note grid_h and grid_w count PATCHES, not pixels, and must both be
///       multiples of spatial_merge_size.
struct qwen3_8mtp_image_t {
    const float* pixel_values = nullptr;
    int          grid_t       = 1;
    int          grid_h       = 0;
    int          grid_w       = 0;
};

/// \brief the images a prefill window refers to, in prompt order
/// \note The engine matches these against the window's image_token_id rows and
///       throws if the counts disagree -- a mismatch would splice one image's
///       rows into another's slots, which every downstream stage accepts.
struct qwen3_8mtp_image_payload_t {
    const qwen3_8mtp_image_t* images     = nullptr;
    int                       num_images = 0;
};

/// \brief optional payload for prefill()
/// \note This is the same `void* payload` seam qwen3_5vl_npu uses for images.
///       Phase 1 needs it for one thing the reference forces: a multimodal
///       window's position ids are NOT 0..L-1. In the dumped prompt 221 tokens
///       occupy only 165 positions, because a 16x16 image block consumes 64
///       token slots while advancing position by 8. Anything that reproduces a
///       captured window must therefore supply positions explicitly.
/// \note Both pointers are borrowed and only read during the call.
struct qwen3_8mtp_payload_t {
    /// [3][L] m-rope position ids, channels t,h,w. Null = pure text, in which
    /// case the engine emits ctx..ctx+L-1 on all three channels.
    const int32_t* position_ids = nullptr;
    /// [L][hidden_size] fp32 hidden states to use instead of the embedding
    /// lookup. Null = look every id up in model.embed_tokens. Phase 2's vision
    /// encoder writes image rows here; the milestone driver uses it to feed a
    /// captured inputs_embeds.
    const float*   embeds       = nullptr;
    /// Raw images for the engine's own vision tower to encode. Appended last so
    /// existing brace-initialised call sites keep compiling.
    /// \note `embeds` wins if both are set: a caller supplying hidden states has
    ///       already resolved its image rows, and running the tower again would
    ///       be work whose result is thrown away.
    const qwen3_8mtp_image_payload_t* images = nullptr;
};

/// \note This is the only engine that overrides causal_lm's speculation hooks.
///       They are defaulted in the base, so the other 15 engines keep the
///       single-token path unchanged -- but their .so files must still be
///       rebuilt in lockstep, because the added virtuals renumber the shared
///       vtable. See the ABI warning in causal_lm.hpp.
class qwen3_8mtp_npu : public causal_lm{
public:
    /// \brief initialize the qwen3_8mtp_npu
    /// \param config the configuration
    /// \param npu_instance the npu instance (unused in phase 1, CPU only)
    /// \param MAX_L the maximum context length
    qwen3_8mtp_npu(LM_Config config, npu_xclbin_manager *npu_instance, int MAX_L = 4096);
    ~qwen3_8mtp_npu();

    /// \brief forward one token
    /// \param ids the token id
    /// \return the last-token logits
    buffer<bf16> forward(int ids) override;

    /// \brief prefill a window of tokens
    /// \param ids the token ids
    /// \param payload reserved for the phase-2 vision path
    /// \return the last-token logits
    /// \note Appends to the KV/SSM state; never resets. _chunked_insert calls
    ///       this repeatedly for one prompt.
    buffer<bf16> prefill(std::vector<int>& ids, void* payload = nullptr) override;

    /// \brief set the context length
    /// \param L the context length
    void set_context_length(int L) override;

    /// \brief load the weights
    /// \param q4nx the q4nx
    void load_weights(Q4NX& q4nx) override;

    /// \brief clear the context
    void clear_context() override;

    /// \brief get the k cache
    /// \note dead ceremony; nothing in the runtime calls it
    buffer<bf16> get_k_cache(int layer_idx, int idx) override;

    /// \brief get the v cache
    /// \note dead ceremony; nothing in the runtime calls it
    buffer<bf16> get_v_cache(int layer_idx, int idx) override;

    /// \brief update the max length
    /// \param MAX_L the max length
    void update_max_length(uint32_t MAX_L) override;

    /// \brief get the current context length
    /// \return the current context length
    int get_current_context_length() override;
    int checkpoint() override;
    int restore() override;

    /// \brief true once a checkpoint carrying the mtp.* tensors is loaded
    /// \note Answers false before load_weights(), and false forever for a
    ///       checkpoint without an MTP head -- that is still a valid base
    ///       model, it just decodes one token at a time.
    bool supports_speculation() const override;

    /// \brief draft with the MTP head, verify against the full stack, commit
    /// \param last_token the token the caller would otherwise pass to forward()
    /// \param max_draft draft depth, clamped to MTP_STEPS (7)
    /// \return the committed tokens (1..max_draft+1), or {} to decline
    /// \note The returned tokens are already in the caches. Feeding them back
    ///       through forward() would double-append them.
    /// \note Greedy only -- see the precondition on causal_lm::speculate.
    std::vector<int> speculate(int last_token, int max_draft) override;

    // ---- speculation statistics -------------------------------------------
    // All non-virtual: they add no vtable slot, so unlike the two speculation
    // hooks on causal_lm these cost no lockstep rebuild of the other engines.
    //
    // Worth reporting at all because a broken draft path is SILENT. Verify
    // overrides every rejected draft with the base model's own argmax, so bad
    // drafting produces correct text and merely burns time -- the hit rate is
    // the only outward symptom. Without it a regression here reads as "the
    // machine feels slow today".

    /// \brief mean drafts accepted per speculative cycle since load
    double mean_accepted_length() const;

    /// \brief fraction of offered drafts the base model agreed with, in [0,1]
    /// \note Denominator is drafts actually offered, which is not
    ///       cycles x MTP_STEPS: k is clamped by max_draft and by the MAX_L
    ///       headroom check, so short cycles would otherwise be charged full.
    /// \note The free token committed at the first mismatch is NOT counted as
    ///       a hit -- it is the base model's argmax, not a correct draft.
    ///       Counting it would floor this at 1/k for a head that never once
    ///       guessed right.
    double draft_hit_rate() const;

    /// \brief speculative cycles run since load; 0 means speculation never ran
    /// \note Check this before reading a rate: with no cycles the rates are
    ///       0.0, which is indistinguishable from "ran and missed everything".
    uint64_t speculation_cycles() const;

    /// \brief one-line summary, or "" if no cycle ever ran
    std::string speculation_stats() const;

    /// \brief print that summary; prints nothing when no cycle ran
    /// \note Engine-side rather than in the runtime decode loop on purpose:
    ///       this model's AutoModel wrapper has its own generate() that does
    ///       not go through _shared_generate, so a runtime-side counter would
    ///       stay silent on the main chat path.
    void report_speculation_stats() const;

    /// \brief multi-line draft/verify/replay time breakdown, or "" if no cycle
    /// \note Lives here rather than in the runtime's profiler because from
    ///       automodel.cpp one cycle IS one speculate() call -- the phase
    ///       boundary does not exist at that level, so DECODING_TIME can only
    ///       ever charge draft, verify and replay to a single number.
    /// \note Indented two levels so it sits under show_profile()'s
    ///       "Statistics:" block. Ends with a newline.
    std::string speculation_timing() const;

    /// \brief (draft + verify + replay) / cycle time, in [0,1]; 0 if no cycle
    /// \note Denominator is the measured cycle, not the sum of the parts, so a
    ///       phase that is double-counted pushes this ABOVE 1.0 instead of
    ///       normalising itself away. That is the whole point of exposing it:
    ///       a breakdown whose parts do not add up to the whole misleads more
    ///       than no breakdown at all, and only this ratio catches it.
    double speculation_phase_fraction() const;

    /// \brief microseconds of the LAST speculate() that were prompt-phase work
    /// \return 0 on every cycle but the first one after each prefill()
    /// \note Step 0 of that first cycle is the draft head absorbing the prompt:
    ///       its catch-up window is the whole last prefill chunk, run once,
    ///       through the prefill bitstream. It is prefill, not decode, and it
    ///       scales with the PROMPT -- so leaving it inside DECODING_TIME makes
    ///       decode tok/s a function of prompt length. The caller moves this
    ///       many microseconds from DECODING_TIME to PREFILL_TIME.
    /// \note Read it immediately after speculate(); the next call overwrites it.
    uint64_t last_speculation_prime_us() const override;

    /// \brief zero the counters, e.g. between benchmark runs
    /// \note Counters are cumulative since load and survive clear_context(),
    ///       so a per-turn rate needs this at the start of each turn.
    /// \note Zeroes the timers too -- a rate and a duration measured over
    ///       different windows do not compose.
    void reset_speculation_stats();

    // ---- phase 2 and 3: MTP draft + verify -------------------------------
    // Not part of causal_lm. The runtime cannot drive speculation yet, so
    // these are the engine-side half only, reachable from the milestone
    // driver and from a future speculate() override.

    /// \brief the raw hidden states the last prefill()/forward() produced
    /// \return [L][hidden_size] fp32, layer 63's output -- BEFORE model.norm.
    ///         Valid until the next prefill()/forward().
    /// \note This is the layer tap, for scoring against prefill.layer_63_out.
    ///       It is NOT what the MTP head consumes -- see last_normed_states().
    const float* last_hidden_states(int& L, int& D) const;

    /// \brief the same states with model.norm applied
    /// \return [L][hidden_size] fp32, equal to the reference's
    ///         prefill.final_norm_out. Valid until the next call.
    /// \note This is what the MTP head's fusion takes as `prev_hidden`, and
    ///       the distinction is load-bearing: the dump's mtp.step0.prev_hidden
    ///       matches final_norm_out exactly (cos 1.0) and the pre-norm tap
    ///       only to cos 0.94. Feeding the raw tap makes the head fuse
    ///       unnormalised states -- it still drafts plausible tokens, so the
    ///       failure shows up only as a quietly poor acceptance rate.
    const float* last_normed_states(int& L, int& D);

    /// \brief draft up to k continuations with the MTP head
    /// \param ids the window's token ids (k=1 in steady state)
    /// \param prev_hidden [T][hidden_size], the states that predicted them
    /// \param T window length
    /// \param k how many tokens to draft
    /// \return the drafted tokens, greedy argmax; empty if no MTP head loaded
    /// \note Greedy is not a default, it is the contract: acceptance is an
    ///       exact integer compare, so a sampled draft would match only by
    ///       coincidence and the speedup would vanish.
    std::vector<int> mtp_draft(const std::vector<int>& ids,
                               const float* prev_hidden, int T, int k);

    /// \brief run the k+1 verify tokens through all 64 layers and accept
    /// \param verify_ids the k+1 tokens: the committed token then the k drafts
    /// \param drafts the k drafted tokens, to compare against
    /// \param out_argmax receives the base model's argmax at all k+1 positions
    /// \return how many drafts were accepted; `accepted + 1` tokens commit
    /// \note Snapshots the stack first and rolls back on a partial accept --
    ///       the delta-net fold has no inverse, so there is no other way back.
    int prefill_verify(const std::vector<int>& verify_ids,
                       const std::vector<int>& drafts,
                       std::vector<int>& out_argmax);

    /// \brief whether the checkpoint carried an MTP head
    bool has_mtp_head() const;

    // ---- module 4: the vision tower ---------------------------------------
    // Not part of causal_lm. prefill() drives the tower itself when the payload
    // carries images; these expose it directly so the milestone driver can
    // score the encoder against a dump without a prompt around it.

    /// \brief true once a checkpoint with vision_weight.q4nx is loaded
    /// \note False before load_weights(), and false for a text-only checkpoint.
    ///       Passing images to prefill() then throws rather than silently
    ///       dropping them.
    bool has_vision_tower() const;

    /// \brief language tokens one grid produces: t*h*w / spatial_merge_size^2
    /// \note The prompt must carry exactly this many image_token_id rows.
    int image_tokens_for(int grid_t, int grid_h, int grid_w) const;

    /// \brief run one image through the tower
    /// \param out [image_tokens_for(...)][hidden_size] fp32
    /// \throws if no vision tower is loaded
    void encode_image(const float* pixel_values, int grid_t, int grid_h,
                      int grid_w, float* out);

    /// \brief which backend the tower's matmuls and attention are running on
    /// \return e.g. "mm=cpu attn=cpu bf16", or "" with no tower loaded
    std::string vision_backend() const;

    /// \brief capture the tower's per-stage intermediates on the next encode
    /// \param fn called as (stage_name, [rows][cols] fp32, rows, cols); the
    ///        pointer dies when the call returns. Pass {} to stop capturing.
    /// \note Costs nothing when unset, so this is the only instrumentation the
    ///       tower carries -- there is no debug build of it.
    void set_vision_tap(std::function<void(const char*, const float*, int, int)> fn);
private:
    struct Impl;
    Impl* _impl;
};
