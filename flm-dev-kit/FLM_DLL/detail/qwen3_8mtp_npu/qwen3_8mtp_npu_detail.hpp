// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file qwen3_8mtp_npu_detail.hpp
/// \brief struct Impl: owns the layer stack and the three modules, and holds
///        the orchestration policy that wires them together.
#pragma once

#include "models/qwen3_8mtp/qwen3_8mtp_npu.hpp"

#include "common/embedding.hpp"
#include "common/layer_stack.hpp"
#include "common/lm_head.hpp"
#include "common/npu/lm_head8_op.hpp"
#include "common/npu/lm_head_op.hpp"
#include "common/rope.hpp"
#include "image/image_module.hpp"
#include "image/mrope.hpp"
#include "image/vision_model_def.hpp"
#include "common/tap_buffer.hpp"
#include "prefill/prefill_module.hpp"
#include "speculate/dflash2_draft.hpp"
#include "qwen3_8mtp_model_def.hpp"
#include "speculate/speculate_module.hpp"
#include "verify/verify_module.hpp"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

/// A saved text context: what hybrid_engine_state holds (delta-net blobs in `state`, attention K/V rows in `kv_k` / `kv_v`,
/// the lengths and the rope position) plus what only this engine keeps.
class qwen38_engine_state : public hybrid_engine_state {
public:
    using hybrid_engine_state::hybrid_engine_state;
    std::vector<int>   ids;        ///< committed_ids: the ids that have K/V rows, size() == context_length
    std::vector<float> hist;       ///< the hidden history the MTP head drafts from, [rows][D]
    int                hist_base = 0;   ///< context position of hist row 0
};

struct qwen3_8mtp_npu::Impl {
    Impl(LM_Config config, npu_xclbin_manager* npu_instance,
         qwen3_8mtp_npu* parent_ptr, int MAX_L = 4096);
    ~Impl();

    // ---- the causal_lm surface --------------------------------------------
    buffer<bf16> forward(int ids);
    buffer<bf16> prefill(std::vector<int>& ids, void* payload = nullptr);

    void set_context_length(int L, int new_general_token_idx);
    void load_weights(Q4NX& q4nx);
    void update_max_length(uint32_t MAX_L);
    void clear_context();

    buffer<bf16> get_k_cache(int layer_idx, int idx);
    buffer<bf16> get_v_cache(int layer_idx, int idx);
    buffer<bf16> get_logits(const float* hidden_row);

    int get_current_context_length();
    int checkpoint();
    int restore();

    // ---- saved states (see the public header) ------------------------------
    std::unique_ptr<hybrid_engine_state> new_state();
    int read_state(hybrid_engine_state& state, int kv_from);
    int write_state(const hybrid_engine_state& state);
    int write_state_kv(const hybrid_engine_state& state);

    // ---- internals ---------------------------------------------------------
    /// Module 1: embed, sweep 64 layers, project the last row. prefill() and
    /// forward() both land here; forward() is the L = 1 case.
    buffer<bf16> _prefill_with_mm(std::vector<int>& ids, void* payload);
    /// Up to L_TOKENS text tokens through the verify pass, committed in full (forward(), and the tail of a prompt
    /// that would not fit the prefill path's padding near the top of the cache).
    buffer<bf16> _verify_window(std::vector<int>& ids);

    /// m-rope position ids [3][L]. Pure text puts general_token_idx + t in all
    /// three channels; a multimodal caller supplies them, because position
    /// advance is not the token count (64 image tokens advance position by 8).
    /// \param spans if non-null, receives where this window's image rows sit,
    ///        which is what _prefill_with_mm splices the encoder's output into.
    ///        Only filled on the walked path -- a caller that hands over
    ///        `position_ids` has taken the whole job, spans included.
    std::vector<int32_t> get_input_idx(std::vector<int>& tokens, void* payload,
                                       std::vector<qwen38::image_span>* spans = nullptr);

    // ---- module 4: vision --------------------------------------------------
    /// Encode every image the payload carries and write its rows over the
    /// placeholder slots in `x`. No-op without images.
    void _splice_image_rows(const qwen3_8mtp_payload_t* p,
                            const std::vector<qwen38::image_span>& spans,
                            float* x, int L);

    void encode_image(const float* pixel_values, int t, int h, int w, float* out);

    /// model.norm + lm_head over `keep` rows of [L][D].
    void project(const float* x, int L, int keep, float* out);

    // ---- modules 2 and 3 ----------------------------------------------------
    const float* last_hidden_states(int& L, int& D) const;
    const float* last_normed_states(int& L, int& D);

    /// Every normed hidden row the draft head has not absorbed yet -- context
    /// positions `[mtp_hist_base, current_context_length)`, oldest first.
    ///
    /// This is what `speculate()` primes from, and it exists because
    /// `last_normed_states()` cannot answer the question. That one reports the
    /// LAST window, and decode is a one-row window: the app runs a `forward()`
    /// between `prefill()` and the first `speculate()`, so by the time the head
    /// asks for the prompt's 350 rows the only one still live is the row
    /// `forward()` just wrote. The other 349 were never wrong -- they were
    /// never kept. See `_mtp_hist_append`.
    const float* mtp_prev_rows(int& rows, int& D);
    /// model.norm `n` rows of a `[n][D]` pre-norm buffer onto the back of the
    /// history. Called from every site that commits rows to the caches, so the
    /// history and `committed_ids` advance together or not at all.
    void _mtp_hist_append(const float* pre_norm, int n);
    std::vector<int> mtp_draft(const std::vector<int>& ids,
                               const float* prev_hidden, int T, int k);
    /// speculate() with the DFlash2 drafter in place of the MTP head. Same verify, commit and statistics.
    std::vector<int> speculate_df(int last_token, int max_draft);
    /// Hand the drafter the taps of the rows [0, n_commit) of the window the last sweep ran, as committed tokens
    /// base_idx + r. `pos_t` is the window's rope position per row.
    void _df_append(int base_idx, const int32_t* pos_t, int n_commit) {
        if (df) df->append(taps, base_idx, pos_t, n_commit);
    }
    int prefill_verify(const std::vector<int>& verify_ids,
                       const std::vector<int>& drafts,
                       std::vector<int>& out_argmax);

    /// The stop ids, or empty when the caller never set them. Held as a plain
    /// vector rather than a set: it has three entries for this checkpoint and
    /// is scanned at most k+1 times per cycle, so a hash would cost more than
    /// it saves.
    std::vector<int> stop_tokens;
    /// see qwen3_8mtp_npu::set_speculation_chooser
    std::function<int(int, const float*, int)> spec_chooser;
    inline bool is_stop_token(int id) const {
        for (int s : stop_tokens) if (s == id) return true;
        return false;
    }
    /// First index in `argmax[0 .. upto]` holding a stop token, or -1. `upto`
    /// is inclusive, and is the accepted count -- positions past it were never
    /// going to commit, so a stop token up there is not a stop at all.
    inline int _first_stop(const std::vector<int>& argmax, int upto) const {
        if (stop_tokens.empty()) return -1;
        for (int i = 0; i <= upto && i < (int)argmax.size(); i++)
            if (is_stop_token(argmax[i])) return i;
        return -1;
    }

    /// How many rows the last prefill_verify() actually committed. Normally
    /// `accepted + 1`; shorter when a stop token truncated the batch. Read by
    /// speculate() to slice the return, which is why prefill_verify() can keep
    /// returning `accepted` for the milestone driver.
    int spec_last_commit = 0;

    /// Verify rows that belong to the tokens the last speculate() returned; 0
    /// when it declined. Kept apart from spec_last_commit because a decline
    /// leaves that, and the verify logits, at the previous cycle's values.
    int spec_logit_rows = 0;

    /// One full cycle: catch the head up, draft, verify, commit. Returns the
    /// committed tokens, or {} to decline.
    std::vector<int> speculate(int last_token, int max_draft);
    std::vector<int> speculate_inner(int last_token, int max_draft);

    /// The window the head must consume to reach the base stack. Its row r
    /// holds position r+1, so this is committed_ids[C-width+1 .. C-1] plus
    /// last_token. Width covers all three regimes: cold start (P), partial
    /// accept (1), and full accept (2) -- the last is the trap, since assuming
    /// a steady state of 1 misaligns the head permanently.
    std::vector<int> mtp_catch_up_window(int last_token, int width) const;

    inline bool is_attn_layer(int layer_idx) const {
        return desc.is_attn_layer(layer_idx);
    }

    /// RMSNorm's +1 is already folded into the q4nx weights. Adding it again
    /// is silent -- output stays fluent and slowly degrades -- so check the
    /// weights look folded (mean near 1) rather than trusting the converter.
    void _assert_norm_weights_folded();

    // ---- configuration -----------------------------------------------------
    LM_Config           config;
    npu_xclbin_manager* npu = nullptr;
    qwen3_8mtp_npu*     parent_ptr = nullptr;
    qwen3_8mtp_desc     desc;
    /// Non-const because the checkpoint corrects the padded MLP width when the
    /// tower's weights are opened, and every op binds to the corrected value.
    qwen3_8mtp_vision_desc vision_desc;
    uint32_t            MAX_L = 0;

    // ---- weights and stack -------------------------------------------------
    /// The mmap'd model.q4nx. Every tensor view points into it, so it must
    /// outlive the stack, the embedding table, the lm_head and the head.
    std::unique_ptr<cpu_ref::safetensors>     weights;
    /// Shared by modules 1 and 3: they are two executors over one set of caches.
    qwen38::layer_stack                       stack;
    std::unique_ptr<cpu_ref::embedding_table> embedding;
    /// The host projection. **Not a backend any more** -- both live projections
    /// are on bitstreams below. This is here for one caller, module 3's shape
    /// guard: a verify batch wider than `npu_lm_head_8::ROWS` has no device
    /// shape, and today's k + 1 = 8 never trips it. See verify/verify_cpu.hpp.
    std::unique_ptr<cpu_ref::lm_head>         head;
    /// The projection on `lm_head.xclbin`, at L = 1 -- the backend for modules
    /// 1 and 2, and never null: a missing file throws in load_weights(). Owned
    /// here rather than by the speculator because the base model and the draft
    /// head share one table and it is 1.35 GB on the device.
    ///
    /// Module 3 is not a caller: it projects k+1 = 8 rows and this bitstream
    /// moves the whole table per row. See get_logits() and `lm_head8_npu`.
    std::unique_ptr<q38npu::npu_lm_head>      lm_head_npu;
    /// The same projection on `lm_head_8.xclbin`, at L = 8 -- module 3's
    /// backend, and only module 3's. Never null, on the same terms.
    ///
    /// A SECOND 1.35 GB copy of the table: a BO belongs to the app that
    /// dispatches it, and these are two xclbins. 2.7 GB resident, and
    /// unconditional now that neither has a host arm to fall back to. See
    /// common/npu/lm_head8_op.hpp.
    std::unique_ptr<q38npu::npu_lm_head_8>    lm_head8_npu;
    std::vector<float>                        final_norm_w;   // model.norm

    // ---- the three modules -------------------------------------------------
    std::unique_ptr<qwen38::prefill_module>   prefill_mod;
    std::unique_ptr<qwen38::speculate_module> spec_mod;   ///< null without mtp.*, and when the DFlash2 drafter is loaded
    /// The DFlash2 drafter (draft package at config "draft_model_path", or $FLM_DFLASH_PATH). When set it REPLACES the
    /// MTP head: speculate() drafts a block with it and verify is unchanged. Null otherwise.
    std::unique_ptr<cpu_ref::safetensors>     df_weights;
    std::unique_ptr<qwen38::dflash2_draft>    df;
    /// The target taps the drafter conditions on: written by the prefill and verify sweeps, consumed right after each.
    qwen38::tap_buffer                        taps;
    std::unique_ptr<qwen38::verify_module>    verify_mod;

    // ---- module 4 ----------------------------------------------------------
    /// The vision tower. Null for a text-only checkpoint, and null until
    /// load_weights() -- its 959 MB of weights are a second file, opened only
    /// when vision_weight.q4nx is actually there.
    std::unique_ptr<qwen38::image_module>     image_mod;
    /// [tokens][D] fp32, one image at a time. Grown, never shrunk.
    std::vector<float>                        image_rows;

    // ---- outputs -----------------------------------------------------------
    /// [vocab_size] bf16, what the runtime samples. Valid until the next
    /// forward()/prefill().
    buffer<bf16>       logits;

    /// The rows the last prefill/verify run left live, pre-norm. The MTP
    /// head's fusion consumes these, so they outlive the call that made them.
    const float*       last_hidden = nullptr;
    int                last_window_len = 0;
    /// model.norm applied to those rows, on demand. The head's `prev_hidden`
    /// is this, not the raw layer-63 tap (cos 1.0 vs 0.94).
    std::vector<float> normed_states;

    // ---- running state -----------------------------------------------------
    int current_context_length = 0;   ///< tokens committed to the caches
    int general_token_idx      = 0;   ///< next m-rope position

    /// The m-rope walk's state, carried ACROSS prefill() calls. _chunked_insert
    /// splits one prompt into several windows, and an image can straddle the
    /// cut: restarting the walk per window would put the second half of an
    /// image back at its first row's position.
    /// \note Reset by clear_context(), alongside general_token_idx.
    qwen38::mrope_cursor mrope;

    /// The ids that have KV rows, so size() == current_context_length always.
    /// Catching the head up means re-feeding ids the base stack already
    /// folded, and the caches hold projections, not ids.
    std::vector<int> committed_ids;

    /// Absolute position of the head's cache row 0. Normally 1 -- the head is
    /// shifted one token forward. It slides only when the head cannot be
    /// primed over the full history (a prompt that arrived in chunks), which
    /// costs context; pinning it at 1 there would put every draft at the wrong
    /// rotary position.
    int mtp_pos_base = 1;

    /// model.norm'd hidden states for context positions
    /// `[mtp_hist_base, current_context_length)`, row major `[rows][D]`.
    ///
    /// The head fuses one hidden row per token it catches up on, so a row it
    /// cannot get is a row it cannot draft from -- that is the whole content of
    /// the `width > rows` clamp in `speculate_inner`. Keeping them here makes
    /// the first cycle after a prompt a full-width prime instead of a one-row
    /// one, which is what both step-0 bitstreams were built for.
    ///
    /// Bounded by construction: `rows == current_context_length - mtp_hist_base`
    /// and the front is dropped to the head's cursor after every cycle, so the
    /// peak is one prompt (≤ MAX_L × D floats) and the steady state is one or
    /// two rows. Never populated when there is no draft head.
    std::vector<float> mtp_hist;
    /// Absolute context position of `mtp_hist` row 0.
    int                mtp_hist_base = 0;

    int _mtp_hist_rows() const {
        return desc.D ? (int)(mtp_hist.size() / (size_t)desc.D) : 0;
    }

    /// Rewind the id log AND the hidden history to `n` committed tokens.
    ///
    /// They are one ledger. `speculate()` pairs `committed_ids[i]` with the
    /// hidden row at position `i-1`, so rewinding one without the other pairs
    /// every token with its neighbour's state -- which still drafts fluently
    /// and shows up only as acceptance decaying. Keep the two in one function
    /// so a future rewind site cannot remember one and forget the other.
    void _trim_committed(int n) {
        if (n < 0) n = 0;
        if ((int)committed_ids.size() > n) committed_ids.resize(n);
        if (df) df->truncate(n);

        if (mtp_hist_base > n) {          // the whole history is in the future
            mtp_hist.clear();
            mtp_hist_base = n;
        } else if (mtp_hist_base + _mtp_hist_rows() > n) {
            mtp_hist.resize((size_t)(n - mtp_hist_base) * (size_t)desc.D);
        }
    }

    /// Drop history rows below context position `upto` -- the ones the head has
    /// folded and can never ask for again. This is what keeps the peak at one
    /// prompt rather than one context.
    ///
    /// Call it with `current_context_length`, NOT with the head's `cache_len()`.
    /// A draft leaves the head's cursor at `C + (k-1)`, because steps 1..k-1
    /// append their own speculative rows, and those sit above every position
    /// the history describes. The two happen to agree today only because the
    /// clamp below saturates; say what is meant instead.
    void _mtp_hist_drop_front(int upto) {
        const int drop = std::min(upto - mtp_hist_base, _mtp_hist_rows());
        if (drop <= 0) return;
        mtp_hist.erase(mtp_hist.begin(),
                       mtp_hist.begin() + (size_t)drop * (size_t)desc.D);
        mtp_hist_base += drop;
    }

    // ---- speculation statistics -------------------------------------------
    /// The cycle costs roughly 1.7 plain decodes, so a mean accepted length
    /// below ~0.7 makes speculation a slowdown -- with output still perfect,
    /// because verify overrides every rejected draft. A broken draft path is
    /// invisible except through these counters.
    uint64_t spec_cycles         = 0;
    /// drafts accepted, raw, as the verifier returned them. NOT clipped to
    /// what the caller ended up emitting: a stop token truncates the commit
    /// but does not un-draft or un-verify anything, so a draft the base model
    /// agreed with past an EOS still counts here.
    uint64_t spec_accepted_total = 0;
    /// draft slots offered, raw. Summed per call rather than derived as
    /// cycles*MTP_STEPS because k is clamped per call, and that form would
    /// overstate any cycle that drafted short.
    uint64_t spec_drafted_total  = 0;
    uint64_t spec_declines       = 0;
    /// tokens returned to the caller, summed. Equals the caller's generated
    /// token count exactly -- that is the invariant the stop truncation in
    /// prefill_verify() exists to hold, and the reason this counter rather
    /// than spec_accepted_total is the one that reconciles against a usage
    /// block. The two differ by the free token per cycle, and past a stop by
    /// the truncated tail as well.
    uint64_t spec_tokens_total   = 0;

    // ---- per-phase timing --------------------------------------------------
    //   draft  -- k serial steps through a 1-layer head at L=1
    //   verify -- one batched pass through 64 layers at L=k+1, where the
    //             dequantised blocks are reused across all k+1 rows
    //   replay -- rollback + re-fold after a rejection: the true cost of a bad
    //             draft, invisible in the hit rate alone
    uint64_t spec_draft_us    = 0;   ///< prime excluded, see spec_prime_us
    uint64_t spec_verify_us   = 0;   ///< batched k+1 pass, replay excluded
    uint64_t spec_replay_us   = 0;
    uint64_t spec_cycle_us    = 0;
    uint64_t spec_draft_steps = 0;   ///< prime steps excluded
    uint64_t spec_verify_rows = 0;
    uint64_t spec_replays     = 0;

    // ---- the prime split ---------------------------------------------------
    // Step 0 of the FIRST cycle after a prompt prefill is not decode. The head
    // enters that cycle knowing nothing about the prompt, so its catch-up
    // window is the whole last prefill chunk -- hundreds of rows through the
    // prefill bitstream, once, to absorb what module 1 just wrote. Every later
    // cycle's step 0 is one or two rows, because crop() leaves the head at
    // C - 1; that is ordinary decode and stays in spec_draft_us.
    //
    // Charged separately because it is the only part of a speculate() call
    // that scales with the PROMPT rather than with the output, and leaving it
    // in DECODING_TIME makes tok/s a function of prompt length.
    //
    // Deliberately NOT armed by the rollback re-prefill in prefill_verify():
    // that one is the price of a bad draft and belongs to decode. Only the
    // public prefill() entry arms it.
    uint64_t spec_prime_us    = 0;
    uint64_t spec_prime_steps = 0;   ///< primes seen; one per prompt prefill
    uint64_t spec_prime_rows  = 0;   ///< step-0 width summed over those primes
    bool     spec_prime_pending = false;

    /// The prime charged to the cycle that just ran, for the runtime to move
    /// out of DECODING_TIME and into PREFILL_TIME. Zero on every other cycle.
    uint64_t spec_last_prime_us = 0;

    /// prefill_verify hands its replay cost up here; it is also called
    /// directly by the milestone driver, which has no cycle to charge it to.
    uint64_t spec_last_replay_us = 0;

    double mean_accepted_length() const {
        return spec_cycles ? (double)spec_accepted_total / (double)spec_cycles : 0.0;
    }
    double draft_hit_rate() const {
        return spec_drafted_total
             ? (double)spec_accepted_total / (double)spec_drafted_total : 0.0;
    }
    /// No tokens_per_cycle() here: it is mean_accepted_length() + 1.0 by
    /// construction, and the stats line prints the accepted mean alone. The
    /// timing block gives the cycle and token totals it would be derived from.
    double cycle_ms() const {
        return spec_cycles ? (double)spec_cycle_us / 1e3 / (double)spec_cycles : 0.0;
    }
    double ms_per_token() const {
        return spec_tokens_total
             ? (double)spec_cycle_us / 1e3 / (double)spec_tokens_total : 0.0;
    }
    /// Denominator is the measured cycle, not the sum of the phases, so what
    /// the three do not account for shows up as a shortfall from 100%.
    double draft_frac() const {
        return spec_cycle_us ? (double)spec_draft_us  / (double)spec_cycle_us : 0.0;
    }
    double verify_frac() const {
        return spec_cycle_us ? (double)spec_verify_us / (double)spec_cycle_us : 0.0;
    }
    double replay_frac() const {
        return spec_cycle_us ? (double)spec_replay_us / (double)spec_cycle_us : 0.0;
    }
    double prime_frac() const {
        return spec_cycle_us ? (double)spec_prime_us  / (double)spec_cycle_us : 0.0;
    }
    /// Per ROW, not per prime: a prime's cost is set by its window width, and
    /// two runs with different prompt lengths have nothing to compare per-prime.
    double ms_per_prime_row() const {
        return spec_prime_rows
             ? (double)spec_prime_us / 1e3 / (double)spec_prime_rows : 0.0;
    }
    /// Cycle time with the prompt-absorbing primes taken out -- what the
    /// runtime left in DECODING_TIME, and the only figure that divides sensibly
    /// by an output token count.
    uint64_t decode_cycle_us() const {
        return spec_cycle_us > spec_prime_us ? spec_cycle_us - spec_prime_us : 0;
    }
    double ms_per_token_decode() const {
        return spec_tokens_total
             ? (double)decode_cycle_us() / 1e3 / (double)spec_tokens_total : 0.0;
    }
    double ms_per_draft_step() const {
        return spec_draft_steps
             ? (double)spec_draft_us / 1e3 / (double)spec_draft_steps : 0.0;
    }
    /// Against ms_per_draft_step this is the batching win made explicit: it
    /// should fall as k rises, while the draft step does not.
    double ms_per_verify_row() const {
        return spec_verify_rows
             ? (double)spec_verify_us / 1e3 / (double)spec_verify_rows : 0.0;
    }

    /// "" when no cycle ever ran, rather than a row of zeros.
    std::string spec_stats_line() const;
    std::string spec_timing_block() const;
    void        report_spec_stats() const;

    // ---- multi-turn rewind -------------------------------------------------
    /// Distinct from the verify snapshot: that one lives and dies inside a
    /// cycle, this spans turns -- hence two state_slots per layer.
    bool is_checkpoint_valid           = false;
    int  checkpoint_context_length     = 0;
    int  checkpoint_general_token_idx  = 0;

    // ---- stale attention rows ----------------------------------------------
    /// Highest row any attention cache has had written since its rows past the cursor were last zero. The kernels stream rows
    /// above the cursor, and what an earlier conversation left there moves the result slightly (the thinking-mode answers of
    /// a cold start depended on the conversation that ran before it), so a cold start and a restored state zero [from, this).
    /// Counts the verify window and the 8-token tile that are written above the cursor.
    int attn_dirty_rows = 0;
    void _zero_attn_rows(int from) {
        const int to = attn_dirty_rows;
        for (size_t i = 0; i < stack.size(); i++) {
            if (stack[(int)i].is_full_attention()) stack[(int)i].zero_rows(from, to);
        }
        attn_dirty_rows = from;
    }
};
