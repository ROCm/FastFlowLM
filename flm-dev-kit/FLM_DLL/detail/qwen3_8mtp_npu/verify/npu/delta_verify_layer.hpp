// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file verify/npu/delta_verify_layer.hpp
/// \brief One whole gated-delta-net block on the NPU: a third implementation of
///        qwen38::decoder_layer, and the delta-net sibling of
///        `verify/npu/attn_layer.hpp`.
///
/// The bitstream is `FLM_Xclbin/Qwen3_8/gate_delta_net_verify`, compiled at
/// L_TOKENS = 8, and it runs the *whole* block -- the six projections, the
/// causal conv, the gated delta rule, the output gate and the MLP residual.
/// Only two things stay on the host, and the whole design of this file is about
/// those two.
///
/// ## The recurrent state does not come back
///
/// A delta-net block's "KV cache" is two tensors: the causal conv window and
/// the recurrent state S. The conv window crosses DDR both ways -- the pass
/// writes out one window per accept length, because at dispatch time nobody
/// knows how many drafted tokens verification will keep.
///
/// S does **not**. It goes in on IT4 MM2S ch0, the cores hold it resident for
/// all L_TOKENS steps, and nothing writes it back. That is not a limitation of
/// the array: the only state the device could report is the accept-*all* one,
/// which is exactly the case speculative decode cannot rely on. So the host
/// rebuilds S for the accepted prefix instead, by replaying
///
///     u_t = beta_t * (v_t - alpha_t * (S_{t-1} @ k_t))
///     S_t = alpha_t * S_{t-1} + k_t (x) u_t
///
/// over the triplet arena the pass already had to write for its own transpose.
/// The replay is a few MB of arithmetic per layer against a ~23 ms dispatch, so
/// it is free if it overlaps and expensive if it does not.
///
/// ## How the overlap is expressed
///
/// `decoder_layer::forward()` is one layer at a time and the stack is a
/// dependent chain -- layer i+1 needs layer i's h2 -- so the dispatches cannot
/// overlap each other. What can overlap is layer i's replay with layer i+1's
/// dispatch: the replay reads layer i's arena and layer i's S, and layer i+1
/// touches neither.
///
/// So `forward()` does not wait for the replay. It dispatches, takes h2,
/// carries the conv window, hands the fold to a worker and returns; the fold is
/// joined lazily, at the next use of *this* layer. With 64 layers and every 4th
/// one an attention block, that join is a full stack traversal away and has
/// always long since finished.
///
/// This is why the arg-0 BO is **per layer** and not shared the way
/// `attn_engine::x` is. One shared arena would mean layer i+1's dispatch
/// overwriting the rows layer i's replay is still reading, which is a data race
/// that produces a plausible-looking state rather than a crash.
/// `FLM_Q38_DELTA_SYNC=1` runs the fold inline for bisecting.
///
/// ## What a window costs
///
/// Per layer: arg 0 ~563 KB, kv_cache ~3.4 MB, aux ~82 KB, rms 20 KB -- about
/// 4 MB, 195 MB across the 48. The 211 MB of projections is **not** counted
/// there: this layer reads the same `npu_delta_weights` BO the prefill phase
/// stages, which is what the a/b tail past `proj_bytes` was appended for.
/// See `prefill/npu/delta_weights.hpp`.
#ifndef __QWEN3_8MTP_NPU_DELTA_VERIFY_LAYER_HPP__
#define __QWEN3_8MTP_NPU_DELTA_VERIFY_LAYER_HPP__

#include "common/layer.hpp"
#include "common/cpu_simd.hpp"             // the vectorised fp32 <-> bf16 marshalling
#include "common/npu/decode_prof.hpp"      // DPROF_*, no-ops without DECODE_PROFILE=1
#include "prefill/npu/delta_proj.hpp"      // q38npu::npu_delta_weights
#include "verify/npu/delta_sequence.hpp"

#include "npu_utils/npu_utils.hpp"
#include "buffer.hpp"
#include "typedef.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <future>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace q38npu {

// ==========================================================================
// staging -- the small weights only this bitstream needs
// ==========================================================================
/// Both layernorm gains into the arg-2 BO: input_layernorm then
/// post_attention_layernorm, adjacent because the rms tile's weight BD is
/// self-chained on one buffer and reloads it between the two phases.
inline void stage_delta_rms_weights(uint16_t* dst, const cpu_ref::safetensors& f,
                                    const delta_verify_layout& t, int layer) {
    const std::string p = "model.layers." + std::to_string(layer) + ".";
    const size_t n = static_cast<size_t>(t.D) * sizeof(uint16_t);
    if (f.nbytes(p + "input_layernorm.weight") < n ||
        f.nbytes(p + "post_attention_layernorm.weight") < n)
        throw std::runtime_error("a layernorm gain is shorter than D bf16");
    std::memcpy(dst,       f.raw(p + "input_layernorm.weight"),          n);
    std::memcpy(dst + t.D, f.raw(p + "post_attention_layernorm.weight"), n);
}

/// The arg-3 BO: | conv taps | ssm_norm |.
///
/// Adjacent, and that is load-bearing: one lengthened transfer on IT5 MM2S ch1
/// feeds MT[5]'s conv-tap BD and its ssm_norm BD back to back, so neither needs
/// a second shim channel. The caller must have zeroed the BO -- `conv_w_size +
/// v_head_dim` is what the descriptor moves and both tensors are exactly that
/// long, but a short `ssm_norm` in some future checkpoint would otherwise
/// stream whatever follows.
///
/// A plain memcpy for the taps and not `reorder_cpy`: the .q4nx stores them
/// tap-major, `[conv_k][conv_dim]`, which is already the memtile's layout --
/// the same assumption `cpu_ref::delta_net_weights::load` makes about the same
/// tensor. `ssm_norm` is one head's worth, replayed by MT[5] for all v_heads.
inline void stage_delta_aux_weights(uint16_t* dst, const cpu_ref::safetensors& f,
                                    const delta_verify_layout& t, int layer) {
    const std::string l = "model.layers." + std::to_string(layer) + ".linear_attn.";
    const size_t cn = t.conv_w_size            * sizeof(uint16_t);
    const size_t nn = (size_t)t.v_head_dim     * sizeof(uint16_t);
    if (f.nbytes(l + "ssm_conv1d.weight") != cn)
        throw std::runtime_error("ssm_conv1d is not [conv_k][conv_dim] bf16");
    if (f.nbytes(l + "ssm_norm.weight") < nn)
        throw std::runtime_error("ssm_norm is shorter than v_head_dim bf16");
    std::memcpy(dst,                 f.raw(l + "ssm_conv1d.weight"), cn);
    std::memcpy(dst + t.conv_w_size, f.raw(l + "ssm_norm.weight"),   nn);
}

// ==========================================================================
/// What the 48 delta-net layers share: the app and its sequence.
///
/// Less than `attn_engine` holds, and deliberately -- there is no shared x BO
/// here (see the header comment) and no `set_ctx()`, because a delta-net pass
/// does not depend on the context length. Its whole history is the conv window
/// and S, both fixed-size, so the sequence is emitted once and never again.
struct delta_engine {
    npu_app             app;
    delta_verify_layout t;
    /// Whether the S replay runs on a worker. False costs ~2-9 ms per layer per
    /// window; it exists so a suspected race can be ruled out in one run.
    bool                overlap = true;

    delta_engine(npu_app a, const delta_verify_layout& layout)
        : app(std::move(a)), t(layout) {
        const char* env = std::getenv("FLM_Q38_DELTA_SYNC");
        overlap = !(env && env[0] == '1');
        gen_delta_layer_seq(app.seq(), t);
    }
};

// ==========================================================================
/// One gated-delta-net layer, executed on the NPU.
///
/// Holds a whole `cpu_ref::delta_net_layer` rather than a weights-only half,
/// because a delta-net block's state and its weights live in one object: the
/// hybrid prefill path reaches it through `layer_stack::delta_half()`, and a
/// rollback restores it there. So this class does not keep a state of its own
/// at all -- it reads S and the conv window out of that object before a
/// dispatch and writes the advanced ones back after, which is exactly the
/// contract `recurrent_state_mut()` / `conv_window_mut()` document. It owns the
/// advance *instead of* `conv_silu` and `delta_rule`, and still owes one
/// `advance(L)`.
class npu_delta_layer final : public qwen38::decoder_layer,
                              private qwen38::layer_state_slots {
public:
    /// \param eng   shared app + sequence; must outlive this
    /// \param q4nx  an open model.q4nx
    /// \param g     block geometry, the one `eng->t` was built from
    /// \param w     this layer's staged projection BO, or null to stage one
    ///              here. The stack passes the prefill phase's
    ///              `npu_delta_weights` when it has one -- 211 MB per layer is
    ///              not worth two copies, and the a/b tail exists precisely so
    ///              one BO serves both bitstreams.
    npu_delta_layer(delta_engine* eng, const cpu_ref::safetensors& q4nx,
                    const cpu_ref::delta_net_geometry& g, int layer,
                    npu_delta_weights* w = nullptr)
        : eng_(eng), layer_(layer), cpu_(q4nx, g, layer), w_(w) {
        const delta_verify_layout& t = eng_->t;
        for (auto& s : shadow_) s.assign(cpu_.state_bytes(), 0);

        if (!w_) {
            own_w_ = std::make_unique<npu_delta_weights>(eng_->app, t.w, q4nx, layer);
            w_ = own_w_.get();
        }

        x_   = eng_->app.create_bo_buffer<bf16>(t.x_words);
        rms_ = eng_->app.create_bo_buffer<bf16>(t.rms_words);
        aux_ = eng_->app.create_bo_buffer<bf16>(t.aux_w_words);
        kv_  = eng_->app.create_bo_buffer<bf16>(t.kv_cache_words);

        // Nothing writes the two pad words after q, and the arena's head slots
        // above v_heads only ever get alpha / beta. Zero once so no core reads
        // a NaN out of a lane it is padding with.
        x_.memset((bf16)0);
        aux_.memset((bf16)0);
        kv_.memset((bf16)0);

        stage_delta_rms_weights(reinterpret_cast<uint16_t*>(rms_.data()),
                                q4nx, t, layer);
        stage_delta_aux_weights(reinterpret_cast<uint16_t*>(aux_.data()),
                                q4nx, t, layer);

        x_.sync_to_device();
        rms_.sync_to_device();
        aux_.sync_to_device();
        kv_.sync_to_device();

        arena_stage_.assign((size_t)t.v_heads * t.L_TOKENS * t.arena_block, (bf16)0);
    }

    ~npu_delta_layer() override { _join(); }

    // ---- decoder_layer -----------------------------------------------------
    /// Any L, through the one 8-wide bitstream: tile, and pad the last tile by
    /// repeating its final row.
    ///
    /// Padding a delta-net pass is a stronger claim than padding an attention
    /// pass, because the padding rows really do advance the device's
    /// recurrence -- there is no mask to keep them out. It is sound anyway,
    /// because nothing past row n-1 is ever read back: S does not return at
    /// all, the conv window is indexed at accept length n, and h2's rows above
    /// n are dropped. The host's replay walks n tokens of the arena and stops.
    ///
    /// The cos/sin arguments are ignored; a delta-net block has no rotary.
    void forward(const float* x, int L, const float* /*cos*/,
                 const float* /*sin*/, float* y) override {
        const delta_verify_layout& t = eng_->t;
        if (L <= 0) return;
        const size_t D = (size_t)t.D;

        for (int base = 0; base < L; base += t.L_TOKENS) {
            const int n = std::min(t.L_TOKENS, L - base);

            // The next window of the SAME layer starts from the state this
            // layer's last window is still folding. Overlap is across layers,
            // never within one -- so the join is here and not at the top.
            DPROF_V(delta_join_us, _join());

            DPROF_V(delta_write_x_us, _write_x(x + (size_t)base * D, n));
            DPROF_V(delta_state_us,   _write_state_in());
            DPROF_COUNT(delta_passes);

            const ert_cmd_state st = DPROF_T(delta_dispatch_us,
                eng_->app(x_, w_->weights(), rms_, aux_, kv_));
            if (st != ERT_CMD_STATE_COMPLETED)
                throw std::runtime_error(
                    "npu_delta_layer(" + std::to_string(layer_) +
                    "): dispatch state " + std::to_string((int)st) +
                    " -- a timeout is a sequence bug, not a numerical one");

            DPROF_V(delta_read_us, x_.sync_from_device(); kv_.sync_from_device());

            const bf16* h2 = x_.data() + t.layer_out_offset;
            float* yi = y + (size_t)base * D;
            DPROF_V(delta_widen_us, q38simd::bf16_to_f32(yi, h2, (size_t)n * D));

            DPROF_V(delta_carry_us, _carry_conv(n); _stage_arena(); cpu_.advance(n));

            // Everything the fold reads is now off the BOs: the arena is in
            // arena_stage_ and S_0 is cpu_'s own heap. So the next layer may
            // dispatch immediately, and this layer's BOs are not touched again
            // until its own next window.
            if (eng_->overlap)
                replay_ = std::async(std::launch::async,
                                     [this, n] { _replay(n); });
            else
                _replay(n);
        }

        // How far back `commit_prefix()` may reach. `arena_stage_` and the
        // conv-out block hold the LAST tile only, so a window that tiled has
        // no recoverable prefix and says so rather than folding the wrong rows.
        last_window_ = (L <= t.L_TOKENS) ? L : -1;
    }

    void reset() override { _join(); cpu_.reset(); last_window_ = -1; }

    int  tokens_seen()       const override { return cpu_.tokens_seen(); }
    bool is_full_attention() const override { return false; }

    /// A no-op: a delta-net block's state is the same size at 128 tokens and at
    /// 32768. Present only because `decoder_layer` requires it -- growing the
    /// context is an attention-cache concern.
    void set_max_length(int /*max_len*/) override {}

    // ---- state -------------------------------------------------------------
    /// ~3.3 MB, because the fold has no inverse. Identical to the CPU layer's,
    /// and stored in the CPU layer, so a rollback that crosses backends works.
    size_t state_bytes() const override { return cpu_.state_bytes(); }

    void save_state(qwen38::state_slot slot) override {
        _join();
        cpu_.save_state(shadow_[qwen38::slot_index(slot)].data());
        _mark_saved(slot);
    }
    void restore_state(qwen38::state_slot slot) override {
        _join();
        _check_saved(slot, "npu_delta_layer");
        cpu_.load_state(shadow_[qwen38::slot_index(slot)].data());
    }

    // ---- a state kept outside the layer (see decoder_layer::export_blob) ---
    size_t blob_bytes() const override { return cpu_.state_bytes(); }
    void export_blob(void* dst) override { _join(); cpu_.save_state(dst); }
    void import_blob(const void* src) override {
        _join();
        cpu_.load_state(src);
        last_window_ = -1;   // no forward() has produced an arena for this state
    }

    /// Land on the accept-`accept_len` state without re-running the block.
    ///
    /// Everything this needs the pass already produced, and all of it is still
    /// on the host: `slot` holds S_0 and the pre-batch conv window, `kv_` holds
    /// a *separate* conv window for every accept length (which is why the
    /// bitstream writes `L_TOKENS + conv_kernel_size - 2` rows rather than
    /// `conv_kernel_size - 1`), and `arena_stage_` holds the k / alpha / v /
    /// beta the cores wrote on the way past. So the conv half is a 60 KB
    /// memcpy at a different offset and the S half is a fold over `accept_len`
    /// tokens -- against the 64-layer weight stream that re-prefilling the
    /// accepted tokens used to cost.
    ///
    /// **The fold is deferred, and into a wider window than `forward()`'s.**
    /// `accept_len` is not known until layer 63 has produced argmaxes, so this
    /// cannot overlap its own pass. It does not need to: the next thing the
    /// engine does is draft the following cycle on the MTP head, which is one
    /// layer with its own private KV cache and touches none of these 48. The
    /// join is already in the right place -- the next cycle opens with
    /// `verify_module::snapshot()`, and `save_state()` joins.
    ///
    /// Owns the advance instead of `conv_silu` / `delta_rule`, and owes exactly
    /// one `advance()`, the same contract `forward()` keeps.
    void commit_prefix(qwen38::state_slot slot, int accept_len) override {
        const delta_verify_layout& t = eng_->t;
        _check_saved(slot, "npu_delta_layer");
        if (accept_len < 0 || accept_len > t.L_TOKENS)
            throw std::runtime_error("npu_delta_layer::commit_prefix: accept_len "
                                     "outside 0 .. L_TOKENS");
        if (last_window_ < 0 || accept_len > last_window_)
            throw std::runtime_error(
                "npu_delta_layer(" + std::to_string(layer_) +
                "): commit_prefix(" + std::to_string(accept_len) +
                ") but the staged arena holds " + std::to_string(last_window_) +
                " tokens -- the last forward() tiled, or none has run");

        _join();
        cpu_.load_state(shadow_[qwen38::slot_index(slot)].data());
        if (accept_len == 0) return;          // exactly restore_state()

        _carry_conv(accept_len);
        cpu_.advance(accept_len);
        if (eng_->overlap)
            replay_ = std::async(std::launch::async,
                                 [this, accept_len] { _replay(accept_len); });
        else
            _replay(accept_len);
    }
    bool can_commit_prefix() const override { return true; }

    // ---- accessors ---------------------------------------------------------
    /// The state of record. Joins the pending fold first, so a caller always
    /// sees a state that has been advanced by exactly `tokens_seen()` tokens.
    cpu_ref::delta_net_layer& inner() { _join(); return cpu_; }

    /// This layer's projection BO, whether it staged it or was handed one. The
    /// prefill phase's four `dequant_mm` dispatches read it at the same byte
    /// offsets; nothing here restages it.
    npu_delta_weights* delta_weights() { return w_; }

    int layer() const { return layer_; }

    // ---- testing -----------------------------------------------------------
    /// The causal conv window the last pass produced, for accept length
    /// `accept_len`: `(conv_kernel_size - 1) * conv_dim` floats, `[tap][channel]`
    /// oldest tap first. This is the device's own layout -- **not** the CPU
    /// layer's 4-row `conv_window()`, whose row 0 is a don't-care.
    std::vector<float> read_conv_out(int accept_len) {
        const delta_verify_layout& t = eng_->t;
        if (accept_len < 1 || accept_len > t.L_TOKENS)
            throw std::runtime_error("npu_delta_layer::read_conv_out: accept_len "
                                     "outside 1 .. L_TOKENS");
        kv_.sync_from_device();
        const size_t n = (size_t)(t.conv_kernel_size - 1) * t.conv_dim;
        const bf16* p = kv_.data() + t.conv_out_offset + t.conv_out_at(accept_len);
        std::vector<float> out(n);
        for (size_t i = 0; i < n; i++) out[i] = (float)p[i];
        return out;
    }

    /// The state after `accept_len` of the last pass's tokens, replayed from
    /// the staged arena. Does not touch `cpu_`, so a driver may ask for every
    /// accept length without disturbing the layer.
    ///
    /// \param S0 the state the pass *started* from, head-major
    ///        [v_heads][qk_head_dim][v_head_dim]
    std::vector<float> replay_state_at(const std::vector<float>& S0,
                                       int accept_len) const {
        const delta_verify_layout& t = eng_->t;
        if (accept_len < 0 || accept_len > t.L_TOKENS)
            throw std::runtime_error("npu_delta_layer::replay_state_at: accept_len "
                                     "outside 0 .. L_TOKENS");
        std::vector<float> S = S0;
        _fold(S.data(), accept_len);
        return S;
    }

    /// One (head, token) arena block as the device left it: k, alpha, v, beta,
    /// q. The direct control for the conv and a/b tiles -- h2 can look right
    /// while beta is a head off.
    void read_triplet(int head, int tok, std::vector<float>& k, float& alpha,
                      std::vector<float>& v, float& beta,
                      std::vector<float>& q) {
        const delta_verify_layout& t = eng_->t;
        x_.sync_from_device();
        const bf16* blk = x_.data() + t.arena_offset + t.arena_at(head, tok);
        const int dk = t.qk_head_dim, dv = t.v_head_dim;
        k.resize(dk); v.resize(dv); q.resize(dk);
        for (int i = 0; i < dk; i++) k[i] = (float)blk[i];
        std::memcpy(&alpha, blk + dk, sizeof(float));
        const bf16* vp = blk + t.triplet_stride;
        for (int i = 0; i < dv; i++) v[i] = (float)vp[i];
        std::memcpy(&beta, vp + dv, sizeof(float));
        const bf16* qp = blk + 2 * t.triplet_stride;
        for (int i = 0; i < dk; i++) q[i] = (float)qp[i];
    }

    buffer<bf16>& hidden_states() { return x_; }
    buffer<bf16>& kv_cache()      { return kv_; }

private:
    // ---- the host side of a pass -------------------------------------------
    /// fp32 -> bf16 for n rows, then pad to L_TOKENS by repeating row n-1.
    /// Repeating and not zeroing: a zero row goes through RMSNorm as
    /// 0 * rsqrt(eps) and a real row cannot produce a NaN.
    void _write_x(const float* xi, int n) {
        const delta_verify_layout& t = eng_->t;
        const size_t D = (size_t)t.D;
        bf16* xd = x_.data();
        q38simd::f32_to_bf16(xd, xi, (size_t)n * D);
        for (int r = n; r < t.L_TOKENS; r++)
            std::memcpy(xd + r * D, xd + (n - 1) * D, D * sizeof(bf16));
        x_.sync_to_device();
    }

    /// Push the CPU layer's two carried states into the arg-4 BO.
    ///
    /// S is f32 and goes in verbatim -- the region is sized in bf16 words but
    /// holds floats, which is the `* 2` in `s_words`.
    ///
    /// The conv window is the interesting half. The device wants
    /// `conv_kernel_size - 1` pre-conv rows, oldest first;
    /// `cpu_ref::delta_net_layer` keeps `conv_kernel_size` of them. The extra
    /// one is its row 0, which `conv_silu()` shifts out before it is ever
    /// multiplied by a tap -- so the device's row k is the CPU's row k + 1, and
    /// the CPU's row 0 is a don't-care in both directions.
    void _write_state_in() {
        const delta_verify_layout& t = eng_->t;
        const std::vector<float>& S = cpu_.recurrent_state();
        const size_t sf = (size_t)t.v_heads * t.qk_head_dim * t.v_head_dim;
        if (S.size() != sf)
            throw std::runtime_error("npu_delta_layer: the CPU layer's state is "
                                     "not [v_heads][qk_head_dim][v_head_dim]");
        std::memcpy(reinterpret_cast<float*>(kv_.data() + t.s_offset),
                    S.data(), sf * sizeof(float));

        // Taps k = 0 .. conv_kernel_size - 2 are contiguous on both sides -- the
        // device's row k is the CPU's row k + 1 and neither is strided -- so the
        // whole shift is one vectorised narrow, not a nest.
        const std::vector<float>& win = cpu_.conv_window();
        const size_t CD = (size_t)t.conv_dim;
        q38simd::f32_to_bf16(kv_.data(), win.data() + CD,
                             (size_t)(t.conv_kernel_size - 1) * CD);

        kv_.sync_to_device();
    }

    /// Carry the device's window for accept length n back into the CPU layer.
    ///
    /// The inverse of the mapping above, and row 0 is written as zero rather
    /// than left alone: a don't-care that is nonetheless deterministic is one a
    /// save/restore round trip and a driver's memcmp can both reason about. A
    /// comparison against a pure-CPU run must skip it -- the CPU has a real
    /// value there and this does not.
    void _carry_conv(int n) {
        const delta_verify_layout& t = eng_->t;
        const size_t CD = (size_t)t.conv_dim;
        const bf16* out = kv_.data() + t.conv_out_offset + t.conv_out_at(n);
        std::vector<float>& win = cpu_.conv_window_mut();
        std::fill(win.begin(), win.begin() + CD, 0.0f);
        q38simd::bf16_to_f32(win.data() + CD, out,
                             (size_t)(t.conv_kernel_size - 1) * CD);
    }

    /// Copy the arena off the BO before handing the fold to a worker.
    ///
    /// Two reasons, and both matter. The fold re-reads each block's k vector
    /// `qk_head_dim` times, and reads that land on an XRT BO cost ~100x what
    /// the same reads cost off ordinary memory -- measured at 9.5 ms against
    /// 0.09 ms for one sweep, with the sync itself only 0.05 ms. And once the
    /// bytes are on the heap the BO is free for this layer's next window with
    /// no ordering to get wrong.
    ///
    /// Only the first `v_heads` heads: heads are the arena's outer index, so
    /// the padded slots the a/b tile writes are a suffix and nothing reads them.
    void _stage_arena() {
        const delta_verify_layout& t = eng_->t;
        std::memcpy(arena_stage_.data(), x_.data() + t.arena_offset,
                    arena_stage_.size() * sizeof(bf16));
    }

    /// Fold the staged arena into the CPU layer's state, in place.
    void _replay(int n) { _fold(cpu_.recurrent_state_mut().data(), n); }

    /// The delta rule over `n` tokens of the staged arena, for every head.
    ///
    /// A replay of what the cores did, not a second derivation of it: beta is
    /// folded into u rather than multiplying the outer product, because
    /// applying it in both places would square it. `alpha * (S @ k)` and
    /// `(alpha * S) @ k` are the same number, which is why this matches
    /// `cpu_ref::delta_net_layer::delta_rule()` even though that scales S first.
    ///
    /// k, v, alpha and beta come off the arena post-conv, post-l2norm and
    /// post-sigmoid -- the cores wrote them there on the way past. So this
    /// reproduces none of the layer's arithmetic except the fold itself, and a
    /// disagreement with the CPU reference is a fold bug or an arena bug and
    /// cannot be a conv or gate bug.
    void _fold(float* S_all, int n) const {
        const delta_verify_layout& t = eng_->t;
        const int dk = t.qk_head_dim, dv = t.v_head_dim;
        const size_t count = (size_t)dk * dv;
        const bf16* arena = arena_stage_.data();

#pragma omp parallel
        {
            std::vector<float> u(dv);
#pragma omp for schedule(static)
            for (int h = 0; h < t.v_heads; h++) {
                float* S = S_all + (size_t)h * count;
                for (int tok = 0; tok < n; tok++) {
                    const bf16* blk = arena + t.arena_at(h, tok);
                    const bf16* v_v = blk + t.triplet_stride;
                    float alpha, beta;
                    std::memcpy(&alpha, blk + dk, sizeof(float));
                    std::memcpy(&beta,  v_v + dv, sizeof(float));

                    // u = beta * (v - alpha * (S @ k)), accumulated against the
                    // INCOMING S -- the core holds S_src fixed for the step.
                    for (int j = 0; j < dv; j++) u[j] = 0.0f;
                    for (int i = 0; i < dk; i++) {
                        const float kk = (float)blk[i];
                        if (kk == 0.0f) continue;
                        const float* Sr = S + (size_t)i * dv;
                        for (int j = 0; j < dv; j++) u[j] += Sr[j] * kk;
                    }
                    for (int j = 0; j < dv; j++)
                        u[j] = beta * ((float)v_v[j] - alpha * u[j]);

                    for (int i = 0; i < dk; i++) {
                        const float kk = (float)blk[i];
                        float* Sr = S + (size_t)i * dv;
                        for (int j = 0; j < dv; j++) Sr[j] = alpha * Sr[j] + kk * u[j];
                    }
                }
            }
        }
    }

    void _join() const {
        if (replay_.valid()) replay_.get();
    }

    delta_engine*             eng_ = nullptr;
    int                       layer_ = 0;
    cpu_ref::delta_net_layer  cpu_;   ///< the state of record, and the weights
    npu_delta_weights*        w_ = nullptr;             // non-owning
    std::unique_ptr<npu_delta_weights> own_w_;          // only if none was given

    buffer<bf16> x_;     ///< arg 0: | x | triplet arena | h2 |
    buffer<bf16> rms_;   ///< arg 2: | input_layernorm | post_attention_layernorm |
    buffer<bf16> aux_;   ///< arg 3: | conv taps | ssm_norm |
    buffer<bf16> kv_;    ///< arg 4: | conv in | S in | conv out |

    /// The arena, off the BO. Sized once; `_fold()` is the only reader.
    std::vector<bf16>     arena_stage_;
    mutable std::future<void> replay_;

    /// Tokens the last `forward()` window ran, or -1 if it tiled. The ceiling
    /// on what `commit_prefix()` can re-derive.
    int                   last_window_ = -1;

    std::array<std::vector<uint8_t>, qwen38::NUM_STATE_SLOTS> shadow_;
};

} // namespace q38npu

#endif  // __QWEN3_8MTP_NPU_DELTA_VERIFY_LAYER_HPP__
