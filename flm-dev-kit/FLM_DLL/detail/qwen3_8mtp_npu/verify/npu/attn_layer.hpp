// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file verify/npu/attn_layer.hpp
/// \brief One gated full-attention block on the NPU: a third implementation
///        of qwen38::decoder_layer, and the one the engine picks whenever it
///        has a device. The CPU attention layer stays as the reference the
///        drivers score against.
///
/// Each layer owns its weight, norm, rope and KV cache BOs; attn_engine holds
/// what all 16 share (the app, its sequence, the layout, the x BO).
///
/// The bitstream is compiled at L_TOKENS = 8 and dispatches nothing else, so
/// forward() tiles any L into 8-row passes and pads the last.
#ifndef __QWEN3_8MTP_NPU_ATTN_LAYER_HPP__
#define __QWEN3_8MTP_NPU_ATTN_LAYER_HPP__

#include "common/layer.hpp"
#include "common/cpu_simd.hpp"   // the vectorised fp32 -> bf16 marshalling
#include "common/npu/decode_prof.hpp"   // DPROF_*, no-ops without DECODE_PROFILE=1
#include "verify/npu/attn_sequence.hpp"

#include "npu_utils/npu_utils.hpp"
#include "buffer.hpp"
#include "typedef.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace q38npu {

// ==========================================================================
/// What the 16 attention layers share: the app, its sequence, and the x BO.
/// Members are public -- this is a shared context, not an abstraction.
struct attn_engine {
    npu_app            app;
    attn_layout        t;
    attn_rtp_addresses rtp;
    buffer<bf16>       x;      ///< arg 0: x in, o / down scratch, h2 out
    int                ctx = 0;

    attn_engine(npu_app a, const attn_layout& layout,
                const attn_rtp_addresses& r = attn_rtp_addresses{})
        : app(std::move(a)), t(layout), rtp(r) {
        x = app.create_bo_buffer<bf16>(t.x_words);
        x.memset((bf16)0);          // the o / down regions are never written
        x.sync_to_device();
        gen_attn_layer_seq(app.seq(), t, ctx, rtp);
    }

    /// Keys already in the cache when the next pass starts. Baked into the
    /// sequence, so changing it regenerates ~1000 descriptors -- once per pass,
    /// not once per layer, since all 16 enter a pass at the same length.
    void set_ctx(int n) {
        if (n == ctx) return;
        // The pass writes L_TOKENS rows from here and replays through them,
        // whatever L the caller asked for -- so the WHOLE of it has to fit, not
        // just its first row. The cache has no apron to absorb the overhang.
        if (n < 0 || n + t.L_TOKENS > t.MAX_L)
            throw std::runtime_error("attn_engine::set_ctx(" + std::to_string(n) +
                                     "): a pass from here writes through " +
                                     std::to_string(n + t.L_TOKENS) +
                                     ", outside a cache of MAX_L " +
                                     std::to_string(t.MAX_L));
        ctx = n;
        gen_attn_layer_seq(app.seq(), t, ctx, rtp);
    }

    /// Grow the layout. Never shrinks. Each layer then regrows its own cache BO.
    void grow(int max_len) {
        // The cache's alignment, not the replay's: PREFILL_CHUNK is what
        // attn_layout::build() requires, and it covers L_CHUNK. A request from
        // anywhere therefore lands on a legal buffer size.
        const int chunk = attn_cache::PREFILL_CHUNK;
        max_len = (max_len + chunk - 1) / chunk * chunk;
        if (max_len <= t.MAX_L) return;
        t = attn_layout::build(max_len, t.L_TOKENS);
        gen_attn_layer_seq(app.seq(), t, ctx, rtp);
    }
};

// ==========================================================================
/// One gated full-attention layer, executed on the NPU.
class npu_attn_layer final : public qwen38::decoder_layer,
                             private qwen38::layer_state_slots {
public:
    /// Allocates this layer's four BOs and fills the three weight ones. 205 MB
    /// and one reorder pass, paid at load; nothing moves per dispatch.
    ///
    /// All four must come from the engine's own app -- a BO from a second app
    /// does not fault, the dispatch simply never completes.
    npu_attn_layer(attn_engine* eng, const cpu_ref::safetensors& q4nx, int layer)
        : eng_(eng), layer_(layer) {
        const attn_layout& t = eng_->t;

        proj_  = eng_->app.create_bo_buffer<u8>(t.proj_bytes);
        rms_   = eng_->app.create_bo_buffer<bf16>(t.rms_words);
        rope_  = eng_->app.create_bo_buffer<bf16>(t.rope_words);
        cache_ = eng_->app.create_bo_buffer<bf16>(t.kv_cache_size);

        rope_.memset((bf16)0);
        cache_.memset((bf16)0);

        // bf16 is a class with a private raw_bits_; the staging functions emit
        // raw bit patterns, so these are copies of the representation.
        stage_attn_proj_weights(proj_.data(), q4nx, t, layer);
        stage_attn_rms_weights(reinterpret_cast<uint16_t*>(rms_.data()),
                               q4nx, t, layer);
        stage_attn_qk_norm(reinterpret_cast<uint16_t*>(rope_.data()) + t.rope_w_offset,
                           q4nx, t, layer);

        proj_.sync_to_device();
        rms_.sync_to_device();
        rope_.sync_to_device();
        cache_.sync_to_device();
    }

    // ---- decoder_layer -----------------------------------------------------
    /// Any L, through the one 8-wide bitstream: tile, and pad the last tile by
    /// repeating its final row.
    ///
    /// Padding is sound because the block mask is causal -- a row at index >= n
    /// reaches only outputs at index >= n, which this drops -- and because the
    /// cursor advances by the true L, so the padding's k/v lands above it. The
    /// replay streams those rows like any other, and the kernels' mask is what
    /// keeps them out: token t reads ctx + t + 1 keys, so the last real token
    /// stops at its own row.
    void forward(const float* x, int L, const float* cos, const float* sin,
                 float* y) override {
        const attn_layout& t = eng_->t;
        if (L <= 0) return;
        if (!cos || !sin)
            throw std::runtime_error("npu_attn_layer: attention needs a rotary table");
        if (cache_.size() != t.kv_cache_size)
            throw std::runtime_error("npu_attn_layer(" + std::to_string(layer_) +
                                     "): set_max_length() was not applied here");
        // Against the PADDED length, not L. The loop below tiles by L_TOKENS
        // and the last tile appends a full L_TOKENS rows however few are real,
        // so a call ending at MAX_L - 1 still writes through
        // round_up(L, L_TOKENS) rows -- and there is no apron under them.
        const int L_pad = (L + t.L_TOKENS - 1) / t.L_TOKENS * t.L_TOKENS;
        if (seen_ + L_pad > t.MAX_L)
            throw std::runtime_error(
                "npu_attn_layer: KV cache overflow -- " + std::to_string(L) +
                " rows from " + std::to_string(seen_) + " append through " +
                std::to_string(seen_ + L_pad) + " (the last tile is padded to "
                "L_TOKENS), past MAX_L " + std::to_string(t.MAX_L));

        const size_t D = (size_t)t.D, R = (size_t)t.ROPE_DIM_, half = R / 2;

        for (int base = 0; base < L; base += t.L_TOKENS) {
            const int n = std::min(t.L_TOKENS, L - base);
            const float* xi = x   + (size_t)base * D;
            const float* ci = cos + (size_t)base * R;
            const float* si = sin + (size_t)base * R;

            DPROF_V(attn_ctx_us, eng_->set_ctx(seen_));
            DPROF_COUNT(attn_passes);
            DPROF_B(stage);

            // Rotary table, head of this layer's rope BO -- the one thing
            // besides x the host still writes per pass, so the one BO besides x
            // still flushed per pass. q_norm / k_norm share the BO but sit past
            // rope_w_offset and were written at load.
            //
            // Both halves of an HF row carry the same angle, so take cos once
            // and put sin behind it. Padding rows repeat row n-1.
            bf16* rp = rope_.data();
            for (int r = 0; r < t.L_TOKENS; r++) {
                const int src = std::min(r, n - 1);
                for (size_t j = 0; j < half; j++) {
                    rp[r * R + j]        = (bf16)ci[src * R + j];
                    rp[r * R + half + j] = (bf16)si[src * R + j];
                }
            }
            rope_.sync_to_device();

            // fp32 -> bf16, the lossy half of the boundary. Padding rows repeat
            // the last real row rather than zeroing: a zero row goes through
            // RMSNorm as 0 * rsqrt(eps), a real row cannot produce a NaN.
            bf16* xd = eng_->x.data();
            q38simd::f32_to_bf16(xd, xi, (size_t)n * D);
            for (int r = n; r < t.L_TOKENS; r++)
                std::memcpy(xd + r * D, xd + (n - 1) * D, D * sizeof(bf16));
            eng_->x.sync_to_device();
            DPROF_E(stage, attn_stage_us);

            const ert_cmd_state st = DPROF_T(attn_dispatch_us,
                eng_->app(eng_->x, proj_, rms_, rope_, cache_));
            if (st != ERT_CMD_STATE_COMPLETED)
                throw std::runtime_error(
                    "npu_attn_layer(" + std::to_string(layer_) +
                    "): dispatch state " + std::to_string((int)st) +
                    " -- a timeout is a sequence bug or a stale RTP address");

            DPROF_V(attn_read_us, eng_->x.sync_from_device());
            const bf16* out = eng_->x.data() + t.h2_offset;
            float* yi = y + (size_t)base * D;
            DPROF_V(attn_widen_us, q38simd::bf16_to_f32(yi, out, (size_t)n * D));

            seen_ += n;
        }
    }

    /// Cursor only, like cpu_ref::attention_layer::reset(). Rows above the
    /// cursor are streamed by the replay but never read, so the stale context
    /// costs nothing -- it is the same state a rejected draft leaves behind.
    void reset() override { seen_ = 0; }

    int  tokens_seen()       const override { return seen_; }
    bool is_full_attention() const override { return true; }

    /// Grow the cache, keeping every byte it holds.
    ///
    /// **Two copies, not one.** Growing changes the plane pitch, so the v plane
    /// moves: a flat prefix copy would land the whole of v on top of the new k
    /// plane's tail and leave v itself zero. Each plane is carried separately,
    /// and each has to be carried whole rather than just [0, seen_) -- the
    /// replay streams up to the L_CHUNK boundary above the cursor, and those
    /// rows move the result at 1e-8.
    void set_max_length(int max_len) override {
        if (max_len < seen_)
            throw std::runtime_error("npu_attn_layer::set_max_length(" +
                                     std::to_string(max_len) + ") is below the " +
                                     std::to_string(seen_) + " rows already cached");

        // The widths do not change across a grow, only the pitch -- so the old
        // pitch is recoverable from the old BO, which is the only record of it
        // once the engine's layout has moved on.
        const size_t DKV = (size_t)eng_->t.DK;
        const size_t old_plane = cache_.size() / (2 * DKV);

        eng_->grow(max_len);                     // the sequence first; it owns MAX_L
        const attn_layout& t = eng_->t;
        if (cache_.size() == t.kv_cache_size) return;

        cache_.sync_from_device();
        std::vector<bf16> keep(cache_.data(), cache_.data() + cache_.size());

        cache_ = eng_->app.create_bo_buffer<bf16>(t.kv_cache_size);
        cache_.memset((bf16)0);
        const size_t new_plane = t.plane_rows();
        const size_t rows = std::min(old_plane, new_plane);   // grow() never shrinks
        std::memcpy(cache_.data() + t.k_at(0),
                    keep.data() + 0,                rows * DKV * sizeof(bf16));
        std::memcpy(cache_.data() + t.v_at(0),
                    keep.data() + old_plane * DKV,  rows * DKV * sizeof(bf16));
        cache_.sync_to_device();
    }

    // ---- state -------------------------------------------------------------
    /// The whole shadow is the cursor: rows above it are streamed by the replay
    /// but never read, so a rejected draft needs nothing cleared.
    size_t state_bytes() const override { return sizeof(int); }

    void save_state(qwen38::state_slot slot) override {
        cursor_[qwen38::slot_index(slot)] = seen_;
        _mark_saved(slot);
    }
    void restore_state(qwen38::state_slot slot) override {
        _check_saved(slot, "npu_attn_layer");
        seen_ = cursor_[qwen38::slot_index(slot)];
    }

    // ---- a state kept outside the layer (see decoder_layer::export_blob) ---
    size_t blob_bytes() const override { return sizeof(int); }
    void export_blob(void* dst) override { std::memcpy(dst, &seen_, sizeof(int)); }
    void import_blob(const void* src) override { std::memcpy(&seen_, src, sizeof(int)); }

    int kv_row_width() const override { return (int)eng_->t.DK; }

    /// K/V rows [from, to) as dense rows: token t's k is at t * DK and its v at MAX_L * DK + t * DV, so the planes are
    /// independent of MAX_L once the rows are out.
    void read_rows(int from, int to, void* k, void* v) override {
        const attn_layout& t = eng_->t;
        const size_t DKV = (size_t)t.DK;
        cache_.sync_from_device();
        for (int r = from; r < to; r++) {
            std::memcpy(static_cast<bf16*>(k) + (size_t)(r - from) * DKV, cache_.data() + t.k_at(r), DKV * sizeof(bf16));
            std::memcpy(static_cast<bf16*>(v) + (size_t)(r - from) * DKV, cache_.data() + t.v_at(r), DKV * sizeof(bf16));
        }
    }
    void write_rows(int from, int to, const void* k, const void* v) override {
        const attn_layout& t = eng_->t;
        if (from < 0 || to > (int)t.plane_rows())
            throw std::runtime_error("npu_attn_layer::write_rows: rows past the cache");
        const size_t DKV = (size_t)t.DK;
        cache_.sync_from_device();
        for (int r = from; r < to; r++) {
            std::memcpy(cache_.data() + t.k_at(r), static_cast<const bf16*>(k) + (size_t)(r - from) * DKV, DKV * sizeof(bf16));
            std::memcpy(cache_.data() + t.v_at(r), static_cast<const bf16*>(v) + (size_t)(r - from) * DKV, DKV * sizeof(bf16));
        }
        cache_.sync_to_device();
    }
    void zero_rows(int from, int to) override {
        const attn_layout& t = eng_->t;
        const int rows = (int)t.plane_rows();
        if (from < 0) from = 0;
        if (to > rows) to = rows;
        if (from >= to) return;
        cache_.sync_from_device();
        std::memset(cache_.data() + t.k_at(from), 0, (size_t)(to - from) * (size_t)t.DK * sizeof(bf16));
        std::memset(cache_.data() + t.v_at(from), 0, (size_t)(to - from) * (size_t)t.DK * sizeof(bf16));
        cache_.sync_to_device();
    }

    /// Keeping the accepted rows costs nothing: the device appended all
    /// `L_TOKENS` of them to the cache BO and the accepted ones were computed
    /// against exactly the causal prefix they will be read with. Rows above the
    /// new cursor stay where they are, which is the same stale context a
    /// rejected draft already leaves behind.
    void commit_prefix(qwen38::state_slot slot, int accept_len) override {
        _check_saved(slot, "npu_attn_layer");
        seen_ = cursor_[qwen38::slot_index(slot)] + accept_len;
    }
    bool can_commit_prefix() const override { return true; }

    // ---- testing -----------------------------------------------------------
    /// Seed the cache from cpu_ref::attention_layer::cache_k() / cache_v(),
    /// fp32 [max_len][DKV]. Real keys, correlated with the queries that will
    /// read them -- random ones put all the softmax mass on the appended row
    /// and make the replay untestable.
    void mirror_from_cpu(const std::vector<float>& k, const std::vector<float>& v,
                         int n) {
        const attn_layout& t = eng_->t;
        const size_t DKV = (size_t)t.DK;
        if (n < 0 || n > t.MAX_L)
            throw std::runtime_error("npu_attn_layer::mirror_from_cpu: n out of range");
        if (k.size() < (size_t)n * DKV || v.size() < (size_t)n * DKV)
            throw std::runtime_error("npu_attn_layer::mirror_from_cpu: short cache");

        for (int r = 0; r < n; r++) {          // row by row: the source is planar
            q38simd::f32_to_bf16(cache_.data() + t.k_at(r),
                                 k.data() + (size_t)r * DKV, DKV);
            q38simd::f32_to_bf16(cache_.data() + t.v_at(r),
                                 v.data() + (size_t)r * DKV, DKV);
        }
        cache_.sync_to_device();
        seen_ = n;
    }

    /// Append `n` CPU-computed k/v rows at the current cursor and advance it.
    ///
    /// `mirror_from_cpu` relative to the cursor rather than absolute, and it is
    /// what lets one cache and one cursor serve both paths: the prefill
    /// attention core (`prefill/npu/prefill_attn_op.hpp`) computes its own
    /// projections and RoPE on the CPU, writes the rows here, and then hands
    /// `cache()` straight to `attn.xclbin`. A later verify pass on the same
    /// layer reads them through the same accessors and cannot tell which path
    /// wrote them.
    ///
    /// The rows are post-norm and post-RoPE -- `cpu_ref::attention_layer`'s
    /// `cache_k()` layout, not raw projections -- packed [n][DK] each.
    ///
    /// Not part of any dispatch; the device's own append is unaffected.
    void append_rows(const float* k, const float* v, int n) {
        const attn_layout& t = eng_->t;
        const size_t DKV = (size_t)t.DK;
        if (n < 0 || seen_ + n > t.MAX_L)
            throw std::runtime_error(
                "npu_attn_layer::append_rows: " + std::to_string(n) +
                " rows at cursor " + std::to_string(seen_) + " passes MAX_L " +
                std::to_string(t.MAX_L));
        // The prefill path calls this once per layer per window with n = L, so
        // it is 16 x L x 2048 conversions and wants the vectorised loop; the
        // verify path's n is 1..4 and takes the same code harmlessly.
        for (int r = 0; r < n; r++) {
            q38simd::f32_to_bf16(cache_.data() + t.k_at(seen_ + r),
                                 k + (size_t)r * DKV, DKV);
            q38simd::f32_to_bf16(cache_.data() + t.v_at(seen_ + r),
                                 v + (size_t)r * DKV, DKV);
        }
        cache_.sync_to_device();
        seen_ += n;
    }

    /// Overwrite this layer's weights with caller-built blobs. For drivers that
    /// want to inject rather than load; forward() never restages.
    void stage(const std::vector<uint8_t>& proj, const std::vector<uint16_t>& rms,
               const std::vector<uint16_t>& qk_norm) {
        if (proj.size() != proj_.size())
            throw std::runtime_error("npu_attn_layer::stage: proj blob is " +
                                     std::to_string(proj.size()) + " B, BO is " +
                                     std::to_string(proj_.size()));
        std::memcpy(proj_.data(), proj.data(), proj.size());
        std::memcpy(reinterpret_cast<uint16_t*>(rms_.data()), rms.data(),
                    rms.size() * sizeof(uint16_t));
        std::memcpy(reinterpret_cast<uint16_t*>(rope_.data()) + eng_->t.rope_w_offset,
                    qk_norm.data(), qk_norm.size() * sizeof(uint16_t));
        proj_.sync_to_device();
        rms_.sync_to_device();
        rope_.sync_to_device();
    }

    /// The k / v rows the last pass appended, L_TOKENS x DK, token-major. The
    /// direct control for the RoPE tile: h2 can look right while k is wrong.
    std::vector<float> read_k(int base) { return _read_plane(0, base, eng_->t.L_TOKENS); }
    std::vector<float> read_v(int base) { return _read_plane(1, base, eng_->t.L_TOKENS); }

    /// The same, for a window the prefill path wrote: n is not L_TOKENS there.
    std::vector<float> read_k(int base, int n) { return _read_plane(0, base, n); }
    std::vector<float> read_v(int base, int n) { return _read_plane(1, base, n); }

    buffer<bf16>& cache() { return cache_; }

    /// This layer's staged projections, for a second bitstream to read.
    ///
    /// `stage_attn_proj_weights` lays them down through
    /// `reorder_cpy(..., vertical_blocks = 2)`, which is the same permutation
    /// `dequant_mm.xclbin`'s own host driver applies -- so
    /// `q38npu::prefill_proj` runs the prefill projections straight out of this
    /// BO at four byte offsets, with nothing restaged and nothing copied. BOs
    /// are device-scoped, so an app over a different xclbin may read it.
    /// See prefill/npu/prefill_proj.hpp for the argument and the invariants it
    /// rests on.
    buffer<u8>& weights() { return proj_; }

    int layer() const { return layer_; }

private:
    /// \param plane 0 = k, 1 = v. Row by row through k_at / v_at rather than a
    ///        flat run: the rows are contiguous within a plane, but going
    ///        through the accessors is what carries a layout change here.
    std::vector<float> _read_plane(int plane, int base, int n) {
        const attn_layout& t = eng_->t;
        cache_.sync_from_device();
        const size_t DKV = (size_t)t.DK;
        std::vector<float> out((size_t)n * DKV);
        for (int r = 0; r < n; r++) {
            const bf16* p = cache_.data()
                          + (plane ? t.v_at(base + r) : t.k_at(base + r));
            for (size_t i = 0; i < DKV; i++) out[(size_t)r * DKV + i] = (float)p[i];
        }
        return out;
    }

    attn_engine* eng_ = nullptr;
    buffer<u8>   proj_;
    buffer<bf16> rms_;
    buffer<bf16> rope_;
    buffer<bf16> cache_;
    int          layer_ = 0;
    int          seen_  = 0;
    std::array<int, qwen38::NUM_STATE_SLOTS> cursor_{};
};

} // namespace q38npu

#endif  // __QWEN3_8MTP_NPU_ATTN_LAYER_HPP__
