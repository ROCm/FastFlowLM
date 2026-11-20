// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file speculate/npu/mtp_layer.hpp
/// \brief The MTP draft head's fused layer on the NPU.
///
/// One object, not the engine/layer pair verify/npu/attn_layer.hpp uses: there
/// are 16 full-attention layers sharing one bitstream and exactly one MTP head,
/// so splitting the shared context out would buy nothing.
///
/// What it computes, per dispatch, is everything between the two halves of the
/// fusion input and h2:
///
///     fc_out = W_fc @ cat[rms(e, w_emb), rms(h, w_hid)]
///     h1     = fc_out + W_o @ (softmax(q K^T / sqrt(DH)) V * sigmoid(z))
///     h2     = h1 + W_down @ (silu(gate(yn)) * up(yn)),  yn = rms(h1, w_post)
///
/// `rms(h2, model.mtp.norm.weight)` and lm_head stay on the host -- they are
/// not in this xclbin, and the engine already owns a cpu_ref::lm_head.
#ifndef __QWEN3_8MTP_NPU_MTP_LAYER_HPP__
#define __QWEN3_8MTP_NPU_MTP_LAYER_HPP__

#include "common/cpu_simd.hpp"   // the vectorised fp32 <-> bf16 marshalling
#include "common/npu/decode_prof.hpp"   // DPROF_*, no-ops without DECODE_PROFILE=1
#include "common/rope.hpp"
#include "speculate/npu/mtp_sequence.hpp"

#include "npu_utils/npu_utils.hpp"
#include "buffer.hpp"
#include "typedef.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace q38npu {

// ==========================================================================
/// The draft head's decoder layer, executed on the NPU.
class npu_mtp_layer {
public:
    /// Allocates the five BOs and fills the three weight ones. ~250 MB and one
    /// reorder pass, paid at load; nothing moves per dispatch but x and the
    /// rotary pair.
    ///
    /// All five must come from this app -- a BO from a second app does not
    /// fault, the dispatch simply never completes.
    npu_mtp_layer(npu_app app, const mtp_layout& layout,
                  const cpu_ref::safetensors& q4nx,
                  const mtp_rtp_addresses& rtp = mtp_rtp_addresses{})
        : app_(std::move(app)), t_(layout), rtp_(rtp) {
        x_     = app_.create_bo_buffer<bf16>(t_.x_words);
        proj_  = app_.create_bo_buffer<u8>(t_.proj_bytes);
        rms_   = app_.create_bo_buffer<bf16>(t_.rms_words);
        rope_  = app_.create_bo_buffer<bf16>(t_.rope_words);
        cache_ = app_.create_bo_buffer<bf16>(t_.kv_cache_size);

        x_.memset((bf16)0);
        rope_.memset((bf16)0);
        cache_.memset((bf16)0);

        // bf16 is a class with a private raw_bits_; the staging functions emit
        // raw bit patterns, so these are copies of the representation.
        stage_mtp_proj_weights(proj_.data(), q4nx, t_);
        stage_mtp_rms_weights(reinterpret_cast<uint16_t*>(rms_.data()), q4nx, t_);
        stage_mtp_qk_norm(reinterpret_cast<uint16_t*>(rope_.data()) + t_.rope_w_offset,
                          q4nx, t_);

        x_.sync_to_device();
        proj_.sync_to_device();
        rms_.sync_to_device();
        rope_.sync_to_device();
        cache_.sync_to_device();

        gen_mtp_layer_seq(app_.seq(), t_, ctx_, rtp_);
    }

    // ---- the pass ----------------------------------------------------------
    /// One drafted token.
    ///
    /// \param embed    the token embedding, D wide, fp32 and **un-normalised**.
    ///                 The passthrough tile owns pre_fc_norm_embedding; norming
    ///                 on the host norms twice.
    /// \param hidden   the state that predicted this token, D wide, likewise raw.
    ///                 Step 0 takes the engine's post-`model.norm` hidden state;
    ///                 later steps take the previous step's post-`mtp.norm` row.
    /// \param position the token's ABSOLUTE rotary position. Not the cache row:
    ///                 the head's cache trails the base stack by one and the two
    ///                 are deliberately decoupled, exactly as in
    ///                 cpu_ref::mtp_head::step().
    /// \param h2_out   D floats, the layer output.
    void forward(const float* embed, const float* hidden, int position,
                 float* h2_out) {
        if (cache_.size() != t_.kv_cache_size)
            throw std::runtime_error("npu_mtp_layer: set_max_length() was not "
                                     "applied to the cache BO");
        if (ctx_ + 1 > t_.MAX_L)
            throw std::runtime_error("npu_mtp_layer: MTP KV cache overflow");

        DPROF_COUNT(mtp_steps);
        DPROF_V(mtp_seq_us, _set_context(ctx_));
        DPROF_B(host);
        _set_rope_position(position);

        // fp32 -> bf16, the lossy half of the boundary. Embedding first: the
        // array sees two packets on one channel and their arrival order IS the
        // concatenation.
        bf16* xd = x_.data();
        const size_t D = (size_t)t_.D;
        q38simd::f32_to_bf16(xd,     embed,  D);
        q38simd::f32_to_bf16(xd + D, hidden, D);
        x_.sync_to_device();
        DPROF_E(host, mtp_host_us);

        const ert_cmd_state st = DPROF_T(mtp_dispatch_us,
            app_(x_, proj_, rms_, rope_, cache_));
        if (st != ERT_CMD_STATE_COMPLETED)
            throw std::runtime_error(
                "npu_mtp_layer: dispatch state " + std::to_string((int)st) +
                " at cache row " + std::to_string(ctx_) + ", position " +
                std::to_string(position) +
                " -- a timeout is a sequence bug or a stale RTP address");

        DPROF_B(read);
        x_.sync_from_device();
        const bf16* out = x_.data() + t_.layer_out_offset;
        q38simd::bf16_to_f32(h2_out, out, D);
        DPROF_E(read, mtp_host_us);

        ctx_++;
    }

    // ---- the prefill seam ----------------------------------------------------
    /// The cache BO, for `q38npu::mtp_prefill_attn_op` to attend over.
    ///
    /// Handing it out is what makes step 0's prefill and steps 1-6's decode ONE
    /// cache with ONE cursor: the op reads rows [0, ctx_) that this object's own
    /// dispatches and `append_rows()` wrote, and writes nothing. BOs are
    /// device-scoped (`npu_app::create_bo_buffer` builds `buffer<T>(*device,
    /// size)`), so a BO allocated from this app is legal in attn.xclbin's.
    buffer<bf16>& cache() { return cache_; }

    /// The projection-weight BO, for `q38npu::mtp_proj` to matmul against.
    ///
    /// Same argument as `cache()`, and the same one `stage_mtp_proj_weights`
    /// already relies on: the 245 MB blob is laid out by `reorder_cpy` with
    /// `vertical_blocks = 2`, which is exactly what `dequant_mm` wants, and
    /// `reorder_cpy`'s permutation never crosses a 64-weight-row boundary, so
    /// per-tensor staging equals staging the concatenation. The five step-0
    /// projections are therefore five byte offsets into THIS buffer -- no second
    /// copy, and no second reorder pass at load.
    ///
    /// Handing it out read-only in practice: `dequant_mm` only ever streams it
    /// MM2S. Nothing here enforces that, so a caller that writes it corrupts
    /// steps 1-6 as well.
    buffer<u8>& weights() { return proj_; }

    /// Append `n` fp32 rows of k and v AT THE CURSOR and advance it.
    ///
    /// `mirror_from_cpu()` relative to the cursor rather than absolute. Step 0's
    /// hybrid path projects on the host, drops the rows here, and only then
    /// dispatches the attention core -- so by the time the core runs, the cache
    /// holds [0, ctx_ + n) exactly as `forward()` would have left it after n
    /// dispatches. Steps 1-6 then resume with no special case.
    ///
    /// The window's own base is the cursor BEFORE this call. Read `cache_len()`
    /// first: passing the post-append cursor to the op lets every query attend
    /// to its own future.
    void append_rows(const float* k, const float* v, int n) {
        const size_t DKV = (size_t)t_.DK;
        if (n < 0 || ctx_ + n > t_.MAX_L)
            throw std::runtime_error(
                "npu_mtp_layer::append_rows: " + std::to_string(n) +
                " rows at cursor " + std::to_string(ctx_) +
                " overflows a cache of MAX_L " + std::to_string(t_.MAX_L));
        if (n == 0) return;

        for (int r = 0; r < n; r++) {      // row by row: the source is planar
            bf16* kp = cache_.data() + t_.k_at(ctx_ + r);
            bf16* vp = cache_.data() + t_.v_at(ctx_ + r);
            q38simd::f32_to_bf16(kp, k + (size_t)r * DKV, DKV);
            q38simd::f32_to_bf16(vp, v + (size_t)r * DKV, DKV);
        }
        cache_.sync_to_device();
        _set_context(ctx_ + n);
    }

    // ---- cache bookkeeping --------------------------------------------------
    int  cache_len() const { return ctx_; }

    /// Cursor only, like cpu_ref::mtp_head::reset(). Rows above the cursor are
    /// streamed by the replay but masked out by the cores, so the stale context
    /// costs nothing -- it is the same state a rejected draft leaves behind.
    void reset() { _set_context(0); }

    /// Put the zeros of a fresh cache back in every row an earlier context wrote. reset() moves only the cursor, and the
    /// rows above it are streamed by the replay: what an earlier conversation left there changes which tokens the head
    /// drafts, and so which verify windows the base model runs -- an answer that depends on the conversation before it.
    void zero_stale() {
        const int to = std::min(hw_, (int)t_.MAX_L);
        hw_ = 0;
        if (to <= 0) return;
        cache_.sync_from_device();
        std::memset(cache_.data() + t_.k_at(0), 0, (size_t)to * (size_t)t_.DK * sizeof(bf16));
        std::memset(cache_.data() + t_.v_at(0), 0, (size_t)to * (size_t)t_.DV * sizeof(bf16));
        cache_.sync_to_device();
    }

    /// Crop the head's cache back to `len`.
    ///
    /// The head's cursor is NOT the base stack's: it trails by exactly one,
    /// permanently, and aligning them makes the head condition on a token the
    /// base model never emitted. Same contract as cpu_ref::mtp_head::crop().
    void crop(int len) { _set_context(std::min(ctx_, std::max(0, len))); }

    /// Grow the cache, keeping every byte it holds.
    ///
    /// **Plane by plane, not a flat prefix copy.** The two planes are laid out
    /// back to back, so growing moves the v plane's base: a flat copy would
    /// leave every v row at its old offset, which is now inside the k plane.
    /// That is silent -- v would read as whatever k happened to hold there --
    /// so the carry-over goes through v_at() on both layouts.
    ///
    /// Each plane is copied WHOLE, not just [0, ctx_): the replay streams up to
    /// the L_CHUNK boundary above the cursor, and those rows move the result at
    /// 1e-8.
    void set_max_length(int max_len) {
        if (max_len < ctx_)
            throw std::runtime_error("npu_mtp_layer::set_max_length(" +
                                     std::to_string(max_len) + ") is below the " +
                                     std::to_string(ctx_) + " rows already cached");
        // The cache's alignment, not the replay's: PREFILL_CHUNK is what
        // mtp_layout::build() requires, and it covers L_CHUNK. A request from
        // anywhere therefore lands on a legal buffer size.
        const int chunk = mtp_layout::PREFILL_CHUNK;
        max_len = (max_len + chunk - 1) / chunk * chunk;
        if (max_len <= t_.MAX_L) return;

        const mtp_layout old = t_;
        t_ = mtp_layout::build(max_len);
        gen_mtp_layer_seq(app_.seq(), t_, ctx_, rtp_);
        seq_ctx_ = ctx_;

        cache_.sync_from_device();
        std::vector<bf16> keep(cache_.data(), cache_.data() + cache_.size());

        cache_ = app_.create_bo_buffer<bf16>(t_.kv_cache_size);
        cache_.memset((bf16)0);
        const size_t k_words = old.plane_rows() * (size_t)old.DK;
        const size_t v_words = old.plane_rows() * (size_t)old.DV;
        std::memcpy(cache_.data() + t_.k_at(0), keep.data() + old.k_at(0),
                    k_words * sizeof(bf16));
        std::memcpy(cache_.data() + t_.v_at(0), keep.data() + old.v_at(0),
                    v_words * sizeof(bf16));
        cache_.sync_to_device();
    }

    const mtp_layout& layout() const { return t_; }

    // ---- testing -------------------------------------------------------------
    /// Seed the cache from cpu_ref::mtp_head's planar fp32 k / v, `n` rows.
    /// Real keys, correlated with the queries that will read them -- random ones
    /// put ~all the softmax mass on the appended row and make the replay
    /// untestable.
    void mirror_from_cpu(const std::vector<float>& k, const std::vector<float>& v,
                         int n) {
        const size_t DKV = (size_t)t_.DK;
        if (n < 0 || n > t_.MAX_L)
            throw std::runtime_error("npu_mtp_layer::mirror_from_cpu: n out of range");
        if (k.size() < (size_t)n * DKV || v.size() < (size_t)n * DKV)
            throw std::runtime_error("npu_mtp_layer::mirror_from_cpu: short cache");

        for (int r = 0; r < n; r++) {          // row by row: the source is planar
            bf16* kp = cache_.data() + t_.k_at(r);
            bf16* vp = cache_.data() + t_.v_at(r);
            for (size_t i = 0; i < DKV; i++) kp[i] = (bf16)k[(size_t)r * DKV + i];
            for (size_t i = 0; i < DKV; i++) vp[i] = (bf16)v[(size_t)r * DKV + i];
        }
        cache_.sync_to_device();
        _set_context(n);
    }

    /// Overwrite the weights with caller-built blobs, for drivers that want to
    /// inject rather than load. forward() never restages.
    void stage(const std::vector<uint8_t>& proj, const std::vector<uint16_t>& rms,
               const std::vector<uint16_t>& qk_norm) {
        if (proj.size() != proj_.size())
            throw std::runtime_error("npu_mtp_layer::stage: proj blob is " +
                                     std::to_string(proj.size()) + " B, BO is " +
                                     std::to_string(proj_.size()));
        std::memcpy(proj_.data(), proj.data(), proj.size());
        std::memcpy(reinterpret_cast<uint16_t*>(rms_.data()), rms.data(),
                    rms.size() * sizeof(uint16_t));
        std::memcpy(reinterpret_cast<uint16_t*>(rope_.data()) + t_.rope_w_offset,
                    qk_norm.data(), qk_norm.size() * sizeof(uint16_t));
        proj_.sync_to_device();
        rms_.sync_to_device();
        rope_.sync_to_device();
    }

    /// Cache row `row`'s k / v, DK wide. The direct control on the RoPE tile:
    /// h2 can look right while k is subtly wrong, because each query's own key
    /// dominates its softmax.
    std::vector<float> read_k(int row) { return _read_half(0, row); }
    std::vector<float> read_v(int row) { return _read_half(1, row); }

private:
    /// Regenerate the sequence only when the cursor actually moved. At T = 1
    /// that is every dispatch in a draft run, but reset()/crop() to a value the
    /// sequence already carries is free.
    void _set_context(int n) {
        ctx_ = n;
        hw_ = std::max(hw_, n + 8);   // a dispatch writes one row above the cursor, a catch-up window a few
        if (n == seq_ctx_) return;
        if (n < 0 || n >= t_.MAX_L)
            throw std::runtime_error("npu_mtp_layer: context " + std::to_string(n) +
                                     " outside a cache of MAX_L " +
                                     std::to_string(t_.MAX_L));
        gen_mtp_layer_seq(app_.seq(), t_, n, rtp_);
        seq_ctx_ = n;
    }

    /// The cos|sin pair for one absolute position, into the head of the rope BO.
    ///
    /// Built through cpu_ref::text_rope_tables() rather than open-coded, so the
    /// head consumes bit-for-bit what cpu_ref::mtp_head consumed -- the angles
    /// are the same formula either way, but the bf16 rounding is not free to
    /// disagree. A 64-wide HF row is 32 slots duplicated, so only the first
    /// half of each table crosses over.
    void _set_rope_position(int position) {
        if (position == rope_pos_) return;
        std::vector<float> cos, sin;
        cpu_ref::text_rope_tables(position, 1, cos, sin);

        const int half = t_.ROPE_DIM_ / 2;
        bf16* p = rope_.data();
        for (int j = 0; j < half; j++) {
            p[j]        = (bf16)cos[j];
            p[half + j] = (bf16)sin[j];
        }
        rope_.sync_to_device();
        rope_pos_ = position;
    }

    /// \param which 0 = k, 1 = v. One contiguous run either way: the two live in
    ///        separate planes, v a whole k plane in. Reading v back is therefore
    ///        the direct check that the append addressed the second plane.
    std::vector<float> _read_half(int which, int row) {
        cache_.sync_from_device();
        const size_t DKV = (size_t)t_.DK;
        std::vector<float> out(DKV);
        const bf16* p = cache_.data() + (which ? t_.v_at(row) : t_.k_at(row));
        for (size_t i = 0; i < DKV; i++) out[i] = (float)p[i];
        return out;
    }

    npu_app           app_;
    mtp_layout        t_;
    mtp_rtp_addresses rtp_;

    buffer<bf16> x_;       ///< arg 0: | e | h | layer_out |
    buffer<u8>   proj_;    ///< arg 1
    buffer<bf16> rms_;     ///< arg 2
    buffer<bf16> rope_;    ///< arg 3
    buffer<bf16> cache_;   ///< arg 4

    int hw_       = 0;     ///< highest row written since the cache was last zero (see zero_stale())
    int ctx_      = 0;     ///< rows the cache holds
    int seq_ctx_  = 0;     ///< what the emitted sequence was built for
    int rope_pos_ = -1;    ///< what the rope BO currently holds
};

} // namespace q38npu

#endif  // __QWEN3_8MTP_NPU_MTP_LAYER_HPP__
