// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file image/npu/vision_mm_op.hpp
/// \brief The vision tower's matmuls on the NPU: one dispatch per bound weight
///        per call, behind `mcv::vision::mm_op`.
///
/// Ported from Qwen3.8-27B's tower (`qwen3_8mtp_npu/image/npu/vision_mm_op.hpp`)
/// over the same two `vision_mm` bitstreams, unchanged. Read that file's
/// comments for the measurements; what is kept here is what still governs, plus
/// the one thing MiniCPM needs that Qwen3.8 never did.
///
/// ### K padding -- new for MiniCPM
///
/// Qwen3.8's `bind()` refused any K that was not a multiple of the k tile of
/// 384, because its checkpoint never had one. MiniCPM's patch embedding does:
/// a 14x14x3 Conv2d is K = 588. Rather than leave the 7k-row patch matmul on
/// the host, this op zero-extends K up to the tile (588 -> 768) in BOTH
/// operands: the weight BO's rows are laid out at stride K_dev with zero
/// columns, and `run()` writes each x row at stride K_dev with a zero tail. A
/// zero column times a zero row contributes nothing, so the result is exact.
///
/// The pre-round runs over the PADDED rows, because the hardware's 8-wide
/// groups start at the padded row: 588 = 73*8 + 4, so one group straddles four
/// real values and four zeros, and the zeros are on any grid. Pre-rounding the
/// unpadded row would put that group boundary somewhere the mmul does not.
///
/// ### Weight-resident, which is the whole point
///
/// The tower binds 115 weights at load -- patch embed, four per block x 27,
/// four in the vit_merger, two in the projector -- 1.1 GB. `bind()` uploads
/// once; `run()` streams activations only.
///
/// ### One app per op; shared staging per design
///
/// `npu_app::operator()` re-runs `_setup_kernel()` whenever the sequence version
/// moves, so ops sharing an app would rebuild the ELF on every dispatch. Each
/// op owns an app and emits its sequence once per M. The x/y staging BOs are
/// shared by every op of a design (a BO is device-scoped and bound per run), so
/// **two ops of one design must not be in flight at once** -- the tower drives
/// them strictly in sequence.
///
/// ### The projector's gelu
///
/// `vision_mm`'s `rtp_act` has no erf gelu. `merger.fc1` asks for one, so it
/// binds `RTP_NONE` and `run()` applies `vision::gelu_erf` on the host. It must
/// NOT bind `RTP_GELU`, which is the tanh form.
///
/// ### Why the tanh gelu is on the host too -- a deviation from Qwen3.8
///
/// Qwen3.8 fuses the block MLPs' gelu_tanh into the device (`rtp_act` 2,
/// `getGeluBf16_nonLUT`). Measured op by op on MiniCPM's reference inputs
/// (`tools/verify_minicpm_v_vision --backend npu`), that fused activation is
/// the dominant device error: every op WITHOUT an activation lands at ~2.7x the
/// bf16 host op's relL2 (the bfp16 mmul floor), while block0.fc1 lands at 11x
/// (0.0226) and vit_merger.fc1 -- 17280 wide -- at 19x (0.0355). Qwen3.8's own
/// N16 check 6 measured the same fused-gelu gap (0.0246) and accepted it; this
/// tower has one more and wider gelu, and its image_embeds came out at relL2
/// 0.106 against Qwen3.8's 0.040.
///
/// So the device computes the pre-activation (`RTP_NONE`) and the gelu is fused
/// into the bf16 -> fp32 widen every dispatch already does, vectorised
/// (`widen_gelu_tanh_par`). Same bitstream, no kernel change.
/// `FLM_MINICPM_V_VISION_DEVICE_GELU=1` puts it back on the device -- the A/B.
#ifndef __MINICPM_V_4_7_NPU_VISION_MM_OP_HPP__
#define __MINICPM_V_4_7_NPU_VISION_MM_OP_HPP__

#include "image/npu/vision_bfp16.hpp"
#include "image/npu/vision_bo.hpp"
#include "image/npu/vision_lane.hpp"
#include "image/npu/vision_mm_sequence.hpp"
#include "image/vision_ops.hpp"
#include "image/vision_par.hpp"
#include "image/vision_profile.hpp"

#include "image/vision_common.hpp"
#include "npu_utils/npu_utils.hpp"
#include "buffer.hpp"
#include "typedef.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace mcv_npu {

// ==========================================================================
/// One design's shared staging: the app manager every op of that design makes
/// its app from, the grow-only x and y BOs they all stream through, the design
/// params and the per-build RTP address. Held by the factory.
// ==========================================================================
class vision_mm_arena {
public:
    vision_mm_arena(npu_app_manager* mgr, const vision_mm_tiles::design_params& dp,
                    uint32_t rtp_address, dispatch_lane* lane = nullptr)
        : mgr_(mgr), dp_(dp), rtp_(rtp_address), lane_(lane) {
        if (!mgr_) throw std::runtime_error("vision_mm_arena: null app manager");
    }

    npu_app_manager* mgr() const { return mgr_; }
    const vision_mm_tiles::design_params& dp() const { return dp_; }
    uint32_t rtp() const { return rtp_; }

    /// Grow-only. A reallocation does not invalidate any op's sequence: the
    /// descriptors carry DDR offsets and BOs are bound per run.
    void ensure(size_t x_words, size_t y_words) {
        if (x_words > x_.size()) {
            x_ = mgr_->create_bo_buffer<bf16>(x_words);
            x_.memset((bf16)0);
            x_.sync_to_device();
        }
        if (y_words > y_.size()) {
            y_ = mgr_->create_bo_buffer<bf16>(y_words);
            y_.memset((bf16)0);
            y_.sync_to_device();
        }
    }

    buffer<bf16>& x() { return x_; }
    buffer<bf16>& y() { return y_; }

    /// The dispatch worker, or null when the tower is not allowed to stream.
    dispatch_lane* lane() const { return lane_; }

    /// Staging slots for the streamed path: slot s is an x and a y BO that one
    /// in-flight chunk owns. Grow-only; serial (called while nothing is queued).
    void ensure_slots(int n, size_t x_words, size_t y_words) {
        if ((int)sx_.size() < n) { sx_.resize((size_t)n); sy_.resize((size_t)n); }
        for (int i = 0; i < n; i++) {
            if (x_words > sx_[(size_t)i].size()) {
                sx_[(size_t)i] = mgr_->create_bo_buffer<bf16>(x_words);
                sx_[(size_t)i].memset((bf16)0);
                sx_[(size_t)i].sync_to_device();
            }
            if (y_words > sy_[(size_t)i].size()) {
                sy_[(size_t)i] = mgr_->create_bo_buffer<bf16>(y_words);
                sy_[(size_t)i].memset((bf16)0);
                sy_[(size_t)i].sync_to_device();
            }
        }
    }
    buffer<bf16>& sx(int slot) { return sx_.at((size_t)slot); }
    buffer<bf16>& sy(int slot) { return sy_.at((size_t)slot); }

private:
    npu_app_manager*               mgr_ = nullptr;
    vision_mm_tiles::design_params dp_{0, 0, 0};
    uint32_t                       rtp_ = 0;
    buffer<bf16>                   x_, y_;
    dispatch_lane*                 lane_ = nullptr;
    std::vector<buffer<bf16>>      sx_, sy_;
};

/// The bfp16 pre-round of both operands (see vision_bfp16.hpp). On by default;
/// `FLM_MINICPM_V_VISION_MM_PREROUND=0` is the A/B arm, not a production knob:
/// on Qwen3.8 it was an 18x accuracy difference on patch_embed.
/// Whether gelu_tanh runs fused on the device (Qwen3.8's choice) rather than in
/// the host widen. Off by default; see the file comment.
inline bool device_gelu_enabled() {
    static const bool on = [] {
        const char* e = std::getenv("FLM_MINICPM_V_VISION_DEVICE_GELU");
        return e && e[0] == '1';
    }();
    return on;
}

/// FLM_MINICPM_V_VISION_MM_C=0 keeps the matmuls on designs A/B even when vision_mm_c.xclbin is present.
inline bool mm_c_disabled() {
    static const bool off = [] {
        const char* e = std::getenv("FLM_MINICPM_V_VISION_MM_C");
        return e && e[0] == '0';
    }();
    return off;
}

/// FLM_MINICPM_V_VISION_MM_P=0 keeps the matmuls off design P even when vision_mm_p.xclbin is present.
inline bool mm_p_disabled() {
    static const bool off = [] {
        const char* e = std::getenv("FLM_MINICPM_V_VISION_MM_P");
        return e && e[0] == '0';
    }();
    return off;
}

inline bool preround_enabled() {
    static const bool on = [] {
        const char* e = std::getenv("FLM_MINICPM_V_VISION_MM_PREROUND");
        return !e || e[0] != '0';
    }();
    return on;
}

// ==========================================================================
/// \brief y[M][N] = act(x[M][K] W[N][K]^T + bias[N]) on `vision_mm`.
///
/// ### Design C (optional)
///
/// With `vision_mm_c.xclbin` loaded, an op whose N is a multiple of 384 can run on design C
/// (512 x 384 array passes, columns split M) instead of A (256 x 384). C is close to A but not
/// bit-identical (measured: a 120-frame tower all on A differs from the same tower on C by up to
/// 0.3, cosine vs the fp32 tower 0.9891 on A, 0.9913 on C) and is 1.2-1.6x faster from 512 rows up, but its M quantum is
/// 512, so a 256-row op would pay for 512. The design is chosen per call from M (`_use_c`) and,
/// for a streamed stretch, once for the whole stretch at `stream_reserve` so that every chunk of
/// every op of the stretch sits on one design: alternating designs would pay a ~2.4 ms hw-context
/// switch per dispatch. The weight BO is shared by A and C (same layout).
///
/// ### Design P (optional, supersedes C)
///
/// With `vision_mm_p.xclbin` loaded, an op whose N is a multiple of 384 runs on P for every M:
/// C's array mapping with x and w handed over already packed as bfp16 (vision_bfp16.hpp), so the
/// cores skip the per-tile conversion C runs -- 16-28% less device time per op at the tower's
/// shapes, C's bits (the pack is the core's own conversion of the pre-rounded values). Such an
/// op keeps only the packed weight, so it never falls back to A or C, and a 256-row call pays for
/// 512. Ops C would have left on A (small M) therefore get C's rounding: a frame's output no longer
/// depends on how many frames share the call, and at 1 frame it is as close to fp32 as before. The host's narrow + pre-round becomes a narrow + pre-round + pack, one pass.
// ==========================================================================
class npu_vision_mm_op final : public mcv::vision::mm_op {
public:
    npu_vision_mm_op(vision_mm_arena* a, vision_mm_arena* b, std::string tag,
                     vision_mm_arena* c = nullptr, vision_mm_arena* p = nullptr)
        : arena_a_(a), arena_b_(b), arena_c_(c), arena_p_(p), tag_(std::move(tag)) {
        if (!arena_a_) throw std::runtime_error("npu_vision_mm_op: null design-A arena");
    }

    /// Upload one weight for the lifetime of the op: N bias elements, then the
    /// [N][K_dev] weight row-major with zero columns past K. A null bias is N
    /// zeros -- the kernel's prologue reads it unconditionally.
    void bind(const uint16_t* w, const uint16_t* bias,
              int K, int N, mcv::vision::activation act) override {
        if (!w) throw std::runtime_error("npu_vision_mm_op::bind(" + tag_ + ") got a null weight");
        if (K <= 0 || N <= 0)
            throw std::runtime_error("npu_vision_mm_op::bind(" + tag_ + "): K=" +
                                     std::to_string(K) + " N=" + std::to_string(N));

        arena_ = _pick(N);
        K_ = K; N_ = N; act_ = act;
        K_dev_ = (int)_round_up((uint32_t)K, arena_->dp().k);

        host_epilogue_ = act == mcv::vision::activation::gelu_erf ||
                         (act == mcv::vision::activation::gelu && !device_gelu_enabled());
        rtp_act_ = host_epilogue_ ? vision_mm_tiles::RTP_NONE : (int32_t)act;

#ifdef MINICPM_V_CPU_SIMD
        // Design P: the packed operands only exist pre-rounded (unrounded, the
        // host's pack would not be the core's conversion, see vision_bfp16.hpp).
        packed_ = arena_p_ && !mm_p_disabled() && preround_enabled() &&
                  N % (int)arena_p_->dp().tile_N() == 0;
        if (packed_) {
            arena_ = arena_p_;
            const size_t bytes = (size_t)N * 2 + bfp_packed_bytes((size_t)N, (size_t)K_dev_);
            w_bo_ = arena_->mgr()->create_bo_buffer<bf16>(bytes / 2);
            uint8_t* d = reinterpret_cast<uint8_t*>(w_bo_.data());
            if (bias) std::memcpy(d, bias, (size_t)N * sizeof(uint16_t));
            else      std::memset(d, 0,    (size_t)N * sizeof(uint16_t));
            pack_bfp16_from_bf16_raw(d + (size_t)N * 2, w, N, N, K, K_dev_, (int)arena_->dp().n);
            w_bo_.sync_to_device();
            tgt_[0].app  = arena_->mgr()->create_app();
            tgt_[0].have = false;
            tgt_[1].have = false;
            bound_       = true;
            return;
        }
#endif

        const size_t w_words = (size_t)N + (size_t)N * (size_t)K_dev_;
        w_bo_ = arena_->mgr()->create_bo_buffer<bf16>(w_words);
        uint16_t* d = reinterpret_cast<uint16_t*>(w_bo_.data());
        if (bias) std::memcpy(d, bias, (size_t)N * sizeof(uint16_t));
        else      std::memset(d, 0,    (size_t)N * sizeof(uint16_t));
        if (K_dev_ == K) {
            std::memcpy(d + N, w, (size_t)N * (size_t)K * sizeof(uint16_t));
        } else {
            for (int n = 0; n < N; n++) {
                uint16_t* row = d + N + (size_t)n * K_dev_;
                std::memcpy(row, w + (size_t)n * K, (size_t)K * sizeof(uint16_t));
                std::memset(row + K, 0, (size_t)(K_dev_ - K) * sizeof(uint16_t));
            }
        }
        // Rows are K_dev contiguous, a multiple of 384, so no 8-group straddles
        // two rows. The bias seeds the accumulator and is not pre-rounded.
        if (preround_enabled())
            preround_bfp16_inplace(d + N, (size_t)N * (size_t)K_dev_);
        w_bo_.sync_to_device();

        tgt_[0].app  = arena_->mgr()->create_app();
        tgt_[0].have = false;
        // design C's app (a hw_context of vision_mm_c) is made on first use
        tgt_[1].have = false;
        bound_    = true;
    }

    /// Run M rows. One dispatch, whatever M is; M is padded to the design's tile_M here.
    void run(const float* x, int M, float* y) override {
        if (!bound_) throw std::runtime_error("npu_vision_mm_op::run(" + tag_ + ") before bind()");
        if (M <= 0) return;
        VTRACE(tag_.c_str());

        target& t = _target(_use_c(M));
        vision_mm_arena* ar = t.arena;
        const vision_mm_tiles::design_params& dp = ar->dp();
        const uint32_t M_pad = _round_up((uint32_t)M, dp.tile_M());
        const size_t   Kd    = (size_t)K_dev_;
        ar->ensure(_x_words(M_pad), (size_t)M_pad * (size_t)N_);

#ifdef MINICPM_V_CPU_SIMD
        if (packed_) {
            // the pack writes all M_pad rows, the padding ones as zero blocks
            { VPROF(mm_x_us);
              pack_bfp16_from_f32(reinterpret_cast<uint8_t*>(ar->x().data()), x, M, (int)M_pad,
                                  K_, K_dev_, (int)dp.m); }
        } else
#endif
        {
        // x: the M real rows, and ROWS [M, M_pad) ZEROED ON EVERY CALL -- there
        // is no mask in this design, so a previous longer call's rows would be
        // multiplied (memory: kv-rows-above-cursor-are-not-inert).
        bf16* xd = ar->x().data();
        {
            VPROF(mm_x_us);
            if (K_dev_ == K_) {
                const size_t real = (size_t)M * Kd;
                if (preround_enabled()) narrow_preround_bfp16(xd, x, real);
                else                    narrow_bf16_par(xd, x, real);
            } else {
                // Padded K: narrow row by row into stride K_dev with a zero
                // tail, then pre-round the padded rows (see the file comment).
                const int K = K_;
#pragma omp parallel for schedule(static) num_threads(mcv::vision::par_threads())
                for (int i = 0; i < M; i++) {
                    bf16* row = xd + (size_t)i * Kd;
                    mcv::simd::f32_to_bf16(row, x + (size_t)i * K, (size_t)K);
                    std::memset(row + K, 0, (Kd - (size_t)K) * sizeof(bf16));
                }
                if (preround_enabled())
                    preround_bfp16_inplace(reinterpret_cast<uint16_t*>(xd), (size_t)M * Kd);
            }
            const size_t tail = ((size_t)M_pad - (size_t)M) * Kd;
            if (tail) std::memset(xd + (size_t)M * Kd, 0, tail * sizeof(bf16));
        }
        }

        // Only the rows this dispatch reads: the arena is sized for the widest
        // op of the largest image (vision_bo.hpp).
        { VPROF(mm_sync_us); sync_to_device_n(ar->x(), _x_words(M_pad)); }

        if (!t.have || M_pad != t.seq_M_pad) _gen_seq(t, M_pad);

        VPROF_COUNT(mm_calls);
        const ert_cmd_state st = VPROF_EXPR(mm_dev_us, t.app(ar->y(), ar->x(), w_bo_));
        if (st != ERT_CMD_STATE_COMPLETED) {
            t.have = false;
            throw std::runtime_error(
                "npu_vision_mm_op::run(" + tag_ + "): dispatch state " +
                std::to_string((int)st) + " at M " + std::to_string(M) + " (padded " +
                std::to_string(M_pad) + ", K " + std::to_string(K_dev_) + ", N " +
                std::to_string(N_) + ") -- a timeout is a sequence bug or a stale RTP "
                "address (diff vision_mm_rtp_addresses against "
                "FLM_Xclbin/Qwen3_8/vision_mm/build/<design>/rtp_addr.txt)");
        }

        { VPROF(mm_sync_us); sync_from_device_n(ar->y(), (size_t)M * (size_t)N_); }
        VPROF(mm_y_us);

        const bf16*  yd    = ar->y().data();
        const size_t n_out = (size_t)M * (size_t)N_;
        if (!host_epilogue_) {
            widen_f32_par(y, yd, n_out);
            return;
        }
        if (act_ == mcv::vision::activation::gelu) {
            // Not rounded back to bf16: the only consumer is the next op's
            // narrow, which rounds it identically.
            widen_gelu_tanh_par(y, yd, n_out);
            return;
        }
        // The projector's erf gelu, 2M elements an image: scalar is fine.
        // Rounding back to bf16 under emulation keeps this arm scoreable
        // against cpu_mm_op on the same bytes.
        const bool emu = mcv::vision::cpu_ops_emulate_bf16();
#pragma omp parallel for schedule(static) num_threads(mcv::vision::par_threads())
        for (int i = 0; i < M; i++) {
            const size_t base = (size_t)i * (size_t)N_;
            for (int j = 0; j < N_; j++) {
                const float v = mcv::vision::apply_activation((float)yd[base + (size_t)j], act_);
                y[base + (size_t)j] = emu ? mcv::ref::round_bf16(v) : v;
            }
        }
    }

    /// Build the sequence and its ELF for M rows now. Touches only this op's
    /// own app, so the tower runs it for all 115 ops concurrently.
    void prepare(int M) override {
        if (!bound_ || M <= 0) return;
        target& t = _target(_use_c(M));
        const uint32_t M_pad = _round_up((uint32_t)M, t.arena->dp().tile_M());
        if (t.have && M_pad == t.seq_M_pad) return;
        _gen_seq(t, M_pad);
        t.app.update_ctrl_seq();
    }

    // ---- streaming interface (see mm_op) --------------------------------------

    bool streamable() const override {
#if defined(FLM_USE_HRX)
        return false;   // HRX has no ranged BO sync and its thread story is unmeasured
#else
        // The streamed stages assume the production arms: host gelu epilogue,
        // pre-round on, no K padding. Any A/B knob turned the other way falls
        // back to run(), which handles all of them.
        return bound_ && arena_->lane() && K_dev_ == K_ && preround_enabled() &&
               !device_gelu_enabled() && act_ != mcv::vision::activation::gelu_erf;
#endif
    }
    int stream_tile_m() const override { return (int)arena_->dp().tile_M(); }

    /// The dispatch tile for a stretch of `rows` rows: design C's when the stretch will run on it.
    int stream_tile_m_for(int rows) const override {
        return (int)(_use_c(rows) ? arena_c_->dp().tile_M() : arena_->dp().tile_M());
    }

    void stream_reserve(int chunk_rows, int slots) override {
        // one design for every chunk of the stretch (see the class comment)
        stream_c_ = _use_c(chunk_rows);
        vision_mm_arena* ar = stream_c_ ? arena_c_ : arena_;
        stream_M_pad_ = _round_up((uint32_t)chunk_rows, ar->dp().tile_M());
        ar->ensure_slots(slots, _x_words(stream_M_pad_), (size_t)stream_M_pad_ * (size_t)N_);
    }

    void stream_load_f32(int slot, const float* x, int rows) override {
        VPROF(mm_x_us);
        bf16* xd = _sarena()->sx(slot).data();
#ifdef MINICPM_V_CPU_SIMD
        if (packed_) {
            pack_bfp16_from_f32(reinterpret_cast<uint8_t*>(xd), x, rows, (int)stream_M_pad_, K_, K_dev_,
                                (int)arena_->dp().m);
            return;
        }
#endif
        narrow_preround_bfp16(xd, x, (size_t)rows * (size_t)K_dev_);
    }

    void stream_load_bf16(int slot, const uint16_t* x, int rows) override {
        VPROF(mm_x_us);
#ifdef MINICPM_V_CPU_SIMD
        if (packed_) {
            pack_bfp16_from_bf16(reinterpret_cast<uint8_t*>(_sarena()->sx(slot).data()), x, rows,
                                 (int)stream_M_pad_, K_dev_, (int)arena_->dp().m);
            return;
        }
#endif
        preround_from_bf16(reinterpret_cast<uint16_t*>(_sarena()->sx(slot).data()), x,
                           (size_t)rows * (size_t)K_dev_);
    }

    std::future<void> stream_submit(int slot, int rows) override {
        if (stream_M_pad_ == 0 || (uint32_t)rows > stream_M_pad_)
            throw std::runtime_error("npu_vision_mm_op::stream_submit(" + tag_ + "): rows " +
                                     std::to_string(rows) + " exceed the reserved chunk of " +
                                     std::to_string(stream_M_pad_));
        _stream_flush_x(slot, rows);
        return arena_->lane()->submit([this, slot, rows] { _stream_job(slot, rows); });
    }

    void stream_store_add(int slot, float* acc, int rows) override {
        _stream_sync_y(slot, rows);
        VPROF(mm_y_us);
        add_widen_bf16(acc, _sarena()->sy(slot).data(), (size_t)rows * (size_t)N_);
    }

    void stream_store_bf16(int slot, uint16_t* dst, size_t ld, int rows) override {
        _stream_sync_y(slot, rows);
        VPROF(mm_y_us);
        copy_flush_bf16_rows(dst, ld, reinterpret_cast<const uint16_t*>(_sarena()->sy(slot).data()),
                             (size_t)N_, (size_t)N_, rows);
    }

    void stream_store_f32(int slot, float* dst, int rows) override {
        if (act_ != mcv::vision::activation::none)
            throw std::runtime_error("npu_vision_mm_op::stream_store_f32(" + tag_ + "): op has an epilogue");
        _stream_sync_y(slot, rows);
        VPROF(mm_y_us);
        widen_f32_par(dst, _sarena()->sy(slot).data(), (size_t)rows * (size_t)N_);
    }

    void stream_store_gelu_to(int slot, mcv::vision::mm_op& next, int rows) override {
        auto* nx = dynamic_cast<npu_vision_mm_op*>(&next);
        if (!nx || nx->_sarena() != _sarena() || nx->K_dev_ != N_ || nx->K_ != N_ ||
            nx->packed_ != packed_ || nx->stream_M_pad_ != stream_M_pad_ ||
            act_ != mcv::vision::activation::gelu || !host_epilogue_)
            throw std::runtime_error("npu_vision_mm_op::stream_store_gelu_to(" + tag_ +
                                     "): the next op does not take this output as its x");
        _stream_sync_y(slot, rows);
        VPROF(mm_y_us);
#ifdef MINICPM_V_CPU_SIMD
        if (packed_) {
            pack_bfp16_gelu(reinterpret_cast<uint8_t*>(_sarena()->sx(slot).data()), _sarena()->sy(slot).data(),
                            rows, (int)stream_M_pad_, N_, (int)arena_->dp().m);
            return;
        }
#endif
        gelu_preround_to(_sarena()->sx(slot).data(), _sarena()->sy(slot).data(),
                         (size_t)rows * (size_t)N_);
    }

    int K() const override { return K_; }
    int N() const override { return N_; }
    const char* backend() const override { return "npu"; }

    bool on_design_b() const { return arena_ && arena_ == arena_b_; }
    bool on_design_p() const { return packed_; }
    bool host_epilogue() const { return host_epilogue_; }
    int  K_dev() const { return K_dev_; }

private:
    static uint32_t _round_up(uint32_t v, uint32_t q) { return (v + q - 1) / q * q; }

    /// x staging words (bf16-sized) for M_pad rows: packed bytes / 2 on P.
    size_t _x_words(uint32_t M_pad) const {
        return packed_ ? bfp_packed_bytes(M_pad, (size_t)K_dev_) / 2 : (size_t)M_pad * (size_t)K_dev_;
    }

    /// One design this op can dispatch on: its arena (the x/y staging), its app (hw_context) and the
    /// sequence the app currently holds.
    struct target {
        vision_mm_arena* arena = nullptr;
        npu_app          app;
        bool             have = false;
        uint32_t         seq_M_pad = 0;
    };

    /// 0 = the op's A/B design, 1 = design C.
    target& _target(bool use_c) {
        if (!use_c) { tgt_[0].arena = arena_; return tgt_[0]; }
        if (!tgt_[1].arena) {
            tgt_[1].arena = arena_c_;
            tgt_[1].app   = arena_c_->mgr()->create_app();
        }
        return tgt_[1];
    }
    vision_mm_arena* _sarena() const { return stream_c_ ? arena_c_ : arena_; }

    /// Design C is on when its xclbin is loaded, FLM_MINICPM_V_VISION_MM_C is not 0, this op's N fits its 384
    /// tile and it is on A (never B), and the 512-row quantum wastes at most a third more rows than A's
    /// 256 would (M = 256 would pay for 512: measured 20% slower there; from 384 rows up it is equal or better).
    bool _use_c(int M) const {
        if (!arena_c_ || arena_ != arena_a_ || mm_c_disabled()) return false;
        if (N_ % (int)arena_c_->dp().tile_N() != 0) return false;
        const uint32_t pa = _round_up((uint32_t)M, arena_a_->dp().tile_M());
        const uint32_t pc = _round_up((uint32_t)M, arena_c_->dp().tile_M());
        return pc * 100 <= pa * 134;
    }

    /// Before a chunk is queued, on the calling thread: zero the rows past `rows` (no mask in this design, as in run())
    /// and flush x. Done here, not in the job, so the lane's thread goes straight from one dispatch to the next: a flush
    /// (or the y invalidate below) on that thread left the device idle for it, ~0.5 ms a dispatch.
    void _stream_flush_x(int slot, int rows) {
        vision_mm_arena* ar = _sarena();
        const uint32_t M_pad = stream_M_pad_;
        const size_t   Kd    = (size_t)K_dev_;
        bf16* xd = ar->sx(slot).data();
        VPROF(mm_sync_us);
        // a packed load already wrote the padding rows as zero blocks
        const size_t tail = packed_ ? 0 : ((size_t)M_pad - (size_t)rows) * Kd;
        if (tail) std::memset(xd + (size_t)rows * Kd, 0, tail * sizeof(bf16));
        sync_to_device_n(ar->sx(slot), _x_words(M_pad));
    }

    /// After the chunk's future is ready, on the calling thread: invalidate the y rows before a store reads them.
    void _stream_sync_y(int slot, int rows) {
        VPROF(mm_sync_us);
        sync_from_device_n(_sarena()->sy(slot), (size_t)rows * (size_t)N_);
    }

    /// The dispatch alone, on the lane's thread (x already flushed by stream_submit, y invalidated by the store).
    void _stream_job(int slot, int rows) {
        target& t = _target(stream_c_);
        vision_mm_arena* ar = t.arena;
        const uint32_t M_pad = stream_M_pad_;
        if (!t.have || M_pad != t.seq_M_pad) {   // normally done by prepare()
            _gen_seq(t, M_pad);
            t.app.update_ctrl_seq();
        }
        VPROF_COUNT(mm_calls);
        const ert_cmd_state st = VPROF_EXPR(mm_dev_us, t.app(ar->sy(slot), ar->sx(slot), w_bo_));
        if (st != ERT_CMD_STATE_COMPLETED) {
            t.have = false;
            throw std::runtime_error(
                "npu_vision_mm_op::stream(" + tag_ + "): dispatch state " +
                std::to_string((int)st) + " at M " + std::to_string(rows) + " (padded " +
                std::to_string(M_pad) + ", K " + std::to_string(K_dev_) + ", N " +
                std::to_string(N_) + ")");
        }
    }

    void _gen_seq(target& t, uint32_t M_pad) {
        const vision_mm_arena* ar = t.arena;
        if (ar->dp().packed)
            gen_vision_mm_p_seq(t.app.seq(), M_pad, (uint32_t)K_dev_, (uint32_t)N_, ar->dp(), rtp_act_, ar->rtp());
        else
            gen_vision_mm_seq(t.app.seq(), M_pad, (uint32_t)K_dev_, (uint32_t)N_, ar->dp(),
                              /*weight_offset=*/0, rtp_act_, ar->rtp());
        t.seq_M_pad = M_pad;
        t.have      = true;
    }

    /// A when N is a multiple of its 384 tile_N (every shape but the projector's
    /// fc2), else B when N is a multiple of 1024. K never decides: all designs
    /// share k = 384 and K is zero-padded up to it. C is an add-on to A (`_use_c`).
    vision_mm_arena* _pick(int N) const {
        if (arena_a_ && N % (int)arena_a_->dp().tile_N() == 0) return arena_a_;
        if (arena_b_ && N % (int)arena_b_->dp().tile_N() == 0) return arena_b_;
        throw std::runtime_error(
            "npu_vision_mm_op::bind(" + tag_ + "): N=" + std::to_string(N) +
            " fits neither vision_mm design (A needs N%384==0, B N%1024==0" +
            std::string(arena_b_ ? "" : "; B is not loaded") + ")");
    }

    vision_mm_arena*          arena_a_ = nullptr;
    vision_mm_arena*          arena_b_ = nullptr;
    vision_mm_arena*          arena_c_ = nullptr;
    vision_mm_arena*          arena_p_ = nullptr;
    vision_mm_arena*          arena_   = nullptr;   // the primary design of this op (A or B)
    std::string               tag_;
    target                    tgt_[2];
    buffer<bf16>              w_bo_;
    int                       K_ = 0, K_dev_ = 0, N_ = 0;
    mcv::vision::activation   act_ = mcv::vision::activation::none;
    int32_t                   rtp_act_ = vision_mm_tiles::RTP_NONE;
    bool                      host_epilogue_ = false;
    bool                      bound_         = false;
    uint32_t                  stream_M_pad_  = 0;
    bool                      stream_c_      = false;
    bool                      packed_        = false;   // on design P (arena_ == arena_p_)
};

} // namespace mcv_npu

#endif  // __MINICPM_V_4_7_NPU_VISION_MM_OP_HPP__
