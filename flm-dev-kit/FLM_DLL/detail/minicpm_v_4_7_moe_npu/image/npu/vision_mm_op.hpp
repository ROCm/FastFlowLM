// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file image/npu/vision_mm_op.hpp
/// \brief The vision tower's matmuls on the NPU: one dispatch per bound weight
///        per call, behind `mcvm::vision::mm_op`.
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
/// (`tools/verify_minicpm_v_moe_vision --backend npu`), that fused activation is
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

namespace mcvm_npu {

// ==========================================================================
/// One design's shared staging: the app manager every op of that design makes
/// its app from, the grow-only x and y BOs they all stream through, the design
/// params and the per-build RTP address. Held by the factory.
// ==========================================================================
class vision_mm_arena {
public:
    vision_mm_arena(npu_app_manager* mgr, const vision_mm_tiles::design_params& dp,
                    uint32_t rtp_address)
        : mgr_(mgr), dp_(dp), rtp_(rtp_address) {
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

private:
    npu_app_manager*               mgr_ = nullptr;
    vision_mm_tiles::design_params dp_{0, 0, 0};
    uint32_t                       rtp_ = 0;
    buffer<bf16>                   x_, y_;
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

inline bool preround_enabled() {
    static const bool on = [] {
        const char* e = std::getenv("FLM_MINICPM_V_VISION_MM_PREROUND");
        return !e || e[0] != '0';
    }();
    return on;
}

// ==========================================================================
/// \brief y[M][N] = act(x[M][K] W[N][K]^T + bias[N]) on `vision_mm`.
// ==========================================================================
class npu_vision_mm_op final : public mcvm::vision::mm_op {
public:
    npu_vision_mm_op(vision_mm_arena* a, vision_mm_arena* b, std::string tag)
        : arena_a_(a), arena_b_(b), tag_(std::move(tag)) {
        if (!arena_a_) throw std::runtime_error("npu_vision_mm_op: null design-A arena");
    }

    /// Upload one weight for the lifetime of the op: N bias elements, then the
    /// [N][K_dev] weight row-major with zero columns past K. A null bias is N
    /// zeros -- the kernel's prologue reads it unconditionally.
    void bind(const uint16_t* w, const uint16_t* bias,
              int K, int N, mcvm::vision::activation act) override {
        if (!w) throw std::runtime_error("npu_vision_mm_op::bind(" + tag_ + ") got a null weight");
        if (K <= 0 || N <= 0)
            throw std::runtime_error("npu_vision_mm_op::bind(" + tag_ + "): K=" +
                                     std::to_string(K) + " N=" + std::to_string(N));

        arena_ = _pick(N);
        K_ = K; N_ = N; act_ = act;
        K_dev_ = (int)_round_up((uint32_t)K, arena_->dp().k);

        host_epilogue_ = act == mcvm::vision::activation::gelu_erf ||
                         (act == mcvm::vision::activation::gelu && !device_gelu_enabled());
        rtp_act_ = host_epilogue_ ? vision_mm_tiles::RTP_NONE : (int32_t)act;

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

        app_      = arena_->mgr()->create_app();
        have_seq_ = false;
        bound_    = true;
    }

    /// Run M rows. One dispatch, whatever M is; M is padded to tile_M here.
    void run(const float* x, int M, float* y) override {
        if (!bound_) throw std::runtime_error("npu_vision_mm_op::run(" + tag_ + ") before bind()");
        if (M <= 0) return;

        const vision_mm_tiles::design_params& dp = arena_->dp();
        const uint32_t M_pad = _round_up((uint32_t)M, dp.tile_M());
        const size_t   Kd    = (size_t)K_dev_;
        arena_->ensure((size_t)M_pad * Kd, (size_t)M_pad * (size_t)N_);

        // x: the M real rows, and ROWS [M, M_pad) ZEROED ON EVERY CALL -- there
        // is no mask in this design, so a previous longer call's rows would be
        // multiplied.
        bf16* xd = arena_->x().data();
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
#pragma omp parallel for schedule(static) num_threads(mcvm::vision::par_threads())
                for (int i = 0; i < M; i++) {
                    bf16* row = xd + (size_t)i * Kd;
                    mcvm::simd::f32_to_bf16(row, x + (size_t)i * K, (size_t)K);
                    std::memset(row + K, 0, (Kd - (size_t)K) * sizeof(bf16));
                }
                if (preround_enabled())
                    preround_bfp16_inplace(reinterpret_cast<uint16_t*>(xd), (size_t)M * Kd);
            }
            const size_t tail = ((size_t)M_pad - (size_t)M) * Kd;
            if (tail) std::memset(xd + (size_t)M * Kd, 0, tail * sizeof(bf16));
        }

        // Only the rows this dispatch reads: the arena is sized for the widest
        // op of the largest image (vision_bo.hpp).
        { VPROF(mm_sync_us); sync_to_device_n(arena_->x(), (size_t)M_pad * Kd); }

        if (!have_seq_ || M_pad != seq_M_pad_) _gen_seq(M_pad);

        VPROF_COUNT(mm_calls);
        const ert_cmd_state st = VPROF_EXPR(mm_dev_us, app_(arena_->y(), arena_->x(), w_bo_));
        if (st != ERT_CMD_STATE_COMPLETED) {
            have_seq_ = false;
            throw std::runtime_error(
                "npu_vision_mm_op::run(" + tag_ + "): dispatch state " +
                std::to_string((int)st) + " at M " + std::to_string(M) + " (padded " +
                std::to_string(M_pad) + ", K " + std::to_string(K_dev_) + ", N " +
                std::to_string(N_) + ") -- a timeout is a sequence bug or a stale RTP "
                "address (diff vision_mm_rtp_addresses against "
                "FLM_Xclbin/Qwen3_8/vision_mm/build/<design>/rtp_addr.txt)");
        }

        { VPROF(mm_sync_us); sync_from_device_n(arena_->y(), (size_t)M * (size_t)N_); }
        VPROF(mm_y_us);

        const bf16*  yd    = arena_->y().data();
        const size_t n_out = (size_t)M * (size_t)N_;
        if (!host_epilogue_) {
            widen_f32_par(y, yd, n_out);
            return;
        }
        if (act_ == mcvm::vision::activation::gelu) {
            // Not rounded back to bf16: the only consumer is the next op's
            // narrow, which rounds it identically.
            widen_gelu_tanh_par(y, yd, n_out);
            return;
        }
        // The projector's erf gelu, 2M elements an image: scalar is fine.
        // Rounding back to bf16 under emulation keeps this arm scoreable
        // against cpu_mm_op on the same bytes.
        const bool emu = mcvm::vision::cpu_ops_emulate_bf16();
#pragma omp parallel for schedule(static) num_threads(mcvm::vision::par_threads())
        for (int i = 0; i < M; i++) {
            const size_t base = (size_t)i * (size_t)N_;
            for (int j = 0; j < N_; j++) {
                const float v = mcvm::vision::apply_activation((float)yd[base + (size_t)j], act_);
                y[base + (size_t)j] = emu ? mcvm::ref::round_bf16(v) : v;
            }
        }
    }

    /// Build the sequence and its ELF for M rows now. Touches only this op's
    /// own app, so the tower runs it for all 115 ops concurrently.
    void prepare(int M) override {
        if (!bound_ || M <= 0) return;
        const uint32_t M_pad = _round_up((uint32_t)M, arena_->dp().tile_M());
        if (have_seq_ && M_pad == seq_M_pad_) return;
        _gen_seq(M_pad);
        app_.update_ctrl_seq();
    }

    int K() const override { return K_; }
    int N() const override { return N_; }
    const char* backend() const override { return "npu"; }

    bool on_design_b() const { return arena_ && arena_ == arena_b_; }
    bool host_epilogue() const { return host_epilogue_; }
    int  K_dev() const { return K_dev_; }

private:
    static uint32_t _round_up(uint32_t v, uint32_t q) { return (v + q - 1) / q * q; }

    void _gen_seq(uint32_t M_pad) {
        gen_vision_mm_seq(app_.seq(), M_pad, (uint32_t)K_dev_, (uint32_t)N_, arena_->dp(),
                          /*weight_offset=*/0, rtp_act_, arena_->rtp());
        seq_M_pad_ = M_pad;
        have_seq_  = true;
    }

    /// A when N is a multiple of its 384 tile_N (every shape but the projector's
    /// fc2), else B when N is a multiple of 1024. K never decides: both designs
    /// share k = 384 and K is zero-padded up to it.
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
    vision_mm_arena*          arena_   = nullptr;
    std::string               tag_;
    npu_app                   app_;
    buffer<bf16>              w_bo_;
    int                       K_ = 0, K_dev_ = 0, N_ = 0;
    mcvm::vision::activation   act_ = mcvm::vision::activation::none;
    int32_t                   rtp_act_ = vision_mm_tiles::RTP_NONE;
    bool                      host_epilogue_ = false;
    bool                      bound_         = false;
    uint32_t                  seq_M_pad_     = 0;
    bool                      have_seq_      = false;
};

} // namespace mcvm_npu

#endif  // __MINICPM_V_4_7_NPU_VISION_MM_OP_HPP__
