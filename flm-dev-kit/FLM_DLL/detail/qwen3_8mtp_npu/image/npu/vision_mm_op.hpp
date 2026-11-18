// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file image/npu/vision_mm_op.hpp
/// \brief The vision tower's matmuls on the NPU: one dispatch per bound weight
///        per image, behind `qwen38::vision::mm_op`.
///
/// The second half of phase 2. `npu_vision_attn_op` moved the attention, which
/// at L = 1200 is ~0.2 s of an ~8 s image; this moves the ~7 s that is left.
///
/// ### Weight-resident, which is the whole point
///
/// `image_cpu` binds 111 weights at load -- one patch embed, four per block x
/// 27 blocks, two merger layers -- and they are **959 MB**. `bind()` uploads
/// once; `run()` streams activations only. A per-call weight pointer would
/// re-upload all of it per image and there would be nothing left to win. That
/// is why the seam splits `bind()` from `run()` at all.
///
/// ### One app per op
///
/// `npu_app::operator()` re-runs `_setup_kernel()` whenever the sequence
/// version moves (`npu_utils.hpp:876`), so 111 ops sharing one app would
/// rebuild the ELF, module and kernel on every dispatch. Each op instead owns
/// an app and emits its sequence once -- an op's shape is fixed for a given
/// image size, so the sequence only moves when the image does.
///
/// What is NOT per op is the staging. `create_bo_buffer` returns
/// `buffer<T>(*device, size)` on both `npu_app` and `npu_app_manager` -- a BO
/// is scoped to the DEVICE, and `operator()` binds it per run -- so x and y can
/// be shared by every op of a design and only the weights are private. Private
/// x/y pairs would be ~2.2 GB of scratch against ~24 MB. The arena's BOs come
/// from the same manager its apps do, so they are same-context as well as
/// same-device.
///
/// The corollary is that **two ops of one design must not be in flight at
/// once**: they share x and y. `image_cpu` drives its ops strictly in sequence,
/// so this costs nothing, but a future overlapped tower would need per-op
/// staging back.
///
/// ### Two designs
///
/// A (m=64,k=384,n=48) serves every shape whose N is a multiple of 384; B
/// (m=32,k=384,n=128) exists for `merger.fc2`, whose N of 5120 is not. They are
/// two bitstreams and therefore two `hw_context`s -- a runlist is bound to one,
/// so they cannot be merged (memory: verify-xclbins-are-two-separate-bitstreams).
///
/// For this checkpoint every K and every N is already a multiple of its
/// design's tile, so `bind()` is a plain memcpy and **only M is padded**:
/// 1536/1152/4608 are all multiples of k=384, 1152/3456/4608 of A's tile_N=384,
/// and 5120 is five of B's 1024.
///
/// ### The merger's gelu
///
/// `vision_mm`'s `rtp_act` has no erf gelu. `merger.fc1` asks for one, so it
/// binds `RTP_NONE` and `run()` applies `vision::gelu_erf` on the host. It must
/// NOT bind `RTP_GELU`: that is the tanh form, the two differ by ~1e-3
/// absolute, and the difference lands on the last rows before the language
/// model where nothing downstream can see it -- measured through `fc2` the two
/// are below the bf16 floor and tanh was *ahead* by 6e-7. That is the entire
/// reason the seam makes `gelu_erf` a separate enumerator instead of a comment.
///
/// The epilogue is 1.38M `erff` over 16 threads, ~1-2 ms per image, against
/// rebuilding both bitstreams and re-extracting both RTP addresses to serve one
/// dispatch out of 111.
#ifndef __QWEN3_8MTP_NPU_VISION_MM_OP_HPP__
#define __QWEN3_8MTP_NPU_VISION_MM_OP_HPP__

#include "image/npu/vision_bfp16.hpp"
#include "image/npu/vision_mm_sequence.hpp"
#include "image/vision_ops.hpp"
#include "image/vision_par.hpp"
#include "image/vision_profile.hpp"

#include "common/cpu_ref_common.hpp"
#include "npu_utils/npu_utils.hpp"
#include "buffer.hpp"
#include "typedef.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace q38npu {

// ==========================================================================
/// One design's shared staging: the app manager every op of that design makes
/// its app from, the grow-only x and y BOs they all stream through, the design
/// params and the per-build RTP address.
///
/// Held by the factory, which outlives the tower. The ops keep a raw pointer.
// ==========================================================================
class vision_mm_arena {
public:
    vision_mm_arena(npu_app_manager* mgr,
                    const vision_mm_tiles::design_params& dp,
                    uint32_t rtp_address)
        : mgr_(mgr), dp_(dp), rtp_(rtp_address) {
        if (!mgr_)
            throw std::runtime_error("vision_mm_arena: null app manager");
    }

    npu_app_manager* mgr() const { return mgr_; }
    const vision_mm_tiles::design_params& dp() const { return dp_; }
    uint32_t rtp() const { return rtp_; }

    /// Grow-only. Every op's `run()` calls this before it fills x.
    ///
    /// A reallocation does not invalidate anyone's sequence: the descriptors
    /// carry DDR *offsets*, and the BO handles are bound per run by
    /// `npu_app::operator()`, so only the SHAPE is baked into a sequence and
    /// that is the op's own business.
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

// ==========================================================================
// Pre-rounding onto the mmul's real grid
// ==========================================================================
/// The kernels themselves are `image/npu/vision_bfp16.hpp`; this is the switch
/// and the reason.
///
/// `vision_mm`'s kernel is built with `-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16`
/// (`makefiles/common.mk:28`), so `aie::mmul<8,8,8,bf16>` does not do bf16
/// arithmetic. It converts each operand block of **8 consecutive k for a fixed
/// n** to a shared exponent with a 7-bit mantissa, and it **truncates toward
/// -infinity** rather than rounding to nearest.
///
/// Truncation is the expensive half. A floor is a BIASED error, so over a
/// K-long dot product it accumulates linearly, while round-to-nearest
/// accumulates as sqrt(K) and cancels. Pre-rounding the weights onto the same
/// grid here makes the hardware's floor a no-op and buys back the bias: the
/// same model on Gemma4's `patch_dense` went cos 0.9872 -> 0.9994.
/// (memory: aie-mmul-bf16-truncates-to-bfp16)
///
/// It runs on **both** operands. `bind()` rounds W once per weight; `run()`
/// rounds x on every call, folded into the fp32 -> bf16 narrow. Gemma4 measured
/// x's half worth another 0.00005 and left it out; on this model it is the
/// larger half -- `merger.fc2` band 3 sits at relL2 0.119 against a host
/// reference that does not model the x floor and 0.032 against one that does,
/// and the weight grid explains none of that gap.
///
/// What it does NOT do is touch the checkpoint. Only the copy in the weight BO
/// moves, so the host reference in the verify driver still scores against the
/// true weights and the improvement is real rather than common-mode
/// (memory: score-dequant-mm-against-its-own-quantized-bytes).
///
/// The clamp matters. Rounding to nearest can push a block's largest magnitude
/// up over the next power of two, which would raise the block exponent the
/// hardware then derives and coarsen all 8 -- so the result is clamped back
/// inside the block's own floored range.
/// `FLM_Q38_VISION_MM_PREROUND=0` turns it off. It is on by default and there
/// is no reason to turn it off in production -- the switch exists so the verify
/// driver can measure the arm it replaces, and so a future kernel that stops
/// emulating bfp16 can be A/B'd against it without a rebuild.
inline bool preround_enabled() {
    static const bool on = [] {
        const char* e = std::getenv("FLM_Q38_VISION_MM_PREROUND");
        return !e || e[0] != '0';
    }();
    return on;
}

// ==========================================================================
/// \brief y[M][N] = act(x[M][K] W[N][K]^T + bias[N]) on `vision_mm`.
// ==========================================================================
class npu_vision_mm_op final : public qwen38::vision::mm_op {
public:
    /// \param a  the design-A arena -- every shape whose N is a multiple of A's
    ///           tile_N of 384
    /// \param b  the design-B arena, or null when only A is loaded. A null B
    ///           makes an N that needs it a readable throw at `bind()` rather
    ///           than a quietly wrong answer
    /// \param tag  the op's name in `image_cpu`, carried only so a throw can
    ///           say which of the 111 it was
    npu_vision_mm_op(vision_mm_arena* a, vision_mm_arena* b, std::string tag)
        : arena_a_(a), arena_b_(b), tag_(std::move(tag)) {
        if (!arena_a_)
            throw std::runtime_error("npu_vision_mm_op: null design-A arena");
    }

    /// Upload one weight for the lifetime of the op.
    ///
    /// The BO layout is `seq_gen.hpp`'s DDR layout and nothing else: N bias
    /// elements, then the `[N][K]` weight row-major. Both come straight out of
    /// the mmap'd checkpoint in `nn.Linear` order, so there is no transpose and
    /// no repacking here -- two memcpys into device memory.
    ///
    /// A null bias is zero-filled rather than skipped. The kernel's prologue
    /// reads the leading `n` elements of the first w chunk unconditionally and
    /// seeds every accumulator from them, so "no bias" has to be an explicit N
    /// zeros (memory: bias-prologue-spans-k-chunks).
    void bind(const uint16_t* w, const uint16_t* bias,
              int K, int N, qwen38::vision::activation act) override {
        if (!w)
            throw std::runtime_error("npu_vision_mm_op::bind(" + tag_ +
                                     ") got a null weight");
        if (K <= 0 || N <= 0)
            throw std::runtime_error("npu_vision_mm_op::bind(" + tag_ +
                                     "): K=" + std::to_string(K) +
                                     " N=" + std::to_string(N));

        arena_ = _pick(K, N);
        K_ = K; N_ = N; act_ = act;

        // vision_mm has no erf gelu. Bind none and keep the epilogue here --
        // NOT RTP_GELU, which is the tanh form. See the file comment.
        host_epilogue_ = (act == qwen38::vision::activation::gelu_erf);
        rtp_act_ = host_epilogue_ ? vision_mm_tiles::RTP_NONE : (int32_t)act;

        const size_t w_words = (size_t)N + (size_t)N * (size_t)K;
        w_bo_ = arena_->mgr()->create_bo_buffer<bf16>(w_words);
        uint16_t* d = reinterpret_cast<uint16_t*>(w_bo_.data());
        if (bias) std::memcpy(d, bias, (size_t)N * sizeof(uint16_t));
        else      std::memset(d, 0,    (size_t)N * sizeof(uint16_t));
        std::memcpy(d + N, w, (size_t)N * (size_t)K * sizeof(uint16_t));

        // The weight rows are contiguous K, which is the axis the mmul groups
        // on, so the whole [N][K] block is one run of 8-element groups and no
        // group ever straddles two rows (every K here is a multiple of 384).
        // The bias is NOT pre-rounded: it never enters the mmul, it seeds the
        // accumulator (memory: bias-prologue-spans-k-chunks).
        if (preround_enabled())
            preround_bfp16_inplace(d + N, (size_t)N * (size_t)K);
        w_bo_.sync_to_device();

        app_      = arena_->mgr()->create_app();
        have_seq_ = false;
        bound_    = true;
    }

    /// Run M rows. **One dispatch, whatever M is.**
    ///
    /// \param x  [M][K] fp32, row-major
    /// \param M  rows. Any positive count; padded here to the design's tile_M
    /// \param y  [M][N] fp32 out
    void run(const float* x, int M, float* y) override {
        if (!bound_)
            throw std::runtime_error("npu_vision_mm_op::run(" + tag_ +
                                     ") before bind()");
        if (M <= 0) return;

        const vision_mm_tiles::design_params& dp = arena_->dp();
        const uint32_t M_pad = _round_up((uint32_t)M, dp.tile_M());

        arena_->ensure((size_t)M_pad * (size_t)K_, (size_t)M_pad * (size_t)N_);

        // x: convert the M real rows and ZERO ROWS [M, M_pad) ON EVERY CALL.
        // The pad rows are not inert -- there is no mask anywhere in this
        // design, so a previous, longer image's activations left there are
        // multiplied and accumulated before their output rows are dropped, and
        // an overflowing bf16 accumulator does not stay in its own row
        // (memory: kv-rows-above-cursor-are-not-inert). Zeroing is one memset
        // on at most tile_M-1 rows and removes the question.
        bf16*        xd   = arena_->x().data();
        const size_t real = (size_t)M * (size_t)K_;
        {
            VPROF(mm_x_us);
            // One traversal, not two. The pre-round used to be a second pass
            // that read this buffer straight back; fusing it removes a read and
            // a write of half a gigabyte per image. Rows are contiguous K and
            // every K here is a multiple of 384, so no group of 8 straddles a
            // row, and the pad rows need no rounding of their own -- a group of
            // eight zeroes has block exponent 0 and the round skips it.
            if (preround_enabled())
                narrow_preround_bfp16(xd, x, real);
            else
                narrow_bf16_par(xd, x, real);
            const size_t tail = ((size_t)M_pad - (size_t)M) * (size_t)K_;
            if (tail) std::memset(xd + real, 0, tail * sizeof(bf16));
        }

        { VPROF(mm_sync_us); arena_->x().sync_to_device(); }

        if (!have_seq_ || M_pad != seq_M_pad_) {
            gen_vision_mm_seq(app_.seq(), M_pad, (uint32_t)K_, (uint32_t)N_,
                              dp, /*weight_offset=*/0, rtp_act_, arena_->rtp());
            seq_M_pad_ = M_pad;
            have_seq_  = true;
        }

        VPROF_COUNT(mm_calls);
        const ert_cmd_state st = VPROF_EXPR(mm_dev_us,
            app_(arena_->y(), arena_->x(), w_bo_));
        if (st != ERT_CMD_STATE_COMPLETED) {
            have_seq_ = false;   // do not reuse a sequence that did not land
            throw std::runtime_error(
                "npu_vision_mm_op::run(" + tag_ + "): dispatch state " +
                std::to_string((int)st) + " at M " + std::to_string(M) +
                " (padded " + std::to_string(M_pad) + ", K " +
                std::to_string(K_) + ", N " + std::to_string(N_) + ", " +
                std::to_string(M_pad / dp.tile_M()) + " x " +
                std::to_string((uint32_t)N_ / dp.tile_N()) + " rounds) -- a "
                "timeout is a sequence bug or a stale RTP address (diff "
                "vision_mm_rtp_addresses against FLM_Xclbin/Qwen3_8/vision_mm/"
                "build/<design>/rtp_addr.txt)");
        }

        { VPROF(mm_sync_us); arena_->y().sync_from_device(); }
        VPROF(mm_y_us);   // the widen and the erf epilogue below, either arm

        // Gather the logical [M][N] block out of the padded output. The row
        // stride is N, not a padded N: this checkpoint's N never needs padding
        // and `gen_vision_mm_seq` throws rather than silently accept one that
        // would, so the valid rows are the leading M*N elements, contiguous.
        const bf16*  yd    = arena_->y().data();
        const size_t n_out = (size_t)M * (size_t)N_;
        if (!host_epilogue_) {
            // 110 of the 111 ops land here, and it was scalar AND unthreaded:
            // 195 ms of a 1662 ms encode, the third-largest stage.
            widen_f32_par(y, yd, n_out);
            return;
        }

        // The merger's erf gelu. Rounding the result back to bf16 under
        // emulation is what keeps this arm scoreable against `cpu_mm_op` on the
        // same bytes -- the device's `none` output is already bf16, so the only
        // new rounding is the activation's
        // (memory: score-dequant-mm-against-its-own-quantized-bytes).
        const bool emu = qwen38::vision::cpu_ops_emulate_bf16();
#pragma omp parallel for schedule(static) num_threads(qwen38::vision::par_threads())
        for (int i = 0; i < M; i++) {
            const size_t base = (size_t)i * (size_t)N_;
            for (int j = 0; j < N_; j++) {
                const float v = qwen38::vision::apply_activation(
                                    (float)yd[base + (size_t)j], act_);
                y[base + (size_t)j] = emu ? cpu_ref::round_bf16(v) : v;
            }
        }
    }

    int K() const override { return K_; }
    int N() const override { return N_; }
    const char* backend() const override { return "npu"; }

    /// Which design `bind()` chose, for the verify driver's geometry check.
    bool on_design_b() const { return arena_ && arena_ == arena_b_; }
    /// Whether the erf epilogue stayed on the host, likewise.
    bool host_epilogue() const { return host_epilogue_; }

private:
    static uint32_t _round_up(uint32_t v, uint32_t q) {
        return (v + q - 1) / q * q;
    }

    /// Pick a design from the shape, as a rule rather than a table, so a
    /// checkpoint with different widths either works or says why it cannot.
    ///
    /// A is tried first. Its tile_N of 384 is the finer of the two, so it
    /// wastes less on a narrow N; B's only advantage is a tile_M of 128 against
    /// A's 256, which would matter for the merger's L/4 rows except that B
    /// cannot express N = 4608 at all (4.5 of its 1024).
    vision_mm_arena* _pick(int K, int N) const {
        const bool a_ok = arena_a_ &&
                          K % (int)arena_a_->dp().k == 0 &&
                          N % (int)arena_a_->dp().tile_N() == 0;
        if (a_ok) return arena_a_;
        const bool b_ok = arena_b_ &&
                          K % (int)arena_b_->dp().k == 0 &&
                          N % (int)arena_b_->dp().tile_N() == 0;
        if (b_ok) return arena_b_;
        throw std::runtime_error(
            "npu_vision_mm_op::bind(" + tag_ + "): K=" + std::to_string(K) +
            " N=" + std::to_string(N) + " fits neither vision_mm design (A "
            "needs K%384==0 and N%384==0; B needs K%384==0 and N%1024==0" +
            std::string(arena_b_ ? "" : "; B is not loaded") +
            ") -- serving it would mean zero-extending N in the weight BO, "
            "which this checkpoint never requires");
    }

    vision_mm_arena*           arena_a_ = nullptr;
    vision_mm_arena*           arena_b_ = nullptr;
    vision_mm_arena*           arena_   = nullptr;
    std::string                tag_;
    npu_app                    app_;
    buffer<bf16>               w_bo_;
    int                        K_ = 0, N_ = 0;
    qwen38::vision::activation act_ = qwen38::vision::activation::none;
    int32_t                    rtp_act_ = vision_mm_tiles::RTP_NONE;
    bool                       host_epilogue_ = false;
    bool                       bound_         = false;
    uint32_t                   seq_M_pad_     = 0;
    bool                       have_seq_      = false;
};

} // namespace q38npu

#endif  // __QWEN3_8MTP_NPU_VISION_MM_OP_HPP__
