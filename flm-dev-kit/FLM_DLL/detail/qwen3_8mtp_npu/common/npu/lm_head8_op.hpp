// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file common/npu/lm_head8_op.hpp
/// \brief The vocabulary projection on `lm_head_8.xclbin`, at L = 8 -- the
///        verify phase's shape.
///
/// The NPU sibling of `cpu_ref::lm_head::norm_and_forward`, and `run()` takes
/// that method's arguments in that order so module 3 swaps backends on one
/// line. The L = 1 sibling is `npu_lm_head`; the two are separate bitstreams,
/// separate hw_contexts and separate weight BOs.
///
/// ## Why a second 1.35 GB copy of the table
///
/// A BO must come from the app that dispatches it -- a BO from another app
/// does not fault, the dispatch simply never completes -- and the two xclbins
/// are two apps. So with both heads enabled the process holds ~2.7 GB of Q8_0
/// weights. That is the price of the L = 8 design and there is no way around
/// it short of one bitstream that serves both shapes.
///
/// ## The norm is on the core, as at L = 1, but split differently
///
/// npu2 core data memory is 64 KB and eight rows of x are 80 KB, so the core
/// never *holds* a whole row -- but it sees every column of one across the k
/// loop, and RMSNorm splits so that this is enough:
///
///     logits[t][m] = s[t] * SUM_i (x[t][i] * nw[i]) * W[m][i]
///
/// `s[t] = 1/sqrt(mean(x[t]^2) + eps)` is a reduction the core accumulates on
/// round 0 and a scalar it applies to the finished accumulator, so it is never
/// needed before x has been consumed. Only `nw` is per-column, and it rides in
/// as an extra lane of the x stream for the kernel to fold in.
///
/// So this bitstream's input is **pre-norm** x, the same as `npu_lm_head`'s,
/// and `run()` really is `cpu_ref::lm_head::norm_and_forward` argument for
/// argument rather than only in shape. The one thing that does not travel is
/// `eps`: it is a kernel constant, and `run()` throws rather than silently
/// ignoring a different one.
///
/// This costs ~1/8 more bytes on the x stream (10 KB per round per column) and
/// one extra multiply per group per token in the inner loop. It buys the host
/// nothing in speed -- the norm was ~0.006% of the arithmetic -- so what it is
/// for is the interface: there is no longer a pair of bitstreams that take
/// opposite inputs and silently produce fluent, wrong logits when swapped.
///
/// ## bf16 logits, and what that costs a verify pass
///
/// The device hands back bf16 -- roughly three decimal digits -- so top-1 ties
/// appear that the fp32 host path never saw (the reference's own row 1 has two
/// tokens at 24.75). On the *draft* path a tie is output-neutral because every
/// drafted token is re-checked. Here it is not: module 3's argmax IS the
/// committed token. A tie broken the other way emits a different token than
/// the CPU backend would have.
///
/// That is a real behavioural difference and it is worth stating plainly
/// rather than burying: this backend is output-equivalent to the CPU one only
/// up to bf16's resolution of the top-1 gap. `top2_margin()` is here so a
/// driver can measure how often that gap is inside a ulp instead of assuming.
/// Measured on the reference rows it is one row in eight, and that row is an
/// exact tie in the reference too. (memory: qwen38-lm-head-untied-and-argmax-ties)
#ifndef __QWEN3_8MTP_NPU_LM_HEAD8_OP_HPP__
#define __QWEN3_8MTP_NPU_LM_HEAD8_OP_HPP__

#include "common/cpu_ref_common.hpp"
#include "common/cpu_simd.hpp"   // the vectorised fp32 <-> bf16 marshalling
#include "common/host_par.hpp"   // host_par_min(), the OpenMP wait-policy guard
#include "common/npu/decode_prof.hpp"   // DPROF_*, no-ops without DECODE_PROFILE=1
#include "common/npu/lm_head8_sequence.hpp"
#include "common/npu/lm_head_weights.hpp"

#include "npu_utils/npu_utils.hpp"
#include "buffer.hpp"
#include "typedef.hpp"

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace q38npu {

class npu_lm_head_8 {
public:
    /// \param a     an app over lm_head_8.xclbin
    /// \param w     the checkpoint. `lm_head.weight` is read once here and the
    ///              reordered copy lives in this object's BO from then on.
    /// \param name  the tensor, so a driver can point this at something else
    npu_lm_head_8(npu_app a, const cpu_ref::safetensors& w,
                  const std::string& name = "lm_head.weight")
        : app_(std::move(a)) {
        const cpu_ref::q4nx_tensor q = cpu_ref::get_q4nx(w, name);
        vocab_  = q.rows;
        D_      = q.cols;
        padded_ = lm_head_vocab_padded(vocab_);

        gen_lm_head8_seq(app_.seq(), (uint32_t)padded_, (uint32_t)D_);

        const size_t w_bytes = lm_head_weight_bytes(padded_, D_);
        weights_ = app_.create_bo_buffer<u8>(w_bytes);
        y_       = app_.create_bo_buffer<bf16>((size_t)ROWS * (size_t)padded_);
        // ROWS + 1 lanes: the token rows plus model.norm.weight.
        x_ = app_.create_bo_buffer<bf16>(
            (size_t)lm_head8_tiles::X_LANES * (size_t)D_);

        y_.memset((bf16)0.0f);
        x_.memset((bf16)0.0f);
        y_.sync_to_device();
        x_.sync_to_device();

        // The same staging routine the L = 1 head uses, unchanged: the two
        // designs share an array topology, so they share a weight layout.
        stage_lm_head_weights(weights_.data(), weights_.size(), q, padded_);
        weights_.sync_to_device();
    }

    /// Token rows one dispatch projects. Fixed by the bitstream.
    static constexpr int ROWS = lm_head8_tiles::L_TOKENS;

    int vocab()        const { return vocab_; }
    int vocab_padded() const { return padded_; }
    int dim()          const { return D_; }

    /// For a driver that wants to check the staged bytes.
    buffer<u8>& weight_bo() { return weights_; }

    /// rms_norm a `keep`-row suffix of x, then project it. Both happen on the
    /// device; this method only interleaves the operands.
    ///
    /// Argument for argument `cpu_ref::lm_head::norm_and_forward`, so module 3
    /// picks a backend rather than a code path.
    ///
    /// \param x       [L][dim()] fp32, the **pre**-norm hidden states
    /// \param L        rows in x
    /// \param norm_w  [dim()] fp32, `model.norm.weight`; goes to the device as
    ///                lane ROWS of the x stream rather than being applied here
    /// \param keep    how many trailing rows to project; 1 .. ROWS
    /// \param logits  [keep][vocab()] fp32, written here
    /// \param eps     RMSNorm epsilon. Baked into the bitstream -- anything but
    ///                `cpu_ref::q38::RMS_EPS` throws.
    ///
    /// A `keep` below ROWS still runs the full eight lanes -- the array has no
    /// shorter shape and the weight stream, which is the entire cost, does not
    /// shrink. The spare lanes are fed a copy of the first kept row rather
    /// than zeros: an all-zero row makes the norm's reciprocal 1/sqrt(eps) and
    /// the cores then do real work on denormal-adjacent values for no reason.
    void run(const float* x, int L, const float* norm_w, int keep,
             float* logits, float eps = cpu_ref::q38::RMS_EPS) {
        if (!x || !norm_w || !logits)
            throw std::runtime_error("npu_lm_head_8::run: null input");
        if (keep <= 0 || keep > L)
            throw std::runtime_error(
                "npu_lm_head_8::run: keep is " + std::to_string(keep) +
                " for " + std::to_string(L) + " rows");
        if (keep > ROWS)
            throw std::runtime_error(
                "npu_lm_head_8::run: this bitstream projects " +
                std::to_string(ROWS) + " rows, asked for " +
                std::to_string(keep));

        _upload_x(x, L, norm_w, keep, eps);

        const ert_cmd_state st = DPROF_T(head_dispatch_us, app_(y_, weights_, x_));
        if (st != ERT_CMD_STATE_COMPLETED)
            throw std::runtime_error(
                "npu_lm_head_8::run: dispatch state " +
                std::to_string((int)st) + " for a " + std::to_string(padded_) +
                " x " + std::to_string(D_) + " projection over " +
                std::to_string(ROWS) + " rows -- a timeout is a sequence bug "
                "or a bitstream whose MODEL_DIM is not " + std::to_string(D_));

        y_.sync_from_device();
        _scatter_y(logits, keep);
    }

    /// Gap between the top two logits of one row of a `run()` result. Below
    /// bf16's resolution at that magnitude means the rounding picked the
    /// token, not the model.
    float top2_margin(const float* row) const {
        float best = row[0], second = -1e30f;
        for (int i = 1; i < vocab_; i++) {
            const float v = row[i];
            if (v > best)        { second = best; best = v; }
            else if (v > second) { second = v; }
        }
        return best - second;
    }

private:
    /// x goes out as [k_block][lane][256] over `X_LANES = ROWS + 1` lanes: the
    /// ROWS token rows, then `model.norm.weight`. The kernel's k-loop wants all
    /// lanes of one column block together, and the memtile replays the whole
    /// buffer per round in exactly this order.
    ///
    /// The rows go out **raw**. The device normalises them -- `eps` is baked
    /// into the bitstream rather than passed, so a caller that wants a
    /// different one has to use the CPU backend. Everything else about this
    /// method's contract is unchanged.
    void _upload_x(const float* x, int L, const float* norm_w, int keep,
                   float eps) {
        // The bitstream's RMS_EPS. A mismatch is a silent accuracy change, not
        // a failure, so it is worth naming rather than ignoring the argument.
        // The tolerance is there only to absorb a JSON round trip: it is three
        // orders of magnitude tighter than the gap to the next plausible value
        // (1e-5), which is what this is for.
        if (std::fabs(eps - cpu_ref::q38::RMS_EPS) > 1e-9f)
            throw std::runtime_error(
                "npu_lm_head_8::run: this bitstream normalises on the core with "
                "eps = " + std::to_string(cpu_ref::q38::RMS_EPS) +
                ", asked for " + std::to_string(eps));

        const int K_BLOCKS = D_ / lm_head8_tiles::K_CHUNK;
        const int C        = lm_head8_tiles::K_CHUNK;
        const int LANES    = lm_head8_tiles::X_LANES;
        const int off      = L - keep;
        bf16* dst = x_.data();

        for (int k = 0; k < K_BLOCKS; k++) {
            for (int t = 0; t < ROWS; t++) {
                // Lanes past `keep` repeat the first kept row; see run()'s note.
                const int src_row = (t < keep) ? (off + t) : off;
                const float* in = x + (size_t)src_row * (size_t)D_ +
                                  (size_t)k * C;
                bf16* out = dst + ((size_t)k * LANES + (size_t)t) * C;
                q38simd::f32_to_bf16(out, in, (size_t)C);
            }
            // Lane ROWS: model.norm.weight. The same bytes every round and
            // every dispatch, but it is re-sent rather than held resident
            // because a one-shot fill BD is what broke this design's second
            // dispatch (memory: a-prologue-bd-breaks-the-second-dispatch).
            const float* nw  = norm_w + (size_t)k * C;
            bf16*        out = dst + ((size_t)k * LANES + (size_t)ROWS) * C;
            q38simd::f32_to_bf16(out, nw, (size_t)C);
        }
        x_.sync_to_device();
    }

    /// The cores emit [token][32] per 32-row weight block, so the DDR image is
    /// [round][col][core][token][32]. Un-interleave into `keep` contiguous
    /// fp32 logit rows, dropping the pad entries past `vocab_` -- the weight
    /// reorder filled those row blocks with a replicated block 0, so they hold
    /// real numbers rather than obvious garbage and an argmax over them would
    /// silently return a token id past the vocabulary.
    void _scatter_y(float* logits, int keep) const {
        const int    BLOCKS = lm_head8_tiles::M_PER_ROUND / lm_head8_tiles::M;
        // Signed: MSVC's /openmp is OpenMP 2.0, which requires a signed
        // integral index on the parallel loop below (C3016).
        const int64_t rounds = (int64_t)padded_ / lm_head8_tiles::M_PER_ROUND;
        const bf16*  src    = y_.data();

        // Gated like every other host row loop in this module: on the default
        // wait policy libgomp leaves the team spinning after the region, and
        // what follows here is 1.5 s of blocking dispatches for the next
        // verify pass. See q38npu::host_par_min().
#pragma omp parallel for schedule(static) \
        if ((size_t)keep * (size_t)vocab_ >= q38npu::host_par_min())
        for (int64_t r = 0; r < rounds; r++) {
            for (int blk = 0; blk < BLOCKS; blk++) {
                const size_t base = (size_t)r * lm_head8_tiles::M_PER_ROUND +
                                    (size_t)blk * lm_head8_tiles::M;
                if (base >= (size_t)vocab_) continue;
                const int n = (base + lm_head8_tiles::M <= (size_t)vocab_)
                                  ? lm_head8_tiles::M
                                  : (int)((size_t)vocab_ - base);
                for (int t = 0; t < keep; t++) {
                    const bf16* in =
                        src + ((((size_t)r * (size_t)BLOCKS + (size_t)blk) *
                                (size_t)ROWS + (size_t)t) * lm_head8_tiles::M);
                    float* out = logits + (size_t)t * (size_t)vocab_ + base;
                    q38simd::bf16_to_f32(out, in, (size_t)n);
                }
            }
        }
    }

    npu_app      app_;
    int          vocab_ = 0, padded_ = 0, D_ = 0;
    buffer<u8>   weights_;
    buffer<bf16> x_, y_;
};

} // namespace q38npu

#endif  // __QWEN3_8MTP_NPU_LM_HEAD8_OP_HPP__
