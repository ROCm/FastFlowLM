// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file common/npu/lm_head_op.hpp
/// \brief The final norm + vocabulary projection on `lm_head.xclbin`, at
///        L = 1.
///
/// The NPU sibling of `common/lm_head.hpp`, and it lives in common/ for the
/// same reason that one does: the checkpoint has no `mtp.lm_head`, so the
/// draft loop and the base model project through one table. The only thing
/// that differs between the two callers is which norm weight they pass, and
/// that is an argument here rather than state.
///
/// Same shape as `speculate/npu/proj_op.hpp` and for the same reasons: the
/// bitstream computes one thing, the shape is pinned at construction and the
/// sequence is emitted once. Unlike that one it **owns its weights** -- the
/// 1.35 GB table is this object's and nothing else reads it.
///
/// ## The norm is on the device, but the host still computes it
///
/// The kernel norms x in place in the input BO (`rms_norm(x, x, x +
/// MODEL_DIM)` in the design's kernel/qmm.cc, epsilon 1e-6, which is
/// `q38::RMS_EPS`). So `run()` takes the **pre**-norm hidden state, and every
/// dispatch has to rewrite both halves of the input: the first because the
/// device destroyed it, the second because 10 KB against a 1.35 GB weight
/// stream is not worth the state needed to skip it.
///
/// The draft loop still runs `cpu_ref::rms_norm` itself, because its *next*
/// step consumes the normed row as a hidden state. That keeps the MTP layer
/// chain bit-identical to the CPU path and confines this object's influence to
/// the token choice. See speculate/speculate_npu.hpp.
///
/// ## bf16 logits
///
/// The device hands back bf16, roughly three decimal digits. Top-1 ties are
/// therefore common where the fp32 host path had none -- the reference host
/// hit one on its second row, two tokens at 24.75. On the draft path that is
/// output-neutral, since `speculate()` verifies every drafted token against
/// the real stack, but it moves the acceptance rate, which is the only thing
/// drafting buys. `top2_margin()` exists so a driver can measure that rather
/// than assume it.
///
/// Measured, it costs nothing: over the reference rows exactly one top-1 gap
/// lands inside a bf16 ulp (row 1, 0.125 against 0.097 -- the tie above), the
/// drafted ids are identical to the host's, and the runtime hit rate is the
/// same 64.3% either way across three runs. That is a property of this
/// distribution, not a guarantee, which is why the margin stays measurable.
#ifndef __QWEN3_8MTP_NPU_LM_HEAD_OP_HPP__
#define __QWEN3_8MTP_NPU_LM_HEAD_OP_HPP__

#include "common/cpu_ref_common.hpp"
#include "common/cpu_simd.hpp"   // argmax_bf16
#include "common/npu/lm_head_sequence.hpp"
#include "common/npu/lm_head_weights.hpp"

#include "npu_utils/npu_utils.hpp"
#include "buffer.hpp"
#include "typedef.hpp"

#include <stdexcept>
#include <string>
#include <utility>

namespace q38npu {

class npu_lm_head {
public:
    /// \param a     an app over lm_head.xclbin. It owns the x and y staging
    ///              BOs, which must come from the app that dispatches them --
    ///              a BO from another app does not fault, the dispatch simply
    ///              never completes.
    /// \param w     the checkpoint. `lm_head.weight` is read once here and the
    ///              reordered copy lives in this object's BO from then on.
    /// \param name  the tensor, so a driver can point this at something else
    npu_lm_head(npu_app a, const cpu_ref::safetensors& w,
                const std::string& name = "lm_head.weight")
        : app_(std::move(a)) {
        const cpu_ref::q4nx_tensor q = cpu_ref::get_q4nx(w, name);
        vocab_  = q.rows;
        D_      = q.cols;
        padded_ = lm_head_vocab_padded(vocab_);

        // Emitted once and never again: both shapes are fixed and they are the
        // whole of the sequence's input. `npu_app::operator()` notices the
        // version and sets the kernel up on the first dispatch.
        gen_lm_head_seq(app_.seq(), (uint32_t)padded_, (uint32_t)D_);

        const size_t w_bytes = lm_head_weight_bytes(padded_, D_);
        weights_ = app_.create_bo_buffer<u8>(w_bytes);
        y_       = app_.create_bo_buffer<bf16>((size_t)padded_);
        x_       = app_.create_bo_buffer<bf16>((size_t)D_ * 2);

        y_.memset((bf16)0.0f);
        x_.memset((bf16)0.0f);
        y_.sync_to_device();
        x_.sync_to_device();

        stage_lm_head_weights(weights_.data(), weights_.size(), q, padded_);
        weights_.sync_to_device();
    }

    /// Real vocabulary; `run()` returns this many valid logits.
    int vocab() const { return vocab_; }
    /// What the device actually projects, rounded up to a whole round. The
    /// entries past `vocab()` are the replicated pad block's -- never read
    /// them, and never argmax over them.
    int vocab_padded() const { return padded_; }
    int dim() const { return D_; }

    /// For a driver that wants to check the staged bytes.
    buffer<u8>& weight_bo() { return weights_; }

    /// One projection.
    ///
    /// \param x       [dim()] fp32, the **pre**-norm hidden state. Rounded to
    ///                bf16 on the way in -- the device norms and reduces in
    ///                bf16, so a reference meant to isolate this kernel's
    ///                error has to round its own x the same way.
    /// \param norm_w  [dim()] fp32, the RMSNorm weight. `model.norm.weight`
    ///                for the base model, `model.mtp.norm.weight` for the
    ///                draft head.
    /// \return        [vocab_padded()] bf16 logits, of which the first
    ///                `vocab()` are real. Valid until the next call.
    const bf16* run(const float* x, const float* norm_w) {
        if (!x || !norm_w)
            throw std::runtime_error("npu_lm_head::run: null input");

        bf16* xd = x_.data();
        for (int j = 0; j < D_; j++)      xd[j]      = (bf16)x[j];
        for (int j = 0; j < D_; j++)      xd[D_ + j] = (bf16)norm_w[j];
        x_.sync_to_device();

        const ert_cmd_state st = app_(y_, weights_, x_);
        if (st != ERT_CMD_STATE_COMPLETED)
            throw std::runtime_error(
                "npu_lm_head::run: dispatch state " + std::to_string((int)st) +
                " for a " + std::to_string(padded_) + " x " +
                std::to_string(D_) + " projection -- a timeout is a sequence "
                "bug or a bitstream whose MODEL_DIM is not " +
                std::to_string(D_));

        y_.sync_from_device();
        return y_.data();
    }

    /// Index of the largest logit, ties to the lowest index -- the same rule
    /// `cpu_ref::argmax` uses, so the two backends differ only where bf16
    /// actually lost the ordering.
    int argmax(const bf16* logits) const {
        return q38simd::argmax_bf16(logits, vocab_);
    }

    /// Gap between the top two logits. A gap below bf16's resolution at that
    /// magnitude means the top-1 was decided by the rounding and not by the
    /// model.
    float top2_margin(const bf16* logits) const {
        float best = (float)logits[0], second = -1e30f;
        for (int i = 1; i < vocab_; i++) {
            const float v = (float)logits[i];
            if (v > best)        { second = best; best = v; }
            else if (v > second) { second = v; }
        }
        return best - second;
    }

private:
    npu_app      app_;
    int          vocab_ = 0, padded_ = 0, D_ = 0;
    buffer<u8>   weights_;
    buffer<bf16> x_, y_;
};

} // namespace q38npu

#endif  // __QWEN3_8MTP_NPU_LM_HEAD_OP_HPP__
