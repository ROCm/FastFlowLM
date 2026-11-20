// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file verify/verify_cpu.hpp
/// \brief Module 3: batch-verify the draft.
///
/// "cpu" names where the *driver* runs, not the arithmetic -- all 64 layers
/// and the L = 8 projection are device ops; what is left here is the sweep
/// over them, the rope table and the argmax.
///
/// Holds no state buffer of its own: each layer carries the shadow it rolls
/// back to, named by state_slot::speculative. This class only says when.
#pragma once

#include "verify/verify_module.hpp"

#include "common/embedding.hpp"
#include "common/layer_stack.hpp"
#include "common/lm_head.hpp"
#include "common/npu/decode_prof.hpp"   // DPROF_*, no-ops without DECODE_PROFILE=1
#include "common/npu/lm_head8_op.hpp"
#include "common/rope.hpp"
#include "common/window.hpp"
#include "qwen3_8mtp_model_def.hpp"

#include <cstdlib>
#include <string>

namespace qwen38 {

/// The 16 full-attention layers -- the seam for the fused attention xclbin.
struct verify_attn_block {
    window_buffers* bufs;

    explicit verify_attn_block(window_buffers* b) : bufs(b) {}

    void forward(decoder_layer& layer, const float* x, int L, float* y) {
        layer.forward(x, L, bufs->cos.data(), bufs->sin.data(), y);
    }
};

/// The 48 gated-delta-net layers -- the seam for the fused deltanet xclbin.
struct verify_linear_block {
    window_buffers* bufs;

    explicit verify_linear_block(window_buffers* b) : bufs(b) {}

    void forward(decoder_layer& layer, const float* x, int L, float* y) {
        layer.forward(x, L, nullptr, nullptr, y);
    }
};

class verify_cpu final : public verify_module {
public:
    /// \param head8  the L = 8 projection, on its own bitstream. Required.
    /// \param head   the host projection, kept for one case only: a batch
    ///               wider than `npu_lm_head_8::ROWS`, which has no shape on
    ///               the device. A shape guard, not a backend choice.
    verify_cpu(const qwen3_8mtp_desc* desc, layer_stack* stack,
               const cpu_ref::embedding_table* embed, cpu_ref::lm_head* head,
               const std::vector<float>* final_norm_w,
               q38npu::npu_lm_head_8* head8)
        : desc_(desc), stack_(stack), embed_(embed), head_(head),
          norm_w_(final_norm_w), head8_(head8),
          attn_block_(&bufs_), linear_block_(&bufs_) {
        if (!head8_)
            throw std::runtime_error(
                "verify: no NPU lm_head_8 -- module 3 projects on the device");
        if (head8_->vocab() != (int)desc_->vocab_size)
            throw std::runtime_error(
                "verify: lm_head_8 projects " + std::to_string(head8_->vocab()) +
                " tokens but the config says " +
                std::to_string(desc_->vocab_size));
        if (head8_->dim() != (int)desc_->D)
            throw std::runtime_error(
                "verify: lm_head_8 is " + std::to_string(head8_->dim()) +
                " wide but the model is " + std::to_string(desc_->D));
        // logits_ is read in rows of vocab_size, whichever head wrote it
        if (head_ && head_->vocab() != (int)desc_->vocab_size)
            throw std::runtime_error(
                "verify: the host lm_head projects " + std::to_string(head_->vocab()) +
                " tokens but the config says " +
                std::to_string(desc_->vocab_size));
    }

    void snapshot() override {
        stack_->save_state(state_slot::speculative);
        held_ = true;
    }

    void rollback() override {
        if (!held_) throw std::runtime_error("verify: rollback without snapshot");
        stack_->restore_state(state_slot::speculative);
        held_ = false;
    }

    void discard() override { held_ = false; }

    /// `FLM_Q38_FAST_ROLLBACK=0` declines the in-place recovery and takes the
    /// re-prefill instead. One binary can then A/B the two, which is what the
    /// bar is: they must agree *bit for bit* on the state they land on, since
    /// both start from the same snapshot and fold the same tokens.
    bool can_commit_prefix() const override {
        return fast_rollback_ && stack_->can_commit_prefix();
    }

    void commit_prefix(int accept_len) override {
        if (!held_)
            throw std::runtime_error("verify: commit_prefix without snapshot");
        stack_->commit_prefix(state_slot::speculative, accept_len);
        held_ = false;
    }

    std::vector<int> run(const std::vector<int>& ids, const int32_t* pos_thw) override {
        const int L = (int)ids.size();
        bufs_.ensure(L, (int)desc_->D, (int)desc_->RoPE_DIM);
        cpu_ref::mrope_table(pos_thw, L, bufs_.cos.data(), bufs_.sin.data(),
                             /*round_to_bf16=*/true);

        std::vector<int32_t> ids32(ids.begin(), ids.end());
        embed_->embed_rows(ids32.data(), L, bufs_.x.data());

        DPROF_RESET();
        DPROF_B(pass);

        float* x = bufs_.x.data();
        float* y = bufs_.y.data();
        const bool tapping = taps_ && taps_->enabled();
        if (tapping) taps_->begin(L);
        for (uint32_t i = 0; i < desc_->num_hidden_layers; i++) {
            decoder_layer& layer = (*stack_)[(int)i];
            if (layer.is_full_attention()) attn_block_.forward(layer, x, L, y);
            else                           linear_block_.forward(layer, x, L, y);
            std::memcpy(x, y, (size_t)L * desc_->D * sizeof(float));
            if (tapping) taps_->capture((int)i, x, L);
        }
        rows_ = L;

        // All k+1 rows, not just the last: position i predicts the token at
        // i+1, so acceptance cannot be read off the final row alone.
        //
        // `resize`, not `assign`: both backends below write every one of the
        // L * vocab entries -- `_scatter_y` covers [0, vocab) for each kept row
        // and `norm_and_forward` is a dense GEMM -- so zeroing first is ~4.9 MB
        // of memset that nothing reads. Same lesson module 1 already applied.
        logits_.resize((size_t)L * desc_->vocab_size);
        // The device projects a fixed eight rows. A shorter batch still
        // dispatches -- the weight stream is the whole cost and it does not
        // shrink -- but a longer one has no shape here, so fall back rather
        // than split it across two dispatches for a saving that would be
        // negative anyway.
        DPROF_B(head);
        if (L <= q38npu::npu_lm_head_8::ROWS)
            head8_->run(x, L, norm_w_->data(), L, logits_.data(),
                        desc_->rms_eps);
        else
            head_->norm_and_forward(x, L, norm_w_->data(), L, logits_.data(),
                                    desc_->rms_eps);
        DPROF_E(head, head_us);

        // Serial on purpose. The rows are independent and this looks like an
        // obvious `parallel for`, but it sits immediately before 1.5 s of
        // blocking dispatches, and on the default wait policy libgomp leaves
        // the team spinning across them -- exactly the shape q38npu::
        // host_par_min() exists to refuse. Threading it measured 7.0 ms here
        // against 0.24 ms for the same eight rows vectorised on one core, so
        // there is nothing left for a second thread to win.
        DPROF_B(am);
        std::vector<int> out(L);
        for (int t = 0; t < L; t++)
            out[t] = cpu_ref::argmax(logits_.data() + (size_t)t * desc_->vocab_size,
                                     (int)desc_->vocab_size);
        DPROF_E(am, argmax_us);
        DPROF_REPORT(pass, "verify pass, " + std::to_string(L) + " rows");
        return out;
    }

    void set_taps(tap_buffer* taps) override { taps_ = taps; }

    const float* hidden() const override { return rows_ ? bufs_.x.data() : nullptr; }
    const float* logits(int& rows, int& vocab) const override {
        rows  = rows_;
        vocab = (int)desc_->vocab_size;
        return rows_ ? logits_.data() : nullptr;
    }
    int          rows()   const { return rows_; }
    size_t       snapshot_bytes() const { return stack_->state_bytes(); }

    /// The base KV cache holds exactly one row per committed token -- every
    /// token in `committed_ids` was forwarded through the stack, and the one
    /// token that was not (the free base-model token at the mismatch) is the
    /// next cycle's `last_token` and is not counted in the context length yet.
    /// The head's cache trails that by at least one, because it is cropped to
    /// `C - 1` after every cycle; further behind is legal and only costs
    /// acceptance, ahead is not.
    ///
    /// Misaligning either still emits perfectly fluent text while acceptance
    /// quietly decays, which is why this is an assertion and not a log line.
    /// `attention_cache_len()` also cross-checks that all 16 layers agree, so a
    /// cursor moved on some of them catches here too.
    void check_invariant(int committed_tokens, int mtp_cache_len) const override {
        const int base_cache = stack_->attention_cache_len();
        if (base_cache < 0) return;

        if (base_cache != committed_tokens)
            throw std::runtime_error(
                "MTP invariant: base KV cache is " + std::to_string(base_cache) +
                " but committed text is " + std::to_string(committed_tokens));
        if (mtp_cache_len >= 0 && mtp_cache_len > base_cache - 1)
            throw std::runtime_error(
                "MTP invariant: head cache is " + std::to_string(mtp_cache_len) +
                " but base cache is " + std::to_string(base_cache) +
                " -- the head must stay behind the base stack");
    }

private:
    const qwen3_8mtp_desc*          desc_;
    layer_stack*                    stack_;
    const cpu_ref::embedding_table* embed_;
    cpu_ref::lm_head*               head_;
    const std::vector<float>*       norm_w_;
    q38npu::npu_lm_head_8*          head8_;

    tap_buffer*         taps_ = nullptr;
    window_buffers      bufs_;
    verify_attn_block   attn_block_;
    verify_linear_block linear_block_;
    std::vector<float>  logits_;
    int                 rows_ = 0;
    bool                held_ = false;
    bool                fast_rollback_ = _env_fast_rollback();

    static bool _env_fast_rollback() {
        const char* e = std::getenv("FLM_Q38_FAST_ROLLBACK");
        return !(e && e[0] == '0');
    }
};

} // namespace qwen38
