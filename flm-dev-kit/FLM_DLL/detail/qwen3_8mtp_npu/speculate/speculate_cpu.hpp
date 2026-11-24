// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file speculate/speculate_cpu.hpp
/// \brief CPU backend for module 2.
#pragma once

#include "speculate/speculate_module.hpp"
#include "speculate/mtp_head.hpp"

#include "common/lm_head.hpp"
#include "common/rope.hpp"
#include "qwen3_8mtp_model_def.hpp"
#include "utils/utils.hpp"

#include <chrono>
#include <cstdint>
#include <stdexcept>

namespace qwen38 {

class speculate_cpu final : public speculate_module {
public:
    /// `head` is the base model's lm_head: there is no mtp.lm_head in the
    /// checkpoint, and mtp_use_dedicated_embeddings is false.
    speculate_cpu(const cpu_ref::safetensors& w, const qwen3_8mtp_desc* desc,
                  const cpu_ref::embedding_table* embed,
                  cpu_ref::lm_head* head, int max_len)
        : desc_(desc), head_(head), mtp_(w, embed, max_len),
          logits_((size_t)desc->vocab_size) {}

    int  max_steps() const override { return cpu_ref::mtp::MTP_STEPS; }
    int  cache_len() const override { return mtp_.cache_len(); }
    void reset() override { mtp_.reset(); }
    void crop(int len) override { mtp_.crop(len); }
    void set_max_length(int n) override { mtp_.set_max_length(n); }

    std::vector<int> draft(const std::vector<int>& ids, const float* prev_hidden,
                           int T, int pos0, int k) override {
        if ((int)ids.size() != T)
            throw std::runtime_error("speculate: ids and T disagree");

        std::vector<int>     drafts;
        std::vector<int32_t> step_ids(ids.begin(), ids.end());
        std::vector<float>   hidden(prev_hidden, prev_hidden + (size_t)T * desc_->D);

        int pos   = pos0;
        int width = T;

        last_step0_us_ = 0;
        const time_utils::time_point t_step0 = time_utils::now();

        for (int s = 0; s < k; s++) {
            const std::vector<int32_t> pids = cpu_ref::text_position_ids(pos, width);

            cpu_ref::mtp_taps tp;
            mtp_.step(step_ids.data(), pids.data(), hidden.data(), width, tp);

            // One row per step, even on the wide priming step: the reference
            // projects norm_out[:, -1:, :] regardless of window width.
            const float* last = tp.norm_out.data() + (size_t)(width - 1) * desc_->D;
            head_->forward(last, 1, logits_.data());
            const int tok = cpu_ref::argmax(logits_.data(), (int)desc_->vocab_size);
            drafts.push_back(tok);

            hidden.assign(last, last + desc_->D);
            step_ids.assign(1, (int32_t)tok);
            pos  += width;
            width = 1;

            // See the NPU backend: bottom of the iteration, so the lm_head and
            // the argmax are inside the measurement.
            if (s == 0)
                last_step0_us_ =
                    (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(
                        time_utils::now() - t_step0).count();
        }
        return drafts;
    }

private:
    const qwen3_8mtp_desc* desc_;
    cpu_ref::lm_head*      head_;    ///< borrowed from the engine
    cpu_ref::mtp_head      mtp_;     ///< the head's own private KV cache
    std::vector<float>     logits_;
};

} // namespace qwen38
