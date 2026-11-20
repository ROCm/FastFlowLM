// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file decision_model.hpp
/// \brief Interface of decision encoders: models that read a prompt once and return a hidden
///        state for an application-side readout, instead of generating text.
/// \note  Deliberately separate from causal_lm.hpp: a decision model has no decode, no logits and
///        no conversation state, and adding virtuals to causal_lm renumbers the vtable of every
///        prebuilt engine.
#pragma once
#include <vector>
#include "buffer.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"

class decision_model{
public:
    virtual ~decision_model() {}

    /// \brief load the quantized weights; call once after construction
    virtual void load_weights(Q4NX& q4nx) = 0;

    /// \brief encode independent prompts; every prompt starts from a fresh state
    /// \param ids one token sequence per prompt, each 1..max_length() long
    /// \param payload model-specific multimodal input shared by every prompt (nullptr: text)
    /// \return one final-norm hidden state (bf16 [hidden_size()]) of each prompt's last token,
    ///         in input order. Equal to calling encode() once per prompt (see the engine's
    ///         README for the exact numerical contract of packing).
    virtual std::vector<buffer<bf16>> encode_batch(const std::vector<std::vector<int>>& ids, void* payload = nullptr) = 0;

    /// \brief encode one prompt
    virtual buffer<bf16> encode(const std::vector<int>& ids, void* payload = nullptr) = 0;

    /// \brief longest prompt one encode accepts, in tokens (nothing is ever truncated)
    virtual int max_length() = 0;

    /// \brief width of the returned hidden state
    virtual int hidden_size() = 0;
};
