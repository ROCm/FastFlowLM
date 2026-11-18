// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file qwen3_5_omni_npu_detail.hpp
/// \brief qwen3_5_omni private implementation (pimpl)
/// \author FastFlowLM Team
/// \note Barebone skeleton — fill in the logic.
#pragma once
#include "models/qwen3_5_omni/qwen3_5_omni.hpp"
#include "modules/dequant.hpp"
#include "modules/gemm.hpp"
#include "modules/mha.hpp"

#include "qwen3_5_omni_thinker/qwen3_5_omni_thinker.hpp"
#include "qwen3_5_omni_talker/qwen3_5_omni_talker.hpp"

struct qwen3_5_omni::Impl {
    /// @brief construct / destruct
    Impl(LM_Config config, npu_xclbin_manager *npu_instance, int MAX_L = 4096);
    ~Impl();

    /// @brief public surface mirrored from qwen3_5_omni
    qwen3_5_omni_thinker_result_t forward(int ids);
    buffer<bf16> say(qwen3_5_omni_thinker_result_t thinker_res);
    qwen3_5_omni_thinker_result_t prefill(std::vector<int>& ids, void* payload = nullptr);

    void set_context_length(int L);
    void load_weights(Q4NX& q4nx);
    void clear_context();
    void update_max_length(uint32_t MAX_L);
    int  get_current_context_length();
    int  checkpoint();
    int  restore();

    /// @brief config / npu handles
    LM_Config config;
    npu_xclbin_manager *npu;

    /// @brief omni sub-modules (write your logic inside these)
    /// @note image/audio embed are owned by the thinker.
    std::unique_ptr<qwen3_5_omni_thinker>     thinker;
    std::unique_ptr<qwen3_5_omni_talker>      talker;

    /// @brief dimensions
    uint32_t D;
    uint32_t DH;
    uint32_t MAX_L;
    uint32_t HIDDEN_SIZE;
    uint32_t INTERMEDIATE_SIZE;

    int vocab_size;
    int vocab_size_padded;
    int current_context_length;
};
