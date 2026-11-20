// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file qwen3_5_omni_talker.hpp
/// \brief qwen3_5_omni talker (speech/token-to-audio head) — simple stub.
/// \author FastFlowLM Team
#pragma once
#include "lm_config.hpp"
#include "npu_utils/npu_utils.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"
#include "tensor_2d.hpp"
#include "utils/utils.hpp"

#include "models/qwen3_5_omni/qwen3_5_omni.hpp"
#include "qwen3_5_omni_talker_def.hpp"

class qwen3_5_omni_talker {
public:
    qwen3_5_omni_talker(LM_Config config, npu_xclbin_manager *npu_instance, int MAX_L = 4096);
    ~qwen3_5_omni_talker();

    buffer<bf16> forward(qwen3_5_omni_thinker_result_t& thinker_res);
    void load_weights(Q4NX& q4nx);
    void set_context_length(int L);
    void clear_context();

private:
    LM_Config config;
    npu_xclbin_manager *npu;
    int current_context_length;

    /// @brief talker head + text backbone config (talker_config[.text_config])
    qwen3_5_omni_talker_desc desc;
};
