// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file qwen3_5_omni_talker.cpp
/// \brief qwen3_5_omni talker implementation — simple stub.
/// \author FastFlowLM Team
#include "qwen3_5_omni_talker.hpp"

qwen3_5_omni_talker::qwen3_5_omni_talker(LM_Config config, npu_xclbin_manager *npu_instance, int MAX_L)
    : config(config), npu(npu_instance), current_context_length(0)
{
    desc.build(config);
}

qwen3_5_omni_talker::~qwen3_5_omni_talker() = default;

buffer<bf16> qwen3_5_omni_talker::forward(qwen3_5_omni_thinker_result_t &thinker_res)
{
    throw std::runtime_error("qwen3_5_omni_talker::forward not implemented");
}

void qwen3_5_omni_talker::load_weights(Q4NX &q4nx) {}

void qwen3_5_omni_talker::set_context_length(int L)
{
    this->current_context_length = L;
}

void qwen3_5_omni_talker::clear_context()
{
    this->set_context_length(0);
}
