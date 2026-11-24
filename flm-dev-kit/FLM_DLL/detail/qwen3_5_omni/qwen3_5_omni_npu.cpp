// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file qwen3_5_omni_npu.cpp
/// \brief qwen3_5_omni implementation — barebone skeleton.
/// \author FastFlowLM Team
#include "qwen3_5_omni_npu_detail.hpp"

// ===============================================
// Impl
// ===============================================
qwen3_5_omni::Impl::Impl(LM_Config config, npu_xclbin_manager *npu_instance, int MAX_L)
    : config(config), npu(npu_instance)
{
    current_context_length = 0;
    // TODO: register xclbins, instantiate sub-modules, allocate buffers.
    this->thinker = std::make_unique<qwen3_5_omni_thinker>(config, npu, MAX_L);
    this->talker  = std::make_unique<qwen3_5_omni_talker>(config, npu, MAX_L);
}

qwen3_5_omni::Impl::~Impl() = default;

void qwen3_5_omni::Impl::load_weights(Q4NX &q4nx)
{
    this->thinker->load_weights(q4nx);
}

qwen3_5_omni_thinker_result_t qwen3_5_omni::Impl::forward(int ids)
{
    qwen3_5_omni_thinker_result_t res;
    res.logits = this->thinker->forward(ids);
    return res;
}

buffer<bf16> qwen3_5_omni::Impl::say(qwen3_5_omni_thinker_result_t thinker_res)
{
    // TODO: talker synthesis from thinker result.
    throw std::runtime_error("qwen3_5_omni::say not implemented");
}

qwen3_5_omni_thinker_result_t qwen3_5_omni::Impl::prefill(std::vector<int> &ids, void *payload)
{
    qwen3_5_omni_thinker_result_t res;
    res.logits = this->thinker->prefill(ids, payload);
    return res;
}

void qwen3_5_omni::Impl::set_context_length(int L)
{
    std::cout << "Setting context length is not supported for qwen3_5_omni_thinker!" << std::endl;
}

void qwen3_5_omni::Impl::clear_context()
{
    this->thinker->clear_context();
}

void qwen3_5_omni::Impl::update_max_length(uint32_t MAX_L)
{
    if (MAX_L <= this->MAX_L) {
        header_print("FLM", "New length is shorter than the current length, no need to update!");
        return;
    }
    this->MAX_L = MAX_L;
    // TODO: reallocate kv-caches / propagate to sub-modules.
}

int qwen3_5_omni::Impl::get_current_context_length()
{
    return this->current_context_length;
}

int qwen3_5_omni::Impl::checkpoint()
{
    return this->thinker->checkpoint();
}

int qwen3_5_omni::Impl::restore()
{
    return this->thinker->restore();
}

// ===============================================
// Externals for qwen3_5_omni
// ===============================================
qwen3_5_omni::qwen3_5_omni(LM_Config config, npu_xclbin_manager *npu_instance, int MAX_L)
{
    this->_impl = new Impl(config, npu_instance, MAX_L);
}

qwen3_5_omni::~qwen3_5_omni()
{
    delete this->_impl;
}

qwen3_5_omni_thinker_result_t qwen3_5_omni::forward(int ids)
{
    return this->_impl->forward(ids);
}

buffer<bf16> qwen3_5_omni::say(qwen3_5_omni_thinker_result_t thinker_res)
{
    return this->_impl->say(thinker_res);
}

qwen3_5_omni_thinker_result_t qwen3_5_omni::prefill(std::vector<int> &ids, void *payload)
{
    try {
        return this->_impl->prefill(ids, payload);
    }
    catch (const std::runtime_error &e) {
        header_print("FLM", e.what());
        throw;
    }
}

void qwen3_5_omni::set_context_length(int L)
{
    this->_impl->set_context_length(L);
}

void qwen3_5_omni::load_weights(Q4NX &q4nx)
{
    this->_impl->load_weights(q4nx);
}

void qwen3_5_omni::clear_context()
{
    this->_impl->clear_context();
}

void qwen3_5_omni::update_max_length(uint32_t MAX_L)
{
    this->_impl->update_max_length(MAX_L);
}

int qwen3_5_omni::get_current_context_length()
{
    return this->_impl->get_current_context_length();
}

int qwen3_5_omni::checkpoint()
{
    return this->_impl->checkpoint();
}

int qwen3_5_omni::restore()
{
    return this->_impl->restore();
}
