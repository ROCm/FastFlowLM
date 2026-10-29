// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#include "iron/causal_lm.hpp"

#include <cstring>
#include <memory>
#include <stdexcept>
#include <utility>

namespace flm::iron {
namespace {

std::uint32_t ValidateContextLength(std::uint32_t length) {
    if (length == 0) {
        throw std::out_of_range("IRON context length must be positive");
    }
    return length;
}

}  // namespace

CausalLM::CausalLM(const std::string& model_path, std::string model,
                   std::uint32_t context_length)
    : worker_(model_path, std::move(model), context_length),
      max_length_(context_length) {}

buffer<bf16> CausalLM::forward(int id) {
    tokens_.push_back(id);
    return run();
}

buffer<bf16> CausalLM::prefill(std::vector<int>& ids, void*) {
    tokens_.insert(tokens_.end(), ids.begin(), ids.end());
    return run();
}

void CausalLM::set_context_length(int length) {
    if (length < 0 || static_cast<std::size_t>(length) > tokens_.size()) {
        throw std::out_of_range("Invalid IRON context length");
    }
    tokens_.resize(static_cast<std::size_t>(length));
}

void CausalLM::load_weights(Q4NX&) {
    throw std::logic_error("IRON loads safetensors in its worker");
}

void CausalLM::update_max_length(std::uint32_t length) {
    max_length_ = length;
}

void CausalLM::clear_context() {
    tokens_.clear();
    worker_.reset();
}

buffer<bf16> CausalLM::get_k_cache(int, int) { return {}; }

buffer<bf16> CausalLM::get_v_cache(int, int) { return {}; }

int CausalLM::get_current_context_length() {
    return static_cast<int>(tokens_.size());
}

int CausalLM::checkpoint() { return static_cast<int>(tokens_.size()); }

int CausalLM::restore() { return 0; }

bool CausalLM::poisoned() const noexcept { return worker_.poisoned(); }

buffer<bf16> CausalLM::run() {
    if (tokens_.empty() || tokens_.size() > max_length_) {
        throw std::out_of_range("IRON token history exceeds the context length");
    }
    auto words = worker_.logits(tokens_);
    buffer<bf16> output(words.size());
    std::memcpy(output.data(), words.data(), words.size() * sizeof(words[0]));
    return output;
}

CausalBackend::CausalBackend(const flm::backend::BackendContext& context,
                             std::string model)
    : engine_(context.model_path, std::move(model),
              ValidateContextLength(context.context_length)) {}

::causal_lm& CausalBackend::engine() { return engine_; }

std::string CausalBackend::id() const {
    return flm::backend::kIronBackendId;
}

std::string CausalBackend::detail() const { return "IRON graph worker"; }

bool CausalBackend::supports_preemption() const { return false; }

bool CausalBackend::poisoned() const noexcept { return engine_.poisoned(); }

flm::backend::BackendFactory causal_factory(std::string model) {
    return [model = std::move(model)](
               const flm::backend::BackendContext& context) {
        return std::make_unique<CausalBackend>(context, model);
    };
}

}  // namespace flm::iron