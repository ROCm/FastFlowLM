// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#include "models/llama3/iron/llama3_iron_backend.hpp"

#include "iron/worker.hpp"

#include <cstring>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <vector>

namespace flm::llama3 {
namespace {

class IronLlama final : public causal_lm {
public:
    IronLlama(const std::string& model_path, std::uint32_t context_length)
        : worker_(model_path, context_length), max_length_(context_length) {}

    buffer<bf16> forward(int id) override {
        tokens_.push_back(id);
        return run();
    }

    buffer<bf16> prefill(std::vector<int>& ids, void*) override {
        tokens_.insert(tokens_.end(), ids.begin(), ids.end());
        return run();
    }

    void set_context_length(int length) override {
        if (length < 0 || static_cast<std::size_t>(length) > tokens_.size()) {
            throw std::out_of_range("Invalid IRON context length");
        }
        tokens_.resize(static_cast<std::size_t>(length));
    }
    void load_weights(Q4NX&) override {
        throw std::logic_error("IRON loads safetensors in its worker");
    }
    void update_max_length(std::uint32_t length) override { max_length_ = length; }
    void clear_context() override {
        tokens_.clear();
        worker_.reset();
    }
    buffer<bf16> get_k_cache(int, int) override { return {}; }
    buffer<bf16> get_v_cache(int, int) override { return {}; }
    int get_current_context_length() override { return static_cast<int>(tokens_.size()); }
    int checkpoint() override { return static_cast<int>(tokens_.size()); }
    int restore() override { return 0; }
    bool poisoned() const noexcept { return worker_.poisoned(); }

private:
    buffer<bf16> run() {
        if (tokens_.empty() || tokens_.size() > max_length_) {
            throw std::out_of_range("IRON token history exceeds the context length");
        }
        auto words = worker_.logits(tokens_);
        buffer<bf16> output(words.size());
        std::memcpy(output.data(), words.data(), words.size() * sizeof(words[0]));
        return output;
    }

    flm::iron::Worker worker_;
    std::vector<int> tokens_;
    std::uint32_t max_length_;
};

class IronBackend final : public flm::backend::ModelBackend {
public:
    explicit IronBackend(const flm::backend::BackendContext& context)
        : engine_(CheckpointPath(context.model_path),
                  ValidateContextLength(context.context_length)) {}
    causal_lm& engine() override { return engine_; }
    std::string id() const override { return flm::backend::kIronBackendId; }
    std::string detail() const override { return "IRON graph worker"; }
    bool supports_preemption() const override { return false; }
    bool poisoned() const noexcept override { return engine_.poisoned(); }

private:
    static std::string CheckpointPath(const std::string& fallback) {
        const char* path = std::getenv("FLM_IRON_MODEL_PATH");
        return path && *path ? path : fallback;
    }
    static std::uint32_t ValidateContextLength(std::uint32_t length) {
        if (length == 0 || length > 32768 || length % 2048 != 0) {
            throw std::out_of_range(
                "IRON Llama context length must be a multiple of 2048 in 2048..32768");
        }
        return length;
    }
    IronLlama engine_;
};

}  // namespace

flm::backend::BackendFactory iron_factory() {
    return [](const flm::backend::BackendContext& context) {
        return std::make_unique<IronBackend>(context);
    };
}

}  // namespace flm::llama3