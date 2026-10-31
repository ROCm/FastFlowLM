// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#pragma once

#include "AutoModel/model_backend.hpp"
#include "iron/worker.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace flm::iron {

class CausalLM final : public ::causal_lm {
public:
    CausalLM(const std::string& model_path, std::string model,
             std::uint32_t context_length);

    buffer<bf16> forward(int id) override;
    buffer<bf16> prefill(std::vector<int>& ids, void*) override;
    void set_context_length(int length) override;
    void load_weights(Q4NX&) override;
    void update_max_length(std::uint32_t length) override;
    void clear_context() override;
    buffer<bf16> get_k_cache(int, int) override;
    buffer<bf16> get_v_cache(int, int) override;
    int get_current_context_length() override;
    int checkpoint() override;
    int restore() override;
    bool poisoned() const noexcept;

private:
    buffer<bf16> run();

    Worker worker_;
    std::vector<int> tokens_;
    std::uint32_t max_length_;
};

class CausalBackend final : public flm::backend::ModelBackend {
public:
    CausalBackend(const flm::backend::BackendContext& context,
                  std::string model);

    ::causal_lm& engine() override;
    std::string id() const override;
    std::string detail() const override;
    bool supports_preemption() const override;
    bool poisoned() const noexcept override;

private:
    CausalLM engine_;
};

flm::backend::BackendFactory causal_factory(std::string model);

inline flm::backend::BackendTraits causal_traits() {
    return flm::backend::BackendTraits{false, false, 0};
}

}  // namespace flm::iron