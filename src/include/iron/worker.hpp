#pragma once
// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#include "nlohmann/json.hpp"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace flm::iron {

// Wire protocol bits; keep in sync with IRON's iron/lm_server.py.
enum class Capability : std::uint32_t {
    CausalLM = 1 << 0,  // logits and reset
    Embeddings = 1 << 1,
};

class Worker {
public:
    Worker(const std::filesystem::path& model_path, std::string model,
           std::uint32_t context_length = 0);
    ~Worker();

    Worker(const Worker&) = delete;
    Worker& operator=(const Worker&) = delete;

    std::vector<std::uint16_t> logits(const std::vector<int>& tokens);
    std::vector<float> embed(const std::string& text, const std::string& task,
                             std::uint32_t dimensions = 768);
    void reset();
    bool supports(Capability capability) const noexcept {
        return (capabilities_ & static_cast<std::uint32_t>(capability)) != 0;
    }
    bool poisoned() const noexcept { return poisoned_; }

private:
    nlohmann::json request(const nlohmann::json& command,
                           const void* payload = nullptr,
                           std::size_t payload_size = 0);
    void write_all(const void* data, std::size_t size);
    void read_all(void* data, std::size_t size);

    struct Impl;
    std::unique_ptr<Impl> impl_;
    bool poisoned_ = false;
    std::uint32_t capabilities_ = 0;
};

}  // namespace flm::iron