#pragma once
// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#include "nlohmann/json.hpp"

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace flm::iron {

class Worker {
public:
        Worker(const std::filesystem::path& model_path, std::uint32_t context_length,
            std::string model = "llama");
    ~Worker();

    Worker(const Worker&) = delete;
    Worker& operator=(const Worker&) = delete;

    std::vector<std::uint16_t> logits(const std::vector<int>& tokens);
    std::vector<float> embed(const std::string& text, const std::string& task,
                             std::uint32_t dimensions = 768);
    void reset();
    bool poisoned() const noexcept { return poisoned_; }

private:
    nlohmann::json request(const nlohmann::json& command,
                           const void* payload = nullptr,
                           std::size_t payload_size = 0);
    void write_all(const void* data, std::size_t size);
    void read_all(void* data, std::size_t size);

    int input_ = -1;
    int output_ = -1;
    int process_ = -1;
    bool poisoned_ = false;
};

}  // namespace flm::iron