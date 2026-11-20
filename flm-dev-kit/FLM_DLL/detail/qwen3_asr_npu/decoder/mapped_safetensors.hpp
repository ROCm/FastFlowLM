// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file mapped_safetensors.hpp
/// \brief Read-only, memory-mapped view of a .safetensors / .q4nx file (tensors are used in place, nothing is copied).
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
#include "nlohmann/json.hpp"
#include "../file_map.hpp"

namespace qwen3_asr_cpu {

struct mapped_tensor {
    std::string dtype;               // "BF16", "I8", "F32", ...
    std::vector<size_t> shape;
    const uint8_t* data = nullptr;   // points into the mapping
    size_t bytes = 0;
};

class mapped_safetensors {
public:
    explicit mapped_safetensors(const std::string& path) : map_(path) {
        base_ = map_.data();
        uint64_t header_len = 0;
        std::memcpy(&header_len, base_, 8);
        auto header = nlohmann::json::parse(std::string((const char*)base_ + 8, header_len));
        const uint8_t* data_start = base_ + 8 + header_len;
        for (auto it = header.begin(); it != header.end(); ++it) {
            if (it.key() == "__metadata__") continue;
            mapped_tensor t;
            t.dtype = it.value()["dtype"].get<std::string>();
            for (auto& d : it.value()["shape"]) t.shape.push_back(d.get<size_t>());
            size_t b = it.value()["data_offsets"][0].get<size_t>();
            size_t e = it.value()["data_offsets"][1].get<size_t>();
            t.data = data_start + b;
            t.bytes = e - b;
            tensors_[it.key()] = std::move(t);
        }
    }
    mapped_safetensors(const mapped_safetensors&) = delete;
    mapped_safetensors& operator=(const mapped_safetensors&) = delete;

    bool has(const std::string& name) const { return tensors_.count(name) != 0; }
    const mapped_tensor& get(const std::string& name) const {
        auto it = tensors_.find(name);
        if (it == tensors_.end()) throw std::runtime_error("tensor not found: " + name);
        return it->second;
    }
    size_t count() const { return tensors_.size(); }

private:
    file_map map_;
    const uint8_t* base_ = nullptr;
    std::map<std::string, mapped_tensor> tensors_;
};

}  // namespace qwen3_asr_cpu
