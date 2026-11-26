// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file inkling2_q4nx_file.hpp
/// \brief the safetensors container behind model.q4nx and vision_weight.q4nx, mapped whole, tensors located in place
#pragma once
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "inkling2_host.hpp"
#include "nlohmann/json.hpp"
#include "tensor_utils/mapped_file.hpp"

namespace inkling2_host {

class q4nx_file {
public:
    struct tensor {
        std::string dtype;
        std::vector<int64_t> shape;
        const u8* data = nullptr;
        size_t bytes = 0;
    };

    void open(const std::string& path) {
        path_ = path;
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("inkling2: cannot open " + path);
        uint64_t n = 0;
        f.read(reinterpret_cast<char*>(&n), 8);
        std::string header(n, '\0');
        f.read(header.data(), (std::streamsize)n);
        if (!f) throw std::runtime_error("inkling2: truncated header in " + path);
        const size_t size = std::filesystem::file_size(path);
        if (!map_.open(path, 0, size)) throw std::runtime_error("inkling2: cannot map " + path);
        const u8* base = map_.data() + 8 + n;
        const auto j = nlohmann::json::parse(header);
        for (auto it = j.begin(); it != j.end(); ++it) {
            if (it.key() == "__metadata__") continue;
            tensor t;
            t.dtype = it.value()["dtype"].get<std::string>();
            t.shape = it.value()["shape"].get<std::vector<int64_t>>();
            const auto off = it.value()["data_offsets"].get<std::vector<size_t>>();
            if (8 + n + off[1] > size) throw std::runtime_error("inkling2: tensor " + it.key() + " past the end of the file");
            t.data = base + off[0];
            t.bytes = off[1] - off[0];
            tensors_[it.key()] = t;
        }
    }

    bool has(const std::string& name) const { return tensors_.count(name) != 0; }

    /// \brief the tensor of this name, whatever its dtype and shape
    const tensor& find(const std::string& name) const {
        const auto it = tensors_.find(name);
        if (it == tensors_.end()) throw std::runtime_error("inkling2: " + path_ + " has no " + name);
        return it->second;
    }

    const tensor& get(const std::string& name, const std::string& dtype, const std::vector<int64_t>& shape) const {
        const auto it = tensors_.find(name);
        if (it == tensors_.end()) throw std::runtime_error("inkling2: " + path_ + " has no " + name);
        if (it->second.dtype != dtype || it->second.shape != shape) {
            std::string s;
            for (auto d : it->second.shape) s += std::to_string(d) + ",";
            throw std::runtime_error("inkling2: " + name + " is " + it->second.dtype + " (" + s + "), expected " + dtype);
        }
        return it->second;
    }

    const float* f32(const std::string& name, const std::vector<int64_t>& shape) const {
        return reinterpret_cast<const float*>(get(name, "F32", shape).data);
    }
    const u16* bf16(const std::string& name, const std::vector<int64_t>& shape) const {
        return reinterpret_cast<const u16*>(get(name, "BF16", shape).data);
    }
    qmat q(const std::string& name, qfmt fmt, int N, int K) const {
        qmat m{fmt, N, K, nullptr};
        m.data = get(name, "I8", {N / QROWS, K / QCOLS, (int64_t)m.block_bytes()}).data;
        return m;
    }

private:
    std::string path_;
    mapped_file map_;
    std::map<std::string, tensor> tensors_;
};

}  // namespace inkling2_host
