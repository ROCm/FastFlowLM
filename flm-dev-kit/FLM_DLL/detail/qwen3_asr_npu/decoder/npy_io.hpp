// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file npy_io.hpp
/// \brief Minimal .npy reader (little-endian float32, C order) for the golden files.
#pragma once
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace qwen3_asr_cpu {

struct npy_f32 {
    std::vector<size_t> shape;
    std::vector<float> data;
};

inline npy_f32 read_npy_f32(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    char magic[6];
    f.read(magic, 6);
    uint8_t ver[2];
    f.read((char*)ver, 2);
    uint32_t header_len = 0;
    if (ver[0] == 1) { uint16_t h; f.read((char*)&h, 2); header_len = h; }
    else { f.read((char*)&header_len, 4); }
    std::string header(header_len, ' ');
    f.read(&header[0], header_len);
    if (header.find("<f4") == std::string::npos) throw std::runtime_error(path + ": only <f4 supported");
    if (header.find("'fortran_order': False") == std::string::npos) throw std::runtime_error(path + ": fortran order not supported");
    npy_f32 out;
    size_t lp = header.find('(', header.find("shape")), rp = header.find(')', lp);
    std::string dims = header.substr(lp + 1, rp - lp - 1);
    size_t pos = 0;
    while (pos < dims.size()) {
        while (pos < dims.size() && (dims[pos] == ' ' || dims[pos] == ',')) ++pos;
        if (pos >= dims.size()) break;
        size_t e = pos;
        while (e < dims.size() && std::isdigit((unsigned char)dims[e])) ++e;
        out.shape.push_back((size_t)std::stoull(dims.substr(pos, e - pos)));
        pos = e;
    }
    size_t n = 1;
    for (size_t d : out.shape) n *= d;
    out.data.resize(n);
    f.read((char*)out.data.data(), n * 4);
    return out;
}

}  // namespace qwen3_asr_cpu
