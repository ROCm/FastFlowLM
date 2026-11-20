// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file image/vision_common.hpp
/// \brief The host-side plumbing the vision tower stands on: bf16 bit
///        conversions, a read-only mmap'd safetensors reader, the error
///        statistics the verify driver prints, and the vectorised fp32<->bf16
///        marshalling every dispatch does.
///
/// Ported from `qwen3_8mtp_npu/common/cpu_ref_common.hpp` and
/// `common/cpu_simd.hpp`, cut down to what the tower uses. It is a COPY under a
/// different namespace on purpose: every engine `.so` carries its own runtime
/// and `flm` loads all of them at once, so two engines that share a header by
/// `#include` across `detail/` dirs would also share inline-function symbols the
/// moment one of them changes (see the `-Wl,-Bsymbolic-functions` comment in
/// this engine's Makefile: without it, ELF interposition can bind even this
/// .so's internal calls to another engine's copy of the same function).
#ifndef __MINICPM_V_VISION_COMMON_HPP__
#define __MINICPM_V_VISION_COMMON_HPP__
#pragma once

#include "typedef.hpp"   // bf16

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#else
#  include <fcntl.h>
#  include <unistd.h>
#  include <sys/mman.h>
#  include <sys/stat.h>
#endif

#if defined(__AVX512F__) && defined(__AVX512DQ__)
#include <immintrin.h>
#define MINICPM_V_CPU_SIMD 1
#endif

namespace mcv {
namespace ref {

// --------------------------------------------------------------------------
// bf16
// --------------------------------------------------------------------------
inline float bf16_to_f32(uint16_t h) {
    uint32_t u = static_cast<uint32_t>(h) << 16;
    float f; std::memcpy(&f, &u, 4); return f;
}

/// Round-to-nearest-even, matching torch.Tensor.bfloat16().
inline uint16_t f32_to_bf16(float f) {
    uint32_t u; std::memcpy(&u, &f, 4);
    uint32_t lsb = (u >> 16) & 1u;
    u += 0x7fffu + lsb;
    return static_cast<uint16_t>(u >> 16);
}

inline float round_bf16(float f) { return bf16_to_f32(f32_to_bf16(f)); }

// --------------------------------------------------------------------------
// Minimal read-only SafeTensors reader
//
//   | u64 header_len | header_len bytes of JSON | payload |
//
// vision_weight.q4nx is ordinary safetensors despite the extension, and every
// tensor in it is BF16, so the tower keeps raw pointers INTO the mapping rather
// than copying 1.1 GB.
// --------------------------------------------------------------------------
struct tensor_info {
    std::string          dtype;
    std::vector<int64_t> shape;
    size_t               off0 = 0, off1 = 0;   // relative to payload start

    int64_t numel() const {
        int64_t n = 1; for (int64_t s : shape) n *= s; return n;
    }
};

class safetensors {
public:
    explicit safetensors(const std::string& path) : path_(path) {
#ifdef _WIN32
        file_ = ::CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ,
                              nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL,
                              nullptr);
        if (file_ == INVALID_HANDLE_VALUE)
            throw std::runtime_error("cannot open " + path);
        LARGE_INTEGER sz{};
        if (!::GetFileSizeEx(file_, &sz))
            throw std::runtime_error("GetFileSizeEx " + path);
        size_ = static_cast<size_t>(sz.QuadPart);
        map_ = ::CreateFileMappingA(file_, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!map_) throw std::runtime_error("CreateFileMapping " + path);
        base_ = static_cast<const uint8_t*>(
            ::MapViewOfFile(map_, FILE_MAP_READ, 0, 0, 0));
        if (!base_) throw std::runtime_error("MapViewOfFile " + path);
#else
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0) throw std::runtime_error("cannot open " + path);
        struct stat st{};
        if (::fstat(fd_, &st) != 0) throw std::runtime_error("fstat " + path);
        size_ = static_cast<size_t>(st.st_size);
        base_ = static_cast<const uint8_t*>(
            ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0));
        if (base_ == MAP_FAILED) throw std::runtime_error("mmap " + path);
#endif
        uint64_t hlen; std::memcpy(&hlen, base_, 8);
        payload_ = base_ + 8 + hlen;
        parse_header(reinterpret_cast<const char*>(base_ + 8), hlen);
    }

    ~safetensors() {
#ifdef _WIN32
        if (base_) ::UnmapViewOfFile(const_cast<uint8_t*>(base_));
        if (map_) ::CloseHandle(map_);
        if (file_ != INVALID_HANDLE_VALUE) ::CloseHandle(file_);
#else
        if (base_ && base_ != MAP_FAILED) ::munmap(const_cast<uint8_t*>(base_), size_);
        if (fd_ >= 0) ::close(fd_);
#endif
    }
    safetensors(const safetensors&) = delete;
    safetensors& operator=(const safetensors&) = delete;

    bool has(const std::string& name) const { return tensors_.count(name) != 0; }

    const tensor_info& info(const std::string& name) const {
        auto it = tensors_.find(name);
        if (it == tensors_.end())
            throw std::runtime_error("no tensor '" + name + "' in " + path_);
        return it->second;
    }

    const uint8_t* raw(const std::string& name) const {
        return payload_ + info(name).off0;
    }

    /// Read a BF16 / F32 tensor out as fp32.
    std::vector<float> f32(const std::string& name) const {
        const tensor_info& t = info(name);
        const size_t n = static_cast<size_t>(t.numel());
        std::vector<float> out(n);
        const uint8_t* p = payload_ + t.off0;
        if (t.dtype == "F32") {
            std::memcpy(out.data(), p, n * 4);
        } else if (t.dtype == "BF16") {
            const uint16_t* h = reinterpret_cast<const uint16_t*>(p);
            for (size_t i = 0; i < n; i++) out[i] = bf16_to_f32(h[i]);
        } else {
            throw std::runtime_error("f32(): unsupported dtype " + t.dtype +
                                     " for '" + name + "'");
        }
        return out;
    }

    std::vector<int32_t> i32(const std::string& name) const {
        const tensor_info& t = info(name);
        if (t.dtype != "I32")
            throw std::runtime_error("i32(): '" + name + "' is " + t.dtype);
        const size_t n = static_cast<size_t>(t.numel());
        std::vector<int32_t> out(n);
        std::memcpy(out.data(), payload_ + t.off0, n * 4);
        return out;
    }

    const std::map<std::string, tensor_info>& tensors() const { return tensors_; }

private:
    void parse_header(const char* p, size_t n) {
        size_t i = 0;
        auto ws = [&] { while (i < n && (p[i]==' '||p[i]=='\n'||p[i]=='\t'||p[i]=='\r')) i++; };
        auto str = [&]() -> std::string {
            ws();
            if (p[i] != '"') throw std::runtime_error("safetensors header: expected string");
            i++;
            std::string s;
            while (i < n && p[i] != '"') {
                if (p[i] == '\\' && i + 1 < n) { s.push_back(p[i+1]); i += 2; }
                else s.push_back(p[i++]);
            }
            i++;
            return s;
        };
        auto skip_value = [&]() {
            ws();
            if (p[i] == '"') { str(); return; }
            if (p[i] == '{' || p[i] == '[') {
                int depth = 0; bool in_str = false;
                for (; i < n; i++) {
                    char c = p[i];
                    if (in_str) { if (c == '\\') i++; else if (c == '"') in_str = false; continue; }
                    if (c == '"') in_str = true;
                    else if (c == '{' || c == '[') depth++;
                    else if (c == '}' || c == ']') { depth--; if (depth == 0) { i++; return; } }
                }
                return;
            }
            while (i < n && p[i] != ',' && p[i] != '}' && p[i] != ']') i++;
        };
        auto num = [&]() -> int64_t {
            ws();
            char* end = nullptr;
            long long v = std::strtoll(p + i, &end, 10);
            i = static_cast<size_t>(end - p);
            return v;
        };

        ws();
        if (p[i] != '{') throw std::runtime_error("safetensors header: expected object");
        i++;
        while (true) {
            ws();
            if (i >= n || p[i] == '}') break;
            std::string key = str();
            ws(); if (p[i] == ':') i++;
            if (key == "__metadata__") {
                skip_value();
            } else {
                ws();
                if (p[i] != '{') throw std::runtime_error("safetensors header: bad entry " + key);
                i++;
                tensor_info t;
                while (true) {
                    ws();
                    if (i >= n || p[i] == '}') { i++; break; }
                    std::string f = str();
                    ws(); if (p[i] == ':') i++;
                    if (f == "dtype") {
                        t.dtype = str();
                    } else if (f == "shape") {
                        ws(); i++;
                        while (true) {
                            ws();
                            if (p[i] == ']') { i++; break; }
                            t.shape.push_back(num());
                            ws(); if (p[i] == ',') i++;
                        }
                    } else if (f == "data_offsets") {
                        ws(); i++;
                        t.off0 = static_cast<size_t>(num()); ws(); if (p[i] == ',') i++;
                        t.off1 = static_cast<size_t>(num()); ws(); if (p[i] == ']') i++;
                    } else {
                        skip_value();
                    }
                    ws(); if (p[i] == ',') i++;
                }
                tensors_.emplace(std::move(key), std::move(t));
            }
            ws(); if (i < n && p[i] == ',') i++;
        }
    }

    std::string                        path_;
#ifdef _WIN32
    HANDLE                             file_   = INVALID_HANDLE_VALUE;
    HANDLE                             map_    = nullptr;
#else
    int                                fd_     = -1;
#endif
    size_t                             size_   = 0;
    const uint8_t*                     base_   = nullptr;
    const uint8_t*                     payload_ = nullptr;
    std::map<std::string, tensor_info> tensors_;
};

// --------------------------------------------------------------------------
// error statistics -- the same four numbers every verify driver prints
// --------------------------------------------------------------------------
struct err_stats { double rel_l1, rel_l2, cosine, rmse; };

inline err_stats error_matrix(const float* a, const float* b, size_t n) {
    double d1 = 0, r1 = 0, d2 = 0, r2 = 0, dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < n; i++) {
        const double x = a[i], y = b[i], d = x - y;
        d1 += std::fabs(d); r1 += std::fabs(y);
        d2 += d * d;        r2 += y * y;
        dot += x * y; na += x * x; nb += y * y;
    }
    err_stats s;
    s.rel_l1 = r1 > 0 ? d1 / r1 : 0.0;
    s.rel_l2 = r2 > 0 ? std::sqrt(d2 / r2) : 0.0;
    s.cosine = (na > 0 && nb > 0) ? dot / (std::sqrt(na) * std::sqrt(nb)) : 0.0;
    s.rmse   = std::sqrt(d2 / static_cast<double>(n));
    return s;
}

}  // namespace ref

// --------------------------------------------------------------------------
// fp32 <-> bf16 marshalling, vectorised
//
// `biovault::bfloat16_t(float)` is an fpclassify switch gcc cannot vectorise;
// these are the same function on 16 lanes. `_cvt_f32_bf16` reproduces the
// constructor's flush of zero/denormal inputs to a signed zero, which is what
// makes it bit-identical to it rather than merely close (see Qwen3.8's
// common/cpu_simd.hpp, where that is checked exhaustively).
// --------------------------------------------------------------------------
namespace simd {

#ifdef MINICPM_V_CPU_SIMD
static inline __m256i _cvt_f32_bf16(__m512 v) {
    const __m512i u     = _mm512_castps_si512(v);
    const __m512i trunc = _mm512_srli_epi32(u, 16);
    const __m512i lsb   = _mm512_and_si512(trunc, _mm512_set1_epi32(1));
    __m512i r = _mm512_srli_epi32(
        _mm512_add_epi32(_mm512_add_epi32(u, _mm512_set1_epi32(0x7FFF)), lsb), 16);
    const __mmask16 zero_or_denorm = _mm512_fpclass_ps_mask(v, 0x02 | 0x04 | 0x20);
    r = _mm512_mask_and_epi32(r, zero_or_denorm, trunc, _mm512_set1_epi32(0x8000));
    return _mm512_cvtepi32_epi16(r);
}

static inline __m512 _cvt_bf16_f32(__m256i b) {
    return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(b), 16));
}
#endif

/// `dst[i] = (bf16)src[i]`.
inline void f32_to_bf16(bf16* dst, const float* src, size_t n) {
    size_t i = 0;
#ifdef MINICPM_V_CPU_SIMD
    for (; i + 16 <= n; i += 16)
        _mm256_storeu_si256((__m256i*)(dst + i), _cvt_f32_bf16(_mm512_loadu_ps(src + i)));
#endif
    for (; i < n; i++) dst[i] = (bf16)src[i];
}

/// `dst[i] = (float)src[i]`.
inline void bf16_to_f32(float* dst, const bf16* src, size_t n) {
    size_t i = 0;
#ifdef MINICPM_V_CPU_SIMD
    for (; i + 16 <= n; i += 16)
        _mm512_storeu_ps(dst + i, _cvt_bf16_f32(_mm256_loadu_si256((const __m256i*)(src + i))));
#endif
    for (; i < n; i++) dst[i] = (float)src[i];
}

}  // namespace simd
}  // namespace mcv

#endif  // __MINICPM_V_VISION_COMMON_HPP__
