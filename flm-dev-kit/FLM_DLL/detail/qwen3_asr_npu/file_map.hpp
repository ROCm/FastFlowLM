// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file file_map.hpp
/// \brief Read-only memory mapping of a whole file (POSIX mmap / Windows file mapping). Used by the host decoder and encoder to read the
///        q4nx weight files in place.
#pragma once
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace qwen3_asr_cpu {

class file_map {
public:
    explicit file_map(const std::string& path) {
#ifdef _WIN32
        // the path is UTF-8
        const int wlen = ::MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
        std::wstring wpath((size_t)(wlen > 0 ? wlen : 1), L'\0');
        ::MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, &wpath[0], wlen);
        file_ = ::CreateFileW(wpath.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (file_ == INVALID_HANDLE_VALUE) throw std::runtime_error("cannot open " + path);
        LARGE_INTEGER sz;
        if (!::GetFileSizeEx(file_, &sz)) { ::CloseHandle(file_); throw std::runtime_error("cannot stat " + path); }
        size_ = (size_t)sz.QuadPart;
        mapping_ = ::CreateFileMappingW(file_, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!mapping_) { ::CloseHandle(file_); throw std::runtime_error("cannot map " + path); }
        data_ = (const uint8_t*)::MapViewOfFile(mapping_, FILE_MAP_READ, 0, 0, 0);
        if (!data_) { ::CloseHandle(mapping_); ::CloseHandle(file_); throw std::runtime_error("cannot map " + path); }
#else
        const int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) throw std::runtime_error("cannot open " + path);
        struct stat sb;
        if (::fstat(fd, &sb) != 0) { ::close(fd); throw std::runtime_error("cannot stat " + path); }
        size_ = (size_t)sb.st_size;
        void* m = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0);
        ::close(fd);
        if (m == MAP_FAILED) throw std::runtime_error("mmap failed for " + path);
        data_ = (const uint8_t*)m;
#endif
    }
    ~file_map() {
#ifdef _WIN32
        if (data_) ::UnmapViewOfFile(data_);
        if (mapping_) ::CloseHandle(mapping_);
        if (file_ && file_ != INVALID_HANDLE_VALUE) ::CloseHandle(file_);
#else
        if (data_) ::munmap((void*)data_, size_);
#endif
    }
    file_map(const file_map&) = delete;
    file_map& operator=(const file_map&) = delete;

    const uint8_t* data() const { return data_; }
    size_t size() const { return size_; }

private:
    const uint8_t* data_ = nullptr;
    size_t size_ = 0;
#ifdef _WIN32
    HANDLE file_ = nullptr;
    HANDLE mapping_ = nullptr;
#endif
};

}  // namespace qwen3_asr_cpu
