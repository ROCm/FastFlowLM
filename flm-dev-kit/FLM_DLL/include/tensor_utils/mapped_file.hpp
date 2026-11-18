// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file mapped_file.hpp
/// \brief Read-only file-backed mapping of a byte range, for lookup-only tensors.
/// \note A tensor that is read one row per token (token / per-layer embeddings) does not
///       need an owned copy: mapping the range keeps only the touched pages resident, and
///       the OS can drop and re-read them under memory pressure.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

class mapped_file {
private:
    void* base = nullptr;          // start of the mapped view (page aligned)
    const uint8_t* ptr = nullptr;  // start of the requested range
    size_t len = 0;
#ifdef _WIN32
    HANDLE file_mapping = nullptr;
#endif

    void release() {
#ifdef _WIN32
        if (this->base) UnmapViewOfFile(this->base);
        if (this->file_mapping) CloseHandle(this->file_mapping);
        this->file_mapping = nullptr;
#else
        if (this->base) munmap(this->base, this->map_len);
#endif
        this->base = nullptr;
        this->ptr = nullptr;
        this->len = 0;
    }
    size_t map_len = 0;

public:
    mapped_file() = default;
    mapped_file(const mapped_file&) = delete;
    mapped_file& operator=(const mapped_file&) = delete;
    mapped_file(mapped_file&& o) noexcept { *this = std::move(o); }
    mapped_file& operator=(mapped_file&& o) noexcept {
        if (this != &o) {
            this->release();
            this->base = o.base; this->ptr = o.ptr; this->len = o.len; this->map_len = o.map_len;
#ifdef _WIN32
            this->file_mapping = o.file_mapping; o.file_mapping = nullptr;
#endif
            o.base = nullptr; o.ptr = nullptr; o.len = 0; o.map_len = 0;
        }
        return *this;
    }
    ~mapped_file() { this->release(); }

    /// \brief Map [offset, offset + length) of the file read-only.
    /// \return false when the range cannot be mapped (the caller falls back to a copy)
    bool open(const std::string& path, size_t offset, size_t length) {
        this->release();
        if (length == 0) return false;
#ifdef _WIN32
        SYSTEM_INFO si; GetSystemInfo(&si);
        const size_t gran = si.dwAllocationGranularity;
        const size_t lead = offset % gran;
        const size_t start = offset - lead;
        HANDLE f = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                               FILE_ATTRIBUTE_NORMAL, nullptr);
        if (f == INVALID_HANDLE_VALUE) return false;
        // a range past the end of a truncated file maps fine and faults on first use: refuse it
        LARGE_INTEGER file_size;
        if (!GetFileSizeEx(f, &file_size) || (unsigned long long)(offset + length) > (unsigned long long)file_size.QuadPart) {
            CloseHandle(f);
            return false;
        }
        this->file_mapping = CreateFileMappingA(f, nullptr, PAGE_READONLY, 0, 0, nullptr);
        CloseHandle(f);  // the mapping object keeps the file alive
        if (!this->file_mapping) return false;
        this->map_len = lead + length;
        this->base = MapViewOfFile(this->file_mapping, FILE_MAP_READ, (DWORD)(start >> 32),
                                   (DWORD)(start & 0xffffffffu), this->map_len);
#else
        const size_t gran = (size_t)sysconf(_SC_PAGESIZE);
        const size_t lead = offset % gran;
        const size_t start = offset - lead;
        int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) return false;
        // a range past the end of a truncated file maps fine and SIGBUSes on first use: refuse it
        struct stat st;
        if (fstat(fd, &st) != 0 || (unsigned long long)(offset + length) > (unsigned long long)st.st_size) {
            ::close(fd);
            return false;
        }
        this->map_len = lead + length;
        void* m = mmap(nullptr, this->map_len, PROT_READ, MAP_PRIVATE, fd, (off_t)start);
        ::close(fd);  // the mapping stays valid after close
        this->base = (m == MAP_FAILED) ? nullptr : m;
#endif
        if (!this->base) { this->release(); return false; }
        this->ptr = static_cast<const uint8_t*>(this->base) + lead;
        this->len = length;
        return true;
    }

    /// \brief Hint that [offset, offset + length) of the range is about to be read, so a cold
    ///        page cache is filled in one batched read instead of one page fault at a time.
    void will_need(size_t offset, size_t length) const {
        if (!this->ptr || offset >= this->len) return;
        length = (offset + length > this->len) ? this->len - offset : length;
#ifdef _WIN32
        WIN32_MEMORY_RANGE_ENTRY range;
        range.VirtualAddress = const_cast<uint8_t*>(this->ptr) + offset;
        range.NumberOfBytes = length;
        PrefetchVirtualMemory(GetCurrentProcess(), 1, &range, 0);
#else
        const size_t gran = (size_t)sysconf(_SC_PAGESIZE);
        const uintptr_t begin = (uintptr_t)(this->ptr + offset) & ~(uintptr_t)(gran - 1);
        const uintptr_t end = (uintptr_t)(this->ptr + offset + length);
        madvise((void*)begin, end - begin, MADV_WILLNEED);
#endif
    }

    const uint8_t* data() const { return this->ptr; }
    size_t size() const { return this->len; }
    explicit operator bool() const { return this->ptr != nullptr; }
};
