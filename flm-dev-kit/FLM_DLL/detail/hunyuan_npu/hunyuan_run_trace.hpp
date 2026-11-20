// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file hunyuan_run_trace.hpp
/// \brief start / finish of every NPU run, for finding the run a hang sits in
/// \note  Build with `make RUN_TRACE=1 BUILD_DIR=...`; compiles to nothing otherwise.
///        The events go into a memory-mapped ring file (HY_RUN_TRACE, default
///        /tmp/hy_run_trace.<pid>), so the last runs can be read while the process
///        hangs. See "Run trace" in README.md.
#ifndef __HUNYUAN_RUN_TRACE_HPP__
#define __HUNYUAN_RUN_TRACE_HPP__

#ifdef HUNYUAN_RUN_TRACE
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

namespace hy_trace {

constexpr size_t SLOTS = 8192;
constexpr size_t SLOT_BYTES = 80;

struct state {
    char* ring = nullptr;
    unsigned long long seq = 0;
    int layer = -1;
    int L_begin = -1;
    int L = -1;
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
};

inline state& get() {
    static state s = [](){
        state n;
        const char* env = std::getenv("HY_RUN_TRACE");
        std::string path = env ? env : "/tmp/hy_run_trace." + std::to_string(getpid());
        int fd = open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
        if (fd < 0) return n;
        if (ftruncate(fd, SLOTS * SLOT_BYTES) == 0) {
            void* p = mmap(nullptr, SLOTS * SLOT_BYTES, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
            if (p != MAP_FAILED) {
                n.ring = static_cast<char*>(p);
                memset(n.ring, ' ', SLOTS * SLOT_BYTES);
            }
        }
        close(fd);
        return n;
    }();
    return s;
}

inline void where(int layer, int L_begin, int L) {
    state& s = get();
    s.layer = layer;
    s.L_begin = L_begin;
    s.L = L;
}

/// @param phase 'S' = submitted, 'D' = done (the wait returned)
/// @param st    the run's ert_cmd_state when the wait returned (4 = completed), -1 = not known
inline void event(const char* what, char phase, int st = -1) {
    state& s = get();
    if (!s.ring) return;
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - s.t0).count();
    char line[SLOT_BYTES + 1];
    int n = snprintf(line, sizeof(line), "%10llu %12.3f %-12s %c layer %3d L_begin %6d L %5d st %2d",
                     s.seq, ms, what, phase, s.layer, s.L_begin, s.L, st);
    if (n < 0) return;
    if ((size_t)n > SLOT_BYTES - 1) n = SLOT_BYTES - 1;
    memset(line + n, ' ', SLOT_BYTES - 1 - n);
    line[SLOT_BYTES - 1] = '\n';
    memcpy(s.ring + (s.seq % SLOTS) * SLOT_BYTES, line, SLOT_BYTES);
    s.seq++;
}

/// @brief test hook: HY_FAKE_FAIL_AT=<n>[,<n>...] makes the n-th checked run (0-based) report a
///        timeout (8), so the run-again path can be tested without a real failure
inline int fake_fail(int st) {
    static std::vector<long> at = [](){
        std::vector<long> v;
        const char* env = std::getenv("HY_FAKE_FAIL_AT");
        for (const char* p = env; p && *p; ) {
            v.push_back(std::strtol(p, nullptr, 10));
            p = std::strchr(p, ',');
            if (p) p++;
        }
        return v;
    }();
    static long n = 0;
    for (long a : at) {
        if (a == n) { n++; return 8; }
    }
    n++;
    return st;
}

} // namespace hy_trace

/// HY_TRACE(what, phase) or HY_TRACE(what, phase, state)
#define HY_TRACE(...) hy_trace::event(__VA_ARGS__)
#define HY_TRACE_WHERE(layer, L_begin, L) hy_trace::where(layer, L_begin, L)
#define HY_TRACE_FAKE_FAIL(st) ((ert_cmd_state)hy_trace::fake_fail((int)(st)))
#else
#define HY_TRACE_FAKE_FAIL(st) (st)
#define HY_TRACE(...) do {} while (0)
#define HY_TRACE_WHERE(layer, L_begin, L) do {} while (0)
#endif

#endif // __HUNYUAN_RUN_TRACE_HPP__
