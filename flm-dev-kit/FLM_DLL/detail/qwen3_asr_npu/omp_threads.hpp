// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file omp_threads.hpp
/// \brief Thread count for the host AVX-512 kernels (encoder, prefill).
/// \note The int8/fp32 AVX-512 loops saturate the vector ports of a core with one thread; the SMT sibling only adds contention (measured
///       on a 12-core / 24-thread Ryzen AI 9 HX 370: 12 threads prefill 255 tokens in 0.50 s, 24 threads in 0.63 s). Default: half of the
///       logical CPUs (all of them below 8). QWEN3_ASR_THREADS overrides.
#pragma once
#include <cstdlib>
#include <omp.h>

namespace qwen3_asr_cpu {
inline int asr_threads() {
    static const int n = [] {
        if (const char* e = std::getenv("QWEN3_ASR_THREADS")) { const int v = std::atoi(e); if (v > 0) return v; }
        const int procs = omp_get_num_procs();
        return procs >= 8 ? procs / 2 : procs;
    }();
    return n;
}
}  // namespace qwen3_asr_cpu
