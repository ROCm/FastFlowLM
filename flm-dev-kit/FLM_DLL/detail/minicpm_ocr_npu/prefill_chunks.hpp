// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file prefill_chunks.hpp
/// \brief Row chunks of one prefill call: an array run over one chunk's rows overlaps the host work on the other
///        chunk. README.md "Host prefill work (WP4)".
#pragma once

#include <algorithm>
#include <cstdlib>

namespace minicpm_ocr_chunks {

constexpr int max_chunks = 4;
#if defined(FLM_USE_HRX)
constexpr int allowed_chunks = 1;
#else
constexpr int allowed_chunks = max_chunks;
#endif
constexpr int default_chunks = std::min(3, allowed_chunks);
constexpr int step = 512;

/// \brief FLM_MINICPM_OCR_PREFILL_CHUNKS=1 runs every projection as one run, as before; default default_chunks. Read once.
///        HRX: always 1 (README.md "Row chunks on HRX").
inline int setting() {
    static const int n = [] {
        const char* env = std::getenv("FLM_MINICPM_OCR_PREFILL_CHUNKS");
        const int v = env != nullptr ? std::atoi(env) : default_chunks;
        return std::clamp(v, 1, allowed_chunks);
    }();
    return n;
}

struct layout {
    int count = 1;
    int begin[max_chunks] = {};
    int padded[max_chunks] = {};

    /// \brief splits L_padded rows into up to `wanted` chunks on 512-row steps, the first chunk the larger one
    void setup(int L_padded, int wanted) {
        const int rounds = L_padded / step;
        count = std::max(1, std::min(wanted, rounds));
        int done = 0;
        for (int c = 0; c < count; c++) {
            const int left = count - c;
            const int in_chunk = (rounds - done + left - 1) / left;
            begin[c] = done * step;
            padded[c] = in_chunk * step;
            done += in_chunk;
        }
    }

    /// \brief the real (not padding) rows of chunk c
    int rows(int c, int L) const {
        return std::max(0, std::min(L, begin[c] + padded[c]) - begin[c]);
    }
};

} // namespace minicpm_ocr_chunks
