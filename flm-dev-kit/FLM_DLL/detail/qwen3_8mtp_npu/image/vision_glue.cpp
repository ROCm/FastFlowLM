// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file image/vision_glue.cpp
#include "image/vision_glue.hpp"
#include "image/vision_par.hpp"

#include "common/cpu_ref_common.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

namespace qwen38 {
namespace vision {

void layer_norm(const float* x, int L, int D,
                const float* gain, const float* bias, float eps, float* y) {
#pragma omp parallel for schedule(static) num_threads(qwen38::vision::par_threads())
    for (int t = 0; t < L; t++) {
        const float* xr = x + (size_t)t * D;
        float*       yr = y + (size_t)t * D;

        double mean = 0.0;
        for (int i = 0; i < D; i++) mean += xr[i];
        mean /= (double)D;

        double var = 0.0;
        for (int i = 0; i < D; i++) {
            const double d = (double)xr[i] - mean;
            var += d * d;
        }
        // Biased variance, matching torch.nn.LayerNorm.
        var /= (double)D;

        const float m   = (float)mean;
        const float inv = 1.0f / std::sqrt((float)var + eps);
        for (int i = 0; i < D; i++)
            yr[i] = (xr[i] - m) * inv * gain[i] + (bias ? bias[i] : 0.0f);
    }
}

void pos_embed_interpolate(const uint16_t* table, int num_grid_per_side, int D,
                           int grid_t, int grid_h, int grid_w,
                           int spatial_merge_size, float* out) {
    const int m = spatial_merge_size;
    if (grid_h % m || grid_w % m)
        throw std::runtime_error("pos_embed_interpolate: grid is not a multiple "
                                 "of spatial_merge_size");
    const int N = num_grid_per_side;

    // linspace(0, N-1, n), then floor/ceil with the ceil clamped to the last
    // row. A grid of 1 collapses to index 0 with weight 1, which is why the
    // denominator is max(n-1, 1) rather than n-1.
    auto axis = [&](int n, std::vector<int>& lo, std::vector<int>& hi,
                    std::vector<float>& frac) {
        lo.resize((size_t)n); hi.resize((size_t)n); frac.resize((size_t)n);
        const float den = (float)std::max(n - 1, 1);
        for (int j = 0; j < n; j++) {
            const float v = (float)j * (float)(N - 1) / den;
            const int   f = (int)v;
            lo[(size_t)j]   = f;
            hi[(size_t)j]   = std::min(f + 1, N - 1);
            frac[(size_t)j] = v - (float)f;
        }
    };

    std::vector<int>   hlo, hhi, wlo, whi;
    std::vector<float> dh, dw;
    axis(grid_h, hlo, hhi, dh);
    axis(grid_w, wlo, whi, dw);

    const int h_blocks = grid_h / m;
    const int w_blocks = grid_w / m;
    const int hw       = grid_h * grid_w;

    // Walk the output in merge-block order so `out` lines up with
    // pixel_values row-for-row.
#pragma omp parallel for schedule(static) num_threads(qwen38::vision::par_threads())
    for (int blk = 0; blk < h_blocks * w_blocks; blk++) {
        const int by = blk / w_blocks;
        const int bx = blk % w_blocks;
        for (int dy = 0; dy < m; dy++) {
            for (int dx = 0; dx < m; dx++) {
                const int hi_ = by * m + dy;
                const int wi_ = bx * m + dx;
                const int s   = ((by * w_blocks + bx) * m + dy) * m + dx;

                const int   idx[4] = {
                    hlo[(size_t)hi_] * N + wlo[(size_t)wi_],
                    hlo[(size_t)hi_] * N + whi[(size_t)wi_],
                    hhi[(size_t)hi_] * N + wlo[(size_t)wi_],
                    hhi[(size_t)hi_] * N + whi[(size_t)wi_],
                };
                const float a = dh[(size_t)hi_], b = dw[(size_t)wi_];
                const float wt[4] = {
                    (1.0f - a) * (1.0f - b),
                    (1.0f - a) * b,
                    a * (1.0f - b),
                    a * b,
                };

                float* o = out + (size_t)s * D;
                for (int d = 0; d < D; d++) {
                    float acc = 0.0f;
                    for (int c = 0; c < 4; c++)
                        acc += wt[c] * cpu_ref::bf16_to_f32(
                                           table[(size_t)idx[c] * D + d]);
                    o[d] = acc;
                }
            }
        }
    }

    // Every frame sees the same spatial grid.
    for (int t = 1; t < grid_t; t++)
        std::memcpy(out + (size_t)t * hw * D, out,
                    sizeof(float) * (size_t)hw * D);
}

void add_inplace(float* x, const float* y, size_t n) {
    // Two per block, 27 blocks, 1.38M elements each: 20 ms of a 644 ms encode
    // on one core. The body is already vectorised by gcc; only the width was
    // missing. A chunk rather than the raw index so the split is coarse.
    constexpr size_t CHUNK = 1u << 16;
    // Signed index: MSVC's /openmp is OpenMP 2.0 and rejects an unsigned loop
    // variable (C3016). See vision_bfp16.hpp's chunk loops.
    const int64_t nchunk = (int64_t)((n + CHUNK - 1) / CHUNK);
#pragma omp parallel for schedule(static) num_threads(qwen38::vision::par_threads())
    for (int64_t c = 0; c < nchunk; c++) {
        const size_t i0 = (size_t)c * CHUNK, i1 = std::min(n, i0 + CHUNK);
        for (size_t i = i0; i < i1; i++) x[i] += y[i];
    }
}

}  // namespace vision
}  // namespace qwen38
