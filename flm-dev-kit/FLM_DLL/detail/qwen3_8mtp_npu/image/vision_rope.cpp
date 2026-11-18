// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file image/vision_rope.cpp
#include "image/vision_rope.hpp"
#include "image/vision_par.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace qwen38 {
namespace vision {

rope_tables build_rope(int grid_t, int grid_h, int grid_w,
                       int spatial_merge_size, int head_dim, float theta) {
    if (head_dim % 4 != 0)
        throw std::runtime_error("vision head_dim " + std::to_string(head_dim) +
                                 " is not a multiple of 4");
    const int m = spatial_merge_size;
    if (grid_h % m || grid_w % m)
        throw std::runtime_error(
            "vision grid " + std::to_string(grid_h) + "x" + std::to_string(grid_w) +
            " is not a multiple of spatial_merge_size " + std::to_string(m) +
            " -- the merge-block walk would run off the end");

    const int rot_dim  = head_dim / 2;
    const int rot_half = rot_dim / 2;
    const int hw       = grid_h * grid_w;
    const int tokens   = grid_t * hw;

    // freqs[p][i] = p / theta^(2i/rot_dim), for every coordinate the grid can
    // reach on either axis.
    const int max_grid = grid_h > grid_w ? grid_h : grid_w;
    std::vector<float> inv_freq((size_t)rot_half);
    for (int i = 0; i < rot_half; i++)
        inv_freq[(size_t)i] =
            1.0f / std::pow(theta, (float)(2 * i) / (float)rot_dim);

    std::vector<float> freqs((size_t)max_grid * rot_half);
    for (int p = 0; p < max_grid; p++)
        for (int i = 0; i < rot_half; i++)
            freqs[(size_t)p * rot_half + i] = (float)p * inv_freq[(size_t)i];

    rope_tables out;
    out.tokens   = tokens;
    out.head_dim = head_dim;
    out.cos.resize((size_t)tokens * head_dim);
    out.sin.resize((size_t)tokens * head_dim);

    const int h_blocks = grid_h / m;
    const int w_blocks = grid_w / m;

    int s = 0;   // token index, walked in merge-block order
    for (int by = 0; by < h_blocks; by++) {
        for (int bx = 0; bx < w_blocks; bx++) {
            for (int dy = 0; dy < m; dy++) {
                for (int dx = 0; dx < m; dx++, s++) {
                    const int hp = by * m + dy;
                    const int wp = bx * m + dx;
                    const float* fh = freqs.data() + (size_t)hp * rot_half;
                    const float* fw = freqs.data() + (size_t)wp * rot_half;

                    float* c = out.cos.data() + (size_t)s * head_dim;
                    float* n = out.sin.data() + (size_t)s * head_dim;
                    for (int i = 0; i < rot_half; i++) {
                        // emb = cat(rotary, rotary), so index i and i+rot_dim
                        // hold the same angle.
                        const float a = fh[i];
                        c[i] = c[i + rot_dim] = std::cos(a);
                        n[i] = n[i + rot_dim] = std::sin(a);
                    }
                    for (int i = 0; i < rot_half; i++) {
                        const float a = fw[i];
                        const int   j = rot_half + i;
                        c[j] = c[j + rot_dim] = std::cos(a);
                        n[j] = n[j + rot_dim] = std::sin(a);
                    }
                }
            }
        }
    }

    // Frames share the spatial coordinates; the table simply repeats.
    for (int t = 1; t < grid_t; t++) {
        const size_t n = (size_t)hw * head_dim;
        std::copy(out.cos.begin(), out.cos.begin() + (long)n,
                  out.cos.begin() + (long)((size_t)t * n));
        std::copy(out.sin.begin(), out.sin.begin() + (long)n,
                  out.sin.begin() + (long)((size_t)t * n));
    }

    return out;
}

void apply_rope(float* x, int L, int heads, const rope_tables& tables) {
    if (L > tables.tokens)
        throw std::runtime_error("apply_rope: L exceeds the built table");

    const int dh   = tables.head_dim;
    const int half = dh / 2;
    const int D    = heads * dh;

#pragma omp parallel for schedule(static) num_threads(qwen38::vision::par_threads())
    for (int t = 0; t < L; t++) {
        const float* c = tables.cos.data() + (size_t)t * dh;
        const float* s = tables.sin.data() + (size_t)t * dh;
        for (int h = 0; h < heads; h++) {
            float* p = x + (size_t)t * D + (size_t)h * dh;
            for (int d = 0; d < half; d++) {
                // rotate_half(q) = [-q2, q1]
                const float q1 = p[d];
                const float q2 = p[d + half];
                p[d]        = q1 * c[d]        - q2 * s[d];
                p[d + half] = q2 * c[d + half] + q1 * s[d + half];
            }
        }
    }
}

}  // namespace vision
}  // namespace qwen38
