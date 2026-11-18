// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file image/vision_glue.cpp
#include "image/vision_glue.hpp"
#include "image/vision_common.hpp"
#include "image/vision_par.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace mco {
namespace vision {

void layer_norm(const float* x, int L, int D,
                const float* gain, const float* bias, float eps, float* y) {
#pragma omp parallel for schedule(static) num_threads(mco::vision::par_threads())
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
        var /= (double)D;   // biased, as torch.nn.LayerNorm

        const float m   = (float)mean;
        const float inv = 1.0f / std::sqrt((float)var + eps);
        for (int i = 0; i < D; i++)
            yr[i] = (xr[i] - m) * inv * gain[i] + (bias ? bias[i] : 0.0f);
    }
}

namespace {

/// `torch.arange(start, end, step)` for float32, value for value.
///
/// ATen's CPU kernel computes in double (`acc_type<float>`) but its vectorised
/// arm is `Vectorized<float>::arange(float(start + step*b), step)`: the block
/// base is rounded to float FIRST and the lanes are `base + j*step` from there.
/// `cpu_serial_kernel_vec` runs that arm two 8-lane vectors at a time and the
/// remainder below 16 elements through the scalar `start + step*i`. Measured
/// against the installed torch (2.13, AVX512 capability) over every n < 600:
/// zero value mismatches with (8 lanes, x2 unroll), hundreds with any other
/// width. The difference only matters on exact bucket ties, but there it does.
std::vector<float> torch_arange_f32(double start, double end, double step) {
    const int64_t cnt = (int64_t)std::ceil((end - start) / step);
    std::vector<float> out((size_t)std::max<int64_t>(cnt, 0));
    constexpr int64_t V = 8, U = 2;
    const int64_t full = (cnt / (V * U)) * (V * U);
    // The products go through volatiles: this file builds with -ffast-math,
    // whose -ffp-contract=fast would fuse `start + step*b` into an fma and round
    // once where torch rounds twice -- exactly the last-bit difference that
    // decides a bucket tie.
    for (int64_t i = 0; i < cnt; i++) {
        if (i < full) {
            const int64_t b = (i / V) * V;
            volatile double pb = step * (double)b;
            const double base = (double)(float)(start + pb);
            volatile double pj = step * (double)(i - b);
            out[(size_t)i] = (float)(base + pj);
        } else {
            volatile double pi = step * (double)i;
            out[(size_t)i] = (float)(start + pi);
        }
    }
    return out;
}

/// bucketize(coords, boundaries, right=True): the count of boundaries <= x.
std::vector<int> buckets(int n, const std::vector<float>& bounds) {
    const std::vector<float> c = torch_arange_f32(0.0, 1.0 - 1e-6, 1.0 / (double)n);
    if ((int)c.size() != n)
        throw std::runtime_error("pos_embed_lookup: arange(0, 1-1e-6, 1/" + std::to_string(n) +
                                 ") has " + std::to_string(c.size()) + " elements");
    std::vector<int> b((size_t)n);
    for (int i = 0; i < n; i++)
        b[(size_t)i] = (int)(std::upper_bound(bounds.begin(), bounds.end(), c[(size_t)i]) -
                             bounds.begin());
    return b;
}

}  // namespace

void pos_embed_lookup(const uint16_t* table, int side, int D, int h, int w, float* out) {
    const std::vector<float> bounds = torch_arange_f32(1.0 / side, 1.0, 1.0 / side);
    const std::vector<int> bh = buckets(h, bounds);
    const std::vector<int> bw = buckets(w, bounds);

#pragma omp parallel for schedule(static) num_threads(mco::vision::par_threads())
    for (int r = 0; r < h; r++) {
        for (int c = 0; c < w; c++) {
            const uint16_t* src = table + ((size_t)bh[(size_t)r] * side + bw[(size_t)c]) * D;
            float* dst = out + ((size_t)r * w + c) * D;
            for (int d = 0; d < D; d++) dst[d] = ref::bf16_to_f32(src[d]);
        }
    }
}

void add_inplace(float* x, const float* y, size_t n) {
    constexpr size_t CHUNK = 1u << 16;
    // Signed index: MSVC's /openmp is OpenMP 2.0 (C3016).
    const int64_t nchunk = (int64_t)((n + CHUNK - 1) / CHUNK);
#pragma omp parallel for schedule(static) num_threads(mco::vision::par_threads())
    for (int64_t c = 0; c < nchunk; c++) {
        const size_t i0 = (size_t)c * CHUNK, i1 = std::min(n, i0 + CHUNK);
        for (size_t i = i0; i < i1; i++) x[i] += y[i];
    }
}

void regroup_2x2(const float* x, int h, int w, int D, float* out, float* mean) {
    if (h % 2 || w % 2)
        throw std::runtime_error("regroup_2x2: view " + std::to_string(h) + "x" +
                                 std::to_string(w) + " is not a multiple of 2");
    const int mh = h / 2, mw = w / 2;
#pragma omp parallel for schedule(static) num_threads(mco::vision::par_threads())
    for (int m = 0; m < mh * mw; m++) {
        const int r = m / mw, c = m % mw;
        float* o = out + (size_t)m * 4 * D;
        const float* src[4] = {
            x + ((size_t)(2 * r) * w + 2 * c) * D,
            x + ((size_t)(2 * r) * w + 2 * c + 1) * D,
            x + ((size_t)(2 * r + 1) * w + 2 * c) * D,
            x + ((size_t)(2 * r + 1) * w + 2 * c + 1) * D,
        };
        for (int s = 0; s < 4; s++) std::memcpy(o + (size_t)s * D, src[s], sizeof(float) * D);
        if (mean) {
            float* mo = mean + (size_t)m * D;
            for (int d = 0; d < D; d++)
                mo[d] = (src[0][d] + src[1][d] + src[2][d] + src[3][d]) * 0.25f;
        }
    }
}

void window_attention(const float* q, const float* k, const float* v, int ld,
                      int h, int w, int heads, int dh, float scale, float* o) {
    if (h % 2 || w % 2)
        throw std::runtime_error("window_attention: view is not a multiple of 2");
    const int    mw = w / 2;
    const int    nwin = (h / 2) * mw;
    const size_t D = (size_t)heads * dh;
    const size_t S = (size_t)ld;
#pragma omp parallel for schedule(static) num_threads(mco::vision::par_threads())
    for (int m = 0; m < nwin; m++) {
        const int r = m / mw, c = m % mw;
        const size_t rows[4] = {
            (size_t)(2 * r) * w + 2 * c,     (size_t)(2 * r) * w + 2 * c + 1,
            (size_t)(2 * r + 1) * w + 2 * c, (size_t)(2 * r + 1) * w + 2 * c + 1,
        };
        for (int hd = 0; hd < heads; hd++) {
            const size_t off = (size_t)hd * dh;
            for (int i = 0; i < 4; i++) {
                const float* qi = q + rows[i] * S + off;
                float s[4];
                for (int j = 0; j < 4; j++) {
                    const float* kj = k + rows[j] * S + off;
                    float acc = 0.0f;
                    for (int d = 0; d < dh; d++) acc += qi[d] * kj[d];
                    s[j] = acc * scale;
                }
                // Seeded from a score, not -INFINITY: the engine builds with
                // -ffast-math, which lets the compiler assume no infinities.
                const float smax = std::max(std::max(s[0], s[1]), std::max(s[2], s[3]));
                float den = 0.0f;
                for (int j = 0; j < 4; j++) { s[j] = std::exp(s[j] - smax); den += s[j]; }
                const float inv = 1.0f / den;
                float* oi = o + rows[i] * D + off;
                for (int d = 0; d < dh; d++) {
                    float acc = 0.0f;
                    for (int j = 0; j < 4; j++) acc += s[j] * v[rows[j] * S + off + d];
                    oi[d] = acc * inv;
                }
            }
        }
    }
}

}  // namespace vision
}  // namespace mco
