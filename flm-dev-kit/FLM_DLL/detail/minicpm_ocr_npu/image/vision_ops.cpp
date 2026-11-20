// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file image/vision_ops.cpp
/// \brief CPU backends for the two vision-tower operators.
#include "image/vision_ops.hpp"
#include "image/vision_par.hpp"

#include <cstdlib>
#include <cstring>

namespace mco {
namespace vision {

bool cpu_ops_emulate_bf16() {
    static const bool v = [] {
        const char* e = std::getenv("FLM_MINICPM_OCR_VISION_FP32");
        return !(e && e[0] == '1');
    }();
    return v;
}

void cpu_mm_op::run(const float* x, int M, float* y) {
    const bool emu = cpu_ops_emulate_bf16();
    const int  K   = K_;
    const int  N   = N_;

    // Round the activations once per call rather than once per (i,j): with
    // N = 4608 the inner reuse is what makes this affordable at all.
    std::vector<float> xr;
    if (emu) {
        xr.resize((size_t)M * K);
        for (size_t i = 0; i < xr.size(); i++) xr[i] = ref::round_bf16(x[i]);
        x = xr.data();
    }

#pragma omp parallel for schedule(static) num_threads(mco::vision::par_threads())
    for (int i = 0; i < M; i++) {
        const float* xrow = x + (size_t)i * K;
        float*       yrow = y + (size_t)i * N;
        for (int j = 0; j < N; j++) {
            const uint16_t* wrow = w_ + (size_t)j * K;
            float acc = bias_ ? ref::bf16_to_f32(bias_[j]) : 0.0f;
            for (int p = 0; p < K; p++)
                acc += xrow[p] * ref::bf16_to_f32(wrow[p]);
            const float o = apply_activation(acc, act_);
            yrow[j] = emu ? ref::round_bf16(o) : o;
        }
    }
}

void cpu_attn_op::run(const float* q, const float* k, const float* v,
                      int L, float* o, const int* head_perm, int ld) {
    if (L > max_L_)
        throw std::runtime_error(
            "cpu_attn_op::run L=" + std::to_string(L) +
            " exceeds the configured max_L=" + std::to_string(max_L_));

    const int    D     = heads_ * dh_;
    const size_t S     = ld > 0 ? (size_t)ld : (size_t)D;   // q/k/v row stride
    const int    dh    = dh_;
    const float  scale = scale_;

    std::memset(o, 0, sizeof(float) * (size_t)L * D);

#pragma omp parallel for schedule(static) num_threads(mco::vision::par_threads())
    for (int h = 0; h < heads_; h++) {
        const int hk = head_perm ? head_perm[h] : h;
        std::vector<float> s((size_t)L);
        for (int t = 0; t < L; t++) {
            const float* qp = q + (size_t)t * S + (size_t)h * dh;

            // No causal horizon: every query sees all L keys of this image.
            float smax = -INFINITY;
            for (int j = 0; j < L; j++) {
                const float* kp = k + (size_t)j * S + (size_t)hk * dh;
                float acc = 0.0f;
                for (int d = 0; d < dh; d++) acc += qp[d] * kp[d];
                acc *= scale;
                s[j] = acc;
                if (acc > smax) smax = acc;
            }

            float denom = 0.0f;
            for (int j = 0; j < L; j++) { s[j] = std::exp(s[j] - smax); denom += s[j]; }
            const float inv = 1.0f / denom;

            float* op = o + (size_t)t * D + (size_t)h * dh;
            for (int j = 0; j < L; j++) {
                const float p = s[j] * inv;
                if (p == 0.0f) continue;
                const float* vp = v + (size_t)j * S + (size_t)hk * dh;
                for (int d = 0; d < dh; d++) op[d] += p * vp[d];
            }
        }
    }
}

double effective_key_count(const float* q, const float* k,
                           int L, int heads, int dh, float scale,
                           int head, int query) {
    const int D = heads * dh;
    const float* qp = q + (size_t)query * D + (size_t)head * dh;
    std::vector<float> s((size_t)L);
    float smax = -INFINITY;
    for (int j = 0; j < L; j++) {
        const float* kp = k + (size_t)j * D + (size_t)head * dh;
        float acc = 0.0f;
        for (int d = 0; d < dh; d++) acc += qp[d] * kp[d];
        acc *= scale;
        s[j] = acc;
        if (acc > smax) smax = acc;
    }
    double denom = 0.0;
    for (int j = 0; j < L; j++) { s[j] = std::exp(s[j] - smax); denom += s[j]; }
    double H = 0.0;
    for (int j = 0; j < L; j++) {
        const double p = s[j] / denom;
        if (p > 0.0) H -= p * std::log(p);
    }
    return std::exp(H);
}

}  // namespace vision
}  // namespace mco
