// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#pragma once
/// \file cpu_ref_vision_attn.hpp
/// \brief fp32 CPU golden for the Qwen3-VL vision encoder attention.
///
/// 16 heads, head_dim 72, no GQA, full (bidirectional) mask - every query
/// attends to every one of the L valid keys. The NPU pads L up to a whole
/// number of 512-row chunks and masks the tail; this reference simply never
/// looks past L, so the two agree only if that masking is correct.

#include <cmath>
#include <vector>
#include "typedef.hpp"

namespace cpu_ref {

/// \brief O[t,h] = softmax(scale * Q[t,h] . K[:,h]) . V[:,h] over the L valid rows.
/// \param q,k,v  [L][heads*dh], head-major within a row (head h at offset h*dh)
/// \param o      [L][heads*dh] output, same layout
/// \param head_perm  if non-null, head h of Q is scored against k/v head
///                   head_perm[h]. Used to build a control: with the identity
///                   this is the reference, with a rotation it is an imposter
///                   that a correct NPU run must NOT match.
inline void vision_attention(
    std::vector<float>& o,
    const std::vector<float>& q,
    const std::vector<float>& k,
    const std::vector<float>& v,
    const int L,
    const int heads,
    const int dh,
    const float scale,
    const int* head_perm = nullptr
){
    const int D = heads * dh;
    o.assign((size_t)L * D, 0.0f);

    std::vector<float> s((size_t)L);

    for (int h = 0; h < heads; h++){
        const int hk = head_perm ? head_perm[h] : h;
        for (int t = 0; t < L; t++){
            const float* qp = q.data() + (size_t)t * D + (size_t)h * dh;

            // scores over all L keys - no causal horizon
            float smax = -INFINITY;
            for (int j = 0; j < L; j++){
                const float* kp = k.data() + (size_t)j * D + (size_t)hk * dh;
                float acc = 0.0f;
                for (int d = 0; d < dh; d++) acc += qp[d] * kp[d];
                acc *= scale;
                s[j] = acc;
                if (acc > smax) smax = acc;
            }

            float denom = 0.0f;
            for (int j = 0; j < L; j++){
                s[j] = std::exp(s[j] - smax);
                denom += s[j];
            }
            const float inv = 1.0f / denom;

            float* op = o.data() + (size_t)t * D + (size_t)h * dh;
            for (int j = 0; j < L; j++){
                const float p = s[j] * inv;
                if (p == 0.0f) continue;
                const float* vp = v.data() + (size_t)j * D + (size_t)hk * dh;
                for (int d = 0; d < dh; d++) op[d] += p * vp[d];
            }
        }
    }
}

/// \brief Effective number of keys carrying the softmax mass, exp(entropy).
///
/// A random q/k pair puts almost all mass on a handful of keys, which makes
/// attention look correct no matter how the keys are addressed. Printing this
/// is what proves the test input actually exercises the reduction: it should
/// be a large fraction of L, not ~1.
inline double effective_key_count(
    const std::vector<float>& q,
    const std::vector<float>& k,
    const int L,
    const int heads,
    const int dh,
    const float scale,
    const int head = 0,
    const int query = 0
){
    const int D = heads * dh;
    const float* qp = q.data() + (size_t)query * D + (size_t)head * dh;
    std::vector<float> s((size_t)L);
    float smax = -INFINITY;
    for (int j = 0; j < L; j++){
        const float* kp = k.data() + (size_t)j * D + (size_t)head * dh;
        float acc = 0.0f;
        for (int d = 0; d < dh; d++) acc += qp[d] * kp[d];
        acc *= scale;
        s[j] = acc;
        if (acc > smax) smax = acc;
    }
    double denom = 0.0;
    for (int j = 0; j < L; j++){ s[j] = std::exp(s[j] - smax); denom += s[j]; }
    double H = 0.0;
    for (int j = 0; j < L; j++){
        const double p = s[j] / denom;
        if (p > 0.0) H -= p * std::log(p);
    }
    return std::exp(H);
}

} // namespace cpu_ref
