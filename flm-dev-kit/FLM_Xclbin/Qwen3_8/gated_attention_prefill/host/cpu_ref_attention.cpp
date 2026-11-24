// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include "cpu_ref_attention.hpp"

#include <cmath>
#include <vector>
#include <algorithm>

namespace cpu_ref {

void gqa_attention(const bf16 *q, const bf16 *k, const bf16 *v, float *out,
                   const attn_geometry &g, int L_begin, int L_q,
                   int window, double *mean_effective_keys) {
    const int DH = g.head_dim;
    const int DQ = g.dq();
    const int DKV = g.dkv();
    const int group = g.gqa_group();

    double eff_sum = 0.0;

#pragma omp parallel for collapse(2) schedule(dynamic) reduction(+ : eff_sum)
    for (int t = 0; t < L_q; t++) {
        for (int h = 0; h < g.n_q_heads; h++) {
            const int kvh = h / group;
            const int q_abs = L_begin + t;
            const int j_end = q_abs + 1; // causal horizon, inclusive of self
            const int j_begin =
                (window > 0 && q_abs - window + 1 > 0) ? (q_abs - window + 1) : 0;
            const int nkey = j_end - j_begin;

            const bf16 *qv = q + (size_t)t * DQ + (size_t)h * DH;

            std::vector<float> s(nkey);
            float smax = -INFINITY;
            for (int n = 0; n < nkey; n++) {
                const bf16 *kv = k + (size_t)(j_begin + n) * DKV + (size_t)kvh * DH;
                float acc = 0.0f;
                for (int d = 0; d < DH; d++) acc += (float)qv[d] * (float)kv[d];
                s[n] = acc * g.scale;
                smax = std::max(smax, s[n]);
            }

            float sum = 0.0f;
            for (int n = 0; n < nkey; n++) {
                s[n] = std::exp(s[n] - smax);
                sum += s[n];
            }
            const float inv = 1.0f / sum;

            float *o = out + (size_t)t * DQ + (size_t)h * DH;
            for (int d = 0; d < DH; d++) o[d] = 0.0f;

            // Softmax perplexity: exp(entropy).  1 == fully collapsed.
            double ent = 0.0;
            for (int n = 0; n < nkey; n++) {
                const float p = s[n] * inv;
                if (p > 0.0f) ent -= (double)p * std::log((double)p);
                if (p == 0.0f) continue;
                const bf16 *vv = v + (size_t)(j_begin + n) * DKV + (size_t)kvh * DH;
                for (int d = 0; d < DH; d++) o[d] += p * (float)vv[d];
            }
            eff_sum += std::exp(ent);
        }
    }

    if (mean_effective_keys)
        *mean_effective_keys = eff_sum / ((double)L_q * g.n_q_heads);
}

} // namespace cpu_ref
