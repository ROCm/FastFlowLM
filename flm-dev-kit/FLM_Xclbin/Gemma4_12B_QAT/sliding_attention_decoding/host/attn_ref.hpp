// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#pragma once
#include <algorithm>
#include <cmath>
#include <vector>
#include "typedef.hpp"

/// \brief Shape of the attention the kernel computes.
struct attn_ref_config {
    int n_q_heads;
    int n_kv_heads;
    int dh;
};

/// \brief fp32 CPU reference: o = softmax(q . k) v per query head, no 1/sqrt(dh) scale.
/// \param q    [rows][n_q_heads * dh]
/// \param k, v [n_keys][n_kv_heads * dh]; every query row sees all n_keys rows
/// \param o    [rows][n_q_heads * dh]
inline void attn_cpu_ref(const bf16 *q, int rows, const bf16 *k, const bf16 *v, int n_keys,
                         const attn_ref_config &cfg, bf16 *o) {
    const int dq = cfg.n_q_heads * cfg.dh;
    const int dk = cfg.n_kv_heads * cfg.dh;
    const int q_heads_per_kv = cfg.n_q_heads / cfg.n_kv_heads;

    std::vector<float> k_f((size_t)n_keys * dk);
    std::vector<float> v_f((size_t)n_keys * dk);
    for (size_t i = 0; i < k_f.size(); i++) {
        k_f[i] = (float)k[i];
        v_f[i] = (float)v[i];
    }

    std::vector<float> q_row(cfg.dh);
    std::vector<float> scores(n_keys);
    std::vector<float> out(cfg.dh);
    for (int r = 0; r < rows; r++) {
        for (int h = 0; h < cfg.n_q_heads; h++) {
            const int kv_col = (h / q_heads_per_kv) * cfg.dh;
            for (int e = 0; e < cfg.dh; e++) {
                q_row[e] = (float)q[(size_t)r * dq + h * cfg.dh + e];
            }

            float max_score = -INFINITY;
            for (int j = 0; j < n_keys; j++) {
                const float *k_row = &k_f[(size_t)j * dk + kv_col];
                float dot = 0.0f;
                for (int e = 0; e < cfg.dh; e++) {
                    dot += q_row[e] * k_row[e];
                }
                scores[j] = dot;
                max_score = std::max(max_score, dot);
            }

            float sum = 0.0f;
            std::fill(out.begin(), out.end(), 0.0f);
            for (int j = 0; j < n_keys; j++) {
                const float p = std::exp(scores[j] - max_score);
                sum += p;
                const float *v_row = &v_f[(size_t)j * dk + kv_col];
                for (int e = 0; e < cfg.dh; e++) {
                    out[e] += p * v_row[e];
                }
            }
            for (int e = 0; e < cfg.dh; e++) {
                o[(size_t)r * dq + h * cfg.dh + e] = (bf16)(out[e] / sum);
            }
        }
    }
}
