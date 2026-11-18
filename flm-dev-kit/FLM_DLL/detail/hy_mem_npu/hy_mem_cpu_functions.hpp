// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file hy_mem_cpu_functions.hpp
/// \brief The small per-row prefill steps of Hy-Memory-30B-A3B, on the host
/// \note Norms, q/k norm + RoPE, the K/V rows into the decode cache, SiLU-GLU,
///       residual, and the MoE router. Rounding follows transformers' bf16 model
///       (the PyTorch reference has the same math): every step reads
///       and writes bf16, fp32 inside.
#pragma once
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>
#include "hy_mem_prefill_ops.hpp"

namespace hy_mem_cpu {

/// \brief out[l] = w * bf16(x[l] * rsqrt(mean(x[l]^2) + eps)), HYV3RMSNorm's rounding
inline void rms_norm(bf16* out, const bf16* x, const bf16* w, int L, int D, float eps) {
    #pragma omp parallel for num_threads(prefill_threads()) schedule(static)
    for (int l = 0; l < L; l++) {
        const bf16* xr = x + (size_t)l * D;
        bf16* o = out + (size_t)l * D;
        double ss = 0.0;
        for (int d = 0; d < D; d++) ss += (double)(float)xr[d] * (float)xr[d];
        const float r = 1.0f / std::sqrt((float)(ss / D) + eps);
        for (int d = 0; d < D; d++) {
            const bf16 n = (bf16)((float)xr[d] * r);
            o[d] = (bf16)((float)n * (float)w[d]);
        }
    }
}

/// \brief per-head RMS norm, then RoPE (rotate_half), in place on one head
/// \note hy_v3 norms q / k *before* RoPE. cos / sin are bf16 as in transformers
///       (and in the decode overlay's RoPE table); the angle is fp32 inv_freq * pos.
inline void head_norm_rope(bf16* h, const bf16* w, int DH, float eps, const float* cos_t, const float* sin_t) {
    std::vector<float> n(DH);
    double ss = 0.0;
    for (int d = 0; d < DH; d++) ss += (double)(float)h[d] * (float)h[d];
    const float r = 1.0f / std::sqrt((float)(ss / DH) + eps);
    for (int d = 0; d < DH; d++) n[d] = (float)(bf16)((float)(bf16)((float)h[d] * r) * (float)w[d]);
    const int half = DH / 2;
    for (int i = 0; i < half; i++) {
        // x * cos + rotate_half(x) * sin, rotate_half(x) = [-x2, x1]
        h[i] = (bf16)(n[i] * cos_t[i] - n[i + half] * sin_t[i]);
        h[i + half] = (bf16)(n[i + half] * cos_t[i] + n[i] * sin_t[i]);
    }
}

/// \brief q / k norm + RoPE on the k|v|q projection rows, in place
/// \param kvq [L][DK + DV + DQ]: k, v, q back to back (the decode proj order)
/// \param pos0 position of row 0
inline void qk_norm_rope(bf16* kvq, int L, const hy_mem_desc& desc, const bf16* q_norm_w, const bf16* k_norm_w, int pos0) {
    const int DH = desc.DH, half = desc.RoPE_DIM / 2;
    const int stride = desc.DK + desc.DV + desc.DQ;
    #pragma omp parallel for num_threads(prefill_threads()) schedule(static)
    for (int l = 0; l < L; l++) {
        std::vector<float> cos_t(half), sin_t(half);
        for (int i = 0; i < half; i++) {
            const float angle = desc.inv_freq[i] * (float)(pos0 + l);
            cos_t[i] = (float)(bf16)std::cos(angle);
            sin_t[i] = (float)(bf16)std::sin(angle);
        }
        bf16* row = kvq + (size_t)l * stride;
        for (uint32_t h = 0; h < desc.num_kv_heads; h++)
            head_norm_rope(row + h * DH, k_norm_w, DH, desc.rms_norm_eps, cos_t.data(), sin_t.data());
        for (uint32_t h = 0; h < desc.num_heads; h++)
            head_norm_rope(row + desc.DK + desc.DV + h * DH, q_norm_w, DH, desc.rms_norm_eps, cos_t.data(), sin_t.data());
    }
}

/// \brief the new K (roped) and V rows into the decode KV cache, rows pos0 .. pos0 + L - 1
inline void store_kv(const kv_cache_view& kv, const bf16* kvq, int L, const hy_mem_desc& desc, int pos0) {
    const int stride = desc.DK + desc.DV + desc.DQ;
    #pragma omp parallel for num_threads(prefill_threads()) schedule(static)
    for (int l = 0; l < L; l++) {
        const bf16* row = kvq + (size_t)l * stride;
        for (uint32_t h = 0; h < desc.num_kv_heads; h++) {
            memcpy(kv.row(0, h, pos0 + l), row + h * desc.DH, desc.DH * sizeof(bf16));
            memcpy(kv.row(1, h, pos0 + l), row + desc.DK + h * desc.DH, desc.DH * sizeof(bf16));
        }
    }
}

/// \brief hid[l] = silu(gate) * up, from rows stored as the decode GLU reads them:
///        [up slab | gate slab] repeated (hy_mem_model_def.cpp interleave_up_gate)
/// \param width the MLP width (6912 dense, 768 experts); fused rows are 2 * width
inline void silu_glu(bf16* hid, const bf16* fused, int L, int width, int slab) {
    #pragma omp parallel for num_threads(prefill_threads()) schedule(static)
    for (int l = 0; l < L; l++) {
        const bf16* f = fused + (size_t)l * 2 * width;
        bf16* o = hid + (size_t)l * width;
        for (int s = 0; s < width / slab; s++) {
            const bf16* up = f + (size_t)s * 2 * slab;
            const bf16* gate = up + slab;
            for (int i = 0; i < slab; i++) {
                const float g = (float)gate[i];
                const float silu = (float)(bf16)(g / (1.0f + std::exp(-g)));
                o[s * slab + i] = (bf16)(silu * (float)up[i]);
            }
        }
    }
}

/// \brief x += y, rows of D, rounded to bf16 (the residual stream is bf16)
inline void add_rows(bf16* x, const bf16* y, size_t n) {
    #pragma omp parallel for num_threads(prefill_threads()) schedule(static)
    for (int64_t i = 0; i < (int64_t)n; i++) x[i] = (bf16)((float)x[i] + (float)y[i]);
}

/// \brief x += bf16(y), y fp32 (the MoE sum is kept in fp32 until it joins the residual)
inline void add_rows_f32(bf16* x, const float* y, size_t n) {
    #pragma omp parallel for num_threads(prefill_threads()) schedule(static)
    for (int64_t i = 0; i < (int64_t)n; i++) x[i] = (bf16)((float)x[i] + (float)(bf16)y[i]);
}

// =============================================================================
// MoE router
// =============================================================================
/// \brief logits[l][e] = x[l] . W[e] in fp32, W in the router's read order
///        (hy_mem_model_def.cpp: 32-expert x 256-input blocks, expert-block major,
///        each block input-major with its 32 experts adjacent)
inline void router_logits(float* logits, const bf16* x, const bf16* w, int L, int D, int E) {
    constexpr int ROW = hy_mem_desc::ROUTER_ROW_BLOCK, COL = hy_mem_desc::ROUTER_COL_BLOCK;
    static_assert(ROW == 32, "two 16-lane accumulators per expert block");
    #pragma omp parallel for num_threads(prefill_threads()) schedule(static)
    for (int l = 0; l < L; l++) {
        const bf16* xr = x + (size_t)l * D;
        for (int eb = 0; eb < E; eb += ROW) {
            // 32 experts = two vectors; each input value scales one row of the block
            __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
            for (int db = 0; db < D; db += COL) {
                const bf16* blk = w + ((size_t)(eb / ROW) * (D / COL) + db / COL) * ROW * COL;
                for (int d = 0; d < COL; d++) {
                    const __m512 xv = _mm512_set1_ps((float)xr[db + d]);
                    a0 = _mm512_fmadd_ps(xv, load_bfloat16_to_m512(blk + d * ROW), a0);
                    a1 = _mm512_fmadd_ps(xv, load_bfloat16_to_m512(blk + d * ROW + 16), a1);
                }
            }
            _mm512_storeu_ps(logits + (size_t)l * E + eb, a0);
            _mm512_storeu_ps(logits + (size_t)l * E + eb + 16, a1);
        }
    }
}

/// \brief tokens routed to one expert, with their routing weights
struct expert_task {
    int expert;
    std::vector<int> tokens;
    std::vector<float> weights;
};

/// \brief sigmoid scores, top k of (score + bias), weight = score / sum * scaling
///        (the bias only picks); grouped into one task per used expert
/// \note Ties in score + bias go to the lower expert index.
inline std::vector<expert_task> route(const float* logits, const std::vector<float>& bias, int L, int E, int top_k, float scaling) {
    std::vector<int> ids((size_t)L * top_k);
    std::vector<float> wts((size_t)L * top_k);
    #pragma omp parallel for num_threads(prefill_threads()) schedule(static)
    for (int l = 0; l < L; l++) {
        std::vector<float> score(E), choice(E);
        for (int e = 0; e < E; e++) {
            score[e] = 1.0f / (1.0f + std::exp(-logits[(size_t)l * E + e]));
            choice[e] = score[e] + bias[e];
        }
        std::vector<int> order(E);
        std::iota(order.begin(), order.end(), 0);
        std::partial_sort(order.begin(), order.begin() + top_k, order.end(), [&](int a, int b) {
            return choice[a] > choice[b] || (choice[a] == choice[b] && a < b);
        });
        float sum = 0.0f;
        for (int k = 0; k < top_k; k++) sum += score[order[k]];
        for (int k = 0; k < top_k; k++) {
            ids[(size_t)l * top_k + k] = order[k];
            wts[(size_t)l * top_k + k] = score[order[k]] / (sum + 1e-20f) * scaling;
        }
    }
    // one task per expert, tokens in order
    std::vector<expert_task> by_expert(E);
    for (int e = 0; e < E; e++) by_expert[e].expert = e;
    for (int l = 0; l < L; l++) {
        for (int k = 0; k < top_k; k++) {
            expert_task& t = by_expert[ids[(size_t)l * top_k + k]];
            t.tokens.push_back(l);
            t.weights.push_back(wts[(size_t)l * top_k + k]);
        }
    }
    std::vector<expert_task> tasks;
    for (auto& t : by_expert)
        if (!t.tokens.empty()) tasks.push_back(std::move(t));
    return tasks;
}

}  // namespace hy_mem_cpu
