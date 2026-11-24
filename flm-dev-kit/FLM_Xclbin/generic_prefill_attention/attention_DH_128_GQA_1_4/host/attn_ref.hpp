// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#pragma once
/// \file attn_ref.hpp
/// \brief CPU golden reference for the DH=128 GQA 1:4 prefill attention kernel.
///
/// The NPU kernel (kernel/attn.cc + kernel/attn_kernel.h) implements flash
/// attention with GQA. This file re-implements the same math in fp32 so the NPU
/// result can be checked against it without a safetensors dump.
///
/// Data layouts (all row major, matching what the host feeds to the NPU):
///   q : [L_q ][n_q_heads  * dh]  row r is global token (L_begin + r)
///   k : [L_kv][n_kv_heads * dh]  row r is global token r
///   v : [L_kv][n_kv_heads * dh]  row r is global token r
///   o : [L_q ][n_q_heads  * dh]  row r is global token (L_begin + r)
///
/// Mask: kernel/attn_kernel.h::apply_mask_and_get_max keeps score(q, k) when
///   k <= q  and  k > q - window_size
/// i.e. causal, plus a sliding window lower bound. The kernel is launched with
/// window_size == L_end when is_sliding_window is false, which makes the lower
/// bound inactive (plain causal attention).

#include <cmath>
#include <limits>
#include <vector>
#include <cassert>
#include "typedef.hpp"

/// \brief shape / masking description of the attention to reproduce
struct attn_ref_config {
    /// \brief head dimension, must match `dh` in kernel/common.h
    int dh = 128;
    /// \brief number of query heads
    int n_q_heads = 32;
    /// \brief number of key/value heads (GQA 1:4)
    int n_kv_heads = 8;
    /// \brief sliding window size; pass >= sequence length for plain causal
    int window_size = 1 << 30;
    /// \brief softmax scale applied to q.k, must match `scale` in kernel/common.h
    float scale = 0.08838834764831843f;
};

/// \brief fp32 causal (optionally windowed) GQA attention reference
/// \param o output buffer, [L_q][n_q_heads * dh], written as bf16
/// \param q query buffer, [L_q][n_q_heads * dh], rows start at global L_begin
/// \param k key buffer, [L_kv][n_kv_heads * dh], rows start at global 0
/// \param v value buffer, [L_kv][n_kv_heads * dh], rows start at global 0
/// \param L_q number of query rows to compute
/// \param L_begin global position of the first query row
/// \param cfg shape / masking description
inline void attn_cpu_ref(
    buffer<bf16>& o,
    const buffer<bf16>& q,
    const buffer<bf16>& k,
    const buffer<bf16>& v,
    const int L_q,
    const int L_begin,
    const attn_ref_config& cfg
){
    const int dh = cfg.dh;
    const int DQ = cfg.n_q_heads * dh;
    const int DK = cfg.n_kv_heads * dh;
    // GQA: q head h reads kv head h / q_per_kv
    const int q_per_kv = cfg.n_q_heads / cfg.n_kv_heads;
    assert(cfg.n_q_heads % cfg.n_kv_heads == 0);
    assert(o.size() >= (size_t)L_q * DQ);
    assert(q.size() >= (size_t)L_q * DQ);

    // Widen the inputs to fp32 up front: the whole reference then runs on
    // contiguous floats, which is what lets the inner loops vectorize.
    const int L_kv = L_begin + L_q;
    assert(k.size() >= (size_t)L_kv * DK);
    assert(v.size() >= (size_t)L_kv * DK);
    std::vector<float> qf((size_t)L_q * DQ);
    std::vector<float> kf((size_t)L_kv * DK);
    std::vector<float> vf((size_t)L_kv * DK);
    for (size_t i = 0; i < qf.size(); i++) qf[i] = (float)q.data()[i];
    for (size_t i = 0; i < kf.size(); i++) kf[i] = (float)k.data()[i];
    for (size_t i = 0; i < vf.size(); i++) vf[i] = (float)v.data()[i];

    std::vector<float> scores;
    std::vector<float> acc(dh);

    for (int r = 0; r < L_q; r++) {
        const int q_pos = L_begin + r;
        // causal upper bound, sliding window lower bound (both inclusive)
        const int k_hi = q_pos;
        const int k_lo = std::max(0, q_pos - cfg.window_size + 1);
        const int n_k = k_hi - k_lo + 1;
        scores.resize(n_k);

        for (int h = 0; h < cfg.n_q_heads; h++) {
            const float* q_row = qf.data() + (size_t)r * DQ + (size_t)h * dh;
            const int kv_h = h / q_per_kv;
            const float* k_base = kf.data() + (size_t)k_lo * DK + (size_t)kv_h * dh;
            const float* v_base = vf.data() + (size_t)k_lo * DK + (size_t)kv_h * dh;

            // s = scale * q . k, keeping the running max for a stable softmax
            float m = -std::numeric_limits<float>::infinity();
            for (int j = 0; j < n_k; j++) {
                const float* k_row = k_base + (size_t)j * DK;
                float dot = 0.0f;
                for (int e = 0; e < dh; e++) {
                    dot += q_row[e] * k_row[e];
                }
                const float s = dot * cfg.scale;
                scores[j] = s;
                m = s > m ? s : m;
            }

            // p = exp(s - m), l = sum(p)
            float l = 0.0f;
            for (int j = 0; j < n_k; j++) {
                const float p = std::exp(scores[j] - m);
                scores[j] = p;
                l += p;
            }

            // y = p . v, o = y / l
            std::fill(acc.begin(), acc.end(), 0.0f);
            for (int j = 0; j < n_k; j++) {
                const float* v_row = v_base + (size_t)j * DK;
                const float p = scores[j];
                for (int e = 0; e < dh; e++) {
                    acc[e] += p * v_row[e];
                }
            }
            const float inv_l = 1.0f / l;
            bf16* o_row = o.data() + (size_t)r * DQ + (size_t)h * dh;
            for (int e = 0; e < dh; e++) {
                o_row[e] = (bf16)(acc[e] * inv_l);
            }
        }
    }
}
