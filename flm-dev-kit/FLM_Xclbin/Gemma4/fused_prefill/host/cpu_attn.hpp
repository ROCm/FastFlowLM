// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __CPU_ATTN_HPP__
#define __CPU_ATTN_HPP__
#pragma once

// CPU model of what the attention columns of the fused overlay compute.
//
// The kernel applies no scaling of its own -- kernel/attn_body.inc feeds q
// straight into _attn_qk -- so q arrives pre-scaled and the reference below is
// plain softmax(q . k^T) v.  The mask is the one apply_mask_and_get_max builds:
// a key is kept when it is causal (k <= q) and inside the window (k > q - W).
// The full-attention column runs with W = L, which makes the window condition
// vacuous and leaves pure causal masking.
//
// Everything accumulates in fp32 and the softmax runs in one pass, unlike the
// kernel's streaming max/correction; that difference is deliberate, since the
// point of the reference is to be independent of the flash recurrence.

#include <cmath>
#include <vector>

#include "typedef.hpp"

/// \brief o = softmax(masked(q . k^T)) . v, one head.
///
/// \param o       L x DH row major output for this head
/// \param q       L x DH row major, already scaled
/// \param k       L x DH row major
/// \param v       L x DH row major
/// \param L       sequence length
/// \param DH      head dim
/// \param window  sliding window; pass >= L for full causal attention
inline void cpu_attn_head(bf16* o, const bf16* q, const bf16* k, const bf16* v,
                          int L, int DH, int window) {
#pragma omp parallel for schedule(static)
    for (int i = 0; i < L; i++) {
        const bf16* q_row = q + (size_t)i * DH;
        const int lo = std::max(0, i - window + 1);

        std::vector<float> s(i - lo + 1);
        float mx = -std::numeric_limits<float>::infinity();
        for (int j = lo; j <= i; j++) {
            const bf16* k_row = k + (size_t)j * DH;
            float acc = 0.0f;
            for (int t = 0; t < DH; t++) acc = std::fma((float)q_row[t], (float)k_row[t], acc);
            s[j - lo] = acc;
            mx = std::max(mx, acc);
        }

        float denom = 0.0f;
        for (float& e : s) { e = std::exp(e - mx); denom += e; }
        const float inv = 1.0f / denom;

        std::vector<float> acc(DH, 0.0f);
        for (int j = lo; j <= i; j++) {
            const float w = s[j - lo];
            const bf16* v_row = v + (size_t)j * DH;
            for (int t = 0; t < DH; t++) acc[t] = std::fma(w, (float)v_row[t], acc[t]);
        }
        bf16* o_row = o + (size_t)i * DH;
        for (int t = 0; t < DH; t++) o_row[t] = bf16(acc[t] * inv);
    }
}

/// \brief Multi-head wrapper over the interleaved [L][Heads * DH] layout the
/// sequence generator reads q from and writes o to.  k and v are MQA here: one
/// kv head shared by every q head, stored as [L][DH].
inline void cpu_attn_mha(bf16* o, const bf16* q, const bf16* k, const bf16* v,
                         int L, int DH, int heads, int window) {
    std::vector<bf16> qh((size_t)L * DH), oh((size_t)L * DH);
    for (int h = 0; h < heads; h++) {
        for (int i = 0; i < L; i++)
            memcpy(qh.data() + (size_t)i * DH, q + (size_t)i * heads * DH + (size_t)h * DH, DH * sizeof(bf16));
        cpu_attn_head(oh.data(), qh.data(), k, v, L, DH, window);
        for (int i = 0; i < L; i++)
            memcpy(o + (size_t)i * heads * DH + (size_t)h * DH, oh.data() + (size_t)i * DH, DH * sizeof(bf16));
    }
}

#endif
