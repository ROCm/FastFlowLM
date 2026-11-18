// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file cpu_ref.hpp
/// \brief fp32 CPU golden reference for the Qwen3.8-27B gated-delta-net conv1d
///        prefill stage, plus the random input generator that feeds it.
///
/// The NPU stage this checks is the middle of Qwen3_5GatedDeltaNet:
///
///     mixed_qkv = silu(causal_depthwise_conv1d_k4(mixed_qkv))
///     q, k, v   = split(mixed_qkv, DQ | DK | DV)
///     q         = l2norm(q_head) * (1/sqrt(head_k_dim))   per 128-wide head
///     k         = l2norm(k_head)                          per 128-wide head
///     v         = untouched
///
/// Both the conv and the norm are fp32 here while the NPU is bf16 with an
/// accfloat accumulator, so the expected agreement is bf16 rounding
/// (cosine ~1 - 1e-3), not bit equality.
///
/// Layout, and the thing that drives the whole host: DDR is token-major,
/// [L][row_stride] row-major, so channel c of token t is at x[t*stride + c].
/// The conv is depthwise -- channel c only ever meets w[k][c] -- which is
/// exactly why the array can shard it 32 ways by channel with no cross-tile
/// traffic, and why `stride` here is the *padded* width while only the first
/// CONV_DIM channels carry real data.
#pragma once

#include <cmath>
#include <cstdint>
#include <random>

#include "typedef.hpp"

namespace cpu_ref {

// ==========================================================================
// geometry
// ==========================================================================
/// Qwen3.8-27B gated delta net, from config.json.  Mirrors the constants in
/// ../gate_delta_net_verify/models/qwen3.8-27b.h.
namespace q38 {
constexpr int QK_HEADS   = 16;
constexpr int V_HEADS    = 48;
constexpr int K_HEAD_DIM = 128;
constexpr int V_HEAD_DIM = 128;
constexpr int DQ         = QK_HEADS * K_HEAD_DIM;   // 2048
constexpr int DK         = DQ;                      // 2048
constexpr int DV         = V_HEADS  * V_HEAD_DIM;   // 6144
/// The depthwise conv's channel count == qkv_proj's row count.
constexpr int CONV_DIM   = DQ + DK + DV;            // 10240
constexpr int CONV_K     = 4;
constexpr float L2N_EPS  = 1e-6f;
/// 1/sqrt(K_HEAD_DIM), folded into q's l2norm.
constexpr float QK_SCALE = 0.08838834764831845f;
/// Heads, counting q then k then v: 16 + 16 + 48 = 80.
constexpr int TOTAL_HEADS = (DQ + DK + DV) / K_HEAD_DIM;
} // namespace q38

// ==========================================================================
// scalar pieces
// ==========================================================================
/// SiLU (swish): x * sigmoid(x).  The NPU computes x / (1 + exp2(-x*log2e)),
/// the same function by a different route.
inline float silu(float x) { return x / (1.0f + std::exp(-x)); }

/// l2norm() of modeling_qwen3_5.py: y = x * extra_scale * rsqrt(sum(x^2)+eps).
/// \param extra_scale q folds 1/sqrt(head_dim) in here; k passes 1.0f.
inline void l2_norm(float* x, int n, float extra_scale) {
    float acc = 0.0f;
    for (int i = 0; i < n; i++) acc += x[i] * x[i];
    const float r = extra_scale / std::sqrt(acc + q38::L2N_EPS);
    for (int i = 0; i < n; i++) x[i] *= r;
}

// ==========================================================================
// the reference
// ==========================================================================
/// Causal depthwise conv1d(k=4) + SiLU + per-head l2norm on q and k.
///
/// \param x      [L_out + CONV_K - 1][stride] bf16, token-major.  The first
///               CONV_K-1 rows are the conv history: zero at the start of a
///               sequence, the trailing tokens of the previous chunk when
///               resuming.  Output token t reads input rows t .. t+CONV_K-1,
///               so input row t+CONV_K-1 is the "current" token -- that
///               off-by-one is the entire reason x carries the extra rows.
/// \param w      [CONV_K][stride] bf16, tap-major, oldest tap first.
/// \param L_out  number of output tokens.
/// \param stride DDR row stride in elements (>= CONV_DIM; the tail is padding
///               that exists only so the 32-core array tiles evenly).
/// \param y      [L_out][stride] fp32 output.  Pad channels are left alone.
inline void conv1d_silu_l2norm(const bf16* x, const bf16* w, int L_out,
                               int stride, float* y) {
    constexpr int CD = q38::CONV_DIM;
    constexpr int K  = q38::CONV_K;

    for (int t = 0; t < L_out; t++) {
        float* out = y + static_cast<size_t>(t) * stride;

        // depthwise conv + silu
        for (int c = 0; c < CD; c++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++) {
                acc += static_cast<float>(x[static_cast<size_t>(t + k) * stride + c]) *
                       static_cast<float>(w[static_cast<size_t>(k) * stride + c]);
            }
            out[c] = silu(acc);
        }

        // q: l2norm per head, with 1/sqrt(head_dim) folded in
        for (int h = 0; h < q38::QK_HEADS; h++) {
            l2_norm(out + h * q38::K_HEAD_DIM, q38::K_HEAD_DIM, q38::QK_SCALE);
        }
        // k: l2norm per head, no extra scale
        for (int h = 0; h < q38::QK_HEADS; h++) {
            l2_norm(out + q38::DQ + h * q38::K_HEAD_DIM, q38::K_HEAD_DIM, 1.0f);
        }
        // v: conv + silu only, already done above
    }
}

// ==========================================================================
// random input
// ==========================================================================
/// Fill `n` elements with N(0, stddev).  Values land in bf16, so the
/// reference and the NPU start from bit-identical inputs and the only
/// difference measured is the NPU datapath.
inline void fill_normal(bf16* dst, size_t n, float stddev, std::mt19937& rng) {
    std::normal_distribution<float> dist(0.0f, stddev);
    for (size_t i = 0; i < n; i++) dst[i] = static_cast<bf16>(dist(rng));
}

} // namespace cpu_ref
