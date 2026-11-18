// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file cpu_ref_vision_mm.hpp
/// \brief CPU reference for the Qwen3.8-27B vision-tower matmul.
///
/// The reference consumes the *same* bf16 bytes the NPU is given and accumulates
/// in fp32. That makes the bf16 rounding of the inputs common-mode, so the only
/// difference left is the NPU's own arithmetic and any addressing mistake - which
/// is what lets the pass bar sit at cosine > 0.999 instead of somewhere vague.
#ifndef __CPU_REF_VISION_MM_HPP__
#define __CPU_REF_VISION_MM_HPP__
#pragma once

#include <cmath>
#include <cstdint>
#include <vector>
#include "typedef.hpp"
#include "buffer.hpp"

namespace vision_mm_ref {

/// Activation applied to the matmul output, matching the kernel's rtp_act.
enum class activation : int32_t {
    none = 0,
    silu = 1,
    gelu = 2,
};

/// gelu_pytorch_tanh, the activation named in the Qwen3.8-27B vision_config.
///   0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
inline float gelu_tanh(float x) {
    constexpr float kSqrt2OverPi = 0.79788456080286535588f;
    constexpr float kBeta = 0.044715f;
    const float inner = kSqrt2OverPi * (x + kBeta * x * x * x);
    return 0.5f * x * (1.0f + std::tanh(inner));
}

inline float silu(float x) {
    return x / (1.0f + std::exp(-x));
}

inline float apply_activation(float x, activation act) {
    switch (act) {
        case activation::silu: return silu(x);
        case activation::gelu: return gelu_tanh(x);
        default:               return x;
    }
}

/// y = act(x @ w^T + bias), with x (M, K) and w (N, K), both row-major.
///
/// \param row_shift rotate the weight rows by this much before multiplying. Zero
///        gives the true reference; a non-zero value gives the control described
///        in verify_shape(). Rotating rows of w permutes the output columns, so a
///        correct NPU result must score *badly* against it.
inline void mm_ref(const bf16* x, const bf16* w, const bf16* bias,
                   bf16* y, int M, int K, int N,
                   activation act = activation::none,
                   int row_shift = 0) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            const int wr = row_shift ? ((j + row_shift) % N + N) % N : j;
            const bf16* w_row = w + (size_t)wr * K;
            const bf16* x_row = x + (size_t)i * K;
            float acc = bias ? (float)bias[wr] : 0.0f;
            for (int p = 0; p < K; p++) {
                acc += (float)x_row[p] * (float)w_row[p];
            }
            y[(size_t)i * N + j] = (bf16)apply_activation(acc, act);
        }
    }
}

}  // namespace vision_mm_ref
#endif
