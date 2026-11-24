// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file common/rope.hpp
/// \brief M-RoPE tables.
#pragma once

#include "common/cpu_ref_common.hpp"

#include <cmath>
#include <cstdint>
#include <vector>

namespace cpu_ref {

/// rope_theta from config.json. Not 1e4 -- copying a qwen3 header blindly
/// gives positions that drift badly past a few hundred tokens.
constexpr float ROPE_THETA = 1e7f;

// ==========================================================================
// partial_rotary_factor 0.25 -> rotary_dim 64 -> 32 frequency slots;
// mrope_interleaved assigns slot i to channel i%3 (t,h,w). Pure text makes all
// three equal and this degenerates to 1-D RoPE.
//
// round_to_bf16 defaults true: transformers returns cos.to(bf16), so that is
// what the model actually saw.
// ==========================================================================
inline void mrope_table(const int32_t* pos_thw, int T, float* cos, float* sin,
                        bool round_to_bf16 = true) {
    const int H = q38::ROPE_DIM / 2;              // 32 frequency slots
    for (int t = 0; t < T; t++) {
        for (int j = 0; j < H; j++) {
            const int   ch  = j % 3;              // 0 = t, 1 = h, 2 = w
            const float pos = static_cast<float>(pos_thw[ch * T + t]);
            const double inv =
                std::pow(static_cast<double>(ROPE_THETA),
                         -static_cast<double>(2 * j) / static_cast<double>(q38::ROPE_DIM));
            const double ang = static_cast<double>(pos) * inv;
            float c = static_cast<float>(std::cos(ang));
            float s = static_cast<float>(std::sin(ang));
            if (round_to_bf16) { c = round_bf16(c); s = round_bf16(s); }
            // the 64-wide table is the 32 slots duplicated, matching rotate_half
            cos[(size_t)t * q38::ROPE_DIM + j]     = c;
            cos[(size_t)t * q38::ROPE_DIM + j + H] = c;
            sin[(size_t)t * q38::ROPE_DIM + j]     = s;
            sin[(size_t)t * q38::ROPE_DIM + j + H] = s;
        }
    }
}

/// Position ids for a pure-text window of T tokens starting at `begin`.
/// All three m-rope channels carry the same value, which is the degenerate
/// case verified against verify.position_ids (t/h/w channels all equal).
/// Layout is [3][T], matching mrope_table's pos_thw[ch * T + t].
inline std::vector<int32_t> text_position_ids(int begin, int T) {
    std::vector<int32_t> pos(static_cast<size_t>(3) * T);
    for (int t = 0; t < T; t++) {
        const int32_t p = static_cast<int32_t>(begin + t);
        pos[(size_t)0 * T + t] = p;
        pos[(size_t)1 * T + t] = p;
        pos[(size_t)2 * T + t] = p;
    }
    return pos;
}

/// cos/sin for a pure-text window, sized [T][64] each.
inline void text_rope_tables(int begin, int T, std::vector<float>& cos,
                             std::vector<float>& sin, bool round_to_bf16 = true) {
    const std::vector<int32_t> pos = text_position_ids(begin, T);
    cos.assign(static_cast<size_t>(T) * q38::ROPE_DIM, 0.0f);
    sin.assign(static_cast<size_t>(T) * q38::ROPE_DIM, 0.0f);
    mrope_table(pos.data(), T, cos.data(), sin.data(), round_to_bf16);
}

} // namespace cpu_ref
