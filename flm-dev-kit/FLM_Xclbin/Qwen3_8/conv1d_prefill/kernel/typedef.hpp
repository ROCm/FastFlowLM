// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

///@file typedef.hpp
///@brief Define the typedefs for the kernels
#ifndef __TYPEDEF_HPP__
#define __TYPEDEF_HPP__

#include <type_traits>
#include <aie_api/aie.hpp>

typedef bfloat16 bf16;

constexpr bfloat16 exp_scale = 1.44269504089;

/// Qwen3.8-27B gated-delta-net geometry, the part this kernel needs.
/// One q/k head, and the unit l2norm works on.
constexpr int HEAD_DIM = 128;
/// Channels one compute tile owns: two heads.
constexpr int CORE_CHANNELS = 256;
/// 1/sqrt(HEAD_DIM), folded into q's l2norm the way cpu_ref's qk_scale() is.
constexpr float LINEAR_ATTN_SCALE = 0.08838834764831845f;
/// l2norm() of modeling_qwen3_5.py: x * rsqrt(sum(x^2) + eps).
constexpr float L2N_EPS = 1e-6f;

#endif // __TYPEDEF_HPP__
