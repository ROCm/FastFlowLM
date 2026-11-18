// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

///@file typedef.hpp
///@brief Define the typedefs for the kernels
#ifndef __TYPEDEF_HPP__
#define __TYPEDEF_HPP__

#include <type_traits>
#include <aie_api/aie.hpp>

typedef bfloat16 bf16;

#define ATTN_IMPL_2x4x1 0
#define ATTN_IMPL_1x8x1 1

#define ATTN_IMPL ATTN_IMPL_1x8x1
constexpr bfloat16 exp_scale = 1.44269504089;

constexpr int   MODEL_DIM         = 2560;
constexpr int   NUM_ATTN_HEADS    = 16;
constexpr int   NUM_KV_HEADS      = 2;
constexpr int   INTERMEDIATE_SIZE = 10240;
constexpr int   GLU_SLICE         = 1024;
constexpr int   DH = 256;
constexpr float ATTN_SCALE        = 1.0f;
constexpr int Q_HEADS_PER_GROUP = NUM_ATTN_HEADS / NUM_KV_HEADS;

#endif // __TYPEDEF_HPP__