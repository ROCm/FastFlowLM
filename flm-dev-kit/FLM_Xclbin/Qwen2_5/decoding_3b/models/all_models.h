// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __ALL_MODELS_H__
#define __ALL_MODELS_H__

/// @brief Attention implementation
/// @note 2x4x1: for 8 kv heads
#define ATTN_IMPL_2x4x1 0
/// @brief Attention implementation
/// @note 2x4x1: for 4 kv heads
#define ATTN_IMPL_1x4x1 1

/// @brief Attention implementation
/// @note 1x8x1: for 8 kv heads
#define ATTN_IMPL_1x8x1 2

/// @brief Activation function
#define A_SILU 0
/// @brief Activation function
#define A_GELU 1

#define QWEN2_3B         0
#define QWEN2VL_3B         1

#ifndef MODEL_TYPE
#define MODEL_TYPE QWEN2_3B
#endif

#include "../models/qwen2_3b.h"
#include "../models/qwen2vl_3b.h"

#endif // __ALL_MODELS_H__