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
/// @note 1x8x1: for 4 kv heads
#define ATTN_IMPL_1x8x1 2

/// @brief Activation function
#define A_SILU 0
/// @brief Activation function
#define A_GELU 1

#define QWEN3_8_27B         0

#define __Q4_1__           0
#define __Q8_0__           1
#define __Q4_K__           2

// For each Q4NX block of Q4NX_ROW_BLOCK_SIZEx Q4NX_COL_BLOCK_SIZE
// The data block are in column-major order, with each column being Q4NX_ROW_STRIDE in length, 
//and the next column start after Q4NX_ROW_STRIDE elements.
/*
The data block is the following layout, where index represent the data element index view in 1D array
[0,Q4NX_ROW_STRIDE    , .... (Q4NX_COL_BLOCK_SIZE-1)*Q4NX_ROW_STRIDE],
[1,Q4NX_ROW_STRIDE + 1, .... (Q4NX_COL_BLOCK_SIZE-1)*Q4NX_ROW_STRIDE + 1],
[2,Q4NX_ROW_STRIDE + 2, .... (Q4NX_COL_BLOCK_SIZE-1)*Q4NX_ROW_STRIDE + 2],
...
*/

constexpr int Q4NX_ROW_STRIDE = 16;

constexpr int Q4NX_ROW_STRIDE_SIZE = Q4NX_ROW_STRIDE; // an alias

#ifndef MODEL_TYPE
#define MODEL_TYPE QWEN3_8_27B
#endif

#include "../models/qwen3.8-27b.h"
#endif // __ALL_MODELS_H__