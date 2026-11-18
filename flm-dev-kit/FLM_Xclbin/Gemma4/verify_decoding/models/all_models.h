// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __ALL_MODELS_H__
#define __ALL_MODELS_H__

/// @brief Attention implementation
/// @note 2x4x1: for 2 kv heads
#define ATTN_IMPL_2x4x1 0
/// @brief Attention implementation
/// @note 1x8x1: for 1 kv heads
#define ATTN_IMPL_1x8x1 1

/// @brief Activation function
#define A_SILU 0
/// @brief Activation function
#define A_GELU 1

#define GEMMA4_E2B         0
#define GEMMA4_E4B         1
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
#define MODEL_TYPE GEMMA4_E4B
#endif

#include "../models/gemma4-e2b.h"
#include "../models/gemma4-e4b.h"
#endif // __ALL_MODELS_H__