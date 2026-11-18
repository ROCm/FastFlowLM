// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#ifndef __MODEL_SPEC__
#define __MODEL_SPEC__

#include "typedef.hpp"
#include "../models/all_models.h"

// Delta-net widths.  There is no DQ/DK/DV and no GQA blocking here: the
// full-attention layer is ../gated_attention_verify's problem.
constexpr int Linear_DQ = L_QK_HEADS * L_Q_HEAD_DIM;
constexpr int Linear_DK = L_QK_HEADS * L_Q_HEAD_DIM;
constexpr int Linear_DV = L_V_HEADS * L_V_HEAD_DIM;

constexpr int L_V_HEADS_PADDED = (L_V_HEADS + QXNX_ROW_BLOCK_SIZE - 1) / QXNX_ROW_BLOCK_SIZE * QXNX_ROW_BLOCK_SIZE;

constexpr int VOCAB_SIZE_PADDED = (VOCAB_SIZE + MODEL_DIM - 1) / MODEL_DIM * MODEL_DIM;

///@brief Q41 block
///@param scales: scales and mins, bf16
///@param mins: scales and mins, bf16
///@param qs: 4-bit quantized values
typedef struct {
    uint8 scales[QXNX_ROW_BLOCK_SIZE * QXNX_COL_BLOCK_SIZE / GGML_GROUP_SIZE]; // per-group scales and mins
    uint8 mins[QXNX_ROW_BLOCK_SIZE * QXNX_COL_BLOCK_SIZE / GGML_GROUP_SIZE];   // per-group scales and mins
    uint4 qs[QXNX_ROW_BLOCK_SIZE * QXNX_COL_BLOCK_SIZE / 2];                      // 4-bit quantized values
    bf16 S[QXNX_ROW_BLOCK_SIZE * QXNX_COL_BLOCK_SIZE / 256];
    bf16 M[QXNX_ROW_BLOCK_SIZE * QXNX_COL_BLOCK_SIZE / 256];
} q4k_block_t;

///@brief Q80 block
///@param scales: scales and mins, bf16
///@param mins: scales and mins, bf16
///@param qs: 4-bit quantized values
typedef struct {
    bf16 scales[QXNX_ROW_BLOCK_SIZE * QXNX_COL_BLOCK_SIZE / GGML_GROUP_SIZE]; // per-group scales and mins
    int8_t qs[QXNX_ROW_BLOCK_SIZE * QXNX_COL_BLOCK_SIZE];                      // 8--bit quants
} q80_block_t;


#if QUANT_TYPE == __Q4_K__
#define QUANT_BLOCK q4k_block_t
#elif QUANT_TYPE == __Q8_0__
#define QUANT_BLOCK q80_block_t
#endif

#if AB_QUANT_TYPE == __Q4_1__
#define AB_QUANT_BLOCK q41_block_t
#elif AB_QUANT_TYPE == __Q8_0__
#define AB_QUANT_BLOCK q80_block_t
#endif

#endif
