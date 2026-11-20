// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __MODEL_SPEC__
#define __MODEL_SPEC__

#include "typedef.hpp"
#include "../models/all_models.h"

constexpr int Q_HEADS_PER_GROUP = NUM_ATTN_HEADS / NUM_KV_HEADS;

constexpr int DQ = NUM_ATTN_HEADS * DH;
constexpr int DK = NUM_KV_HEADS * DH;
constexpr int DV = DK;

// There is no Linear_DQ/DK/DV here: the gated delta-net layers are a separate
// xclbin, and qwen3.8-27b.h deliberately does not define the L_* constants
// they would need. The MTP draft head is one full-attention layer.

const int GQA_R = 8;
const int GQA_S = 8;
const int GQA_T = 8;
constexpr int GQA_SEGMENT_SIZE = 8; // 8x8x8, must be multiple of 8
constexpr int KV_HEADS_PER_CU  = 1;
constexpr int Q_HEADS_PER_CU  = Q_HEADS_PER_GROUP * 1;
constexpr int ATTN_GROUPS_PADDING = ((Q_HEADS_PER_CU + GQA_SEGMENT_SIZE - 1) / GQA_SEGMENT_SIZE) * GQA_SEGMENT_SIZE - Q_HEADS_PER_CU;

constexpr int Q_HEADS_PER_GROUP_PADDED = Q_HEADS_PER_GROUP + ATTN_GROUPS_PADDING;

constexpr int Q_HEADS_PADDED_PER_CU = KV_HEADS_PER_CU * Q_HEADS_PER_GROUP_PADDED;

constexpr int VOCAB_SIZE_PADDED = (VOCAB_SIZE + MODEL_DIM - 1) / MODEL_DIM * MODEL_DIM;

constexpr int ROPE_BY_PASS = DH - ROPE_DIM;

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
