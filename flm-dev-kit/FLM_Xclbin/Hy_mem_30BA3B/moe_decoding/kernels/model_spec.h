// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#ifndef __MODEL_SPEC__
#define __MODEL_SPEC__

#include "typedef.hpp"
#include "../models/all_models.h"

constexpr int Q_HEADS_PER_GROUP = NUM_ATTN_HEADS / NUM_KV_HEADS;

constexpr int DQ = NUM_ATTN_HEADS * DH;
constexpr int DK = NUM_KV_HEADS * DH;
constexpr int DV = DK;

// Largest reduction dimension any mvm() phase runs with. o_proj uses DQ,
// which can exceed MODEL_DIM.
constexpr int MAX_MVM_K_FFN = MODEL_DIM > INTERMEDIATE_SIZE ? MODEL_DIM : INTERMEDIATE_SIZE;
constexpr int MAX_MVM_K     = MAX_MVM_K_FFN > DQ ? MAX_MVM_K_FFN : DQ;

// Analize the GQA patterns;

#if ATTN_IMPL == ATTN_IMPL_2x4x1
    const int GQA_R = 8;
    const int GQA_S = 8;
    const int GQA_T = 8;
    constexpr int GQA_SEGMENT_SIZE = 4; // 8x8x8 or 4x8x8, must be multiple of 4
    constexpr int KV_HEADS_PER_CU  = 2;
    constexpr int Q_HEADS_PER_CU  = Q_HEADS_PER_GROUP * 2;
    constexpr int ATTN_GROUPS_PADDING = ((Q_HEADS_PER_GROUP + GQA_SEGMENT_SIZE - 1) / GQA_SEGMENT_SIZE) * GQA_SEGMENT_SIZE - Q_HEADS_PER_GROUP;
#elif ATTN_IMPL == ATTN_IMPL_1x4x1
    const int GQA_R = 4;
    const int GQA_S = 8;
    const int GQA_T = 8;
    constexpr int GQA_SEGMENT_SIZE = 4; // 4x8x8, must be multiple of 4
    constexpr int KV_HEADS_PER_CU  = 1;
    constexpr int Q_HEADS_PER_CU  = Q_HEADS_PER_GROUP * 1;
    constexpr int ATTN_GROUPS_PADDING = ((Q_HEADS_PER_CU + GQA_SEGMENT_SIZE - 1) / GQA_SEGMENT_SIZE) * GQA_SEGMENT_SIZE - Q_HEADS_PER_CU;
#elif ATTN_IMPL == ATTN_IMPL_1x8x1
    const int GQA_R = 8;
    const int GQA_S = 8;
    const int GQA_T = 8;
    constexpr int GQA_SEGMENT_SIZE = 8; // 8x8x8, must be multiple of 8
    constexpr int KV_HEADS_PER_CU  = 1;
    constexpr int Q_HEADS_PER_CU  = Q_HEADS_PER_GROUP * 1;
    constexpr int ATTN_GROUPS_PADDING = ((Q_HEADS_PER_CU + GQA_SEGMENT_SIZE - 1) / GQA_SEGMENT_SIZE) * GQA_SEGMENT_SIZE - Q_HEADS_PER_CU;
#endif


constexpr int Q_HEADS_PER_GROUP_PADDED = Q_HEADS_PER_GROUP + ATTN_GROUPS_PADDING;

constexpr int Q_HEADS_PADDED_PER_CU = KV_HEADS_PER_CU * Q_HEADS_PER_GROUP_PADDED;

constexpr int ROPE_BY_PASS = DH - ROPE_DIM;

///@brief Q4K block
///@param scales: per-group uint8 scale s'
///@param mins: per-group uint8 min m'
///@param qs: 4-bit quantized values
///@param S: per-super-block bf16 scale S'
///@param M: per-super-block bf16 min M' (stored negated; the kernel adds it)
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

#endif
