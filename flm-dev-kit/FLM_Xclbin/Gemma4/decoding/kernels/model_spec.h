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

constexpr int SWA_DQ = NUM_ATTN_HEADS * SWA_DH;
constexpr int SWA_DK = NUM_KV_HEADS * SWA_DH;
constexpr int SWA_DV = SWA_DK;

// Analize the GQA patterns;

#if ATTN_IMPL == ATTN_IMPL_2x4x1
    const int GQA_R = 8;
    const int GQA_S = 8;
    const int GQA_T = 8;
    constexpr int GQA_SEGMENT_SIZE = 4; // 8x8x8 or 4x8x8, must be multiple of 4
    constexpr int KV_HEADS_PER_CU  = 2;
    constexpr int Q_HEADS_PER_CU  = Q_HEADS_PER_GROUP * 2;
    constexpr int ATTN_GROUPS_PADDING = ((Q_HEADS_PER_GROUP + GQA_SEGMENT_SIZE - 1) / GQA_SEGMENT_SIZE) * GQA_SEGMENT_SIZE - Q_HEADS_PER_GROUP;
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

constexpr int VOCAB_SIZE_PADDED = (VOCAB_SIZE + MODEL_DIM - 1) / MODEL_DIM * MODEL_DIM;


#ifndef Q4_0
///@brief Q4K block
///@param scales: scales and mins, bf16
///@param mins: scales and mins, bf16
///@param qs: 4-bit quantized values
typedef struct {
    bf16 scales[Q4NX_ROW_BLOCK_SIZE * Q4NX_COL_BLOCK_SIZE / Q4NX_GROUP_SIZE]; // per-group scales and mins
    bf16 mins[Q4NX_ROW_BLOCK_SIZE * Q4NX_COL_BLOCK_SIZE / Q4NX_GROUP_SIZE];   // per-group scales and mins
    uint4 qs[Q4NX_ROW_BLOCK_SIZE * Q4NX_COL_BLOCK_SIZE];                      // 4-bit quantized values
} q4k_block_t;
#else

///@brief Q4K block
///@param scales: scales and mins, bf16
///@param mins: scales and mins, bf16
///@param qs: 4-bit quantized values
typedef struct {
    bf16 scales[Q4NX_ROW_BLOCK_SIZE * Q4NX_COL_BLOCK_SIZE / Q4NX_GROUP_SIZE]; // per-group scales and mins
    bf16 mins[Q4NX_ROW_BLOCK_SIZE * Q4NX_COL_BLOCK_SIZE / Q4NX_GROUP_SIZE];   // per-group scales and mins
    int4 qs[Q4NX_ROW_BLOCK_SIZE * Q4NX_COL_BLOCK_SIZE];                      // 4-bit quantized values
} q4k_block_t;
#endif

#endif
