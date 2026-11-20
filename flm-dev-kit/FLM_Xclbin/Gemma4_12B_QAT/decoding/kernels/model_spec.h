// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __MODEL_SPEC__
#define __MODEL_SPEC__

#include "typedef.hpp"
#include "../models/all_models.h"
#include "attn_role.h"

// The GQA geometry below is per-role: ROLE_KV_HEADS, Q_PASSES_PER_CU and
// ROLE_IMPL all come from attn_role.h and resolve to the global flavor unless
// the translation unit was compiled with -DATTN_ROLE=ATTN_ROLE_SWA.
constexpr int Q_HEADS_PER_GROUP = NUM_ATTN_HEADS / ROLE_KV_HEADS / Q_PASSES_PER_CU;
constexpr int SWA_Q_HEADS_PER_GROUP = NUM_ATTN_HEADS / SWA_NUM_KV_HEADS;

constexpr int DQ = NUM_ATTN_HEADS * DH;
constexpr int DK = NUM_KV_HEADS * DH;
constexpr int DV = DK;

constexpr int SWA_DQ = NUM_ATTN_HEADS * SWA_DH;
constexpr int SWA_DK = SWA_NUM_KV_HEADS * SWA_DH;
constexpr int SWA_DV = SWA_DK;

// Analize the GQA patterns;

#if ROLE_IMPL == ATTN_IMPL_2x4x1
    const int GQA_R = 8;
    const int GQA_S = 8;
    const int GQA_T = 8;
    constexpr int GQA_SEGMENT_SIZE = 4; // 8x8x8 or 4x8x8, must be multiple of 4
    constexpr int KV_HEADS_PER_CU  = 2;
    constexpr int Q_HEADS_PER_CU  = Q_HEADS_PER_GROUP * 2;
    constexpr int ATTN_GROUPS_PADDING = ((Q_HEADS_PER_GROUP + GQA_SEGMENT_SIZE - 1) / GQA_SEGMENT_SIZE) * GQA_SEGMENT_SIZE - Q_HEADS_PER_GROUP;

#elif ROLE_IMPL == ATTN_IMPL_4x2x1
    const int GQA_R = 8;
    const int GQA_S = 8;
    const int GQA_T = 8;
    constexpr int GQA_SEGMENT_SIZE = 2; // 8x8x8 or 4x8x8, must be multiple of 4
    constexpr int KV_HEADS_PER_CU  = 4;
    constexpr int Q_HEADS_PER_CU  = Q_HEADS_PER_GROUP * 4;
    constexpr int ATTN_GROUPS_PADDING = ((Q_HEADS_PER_GROUP + GQA_SEGMENT_SIZE - 1) / GQA_SEGMENT_SIZE) * GQA_SEGMENT_SIZE - Q_HEADS_PER_GROUP;

#elif ROLE_IMPL == ATTN_IMPL_1x8x1
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

// The KxQx1 in the impl name is the shape of a single mmul's output lanes: a CU
// holds K KV heads, each with Q Q-heads, and K*Q must fill all t lanes exactly.
static_assert(KV_HEADS_PER_CU * Q_HEADS_PER_GROUP_PADDED == GQA_T,
              "ATTN_IMPL_KxQx1 must cover exactly GQA_T output lanes");
// A CU walks the head dim eight elements at a time.
static_assert(HEAD_DIM % 8 == 0, "HEAD_DIM must be a multiple of 8");
// Every Q head of the role must land in some (KV head, pass) slot.
static_assert(NUM_ATTN_HEADS % (ROLE_KV_HEADS * Q_PASSES_PER_CU) == 0,
              "Q heads do not divide evenly over KV heads and Q passes");

constexpr int VOCAB_SIZE_PADDED = (VOCAB_SIZE + MODEL_DIM - 1) / MODEL_DIM * MODEL_DIM;


///@brief Q40 block
///@param scales: scales and mins, bf16
///@param qs: 4-bit quantized values
typedef struct {
    bf16 scales[QXNX_ROW_BLOCK_SIZE * QXNX_COL_BLOCK_SIZE / GGML_GROUP_SIZE]; // per-group scales and mins
    int4 qs[QXNX_ROW_BLOCK_SIZE * QXNX_COL_BLOCK_SIZE / 2];                      // 4-bit quantized values
} q40_block_t;

///@brief Q80 block
///@param scales: scales and mins, bf16
///@param mins: scales and mins, bf16
///@param qs: 4-bit quantized values
typedef struct {
    bf16 scales[QXNX_ROW_BLOCK_SIZE * QXNX_COL_BLOCK_SIZE / GGML_GROUP_SIZE]; // per-group scales and mins
    int8_t qs[QXNX_ROW_BLOCK_SIZE * QXNX_COL_BLOCK_SIZE];                      // 8--bit quants
} q80_block_t;


#if QUANT_TYPE == __Q4_0__
#define QUANT_BLOCK q40_block_t
#elif QUANT_TYPE == __Q8_0__
#define QUANT_BLOCK q80_block_t
#endif

#if AB_QUANT_TYPE == __Q4_0__
#define AB_QUANT_BLOCK q40_block_t
#elif AB_QUANT_TYPE == __Q8_0__
#define AB_QUANT_BLOCK q80_block_t
#endif
#endif
