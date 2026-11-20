// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __MODEL_SPEC__
#define __MODEL_SPEC__

#include "typedef.hpp"
#include "../models/all_models.h"

constexpr int Q_HEADS_PER_GROUP = NUM_ATTN_HEADS / NUM_KV_HEADS;

constexpr int DQ = NUM_ATTN_HEADS * DH;
constexpr int DK = NUM_KV_HEADS * DH;
constexpr int DV = DK;

constexpr int Linear_DQ = L_QK_HEADS * L_Q_HEAD_DIM;
constexpr int Linear_DK = L_QK_HEADS * L_Q_HEAD_DIM;
constexpr int Linear_DV = L_V_HEADS * L_V_HEAD_DIM;

constexpr int L_V_HEADS_PADDED = (L_V_HEADS + QXNX_ROW_BLOCK_SIZE - 1) / QXNX_ROW_BLOCK_SIZE * QXNX_ROW_BLOCK_SIZE;

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

constexpr int VOCAB_SIZE_PADDED = (VOCAB_SIZE + MODEL_DIM - 1) / MODEL_DIM * MODEL_DIM;

constexpr int ROPE_BY_PASS = DH - ROPE_DIM;

///@brief Q4_1 block: uint4 with a bf16 scale and a bf16 min per group.
///
/// w = s q + m, the signed-min convention. 4 + 16/32 + 16/32 = 5 bits per
/// weight, i.e. 5120 bytes for a 32x256 block. Packed plain column major over
/// the whole 32-row span, so one column of qs is exactly PARALLEL_ROWS nibbles
/// and one group of scales/mins is exactly PARALLEL_ROWS entries.
///@param scales one bf16 per group, group major: scales[g * ROW_BLOCK + r]
///@param mins   same layout as scales
///@param qs     4-bit quantized values, two per byte
typedef struct {
    bf16 scales[QXNX_ROW_BLOCK_SIZE * QXNX_COL_BLOCK_SIZE / GGML_GROUP_SIZE];
    bf16 mins[QXNX_ROW_BLOCK_SIZE * QXNX_COL_BLOCK_SIZE / GGML_GROUP_SIZE];
    uint4 qs[QXNX_ROW_BLOCK_SIZE * QXNX_COL_BLOCK_SIZE / 2];
} q41_block_t;

///@brief Q4_K block: uint4 with a two-level scale/min hierarchy.
///
/// w = S' s'_j q + M' m'_j: a uint8 scale s' and min m' per group, a bf16 S'
/// and M' per 256-weight super-block. One block is exactly one super-block.
/// 4 + 8/32 + 8/32 + 2*16/256 = 4.625 bits per weight, i.e. 4736 bytes for a
/// 32x256 block.
///@param scales uint8 s'_j, group major: scales[g * ROW_BLOCK + r]
///@param mins   uint8 m'_j, same layout as scales
///@param qs     4-bit quantized values, two per byte
///@param S      bf16 super-block scale, one per row
///@param M      bf16 super-block min, one per row
typedef struct {
    uint8 scales[QXNX_ROW_BLOCK_SIZE * QXNX_COL_BLOCK_SIZE / GGML_GROUP_SIZE];
    uint8 mins[QXNX_ROW_BLOCK_SIZE * QXNX_COL_BLOCK_SIZE / GGML_GROUP_SIZE];
    uint4 qs[QXNX_ROW_BLOCK_SIZE * QXNX_COL_BLOCK_SIZE / 2];
    bf16 S[QXNX_ROW_BLOCK_SIZE * QXNX_COL_BLOCK_SIZE / 256];
    bf16 M[QXNX_ROW_BLOCK_SIZE * QXNX_COL_BLOCK_SIZE / 256];
} q4k_block_t;

///@brief Q8_0 block: symmetric int8, one bf16 scale per group, no min.
///
/// w = s q, q signed. 8 + 16/32 = 8.5 bits per weight, i.e. 8704 bytes for a
/// 32x256 block.
///@param scales one bf16 per group, group major: scales[g * ROW_BLOCK + r]
///@param qs     8--bit quants, one per byte
typedef struct {
    bf16 scales[QXNX_ROW_BLOCK_SIZE * QXNX_COL_BLOCK_SIZE / GGML_GROUP_SIZE]; // per-group scales and mins
    int8_t qs[QXNX_ROW_BLOCK_SIZE * QXNX_COL_BLOCK_SIZE];                      // 8--bit quants
} q80_block_t;

///@brief Q8_0 half block for lm_head: 32 rows x 128 columns, same layout as q80_block_t.
///
/// 4352 bytes; padded to one Q4_K block (4736 bytes) in DDR so weight buffers and DMA lengths stay the same.
constexpr int Q80_HALF_COLS = QXNX_COL_BLOCK_SIZE / 2;
typedef struct {
    bf16 scales[QXNX_ROW_BLOCK_SIZE * Q80_HALF_COLS / GGML_GROUP_SIZE];
    int8_t qs[QXNX_ROW_BLOCK_SIZE * Q80_HALF_COLS];
} q80_half_block_t;


#if QUANT_TYPE == __Q4_1__
#define QUANT_BLOCK q41_block_t
#elif QUANT_TYPE == __Q4_K__
#define QUANT_BLOCK q4k_block_t
#elif QUANT_TYPE == __Q8_0__
#define QUANT_BLOCK q80_block_t
#else
#error "QUANT_TYPE must be one of __Q4_1__, __Q4_K__, __Q8_0__"
#endif

#if AB_QUANT_TYPE == __Q4_1__
#define AB_QUANT_BLOCK q41_block_t
#elif AB_QUANT_TYPE == __Q8_0__
#define AB_QUANT_BLOCK q80_block_t
#else
#error "AB_QUANT_TYPE must be one of __Q4_1__, __Q8_0__"
#endif

// The DMA hands the cores exactly one packed block, so a mismatch between these
// structs and the packer is a silently misread weight rather than a fault.
static_assert(sizeof(q41_block_t) == QXNX_ROW_BLOCK_SIZE * QXNX_COL_BLOCK_SIZE * 5 / 8,
              "q41_block_t must be 5 bits per weight");
static_assert(sizeof(q4k_block_t) == QXNX_ROW_BLOCK_SIZE * QXNX_COL_BLOCK_SIZE * 37 / 64,
              "q4k_block_t must be 4.625 bits per weight");
static_assert(sizeof(q80_block_t) == QXNX_ROW_BLOCK_SIZE * QXNX_COL_BLOCK_SIZE * 17 / 16,
              "q80_block_t must be 8.5 bits per weight");

#endif
