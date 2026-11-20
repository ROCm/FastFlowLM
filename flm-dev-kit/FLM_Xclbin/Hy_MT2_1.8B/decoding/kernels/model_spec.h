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

// Analize the GQA patterns
constexpr int NUM_MHA_CU      = 4;
constexpr int KV_HEADS_PER_CU = NUM_KV_HEADS / NUM_MHA_CU;
constexpr int Q_HEADS_PER_CU  = NUM_ATTN_HEADS / NUM_MHA_CU;

#if ATTN_IMPL == ATTN_IMPL_2x4x1
    const int GQA_R = 8;
    const int GQA_S = 8;
    const int GQA_T = 8;
    constexpr int GQA_SEGMENT_SIZE = 4; // 8x8x8 or 4x8x8, must be multiple of 4
    constexpr int ATTN_GROUPS_PADDING = ((Q_HEADS_PER_GROUP + GQA_SEGMENT_SIZE - 1) / GQA_SEGMENT_SIZE) * GQA_SEGMENT_SIZE - Q_HEADS_PER_GROUP;
#elif ATTN_IMPL == ATTN_IMPL_1x4x1
    const int GQA_R = 4;
    const int GQA_S = 8;
    const int GQA_T = 8;
    constexpr int GQA_SEGMENT_SIZE = 4; // 4x8x8, must be multiple of 4
    constexpr int ATTN_GROUPS_PADDING = ((Q_HEADS_PER_GROUP + GQA_SEGMENT_SIZE - 1) / GQA_SEGMENT_SIZE) * GQA_SEGMENT_SIZE - Q_HEADS_PER_GROUP;
#elif ATTN_IMPL == ATTN_IMPL_1x8x1
    const int GQA_R = 8;
    const int GQA_S = 8;
    const int GQA_T = 8;
    constexpr int GQA_SEGMENT_SIZE = 8; // 8x8x8, must be multiple of 8
    constexpr int ATTN_GROUPS_PADDING = ((Q_HEADS_PER_GROUP + GQA_SEGMENT_SIZE - 1) / GQA_SEGMENT_SIZE) * GQA_SEGMENT_SIZE - Q_HEADS_PER_GROUP;
#endif


constexpr int Q_HEADS_PER_GROUP_PADDED = Q_HEADS_PER_GROUP + ATTN_GROUPS_PADDING;

// ATTN_IMPL_1x4x1's kernel bodies (attn_qk.cc, attn_kv.cc) are not generic in
// the group size the way the other two are: they emit exactly two score rows,
// zero-fill the remaining mmul rows, and attn_kv's caller hardcodes an 8*DH
// scale_div_aie that overruns a 2-head O buffer.  That was written for
// gemma3-4b's 8/4 shape and never generalized, so a 4-Q-head group silently
// lost heads 2 and 3.  Use ATTN_IMPL_1x8x1 for those -- it pads to 8 and its
// kernels honour Q_HEADS_PER_CU -- until someone finishes 1x4x1.
#if ATTN_IMPL == ATTN_IMPL_1x4x1
static_assert(Q_HEADS_PER_GROUP == 2 && KV_HEADS_PER_CU == 1,
              "ATTN_IMPL_1x4x1 only implements 2 Q heads per CU; use ATTN_IMPL_1x8x1");
#endif

constexpr int Q_HEADS_PADDED_PER_CU = KV_HEADS_PER_CU * Q_HEADS_PER_GROUP_PADDED;

constexpr int VOCAB_SIZE_PADDED = (VOCAB_SIZE + MODEL_DIM - 1) / MODEL_DIM * MODEL_DIM;

// Opt-in, from models/<model>.h: the cores sequence the token themselves --
// NUM_HIDDEN_LAYERS decoder layers then the LM head, per core-loop iteration --
// instead of reading the mode out of an RTP buffer.  See all_models.h.
#ifdef SELF_SEQUENCED_LM_HEAD
static_assert(NUM_HIDDEN_LAYERS > 0,
              "SELF_SEQUENCED_LM_HEAD needs NUM_HIDDEN_LAYERS from models/<model>.h");
#endif

///@brief Q4_1 block: uint4 with a bf16 scale and a bf16 min per group.
///
/// w = s q + m, the signed-min convention. 4 + 16/32 + 16/32 = 5 bits per
/// weight, i.e. 5120 bytes for a 32x256 block. Column major with a
/// PARALLEL_ROWS-row column (see all_models.h). This is the format the layer has
/// shipped from the start.
///@param scales one bf16 per group, group major: scales[g * ROW_BLOCK + r]
///@param mins   same layout as scales
///@param qs     4-bit quants, two per byte
typedef struct {
    bf16 scales[Q4NX_ROW_BLOCK_SIZE * Q4NX_COL_BLOCK_SIZE / Q4NX_GROUP_SIZE];
    bf16 mins[Q4NX_ROW_BLOCK_SIZE * Q4NX_COL_BLOCK_SIZE / Q4NX_GROUP_SIZE];
    uint4 qs[Q4NX_ROW_BLOCK_SIZE * Q4NX_COL_BLOCK_SIZE / 2];
} q41_block_t;

///@brief Q4_K block: uint4 with a two-level scale/min hierarchy.
///
/// w = S' s'_j q - M' m'_j: a uint8 scale s' and min m' per group, a bf16 S'
/// and M' per 256-weight super-block (FLM_Q4NX_Converter/quant.md). One block
/// is exactly one super-block. 4 + 8/32 + 8/32 + 2*16/256 = 4.625 bits per
/// weight, i.e. 4736 bytes for a 32x256 block.
///
/// Q4_K is only ever packed plain column major over the whole 32-row span
/// (FLM_Q4NX_Converter's _pack_q4k), so it requires PARALLEL_ROWS ==
/// Q4NX_ROW_BLOCK_SIZE and the kernel needs one pass, not two.
///@param scales uint8 s'_j, group major: scales[g * ROW_BLOCK + r]
///@param mins   uint8 m'_j, same layout as scales
///@param qs     4-bit quants, two per byte
///@param S      bf16 super-block scale, one per row
///@param M      bf16 super-block min, one per row
typedef struct {
    uint8 scales[Q4NX_ROW_BLOCK_SIZE * Q4NX_COL_BLOCK_SIZE / Q4NX_GROUP_SIZE];
    uint8 mins[Q4NX_ROW_BLOCK_SIZE * Q4NX_COL_BLOCK_SIZE / Q4NX_GROUP_SIZE];
    uint4 qs[Q4NX_ROW_BLOCK_SIZE * Q4NX_COL_BLOCK_SIZE / 2];
    bf16 S[Q4NX_ROW_BLOCK_SIZE * Q4NX_COL_BLOCK_SIZE / 256];
    bf16 M[Q4NX_ROW_BLOCK_SIZE * Q4NX_COL_BLOCK_SIZE / 256];
} q4k_block_t;

///@brief Q8_0 block: symmetric int8, one bf16 scale per group, no min.
///
/// w = s q, q signed. 8 + 16/32 = 8.5 bits per weight, i.e. 8704 bytes for a
/// 32x256 block -- nearly twice Q4_1, so this is the accuracy-first format, not
/// a bandwidth one. Packed plain column major over the whole row span
/// (FLM_Q4NX_Converter's _pack_q8nx), like Q4_K.
///@param scales one bf16 per group, group major: scales[g * ROW_BLOCK + r]
///@param qs     8-bit quants, one per byte
typedef struct {
    bf16 scales[Q4NX_ROW_BLOCK_SIZE * Q4NX_COL_BLOCK_SIZE / Q4NX_GROUP_SIZE];
    int8 qs[Q4NX_ROW_BLOCK_SIZE * Q4NX_COL_BLOCK_SIZE];
} q80_block_t;

#if QUANT_TYPE == __Q4_1__
    #define QUANT_BLOCK q41_block_t
    #define QUANT_HAS_MIN 1
#elif QUANT_TYPE == __Q4_K__
    #define QUANT_BLOCK q4k_block_t
    #define QUANT_HAS_MIN 1
#elif QUANT_TYPE == __Q8_0__
    #define QUANT_BLOCK q80_block_t
    /// @brief Q8_0 is symmetric, so there is no min path to accumulate.
    #define QUANT_HAS_MIN 0
#else
    #error "QUANT_TYPE must be one of __Q4_1__, __Q4_K__, __Q8_0__"
#endif

// The DMA hands the cores exactly one packed block, so a mismatch here is a
// silently misread weight rather than a fault. Pin all three sizes.
static_assert(sizeof(q41_block_t) == Q4NX_ROW_BLOCK_SIZE * Q4NX_COL_BLOCK_SIZE * 5 / 8,
              "q41_block_t must be 5 bits per weight");
static_assert(sizeof(q4k_block_t) == Q4NX_ROW_BLOCK_SIZE * Q4NX_COL_BLOCK_SIZE * 37 / 64,
              "q4k_block_t must be 4.625 bits per weight");
static_assert(sizeof(q80_block_t) == Q4NX_ROW_BLOCK_SIZE * Q4NX_COL_BLOCK_SIZE * 17 / 16,
              "q80_block_t must be 8.5 bits per weight");

/// @brief Packed size of one weight block in bytes, for the DMA descriptors.
constexpr int QUANT_BLOCK_BYTES = sizeof(QUANT_BLOCK);

///@brief Half of a Q8_0 32x256 block: 32 rows x 128 columns, for the lm_head.
///
/// The lm_head is Q8_0, which does not fit one QUANT_BLOCK buffer, so the host
/// splits every 32x256 block into two halves and pads each to QUANT_BLOCK_BYTES
/// (FLM_DLL hunyuan_desc::reorder_cpy_lm_head). The core reads two weight
/// buffers per x buffer for it.
///@param scales one bf16 per group, group major: scales[g * ROW_BLOCK + r]
///@param qs     int8 quants, column major, 32 rows per column
constexpr int Q80_HALF_COLS = Q4NX_COL_BLOCK_SIZE / 2;
typedef struct {
    bf16 scales[Q4NX_ROW_BLOCK_SIZE * Q80_HALF_COLS / Q4NX_GROUP_SIZE];
    int8 qs[Q4NX_ROW_BLOCK_SIZE * Q80_HALF_COLS];
} q80_half_block_t;

static_assert(sizeof(q80_half_block_t) <= QUANT_BLOCK_BYTES,
              "a Q8_0 half block must fit one weight buffer");

#endif
