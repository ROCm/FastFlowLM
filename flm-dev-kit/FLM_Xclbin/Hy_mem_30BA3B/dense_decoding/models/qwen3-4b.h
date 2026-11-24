// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

///@file qwen3-4b.h
///@brief Define the model parameters for the qwen3-4b model
#ifndef __QWEN3_4B_H__
#define __QWEN3_4B_H__

#if MODEL_TYPE == QWEN3_4B
constexpr int   MODEL_DIM         = 2560;
constexpr int   NUM_ATTN_HEADS    = 32;
constexpr int   NUM_KV_HEADS      = 8;
constexpr int   INTERMEDIATE_SIZE = 9728;
constexpr int   GLU_SLICE         = 1024;
constexpr int   DH                = 128;
constexpr int   VOCAB_SIZE        = 151936;
constexpr float ATTN_SCALE        = 0.08838834764831843f;

#define ATTN_IMPL ATTN_IMPL_2x4x1
#define A_FUNC    A_SILU
#define HAS_QK_NORM

constexpr int   Q4NX_ROW_BLOCK_SIZE = 32;
constexpr int   Q4NX_COL_BLOCK_SIZE = 256;
constexpr int   Q4NX_GROUP_SIZE     = 32;
// projection weight format, and the row stride it was packed with
// (16 = two 16-row planes, 32 = plain column major; see all_models.h)
#define QUANT_TYPE __Q4_1__
constexpr int   PARALLEL_ROWS       = 16;
#endif

#endif // __QWEN3_4B_H__