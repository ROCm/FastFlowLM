// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

///@file qwen2-3b.h
///@brief Define the model parameters for the qwen2-3b model
#ifndef __QWEN2_3B_H__
#define __QWEN2_3B_H__

#if MODEL_TYPE == QWEN2_3B
constexpr int   MODEL_DIM         = 2048;
constexpr int   NUM_ATTN_HEADS    = 16;
constexpr int   NUM_KV_HEADS      = 2;
constexpr int   INTERMEDIATE_SIZE = 11008 + 256; // padding to multiple of 512
constexpr int   GLU_SLICE         = 1024;
constexpr int   DH                = 128;
constexpr int   VOCAB_SIZE        = 151936;
constexpr float ATTN_SCALE        = 0.08838834764831843f;

#define ATTN_IMPL ATTN_IMPL_1x8x1
#define A_FUNC    A_SILU
#define HAS_QKV_BIAS

#define Q4_0

constexpr int   Q4NX_ROW_BLOCK_SIZE = 32;
constexpr int   Q4NX_COL_BLOCK_SIZE = 256;
constexpr int   Q4NX_GROUP_SIZE     = 32;
#endif

#endif // __QWEN2_3B_H__
