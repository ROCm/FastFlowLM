// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

///@file gemma4-12b.h
///@brief Define the model parameters for the gemma4-e4b model
#ifndef __GEMMA4_12B_H__
#define __GEMMA4_12B_H__

#if MODEL_TYPE == GEMMA4_12B
constexpr int   MODEL_DIM         = 3840;
constexpr int   NUM_ATTN_HEADS    = 16;
constexpr int   NUM_KV_HEADS      = 1;
constexpr int   SWA_NUM_KV_HEADS  = 8;
constexpr int   INTERMEDIATE_SIZE = 15360;
constexpr int   GLU_SLICE         = 1024;
constexpr int   PLI_D             = 256; // PLI stands for Per Layer Input
constexpr int   SWA_DH            = 256;
constexpr int   DH                = 512;
constexpr int   VOCAB_SIZE        = 262144;
constexpr float SWA_ATTN_SCALE    = 0.0625f;
constexpr float ATTN_SCALE        = 0.04419417382415922f;
constexpr float PER_LAYER_MODEL_PROJECTION_SCALE = 0.7071067811865476f;
constexpr float PER_LAYER_INPUT_SCALE            = 0.01976423537605237f;

#define ATTN_IMPL ATTN_IMPL_1x8x1
#define SWA_ATTN_IMPL ATTN_IMPL_4x2x1
#define A_FUNC    A_GELU
#define HAS_QK_NORM
#define QUANT_TYPE __Q4_0__

constexpr int   QXNX_ROW_BLOCK_SIZE = 16;
constexpr int   QXNX_COL_BLOCK_SIZE = 256;
constexpr int   GGML_GROUP_SIZE     = 32;

#endif


#endif // __GEMMA4_E4B_H__