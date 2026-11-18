// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

///@file qwen3.8-27b.h
///@brief Define the model parameters for the qwen3.8-27b model
#ifndef __QWEN3_8_27B_H__
#define __QWEN3_8_27B_H__

// Delta-net (linear attention) layers only.  Qwen3.8-27B's full-attention
// layers -- every 4th one -- are a separate xclbin, ../gated_attention_verify,
// and the constants that only they need (NUM_ATTN_HEADS, DH, ROPE_DIM,
// ATTN_SCALE, ATTN_IMPL, HAS_QK_NORM) live there, not here.
#if MODEL_TYPE == QWEN3_8_27B
constexpr int   MODEL_DIM         = 5120;
constexpr int   INTERMEDIATE_SIZE = 17408;
constexpr int   GLU_SLICE         = 1024;
constexpr int   VOCAB_SIZE        = 248320;
constexpr int   L_Q_HEAD_DIM      = 128;
constexpr int   L_V_HEAD_DIM      = 128;
constexpr int   L_QK_HEADS        = 16;
constexpr int   L_V_HEADS         = 48;
constexpr float LINEAR_ATTN_SCALE = 0.08838834764831845f;

#define QUANT_TYPE __Q4_K__
#define AB_QUANT_TYPE __Q8_0__
#define A_FUNC    A_SILU

constexpr int   QXNX_ROW_BLOCK_SIZE = 32;
constexpr int   QXNX_COL_BLOCK_SIZE = 256;
constexpr int   GGML_GROUP_SIZE     = 32;
#endif

#endif // __QWEN3_8_27B_H__
