// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

///@file qwen3.8-27b.h
///@brief Define the model parameters for the qwen3.8-27b model
#ifndef __QWEN3_8_27B_H__
#define __QWEN3_8_27B_H__

// Full-attention (softmax) layers only.  Qwen3.8-27B's gated delta-net layers
// -- three in every four -- are a separate xclbin, ../gate_delta_net_verify,
// and the constants that only they need (L_QK_HEADS, L_V_HEADS, L_Q_HEAD_DIM,
// L_V_HEAD_DIM, LINEAR_ATTN_SCALE) live there, not here.
#if MODEL_TYPE == QWEN3_8_27B
constexpr int   MODEL_DIM         = 5120;
constexpr int   NUM_ATTN_HEADS    = 24;
constexpr int   NUM_KV_HEADS      = 4;
constexpr int   INTERMEDIATE_SIZE = 17408;
constexpr int   GLU_SLICE         = 1024;
constexpr int   DH                = 256;
constexpr int   VOCAB_SIZE        = 248320;
constexpr int   ROPE_DIM          = 64;
constexpr float ATTN_SCALE        = 0.0625f;

#define QUANT_TYPE __Q4_K__
#define AB_QUANT_TYPE __Q8_0__
// 24/4 = 6 q heads per group, so the hardcoded 4-wide 2x4x1 path does not fit.
#define ATTN_IMPL ATTN_IMPL_1x8x1
#define A_FUNC    A_SILU
#define HAS_QK_NORM

constexpr int   QXNX_ROW_BLOCK_SIZE = 32;
constexpr int   QXNX_COL_BLOCK_SIZE = 256;
constexpr int   GGML_GROUP_SIZE     = 32;
#endif

#endif // __QWEN3_8_27B_H__
