// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

///@file qwen3.6-35b-a3b.h
///@brief Define the model parameters for the qwen3-5-35b-a3b model
#ifndef __QWEN3_6_35B_A3B_H__
#define __QWEN3_6_35B_A3B_H__

#if MODEL_TYPE == QWEN3_6_35B_A3B
constexpr int   MODEL_DIM         = 2048;
constexpr int   NUM_ATTN_HEADS    = 16;
constexpr int   NUM_KV_HEADS      = 2;
constexpr int   INTERMEDIATE_SIZE = 512;
constexpr int   GLU_SLICE         = 1024;
constexpr int   UP_GATE_OUTDIM_REORDEER = 512;
constexpr int   EXP_UP_GATE_OUTDIM_REORDER = 128;


constexpr int   DH                = 256;
constexpr int   VOCAB_SIZE        = 248320;
constexpr int   ROPE_DIM          = 64;
constexpr int   L_Q_HEAD_DIM      = 128;
constexpr int   L_V_HEAD_DIM      = 128;
constexpr int   L_QK_HEADS        = 16;
constexpr int   L_V_HEADS         = 32;
constexpr int   LAYER_PERIOD      = 4;
constexpr int   NUM_EXPERTS       = 256;
constexpr int   BF16MVM_COL       = 64;
constexpr int   NUM_ACT_EXPERTS   = 8;
constexpr float ATTN_SCALE        = 0.0625f;
constexpr float LINEAR_ATTN_SCALE = 0.08838834764831845f;

#define QUANT_TYPE __Q4_K__
#define ATTN_IMPL ATTN_IMPL_1x8x1
#define A_FUNC    A_SILU
#define HAS_QK_NORM

constexpr int   QXNX_ROW_BLOCK_SIZE = 32;
constexpr int   QXNX_COL_BLOCK_SIZE = 256;
constexpr int   GGML_GROUP_SIZE     = 32;

constexpr int   PROJ_X_SLICE = QXNX_COL_BLOCK_SIZE;
// for decode reorder only
constexpr int router_weight_row_block= 32;
constexpr int router_weight_col_block = 256;


constexpr int ab_proj_weight_row_block= 32;
constexpr int ab_proj_weight_col_block = 256;
#endif

#endif // __QWEN3_6_35B_A3B_H__