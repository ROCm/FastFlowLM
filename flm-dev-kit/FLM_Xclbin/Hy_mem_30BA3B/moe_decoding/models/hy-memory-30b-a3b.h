// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

///@file hy-memory-30b-a3b.h
///@brief Model parameters for tencent/Hy-Memory-30B-A3B (hy_v3), sparse (MoE) layers
#ifndef __HY_MEMORY_30B_A3B_H__
#define __HY_MEMORY_30B_A3B_H__

#if MODEL_TYPE == HY_MEMORY_30B_A3B
constexpr int   MODEL_DIM         = 2048;
constexpr int   NUM_ATTN_HEADS    = 32;
constexpr int   NUM_KV_HEADS      = 4;
constexpr int   DH                = 128;
constexpr int   ROPE_DIM          = 128;               // full RoPE
constexpr float ATTN_SCALE        = 0.08838834764831845f; // 1 / sqrt(DH)
constexpr float RMS_EPS           = 1e-5f;             // input/post norms and q/k norms

// ---- MoE ----
// Experts and the shared expert are 768 wide (config expert_hidden_dim, not
// moe_intermediate_size). One GLU slice is one proj round: 16 cores x 32 rows
// = 512 values, and the host orders the up/gate weight rows so that every
// round holds matching up and gate rows:
//   shared expert: [up 256 | gate 256]                       -> 256 h per round, 3 rounds
//   routed experts: per column (one expert each) [up 64 | gate 64]
//                   -> 64 h per expert per round, 12 rounds
// (2 * 768 / 512 = 3 rounds would not split into 1024-value slices.)
constexpr int   INTERMEDIATE_SIZE = 768;
constexpr int   GLU_SLICE         = 512;
constexpr int   UP_GATE_OUTDIM_REORDEER = 256;         // shared: up (or gate) rows per round
constexpr int   EXP_UP_GATE_OUTDIM_REORDER = 64;       // routed: up (or gate) rows per column per round
constexpr int   NUM_EXPERTS       = 128;
constexpr int   NUM_ACT_EXPERTS   = 8;
// router: sigmoid scores, top 8 of (score + fp32 bias), weight = score / sum * this
constexpr float ROUTER_SCALING    = 2.826f;
// layers the router keeps a weight base address for (num_hidden_layers)
constexpr int   MAX_LAYERS        = 48;

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
#endif

#endif // __HY_MEMORY_30B_A3B_H__
