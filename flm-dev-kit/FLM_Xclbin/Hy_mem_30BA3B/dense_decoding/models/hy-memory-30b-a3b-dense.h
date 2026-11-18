// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

///@file hy-memory-30b-a3b-dense.h
///@brief Model parameters for tencent/Hy-Memory-30B-A3B (hy_v3), dense layer
///
/// Only layer 0 is dense (config first_k_dense_replace = 1); layers 1-47 are
/// MoE and run on ../moe_decoding.  The dense layer is a plain GQA attention +
/// SwiGLU MLP, so it runs on this generic overlay with two things to know:
///
///   * Q/K RMS norms run *before* RoPE (like Qwen3, unlike Hy-MT2), so the
///     RoPE tile uses the default rope.o, not rope_hunyuan.o.
///   * rms_norm_eps is 1e-5 (the tree default is 1e-6), see RMS_EPS below.
#ifndef __HY_MEMORY_30B_A3B_DENSE_H__
#define __HY_MEMORY_30B_A3B_DENSE_H__

#if MODEL_TYPE == HY_MEMORY_30B_A3B_DENSE
constexpr int   MODEL_DIM         = 2048;
constexpr int   NUM_ATTN_HEADS    = 32;
constexpr int   NUM_KV_HEADS      = 4;
constexpr int   INTERMEDIATE_SIZE = 6912;
// 2 * 6912 = 13824 does not split into 1024-value slices, so one GLU slice is
// one proj round (16 cores x 32 rows = 512 values): the host orders the up/gate
// rows [up 256 | gate 256] per round, the same as moe_decoding's shared expert.
constexpr int   GLU_SLICE         = 512;
constexpr int   DH                = 128;
constexpr int   VOCAB_SIZE        = 120832;
constexpr float ATTN_SCALE        = 0.08838834764831845f;  // DH^-0.5

// input/post-attention norms and the per-head q/k norms (model_spec.h default 1e-6)
#define RMS_EPS 1e-5f

// 4 KV heads -> 1 KV head per attention CU, and 32/4 = 8 Q heads per group:
// exactly one 1x8x1 segment, so no pad heads (iron q_heads_padding = 8).
#define ATTN_IMPL ATTN_IMPL_1x8x1
#define A_FUNC    A_SILU
#define HAS_QK_NORM

constexpr int   Q4NX_ROW_BLOCK_SIZE = 32;
constexpr int   Q4NX_COL_BLOCK_SIZE = 256;
constexpr int   Q4NX_GROUP_SIZE     = 32;
// projection weight format, and the row stride it was packed with
// (16 = two 16-row planes, 32 = plain column major; see all_models.h).
// FLM_Q4NX_Converter configs/hy_mem_30b_a3b.json: Q4_K, 4736-byte 32x256
// blocks, plain column major.  lm_head is Q8_0, so do NOT run this build's
// LM-head mode on it.
#define QUANT_TYPE __Q4_K__
constexpr int   PARALLEL_ROWS       = 32;
#endif

#endif // __HY_MEMORY_30B_A3B_DENSE_H__
