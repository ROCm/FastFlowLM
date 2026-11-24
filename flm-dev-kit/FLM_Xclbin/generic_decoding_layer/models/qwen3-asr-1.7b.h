// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

///@file qwen3-asr-1.7b.h
///@brief Define the model parameters for the Qwen3-ASR-1.7B text decoder (Confucius4-R2T2)
///
/// Same shape as qwen3-1.7b.h, but the weights are Q8_0: 4-bit round-to-nearest (Q4_0 / Q4_1 / Q4_K) breaks this
/// fine-tune (empty or wrong-language transcripts, see the R2T2 plan), Q8_0 is lossless. Q8_0 needs plain 32-row
/// column-major blocks, i.e. PARALLEL_ROWS == Q4NX_ROW_BLOCK_SIZE, which is what FLM_Q4NX_Converter's _pack_q8nx
/// emits with configs/qwen3_asr.json (parallel_size 32).
#ifndef __QWEN3_ASR_1_7B_H__
#define __QWEN3_ASR_1_7B_H__

#if MODEL_TYPE == QWEN3_ASR_1_7B
constexpr int   MODEL_DIM         = 2048;
constexpr int   NUM_ATTN_HEADS    = 16;
constexpr int   NUM_KV_HEADS      = 8;
constexpr int   INTERMEDIATE_SIZE = 6144;
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
// projection weight format (also the LM head), and the row stride it was packed with
// (32 = plain column major, what Q8_0 / Q4_K need; see all_models.h)
#define QUANT_TYPE __Q8_0__
constexpr int   PARALLEL_ROWS       = 32;
// The default (0) rounds every 32-column group sum to bf16 before it meets the scale: ~2e-3 relative, correlated across the
// groups, not averaged away over the K loop. This fine-tune is sensitive to it (English WER +0.95 pts vs the lossless
// CPU path on the R2T2 eval set), so keep two bf16 limbs (~17 bits) at +28% in the projection loop. See README 5.3.
#define QMM_LIMB_PRECISION 1
#endif

#endif // __QWEN3_ASR_1_7B_H__
