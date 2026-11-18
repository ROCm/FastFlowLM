// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

///@file qwen3-asr-1.7b-mix.h
///@brief Qwen3-ASR-1.7B text decoder (Confucius4-R2T2) with mixed 4/8-bit projection weights
///
/// Same shape as qwen3-asr-1.7b.h. 4-bit everywhere breaks this fine-tune (empty or wrong-language transcripts), but Q4_1 for
/// q/k/o/gate/up with v_proj and down_proj at Q8_0 matches BF16 on the R2T2 eval set; the decode reads ~25 % fewer weight bytes.
/// Every weight DMA is one 10240-byte unit (QUANT_TYPE __Q41_Q80_MIX__): two Q4_1 32x256 blocks side by side (a 32x512 span) or one
/// Q8_0 32x256 block and 1536 bytes of padding. The MIX_*_Q8 switches below say which projections are Q8_0; the host packs the
/// units the same way (FLM_DLL/detail/qwen3_asr_npu).
#ifndef __QWEN3_ASR_1_7B_MIX_H__
#define __QWEN3_ASR_1_7B_MIX_H__

#if MODEL_TYPE == QWEN3_ASR_1_7B_MIX
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
#define QUANT_TYPE __Q41_Q80_MIX__
// both formats plain column major over the 32-row span (one full-width pass)
constexpr int   PARALLEL_ROWS       = 32;
// two bf16 limbs on the group sums, as the Q8_0 build (see qwen3-asr-1.7b.h)
#define QMM_LIMB_PRECISION 1
// which projections are Q8_0; the others are Q4_1
#define MIX_V_Q8    1
#define MIX_DOWN_Q8 1
#define MIX_HEAD_Q8 0
#endif

#endif // __QWEN3_ASR_1_7B_MIX_H__
