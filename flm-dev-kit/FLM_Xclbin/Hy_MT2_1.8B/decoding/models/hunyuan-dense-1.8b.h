// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

///@file hunyuan-dense-1.8b.h
///@brief Define the model parameters for the Hy-MT2-1.8B model
///
/// `hunyuan-dense` is a plain Llama-shaped GQA decoder (see
/// models_simple/HUNYUAN_V1_DENSE.md) with exactly two deviations, only one of
/// which the overlay can see:
///
///   * QK norm is applied *after* RoPE, not before.  Every other QK-norm model
///     in the tree (Qwen3) normalizes first, so this model links the RoPE tile
///     against rope_hunyuan.o instead of rope.o -- see iron/models/hunyuan.py.
///   * The DynamicNTKAlpha rope base rescale is already folded into the GGUF's
///     rope.freq_base (11158840.0, scaling type "none"), so it only affects the
///     host's cos/sin table and nothing here.
#ifndef __HUNYUAN_DENSE_1_8B_H__
#define __HUNYUAN_DENSE_1_8B_H__

#if MODEL_TYPE == HUNYUAN_DENSE_1_8B
constexpr int   MODEL_DIM         = 2048;
constexpr int   NUM_ATTN_HEADS    = 16;
constexpr int   NUM_KV_HEADS      = 4;
constexpr int   INTERMEDIATE_SIZE = 6144;
constexpr int   GLU_SLICE         = 1024;
constexpr int   DH                = 128;
constexpr int   VOCAB_SIZE        = 120818;
constexpr float ATTN_SCALE        = 0.08838834764831843f;  // DH^-0.5

// 4 KV heads -> 1 KV head per attention CU, and 16/4 = 4 Q heads per group.
//
// Not 1x4x1, despite the name fitting: that variant's kernel bodies
// (kernels/attn_qk.cc, kernels/attn_kv.cc) are written for the *two* real Q
// heads gemma3-4b had, zero-filling the other two mmul rows and emitting only
// mout0.first.  Selecting it here computed Q heads 0 and 1 of every group and
// left 2 and 3 unwritten.  1x8x1 pads the group of 4 out to a segment of 8
// (ATTN_GROUPS_PADDING == 4) and is the variant nanbeige-3b and qwen2.5-7b run;
// its kernels emit Q_HEADS_PER_CU real heads and drop the pad in scale_div_aie.
// iron/models/hunyuan.py must carry the matching q_heads_padding = 8.
#define ATTN_IMPL ATTN_IMPL_1x8x1
#define A_FUNC    A_SILU
#define HAS_QK_NORM

constexpr int   Q4NX_ROW_BLOCK_SIZE = 32;
constexpr int   Q4NX_COL_BLOCK_SIZE = 256;
constexpr int   Q4NX_GROUP_SIZE     = 32;
// projection weight format, and the row stride it was packed with
// (32 = plain column major; see all_models.h). The .q4nx carries 4736-byte
// Q4_K projection blocks (q4k_block_t). The lm_head is Q8_0 and runs on the same
// cores as padded half blocks (q80_half_block_t, see README.md "Q4_K body").
#define QUANT_TYPE __Q4_K__
constexpr int   PARALLEL_ROWS       = 32;
// two-limb group sums and column sums: free here (decode is memory bound)
#define QMM_LIMB_PRECISION 1
#endif

#endif // __HUNYUAN_DENSE_1_8B_H__
