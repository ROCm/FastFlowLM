// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __ALL_MODELS_H__
#define __ALL_MODELS_H__

/// @brief Attention implementation
/// @note 2x4x1: for 8 kv heads
#define ATTN_IMPL_2x4x1 0
/// @brief Attention implementation
/// @note 2x4x1: for 4 kv heads
#define ATTN_IMPL_1x4x1 1
/// @brief Attention implementation
/// @note 1x8x1: for 4 kv heads
#define ATTN_IMPL_1x8x1 2

/// @brief Activation function
#define A_SILU 0
/// @brief Activation function
#define A_GELU 1

/// @brief Projection weight formats.
/// @note Values match the Qwen3_5 decoding layer's all_models.h so a model
///       header can be moved between the two without renumbering.
#define __Q4_1__           0
#define __Q8_0__           1
#define __Q4_K__           2
#define __Q4_0__           3

// A weight block is Q4NX_ROW_BLOCK_SIZE x Q4NX_COL_BLOCK_SIZE and is packed
// column major, with each column being PARALLEL_ROWS elements long and the next
// column starting PARALLEL_ROWS elements later:
/*
[0, PARALLEL_ROWS    , .... (Q4NX_COL_BLOCK_SIZE-1)*PARALLEL_ROWS    ],
[1, PARALLEL_ROWS + 1, .... (Q4NX_COL_BLOCK_SIZE-1)*PARALLEL_ROWS + 1],
[2, PARALLEL_ROWS + 2, .... (Q4NX_COL_BLOCK_SIZE-1)*PARALLEL_ROWS + 2],
...
*/
// PARALLEL_ROWS is therefore both the packing stride and how many rows of a
// block one vector operand covers, and the block's rows take
// ROW_PASSES = Q4NX_ROW_BLOCK_SIZE / PARALLEL_ROWS passes.  It is a *per model*
// constant, declared in models/<model>.h next to QUANT_TYPE, because it is a
// property of how FLM_Q4NX_Converter packed that particular checkpoint:
//
//   16 -> the parallel-strided re-order: a 32-row block arrives as two 16-row
//         planes and the MVM cores walk it in two half-width passes.  This is
//         what _pack_q4nx emits, i.e. every Q4_0 / Q4_1 model in the tree.
//   32 -> plain column major over the whole 32-row span, one full-width pass.
//         This is what _pack_q4k and _pack_q8nx emit, so Q4_K and Q8_0 require
//         it; aie_array_layout.h asserts that.

#define LLAMA_3_2_1B         0
#define LLAMA_3_2_3B         1
#define LLAMA_3_1_8B         2
#define QWEN3_0_6B           3
#define QWEN3_1_7B           4
#define QWEN3_4B             5
#define QWEN3_8B             6
#define PHI4_4B              7
#define NANBEIGE_3B          8
#define HUNYUAN_DENSE_1_8B   9
#define HY_MEMORY_30B_A3B_DENSE 10

#ifndef MODEL_TYPE
#define MODEL_TYPE HY_MEMORY_30B_A3B_DENSE
#endif

#include "../models/llama3.2-1b.h"
#include "../models/llama3.2-3b.h"
#include "../models/llama3.1-8b.h"
#include "../models/qwen3-0.6b.h"
#include "../models/qwen3-1.7b.h"
#include "../models/qwen3-4b.h"
#include "../models/qwen3-8b.h"
#include "../models/phi4-4b.h"
#include "../models/nanbeige-3b.h"
#include "../models/hunyuan-dense-1.8b.h"
#include "../models/hy-memory-30b-a3b-dense.h"

// SELF_SEQUENCED_LM_HEAD is opt-in, per model.  Defining it (and, with it,
// NUM_HIDDEN_LAYERS) makes proj_main.cc and rms_residual.cc run
// NUM_HIDDEN_LAYERS decoder layers and then the LM head per core-loop
// iteration, deciding the mode from their own position in that loop rather than
// reading IS_ATTN out of an RTP buffer.  A model that does not define it keeps
// the RTP protocol unchanged.  The two are mutually exclusive at run time: with
// the switch on, the host must stop writing RTP_*_IS_ATTN_BUFFER and stop
// releasing lock 6, because nothing acquires it any more.  RTP_L_attn_* (the
// context length) is a different buffer and is unaffected.

// Every model header picks its own QUANT_TYPE and PARALLEL_ROWS.  Only the
// format falls back to a default -- the layer has shipped Q4_1 from the start --
// while a missing PARALLEL_ROWS is deliberately left as a compile error: there is
// no value for it that is safe to guess, a wrong one silently misreads weights.
#ifndef QUANT_TYPE
#define QUANT_TYPE __Q4_1__
#endif

#endif // __ALL_MODELS_H__