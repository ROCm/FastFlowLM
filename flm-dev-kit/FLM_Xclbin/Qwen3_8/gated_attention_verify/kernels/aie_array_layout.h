// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#ifndef __AIE_ARRAY_LAYOUT_H__
#define __AIE_ARRAY_LAYOUT_H__

#include "typedef.hpp"
#include "model_spec.h"

// calculate the number of repeats of x;
// Each round, the MVM engine computes QKV_ROW_BLOCK_SIZE * MVM_CORES of rows for the output;
constexpr int MVM_CORES = 16;
// The two mvms proj_main() runs off the one resident x: the fused qkv
// projection and the output gate z. Their sum must equal the `replays` of the
// matching x_replay_phase in iron/layer_aie.py -- nothing cross-checks the two
// files, and the engine parks on the x acquire if they disagree.
//
// z reduces over the same MODEL_DIM as qkv and lands in the same replay
// buffer, so it costs a longer replay rather than a second phase. That is the
// whole reason it is the cheapest stage to add: o_proj and the MLP reduce over
// DQ and INTERMEDIATE_SIZE and each need a phase of their own.
//
// O_DOWN_REPEATS is that phase's count for o_proj, which is MODEL_DIM rows over
// K = DQ. down_proj shares the constant because it is the same output width.
constexpr int QKV_ONLY_REPEATS = (DQ + DK + DV) / MVM_CORES / QXNX_ROW_BLOCK_SIZE;  // 16
constexpr int Z_REPEATS         = DQ / MVM_CORES / QXNX_ROW_BLOCK_SIZE;             // 12
constexpr int QKV_REPEATS       = QKV_ONLY_REPEATS + Z_REPEATS;                     // 28
constexpr int O_DOWN_REPEATS = MODEL_DIM / MVM_CORES / QXNX_ROW_BLOCK_SIZE;         // 10
constexpr int UP_GATE_REPEATS = (2 * INTERMEDIATE_SIZE) / MVM_CORES / QXNX_ROW_BLOCK_SIZE;

// The two ping/pong chains in proj_main.cc are static across invocations, so
// what has to stay even is the *sum* over every mvm the kernel runs, not each
// term. Three different Ks now appear -- MODEL_DIM, DQ and INTERMEDIATE_SIZE --
// so the x-slice total is a sum of products rather than one.
//
//   row blocks  28 (qkv|z) + 10 (o) + 68 (up|gate) + 10 (down)      = 116
//   x slices    28*20 + 10*24 + 68*20 + 10*68 = 560+240+1360+680    = 2840
//
// O_DOWN_REPEATS covers both o_proj and down_proj: same output width, different
// K.
constexpr int TOTAL_ROW_BLOCKS = QKV_REPEATS + O_DOWN_REPEATS
                               + UP_GATE_REPEATS + O_DOWN_REPEATS;
constexpr int TOTAL_X_SLICES   = QKV_REPEATS     * (MODEL_DIM / QXNX_COL_BLOCK_SIZE)
                               + O_DOWN_REPEATS  * (DQ / QXNX_COL_BLOCK_SIZE)
                               + UP_GATE_REPEATS * (MODEL_DIM / QXNX_COL_BLOCK_SIZE)
                               + O_DOWN_REPEATS  * (INTERMEDIATE_SIZE / QXNX_COL_BLOCK_SIZE);

constexpr int M_PER_ROUND = QXNX_ROW_BLOCK_SIZE * MVM_CORES;

// The y tree's ids are one-hot, and that is load-bearing -- see the pkt_ids
// comment in iron/proj_tile.py. Keep in step with proj_pkt_id in
// iron/constants.py; the kernel writes these into the payload, so a change here
// is not local to this file.
constexpr int pkt_id_to_rms      = 2;    // o
constexpr int pkt_id_to_rope     = 4;    // qkv
constexpr int pkt_id_to_aux      = 8;    // z
constexpr int pkt_id_to_glu      = 16;   // up|gate

// The ids the rms kernel stamps on its y rows, one per phase. Keep in step with
// rms_pkt_id in iron/constants.py; the kernel writes these into the payload, so
// a change here is not local to this file.
constexpr int pkt_id_rms_to_proj = 0;   // phase 1, the input norm     -> MT[2]
constexpr int pkt_id_rms_to_st   = 1;   // phase 2, the post-attn norm -> MT[2]
constexpr int pkt_id_rms_to_out  = 4;   // phase 3, h2, no norm        -> IT[2]


constexpr bfloat16 exp_scale = 1.44269504089;
constexpr int W_SLICE_SIZE = 512;
constexpr int Q4_WEIGHTS_COUNT = W_SLICE_SIZE * 2;
constexpr int Q8_WEIGHTS_COUNT = W_SLICE_SIZE;
// The MVM cores hold a whole weight block's rows at once: one column of a block
// is exactly PARALLEL_ROWS elements, so this is a layout constant tied to the
// block shape, not a tuning knob.
constexpr int PARALLEL_ROWS = QXNX_ROW_BLOCK_SIZE;
// proj related

constexpr int Q4_COL_PER_SLICE = W_SLICE_SIZE / PARALLEL_ROWS * 2;
constexpr int Q8_COL_PER_SLICE = W_SLICE_SIZE / PARALLEL_ROWS;

#endif // __AIE_ARRAY_LAYOUT_H__