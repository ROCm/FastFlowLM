// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __AIE_ARRAY_LAYOUT_H__
#define __AIE_ARRAY_LAYOUT_H__

#include "typedef.hpp"
#include "model_spec.h"

// calculate the number of repeats of x;
// Each round, the MVM engine computes QKV_ROW_BLOCK_SIZE * MVM_CORES of rows for the output;
constexpr int MVM_CORES = 16;
constexpr int L_QKV_REPEATS = (Linear_DQ + Linear_DK + Linear_DV + Linear_DV) / MVM_CORES / QXNX_ROW_BLOCK_SIZE;
constexpr int O_DOWN_REPEATS = MODEL_DIM / MVM_CORES / QXNX_ROW_BLOCK_SIZE;
constexpr int UP_GATE_REPEATS = (2 * INTERMEDIATE_SIZE) / MVM_CORES / QXNX_ROW_BLOCK_SIZE;

constexpr int M_PER_ROUND = QXNX_ROW_BLOCK_SIZE * MVM_CORES;

// The y tree's ids are one-hot, and that is load-bearing -- see the pkt_ids
// comment in iron/proj_tile.py. Keep in step with proj_pkt_id in
// iron/constants.py.
constexpr int pkt_id_to_rms      = 2;    // o and down
constexpr int pkt_id_to_conv     = 4;    // qkv
constexpr int pkt_id_to_aux      = 8;    // z
constexpr int pkt_id_to_glu      = 16;   // up|gate

// The rms tile's three phases, one id each; see rms_pkt_id in iron/constants.py.
constexpr int pkt_id_rms_to_proj = 0;   // phase 1, the input norm
constexpr int pkt_id_rms_to_it   = 1;   // phase 2, the post-attention norm
constexpr int pkt_id_rms_to_out  = 4;   // phase 3, h2, the layer's output


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