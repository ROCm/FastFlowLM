// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __AIE_ARRAY_LAYOUT_H__
#define __AIE_ARRAY_LAYOUT_H__

#include "typedef.hpp"
#include "model_spec.h"

// calculate the number of repeats of x;
// Each round, the MVM engine computes QKV_ROW_BLOCK_SIZE * MVM_CORES of rows for the output;
constexpr int MVM_CORES = 16;
constexpr int QKV_REPEATS = (DQ + DK + DV + DQ) / MVM_CORES / QXNX_ROW_BLOCK_SIZE;
constexpr int L_QKV_REPEATS = (Linear_DQ + Linear_DK + Linear_DV + Linear_DV) / MVM_CORES / QXNX_ROW_BLOCK_SIZE;
constexpr int O_DOWN_REPEATS = MODEL_DIM / MVM_CORES / QXNX_ROW_BLOCK_SIZE;
constexpr int UP_GATE_REPEATS = (2 * INTERMEDIATE_SIZE) / MVM_CORES / QXNX_ROW_BLOCK_SIZE;

constexpr int M_PER_ROUND = QXNX_ROW_BLOCK_SIZE * MVM_CORES;
constexpr int LM_HEAD_ROWS = (VOCAB_SIZE + M_PER_ROUND - 1) / M_PER_ROUND * M_PER_ROUND;

// what one proj_main run computes, read from the RTP written by the host
constexpr int PROJ_MODE_LINEAR  = 0;
constexpr int PROJ_MODE_ATTN    = 1;
constexpr int PROJ_MODE_LM_HEAD = 2;

// what one rms_residual run does, read from its RTP
constexpr int RMS_MODE_FIRST_LINEAR = 0;
constexpr int RMS_MODE_LINEAR       = 1;
constexpr int RMS_MODE_ATTN         = 2;
constexpr int RMS_MODE_LM_HEAD      = 3;

// what one gate run does (STATELESS_X builds read it from the gate core's RTP; the other builds count the layers of a period inside the core)
constexpr int GATE_MODE_LINEAR = 0;   // the linear layer's z gate: SiLU
constexpr int GATE_MODE_ATTN   = 1;   // the attention layer's output gate: sigmoid

constexpr int pkt_id_to_rope     = 17;
constexpr int pkt_id_to_out      = 12;
constexpr int pkt_id_to_conv     = 4;
constexpr int pkt_id_to_aux      = 8;
constexpr int pkt_id_to_glu      = 15;

constexpr int pkt_id_rms_to_proj = 0;
constexpr int pkt_id_rms_to_it   = 1;


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