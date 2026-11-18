// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __AIE_ARRAY_LAYOUT_H__
#define __AIE_ARRAY_LAYOUT_H__

#include "typedef.hpp"
#include "model_spec.h"

// calculate the number of repeats of x;
// Each round, the MVM engine computes QKV_ROW_BLOCK_SIZE * MVM_CORES of rows for the output;
constexpr int MVM_CORES = 16;
constexpr int QKV_REPEATS = (DQ + DK + DV) / MVM_CORES / QXNX_ROW_BLOCK_SIZE;
constexpr int Q_REPEATS = DQ / MVM_CORES / QXNX_ROW_BLOCK_SIZE;
constexpr int QK_REPEATS = (DQ + DK) / MVM_CORES / QXNX_ROW_BLOCK_SIZE;
constexpr int SWA_QKV_REPEATS = (SWA_DQ + SWA_DK + SWA_DV) / MVM_CORES / QXNX_ROW_BLOCK_SIZE;
constexpr int SWA_Q_REPEATS = SWA_DQ / MVM_CORES / QXNX_ROW_BLOCK_SIZE;
constexpr int O_DOWN_REPEATS = MODEL_DIM / MVM_CORES / QXNX_ROW_BLOCK_SIZE;
constexpr int UP_GATE_REPEATS = (2 * INTERMEDIATE_SIZE) / MVM_CORES / QXNX_ROW_BLOCK_SIZE;

constexpr int M_PER_ROUND = QXNX_ROW_BLOCK_SIZE * MVM_CORES;

constexpr int pkt_id_to_rope     = 1;
constexpr int pkt_id_to_swa_rope = 2;
constexpr int pkt_id_to_rms      = 4;
constexpr int pkt_id_to_glu      = 8;

constexpr int pkt_id_rms_to_proj = 0;
constexpr int pkt_id_rms_to_it   = 1;


constexpr bfloat16 exp_scale = 1.44269504089;
constexpr int W_SLICE_SIZE = 512;
constexpr int Q4_WEIGHTS_COUNT = W_SLICE_SIZE * 2;
constexpr int Q8_WEIGHTS_COUNT = W_SLICE_SIZE;
constexpr int PARALLEL_ROWS = 16;
// proj related

constexpr int Q4_COL_PER_SLICE = W_SLICE_SIZE / PARALLEL_ROWS * 2;
constexpr int Q8_COL_PER_SLICE = W_SLICE_SIZE / PARALLEL_ROWS;

#endif // __AIE_ARRAY_LAYOUT_H__