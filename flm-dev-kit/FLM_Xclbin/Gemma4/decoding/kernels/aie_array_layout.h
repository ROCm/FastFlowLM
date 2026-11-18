// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __AIE_ARRAY_LAYOUT_H__
#define __AIE_ARRAY_LAYOUT_H__

#include "typedef.hpp"
#include "model_spec.h"

// calculate the number of repeats of x;
// Each round, the MVM engine computes QKV_ROW_BLOCK_SIZE * MVM_CORES of rows for the output;
constexpr int MVM_CORES = 16;
constexpr int QKV_REPEATS = (DQ + DK + DV) / MVM_CORES / Q4NX_ROW_BLOCK_SIZE;
constexpr int Q_REPEATS = DQ / MVM_CORES / Q4NX_ROW_BLOCK_SIZE;
constexpr int SWA_QKV_REPEATS = (SWA_DQ + SWA_DK + SWA_DV) / MVM_CORES / Q4NX_ROW_BLOCK_SIZE;
constexpr int SWA_Q_REPEATS = SWA_DQ / MVM_CORES / Q4NX_ROW_BLOCK_SIZE;
constexpr int O_DOWN_REPEATS = MODEL_DIM / MVM_CORES / Q4NX_ROW_BLOCK_SIZE;
constexpr int UP_GATE_REPEATS = (2 * INTERMEDIATE_SIZE) / MVM_CORES / Q4NX_ROW_BLOCK_SIZE;

constexpr int M_PER_ROUND = Q4NX_ROW_BLOCK_SIZE * MVM_CORES;


constexpr int pkt_id_rms_to_proj = 0;
constexpr int pkt_id_rms_to_it   = 1;


constexpr bfloat16 exp_scale = 1.44269504089;

#endif // __AIE_ARRAY_LAYOUT_H__