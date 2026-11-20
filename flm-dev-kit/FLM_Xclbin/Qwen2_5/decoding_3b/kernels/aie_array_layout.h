// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __AIE_ARRAY_LAYOUT_H__
#define __AIE_ARRAY_LAYOUT_H__

#include "typedef.hpp"
#include "model_spec.h"

// calculate the number of repeats of x;
// Each round, the MVM engine computes QKV_ROW_BLOCK_SIZE * MVM_CORES of rows for the output;
constexpr int MVM_CORES = 16;
constexpr int QKV_REPEATS = (DQ + DK + DV) / MVM_CORES / Q4NX_ROW_BLOCK_SIZE;
constexpr int UP_GATE_REPEATS = (2 * INTERMEDIATE_SIZE) / MVM_CORES / Q4NX_ROW_BLOCK_SIZE;

constexpr int M_PER_ROUND = 32 * MVM_CORES;

constexpr int pkt_id_to_rope     = 1;
constexpr int pkt_id_to_rms_norm = 4;
constexpr int pkt_id_to_glu      = 8;

constexpr int pkt_id_rms_to_proj = 0;
constexpr int pkt_id_rms_to_it   = 1;

#endif // __AIE_ARRAY_LAYOUT_H__