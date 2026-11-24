// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

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

// Rows per dispatch of the verify design (1 = the decode design). Set by -DL_TOKENS from the Makefile.
#ifndef L_TOKENS
#define L_TOKENS 1
#endif
static_assert(L_TOKENS >= 1 && L_TOKENS <= 8, "L_TOKENS must be 1..8");
// every projection phase must be a whole number of 512-row rounds (16 cores x 32 rows)
static_assert((DQ + DK + DV) % M_PER_ROUND == 0, "qkv rows must be a multiple of the round size");
static_assert(DQ % M_PER_ROUND == 0, "q rows must be a multiple of the round size");
static_assert((SWA_DQ + SWA_DK + SWA_DV) % M_PER_ROUND == 0, "swa qkv rows must be a multiple of the round size");
static_assert(SWA_DQ % M_PER_ROUND == 0, "swa q rows must be a multiple of the round size");
static_assert(MODEL_DIM % M_PER_ROUND == 0, "o/down rows must be a multiple of the round size");
static_assert((2 * INTERMEDIATE_SIZE) % M_PER_ROUND == 0, "up/gate rows must be a multiple of the round size");
// proj_main keeps its ping/pong toggles across layers: the number of row blocks per layer must be even (decode_design_map.md section 3)
static_assert((QKV_REPEATS + 2 * O_DOWN_REPEATS + UP_GATE_REPEATS) % 2 == 0, "round parity, global layer");
static_assert((Q_REPEATS + 2 * O_DOWN_REPEATS + UP_GATE_REPEATS) % 2 == 0, "round parity, global skip layer");
static_assert((SWA_QKV_REPEATS + 2 * O_DOWN_REPEATS + UP_GATE_REPEATS) % 2 == 0, "round parity, swa layer");
static_assert((SWA_Q_REPEATS + 2 * O_DOWN_REPEATS + UP_GATE_REPEATS) % 2 == 0, "round parity, swa skip layer");


#if L_TOKENS > 1
// x replay (iron/proj_tile.py proj_x_replay_mt) sends every projection phase a fixed number of times; the layer variant that needs fewer
// qkv rounds drains the surplus first (kernels/proj_main.cc). The fixed count is the global layer's.
static_assert(SWA_QKV_REPEATS <= QKV_REPEATS && Q_REPEATS <= QKV_REPEATS && SWA_Q_REPEATS <= QKV_REPEATS, "qkv replays must cover every variant");
// x slices (256 columns) per phase: proj_main keeps its x ping/pong toggle across phases, so every phase must be a whole even number of slices
static_assert((QKV_REPEATS * (MODEL_DIM / 256)) % 2 == 0, "qkv slices");
static_assert((O_DOWN_REPEATS * (DQ / 256)) % 2 == 0, "o (global) slices");
static_assert((O_DOWN_REPEATS * (SWA_DQ / 256)) % 2 == 0, "o (swa) slices");
static_assert((UP_GATE_REPEATS * (MODEL_DIM / 256)) % 2 == 0, "up|gate slices");
static_assert((O_DOWN_REPEATS * (INTERMEDIATE_SIZE / 256)) % 2 == 0, "down slices");
#endif

constexpr int pkt_id_rms_to_proj = 0;
constexpr int pkt_id_rms_to_it   = 1;


constexpr bfloat16 exp_scale = 1.44269504089;

#endif // __AIE_ARRAY_LAYOUT_H__