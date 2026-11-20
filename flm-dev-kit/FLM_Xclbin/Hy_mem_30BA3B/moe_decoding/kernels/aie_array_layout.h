// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#ifndef __AIE_ARRAY_LAYOUT_H__
#define __AIE_ARRAY_LAYOUT_H__

#include "typedef.hpp"
#include "model_spec.h"

// calculate the number of repeats of x;
// Each round, the MVM engine computes QKV_ROW_BLOCK_SIZE * MVM_CORES of rows for the output;
constexpr int MVM_CORES = 16;
constexpr int MVM_CORES_Per_Column = 4;
constexpr int NUM_MVM_COLUMN =MVM_CORES /  MVM_CORES_Per_Column;

constexpr int QKV_REPEATS = (DQ + DK + DV) / MVM_CORES / QXNX_ROW_BLOCK_SIZE;
constexpr int O_DOWN_REPEATS = MODEL_DIM / MVM_CORES / QXNX_ROW_BLOCK_SIZE;
constexpr int UP_GATE_REPEATS = (2 * INTERMEDIATE_SIZE) / MVM_CORES / QXNX_ROW_BLOCK_SIZE;



//NOTE: For q4nx
// q4_k: 4 bits of quant per weight, a uint8 scale and a uint8 min per group of
// 32, and a bf16 S / bf16 M per super-block of 256, i.e. 37/64 byte = 4.625 bpw.
// With the shipping 32x256 block that is 4096 + 512 + 128 = 4736 B.
constexpr uint32_t Q4_K_block_byte = (QXNX_ROW_BLOCK_SIZE* QXNX_COL_BLOCK_SIZE* 37) /64;
constexpr uint32_t  Q4_K_GATE_UP_WEIGHT_BYTE_PER_EXPERT = 2* (INTERMEDIATE_SIZE/QXNX_ROW_BLOCK_SIZE) * (MODEL_DIM/QXNX_COL_BLOCK_SIZE)* Q4_K_block_byte;
constexpr uint32_t  Q4_K_DOWN_WEIGHT_BYTE_PER_EXPERT = (MODEL_DIM/QXNX_ROW_BLOCK_SIZE) * (INTERMEDIATE_SIZE/QXNX_COL_BLOCK_SIZE)* Q4_K_block_byte;

// one routed expert in DDR: [channel 0 stream | channel 1 stream], each stream
// its half of the gate/up blocks then its half of the down blocks (see the host)
constexpr uint32_t  Q4_K_EXPERT_BYTE = Q4_K_GATE_UP_WEIGHT_BYTE_PER_EXPERT + Q4_K_DOWN_WEIGHT_BYTE_PER_EXPERT;
constexpr uint32_t  Q4_K_EXPERT_BYTE_PER_CHANNEL = Q4_K_EXPERT_BYTE / 2;

constexpr int UP_GATE_EXPERT_REPEATS =(2 * INTERMEDIATE_SIZE )/ MVM_CORES_Per_Column   / QXNX_ROW_BLOCK_SIZE;
constexpr int DOWN_EXPERT_REPEATS = MODEL_DIM /MVM_CORES_Per_Column/QXNX_ROW_BLOCK_SIZE;

constexpr int PER_PROJ_COLUMN_OUTPUT_SLICE = MVM_CORES_Per_Column*QXNX_ROW_BLOCK_SIZE;
constexpr int PER_PROJ_OUTPUT_SLICE = NUM_MVM_COLUMN* PER_PROJ_COLUMN_OUTPUT_SLICE;

constexpr int NUM_PROJ_OUT_PER_GLU_SLICE = GLU_SLICE / PER_PROJ_OUTPUT_SLICE;

// one proj round of a column (its 4 cores x 32 rows) is one expert's [up | gate]
static_assert(2 * EXP_UP_GATE_OUTDIM_REORDER == QXNX_ROW_BLOCK_SIZE * MVM_CORES_Per_Column);
// one GLU slice is exactly one proj round
static_assert(GLU_SLICE == PER_PROJ_OUTPUT_SLICE);


constexpr int pkt_id_to_rope     = 1;
constexpr int pkt_id_to_aux      = 8;
constexpr int pkt_id_to_glu      = 15;

constexpr int pkt_id_rms_to_proj = 0;
constexpr int pkt_id_rms_to_it   = 1;
constexpr int pkt_id_rms_to_glu = 7;
constexpr int pkt_id_rms_from_glu = 11;

// packet ids into the proj engine's x input (same values as proj_x_pkt_id in iron/constants.py)
// GLU uses one id for everything it sends to the proj engine: shared expert h,
// routed up/gate x, and the 4 routed experts' h (round robin). Two ids on the
// same path made the router merge them into one masked rule that also caught
// GLU's other packets (id 11, to rms).
constexpr int pkt_id_glu_to_proj_x      = 9;




// configuration for control-packet 
// routed expert BDs on shims 0, 1, 6, 7: channel 0 uses the id below, channel 1 the
// next one (round 0: 4, 5; round 1: 12, 13). Must match _gen_sequence on the host.
constexpr uint32_t SHIM_X_expert_BD_ID = 4;
constexpr uint32_t Round_2_SHIM_X_expert_BD_ID = 12;

constexpr uint32_t SHIM_0_CONTROL_PKT = 29;
constexpr uint32_t SHIM_1_CONTROL_PKT = 25;
constexpr uint32_t SHIM_6_CONTROL_PKT = 22;
constexpr uint32_t SHIM_7_CONTROL_PKT = 31;

constexpr uint32_t SHIM_6_CONTROL_RES_PKT = 30;

constexpr bfloat16 exp_scale = 1.44269504089;
// The MVM cores hold a whole weight block's rows at once: one column of a block
// is exactly PARALLEL_ROWS elements, so this is a layout constant tied to the
// block shape, not a tuning knob.
constexpr int PARALLEL_ROWS = QXNX_ROW_BLOCK_SIZE;

#endif // __AIE_ARRAY_LAYOUT_H__