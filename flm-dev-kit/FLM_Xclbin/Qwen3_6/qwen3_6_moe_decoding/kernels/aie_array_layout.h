// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __AIE_ARRAY_LAYOUT_H__
#define __AIE_ARRAY_LAYOUT_H__

#include "typedef.hpp"
#include "model_spec.h"

// calculate the number of repeats of x;
// Each round, the MVM engine computes QKV_ROW_BLOCK_SIZE * MVM_CORES of rows for the output;
constexpr int MVM_CORES = 16;
constexpr int MVM_CORES_Per_Column = 4;
constexpr int NUM_MVM_COLUMN =MVM_CORES /  MVM_CORES_Per_Column;

constexpr int QKV_REPEATS = (DQ + DK + DV + DQ) / MVM_CORES / QXNX_ROW_BLOCK_SIZE;
constexpr int L_QKV_REPEATS = (Linear_DQ + Linear_DK + Linear_DV + Linear_DV) / MVM_CORES / QXNX_ROW_BLOCK_SIZE;
constexpr int O_DOWN_REPEATS = MODEL_DIM / MVM_CORES / QXNX_ROW_BLOCK_SIZE;
constexpr int UP_GATE_REPEATS = (2 * INTERMEDIATE_SIZE) / MVM_CORES / QXNX_ROW_BLOCK_SIZE;



//NOTE: For q4nx
// q4_k: 4 bits of quant per weight, a uint8 scale and a uint8 min per group of
// 32, and a bf16 S / bf16 M per super-block of 256, i.e. 37/64 byte = 4.625 bpw.
// With the shipping 32x256 block that is 4096 + 512 + 128 = 4736 B.
constexpr uint32_t Q4_K_block_byte = (QXNX_ROW_BLOCK_SIZE* QXNX_COL_BLOCK_SIZE* 37) /64;
constexpr uint32_t  Q4_K_GATE_UP_WEIGHT_BYTE_PER_EXPERT = 2* (INTERMEDIATE_SIZE/QXNX_ROW_BLOCK_SIZE) * (MODEL_DIM/QXNX_COL_BLOCK_SIZE)* Q4_K_block_byte;
constexpr uint32_t  Q4_K_DOWN_WEIGHT_BYTE_PER_EXPERT = (MODEL_DIM/QXNX_ROW_BLOCK_SIZE) * (INTERMEDIATE_SIZE/QXNX_COL_BLOCK_SIZE)* Q4_K_block_byte;

// constexpr uint32_t  GATE_UP_DOWN_WEIGHT_BYTE = NUM_OF_EXPERT*(GATE_UP_WEIGHT_BYTE_PER_EXPERT+DOWN_WEIGHT_BYTE_PER_EXPERT);





constexpr int UP_GATE_EXPERT_REPEATS =(2 * INTERMEDIATE_SIZE )/ MVM_CORES_Per_Column   / QXNX_ROW_BLOCK_SIZE;
constexpr int DOWN_EXPERT_REPEATS = MODEL_DIM /MVM_CORES_Per_Column/QXNX_ROW_BLOCK_SIZE;

constexpr int M_PER_ROUND = QXNX_ROW_BLOCK_SIZE * MVM_CORES;


constexpr int PER_PROJ_COLUMN_OUTPUT_SLICE = MVM_CORES_Per_Column*QXNX_ROW_BLOCK_SIZE;
constexpr int PER_PROJ_OUTPUT_SLICE = NUM_MVM_COLUMN* PER_PROJ_COLUMN_OUTPUT_SLICE;

constexpr int NUM_PROJ_OUT_PER_GLU_SLICE = GLU_SLICE / PER_PROJ_OUTPUT_SLICE;

static_assert(EXP_UP_GATE_OUTDIM_REORDER  == QXNX_ROW_BLOCK_SIZE *MVM_CORES_Per_Column);


constexpr int pkt_id_to_rope     = 1;
constexpr int pkt_id_to_conv     = 4;
constexpr int pkt_id_to_aux      = 8;
constexpr int pkt_id_to_glu      = 15;

constexpr int pkt_id_rms_to_proj = 0;
constexpr int pkt_id_rms_to_it   = 1;
constexpr int pkt_id_rms_to_glu = 7;
constexpr int pkt_id_rms_from_glu = 11;

constexpr int aux_to_proj_x_in_pkt_id_glu_to_aux_x_in  = 9;
constexpr int x_repeat_to_aux_x_proj_id = 18;




// configuration for control-packet 
constexpr uint32_t SHIM_X_gateup_BD_ID = 4;
constexpr uint32_t SHIM_X2_gateup_BD_ID = 5;
constexpr uint32_t SHIM_X_down_BD_ID = 6;
constexpr uint32_t SHIM_X2_down_BD_ID = 7;



 
constexpr uint32_t Round_2_SHIM_X_gateup_BD_ID = 12;
constexpr uint32_t Round_2_SHIM_X2_gateup_BD_ID = 13;
constexpr uint32_t Round_2_SHIM_X_down_BD_ID = 14;
constexpr uint32_t Round_2_SHIM_X2_down_BD_ID = 15;






// constexpr uint32_t SHIM_MM2S_PORT = 0;
constexpr uint32_t SHIM_0_CONTROL_PKT = 29;
constexpr uint32_t SHIM_1_CONTROL_PKT = 25;
constexpr uint32_t SHIM_6_CONTROL_PKT = 22;
constexpr uint32_t SHIM_7_CONTROL_PKT = 31;


// constexpr uint32_t SHIM_2_CONTROL_RES_PKT = 24;
// constexpr uint32_t SHIM_3_CONTROL_RES_PKT = 26;
// constexpr uint32_t SHIM_4_CONTROL_RES_PKT = 28;
constexpr uint32_t SHIM_6_CONTROL_RES_PKT = 30;

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