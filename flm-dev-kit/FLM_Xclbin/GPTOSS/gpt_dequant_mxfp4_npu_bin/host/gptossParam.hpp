// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __GPTOSS_PARAM_HPP_
#define __GPTOSS_PARAM_HPP_
#include <cstdint>

constexpr uint32_t TOTAL_LAYER = 24;
constexpr float    GPTOSS_RMS_NORM_EP       = 1e-5f;
constexpr uint32_t GPTOSS_NUM_EXPERT        = 32;
constexpr uint32_t GPTOSS_HIDDEN_SIZE       = 2880;
constexpr uint32_t GPTOSS_INTERMEDIATE_SIZE = 2880;
constexpr double   GPTOSS_ROPE_SCALING      = 1.3465735902799727;
constexpr uint32_t GPTOSS_HEAD_DIM          = 64;

constexpr  uint32_t GPTOSS_DQ = 4096;
constexpr  uint32_t GPTOSS_DK = 512;
constexpr  uint32_t GPTOSS_DV = 512;


constexpr int NUM_ATTN_HEADS = 64;
constexpr int NUM_KV_HEADS = 8;
constexpr int ATTN_GROUPS = NUM_ATTN_HEADS / NUM_KV_HEADS;

constexpr uint32_t  D = GPTOSS_HIDDEN_SIZE;
constexpr uint32_t DH = GPTOSS_HEAD_DIM;

// MOSTLY for NPU AIE
constexpr uint32_t  Q4NX_ROW = 32;
constexpr uint32_t  Q4NX_COL =128;
constexpr uint32_t  Q4NX_ROWL_TILE = 16;
constexpr uint32_t  Q4NX_BLOCK_SIZE =Q4NX_ROW*Q4NX_COL;

constexpr uint32_t Q4NX_BLOCK_BYTE_SIZE = (Q4NX_ROW * (Q4NX_COL/32 ) )*4 + (Q4NX_BLOCK_SIZE/2); // div2 because each value is 4bit

constexpr uint32_t D_PADDED_Q4NX_COL = ((((D) + (Q4NX_COL) - 1) / (Q4NX_COL)) * (Q4NX_COL));
constexpr uint32_t D_PADDED_FOUR_Q4NX_ROW = (((( D) + (4*Q4NX_ROW) - 1) / (4*Q4NX_ROW)) * (4*Q4NX_ROW));
constexpr uint32_t GPTOSS_HIDDEN_SIZE_PADDED = D_PADDED_Q4NX_COL;

constexpr uint32_t INTERMEDIATE_PADDED_FOUR_Q4NX_ROW= (((( GPTOSS_INTERMEDIATE_SIZE) + (4*Q4NX_ROW) - 1) / (4*Q4NX_ROW)) * (4*Q4NX_ROW));
constexpr uint32_t INTERMEDIATE_PADDED_Q4NX_COL = ((((GPTOSS_INTERMEDIATE_SIZE) + (Q4NX_COL) - 1) / (Q4NX_COL)) * (Q4NX_COL));

constexpr uint32_t  MVM_cores= 16;
static_assert(GPTOSS_DQ % Q4NX_ROW == 0);
static_assert(GPTOSS_DK % Q4NX_ROW == 0);
static_assert(GPTOSS_DV% Q4NX_ROW == 0);

constexpr uint32_t X_QKV_REPEAT =  (GPTOSS_DQ + GPTOSS_DK + GPTOSS_DV  )/MVM_cores/Q4NX_ROW;
constexpr uint32_t X_UP_GATE_REPEAT =  (2*INTERMEDIATE_PADDED_FOUR_Q4NX_ROW) / 4/Q4NX_ROW;

constexpr uint32_t  Q_WEIGHT_BYTE =  (GPTOSS_DQ/Q4NX_ROW) * (D_PADDED_Q4NX_COL/Q4NX_COL) * Q4NX_BLOCK_BYTE_SIZE;
constexpr uint32_t  K_WEIGHT_BYTE  =(GPTOSS_DK/Q4NX_ROW) * (D_PADDED_Q4NX_COL/Q4NX_COL) * Q4NX_BLOCK_BYTE_SIZE;
constexpr uint32_t  V_WEIGHT_BYTE  = (GPTOSS_DV/Q4NX_ROW) * (D_PADDED_Q4NX_COL/Q4NX_COL) * Q4NX_BLOCK_BYTE_SIZE;
constexpr uint32_t  O_WEIGHT_BYTE =  (D_PADDED_Q4NX_COL/Q4NX_COL)* (GPTOSS_DQ/Q4NX_ROW) * Q4NX_BLOCK_BYTE_SIZE;


//MVM parameter
constexpr uint32_t MVM_INTERLEAVE_SIZE = 128; // means when rms/router send out X for MVM
// The data need to repeat itself every 128 count for (MVM_cores/4), aka Repeat the data every 128 times for each MVM column

constexpr uint32_t EXPERT_BLOCK_ROWS = 32;
constexpr uint32_t EXPERT_BLOCK_COLS = 64;
constexpr uint32_t EXPERT_BLOCK_TILE_ROWS = 16;
constexpr uint32_t NUM_OF_EXPERT = 32;
constexpr uint32_t TOP_K = 4;
constexpr uint32_t EXPERT_BLOCK_SIZE = EXPERT_BLOCK_ROWS *EXPERT_BLOCK_COLS;
constexpr uint32_t EXPERT_BIAS_BLOCK = (NUM_OF_EXPERT + EXPERT_BLOCK_ROWS -1) / EXPERT_BLOCK_ROWS; // how many MAT_BLOCK_SIZE to load for bias


constexpr uint32_t  GATE_UP_WEIGHT_BYTE_PER_EXPERT = 2* (INTERMEDIATE_PADDED_FOUR_Q4NX_ROW/Q4NX_ROW) * (D_PADDED_Q4NX_COL/Q4NX_COL)* Q4NX_BLOCK_BYTE_SIZE;
constexpr uint32_t  DOWN_WEIGHT_BYTE_PER_EXPERT = (INTERMEDIATE_PADDED_FOUR_Q4NX_ROW/Q4NX_ROW) * (D_PADDED_Q4NX_COL/Q4NX_COL)* Q4NX_BLOCK_BYTE_SIZE;

constexpr uint32_t  GATE_UP_DOWN_WEIGHT_BYTE = NUM_OF_EXPERT*(GATE_UP_WEIGHT_BYTE_PER_EXPERT+DOWN_WEIGHT_BYTE_PER_EXPERT);


constexpr uint32_t GELU_SLICE = 2* Q4NX_ROW *  4*4; // note: weight specific
constexpr float MLP_ALPHA = 1.702;
constexpr float MLP_LIMIT = 7.0;


// configuration for control-packet 
constexpr uint32_t SHIM_X_BD_ID = 14;
constexpr uint32_t SHIM_MM2S_PORT = 0;
constexpr uint32_t SHIM_2_CONTROL_PKT = 28;
constexpr uint32_t SHIM_3_CONTROL_PKT = 27;
constexpr uint32_t SHIM_4_CONTROL_PKT = 29;
constexpr uint32_t SHIM_5_CONTROL_PKT = 31;


constexpr uint32_t SHIM_2_CONTROL_RES_PKT = 24;
constexpr uint32_t SHIM_3_CONTROL_RES_PKT = 26;
constexpr uint32_t SHIM_4_CONTROL_RES_PKT = 28;
constexpr uint32_t SHIM_5_CONTROL_RES_PKT = 30;

#endif