// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#pragma once

/*
RMS O BIAS BUFFER ARRANGEMENT:
    1. q_proj_weights: Q_WEIGHT_BYTE
    2. k_proj_weights: K_WEIGHT_BYTE
    3. v_proj_weights: V_WEIGHT_BYTE
    4. o_proj_weights: O_WEIGHT_BYTE
    5. q_proj_bias:    GPTOSS_DQ*sizeof(bf16)
    6. k_proj_bias:    GPTOSS_DK*sizeof(bf16)
    7. v_proj_bias:    GPTOSS_DV*sizeof(bf16)
    8. sinks_weights:  NUM_ATTN_HEADS * sizeof(bf16)
    9. router_bias:    EXPERT_BIAS_SIZE_IN_BYTE
    10. router_weight: EXPERT_WEIGHT_SIZE_IN_BYTE
    11. rms_o_proj_bias: 3 * GPTOSS_HIDDEN_SIZE * sizeof(bf16)
*/

constexpr float    GPTOSS_RMS_NORM_EP       = 1e-5f;
constexpr uint32_t GPTOSS_NUM_EXPERT        = 32;
constexpr uint32_t GPTOSS_HIDDEN_SIZE       = 2880;
constexpr uint32_t GPTOSS_INTERMEDIATE_SIZE = 2880;
constexpr float    GPTOSS_ROPE_SCALING      = 1.3465735902799727;
constexpr uint32_t GPTOSS_HEAD_DIM          = 64;

constexpr  uint32_t GPTOSS_DQ = 4096;
constexpr  uint32_t GPTOSS_DK = 512;
constexpr  uint32_t GPTOSS_DV = 512;

constexpr int NUM_ATTN_HEADS = 64;
constexpr int NUM_KV_HEADS = 8;
constexpr int ATTN_GROUPS = NUM_ATTN_HEADS / NUM_KV_HEADS;

constexpr uint32_t  D = GPTOSS_HIDDEN_SIZE;
constexpr uint32_t  DH = GPTOSS_HEAD_DIM;
constexpr uint32_t  MVM_cores= 16;

// MOSTLY for NPU AIE
constexpr uint32_t  Q4NX_ROW = 32;
constexpr uint32_t  Q4NX_COL =128;
constexpr uint32_t  Q4NX_ROWL_TILE = 16;
constexpr uint32_t  Q4NX_BLOCK_SIZE =Q4NX_ROW*Q4NX_COL;

constexpr uint32_t Q4NX_BLOCK_BYTE_SIZE = (Q4NX_ROW * (Q4NX_COL/32 ) )*4 + (Q4NX_BLOCK_SIZE/2); // div2 because each value is 4bit

constexpr uint32_t D_PADDED_Q4NX_COL = ((((D) + (Q4NX_COL) - 1) / (Q4NX_COL)) * (Q4NX_COL));
constexpr uint32_t D_PADDED_FOUR_Q4NX_ROW = (((( D) + (4*Q4NX_ROW) - 1) / (4*Q4NX_ROW)) * (4*Q4NX_ROW));
constexpr uint32_t GPTOSS_HIDDEN_SIZE_PADDED = D_PADDED_Q4NX_COL;

constexpr uint32_t D_ATTEN_O_PADDED_MVM_SIZE = ((((D) + (MVM_cores * Q4NX_ROW) - 1) / (MVM_cores * Q4NX_ROW)) * (MVM_cores * Q4NX_ROW));

constexpr uint32_t INTERMEDIATE_PADDED_FOUR_Q4NX_ROW= (((( GPTOSS_INTERMEDIATE_SIZE) + (4*Q4NX_ROW) - 1) / (4*Q4NX_ROW)) * (4*Q4NX_ROW));
constexpr uint32_t INTERMEDIATE_PADDED_Q4NX_COL = ((((GPTOSS_INTERMEDIATE_SIZE) + (Q4NX_COL) - 1) / (Q4NX_COL)) * (Q4NX_COL));

static_assert(GPTOSS_DQ % Q4NX_ROW == 0);
static_assert(GPTOSS_DK % Q4NX_ROW == 0);
static_assert(GPTOSS_DV% Q4NX_ROW == 0);

constexpr uint32_t X_QKV_REPEAT =  (GPTOSS_DQ + GPTOSS_DK + GPTOSS_DV  )/MVM_cores/Q4NX_ROW;
constexpr uint32_t X_UP_GATE_REPEAT =  (2*INTERMEDIATE_PADDED_FOUR_Q4NX_ROW) / 4/Q4NX_ROW;

//NOTE: assume DQ, DK, DV is divisible by FOUR_Q4NX_ROW
static_assert(GPTOSS_DQ % (MVM_cores*Q4NX_ROW) == 0);
static_assert(GPTOSS_DK % (MVM_cores*Q4NX_ROW) == 0);
static_assert(GPTOSS_DV % (MVM_cores*Q4NX_ROW) == 0);
static_assert(D_ATTEN_O_PADDED_MVM_SIZE  % (MVM_cores*Q4NX_ROW) == 0);
constexpr uint32_t  Q_WEIGHT_BYTE =  (GPTOSS_DQ/Q4NX_ROW) * (D_PADDED_Q4NX_COL/Q4NX_COL) * Q4NX_BLOCK_BYTE_SIZE;
constexpr uint32_t  K_WEIGHT_BYTE  =(GPTOSS_DK/Q4NX_ROW) * (D_PADDED_Q4NX_COL/Q4NX_COL) * Q4NX_BLOCK_BYTE_SIZE;
constexpr uint32_t  V_WEIGHT_BYTE  = (GPTOSS_DV/Q4NX_ROW) * (D_PADDED_Q4NX_COL/Q4NX_COL) * Q4NX_BLOCK_BYTE_SIZE;
constexpr uint32_t  O_WEIGHT_BYTE =  (D_ATTEN_O_PADDED_MVM_SIZE/Q4NX_ROW)* (GPTOSS_DQ/Q4NX_COL) * Q4NX_BLOCK_BYTE_SIZE;

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

constexpr uint32_t GATE_UP_WEIGHT_BYTE_PER_EXPERT = 2* (INTERMEDIATE_PADDED_FOUR_Q4NX_ROW/Q4NX_ROW) * (D_PADDED_Q4NX_COL/Q4NX_COL)* Q4NX_BLOCK_BYTE_SIZE;
constexpr uint32_t  DOWN_WEIGHT_BYTE_PER_EXPERT = (INTERMEDIATE_PADDED_FOUR_Q4NX_ROW/Q4NX_ROW) * (D_PADDED_Q4NX_COL/Q4NX_COL)* Q4NX_BLOCK_BYTE_SIZE;

constexpr uint32_t  GATE_UP_DOWN_WEIGHT_BYTE = NUM_OF_EXPERT*(GATE_UP_WEIGHT_BYTE_PER_EXPERT+DOWN_WEIGHT_BYTE_PER_EXPERT);
constexpr uint32_t  EXPERT_BIAS_SIZE_IN_BYTE = 2 * (EXPERT_BIAS_BLOCK * EXPERT_BLOCK_SIZE);
constexpr uint32_t  EXPERT_WEIGHT_SIZE_IN_BYTE = 2*(NUM_OF_EXPERT*GPTOSS_HIDDEN_SIZE);
constexpr size_t EXPERT_WEIGHT_OFFSET_IN_BYTE = EXPERT_WEIGHT_SIZE_IN_BYTE + EXPERT_BIAS_SIZE_IN_BYTE;

constexpr uint32_t GELU_SLICE = 2* Q4NX_ROW *  4*4; // note: weight specific
constexpr float MLP_ALPHA = 1.702;
constexpr float MLP_LIMIT = 7.0;

constexpr size_t PROJ_BIAS_OFFSET_IN_BYTE = 0;

// configuration for control-packet
constexpr uint32_t SHIM_X_gateup_BD_ID = 12;
constexpr uint32_t SHIM_X2_gateup_BD_ID = 13;
constexpr uint32_t SHIM_X_down_BD_ID = 14;
constexpr uint32_t SHIM_X2_down_BD_ID = 15;

constexpr uint32_t SHIM_MM2S_PORT = 0;
constexpr uint32_t SHIM_2_CONTROL_PKT = 28;
constexpr uint32_t SHIM_3_CONTROL_PKT = 27;
constexpr uint32_t SHIM_4_CONTROL_PKT = 29;
constexpr uint32_t SHIM_5_CONTROL_PKT = 31;

constexpr uint32_t SHIM_5_CONTROL_RES_PKT = 30;

// some simplified constants
constexpr uint32_t DQ = GPTOSS_DQ;
constexpr uint32_t DK = GPTOSS_DK;
constexpr uint32_t DV = GPTOSS_DV;
constexpr uint32_t D_PADDED_128 = D_PADDED_Q4NX_COL;
constexpr uint32_t D_PADDED_256 = D_PADDED_FOUR_Q4NX_ROW;
// constexpr uint32_t D_PADDED_512 = (D + 512 - 1) / 512 * 512;

constexpr uint32_t rms_o_bias_offset_in_byte =
PROJ_BIAS_OFFSET_IN_BYTE + EXPERT_WEIGHT_SIZE_IN_BYTE + EXPERT_BIAS_SIZE_IN_BYTE +
(GPTOSS_DQ + GPTOSS_DK + GPTOSS_DV + NUM_ATTN_HEADS) * 2;

constexpr uint32_t qkv_weight_offset_in_byte = rms_o_bias_offset_in_byte + 3 * GPTOSS_HIDDEN_SIZE * 2;

constexpr int SLIDING_WINDOW = 128;

constexpr uint32_t VOCAB_SIZE = 201088;
constexpr uint32_t MVM_DECODE_CORES = 16;
constexpr uint32_t VOCAB_SIZE_PADDED = (((VOCAB_SIZE) + (D_ATTEN_O_PADDED_MVM_SIZE) - 1) / (D_ATTEN_O_PADDED_MVM_SIZE)) * (D_ATTEN_O_PADDED_MVM_SIZE);
constexpr size_t D_PADDED_DIV_Q4NX_COL = D_PADDED_Q4NX_COL/Q4NX_COL;

static_assert(VOCAB_SIZE_PADDED % Q4NX_ROW  == 0);
static_assert(VOCAB_SIZE_PADDED % (Q4NX_ROW*MVM_DECODE_CORES)  == 0);
constexpr size_t VOCAB_PADDED_DIV_Q4NX_ROW = VOCAB_SIZE_PADDED/Q4NX_ROW;

constexpr uint32_t LM_HEAD_LM_HEAD_WEIGHT_BUFFER_SIZE = VOCAB_PADDED_DIV_Q4NX_ROW * D_PADDED_DIV_Q4NX_COL *  Q4NX_BLOCK_BYTE_SIZE;
#define PREFILL_THREADS_BUDGET 2
#define CPU_MAX_LENGTH 0
#define NPU_MIN_LENGTH 0

// Upper bound (in routed tokens, before MM padding) on the second ping-pong
// expert buffer set ("slot 1") used during the prefill expert phase. slot 0 is
// always sized to the largest routed expert of the layer and can hold any
// expert; slot 1 is capped here so that two similarly large ("co-dominant")
// experts cannot force two max-sized expert buffer sets to be resident on the
// device at the same time and exhaust device memory. Experts that are larger
// than this cap are not pipelined through slot 1 -- they are run serially on
// the slot-0 buffers instead. This bounds the peak expert-buffer footprint to
// one max-sized set plus one capped set regardless of how skewed routing is.
//
// Tuning knob: larger values pipeline more of the large experts (faster) but
// raise the peak expert-buffer footprint; smaller values bound memory harder
// but serialize more of the largest experts. The default is chosen to comfortably
// hold the second-largest expert under typical skewed routing while keeping the
// extra footprint small, and was validated against on-device prefill benchmarks.
#define GPTOSS_PINGPONG_SLOT1_MAX_LEN 512
