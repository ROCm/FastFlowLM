// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

///@file typedef.hpp
///@brief Define the typedefs for the kernels
#ifndef __COMMON_HPP__
#define __COMMON_HPP__

#include <type_traits>
#include <aie_api/aie.hpp>

typedef bfloat16 bf16;

constexpr int   QXNX_ROW_BLOCK_SIZE = 32;
constexpr int   QXNX_COL_BLOCK_SIZE = 256;
constexpr int   GGML_GROUP_SIZE     = 32;

// This kernel reads the weights of the mixed-precision decode layer in place (Q4_1 and Q8_0 blocks, see
// models/qwen3-asr-1.7b-mix.h of generic_decoding_layer): their quants are plain column major over the whole
// 32-row block, so the two cores of a row pair read the same bytes and unzip their own 16 rows.
constexpr int PARALLEL_ROWS = QXNX_ROW_BLOCK_SIZE;

constexpr bfloat16 exp_scale = 1.44269504089;
constexpr int NPU_M_TILE = 128;
constexpr int NPU_K_TILE = 512;
constexpr int NPU_N_TILE = 64;
constexpr int m_slice = 16;
constexpr int k_slice = QXNX_COL_BLOCK_SIZE;
constexpr int n_slice = QXNX_ROW_BLOCK_SIZE;
// the output tile is handed to the DMA in pieces of this many bf16 values (one lock handshake per piece)
constexpr int y_slice = 512;
constexpr int y_num = NPU_M_TILE * NPU_N_TILE / y_slice;
// weight chunk formats (K_div_k[1])
constexpr int FMT_Q4_1 = 0;
constexpr int FMT_Q8_0 = 1;
// rows of one weight slab: a pair of 32-row blocks, the w_buffer of a row pair
constexpr int SLAB_ROWS = 2 * QXNX_ROW_BLOCK_SIZE;

#endif 