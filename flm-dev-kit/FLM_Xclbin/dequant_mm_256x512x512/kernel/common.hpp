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

// A packed q4_k column holds the block's whole row span, so the quants are
// column major with PARALLEL_ROWS nibbles per column.
constexpr int PARALLEL_ROWS = QXNX_ROW_BLOCK_SIZE;

constexpr bfloat16 exp_scale = 1.44269504089;
constexpr int NPU_M_TILE = 64;
constexpr int NPU_K_TILE = 512;
constexpr int NPU_N_TILE = 128;
constexpr int m_slice = 16;
constexpr int k_slice = QXNX_COL_BLOCK_SIZE;
constexpr int n_slice = QXNX_ROW_BLOCK_SIZE;
constexpr int y_slice = 64;
constexpr int y_num = NPU_M_TILE * NPU_N_TILE / y_slice;

#endif 