// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

///@file typedef.hpp
///@brief Define the typedefs for the kernels
#ifndef __COMMON_HPP__
#define __COMMON_HPP__

#include <type_traits>
#include <aie_api/aie.hpp>

typedef bfloat16 bf16;

constexpr int   QXNX_ROW_BLOCK_SIZE = 16;
constexpr int   QXNX_COL_BLOCK_SIZE = 256;
constexpr int   GGML_GROUP_SIZE     = 32;

constexpr int PARALLEL_ROWS = 16;

constexpr bfloat16 exp_scale = 1.44269504089;
constexpr int NPU_M_TILE = 128;
constexpr int NPU_K_TILE = 256;
constexpr int NPU_N_TILE = 64;
constexpr int m_slice = 16;
constexpr int k_slice = QXNX_COL_BLOCK_SIZE;
constexpr int n_slice = 32;
constexpr int y_slice = 64;
constexpr int y_num = NPU_M_TILE * NPU_N_TILE / y_slice;

#endif 