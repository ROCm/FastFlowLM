// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

///@file typedef.hpp
///@brief Define the typedefs for the kernels
#ifndef __COMMON_HPP__
#define __COMMON_HPP__

#include <type_traits>
#include <aie_api/aie.hpp>

typedef bfloat16 bf16;

// A q4nx q4_k block covers 32 output rows x 256 input columns in 4736 B (see q4_core.h).
constexpr int   QXNX_ROW_BLOCK_SIZE = 32;
constexpr int   QXNX_COL_BLOCK_SIZE = 256;
constexpr int   GGML_GROUP_SIZE     = 32;

// rows a single _Q4_K_bf16 call dequantises: the whole block, since the two cores
// split it by K rather than by rows
constexpr int PARALLEL_ROWS = 32;

constexpr bfloat16 exp_scale = 1.44269504089;
// One core column owns NPU_N_TILE output rows == NPU_N_TILE / QXNX_ROW_BLOCK_SIZE = 4
// q4_k row bands, and walks Din in NPU_K_TILE = 2 * QXNX_COL_BLOCK_SIZE steps.  The
// weight-block visit order is therefore j (k half) outer, p (row band) inner; see
// gen_dequant_mm's send_w, which gathers exactly that order out of reorder_cpy(V = 2).
constexpr int NPU_M_TILE = 32;
constexpr int NPU_K_TILE = 512;
constexpr int NPU_N_TILE = 128;
constexpr int m_slice = 16;
constexpr int k_slice = QXNX_COL_BLOCK_SIZE;
constexpr int n_slice = 32;
constexpr int y_slice = 64;
constexpr int y_num = NPU_M_TILE * NPU_N_TILE / y_slice;

#endif 