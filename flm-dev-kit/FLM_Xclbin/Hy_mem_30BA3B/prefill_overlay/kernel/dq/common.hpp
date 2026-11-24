// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

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
// Halved from dequant_mm_512x512x512 so every Hy-Memory matrix fits: K (Din)
// now only has to be a multiple of one Q4_K block (256) -- the 768 / 6912
// "down" projections -- and a round is 256 tokens (4 core rows x 64), half the
// padding on the routed experts. The loops below derive their trip counts
// from these (k_div_k_slice = 1 now).
constexpr int NPU_M_TILE = 64;
constexpr int NPU_K_TILE = 256;
constexpr int NPU_N_TILE = 64;
constexpr int m_slice = 16;
constexpr int k_slice = QXNX_COL_BLOCK_SIZE;
constexpr int n_slice = QXNX_ROW_BLOCK_SIZE;
constexpr int y_slice = 64;
constexpr int y_num = NPU_M_TILE * NPU_N_TILE / y_slice;

#endif 