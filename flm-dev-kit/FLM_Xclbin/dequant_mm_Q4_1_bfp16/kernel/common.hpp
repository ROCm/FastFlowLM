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

// Q4_1's quant plane is parallel-strided, not plain column major: the block's
// QXNX_ROW_BLOCK_SIZE rows are packed as QXNX_ROW_BLOCK_SIZE / PARALLEL_ROWS
// planes of PARALLEL_ROWS rows, and each plane is laid out column major on its
// own.  PARALLEL_ROWS is the converter's `parallel_size` (Q4NX_ROW_STRIDE = 16
// for Q4_0 / Q4_1); it must match, or the nibbles land on the wrong rows.
//
// The two cores of a row pair own one plane each, so a core's quants are a
// contiguous, 512 bit aligned region and one 512 bit load is exactly
// 128 / PARALLEL_ROWS = 8 whole columns, i.e. one k block of the mmul B tile.
// (Q4_K packs with PARALLEL_ROWS == QXNX_ROW_BLOCK_SIZE instead, which is why
// its kernel needs an extra unzip to separate the row pair's two halves.)
constexpr int PARALLEL_ROWS = 16;

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

#endif 