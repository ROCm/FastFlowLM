// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

///@file typedef.hpp
///@brief Define the typedefs for the kernels
#ifndef __TYPEDEF_HPP__
#define __TYPEDEF_HPP__

#include <type_traits>
#include <aie_api/aie.hpp>

typedef bfloat16 bf16;

constexpr bfloat16 exp_scale = 1.44269504089;

// The array is 4 rows x 8 cols of compute tiles. COLUMNS split M, ROWS split N, so an
// array tile is (8 * NPU_M_TILE) x (4 * NPU_N_TILE).
//
//   C  64 x 384 x 96 per core -> 512 x 384 per array.
//      Covers every vision-tower matmul whose N is a multiple of 384 (see
//      iron/mm_config.py).
#define VISION_MM_C 3

#ifndef MODEL_TYPE
#define MODEL_TYPE VISION_MM_C
#endif

constexpr int NPU_M_TILE = 64;
constexpr int NPU_K_TILE = 384;
constexpr int NPU_N_TILE = 96;

constexpr int m_slice = 16;
constexpr int k_slice = 64;
constexpr int n_slice = NPU_N_TILE;
constexpr int y_slice = 64;
constexpr int y_num = NPU_M_TILE * NPU_N_TILE / y_slice;

// The mmul is 8x8x8, and the y DMA moves y_slice elements at a time.
static_assert(NPU_M_TILE % 8 == 0, "M tile must be a multiple of the mmul row block");
static_assert(NPU_N_TILE % 8 == 0, "N tile must be a multiple of the mmul col block");
static_assert(NPU_K_TILE % k_slice == 0, "K tile must divide into k_slice chunks");
static_assert(k_slice % 8 == 0, "k_slice must be a multiple of the mmul depth");
static_assert((NPU_M_TILE * NPU_N_TILE) % y_slice == 0, "y tile must divide into y_slice");

#endif
