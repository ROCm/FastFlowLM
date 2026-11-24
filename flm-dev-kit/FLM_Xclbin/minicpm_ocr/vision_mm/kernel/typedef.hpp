// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

///@file typedef.hpp
///@brief Define the typedefs for the kernels
#ifndef __TYPEDEF_HPP__
#define __TYPEDEF_HPP__

#include <type_traits>
#include <aie_api/aie.hpp>

typedef bfloat16 bf16;

constexpr bfloat16 exp_scale = 1.44269504089;

// The array is 4 rows x 8 cols of compute tiles. Rows split M, cols split N, so an
// array tile is (4 * NPU_M_TILE) x (8 * NPU_N_TILE).
constexpr int total_rows = 4;
constexpr int total_cols = 8;

// Two designs cover the Qwen3.8-27B vision tower. Selected by -DMODEL_TYPE.
//
//   A  64 x 384 x 48  per core -> 256 x 384 per array.
//      Covers attn.qkv (N=3456), attn.proj (N=1152), mlp.fc1 (N=4304->4608),
//      mlp.fc2 (K=4304->4608), merger.fc1 (4608x4608). All N divide 384.
//
//   B  32 x 384 x 128 per core -> 128 x 1024 per array.
//      Covers merger.fc2 (4608 x 5120); 5120 = 5 * 1024.
//
// m is 32 rather than 64 in design B because the core has only 64 KB of local
// memory: at m=64,n=128 the fp32 accumulator alone is 32 KB and the double
// buffered w tile another 32 KB, which overflows before x and the stack. Keeping
// k_slice at 64 in both designs leaves the mmul inner loop trip count unchanged.
#define VISION_MM_A 1
#define VISION_MM_B 2

#ifndef MODEL_TYPE
#define MODEL_TYPE VISION_MM_A
#endif

#if MODEL_TYPE == VISION_MM_B
constexpr int NPU_M_TILE = 32;
constexpr int NPU_K_TILE = 384;
constexpr int NPU_N_TILE = 128;
#else
constexpr int NPU_M_TILE = 64;
constexpr int NPU_K_TILE = 384;
constexpr int NPU_N_TILE = 48;
#endif

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
