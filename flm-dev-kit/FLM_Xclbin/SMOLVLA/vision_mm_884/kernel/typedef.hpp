// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

///@file typedef.hpp
///@brief Define the typedefs for the kernels
#ifndef __TYPEDEF_HPP__
#define __TYPEDEF_HPP__

#include <type_traits>
#include <aie_api/aie.hpp>

typedef bfloat16 bf16;

constexpr bfloat16 exp_scale = 1.44269504089;
constexpr int NPU_M_TILE = 128;
constexpr int NPU_K_TILE = 256;
constexpr int NPU_N_TILE = 64;
constexpr int m_slice = 16;
constexpr int k_slice = 64;
constexpr int n_slice = 64;
constexpr int y_slice = 32;
constexpr int y_num = NPU_M_TILE * NPU_N_TILE / y_slice;
#endif 