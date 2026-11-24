// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __QMM_H__
#define __QMM_H__
#include <stdint.h>
#include <aie_api/aie.hpp>
#include <aie_api/detail/lut.hpp>
#include <aie_api/detail/parallel_lookup.hpp>


constexpr int SUB_BLOCKS = 4;
constexpr int SUB_BLOCK_SIZE = 32;

constexpr int vec_factor = 8;
typedef bfloat16 bf16;




constexpr int QK_K = 128;
typedef uint16_t ggml_half;

#define BUG_SOLVED 0

constexpr int NPU_M_TILE = 16;
constexpr int NPU_K_TILE = 128;

#endif // __COMMON_H__
