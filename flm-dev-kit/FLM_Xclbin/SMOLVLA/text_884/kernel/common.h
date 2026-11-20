// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#ifndef COMMON_H
#define COMMON_H

#include <aie_api/aie.hpp>
typedef bfloat16 bf16;

// .bss is not zeroed when a Peano-built core is loaded. AIE_PERSIST_ZERO puts a
// zero-initialized static that must keep its value across calls in .data.
// AIE_TRANSIENT drops `static` where the value does not need to persist, so no
// init guard in .bss is involved. Both do nothing under chess.
#if defined(__chess__)
#define AIE_PERSIST_ZERO
#define AIE_TRANSIENT static
#else
#define AIE_PERSIST_ZERO __attribute__((section(".data")))
#define AIE_TRANSIENT
#endif

constexpr int lq = 64;
constexpr int lk = 64;
constexpr int dh = 64;

const float scale = 0.125f; // 1 / sqrt(64)

constexpr bf16 exp_scale = (bf16)1.4426950408889634f;

constexpr int NPU_M_TILE = 64;
constexpr int NPU_K_TILE = 320;
constexpr int NPU_N_TILE = 80;
constexpr int m_slice = 16;
constexpr int k_slice = 64;
constexpr int n_slice = 80;
constexpr int y_slice = 32;
constexpr int y_num = NPU_M_TILE * NPU_N_TILE / y_slice;

#endif
