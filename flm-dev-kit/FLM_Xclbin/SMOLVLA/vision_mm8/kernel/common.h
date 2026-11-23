// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

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

// attention
constexpr int L = 1024;
constexpr int lk_mt = 512;
constexpr int chunk_size = lk_mt;
constexpr int chunks = L / chunk_size;
constexpr int lq = 64;
constexpr int lk = 16;
constexpr int dh = 64;

constexpr bf16 exp_scale = (bf16)1.4426950408889634f;
constexpr float scale = 0.125f;

// The 1/sqrt(dh) scale is folded into Q at staging time rather than applied to
// the scores (see scale_copy_aie). That is only bit-exact while scale is a
// power of two, where it just shifts the exponent and leaves the bf16 mantissa
// untouched; for any other value the fold changes results. Keep them in sync.
static_assert(scale * scale * dh == 1.0f, "scale must be 1/sqrt(dh)");
static_assert(scale == 0.125f, "Q-side scale fold requires a power-of-two scale");

// mm
constexpr int NPU_M_TILE = 128;
constexpr int NPU_K_TILE = 256;
constexpr int NPU_N_TILE = 48;
constexpr int m_slice = 16;
constexpr int k_slice = 64;
constexpr int n_slice = 48;
constexpr int y_slice = 64;
constexpr int y_num = NPU_M_TILE * NPU_N_TILE / y_slice;

#endif
