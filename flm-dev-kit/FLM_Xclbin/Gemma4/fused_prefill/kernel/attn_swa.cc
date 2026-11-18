// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

// Sliding-window attention core, column 7 of the fused Gemma4 prefill array.
// Gemma4-E2B's sliding layers have head_dim 256 and a 512-token window.
//
// Half the head width of the full-attention column, so the standalone lk_mt of
// 128 still fits one memtile (256 KB of k/v + 64 KB q + 64 KB o).
#define ATTN_LQ 16
#define ATTN_LK 16
#define ATTN_DH 256
#define ATTN_LK_MT 128
#define ATTN_ZERO_R 16
#define ATTN_HAS_REORDER_S

#include <aie_api/aie.hpp>
#include "common.h"
#include <math.h>
#include "attn_kernel_dh256.h"

#define ATTN_FUNC_NAME attn_swa
#include "attn_body.inc"
