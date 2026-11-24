// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

// Full (global-causal) attention core, column 0 of the fused Gemma4 prefill
// array.  Gemma4-E2B's full-attention layers have head_dim 512.
//
// lk_mt is 64 rather than the standalone design's 128: a fused column stages q,
// k, v and o in a single 512 KB memtile (2 * 2 * 64 * 512 * 2 B of k/v, plus
// 64 KB of q and 64 KB of o = 384 KB), where the standalone build spread the
// same traffic over four memtiles.
#define ATTN_LQ 8
#define ATTN_LK 8
#define ATTN_DH 512
#define ATTN_LK_MT 64
#define ATTN_ZERO_R 8

#include <aie_api/aie.hpp>
#include "common.h"
#include <math.h>
#include "attn_kernel_dh512.h"

#define ATTN_FUNC_NAME attn_full
#include "attn_body.inc"
