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
// Must match m / k / n in iron/dequant_mm_aie.py -- the host, the array's DMA
// descriptors and this core loop all size their tile from these three, so a
// mismatch deadlocks rather than producing wrong numbers (the core waits on x
// data the shim never sends). This build is the m = 64 sibling of
// dequant_mm_Q4_1_512x512x512, which keeps NPU_M_TILE = 128.
constexpr int NPU_M_TILE = 32;
constexpr int NPU_K_TILE = 256;
constexpr int NPU_N_TILE = 64;
constexpr int m_slice = 16;
constexpr int k_slice = QXNX_COL_BLOCK_SIZE;
constexpr int n_slice = QXNX_ROW_BLOCK_SIZE;
constexpr int y_slice = 64;
constexpr int y_num = NPU_M_TILE * NPU_N_TILE / y_slice;

// bf16 passthrough mode (see `_bf16_w_copy`).  A q4_1 packet is 5120 B, of which
// only BF16_USABLE_BYTES may be used for a bf16 tile -- 4096 divides the 16384 B
// w tile evenly, 5120 does not.  So one w tile arrives as BF16_PACKETS packets,
// each carrying BF16_PLANE_BYTES for each core of the row pair plus a 1024 B pad.
// The pad is dead DMA traffic; it is the price of not re-cutting every BD, and
// the bf16 gemms it serves already moved raw bf16 on the other overlay anyway.
constexpr int BF16_TILE_ELEMS   = QXNX_ROW_BLOCK_SIZE * QXNX_COL_BLOCK_SIZE;   // 8192
constexpr int BF16_TILE_BYTES   = BF16_TILE_ELEMS * (int)sizeof(bf16);        // 16384
constexpr int BF16_USABLE_BYTES = 4096;
constexpr int BF16_PACKETS      = BF16_TILE_BYTES / BF16_USABLE_BYTES;        // 4
constexpr int BF16_PLANE_BYTES  = BF16_USABLE_BYTES / 2;                      // 2048
static_assert(BF16_TILE_BYTES % BF16_USABLE_BYTES == 0);

#endif
