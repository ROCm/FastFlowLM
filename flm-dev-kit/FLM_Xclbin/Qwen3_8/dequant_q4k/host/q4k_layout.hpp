// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef Q4K_LAYOUT_HPP
#define Q4K_LAYOUT_HPP

#pragma once

#include <cassert>
#include <cstring>

#include "typedef.hpp"
#include "weight_desc.hpp"

/// \file q4k_layout.hpp
/// \brief On-disk layout of a q4nx Q4_K block, plus a scalar CPU decode of it.
///
/// This mirrors `_pack_q4k` in FLM_Q4NX_Converter/q4nx/model_converter.py and
/// `q4k_block_t` in FLM_Xclbin/Qwen3_8/gate_delta_net_verify/kernels/model_spec.h
/// (and its identical copy in ../gated_attention_verify),
/// at QXNX_ROW_BLOCK_SIZE = 32:
///
///     scales  8 groups x 32 rows, uint8      256 B
///     mins    8 groups x 32 rows, uint8      256 B
///     qs      256 columns x 32 rows, uint4  4096 B, column major
///     S       32 rows, bf16                   64 B
///     M       32 rows, bf16                   64 B, stored *negated*
///
///     w[r][c] = S[r] * scales[c / 32][r] * q[c][r] + M[r] * mins[c / 32][r]
///
/// Note the `+`: GGUF's Q4_K subtracts its min, this format folds the sign into M.
///
/// The block size is spelled out here rather than taken from
/// `get_quantization_byte_size`, whose q4 branch describes q4_1 (4608 B) and is
/// shared with every other app -- widening it for Q4_K would change their sizes.

constexpr int Q4K_GROUP_SIZE   = 32;
constexpr int Q4K_GROUPS       = QXNX_COL_BLOCK_SIZE / Q4K_GROUP_SIZE; // 8

constexpr size_t Q4K_SCALES_BYTES = (size_t)Q4K_GROUPS * QXNX_ROW_BLOCK_SIZE;
constexpr size_t Q4K_MINS_BYTES   = (size_t)Q4K_GROUPS * QXNX_ROW_BLOCK_SIZE;
constexpr size_t Q4K_QS_BYTES     = (size_t)QXNX_ROW_BLOCK_SIZE * QXNX_COL_BLOCK_SIZE / 2;
constexpr size_t Q4K_S_BYTES      = (size_t)QXNX_ROW_BLOCK_SIZE * sizeof(bf16);
constexpr size_t Q4K_M_BYTES      = (size_t)QXNX_ROW_BLOCK_SIZE * sizeof(bf16);

constexpr size_t Q4K_SCALES_OFFSET = 0;
constexpr size_t Q4K_MINS_OFFSET   = Q4K_SCALES_OFFSET + Q4K_SCALES_BYTES;  // 256
constexpr size_t Q4K_QS_OFFSET     = Q4K_MINS_OFFSET   + Q4K_MINS_BYTES;    // 512
constexpr size_t Q4K_S_OFFSET      = Q4K_QS_OFFSET     + Q4K_QS_BYTES;      // 4608
constexpr size_t Q4K_M_OFFSET      = Q4K_S_OFFSET      + Q4K_S_BYTES;       // 4672

constexpr size_t Q4K_BLOCK_BYTES = Q4K_M_OFFSET + Q4K_M_BYTES;              // 4736

static_assert(Q4K_BLOCK_BYTES == 4736, "q4k block stride changed");

/// \brief Read one bf16 out of an unaligned byte stream.
inline float q4k_bf16_at(const u8 *p, size_t index) {
    u16 raw;
    std::memcpy(&raw, p + index * sizeof(u16), sizeof(u16));
    return (float)*reinterpret_u16_as_bf16(&raw, 1);
}

/// \brief Scalar decode of one 32x256 block into a row-major destination.
///
/// This is the exact target for the NPU kernel: same arithmetic, no vector
/// rounding differences beyond the final bf16 store. Comparing the NPU against
/// this separates a kernel/DMA bug from Q4_K simply being a lossy format.
///
/// \param dst        first element of the block in the output matrix
/// \param dst_stride elements between consecutive output rows (i.e. Din)
/// \param blk        the 4736-byte block
inline void dequant_q4k_block_ref(bf16 *dst, size_t dst_stride, const u8 *blk) {
    const u8 *scales = blk + Q4K_SCALES_OFFSET;
    const u8 *mins   = blk + Q4K_MINS_OFFSET;
    const u8 *qs     = blk + Q4K_QS_OFFSET;

    for (int r = 0; r < QXNX_ROW_BLOCK_SIZE; r++) {
        const float S = q4k_bf16_at(blk + Q4K_S_OFFSET, r);
        const float M = q4k_bf16_at(blk + Q4K_M_OFFSET, r);
        for (int c = 0; c < QXNX_COL_BLOCK_SIZE; c++) {
            const int g = c / Q4K_GROUP_SIZE;
            // one column is QXNX_ROW_BLOCK_SIZE nibbles: byte b holds row 2b in
            // the low nibble and row 2b+1 in the high one
            const u8 byte = qs[(size_t)c * (QXNX_ROW_BLOCK_SIZE / 2) + r / 2];
            const int q = (r & 1) ? (byte >> 4) : (byte & 0xF);
            const float w = S * (float)scales[g * QXNX_ROW_BLOCK_SIZE + r] * (float)q
                          + M * (float)mins  [g * QXNX_ROW_BLOCK_SIZE + r];
            dst[(size_t)r * dst_stride + c] = (bf16)w;
        }
    }
}

/// \brief Scalar decode of a whole tensor, from q4nx file order.
///
/// The file stores blocks row-block major: block (rb, cb) at
/// (rb * blocks_per_row + cb) * Q4K_BLOCK_BYTES. This is the *unreordered*
/// order, i.e. before `reorder_cpy` interleaves row blocks for the 4 core rows.
inline void dequant_q4k_tensor_ref(buffer<bf16> &dst, const buffer<u8> &src,
                                   const int Dout, const int Din) {
    const size_t blocks_per_row = (size_t)Din / QXNX_COL_BLOCK_SIZE;
    const size_t row_blocks     = (size_t)Dout / QXNX_ROW_BLOCK_SIZE;
    assert(src.size() == row_blocks * blocks_per_row * Q4K_BLOCK_BYTES);
    assert(dst.size() >= (size_t)Dout * Din);

    for (size_t rb = 0; rb < row_blocks; rb++) {
        for (size_t cb = 0; cb < blocks_per_row; cb++) {
            const u8 *blk = src.data() + (rb * blocks_per_row + cb) * Q4K_BLOCK_BYTES;
            bf16 *out = dst.data()
                      + rb * QXNX_ROW_BLOCK_SIZE * (size_t)Din
                      + cb * QXNX_COL_BLOCK_SIZE;
            dequant_q4k_block_ref(out, (size_t)Din, blk);
        }
    }
}

#endif
