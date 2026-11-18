// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file verify/npu/q4nx.hpp
/// \brief q4nx arithmetic shared by every NPU layer here: the two block-size
///        formulae and the row-block interleave.
///
/// Duplicated from the IRON tree's quant_size.hpp and
/// decoding_layer::reorder_cpy rather than included, since the DLL does not
/// otherwise reach under FLM_Xclbin. A divergence dequantises garbage rather
/// than failing to compile -- hence the static assertions below.
#ifndef __QWEN3_8MTP_NPU_Q4NX_HPP__
#define __QWEN3_8MTP_NPU_Q4NX_HPP__

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace q38npu {

/// On-device byte footprint of a q4_k tensor, per kernels/model_spec.h.
/// 4.625 bits/weight, so it is not a whole number of 512 B slices -- anything
/// that walks it must step by whole 32x256 blocks.
constexpr size_t q4k_bytes(size_t elems) {
    return elems / 2            // 4-bit quants
         + elems / 32 * 2       // uint8 scale + uint8 min per group
         + elems / 256 * 4;     // bf16 S + bf16 M per super-block
}

/// On-device byte footprint of a q8_0 tensor. Unused by the attention layer;
/// the delta-net layer's alpha/beta projections are q8_0.
constexpr size_t q80_bytes(size_t elems) {
    return elems                // 8-bit quants
         + elems / 32 * 2;      // bf16 scale per group
}

static_assert(q4k_bytes(32 * 256) == 4736, "a q4_k 32x256 block is 4736 B");
static_assert(q80_bytes(32 * 256) == 8704, "a q8_0 32x256 block is 8704 B");

/// The engine's weight block: 32 output rows by 256 reduction columns.
constexpr size_t W_BLOCK_BYTES = q4k_bytes(32 * 256);

// ==========================================================================
/// Interleave whole 32x256 blocks between `vertical_blocks` row-blocks.
///
/// The projection memtile fills w_buffer[0 : 2*W_BLOCK_SIZE] contiguously and
/// hands half to each of its two row-blocks, so DDR must alternate whole blocks
/// between them. Byte-identical to decoding_layer::reorder_cpy; `src` is a raw
/// pointer plus a length here rather than a buffer<u8>, so it can read straight
/// out of the mmap'd .q4nx instead of a heap copy.
///
/// **`col` is the matrix's K, not always D.** It is how many 256-wide blocks
/// make one row of the source. o_proj reduces over the gated row (K = DQ) and
/// down_proj over the MLP hidden (K = intermediate_size); passing D for either
/// strides the wrong way through a matrix of exactly the same byte count, so
/// nothing asserts -- the cores just dequantise the wrong rows.
inline void reorder_cpy(uint8_t* dst, const uint8_t* src, size_t src_bytes,
                        int col, int vertical_blocks = 2) {
    const size_t a_block_size   = W_BLOCK_BYTES;
    const size_t blocks_per_row = static_cast<size_t>(col) / 256;
    const size_t rows = src_bytes / a_block_size / blocks_per_row;

    uint8_t* dst_ptr = dst;
    std::vector<const uint8_t*> src_ptr(vertical_blocks);
    for (int i = 0; i < vertical_blocks; i++)
        src_ptr[i] = src + static_cast<size_t>(i) * a_block_size * blocks_per_row;

    const uint8_t* src_end = src + src_bytes;
    for (size_t r = 0; r < rows; r += vertical_blocks) {
        for (size_t c = 0; c < blocks_per_row; c++) {
            for (int i = 0; i < vertical_blocks; i++) {
                std::memcpy(dst_ptr, src_ptr[i], a_block_size);
                dst_ptr += a_block_size;
                src_ptr[i] += a_block_size;
            }
        }
        for (int i = 0; i < vertical_blocks; i++) {
            src_ptr[i] += static_cast<size_t>(vertical_blocks - 1)
                        * a_block_size * blocks_per_row;
            if (src_ptr[i] + a_block_size * blocks_per_row > src_end)
                src_ptr[i] = src;   // useless padding
        }
    }
}

} // namespace q38npu

#endif  // __QWEN3_8MTP_NPU_Q4NX_HPP__
