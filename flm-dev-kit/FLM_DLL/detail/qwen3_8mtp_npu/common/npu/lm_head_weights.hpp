// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file common/npu/lm_head_weights.hpp
/// \brief Stage `lm_head.weight` into the layout the lm_head bitstream reads.
///
/// Port of `reorder_cpy_q80()` from
/// FLM_Xclbin/Qwen3_8/lm_head_npu_bin/host/host.cpp, reading a
/// `cpu_ref::q4nx_tensor` instead of a flat `buffer<u8>`. The two source
/// layouts are the same thing: `q4nx_tensor::blocks` is `[nrb][ncb][block_bytes]`
/// and `Q80_BLOCK_BYTES` is 8704 = 32*256 int8 + 256 bf16, which is exactly
/// what upstream indexes with `(row * blocks_per_row + c) * a_block_size`. So
/// the permutation is unchanged and only the addressing of the source moved.
///
/// The permutation itself: a column's weights arrive on two shim MM2S
/// channels, one per pair of cores (see lm_head_sequence.hpp), so each group
/// of four row blocks has to be two contiguous halves in DDR rather than one
/// run with all four cores' blocks interleaved -- row blocks r+0/r+1 for the
/// stream that feeds cores 0 and 1, r+2/r+3 for the stream that feeds cores 2
/// and 3. Within a half the order is k-major, one block per core, because that
/// is exactly one memtile slot.
///
/// Getting this wrong does not fault: the cores dequantise whatever bytes
/// arrive. See `_move_weights' offset is in elements, not bytes` -- an
/// addressing error in a weight stage shows up as arithmetic nonsense, so the
/// milestone driver checks the destination offsets directly rather than
/// inferring them from a cosine.
#ifndef __QWEN3_8MTP_NPU_LM_HEAD_WEIGHTS_HPP__
#define __QWEN3_8MTP_NPU_LM_HEAD_WEIGHTS_HPP__

#include "common/cpu_ref_common.hpp"
#include "common/npu/lm_head_sequence.hpp"

#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <string>

namespace q38npu {

/// Row blocks that share a memtile slot group: four cores, one block each.
constexpr int LM_HEAD_VERTICAL_BLOCKS = 4;

/// Where block `(rb, cb)` of the source lands in the reordered BO.
///
/// Factored out of the copy so the driver can recompute it independently --
/// a staging routine checked against itself checks nothing.
///
/// \param rb  source row block, 0 .. padded_nrb-1
/// \param cb  source column block, 0 .. ncb-1
/// \return    byte offset into the destination
inline size_t lm_head_block_offset(int rb, int cb, int ncb) {
    const int V    = LM_HEAD_VERTICAL_BLOCKS;
    const int half = (rb % V) / (V / 2);        // which shim stream
    const int i    = (rb % V) % (V / 2);        // which core within the pair
    const size_t group = (size_t)(rb / V);
    const size_t slot  = ((group * 2 + (size_t)half) * (size_t)ncb +
                          (size_t)cb) * (size_t)(V / 2) + (size_t)i;
    return slot * (size_t)cpu_ref::Q80_BLOCK_BYTES;
}

/// Copy `w` into `dst` in the bitstream's order.
///
/// \param dst          the weight BO, at least `lm_head_weight_bytes(...)` long
/// \param dst_bytes    its size, checked
/// \param w            `lm_head.weight`, Q8_0
/// \param vocab_padded from `lm_head_vocab_padded`; the row blocks past the
///                     real table are filled with a replicated block 0, which
///                     is what upstream's "useless padding" branch does. Those
///                     logits are read by nobody -- `npu_lm_head::run` hands
///                     back only the first `vocab` -- but the destination has
///                     to be written, because an unwritten BO region is
///                     whatever the allocator left there and denormal garbage
///                     in a Q8_0 scale costs the cores real cycles.
inline void stage_lm_head_weights(uint8_t* dst, size_t dst_bytes,
                                  const cpu_ref::q4nx_tensor& w,
                                  int vocab_padded) {
    if (w.fmt != cpu_ref::q4nx_format::q8_0)
        throw std::runtime_error(
            "stage_lm_head_weights: lm_head.weight is " +
            std::string(cpu_ref::format_name(w.fmt)) +
            ", the bitstream dequantises Q8_0 only");
    if (w.block_bytes != cpu_ref::Q80_BLOCK_BYTES)
        throw std::runtime_error(
            "stage_lm_head_weights: block is " + std::to_string(w.block_bytes) +
            " B, expected " + std::to_string(cpu_ref::Q80_BLOCK_BYTES));
    if (vocab_padded % (LM_HEAD_VERTICAL_BLOCKS * cpu_ref::Q4NX_ROW_BLOCK) != 0)
        throw std::runtime_error(
            "stage_lm_head_weights: padded vocab " + std::to_string(vocab_padded) +
            " is not a whole number of 4-block groups");

    const int    padded_nrb = vocab_padded / cpu_ref::Q4NX_ROW_BLOCK;
    const size_t need = (size_t)padded_nrb * (size_t)w.ncb *
                        (size_t)cpu_ref::Q80_BLOCK_BYTES;
    if (dst_bytes < need)
        throw std::runtime_error(
            "stage_lm_head_weights: destination is " + std::to_string(dst_bytes) +
            " B, the padded table needs " + std::to_string(need));
    if (padded_nrb < w.nrb)
        throw std::runtime_error(
            "stage_lm_head_weights: padded vocab " + std::to_string(vocab_padded) +
            " is shorter than the table's " + std::to_string(w.rows) + " rows");

    // 1.35 GB of memcpy at load time, and the destination offset is a pure
    // function of (rb, cb), so the outer loop parallelises without a reduction.
#pragma omp parallel for schedule(static)
    for (int rb = 0; rb < padded_nrb; rb++) {
        for (int cb = 0; cb < w.ncb; cb++) {
            const uint8_t* src =
                (rb < w.nrb)
                    ? w.blocks + ((size_t)rb * (size_t)w.ncb + (size_t)cb) *
                                     (size_t)w.block_bytes
                    : w.blocks;   // the pad; see the note above
            std::memcpy(dst + lm_head_block_offset(rb, cb, w.ncb), src,
                        (size_t)cpu_ref::Q80_BLOCK_BYTES);
        }
    }
}

} // namespace q38npu

#endif  // __QWEN3_8MTP_NPU_LM_HEAD_WEIGHTS_HPP__
