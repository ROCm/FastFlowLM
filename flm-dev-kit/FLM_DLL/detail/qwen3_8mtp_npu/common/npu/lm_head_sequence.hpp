// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file common/npu/lm_head_sequence.hpp
/// \brief Instruction sequence for the fused final-norm + vocabulary
///        projection, at L = 1.
///
/// Port of `generate_seq()` from
/// FLM_Xclbin/Qwen3_8/lm_head_npu_bin/host/host.cpp, descriptor for
/// descriptor. A changed number here changes hardware behaviour, not an API:
/// the packet ids and the channel assignment below are a contract with
/// `X_PKT_ID` / `W_PKT_ID` and `lm_head_memtile` in that design's
/// iron/q4k_mm.py, and nothing in the toolchain checks the two agree.
///
/// The design computes
///
///     logits[VOCAB] = rms_norm(x[D], norm_w[D]) . W^T
///
/// with **the norm fused into the kernel**, over a 4 row x 8 column array.
/// One round covers `M_PER_ROUND = 1024` vocabulary entries, 128 per column,
/// 32 per core; the 243 rounds tile the padded vocabulary. So the input is the
/// *pre*-norm hidden state, and the kernel's epsilon (`1e-6` in kernel/qmm.cc)
/// has to match `q38::RMS_EPS` -- it does.
///
/// ## The weight stream is the whole cost
///
/// Per dispatch this moves `vocab_padded x D x 17/16` bytes of weights --
/// 1.35 GB for 248832 x 5120 -- against 20 KB of x and 486 KB of logits. The
/// projection is therefore DRAM-bound and not MAC-bound, on either engine:
/// 26 ms on the device is 52 GB/s, and the 32-thread host reference runs the
/// same table at very nearly the same rate. Nothing about how this sequence is
/// emitted moves that number; only reading fewer bytes would.
///
/// That is also why there is no L: a second row would re-read the same 1.35 GB
/// unless the *kernel* held both rows resident, which this one does not. The
/// L = 8 variant is a separate design (FLM_Xclbin/Qwen3_8/lm_head_8) and a
/// separate bitstream.
///
/// ## Two shim channels per column
///
/// A column's weights arrive on two shim MM2S channels, one per pair of cores,
/// so `reorder_cpy_q80` (lm_head_weights.hpp) has to lay each column-round out
/// as two contiguous halves rather than one run with all four cores' blocks
/// interleaved. IT0's channel 0 is also the port the x broadcast enters on, so
/// that one flow -- and only that one -- carries a packet id.
#ifndef __QWEN3_8MTP_NPU_LM_HEAD_SEQUENCE_HPP__
#define __QWEN3_8MTP_NPU_LM_HEAD_SEQUENCE_HPP__

#include "npu_utils/npu_utils.hpp"
#include "typedef.hpp"

#include <cstddef>
#include <cstdint>

namespace q38npu {

namespace lm_head_tiles {

/// Vocabulary entries one round covers: 32 per core, 4 cores per column,
/// 8 columns.
constexpr int M              = 32;
constexpr int COLUMNS        = 8;
constexpr int M_PER_COL      = M * 4;              // 128
constexpr int M_PER_ROUND    = M_PER_COL * COLUMNS; // 1024

/// Rounds of shim BD ping-pong; also how far the y waits lag the y issues.
constexpr int LOOKAHEAD      = 3;

/// Bits per weight, x16. Q8_0 packs a 32 x 256 chunk as 32*256 int8 plus 256
/// bf16 scales = 8704 B for 8192 weights, i.e. 8.5 bits = 17/16 bytes each.
/// Written as a ratio rather than as 8.5/8 so every size below is exact
/// integer arithmetic.
constexpr size_t Q80_NUM     = 17;
constexpr size_t Q80_DEN     = 16;

} // namespace lm_head_tiles

/// The vocabulary the device actually projects: rounded up to a whole round,
/// because the sequence emits `vocab_padded / M_PER_ROUND` of them and a
/// partial round has no descriptor shape. The tail entries are garbage -- the
/// weight reorder fills them with a replicated block -- so the caller must
/// read only the first `vocab` logits.
inline int lm_head_vocab_padded(int vocab) {
    const int R = lm_head_tiles::M_PER_ROUND;
    return (vocab + R - 1) / R * R;
}

/// Bytes the weight BO must hold: the padded table in Q8_0.
inline size_t lm_head_weight_bytes(int vocab_padded, int D) {
    return (size_t)vocab_padded * (size_t)D * lm_head_tiles::Q80_NUM /
           lm_head_tiles::Q80_DEN;
}

/// Emit the pass. Once per object -- both shapes are pinned and they are the
/// whole of the sequence's input.
///
/// \param seq          the app's sequence
/// \param vocab_padded from `lm_head_vocab_padded`; must be a multiple of
///                     `M_PER_ROUND`
/// \param D            hidden size. Baked into the kernel as MODEL_DIM, so a
///                     build is not interchangeable across checkpoints; this
///                     argument only sizes the descriptors
void gen_lm_head_seq(npu_sequence* seq, uint32_t vocab_padded, uint32_t D);

} // namespace q38npu

#endif  // __QWEN3_8MTP_NPU_LM_HEAD_SEQUENCE_HPP__
