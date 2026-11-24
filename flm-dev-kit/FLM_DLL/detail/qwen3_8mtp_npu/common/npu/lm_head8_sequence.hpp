// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file common/npu/lm_head8_sequence.hpp
/// \brief Instruction sequence for the vocabulary projection at L = 8, the
///        verify phase's shape.
///
/// Port of `generate_seq()` from
/// FLM_Xclbin/Qwen3_8/lm_head_8/host/host.cpp, descriptor for descriptor, the
/// same way `lm_head_sequence.hpp` ports the L = 1 design. A changed number
/// here changes hardware behaviour, not an API: `W_PKT_ID`, the channel
/// assignment and `LOOKAHEAD` are a contract with that design's
/// iron/q4k_mm.py, and nothing in the toolchain checks the two agree.
///
/// ## What this shares with L = 1, and what it does not
///
/// The array topology is the same -- 4 rows x 8 columns, `M_PER_ROUND = 1024`
/// vocabulary entries, 243 rounds over the padded table -- so the *weight
/// layout is identical* and `stage_lm_head_weights()` stages for both designs
/// unchanged. (Verified: `reorder_cpy_q80` is byte-for-byte the same function
/// in the two host.cpp files.) That is why this header reuses
/// `lm_head_vocab_padded()` and `lm_head_weight_bytes()` rather than
/// redefining them, and static_asserts the two tilings really do match.
///
/// Three things do differ, and all three come from the same fact: npu2 core
/// data memory is 64 KB and 8 x 5120 x 2 = 80 KB of activation will not fit.
///
///   1. **x streams per round** instead of being broadcast once. It is an
///      ordinary producer/consumer slot on the same shim channel as the
///      group-a weights, alternating with them. That is what makes the design
///      re-dispatchable -- the first version filled x from a one-shot prologue
///      BD, which the memtile ring steps over after the first pass, so
///      dispatch 2's x landed in a weight BD and every logit came back NaN
///      while the command still reported COMPLETED.
///   2. **The final rms_norm is split across the k loop.** The core never
///      holds a whole row, but it does see every column of one across the k
///      loop, so the sum of squares is an ordinary reduction and the
///      reciprocal it yields is a per-token scalar that commutes through the
///      matmul -- applied to the finished accumulator, not to x. Only the
///      per-column `model.norm.weight` has to meet x elementwise, and it rides
///      in as lane `L_TOKENS` of the x stream. So this bitstream's input is
///      *pre*-norm x, the same as the L = 1 one, and `npu_lm_head_8::run` is
///      `cpu_ref::lm_head::norm_and_forward` argument for argument.
///   3. **LOOKAHEAD is 2, not 3.** A round issues four shim BDs (x, y, wa, wb)
///      where L = 1 issued three; `LOOKAHEAD + 1 = 4` rounds in flight would
///      need all 16 of the shim's BDs and leave the sequencer no free id to
///      stage into, which hangs. Twelve BDs costs ~3%.
///
/// y comes back interleaved as `[round][col][core][token][32]`, because each
/// core emits all L tokens of its 32-row weight block together. The host
/// un-interleaves it.
#ifndef __QWEN3_8MTP_NPU_LM_HEAD8_SEQUENCE_HPP__
#define __QWEN3_8MTP_NPU_LM_HEAD8_SEQUENCE_HPP__

#include "common/npu/lm_head_sequence.hpp"

#include "npu_utils/npu_utils.hpp"
#include "typedef.hpp"

#include <cstddef>
#include <cstdint>

namespace q38npu {

namespace lm_head8_tiles {

/// Token rows per dispatch. Baked into the bitstream as `L_TOKENS` in both
/// kernel/qmm.cc and iron/lm_head_aie.py; this constant only shapes the
/// descriptors, so changing it here does not change what the array computes.
constexpr int L_TOKENS = 8;

/// Lanes in the x stream: the L token rows plus one carrying
/// `model.norm.weight`. The kernel folds the weight into x as it loads it --
/// see the header comment -- and it rides in the activation stream rather than
/// a buffer of its own so that nothing needs a one-shot fill. `X_LANES` in
/// kernel/qmm.cc and iron/lm_head_aie.py.
constexpr int X_LANES = L_TOKENS + 1;

/// The Q8_0 column block the kernel's k-loop walks, and the unit x is
/// interleaved by. `Q4NX_COL_BLOCK_SIZE` in the design.
constexpr int K_CHUNK = 256;

/// Same array tiling as L = 1 -- that is what lets the weight staging be
/// shared. Named locally so the descriptors below read as one design.
constexpr int M           = lm_head_tiles::M;
constexpr int COLUMNS     = lm_head_tiles::COLUMNS;
constexpr int M_PER_COL   = lm_head_tiles::M_PER_COL;
constexpr int M_PER_ROUND = lm_head_tiles::M_PER_ROUND;

/// Two, not the L = 1 design's three. See the header comment.
constexpr int LOOKAHEAD = 2;

static_assert(M_PER_ROUND == 32 * 32,
              "lm_head_8's host.cpp pads to 32*32 per round; the two tilings "
              "must agree or the shared weight staging is wrong");

} // namespace lm_head8_tiles

/// Emit the pass. Once per object -- both shapes are pinned at construction.
///
/// \param seq          the app's sequence
/// \param vocab_padded from `lm_head_vocab_padded()`, shared with the L = 1
///                     design because the round is the same 1024 entries
/// \param D            hidden size; baked into the kernel as MODEL_DIM, so
///                     this argument only sizes the descriptors
void gen_lm_head8_seq(npu_sequence* seq, uint32_t vocab_padded, uint32_t D);

} // namespace q38npu

#endif  // __QWEN3_8MTP_NPU_LM_HEAD8_SEQUENCE_HPP__
