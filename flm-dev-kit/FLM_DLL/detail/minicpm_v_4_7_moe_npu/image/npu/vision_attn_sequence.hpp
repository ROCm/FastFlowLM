// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file image/npu/vision_attn_sequence.hpp
/// \brief Instruction sequence for the vision tower's attention core.
///
/// Port of `gen_attn_seq()` from
/// FLM_Xclbin/Qwen3_8/vision_attn/host/seq_gen.hpp, descriptor for descriptor,
/// via Qwen3.8-27B's tower. The bitstream is REUSED UNCHANGED for MiniCPM: its
/// baked head_dim of 72 and 16 heads are exactly SigLIP-so400m's.
/// A changed number here changes hardware behaviour, not an API.
///
/// **This is not the language model's attention.** `vision_attn.xclbin`
/// computes only
///
///     o = softmax(q K^T / sqrt(72)) V
///
/// with a FULL bidirectional mask -- no causal horizon, no GQA, no cache, no
/// projections and no RoPE. Projections are the caller's matmuls; SigLIP has no
/// RoPE at all. The
/// geometry is baked into the bitstream: `kernel/common.h` fixes `dh = 72`,
/// `lq = 64`, `lk_mt = 512` and `scale = 1/sqrt(72)` as compile-time constants,
/// so the only run-time parameters are the two lengths, and they arrive as
/// RTP writes rather than as sequence shape.
///
/// The array is 8 columns x 4 rows = 4 CUs, one per column pair. Each round a
/// CU works on ONE head and 512 query rows: the even column of the pair takes
/// rows [0,256) of the round and the odd column rows [256,512), and inside a
/// column the four CTs take 64 rows each. So one round covers 4 heads (one per
/// CU) and `Heads/4` outer passes cover all 16. k and v are re-sent whole on
/// every round, from the odd shim of the pair -- the design has no residency,
/// which is why a short image is a fixed cost rather than a cheap one.
///
/// `attn.cc`'s core loop trip count is `L_buffer[0] >> 9`, i.e. `L_pad / 512`,
/// and the kernel runs inside `for _ in range_(0xFFFFFFFF)`. So the sequence
/// must emit EXACTLY `L_pad / 512` rounds per head group and exactly
/// `Heads / 4` head groups; a round more or fewer leaves the array a pass out
/// of step for the next dispatch.
#ifndef __MINICPM_V_4_7_NPU_VISION_ATTN_SEQUENCE_HPP__
#define __MINICPM_V_4_7_NPU_VISION_ATTN_SEQUENCE_HPP__

#include "npu_utils/npu_utils.hpp"
#include "typedef.hpp"

#include <cstdint>

namespace mcvm_npu {

// ==========================================================================
/// The vision design's floorplan.
// ==========================================================================
/// Its own namespace: this design disagrees with `prefill_attn_tiles` about
/// what every shim is for, and shares nothing with it but the tile names.
namespace vision_attn_tiles {

/// The four CUs' shim pairs. CU c owns columns 2c and 2c+1. q goes in and y
/// comes back on BOTH shims of the pair; k and v go in on the odd one only.
constexpr npu_tiles compute_tiles[4][2] = {
    {IT0, IT1}, {IT2, IT3}, {IT4, IT5}, {IT6, IT7}
};

/// Positional against the xclbin's four buffer arguments. `host.cpp` calls
/// `app_0(o_0, q_0, k_0, v_0)`, so o is 0.
constexpr int o_arg_id = 0;   ///< [L_pad][heads*DH] bf16 out
constexpr int q_arg_id = 1;   ///< [L_pad][heads*DH] bf16 in
constexpr int k_arg_id = 2;   ///< [L_pad][heads*DH] bf16 in
constexpr int v_arg_id = 3;   ///< [L_pad][heads*DH] bf16 in

/// Query rows per CT, `lq` in kernel/common.h.
constexpr int LQ = 64;
/// Query rows one round covers: 8 CTs of a CU, `lq * 8`.
constexpr int ROWS_PER_ROUND = LQ * 8;              // 512
/// Key rows one k/v descriptor block carries, `lk_mt` in kernel/common.h.
/// Numerically equal to ROWS_PER_ROUND and independently so -- the k side is
/// the memtile's double buffer, the q side is the CT count.
constexpr int KV_CHUNK = 512;
/// The head_dim the kernel is compiled for. Not a parameter.
constexpr int DH_BAKED = 72;
/// Heads per round, one per CU.
constexpr int HEADS_PER_PASS = 4;

} // namespace vision_attn_tiles

// ==========================================================================
/// Where each attention core keeps its two run-time lengths.
///
/// **This is per-build.** It comes out of the bitstream's
/// `build/bitstream/from_iron/attn.mlir.prj/input_with_addresses.mlir` as
/// `L_attn_core_2_0`, and `FLM_Xclbin/Qwen3_8/vision_attn/tools/
/// extract_rtp_addr.sh` is what reads it. A stale value does not fault -- it
/// writes into some other buffer and the dispatch hangs in
/// `qds_device::wait()`.
///
/// Whenever vision_attn.xclbin is rebuilt, diff this against
/// `build/rtp_addr.txt`. The N15 driver prints both for exactly that reason.
struct vision_attn_rtp_addresses {
    uint32_t l = 57344;        ///< word 0: L_pad. word 1 (l + 4): L_valid.
};

// ==========================================================================
/// Emit one image's whole attention into `seq`.
///
/// Must be re-emitted whenever `L_pad`, `L_valid` or `head_perm` changes: all
/// three are baked into the RTP writes or into the k/v descriptors.
/// `npu_app::operator()` notices the sequence version moved and re-sets up the
/// kernel.
///
/// \param seq      the app's sequence, cleared and rebuilt
/// \param L_pad    query AND key rows the array walks. Must be a positive
///                 multiple of `ROWS_PER_ROUND`; the caller pads, the kernel
///                 cannot
/// \param L_valid  the real patch count. Keys at or past it are masked off by
///                 `attn.cc`'s `mask_tail_and_get_max`; QUERIES past it are
///                 computed anyway and the caller drops them
/// \param DH       head_dim. Only 72 is legal -- the kernel is compiled for it
/// \param Heads    query heads. Must be a positive multiple of 4
/// \param rtp      the per-build RTP address
/// \param head_perm null for the real run; non-null sends k/v head
///                 `head_perm[h]` to the CU computing q head `h`. This is the
///                 cross-head control, and it is supported on the device
///                 rather than refused because a correct run has to be scored
///                 against an imposter that a *consistent permutation* would
///                 agree with
/// \param n_seg    segments in one dispatch -- MiniCPM's batching of same-shape
///                 views (added in this port; 1 is Qwen3.8's single image).
///                 Segment s occupies rows [s*L_pad, (s+1)*L_pad) of q/k/v/o and
///                 gets its own `Heads/4` passes, each attending only within
///                 that segment. The kernel cannot tell: it runs one pass per
///                 outer iteration forever and reads the SAME two RTP lengths
///                 for every pass, which is why all segments of a dispatch
///                 must share L_pad AND L_valid. No RTP is rewritten
///                 mid-sequence.
void gen_vision_attn_seq(npu_sequence* seq,
                         int L_pad, int L_valid, int DH, int Heads,
                         const vision_attn_rtp_addresses& rtp,
                         const int* head_perm = nullptr, int n_seg = 1);

} // namespace mcvm_npu

#endif  // __MINICPM_V_4_7_NPU_VISION_ATTN_SEQUENCE_HPP__
