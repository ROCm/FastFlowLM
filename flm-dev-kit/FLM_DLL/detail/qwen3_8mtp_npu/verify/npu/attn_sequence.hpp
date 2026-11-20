// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file verify/npu/attn_sequence.hpp
/// \brief Instruction sequence for the gated full-attention layer.
///
/// Port of decoding_layer::_gen_sequence() from
/// FLM_Xclbin/Qwen3_8/gated_attention_verify/host, argument for argument: a
/// changed number here changes hardware behaviour, not an API.
///
/// `context_offset` is a parameter because it is baked into the per-token key
/// counts, the replay length and the append address -- the sequence must be
/// rebuilt whenever the cache length changes.
///
/// The constants below are the attention design's floorplan and must not be
/// unified with the delta-net design's (whose xyr_tile is IT3, not IT2).
///
/// Ordering rules. All of these deadlock rather than answer wrongly, which is
/// the good case: a hang points at the sequence, a wrong number points
/// anywhere.
///
///   1. _move_weights() calls are POSITIONAL against proj_main()'s mvms --
///      q|k|v, z, o, up|gate, down, in that order and all present. An mvm with
///      no weights behind it stalls the cores; weights with no mvm to consume
///      them stall the DMA wait inside _move_weights().
///   2. Every read is posted before any weights move: the cores stall on their
///      y locks until the shim is draining.
///   3. The k|v append is waited on, per column, BEFORE the replay is posted --
///      and both sit on the same two shims, because npu_dma_wait does not
///      reach across shim tiles. Without that the CUs attend to stale rows.
///   4. o_proj's weight round goes AFTER the replay, and up|gate and down after
///      it, because the x each reduces over does not exist until the stage
///      before it has run. Issuing o's round early is a cycle: the host blocks
///      on a weight round the array cannot consume yet, before posting the
///      replay the array is waiting for.
#ifndef __QWEN3_8MTP_NPU_ATTN_SEQUENCE_HPP__
#define __QWEN3_8MTP_NPU_ATTN_SEQUENCE_HPP__

#include "verify/npu/attn_weights.hpp"

#include "npu_utils/npu_utils.hpp"
#include "typedef.hpp"

#include <cstdint>
#include <string>

namespace q38npu {

// ==========================================================================
// the attention design's floorplan
// ==========================================================================
// From decoding_layer.hpp:109-129.
namespace attn_tiles {

/// The norm tile's shim, column 2: x in on MM2S ch0, both layernorm weight
/// vectors on ch1. Takes the assembled gated row on S2MM ch0 and the post-norm
/// layer output on S2MM ch1.
constexpr npu_tiles xyr_tile = IT2;
/// Column 3's shim: the RoPE weight block in on MM2S ch0. The staging memtile
/// is in this column, directly south of the rms tile.
constexpr npu_tiles attn_tile = IT3;
/// The two attention columns' shims: the replayed context in on MM2S ch0 (k)
/// and ch1 (v), the k|v append back on S2MM ch0.
constexpr npu_tiles attn_col_tiles[2] = {IT4, IT5};
/// The eight attention cores, so the per-token L can be written to each. A CU
/// is a qk core with its kv core directly north of it -- the qk core writes the
/// scores into the kv tile's memory. Getting this pair the wrong way round does
/// not hang; it silently produces zeros.
constexpr npu_tiles attn_qk_tiles[4] = {CT04, CT24, CT05, CT25};
constexpr npu_tiles attn_kv_tiles[4] = {CT14, CT34, CT15, CT35};
/// The four projection columns' shims, in engine order.
constexpr npu_tiles mvm_tiles[4] = {IT0, IT1, IT6, IT7};

/// Positional against the xclbin's five buffer arguments.
constexpr int x_arg_id        = 0;   ///< | x | o | down | h2 |
constexpr int proj_arg_id     = 1;   ///< q4_k projection weights
constexpr int rms_arg_id      = 2;   ///< both layernorm gains
constexpr int rope_arg_id     = 3;   ///< cos|sin per token | q_norm | k_norm
constexpr int kv_cache_arg_id = 4;   ///< | k plane | v plane |

} // namespace attn_tiles

// ==========================================================================
/// Where each attention core keeps its per-token key-count array.
///
/// **These are per-build.** They come out of the bitstream's
/// input_with_addresses.mlir and land in xclbins/rtp_addresses.txt; a stale
/// value does not fault, it writes into some other buffer and the run hangs in
/// the wait. The defaults below are the ones this repository's
/// gated_attention_verify build emits -- and are deliberately NOT the
/// qwen3_5vl header's 53248 / 53536, which belong to a different design.
///
/// Whenever the xclbin is rebuilt, diff rtp_addresses.txt against these.
struct attn_rtp_addresses {
    uint32_t qk = 57344;   ///< RTP_L_attn_qk_core_{2,4}_{4,5}
    uint32_t kv = 36864;   ///< RTP_L_attn_kv_core_{3,5}_{4,5}
};

// ==========================================================================
/// Emit the whole pass into `seq`.
///
/// Must be re-emitted whenever `context_offset` or `t.MAX_L` changes; both are
/// baked into descriptors. npu_app::operator() notices the sequence version
/// changed and re-sets up the kernel, so there is nothing else to do after.
///
/// \param seq             the app's sequence, cleared and rebuilt
/// \param t               layout, already validated by attn_layout::build()
/// \param context_offset  how many keys the cache already holds. Token t of
///                        this pass attends to context_offset + t + 1 of them,
///                        and the new k|v are appended at this offset.
/// \param rtp             the per-build core RTP addresses
void gen_attn_layer_seq(npu_sequence* seq, const attn_layout& t,
                        int context_offset, const attn_rtp_addresses& rtp);

} // namespace q38npu

#endif  // __QWEN3_8MTP_NPU_ATTN_SEQUENCE_HPP__
