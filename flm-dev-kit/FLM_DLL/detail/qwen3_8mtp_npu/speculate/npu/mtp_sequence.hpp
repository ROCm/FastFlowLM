// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file speculate/npu/mtp_sequence.hpp
/// \brief Instruction sequence for the MTP draft head's fused layer.
///
/// Port of decoding_layer::_gen_sequence() from
/// FLM_Xclbin/Qwen3_8/MTP_decoding/host, argument for argument: a changed
/// number here changes hardware behaviour, not an API.
///
/// `context_offset` is a parameter because it is baked into the attention key
/// count, the replay length and the append address -- the sequence must be
/// rebuilt whenever the head's cache length changes. At T = 1 that is every
/// dispatch, which is why it is cheap here and expensive in the attention port:
/// this design has ~1/8 the descriptors.
///
/// The constants below are the MTP design's floorplan. It is close to the
/// gated_attention_verify design's but **not** the same: this one has a
/// passthrough tile (IT3) in front carrying the fusion, and its attention cores
/// keep L at a different RTP address. Do not unify the two.
///
/// Ordering rules. All of these deadlock rather than answer wrongly, which is
/// the good case: a hang points at the sequence, a wrong number points anywhere.
///
///   1. _move_weights() calls are POSITIONAL against proj_main()'s mvms --
///      fc, k|v|q, z, o, up|gate, down, in that order and all present. An mvm
///      with no weights behind it stalls the cores; weights with no mvm to
///      consume them stall the DMA wait inside _move_weights().
///   2. Every read is posted before any weights move: the cores stall on their
///      y locks until the shim is draining.
///   3. The k|v append is waited on, per column AND per channel, BEFORE the
///      replay is posted -- and both sit on the same two shims, because
///      npu_dma_wait does not reach across shim tiles. k and v arrive on
///      separate channels, so that is two waits per column and not one.
///   4. o_proj's weight round goes AFTER the replay, and up|gate and down after
///      it, because the x each reduces over does not exist until the stage
///      before it has run.
#ifndef __QWEN3_8MTP_NPU_MTP_SEQUENCE_HPP__
#define __QWEN3_8MTP_NPU_MTP_SEQUENCE_HPP__

#include "speculate/npu/mtp_weights.hpp"

#include "npu_utils/npu_utils.hpp"
#include "typedef.hpp"

#include <cstdint>

namespace q38npu {

// ==========================================================================
// the MTP design's floorplan
// ==========================================================================
// From MTP_decoding/host/decoding_layer.hpp:119-162.
namespace mtp_tiles {

/// Column 3's shim, feeding the fusion's passthrough tile CT[3][3]: the two
/// raw D-wide input halves on MM2S ch0, the two pre-fc norm gains on ch1. The
/// passthrough tile owns both norms, which is why the host sends x *raw* --
/// pre-norming on the host would norm twice.
constexpr npu_tiles pass_tile = IT3;
/// The norm tile's shim, column 2: the layer's two layernorm gains on MM2S ch1
/// and the RoPE weight block on ch0. It also takes the layer output back on
/// S2MM ch1.
constexpr npu_tiles xyr_tile = IT2;
/// h2, the layer's scored output -- the rms tile sits in column 2 and its y
/// port reaches this shim one hop straight south.
constexpr npu_tiles layer_out_tap_tile = IT2;
/// The two attention columns' shims: the replayed context in on MM2S ch0 (k)
/// and ch1 (v), the append back on S2MM ch0 (k) and ch1 (v).
///
/// The append is split by plane because the RoPE tile stamps k and v with
/// different packet ids, which the array routes to those two channels. That is
/// what lets each plane be written by a BD whose only stride is one row -- see
/// mtp_weights.hpp::MAX_L_LIMIT -- and it means the sequence must drain BOTH
/// channels before the replay, not just ch0.
constexpr npu_tiles attn_col_tiles[2] = {IT4, IT5};
/// The four CUs, in CU order -- column 4's two, then column 5's, matching the
/// q fan-out in iron/attn_hub.py. A CU is a qk core with its kv core directly
/// north of it; getting the pair the wrong way round does not hang, it
/// silently produces zeros.
constexpr npu_tiles attn_qk_tiles[4] = {CT04, CT24, CT05, CT25};
constexpr npu_tiles attn_kv_tiles[4] = {CT14, CT34, CT15, CT35};
/// The four projection columns' shims, in engine order.
constexpr npu_tiles mvm_tiles[4] = {IT0, IT1, IT6, IT7};

/// Positional against the xclbin's five buffer arguments.
constexpr int x_arg_id        = 0;   ///< | e | h | layer_out |
constexpr int proj_arg_id     = 1;   ///< q4_k projection weights
constexpr int rms_arg_id      = 2;   ///< four norm gains, D each
constexpr int rope_arg_id     = 3;   ///< cos|sin | q_norm | k_norm
constexpr int kv_cache_arg_id = 4;   ///< two planes: [MAX_L][DK] then [MAX_L][DV]

/// Packet ids, from iron/constants.py's it1_pkt_id. These select a destination
/// inside the column rather than a port, so a wrong one is delivered somewhere
/// plausible instead of failing.
///
/// There is deliberately no id for the fusion input. IT[3].MM2S[0] reaches one
/// destination, so that link is circuit-switched and `_send_fc_input` sends -1;
/// `it1_pkt_id.to_pass` exists in constants.py but no switch routes it, and
/// using it hung every dispatch after the first.
constexpr int rms_w_pkt_id  = 0;   ///< to_rms
constexpr int rope_w_pkt_id = 1;   ///< to_rope

} // namespace mtp_tiles

// ==========================================================================
/// Where each MTP attention core keeps its key count.
///
/// **These are per-build, and they are NOT attn_rtp_addresses.** The MTP
/// design's qk address is 9216 where gated_attention_verify's is 57344; the kv
/// address happens to coincide at 36864, which is exactly the kind of partial
/// agreement that makes a shared struct look like it works. A stale address
/// does not fault -- it writes the key count into some other buffer, the cores
/// never see their L, and the run hangs in qds_device::wait().
///
/// Source: MTP_decoding/build/QWEN3_8_27B/xclbins/rtp_addresses.txt. Whenever
/// that xclbin is rebuilt, diff it against these.
struct mtp_rtp_addresses {
    uint32_t qk = 9216;    ///< RTP_L_attn_qk_core_{2,4}_{4,5}
    uint32_t kv = 36864;   ///< RTP_L_attn_kv_core_{3,5}_{4,5}
};

// ==========================================================================
/// Emit one T = 1 pass of the draft head into `seq`.
///
/// Must be re-emitted whenever `context_offset` or `t.MAX_L` changes; both are
/// baked into descriptors. npu_app::operator() notices the sequence version
/// changed and re-sets up the kernel, so there is nothing else to do after.
///
/// \param seq             the app's sequence, cleared and rebuilt
/// \param t               layout, already validated by mtp_layout::build()
/// \param context_offset  how many keys the head's cache already holds. The
///                        drafted token attends to context_offset + 1 of them
///                        and its own k|v are appended at this row.
/// \param rtp             the per-build core RTP addresses
void gen_mtp_layer_seq(npu_sequence* seq, const mtp_layout& t,
                       int context_offset, const mtp_rtp_addresses& rtp);

} // namespace q38npu

#endif  // __QWEN3_8MTP_NPU_MTP_SEQUENCE_HPP__
