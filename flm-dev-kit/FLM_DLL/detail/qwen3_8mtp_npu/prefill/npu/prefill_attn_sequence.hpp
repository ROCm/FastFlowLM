// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file prefill/npu/prefill_attn_sequence.hpp
/// \brief Instruction sequence for the prefill attention core.
///
/// Port of `gen_seq_1_4()` from
/// FLM_Xclbin/Qwen3_8/gated_attention_prefill/host/seq_gen.hpp, argument for
/// argument. A changed number here changes hardware behaviour, not an API.
///
/// **This is not a layer.** Unlike `gen_attn_layer_seq` (verify/npu/), which
/// drives a fused block from hidden states to hidden states, this design
/// computes only
///
///     o = softmax(q K^T / sqrt(d) + causal_mask) V
///
/// from a q buffer and an already-populated KV cache. The projections, the
/// q/k RMSNorm, RoPE, the output gate, o_proj and the MLP are the caller's --
/// on the CPU, in this phase. What it buys is the shape: one dispatch for a
/// whole 128-aligned window of a layer, where the verify bitstream needs
/// ceil(L / 8).
///
/// The array is 8 columns x 4 rows = 4 CUs, one per column pair. The 8 CTs of
/// a CU split one q head's 128-token chunk 16 tokens apiece. k/v is broadcast
/// from two shims, IT3 -> columns 0..3 and IT7 -> columns 4..7, so exactly two
/// kv heads are resident per round and each feeds two q heads -- the 1:2 GQA
/// mapping. 24 q heads over 4 kv heads (group 6) are covered in 6 passes:
///
///     pass p:  pair = p / 3,  pp = p % 3
///     CU c  :  kvh  = 2*pair + c/2
///              head = kvh * 6 + 2*pp + (c & 1)
///
/// which enumerates heads 0..23 exactly once and always satisfies
/// head / 6 == kvh. Matching a CPU reference does not prove this mapping --
/// a consistent permutation agrees with itself -- so the driver scores a
/// rotated-kv cross-head control as well.
///
/// The cache is `q38npu::attn_layout`'s, unchanged: two contiguous planes,
/// `k_at(t) = t*DK` and `v_at(t) = plane_rows()*DK + t*DV`. That is the whole
/// reason this can share a BO with the verify design and the decode designs --
/// one cache, one cursor, either bitstream.
///
/// Nothing here scales a *stride* with the plane pitch. The k->v hop is a base
/// offset, which npu_dma_memcpy_nd lowers to a DDR address patch rather than
/// one of the BD's 20-bit step fields, so it costs nothing to make large --
/// unlike the append side of the decode designs, where that same hop as a
/// stride masked silently past 2048 rows. The only pitch-dependent BD field
/// left is the key-block count `kv_length / 128` on a 10-bit wrap.
#ifndef __QWEN3_8MTP_NPU_PREFILL_ATTN_SEQUENCE_HPP__
#define __QWEN3_8MTP_NPU_PREFILL_ATTN_SEQUENCE_HPP__

#include "verify/npu/attn_weights.hpp"      // q38npu::attn_layout

#include "npu_utils/npu_utils.hpp"
#include "typedef.hpp"

#include <cstdint>

namespace q38npu {

// ==========================================================================
// the prefill design's floorplan
// ==========================================================================
/// Deliberately its own namespace rather than a reuse of `attn_tiles`: the two
/// designs disagree on what every shim is for, and the only thing they share
/// is the cache layout.
namespace prefill_attn_tiles {

/// The four CUs' shim pairs. CU c owns columns 2c and 2c+1; q goes in on
/// IT[c][0] (both channels, two half-chunks), y comes back on both.
constexpr npu_tiles compute_tiles[4][2] = {
    {IT0, IT1}, {IT2, IT3}, {IT4, IT5}, {IT6, IT7}
};

/// The two kv broadcast sources, indexed by column group: group 0 feeds
/// columns 0..3, group 1 columns 4..7. k on MM2S ch0, v on ch1.
constexpr npu_tiles kv_tiles[2] = {IT3, IT7};

/// Positional against the xclbin's three buffer arguments.
constexpr int o_arg_id  = 0;   ///< [L_wanted][DQ] bf16, row 0 is token L_begin
constexpr int q_arg_id  = 1;   ///< [L_wanted][DQ] bf16, same indexing
constexpr int kv_arg_id = 2;   ///< | k plane | v plane |, attn_layout's

/// Query rows per round, `lc * 8` upstream: 8 CTs of 16 tokens each. Baked
/// into the bitstream and into attn.cc's `pointer_block_q = L_begin + i*128`.
constexpr int Q_CHUNK = attn_cache::PREFILL_CHUNK;   // 128
/// Tokens per CT, `lc` upstream.
constexpr int TOKENS_PER_CT = Q_CHUNK / 8;           // 16

} // namespace prefill_attn_tiles

// ==========================================================================
/// Where each attention core keeps its three scalar run-time parameters.
///
/// **These are per-build.** They come out of the bitstream's
/// `build/bitstream/from_iron/attn.mlir.prj/input_with_addresses.mlir` as
/// `L_begin_attn_core_2_0`, `L_chunks_attn_core_2_0` and
/// `window_size_attn_core_2_0`. A stale value does not fault -- it writes into
/// some other buffer and the dispatch hangs in `qds_device::wait()`.
///
/// Whenever the xclbin is rebuilt, diff them against these. The N4 driver
/// prints them for exactly that reason.
struct prefill_attn_rtp_addresses {
    uint32_t l_begin     = 61568;   ///< absolute position of q row 0
    uint32_t l_chunks    = 11904;   ///< how many 128-token rounds to walk
    uint32_t window_size = 61600;   ///< sliding window; the full-attention
                                    ///< layers pass the plane pitch, i.e. off
};

// ==========================================================================
/// Emit the whole window's attention core into `seq`.
///
/// Must be re-emitted whenever `L_begin` or `L_wanted` changes: both are baked
/// into the RTP writes and into every q/y/kv descriptor. npu_app::operator()
/// notices the sequence version changed and re-sets up the kernel.
///
/// Generic over the layout type, and deliberately so. `q38npu::attn_layout`
/// (the 16 base full-attention layers) and `q38npu::mtp_layout` (the draft
/// head) agree field for field on everything read below -- `DH`, `DQ`, `DK`,
/// `DV`, `num_kv_heads`, `MAX_L` and the `k_at`/`v_at`/
/// `plane_rows()` algebra -- because they describe the same cache geometry for
/// the same 24/4 head split. So one bitstream serves both, and the only thing
/// the two layouts disagree about is which *other* design owns the BO. The
/// definition stays in the .cpp with an explicit instantiation per layout; the
/// driver's geometry guard is what keeps the two from drifting apart.
///
/// \param seq       the app's sequence, cleared and rebuilt
/// \param t         layout, already validated by its own `build()`. Only the
///                  geometry and `plane_rows()` are read -- **the design's own
///                  MAX_L is `t.plane_rows()`, not `t.MAX_L`**, because it is
///                  the pitch between the two planes and the apron is part of
///                  it.
/// \param L_begin   absolute position of q row 0. Unconstrained: attn.cc masks
///                  on absolute position, so a ragged cursor is legal.
/// \param L_wanted  query rows, a multiple of Q_CHUNK. The caller pads.
/// \param rtp       the per-build core RTP addresses
template <class Layout>
void gen_prefill_attn_seq(npu_sequence* seq, const Layout& t,
                          int L_begin, int L_wanted,
                          const prefill_attn_rtp_addresses& rtp);

} // namespace q38npu

#endif  // __QWEN3_8MTP_NPU_PREFILL_ATTN_SEQUENCE_HPP__
