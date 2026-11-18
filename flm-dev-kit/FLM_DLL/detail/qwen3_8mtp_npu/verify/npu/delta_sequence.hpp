// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file verify/npu/delta_sequence.hpp
/// \brief Instruction sequence for the whole gated-delta-net layer.
///
/// Port of `linear_decoding_layer::_gen_linear_sequence()` from
/// FLM_Xclbin/Qwen3_8/gate_delta_net_verify/host, argument for argument: a
/// changed number here changes hardware behaviour, not an API.
///
/// The sibling of `verify/npu/attn_sequence.hpp`, and the two floorplans are
/// **not** interchangeable -- this design's `xyr_tile` is IT3, the attention
/// design's is IT2. Nothing is shared between the two namespaces on purpose.
///
/// Unlike the attention sequence, nothing here depends on the context length:
/// a delta-net layer's whole history lives in the recurrent state and the conv
/// window, both of which are fixed-size. The sequence is therefore built once
/// per layer object and never re-emitted.
///
/// ## What crosses DDR, and why
///
/// Two results: h2 (IT3 S2MM ch1) and the causal conv window (IT5 S2MM ch0).
/// The recurrent state goes **in** on IT4 MM2S ch0 and does not come back --
/// the cores can only report S after all L_TOKENS steps, and speculative decode
/// needs S after whichever prefix verification accepts, so the host replays it
/// from the triplet arena instead.
///
/// The arena is the third DDR region and it is not a readback at all: it is the
/// token-major -> head-major transpose that no tile on the conv path can do.
/// The conv tile scatters k|v|q into it on IT6 S2MM ch0, the a/b tile scatters
/// alpha|beta into the two-word holes on IT5 S2MM ch1, and IT4 MM2S ch1 reads
/// the whole thing back linearly. That is what the two mid-sequence waits are.
///
/// ## Ordering rules
///
/// All of these deadlock rather than answer wrongly, which is the good case.
///
///   1. `_move_weights()` calls are POSITIONAL against proj_main()'s mvms --
///      qkv, z, o, up|gate, down, in that order and all present. An mvm with no
///      weights behind it stalls the cores; weights with no mvm to consume them
///      stall the DMA wait inside `_move_weights()`.
///   2. Every read is posted before any weights move: the cores stall on their
///      y locks until the shim is draining.
///   3. The a/b weights go before the qkv weights and are re-sent once per
///      token -- the a/b tile holds one x row and streams a whole set past it.
///      The rms tile cannot finish row t until the a/b tile has taken it, and
///      the engine will not drain a qkv weight until all L_TOKENS rows are in
///      MT[1].
///   4. o's weight round goes after the arena readback, up|gate after o, down
///      after up|gate: the x each reduces over does not exist until the stage
///      before it has run. Issuing a round early is a cycle -- the host blocks
///      on a weight round the array cannot consume yet, before posting the
///      transfer the array is waiting for.
///   5. The bias block leads *every* a/b weight send, inside the per-token
///      loop. Sending it once ahead of the loop is a prologue BD, and a
///      prologue works for exactly one dispatch per xclbin load.
#ifndef __QWEN3_8MTP_NPU_DELTA_SEQUENCE_HPP__
#define __QWEN3_8MTP_NPU_DELTA_SEQUENCE_HPP__

#include "common/cpu_ref_delta_net_layer.hpp"  // cpu_ref::delta_net_geometry
#include "prefill/npu/delta_weights.hpp"       // q38npu::delta_layout

#include "npu_utils/npu_utils.hpp"
#include "typedef.hpp"

#include <cstddef>
#include <cstdint>

namespace q38npu {

// ==========================================================================
// the delta-net design's floorplan
// ==========================================================================
// From linear_decoding_layer.hpp:176-182 and the channel map at
// _gen_linear_sequence():760-770.
namespace delta_tiles {

/// The rms tile's own shim, column 3: x in on MM2S ch0, both layernorm weight
/// vectors on ch1, h2 back out on S2MM ch1. (o and the post-attention norm stay
/// on the array -- see ARCHITECTURE.md §7.)
constexpr npu_tiles xyr_tile = IT3;
/// Column 4's shim: the recurrent state in on MM2S ch0, the assembled arena
/// back in on MM2S ch1. Nothing comes out of this column.
constexpr npu_tiles state_tile = IT4;
/// Column 5's shim: the q8_0 a/b weights in on MM2S ch0, the conv taps +
/// ssm_norm and then the conv cache on MM2S ch1; the conv window out on
/// S2MM ch0 and the alpha|beta scatter on S2MM ch1.
constexpr npu_tiles conv_tile = IT5;
/// The four projection columns' shims, in engine order. IT6 doubles as the
/// conv tile's triplet scatter on S2MM ch0 -- it only ever *sends* q4_k
/// weights, so both of its S2MM channels were idle.
constexpr npu_tiles mvm_tiles[4] = {IT0, IT1, IT6, IT7};
/// The shim the conv tile's k|v|q triplets land on.
constexpr npu_tiles arena_in_tile = IT6;

/// Positional against the xclbin's five buffer arguments.
constexpr int x_arg_id        = 0;   ///< | x | triplet arena | h2 |
constexpr int proj_arg_id     = 1;   ///< q4_k projections, then the q8_0 a/b tail
constexpr int rms_arg_id      = 2;   ///< both layernorm gains
constexpr int aux_w_arg_id    = 3;   ///< | conv taps | ssm_norm |
constexpr int kv_cache_arg_id = 4;   ///< | conv in | S in | conv out |

} // namespace delta_tiles

// ==========================================================================
// geometry
// ==========================================================================
/// One delta-net verify layer's BO sizes and offsets, in **bf16 words**.
///
/// `delta_layout` covers the weight BO, which the prefill phase shares; this
/// covers the four activation BOs, which only this bitstream has. Every field
/// is `linear_decoding_layer`'s member of the same name.
struct delta_verify_layout {
    delta_layout w;      ///< the weight BO, shared with the prefill phase

    int L_TOKENS = 0;    ///< query rows per pass; the bitstream's -DL_TOKENS
    int D = 0;
    int intermediate_size = 0;
    int DQ = 0, DK = 0, DV = 0;
    int conv_dim = 0;            ///< DQ + DK + DV, the depthwise conv's channels
    int conv_kernel_size = 0;
    int qk_heads = 0, v_heads = 0, v_heads_padded = 0;
    int qk_head_dim = 0, v_head_dim = 0;

    /// Rows each projection emits per token. Weight-stream shapes, not DDR
    /// shapes: neither qkv nor z ever reaches the host.
    int qkv_out = 0;             ///< conv_dim
    int z_out   = 0;             ///< DV
    int ab_out  = 0;             ///< 2 * v_heads_padded, alpha then beta, f32

    // --- arg 0: | x | arena | h2 | ------------------------------------------
    /// The DDR triplet arena. One (head, token) block is
    ///
    ///   [ k(128) | alpha(2) | v(128) | beta(2) | q(128) pad(2) ]
    ///
    /// i.e. three `triplet_stride`-word sub-blocks. Blocks are ordered
    /// [head][token], so reading heads 0 .. v_heads back out is a single linear
    /// DMA. Every offset in it is an even word count by construction (128 + 2,
    /// 3 * 130 and arena_offset are all even), which is what lets the a/b tile
    /// write f32 into the two-word holes.
    int    triplet_stride = 0;   ///< v_head_dim + 2
    int    arena_block    = 0;   ///< 3 * triplet_stride
    int    arena_heads    = 0;   ///< v_heads_padded -- a/b emits padded heads
    size_t arena_offset     = 0;
    size_t layer_out_offset = 0;
    size_t x_words          = 0; ///< total size of the arg-0 BO

    // --- arg 2 and arg 3 ----------------------------------------------------
    size_t rms_words   = 0;      ///< 2 * D
    size_t conv_w_size = 0;      ///< conv_kernel_size * conv_dim
    /// | conv taps | ssm_norm |, adjacent so one lengthened transfer on
    /// IT5 MM2S ch1 feeds both of MT[5]'s BDs.
    size_t aux_w_words = 0;

    // --- arg 4: | conv in | S in | conv out | --------------------------------
    /// The window going *in*: the last conv_kernel_size - 1 **pre-conv** qkv
    /// rows, which is what the next pass convolves its first token against.
    size_t conv_words     = 0;
    size_t s_offset       = 0;
    size_t s_words        = 0;   ///< one full recurrent state, f32 in bf16 words
    size_t conv_out_offset = 0;
    /// The window coming *out*, which is larger. At dispatch time nobody knows
    /// how many drafted tokens will survive verification, so one window is not
    /// enough: accept-j needs the last conv_kernel_size - 1 pre-conv rows as of
    /// token j. Those runs overlap, so their union is one contiguous
    /// L_TOKENS + conv_kernel_size - 2 row block and the host just indexes.
    size_t conv_out_words = 0;
    size_t kv_cache_words = 0;   ///< total size of the arg-4 BO

    /// Where accept length `j` (1 .. L_TOKENS) reads its window, in words from
    /// `conv_out_offset`. Row j of the out block is the oldest tap of the
    /// window as of token j; row 0 is the one no accept length can want.
    size_t conv_out_at(int accept_len) const {
        return (size_t)(accept_len - 1) * (size_t)conv_dim;
    }
    /// Where head `h`'s recurrent state begins, in f32 elements from s_offset.
    size_t s_at(int head) const {
        return (size_t)head * (size_t)qk_head_dim * (size_t)v_head_dim;
    }
    /// Where (head, token)'s arena block begins, in words from arena_offset.
    size_t arena_at(int head, int tok) const {
        return ((size_t)head * (size_t)L_TOKENS + (size_t)tok) * (size_t)arena_block;
    }

    /// \param g  the layer geometry, already validated
    /// \param L  query rows per pass; must match the xclbin's -DL_TOKENS
    static delta_verify_layout build(const cpu_ref::delta_net_geometry& g, int L);
};

// ==========================================================================
/// Emit the whole pass into `seq`.
///
/// \param seq     the app's sequence, cleared and rebuilt
/// \param t       layout, already validated by delta_verify_layout::build()
/// \param stages  -1 builds the whole pass; >= 0 stops after that many numbered
///                steps. Truncation is a debugging lever, not a mode: a short
///                sequence leaves the array mid-pass, so a process that runs
///                one is good for exactly one measurement afterwards. What it
///                buys is the only way to find *which* transfer a wedged array
///                stops on -- dispatch the full pass, rebuild at a prefix,
///                dispatch again, and the shortest prefix that times out names
///                the step.
void gen_delta_layer_seq(npu_sequence* seq, const delta_verify_layout& t,
                         int stages = -1);

} // namespace q38npu

#endif  // __QWEN3_8MTP_NPU_DELTA_SEQUENCE_HPP__
