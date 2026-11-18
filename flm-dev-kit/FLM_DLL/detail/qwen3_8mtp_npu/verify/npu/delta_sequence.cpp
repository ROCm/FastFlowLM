// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file verify/npu/delta_sequence.cpp
/// \brief The whole gated-delta-net pass, descriptor for descriptor.
///
/// Every helper below is `linear_decoding_layer`'s method of the same name with
/// the member reads replaced by `t.`. The comments on why a descriptor is
/// shaped the way it is are kept: they are not recoverable from the numbers.
#include "verify/npu/delta_sequence.hpp"

#include <cassert>
#include <stdexcept>
#include <string>

namespace q38npu {

using namespace delta_tiles;

// ==========================================================================
// geometry
// ==========================================================================
delta_verify_layout delta_verify_layout::build(const cpu_ref::delta_net_geometry& g,
                                               int L) {
    delta_verify_layout t;
    t.w = delta_layout::build(g);

    t.L_TOKENS          = L;
    t.D                 = g.model_dim;
    t.intermediate_size = g.intermediate_size;
    t.DQ                = g.dq();
    t.DK                = g.dk();
    t.DV                = g.dv();
    t.conv_dim          = g.conv_dim();
    t.conv_kernel_size  = g.conv_k;
    t.qk_heads          = g.qk_heads;
    t.v_heads           = g.v_heads;
    t.v_heads_padded    = t.w.v_heads_padded;
    t.qk_head_dim       = g.k_head_dim;
    t.v_head_dim        = g.v_head_dim;

    t.qkv_out = t.conv_dim;
    t.z_out   = t.DV;
    t.ab_out  = 2 * t.v_heads_padded;

    t.triplet_stride = t.v_head_dim + 2;
    t.arena_block    = 3 * t.triplet_stride;
    t.arena_heads    = t.v_heads_padded;

    t.arena_offset     = (size_t)L * (size_t)t.D;
    t.layer_out_offset = t.arena_offset +
                         (size_t)t.arena_heads * (size_t)L * (size_t)t.arena_block;
    t.x_words          = t.layer_out_offset + (size_t)L * (size_t)t.D;

    t.rms_words   = 2 * (size_t)t.D;
    t.conv_w_size = (size_t)t.conv_kernel_size * (size_t)t.conv_dim;
    t.aux_w_words = t.conv_w_size + (size_t)t.v_head_dim;

    // f32 counted in bf16 words, hence the * 2. The conv window is bf16
    // pre-conv rows, so it is not doubled.
    t.conv_words      = (size_t)(t.conv_kernel_size - 1) * (size_t)t.conv_dim;
    t.s_offset        = t.conv_words;
    t.s_words         = (size_t)t.v_heads * (size_t)t.v_head_dim *
                        (size_t)t.qk_head_dim * 2;
    t.conv_out_offset = t.s_offset + t.s_words;
    t.conv_out_words  = (size_t)(L + t.conv_kernel_size - 2) * (size_t)t.conv_dim;
    t.kv_cache_words  = t.conv_out_offset + t.conv_out_words;

    // --- the shapes the descriptors below cannot express ---------------------
    auto bad = [](const std::string& m) {
        throw std::runtime_error("delta_verify_layout: " + m);
    };
    // _receive_layer_out()'s iter_size field is 6 bits, and it splits a row
    // into whole 1024-word runs because a BD dimension caps at 1024 after the
    // bf16 fold.
    if (L < 1 || L >= 64)        bad("L_TOKENS must be 1 .. 63");
    if (t.D % 1024 != 0)         bad("D is not a whole number of 1024-word runs");
    // Every arena offset reaches a descriptor, which folds bf16 to u32 and
    // halves it; an odd one silently truncates.
    if (t.arena_offset % 2 || t.triplet_stride % 2 || t.arena_block % 2)
        bad("an arena offset or stride is not an even bf16 word");
    // `triplet_stride` is v_head_dim + 2, and alpha's hole sits at qk_head_dim
    // words into the k sub-block. The two agree only when the head dims do, and
    // if they ever stop agreeing the readers find alpha inside k rather than
    // failing -- so say so here instead.
    if (t.qk_head_dim != t.v_head_dim)
        bad("the triplet arena assumes qk_head_dim == v_head_dim");
    // _move_weights() hands 32 output rows to each of 16 cores per round.
    for (int dout : {t.qkv_out, t.z_out, t.D, 2 * t.intermediate_size})
        if (dout % (32 * 16) != 0)
            bad("a projection's output width is not a whole 512-row round");
    // _move_alpha_beta_weights() walks whole 32x256 q8_0 blocks.
    if (t.ab_out % 32 != 0 || t.D % 256 != 0)
        bad("the a/b matrix is not a whole number of 32x256 blocks");
    return t;
}

namespace {

// ==========================================================================
// arg 0 -- hidden states in, arena both ways, h2 out
// ==========================================================================
/// All L_TOKENS rows as one contiguous transfer. The rms tile's S2MM chain
/// splits it back into D-element rows across its ping/pong buffers, so the shim
/// does not need to know about the row boundaries.
void _send_hidden_states(npu_sequence* seq, const delta_verify_layout& t) {
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        x_arg_id,
        MM2S,
        xyr_tile,
        bd_0,
        it_channel_0,
        {0, 0, 0, 0},
        {1, 1, 1, (uint32_t)((size_t)t.L_TOKENS * t.D)},
        {0, 0, 0, 1},
        0, 0, false, aggressive_cache
    );
}

/// Both norms, input_layernorm then post_attention_layernorm, as one transfer.
///
/// The rms tile's weight BD is self-chained on a single w_buffer and the kernel
/// acquires the weight lock once per phase, so the second D words simply
/// overwrite the first between the two phases -- which is the only reason a
/// tile with four banks can do two different norms.
void _send_rms_weights(npu_sequence* seq, const delta_verify_layout& t) {
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        rms_arg_id,
        MM2S,
        xyr_tile,
        bd_1,
        it_channel_1,
        {0, 0, 0, 0},
        {1, 1, 1, (uint32_t)t.rms_words},
        {0, 0, 0, 1},
        0, 0, false, aggressive_cache
    );
}

/// Scatter the a / b tile's output into the arena's alpha and beta holes.
///
/// The tile emits, per token, v_heads_padded alpha floats then as many beta
/// floats; only the BD's access pattern makes an arena out of them. That is why
/// the L=1 design's a/b -> conv handoff could go: the scalars meet k and v in
/// DDR instead of on the array, and the shim's address generator does it free.
///
///   iter (L)    token       stride arena_block     -- one token per replay
///   dim2 (2)    alpha|beta  stride triplet_stride  -- alpha one sub-block
///                                                     before beta
///   dim1 (heads) head       stride L*arena_block
///   dim0 (2)    the two bf16 words of one f32
///
/// starting at +v_head_dim, i.e. right after k in each block.
void _receive_ab(npu_sequence* seq, const delta_verify_layout& t) {
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        x_arg_id,
        S2MM,
        conv_tile,
        bd_11,
        it_channel_1,
        {0, 0, 0, (uint32_t)(t.arena_offset + t.v_head_dim)},
        {(uint32_t)t.L_TOKENS, 2, (uint32_t)t.arena_heads, 2},
        {(uint32_t)t.arena_block, (uint32_t)t.triplet_stride,
         (uint32_t)((size_t)t.L_TOKENS * t.arena_block), 1},
        -1, 0, true
    );
}

/// Scatter the conv tile's packed triplets into the arena.
///
/// The tile emits, per token, v_heads dense 3*v_head_dim-word (k | v | q)
/// packets in head order; this spreads each packet's three runs across the
/// block's three triplet_stride-word sub-blocks, leaving the alpha and beta
/// holes for _receive_ab() to fill. **v_heads, not v_heads_padded**: the conv
/// tile emits real heads only, and the pad slots carry alpha and beta alone.
///
///   iter (L)          token           stride arena_block
///   dim2 (v heads)    head            stride L*arena_block
///   dim1 (3)          k | v | q       stride triplet_stride
///   dim0 (head dim)   channel         stride 1
///
/// IT[6] only ever sends q4_k weights, so both of its S2MM channels are idle.
void _receive_arena(npu_sequence* seq, const delta_verify_layout& t) {
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        x_arg_id,
        S2MM,
        arena_in_tile,
        bd_4,
        it_channel_0,
        {0, 0, 0, (uint32_t)t.arena_offset},
        {(uint32_t)t.L_TOKENS, (uint32_t)t.v_heads, 3, (uint32_t)t.v_head_dim},
        {(uint32_t)t.arena_block, (uint32_t)((size_t)t.L_TOKENS * t.arena_block),
         (uint32_t)t.triplet_stride, 1},
        -1, 0, true
    );
}

/// Broadcast the finished arena to the four delta cores.
///
/// Linear, because the arena is head-major and the cores want it head-major:
/// heads 0 .. v_heads are one contiguous run, and inside it every
/// triplet_stride-word sub-block is exactly one of the cores' k, v or q BDs.
/// The padded head slots above v_heads carry only alpha and beta and are simply
/// not read.
void _send_arena_to_delta(npu_sequence* seq, const delta_verify_layout& t) {
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        x_arg_id,
        MM2S,
        state_tile,
        bd_4,
        it_channel_1,
        {0, 0, 0, (uint32_t)t.arena_offset},
        {1, 1, 1, (uint32_t)((size_t)t.v_heads * t.L_TOKENS * t.arena_block)},
        {0, 0, 0, 1},
        -1, 0, false, no_cache
    );
}

/// h2 = h1 + mlp.down_proj(hid), the layer's output and the next layer's input.
///
/// Flat: this comes off the rms tile, one packet per token in token order, not
/// off the engine's y tree, so there is no row-block permute to undo -- MT[3]
/// already did that for down's y on its way in. Split into 1024-word runs only
/// because a BD dimension caps at 1024 after the bf16 fold.
///
/// Its own channel rather than a second BD behind the norm's flow, which MT[2]
/// takes as up|gate's x. The rms tile stamps a distinct packet id per phase, so
/// the router keeps the two apart; see rms_pkt_id in iron/constants.py.
void _receive_layer_out(npu_sequence* seq, const delta_verify_layout& t) {
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        x_arg_id,
        S2MM,
        xyr_tile,
        bd_9,
        it_channel_1,
        {0, 0, 0, (uint32_t)t.layer_out_offset},
        {(uint32_t)t.L_TOKENS, 1, (uint32_t)(t.D / 1024), 1024},
        {(uint32_t)t.D, 0, 1024, 1},
        -1, 0, true
    );
}

// ==========================================================================
// arg 3 and arg 4 -- the small weights and the two carried states
// ==========================================================================
/// The conv1d taps and the gated norm's weight, in one transfer.
///
/// First on IT5 MM2S ch1: the memtile's BD chain there is conv taps, then
/// ssm_norm, then the conv cache, cycling. Because the aux BO stores conv_w and
/// ssm_norm adjacently, the two memtile BDs can be fed by a single contiguous
/// send -- no second shim channel and no second helper. MT[5] then replays the
/// ssm_norm words to the norm tile once per token, which is what that kernel
/// expects (it re-acquires its weight lock every invocation).
void _send_conv_weights(npu_sequence* seq, const delta_verify_layout& t) {
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        aux_w_arg_id,
        MM2S,
        conv_tile,
        bd_2,
        it_channel_1,
        {0, 0, 0, 0},
        {1, 1, 1, (uint32_t)t.aux_w_words},
        {0, 0, 0, 1},
        -1, 0, true, no_cache
    );
    seq->npu_dma_wait(conv_tile, MM2S, it_channel_1);
}

/// The incoming conv window, behind the taps on the same channel.
///
/// Second on IT5 MM2S ch1 and that order is the memtile's BD chain, not a
/// preference: both transfers block on their own DMA wait, so swapping them
/// feeds the cache to the tap BD.
void _move_conv_cache(npu_sequence* seq, const delta_verify_layout& t) {
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        kv_cache_arg_id,
        MM2S,
        conv_tile,
        bd_2,
        it_channel_1,
        {0, 0, 0, 0},
        {1, 1, 1, (uint32_t)t.conv_words},
        {0, 0, 0, 1},
        -1, 0, true, no_cache
    );
    seq->npu_dma_wait(conv_tile, MM2S, it_channel_1);
}

/// Every accept length's conv window for the next pass, off MT[5]'s MM2S ch4.
///
/// Pre-conv qkv rows 1 .. L + CONV_SIZE - 2 of the tile's staging buffer, one
/// contiguous run. Each accept length's window is a CONV_SIZE - 1 row slice of
/// it, in the same layout _move_conv_cache() sends the cache back in with;
/// `conv_out_at()` does the indexing.
///
/// IT5 S2MM ch0 is the one free channel on this shim: ch1 is the a/b scatter
/// into the arena. bd_14 because BD ids are a single per-shim pool across both
/// directions, and IT5 already spends 2, 11 and 13 on the MM2S side.
///
/// Posted with the other reads, before any weights move -- the memtile will not
/// release the window until the L token rows are in, and a read issued late
/// would stall the tile on a full channel.
void _receive_conv_out(npu_sequence* seq, const delta_verify_layout& t) {
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        kv_cache_arg_id,
        S2MM,
        conv_tile,
        bd_14,
        it_channel_0,
        {0, 0, 0, (uint32_t)t.conv_out_offset},
        {1, 1, 1, (uint32_t)t.conv_out_words},
        {0, 0, 0, 1},
        -1, 0, true, no_cache
    );
}

/// The recurrent state in, one linear run. There is no run back.
///
/// DDR holds it [head][dk][dv] f32, which is both MT[4]'s per-head block layout
/// and cpu_ref's, so neither side needs an access pattern.
///
/// No token is issued and nothing waits on this leg: the barrier that a
/// write-back would have given is the h2 wait at the end of the sequence.
void _move_s(npu_sequence* seq, const delta_verify_layout& t) {
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        kv_cache_arg_id,
        MM2S,
        state_tile,
        bd_2,
        it_channel_0,
        {0, 0, 0, (uint32_t)t.s_offset},
        {1, 1, 1, (uint32_t)t.s_words},
        {0, 0, 0, 1},
        -1, 0, false, no_cache
    );
}

// ==========================================================================
// arg 1 -- the weight rounds
// ==========================================================================
/// One projection's whole weight stream, 512 output rows per round.
///
/// `w_words` is a **bf16 word** offset, not a byte offset: this descriptor's
/// element size is sizeof(bf16) and `a_block_size` is divided down to match.
/// A byte offset here reads past the end of the BO and the cores dequantise it
/// into NaN. (memory: move-weights-offset-is-in-elements)
void _move_weights(npu_sequence* seq, const delta_verify_layout& t,
                   size_t Dout, size_t Din, size_t w_words) {
    const int      m              = 32;
    const int      k              = 256;
    const uint32_t columns        = 4;
    const uint32_t a_block_size   = (uint32_t)(q4k_bytes(m * k) / sizeof(bf16));
    const uint32_t blocks_per_row = (uint32_t)(Din / k);
    const uint32_t cores          = columns * 4;
    assert(Dout % (m * cores) == 0 && Dout / m / cores > 0);
    const size_t rounds = Dout / m / cores;
    (void)t;

    for (size_t round = 0; round < rounds; round++) {
        const uint32_t bd_offset = (uint32_t)(round % 2) * 8;
        for (uint32_t col = 0; col < columns; col++) {
            seq->npu_dma_memcpy_nd(
                sizeof(bf16),
                proj_arg_id,
                MM2S,
                mvm_tiles[col],
                npu_bd_id(bd_1 + bd_offset),
                it_channel_0,
                {0, 0, 0, (uint32_t)((round * cores + col * 4) * a_block_size *
                                         blocks_per_row + w_words)},
                {1, 1, 1, 2 * blocks_per_row * a_block_size},
                {0, 0, 0, 1},
                -1, 0, true, aggressive_cache
            );
            seq->npu_dma_memcpy_nd(
                sizeof(bf16),
                proj_arg_id,
                MM2S,
                mvm_tiles[col],
                npu_bd_id(bd_2 + bd_offset),
                it_channel_1,
                {0, 0, 0, (uint32_t)((round * cores + col * 4 + 2) * a_block_size *
                                         blocks_per_row + w_words)},
                {1, 1, 1, 2 * blocks_per_row * a_block_size},
                {0, 0, 0, 1},
                -1, 0, true, aggressive_cache
            );
        }
        // Round n's pair of BDs is round n-2's, so the wait for round n-1
        // is what makes reusing the id safe. Round 0 has nothing to wait on.
        if (round > 0)
            for (uint32_t col = 0; col < columns; col++) {
                seq->npu_dma_wait(mvm_tiles[col], MM2S, it_channel_0);
                seq->npu_dma_wait(mvm_tiles[col], MM2S, it_channel_1);
            }
    }
    for (uint32_t col = 0; col < columns; col++) {
        seq->npu_dma_wait(mvm_tiles[col], MM2S, it_channel_0);
        seq->npu_dma_wait(mvm_tiles[col], MM2S, it_channel_1);
    }
}

/// One token's whole a/b weight set, bias block first.
///
/// The offset is `w_ab_bias_words()`, not `w_ab_words()`: the leading block
/// carries the activation bias and the tile takes it through the same ping/pong
/// BDs as everything else. That is what makes the design re-dispatchable -- the
/// chain on this channel has no prologue BD, so every pass puts exactly
/// 1 + total_blocks blocks on it and the ping/pong parity never drifts. A
/// sequence that sent the bias once, ahead of the loop, worked for exactly one
/// dispatch per xclbin load.
void _move_alpha_beta_weights(npu_sequence* seq, const delta_verify_layout& t) {
    const int      m            = 32;
    const int      k            = 256;
    const uint32_t block_size   = (uint32_t)(q80_bytes(m * k) / sizeof(bf16));
    const uint32_t total_blocks = 1 + (uint32_t)(((size_t)t.ab_out / m) *
                                                 ((size_t)t.D / k));
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        proj_arg_id,
        MM2S,
        conv_tile,
        npu_bd_id(bd_13),
        it_channel_0,
        {0, 0, 0, (uint32_t)t.w.w_ab_bias_words()},
        {1, 1, 1, block_size * total_blocks},
        {0, 0, 0, 1},
        -1, 0, true, aggressive_cache
    );
    seq->npu_dma_wait(conv_tile, MM2S, it_channel_0);
}

} // namespace

// ==========================================================================
// the pass
// ==========================================================================
void gen_delta_layer_seq(npu_sequence* seq, const delta_verify_layout& t,
                         int stages) {
    seq->clear_cmds();

    int  step = 0;
    auto go   = [&]() { return stages < 0 || ++step <= stages; };
#define DELTA_STEP() do { if (!go()) { seq->cmds2seq(); return; } } while (0)

    // Shim channel maps:
    //
    //   IT3  MM2S ch0  x, to the rms tile (phase 1 only; o comes from MT[3])
    //        MM2S ch1  the input and post-attention layernorm weights
    //        S2MM ch1  h2 out, the end of the pass
    //   IT4  MM2S ch0  the recurrent state in (one way: nothing comes back)
    //        MM2S ch1  the arena read-back
    //   IT5  MM2S ch0  the q8_0 a/b weights, bias block first, once per token
    //        MM2S ch1  conv taps + ssm_norm, then the conv cache
    //        S2MM ch0  the outgoing conv window
    //        S2MM ch1  alpha | beta, scattered into the arena
    //   IT6  S2MM ch0  the conv tile's k|v|q triplets, into the arena
    //
    // The ordering rules are in the header. The two mid-sequence waits below
    // are the arena round trip: the read-back cannot be issued until both of
    // its writers have drained.
    DELTA_STEP(); _send_hidden_states(seq, t);
    DELTA_STEP(); _send_rms_weights(seq, t);
    DELTA_STEP(); _receive_ab(seq, t);
    DELTA_STEP(); _receive_arena(seq, t);
    DELTA_STEP(); _receive_layer_out(seq, t);
    // The conv window out. Like S, independent of the arena, so it is posted
    // with the other reads; MT[5] holds it back until the L token rows land.
    DELTA_STEP(); _receive_conv_out(seq, t);
    // S in. Independent of the arena too: MT[4] fills its state buffers
    // straight away and the cores park on their k/v/q locks until the read-back
    // starts. The cores need S_0 streamed in every pass -- it is only the
    // write-back that is gone.
    DELTA_STEP(); _move_s(seq, t);
    // Prime MT[5] before the projection engine starts pushing qkv at it.
    DELTA_STEP(); _send_conv_weights(seq, t);
    DELTA_STEP(); _move_conv_cache(seq, t);
    for (int tok = 0; tok < t.L_TOKENS; tok++) {
        DELTA_STEP(); _move_alpha_beta_weights(seq, t);
    }
    DELTA_STEP(); _move_weights(seq, t, t.qkv_out, t.D, t.w.w_qkv_words());
    DELTA_STEP(); _move_weights(seq, t, t.z_out,   t.D, t.w.w_z_words());
    // The arena is complete once the conv dump and the a/b scatter have both
    // drained. Everything that feeds them has been issued above, so neither
    // wait can deadlock.
    DELTA_STEP(); seq->npu_dma_wait(arena_in_tile, S2MM, it_channel_0);
    DELTA_STEP(); seq->npu_dma_wait(conv_tile,     S2MM, it_channel_1);
    DELTA_STEP(); _send_arena_to_delta(seq, t);
    // ssm_out_proj. Third _move_weights(), so third mvm() in proj_main.cc --
    // the cores consume the weight stream positionally. K is DV here, not D:
    // this projection reduces over the gated rows.
    DELTA_STEP(); _move_weights(seq, t, t.D, t.DV, t.w.w_out_words());
    // The conv window. There is no wait on the state coming back any more --
    // nothing writes that channel, so waiting on it would block forever. This
    // wait is the same barrier: MT[5] does not release the window until all L
    // sliding windows have been taken, which needs the same L_TOKENS steps the
    // scan does, and h2 is downstream of the scan through o anyway.
    DELTA_STEP(); seq->npu_dma_wait(conv_tile, S2MM, it_channel_0);

    // The MLP: up | gate off the post-attention norm, then down off the SwiGLU.
    // Fourth and fifth _move_weights(), so fourth and fifth mvm().
    DELTA_STEP(); _move_weights(seq, t, 2 * (size_t)t.intermediate_size, t.D,
                                t.w.w_upgate_words());
    DELTA_STEP(); _move_weights(seq, t, t.D, t.intermediate_size,
                                t.w.w_down_words());

    // h2, the end of the pass.
    DELTA_STEP(); seq->npu_dma_wait(xyr_tile, S2MM, it_channel_1);

#undef DELTA_STEP
    seq->cmds2seq();
}

} // namespace q38npu
