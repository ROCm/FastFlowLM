// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file speculate/npu/mtp_sequence.cpp
/// \brief The MTP draft head's pass, descriptor for descriptor.
///
/// Every helper below is decoding_layer's method of the same name with the
/// member reads replaced by `t.` and `context_offset`. The comments on why a
/// descriptor is shaped the way it is are kept: they are not recoverable from
/// the numbers alone.
///
/// Every helper IS a transcription again. _receive_kv_cache and _move_kv_cache
/// used to address an interleaved cache where the IRON host addressed two
/// planes -- a deliberate divergence, to dodge a 20-bit stride field that capped
/// the cache at MAX_L = 2048. The bitstream removed that hop at the producer
/// (k and v now ride separate shim channels), so the divergence bought nothing
/// and the port is two planes again. See mtp_layout::MAX_L_LIMIT.
#include "speculate/npu/mtp_sequence.hpp"

#include <cassert>
#include <stdexcept>

namespace q38npu {

using namespace mtp_tiles;

namespace {

// ==========================================================================
/// The pass's key count, to all eight attention cores. One word each.
///
/// T = 1, so there is no arithmetic sequence to compress: the drafted token
/// attends to context_offset + 1 keys and that single number is both the mask
/// bound and the chunk count ceil(keys / L_CHUNK) the cores walk. It has to
/// equal the count _move_kv_cache() sends, or the surplus chunks stay queued in
/// the memtile and the *next* dispatch consumes them in place of its own.
///
/// That failure is the one the base attention layer hit on 2026-09-17, and the
/// draft loop is exactly the workload that exposes it: seven consecutive
/// dispatches at consecutive cursors, one per step.
void _write_attn_L(npu_sequence* seq, const mtp_layout& t, int context_offset,
                   const mtp_rtp_addresses& rtp) {
    const uint32_t keys = (uint32_t)(context_offset + t.L_TOKENS);
    const int cus = t.num_attn_columns * t.cus_per_column;
    for (int i = 0; i < cus; i++) {
        seq->rtp_write(attn_qk_tiles[i], rtp.qk, keys);
        seq->rtp_write(attn_kv_tiles[i], rtp.kv, keys);
    }
}

/// The two pre-fc norm gains to the passthrough tile, embedding then hidden.
///
/// Two descriptors and not one: the tile holds them in separate w buffers and
/// acquires each once per pass, so a single 2*D stream would leave the second
/// half sitting in the channel.
void _send_fc_norm_weights(npu_sequence* seq, const mtp_layout& t) {
    for (int half = 0; half < 2; half++) {
        seq->npu_dma_memcpy_nd(
            sizeof(bf16),
            rms_arg_id,
            MM2S,
            pass_tile,
            half == 0 ? bd_1 : bd_2,
            it_channel_1,
            {0, 0, 0, (uint32_t)((size_t)half * t.D)},
            {1, 1, 1, (uint32_t)t.D},
            {0, 0, 0, 1},
            -1, 0, false, aggressive_cache
        );
    }
}

/// The fusion's two halves, **raw**: e = the token embedding, then h = the
/// previous hidden state.
///
/// The concat exists only on the host. The array sees two D-wide packets on one
/// channel and **the order in which they arrive is the concatenation** -- so
/// swapping these two descriptors is a silent wrong answer, not a failure.
/// (cpu_ref::mtp_head builds cat([norm_embed, norm_hidden]) the same way round.)
///
/// Raw because the passthrough tile applies pre_fc_norm_embedding and
/// pre_fc_norm_hidden on the way past; pre-norming on the host norms twice.
///
/// **pkt_id is -1: this link is circuit-switched, not packet-switched.**
/// IT[3].MM2S[0] goes to exactly one place -- pass_core.X_in -- so there is
/// nothing for an id to select between, and tagging the stream `to_pass`
/// routed it at a switch that was never told about the id. The first dispatch
/// survived it; every one after it timed out. Both ends had to move together:
/// iron/MTP_layer_aie.py drops the `pkt_id =` from that connect_port.
void _send_fc_input(npu_sequence* seq, const mtp_layout& t) {
    const npu_bd_id bds[2] = {bd_0, bd_3};
    for (int half = 0; half < 2; half++) {
        seq->npu_dma_memcpy_nd(
            sizeof(bf16),
            x_arg_id,
            MM2S,
            pass_tile,
            bds[half],
            it_channel_0,
            {0, 0, 0, (uint32_t)((size_t)half * t.D)},
            {1, 1, 1, (uint32_t)t.D},
            {0, 0, 0, 1},
            -1, 0, false, aggressive_cache
        );
    }
}

/// The layer's two layernorm gains as one stream: input_layernorm, then
/// post_attention_layernorm overwriting it in the rms tile's single w buffer.
///
/// Offset 2 * D, because the first two D-blocks of arg 2 belong to the fusion.
/// The kernel acquires the weight lock once per *norm*, outside each row loop,
/// so the self-chained w BD fires exactly twice per invocation -- this length
/// has to track the live phase count, or the surplus sits in the channel
/// unconsumed and desynchronises the next pass.
void _send_rms_weights(npu_sequence* seq, const mtp_layout& t) {
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        rms_arg_id,
        MM2S,
        xyr_tile,
        bd_1,
        it_channel_1,
        {0, 0, 0, (uint32_t)(2 * (size_t)t.D)},
        {1, 1, 1, (uint32_t)(2 * t.D)},
        {0, 0, 0, 1},
        rms_w_pkt_id, 0, false, aggressive_cache
    );
}

/// The RoPE weight block: the cos|sin pair for this position, then q_norm and
/// k_norm.
///
/// Once per invocation, not once per token -- T = 1, so those coincide, but the
/// kernel still acquires it outside its token loop. `no_cache`, because the
/// cos/sin half is rewritten between dispatches: the draft loop advances the
/// position every step.
///
/// Waited on immediately. The rotation is the first thing downstream of the
/// projections that the host can change between dispatches, and the wait is
/// what keeps a dispatch from rotating with its predecessor's angle.
void _send_rope_weights(npu_sequence* seq, const mtp_layout& t) {
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        rope_arg_id,
        MM2S,
        xyr_tile,
        bd_2,
        it_channel_0,
        {0, 0, 0, 0},
        {1, 1, 1, (uint32_t)t.rope_words},
        {0, 0, 0, 1},
        rope_w_pkt_id, 0, true, no_cache
    );
    seq->npu_dma_wait(xyr_tile, MM2S, it_channel_0);
}

/// The drafted token's k and v into the DDR cache, **two** descriptors per
/// column.
///
/// The RoPE tile stamps k and v with different packet ids and they land on
/// different S2MM channels of the column's shim (iron/RoPE.py for the stamping,
/// iron/MTP_layer_aie.py for the routing), so each plane gets its own BD:
///
///     k(t) at t * DK        v(t) at plane_rows * DK + t * DV
///
/// Only one token is appended per pass -- T = 1 is the whole point of this
/// design -- so each descriptor is a single contiguous run of the column's two
/// heads. Neither carries a stride at all.
///
/// **This is what removed the MAX_L ceiling, and what retired the port's
/// interleave.** k and v used to arrive together on one channel, which forced
/// one BD to describe both planes: the column's heads on one axis and "k or v"
/// on another, whose stride was the whole k plane. A shim BD's stride field is
/// 20 bits and bf16 strides are halved on lowering, so MAX_L capped at 2048 --
/// and it did not fault. The port dodged that by interleaving the row; the
/// producer split removes the hop outright, so the two sides are one layout
/// again. See mtp_layout::MAX_L_LIMIT.
///
/// It is on the attention shims and not IT3 because npu_dma_wait only orders
/// transfers on one shim tile. The replay reads from these same shims, so this
/// is what lets the sequence wait for the append before replaying -- and it now
/// has to wait on BOTH channels.
void _receive_kv_cache(npu_sequence* seq, const mtp_layout& t, int context_offset) {
    const uint32_t width = (uint32_t)(t.kv_heads_per_column * t.DH);
    assert(t.v_at(context_offset) + (size_t)t.DV <= t.kv_cache_size &&
           "cache slot is past the end of the cache");
    assert(width <= 1023 * 2);   // 10-bit dim, doubled for bf16

    for (int c = 0; c < t.num_attn_columns; c++) {
        const size_t head_off = (size_t)c * width;
        // k on channel 0, v on channel 1 -- the same two channels the ids are
        // routed to. Each plane's base is a full-width byte offset patched into
        // the descriptor, not a BD field, so it does not overflow.
        seq->npu_dma_memcpy_nd(
            sizeof(bf16),
            kv_cache_arg_id,
            S2MM,
            attn_col_tiles[c],
            bd_11,
            it_channel_0,
            {0, 0, 0, (uint32_t)(t.k_at(context_offset) + head_off)},
            {1, 1, 1, width},
            {0, 0, 0, 1},
            -1, 0, true
        );
        seq->npu_dma_memcpy_nd(
            sizeof(bf16),
            kv_cache_arg_id,
            S2MM,
            attn_col_tiles[c],
            bd_12,
            it_channel_1,
            {0, 0, 0, (uint32_t)(t.v_at(context_offset) + head_off)},
            {1, 1, 1, width},
            {0, 0, 0, 1},
            -1, 0, true
        );
    }
}

/// The whole context out of the DDR cache and into the two attention columns,
/// in chunks of L_CHUNK keys.
///
/// Each column takes the two kv heads it owns, and its memtile splits those
/// again per CU. The cache is token-major over all four heads, so a column's
/// slice is a strided walk -- kv_heads_per_column * DH out of every row -- with
/// column c starting at c * kv_heads_per_column * DH.
///
/// **No iteration-dimension repeat here, unlike the base attention layer's
/// replay.** That one re-streams the context once per query row and puts the
/// repeat in size[0]; T = 1 leaves nothing to repeat, so dim0 is 1 and the
/// chunk walk is the whole descriptor.
///
///   dim1 (chunks)   successive L_CHUNKs of context, stride L_CHUNK * row stride
///   dim2 (L_CHUNK)  one chunk's rows, stride one row
///   dim3 (width)    the column's two heads, contiguous
///
/// Each plane is walked on its own descriptor, so the row stride is just DK (or
/// DV) and nothing here depends on MAX_L. Only the base offsets differ --
/// k_at(0) is 0 and v_at(0) is a whole k plane in. The append is symmetric now:
/// it too puts k and v on separate channels, so no descriptor anywhere carries
/// a MAX_L-scaled stride. See _receive_kv_cache.
///
/// **What does bound the context is right here.** dim1 counts whole L_CHUNKs on
/// a 10-bit field, so one pass replays at most CHUNKS_LIMIT chunks -- that is
/// mtp_layout::MAX_L_LIMIT, and an oversized dim1 masks exactly as quietly as
/// the old stride did.
void _move_kv_cache(npu_sequence* seq, const mtp_layout& t, int context_offset) {
    const int      L_CHUNK = mtp_layout::L_CHUNK;
    const uint32_t width   = (uint32_t)(t.kv_heads_per_column * t.DH);
    const uint32_t keys    = (uint32_t)(context_offset + t.L_TOKENS);
    const uint32_t rows    = (keys + L_CHUNK - 1) / L_CHUNK * L_CHUNK;
    const uint32_t chunks  = rows / (uint32_t)L_CHUNK;
    const uint32_t k_row_stride   = (uint32_t)t.k_row_stride();
    const uint32_t v_row_stride   = (uint32_t)t.v_row_stride();
    const uint32_t k_chunk_stride = (uint32_t)(L_CHUNK * t.k_row_stride());
    const uint32_t v_chunk_stride = (uint32_t)(L_CHUNK * t.v_row_stride());
    // The backstop on the caller's cursor guard. `rows` rounds up to an
    // L_CHUNK and there is no apron above MAX_L, so this holds only because
    // context_offset + 1 <= MAX_L and MAX_L is a whole number of chunks --
    // the round-up lands ON the top of the buffer, never past it.
    assert(rows <= (uint32_t)t.plane_rows());
    assert(width <= 1023 * 2);                            // 10-bit dims, doubled for bf16
    assert(chunks <= (uint32_t)mtp_layout::CHUNKS_LIMIT); // dim1 is 10 bits

    for (int c = 0; c < t.num_attn_columns; c++) {
        const uint32_t head_off = (uint32_t)(c * (int)width);
        seq->npu_dma_memcpy_nd(
            sizeof(bf16),
            kv_cache_arg_id,
            MM2S,
            attn_col_tiles[c],
            bd_0,
            it_channel_0,
            {0, 0, 0, (uint32_t)t.k_at(0) + head_off},
            {1, chunks,         (uint32_t)L_CHUNK, width},
            {0, k_chunk_stride, k_row_stride,      1},
            // no_cache: these are rows the append wrote earlier in this same
            // sequence, so a cached read can serve the pre-append contents.
            -1, 0, false, no_cache
        );
        seq->npu_dma_memcpy_nd(
            sizeof(bf16),
            kv_cache_arg_id,
            MM2S,
            attn_col_tiles[c],
            bd_1,
            it_channel_1,
            {0, 0, 0, (uint32_t)t.v_at(0) + head_off},
            {1, chunks,         (uint32_t)L_CHUNK, width},
            {0, v_chunk_stride, v_row_stride,      1},
            -1, 0, false, no_cache
        );
    }
}

/// The layer's output, h2 = h + W_down @ SwiGLU(...), D wide.
///
/// D = 5120 split across two axes of one descriptor because a 10-bit dimension
/// tops out at 2048 once bf16 sizes are halved on lowering. Never two BDs.
///
/// No intermediate stage is separately observable: the post-attention norm, the
/// SwiGLU row and down each became the next stage's x, and no producer's single
/// MM2S had room for a second destination. This row covers all of them.
void _receive_layer_out(npu_sequence* seq, const mtp_layout& t) {
    const uint32_t outer = 4;
    const uint32_t inner = (uint32_t)t.D / outer;
    assert(t.D % (int)outer == 0 && inner <= 2048 &&
           "a BD dimension is 10 bits, and bf16 sizes are halved");
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        x_arg_id,
        S2MM,
        layer_out_tap_tile,
        bd_10,
        it_channel_1,
        {0, 0, 0, (uint32_t)t.layer_out_offset},
        {1, 1, outer, inner},
        {0, 0, inner, 1},
        -1, 0, true
    );
}

/// One projection round set: `Dout x Din` weights, streamed to the four mvm
/// columns two row-blocks at a time.
///
/// \param w_offset **in bf16 elements, not bytes.** The descriptor's element
///        size is sizeof(bf16) and a_block_size is likewise divided down, so
///        the offset has to be in the same unit as everything else in this
///        arithmetic. Passing a byte offset addresses twice as far into the
///        buffer and streams whatever is past the end -- which the cores
///        dequantise into NaN rather than rejecting. mtp_layout's w_*_words()
///        accessors are the only sanctioned source.
void _move_weights(npu_sequence* seq, size_t Dout, size_t Din, size_t w_offset) {
    const int      m = 32;
    const int      k = 256;
    const uint32_t columns = 4;
    const uint32_t elems = m * k;
    const uint32_t a_block_size = q4k_bytes(elems) / sizeof(bf16);   // W_BLOCK_SIZE
    const uint32_t blocks_per_row = Din / k;
    const uint32_t cores = columns * 4;
    assert(Dout / m / cores > 0);
    assert(Dout % (m * cores) == 0);
    const size_t rounds = Dout / m / cores;
    for (size_t round = 0; round < rounds; round++) {
        const uint32_t bd_offset = (round % 2) * 8;
        for (uint32_t col = 0; col < columns; col++) {
            seq->npu_dma_memcpy_nd(
                sizeof(bf16),
                proj_arg_id,
                MM2S,
                mvm_tiles[col],
                npu_bd_id(bd_1 + bd_offset),
                it_channel_0,
                {0, 0, 0, (uint32_t)((round * cores + col * 4) * a_block_size * blocks_per_row + (uint32_t)w_offset)},
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
                {0, 0, 0, (uint32_t)((round * cores + col * 4 + 2) * a_block_size * blocks_per_row + (uint32_t)w_offset)},
                {1, 1, 1, 2 * blocks_per_row * a_block_size},
                {0, 0, 0, 1},
                -1, 0, true, aggressive_cache
            );
        }
        if (round > 0) {
            for (uint32_t col = 0; col < columns; col++) {
                seq->npu_dma_wait(mvm_tiles[col], MM2S, it_channel_0);
                seq->npu_dma_wait(mvm_tiles[col], MM2S, it_channel_1);
            }
        }
    }
    for (uint32_t col = 0; col < columns; col++) {
        seq->npu_dma_wait(mvm_tiles[col], MM2S, it_channel_0);
        seq->npu_dma_wait(mvm_tiles[col], MM2S, it_channel_1);
    }
}

} // namespace

// ==========================================================================
void gen_mtp_layer_seq(npu_sequence* seq, const mtp_layout& t,
                       int context_offset, const mtp_rtp_addresses& rtp) {
    // The whole pass has to fit, not just its first row -- there is no apron
    // above MAX_L. T = 1 here, so `< MAX_L` IS `context_offset + L_TOKENS <=
    // MAX_L`; the verify design states the same condition with its 8.
    if (context_offset < 0 || context_offset + mtp_layout::L_TOKENS > t.MAX_L)
        throw std::runtime_error(
            "gen_mtp_layer_seq: a pass at context_offset " +
            std::to_string(context_offset) + " appends through " +
            std::to_string(context_offset + mtp_layout::L_TOKENS) +
            ", outside a cache of MAX_L " + std::to_string(t.MAX_L));
    if (rtp.qk == 0 || rtp.kv == 0)
        throw std::runtime_error("gen_mtp_layer_seq: RTP addresses are unset");

    seq->clear_cmds();

    _write_attn_L(seq, t, context_offset, rtp);

    // --- phase 1: the fusion, 2 * D -> D -----------------------------------
    _send_fc_norm_weights(seq, t);
    _send_fc_input(seq, t);

    // --- phase 2: input_layernorm, then k|v|q ------------------------------
    _send_rms_weights(seq, t);

    // --- phase 3: q/k norm and the rotation --------------------------------
    _send_rope_weights(seq, t);

    _receive_kv_cache(seq, t, context_offset);
    _receive_layer_out(seq, t);

    _move_weights(seq, (size_t)t.D,       (size_t)t.FC_IN, t.w_fc_words());
    _move_weights(seq, (size_t)t.qkv_out, (size_t)t.D,     t.w_qkv_words());
    _move_weights(seq, (size_t)t.DQ,      (size_t)t.D,     t.w_gate_words());

    // --- phase 4: attention -------------------------------------------------
    // The append, waited on per column. This only orders the replay because
    // both are on these same two shims -- npu_dma_wait does not reach across
    // shim tiles. Everything after this point may assume the cache holds
    // context_offset + 1 valid rows.
    //
    // The append is two transfers per column, k on channel 0 and v on channel
    // 1, so both have to be drained. A wait is per shim *and* per channel:
    // waiting only on channel 0 would let the replay start with v still in
    // flight, and the damage would be a plausible-looking output rather than a
    // hang -- and, since the next pass replays the row, drift across dispatches
    // rather than one wrong answer.
    for (int c = 0; c < t.num_attn_columns; c++) {
        seq->npu_dma_wait(attn_col_tiles[c], S2MM, it_channel_0);   // k
        seq->npu_dma_wait(attn_col_tiles[c], S2MM, it_channel_1);   // v
    }

    _move_kv_cache(seq, t, context_offset);

    // --- phase 5: o_proj ----------------------------------------------------
    // Dout is D and Din is DQ: K is the gated row here. Not folded in with the
    // three rounds above -- _move_weights waits on its own transfers, and the
    // cores cannot touch o's weights until the gate has filled o's replay
    // phase, which needs the attention output, which needs the replay.
    _move_weights(seq, (size_t)t.D, (size_t)t.DQ, t.w_o_words());

    // --- phase 6: the MLP's up | gate ---------------------------------------
    _move_weights(seq, 2 * (size_t)t.intermediate_size, (size_t)t.D,
                  t.w_upgate_words());

    // --- phase 7: down, then the MLP residual -------------------------------
    _move_weights(seq, (size_t)t.D, (size_t)t.intermediate_size, t.w_down_words());

    seq->npu_dma_wait(layer_out_tap_tile, S2MM, it_channel_1);

    seq->cmds2seq();
}

} // namespace q38npu
