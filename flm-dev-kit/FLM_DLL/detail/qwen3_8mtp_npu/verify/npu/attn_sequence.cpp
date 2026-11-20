// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file verify/npu/attn_sequence.cpp
/// \brief The gated full-attention pass, descriptor for descriptor.
///
/// Every helper below is decoding_layer's method of the same name with the
/// member reads replaced by `t.` and `context_offset`. The comments on why a
/// descriptor is shaped the way it is are kept: they are not recoverable from
/// the numbers alone.
#include "verify/npu/attn_sequence.hpp"

#include <cassert>
#include <stdexcept>

namespace q38npu {

using namespace attn_tiles;

namespace {

// ==========================================================================
/// The pass's key count, to all eight attention cores. One word each.
///
/// Token t attends to context_offset + t + 1 keys: everything already in the
/// cache, plus the tokens of this pass up to and including itself. That is the
/// causal mask, in the only form the cores can see it -- but it is an
/// arithmetic sequence, so **the last token's count is the whole of it**. The
/// cores keep their own token counter and take
/// keys = L_max - (L_TOKENS - 1) + t.
///
/// The same word is the chunk count both cores walk, ceil(L_max / L_CHUNK), and
/// it has to equal the count _replay_kv_cache() sends. It did not, before
/// 2026-09-17: the kernels stopped at their own ceil(L[t] / L_CHUNK) while the
/// host sends every token the whole context, so at
/// context_offset % L_CHUNK >= L_CHUNK - L_TOKENS + 1 -- 7 cursors in 16 -- the
/// surplus chunks stayed queued in the memtile and the next token consumed them
/// in place of its own. Prefill only ever dispatches at ctx % 16 in {0, 8} and
/// was exact; speculation resumes at an arbitrary cursor and collapsed. See
/// FLM_Xclbin/Qwen3_8/gated_attention_verify/KV_CHUNK_BUG.md.
///
/// **This must be kept in step with kernels/attn_qk.cc and attn_kv.cc.** They
/// now read one int and derive everything from it, so writing the old
/// L_TOKENS-entry burst would leave them reading L[0] = context_offset + 1 and
/// walking far too few chunks.
void _write_attn_L(npu_sequence* seq, const attn_layout& t, int context_offset,
                   const attn_rtp_addresses& rtp) {
    const uint32_t l_max = static_cast<uint32_t>(context_offset + t.L_TOKENS);
    for (int c = 0; c < 4; c++) {
        seq->rtp_write(attn_qk_tiles[c], rtp.qk, l_max);
        seq->rtp_write(attn_kv_tiles[c], rtp.kv, l_max);
    }
}

/// All L_TOKENS rows as one contiguous transfer. The rms tile's S2MM chain
/// splits it back into D-element rows across its ping/pong buffers, so the shim
/// does not need to know about the row boundaries.
void _send_hidden_states(npu_sequence* seq, const attn_layout& t) {
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        x_arg_id,
        MM2S,
        xyr_tile,
        bd_0,
        it_channel_0,
        {0, 0, 0, 0},
        {1, 1, 1, (uint32_t)(t.L_TOKENS * t.D)},
        {0, 0, 0, 1},
        0, 0, false, aggressive_cache
    );
}

/// Both norm weights as one stream: input_layernorm, then
/// post_attention_layernorm overwriting it in the tile's single w buffer.
///
/// The kernel acquires the weight lock once per *norm*, outside each row loop,
/// so the self-chained w BD fires exactly twice per invocation -- this length
/// has to track the live phase count, or the surplus sits in the channel
/// unconsumed and desynchronises the next pass.
void _send_rms_weights(npu_sequence* seq, const attn_layout& t) {
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        rms_arg_id,
        MM2S,
        xyr_tile,
        bd_1,
        it_channel_1,
        {0, 0, 0, 0},
        {1, 1, 1, (uint32_t)(2 * t.D)},
        {0, 0, 0, 1},
        0, 0, false, aggressive_cache
    );
}

/// The RoPE weight block: one cos|sin pair per token, then q_norm and k_norm.
///
/// Once per invocation, not once per token -- the kernel acquires it outside
/// its token loop and indexes the pair it wants. Linear: the host already built
/// it in the order the kernel reads. `no_cache`, because the cos/sin half is
/// rewritten between passes whenever the positions change.
void _send_rope_weights(npu_sequence* seq, const attn_layout& t) {
    const uint32_t len = (uint32_t)t.rope_words;
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        rope_arg_id,
        MM2S,
        attn_tile,
        bd_3,
        it_channel_0,
        {0, 0, 0, 0},
        {1, 1, 1, len},
        {0, 0, 0, 1},
        -1, 0, false, no_cache
    );
}

/// k and v into the two-plane DDR cache, in one descriptor per column.
///
/// The RoPE tile emits k then v for each token, alternating, but stamps each
/// with its own packet id, so the two arrive on two S2MM channels of the same
/// shim rather than interleaved on one:
///
///     ch0   k(0) k(1) ... k(L-1)   -> k(t) at t * DK
///     ch1   v(0) v(1) ... v(L-1)   -> v(t) at plane_rows * DK + t * DV
///
/// Each is then a plain strided write, one row per token:
///
///   dim2 (L_TOKENS)  token   stride DK (or DV) -- advance a row of the plane
///   dim3 (width)     element stride 1
///
/// **This used to be one descriptor, and it was what capped MAX_L at 2032.**
/// Catching the interleaved stream meant describing both planes at once, with
/// an extra axis of size 2 whose stride was the whole k plane -- a 20-bit
/// field, halved for bf16, so plane_rows <= 2048 at DK = 1024. npu_instr_utils
/// only *prints* when a stride overflows before masking it, so above the cap
/// every v row landed at a wrapped address and nothing said so until a later
/// pass read it back.
///
/// The split removes the axis rather than widening it: the only strides left
/// are DK and DV, constants of the model, and the plane base is a byte offset
/// patched into the BD's address at full width. What bounds MAX_L now is the
/// replay's 10-bit chunk count; see attn_layout::MAX_L_LIMIT.
///
/// The token axis stays in dims 1..3 with the iteration dimension at 1. The
/// iteration count is *not* part of the transfer length -- npu_dma_memcpy_nd()
/// builds buffer_length from dims 1..3 only, and the hardware derives dim2_size
/// from that length -- so putting the tokens there would silently transfer one
/// token's worth and stop.
///
/// It is on the attention shims and not IT3 because npu_dma_wait only orders
/// transfers on one shim tile. The replay reads from these same shims, so this
/// is what lets the sequence wait for the append before replaying; an append on
/// IT3 would be unordered against the replay and the CUs would attend to stale
/// rows. Both channels have to be waited on now -- see gen_attn_layer_seq.
void _receive_kv_cache(npu_sequence* seq, const attn_layout& t, int context_offset) {
    const uint32_t k_row_stride = (uint32_t)t.k_row_stride();
    const uint32_t v_row_stride = (uint32_t)t.v_row_stride();
    const uint32_t width = (uint32_t)(t.kv_heads_per_column * t.DH);
    assert(t.v_at(context_offset + t.L_TOKENS - 1) + (size_t)t.DV <= t.kv_cache_size &&
           "the append walks past the end of the cache");
    assert(t.L_TOKENS <= 1023 && width <= 1023 * 2);   // 10-bit dim, doubled for bf16
    // The only strides left are one row of one plane. Both are constants of the
    // model, not of MAX_L, so there is nothing here to overflow however deep
    // the cache gets -- which is the entire point of the two channels.
    assert(k_row_stride / 2 <= (1u << 20) && v_row_stride / 2 <= (1u << 20));

    for (int c = 0; c < 2; c++) {
        const size_t head_off = (size_t)c * width;
        seq->npu_dma_memcpy_nd(
            sizeof(bf16),
            kv_cache_arg_id,
            S2MM,
            attn_col_tiles[c],
            bd_3,
            it_channel_0,
            {0, 0, 0, (uint32_t)(t.k_at(context_offset) + head_off)},
            {1, 1, (uint32_t)t.L_TOKENS, width},
            {0, 0, k_row_stride,         1},
            -1, 0, true
        );
        seq->npu_dma_memcpy_nd(
            sizeof(bf16),
            kv_cache_arg_id,
            S2MM,
            attn_col_tiles[c],
            bd_4,
            it_channel_1,
            {0, 0, 0, (uint32_t)(t.v_at(context_offset) + head_off)},
            {1, 1, (uint32_t)t.L_TOKENS, width},
            {0, 0, v_row_stride,         1},
            -1, 0, true
        );
    }
}

/// The layer's output, h2 = h1 + down_proj(SwiGLU), L_TOKENS x D.
///
/// One producer, one whole token-major row per packet, and D = 5120 split
/// across two axes of one descriptor because a 10-bit dimension tops out at
/// 2048 once bf16 sizes are halved on lowering. Never two BDs -- the first
/// would swallow whole tokens.
void _receive_h2(npu_sequence* seq, const attn_layout& t) {
    const uint32_t inner = (uint32_t)(t.D / 4);
    assert(t.D % 4 == 0 && inner <= 2048 &&
           "a BD dimension is 10 bits, and bf16 sizes are halved");
    assert(t.L_TOKENS <= 1024);

    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        x_arg_id,
        S2MM,
        xyr_tile,
        bd_6,
        it_channel_1,
        {0, 0, 0, (uint32_t)t.h2_offset},
        {1, (uint32_t)t.L_TOKENS, 4,     inner},
        {0, (uint32_t)t.D,        inner, 1},
        -1, 0, true
    );
}

/// The whole context out of the DDR cache and into the two attention columns,
/// in chunks of L_CHUNK keys.
///
/// Each column takes the two kv heads it owns, and its memtile splits those
/// again per CU. The cache is token-major over all four heads:
///
///     k[t * DK + head * DH .. + DH)
///
/// so a column's slice is a strided walk -- kv_heads_per_column * DH out of
/// every DK -- with column c starting at c * kv_heads_per_column * DH.
///
/// Every token re-reads the whole context, and **that repeat belongs in the
/// iteration dimension** -- size[0], the first of the four. It is the axis the
/// hardware provides for exactly this, a BD replayed from the same address, and
/// its counter is 6 bits, so the repeat count has to stay under 64.
///
/// It cannot be an ordinary axis with a zero stride. npu_dma_memcpy_nd reads a
/// zero stride as "this axis does not advance" and collapses it, and the
/// iteration count is deliberately left out of buffer_length, so writing the
/// repeat as size L with stride 0 on dim1 sends the context once and the cores
/// wait forever for the rest. That is what hung the first version of this: the
/// run sat until XRT gave up after 63 s -- against 6 ms for the same design
/// without attention -- and the output region was never written at all.
///
/// Each pass is sized to the *longest* token, which is why the repeat is per
/// token rather than one long stream. **Every token therefore receives the same
/// number of chunks, and both kernels consume that same number** -- they derive
/// it from the one RTP word exactly as `chunks` is derived here, and mask the
/// keys past their own L[t] rather than stopping early. See the assert below.
///
///   dim0 (L_TOKENS) the repeat, carried by the queue push
///   dim1 (chunks)   successive L_CHUNKs of context, stride L_CHUNK * row stride
///   dim2 (L_CHUNK)  one chunk's context rows, stride one row
///   dim3 (width)    the column's two heads, contiguous
///
/// Each plane is walked on its own descriptor, so the row stride is just DK (or
/// DV) and nothing here depends on MAX_L. Only the base offsets differ -- k_at(0)
/// is 0 and v_at(0) is a whole k plane in. The append is symmetric now: it too
/// puts k and v on separate channels, so no descriptor anywhere carries a
/// MAX_L-scaled stride; see _receive_kv_cache.
///
/// What *does* bound the context is right here: dim1 counts whole L_CHUNKs and a
/// BD size field is 10 bits, so a pass can replay at most 1024 chunks. That is
/// attn_cache::CHUNKS_LIMIT, and attn_layout::build turns it into the max_len
/// cap -- against the deepest pass's reach, context_offset + L_TOKENS, not
/// against max_len alone.
void _replay_kv_cache(npu_sequence* seq, const attn_layout& t, int context_offset) {
    const int      L_CHUNK = attn_layout::L_CHUNK;
    const uint32_t width   = (uint32_t)(t.kv_heads_per_column * t.DH);
    const uint32_t rows    = (uint32_t)(((context_offset + t.L_TOKENS) + L_CHUNK - 1)
                                        / L_CHUNK * L_CHUNK);
    const uint32_t chunks  = rows / (uint32_t)L_CHUNK;
    const uint32_t k_row_stride   = (uint32_t)t.k_row_stride();
    const uint32_t v_row_stride   = (uint32_t)t.v_row_stride();
    const uint32_t k_chunk_stride = (uint32_t)(L_CHUNK * t.k_row_stride());
    const uint32_t v_chunk_stride = (uint32_t)(L_CHUNK * t.v_row_stride());
    // The backstop on the caller's guard. `rows` rounds up to an L_CHUNK, and
    // there is no apron above MAX_L to absorb the overshoot -- it is sound only
    // because context_offset + L_TOKENS <= MAX_L (gen_attn_layer_seq checks it)
    // and MAX_L is a whole number of chunks, so the round-up lands ON the top.
    assert(rows <= (uint32_t)t.plane_rows());
    assert(chunks <= 1024);      // dim1 is 10 bits
    assert(width <= 1023 * 2);   // 10-bit dims, doubled for bf16

    // **One pass over the whole context per token; the repeat covers the
    // tokens.** The memtile's inbound BD holds one L_CHUNK x kv_row buffer, so
    // it consumes a chunk per firing -- but the *descriptor* still has to walk
    // every chunk, which is what dim1 does. Describing only one chunk and
    // folding the chunk count into the repeat sends chunk 0 over and over and
    // never reaches row 16, because the repeat replays a BD from the same
    // address. That is invisible at context_offset = 0, where L_TOKENS = 8 rows
    // round up to exactly one chunk and the two forms coincide.
    assert(t.L_TOKENS < 64 && "the queue's repeat count is 6 bits");
    assert(width * L_CHUNK <= 1023 * 2 * 1024 && "dim sizes are 10 bits");

    // Every token gets the SAME `chunks`, because the repeat is the iteration
    // dimension and that replays one descriptor -- a per-token length is not
    // expressible here, and issuing eight separate BDs instead deadlocks the
    // channel against the cores. So the surplus is the kernels' to absorb, and
    // this is the assert iron/attn.py's attn_memtile refers to: both sides
    // derive the count from the same number. A kernel that stopped at its own
    // ceil(L[t] / L_CHUNK) would leave its unread chunks queued for the next
    // token -- that was the bug fixed on 2026-09-17, invisible to prefill,
    // which only ever dispatches at ctx % 16 in {0, 8}, and fatal to
    // speculation, which resumes wherever acceptance left the cursor.
    const uint32_t kernel_chunks =
        ((uint32_t)(context_offset + t.L_TOKENS) + L_CHUNK - 1) / L_CHUNK;
    assert(chunks == kernel_chunks && "sent != consumed; see kernels/attn_qk.cc");
    (void)kernel_chunks;

    for (int c = 0; c < 2; c++) {
        const uint32_t head_off = (uint32_t)(c * t.kv_heads_per_column * t.DH);
        seq->npu_dma_memcpy_nd(
            sizeof(bf16),
            kv_cache_arg_id,
            MM2S,
            attn_col_tiles[c],
            bd_0,
            it_channel_0,
            {0, 0, 0, (uint32_t)t.k_at(0) + head_off},
            {(uint32_t)t.L_TOKENS, chunks,         (uint32_t)L_CHUNK, width},
            {0,                    k_chunk_stride, k_row_stride,      1},
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
            {(uint32_t)t.L_TOKENS, chunks,         (uint32_t)L_CHUNK, width},
            {0,                    v_chunk_stride, v_row_stride,      1},
            -1, 0, false, no_cache
        );
    }
}

/// One projection round set: `Dout x Din` weights, streamed to the four mvm
/// columns two row-blocks at a time.
///
/// \param w_offset **in bf16 elements, not bytes.** The descriptor's element
///        size is sizeof(bf16) and a_block_size is likewise divided down, so
///        the offset has to be in the same unit as everything else in this
///        arithmetic. Passing a byte offset addresses twice as far into the
///        buffer and streams whatever is past the end -- which the cores
///        dequantise into NaN rather than rejecting. attn_layout's w_*_words()
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
void gen_attn_layer_seq(npu_sequence* seq, const attn_layout& t,
                        int context_offset, const attn_rtp_addresses& rtp) {
    // **The whole pass, not just its first row.** The append writes L_TOKENS
    // rows from here and the replay streams through them; the cache has no
    // apron above MAX_L, so both ends have to be inside it.
    if (context_offset < 0 || context_offset + t.L_TOKENS > t.MAX_L)
        throw std::runtime_error(
            "gen_attn_layer_seq: a pass at context_offset " +
            std::to_string(context_offset) + " appends through " +
            std::to_string(context_offset + t.L_TOKENS) +
            ", outside a cache of MAX_L " + std::to_string(t.MAX_L));
    if (rtp.qk == 0 || rtp.kv == 0)
        throw std::runtime_error("gen_attn_layer_seq: RTP addresses are unset");

    seq->clear_cmds();

    _write_attn_L(seq, t, context_offset, rtp);

    _send_hidden_states(seq, t);
    _send_rms_weights(seq, t);
    _send_rope_weights(seq, t);
    _receive_kv_cache(seq, t, context_offset);
    _receive_h2(seq, t);

    _move_weights(seq, t.qkv_out, t.D, 0);
    _move_weights(seq, t.DQ, t.D, t.w_z_words());

    // The append, waited on per column **and per plane**. This only orders the
    // replay because both are on these same two shims -- npu_dma_wait does not
    // reach across shim tiles, which is why the k|v sink was moved here from
    // IT3. Everything after this point may assume the cache holds
    // context_offset + L_TOKENS valid rows.
    //
    // Two channels each: k lands on S2MM ch0 and v on ch1 (iron/RoPE.py stamps
    // them with different packet ids). Waiting only on ch0 would order the
    // replay against the k plane alone and leave v racing -- the same class of
    // bug as waiting on the wrong shim, and just as quiet, since the CUs use
    // the v they computed rather than the v they read back.
    for (int c = 0; c < 2; c++) {
        seq->npu_dma_wait(attn_col_tiles[c], S2MM, it_channel_0);   // k
        seq->npu_dma_wait(attn_col_tiles[c], S2MM, it_channel_1);   // v
    }

    _replay_kv_cache(seq, t, context_offset);

    // o_proj's weights, and **not with the other two rounds**. _move_weights
    // waits on its own transfers between rounds, and the cores cannot touch o's
    // weights until the gate has filled o's replay phase -- which needs the
    // attention output, which needs the replay above.
    //
    // Dout is D and Din is DQ: K is the gated row here.
    _move_weights(seq, t.D, t.DQ, t.w_o_words());

    // The MLP: up|gate off the post-attention norm, then down off the SwiGLU.
    // Fourth and fifth rounds, so fourth and fifth mvm(). Both are behind
    // o_proj's round for the same reason it was behind the replay -- the x each
    // reduces over does not exist until the stage before it has run.
    _move_weights(seq, 2 * (size_t)t.intermediate_size, t.D, t.w_upgate_words());
    _move_weights(seq, t.D, t.intermediate_size, t.w_down_words());

    seq->npu_dma_wait(xyr_tile, S2MM, it_channel_1);

    seq->cmds2seq();
}

} // namespace q38npu
