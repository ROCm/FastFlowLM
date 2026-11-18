// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file verify/npu/attn_weights.hpp
/// \brief Device-side buffer layout for one gated full-attention layer, and
///        the staging that fills it.
///
/// Port of decoding_layer's constructor arithmetic and load_weights(). The
/// source is the mmap'd cpu_ref::safetensors rather than Q4NX, so reorder_cpy
/// reads straight out of the page cache instead of several GB of copies.
///
/// Geometry comes from cpu_ref::q38, the same source cpu_ref::attention_layer
/// uses, so the reference and the device layout cannot disagree.
///
/// These functions only fill a caller-supplied buffer; they own no BO.
#ifndef __QWEN3_8MTP_NPU_ATTN_WEIGHTS_HPP__
#define __QWEN3_8MTP_NPU_ATTN_WEIGHTS_HPP__

#include "common/cpu_ref_common.hpp"        // cpu_ref::safetensors, cpu_ref::q38
#include "verify/npu/q4nx.hpp"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

namespace q38npu {

// ==========================================================================
// the KV cache's shape constants
// ==========================================================================
/// Everything the two-plane cache's bound is derived from, at namespace scope
/// so `attn_layout` can use it in the initialiser of its own static members --
/// a constexpr member function is not usable inside the class that declares it.
namespace attn_cache {

/// Keys are streamed to the CUs this many at a time. Baked into the kernels --
/// attn_qk builds its causal mask from a 16-lane vector -- so this names the
/// constant rather than choosing it. Matches L_CHUNK in iron/attn.py and in
/// decoding_layer.hpp.
constexpr int L_CHUNK = 16;

/// The largest stride a shim BD can encode: the field is 20 bits and holds
/// stride - 1, so 1 << 20 u32 words.
///
/// **Nothing in this layout is bounded by it any more**, which is the point of
/// the note below. It is kept because build() still checks against it, and a
/// check that can never fire is cheaper than one that was never written.
constexpr size_t BD_STRIDE_LIMIT = size_t(1) << 20;

/// The most chunks the replay's dim1 can walk: a 10-bit BD dimension.
constexpr int CHUNKS_LIMIT = 1 << 10;

/// The prefill design's query chunk, `lc * 8` in
/// FLM_Xclbin/Qwen3_8/gated_attention_prefill/host/seq_gen.hpp. A prefill pass
/// covers a whole number of these and its kv broadcast walks 128-row blocks.
constexpr int PREFILL_CHUNK = 128;

/// **The cache has no apron: `MAX_L` is the buffer, the pitch, and the cap.**
///
/// This is `qwen3_5vl_npu`'s layout (`get_kv_cache_size(MAX_L) =
/// MAX_L * (DK + DV)`) and the IRON scaffold's, and the invariant that makes it
/// sound is the alignment asserted in build(): `MAX_L % PREFILL_CHUNK == 0`.
///
/// Three passes would otherwise run off the top of the cache. Each is closed by
/// a guard on the WHOLE pass rather than on its first row:
///
///   the write   `_receive_kv_cache` appends L_TOKENS rows however few the
///               caller asked for, so a pass at `MAX_L - 1` would write
///               through `MAX_L + L_TOKENS - 2`. gen_attn_layer_seq and
///               npu_attn_layer::forward require `ctx + L_TOKENS <= MAX_L`.
///               The append cannot simply be shortened to the real L: the RoPE
///               tile emits L_TOKENS rows either way and a short S2MM
///               descriptor strands the rest in the stream.
///   the replay  `_replay_kv_cache` streams `ctx + L_TOKENS` rows rounded up to
///               a whole L_CHUNK. Given the guard above and
///               `MAX_L % L_CHUNK == 0`, that round-up lands on MAX_L at worst.
///   the prefill `gen_prefill_attn_seq`'s kv broadcast walks
///               `kv_end = round_up(L_begin + round_up(L, PREFILL_CHUNK),
///               PREFILL_CHUNK)`. prefill_attn_op::run requires that of MAX_L,
///               so the padded window has to fit, not just the real one. It
///               cannot be clamped host-side either: attn.cc derives the same
///               count itself (`kv_blocks`), and a host-only clamp would leave
///               blocks queued for the next dispatch.
///
/// The rows past the cursor are not slack the kernel ignores -- the broadcast
/// BD's dim1 count is `kv_length / 128`, so they are really fetched. They are
/// masked out of the softmax, but reading them off the end of the BO would not
/// fault: it would quietly be the next layer's cache.
///
/// The engine is what keeps `MAX_L` large enough that those guards never bite a
/// context a caller can reach -- see qwen3_8mtp_npu.cpp, which rounds the
/// requested length up to a power of two and then adds one PREFILL_CHUNK, or,
/// above 8192, adds the PREFILL_CHUNK first and rounds up to a 4096-row page,
/// clamped to max_context(). Where that clamp leaves less than a PREFILL_CHUNK
/// of headroom, the engine runs a prompt's last rows through the verify pass.

/// The largest `MAX_L` build() accepts: 16384.
///
/// **This used to be a stride, and it used to be 2041.** The append crossed
/// from a token's k to its v inside one BD, a hop of `MAX_L * DK`, which is
/// BD_STRIDE_LIMIT halved for bf16 -- so the plane could not exceed 2048 rows,
/// and above that npu_dma_memcpy_nd masked the stride instead of faulting.
/// iron/RoPE.py now stamps k and v with different packet ids so they land on
/// two S2MM channels of the column's shim, and each plane is written by its own
/// BD whose only stride is one row. See _receive_kv_cache.
///
/// What is left is the replay: its dim1 carries the chunk count, a 10-bit
/// field. The deepest pass is now bounded by the guard above rather than by an
/// apron -- it reaches exactly `MAX_L` -- so the bound is `MAX_L / L_CHUNK <=
/// CHUNKS_LIMIT` and comes out as one number, the same one and for the same
/// reason as mtp_layout::MAX_L_LIMIT.
constexpr int max_context() { return CHUNKS_LIMIT * L_CHUNK; }

} // namespace attn_cache

// ==========================================================================
// geometry
// ==========================================================================
/// Every size the attention xclbin's five buffer arguments need.
///
/// Mirrors decoding_layer's constructor. Kept as a plain struct so the engine
/// can compute it once and both the staging code and the sequence generator
/// stay pure functions of it.
struct attn_layout {
    // --- from the model ----------------------------------------------------
    int D  = 0;
    int DH = 0;
    int DQ = 0, DK = 0, DV = 0;
    int num_kv_heads = 0;
    int intermediate_size = 0;
    int qkv_out = 0;      ///< DQ + DK + DV, the fused projection's width
    int DQ_PADDED = 0;    ///< what leaves the RoPE tile, GQA padding included
    int ROPE_DIM_ = 0;    ///< head_dim * partial_rotary_factor, 64 here

    // --- attention partitioning --------------------------------------------
    int q_heads_per_group   = 0;   ///< 6: what a CU really computes
    int cus_per_column      = 2;
    int kv_heads_per_column = 2;
    int attn_col_width      = 0;   ///< one column's share of a row of output

    /// Keys are streamed to the CUs this many at a time. Baked into the
    /// kernels -- attn_qk builds its causal mask from a 16-lane vector -- so
    /// this names the constant rather than choosing it. Matches L_CHUNK in
    /// iron/attn.py and in decoding_layer.hpp.
    static constexpr int L_CHUNK = attn_cache::L_CHUNK;

    /// Query rows per pass. Compiled into the bitstream (Makefile: L_TOKENS ?= 8)
    /// and into the DMA descriptor sizes.
    int L_TOKENS = 8;

    /// **The largest MAX_L this cache layout can address: 16384 tokens.**
    ///
    /// It was 2032, and the thing that bound it was a stride. The append BD
    /// used to write a token's k AND its v in ONE descriptor, because the RoPE
    /// tile interleaved them per token on a single stream, so it crossed
    /// between the planes with stride[2] = MAX_L * DK -- a 20-bit field,
    /// halved for bf16, so the plane could not exceed 2048 rows. Worse,
    /// overshooting did not fault: npu_dma_memcpy_nd printed "stride_2 out of
    /// range" and masked the value, leaving every v row at a wrapped address,
    /// invisible until the next pass read it back.
    ///
    /// iron/RoPE.py now gives k and v distinct packet ids, so they land on two
    /// S2MM channels of the column's shim and each plane is appended by its own
    /// BD. Neither has a stride that depends on MAX_L; the plane base is a
    /// byte offset patched in at full width. The IRON reference carries the
    /// same split, so a descriptor diffed between this and
    /// FLM_Xclbin/Qwen3_8/gated_attention_verify still agrees on the cache.
    ///
    /// What binds now is the replay's chunk count on a 10-bit dim. Because the
    /// guards refuse a pass that would reach past MAX_L, the deepest replay
    /// ends exactly at MAX_L, so the bound is simply
    /// `MAX_L <= CHUNKS_LIMIT * L_CHUNK`. This one is asserted, not masked.
    /// **It is above the engine's MAX_L floor of 4096** (qwen3_8mtp_npu.cpp),
    /// so a default-configured engine constructs; before the split it did not.
    static constexpr int MAX_L_LIMIT = attn_cache::max_context();
    /// **How many positions the KV cache BO holds -- the buffer, in tokens.**
    ///
    /// This is the one number the cache is described by: the height of each
    /// plane, the pitch from the k plane to the v plane, and half the BO. The
    /// same meaning `MAX_L` carries in qwen3_5vl_npu, whose
    /// `get_kv_cache_size(MAX_L)` is `MAX_L * (DK + DV)`.
    ///
    /// It is a buffer size, not the usable context: a pass is legal only if the
    /// WHOLE of it fits, which costs L_TOKENS rows on the verify path and up to
    /// PREFILL_CHUNK - 1 on the prefill path. The engine is what leaves room
    /// for that -- see qwen3_8mtp_npu.cpp. Changing this means regenerating the
    /// sequence AND reallocating the cache BO.
    ///
    /// Must be a multiple of PREFILL_CHUNK; build() asserts it.
    int MAX_L = MAX_L_LIMIT;

    // --- arg 0: | x | o | down | h2 |, each L_TOKENS x D --------------------
    // One buffer; the sequence reaches every region through x_arg_id. h2 is
    // what the layer produces. o and down are diagnostics the shipped sequence
    // no longer taps (decoding_layer::_receive_o is commented out there), but
    // the regions stay so the offsets, and any future tap, match the harness.
    size_t o_offset = 0, down_offset = 0, h2_offset = 0;
    size_t x_words = 0;         ///< total size of the arg-0 BO, in bf16 words

    // --- arg 1: all_proj_weight, BYTE offsets -------------------------------
    // **Bytes, not bf16 words.** decoding_layer keeps them this way and divides
    // by sizeof(bf16) at each _move_weights() call site, because _move_weights
    // builds its descriptor with an element size of sizeof(bf16) and its own
    // a_block_size divided down to match. The w_*_words() accessors below are
    // the single place that conversion happens; never hand a byte offset
    // straight to the sequence generator. (Doing so addresses twice as far into
    // the buffer and streams whatever is past the end, which the cores
    // dequantise into NaN rather than rejecting.)
    size_t w_z_offset = 0;      ///< output gate z, one q4k(DQ x D) past q|k|v
    size_t w_o_offset = 0;      ///< o_proj, D x DQ
    size_t w_upgate_offset = 0; ///< the woven up | gate
    size_t w_down_offset = 0;   ///< down_proj, D x intermediate_size
    size_t proj_bytes = 0;      ///< total size of the arg-1 BO

    size_t w_z_words()      const { return w_z_offset      / sizeof(uint16_t); }
    size_t w_o_words()      const { return w_o_offset      / sizeof(uint16_t); }
    size_t w_upgate_words() const { return w_upgate_offset / sizeof(uint16_t); }
    size_t w_down_words()   const { return w_down_offset   / sizeof(uint16_t); }

    // --- arg 2 and arg 3 -----------------------------------------------------
    size_t rms_words = 0;       ///< 2 * D: input_layernorm then post_attention
    /// Where q_norm.weight starts in the RoPE block:
    ///   [t * ROPE_DIM              .. + ROPE_DIM/2)   cos for token t
    ///   [t * ROPE_DIM + ROPE_DIM/2 .. + ROPE_DIM/2)   sin for token t
    ///   [L_TOKENS * ROPE_DIM       .. + DH)           q_norm.weight
    ///   [L_TOKENS * ROPE_DIM + DH  .. + DH)           k_norm.weight
    size_t rope_w_offset = 0;
    size_t rope_words = 0;      ///< total size of the arg-3 BO

    // --- arg 4 ---------------------------------------------------------------
    /// The KV cache BO, in bf16 elements: 2 * MAX_L * DK.
    ///
    /// **Token-major, TWO CONTIGUOUS PLANES**, exactly as
    /// FLM_Xclbin/Qwen3_8/gated_attention_verify lays it out: every row of k
    /// first, then every row of v starting at `MAX_L * DK`. Token `t`'s k is at
    /// `t * DK` and its v at `MAX_L * DK + t * DV`. The five accessors below
    /// are the single place the layout is expressed -- flip them and every
    /// descriptor, the seeder, the read-back and the grow follow.
    ///
    /// The cap this carries is MAX_L_LIMIT; read it before raising MAX_L.
    size_t kv_cache_size = 0;
    /// The pitch of each plane, in tokens. The two planes are the same height,
    /// so this is also where the v plane starts, measured in rows -- and with
    /// no apron it is just MAX_L. Kept as an accessor because every descriptor
    /// generator asks for the pitch by name.
    size_t plane_rows() const { return (size_t)MAX_L; }
    /// Elements from one k row to the next, and from one v row to the next. The
    /// planes are walked separately, so each has its own -- and neither depends
    /// on MAX_L. No descriptor does any more: the plane base is patched in as a
    /// full-width byte offset, which is why MAX_L_LIMIT is no longer a stride.
    size_t k_row_stride() const { return (size_t)DK; }
    size_t v_row_stride() const { return (size_t)DV; }
    /// Where token `t`'s k begins.
    size_t k_at(int t) const { return (size_t)t * k_row_stride(); }
    /// Where token `t`'s v begins -- past the whole k plane.
    size_t v_at(int t) const {
        return plane_rows() * (size_t)DK + (size_t)t * v_row_stride();
    }

    /// \param max_len  the cache buffer's height in tokens. Taken as given: a
    ///        whole multiple of PREFILL_CHUNK, at most MAX_L_LIMIT
    /// \param L        query rows per pass; must match the bitstream's L_TOKENS
    static attn_layout build(int max_len = MAX_L_LIMIT, int L = 8) {
        using namespace cpu_ref::q38;
        attn_layout t;
        t.D            = MODEL_DIM;
        t.DH           = HEAD_DIM;
        // Qualified: the unqualified name would find the member, not the
        // constant -- the using-directive injects into the enclosing namespace,
        // and class scope is searched first.
        t.DQ           = cpu_ref::q38::DQ;
        t.DK           = DKV;
        t.DV           = DKV;
        t.num_kv_heads = NUM_KV_HEADS;
        t.intermediate_size = INTERMEDIATE_SIZE;
        t.qkv_out      = t.DQ + t.DK + t.DV;
        t.ROPE_DIM_    = ROPE_DIM;
        t.L_TOKENS     = L;
        t.MAX_L        = max_len;

        // The GLU tile takes one engine round at a time, and a round has to be
        // all up or all gate -- which is what the weave in
        // stage_attn_proj_weights() below assumes.
        if (t.intermediate_size % 512 != 0)
            throw std::runtime_error("a projection round is 512 output rows; "
                                     "up|gate is woven at that grain");
        // The engine partitions output rows 512 at a time (16 cores x 32 rows)
        // and the weight DDR layout pairs adjacent row blocks. q, k and v are
        // each a whole number of both, so concatenating three separately
        // reordered matrices is byte for byte one reorder of the fused matrix.
        if (t.DQ % 512 || t.DK % 512 || t.DV % 512)
            throw std::runtime_error("q/k/v are not whole engine rounds");

        // What the RoPE tile actually emits. 24 q heads over 4 kv heads is 6
        // per group, padded to 8 for the 1x8x1 attention CU, so q comes out a
        // third wider than DQ with the tail of each group zeroed. Must match
        // DQ_PADDED in iron/RoPE.py, which computes it the same way.
        t.q_heads_per_group = NUM_ATTN_HEADS / NUM_KV_HEADS;
        const int q_heads_padding = 8;
        t.DQ_PADDED = (t.q_heads_per_group + q_heads_padding - 1) / q_heads_padding
                    * q_heads_padding * t.num_kv_heads * t.DH;

        // One kv head per CU, two CUs per column, two columns. Has to agree
        // with iron/model_config.py's kv_heads_per_cu and cus_per_column,
        // which is what iron/attn.py builds the BDs from.
        t.cus_per_column      = 2;
        t.kv_heads_per_column = t.cus_per_column;
        t.attn_col_width      = t.cus_per_column * t.q_heads_per_group * t.DH;
        if (t.num_kv_heads != 4 || 2 * t.attn_col_width != t.DQ)
            throw std::runtime_error("attention partitioning is two columns "
                                     "x two CUs x one kv head");

        const size_t LD = static_cast<size_t>(L) * t.D;
        t.o_offset    = LD;
        t.down_offset = t.o_offset + LD;
        t.h2_offset   = t.down_offset + LD;
        t.x_words     = t.h2_offset + LD;

        const size_t Dz = static_cast<size_t>(t.D);
        t.w_z_offset      = q4k_bytes(static_cast<size_t>(t.qkv_out) * Dz);
        t.w_o_offset      = t.w_z_offset + q4k_bytes(static_cast<size_t>(t.DQ) * Dz);
        t.w_upgate_offset = t.w_o_offset + q4k_bytes(Dz * t.DQ);
        t.w_down_offset   = t.w_upgate_offset
                          + q4k_bytes(2 * static_cast<size_t>(t.intermediate_size) * Dz);
        t.proj_bytes      = t.w_down_offset
                          + q4k_bytes(Dz * t.intermediate_size);

        // Every offset the DMA sees must be an even bf16 word count:
        // npu_dma_memcpy_nd folds bf16 into 32-bit words, halving the offset
        // and every stride, so an odd one silently truncates.
        if (t.w_z_offset % 4 || t.w_o_offset % 4 || t.w_upgate_offset % 4 ||
            t.w_down_offset % 4)
            throw std::runtime_error("a projection offset is not an even bf16 "
                                     "word -- the DMA folds it to u32");

        t.rms_words     = 2 * static_cast<size_t>(t.D);
        t.rope_w_offset = static_cast<size_t>(L) * t.ROPE_DIM_;
        t.rope_words    = t.rope_w_offset + 2 * static_cast<size_t>(t.DH);

        // **The buffer, and the one alignment the apron-free layout rests on.**
        //
        // `max_len` is taken as given -- it IS the plane height and the BO is
        // two of them, exactly as qwen3_5vl_npu's get_kv_cache_size(MAX_L).
        // Nothing is added here, so `MAX_L` never means two different numbers.
        //
        // What makes that sound is the alignment: a whole PREFILL_CHUNK (which
        // implies a whole L_CHUNK, 128 being a multiple of 16), so neither the
        // replay's round-up nor the prefill broadcast's can carry a legal pass
        // past the top. The scaffold asserts the L_CHUNK half of this for the
        // same reason; see speculate/npu/mtp_weights.hpp.
        if (max_len <= 0)
            throw std::runtime_error("attn_layout: max_len must be positive");
        if (max_len % attn_cache::PREFILL_CHUNK != 0)
            throw std::runtime_error(
                "attn_layout: max_len " + std::to_string(max_len) +
                " is not a whole PREFILL_CHUNK (" +
                std::to_string(attn_cache::PREFILL_CHUNK) + "). The cache has "
                "no apron, so the replay's and the prefill broadcast's "
                "round-ups have to land ON the top of the buffer -- round the "
                "request up before building, as attn_engine::grow and "
                "qwen3_8mtp_npu.cpp do.");
        if (t.DK != t.DV)
            throw std::runtime_error("one BD writes both k and v; "
                                     "they have to be the same width");
        t.kv_cache_size = 2 * (size_t)t.MAX_L * static_cast<size_t>(t.DK);

        // **The cap.** See MAX_L_LIMIT. The replay walks the deepest pass's
        // whole reach, and the guards (gen_attn_layer_seq, npu_attn_layer::
        // forward, prefill_attn_op::run) refuse a pass reaching past MAX_L --
        // so the deepest replay ends exactly at MAX_L, whatever L_TOKENS is,
        // and the chunk count that 10-bit BD dimension has to carry is
        // MAX_L / L_CHUNK.
        const size_t reach_chunks = (size_t)max_len / L_CHUNK;
        if (reach_chunks > (size_t)attn_cache::CHUNKS_LIMIT)
            throw std::runtime_error(
                "attn_layout: max_len " + std::to_string(max_len) +
                " puts the deepest pass's replay at " +
                std::to_string(reach_chunks) + " chunks, past the " +
                std::to_string(attn_cache::CHUNKS_LIMIT) + " its 10-bit BD "
                "dimension can walk. max_len is at most " +
                std::to_string(attn_cache::max_context()) + " (MAX_L_LIMIT).");

        // The append's strides, which no longer depend on max_len at all -- k
        // and v land on separate shim channels, so neither BD crosses between
        // the planes. This is the check that used to be the cap; it is kept
        // because npu_dma_memcpy_nd masks rather than faults, so the day
        // something does reintroduce an MAX_L-dependent stride, it should be
        // this that says so and not a wrong answer three passes later.
        if ((size_t)t.DK / 2 > attn_cache::BD_STRIDE_LIMIT ||
            (size_t)t.DV / 2 > attn_cache::BD_STRIDE_LIMIT)
            throw std::runtime_error(
                "attn_layout: a cache row is wider than a 20-bit BD stride");
        return t;
    }
};

// ==========================================================================
// staging
// ==========================================================================
/// Stage one attention layer's q4_k projections into the arg-1 BO.
///
/// Block order matches what the sequence streams: q|k|v, z, o, up|gate, down.
/// The cores consume the weight stream POSITIONALLY, so this order is not
/// cosmetic -- a permutation dequantises the wrong rows and produces plausible
/// garbage, which is why the gate on this function is a byte compare against
/// the IRON harness's buffer and not a cosine.
inline void stage_attn_proj_weights(uint8_t* dst, const cpu_ref::safetensors& f,
                                    const attn_layout& t, int layer) {
    const std::string p = "model.layers." + std::to_string(layer) + ".";
    const size_t Dz = static_cast<size_t>(t.D);

    {
        // self_attn.q_proj.weight is 2 * DQ rows, not DQ: rows [0, DQ) are q
        // and rows [DQ, 2*DQ) are the output gate z. HF interleaves the two per
        // head and the q4nx converter de-interleaves them, which is what lets
        // the array split the matrix between two mvms. Because q4nx blocks are
        // row-block major, the split is a plain byte offset.
        //
        // They are laid down as two separate rounds -- q|k|v, then z -- because
        // that is the order proj_main() runs its mvms and the sequence streams
        // them positionally. z is *not* contiguous with q here, even though it
        // is in the checkpoint.
        const std::string qn = p + "self_attn.q_proj.weight";
        const std::string kn = p + "self_attn.k_proj.weight";
        const std::string vn = p + "self_attn.v_proj.weight";

        const size_t q_bytes = q4k_bytes(static_cast<size_t>(t.DQ) * Dz);
        const size_t k_bytes = q4k_bytes(static_cast<size_t>(t.DK) * Dz);
        const size_t v_bytes = q4k_bytes(static_cast<size_t>(t.DV) * Dz);
        if (f.nbytes(qn) != 2 * q_bytes)
            throw std::runtime_error(qn + " is not 2 * DQ rows -- q and the "
                                     "output gate should share one tensor");
        if (f.nbytes(kn) != k_bytes || f.nbytes(vn) != v_bytes)
            throw std::runtime_error("k_proj / v_proj are not [DKV][D] q4_k");
        if (t.w_z_offset != q_bytes + k_bytes + v_bytes)
            throw std::runtime_error("w_z_offset does not follow q|k|v");

        const uint8_t* wq = f.raw(qn);
        reorder_cpy(dst,                     wq,            q_bytes, t.D);
        reorder_cpy(dst + q_bytes,           f.raw(kn),     k_bytes, t.D);
        reorder_cpy(dst + q_bytes + k_bytes, f.raw(vn),     v_bytes, t.D);
        reorder_cpy(dst + t.w_z_offset,      wq + q_bytes,  q_bytes, t.D);
    }
    {
        // o_proj. **The reorder's column count is DQ, not D**: it walks the
        // matrix in 32x256 blocks and has to know how many make a row, and
        // o_proj reduces over the gated rows rather than over the hidden state.
        const std::string on = p + "self_attn.o_proj.weight";
        if (f.nbytes(on) != q4k_bytes(Dz * t.DQ))
            throw std::runtime_error(on + " is not [D][DQ] q4_k");
        reorder_cpy(dst + t.w_o_offset, f.raw(on), f.nbytes(on), t.DQ);
    }
    {
        // mlp.up_proj and mlp.gate_proj, **woven 512 output rows at a time
        // rather than concatenated**. The GLU tile takes one engine round at a
        // time and needs a round to be all up or all gate, so the two
        // alternate: round 2p is up, round 2p+1 is gate, both covering
        // intermediate columns [p*512, (p+1)*512).
        //
        // 512 rows is an even number of the engine's 32-row blocks, so this is
        // byte for byte what one reorder_cpy over the woven matrix would
        // produce -- the weave and the row-block pairing do not interfere.
        const std::string un = p + "mlp.up_proj.weight";
        const std::string gn = p + "mlp.gate_proj.weight";
        const size_t blocks = 1024 / 2 / 32 * (Dz / 256);
        const size_t chunk  = blocks * W_BLOCK_BYTES;
        const size_t total  = q4k_bytes(static_cast<size_t>(t.intermediate_size) * Dz);
        const size_t phases = total / chunk;
        if (phases * chunk != total)
            throw std::runtime_error("up|gate does not divide into whole phases");
        if (f.nbytes(un) != total || f.nbytes(gn) != total)
            throw std::runtime_error("up/gate are not [intermediate_size][D] q4_k");

        const uint8_t* up   = f.raw(un);
        const uint8_t* gate = f.raw(gn);
        uint8_t* w = dst + t.w_upgate_offset;
        for (size_t i = 0; i < phases; i++) {
            reorder_cpy(w, up   + i * chunk, chunk, t.D); w += chunk;
            reorder_cpy(w, gate + i * chunk, chunk, t.D); w += chunk;
        }
    }
    {
        // mlp.down_proj, D x intermediate_size. Like o_proj, its K is not D --
        // hence the intermediate_size column count.
        const std::string dn = p + "mlp.down_proj.weight";
        if (f.nbytes(dn) != q4k_bytes(Dz * t.intermediate_size))
            throw std::runtime_error(dn + " is not [D][intermediate_size] q4_k");
        reorder_cpy(dst + t.w_down_offset, f.raw(dn), f.nbytes(dn),
                    t.intermediate_size);
    }
}

/// Both layernorm gains, input then post-attention, into the arg-2 BO.
///
/// One transfer feeds both: the rms tile has a single D-sized w buffer and its
/// self-chained BD reloads it between the two phases, so this is a stream and
/// not two addressable regions.
inline void stage_attn_rms_weights(uint16_t* dst, const cpu_ref::safetensors& f,
                                   const attn_layout& t, int layer) {
    const std::string p = "model.layers." + std::to_string(layer) + ".";
    const size_t n = static_cast<size_t>(t.D) * sizeof(uint16_t);
    if (f.nbytes(p + "input_layernorm.weight") < n ||
        f.nbytes(p + "post_attention_layernorm.weight") < n)
        throw std::runtime_error("a layernorm gain is shorter than D bf16");
    std::memcpy(dst,       f.raw(p + "input_layernorm.weight"),          n);
    std::memcpy(dst + t.D, f.raw(p + "post_attention_layernorm.weight"), n);
}

/// The two per-head norm gains, into the TAIL of the arg-3 BO.
///
/// Only the tail: the cos/sin half in front of them is per-pass and is written
/// by attn_engine::set_rope_table(). Staging the whole block here would clobber
/// the rotary table every time the layer changes.
inline void stage_attn_qk_norm(uint16_t* rope_tail, const cpu_ref::safetensors& f,
                               const attn_layout& t, int layer) {
    const std::string p = "model.layers." + std::to_string(layer) + ".";
    const size_t n = static_cast<size_t>(t.DH) * sizeof(uint16_t);
    if (f.nbytes(p + "self_attn.q_norm.weight") < n ||
        f.nbytes(p + "self_attn.k_norm.weight") < n)
        throw std::runtime_error("a q/k norm gain is shorter than head_dim bf16");
    std::memcpy(rope_tail,        f.raw(p + "self_attn.q_norm.weight"), n);
    std::memcpy(rope_tail + t.DH, f.raw(p + "self_attn.k_norm.weight"), n);
}

} // namespace q38npu

#endif  // __QWEN3_8MTP_NPU_ATTN_WEIGHTS_HPP__
