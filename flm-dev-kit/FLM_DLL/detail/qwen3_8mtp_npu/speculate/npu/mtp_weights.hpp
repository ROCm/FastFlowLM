// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file speculate/npu/mtp_weights.hpp
/// \brief Device-side buffer layout for the MTP draft head's fused layer, and
///        the staging that fills it.
///
/// Port of MTP_decoding/host/decoding_layer's constructor arithmetic and
/// load_weights(). The source is the mmap'd cpu_ref::safetensors rather than
/// Q4NX, so reorder_cpy reads straight out of the page cache.
///
/// This is *not* verify/npu/attn_weights.hpp with different tensor names. The
/// two designs differ in three ways that each break the other's layout:
///
///   - the head has a fusion projection in front of the layer, so arg 1 opens
///     with model.mtp.fc.weight (D x 2D) and arg 2 carries FOUR norm gains;
///   - the qkv walk is **k, v, q**, not q, k, v -- the RoPE tile consumes kv
///     before q and the arrival order does the split for free;
///   - the bitstream is decode-only, T = 1, so arg 0 is one fusion row in and
///     one D-wide row out rather than an L_TOKENS x D tile.
///
/// Geometry comes from cpu_ref::q38, the same source cpu_ref::mtp_head uses,
/// so the reference and the device layout cannot disagree.
///
/// These functions only fill a caller-supplied buffer; they own no BO.
#ifndef __QWEN3_8MTP_NPU_MTP_WEIGHTS_HPP__
#define __QWEN3_8MTP_NPU_MTP_WEIGHTS_HPP__

#include "common/cpu_ref_common.hpp"        // cpu_ref::safetensors, cpu_ref::q38
#include "verify/npu/q4nx.hpp"              // q4k_bytes, reorder_cpy, W_BLOCK_BYTES

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

namespace q38npu {

// ==========================================================================
// geometry
// ==========================================================================
/// Every size the MTP xclbin's five buffer arguments need.
struct mtp_layout {
    // --- from the model ----------------------------------------------------
    int D  = 0;
    int DH = 0;
    int DQ = 0, DK = 0, DV = 0;
    int num_kv_heads = 0;
    int intermediate_size = 0;
    int qkv_out = 0;      ///< DQ + DK + DV, the fused projection's width
    int ROPE_DIM_ = 0;    ///< head_dim * partial_rotary_factor, 64 here
    int FC_IN = 0;        ///< 2 * D, the fusion's concatenated input

    // --- attention partitioning --------------------------------------------
    int q_heads_per_group   = 0;   ///< 6
    int num_attn_columns    = 2;
    int cus_per_column      = 2;
    int kv_heads_per_column = 0;   ///< 2: one column's share of a cache row

    /// Keys per streamed chunk. Baked into the kernels -- attn_qk builds its
    /// causal mask from a 16-lane vector -- so this names the constant rather
    /// than choosing it. Matches L_CHUNK in iron/attn.py and decoding_layer.hpp.
    static constexpr int L_CHUNK = 16;

    /// **The bitstream is decode-only.** Every descriptor in mtp_sequence.cpp
    /// moves exactly one token: one fusion row in, one k|v row appended, one
    /// D-wide layer output back. A wider draft window is tiled into this many
    /// sequential dispatches by speculate_npu -- which is exact, because each
    /// dispatch appends its own k|v before the next one replays the cache.
    static constexpr int L_TOKENS = 1;

    /// The most chunks the replay's dim1 can walk: a 10-bit BD dimension.
    static constexpr int CHUNKS_LIMIT = 1 << 10;

    /// The OTHER design's query chunk. `attn.xclbin` computes step 0's
    /// attention core over this cache, and it covers a whole number of
    /// 128-token rounds with a kv broadcast that walks 128-row blocks. Named
    /// here rather than included from attn_cache for the same reason L_CHUNK
    /// and CHUNKS_LIMIT are: this struct states the constants it is built from,
    /// and the two must agree; nothing in the engine asserts it. It is also the alignment build() requires of MAX_L.
    static constexpr int PREFILL_CHUNK = 128;

    /// **The largest MAX_L this cache can address: 16384 tokens.**
    ///
    /// Same bound and the same reason as attn_layout::MAX_L_LIMIT: MAX_L is a
    /// whole number of L_CHUNKs and no legal pass replays past it, so the chunk
    /// count the 10-bit dim1 carries is MAX_L / L_CHUNK. Hence
    /// CHUNKS_LIMIT * L_CHUNK exactly.
    ///
    /// **This used to be a stride, and the port used to interleave to dodge it.**
    /// The append crossed from a token's k to its v inside one BD, a hop of
    /// MAX_L * DK; a shim BD's stride field is 20 bits and bf16 strides are
    /// halved on lowering, so the plane could not exceed 2048 rows -- and above
    /// that npu_dma_memcpy_nd printed "stride_2 out of range" and masked instead
    /// of faulting, so every v row landed at a wrapped address and only the NEXT
    /// pass saw the damage. iron/RoPE.py now stamps k and v with different
    /// packet ids (kv_out_pkt_id 12..15) and iron/MTP_layer_aie.py routes them to
    /// two S2MM channels of the column's shim, so each plane is appended by its
    /// own BD whose only stride is one row. The hop is gone rather than raised,
    /// the interleave is no longer buying anything, and this cache is two planes
    /// again -- byte for byte the scaffold's.
    static constexpr int MAX_L_LIMIT = CHUNKS_LIMIT * L_CHUNK;
    /// **How many positions the head's KV cache BO holds -- the buffer.**
    ///
    /// The one number the cache is described by: the height of each plane, the
    /// pitch from the k plane to the v plane, and half the BO. The scaffold's
    /// meaning, and qwen3_5vl_npu's, where `get_kv_cache_size(MAX_L)` is just
    /// `MAX_L * (DK + DV)`. There is no apron.
    ///
    /// TWO designs read this cache, and each rounds its trip count up:
    ///
    ///   MTP_layer.xclbin  the append writes the single row it was given, but
    ///                     _move_kv_cache streams context_offset + 1 rows
    ///                     ROUNDED UP to a whole L_CHUNK.
    ///   attn.xclbin       step 0's prefill core streams through
    ///                     ceil((L_begin + L_pad) / 128) * 128 rows.
    ///
    /// Both land ON the top of the buffer rather than past it because build()
    /// requires `MAX_L % PREFILL_CHUNK == 0` (128 being a multiple of 16, that
    /// covers the L_CHUNK round-up too) and because the guards refuse a pass
    /// whose WHOLE extent does not fit. Getting either wrong means reading past
    /// the end of the BO into whatever the allocator put next -- which does not
    /// fault.
    int MAX_L = MAX_L_LIMIT;

    // --- arg 0: | e | h | layer_out | ---------------------------------------
    // The fusion's two raw halves, then the layer's output behind them. Behind
    // rather than over: the cores are still reading x when the first row blocks
    // come back.
    size_t layer_out_offset = 0;   ///< = FC_IN
    size_t x_words = 0;            ///< total size of the arg-0 BO, in bf16 words

    // --- arg 1: all_proj_weight, BYTE offsets -------------------------------
    // **Bytes, not bf16 words.** _move_weights builds its descriptor with an
    // element size of sizeof(bf16), so every offset it sees must be in the same
    // unit. The w_*_words() accessors are the single place that conversion
    // happens; never hand a byte offset straight to the sequence generator.
    size_t w_fc_offset     = 0;   ///< model.mtp.fc.weight, D x 2D
    size_t w_qkv_offset    = 0;   ///< k | v | q, in that order
    size_t w_gate_offset   = 0;   ///< output gate z, q_proj's second DQ rows
    size_t w_o_offset      = 0;   ///< o_proj, D x DQ
    size_t w_upgate_offset = 0;   ///< the woven up | gate
    size_t w_down_offset   = 0;   ///< down_proj, D x intermediate_size
    size_t proj_bytes      = 0;   ///< total size of the arg-1 BO

    size_t w_fc_words()     const { return w_fc_offset     / sizeof(uint16_t); }
    size_t w_qkv_words()    const { return w_qkv_offset    / sizeof(uint16_t); }
    size_t w_gate_words()   const { return w_gate_offset   / sizeof(uint16_t); }
    size_t w_o_words()      const { return w_o_offset      / sizeof(uint16_t); }
    size_t w_upgate_words() const { return w_upgate_offset / sizeof(uint16_t); }
    size_t w_down_words()   const { return w_down_offset   / sizeof(uint16_t); }

    // --- arg 2: four norm gains, D each --------------------------------------
    //   [0 * D]  pre_fc_norm_embedding   -- passthrough tile, e half
    //   [1 * D]  pre_fc_norm_hidden      -- passthrough tile, h half
    //   [2 * D]  input_layernorm         -- rms tile, phase 1
    //   [3 * D]  post_attention_layernorm-- rms tile, phase 2
    // The first two go to a different tile than the last two, which is why the
    // sequence sends this BO as three separate transfers over two shims.
    size_t rms_words = 0;

    // --- arg 3 ----------------------------------------------------------------
    /// Where q_norm.weight starts in the RoPE block:
    ///   [0 .. ROPE_DIM/2)                  cos for the drafted token
    ///   [ROPE_DIM/2 .. ROPE_DIM)           sin
    ///   [ROPE_DIM .. + DH)                 q_norm.weight
    ///   [ROPE_DIM + DH .. + DH)            k_norm.weight
    /// T = 1, so the cos/sin half is one pair and not a table.
    size_t rope_w_offset = 0;
    size_t rope_words = 0;

    // --- arg 4 ---------------------------------------------------------------
    /// The KV cache BO, in bf16 elements: 2 * MAX_L * DK.
    ///
    /// **Token-major, TWO CONTIGUOUS PLANES**, the scaffold's layout: every row
    /// of k first, then every row of v starting at `MAX_L * DK`. The four
    /// accessors below are the single place that is written down -- flip them
    /// and every descriptor, mirror_from_cpu and read_k/read_v follow.
    ///
    /// The cap this carries is MAX_L_LIMIT; read it before raising MAX_L.
    size_t kv_cache_size = 0;
    /// The pitch of each plane, in tokens. The two planes are the same height,
    /// so this is also where the v plane starts, measured in rows -- and with
    /// no apron it is just MAX_L, so `v_at(0)` is exactly `MAX_L * DK` and the
    /// descriptors agree with the scaffold's on the nose. Kept as an accessor
    /// because every descriptor generator asks for the pitch by name.
    size_t plane_rows() const { return (size_t)MAX_L; }
    /// Elements from one k row to the next, and from one v row to the next. The
    /// planes are walked separately, so each has its own -- and neither depends
    /// on MAX_L, which is what removed the cap. See MAX_L_LIMIT.
    size_t k_row_stride() const { return (size_t)DK; }
    size_t v_row_stride() const { return (size_t)DV; }
    /// Element offset of row `t`'s k.
    size_t k_at(int t) const { return (size_t)t * k_row_stride(); }
    /// Element offset of row `t`'s v -- past the whole k plane.
    size_t v_at(int t) const {
        return plane_rows() * (size_t)DK + (size_t)t * v_row_stride();
    }

    /// \param max_len the cache buffer's height in tokens. Taken as given: a
    ///        whole multiple of PREFILL_CHUNK, at most MAX_L_LIMIT
    static mtp_layout build(int max_len = MAX_L_LIMIT) {
        using namespace cpu_ref::q38;
        mtp_layout t;
        t.D  = MODEL_DIM;
        t.DH = HEAD_DIM;
        // Qualified: the unqualified name would find the member, not the
        // constant -- class scope is searched before the using-directive's.
        t.DQ = cpu_ref::q38::DQ;
        t.DK = DKV;
        t.DV = DKV;
        t.num_kv_heads      = NUM_KV_HEADS;
        t.intermediate_size = INTERMEDIATE_SIZE;
        t.qkv_out   = t.DQ + t.DK + t.DV;
        t.ROPE_DIM_ = ROPE_DIM;
        t.FC_IN     = 2 * t.D;
        t.MAX_L     = max_len;

        // The GLU tile takes one engine round at a time and a round has to be
        // all up or all gate, which is what the weave below assumes.
        if (t.intermediate_size % 512 != 0)
            throw std::runtime_error("a projection round is 512 output rows; "
                                     "up|gate is woven at that grain");
        // The engine partitions output rows 512 at a time (16 cores x 32 rows).
        // k, v and q are each a whole number of rounds, which is what lets the
        // three be reordered separately and still read as one fused matrix --
        // and what keeps the k|v|q split off a round boundary from feeding the
        // RoPE tile garbage.
        if (t.DQ % 512 || t.DK % 512 || t.DV % 512)
            throw std::runtime_error("k/v/q are not whole engine rounds");

        t.q_heads_per_group   = NUM_ATTN_HEADS / NUM_KV_HEADS;
        t.num_attn_columns    = 2;
        t.cus_per_column      = 2;
        t.kv_heads_per_column = t.num_kv_heads / t.num_attn_columns;
        if (t.kv_heads_per_column * t.num_attn_columns != t.num_kv_heads)
            throw std::runtime_error("the kv heads do not divide evenly across "
                                     "the attention columns");

        t.layer_out_offset = (size_t)t.FC_IN;
        t.x_words          = t.layer_out_offset + (size_t)t.D;

        const size_t Dz = (size_t)t.D;
        t.w_fc_offset     = 0;
        t.w_qkv_offset    = q4k_bytes((size_t)t.FC_IN * Dz);
        t.w_gate_offset   = t.w_qkv_offset  + q4k_bytes((size_t)t.qkv_out * Dz);
        t.w_o_offset      = t.w_gate_offset + q4k_bytes((size_t)t.DQ * Dz);
        t.w_upgate_offset = t.w_o_offset    + q4k_bytes(Dz * t.DQ);
        t.w_down_offset   = t.w_upgate_offset
                          + q4k_bytes(2 * (size_t)t.intermediate_size * Dz);
        t.proj_bytes      = t.w_down_offset + q4k_bytes(Dz * t.intermediate_size);

        // Every offset the DMA sees must be an even bf16 word count:
        // npu_dma_memcpy_nd folds bf16 into 32-bit words, halving the offset and
        // every stride, so an odd one silently truncates.
        if (t.w_qkv_offset % 4 || t.w_gate_offset % 4 || t.w_o_offset % 4 ||
            t.w_upgate_offset % 4 || t.w_down_offset % 4)
            throw std::runtime_error("a projection offset is not an even bf16 "
                                     "word -- the DMA folds it to u32");

        t.rms_words     = 4 * Dz;
        t.rope_w_offset = (size_t)t.ROPE_DIM_;
        t.rope_words    = t.rope_w_offset + 2 * (size_t)t.DH;

        // **The buffer, and the one alignment the apron-free layout rests on.**
        //
        // `max_len` is taken as given: it IS the plane height, and the BO is two
        // of them -- the scaffold's shape, and qwen3_5vl_npu's
        // `get_kv_cache_size(MAX_L) = MAX_L * (DK + DV)`. Nothing is added here,
        // so `MAX_L` never means two different numbers in one system.
        //
        // What makes it sound is the alignment. TWO designs read this cache and
        // each rounds its trip count up at the top:
        //
        //   the replay  MTP_layer.xclbin appending at MAX_L - 1 replays
        //               ceil(MAX_L / L_CHUNK) * L_CHUNK rows
        //   the prefill attn.xclbin's step 0 streams
        //               ceil((L_begin + L_pad) / 128) * 128 rows
        //
        // A whole PREFILL_CHUNK covers both (128 is a multiple of 16), so each
        // round-up lands ON the top of the buffer. The other half of the
        // argument is that no legal pass starts high enough to need more, which
        // is what mtp_layer's cursor guard and prefill_attn_op::run enforce --
        // the latter on the PADDED window, since `_use_proj` has no width floor
        // and `run` takes any width by rounding it up to PREFILL_CHUNK.
        if (max_len <= 0)
            throw std::runtime_error("mtp_layout: max_len must be positive");
        if (max_len % PREFILL_CHUNK != 0)
            throw std::runtime_error(
                "mtp_layout: max_len " + std::to_string(max_len) +
                " is not a whole PREFILL_CHUNK (" +
                std::to_string(PREFILL_CHUNK) + "). The cache has no apron, so "
                "the replay's and the prefill broadcast's round-ups have to "
                "land ON the top of the buffer -- round the request up before "
                "building, as npu_mtp_layer::set_max_length and "
                "qwen3_8mtp_npu.cpp do.");
        t.kv_cache_size = 2 * (size_t)t.MAX_L * (size_t)t.DK;
        if (t.DK != t.DV)
            throw std::runtime_error("the two planes are sized as one height "
                                     "times DK; k and v have to be equally wide");
        // See MAX_L_LIMIT. Derived from CHUNKS_LIMIT rather than compared
        // against the constant, so that a different L_CHUNK gets the right
        // bound and not this one's. The replay's dim1 masks into 10 bits as
        // quietly as the old stride did -- it would score against a truncated
        // history -- so this throws rather than trusting the descriptor.
        const size_t max_chunks = (size_t)CHUNKS_LIMIT;
        if ((size_t)(max_len / L_CHUNK) > max_chunks)
            throw std::runtime_error(
                "mtp_layout: max_len " + std::to_string(max_len) + " exceeds " +
                std::to_string(max_chunks * L_CHUNK) + ", the most context rows "
                "the replay's 10-bit chunk dimension can walk.");
        return t;
    }
};

// ==========================================================================
// staging
// ==========================================================================
/// Stage the draft head's q4_k projections into the arg-1 BO.
///
/// Block order matches what the sequence streams: fc, k|v|q, z, o, up|gate,
/// down. The cores consume the weight stream POSITIONALLY, so this order is not
/// cosmetic -- a permutation dequantises the wrong rows and produces plausible
/// garbage.
///
/// Two orderings differ from the base model's attention layers and both are
/// load-bearing:
///   - **fc comes first**, because the fusion is the layer's first mvm;
///   - **k, then v, then q**, because the RoPE tile consumes kv before q. The
///     attention port's q|k|v here feeds the rotation the output gate's rows.
inline void stage_mtp_proj_weights(uint8_t* dst, const cpu_ref::safetensors& f,
                                   const mtp_layout& t) {
    const std::string p  = "model.mtp.";
    const std::string pl = p + "layers.0.";
    const size_t Dz = (size_t)t.D;

    {
        // The fusion, D x 2D. **col is FC_IN, not D**: reorder_cpy walks the
        // matrix in 32x256 blocks and has to know how many make one row, and
        // fc reduces over cat(norm_embed, norm_hidden).
        const std::string fn = p + "fc.weight";
        const size_t fc_bytes = q4k_bytes(Dz * t.FC_IN);
        if (f.nbytes(fn) != fc_bytes)
            throw std::runtime_error(fn + " is " + std::to_string(f.nbytes(fn)) +
                                     " B, expected " + std::to_string(fc_bytes));
        reorder_cpy(dst + t.w_fc_offset, f.raw(fn), fc_bytes, t.FC_IN);
    }
    {
        // k | v | q, in that order. q_proj is 2 * DQ rows, not DQ: rows [0, DQ)
        // are q and rows [DQ, 2*DQ) are the output gate z. HF interleaves the
        // two per head and the q4nx converter de-interleaves them, which is
        // what lets the array split the matrix between two mvms -- and what
        // makes the split a plain byte offset here.
        // (cpu_ref::mtp_head::q_interleaved is false for the same reason.)
        const std::string qn = pl + "self_attn.q_proj.weight";
        const std::string kn = pl + "self_attn.k_proj.weight";
        const std::string vn = pl + "self_attn.v_proj.weight";

        const size_t q_bytes = q4k_bytes((size_t)t.DQ * Dz);
        const size_t k_bytes = q4k_bytes((size_t)t.DK * Dz);
        const size_t v_bytes = q4k_bytes((size_t)t.DV * Dz);
        if (f.nbytes(qn) != 2 * q_bytes)
            throw std::runtime_error(qn + " is not 2 * DQ rows -- q and the "
                                     "output gate should share one tensor");
        if (f.nbytes(kn) != k_bytes || f.nbytes(vn) != v_bytes)
            throw std::runtime_error("mtp k_proj / v_proj are not [DKV][D] q4_k");
        if (t.w_gate_offset != t.w_qkv_offset + k_bytes + v_bytes + q_bytes)
            throw std::runtime_error("w_gate_offset does not follow k|v|q");

        const uint8_t* wq = f.raw(qn);
        uint8_t* w = dst + t.w_qkv_offset;
        reorder_cpy(w, f.raw(kn), k_bytes, t.D); w += k_bytes;
        reorder_cpy(w, f.raw(vn), v_bytes, t.D); w += v_bytes;
        reorder_cpy(w, wq,        q_bytes, t.D);

        // z, the output gate: q_proj's second DQ rows, laid down as their own
        // round because that is a separate mvm.
        reorder_cpy(dst + t.w_gate_offset, wq + q_bytes, q_bytes, t.D);
    }
    {
        // o_proj. **col is DQ, not D**: o reduces over the gated rows.
        const std::string on = pl + "self_attn.o_proj.weight";
        if (f.nbytes(on) != q4k_bytes(Dz * t.DQ))
            throw std::runtime_error(on + " is not [D][DQ] q4_k");
        reorder_cpy(dst + t.w_o_offset, f.raw(on), f.nbytes(on), t.DQ);
    }
    {
        // mlp.up_proj and mlp.gate_proj, **woven 512 output rows at a time
        // rather than concatenated**. The GLU tile takes one engine round at a
        // time and needs a round to be all up or all gate, so the two
        // alternate: round 2i is up, round 2i+1 is gate, both covering
        // intermediate columns [i*512, (i+1)*512).
        //
        // 512 rows is an even number of the engine's 32-row blocks, so this is
        // byte for byte what one reorder_cpy over the woven matrix would
        // produce -- the weave and the row-block pairing do not interfere.
        const std::string un = pl + "mlp.up_proj.weight";
        const std::string gn = pl + "mlp.gate_proj.weight";
        const size_t blocks = 512 / 32 * (Dz / 256);
        const size_t chunk  = blocks * W_BLOCK_BYTES;
        const size_t total  = q4k_bytes((size_t)t.intermediate_size * Dz);
        const size_t phases = total / chunk;
        if (phases * chunk != total)
            throw std::runtime_error("mtp up|gate does not divide into whole phases");
        if (f.nbytes(un) != total || f.nbytes(gn) != total)
            throw std::runtime_error("mtp up/gate are not [intermediate_size][D] q4_k");

        const uint8_t* up   = f.raw(un);
        const uint8_t* gate = f.raw(gn);
        uint8_t* w = dst + t.w_upgate_offset;
        for (size_t i = 0; i < phases; i++) {
            reorder_cpy(w, up   + i * chunk, chunk, t.D); w += chunk;
            reorder_cpy(w, gate + i * chunk, chunk, t.D); w += chunk;
        }
    }
    {
        // mlp.down_proj, D x intermediate_size. Like o_proj, its K is not D.
        const std::string dn = pl + "mlp.down_proj.weight";
        if (f.nbytes(dn) != q4k_bytes(Dz * t.intermediate_size))
            throw std::runtime_error(dn + " is not [D][intermediate_size] q4_k");
        reorder_cpy(dst + t.w_down_offset, f.raw(dn), f.nbytes(dn),
                    t.intermediate_size);
    }
}

/// All four norm gains into the arg-2 BO, in the order the sequence reads them.
///
/// The first two feed the passthrough tile (the fusion's two halves), the last
/// two the rms tile (the layer's two phases). Three transfers, two shims -- see
/// mtp_sequence.cpp -- so unlike the attention layer's arg 2 these ARE four
/// separately addressed regions and not one stream.
inline void stage_mtp_rms_weights(uint16_t* dst, const cpu_ref::safetensors& f,
                                  const mtp_layout& t) {
    const std::string p  = "model.mtp.";
    const std::string pl = p + "layers.0.";
    const size_t n = (size_t)t.D * sizeof(uint16_t);
    const std::string s[4] = {
        p  + "pre_fc_norm_embedding.weight",
        p  + "pre_fc_norm_hidden.weight",
        pl + "input_layernorm.weight",
        pl + "post_attention_layernorm.weight",
    };
    for (int i = 0; i < 4; i++) {
        if (f.nbytes(s[i]) < n)
            throw std::runtime_error(s[i] + " is shorter than D bf16");
        std::memcpy(dst + (size_t)i * t.D, f.raw(s[i]), n);
    }
}

/// The two per-head norm gains, into the TAIL of the arg-3 BO.
///
/// Only the tail: the cos/sin pair in front of them is per-dispatch and is
/// written by npu_mtp_layer::_set_rope_position(). Staging the whole block here
/// would clobber the rotation every time the position moves.
inline void stage_mtp_qk_norm(uint16_t* rope_tail, const cpu_ref::safetensors& f,
                              const mtp_layout& t) {
    const std::string pl = "model.mtp.layers.0.";
    const size_t n = (size_t)t.DH * sizeof(uint16_t);
    if (f.nbytes(pl + "self_attn.q_norm.weight") < n ||
        f.nbytes(pl + "self_attn.k_norm.weight") < n)
        throw std::runtime_error("an mtp q/k norm gain is shorter than head_dim bf16");
    std::memcpy(rope_tail,        f.raw(pl + "self_attn.q_norm.weight"), n);
    std::memcpy(rope_tail + t.DH, f.raw(pl + "self_attn.k_norm.weight"), n);
}

} // namespace q38npu

#endif  // __QWEN3_8MTP_NPU_MTP_WEIGHTS_HPP__
