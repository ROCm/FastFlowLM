// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __SMOLVLA__EXPERT__EXEC__HPP__
#define __SMOLVLA__EXPERT__EXEC__HPP__

#include <memory>
#include <string>
#include <cstring>

#include "buffer.hpp"
#include "typedef.hpp"
#include "SmolVLA_config.hpp"
#include "SmolVLA_expert_def.hpp"
#include "npu_utils/npu_utils.hpp"
#include "cpu_functions.hpp"
#include "utils/prof_utils.hpp"
#include "utils/xclbin_path.hpp"
#include "seq_gen.hpp"


// ---------------------------------------------------------------------------
// SmolVLA action-expert execution contexts.
//
// The expert is a stack of interleaved transformer layers over a fixed-length
// action-token chunk (L == chunk_size):
//   even layers -> self-attention  (q/k/v from the expert hidden state; the
//                  attention keys/values are the expert's own states prepended
//                  with the cached VLM prefix kv, block-causal)
//   odd  layers -> cross-attention (q from the expert hidden state; k/v are the
//                  cached VLM kv states re-projected through this layer's k/v
//                  proj, attended over the whole prefix, non-causal)
//
// Neither layer kind's view of the VLM prefix changes across a denoise run, so
// both hoist it out of the loop into per-layer storage that prepare_prefix_kv()
// fills once (see SmolVLA_expert_cross_attn_block for the RTP-ordering caveat):
//   cross -> the re-projected K/V depend only on the kv cache and the layer
//            weights, so the whole projection is hoisted and only q is per-step.
//   self  -> the first P rows of the [prefix ; chunk] K/V are a copy of the kv
//            cache, so they are written once and each step overwrites only the
//            L rows appended after them.
// Between them that removes 8 mm launches, 32 prefix row-copies and 32 full-
// buffer host->device syncs from every one of the 10 denoise steps.
// Every layer is followed by a gated-MLP (SwiGLU). Q/K get RoPE with a position
// offset equal to the VLM prefix length P (the action chunk sits after the
// prefix). The VLM keys are cached post-rope, the values pre-rope.
//
// Weights are stored padded to PAD_ALIGN, so every GEMM passes the padded input
// dim as _cpu_mm's w_stride while K/N stay real.
//
// Every expert ACTIVATION is padded the same way: a [L, pX] buffer holds X real
// columns per row with zeros to the right. So each kernel takes the padded dim as
// a row stride while the logical extent (K, N, D) stays real:
//   _cpu_mm    -> w_stride / x_stride / y_stride
//   _rms_norm  -> y_stride / x_stride
//   _rope      -> row_stride
//   _gqa_mha   -> q_stride / kv_stride / out_stride
// Nothing ever accumulates over the padding, so the zeros written once at
// construction stay valid for the life of the buffer.
//
// The two boundaries are unpadded by necessity: `residual` (the chunk the caller
// owns, [L, D]) and the VLM kv cache (`vlm_k`/`vlm_v`, [P, DK]), which the
// language tower writes packed. Both are named explicitly at their call sites.
// ---------------------------------------------------------------------------

// scratch sizing: the action chunk and the VLM prefix it attends over are both
// bounded by smolvla_cfg, so every block below allocates once at its worst case.
static constexpr int EXPERT_MAX_CHUNK  = smolvla_cfg::MAX_ACTION_CHUNK;
static constexpr int EXPERT_MAX_PREFIX = smolvla_cfg::MAX_PREFIX_LEN;
static constexpr int EXPERT_MAX_KV     = smolvla_cfg::MAX_EXPERT_KV;
/// @brief allocate a padded activation buffer and zero it once.
///
/// Every expert activation is [rows, padded_cols] with real values only in the
/// leading columns. No kernel writes past the real width, so clearing the whole
/// thing here is what keeps the padding zero for the life of the buffer.
static inline buffer<bf16> _padded_activation(size_t rows, size_t padded_cols) {
    buffer<bf16> b(rows * padded_cols);
    memset(b.data(), 0, b.size() * sizeof(bf16));
    return b;
}
/// @brief copy an unpadded [rows, cols] source into a padded [rows, dst_stride]
///        destination, leaving each row's trailing columns as they were (zero).
static inline void _copy_rows_padded(bf16* dst, const bf16* src,
                                     size_t rows, size_t cols, size_t dst_stride) {
    for (size_t r = 0; r < rows; r++)
        memcpy(dst + r * dst_stride, src + r * cols, cols * sizeof(bf16));
}

/// @brief push only rows [first_row, first_row + rows) of a [*, row_stride]
///        buffer to the device.
///
/// A full sync_to_device() on a [MAX_KV, pDK] buffer moves the whole 320 KB even
/// when the step only rewrote the 50-row tail. Rows are contiguous at the padded
/// stride, so a row range is one contiguous byte range and xrt::bo::sync takes
/// the (size, offset) pair directly.
static inline void _sync_rows_to_device(buffer<bf16>& b, size_t first_row,
                                        size_t rows, size_t row_stride) {
    if (rows == 0) return;
    const size_t elem_off = first_row * row_stride;
    const size_t nbytes   = rows * row_stride * sizeof(bf16);
    assert((elem_off + rows * row_stride) <= b.size());
    b.bo().sync(XCL_BO_SYNC_BO_TO_DEVICE, nbytes, elem_off * sizeof(bf16));
}

/// @brief self-attention block (even layers). q/k/v projected from the expert
///        hidden state; attention runs over [VLM prefix kv ; expert kv],
///        block-causal (each action token sees the full prefix + causally the
///        preceding action tokens).
///
/// The K/V the mha reads is [VLM prefix (P rows) ; expert chunk (L rows)]. Only
/// the L appended rows are a function of the denoise step -- the P prefix rows
/// are a straight re-spacing of this layer's VLM kv cache into the padded row
/// stride, and that cache is frozen once the prefill has run. So the prefix is
/// written once per (prefix, layer) by prepare_kv() and every step touches only
/// the tail. With P == 256 and L == 50 that is the difference between copying
/// and syncing 320 rows per layer per step and copying 50.
///
/// Hoisting it means the prefix cannot live in one shared buffer any more: each
/// self layer reads a different slice of the kv cache, so each needs its own
/// K/V to keep its prefix across the run -- hence k_layers/v_layers, indexed by
/// the layer's slot exactly like the cross block's.
struct SmolVLA_expert_self_attn_block {
    SmolVLA_Expert_def* def;

    npu_app q_proj;
    npu_app k_proj;
    npu_app v_proj;
    npu_app o_proj;
    npu_app mha;

    buffer<bf16> q;      // [MAX_CHUNK, pDQ]
    buffer<bf16> k;      // [MAX_CHUNK, pDK]
    buffer<bf16> v;      // [MAX_CHUNK, pDV]
    buffer<bf16> attn;   // [MAX_CHUNK, pDQ]
    buffer<bf16> o;      // [MAX_CHUNK, pD]

    // Per-self-layer [VLM prefix ; expert chunk] K/V. The prefix rows are filled
    // by prepare_kv() and then left alone; forward() rewrites only the tail.
    std::vector<buffer<bf16>> k_layers;  // [num_self][MAX_KV, pDK]
    std::vector<buffer<bf16>> v_layers;  // [num_self][MAX_KV, pDV]

    xrt::runlist rl;
    // prefix length the cached mha sequence was built for; -1 = not yet built.
    int last_P = -1;
    // prefix length whose rows are currently resident in k_layers/v_layers, and
    // the tail row each step writes at (num_blocks * 64). Set by prepare_kv().
    int prepared_P = -1;
    int tail_row   = 0;

    SmolVLA_expert_self_attn_block(
        SmolVLA_Expert_def* def,
        npu_app_manager* expert_app_manager
    ) : def(def) {

        this->q_proj = expert_app_manager->create_app();
        this->k_proj = expert_app_manager->create_app();
        this->v_proj = expert_app_manager->create_app();
        this->o_proj = expert_app_manager->create_app();
        this->mha    = expert_app_manager->create_app();

        q     = this->q_proj.create_bo_buffer<bf16>(EXPERT_MAX_CHUNK * def->pDQ);
        k     = this->k_proj.create_bo_buffer<bf16>(EXPERT_MAX_CHUNK * def->pDK);
        v     = this->v_proj.create_bo_buffer<bf16>(EXPERT_MAX_CHUNK * def->pDV);
        attn  = this->o_proj.create_bo_buffer<bf16>(EXPERT_MAX_CHUNK * def->pDQ);
        o     = this->o_proj.create_bo_buffer<bf16>(EXPERT_MAX_CHUNK * def->pD);

        memset(q.data(), 0, q.size() * sizeof(bf16));
        memset(k.data(), 0, k.size() * sizeof(bf16));
        memset(v.data(), 0, v.size() * sizeof(bf16));
        memset(attn.data(), 0, attn.size() * sizeof(bf16));
        memset(o.data(), 0, o.size() * sizeof(bf16));

        // one K/V slot per self (even) layer; see the cross block's ctor for why
        // these are moved into place rather than resize()d.
        const uint32_t num_self = (def->num_layers + smolvla_cfg::SELF_ATTN_EVERY_N_LAYERS - 1)
                                  / smolvla_cfg::SELF_ATTN_EVERY_N_LAYERS;
        k_layers.resize(num_self);
        v_layers.resize(num_self);
        for (uint32_t s = 0; s < num_self; s++) {
            k_layers[s] = this->o_proj.create_bo_buffer<bf16>(EXPERT_MAX_KV * def->pDK);
            v_layers[s] = this->o_proj.create_bo_buffer<bf16>(EXPERT_MAX_KV * def->pDV);
            memset(k_layers[s].data(), 0, k_layers[s].size() * sizeof(bf16));
            memset(v_layers[s].data(), 0, v_layers[s].size() * sizeof(bf16));
            k_layers[s].sync_to_device();
            v_layers[s].sync_to_device();
        }

        rl = expert_app_manager->create_runlist();
    }

    /// @brief build the four mm sequences. Split out of the constructor because
    ///        the mm generators drop RTP writes that the previously executing
    ///        sequence already left in the tiles, which only works if generation
    ///        runs in execution order -- see the note in seq_gen.hpp. The owner
    ///        calls this on self, then mlp, then cross.
    void gen_seqs() {
        smolvla_expert_seq::gen_expert_mm_seq(this->q_proj.seq(), EXPERT_MAX_CHUNK, def->self_layer_def.q_proj, 0);
        smolvla_expert_seq::gen_expert_mm_seq(this->k_proj.seq(), EXPERT_MAX_CHUNK, def->self_layer_def.k_proj, 0);
        smolvla_expert_seq::gen_expert_mm_seq(this->v_proj.seq(), EXPERT_MAX_CHUNK, def->self_layer_def.v_proj, 0);
        smolvla_expert_seq::gen_expert_mm_seq(this->o_proj.seq(), EXPERT_MAX_CHUNK, def->self_layer_def.o_proj, 0);
    }

    /// @brief write one self layer's VLM prefix rows into its persistent K/V.
    ///        Call once per (prefix, layer) before the denoise loop.
    ///
    /// The mha reads K/V as ceil(P/64) blocks of 64 rows followed by the expert's
    /// own L rows, so the prefix occupies rows [0, P) and the tail starts at
    /// num_blocks*64 -- the gap rows in between keep the zeros from construction.
    /// Only the prefix is written here; forward() fills the tail each step.
    ///
    /// @param slot     index of this layer among the self layers (l / 2)
    /// @param vlm_k/v  this layer's cached VLM kv states, [P, DK]/[P, DV]
    /// @param P        VLM prefix length
    void prepare_kv(uint32_t slot, bf16* vlm_k, bf16* vlm_v, int P) {
        const int DK  = (int)def->DK,  DV  = (int)def->DV;
        const int pDK = (int)def->pDK, pDV = (int)def->pDV;
        const int tail = ((P + 63) / 64) * 64;

        // The VLM cache is packed [P, DK]; the mha reads a padded row stride, so
        // the rows are re-spaced on the way in.
        _copy_rows_padded(k_layers[slot].data(), vlm_k, P, DK, pDK);
        _copy_rows_padded(v_layers[slot].data(), vlm_v, P, DV, pDV);
        // Rows [P, tail) round the prefix up to the block the mha reads in. They
        // are zero at construction, but a second prepare over a shorter prefix
        // would otherwise leave the previous one's rows exposed there, so clear
        // them rather than rely on the last-block masking to skip them.
        memset(k_layers[slot].data() + (size_t)P * pDK, 0, (size_t)(tail - P) * pDK * sizeof(bf16));
        memset(v_layers[slot].data() + (size_t)P * pDV, 0, (size_t)(tail - P) * pDV * sizeof(bf16));
        // Only the prefix rows changed, so only they go to the device. The tail
        // rows are pushed per step by forward().
        _sync_rows_to_device(k_layers[slot], 0, tail, pDK);
        _sync_rows_to_device(v_layers[slot], 0, tail, pDV);

        prepared_P = P;
        tail_row   = tail;
    }

    /// @param L      action chunk length
    /// @param input  rms-normed expert hidden state, [L, pD] (D real columns)
    /// @param weights packed per-layer weight buffer (self layer)
    /// @param ld     this layer's weight descriptors (self_layer_def)
    /// @param slot   index of this layer among the self layers, selecting the
    ///               K/V whose prefix rows prepare_kv() already filled
    /// @param P      VLM prefix length
    /// @return attention output projected back to model dim, [L, pD]
    buffer<bf16> forward(int L, buffer<bf16>& input, buffer<bf16>& weights,
                         SmolVLA_expert_layer& ld, const float* inv_freq,
                         uint32_t slot, int P) {
        const int pDQ = (int)def->pDQ, pDK = (int)def->pDK, pDV = (int)def->pDV;

        int num_blocks = (P + 63) / 64;
        int effect = P - (P / 64) * 64;

        // the prefix rows in k_layers/v_layers have to be the ones this P names.
        assert(prepared_P == P && "self_attn::prepare_kv not run for this prefix");

        // The attention sequence depends on the shapes only through P, so it is
        // identical for every self layer and every denoise step of a run. Build
        // it on the first call and whenever the prefix length changes.
        //
        // force_rtp is unconditional here: a self layer is always layer 0, so
        // this is the first sequence to run after the switch to expert.xclbin,
        // which wipes whatever the tiles were holding. Tracking would elide
        // exactly the writes needed to repopulate them.
        if (P != last_P) {
            smolvla_expert_seq::gen_expert_attn_seq(this->mha.seq(), def->head_dim, pDQ, pDK,
                                                    num_blocks, effect, 1, /*force_rtp=*/true);
            last_P = P;
        }
        buffer<bf16>& k_cat = k_layers[slot];
        buffer<bf16>& v_cat = v_layers[slot];

        PROF_INIT();
        rl.reset();
        rl.add(q_proj.create_run(q, input, weights));
        rl.add(v_proj.create_run(v, input, weights));
        rl.add(k_proj.create_run(k, input, weights));
        rl.execute();
        rl.wait();
        q.sync_from_device();
        k.sync_from_device();
        v.sync_from_device();
        cpu_func::_rope(q.data(), L, def->num_attn_heads, def->head_dim, inv_freq, P, pDQ);
        q.sync_to_device();
        // K/V = [VLM prefix (rows 0..P) ; expert (L rows at tail_row)]. The
        // prefix half is already in place and already on the device from
        // prepare_kv(), so only the expert's own rows are copied and synced --
        // they are at the padded stride already, so they go as one block.
        memcpy(k_cat.data() + (size_t)tail_row * pDK, k.data(), (size_t)L * pDK * sizeof(bf16));
        memcpy(v_cat.data() + (size_t)tail_row * pDV, v.data(), (size_t)L * pDV * sizeof(bf16));
        cpu_func::_rope(k_cat.data() + (size_t)tail_row * pDK, L, def->num_kv_heads, def->head_dim, inv_freq, P, pDK);
        _sync_rows_to_device(k_cat, tail_row, L, pDK);
        _sync_rows_to_device(v_cat, tail_row, L, pDV);
        PROF_TICK("expert.self_attn:qkv");

        mha(attn, q, k_cat, v_cat);

        PROF_TICK("expert.self_attn:mha");

        o_proj(o, attn, weights);
        o.sync_from_device();
        PROF_TICK("expert.self_attn:o_proj");
        return o;
    }
};

/// @brief cross-attention block (odd layers). q from the expert hidden state;
///        k/v are the cached VLM kv states re-projected through this layer's
///        k/v proj (DK->DK, DV->DV). Keys are NOT re-roped (the cache is already
///        post-rope). Attends over the full prefix, non-causal.
///
/// The re-projected K/V are a function of (VLM kv cache, layer weights) only --
/// nothing in them depends on the denoise step, which enters solely through q.
/// Both inputs are frozen once the VLM prefill has run, so prepare_prefix_kv()
/// projects every cross layer's K/V once per prefix into k_layers/v_layers and
/// forward() just points the mha at the right slot. The per-step cost of a cross
/// layer drops from three mm launches plus two prefix copies and two host->device
/// syncs to a single q_proj.
struct SmolVLA_expert_cross_attn_block {
    SmolVLA_Expert_def* def;

    npu_app q_proj;
    npu_app k_proj;
    npu_app v_proj;
    npu_app o_proj;
    npu_app mha;

    buffer<bf16> q;     // [MAX_CHUNK, pDQ]
    buffer<bf16> k_cat;     // [MAX_PREFIX, pDK]  staging: one layer's prefix keys
    buffer<bf16> v_cat;     // [MAX_PREFIX, pDV]
    buffer<bf16> attn;  // [MAX_CHUNK, pDQ]
    buffer<bf16> o;     // [MAX_CHUNK, pD]

    // Per-cross-layer re-projected prefix K/V, filled by prepare_kv() and read by
    // every denoise step. bo-backed because the mha kernel reads them directly.
    // k_proj/v_proj write them on the device and the host never reads them back,
    // so they stay device-resident for the whole run -- no sync per step.
    std::vector<buffer<bf16>> k_layers;  // [num_cross][MAX_PREFIX, pDK]
    std::vector<buffer<bf16>> v_layers;  // [num_cross][MAX_PREFIX, pDV]

    xrt::runlist rl;

    // prefix length the cached mha sequence was built for; -1 = not yet built.
    int last_P = -1;

    SmolVLA_expert_cross_attn_block(
        SmolVLA_Expert_def* def,
        npu_app_manager* expert_app_manager
    ) : def(def) {

        this->q_proj = expert_app_manager->create_app();
        this->k_proj = expert_app_manager->create_app();
        this->v_proj = expert_app_manager->create_app();
        this->o_proj = expert_app_manager->create_app();
        this->mha    = expert_app_manager->create_app();

        q     = this->q_proj.create_bo_buffer<bf16>(EXPERT_MAX_CHUNK * def->pDQ);
        k_cat = this->k_proj.create_bo_buffer<bf16>(EXPERT_MAX_PREFIX * def->pDK);
        v_cat = this->v_proj.create_bo_buffer<bf16>(EXPERT_MAX_PREFIX * def->pDV);
        attn  = this->mha.create_bo_buffer<bf16>(EXPERT_MAX_CHUNK * def->pDQ);
        o     = this->o_proj.create_bo_buffer<bf16>(EXPERT_MAX_CHUNK * def->pD);

        memset(q.data(), 0, q.size() * sizeof(bf16));
        memset(k_cat.data(), 0, k_cat.size() * sizeof(bf16));
        memset(v_cat.data(), 0, v_cat.size() * sizeof(bf16));
        memset(attn.data(), 0, attn.size() * sizeof(bf16));
        memset(o.data(), 0, o.size() * sizeof(bf16));

        // one K/V slot per cross (odd) layer. buffer<T> has no move ctor that
        // resize() can use (its copy ctor is shallow), so size the vectors with
        // empty buffers first and move each bo-backed buffer into place -- the
        // same dance SmolVLA_expert::alloc_weights does.
        const uint32_t num_cross = def->num_layers / smolvla_cfg::SELF_ATTN_EVERY_N_LAYERS;
        k_layers.resize(num_cross);
        v_layers.resize(num_cross);
        for (uint32_t c = 0; c < num_cross; c++) {
            k_layers[c] = this->k_proj.create_bo_buffer<bf16>(EXPERT_MAX_PREFIX * def->pDK);
            v_layers[c] = this->v_proj.create_bo_buffer<bf16>(EXPERT_MAX_PREFIX * def->pDV);
            memset(k_layers[c].data(), 0, k_layers[c].size() * sizeof(bf16));
            memset(v_layers[c].data(), 0, v_layers[c].size() * sizeof(bf16));
            k_layers[c].sync_to_device();
            v_layers[c].sync_to_device();
        }

        rl = expert_app_manager->create_runlist();
    }

    /// @brief build the four mm sequences; see the self block's gen_seqs().
    ///
    /// k_proj is forced to emit all its RTPs. In the steady-state layer cycle it
    /// followed q_c and could inherit q_c's act=0, but the kv prologue now runs
    /// k_proj/v_proj before any other expert mm -- right after the switch to
    /// expert.xclbin wipes the tiles -- so there is no predecessor to inherit
    /// from. v_proj still inherits from k_proj, which immediately precedes it in
    /// both the prologue and the generation order, and o_proj re-sends the one
    /// slot (K_div_k) that differs from its new predecessor q_c anyway.
    void gen_seqs() {
        smolvla_expert_seq::gen_expert_mm_seq(this->q_proj.seq(), EXPERT_MAX_CHUNK, def->cross_layer_def.q_proj, 0);
        smolvla_expert_seq::gen_expert_mm_seq(this->k_proj.seq(), EXPERT_MAX_PREFIX, def->cross_layer_def.k_proj, 0, /*force_rtp=*/true);
        smolvla_expert_seq::gen_expert_mm_seq(this->v_proj.seq(), EXPERT_MAX_PREFIX, def->cross_layer_def.v_proj, 0);
        smolvla_expert_seq::gen_expert_mm_seq(this->o_proj.seq(), EXPERT_MAX_CHUNK, def->cross_layer_def.o_proj, 0);
    }

    /// @brief re-project one cross layer's VLM prefix kv into its persistent
    ///        K/V slot. Call once per (prefix, layer) before the denoise loop.
    /// @param slot     index of this layer among the cross layers (l / 2)
    /// @param weights  this cross layer's packed weight buffer
    /// @param vlm_k/v  this layer's cached VLM kv states, [P, DK]/[P, DV]
    /// @param P        VLM prefix length
    void prepare_kv(uint32_t slot, buffer<bf16>& weights,
                    bf16* vlm_k, bf16* vlm_v, int P) {
        const int DK  = (int)def->DK,  DV  = (int)def->DV;
        const int pDK = (int)def->pDK, pDV = (int)def->pDV;

        // The VLM cache is packed [P, DK]; the mm reads a padded row stride, so
        // the rows are re-spaced into k_cat/v_cat on the way in. Rows past P keep
        // the zeros from construction, and the mha never reads them (its sequence
        // is built for exactly P).
        _copy_rows_padded(k_cat.data(), vlm_k, P, DK, pDK);
        _copy_rows_padded(v_cat.data(), vlm_v, P, DV, pDV);
        k_cat.sync_to_device();
        v_cat.sync_to_device();

        rl.reset();
        rl.add(k_proj.create_run(k_layers[slot], k_cat, weights));
        rl.add(v_proj.create_run(v_layers[slot], v_cat, weights));
        rl.execute();
        rl.wait();
        // the results stay on the device for the mha to read; no sync back.
    }

    /// @param slot index of this layer among the cross layers, selecting the K/V
    ///             prepare_kv() projected for it.
    buffer<bf16> forward(int L, buffer<bf16>& input, buffer<bf16>& weights,
                         SmolVLA_expert_layer& ld, const float* inv_freq,
                         uint32_t slot, int P) {
        const int pDQ = (int)def->pDQ, pDK = (int)def->pDK;

        int num_blocks = (P + 63) / 64;
        int effect = P - (P / 64) * 64;

        // Cached on P like the self block's. No force_rtp: the self sequence
        // always runs first after the xclbin switch and repopulates the tiles,
        // so the only slot this one has to correct is is_self -- which is
        // exactly what the tracker in gen_expert_attn_seq leaves it emitting.
        if (P != last_P) {
            smolvla_expert_seq::gen_expert_attn_seq(this->mha.seq(), def->head_dim, pDQ, pDK,
                                                    num_blocks, effect, 0);
            last_P = P;
        }

        PROF_INIT();
        // K/V were re-projected once per prefix by prepare_kv(); only q depends
        // on the denoise step, so it is all that runs here.
        q_proj(q, input, weights);
        q.sync_from_device();
        // Cross layers rope the query at positions 0..L-1, NOT offset by the VLM
        // prefix P (the reference subtracts the prefix min so the chunk "starts
        // from 0"). Self layers use offset P; cross layers use 0.
        cpu_func::_rope(q.data(), L, def->num_attn_heads, def->head_dim, inv_freq, 0, pDQ);
        q.sync_to_device();
        PROF_TICK("expert.cross_attn:q+rope");

        mha(attn, q, k_layers[slot], v_layers[slot]);
        PROF_TICK("expert.cross_attn:mha");

        o_proj(o, attn, weights);
        o.sync_from_device();
        PROF_TICK("expert.cross_attn:o_proj");
        return o;
    }
};

/// @brief gated-MLP block (shared by every layer): SwiGLU over gate/up_proj then
///        down_proj.
struct SmolVLA_expert_mlp_block {
    SmolVLA_Expert_def* def;

    npu_app gate_proj;
    npu_app up_proj;
    npu_app down_proj;

    buffer<bf16> gate;  // [MAX_CHUNK, pinter]
    buffer<bf16> up;    // [MAX_CHUNK, pinter]
    buffer<bf16> hid;   // [MAX_CHUNK, pinter]
    buffer<bf16> o;     // [MAX_CHUNK, pD]

    xrt::runlist rl;

    SmolVLA_expert_mlp_block(
        SmolVLA_Expert_def* def,
        npu_app_manager* expert_app_manager
    ) : def(def) {
        
        this->gate_proj = expert_app_manager->create_app();
        this->up_proj = expert_app_manager->create_app();
        this->down_proj = expert_app_manager->create_app();

        gate = this->gate_proj.create_bo_buffer<bf16>(EXPERT_MAX_CHUNK * def->pintermediate_size);
        up   = this->up_proj.create_bo_buffer<bf16>(EXPERT_MAX_CHUNK * def->pintermediate_size);
        hid  = this->down_proj.create_bo_buffer<bf16>(EXPERT_MAX_CHUNK * def->pintermediate_size);
        o    = this->down_proj.create_bo_buffer<bf16>(EXPERT_MAX_CHUNK * def->pD);
        memset(gate.data(), 0, gate.size() * sizeof(bf16));
        memset(up.data(), 0, up.size() * sizeof(bf16));
        memset(hid.data(), 0, hid.size() * sizeof(bf16));
        memset(o.data(), 0, o.size() * sizeof(bf16));

        rl = expert_app_manager->create_runlist();
    }

    /// @brief build the three mm sequences; see the self block's gen_seqs().
    ///        One mlp block serves both layer kinds, so these sequences run with
    ///        either o_proj as predecessor -- sound only because the self and
    ///        cross o_proj are the same shape with the same act, so they leave
    ///        the same mm RTP state behind.
    void gen_seqs() {
        smolvla_expert_seq::gen_expert_mm_seq(this->gate_proj.seq(), EXPERT_MAX_CHUNK, def->mlp_def.gate_proj, 1);
        smolvla_expert_seq::gen_expert_mm_seq(this->up_proj.seq(), EXPERT_MAX_CHUNK, def->mlp_def.up_proj, 0);
        smolvla_expert_seq::gen_expert_mm_seq(this->down_proj.seq(), EXPERT_MAX_CHUNK, def->mlp_def.down_proj, 0);
    }

    buffer<bf16> forward(int L, buffer<bf16>& input, buffer<bf16>& weights) {
        SmolVLA_expert_mlp& ld = def->mlp_def;
        const int D = (int)def->D, inter = (int)def->intermediate_size;
        const int pD = (int)def->pD, pinter = (int)def->pintermediate_size;

        PROF_INIT();
        // input is [L, pD]; gate/up are [L, pinter].
        rl.reset();
        rl.add(gate_proj.create_run(gate, input, weights));
        rl.add(up_proj.create_run(up, input, weights));
        rl.execute();
        rl.wait();
        gate.sync_from_device();
        up.sync_from_device(); 
        PROF_TICK("expert.mlp:gate+up");

        const int act_len = L * pinter;
        cpu_func::_elementwise_mul(hid.data(), gate.data(), up.data(), act_len);
        PROF_TICK("expert.mlp:swiglu");
        
        hid.sync_to_device();
        down_proj(o, hid, weights);
        o.sync_from_device();
        PROF_TICK("expert.mlp:down");
        return o;
    }
};

/// @brief top-level driver: runs the full interleaved expert stack over a single
///        action-token chunk for one denoising step.
struct SmolVLA_expert_exec {
    SmolVLA_Expert_def* def;

    std::unique_ptr<SmolVLA_expert_self_attn_block>  self_attn;
    std::unique_ptr<SmolVLA_expert_cross_attn_block> cross_attn;
    std::unique_ptr<SmolVLA_expert_mlp_block>        mlp;

    // RoPE inverse frequencies for head_dim, theta=1e4 (LLaMA rotate_half).
    // Same schedule the VLM text tower uses; shared by self and cross attention.
    std::vector<float> inv_freq;

    // xclbin app managers
    npu_app_manager* fused_xclbin = nullptr;

    buffer<bf16> x;   // rms-normed hidden state fed into each block, [MAX_CHUNK, D]

    SmolVLA_expert_exec() {}
    SmolVLA_expert_exec(npu_xclbin_manager* npu, SmolVLA_Expert_def* def)
        : def(def) {
        this->inv_freq = cpu_func::_rope_inv_freq(def->head_dim, smolvla_cfg::ROPE_THETA);
        x = npu->create_bo_buffer<bf16>(EXPERT_MAX_CHUNK * def->pD);
        x.memset((bf16)(0.0f));
        // register xclbin
        this->fused_xclbin = npu->register_xclbin(utils::xclbin_path("expert.xclbin"));
        this->self_attn  = std::make_unique<SmolVLA_expert_self_attn_block>(def, fused_xclbin);
        this->cross_attn = std::make_unique<SmolVLA_expert_cross_attn_block>(def, fused_xclbin);
        this->mlp        = std::make_unique<SmolVLA_expert_mlp_block>(def, fused_xclbin);

        // The mm sequences elide RTP writes whose value the previous sequence
        // already left in the tiles, so they must be generated in the order they
        // execute -- self attn, mlp, cross attn, mlp -- not in construction
        // order. The trailing mlp repeat is implicit: its sequences are built
        // once and the self/cross o_proj leave identical state (see
        // SmolVLA_expert_mlp_block::gen_seqs).
        this->self_attn->gen_seqs();
        this->mlp->gen_seqs();
        this->cross_attn->gen_seqs();
    }

    /// @brief hoist every layer's view of the VLM prefix out of the denoise loop,
    ///        once per prefix. Cross layers get their kv re-projected through the
    ///        layer's k/v proj; self layers get the prefix half of their
    ///        [prefix ; chunk] K/V filled. Both are frozen for the whole denoise
    ///        run (see the header note), so this must be called after the VLM
    ///        prefill and before the first forward() over that prefix.
    /// @param cross_weights one packed weight buffer per cross (odd) layer
    /// @param kv_cache      VLM kv cache, (2, num_vlm_layers, MAX_L, DK)
    /// @param P             valid VLM prefix length
    /// @param num_vlm_layers layer count (== expert num_layers)
    /// @param MAX_L         cache row stride per layer
    void prepare_prefix_kv(std::vector<buffer<bf16>>& cross_weights,
                           buffer<bf16>& kv_cache, int P, int num_vlm_layers, int MAX_L) {
        const int DK = (int)def->DK;
        const size_t layer_stride = (size_t)MAX_L * DK;                 // DK == DV
        const size_t value_base   = (size_t)num_vlm_layers * layer_stride;

        PROF_INIT();
        for (uint32_t l = 0; l < def->num_layers; l++) {
            const uint32_t slot = SmolVLA_Expert_def::layer_slot(l);
            bf16* vlm_k = kv_cache.data() + (size_t)l * layer_stride;
            bf16* vlm_v = kv_cache.data() + value_base + (size_t)l * layer_stride;
            if (def->is_self_layer(l)) {
                self_attn->prepare_kv(slot, vlm_k, vlm_v, P);
            } else {
                cross_attn->prepare_kv(slot, cross_weights[slot], vlm_k, vlm_v, P);
            }
        }
        PROF_TICK("expert.prefix_kv_prep");
    }

    /// @brief run one expert layer, updating residual in place.
    /// @param is_self   true for self (even) layers, false for cross (odd)
    /// @param residual  running hidden state, [L, D] (unpadded: the caller's chunk)
    /// @param L         chunk length
    /// @param weights   this layer's packed weight buffer
    /// @param ld        this layer's weight descriptors
    /// @param slot      this layer's index within its own kind (l / 2). Selects
    ///                  the K/V prepare_prefix_kv() set up for this layer --
    ///                  re-projected for cross layers, prefix-filled for self.
    /// @param P         VLM prefix length
    buffer<bf16> forward(bool is_self, buffer<bf16>& residual, int L,
                         buffer<bf16>& weights, SmolVLA_expert_layer& ld,
                         uint32_t slot, int P) {
        constexpr float eps = smolvla_cfg::EXPERT_NORM_EPS;
        const int D = (int)def->D, pD = (int)def->pD;

        PROF_INIT();
        // residual is the caller's unpadded [L, D] chunk; everything the blocks
        // produce is padded [L, pD]. Only the strides differ, D stays real.
        buffer<bf16> ln1_w = ld.input_layernorm.locate_myself<bf16>(weights);
        cpu_func::_rms_norm(x, residual, ln1_w, L, D, eps, pD);
        x.sync_to_device();
        PROF_TICK("expert.layer:rms_norm1");
        buffer<bf16> attn_out = is_self
            ? self_attn->forward(L, x, weights, ld, inv_freq.data(), slot, P)
            : cross_attn->forward(L, x, weights, ld, inv_freq.data(), slot, P);
        cpu_func::_residual_add(residual.data(), attn_out.data(), residual.data(),
                                L, D, /*dest=*/D, /*src=*/pD, /*res=*/D);
        // self and cross layers alternate; keep them in separate rows so their
        // (quite different) costs do not average together.
        PROF_TICK(is_self ? "expert.layer:self_attn+residual"
                          : "expert.layer:cross_attn+residual");

        buffer<bf16> ln2_w = ld.post_attention_layernorm.locate_myself<bf16>(weights);
        cpu_func::_rms_norm(x, residual, ln2_w, L, D, eps, pD);
        PROF_TICK("expert.layer:rms_norm2");
        x.sync_to_device();
        buffer<bf16> mlp_out = mlp->forward(L, x, weights);
        cpu_func::_residual_add(residual.data(), mlp_out.data(), residual.data(),
                                L, D, /*dest=*/D, /*src=*/pD, /*res=*/D);
        PROF_TICK("expert.layer:mlp+residual");
        return residual;
    }

    /// @brief run the full interleaved expert stack for one denoising step.
    /// @param chunk         action-token hidden state, [L, D] (mutated in place)
    /// @param L             chunk length
    /// @param self_weights  one packed buffer per self (even) layer
    /// @param cross_weights one packed buffer per cross (odd) layer
    /// @param non_layer_weights top-level expert weights (holds expert_norm)
    /// @param P             valid VLM prefix length
    /// @return the chunk after the final expert norm, [L, D]
    /// @note the VLM kv cache is not read here: both layer kinds take their view
    ///       of the prefix from what prepare_prefix_kv() left in their slot.
    buffer<bf16> forward(buffer<bf16>& chunk, int L,
                         std::vector<buffer<bf16>>& self_weights,
                         std::vector<buffer<bf16>>& cross_weights,
                         buffer<u8>& non_layer_weights, int P) {
        for (uint32_t l = 0; l < def->num_layers; l++) {
            const bool is_self = def->is_self_layer(l);
            const uint32_t slot = SmolVLA_Expert_def::layer_slot(l);
            buffer<bf16>& w = is_self ? self_weights[slot] : cross_weights[slot];
            SmolVLA_expert_layer& ld = is_self
                ? (SmolVLA_expert_layer&)def->self_layer_def
                : (SmolVLA_expert_layer&)def->cross_layer_def;
            forward(is_self, chunk, L, w, ld, slot, P);
        }
        PROF_INIT();

        // final expert norm (top-level lm_expert weight).
        buffer<bf16> norm_w = def->expert_norm.locate_myself<bf16>(non_layer_weights);
        cpu_func::_rms_norm(chunk, chunk, norm_w, L, (int)def->D, smolvla_cfg::EXPERT_NORM_EPS);
        PROF_TICK("expert.final_norm");
        return chunk;
    }
};

#endif // __SMOLVLA__EXPERT__EXEC__HPP__
