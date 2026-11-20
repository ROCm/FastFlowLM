// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __QKV_DECODING_LAYER_HPP__
#define __QKV_DECODING_LAYER_HPP__

#include "models.h"
#include "lm_config.hpp"
#include "npu_utils/npu_utils.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"
#include "utils/utils.hpp"
#include "tensor_2d.hpp"
#include "quant_size.hpp"

#ifndef L_TOKENS
#define L_TOKENS 8
#endif

class decoding_layer {
private:
    model_list::model_t model_type;
    LM_Config& lm_config;
    npu_app_manager* layer_app_mgr;

    npu_app layer_app;

    int x_arg_id = 0;
    int proj_arg_id = 1;
    int rms_arg_id = 2;
    int rope_arg_id = 3;
    int kv_cache_arg_id = 4;

    buffer<bf16> hidden_states_buffer;
    buffer<u8> all_proj_weight_buffer;
    buffer<bf16> rms_weights_buffer;
    buffer<bf16> rope_weights_buffer;
    buffer<bf16> kv_cache;

    int num_hidden_layers;

    void _gen_sequence(npu_sequence* seq);
    void _send_hidden_states(npu_sequence* seq);
    void _send_rms_weights(npu_sequence* seq);
    void _send_rope_weights(npu_sequence* seq);
    void _receive_kv_cache(npu_sequence* seq);
    void _receive_o(npu_sequence* seq);
    void _receive_h2(npu_sequence* seq);
    void _write_attn_L(npu_sequence* seq);
    void _replay_kv_cache(npu_sequence* seq);
    void _move_weights(npu_sequence* seq, const size_t Dout, const size_t Din, const size_t w_offset);

    void _load_rtp_addresses();

    // model specific config
    int D;
    int DH;
    int DQ;
    int DK;
    int DV;
    int DQ_PADDED;        ///< what leaves the RoPE tile, GQA padding included
    int ROPE_DIM_;        ///< head_dim * rope_partial_factor, 64 here
    int num_kv_heads;
    int qkv_out;          ///< DQ + DK + DV, the fused projection's width
    int intermediate_size; ///< the MLP's hidden width, K of down_proj
    size_t o_offset;      ///< where o_proj's output lands in hidden_states_buffer
    size_t down_offset;
    size_t h2_offset;     ///< where the layer output lands, one L x D past down
    size_t w_z_offset;    ///< where the output gate's weights start in all_proj_weight_buffer
    size_t w_o_offset;    ///< where o_proj's weights start, one q4k(DQ x D) past z's
    size_t w_upgate_offset; ///< where the woven up | gate weights start
    size_t w_down_offset;   ///< where down_proj's weights start
    size_t rope_w_offset; ///< where q_norm.weight starts in rope_weights_buffer
    /// The KV cache BO, in bf16 elements: MAX_L * (DK + DV).
    ///
    /// **Token-major, TWO CONTIGUOUS PLANES**: all MAX_L rows of k first, then
    /// all MAX_L rows of v, starting at MAX_L * DK. Row `t`'s k is at `t * DK`
    /// and its v at `MAX_L * DK + t * DV`. The four accessors below are the
    /// single place the layout is expressed -- flip them and every descriptor,
    /// the seeder and the read-back follow.
    ///
    /// The DLL port matches this: `q38npu::attn_layout::k_at/v_at` are these
    /// two accessors, so a descriptor diffed between this bench and the port
    /// agrees on the cache.
    ///
    /// **The cap this layout used to carry, and why it is gone.** The append
    /// once crossed from a row's k to its v inside one BD, a stride of
    /// MAX_L * DK. A shim BD's stride field is 20 bits and bf16 strides are
    /// halved on lowering, so that stopped encoding past MAX_L = 2048 -- and it
    /// did not fault: npu_instr_utils prints "stride_2 out of range" to stderr,
    /// masks the value into 20 bits, and every v row lands at a wrapped
    /// address, so only the *next* pass sees the damage.
    ///
    /// k and v now arrive on two S2MM channels of the column's shim instead of
    /// interleaved on one (iron/RoPE.py stamps them with different packet ids),
    /// so each plane is written by its own BD whose only stride is one row --
    /// DK or DV, constants of the model. Nothing in the append scales with
    /// MAX_L any more. The binding limit is the replay's 10-bit chunk count,
    /// MAX_L <= 1024 * L_CHUNK = 16384, asserted in _replay_kv_cache.
    size_t kv_cache_size;
    /// Elements from one k row to the next, and from one v row to the next.
    /// The planes are walked separately, so each has its own.
    size_t k_row_stride() const { return (size_t)DK; }
    size_t v_row_stride() const { return (size_t)DV; }
    /// Element offset of row `t`'s k.
    size_t k_at(int t) const { return (size_t)t * k_row_stride(); }
    /// Element offset of row `t`'s v -- past the whole k plane.
    size_t v_at(int t) const {
        return (size_t)MAX_L * (size_t)DK + (size_t)t * v_row_stride();
    }

    // ---- attention geometry ------------------------------------------------
    int q_heads_per_group;      ///< 6: what a CU really computes
    int kv_heads_per_column;    ///< 2
    int cus_per_column;         ///< 2
    int attn_col_width;         ///< one column's share of a row of output

    /// Keys are streamed to the CUs this many at a time. Baked into the
    /// kernels -- attn_qk builds its causal mask from a 16-lane vector -- so
    /// this names the constant rather than choosing it. Matches L_CHUNK in
    /// iron/attn.py.
    static constexpr int L_CHUNK = 16;

    /// Per-build RTP addresses, read from rtp_addresses.txt.
    uint32_t l_qk_address = 0;
    uint32_t l_kv_address = 0;

    /// How many positions the cache holds. The replay walks context_offset +
    /// L_TOKENS of them rounded up to L_CHUNK, so this bounds how much past
    /// context a pass can carry.
    int MAX_L = 256;

    /// Absolute position of token 0 -- how many keys are already in the cache
    /// when the pass starts. Token t then attends to context_offset + t + 1
    /// keys, which is what goes into the per-token L RTP array, and the new k/v
    /// are appended at this offset.
    int context_offset = 0;

    /// The layer this bench pins itself to. Qwen3.8-27B is full attention iff
    /// (i + 1) % 4 == 0, so 3 is the first one; the CPU golden in host.cpp reads
    /// the same index.
    int layer_idx = 3;

    // npu configs

    /// The norm tile's shim, column 2: x in on MM2S ch0, both layernorm weight
    /// vectors on ch1. Takes the assembled gated row on S2MM ch0 and the
    /// post-norm layer output on S2MM ch1.
    static constexpr npu_tiles xyr_tile = IT2;
    /// Column 3's shim: the RoPE weight block in on MM2S ch0, o_proj's output
    /// out on S2MM ch1 -- the staging memtile is in this column, directly south
    /// of the rms tile. q and the k | v append no longer pass through here.
    static constexpr npu_tiles attn_tile = IT3;
    /// The two attention columns' shims: the replayed context in on MM2S ch0
    /// (k) and ch1 (v), each column's half of the output back on S2MM ch0.
    static constexpr npu_tiles attn_col_tiles[2] = {IT4, IT5};
    /// The eight attention cores, so the per-token L can be written to each. A
    /// CU is a qk core with its kv core directly north of it -- the qk core
    /// writes the scores into the kv tile's memory. Getting this pair the wrong
    /// way round does not hang; it silently produces zeros.
    static constexpr npu_tiles attn_qk_tiles[4] = {CT04, CT24, CT05, CT25};
    static constexpr npu_tiles attn_kv_tiles[4] = {CT14, CT34, CT15, CT35};
    /// The gate emitter's column shim. One, not two: the two gate tiles are
    /// vertically adjacent in column 2 and the helper writes its half into the
    /// emitter's buffer through the tile boundary, so the whole assembled DQ row
    /// comes back on this single channel.
    static constexpr npu_tiles gate_tile_shim = IT2;
    /// The four projection columns' shims, in engine order.
    static constexpr npu_tiles mvm_tiles[4] = {IT0, IT1, IT6, IT7};

public:
    decoding_layer(model_list::model_t, LM_Config&, npu_app_manager*);

    /// \brief load weights
    /// \param q4nx the quantised checkpoint
    void load_weights(Q4NX&);

    /// \brief Set the per-token rotary table from the reference dump.
    /// \param cos,sin L_TOKENS x ROPE_DIM each, as dumped. Only the first
    ///        ROPE_DIM/2 of each row is used -- the dumped halves are identical,
    ///        and the kernel pairs dim j with dim j + 32.
    void set_rope_table(const std::vector<float>& cos, const std::vector<float>& sin);

    /// \brief Run one pass over L_TOKENS rows.
    /// \param input L_TOKENS x D hidden states, row-major.
    /// \return L_TOKENS x D bf16, token-major: h2, the layer's output and the
    ///         next layer's input.
    buffer<bf16> operator()(buffer<bf16>&);

    /// \brief o_proj's output for the last pass, L_TOKENS x D, token-major.
    ///        The last diagnostic before the residual.
    buffer<bf16> get_o();

    /// \brief down_proj's output for the last pass, L_TOKENS x D, token-major.
    ///        The last diagnostic before the MLP residual.
    buffer<bf16> get_down();

    /// \brief h2 = h1 + down_proj(SwiGLU), L_TOKENS x D, token-major: the
    ///        layer's output and the next layer's input.
    buffer<bf16> get_h2();

    /// \brief How many keys are already in the cache. Token t of the next pass
    ///        attends to `n + t + 1` of them. Any offset works, aligned or not
    ///        -- see the note in the definition. (It once said a multiple of
    ///        L_CHUNK was required; it never was, and the offsets that actually
    ///        misbehaved, `n % L_CHUNK >= L_CHUNK - L_TOKENS + 1`, were a
    ///        kernel bug. See KV_CHUNK_BUG.md.)
    void set_context_offset(int n);

    int context_length() const { return context_offset; }

    /// How many positions the cache holds. A caller that walks the cursor over
    /// several dispatches needs this to stop before set_context_offset's assert.
    int max_context() const { return MAX_L; }

    /// \brief Overwrite the cache's first `n` positions, for testing a pass that
    ///        starts from real past context.
    /// \param k,v n x DK and n x DV, token-major.
    void seed_kv_cache(const std::vector<float>& k, const std::vector<float>& v, int n);

    /// \brief The k rows the array wrote this pass, L_TOKENS x DK, token-major.
    buffer<bf16> get_k();
    /// \brief The v rows the array wrote this pass, L_TOKENS x DV, token-major.
    buffer<bf16> get_v();

    int q_padded_width() const { return DQ_PADDED; }

    void reorder_cpy(u8 *dst, buffer<u8> &src, const int col, const int vertical_blocks = 2);

    /// \brief Steady-state latency: num_hidden_layers passes on one runlist, so
    ///        XRT dispatch is amortised rather than dominating.
    float bench(buffer<bf16>&);
};

#endif
