// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __SMOLVLA__LANGUAGE__EXEC__HPP__
#define __SMOLVLA__LANGUAGE__EXEC__HPP__

#include <memory>
#include <string>

#include "buffer.hpp"
#include "typedef.hpp"
#include "SmolVLA_config.hpp"
#include "SmolVLA_language_def.hpp"
#include "npu_utils/npu_utils.hpp"
#include "utils/prof_utils.hpp"
#include "utils/xclbin_path.hpp"
#include "cpu_functions.hpp"
#include "seq_gen.hpp"

// ---------------------------------------------------------------------------
// SmolVLA text-model (SmolVLM2 language tower) execution contexts.
//
// The text model is a stack of identical LLaMA-style pre-norm transformer
// layers over the multimodal token sequence (image tokens from the connector +
// language tokens). Each layer is GQA self-attention followed by a gated
// SwiGLU MLP. During SmolVLA prefill the per-layer key/value states are cached
// so the action expert's cross-attention blocks can attend over them.
//
// All activations flowing between blocks are buffer<bf16>. Each block owns its
// npu_app handles + scratch buffers and exposes a forward() that takes the
// (pre-normed) hidden state and returns the block output (pre residual-add).
// ---------------------------------------------------------------------------


/// @brief GQA self-attention block. Separate q/k/v/o projections; keys/values
///        are cached per layer for the expert's cross-attention.
struct SmolVLA_language_attn_block {
    static constexpr int MAX_L = smolvla_cfg::MAX_PREFIX_LEN;

    // row tile of the text mm NPU sequence (gen_text_mm_seq's `m`); the M it is
    // handed has to be a whole number of these.
    static constexpr int MM_TILE_M = 64;

    SmolVLA_Language_def* def;

    // RoPE inverse frequencies (LLaMA rotate_half, theta=1e4, no q/k-norm).
    std::vector<float> inv_freq;

    npu_app q_proj;
    npu_app k_proj;
    npu_app v_proj;
    npu_app o_proj;
    npu_app mha;

    int last_L = -1;

    buffer<bf16> q;
    buffer<bf16> k;
    buffer<bf16> v;
    buffer<bf16> attn;
    buffer<bf16> o;

    buffer<bf16> temp_attn_out;

    SmolVLA_language_attn_block(
        SmolVLA_Language_def* def,
        npu_app_manager* text_app_manager
        )
        : def(def) {
        this->q_proj = text_app_manager->create_app();
        this->k_proj = text_app_manager->create_app();
        this->v_proj = text_app_manager->create_app();
        this->o_proj = text_app_manager->create_app();
        this->mha    = text_app_manager->create_app();

        this->inv_freq = cpu_func::_rope_inv_freq(def->head_dim, smolvla_cfg::ROPE_THETA);

        this->q    = this->q_proj.create_bo_buffer<bf16>(MAX_L * def->DQ);
        this->k    = this->k_proj.create_bo_buffer<bf16>(MAX_L * def->DK);
        this->v    = this->v_proj.create_bo_buffer<bf16>(MAX_L * def->DV);
        this->attn = this->mha.create_bo_buffer<bf16>(MAX_L * def->DQ);
        this->o    = this->o_proj.create_bo_buffer<bf16>(MAX_L * def->D);

        

        this->temp_attn_out = buffer<bf16>(def->DQ);

        smolvla_vlm_text_seq::gen_text_mm_seq(this->q_proj.seq(), MAX_L, def->layer_def.q_proj, 0);
        smolvla_vlm_text_seq::gen_text_mm_seq(this->k_proj.seq(), MAX_L, def->layer_def.k_proj, 0);
        smolvla_vlm_text_seq::gen_text_mm_seq(this->v_proj.seq(), MAX_L, def->layer_def.v_proj, 0);
        smolvla_vlm_text_seq::gen_text_mm_seq(this->o_proj.seq(), MAX_L, def->layer_def.o_proj, 0);
    }

    /// @brief run GQA self-attention over the token sequence.
    /// @param L        number of valid tokens
    /// @param input    pre-normed hidden state, [L, D]
    /// @param weights  this layer's packed weight buffer
    /// @param k_cache_dst  destination for the pre-rope keys,   [L, DK] (or null)
    /// @param v_cache_dst  destination for the pre-rope values, [L, DV] (or null)
    /// @param early_stop   return as soon as the kv cache is populated, skipping
    ///                     rope(q) / mha / o_proj. See SmolVLA_language_exec::
    ///                     forward for why the last layer can do this.
    /// @return attention output projected back to model dim, [L, D]. Under
    ///         early_stop nothing computes it, so the returned buffer holds the
    ///         previous layer's output and must not be read.
    buffer<bf16> forward(int L, buffer<bf16>& input, buffer<bf16>& weights,
                         bf16* k_cache_dst = nullptr, bf16* v_cache_dst = nullptr,
                         bool early_stop = false) {
        SmolVLA_language_layer& ld = def->layer_def;
        int L_padded = (L + 63) / 64 * 64;
        if (last_L != L) {
            smolvla_vlm_text_seq::gen_text_attn_seq(this->mha.seq(), def->head_dim, def->num_attn_heads, def->num_kv_heads, L_padded / 64, L_padded / 64 - 1, L - (L / 64) * 64 - 1);
            last_L = L;
        }
        const int D  = (int)def->D;
        const int DQ = (int)def->DQ;
        const int DK = (int)def->DK;
        const int DV = (int)def->DV;

        PROF_INIT();
        // 2. q = q_proj(input); k = k_proj(input); v = v_proj(input)  // GQA
        //
        // q_proj is launched even under early_stop, where q itself is dead. The
        // mm sequences share RTP state across launches and only re-send a
        // parameter when it changes (see language/seq_gen.hpp), which makes
        // q_proj's sequence the one carrying the K_div_k write that k_proj and
        // v_proj rely on. Dropping the launch would leave the tiles holding
        // down_proj's K_div_k from the previous layer and silently corrupt k/v.
        auto q_run = q_proj.create_run(q, input, weights);
        auto k_run = k_proj.create_run(k, input, weights);
        auto v_run = v_proj.create_run(v, input, weights);

        q_run.start();
        q_run.wait();

        // 3. apply rope to q and k (LLaMA rotate_half, theta=1e4, no q/k-norm).
        k_run.start();
        if (!early_stop) {
            // rope(q) overlaps the k_proj launch; it only feeds mha, so it goes
            // away with mha under early_stop.
            q.sync_from_device();
            cpu_func::_rope(q.data(), L, def->num_attn_heads, def->head_dim, inv_freq.data());
            q.sync_to_device();
        }
        k_run.wait();
        k.sync_from_device();

        v_run.start();
        cpu_func::_rope(k.data(), L, def->num_kv_heads,   def->head_dim, inv_freq.data());
        k.sync_to_device();
        v_run.wait();
        v.sync_from_device();

        PROF_TICK("language.attn:qkv");

        // cache the pre-rope values (V is never roped). The expert re-projects
        // these for its cross layers and concatenates them for its self layers.
        if (k_cache_dst) memcpy(k_cache_dst, k.data(), (size_t)L * DK * sizeof(bf16));
        if (v_cache_dst) memcpy(v_cache_dst, v.data(), (size_t)L * DV * sizeof(bf16));


        // cache the post-rope keys: the expert attends over these directly (both
        // its cross layers, which re-project them, and its self layers, which
        // concatenate them) so the rope must already be baked in.
        PROF_TICK("language.attn:rope+kv_cache");

        // the kv cache is now complete for this layer; everything below only
        // feeds the residual stream.
        if (early_stop) return o;

        // attention over the full prefix (image + text + state). The VLM prefix
        // attends bidirectionally, so causal=false.
        const int q_last = L - 1;
        auto mha_run = mha.create_run(attn, q, k, v);
        mha_run.start();
        cpu_func::_gqa_mha(temp_attn_out.data(), q.data() + (size_t)q_last * DQ,
                           k.data(), v.data(),
                           /*L_q=*/1, /*L_kv=*/L,
                           def->num_attn_heads, def->num_kv_heads, def->head_dim,
                           /*causal=*/false);
        mha_run.wait();
        attn.sync_from_device();
        memcpy(attn.data() + (size_t)q_last * DQ, temp_attn_out.data(), DQ * sizeof(bf16));
        PROF_TICK("language.attn:mha");

        attn.sync_to_device();
        // 5. out = o_proj(attn) for the state row only            // [1, D]
        o_proj(o, attn, weights);
        o.sync_from_device();
        PROF_TICK("language.attn:o_proj");
        return o;
    }
};

/// @brief gated-MLP block (shared by every layer): SwiGLU over gate/up_proj
///        then down_proj.
struct SmolVLA_language_mlp_block {
    static constexpr int MAX_L = smolvla_cfg::MAX_PREFIX_LEN;

    SmolVLA_Language_def* def;

    buffer<bf16> gate;    // gate projection, [L, intermediate]
    buffer<bf16> up;      // up projection, [L, intermediate]
    buffer<bf16> hid;     // activated hidden, [L, intermediate]
    buffer<bf16> o;       // down projection output, [L, D]

    npu_app gate_proj;
    npu_app up_proj;
    npu_app down_proj;


    xrt::runlist rl;

    SmolVLA_language_mlp_block(
        SmolVLA_Language_def* def,
        npu_app_manager* fused_xclbin)
        : def(def) {

        this->gate_proj = fused_xclbin->create_app();
        this->up_proj   = fused_xclbin->create_app();
        this->down_proj = fused_xclbin->create_app();

        this->gate = this->gate_proj.create_bo_buffer<bf16>(MAX_L * def->intermediate_size);
        this->up   = this->up_proj.create_bo_buffer<bf16>(MAX_L * def->intermediate_size);
        this->hid  = this->down_proj.create_bo_buffer<bf16>(MAX_L * def->intermediate_size);
        this->o    = this->down_proj.create_bo_buffer<bf16>(MAX_L * def->D);

        rl = fused_xclbin->create_runlist();
        
        smolvla_vlm_text_seq::gen_text_mm_seq(this->gate_proj.seq(), MAX_L, def->layer_def.gate_proj, 1);
        smolvla_vlm_text_seq::gen_text_mm_seq(this->up_proj.seq(), MAX_L, def->layer_def.up_proj, 0);
        smolvla_vlm_text_seq::gen_text_mm_seq(this->down_proj.seq(), MAX_L, def->layer_def.down_proj, 0);
    }

    /// @brief run the gated MLP.
    /// @param L       number of valid tokens
    /// @param input   pre-normed hidden state, [L, D]
    /// @param weights this layer's packed weight buffer (holds gate/up/down)
    /// @return mlp output, [L, D]
    buffer<bf16> forward(int L, buffer<bf16>& input, buffer<bf16>& weights) {
        SmolVLA_language_layer& ld = def->layer_def;
        const int D     = (int)def->D;
        const int inter = (int)def->intermediate_size;

        PROF_INIT();
        // 1. gate = gate_proj(input); up = up_proj(input)  // [L, intermediate]
        rl.reset();
        rl.add(gate_proj.create_run(gate, input, weights));
        rl.add(up_proj.create_run(up, input, weights));
        rl.execute();
        rl.wait();
        gate.sync_from_device();
        up.sync_from_device();
        PROF_TICK("language.mlp:gate+up");

        // 2. hid = silu(gate) * up = gate * sigmoid(gate) * up   (SwiGLU)
        cpu_func::_elementwise_mul(hid.data(), gate.data(), up.data(), L * inter);
        hid.sync_to_device();
        const int act_len = L * inter;
        PROF_TICK("language.mlp:swiglu");

        // 3. out = down_proj(hid), [L, D]
        down_proj(o, hid, weights);
        PROF_TICK("language.mlp:down");
        o.sync_from_device();
        return o;
    }
};

/// @brief top-level driver: runs the full text-model stack over the multimodal
///        token sequence (prefill), populating the per-layer kv cache.
struct SmolVLA_language_exec {
    SmolVLA_Language_def* def;

    std::unique_ptr<SmolVLA_language_attn_block> attn;
    std::unique_ptr<SmolVLA_language_mlp_block>  mlp;

    // xclbin app managers
    npu_app_manager* fused_xclbin = nullptr;

    buffer<bf16> x;

    SmolVLA_language_exec() {}
    SmolVLA_language_exec(npu_xclbin_manager* npu, SmolVLA_Language_def* def)
        : def(def) {
        x = npu->create_bo_buffer<bf16>(smolvla_cfg::MAX_PREFIX_LEN * def->D);
        this->fused_xclbin = npu->register_xclbin(utils::xclbin_path("text.xclbin"));

        this->attn = std::make_unique<SmolVLA_language_attn_block>(def, fused_xclbin);
        this->mlp  = std::make_unique<SmolVLA_language_mlp_block>(def, fused_xclbin);
    }

    /// @brief run one text-model layer.
    /// @param residual     running hidden state, updated in place, [L, D]
    /// @param norm_weights this layer's norm weights
    /// @param weights      packed weights for this layer
    /// @param L            token count
    /// @param k_cache_dst  destination for this layer's pre-rope keys,   [L, DK]
    /// @param v_cache_dst  destination for this layer's pre-rope values, [L, DV]
    /// @param early_stop   stop once this layer's kv cache is written, skipping
    ///                     the rest of the attention sub-block and the whole mlp
    ///                     sub-block (both residual adds, layer_norm2, o_proj,
    ///                     mha). Only valid on the LAST layer: the prefill's sole
    ///                     output is the kv cache (SmolVLA_language::forward
    ///                     discards the returned residual, and the expert reads
    ///                     only kv_cache), so nothing downstream can observe the
    ///                     hidden state the tail would have produced.
    /// @return updated residual hidden state, [L, D]. Under early_stop the
    ///         residual is left mid-layer -- valid memory, but not this layer's
    ///         output, so the caller must not use it.
    buffer<bf16> forward(buffer<bf16>& residual, buffer<bf16>& norm_weights, buffer<bf16>& weights, int L,
                         bf16* k_cache_dst = nullptr, bf16* v_cache_dst = nullptr,
                         bool early_stop = false) {
        SmolVLA_language_layer& ld = def->layer_def;
        auto N = [&](weight_desc_t& d) { return d.locate_myself<bf16>(norm_weights); };

        constexpr float eps = smolvla_cfg::TEXT_NORM_EPS;
        const int D = (int)def->D;
        PROF_INIT();
        buffer<bf16> ln1_w = N(ld.input_layernorm);
        cpu_func::_layer_norm(x, residual, ln1_w, L, D, eps);
        x.sync_to_device();
        PROF_TICK("language.layer:layer_norm1");
        buffer<bf16> attn_out = this->attn->forward(L, x, weights, k_cache_dst, v_cache_dst, early_stop);
        if (early_stop) {
            PROF_TICK("language.layer:attn(early_stop)");
            return residual;
        }

        // 3. residual = residual + attn_out (state row only, matching attn).
        cpu_func::_residual_add(residual, attn_out, residual, (size_t)L * D);
        PROF_TICK("language.layer:attn+residual");

        // pre-norm for the mlp sub-block, over the full sequence.
        buffer<bf16> ln2_w = N(ld.post_attention_layernorm);
        cpu_func::_layer_norm(x, residual, ln2_w, L, D, eps);
        PROF_TICK("language.layer:layer_norm2");
        // 4. x = rms_norm(residual, post_attention_layernorm)   // full length
        x.sync_to_device();
        // 5. mlp_out = mlp->forward(L, x, weights)              // full length
        buffer<bf16> mlp_out = this->mlp->forward(L, x, weights);
        cpu_func::_residual_add(residual, mlp_out, residual, (size_t)L * D);
        PROF_TICK("language.layer:mlp+residual");
        return residual;
    }
};

#endif // __SMOLVLA__LANGUAGE__EXEC__HPP__
