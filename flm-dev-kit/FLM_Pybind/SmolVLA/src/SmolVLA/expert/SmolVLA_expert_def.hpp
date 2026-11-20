// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __SMOLVLA__EXPERT__DEF__HPP__
#define __SMOLVLA__EXPERT__DEF__HPP__

#include <vector>

#include "typedef.hpp"
#include "buffer.hpp"
#include "weight_desc.hpp"
#include "tensor_utils/safe_tensors.hpp"
#include "nlohmann/json.hpp"
#include "SmolVLA_config.hpp"

/// @brief the gated MLP (SwiGLU) weight descriptors. Identical for self and
///        cross layers (same shapes, same name template), so a single instance
///        describes the MLP of every layer; see SmolVLA_Expert_def::mlp_def.
struct SmolVLA_expert_mlp {
    weight_desc_t gate_proj;
    weight_desc_t up_proj;
    weight_desc_t down_proj;

    std::vector<weight_desc_t*> all() {
        return { &gate_proj, &up_proj, &down_proj };
    }
};

/// @brief One expert layer's attention weight descriptors. Self (even) and cross
///        (odd) layers carry the same set of tensors and differ only in the
///        shapes of k_proj/v_proj: self layers project from the expert hidden
///        state (D->DK), cross layers re-project the cached VLM kv states
///        (DK->DK). Two instances of this type describe both kinds; see
///        SmolVLA_Expert_def::self_layer_def / cross_layer_def. The MLP is not
///        stored here — it is shared across layer kinds (see mlp_def).
struct SmolVLA_expert_layer {
    weight_desc_t input_layernorm;
    weight_desc_t post_attention_layernorm;
    // separate q/k/v/o projections (raw model.safetensors layout)
    weight_desc_t q_proj;
    weight_desc_t k_proj;
    weight_desc_t v_proj;
    weight_desc_t o_proj;
};

struct SmolVLA_Expert_def {
    weight_desc_t expert_norm;

    SmolVLA_expert_layer cross_layer_def;
    SmolVLA_expert_layer self_layer_def;
    // the gated MLP is identical for self and cross layers, so a single
    // descriptor set covers every layer (its %d name template loads per-layer).
    SmolVLA_expert_mlp   mlp_def;

    // parameters
    uint32_t D;
    // S stands for self
    uint32_t DQ;
    uint32_t DK;
    uint32_t DV;

    // padded parameters (each padded up to a multiple of smolvla_cfg::PAD_ALIGN)
    uint32_t pD;
    uint32_t pDQ;
    uint32_t pDK;
    uint32_t pDV;
    uint32_t pintermediate_size;

    uint32_t num_attn_heads;
    uint32_t num_kv_heads;
    uint32_t head_dim;
    uint32_t intermediate_size;
    uint32_t action_dim;        // padded action/state dimension
    uint32_t num_layers;        // total layers (interleaved self / cross)

    // inference parameters
    uint32_t chunk_size;

    // aggregated buffer byte sizes (one representative layer of each kind)
    size_t cross_layer_byte_size = 0;
    size_t self_layer_byte_size = 0;
    size_t non_layer_byte_size = 0;

    /// @brief default constructor
    SmolVLA_Expert_def() {}

    /// @brief the non-in-layer (top-level) weight descriptors. The action/state
    /// adapters live on the top-level SmolVLA (they are model.* tensors, not
    /// under lm_expert), so only the final expert norm remains here.
    std::vector<weight_desc_t*> non_layer_all() {
        return { &expert_norm };
    }

    /// @brief all descriptors that live in one layer's packed buffer, in offset
    /// order: layernorms, then the shared MLP, then attention. The MLP comes from
    /// the single shared mlp_def (identical for self and cross layers).
    std::vector<weight_desc_t*> layer_all(SmolVLA_expert_layer& attn) {
        return {
            &attn.input_layernorm, &attn.post_attention_layernorm,
            &mlp_def.gate_proj, &mlp_def.up_proj, &mlp_def.down_proj,
            &attn.q_proj, &attn.k_proj, &attn.v_proj, &attn.o_proj,
        };
    }
    std::vector<weight_desc_t*> self_layer_all()  { return layer_all(self_layer_def); }
    std::vector<weight_desc_t*> cross_layer_all() { return layer_all(cross_layer_def); }

    /// @brief the expert alternates self- and cross-attention layers.
    static inline bool is_self_layer(uint32_t layer_idx) {
        return (layer_idx % smolvla_cfg::SELF_ATTN_EVERY_N_LAYERS) == 0;
    }

    /// @brief index of a layer within the buffers of its own kind.
    static inline uint32_t layer_slot(uint32_t layer_idx) {
        return layer_idx / smolvla_cfg::SELF_ATTN_EVERY_N_LAYERS;
    }

    static inline uint32_t pad_to(uint32_t v) { return smolvla_cfg::pad_to(v); }

    /// @brief parse the expert configuration
    /// @param ecfg the "expert_config" object from config.json
    void build(const nlohmann::json& ecfg) {
        this->D                 = ecfg.at("hidden_size").get<uint32_t>();
        this->intermediate_size = ecfg.at("intermediate_size").get<uint32_t>();
        this->num_attn_heads    = ecfg.at("num_attn_heads").get<uint32_t>();
        this->num_kv_heads      = ecfg.at("num_key_value_heads").get<uint32_t>();
        this->head_dim          = ecfg.at("head_dim").get<uint32_t>();
        this->num_layers        = ecfg.at("num_expert_layers").get<uint32_t>();

        // attention projection dims: Q spans all heads, K/V span the kv heads (GQA)
        this->DQ = this->num_attn_heads * this->head_dim;
        this->DK = this->num_kv_heads   * this->head_dim;
        this->DV = this->num_kv_heads   * this->head_dim;

        // padded dims (multiple of PAD_ALIGN)
        this->pD                 = pad_to(this->D);
        this->pDQ                = pad_to(this->DQ);
        this->pDK                = pad_to(this->DK);
        this->pDV                = pad_to(this->DV);
        this->pintermediate_size = pad_to(this->intermediate_size);

        this->chunk_size = ecfg.value("chunk_size", 50u);
        this->action_dim = ecfg.at("action_dim").get<uint32_t>();

        assert(this->D > 0);
        assert(this->num_attn_heads > 0);
        assert(this->num_kv_heads > 0);
        assert(this->num_attn_heads % this->num_kv_heads == 0);
        assert(this->head_dim > 0);
        assert(this->intermediate_size > 0);
        assert(this->action_dim > 0);


        // build weight descriptors. shapes are {in, out} (transpose of the
        // stored [out, in] safetensor layout). dims are padded to smolvla_cfg::PAD_ALIGN.
        // layers interleave: even = self-attn, odd = cross-attn; %d is the
        // layer index, filled in per-instance via weight_desc_t::format_name.
        // names are fully-qualified paths within model.safetensors.
        const std::string LME = "model.vlm_with_expert.lm_expert.";
        const std::string LYR = LME + "layers.%d.";

        // final expert norm (top-level lm_expert weight)
        expert_norm = weight_desc_t(flm_bf16, {pD}, LME + "norm.weight");

        // ---- gated MLP (SwiGLU): shared by every layer (self and cross carry an
        // identical MLP, so one descriptor set covers both kinds). ----
        mlp_def.gate_proj = weight_desc_t(flm_bf16, {pD, pintermediate_size}, LYR + "mlp.gate_proj.weight");
        mlp_def.up_proj   = weight_desc_t(flm_bf16, {pD, pintermediate_size}, LYR + "mlp.up_proj.weight");
        mlp_def.down_proj = weight_desc_t(flm_bf16, {pintermediate_size, pD}, LYR + "mlp.down_proj.weight");

        // ---- cross-attention layers (odd): Q from expert hidden, K/V from the
        // cached VLM kv states (DK->DK / DV->DV). ----
        cross_layer_def.input_layernorm          = weight_desc_t(flm_bf16, {pD}, LYR + "input_layernorm.weight");
        cross_layer_def.post_attention_layernorm = weight_desc_t(flm_bf16, {pD}, LYR + "post_attention_layernorm.weight");
        cross_layer_def.q_proj    = weight_desc_t(flm_bf16, {pD,  pDQ}, LYR + "self_attn.q_proj.weight");
        cross_layer_def.k_proj    = weight_desc_t(flm_bf16, {pDK, pDK}, LYR + "self_attn.k_proj.weight");
        cross_layer_def.v_proj    = weight_desc_t(flm_bf16, {pDV, pDV}, LYR + "self_attn.v_proj.weight");
        cross_layer_def.o_proj    = weight_desc_t(flm_bf16, {pDQ, pD},  LYR + "self_attn.o_proj.weight");

        // ---- self-attention layers (even): K/V projected from the expert
        // hidden state (D->DK / D->DV), stored in BF16. ----
        self_layer_def.input_layernorm          = weight_desc_t(flm_bf16, {pD}, LYR + "input_layernorm.weight");
        self_layer_def.post_attention_layernorm = weight_desc_t(flm_bf16, {pD}, LYR + "post_attention_layernorm.weight");
        self_layer_def.q_proj    = weight_desc_t(flm_bf16, {pD, pDQ}, LYR + "self_attn.q_proj.weight");
        self_layer_def.k_proj    = weight_desc_t(flm_bf16, {pD, pDK}, LYR + "self_attn.k_proj.weight");
        self_layer_def.v_proj    = weight_desc_t(flm_bf16, {pD, pDV}, LYR + "self_attn.v_proj.weight");
        self_layer_def.o_proj    = weight_desc_t(flm_bf16, {pDQ, pD}, LYR + "self_attn.o_proj.weight");

        // group weights into containers and record the byte size of each group.
        // Offset order per layer: layernorms, then shared MLP, then attention.
        weight_container cross_layer_helper;
        for (weight_desc_t* w : cross_layer_all()) {
            cross_layer_helper.add_weight(*w);
        }
        this->cross_layer_byte_size = cross_layer_helper.get_size();

        weight_container self_layer_helper;
        for (weight_desc_t* w : self_layer_all()) {
            self_layer_helper.add_weight(*w);
        }
        this->self_layer_byte_size = self_layer_helper.get_size();

        weight_container non_layer_helper;
        for (weight_desc_t* w : non_layer_all()) {
            non_layer_helper.add_weight(*w);
        }
        this->non_layer_byte_size = non_layer_helper.get_size();

        assert(this->num_layers > 0);
    }

    /// @brief load one weight into its padded slot within a destination buffer.
    ///
    /// The descriptor's shape carries the *padded* dims (each rounded up to
    /// smolvla_cfg::PAD_ALIGN), but the safetensor on disk is unpadded. A flat read would
    /// misalign every row after the first, so we read the unpadded tensor into a
    /// temp buffer and copy it row by row into the wider padded row stride,
    /// leaving the padding (trailing columns and rows) zeroed.
    /// @param st        opened SafeTensors handle
    /// @param w         weight descriptor (its shape holds the padded dims)
    /// @param name      the (formatted) tensor name to read
    /// @param dst_buf   destination buffer; the weight goes at w->offset
    void load_padded(SafeTensors& st, weight_desc_t* w,
                     const std::string& name, bytes& dst_buf) {
        tensor_metadata meta = st.get_tensor_metadata(name);

        // read the raw unpadded tensor into a temp buffer
        bytes temp;
        st.load_weights(temp, name);

        const size_t esize = get_dtype_byte(w->dtype);
        // real (unpadded) row/col counts from the safetensor shape
        const size_t rows = (meta.shape.size() >= 2) ? meta.shape[0] : 1;
        const size_t cols = (meta.shape.size() >= 2) ? meta.shape[1] : meta.shape[0];
        // padded dims from the descriptor: shape[0] is the padded inner (column)
        // stride, shape[1] the padded outer (row) count.
        const size_t pcols = static_cast<size_t>(w->shape[0]);
        const size_t prows = static_cast<size_t>(w->shape[1]);
        assert(cols <= pcols && rows <= prows);

        uint8_t* dst = dst_buf.data() + w->offset;
        // zero the full padded region first, then fill the real rows
        std::memset(dst, 0, prows * pcols * esize);
        for (size_t r = 0; r < rows; r++) {
            std::memcpy(dst + r * pcols * esize,
                        temp.data() + r * cols * esize,
                        cols * esize);
        }
    }

    /// @brief load all expert weights from the safetensors file into the
    ///        caller-provided host/device buffers (owned by SmolVLA_expert).
    /// @param st an opened SafeTensors handle for expert_weights.safetensors
    /// @param self_layer_weights  one buffer per self (even) layer
    /// @param cross_layer_weights one buffer per cross (odd) layer
    /// @param non_layer_weights   buffer for the top-level (non-in-layer) weights
    void load(SafeTensors& st,
              std::vector<buffer<bf16>>& self_layer_weights,
              std::vector<buffer<bf16>>& cross_layer_weights,
              buffer<u8>& non_layer_weights) {
        // top-level (non-in-layer) weights
        for (weight_desc_t* w : non_layer_all()) {
            load_padded(st, w, w->name, non_layer_weights);
            w->load();
        }

        // per-layer weights: each layer goes into its own buffer, indexed by
        // its slot within its kind (self = even indices, cross = odd indices).
        for (uint32_t l = 0; l < num_layers; l++) {
            uint32_t slot = layer_slot(l);
            if (is_self_layer(l)) {
                for (weight_desc_t* w : self_layer_all()) {
                    load_padded(st, w, w->format_name(l), self_layer_weights[slot]);
                }
            } else {
                for (weight_desc_t* w : cross_layer_all()) {
                    load_padded(st, w, w->format_name(l), cross_layer_weights[slot]);
                }
            }
            self_layer_weights[slot].sync_to_device();
            cross_layer_weights[slot].sync_to_device();
        }

        for (weight_desc_t* w : self_layer_all())  w->load();
        for (weight_desc_t* w : cross_layer_all()) w->load();
    }
};

#endif // __SMOLVLA__EXPERT__DEF__HPP__