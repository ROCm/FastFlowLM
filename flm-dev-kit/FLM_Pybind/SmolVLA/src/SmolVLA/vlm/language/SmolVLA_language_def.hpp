// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __SMOLVLA__LANGUAGE__DEF__HPP__
#define __SMOLVLA__LANGUAGE__DEF__HPP__

#include <vector>
#include <cmath>

#include "typedef.hpp"
#include "buffer.hpp"
#include "weight_desc.hpp"
#include "tensor_utils/safe_tensors.hpp"
#include "nlohmann/json.hpp"

/// @brief A single text-model layer (LLaMA-style, pre-norm, GQA + SwiGLU MLP).
///        All text layers are identical, so one template describes them all;
///        the layer index is filled in via weight_desc_t::format_name(layer).
struct SmolVLA_language_layer {
    weight_desc_t input_layernorm;
    weight_desc_t post_attention_layernorm;

    weight_desc_t q_proj;
    weight_desc_t k_proj;
    weight_desc_t v_proj;
    weight_desc_t o_proj;
    // q|k|v seen as one [D -> DQ+DK+DV] matrix (they are adjacent in the layer buffer), so the three
    // projections run as a single matmul. Not part of proj_all().
    weight_desc_t qkv_proj;

    weight_desc_t gate_proj;
    weight_desc_t up_proj;
    weight_desc_t down_proj;

    std::vector<weight_desc_t*> proj_all() {
        return {
            &q_proj, &k_proj, &v_proj, &o_proj,
            &gate_proj, &up_proj, &down_proj,
        };
    }
    std::vector<weight_desc_t*> norm_all() {
        return {
            &input_layernorm, &post_attention_layernorm
        };
    }
};

struct SmolVLA_Language_def {
    // embeddings / final norm
    weight_desc_t embed_tokens;   // [vocab, D]
    weight_desc_t norm;           // [D]

    // all layers are identical: one template covers every layer
    SmolVLA_language_layer layer_def;

    // parameters
    uint32_t D;                 // hidden_size
    uint32_t DQ;
    uint32_t DK;
    uint32_t DV;
    uint32_t num_attn_heads;
    uint32_t num_kv_heads;
    uint32_t head_dim;
    uint32_t intermediate_size;
    uint32_t vocab_size;
    uint32_t num_layers;

    // sqrt(hidden_size): SmolVLA scales image/text embeddings by this before the
    // language model. Precomputed at build time to keep it off the hot path.
    float embed_scale;

    // aggregated buffer byte sizes (one representative layer + the rest)
    size_t layer_byte_size = 0;
    size_t norm_byte_size = 0;
    size_t non_layer_byte_size = 0;

    /// @brief default constructor
    SmolVLA_Language_def() {}

    /// @brief the non-in-layer (embedding / final-norm) weight descriptors
    std::vector<weight_desc_t*> non_layer_all() {
        return { &embed_tokens, &norm };
    }

    /// @brief parse the text-model configuration.
    ///
    /// SmolVLA's config.json does not carry a dedicated text_config block; the
    /// text tower is the SmolVLM2-500M language model whose dims are fixed. Only
    /// the layer count is configurable (top-level "num_vlm_layers").
    /// @param num_layers number of text-model layers to build
    void build(uint32_t num_layers) {
        this->D                 = 960;
        this->intermediate_size = 2560;
        this->num_attn_heads    = 15;
        this->num_kv_heads      = 5;
        this->head_dim          = 64;
        this->vocab_size        = 49280;
        this->num_layers        = num_layers;
        this->embed_scale       = std::sqrt((float)this->D);

        // attention projection dims: Q spans all heads, K/V span the kv heads
        this->DQ = this->num_attn_heads * this->head_dim;   // 960
        this->DK = this->num_kv_heads   * this->head_dim;   // 320
        this->DV = this->num_kv_heads   * this->head_dim;   // 320

        assert(this->D > 0);
        assert(this->num_attn_heads % this->num_kv_heads == 0);
        assert(this->num_layers > 0);

        // build weight descriptors. shapes are {in, out} (transpose of the
        // stored [out, in] safetensor layout). all text weights are bf16.
        // names are fully-qualified paths within model.safetensors.
        const std::string TM  = "model.vlm_with_expert.vlm.model.text_model.";
        const std::string LYR = TM + "layers.%d.";

        // embeddings / final norm
        embed_tokens = weight_desc_t(flm_bf16, {vocab_size, D}, TM + "embed_tokens.weight");
        norm         = weight_desc_t(flm_bf16, {D},             TM + "norm.weight");

        // per-layer norms
        layer_def.input_layernorm          = weight_desc_t(flm_bf16, {D}, LYR + "input_layernorm.weight");
        layer_def.post_attention_layernorm = weight_desc_t(flm_bf16, {D}, LYR + "post_attention_layernorm.weight");

        // self-attention (GQA: DQ != DK == DV)
        layer_def.q_proj = weight_desc_t(flm_bf16, {D, DQ}, LYR + "self_attn.q_proj.weight");
        layer_def.k_proj = weight_desc_t(flm_bf16, {D, DK}, LYR + "self_attn.k_proj.weight");
        layer_def.v_proj = weight_desc_t(flm_bf16, {D, DV}, LYR + "self_attn.v_proj.weight");
        layer_def.o_proj = weight_desc_t(flm_bf16, {DQ, D}, LYR + "self_attn.o_proj.weight");

        // gated MLP (SwiGLU): gate/up project up, down projects back
        layer_def.gate_proj = weight_desc_t(flm_bf16, {D, intermediate_size}, LYR + "mlp.gate_proj.weight");
        layer_def.up_proj   = weight_desc_t(flm_bf16, {D, intermediate_size}, LYR + "mlp.up_proj.weight");
        layer_def.down_proj = weight_desc_t(flm_bf16, {intermediate_size, D}, LYR + "mlp.down_proj.weight");

        // group weights into containers and record the byte size of each group
        weight_container layer_helper;
        for (weight_desc_t* w : layer_def.proj_all()) {
            layer_helper.add_weight(*w);
        }
        this->layer_byte_size = layer_helper.get_size();
        assert(layer_def.k_proj.offset == layer_def.q_proj.offset + (size_t)D * DQ * sizeof(bf16));
        assert(layer_def.v_proj.offset == layer_def.k_proj.offset + (size_t)D * DK * sizeof(bf16));
        layer_def.qkv_proj = weight_desc_t(flm_bf16, {D, DQ + DK + DV}, LYR + "self_attn.qkv_proj.weight");
        layer_def.qkv_proj.offset = layer_def.q_proj.offset;
        layer_def.qkv_proj.added  = true;

        weight_container norm_helper;
        for (weight_desc_t* w : layer_def.norm_all()) {
            norm_helper.add_weight(*w);
        }
        this->norm_byte_size = norm_helper.get_size();

        weight_container non_layer_helper;
        for (weight_desc_t* w : non_layer_all()) {
            non_layer_helper.add_weight(*w);
        }
        this->non_layer_byte_size = non_layer_helper.get_size();
    }

    /// @brief load all text weights from the safetensors file into the
    ///        caller-provided host/device buffers (owned by SmolVLA_language).
    /// @param st an opened SafeTensors handle for model.safetensors
    /// @param layer_weights     one buffer per text layer (layer_byte_size each)
    /// @param non_layer_weights buffer for the embedding / final-norm weights
    void load(SafeTensors& st,
              std::vector<buffer<bf16>>& layer_weights,
              std::vector<buffer<bf16>>& norm_weights,
              buffer<u8>& non_layer_weights) {
        // embeddings / final norm
        for (weight_desc_t* w : non_layer_all()) {
            st.load_weights(non_layer_weights, w->name, w->offset);
            w->load();
        }

        // per-layer weights: each layer goes into its own buffer, offsets are
        // relative to the start of a single layer's block.
        for (uint32_t l = 0; l < num_layers; l++) {
            for (weight_desc_t* w : layer_def.proj_all()) {
                st.load_weights(layer_weights[l], w->format_name(l), w->offset);
            }
            for (weight_desc_t* w : layer_def.norm_all()) {
                st.load_weights(norm_weights[l], w->format_name(l), w->offset);
            }
            layer_weights[l].sync_to_device();
        }
        for (weight_desc_t* w : layer_def.proj_all()) w->load();
        for (weight_desc_t* w : layer_def.norm_all()) w->load();
    }
};

#endif // __SMOLVLA__LANGUAGE__DEF__HPP__
