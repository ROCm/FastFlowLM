// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __SMOLVLA__VISION__DEF__HPP__
#define __SMOLVLA__VISION__DEF__HPP__

#include <vector>

#include "typedef.hpp"
#include "buffer.hpp"
#include "SmolVLA_config.hpp"
#include "weight_desc.hpp"
#include "tensor_utils/safe_tensors.hpp"
#include "nlohmann/json.hpp"
#include "utils/utils.hpp"

/// @brief A single vision encoder layer (SigLIP-style, pre-norm transformer).
///        All vision layers are identical, so one template describes them all;
///        the layer index is filled in via weight_desc_t::format_name(layer).
struct SmolVLA_vision_layer {
    weight_desc_t layer_norm1;
    weight_desc_t layer_norm1_bias;
    weight_desc_t layer_norm2;
    weight_desc_t layer_norm2_bias;

    weight_desc_t q_proj;
    weight_desc_t q_proj_bias;
    weight_desc_t k_proj;
    weight_desc_t k_proj_bias;
    weight_desc_t v_proj;
    weight_desc_t v_proj_bias;
    weight_desc_t out_proj;
    weight_desc_t out_proj_bias;

    weight_desc_t fc1;
    weight_desc_t fc1_bias;
    weight_desc_t fc2;
    weight_desc_t fc2_bias;

    /// @brief the NPU-executed weights (attention + mlp projections), in the
    ///        order they are packed into a layer buffer: each bias immediately
    ///        precedes its weight, which is the layout the mm sequences assume.
    ///        The four layer_norm tensors are excluded: they are only consumed by
    ///        the CPU layer_norm and live in a separate host buffer (norm_all()).
    std::vector<weight_desc_t*> all() {
        return {
            &q_proj_bias, &q_proj, &k_proj_bias, &k_proj,
            &v_proj_bias, &v_proj, &out_proj_bias, &out_proj,
            &fc1_bias, &fc1, &fc2_bias, &fc2,
        };
    }

    /// @brief the CPU-only layer-norm weights (never uploaded to the NPU).
    std::vector<weight_desc_t*> norm_all() {
        return { &layer_norm1, &layer_norm1_bias, &layer_norm2, &layer_norm2_bias };
    }
};

struct SmolVLA_Vision_def {
    // embeddings
    weight_desc_t patch_embedding;      // conv2d [D, Cin, k, k]
    weight_desc_t patch_embedding_bias;
    weight_desc_t position_embedding;   // [num_positions, D]
    weight_desc_t post_layernorm;
    weight_desc_t post_layernorm_bias;

    // all layers are identical: one template covers every layer
    SmolVLA_vision_layer layer_def;

    // parameters
    uint32_t D;                 // hidden_size
    uint32_t DQ;
    uint32_t DK;
    uint32_t DV;
    uint32_t num_attn_heads;
    uint32_t num_kv_heads;
    uint32_t head_dim;
    uint32_t intermediate_size;
    uint32_t num_layers;

    // aggregated buffer byte sizes (one representative layer + the rest)
    size_t layer_byte_size = 0;      // NPU projections, per layer
    size_t norm_byte_size = 0;        // CPU layer-norm weights, per layer
    size_t non_layer_byte_size = 0;

    /// @brief default constructor
    SmolVLA_Vision_def() {}

    /// @brief the non-in-layer (embedding / final-norm) weight descriptors
    std::vector<weight_desc_t*> non_layer_all() {
        return {
            &patch_embedding, &patch_embedding_bias, &position_embedding,
            &post_layernorm, &post_layernorm_bias,
        };
    }

    /// @brief parse the vision configuration
    /// @param vcfg the "vision_config" object from config.json
    void build(const nlohmann::json& vcfg) {
        this->D                 = vcfg.at("hidden_size").get<uint32_t>();
        this->intermediate_size = vcfg.at("intermediate_size").get<uint32_t>();
        this->num_attn_heads    = vcfg.at("num_attn_heads").get<uint32_t>();
        this->num_kv_heads      = vcfg.at("num_key_value_heads").get<uint32_t>();
        this->head_dim          = vcfg.at("head_dim").get<uint32_t>();
        this->num_layers        = vcfg.at("num_vlm_layers").get<uint32_t>();

        // attention projection dims: Q spans all heads, K/V span the kv heads
        this->DQ = this->num_attn_heads * this->head_dim;
        this->DK = this->num_kv_heads  * this->head_dim;
        this->DV = this->num_kv_heads  * this->head_dim;

        assert(this->D > 0);
        assert(this->num_attn_heads > 0);
        assert(this->num_kv_heads > 0);
        assert(this->num_attn_heads % this->num_kv_heads == 0);
        assert(this->head_dim > 0);
        assert(this->intermediate_size > 0);
        assert(this->num_layers > 0);

        constexpr uint32_t PATCH_CIN     = smolvla_cfg::PATCH_CIN;
        constexpr uint32_t PATCH_KERNEL  = smolvla_cfg::PATCH_SIZE;
        constexpr uint32_t NUM_POSITIONS = smolvla_cfg::NUM_POSITIONS;

        // build weight descriptors. shapes are {in, out} (transpose of the
        // stored [out, in] safetensor layout). all vision weights are bf16.
        // %d is the layer index, filled in per-instance via format_name.
        // names are fully-qualified paths within model.safetensors.
        const std::string VM  = "model.vlm_with_expert.vlm.model.vision_model.";
        const std::string EMB = VM + "embeddings.";
        const std::string LYR = VM + "encoder.layers.%d.";

        // embeddings / final norm
        patch_embedding      = weight_desc_t(flm_bf16, {D, PATCH_CIN, PATCH_KERNEL, PATCH_KERNEL}, EMB + "patch_embedding.weight");
        patch_embedding_bias = weight_desc_t(flm_bf16, {D},                    EMB + "patch_embedding.bias");
        position_embedding   = weight_desc_t(flm_bf16, {D, NUM_POSITIONS},     EMB + "position_embedding.weight");
        post_layernorm       = weight_desc_t(flm_bf16, {D},                    VM + "post_layernorm.weight");
        post_layernorm_bias  = weight_desc_t(flm_bf16, {D},                    VM + "post_layernorm.bias");

        // per-layer norms
        layer_def.layer_norm1      = weight_desc_t(flm_bf16, {D}, LYR + "layer_norm1.weight");
        layer_def.layer_norm1_bias = weight_desc_t(flm_bf16, {D}, LYR + "layer_norm1.bias");
        layer_def.layer_norm2      = weight_desc_t(flm_bf16, {D}, LYR + "layer_norm2.weight");
        layer_def.layer_norm2_bias = weight_desc_t(flm_bf16, {D}, LYR + "layer_norm2.bias");

        // self-attention (separate q/k/v, no GQA: DQ == DK == DV == D)
        layer_def.q_proj        = weight_desc_t(flm_bf16, {D, DQ}, LYR + "self_attn.q_proj.weight");
        layer_def.q_proj_bias   = weight_desc_t(flm_bf16, {DQ},    LYR + "self_attn.q_proj.bias");
        layer_def.k_proj        = weight_desc_t(flm_bf16, {D, DK}, LYR + "self_attn.k_proj.weight");
        layer_def.k_proj_bias   = weight_desc_t(flm_bf16, {DK},    LYR + "self_attn.k_proj.bias");
        layer_def.v_proj        = weight_desc_t(flm_bf16, {D, DV}, LYR + "self_attn.v_proj.weight");
        layer_def.v_proj_bias   = weight_desc_t(flm_bf16, {DV},    LYR + "self_attn.v_proj.bias");
        layer_def.out_proj      = weight_desc_t(flm_bf16, {DQ, D}, LYR + "self_attn.out_proj.weight");
        layer_def.out_proj_bias = weight_desc_t(flm_bf16, {D},     LYR + "self_attn.out_proj.bias");

        // mlp
        layer_def.fc1      = weight_desc_t(flm_bf16, {D, intermediate_size}, LYR + "mlp.fc1.weight");
        layer_def.fc1_bias = weight_desc_t(flm_bf16, {intermediate_size},    LYR + "mlp.fc1.bias");
        layer_def.fc2      = weight_desc_t(flm_bf16, {intermediate_size, D}, LYR + "mlp.fc2.weight");
        layer_def.fc2_bias = weight_desc_t(flm_bf16, {D},                    LYR + "mlp.fc2.bias");

        // group weights into containers and record the byte size of each group
        weight_container layer_helper;
        for (weight_desc_t* w : layer_def.all()) {
            layer_helper.add_weight(*w);
        }
        this->layer_byte_size = layer_helper.get_size();

        // layer-norm weights live in a separate CPU-only buffer (not uploaded
        // to the NPU): they are consumed exclusively by the CPU layer_norm.
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

    /// @brief load all vision weights from the safetensors file into the
    ///        caller-provided host/device buffers (owned by SmolVLA_vision).
    /// @param st an opened SafeTensors handle for vision_weights.safetensors
    /// @param layer_weights     one buffer per encoder layer (layer_byte_size each)
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
        // relative to the start of a single layer's block. NPU projections go to
        // layer_weights (uploaded to device); the CPU-only layer-norm weights go
        // to norm_weights (host memory, never synced to device).
        for (uint32_t l = 0; l < num_layers; l++) {
            for (weight_desc_t* w : layer_def.all()) {
                st.load_weights(layer_weights[l], w->format_name(l), w->offset);
            }
            for (weight_desc_t* w : layer_def.norm_all()) {
                st.load_weights(norm_weights[l], w->format_name(l), w->offset);
            }
            layer_weights[l].sync_to_device();
        }
        for (weight_desc_t* w : layer_def.all())     w->load();
        for (weight_desc_t* w : layer_def.norm_all()) w->load();
    }
};

#endif // __SMOLVLA__VISION__DEF__HPP__
