// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __SMOLVLA__VLM__DEF__HPP__
#define __SMOLVLA__VLM__DEF__HPP__

#include <vector>

#include "typedef.hpp"
#include "buffer.hpp"
#include "weight_desc.hpp"
#include "tensor_utils/safe_tensors.hpp"

/// @brief Non-layer VLM weights that live directly under vlm_with_expert.vlm:
///        the vision->text connector (modality projection) and the LM head.
///        The vision encoder and text model own their own descriptors.
struct SmolVLA_Vlm_def {
    weight_desc_t connector;    // modality_projection.proj.weight [D_text, N*D_vision]
    weight_desc_t lm_head;      // [vocab, D_text]

    // dims (text-tower hidden / vocab)
    uint32_t D_text;
    uint32_t vocab_size;
    uint32_t connector_in;      // flattened vision feature width feeding the connector

    size_t non_layer_byte_size = 0;

    SmolVLA_Vlm_def() {}

    std::vector<weight_desc_t*> non_layer_all() {
        return { &connector, &lm_head };
    }

    /// @brief build the non-layer vlm descriptors.
    /// @param D_text       text-tower hidden size (960)
    /// @param vocab_size   token vocabulary (49280)
    /// @param connector_in flattened vision feature width into the connector (12288)
    void build(uint32_t D_text, uint32_t vocab_size, uint32_t connector_in) {
        this->D_text       = D_text;
        this->vocab_size   = vocab_size;
        this->connector_in = connector_in;

        const std::string VLM = "model.vlm_with_expert.vlm.";

        connector = weight_desc_t(flm_bf16, {connector_in, D_text}, VLM + "model.connector.modality_projection.proj.weight");
        lm_head   = weight_desc_t(flm_bf16, {D_text, vocab_size},   VLM + "lm_head.weight");

        weight_container helper;
        for (weight_desc_t* w : non_layer_all()) {
            helper.add_weight(*w);
        }
        this->non_layer_byte_size = helper.get_size();
    }

    /// @brief load the non-layer vlm weights from model.safetensors
    void load(SafeTensors& st, buffer<u8>& non_layer_weights) {
        for (weight_desc_t* w : non_layer_all()) {
            st.load_weights(non_layer_weights, w->name, w->offset);
            w->load();
        }
    }
};

#endif // __SMOLVLA__VLM__DEF__HPP__
