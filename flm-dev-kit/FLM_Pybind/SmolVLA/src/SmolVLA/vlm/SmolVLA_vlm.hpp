// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __SMOLVLA__VLM__HPP__
#define __SMOLVLA__VLM__HPP__

#include <memory>

#include "nlohmann/json.hpp"
#include "buffer.hpp"
#include "typedef.hpp"
#include "SmolVLA_config.hpp"
#include "tensor_utils/safe_tensors.hpp"
#include "SmolVLA_vlm_def.hpp"
#include "vision/SmolVLA_vision.hpp"
#include "language/SmolVLA_language.hpp"

class npu_xclbin_manager;

typedef struct{
    std::vector<buffer<bf16>> pixel_values;
    std::vector<int> input_tokens;
} smolvla_vlm_input_t;

/// @brief SmolVLA vision-language model (SmolVLM2). Owns the vision encoder and
///        the text model, plus the shared non-layer weights (vision->text
///        connector and the LM head).
class SmolVLA_vlm {
public:
    SmolVLA_Vlm_def def;
    npu_xclbin_manager* xclbin_mgr;

    std::unique_ptr<SmolVLA_vision>   vision;
    std::unique_ptr<SmolVLA_language> language;

    buffer<bf16> language_in;
    bool conn_ready = false;       // connector weight uploaded to the NPU app

    buffer<u8> non_layer_weights;   // connector + lm_head

    SmolVLA_vlm(npu_xclbin_manager* xclbin_mgr) :
        xclbin_mgr(xclbin_mgr),
        vision(std::make_unique<SmolVLA_vision>(xclbin_mgr)),
        language(std::make_unique<SmolVLA_language>(xclbin_mgr)) {}

    /// @brief build the vlm from config.json.
    /// @param vcfg          the "vision_config" object
    /// @param num_vlm_layers number of text-model layers (top-level "num_vlm_layers")
    void build(const nlohmann::json& vcfg, uint32_t num_vlm_layers);

    /// @brief allocate all runtime weight buffers (call after build())
    void alloc_weights();

    /// @brief load all vlm weights from an already-open model safetensors
    /// @param st opened SafeTensors handle for model.safetensors
    void load_weights(SafeTensors& st);

    /// @brief run vlm prefill: vision encode -> connector -> language prefill.
    /// @param vlm_in   one pixel buffer per camera plus the prompt token ids
    /// @param state_in robot state embedding appended to the prefix, [1, D_text]
    /// @return the language kv cache for the expert to cross-attend over,
    ///         laid out (2, num_layers, MAX_PREFIX_LEN, DK)
    buffer<bf16> forward(smolvla_vlm_input_t& vlm_in, buffer<bf16>& state_in);
};

#endif // __SMOLVLA__VLM__HPP__
