// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __SMOLVLA__LANGUAGE__HPP__
#define __SMOLVLA__LANGUAGE__HPP__

#include <string>
#include <vector>

#include "nlohmann/json.hpp"
#include "buffer.hpp"
#include "typedef.hpp"
#include "tensor_utils/safe_tensors.hpp"
#include "SmolVLA_language_def.hpp"
#include "SmolVLA_language_exec.hpp"

class npu_xclbin_manager;

/// @brief SmolVLA text model (SmolVLM2 language tower). Owns the parsed
///        weight/parameter description and the runtime weight buffers.
class SmolVLA_language {
public:
    SmolVLA_Language_def def;
    npu_xclbin_manager* xclbin_mgr;

    // max context length the kv cache is sized for (matches the attn block).
    static constexpr int MAX_L = smolvla_cfg::MAX_PREFIX_LEN;

    // weight buffers. one buffer per text layer, plus the non-layer weights.
    std::vector<buffer<bf16>> layer_weights;
    std::vector<buffer<bf16>> norm_weights;
    buffer<u8> non_layer_weights;

    // full kv cache for all layers. Laid out as (2, num_layers, MAX_L, DK): all
    // layer keys first (layer.0.k, layer.1.k, ...), then all layer values
    // (layer.0.v, layer.1.v, ...). Keys are post-rope, values are pre-rope. This
    // is what the action expert attends over.
    buffer<bf16> kv_cache;

    // valid (non-padded) prefix length of the last forward pass: image tokens +
    // real text tokens + 1 state token. The expert attends over these P prefix
    // positions of the kv cache.
    int valid_prefix_len = 0;

    // runtime execution context (compute skeleton)
    SmolVLA_language_exec exec;

    SmolVLA_language(npu_xclbin_manager* xclbin_mgr) : xclbin_mgr(xclbin_mgr) {}

    /// @brief build the text model from its layer count
    /// @param num_layers number of text-model layers (top-level "num_vlm_layers")
    void build(uint32_t num_layers);

    /// @brief allocate the runtime weight buffers (call after build())
    void alloc_weights();

    /// @brief load the text weights from an already-open model safetensors
    /// @param st opened SafeTensors handle for model.safetensors
    void load_weights(SafeTensors& st);

    /// @brief run text-model prefill over the multimodal token sequence,
    ///        populating the full kv cache for every layer.
    /// @param embeds          scratch holding the connector's image tokens; the
    ///                        text and state embeddings are appended in place, [L, D]
    /// @param state           projected robot-state token, [1, D]
    /// @param language_tokens prompt token ids (pad-terminated)
    /// @param num_images      number of cameras in this prefix
    /// @param L               padded token count (recomputed to the exact length)
    /// @return the full kv cache, (2, num_layers, MAX_L, DK) (non-owning view).
    buffer<bf16> forward(buffer<bf16>& embeds, buffer<bf16>& state, std::vector<int>& language_tokens, int num_images, int L);
};

#endif // __SMOLVLA__LANGUAGE__HPP__
