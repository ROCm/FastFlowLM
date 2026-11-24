// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __SMOLVLA__VISION__HPP__
#define __SMOLVLA__VISION__HPP__

#include <string>
#include <vector>

#include "nlohmann/json.hpp"
#include "buffer.hpp"
#include "typedef.hpp"
#include "tensor_utils/safe_tensors.hpp"
#include "SmolVLA_vision_def.hpp"
#include "SmolVLA_vision_exec.hpp"
#include "tensor_2d.hpp"
class npu_xclbin_manager;

/// @brief SmolVLA vision encoder. Owns the parsed weight/parameter description
///        and the runtime weight buffers needed to execute it.
class SmolVLA_vision {
public:
    SmolVLA_Vision_def def;
    npu_xclbin_manager* xclbin_mgr;

    // weight buffers. one buffer per encoder layer, plus the non-layer weights.
    // norm_weights holds the CPU-only layer-norm weights (host memory, never
    // uploaded to the NPU), one buffer per encoder layer.
    std::vector<buffer<bf16>> layer_weights;
    std::vector<buffer<bf16>> norm_weights;
    buffer<u8> non_layer_weights;
    // encoder input: patch/pos-embedded tokens for up to MAX_IMAGES images,
    // [MAX_VISION_TOKENS, D]. in_tensor[i] views the i-th image's slice.
    buffer<bf16> in_buffer;
    tensor_2d<bf16> in_tensor;
    // connector output: pixel-shuffled tokens for up to MAX_IMAGES images,
    // [MAX_IMAGES*VISION_TOKENS_PER_IMAGE, D*sf^2]. out_tensor[i] views image i.
    buffer<bf16> out_buffer;
    tensor_2d<bf16> out_tensor;

    // runtime execution context (compute skeleton)
    SmolVLA_vision_exec exec;

    SmolVLA_vision(npu_xclbin_manager* xclbin_mgr) : xclbin_mgr(xclbin_mgr) {}

    /// @brief build the vision encoder from the "vision_config" object
    /// @param vcfg the "vision_config" object from config.json
    void build(const nlohmann::json& vcfg);

    /// @brief allocate the runtime weight buffers (call after build())
    void alloc_weights();

    /// @brief load the vision weights from an already-open model safetensors
    /// @param st opened SafeTensors handle for model.safetensors
    void load_weights(SafeTensors& st);

    /// @brief encode a batch of images into visual-token sequences. Each image is
    ///        patch-embedded into the shared in_buffer, then the whole batch runs
    ///        through the encoder stack together; post-layernorm and connector
    ///        pixel-shuffle are applied per image into out_buffer.
    /// @param pixel_tokens one raw pixel buffer per image (num_images entries)
    /// @return concatenated pixel-shuffled tokens for all images,
    ///         [num_images*VISION_TOKENS_PER_IMAGE, D*sf^2]
    buffer<bf16> forward(std::vector<buffer<bf16>>& pixel_tokens);
};

#endif // __SMOLVLA__VISION__HPP__
