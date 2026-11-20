// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include "SmolVLA_vision.hpp"

#include "tensor_utils/safe_tensors.hpp"
#include "npu_utils/npu_utils.hpp"
#include "utils/prof_utils.hpp"

void SmolVLA_vision::build(const nlohmann::json& vcfg) {
    this->def.build(vcfg);
    this->exec = SmolVLA_vision_exec(this->xclbin_mgr, &this->def);
    // encoder input: TOKENS_PER_IMAGE rows of D per image.
    const size_t in_per_image = (size_t)smolvla_cfg::TOKENS_PER_IMAGE * def.D;
    // connector output: VISION_TOKENS_PER_IMAGE rows of D*sf^2 per image, which
    // is the same element count (the pixel shuffle only regroups).
    const size_t out_per_image = (size_t)smolvla_cfg::VISION_TOKENS_PER_IMAGE * def.D
                               * smolvla_cfg::SHUFFLE_FACTOR * smolvla_cfg::SHUFFLE_FACTOR;

    in_buffer  = buffer<bf16>(in_per_image * smolvla_cfg::MAX_IMAGES);
    in_tensor  = tensor_2d<bf16>(in_buffer, in_per_image);
    out_buffer = buffer<bf16>(out_per_image * smolvla_cfg::MAX_IMAGES);
    out_tensor = tensor_2d<bf16>(out_buffer, out_per_image);
}

void SmolVLA_vision::alloc_weights() {
    // buffer<T> has no move ctor (its copy ctor is shallow), so resize to
    // empty buffers first, then move-assign each bo-backed buffer into place.
    this->layer_weights.resize(this->def.num_layers);
    this->norm_weights.resize(this->def.num_layers);
    for (uint32_t l = 0; l < this->def.num_layers; l++) {
        this->layer_weights[l] =
            this->xclbin_mgr->create_bo_buffer<bf16>(this->def.layer_byte_size / sizeof(bf16));
        // CPU-only layer-norm weights: plain host buffer, no bo backing.
        this->norm_weights[l] = buffer<bf16>(this->def.norm_byte_size / sizeof(bf16));
    }
    this->non_layer_weights = buffer<u8>(this->def.non_layer_byte_size);
}

void SmolVLA_vision::load_weights(SafeTensors& st) {
    this->def.load(st, this->layer_weights, this->norm_weights, this->non_layer_weights);
    // prepack conv patch-embedding weights now (weight-dependent, image-
    // independent) so it is excluded from per-image patch_embed timing.
    this->exec.init(this->non_layer_weights);
}

buffer<bf16> SmolVLA_vision::forward(std::vector<buffer<bf16>>& pixel_tokens) {
    PROF_RESET();
    // patch + position embedding, one image at a time
    // -> in_buffer [num_images*TOKENS_PER_IMAGE, D]
    PROF_INIT();
    const int num_images = (int)pixel_tokens.size();

    for (int i = 0; i < num_images; i++) {
        this->exec.patch_embed(in_tensor[i], pixel_tokens[i], non_layer_weights);
    }
    PROF_TICK("Patch embedding");
    // num_layers identical pre-norm transformer layers (residual updated in
    // place). Each layer's inner ticks accumulate by name, so the report shows
    // one aggregated row per stage instead of one print per layer.
    for (uint32_t layer_idx = 0; layer_idx < def.num_layers; layer_idx++) {
        this->exec.forward(in_buffer, this->layer_weights[layer_idx], this->norm_weights[layer_idx], num_images);
    }
    PROF_TICK("All layers");

    // per image: final post-layernorm, then connector pixel-shuffle
    // [TOKENS_PER_IMAGE, D] -> [VISION_TOKENS_PER_IMAGE, D*sf^2] into out_buffer
    for (int i = 0; i < num_images; i++) {
        this->exec.post_layernorm(in_tensor[i], non_layer_weights);
        this->exec.pixel_shuffle(out_tensor[i], in_tensor[i]);
    }
    PROF_TICK("Pixel Shuffle!");

    PROF_REPORT();
    return out_buffer;
}
