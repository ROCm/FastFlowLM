// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include "SmolVLA_vlm.hpp"

#include "tensor_utils/safe_tensors.hpp"
#include "npu_utils/npu_utils.hpp"
#include "utils/prof_utils.hpp"

void SmolVLA_vlm::build(const nlohmann::json& vcfg, uint32_t num_vlm_layers) {
    this->vision->build(vcfg);
    this->language->build(num_vlm_layers);

    // connector maps flattened vision features into the text hidden size. Its
    // input width is fixed by the SmolVLM2 pixel-shuffle (D_vision * sf^2).
    const uint32_t D_text     = this->language->def.D;
    const uint32_t vocab_size = this->language->def.vocab_size;
    const uint32_t conn_in    = smolvla_cfg::CONNECTOR_IN;

    language_in.resize(smolvla_cfg::MAX_PREFIX_LEN * D_text);
    this->def.build(D_text, vocab_size, conn_in);
}

void SmolVLA_vlm::alloc_weights() {
    this->vision->alloc_weights();
    this->language->alloc_weights();
    this->non_layer_weights = buffer<u8>(this->def.non_layer_byte_size);
}

void SmolVLA_vlm::load_weights(SafeTensors& st) {
    this->def.load(st, this->non_layer_weights);
    this->vision->load_weights(st);
    this->language->load_weights(st);
}

buffer<bf16> SmolVLA_vlm::forward(smolvla_vlm_input_t& vlm_in, buffer<bf16>& state_in) {
    const int num_images = (int)vlm_in.pixel_values.size();
    // one prefix chunk per camera plus one for the text prompt + robot state.
    const int chunks_required = num_images + 1;
    const int L = chunks_required * smolvla_cfg::VISION_TOKENS_PER_IMAGE;

    // Vision and language each own the shared registry for the span of their
    // own report (they PROF_RESET on entry), so the tower totals here are kept
    // in locals instead — they would otherwise be wiped by the child resets.
    PROF_INIT();
    PROF_ACC_DECL(vision);
    PROF_ACC_DECL(connector);
    PROF_ACC_DECL(language);

    // do image embedding
    buffer<bf16> connector_mm_w = this->def.connector.locate_myself<bf16>(this->non_layer_weights);
    auto last_hidden = this->vision->forward(vlm_in.pixel_values);
    PROF_ACC(vision);
    cpu_func::_cpu_mm(language_in.data(), connector_mm_w.data(), last_hidden.data(),
                      smolvla_cfg::VISION_TOKENS_PER_IMAGE * num_images,
                      smolvla_cfg::CONNECTOR_IN, this->language->def.D);
    PROF_ACC(connector);
    buffer<bf16> kv_cache = this->language->forward(language_in, state_in, vlm_in.input_tokens, num_images, L);
    PROF_ACC(language);

    PROF_ACC_PRINT("vlm.vision   ", vision);
    PROF_ACC_PRINT("vlm.connector", connector);
    PROF_ACC_PRINT("vlm.language ", language);
    return kv_cache;
}
