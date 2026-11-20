// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include "SmolVLA_expert.hpp"

#include "tensor_utils/safe_tensors.hpp"
#include "npu_utils/npu_utils.hpp"

void SmolVLA_expert::build(const nlohmann::json& ecfg) {
    this->def.build(ecfg);
    this->exec = SmolVLA_expert_exec(this->xclbin_mgr, &this->def);
}

void SmolVLA_expert::alloc_weights() {
    uint32_t num_self  = (this->def.num_layers + 1) / 2;  // even indices
    uint32_t num_cross = this->def.num_layers / 2;        // odd indices

    // buffer<T> has no move ctor (its copy ctor is shallow), so resize to
    // empty buffers first, then move-assign each bo-backed buffer into place.
    this->self_layer_weights.resize(num_self);
    for (uint32_t s = 0; s < num_self; s++) {
        this->self_layer_weights[s] =
            this->xclbin_mgr->create_bo_buffer<bf16>(this->def.self_layer_byte_size / sizeof(bf16));
    }
    this->cross_layer_weights.resize(num_cross);
    for (uint32_t c = 0; c < num_cross; c++) {
        this->cross_layer_weights[c] =
            this->xclbin_mgr->create_bo_buffer<bf16>(this->def.cross_layer_byte_size / sizeof(bf16));
    }
    this->non_layer_weights = this->xclbin_mgr->create_bo_buffer<u8>(this->def.non_layer_byte_size);
}

void SmolVLA_expert::load_weights(SafeTensors& st) {
    this->def.load(st, this->self_layer_weights, this->cross_layer_weights, this->non_layer_weights);
}

void SmolVLA_expert::prepare_prefix_kv(buffer<bf16>& kv_cache, int P,
                                       int num_vlm_layers, int MAX_L) {
    this->exec.prepare_prefix_kv(this->cross_layer_weights, kv_cache, P,
                                 num_vlm_layers, MAX_L);
}

buffer<bf16> SmolVLA_expert::forward(buffer<bf16>& chunk, int L, int P) {
    return this->exec.forward(chunk, L, this->self_layer_weights,
                              this->cross_layer_weights, this->non_layer_weights, P);
}
