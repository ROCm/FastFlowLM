// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __SMOLVLA__EXPERT__HPP__
#define __SMOLVLA__EXPERT__HPP__

#include <string>
#include <vector>

#include "nlohmann/json.hpp"
#include "buffer.hpp"
#include "typedef.hpp"
#include "tensor_utils/safe_tensors.hpp"
#include "SmolVLA_expert_def.hpp"
#include "SmolVLA_expert_exec.hpp"

class npu_xclbin_manager;

/// @brief SmolVLA action expert. Owns the parsed weight/parameter description
///        and the runtime weight buffers needed to execute it.
class SmolVLA_expert {
public:
    SmolVLA_Expert_def def;
    npu_xclbin_manager* xclbin_mgr;

    // weight buffers. one buffer per layer of each kind (self = even indices,
    // cross = odd indices), plus the top-level non-layer weights.
    std::vector<buffer<bf16>> self_layer_weights;
    std::vector<buffer<bf16>> cross_layer_weights;
    buffer<u8> non_layer_weights;

    // runtime execution context (compute skeleton)
    SmolVLA_expert_exec exec;

    SmolVLA_expert(npu_xclbin_manager* xclbin_mgr) : xclbin_mgr(xclbin_mgr) {}

    /// @brief build the expert from the "expert_config" object
    /// @param ecfg the "expert_config" object from config.json
    void build(const nlohmann::json& ecfg);

    /// @brief allocate the runtime weight buffers (call after build())
    void alloc_weights();

    /// @brief load the expert weights from an already-open model safetensors
    /// @param st opened SafeTensors handle for model.safetensors
    void load_weights(SafeTensors& st);

    /// @brief hoist every layer's view of the VLM prefix out of the denoise loop.
    ///        Cross layers re-project the prefix kv through their k/v proj; self
    ///        layers take the prefix half of their [prefix ; chunk] K/V. Both
    ///        depend only on the kv cache and the (frozen) layer weights, so they
    ///        are computed once here and reused by every denoising step. Must be
    ///        called after the vlm prefill and before forward().
    /// @param kv_cache  the VLM language kv cache, (2, num_vlm_layers, MAX_L, DK)
    /// @param P         valid VLM prefix length
    /// @param num_vlm_layers layer count of the VLM cache (== expert num_layers)
    /// @param MAX_L     per-layer row stride of the VLM cache
    void prepare_prefix_kv(buffer<bf16>& kv_cache, int P, int num_vlm_layers, int MAX_L);

    /// @brief run the interleaved expert stack over one action-token chunk for a
    ///        single denoising step. Requires a prior prepare_prefix_kv() over the
    ///        same kv cache and prefix length -- the cache is not re-read here.
    /// @param chunk     conditioned action-token hidden state, [L, D] (in place)
    /// @param L         chunk length
    /// @param P         valid VLM prefix length
    /// @return updated action-token hidden state (post final norm), [L, D]
    buffer<bf16> forward(buffer<bf16>& chunk, int L, int P);
};

#endif // __SMOLVLA__EXPERT__HPP__
