// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __SMOLVLA__HPP__
#define __SMOLVLA__HPP__

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"
#include "buffer.hpp"
#include "typedef.hpp"
#include "weight_desc.hpp"
#include "expert/SmolVLA_expert.hpp"
#include "vlm/SmolVLA_vlm.hpp"

class npu_xclbin_manager;

typedef struct : public smolvla_vlm_input_t {
    buffer<bf16> states;
} smolvla_input_t;

/// @brief Top-level SmolVLA policy. Owns the vision-language model (vlm) and the
///        action expert, plus the F32 action/state adapters that bridge the
///        robot action/state space and the expert hidden space.
class SmolVLA {
public:
    npu_xclbin_manager* xclbin_mgr;
    buffer<bf16> state_in;
    std::unique_ptr<SmolVLA_vlm>    vlm;
    std::unique_ptr<SmolVLA_expert> expert;

    // top-level F32 adapters (model.* tensors): action in/out projections, the
    // time-conditioned action MLP, and the state projection.
    weight_desc_t action_in_proj;
    weight_desc_t action_in_proj_bias;
    weight_desc_t action_out_proj;
    weight_desc_t action_out_proj_bias;
    weight_desc_t action_time_mlp_in;
    weight_desc_t action_time_mlp_in_bias;
    weight_desc_t action_time_mlp_out;
    weight_desc_t action_time_mlp_out_bias;
    weight_desc_t state_proj;
    weight_desc_t state_proj_bias;

    buffer<u8> adapter_weights;   // holds all of the F32 adapters above
    size_t adapter_byte_size = 0;

    buffer<bf16> out;

    // embed_actions folds action_in_proj into the first half of action_time_mlp_in (both are linear, nothing
    // between them but the bf16 rounding of action_emb): Wc = W1[:, :D] @ Wa [D_expert, action_dim] and
    // bias0 = W1[:, :D] @ ba, built once from the adapter weights. time_bias caches the per-timestep
    // time-embedding term. SMOLVLA_EMBED_FOLD=0 keeps the two separate GEMMs.
    std::vector<float> embed_wc, embed_bias0;
    std::map<float, std::vector<float>> embed_time_bias;


    SmolVLA(npu_xclbin_manager* xclbin_mgr) :
        xclbin_mgr(xclbin_mgr),
        vlm(std::make_unique<SmolVLA_vlm>(xclbin_mgr)),
        expert(std::make_unique<SmolVLA_expert>(xclbin_mgr)) {}

    /// @brief load config.json from a model directory and build the sub-modules
    /// @param model_path path to the model directory containing config.json
    void from_pretrained(const std::string& model_path);

    /// @brief load all weights (adapters + vlm + expert) from model.safetensors
    void load_weights();

    /// @brief run the vlm prefill over the observation, filling the language kv
    ///        cache (returned as a non-owning view). The expert compute is driven
    ///        separately (see expert_core / read_velocity / denoise) so the kv
    ///        handoff can be verified in isolation.
    buffer<bf16> predict(smolvla_input_t& model_input);

    /// @brief full policy step: run the vlm prefill (vision + connector +
    ///        language) over the observation, then integrate the flow-matching
    ///        denoise loop (10 Euler steps) over the action expert. Returns the
    ///        denoised action chunk [chunk_size, action_dim] (bf16).
    /// @param model_input the observation (images + language tokens + state)
    /// @param noise optional captured noise [chunk_size, action_dim] to seed the
    ///        denoise loop; if null, noise is sampled (Box-Muller from rand()).
    /// @param skip_prefill if true, reuse the kv cache already populated by a
    ///        prior predict()/forward() (or externally injected) instead of
    ///        re-running the vlm prefill. Lets a caller substitute a reference kv
    ///        cache to verify the expert in isolation.
    buffer<bf16> forward(smolvla_input_t& model_input, buffer<bf16>* noise = nullptr,
                         bool skip_prefill = false);

    /// @brief lift a noisy action chunk into the expert hidden space and add the
    ///        sinusoidal time embedding via the action_time MLP. [L, action_dim]
    ///        (bf16) -> [L, D_expert].
    /// @param x_t      noisy action chunk, [L, action_dim] (bf16)
    /// @param L        chunk length
    /// @param timestep flow-matching time in [0, 1]
    buffer<bf16> embed_actions(buffer<bf16>& x_t, int L, float timestep);

    /// @brief hoist every expert layer's view of the cached VLM kv states out of
    ///        the denoise loop: cross layers re-project them through their k/v
    ///        proj, self layers take them as the prefix half of their
    ///        [prefix ; chunk] K/V. Both depend only on the kv cache and the
    ///        frozen layer weights -- not on the denoise step -- so they are
    ///        hoisted here and reused by every expert_core() call over the same
    ///        prefix. Call after the kv cache is final (prefill or injection) and
    ///        before the first expert_core().
    void prepare_expert_prefix_kv();

    /// @brief run the action expert over one conditioned chunk, attending the
    ///        cached VLM kv states from the last predict(). [L, D_expert] in place.
    /// @note requires a prior prepare_expert_prefix_kv() over the current kv cache.
    buffer<bf16> expert_core(buffer<bf16>& chunk, int L);

    /// @brief read the predicted velocity out of the expert hidden state via
    ///        action_out_proj. [L, D_expert] -> [L, action_dim] (bf16).
    buffer<bf16> read_velocity(buffer<bf16>& hidden, int L);

private:
    std::string model_path;

    /// @brief the top-level F32 adapter descriptors
    std::vector<weight_desc_t*> adapter_all();

    /// @brief build the model from an already-parsed config.json object
    /// @param jcfg the parsed config.json
    void build(nlohmann::json& jcfg);
};

#endif // __SMOLVLA__HPP__
