/// \file hy_mem_npu.hpp
/// \brief hy_mem_npu class
/// \author FastFlowLM Team
/// \date 2026-10-26
/// \note The engine for Hy-Memory-30B-A3B (hy_v3): 48 layers, layer 0 dense and
///       layers 1..47 MoE (128 experts, top 8, + a shared expert), Q4_K body,
///       Q8_0 lm_head. Text only. Decode runs on three bitstreams
///       (dense_layer / moe_layer / lm_head .xclbin); prefill feeds the prompt
///       through the same decode overlays one token at a time.
#pragma once
#include "lm_config.hpp"
#include "npu_utils/npu_utils.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"
#include "modules/embedding.hpp"
#include "tensor_2d.hpp"
#include "utils/utils.hpp"
#include "causal_lm.hpp"


class hy_mem_npu : public causal_lm{
public:
    /// \brief  initialize the hy_mem_npu
    /// \param config the configuration
    /// \param npu_instance the npu instance
    /// \param MAX_L KV cache length in tokens; rounded up to a power of two in
    ///        [4096, 131072]
    hy_mem_npu(LM_Config config, npu_xclbin_manager *npu_instance, int MAX_L = 16384);
    ~hy_mem_npu();

    /// \brief decode one token
    /// \param ids the token
    /// \return the logits (vocab_size)
    buffer<bf16> forward(int ids) override;

    /// \brief run the prompt into the KV cache
    /// \return the logits of the last prompt token
    buffer<bf16> prefill(std::vector<int>& ids, void* payload = nullptr) override;

    /// \brief set the context length
    /// \param L the context length
    void set_context_length(int L) override;

    /// \brief load the weights
    /// \param q4nx the q4nx
    void load_weights(Q4NX& q4nx) override;

    /// \brief clear the context
    void clear_context() override;

    /// \brief get the k cache
    /// \param layer_idx the layer index
    /// \param idx the token index
    /// \return K of that token, all KV heads in head order
    buffer<bf16> get_k_cache(int layer_idx, int idx) override;

    /// \brief get the v cache
    /// \param layer_idx the layer index
    /// \param idx the token index
    /// \return V of that token, all KV heads in head order
    buffer<bf16> get_v_cache(int layer_idx, int idx) override;

    /// \brief grow the KV caches, keeping the conversation
    /// \param MAX_L the max length
    void update_max_length(uint32_t MAX_L) override;

    /// \brief get the current context length
    /// \return the current context length
    int get_current_context_length() override;
    int checkpoint() override;
    int restore() override;
private:
    struct Impl;
    Impl* _impl;
};
