// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file gpt_oss_npu.hpp
/// \brief gpt_oss_npu class
/// \author FastFlowLM Team
/// \date 2026-01-23
/// \version 0.9.28
/// \note This is a header file for the gpt_oss_npu class
#pragma once
#include "lm_config.hpp"
#include "npu_utils/npu_utils.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"
#include "models/gpt_oss/gpt_oss_npu_sequence.hpp"
#include "modules/embedding.hpp"
#include "modules/lm_head.hpp"
#include "modules/gemm.hpp"
#include "modules/dequant.hpp"
#include "tensor_2d.hpp"
#include "utils/utils.hpp"
#include "causal_lm.hpp"
#include "utils/hybrid_state.hpp"
#if USEAVX2
#include <immintrin.h>  // For AVX intrinsics
#endif


class gpt_oss_npu : public causal_lm{
public:
    /// \brief  initialize the gpt_oss_npu
    /// \param config the configuration
    /// \param npu_instance the npu instance
    gpt_oss_npu(LM_Config config, npu_xclbin_manager *npu_instance, int MAX_L = 4096);
    ~gpt_oss_npu();

    /// \brief forward the gpt_oss_npu
    /// \param ids the ids
    /// \return the output tensor
    buffer<bf16> forward(int ids) override;
    buffer<bf16> forward(buffer<bf16>& x);
    buffer<bf16> prefill(std::vector<int>& ids, void* payload = nullptr) override;

    /// \brief set the context length
    /// \param L the context length
    void set_context_length(int L) override;

    /// \brief load the weights
    /// \param q4nx the q4nx
    void load_weights(Q4NX& q4nx) override;

    /// \brief update the max length
    void clear_context() override;

    /// \brief get the k cache
    /// \param layer_idx the layer index
    /// \param idx the index
    /// \return the k cache
    buffer<bf16> get_k_cache(int layer_idx, int idx) override;

    /// \brief get the v cache
    /// \param layer_idx the layer index
    /// \param idx the index
    /// \return the v cache
    buffer<bf16> get_v_cache(int layer_idx, int idx) override;

    /// \brief update the max length
    /// \param MAX_L the max length
    void update_max_length(uint32_t MAX_L) override;

    /// \brief get the current context length
    /// \return the current context length
    int get_current_context_length() override;
    int checkpoint() override;
    int restore() override;

    // ---- saved states (see flm_state_api.h) --------------------------------
    // A state holds the 128-row ring of every sliding-window layer (valid at exactly the length it was read at:
    // a ring cannot be rewound) and the K/V rows of the global layers. read_state() copies them out of the live
    // buffers, write_state() puts a state back (and sets the context length), write_state_kv() puts back only the
    // global rows (an ancestor's head, before a state taken further on is written).
    std::unique_ptr<hybrid_engine_state> new_state();
    int read_state(hybrid_engine_state& state, int kv_from);
    int write_state(const hybrid_engine_state& state);
    int write_state_kv(const hybrid_engine_state& state);
private:
    struct Impl;
    Impl* _impl;
};

/// \brief the saved-state table of this engine (flm_state_api.h); nullptr if `engine` is not a gpt_oss_npu
extern "C" const flm_state_api* gpt_oss_npu_state_api(void* engine);

