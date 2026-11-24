// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file qwen3_5_omni_thinker.hpp
/// \brief qwen3_5_omni thinker (text LM backbone).
/// \author FastFlowLM Team
/// \note Language backbone migrated from qwen3_5vl_npu. Vision/audio/talker
///       are intentionally omitted for now.
#pragma once
#include "lm_config.hpp"
#include "npu_utils/npu_utils.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"
#include "modules/embedding.hpp"
#include "modules/lm_head.hpp"
#include "modules/gemm.hpp"
#include "modules/dequant.hpp"
#include "tensor_2d.hpp"
#include "utils/utils.hpp"
#include "embedding_q8_0.hpp"

#include "models/qwen3_5_omni/qwen3_5_omni.hpp"

class qwen3_5_omni_thinker {
public:
    qwen3_5_omni_thinker(LM_Config config, npu_xclbin_manager *npu_instance, int MAX_L = 4096);
    ~qwen3_5_omni_thinker();

    buffer<bf16> forward(int ids);
    buffer<bf16> prefill(std::vector<int>& ids, void* payload = nullptr);

    void set_context_length(int L);
    void load_weights(Q4NX& q4nx);
    void clear_context();

    buffer<bf16> get_k_cache(int layer_idx, int idx);
    buffer<bf16> get_v_cache(int layer_idx, int idx);

    void update_max_length(uint32_t MAX_L);

    int get_current_context_length();
    int checkpoint();
    int restore();

private:
    struct Impl;
    Impl* _impl;
};
