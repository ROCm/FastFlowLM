// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __QWEN3VL_NPU_DETAIL_HPP__
#define __QWEN3VL_NPU_DETAIL_HPP__
#include "models/qwen3vl/qwen3vl_npu.hpp"
#include "qwen3vl_npu_sequence.hpp"
#include "qwen3vl_image.hpp"
#include "qwen3vl_model_def.hpp"
#include "qwen3vl_prefill.hpp"
#include "utils/kv_rows_state.hpp"

struct qwen3vl_npu::Impl{

    /// @brief token id standing in for one vision patch in the prompt
    static constexpr int image_soft_token_id = 151655;

    static constexpr int LC = 16;
    /// @brief  initialize the qwen3vl_npu
    /// @param config
    /// @param npu_instance
    Impl(LM_Config config, npu_xclbin_manager *npu_instance, int MAX_L = 4096);
    ~Impl();

    /// @brief forward the qwen3vl_npu
    buffer<bf16> forward(int ids);
    buffer<bf16> prefill(std::vector<int>& ids, void* payload = nullptr);
    buffer<bf16> _prefill_with_mv(std::vector<int>& ids);
    buffer<bf16> _prefill_with_mm(std::vector<int>& ids, void* payload = nullptr);

    void set_context_length(int L, int new_general_token_idx);
    void load_weights(Q4NX& q4nx);
    void update_max_length(uint32_t MAX_L);
    void clear_context();

    buffer<bf16> get_k_cache(int layer_idx, int idx);
    buffer<bf16> get_v_cache(int layer_idx, int idx);
    buffer<bf16> get_logits(buffer<bf16>& x);

    int get_current_context_length();
    int checkpoint();
    int restore();
    void _allocate_checkpoint_buffers();

    /// @brief saved states (see flm_state_api.h)
    kv_row_layout kv_layout() const;
    std::unique_ptr<hybrid_engine_state> new_state();
    int read_state(hybrid_engine_state& state, int kv_from);
    int write_state(const hybrid_engine_state& state);
    int write_state_kv(const hybrid_engine_state& state);
    /// @brief highest context length reached since the K/V rows past it were last zero (see kv_rows_zero)
    size_t dirty_rows = 0;
    bool is_checkpoint_valid;
    bool is_vlm;
    int checkpoint_context_length;
    int checkpoint_general_token_idx;
    std::vector<buffer<bf16>> kv_checkpoint;

    size_t K03_offset;
    size_t K47_offset;
    size_t V03_offset;
    size_t V47_offset;

    LM_Config config;
    npu_xclbin_manager *npu;

    /// @brief model dimensions, weight descriptors and weight loading
    qwen3vl_desc desc;
    /// @brief everything the mm-based prefill path needs
    std::unique_ptr<qwen3vl_prefill_context> pctx;

    /// @brief Walks the prompt, marking the vision tokens and assigning each
    ///        token its three mrope indices; advances general_token_idx.
    std::vector<int> get_input_idx(std::vector<bool>& vision_mask, std::vector<int>& tokens, qwen3vl_image_payload_t* image_payload);
    void _set_rope_rms_weights(int idx);

    /// @brief decoding path: one xclbin runs a whole layer
    npu_app_manager* layer_app_manager;
    npu_app layer;
    npu_app lm_head;

    std::unique_ptr<qwen3vl_npu_sequence> sequence;
    std::unique_ptr<Qwen3VLImageEncoder> qwen3vl_image_encoder;
    uint32_t MAX_L;

    std::vector<buffer<bf16>> rms_weights;
    /// @brief per layer: the rope cos/sin table, then the q_norm and k_norm rows
    std::vector<buffer<bf16>> rope_rms_weights;
    std::vector<buffer<u8>> proj_weights;
    std::vector<buffer<bf16>> kv_caches;

    buffer<bf16> x;
    std::unique_ptr<Embedding> embedding;

    buffer<bf16> final_rms_weight;
    buffer<bf16> logits;
    buffer<bf16> logits_valid;
    buffer<u8> lm_head_weights;

    flm_rt::runlist layers_run;
    flm_rt::run lm_head_run;
    flm_rt::runlist dequant_all;
    int current_context_length;
    /// @brief the temporal mrope clock; images advance it by their grid width
    int general_token_idx;
};

#endif
