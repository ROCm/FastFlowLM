// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#ifndef __LFM2_NPU_DETAIL_HPP__
#define __LFM2_NPU_DETAIL_HPP__
#include "models/lfm2/lfm2_npu.hpp"
#include "lfm2_model_def.hpp"
#include "lfm2_prefill.hpp"
#include "utils/kv_rows_state.hpp"

struct lfm2_npu::Impl{

    static constexpr int LC = 16;
    /// @brief  initialize the lfm2_npu
    /// @param config
    /// @param npu_instance
    Impl(LM_Config config, npu_xclbin_manager *npu_instance, int MAX_L = 4096);
    ~Impl();

    /// @brief forward the lfm2_npu
    buffer<bf16> forward(int ids);
    buffer<bf16> prefill(std::vector<int>& ids, void* payload = nullptr);
    buffer<bf16> _prefill_with_mv(std::vector<int>& ids);
    buffer<bf16> _prefill_with_mm(std::vector<int>& ids);

    void set_context_length(int L);
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

    kv_row_layout kv_layout() const;
    std::unique_ptr<hybrid_engine_state> new_state();
    int read_state(hybrid_engine_state& state, int kv_from);
    int write_state(const hybrid_engine_state& state);
    int write_state_kv(const hybrid_engine_state& state);
    /// @brief one flag per layer: attention (keeps K/V rows) or convolution (keeps a fixed blob)
    std::vector<bool> attn_mask;
    /// @brief highest context length reached since the rows past the length were last zero (see kv_rows_zero)
    size_t dirty_rows = 0;
    bool is_checkpoint_valid;
    int checkpoint_context_length;
    std::vector<buffer<bf16>> kv_checkpoint;

    size_t K03_offset;
    size_t K47_offset;
    size_t V03_offset;
    size_t V47_offset;

    LM_Config config;
    npu_xclbin_manager *npu;

    /// @brief model dimensions, weight descriptors and weight loading
    lfm2_desc desc;
    /// @brief everything the mm-based prefill path needs
    std::unique_ptr<lfm2_prefill_context> pctx;

    inline bool is_attn_layer(uint32_t layer_idx) const { return desc.is_attn_layer(layer_idx); }

    /// @brief decoding path: one xclbin runs a whole layer, in two flavours
    npu_app_manager* layer_app_manager;
    npu_app attn_layer;
    npu_app conv_layer;
    npu_app lm_head;

    std::unique_ptr<lfm2_npu_sequence> sequence;
    uint32_t MAX_L;

    std::vector<buffer<bf16>> rms_weights;
    /// @brief per layer: the rope cos/sin table plus q_norm and k_norm on an
    ///        attention layer, the transposed convolution kernel on a conv layer
    std::vector<buffer<bf16>> rope_rms_weights;
    std::vector<buffer<u8>> proj_weights;
    /// @brief the kv cache on an attention layer, the two conv history rows on a
    ///        conv layer
    std::vector<buffer<bf16>> kv_caches;

    buffer<bf16> x;
    std::unique_ptr<Embedding> embedding;

    buffer<bf16> final_rms_weight;
    buffer<bf16> logits;
    buffer<u8> lm_head_weights;

    flm_rt::runlist dequant_all;
    flm_rt::runlist layers_run;
    flm_rt::run lm_head_run;
    int current_context_length;
};

#endif
