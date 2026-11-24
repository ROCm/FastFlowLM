// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#include <array>
#include "utils/hybrid_state.hpp"
#pragma once
#include "models/minicpm_v_4_7_moe/minicpm_v_4_7_moe_npu.hpp"
#include "minicpm_v_4_7_moe_npu_sequence.hpp"
#include "image/image_module.hpp"
#include "image/mrope.hpp"
#include "image/vision_model_def.hpp"
#include "modules/gemm.hpp"
#include "minicpm_v_4_7_moe_model_def.hpp"
#include "minicpm_v_4_7_moe_prefill.hpp"
#include "utils/avx512_util.hpp"

struct minicpm_v_4_7_moe_npu::Impl{
    /// @brief constexprs
    static constexpr f32 inv_freq[32] = {
        1.000000000e+00, 6.042963862e-01, 3.651741445e-01, 2.206733972e-01, 1.333521456e-01, 8.058421314e-02, 4.869675264e-02, 2.942727320e-02,
        1.778279431e-02, 1.074607857e-02, 6.493816618e-03, 3.924189601e-03, 2.371373819e-03, 1.433012658e-03, 8.659643354e-04, 5.232990952e-04,
        3.162277862e-04, 1.910952997e-04, 1.154781930e-04, 6.978305464e-05, 4.216965317e-05, 2.548296652e-05, 1.539926416e-05, 9.305720596e-06,
        5.623413472e-06, 3.398208264e-06, 2.053524895e-06, 1.240937763e-06, 7.498942409e-07, 4.531583784e-07, 2.738419767e-07, 1.654817083e-07,
    };

    static constexpr int image_soft_token_id = 248056;
    static constexpr int full_attention_interval = 4;
    static constexpr int LC = 16;

    LM_Config config;
    npu_xclbin_manager *npu;
    bool is_vlm;
    minicpm_v_4_7_moe_desc desc;
    std::unique_ptr<minicpm_v_4_7_moe_prefill_context> pctx;
    uint32_t MAX_L;
    /// @brief the one copy checkpoint()/restore() keep; saved states of any other kind live with the caller
    std::unique_ptr<hybrid_engine_state> legacy_checkpoint;

    npu_app_manager* layer_app_manager;
    npu_app_manager* lm_head_app_manager;

    npu_app layer;
    npu_app first_linear_layer;
    npu_app linear_layer;
    npu_app set_layer_rtp;
    npu_app lm_head;

    int vocab_size;
    int vocab_size_padded;
    uint32_t V_offset;

    std::unique_ptr<minicpm_v_4_7_moe_npu_sequence> sequence;
    minicpm_v_moe_vision_desc vision_desc;
    std::unique_ptr<mcvm::image_module> image_mod;
    mcvm::canvas_token_ids canvas_ids;

    minicpm_v_4_7_moe_npu* parent_ptr;

    // size
    std::vector<buffer<bf16>> rms_weights;
    std::vector<buffer<bf16>> rope_rms_weights;
    std::vector<buffer<u8>> proj_weights;
    std::vector<buffer<bf16>> kv_caches;

    buffer<bf16> x;
    std::unique_ptr<Embedding> embedding;

    buffer<bf16> logits;
    buffer<bf16> logits_valid;
    buffer<u8> lm_head_weights;

    flm_rt::runlist layers_run;
    flm_rt::run lm_head_run;
    int current_context_length;
    int general_token_idx;

    /// @brief  initialize the qwen_npu
    /// @param config
    /// @param npu_instance
    Impl(LM_Config config, npu_xclbin_manager *npu_instance, minicpm_v_4_7_moe_npu* parent_ptr, int MAX_L = 4096);
    ~Impl();

    /// @brief forward the qwen_npu
    buffer<bf16> forward(int ids);
    buffer<bf16> prefill(std::vector<int>& ids, void* payload = nullptr);
    buffer<bf16> _prefill_with_mv(std::vector<int>& ids, void* payload = nullptr);
    buffer<bf16> _prefill_with_mm(std::vector<int>& ids, void* payload = nullptr);

    void set_context_length(int L, int new_general_token_idx);
    void load_weights(Q4NX& q4nx);
    void update_max_length(uint32_t MAX_L);
    void clear_context();

    buffer<bf16> get_k_cache(int layer_idx, int idx);
    buffer<bf16> get_v_cache(int layer_idx, int idx);
    buffer<bf16> get_logits(buffer<bf16>& x);

    int get_current_context_length();

    static constexpr int video_token_id = 248057;
    void _load_vision();
    std::vector<bf16> _encode_images(const minicpm_v_moe_image_payload_t* p, int n_image_tokens,
                                     std::vector<mcvm::view_grid>& views);
    void _set_rope_rms_weights(int idx);
    inline bool is_attn_layer(int layer_idx) {return (layer_idx % full_attention_interval) == (full_attention_interval - 1); }
    inline bool is_first_linear_layer(int layer_idx) {return (layer_idx % full_attention_interval) == 0; }

    int read_state(hybrid_engine_state& state, int kv_from);
    int write_state(const hybrid_engine_state& state);
    int write_state_kv(const hybrid_engine_state& state);
    std::unique_ptr<hybrid_engine_state> new_state();
    /// @brief host bytes one saved K/V row costs, summed over the attention layers (K and V, bf16)
    size_t state_row_bytes();
    /// @brief host bytes of the recurrent state, summed over the linear layers
    size_t state_fixed_bytes();
};
