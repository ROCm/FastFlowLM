// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#pragma once
#include "models/qwen3_5_omni/qwen3_5_omni.hpp"
#include "qwen3_5_omni_thinker.hpp"
#include "qwen3_5_omni_thinker_sequence.hpp"
#include "modules/gemm.hpp"
#include "qwen3_5_omni_thinker_def.hpp"
#include "qwen3_5_omni_thinker_prefill.hpp"
#include "avx512_util.hpp"
#include "qwen3_5_omni_image_embed/qwen3_5_omni_image_embed.hpp"
#include "qwen3_5_omni_audio_embed/qwen3_5_omni_audio_embed.hpp"
#include "embedding_q8_0.hpp"

struct qwen3_5_omni_thinker::Impl{
    /// @brief constexprs
    static constexpr int LC = 16;

    LM_Config config;
    npu_xclbin_manager *npu;
    bool is_checkpoint_valid;
    bool is_vlm;
    bool is_audio;
    int checkpoint_context_length;
    int checkpoint_general_token_idx;
    qwen3_5_omni_thinker_desc desc;
    std::unique_ptr<qwen3_5_omni_thinker_prefill_context> pctx;
    uint32_t MAX_L;
    std::vector<buffer<bf16>> cache_checkpoint;

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

    std::unique_ptr<qwen3_5_omni_thinker_sequence> sequence;
    std::unique_ptr<qwen3_5_omni_image_embed> image_encoder;
    std::unique_ptr<qwen3_5_omni_audio_embed> audio_encoder;
    // size
    std::vector<buffer<bf16>> rms_weights;
    std::vector<buffer<bf16>> rope_rms_weights;
    std::vector<buffer<u8>> proj_weights;
    std::vector<buffer<bf16>> kv_caches;
    std::vector<buffer<bf16>> alpha_proj_cpu;
    std::vector<buffer<bf16>> beta_proj_cpu;

    buffer<bf16> x;
    /// @brief bf16 token embedding table, used when the checkpoint carries
    ///        model.embed_tokens.weight. Exactly one of these two is non-null.
    std::unique_ptr<Embedding> embedding;
    /// @brief fallback table dequantized out of the tied q8_0 lm_head.
    std::unique_ptr<embedding_q8_0> embedding_q8;

    /// @brief token embedding lookup against whichever table was loaded
    void embed_forward(int id, vdtype& y){
        if (this->embedding){
            this->embedding->forward(id, y);
        }
        else{
            this->embedding_q8->forward(id, y);
        }
    }

    buffer<bf16> logits;
    buffer<bf16> logits_valid;
    buffer<u8> lm_head_weights;

    flm_rt::runlist layers_run;
    flm_rt::run lm_head_run;
    int current_context_length;
    int general_token_idx;

    /// @brief  initialize the thinker
    /// @param config
    /// @param npu_instance
    Impl(LM_Config config, npu_xclbin_manager *npu_instance, int MAX_L = 4096);
    ~Impl();

    /// @brief forward the thinker
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

    std::vector<int> get_input_idx(std::vector<bool>& vision_mask, std::vector<int>& tokens, qwen3_5_omni_image_payload_t* image_payload);
    void _set_rope_rms_weights(int idx);
    inline bool is_attn_layer(int layer_idx) {return desc.is_attn_layer(layer_idx); }
    inline bool is_first_linear_layer(int layer_idx) {return (layer_idx % desc.full_attention_interval) == 0; }

    int checkpoint();
    int restore();
    void _allocate_checkpoint_buffers();
};
