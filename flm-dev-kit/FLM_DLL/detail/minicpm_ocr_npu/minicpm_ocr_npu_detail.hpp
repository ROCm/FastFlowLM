// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#pragma once
#include "models/minicpm_ocr/minicpm_ocr_npu.hpp"
#include "minicpm_ocr_npu_sequence.hpp"
#include "image/image_module.hpp"
#include "image/mrope.hpp"
#include "image/vision_model_def.hpp"
#include "modules/gemm.hpp"
#include "minicpm_ocr_model_def.hpp"
#include "minicpm_ocr_prefill.hpp"
#include "utils/avx512_util.hpp"
#include "embedding_q8_0.hpp"
#include "decode_kv_bfp16.hpp"

struct minicpm_ocr_npu::Impl{
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
    bool is_checkpoint_valid;
    bool is_vlm;
    int checkpoint_context_length;
    int checkpoint_general_token_idx;
    minicpm_ocr_desc desc;
    std::unique_ptr<minicpm_ocr_prefill_context> pctx;
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

    std::unique_ptr<minicpm_ocr_npu_sequence> sequence;

    /// @brief the vision tower (image/). The desc must outlive the tower, which
    ///        holds it by reference -- hence declared first.
    minicpm_ocr_vision_desc vision_desc;
    std::unique_ptr<mco::image_module> image_mod;
    mco::canvas_token_ids canvas_ids;
    static constexpr int video_token_id = 248057;

    minicpm_ocr_npu* parent_ptr;

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
    /// @brief true: the lm_head runs on layer_with_lm_head.xclbin (README "lm_head inside layer.xclbin");
    ///        false: on lm_head.xclbin, after layer.xclbin
    bool lm_head_in_layer = false;
    /// @brief model.norm for the lm_head_in_layer run
    buffer<bf16> lm_head_norm;
    /// @brief true: decode runs on layer_with_lm_head_kvb.xclbin, whose attention reads the finished KV chunks from
    ///        decode_kv as bfp16 (README "Decode attention on bfp16 K and V")
    bool decode_kv_bfp16 = false;
    minicpm_ocr_decode_kv::store decode_kv;
    /// @brief the one-token hidden state of a host decode (MINICPM_OCR_DECODE_ON_NPU == 0)
    buffer<bf16> decode_hidden;

    flm_rt::runlist layers_run;
    flm_rt::run lm_head_run;
    int current_context_length;
    int general_token_idx;

    /// @brief  initialize the minicpm_ocr_npu
    /// @param config
    /// @param npu_instance
    Impl(LM_Config config, npu_xclbin_manager *npu_instance, minicpm_ocr_npu* parent_ptr, int MAX_L = 4096);
    ~Impl();

    /// @brief forward the minicpm_ocr_npu
    buffer<bf16> forward(int ids);
    buffer<bf16> prefill(std::vector<int>& ids, void* payload = nullptr);
    buffer<bf16> _prefill_with_mm(std::vector<int>& ids, void* payload = nullptr);

    void set_context_length(int L, int new_general_token_idx);
    void load_weights(Q4NX& q4nx);
    void update_max_length(uint32_t MAX_L);
    void clear_context();

    buffer<bf16> get_k_cache(int layer_idx, int idx);
    buffer<bf16> get_v_cache(int layer_idx, int idx);
    buffer<bf16> get_logits(buffer<bf16>& x);

    int get_current_context_length();

    /// @brief build the vision tower on the array, or leave the engine text-only
    void _load_vision();
    /// @brief encode the payload's views into [rows][D] bf16 rows for the prompt's
    ///        image_token_id positions; fills `views` with the payload's grids
    std::vector<bf16> _encode_images(const minicpm_ocr_image_payload_t* p, int n_image_tokens,
                                     std::vector<mco::view_grid>& views);
    void _set_rope_rms_weights(int idx);
    /// @brief decode_kv up to date for a run with L cached tokens; true if it had to grow (runs must be made again)
    bool _decode_kv_catch_up(int L);
    void _check_descriptors(Q4NX& q4nx);
    void _lm_head_host();
    inline bool is_attn_layer(int layer_idx) {return (layer_idx % full_attention_interval) == (full_attention_interval - 1); }
    inline bool is_first_linear_layer(int layer_idx) {return (layer_idx % full_attention_interval) == 0; }

    int checkpoint();
    int restore();
    void _allocate_checkpoint_buffers();

    std::unique_ptr<hybrid_engine_state> new_state();
    int read_state(hybrid_engine_state& state, int kv_from);
    int write_state(const hybrid_engine_state& state);
    int write_state_kv(const hybrid_engine_state& state);
    size_t state_row_bytes();
    size_t state_fixed_bytes();
};
