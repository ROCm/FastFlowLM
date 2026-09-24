#pragma once
#include "models/gemma4e/gemma4e_npu.hpp"
#include "gemma4e_npu_sequence.hpp"
#include "gemma4e_image.hpp"
#include "modules/gemm.hpp"
#include "reorder_cpy.hpp"
#include "avx512_util.hpp"
#include "gemma4e_audio.hpp"
#include "embedding_q8_0.hpp"
#include "gemma4e_prefill.hpp"

struct gemma4e_npu::Impl{
    /// @brief constexprs

    // some model specific template variables
    static constexpr int boi_token_id = 255999; // begin of image token id
    static constexpr int image_token_id = 258880; // image token id
    static constexpr int eoi_token_id = 258882; // end of image token id

    static constexpr int boa_token_id = 256000; // begin of audio token id
    static constexpr int audio_token_id = 258881; // audio token id
    static constexpr int eoa_token_id = 258883; // end of audio token id
    /// @brief constexprs
    //TODO: FIXME:

    static constexpr int LC = 16;

    size_t V_offset;

    int vocab_size;
    int vocab_size_padded;
    bool is_preload_launched;

    LM_Config config;
    npu_xclbin_manager *npu;

    npu_app_manager* layer_app_manager;
    npu_app_manager* lm_head_app_manager;

    npu_app global_layer;
    npu_app swa_layer;
    npu_app global_skip_layer;
    npu_app swa_skip_layer;
    npu_app layer_pre_load;
    flm_rt::run pre_load_run;

    // per layer input
    npu_app per_layer_input_down_proj;
    npu_app per_layer_input_up_proj;

    npu_app lm_head;

    std::unique_ptr<gemma4e_npu_sequence> sequence;
    std::unique_ptr<Gemma4e_ImageEncoder> gemma4e_image_encoder;
    std::unique_ptr<Gemma4e_AudioEncoder> gemma4e_audio_encoder;

    gemma4e_npu* parent_ptr;

    int D;
    int DH;
    int DQ;
    int DK;
    int DV;
    int SWA_DH;
    int SWA_DQ;
    int SWA_DK;
    int SWA_DV;

    int MAX_L;
    int HIDDEN_SIZE;
    int INTERMEDIATE_SIZE;
    int PLI_D;
    int SLIDING_LENGTH;

    int sliding_layer_interval;
    int num_hidden_layers;
    int num_kv_shared_layers;
    int non_skip_layers;
    int global_layer_period;
    int last_swa_kv_cache_layer_idx;
    int last_global_kv_cache_layer_idx;
    float final_logit_softcapping;

    bool enable_double_wide_mlp;

    /// \brief Config-derived dimensions and every weight descriptor of the model.
    /// \note Owns the buffer layout: the dequant sequences, the decode sequence
    ///       generator and load_weights all address weights through it, so the
    ///       quantization type is switchable from gemma4e_desc::PROJ_DTYPE alone.
    gemma4e_desc desc;
    std::vector<gemma4e_layer_type_t> layer_types;

    // size
    std::vector<buffer<bf16>> rms_weights;
    std::vector<buffer<bf16>> rope_rms_weights;
    std::vector<buffer<u8>> proj_weights;
    std::vector<buffer<bf16>> kv_caches;
    std::vector<buffer<bf16>> pli_gate_up_weights;

    std::vector<buffer<bf16>> kv_checkpoint;

    buffer<bf16> pli_down_weights;
    buffer<float> layer_scale;
    /// Everything the prefill path needs: its own xclbins, sequences and buffers.
    std::unique_ptr<gemma4e_prefill_context> prefill_ctx;

    buffer<bf16> x;
    std::unique_ptr<embedding_q8_0> embedding;
    std::unique_ptr<embedding_q8_0> pli_embedding;

    buffer<bf16> logits;
    buffer<bf16> logits_valid;
    buffer<u8> lm_head_weights;

    flm_rt::runlist layers_run;
    flm_rt::run lm_head_run;
    flm_rt::runlist dequant_all;
    int current_context_length;

    int checkpoint_context_length;

    /// @brief  initialize the qwen_npu
    /// @param config
    /// @param npu_instance
    Impl(LM_Config config, npu_xclbin_manager *npu_instance, gemma4e_npu* parent_ptr, int MAX_L = 4096);
    ~Impl(); // waits for any in-flight preload run before tearing down

    /// @brief forward the qwen_npu
    buffer<bf16> forward(int ids);
    buffer<bf16> prefill(std::vector<int>& ids, void* payload = nullptr);
    buffer<bf16> _prefill_with_mv(std::vector<int>& ids, void* payload = nullptr);
    buffer<bf16> _prefill_with_mm(std::vector<int>& ids, void* payload = nullptr);
    void _load_attn_layer_weights(Q4NX& q4nx, int layer_idx);
    void _load_linear_layer_weights(Q4NX& q4nx, int layer_idx);

    void set_context_length(int L);
    void load_weights(Q4NX& q4nx);
    void update_max_length(uint32_t MAX_L);
    void clear_context();

    bool is_checkpoint_valid;
    bool is_vlm;
    bool is_audio;
    void _allocate_checkpoint_buffers();
    int checkpoint();
    int restore();

    buffer<bf16> get_k_cache(int layer_idx, int idx);
    buffer<bf16> get_v_cache(int layer_idx, int idx);
    buffer<bf16> get_logits(buffer<bf16>& x);

    int get_current_context_length();

    void _set_rope_rms_weights(int idx);

    inline bool is_global_layer_idx(int layer_idx) {return (layer_idx % global_layer_period) == (global_layer_period - 1); }
    inline void _process_embedding(int idx){
        buffer<bf16> embedding_output = this->embedding->forward(idx);
        DEBUG_BLOCK(2,
            header_print("debug", "Embedding output for token idx " + std::to_string(idx));
            utils::print_matrix(embedding_output, D);
        )
        memcpy(this->x.data(), embedding_output.data(), D * sizeof(bf16));
        memcpy(this->x.data() + D * 2, embedding_output.data(), D * sizeof(bf16)); // copy to the second half for swa
        this->x.sync_to_device();
        buffer<bf16> pli_embedding_output = this->pli_embedding->forward(idx);

        DEBUG_BLOCK(2,
            header_print("debug", "PLI Embedding output for token idx " + std::to_string(idx));
            utils::print_matrix(pli_embedding_output, PLI_D);
        )

        bf16* p_pli = pli_embedding_output.data();
        for (int i = 0; i < num_hidden_layers; i++) {
            memcpy(this->rope_rms_weights[i].data() + desc.get_pli_embed_offset(layer_types[i]), p_pli, PLI_D * sizeof(bf16));
            p_pli += PLI_D;
            this->rope_rms_weights[i].sync_to_device();
        }
    }

};
