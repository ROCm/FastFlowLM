// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#pragma once
#include "models/gemma4_12b/gemma4_12b_npu.hpp"
#include "gemma4_12b_npu_sequence.hpp"
#include "gemma4_12b_npu_def.hpp"
#include "gemma4_12b_prefill.hpp"
#include "gemma4_12b_mm_projector.hpp"
#include "avx512_util.hpp"
#include "utils/kv_rows_state.hpp"
#include "embedding_bf16_mapped.hpp"

struct gemma4_12b_npu::Impl{
    // model specific token ids
    static constexpr int boi_token_id = 255999; // begin of image token id
    static constexpr int image_token_id = 258880; // image token id
    static constexpr int eoi_token_id = 258882; // end of image token id

    static constexpr int boa_token_id = 256000; // begin of audio token id
    static constexpr int audio_token_id = 258881; // audio token id
    static constexpr int eoa_token_id = 258883; // end of audio token id

    static constexpr int video_token_id = 258884; // video soft token id

    static constexpr int max_prefill_thread = 4;

    int vocab_size;
    int vocab_size_padded;
    bool is_preload_launched;

    LM_Config config;
    npu_xclbin_manager *npu;

    npu_app_manager* layer_app_manager;
    npu_app_manager* lm_head_app_manager;

    npu_app global_layer;
    npu_app swa_layer;
    npu_app layer_pre_load; //!< empty sequence, used to keep the layer xclbin resident
    //!< one tiny app per layer, writes that layer's output scale RTP before the layer runs
    std::vector<npu_app> layer_scale_apps;
    flm_rt::run pre_load_run;

    //!< the batched (matmul) prefill graph: dequant_mm.xclbin, one attention/mlp block
    //!< pair per layer kind and the activation scratch they share. Built by
    //!< _init_prefill_mm() on the first prefill, see gemma4_12b_prefill.hpp.
    std::unique_ptr<gemma4_12b_prefill_mm_context> prefill_mm;

    //!< the image/audio front end: image_audio_weights.q4nx and the two host (or, later,
    //!< array) towers that turn raw patches and raw PCM frames into language embeddings.
    //!< Built by _init_mm_projector() on the first prompt that actually carries media, so
    //!< a text only session never reads the ~105 MB file. See gemma4_12b_mm_projector.hpp.
    std::unique_ptr<gemma4_12b_mm_projector> mm_projector;

    npu_app lm_head;

    std::unique_ptr<gemma4_12b_npu_sequence> sequence;

    gemma4_12b_npu* parent_ptr;

    /// @brief model description: dimensions + every weight descriptor, built from config.json
    gemma4_12b_desc desc;

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
    int SLIDING_LENGTH;

    int num_hidden_layers;
    int num_kv_shared_layers;
    int global_layer_period;
    float final_logit_softcapping;

    std::vector<gemma4_12b_buffer_offsets_t> proj_offsets;
    std::vector<gemma4_12b_layer_type_t> layer_types;

    // size
    std::vector<buffer<bf16>> rms_weights;
    std::vector<buffer<bf16>> rope_rms_weights;
    std::vector<buffer<u8>> proj_weights;
    std::vector<buffer<bf16>> kv_caches;
    std::vector<buffer<bf16>> kv_checkpoint; //!< host copy of the sliding kv caches

    buffer<bf16> layer_scale; //!< per layer output scale, one bf16 per layer

    buffer<bf16> x;
    /// @note model.embed_tokens.weight is stored as bf16 in the checkpoint (there is no
    ///       companion .scale tensor), so the plain bf16 embedding is used here.
    std::unique_ptr<EmbeddingBF16Mapped> embedding;

    buffer<bf16> logits;
    buffer<bf16> logits_valid;
    buffer<u8> lm_head_weights;

    flm_rt::runlist layers_run;
    flm_rt::run lm_head_run;
    int current_context_length;

    int checkpoint_context_length;

    Impl(LM_Config config, npu_xclbin_manager *npu_instance, gemma4_12b_npu* parent_ptr, int MAX_L = 4096);
    ~Impl(); // waits for any in-flight preload run before tearing down

    /// @brief Block until the keep-resident preload run started by the previous token
    ///        has retired, so its xrt::run handle can be reassigned or destroyed.
    /// @note  Every site that re-arms `pre_load_run`, and the destructor, must call this
    ///        first: overwriting a handle whose run is still executing tears down the
    ///        command buffer the device is reading.
    void _drain_preload();

    /// @brief decode one token
    buffer<bf16> forward(int ids);
    buffer<bf16> prefill(std::vector<int>& ids, void* payload = nullptr);
    buffer<bf16> _prefill_with_mv(std::vector<int>& ids, void* payload = nullptr);
#ifndef MVPREFILL
    buffer<bf16> _prefill_with_mm(std::vector<int>& ids, void* payload = nullptr);
    /// @brief Build `prefill_mm`, which registers dequant_mm.xclbin, generates the
    ///        projection control sequences and allocates the mm scratch buffers.
    /// @note  Called from the first _prefill_with_mm() rather than from the constructor,
    ///        so a decode only session never loads the second xclbin.
    void _init_prefill_mm();
#endif

    /// @brief Build `mm_projector` and read its weights, once.
    /// @note  Called from the first prefill that has a multimodal payload rather than from
    ///        the constructor, so a text only session pays neither the file read nor (on
    ///        an array backend) a second xclbin.
    void _init_mm_projector();

    /// @brief Project the payload's media and overwrite the soft token rows of `residual`.
    /// @param tokens    the prompt ids, `L_in` of them
    /// @param residual  [L_in][D] the embedding rows, already filled from the token table
    /// @param payload   gemma4_12b_multi_modal_payload_t*, or null for a text only prompt
    /// @note  The rows written here must NOT be scaled by sqrt(hidden_size): that scale
    ///        belongs to the token embedding table alone, and Embedding::forward() has
    ///        already applied it to the text rows this overwrites.
    void _embed_media(const int* tokens, int L_in, buffer<bf16>& residual, void* payload);

    void set_context_length(int L);
    void load_weights(Q4NX& q4nx);
    void update_max_length(uint32_t MAX_L);
    void clear_context();

    bool is_checkpoint_valid;
    void _allocate_checkpoint_buffers();

    kv_row_layout kv_layout() const;
    std::unique_ptr<hybrid_engine_state> new_state();
    int read_state(hybrid_engine_state& state, int kv_from);
    int write_state(const hybrid_engine_state& state);
    int write_state_kv(const hybrid_engine_state& state);
    /// @brief one flag per layer: global attention (keeps K/V rows) or sliding window (keeps a ring)
    std::vector<bool> attn_mask;
    /// @brief highest row (plus the prefill's padding) reached since the global rows past the length were last zero
    size_t dirty_rows = 0;
    int checkpoint();
    int restore();

    /// @brief Row index of absolute position `idx` in `layer_idx`'s cache: the position
    ///        itself on a global layer, the ring slot on a sliding one.
    size_t _kv_cache_row(int layer_idx, int idx) const;
    buffer<bf16> get_k_cache(int layer_idx, int idx);
    buffer<bf16> get_v_cache(int layer_idx, int idx);

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
    }
};
