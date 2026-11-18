// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#pragma once
#include "models/gemma4e_dflash/gemma4e_dflash.hpp"
#include "gemma4e_dflash_sequence.hpp"
#include "gemma4e_dflash_swa.hpp"
#include "gemma4e_image.hpp"
#include "modules/gemm.hpp"
#include "reorder_cpy.hpp"
#include "avx512_util.hpp"
#include "gemma4e_audio.hpp"
#include "embedding_q8_0.hpp"
#include "gemma4e_prefill.hpp"
#include "utils/kv_rows_state.hpp"

struct gemma4e_dflash::Impl{
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

    std::unique_ptr<gemma4e_dflash_sequence> sequence;
    std::unique_ptr<Gemma4e_ImageEncoder> gemma4e_image_encoder;
    std::unique_ptr<Gemma4e_AudioEncoder> gemma4e_audio_encoder;

    gemma4e_dflash* parent_ptr;

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
    Impl(LM_Config config, npu_xclbin_manager *npu_instance, gemma4e_dflash* parent_ptr, int MAX_L = 4096);
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

    kv_row_layout kv_layout() const;
    std::unique_ptr<hybrid_engine_state> new_state();
    int read_state(hybrid_engine_state& state, int kv_from);
    int write_state(const hybrid_engine_state& state);
    int write_state_kv(const hybrid_engine_state& state);
    /// @brief one flag per kv cache (non-skip layer): global attention (keeps K/V rows) or sliding window (keeps a ring)
    std::vector<bool> attn_mask;
    /// @brief highest context length reached since the global rows past the length were last zero (see kv_rows_zero)
    size_t dirty_rows = 0;
    // ---- sliding-window K/V buffers: linear window with host compaction (gemma4e_dflash_swa.hpp) ----
    /// @brief position of row 0 of every sliding layer buffer
    int swa_base = 0;
    /// @brief elements of one sliding layer buffer (K plane | V plane, kSwaBufferRows rows each)
    size_t swa_buffer_elems() const { return (size_t)kSwaBufferRows * (SWA_DK + SWA_DV); }
    /// @brief elements of a saved sliding window (K rows | V rows, SLIDING_LENGTH rows each, oldest row first)
    size_t swa_window_elems() const { return (size_t)SLIDING_LENGTH * (SWA_DK + SWA_DV); }
    /// @brief rows a sliding layer holds at context length L, and the first position among them
    int swa_window_rows(int L) const { return std::min<int>(L, SLIDING_LENGTH); }
    /// @brief move the last window of every sliding buffer to the front and advance `base` (between two steps only)
    void swa_compact(int L);
    /// @brief debug (FLM_DFLASH_CHECK_SWA=1): report non-zero rows behind the context in the sliding buffers and the global caches
    void swa_check_zero(const char* where, int ctx);
    /// @brief zero rows [from_row, to_row) of both planes of sliding buffer `layer`
    void swa_zero_rows(int layer, int from_row, int to_row);
    /// @brief copy the window rows [L - n, L) of buffer `layer` into `dst` (swa_window_elems(), oldest first, rest zero)
    void swa_extract(int layer, int L, bf16* dst);
    /// @brief lay a saved window (context length L) into buffer `layer` at base `new_base`; rows past it are zeroed up to `zero_hi`
    void swa_relay(int layer, int L, const bf16* src, int new_base, int zero_hi);
    /// @brief put a window saved at length L back into every sliding buffer, base = swa_base_after_compaction(L)
    void swa_restore_all(int L, const std::vector<buffer<bf16>>& windows);
    int checkpoint();
    int restore();

    buffer<bf16> get_k_cache(int layer_idx, int idx);
    buffer<bf16> get_v_cache(int layer_idx, int idx);
    buffer<bf16> get_logits(buffer<bf16>& x);

    int get_current_context_length();

    void _set_rope_rms_weights(int idx);

    // ---- DFlash speculative decoding (gemma4e_dflash_spec.cpp). All state lives in `spec` so this struct stays small. ----
    struct Spec;
    Spec* spec = nullptr;
    void spec_create();    // defined next to Spec in gemma4e_dflash_spec.cpp; called from the constructor
    // ---- fused 8-row verify (gemma4e_dflash_verify.cpp) ----
    /// \brief register verify.xclbin + lm_head_8.xclbin from the xclbin folder and allocate the per-layer verify BOs; false = stay sequential
    /// \brief set when the decode instruction streams / runlists no longer match current_context_length (a fused verify commit moves the context
    ///        without a plain step following): forward() regenerates them lazily
    bool decode_seq_stale = false;
    void verify_destroy();
    /// \brief prepare the next fused verify for context length ctx0 (SWA compaction, instruction streams, the 42-run chain); runs concurrently with the host draft
    void verify_prepare(int ctx0);
    void verify_prepare_async(int ctx0);
    bool verify_load();
    /// \brief the sequential verify (n+1 ordinary forward passes), also the fallback of the fused one
    const bf16* verify_seq_impl(int last_token, const int* drafts, int n_drafts);
    bool verify_fused_ready() const;
    /// \brief one fused verify: [last_token, drafts..., padding] -> 8 logits rows, taps, K/V rows ctx..ctx+7; nullptr = declined
    const bf16* verify_fused(int last_token, const int* drafts, int n_drafts);
    /// \brief draft head: 7 pre-norm hidden rows -> 7 token ids through ONE lm_head_8 dispatch (norm weight = draft output_norm)
    bool verify_draft_head(const bf16* hidden, const bf16* norm_w, int* ids_out, int* topk_out = nullptr, int K = 0);
    void verify_gen_seq(npu_sequence* seq, gemma4e_layer_type_t type, int ctx0);
    void spec_destroy();   // called from ~Impl
    // hooks of the plain paths (no-ops until a draft is loaded)
    bool spec_tap_mode() const;                       // true: the 42 layer runs are split into 6 segments so taps can be read
    int  spec_tap_segment(int layer) const;
    int  spec_tap_index(int layer) const;             // 0..4 for layers 5,11,17,23,35, else -1
    void spec_capture_tap(int t);
    void spec_forward_begin(int id);
    void spec_forward_end();
    void spec_prefill_begin(const std::vector<int>& ids, int L);
    void spec_prefill_tap(int layer, int L);
    int spec_last_tap_layer() const;    // last layer whose hidden state the draft reads
    bool spec_taps_wanted() const;      // a draft is loaded: prefill must hand over the tap rows of every layer
    void spec_prefill_end(int L);
    void spec_cold(int length);
    void spec_join_inject();            // wait for the background draft-context injection (no-op if none runs)
    void spec_async_inject(int ctx0, int rows);   // start it for the verify rows [ctx0, ctx0+rows)
    void spec_sync_draft_ctx(int ctx);
    int  spec_lm_head_argmax(const bf16* hidden, const bf16* norm_w);
    std::vector<flm_rt::runlist> seg_runs;            // tap mode: runs of layers [0..5] [6..11] [12..17] [18..23] [24..35] [36..41]
    bool load_draft(const std::string& draft_dir);
    bool draft_ready() const;
    void set_stop_tokens(const std::vector<int>& ids);
    int dflash_draft(int last_token, int* drafts);
    const bf16* dflash_verify(int last_token, const int* drafts, int n_drafts);
    const bf16* dflash_verify_impl(int last_token, const int* drafts, int n_drafts);
    void dflash_commit(int n_keep);
    int spec_vocab_size() const;
    uint64_t last_speculation_prime_us() const;
    const bf16* last_speculation_logits(int& rows, int& vocab) const;
    bool supports_speculation() const;
    std::vector<int> speculate(int last_token, int max_draft);
    double mean_accepted_length() const;
    double draft_hit_rate() const;
    int speculation_cycles() const;
    std::string speculation_stats() const;
    std::string speculation_timing() const;
    void reset_speculation_stats();
    /// @brief Runs the layers that share a kv cache for one token with the decode kernels, in place on `x`.
    /// @param token_id  the token whose hidden state `hidden` is (it also supplies the per-layer inputs)
    /// @param hidden    the token's hidden state after the last layer that owns a cache
    /// @param L_end     context length including this token; the token sits at position L_end - 1
    /// @param first_layer  first layer to run (all layers from here on share a kv cache)
    void _decode_shared_layers(int token_id, buffer<bf16>& hidden, int L_end, uint32_t first_layer);

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
            // only this layer's slice of the per-layer inputs changes from token to token: sync just those bytes
            const size_t pli_offset = desc.get_pli_embed_offset(layer_types[i]);
            memcpy(this->rope_rms_weights[i].data() + pli_offset, p_pli, PLI_D * sizeof(bf16));
            p_pli += PLI_D;
            this->rope_rms_weights[i].sync_range_to_device(pli_offset * sizeof(bf16), PLI_D * sizeof(bf16));
        }
    }

};
