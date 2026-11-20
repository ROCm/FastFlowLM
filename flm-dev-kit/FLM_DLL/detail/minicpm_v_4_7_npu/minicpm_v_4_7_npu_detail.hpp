// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#pragma once
#include "models/minicpm_v_4_7/minicpm_v_4_7_npu.hpp"
#include "minicpm_v_4_7_npu_sequence.hpp"
#include "image/image_module.hpp"
#include "image/mrope.hpp"
#include "image/vision_model_def.hpp"
#include "modules/gemm.hpp"
#include "minicpm_v_model_def.hpp"
#include "minicpm_v_prefill.hpp"
#include "utils/avx512_util.hpp"
#include "embedding_q8_0.hpp"
#include "minicpm_v_decode_wait.hpp"
#include "embedding_bf16_mapped.hpp"

struct minicpm_v_4_7_npu::Impl{
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
    minicpm_v_desc desc;
    std::unique_ptr<minicpm_v_prefill_context> pctx;
    uint32_t MAX_L;
    std::vector<buffer<bf16>> cache_checkpoint;

    npu_app_manager* layer_app_manager;
    npu_app_manager* lm_head_app_manager = nullptr;

    npu_app layer;
    npu_app first_linear_layer;
    npu_app linear_layer;
    npu_app set_layer_rtp;
    npu_app lm_head;

    int vocab_size;
    int vocab_size_padded;
    uint32_t V_offset;

    std::unique_ptr<minicpm_v_4_7_npu_sequence> sequence;

    /// @brief the vision tower (image/). The desc must outlive the tower, which
    ///        holds it by reference -- hence declared first.
    minicpm_v_vision_desc vision_desc;
    std::unique_ptr<mcv::image_module> image_mod;
    mcv::canvas_token_ids canvas_ids;
    static constexpr int video_token_id = 248057;

    minicpm_v_4_7_npu* parent_ptr;

    // size
    std::vector<buffer<bf16>> rms_weights;
    std::vector<buffer<bf16>> rope_rms_weights;
    std::vector<buffer<u8>> proj_weights;
    std::vector<buffer<bf16>> kv_caches;
    std::vector<buffer<bf16>> alpha_proj_cpu;
    std::vector<buffer<bf16>> beta_proj_cpu;

    buffer<bf16> x;
    /// @brief bf16 token embedding table, used when the checkpoint carries
    ///        model.embed_tokens.weight (mapped from model.q4nx, copied if it cannot be
    ///        mapped). Exactly one of these two is non-null.
    std::unique_ptr<EmbeddingBF16Mapped> embedding;
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

    /// @brief batch the page-ins of a prompt's embedding rows (no-op unless the table is mapped)
    void embed_prefetch(const int* ids, size_t n){
        if (this->embedding) this->embedding->prefetch(ids, n);
    }

    buffer<bf16> logits;
    buffer<bf16> logits_valid;
    buffer<u8> lm_head_weights;
    /// @brief FLM_MINICPM_LM_IN_LAYER (default on when the model folder ships layer_lm.xclbin; =0 forces the two-xclbin path): the lm_head is a run
    ///        of the layer design (mode 2 of layer_lm.xclbin), so a decode token needs one xclbin / hw context and no reconfiguration. The weights are
    ///        then in the projection engine's layout (reorder_cpy_lm_head_in_layer) and the final norm weight sits in lm_head_norm.
    bool lm_in_layer = false;
    bool lm_stateless = false;   //!< layer_lm_sl.xclbin
    buffer<bf16> lm_head_norm;
    /// @brief the one-token hidden state of a host decode (MINICPM_V_DECODE_ON_NPU == 0)
    buffer<bf16> decode_hidden;

    flm_rt::runlist layers_run;
    minicpm_wait::Waiter wait_layers, wait_lm;
#if MINICPM_V_DECODE_PROFILE
    std::vector<flm_rt::run> prof_runs;
#endif
    flm_rt::run lm_head_run;
#ifdef FLM_MINICPM_FAULT_RELOAD
    std::unique_ptr<npu_app> fault_dummy;   //!< an empty sequence on conv.xclbin: dispatching it makes the firmware reload the layer design
    void fault_layers();                    //!< one token's layer runs, one by one, with the reloads FLM_MINICPM_FAULT_RELOAD asks for
    void fault_before_lm_head();
#endif
#if MINICPM_V_DECODE_ON_NPU && FLM_MINICPM_RECONFIG_SUPPORTED
    // FLM_MINICPM_DEC_SINGLE_CTX=1 (opt-in, default off): a decode token stays in layer.xclbin's hw_context. The array is reconfigured to the lm_head design
    // and back from inside the instruction stream (the quiesced CDO replay of minicpm_v_reconfig.hpp) instead of two hw-context switches (2 x ~2.4 ms).
    //   round 1 (one runlist): rtp, the 24 layer runs, reconfigure to lm_head
    //   round 2 (one runlist): the lm_head run, reconfigure back to the layer design; set_context_length for the next token overlaps it
    // Three gates, each falling back to the ordinary two-context decode: built (CDOs parse) -> validated (known-answer test at weight load, longer
    // than the 70 tokens at which the unwaited GDN state tokens used to hang the array) -> usable per token (not disabled, this process alone on the NPU).
    using dec_chain_t = minicpm_reconfig::Chain<npu_app, xrt::run>;
    dec_chain_t dec_to_lm, dec_to_layer;
    npu_app lm_head_sc;                     //!< the lm_head sequence as an app of the layer context
    xrt::runlist dec_rl1, dec_rl2;          //!< persistent: reset() and refilled per token
    bool dec_rl_ready = false, dec_sc_built = false, dec_sc_validated = false, dec_sc_testing = false, dec_force_two_ctx = false;
    long dec_sc_tokens = 0;
    static bool dec_single_ctx_wanted() { static const bool v = [] { const char* e = std::getenv("FLM_MINICPM_DEC_SINGLE_CTX"); return e && std::atoi(e) != 0; }(); return v; }
    bool dec_sc_usable() {
        return dec_sc_built && (dec_sc_validated || dec_sc_testing) && !dec_force_two_ctx && !minicpm_reconfig::disabled() && minicpm_reconfig::npu_exclusive();
    }
    buffer<bf16> forward_single_ctx();      //!< a decode token after the embedding, single-context
    void dec_sc_selftest();                 //!< known-answer test against the two-context path (load time)
    void dec_sc_recover();                  //!< force the firmware to reload the layer PDI after a failed round
#endif
    int current_context_length;
    /// @brief K/V rows (per attention layer) that any request since the last clear_context() may have written
    int kv_hwm_rows = 0;
    int general_token_idx;

    /// @brief  initialize the minicpm_v_4_7_npu
    /// @param config
    /// @param npu_instance
    Impl(LM_Config config, npu_xclbin_manager *npu_instance, minicpm_v_4_7_npu* parent_ptr, int MAX_L = 4096);
    ~Impl();

    /// @brief forward the minicpm_v_4_7_npu
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
    /// @brief (re)create the persistent lm_head run from the lm_head app and the engine's buffers
    void make_lm_head_run();
    /// @brief after a request failed with an exception: a persistent run that was started but never waited for (an exception between start() and wait())
    ///        or ended in an error state cannot be started again ("bad command state, can't launch"), so every later request would fail until the server
    ///        restarted. Waits for an in-flight lm_head run, parks it (never destroys it: XRT aborts the process for a run destroyed while in flight), creates
    ///        a new one, resets the persistent runlists and releases the single-context chains. Never throws.
    void npu_recover(const char* where);
    int npu_recoveries = 0;
#ifdef FLM_MINICPM_FAULT_RUNSTATE
    /// test builds only: FLM_MINICPM_FAULT_RUNSTATE=p:K (K-th prefill-end logits run) or d:K (K-th decode lm_head round) throws between start() and wait() of the lm_head run, which leaves the persistent run object started but never waited for
    int fault_calls_p = 0, fault_calls_d = 0;
    void fault_runstate(const char* kind){
        const char* e = std::getenv("FLM_MINICPM_FAULT_RUNSTATE");
        if (!e || e[0] != kind[0] || e[1] != ':') return;
        int& calls = (kind[0] == 'p') ? fault_calls_p : fault_calls_d;
        if (++calls != std::atoi(e + 2)) return;
        throw std::runtime_error("injected fault: an exception while the lm_head run was in flight");
    }
#endif

    int get_current_context_length();

    /// @brief build the vision tower on the array, or leave the engine text-only
    void _load_vision();
    /// @brief encode the payload's views into [rows][D] bf16 rows for the prompt's
    ///        image_token_id positions; fills `views` with the payload's grids
    std::vector<bf16> _encode_images(const minicpm_v_image_payload_t* p, int n_image_tokens,
                                     std::vector<mcv::view_grid>& views);
    void _set_rope_rms_weights(int idx);
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
