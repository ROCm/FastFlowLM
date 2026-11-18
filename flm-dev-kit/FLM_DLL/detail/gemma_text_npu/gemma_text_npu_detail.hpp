// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#pragma once
#include "models/gemma_text/gemma_text_npu.hpp"
#include "modules/mha.hpp"
#include "utils/kv_rows_state.hpp"
typedef struct {
    bf16 scales[4][16];
    bf16 mins[4][16];
    uint8_t qs[128][8];
} qblock;

struct gemma_text_npu::Impl{
    /// @brief constexprs
    static constexpr f32 inv_freq[128] = {
        1.0000e+00, 8.9769e-01, 8.0584e-01, 7.2339e-01, 6.4938e-01, 5.8294e-01, 5.2330e-01, 4.6976e-01, 4.2170e-01, 3.7855e-01, 3.3982e-01, 3.0505e-01, 2.7384e-01, 2.4582e-01, 2.2067e-01, 1.9810e-01,
        1.7783e-01, 1.5963e-01, 1.4330e-01, 1.2864e-01, 1.1548e-01, 1.0366e-01, 9.3057e-02, 8.3536e-02, 7.4989e-02, 6.7317e-02, 6.0430e-02, 5.4247e-02, 4.8697e-02, 4.3714e-02, 3.9242e-02, 3.5227e-02,
        3.1623e-02, 2.8387e-02, 2.5483e-02, 2.2876e-02, 2.0535e-02, 1.8434e-02, 1.6548e-02, 1.4855e-02, 1.3335e-02, 1.1971e-02, 1.0746e-02, 9.6466e-03, 8.6596e-03, 7.7737e-03, 6.9783e-03, 6.2643e-03,
        5.6234e-03, 5.0481e-03, 4.5316e-03, 4.0679e-03, 3.6517e-03, 3.2781e-03, 2.9427e-03, 2.6416e-03, 2.3714e-03, 2.1288e-03, 1.9110e-03, 1.7154e-03, 1.5399e-03, 1.3824e-03, 1.2409e-03, 1.1140e-03,
        1.0000e-03, 8.9769e-04, 8.0584e-04, 7.2339e-04, 6.4938e-04, 5.8294e-04, 5.2330e-04, 4.6976e-04, 4.2170e-04, 3.7855e-04, 3.3982e-04, 3.0505e-04, 2.7384e-04, 2.4582e-04, 2.2067e-04, 1.9810e-04,
        1.7783e-04, 1.5963e-04, 1.4330e-04, 1.2864e-04, 1.1548e-04, 1.0366e-04, 9.3057e-05, 8.3536e-05, 7.4989e-05, 6.7317e-05, 6.0430e-05, 5.4247e-05, 4.8697e-05, 4.3714e-05, 3.9242e-05, 3.5227e-05,
        3.1623e-05, 2.8387e-05, 2.5483e-05, 2.2876e-05, 2.0535e-05, 1.8434e-05, 1.6548e-05, 1.4855e-05, 1.3335e-05, 1.1971e-05, 1.0746e-05, 9.6466e-06, 8.6596e-06, 7.7737e-06, 6.9783e-06, 6.2643e-06,
        5.6234e-06, 5.0481e-06, 4.5316e-06, 4.0679e-06, 3.6517e-06, 3.2781e-06, 2.9427e-06, 2.6416e-06, 2.3714e-06, 2.1288e-06, 1.9110e-06, 1.7154e-06, 1.5399e-06, 1.3824e-06, 1.2409e-06, 1.1140e-06
    };

    static constexpr f32 inv_freq_local[128] = {
        1.0000e+00, 9.3057e-01, 8.6596e-01, 8.0584e-01, 7.4989e-01, 6.9783e-01, 6.4938e-01, 6.0430e-01, 5.6234e-01, 5.2330e-01, 4.8697e-01, 4.5316e-01, 4.2170e-01, 3.9242e-01, 3.6517e-01, 3.3982e-01,
        3.1623e-01, 2.9427e-01, 2.7384e-01, 2.5483e-01, 2.3714e-01, 2.2067e-01, 2.0535e-01, 1.9110e-01, 1.7783e-01, 1.6548e-01, 1.5399e-01, 1.4330e-01, 1.3335e-01, 1.2409e-01, 1.1548e-01, 1.0746e-01,
        1.0000e-01, 9.3057e-02, 8.6596e-02, 8.0584e-02, 7.4989e-02, 6.9783e-02, 6.4938e-02, 6.0430e-02, 5.6234e-02, 5.2330e-02, 4.8697e-02, 4.5316e-02, 4.2170e-02, 3.9242e-02, 3.6517e-02, 3.3982e-02,
        3.1623e-02, 2.9427e-02, 2.7384e-02, 2.5483e-02, 2.3714e-02, 2.2067e-02, 2.0535e-02, 1.9110e-02, 1.7783e-02, 1.6548e-02, 1.5399e-02, 1.4330e-02, 1.3335e-02, 1.2409e-02, 1.1548e-02, 1.0746e-02,
        1.0000e-02, 9.3057e-03, 8.6596e-03, 8.0584e-03, 7.4989e-03, 6.9783e-03, 6.4938e-03, 6.0430e-03, 5.6234e-03, 5.2330e-03, 4.8697e-03, 4.5316e-03, 4.2170e-03, 3.9242e-03, 3.6517e-03, 3.3982e-03,
        3.1623e-03, 2.9427e-03, 2.7384e-03, 2.5483e-03, 2.3714e-03, 2.2067e-03, 2.0535e-03, 1.9110e-03, 1.7783e-03, 1.6548e-03, 1.5399e-03, 1.4330e-03, 1.3335e-03, 1.2409e-03, 1.1548e-03, 1.0746e-03,
        1.0000e-03, 9.3057e-04, 8.6596e-04, 8.0584e-04, 7.4989e-04, 6.9783e-04, 6.4938e-04, 6.0430e-04, 5.6234e-04, 5.2330e-04, 4.8697e-04, 4.5316e-04, 4.2170e-04, 3.9242e-04, 3.6517e-04, 3.3982e-04,
        3.1623e-04, 2.9427e-04, 2.7384e-04, 2.5483e-04, 2.3714e-04, 2.2067e-04, 2.0535e-04, 1.9110e-04, 1.7783e-04, 1.6548e-04, 1.5399e-04, 1.4330e-04, 1.3335e-04, 1.2409e-04, 1.1548e-04, 1.0746e-04
    };

    static constexpr int LC = 16;
    /// @brief  initialize the gemma_npu
    /// @param config
    /// @param npu_instance
    Impl(LM_Config config, npu_xclbin_manager *npu_instance, int MAX_L = 4096);
    ~Impl();

    /// @brief forward the gemma_npu
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
    /// @brief one flag per layer: global attention (keeps K/V rows) or sliding window (keeps a ring)
    std::vector<bool> attn_mask;
    /// @brief highest context length reached since the global rows past the length were last zero (see kv_rows_zero)
    size_t dirty_rows = 0;
    /// @brief sliding_temp_cache holds what an earlier prefill left there
    bool temp_dirty = false;
    void _zero_temp_cache();
    bool is_checkpoint_valid;
    int checkpoint_context_length;
    std::vector<buffer<bf16>> kv_checkpoint;

    size_t K_offset;
    size_t V_offset;
    size_t sliding_K_offset;
    size_t sliding_V_offset;

    LM_Config config;
    npu_xclbin_manager *npu;

    std::unique_ptr<GemmaTextLMHead> lm_head;
    void _set_rope_rms_weights(int idx);

    inline void _rms_norm(buffer<bf16>& y, buffer<bf16>& x, buffer<bf16>& rms_weight);
    inline void _rope_rms(buffer<bf16>& x, int idx, buffer<bf16>& rms_weight, int layer_idx);
    inline void _residual_add(buffer<bf16>& dest, buffer<bf16>& src, buffer<bf16>& residual);
    inline __m256 _mm256_exp_ps(__m256 x);
    inline void _gelu(buffer<bf16>& dest, buffer<bf16>& up_gate, int L_offset, int L_effective);
    void fill_kv_cache(buffer<bf16>& buffer_k, buffer<bf16>& buffer_v, int L_offset, int L_begin, int L_effective, int layer_idx);
    void fill_kv_cache_sliding(buffer<bf16>& local_kv_cache, buffer<bf16>& buffer_k, buffer<bf16>& buffer_v, int L_offset, int L_begin, int L_end, int L_effective, int sliding_l_begin, int slidding_l_end, int layer_idx);
    inline bool is_sliding_window(int layer_idx) {return (layer_idx + 1) % SLIDING_WINDOW_PATTERN != 0;}
    inline int linear2ring(int l) {return l % SLIDING_WINDOW;}
    void check_consistency(buffer<bf16>& x, std::string file_name);
    float total_error;

    npu_app_manager* layer_app_manager;
    npu_app_manager* gemm_app_manager;
    npu_app_manager* dequant_app_manager;
    npu_app_manager* mha_app_manager;

    npu_app layer;
    npu_app sliding_layer;
    npu_app dequant_0;
    npu_app dequant_1;
    npu_app dequant_2;
    npu_app dequant_3;
    npu_app q_proj;
    npu_app k_proj;
    npu_app v_proj;
    npu_app o_proj;
    npu_app up_gate_proj;
    npu_app down_proj;
    npu_app mha_engine;
    npu_app sliding_mha_engine;

    std::unique_ptr<gemma_text_npu_sequence> sequence;
    std::unique_ptr<GemmaTextGemm> gemm;
    std::unique_ptr<MHA> mha_sequence_gen;
    std::unique_ptr<GemmaTextDequant> dequant;
    uint32_t D;
    uint32_t DH;
    uint32_t DQ;
    uint32_t DK;
    uint32_t DV;
    uint32_t MAX_L;
    uint32_t HIDDEN_SIZE;
    uint32_t INTERMEDIATE_SIZE;
    uint32_t num_hidden_layers;
    int SLIDING_WINDOW;
    int SLIDING_WINDOW_PATTERN;
    // size
    std::vector<buffer<bf16>> rms_weights;
    std::vector<buffer<bf16>> rope_rms_weights;
    std::vector<buffer<u8>> proj_weights;
    std::vector<buffer<bf16>> kv_caches;

    buffer<bf16> weights_p0; // QKV
    buffer<bf16> weights_p1; // O
    buffer<bf16> weights_p2; // UP, Gate
    buffer<bf16> weights_p3; // Down

    buffer<bf16> sliding_temp_cache;

    buffer<bf16> x;
    buffer<bf16> x_lm_head_exposed;
    std::unique_ptr<Embedding> embedding;

    buffer<bf16> final_rms_weight;
    
    flm_rt::runlist layers_run;
    flm_rt::runlist dequant_all;
    int current_context_length;

    buffer<bf16> logits_temp;
};
