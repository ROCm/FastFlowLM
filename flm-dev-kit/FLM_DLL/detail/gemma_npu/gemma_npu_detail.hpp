// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#pragma once
#include "models/gemma/gemma_npu.hpp"
#include "modules/gemm.hpp"
#include "modules/dequant.hpp"
#include "modules/mha.hpp"
#include "gamma_image.hpp"
#include "utils/kv_rows_state.hpp"
#include "gemma_bfp16_pack.hpp"

struct gemma_npu::Impl{
    /// @brief constexprs
    static constexpr f32 inv_freq[128] = {
        1.2500e-01, 1.1221e-01, 1.0073e-01, 9.0424e-02, 8.1173e-02, 7.2868e-02, 6.5412e-02, 5.8720e-02, 5.2712e-02, 4.7319e-02, 4.2478e-02, 3.8132e-02, 3.4230e-02, 3.0728e-02, 2.7584e-02, 2.4762e-02,
        2.2228e-02, 1.9954e-02, 1.7913e-02, 1.6080e-02, 1.4435e-02, 1.2958e-02, 1.1632e-02, 1.0442e-02, 9.3737e-03, 8.4146e-03, 7.5537e-03, 6.7809e-03, 6.0871e-03, 5.4643e-03, 4.9052e-03, 4.4034e-03,
        3.9528e-03, 3.5484e-03, 3.1854e-03, 2.8595e-03, 2.5669e-03, 2.3043e-03, 2.0685e-03, 1.8569e-03, 1.6669e-03, 1.4964e-03, 1.3433e-03, 1.2058e-03, 1.0825e-03, 9.7171e-04, 8.7229e-04, 7.8304e-04,
        7.0293e-04, 6.3101e-04, 5.6645e-04, 5.0849e-04, 4.5647e-04, 4.0977e-04, 3.6784e-04, 3.3021e-04, 2.9642e-04, 2.6609e-04, 2.3887e-04, 2.1443e-04, 1.9249e-04, 1.7280e-04, 1.5512e-04, 1.3925e-04,
        1.2500e-04, 1.1221e-04, 1.0073e-04, 9.0424e-05, 8.1173e-05, 7.2868e-05, 6.5412e-05, 5.8720e-05, 5.2712e-05, 4.7319e-05, 4.2478e-05, 3.8132e-05, 3.4230e-05, 3.0728e-05, 2.7584e-05, 2.4762e-05,
        2.2228e-05, 1.9954e-05, 1.7913e-05, 1.6080e-05, 1.4435e-05, 1.2958e-05, 1.1632e-05, 1.0442e-05, 9.3737e-06, 8.4146e-06, 7.5537e-06, 6.7809e-06, 6.0871e-06, 5.4643e-06, 4.9052e-06, 4.4034e-06,
        3.9528e-06, 3.5484e-06, 3.1854e-06, 2.8595e-06, 2.5669e-06, 2.3043e-06, 2.0685e-06, 1.8569e-06, 1.6669e-06, 1.4964e-06, 1.3433e-06, 1.2058e-06, 1.0825e-06, 9.7171e-07, 8.7229e-07, 7.8304e-07,
        7.0293e-07, 6.3101e-07, 5.6645e-07, 5.0849e-07, 4.5647e-07, 4.0977e-07, 3.6784e-07, 3.3021e-07, 2.9642e-07, 2.6609e-07, 2.3887e-07, 2.1443e-07, 1.9249e-07, 1.7280e-07, 1.5512e-07, 1.3925e-07
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

    static constexpr int img_soft_token_id = 262144;

    /// @brief  initialize the gemma_npu
    /// @param config
    /// @param npu_instance
    Impl(LM_Config config, npu_xclbin_manager *npu_instance, int MAX_L = 4096);
    ~Impl();

    /// @brief forward the gemma_npu
    buffer<bf16> forward(int ids);
    buffer<bf16> prefill(std::vector<int>& ids, void* payload = nullptr);
    buffer<bf16> _prefill_with_mv(std::vector<int>& ids, void* payload = nullptr);
    buffer<bf16> _prefill_with_mm(std::vector<int>& ids, void* payload = nullptr);

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

    // bfp16 copy of the KV cache (README "bfp16 KV cache"): every finished 16-row chunk, packed by the host
    static constexpr int KV_CHUNK = 16;
    size_t kv_bf16_elems(uint32_t layer);
    void _kv_bfp16_invalidate_all();
    void _kv_bfp16_invalidate_positions(uint32_t layer, size_t begin, size_t end);
    void _kv_bfp16_ensure(int L_next);
    std::unique_ptr<hybrid_engine_state> new_state();
    int read_state(hybrid_engine_state& state, int kv_from);
    int write_state(const hybrid_engine_state& state);
    int write_state_kv(const hybrid_engine_state& state);
    /// @brief one flag per layer: global attention (keeps K/V rows) or sliding window (keeps a ring)
    std::vector<bool> attn_mask;
    /// @brief highest context length reached since the global rows past the length were last zero (see kv_rows_zero)
    size_t dirty_rows = 0;
    /// @brief sliding_temp_cache holds what an earlier prefill left there (or was never zeroed)
    bool temp_dirty = true;
    void _zero_temp_cache();
    bool is_checkpoint_valid;
    bool is_vlm;
    std::vector<buffer<bf16>> kv_checkpoint;

    size_t K01_offset;
    size_t K23_offset;
    size_t V01_offset;
    size_t V23_offset;
    size_t sliding_K01_offset;
    size_t sliding_K23_offset;
    size_t sliding_V01_offset;
    size_t sliding_V23_offset;

    LM_Config config;
    npu_xclbin_manager *npu;

    void _set_rope_rms_weights(int idx);

    inline void _rms_norm(buffer<bf16>& y, buffer<bf16>& x, buffer<bf16>& rms_weight);
    inline void _rope_rms(buffer<bf16>& x, int idx, buffer<bf16>& rms_weight, int layer_idx);
    inline void _residual_add(buffer<bf16>& dest, buffer<bf16>& src, buffer<bf16>& residual);
    inline void _elementwise_mul(buffer<bf16>& dest, buffer<bf16>& a, buffer<bf16>& b);
    void fill_kv_cache(buffer<bf16>& buffer_k, buffer<bf16>& buffer_v, int L_offset, int L_begin, int L_effective, int layer_idx);
    void fill_kv_cache_sliding(buffer<bf16>& local_kv_cache, buffer<bf16>& buffer_k, buffer<bf16>& buffer_v, int L_offset, int L_begin, int L_end, int L_effective, int sliding_l_begin, int slidding_l_end, int layer_idx);
    inline bool is_sliding_window(int layer_idx) {return (layer_idx + 1) % SLIDING_WINDOW_PATTERN != 0;}
    inline int linear2ring(int l) {return l % SLIDING_WINDOW;}

    npu_app_manager* layer_app_manager;
    npu_app_manager* gemm_app_manager;
    npu_app_manager* dequant_app_manager;
    npu_app_manager* mha_app_manager;

    npu_app layer;
    npu_app sliding_layer;
    npu_app qkv_dequant;
    npu_app o_dequant;
    npu_app gate_dequant;
    npu_app up_dequant;

    npu_app down_dequant;

    npu_app q_proj;
    npu_app k_proj;
    npu_app v_proj;
    npu_app o_proj;
    npu_app up_proj;
    npu_app gate_proj;
    npu_app down_proj;
    npu_app mha_engine;
    npu_app sliding_mha_engine;
    npu_app lm_head;

    std::unique_ptr<gemma_npu_sequence> sequence;
    std::unique_ptr<Gemm> gemm_seq_generator;
    std::unique_ptr<Dequant> dequant_seq_generator;
    std::unique_ptr<MHA> mha_seq_generator;
    std::unique_ptr<GemmaImageEncoder> gemma_image_encoder;
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
    uint32_t vocab_size;
    // size
    std::vector<buffer<bf16>> rms_weights;
    std::vector<buffer<bf16>> rope_rms_weights;
    std::vector<buffer<u8>> proj_weights;
    std::vector<buffer<bf16>> kv_caches;
    std::vector<std::vector<uint8_t>> kv_packed_valid;   // per layer, per chunk: the bfp16 copy matches the rows

    buffer<bf16> dequanted_qkv_weights; // QKV
    buffer<bf16> dequanted_o_weights; // O
    buffer<bf16> dequanted_up_weights; // UP,
    buffer<bf16> dequanted_gate_weights; // Gate
    buffer<bf16> dequanted_down_weights; // Down

    buffer<bf16> sliding_temp_cache;

    buffer<bf16> x;
    std::unique_ptr<Embedding> embedding;

    buffer<bf16> logits;
    buffer<bf16> logits_valid;
    buffer<u8> lm_head_weights;
    buffer<bf16> lm_head_norm;
    buffer<bf16> lm_head_unused;
    buffer<bf16> final_rms_weight;
    
    flm_rt::runlist layers_run;
    flm_rt::run lm_head_run;
    flm_rt::runlist dequant_all;

    int current_context_length;
    int checkpoint_context_length;
};
