// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#pragma once
#include "models/hy_mem/hy_mem_npu.hpp"
#include "hy_mem_model_def.hpp"
#include "hy_mem_npu_sequence.hpp"
#include "hy_mem_prefill.hpp"
#include "utils/kv_rows_state.hpp"

/// @brief everything the engine owns
/// @note  One token = three runs on three hw contexts, in order:
///          dense_run                      layer 0                (dense_layer.xclbin)
///          moe_runlist                    47 x [router RTP, layer] (moe_layer.xclbin)
///          lm_head_run                    final norm + logits    (lm_head.xclbin)
///        The layer runs all read and write the same x buffer on the device, so
///        nothing goes back to the host between them.
struct hy_mem_npu::Impl{
    LM_Config config;
    npu_xclbin_manager *npu;
    hy_mem_desc desc;
    std::unique_ptr<hy_mem_npu_sequence> sequence;
    std::unique_ptr<hy_mem_prefill_context> pctx;   // the batched prefill
    std::unique_ptr<hy_mem_prefill_overlay> overlay; // its NPU stages (prefill_overlay.xclbin)
    uint32_t MAX_L;
    // tokens per batched prefill chunk: bounds the scratch (~100 KB a token, ~400 MB a chunk)
    static constexpr int PREFILL_CHUNK = 4096;

    npu_app_manager* dense_app_manager;
    npu_app_manager* moe_app_manager;
    npu_app_manager* lm_head_app_manager;

    npu_app dense_layer;               // sequence depends on L, rebuilt per token
    npu_app moe_layer;                 // one sequence for all MoE layers, rebuilt per token
    std::vector<npu_app> router_rtp;   // per MoE layer: layer number + expert bias, built once
    npu_app lm_head;

    // per layer
    std::vector<buffer<u8>> proj_weights;
    // Dense layers only (~35 MB for layer 0): the same projections in the prefill
    // overlay's block order. The 47 MoE layers are already stored that way for the
    // int8-mmul decode kernel, so they are read in place and get no copy here.
    std::vector<buffer<u8>> prefill_proj;
    std::vector<buffer<bf16>> rms_weights;
    std::vector<buffer<bf16>> rope_rms_weights;   // [cos | sin | q_norm | k_norm]
    std::vector<buffer<bf16>> kv_caches;          // [K col 0][K col 1][V col 0][V col 1]
    std::vector<std::vector<float>> router_bias;

    // x: [0, D) layer input, then layer output; sized for the MoE overlay, which
    // owns [D, D + DQ) for its attention tap build
    static constexpr size_t X_ELEMS = 8192;
    buffer<bf16> x;
    buffer<bf16> lm_x;                 // [x | final norm weight]
    buffer<u8> lm_head_weights;
    buffer<bf16> logits;
    buffer<bf16> logits_valid;         // first vocab_size entries of logits
    std::unique_ptr<Embedding> embedding;

    flm_rt::run dense_run;
    flm_rt::runlist moe_runlist;
    flm_rt::run lm_head_run;

    int current_context_length;
    int checkpoint_context_length;
    bool is_checkpoint_valid;
    /// @brief highest context length reached since the rows were last known to be zero
    /// @note  Rows below it hold some conversation's K/V. write_state() zeroes
    ///        [restored length, dirty_rows) so a restored context looks like a fresh
    ///        one; measured, not assumed (check_engine.cpp check 6).
    size_t dirty_rows = 0;

    Impl(LM_Config config, npu_xclbin_manager *npu_instance, int MAX_L);
    ~Impl();

    buffer<bf16> forward(int ids);
    buffer<bf16> prefill(std::vector<int>& ids, void* payload = nullptr);
    buffer<bf16> _prefill_batched(std::vector<int>& ids);   // default
    buffer<bf16> _prefill_with_mv(std::vector<int>& ids);   // PREFILL_WITH_MV: decode overlays, token by token
    void set_context_length(int L);
    void load_weights(Q4NX& q4nx);
    void update_max_length(uint32_t MAX_L);
    void clear_context();
    buffer<bf16> get_k_cache(int layer_idx, int idx);
    buffer<bf16> get_v_cache(int layer_idx, int idx);
    int get_current_context_length() { return current_context_length; }
    int checkpoint();
    int restore();

    /// @brief where this layer's four K/V planes sit, for kv_rows_state.hpp
    /// @note  Rebuilt on every call: the plane offsets scale with MAX_L, which
    ///        update_max_length() can grow under a state that was read earlier.
    kv_row_layout kv_layout() const;
    std::unique_ptr<hybrid_engine_state> new_state();
    int read_state(hybrid_engine_state& state, int kv_from);
    int write_state(const hybrid_engine_state& state);
    int write_state_kv(const hybrid_engine_state& state);

    /// @brief round a requested KV length up to a power of two in [4096, MAX_CONTEXT]
    static uint32_t _normalize_max_length(uint32_t requested);
    /// @brief embedding of `id` into x, then all 48 layers; x holds the last hidden state
    void _run_layers(int id);
    /// @brief x -> final norm + logits
    void _start_lm_head();
    void _finish_lm_head();
    /// @brief cos / sin of position `pos` into every layer's rope buffer
    void _set_rope(int pos);
    /// @brief K (kv = 0) or V (kv = 1) row `idx` of a layer, all KV heads in head order
    buffer<bf16> _gather_kv_row(int layer_idx, int kv, int idx);
};
