// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file hunyuan_npu_detail.hpp
/// \brief hunyuan_npu::Impl
/// \author FastFlowLM Team
/// \date 2026-09-01
#ifndef __HUNYUAN_NPU_DETAIL_HPP__
#define __HUNYUAN_NPU_DETAIL_HPP__
#include "models/hunyuan/hunyuan_npu.hpp"
#include "hunyuan_npu_sequence.hpp"
#include "modules/gemm.hpp"
#include "hunyuan_model_def.hpp"
#include "hunyuan_prefill.hpp"
#include "hunyuan_bfp16.hpp"
#include "utils/avx512_util.hpp"

struct hunyuan_npu::Impl{
    LM_Config config;
    npu_xclbin_manager *npu;
    hunyuan_desc desc;
    std::unique_ptr<hunyuan_npu_sequence> sequence;
    std::unique_ptr<hunyuan_prefill_context> pctx;
    hunyuan_npu* parent_ptr;

    uint32_t MAX_L;
    int vocab_size;
    int vocab_size_padded;
    /// @brief bases of the four kv-cache quadrants (K heads 0..1, K heads 2..3,
    ///        V heads 0..1, V heads 2..3), mirrored from hunyuan_npu_sequence.
    size_t K03_offset;
    size_t K47_offset;
    size_t V03_offset;
    size_t V47_offset;

    int current_context_length;

    bool is_checkpoint_valid;
    int checkpoint_context_length;
    std::vector<buffer<bf16>> cache_checkpoint;

    /// @brief layer.xclbin backs both the decode layer and the lm_head; the
    ///        is_attn RTP is what tells the two apart.
#if HUNYUAN_DECODE_ON_NPU || HUNYUAN_LM_HEAD_ON_NPU
    npu_app_manager* layer_app_manager;
#endif
#if HUNYUAN_DECODE_ON_NPU
    /// @brief one half of the decode ping-pong: everything a single token's
    ///        layer pass needs, built for one context length.
    /// @note  The expensive part of preparing a token is not the array work, it
    ///        is the host work behind create_run() -- npu_app::_setup_kernel()
    ///        reassembles the elf every time the sequence changes, and the
    ///        sequence changes on every token because the RTPs carry L. With
    ///        two slots the next token's elf is assembled while the current
    ///        token's runlist is on the array, so it never sits between tokens.
    struct decode_slot_t{
        npu_app layer;
        npu_app set_layer_rtp;
        flm_rt::runlist run;
        /// @brief the same runs plus the lm_head at the end, for tokens whose
        ///        logits are read: one submission per token instead of two
        flm_rt::runlist run_logits;
        /// @brief cos | sin | q_norm | k_norm, per layer. Per slot, not shared:
        ///        the cos/sin half is rewritten for position L+1 while the other
        ///        slot's layers are still reading position L out of theirs.
        std::vector<buffer<bf16>> rope;
        /// @brief the context length this slot is built for, -1 when invalid.
        ///        The sequences and the rope table depend on nothing else, so a
        ///        matching L makes a slot reusable no matter how the context
        ///        length got there.
        int L = -1;
    };
    decode_slot_t decode_slots[2];
    /// @brief the slot holding the sequence for current_context_length
    int active_slot = 0;
#endif
#if HUNYUAN_LM_HEAD_ON_NPU
    npu_app lm_head;
    flm_rt::run lm_head_run;
#endif

    /// @brief per layer weights. proj_weights holds the reordered q4_k body in
    ///        the order hunyuan_desc::load_layer_weights added it; rms_weights
    ///        holds input_layernorm at 0 and post_attention_layernorm at +D;
    ///        rope_rms_weights holds cos | sin | q_norm | k_norm.
    std::vector<buffer<u8>> proj_weights;
    std::vector<buffer<bf16>> rms_weights;
    std::vector<buffer<bf16>> rope_rms_weights;
    /// @brief per layer: the bf16 cache (four quadrants), then the bfp16 V^T copy of the
    ///        finished 16-key chunks that decode attention reads (sequence Vb03/Vb47_offset).
    ///        See "bfp16 KV copy" in README.md.
    std::vector<buffer<bf16>> kv_caches;
    /// @brief finished chunks converted to bfp16 so far (the same for every layer)
    int bfp16_chunks = 0;
    /// @brief rows [0, kv_rows_used) of the bf16 cache may be non-zero; every row past it is zero
    int kv_rows_used = 0;

    /// @brief the decode hidden state, [D]. Both the layer sequence and the
    ///        lm_head sequence read it in place.
    buffer<bf16> x;
    /// @brief the final model.norm weight, [D]. Its own BO because the lm_head
    ///        sequence binds it as a separate kernel argument.
    buffer<bf16> final_rms_weight;
    buffer<bf16> decode_hidden;
    std::unique_ptr<Embedding> embedding;

    buffer<bf16> logits;
    buffer<bf16> logits_valid;
    buffer<u8> lm_head_weights;

    Impl(LM_Config config, npu_xclbin_manager *npu_instance, hunyuan_npu* parent_ptr, int MAX_L = 4096);
    ~Impl();

    /// @param need_logits false when the token is only being pushed into the kv
    ///        cache -- a prefill tail run through the decode path -- so the
    ///        lm_head can be skipped for every token but the last.
    buffer<bf16> forward(int ids, bool need_logits = true);
    buffer<bf16> prefill(std::vector<int>& ids, void* payload = nullptr);
    buffer<bf16> _prefill_short(std::vector<int>& ids);
    buffer<bf16> _prefill_long(std::vector<int>& ids);
    buffer<bf16> _prefill_batched(std::vector<int>& ids, bool need_logits, bool small_rows = false);

    void set_context_length(int L);
    void load_weights(Q4NX& q4nx);
    void update_max_length(uint32_t MAX_L);
    void clear_context();

    buffer<bf16> get_k_cache(int layer_idx, int idx);
    buffer<bf16> get_v_cache(int layer_idx, int idx);
    buffer<bf16> get_logits(buffer<bf16>& predict);

    int get_current_context_length();

    void _set_rope_rms_weights(int idx, std::vector<buffer<bf16>>& rope);
#if HUNYUAN_DECODE_ON_NPU
    /// @brief (re)build @p slot for context length @p L; a no-op if it already is
    void _build_slot(int slot, int L);
    /// @brief return the slot built for @p L, building the active one if neither is
    int _select_slot(int L);
    /// @brief drop both slots, e.g. after the kv cache buffers were reallocated
    void _invalidate_slots();
    /// @brief seed both slots' rope tables from the loaded q_norm / k_norm
    void _sync_slot_rope_weights();
#endif

    int checkpoint();
    int restore();
    /// @brief zero the bf16 cache rows [first, kv_rows_used) of every layer (HY_ZERO_STALE_KV)
    void _zero_stale_kv(int first);

    /// @brief convert every chunk that is complete in the first @p context_len rows
    void _update_bfp16(int context_len);
    /// @brief chunks [first, target) of one layer; the prefill runs it while that
    ///        layer's attention is on the array
    void _update_bfp16_layer(uint32_t layer_idx, int first, int target);
    /// @brief HY_BFP16_CHECK=1: re-pack the whole cache and compare with the incremental copy
    void _check_bfp16();
    void _allocate_checkpoint_buffers();
};

#endif // __HUNYUAN_NPU_DETAIL_HPP__
