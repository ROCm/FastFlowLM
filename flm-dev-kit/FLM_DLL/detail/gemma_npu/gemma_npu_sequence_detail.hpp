// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#pragma once

#include "npu_utils/npu_instr_utils.hpp"
#include "lm_config.hpp"
#include "models/gemma/gemma_npu_sequence.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"
#include "models/gemma/gemma_npu.hpp"

struct gemma_npu_sequence::Impl{
    /// @brief constexprs
    static constexpr npu_tiles kv_tiles[] = {IT0, IT7};
    static constexpr npu_tiles xyr_tile = IT1;
    static constexpr npu_tiles mvm_tiles[] = {IT2, IT3, IT4, IT5};
    static constexpr npu_tiles attn_qk_tiles[] = {CT00, CT20, CT07, CT27};
    static constexpr npu_tiles attn_kv_tiles[] = {CT10, CT30, CT17, CT37};
    static constexpr npu_tiles proj_tiles[] = {CT02, CT12, CT22, CT32,
                                               CT03, CT13, CT23, CT33,
                                               CT04, CT14, CT24, CT34,
                                               CT05, CT15, CT25, CT35
                                            };
    static constexpr npu_tiles rms_tile = CT01;
    static constexpr int xy_arg_idx = 0;
    static constexpr int w_proj_arg_idx = 1;
    static constexpr int w_rms_arg_idx = 2;
    static constexpr int w_rope_rms_arg_idx = 3;
    static constexpr int kv_cache_arg_idx = 4;

    static constexpr int m = 32;
    static constexpr int k = 256;
    static constexpr int L_CHUNK = 16;
    uint32_t D;
    uint32_t DH;
    uint32_t DK;
    uint32_t DV;
    uint32_t DQ;
    uint32_t MAX_L;
    uint32_t vocab_size;

    uint32_t HIDDEN_SIZE;
    uint32_t kv_cache_size;
    uint32_t w_qkv_offset;
    uint32_t w_o_offset;
    uint32_t w_up_offset;
    uint32_t w_down_offset;
    uint32_t sliding_window;
    uint32_t sliding_window_pattern;

    // RTP addresses, fixed for every supported Gemma3 (vision) model.
    static constexpr uint32_t l_qk_address = 9216;
    static constexpr uint32_t proj_addr = 33280;
    static constexpr uint32_t l_kv_address = 33792;
    static constexpr uint32_t l_begin_mha_address = 15872;
    static constexpr uint32_t l_end_mha_address = 15904;
    static constexpr uint32_t window_size_address = 15936;
    static constexpr uint32_t rms_addr = 54272;
    // the same RTPs in the KV_BFP16 build of the kernel (README "bfp16 KV cache"); L[1] = packed chunks
    static constexpr uint32_t l_qk_address_kvb = 20992;
    static constexpr uint32_t l_kv_address_kvb = 50176;
    static constexpr uint32_t kv_bfp16_unit_elems = 4608;   // one stream's 16-row chunk as bfp16: 9,216 B

    // lm_head inside layer_lm.xclbin (gemma3:4b): mode values for the proj and rms RTPs, the vocab size the
    // kernel is built for (common.h), and its padded row count (whole rounds of D logits).
    static constexpr uint32_t lm_head_proj_mode = 0;
    static constexpr uint32_t lm_head_rms_passthrough_mode = 2;
    static constexpr uint32_t lm_head_in_layer_vocab = 262208;
    static constexpr int lm_head_in_layer_logits_arg_idx = 4;
    uint32_t lm_head_in_layer_rows;

    // for overlapping the sequence generation

    void _send_x(npu_sequence* seq);
    void _send_rms_weights(npu_sequence* seq, uint32_t w_offset);
    void _send_rope_rms_weights(npu_sequence* seq);
    void _receive_kv_cache(npu_sequence* seq, const int L, bool is_sliding_window);
    void _move_weights(npu_sequence* seq, const size_t Dout, const size_t Din, const size_t w_offset);
    void _move_kv_cache(npu_sequence* seq, const size_t L, bool is_sliding_window);
    void _move_kv_cache_bfp16(npu_sequence* seq, const size_t L, bool is_sliding_window);
    uint32_t kv_bfp16_packed_chunks(const uint32_t L, bool is_sliding_window) const;
    size_t kv_bfp16_offset(bool is_sliding_window, int stream) const;

    size_t K01_offset;
    size_t K23_offset;
    size_t V01_offset;
    size_t V23_offset;
    size_t K01_sliding_offset;
    size_t K23_sliding_offset;
    size_t V01_sliding_offset;
    size_t V23_sliding_offset;
    Impl(){}
    Impl(LM_Config config, uint32_t MAX_L);
    ~Impl();

    void gen_layer_seq(npu_sequence* seq, const uint32_t L, bool is_sliding_window, bool kv_bfp16);
    void gen_lm_head_in_layer_seq(npu_sequence* seq);
    void gen_mha_engine_seq(npu_sequence* seq, const uint32_t L_begin, const uint32_t L_end, bool is_sliding_window, int buffer_length);

    void set_max_length(const uint32_t MAX_L);

    void set_context_length(int L);
    void update_max_length(uint32_t MAX_L);
    void clear_context();
};
