// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#pragma once

#include "npu_utils/npu_instr_utils.hpp"
#include "lm_config.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"
#include "models/qwen3vl/qwen3vl_npu.hpp"
#include "qwen3vl_model_def.hpp"

struct qwen3vl_npu_sequence{
    /// @brief constexprs
    static constexpr npu_tiles xyr_tile = IT2;
    static constexpr npu_tiles attn_tiles[2] = {IT3, IT4};
    static constexpr npu_tiles mvm_tiles[4] = {IT0, IT1, IT6, IT7};
    static constexpr npu_tiles attn_qk_tiles[] = {CT03, CT23, CT04, CT24};
    static constexpr npu_tiles attn_kv_tiles[] = {CT13, CT33, CT14, CT34};
    static constexpr npu_tiles proj_tiles[] = {CT00, CT10, CT20, CT30,
                                               CT01, CT11, CT21, CT31,
                                               CT06, CT16, CT26, CT36,
                                               CT07, CT17, CT27, CT37
                                            };
    static constexpr npu_tiles rms_tile = CT02;

    static constexpr int xy_arg_idx = 0;
    static constexpr int w_proj_arg_idx = 1;
    static constexpr int w_rms_arg_idx = 2;
    static constexpr int w_rope_rms_arg_idx = 3;
    static constexpr int kv_cache_arg_idx = 4;

    static constexpr int m = 32;
    static constexpr int k = 256;
    static constexpr int L_CHUNK = 16;

    static constexpr int rms_addr = 54272;
    uint32_t D;
    uint32_t DH;
    uint32_t DK;
    uint32_t DV;
    uint32_t DQ;
    uint32_t MAX_L;

    uint32_t HIDDEN_SIZE;
    uint32_t kv_cache_size;

    /// @brief model definition owned by qwen3vl_npu::Impl; the weight descriptors
    ///        it holds carry the byte offsets the weight DMAs index by.
    const qwen3vl_desc* desc = nullptr;

    // RTP addresses, fixed for every supported Qwen3-VL model.
    static constexpr uint32_t l_qk_address = 9216;
    static constexpr uint32_t proj_addr = 33280;
    static constexpr uint32_t l_kv_address = 34816;
    static constexpr uint32_t l_begin_mha_address = 54272;
    static constexpr uint32_t l_end_mha_address = 25600;

    int vocab_size;
    int vocab_size_padded;

    void _send_x(npu_sequence* seq);
    void _send_rms_weights(npu_sequence* seq);
    void _send_rope_rms_weights(npu_sequence* seq);
    void _receive_kv_cache(npu_sequence* seq, const int L);
    void _move_weights(npu_sequence* seq, const size_t Dout, const size_t Din, const weight_desc_t& weight);
    void _move_kv_cache(npu_sequence* seq, const size_t L);

    void gen_dequant_seq(npu_sequence* seq, const size_t D_in, const size_t D_out, const size_t weight_offset);
    void gen_lm_head_seq(npu_sequence* seq);

    size_t K03_offset;
    size_t K47_offset;
    size_t V03_offset;
    size_t V47_offset;
    qwen3vl_npu_sequence(){}
    qwen3vl_npu_sequence(qwen3vl_desc& desc, LM_Config config, uint32_t MAX_L);
    ~qwen3vl_npu_sequence();

    void gen_layer_seq(npu_sequence* seq, const uint32_t L);

    void gen_mha_engine_seq(npu_sequence* seq, const uint32_t L_begin, const uint32_t L_end);

    void set_max_length(const uint32_t MAX_L);

    void set_context_length(int L);
    void update_max_length(uint32_t MAX_L);
    void clear_context();

    size_t get_k03_offset() { return this->K03_offset; }
    size_t get_k47_offset() { return this->K47_offset; }
    size_t get_v03_offset() { return this->V03_offset; }
    size_t get_v47_offset() { return this->V47_offset; }
};
