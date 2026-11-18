// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#pragma once

#include "npu_utils/npu_instr_utils.hpp"
#include "lm_config.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"
#include "models/lfm2/lfm2_npu.hpp"
#include "lfm2_model_def.hpp"

struct lfm2_npu_sequence{
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
    static constexpr npu_tiles rope_tile = CT21;
    static constexpr int xy_arg_idx = 0;
    static constexpr int w_proj_arg_idx = 1;
    static constexpr int w_rms_arg_idx = 2;
    static constexpr int w_rope_arg_idx = 3;
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

    /// @brief model definition owned by lfm2_npu::Impl; the weight descriptors
    ///        it holds carry the byte offsets the weight DMAs index by.
    const lfm2_desc* desc = nullptr;

    // RTP addresses, fixed for every supported LFM2 model.
    static constexpr uint32_t l_qk_address = 5120;
    static constexpr uint32_t proj_addr = 33280;
    static constexpr uint32_t l_kv_address = 33792;
    static constexpr uint32_t l_begin_mha_address = 20480;
    static constexpr uint32_t l_end_mha_address = 35328;
    static constexpr uint32_t rms_addr = 36864;
    static constexpr uint32_t rope_addr = 22528;
    static constexpr uint32_t l_conv_address = 16512;

    // for overlapping the sequence generation

    void _send_x(npu_sequence* seq);
    void _send_rms_weights(npu_sequence* seq);
    void _send_rope_weights(npu_sequence* seq);
    void _send_conv_cache(npu_sequence* seq, const int L);
    void _receive_conv_cache(npu_sequence* seq, const int L);
    void _receive_kv_cache(npu_sequence* seq, const int L);
    void _move_weights(npu_sequence* seq, const size_t Dout, const size_t Din, const weight_desc_t& weight);
    void _move_kv_cache(npu_sequence* seq, const size_t L);
    // void generate_dequant_seq(npu_sequence* seq, const u32 D_in, const u32 D_out, const u32 weight_offset);
    void gen_lm_head_seq(npu_sequence* seq);

    size_t K03_offset;
    size_t K47_offset;
    size_t V03_offset;
    size_t V47_offset;
    lfm2_npu_sequence(){}
    lfm2_npu_sequence(lfm2_desc& desc, LM_Config config, uint32_t MAX_L);
    ~lfm2_npu_sequence();

    void gen_attn_layer_seq(npu_sequence* seq, const uint32_t L);
    void gen_conv_layer_seq(npu_sequence* seq, const uint32_t L);

    void gen_mha_engine_seq(npu_sequence* seq, const uint32_t L_begin, const uint32_t L_end);

    void gen_conv1d_engine_seq(npu_sequence* seq,
        const uint32_t L_PADDED, // assume padded
        const uint32_t D,
        const uint32_t L_OUT,  // caller handle the calculation of L_OUT through kernel size and stride...
        const uint32_t L_PADDED_REQUIREMENT = 1,
        const uint32_t weight_offset = 0
    );

    void set_max_length(const uint32_t MAX_L);

    void set_context_length(int L);
    void update_max_length(uint32_t MAX_L);
    void clear_context();

    inline int get_k03_offset() {
        return this->K03_offset;
    }

    inline int get_k47_offset() {
        return this->K47_offset;
    }

    inline int get_v03_offset() {
        return this->V03_offset;
    }

    inline int get_v47_offset() {
        return this->V47_offset;
    }
};
