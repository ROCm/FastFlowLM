// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#pragma once
#include "typedef.hpp"
#include "npu_utils/npu_instr_utils.hpp"
#include "lm_config.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"

class qwen2vl_npu_sequence{
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

    uint32_t HIDDEN_SIZE;
    uint32_t kv_cache_size;
    uint32_t w_qkv_offset;
    uint32_t w_o_offset;
    uint32_t w_up_offset;
    uint32_t w_down_offset;

    // RTP addresses, fixed for every supported Qwen2.5-VL model.
    static constexpr uint32_t rms_addr = 53248;
    static constexpr uint32_t l_qk_address = 5120;
    static constexpr uint32_t proj_addr = 33280;
    static constexpr uint32_t l_kv_address = 34816;
    static constexpr uint32_t l_begin_mha_address = 15360;
    static constexpr uint32_t l_end_mha_address = 25088;
    static constexpr uint32_t window_size_address = 36992;

    int vocab_size;
    int vocab_size_padded;
    int intermediate_size;

    // for overlapping the sequence generation

public:

    void _send_x(npu_sequence* seq);
    void _send_rms_weights(npu_sequence* seq);
    void _send_rope_rms_weights(npu_sequence* seq);
    void _receive_kv_cache(npu_sequence* seq, const int L);
    void _move_weights(npu_sequence* seq, const size_t Dout, const size_t Din, const size_t w_offset);
    void _move_kv_cache(npu_sequence* seq, const size_t L);

    void gen_dequant_seq(npu_sequence* seq, const size_t D_in, const size_t D_out, const size_t weight_offset);
    void gen_lm_head_seq(npu_sequence* seq);

    size_t V_offset;

    qwen2vl_npu_sequence(){}
    qwen2vl_npu_sequence(LM_Config config, uint32_t MAX_L);
    ~qwen2vl_npu_sequence();

    void gen_layer_seq(npu_sequence* seq, const uint32_t L);

    void gen_mha_engine_seq(npu_sequence* seq, const uint32_t L_begin, const uint32_t L_end);

    void set_max_length(const uint32_t MAX_L);

    void update_max_length(uint32_t MAX_L){
        this->MAX_L = MAX_L;
        this->kv_cache_size = MAX_L * (DK + DV);
        this->V_offset = kv_cache_size / 2;
    }

    int get_V_Offset(){return V_offset;}
};
