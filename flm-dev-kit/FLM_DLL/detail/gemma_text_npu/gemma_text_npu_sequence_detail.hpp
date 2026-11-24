// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#pragma once

#include "npu_utils/npu_instr_utils.hpp"
#include "lm_config.hpp"
#include "models/gemma_text/gemma_text_npu_sequence.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"
#include "models/gemma_text/gemma_text_npu.hpp"

struct gemma_text_npu_sequence::Impl{
    /// @brief constexprs
    static constexpr npu_tiles kv_tiles = IT0;
    static constexpr npu_tiles xyr_tile = IT1;
    static constexpr npu_tiles mvm_tiles[] = {IT2, IT3, IT4, IT5};
    static constexpr npu_tiles attn_qk_tiles = CT00;
    static constexpr npu_tiles attn_kv_tiles = CT10;

    static constexpr int xy_arg_idx = 0;
    static constexpr int w_proj_arg_idx = 1;
    static constexpr int w_rms_arg_idx = 2;
    static constexpr int w_rope_rms_arg_idx = 3;
    static constexpr int kv_cache_arg_idx = 4;

    static constexpr int m = 16;
    static constexpr int k = 128;
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
    uint32_t sliding_window;
    uint32_t sliding_window_pattern;

    // RTP addresses, fixed for every supported Gemma3-text model.
    static constexpr uint32_t l_qk_address = 9216;
    static constexpr uint32_t l_kv_address = 34816;
    static constexpr uint32_t l_begin_mha_address = 15872;
    static constexpr uint32_t l_end_mha_address = 15904;
    static constexpr uint32_t window_size_address = 15936;

    // for overlapping the sequence generation

    void _send_x(npu_sequence* seq);
    void _send_rope_rms_weights(npu_sequence* seq);
    void _receive_kv_cache(npu_sequence* seq, const int L, bool is_sliding_window);
    void _move_weights(npu_sequence* seq, const size_t Dout, const size_t Din, const size_t w_offset);
    void _move_kv_cache(npu_sequence* seq, const size_t L, bool is_sliding_window);

    size_t K_offset;
    size_t V_offset;
    size_t sliding_K_offset;
    size_t sliding_V_offset;

    Impl(){}
    Impl(LM_Config config, uint32_t MAX_L);
    ~Impl();

    void gen_layer_seq(npu_sequence* seq, const uint32_t L, bool is_sliding_window);

    void gen_mha_engine_seq(npu_sequence* seq, const uint32_t L_begin, const uint32_t L_end, bool is_sliding_window, int buffer_length);

    void set_max_length(const uint32_t MAX_L);

    void set_context_length(int L);
    void update_max_length(uint32_t MAX_L);
    void clear_context();
};
