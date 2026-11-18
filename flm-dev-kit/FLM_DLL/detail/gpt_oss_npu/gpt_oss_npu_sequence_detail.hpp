// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#pragma once

#include "npu_utils/npu_instr_utils.hpp"
#include "lm_config.hpp"
#include "models/gpt_oss/gpt_oss_npu_sequence.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"
#include "models/gpt_oss/gpt_oss_npu.hpp"
#include "gpt_oss_params.hpp"

struct gpt_oss_npu_sequence::Impl{
    /// @brief constexprs
    static constexpr npu_tiles kv_tiles[] = {IT0, IT7};
    static constexpr npu_tiles xyr_tile = IT1;
    static constexpr npu_tiles mvm_tiles[] = {IT2, IT3, IT4, IT5};
    static constexpr npu_tiles attn_qk_tiles[] = {CT00, CT20, CT07, CT27};
    static constexpr npu_tiles attn_kv_tiles[] = {CT10, CT30, CT17, CT37};

    static constexpr int xy_arg_idx = 1;
    static constexpr int w_proj_arg_idx = 0;
    static constexpr int w_rms_arg_idx = 2;
    static constexpr int w_rope_arg_idx = 3;
    static constexpr int kv_cache_arg_idx = 4;

    static constexpr uint32_t l_begin_mha_address = 9216;
    static constexpr uint32_t l_end_mha_address = 20992;
    static constexpr uint32_t window_size_address = 36992;
    static constexpr uint32_t sinks_address = 51328;

    static constexpr int m = Q4NX_ROW;
    static constexpr int k = Q4NX_COL;
    static constexpr int L_CHUNK = 16;

    // for overlapping the sequence generation

    int MAX_L;
    uint32_t kv_cache_size;

    void _send_x(npu_sequence* seq);
    void _send_rms_weights(npu_sequence* seq);
    void _send_rope_weights(npu_sequence* seq);
    void _receive_kv_cache(npu_sequence* seq, const int L, bool is_sliding_window);
    void _move_weights(npu_sequence* seq, const size_t Dout, const size_t Din, const size_t w_offset);
    void _move_kv_cache(npu_sequence* seq, const int L, bool is_sliding_window);
    void generate_lm_head_seq(npu_sequence* seq);

    size_t K03_offset;
    size_t K47_offset;
    size_t V03_offset;
    size_t V47_offset;
    size_t K03_sliding_offset;
    size_t K47_sliding_offset;
    size_t V03_sliding_offset;
    size_t V47_sliding_offset;
    Impl(){}
    Impl(LM_Config config, uint32_t MAX_L);
    ~Impl();

    void gen_layer_seq(npu_sequence* seq, const int L, bool is_sliding_window);

    void gen_mha_engine_seq(npu_sequence* seq, const uint32_t L_begin, const uint32_t L_end, bf16* sinks, bool is_sliding_window = true);

    void set_max_length(const uint32_t MAX_L);

    void set_context_length(int L);
    void update_max_length(uint32_t MAX_L);
    void clear_context();
};
