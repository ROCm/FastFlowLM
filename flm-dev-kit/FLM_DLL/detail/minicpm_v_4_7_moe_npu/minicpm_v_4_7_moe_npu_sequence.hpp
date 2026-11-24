// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#pragma once

#include "npu_utils/npu_instr_utils.hpp"
#include "lm_config.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"
#include "models/minicpm_v_4_7_moe/minicpm_v_4_7_moe_npu.hpp"
#include "minicpm_v_4_7_moe_model_def.hpp"
struct minicpm_v_4_7_moe_npu_sequence{
    typedef enum: int{
        NORMAL_DEQUANT = 0,
        UP_MATRIX      = 1,
        GATE_MATRIX    = 2
    } dequant_output_mode_t;
    /// @brief constexprs
    static constexpr npu_tiles xyr_tile = IT2;
    static constexpr npu_tiles logits_out_tile = IT3;
    static constexpr npu_tiles attn_tile = IT3;
    static constexpr npu_tiles mvm_tiles[4] = {IT0, IT1, IT6, IT7};
    static constexpr npu_tiles attn_qk_tiles[] = {CT03, CT23};
    static constexpr npu_tiles attn_kv_tiles[] = {CT13, CT33};
    static constexpr npu_tiles proj_tiles[] = {CT00, CT10, CT20, CT30,
                                               CT01, CT11, CT21, CT31,
                                               CT06, CT16, CT26, CT36,
                                               CT07, CT17, CT27, CT37
                                            };
    static constexpr npu_tiles rms_tile = CT02;
    static constexpr npu_tiles gate_tile = CT22;
    static constexpr npu_tiles ab_proj_tile = CT15;

    static constexpr int w_proj_arg_idx = 0;
    static constexpr int xy_arg_idx = 1;
    static constexpr int w_rms_arg_idx = 2;
    static constexpr int w_rope_rms_arg_idx = 3;
    static constexpr int kv_cache_arg_idx = 4;

    static constexpr int L_CHUNK = 16;

    static constexpr uint32_t l_qk_address = 9216;
    static constexpr uint32_t l_kv_address = 36864;
    // expert-MM RTP addresses; the CT kernel differs between even and odd rows.
    static constexpr uint32_t CT_rtp_address_even_row = 54784;
    static constexpr uint32_t act_rtp_address_even_row = 54848;
    static constexpr uint32_t CT_rtp_address_odd_row = 61696;
    static constexpr uint32_t act_rtp_address_odd_row = 61760;
    // dense-MM RTP addresses; one CT kernel for every active row.
    static constexpr uint32_t mm_CT_rtp_address = 43008;
    static constexpr uint32_t mm_act_rtp_address = 57472;

    /// @brief shared model-shape definition, owned by minicpm_v_4_7_moe_npu::Impl.
    ///        Dimensions such as D, DK, Linear_DQ, ... are read from here
    ///        instead of being duplicated in this class.
    const minicpm_v_4_7_moe_desc* desc = nullptr;

    uint32_t MAX_L;
    uint32_t kv_cache_size;

    size_t linear_kv_cache_size;
    size_t conv_w_size_bf16;

    size_t s_offset;

    static constexpr uint32_t l_begin_mha_address = 61568;
    static constexpr uint32_t l_end_mha_address = 11904;
    static constexpr uint32_t window_size_address = 61600;

    void _gen_sequence(npu_sequence* seq, int L);
    void _send_hidden_states(npu_sequence* seq);
    void _send_rms_weights(npu_sequence* seq);
    void _send_rope_rms_weights(npu_sequence* seq);
    void _receive_kv_cache(npu_sequence* seq, const int L);
    void _move_kv_cache(npu_sequence* seq, const size_t L);
    void _move_weights(npu_sequence* seq, const uint32_t Dout, const uint32_t Din, const weight_desc_t& weight, const int arg_id = w_proj_arg_idx);
    void gen_lm_head_seq(npu_sequence* seq, int VOCAB_SIZE, int D);

    void _gen_linear_sequence(npu_sequence* seq, bool is_first = true);
    void _send_linear_conv_weights(npu_sequence* seq);
    void _receive_linear_kv_cache(npu_sequence* seq);
    void _move_linear_kv_cache(npu_sequence* seq);
    void _move_s(npu_sequence* seq);
    void _move_alpha_beta_weights(npu_sequence* seq, const uint32_t Dout, const uint32_t Din, const uint32_t w_offset);
    void _send_router_w_and_share_exp_gate(npu_sequence* seq, const size_t NUM_expert, const size_t D, const size_t W_offset_in_byte, mcvmoe_layer_weright_def* _weight_desc_ptr);
    void setup_expert_down_gate_q4k(npu_sequence*seq,
        int col, // the shimtile for this expert, 0-3
        int intermediate, int D,
        int bd_id_channel_0, int bd_id_channel_1,
        size_t const_w_byte_offset
    );
    void setup_expert_up_gate_q4k(
        npu_sequence*seq,
        int col, // the shimtile for this expert, 0-3
        int intermediate, int D,
        int bd_id_channel_0, int bd_id_channel_1,
        size_t const_w_byte_offset
    );
    void send_manual_expert_up_gate_q4k(
        npu_sequence*seq,
        int col, // the shimtile for this expert, 0-3
        int intermediate, int D,
        int expert_id, size_t const_w_byte_offset
    );
    void send_manual_expert_down_gate_q4k(
        npu_sequence*seq,
        int col, // the shimtile for this expert, 0-3
        int intermediate, int D,
        int expert_id, size_t const_w_byte_offset
    );
    inline size_t q8_0_size(size_t elems) { return elems + elems / 32 * 2;}

    minicpm_v_4_7_moe_npu_sequence(){}
    minicpm_v_4_7_moe_npu_sequence(minicpm_v_4_7_moe_desc& desc, LM_Config config, uint32_t MAX_L);
    ~minicpm_v_4_7_moe_npu_sequence();

    void gen_set_context(npu_sequence* seq, const uint32_t L) {
        for (int i = 0; i < 2; i++){
            seq->rtp_write(attn_qk_tiles[i], l_qk_address, L);
            seq->rtp_write(attn_kv_tiles[i], l_kv_address, L);
        }
    }
    void gen_layer_seq(npu_sequence* seq, const uint32_t L, bool is_attn_layer, bool is_first = true);
    void gen_mha_engine_seq(npu_sequence* seq, const uint32_t L_begin, uint32_t L);
    void gen_dequant_mm(npu_sequence* seq, const uint32_t L, const uint32_t Din, const uint32_t Dout, const size_t weight_offset, const int rtp_act, flm_dtype_t dtype);
    void gen_dequant_mm_512(npu_sequence* seq, const uint32_t L, const uint32_t Din, const uint32_t Dout, const size_t weight_offset, const size_t out_offset, const int rtp_act, flm_dtype_t dtype);
    void gen_mm_seq(npu_sequence* seq, const uint32_t M, const uint32_t K, const uint32_t N, const uint32_t weight_offset, uint32_t out_offset, int32_t rtp_act = 0);
    void gen_seq_conv1d(
        npu_sequence* seq,
        const uint32_t L_PADDED, // assume padded
        const uint32_t D,
        const uint32_t L_OUT,
        const uint32_t weight_offset
    );
    void gen_seq_conv1d_small(
        npu_sequence* seq,
        const uint32_t L_PADDED, // assume padded
        const uint32_t D,
        const uint32_t L_OUT,
        const uint32_t weight_offset
    );
    size_t get_V_offset() { return this->kv_cache_size / 2; }

    void set_max_length(const uint32_t MAX_L);
};
