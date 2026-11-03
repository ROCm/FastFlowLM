#pragma once

#include "npu_utils/npu_instr_utils.hpp"
#include "lm_config.hpp"
#include "models/qwen3_5vl/qwen3_5vl_npu.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"
#include "qwen3_5vl_model_def.hpp"

struct qwen3_5vl_npu_sequence{
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

    static constexpr int xy_arg_idx = 0;
    static constexpr int w_proj_arg_idx = 1;
    static constexpr int w_rms_arg_idx = 2;
    static constexpr int w_rope_rms_arg_idx = 3;
    static constexpr int kv_cache_arg_idx = 4;
    // lm_head run (mode 2): x, lm_head weights, model.norm, (unused), logits
    static constexpr int logits_arg_idx = 4;
    static constexpr npu_tiles lm_head_logits_tile = IT1;

    static constexpr int L_CHUNK = 16;

    static constexpr uint32_t attn_qk_addr = 53248;
    static constexpr uint32_t attn_kv_addr = 53536;
    // projection cores read their mode (PROJ_MODE_* in kernels/aie_array_layout.h) after the host releases lock 6
    static constexpr uint32_t proj_mode_addr = 33280;
    static constexpr uint32_t ct_lock_base = 0x1F000;
    static constexpr int proj_rtp_lock_id = 6;
    static constexpr int proj_mode_linear = 0;
    static constexpr int proj_mode_attn = 1;
    static constexpr int proj_mode_lm_head = 2;
    // the rms core reads its mode (RMS_MODE_* in kernels/aie_array_layout.h) the same way
    static constexpr uint32_t rms_mode_addr = 36864;
    static constexpr int rms_mode_first_linear = 0;
    static constexpr int rms_mode_linear = 1;
    static constexpr int rms_mode_attn = 2;
    static constexpr int rms_mode_lm_head = 3;
    bool write_proj_mode = true;
    // expert-MM RTP addresses; the CT kernel differs between even and odd rows.
    static constexpr uint32_t CT_rtp_address_even_row = 54784;
    static constexpr uint32_t act_rtp_address_even_row = 54848;
    static constexpr uint32_t CT_rtp_address_odd_row = 61696;
    static constexpr uint32_t act_rtp_address_odd_row = 61760;

    /// @brief shared model-shape definition, owned by qwen3_5vl_npu::Impl.
    ///        Dimensions such as D, DK, Linear_DQ, ... are read from here
    ///        instead of being duplicated in this class.
    const qwen3_5vl_desc* desc = nullptr;

    uint32_t MAX_L;
    uint32_t kv_cache_size;

    size_t linear_kv_cache_size;
    size_t conv_w_size_bf16;

    size_t s_offset;

    static constexpr uint32_t l_begin_mha_address = 61568;
    static constexpr uint32_t l_end_mha_address = 11904;
    static constexpr uint32_t window_size_address = 61600;

    void _gen_sequence(npu_sequence* seq, int L);
    void _set_modes(npu_sequence* seq, int proj_mode, int rms_mode);
    void _send_hidden_states(npu_sequence* seq);
    void _send_rms_weights(npu_sequence* seq);
    void _send_rope_rms_weights(npu_sequence* seq);
    void _receive_kv_cache(npu_sequence* seq, const int L);
    void _move_kv_cache(npu_sequence* seq, const size_t L);
    void _move_weights(npu_sequence* seq, const size_t Dout, const size_t Din, const weight_desc_t& weight);
    void gen_lm_head_seq(npu_sequence* seq, int VOCAB_SIZE, int D);
    void gen_lm_head_in_layer_seq(npu_sequence* seq);

    void _gen_linear_sequence(npu_sequence* seq, bool is_first = true);
    void _send_linear_conv_weights(npu_sequence* seq);
    void _receive_linear_kv_cache(npu_sequence* seq);
    void _move_linear_kv_cache(npu_sequence* seq);
    void _move_s(npu_sequence* seq);
    void _move_alpha_beta_weights(npu_sequence* seq, const size_t Dout, const size_t Din, const size_t w_offset);
    void gen_dequant_mm_512(npu_sequence* seq, const uint32_t L, const uint32_t Din, const uint32_t Dout, const size_t weight_offset, const size_t out_offset, const int rtp_act, const size_t x_offset_base = 0);
    inline size_t q8_0_size(size_t elems) { return elems + elems / 32 * 2;}

    qwen3_5vl_npu_sequence(){}
    qwen3_5vl_npu_sequence(qwen3_5vl_desc& desc, LM_Config config, uint32_t MAX_L);
    ~qwen3_5vl_npu_sequence();

    void gen_set_context(npu_sequence* seq, const uint32_t L) {
        for (int i = 0; i < 2; i++){
            seq->rtp_write(attn_qk_tiles[i], attn_qk_addr, L);
            seq->rtp_write(attn_kv_tiles[i], attn_kv_addr, L);
        }
    }
    void gen_layer_seq(npu_sequence* seq, const uint32_t L, bool is_attn_layer, bool is_first = true);
    void gen_mha_engine_seq(npu_sequence* seq, const uint32_t L_begin, const uint32_t L_wanted);
    void gen_seq_conv1d(
        npu_sequence* seq,
        const uint32_t L_PADDED, // assume padded
        const uint32_t D,
        const uint32_t L_OUT,
        const uint32_t weight_offset
    );
    void gen_seq_conv1d_v2(
        npu_sequence* seq,
        const uint32_t L_PADDED, // L_OUT + (window_size - 1)
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
