// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#pragma once

#include "npu_utils/npu_instr_utils.hpp"
#include "lm_config.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"
#include "models/minicpm_ocr/minicpm_ocr_npu.hpp"
#include "minicpm_ocr_model_def.hpp"

struct minicpm_ocr_npu_sequence{
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

    static constexpr int L_CHUNK = 16;

    static constexpr uint32_t attn_qk_addr = 53248;
    static constexpr uint32_t attn_kv_addr = 53536;
    // layer_with_lm_head_kvb.xclbin (decode_kv_bfp16.hpp): its own attention RTP addresses (L[0] keys, L[1] finished
    // chunks read as bfp16) and the bfp16 K / V units in arg 3 after the head area (decode_kv_bfp16.hpp)
    bool kv_bfp16 = false;
    static constexpr uint32_t attn_qk_addr_kvb = 25600;
    static constexpr uint32_t attn_kv_addr_kvb = 53248;
    static constexpr uint32_t kv_bfp16_unit_bytes = 9216;
    static constexpr uint32_t kv_bfp16_lead_rows = 2;
    static constexpr size_t kv_bfp16_k_base = 4096;
    size_t kv_bfp16_cap = 0;
    // L[2] of the P.V core: 1 = no emulated multiply by 1.0 of y when no row max moved (README "Rescale skip")
    bool kv_bfp16_rescale_skip = false;
    size_t kv_bfp16_v_base() const { return kv_bfp16_k_base + kv_bfp16_cap * kv_bfp16_unit_bytes; }

    // layer_with_lm_head.xclbin (FLM_Xclbin/minicpm_ocr/decoding): every run starts by writing a mode RTP on the
    // projection cores and the rms core and releasing their lock 6. Off for the old layer.xclbin.
    bool write_modes = false;
    static constexpr uint32_t proj_mode_addr = 33280;
    static constexpr uint32_t rms_mode_addr = 34816;
    static constexpr uint32_t ct_lock_base = 0x1F000;
    static constexpr int mode_rtp_lock_id = 6;
    static constexpr int proj_mode_linear = 0;
    static constexpr int proj_mode_attn = 1;
    static constexpr int proj_mode_lm_head = 2;
    static constexpr int rms_mode_first_linear = 0;
    static constexpr int rms_mode_linear = 1;
    static constexpr int rms_mode_attn = 2;
    static constexpr int rms_mode_lm_head = 3;
    // lm_head run on layer_with_lm_head.xclbin: x, lm_head weights, model.norm, (unused), logits
    static constexpr int logits_arg_idx = 4;
    static constexpr npu_tiles lm_head_logits_tile = IT1;
    // expert-MM RTP addresses; the CT kernel differs between even and odd rows.
    static constexpr uint32_t CT_rtp_address_even_row = 54784;
    static constexpr uint32_t act_rtp_address_even_row = 54848;
    static constexpr uint32_t CT_rtp_address_odd_row = 61696;
    static constexpr uint32_t act_rtp_address_odd_row = 61760;

    /// @brief shared model-shape definition, owned by minicpm_ocr_npu::Impl.
    ///        Dimensions such as D, DK, Linear_DQ, ... are read from here
    ///        instead of being duplicated in this class.
    const minicpm_ocr_desc* desc = nullptr;

    uint32_t MAX_L;
    uint32_t kv_cache_size;

    size_t linear_kv_cache_size;
    size_t conv_w_size_bf16;

    size_t s_offset;

    static constexpr uint32_t l_begin_mha_address = 61568;
    static constexpr uint32_t l_end_mha_address = 11904;
    static constexpr uint32_t window_size_address = 61600;
    // attn_bfp16.xclbin (FLM_Xclbin/minicpm_ocr/gated_attention_prefill)
    static constexpr uint32_t mha_bfp16_l_begin = 64256;
    static constexpr uint32_t mha_bfp16_l_end = 64288;
    static constexpr uint32_t mha_bfp16_window_size = 64320;
    // mm_bfp16.xclbin (FLM_Xclbin/minicpm_ocr/dequant_mm_prefill): K_div_k / rtp_act, odd rows +8960
    static constexpr uint32_t mm_bfp16_ct_even = 55552;
    static constexpr uint32_t mm_bfp16_act_even = 55616;
    static constexpr uint32_t mm_bfp16_ct_odd = 64512;
    static constexpr uint32_t mm_bfp16_act_odd = 64576;
    // mm_bfp16_glu.xclbin: rtp_act 2 = GLU epilogue; hid goes to one 16 KB slot per (128 rows, 32 features), its
    // first glu_slot_used bytes the down projection's bfp16 x tiles (FLM_Xclbin/minicpm_ocr/dequant_mm_prefill README)
    static constexpr int mm_bfp16_act_glu = 2;
    static constexpr uint32_t glu_slot_bytes = 16384;
    static constexpr uint32_t glu_slot_used = 4608;
    static constexpr uint32_t glu_slot_features = 32;
    static size_t glu_slots_bytes(size_t L_padded, size_t IS) { return L_padded / 128 * (IS / glu_slot_features) * glu_slot_bytes; }

    void _gen_sequence(npu_sequence* seq, int L);
    void _send_hidden_states(npu_sequence* seq);
    void _send_rms_weights(npu_sequence* seq);
    void _send_rope_rms_weights(npu_sequence* seq);
    void _receive_kv_cache(npu_sequence* seq, const int L);
    void _move_kv_cache(npu_sequence* seq, const size_t L);
    void _move_kv_cache_bfp16(npu_sequence* seq, const size_t L);
    void _move_weights(npu_sequence* seq, const size_t Dout, const size_t Din, const weight_desc_t& weight);
    void gen_lm_head_seq(npu_sequence* seq, int VOCAB_SIZE, int D);
    void _set_modes(npu_sequence* seq, int proj_mode, int rms_mode);
    void gen_lm_head_in_layer_seq(npu_sequence* seq);

    void _gen_linear_sequence(npu_sequence* seq, bool is_first = true);
    void _send_linear_conv_weights(npu_sequence* seq);
    void _receive_linear_kv_cache(npu_sequence* seq);
    void _move_linear_kv_cache(npu_sequence* seq);
    void _move_s(npu_sequence* seq);
    void _move_alpha_beta_weights(npu_sequence* seq, const size_t Dout, const size_t Din, const size_t w_offset);
    void gen_dequant_mm_512(npu_sequence* seq, const uint32_t L, const uint32_t Din, const uint32_t Dout, const size_t weight_offset, const size_t out_offset, const int rtp_act);
    void gen_dequant_mm_bfp16(npu_sequence* seq, const uint32_t L, const uint32_t Din, const uint32_t Dout, const size_t weight_offset, const size_t out_offset, const int rtp_act, const bool x_from_glu_slots = false, const uint32_t x_row_begin = 0);
    void gen_dequant_mm_bfp16_glu(npu_sequence* seq, const uint32_t L, const uint32_t Din, const uint32_t IS, const size_t weight_offset, const uint32_t x_row_begin = 0);
    inline size_t q8_0_size(size_t elems) { return elems + elems / 32 * 2;}

    minicpm_ocr_npu_sequence(){}
    minicpm_ocr_npu_sequence(minicpm_ocr_desc& desc, LM_Config config, uint32_t MAX_L);
    ~minicpm_ocr_npu_sequence();

    void gen_set_context(npu_sequence* seq, const uint32_t L) {
        const uint32_t qk_addr = kv_bfp16 ? attn_qk_addr_kvb : attn_qk_addr;
        const uint32_t kv_addr = kv_bfp16 ? attn_kv_addr_kvb : attn_kv_addr;
        for (int i = 0; i < 2; i++){
            seq->rtp_write(attn_qk_tiles[i], qk_addr, L);
            seq->rtp_write(attn_kv_tiles[i], kv_addr, L);
            if (kv_bfp16){
                seq->rtp_write(attn_qk_tiles[i], qk_addr + 4, (L - 1) / L_CHUNK);
                seq->rtp_write(attn_kv_tiles[i], kv_addr + 4, (L - 1) / L_CHUNK);
                seq->rtp_write(attn_kv_tiles[i], kv_addr + 8, kv_bfp16_rescale_skip ? 1 : 0);
            }
        }
    }
    void gen_layer_seq(npu_sequence* seq, const uint32_t L, bool is_attn_layer, bool is_first = true);
    void gen_mha_engine_seq(npu_sequence* seq, const uint32_t L_begin, const uint32_t L_wanted);
    void gen_mha_bfp16_seq(npu_sequence* seq, const uint32_t L_begin, const uint32_t L_wanted, const uint32_t kv_bfp16_rows);
    void gen_seq_conv1d(
        npu_sequence* seq,
        const uint32_t L_PADDED, // assume padded
        const uint32_t D,
        const uint32_t L_OUT,
        const uint32_t weight_offset
    );
    size_t get_V_offset() { return this->kv_cache_size / 2; }

    void set_max_length(const uint32_t MAX_L);
};
