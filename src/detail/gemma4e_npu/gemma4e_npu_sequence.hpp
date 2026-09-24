#pragma once

#include "npu_utils/npu_instr_utils.hpp"
#include "lm_config.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"
#include "models/gemma4e/gemma4e_npu.hpp"
#include "weight_desc.hpp"
#include "gemma4e_npu_def.hpp"

#define MIN_BF16_PAD 32 // minimum padding in number of bf16 elements to avoid NPU OOM when processing long sequences, which is determined empirically

#define LAYER_SPECIFIC_DIM(type) \
    int _DH = is_global_layer(type) ? DH : SWA_DH; \
    int _DQ = is_global_layer(type) ? DQ : SWA_DQ; \
    int _DK = is_global_layer(type) ? DK : SWA_DK; \
    int _DV = is_global_layer(type) ? DV : SWA_DV; \
    int _INTERMEDIATE_SIZE = (is_skip_layer(type) && enable_double_wide_mlp) ? INTERMEDIATE_SIZE * 2 : INTERMEDIATE_SIZE;

typedef struct {
    int D;
    int DH;
    int DQ;
    int DK;
    int DV;
    int SWA_DH;
    int SWA_DQ;
    int SWA_DK;
    int SWA_DV;
    int PLI_D;
    int INTERMEDIATE_SIZE;
    int NUM_ATTENTION_HEADS;
    int NUM_KEY_VALUE_HEADS;
    int SLIDING_WINDOW_SIZE;
    int VOCAB_SIZE_PADDED;
    bool enable_double_wide_mlp;
} gemma4e_seq_gen_parameters_t;

struct gemma4e_npu_sequence{
    typedef enum: int{
        NORMAL_DEQUANT = 0,
        UP_MATRIX      = 1,
        GATE_MATRIX    = 2
    } dequant_output_mode_t;

    typedef struct {
        // for decoding layer
        uint32_t l_qk_address;
        uint32_t l_kv_address;
        uint32_t swa_l_qk_address;
        uint32_t swa_l_kv_address;
        uint32_t proj_swa_address;
        uint32_t proj_skip_address;
        uint32_t rms_swa_address;
        uint32_t rms_skip_address;
        uint32_t rope_skip_kv_address;
        uint32_t swa_rope_skip_kv_address;
        uint32_t glu_skip_address;
        uint32_t lm_head_final_tune_address;
        // for others
        uint32_t place_holder_0;
    } rtp_address_book_t;

    static constexpr rtp_address_book_t e2b_rtp_addresses = {
        .l_qk_address = 57344,
        .l_kv_address = 14976,
        .swa_l_qk_address = 9216,
        .swa_l_kv_address = 40960,
        .proj_swa_address = 33280,
        .proj_skip_address = 49664,
        .rms_swa_address = 52224,
        .rms_skip_address = 25600,
        .rope_skip_kv_address = 33792,
        .swa_rope_skip_kv_address = 33280,
        .glu_skip_address = 34816,
        .lm_head_final_tune_address = 49152,
        .place_holder_0 = 0x20050625
    };

    static constexpr rtp_address_book_t e4b_rtp_addresses = {
        .l_qk_address = 57344,
        .l_kv_address = 14976,
        .swa_l_qk_address = 53248,
        .swa_l_kv_address = 57664,
        .proj_swa_address = 33280,
        .proj_skip_address = 49664,
        .rms_swa_address = 55328,
        .rms_skip_address = 55392,
        .rope_skip_kv_address = 34816,
        .swa_rope_skip_kv_address = 33792,
        .glu_skip_address = 30720,
        .lm_head_final_tune_address = 10240,
        .place_holder_0 = 0x20050625
    };
    /// @brief constexprs
    static constexpr npu_tiles xr_tile  = IT3;
    static constexpr npu_tiles out_tile = IT4;
    static constexpr npu_tiles attn_tile = IT2;
    static constexpr npu_tiles mvm_tiles[4] = {IT0, IT1, IT6, IT7};
    static constexpr npu_tiles rms_tile = CT03;
    static constexpr npu_tiles glu_tile = CT13;
    static constexpr npu_tiles rope_ct = CT23;
    static constexpr npu_tiles swa_rope_ct = CT33;
    static constexpr npu_tiles attn_qk_tile = CT02;
    static constexpr npu_tiles attn_kv_tile = CT12;
    static constexpr npu_tiles swa_attn_qk_tile = CT22;
    static constexpr npu_tiles swa_attn_kv_tile = CT32;
    static constexpr npu_tiles proj_tiles[] = {CT00, CT10, CT20, CT30,
                                               CT01, CT11, CT21, CT31,
                                               CT06, CT16, CT26, CT36,
                                               CT07, CT17, CT27, CT37
                                            };
    static constexpr npu_tiles pli_tile = CT05;

    static constexpr int x_arg_id = 0;
    static constexpr int proj_arg_id = 1;
    static constexpr int rms_arg_id = 2;
    static constexpr int rope_rms_arg_id = 3;
    static constexpr int kv_cache_arg_id = 4;

    static constexpr int L_CHUNK = 16;

    /// @brief dequant kernel RTP offset, identical for E2B and E4B.
    static constexpr uint32_t dequant_rtp_address = 54272;

    /// \brief The model description; owns every weight descriptor this generator addresses.
    /// \note Not owned. Set by set_desc() before any sequence is generated.
    gemma4e_desc* desc = nullptr;
    rtp_address_book_t rtp_addresses;

    uint32_t D;
    uint32_t DH;
    uint32_t DQ;
    uint32_t DK;
    uint32_t DV;
    uint32_t SWA_DH;
    uint32_t SWA_DQ;
    uint32_t SWA_DK;
    uint32_t SWA_DV;

    uint32_t MAX_L;
    uint32_t HIDDEN_SIZE;
    uint32_t INTERMEDIATE_SIZE;
    uint32_t PLI_D;
    uint32_t SLIDING_LENGTH;

    bool enable_double_wide_mlp;

    int num_attn_heads;
    int num_kv_heads;
    int num_kv_per_round;

    uint32_t rms_addr;

    int VOCAB_SIZE;

    gemma4e_npu_sequence(){}
    gemma4e_npu_sequence(gemma4e_seq_gen_parameters_t params, uint32_t MAX_L);
    ~gemma4e_npu_sequence();
    /// \brief Human-readable name of a layer kind, for debug output.
    static const char* layer_type_name(gemma4e_layer_type_t t) {
        switch (t) {
            case e_gemma4e_swa_layer:         return "swa";
            case e_gemma4e_global_layer:      return "global";
            case e_gemma4e_swa_layer_skip:    return "swa_skip";
            case e_gemma4e_global_layer_skip: return "global_skip";
            default:                          return "unknown";
        }
    }

    /// \brief Point the generator at the model description.
    /// \param desc the description whose weight descriptors name every DMA source
    void set_desc(gemma4e_desc* desc) {
        this->desc = desc;
        DEBUG_BLOCK(2,
            header_print("info", "Sequence generator bound to gemma4e_desc; proj layout (bytes):");
            for (int t = 0; t < 4; t++){
                gemma4e_layer_weight_def& L = desc->weight_desc(static_cast<gemma4e_layer_type_t>(t));
                std::cout << "  " << layer_type_name(static_cast<gemma4e_layer_type_t>(t))
                          << ": qkv=" << L.attn_qkv.offset
                          << ", o=" << L.attn_output.offset
                          << ", upgate=" << L.ffn_up_gate.offset
                          << ", down=" << L.ffn_down.offset
                          << ", pli_down=" << L.pli_down_proj.offset
                          << ", pli_gate=" << L.pli_gate_proj.offset
                          << ", pli_up=" << L.pli_up_proj.offset << std::endl;
            }
        )
    }

    /// \brief The weight layout of a layer kind.
    gemma4e_layer_weight_def& layer_weights(gemma4e_layer_type_t layer_type) {
        assert(desc != nullptr && "set_desc() must run before any sequence is generated");
        return desc->weight_desc(layer_type);
    }

    /// \brief A weight's start, in bf16 elements, which is how the DMA ports address it.
    static uint32_t weight_elem_offset(weight_desc_t& weight) {
        return (uint32_t)(weight.offset / sizeof(bf16));
    }

    void _send_hidden_states(npu_sequence* seq);
    void _send_rms_weights(npu_sequence* seq);
    void _send_rope_rms_weights(npu_sequence* seq, gemma4e_layer_type_t layer_type);
    void _receive_kv_cache(npu_sequence* seq, const int L, gemma4e_layer_type_t layer_type);
    void _move_kv_cache(npu_sequence* seq, const size_t L, gemma4e_layer_type_t layer_type);
    void _move_weights(npu_sequence* seq, weight_desc_t& weight);
    void _gen_pli_path_seq(npu_sequence* seq_ptr, gemma4e_layer_type_t layer_type);
    void gen_lm_head_seq(npu_sequence* seq, float final_scale);

    void gen_layer_seq(npu_sequence* seq, const uint32_t L, gemma4e_layer_type_t layer_type);
    void gen_mha_engine_seq(npu_sequence* seq, const uint32_t L_begin, const uint32_t L_end);
    void gen_swa_engine_seq(npu_sequence* seq, const uint32_t L_begin, const uint32_t L_end);
    void generate_dequant_seq(npu_sequence* seq_ptr, weight_desc_t& weight, dequant_output_mode_t output_mode);

    void set_max_length(const uint32_t MAX_L);
};
