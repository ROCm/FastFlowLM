// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __DECODING_LAYER_HPP__
#define __DECODING_LAYER_HPP__

#include "models.h"
#include "lm_config.hpp"
#include "npu_utils/npu_utils.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"
#include "utils/utils.hpp"
#include "tensor_2d.hpp"
class decoding_layer {
private:
    model_list::model_t model_type;
    LM_Config& lm_config;
    npu_app_manager* layer_app_mgr;

    npu_app layer_app;
    npu_app lm_head;

    buffer<float> rope_weights;
    float rope_scaling;

    int x_arg_id = 0;
    int proj_arg_id = 1;
    int rms_arg_id = 2;
    int rope_rms_arg_id = 3;
    int kv_cache_arg_id = 4;

    int x_gropu_id = 3;
    int proj_group_id = 4;
    int rms_group_id = 5;
    int rope_rms_group_id = 6;
    int kv_cache_group_id = 7;

    buffer<bf16> hidden_states_buffer;
    buffer<u8> all_proj_weight_buffer;
    buffer<bf16> rms_weights_buffer;
    buffer<bf16> rope_rms_buffer;
    buffer<bf16> kv_cache;

    int current_context_length;
    int num_hidden_layers;

    void _gen_sequence(npu_sequence* seq, int L);
    void _set_rope_weights(int);

    void _send_hidden_states(npu_sequence* seq);
    void _send_rms_weights(npu_sequence* seq);
    void _send_rope_rms_weights(npu_sequence* seq);
    void _receive_kv_cache(npu_sequence* seq, const int L);
    void _move_kv_cache(npu_sequence* seq, const size_t L);
    void _move_weights(npu_sequence* seq, const size_t Dout, const size_t Din, const size_t w_offset);
    void _move_weights_q80(npu_sequence* seq, const size_t Dout, const size_t Din, const size_t w_offset);


    // model specific config
    int DH;
    int D;
    int DQ;
    int DK;
    int DV;
    int RoPE_DIM;
    int num_kv_heads;
    int num_kv_per_round;
    bool has_qk_norm;
    int intermediate_size;
    int MAX_L;

    size_t kv_cache_size;
    size_t w_qkv_offset;
    size_t w_gate_offset;
    size_t w_o_offset;
    size_t w_up_offset;
    size_t w_down_offset;

    uint32_t l_qk_address;
    uint32_t l_kv_address;
    uint32_t l_begin_mha_address;
    uint32_t l_end_mha_address;
    uint32_t kk_address;
    uint32_t rms_address;

    // npu configs

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

    int L_CHUNK = 16;

public:
    /// \brief constructor
    /// \param model_type the model type
    /// \param lm_config the lm config
    /// \param xclbin_mgr the xclbin manager
    /// \param context_length the context length
    decoding_layer(model_list::model_t, LM_Config&, npu_app_manager*, int);

    /// \brief load weights
    /// \param q4nx_model the q4nx model
    void load_weights(Q4NX&);

    /// \brief operator()
    /// \param input the hidden states
    /// \return the output buffer
    buffer<bf16> operator()(buffer<bf16>&);

    
    void reorder_cpy(u8 *dst, buffer<u8> &src, const int col, const int vertical_blocks = 2);
    void reorder_cpy_q80(u8 *dst, buffer<u8> &src, const int col, const int vertical_blocks = 2);
    float bench(buffer<bf16>&);

    
    /// \brief set context length
    /// \param length the context length
    void set_context_length(int length);

    buffer<bf16> get_k_cache(int idx);
    buffer<bf16> get_v_cache(int idx);

    void set_rope_frequency(buffer<float>& freqs);
};

#endif