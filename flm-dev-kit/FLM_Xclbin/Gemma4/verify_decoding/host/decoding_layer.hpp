// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#ifndef __DECODING_LAYER_HPP__
#define __DECODING_LAYER_HPP__

#include "models.h"
#include "lm_config.hpp"
#include "npu_utils/npu_utils.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"
#include "utils/utils.hpp"
#include "tensor_2d.hpp"
#include "embedding.hpp"
#ifndef L_TOKENS
#define L_TOKENS 1
#endif
#include <chrono>
#include <string>
#include <memory>

class decoding_layer {
public:
    /// rtp_addresses.txt written by the xclbin build (S6.1); empty = use the compiled-in constants
    static inline std::string rtp_file = "";
    /// dispatch timeout in ms (a hang must never look like a pass)
    static inline int dispatch_timeout_ms = 20000;
    static constexpr int SWA_WINDOW = 512;
    double last_dispatch_us = 0;
    ert_cmd_state last_state = ERT_CMD_STATE_NEW;
    void _load_rtp_file();
    /// read K (plane 0) or V (plane 1) row of absolute position pos from the cache (ring for SWA layers), after sync_from_device
    buffer<bf16> kv_row(int plane, int pos);
    int kv_rows() const { return is_swa_layer ? SWA_WINDOW : MAX_L; }
    int ring_slot(int pos) const { return is_swa_layer ? pos % SWA_WINDOW : pos; }

    model_list::model_t model_type;
    LM_Config& lm_config;
    npu_app_manager* layer_app_mgr;

    npu_app layer_app;
    npu_app lm_head;

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

    int full_attn_period;

    buffer<bf16> hidden_states_buffer;
    buffer<u8> all_proj_weight_buffer;
    buffer<bf16> rms_weights_buffer;
    buffer<bf16> rope_rms_buffer;
    buffer<bf16> kv_cache;
    
    // The embedding tables are the same for every layer (3.5 GB with the per layer table): one shared copy per process, loaded once.
    static inline std::shared_ptr<embedding_module> s_token_embed, s_pli_embed;
    static inline bool s_embed_loaded = false;
    std::shared_ptr<embedding_module> token_embed;
    std::shared_ptr<embedding_module> pli_embed;

    int current_context_length;
    int num_hidden_layers;
    // model specific config
    int DH;
    int D;
    int DQ;
    int DK;
    int DV;
    int hidden_size_per_layer_input;
    int num_kv_heads;
    int num_kv_per_round;
    int num_kv_shared_layers;
    int kv_caches_required;
    int layer_idx;
    int final_kv_cache_layer;
    bool is_swa_layer;
    bool shall_skip_kv;
    bool has_qk_norm;
    int intermediate_size;
    int MAX_L;
    int tb_size;
    bool use_double_wide_mlp;

    size_t kv_cache_size;
    size_t w_qkv_offset;
    size_t w_o_offset;
    size_t w_up_offset;
    size_t w_down_offset;
    size_t w_per_layer_input_proj_offset;
    size_t w_inp_gate_offset;
    size_t w_inp_up_offset;

    uint32_t l_qk_address;
    uint32_t l_kv_address;
    uint32_t swa_l_qk_address;
    uint32_t swa_l_kv_address;
    uint32_t proj_swa_address;
    uint32_t proj_skip_address;
    uint32_t kk_address;
    uint32_t rms_swa_address;
    uint32_t rms_skip_address;
    uint32_t rope_skip_kv_address;
    uint32_t swa_rope_skip_kv_address;
    uint32_t pli_scale_address;
    uint32_t glu_skip_address;
    float layer_scale;

    // npu configs

    static constexpr npu_tiles xyr_tile = IT3;
    static constexpr npu_tiles logits_out_tile = IT3;
    static constexpr npu_tiles attn_tile = IT2;
    static constexpr npu_tiles mvm_tiles[4] = {IT0, IT1, IT6, IT7};
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
    static constexpr npu_tiles rms_tile = CT03;
    static constexpr npu_tiles gate_tile = CT23;
    static constexpr npu_tiles pli_tile = CT05;
    static constexpr npu_tiles glu_tile = CT13;

    int L_CHUNK = 16;

    bool is_sliding_window_layer (int layer_idx) {return (layer_idx % full_attn_period) < (full_attn_period - 1);}
    bool skip_kv (int layer_idx) {return layer_idx >= (num_hidden_layers - num_kv_shared_layers);}
    int last_kv_cache_layer() {
        if (kv_caches_required == 0) return 0;
        if (is_swa_layer) {
            for (int i = kv_caches_required - 1; i >= 0; i--) {
                if (is_sliding_window_layer(i)) {
                    return i;
                }
            }
        }
        else {
            for (int i = kv_caches_required - 1; i >= 0; i--) {
                if (!is_sliding_window_layer(i)) {
                    return i;
                }
            }
        }
    }
    void _gen_sequence(npu_sequence* seq, int L);
    void _set_rope_weights(int);

    void _send_hidden_states(npu_sequence* seq);
    void _send_rms_weights(npu_sequence* seq);
    void _send_rope_rms_weights(npu_sequence* seq);
    void _receive_kv_cache(npu_sequence* seq, const int L);
    void _move_kv_cache(npu_sequence* seq, const size_t L);
    void _move_weights(npu_sequence* seq, const size_t Dout, const size_t Din, const size_t w_offset);

    void _per_layer_input_path(npu_sequence* seq);


    /// \brief constructor
    /// \param model_type the model type
    /// \param lm_config the lm config
    /// \param xclbin_mgr the xclbin manager
    /// \param layer_idx the layer index
    /// \param context_length the context length
    decoding_layer(model_list::model_t, LM_Config&, npu_app_manager*, int, int);

    /// \brief load weights
    /// \param q4nx_model the q4nx model
    void load_weights(Q4NX&);

    /// \brief operator()
    /// \param input the hidden states
    /// \return the output buffer
    buffer<bf16> operator()(buffer<bf16>& x, int idx);

    
    void reorder_cpy(u8 *dst, buffer<u8> &src, const int col, const int vertical_blocks = 2);
    float bench(buffer<bf16>&);

    buffer<bf16> embedding(int idx);
    /// \brief set context length
    /// \param length the context length
    void set_context_length(int length);

    buffer<bf16> get_k_cache(int idx);
    buffer<bf16> get_v_cache(int idx);

    void load_kv_cache(buffer<bf16>& k_cache, buffer<bf16>& v_cache, int L);

    void set_rope_frequency(buffer<float>& freqs);
};

#endif