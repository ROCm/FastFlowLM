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
#include "weight_desc.hpp"


struct moe_attn_layer_weights{
public:
    weight_desc_t attn_k;
    weight_desc_t attn_k_norm;
    weight_desc_t input_layer_norm;
    weight_desc_t attn_q;
    weight_desc_t attn_q_norm;
    weight_desc_t attn_v;
    weight_desc_t ffn_down_exps;
    weight_desc_t ffn_down_shexp;
    weight_desc_t ffn_gate_exps;
    weight_desc_t ffn_gate_shexp;
    weight_desc_t ffn_up_exps;
    weight_desc_t ffn_up_shexp;
    weight_desc_t ffn_gate_inp;
    weight_desc_t post_attntion_norm;
    weight_desc_t attn_output;

    std::vector<weight_desc_t*> all() {
        return {
            &attn_k, &attn_k_norm, &input_layer_norm, &attn_q, &attn_q_norm,
            &attn_v, &ffn_down_exps, &ffn_down_shexp, &ffn_gate_exps,
            &ffn_gate_shexp, &ffn_up_exps, &ffn_up_shexp, &ffn_gate_inp,
            &post_attntion_norm, &attn_output,
        };
    }
};

class decoding_layer {
public:
    model_list::model_t model_type;
    LM_Config& lm_config;
    npu_app_manager* layer_app_mgr;

    npu_app layer_app;

    buffer<float> rope_weights;
    float rope_scaling;

    int xy_arg_idx = 0;
    int w_proj_arg_idx = 1;
    int w_rms_arg_idx = 2;
    int w_rope_rms_arg_idx = 3;
    int kv_cache_arg_idx = 4;

    buffer<bf16> hidden_states_buffer;
    buffer<u8> all_proj_weight_buffer;
    buffer<bf16> rms_weights_buffer;
    buffer<bf16> rope_rms_buffer;
    buffer<bf16> kv_cache;

    int current_context_length;
    int num_hidden_layers;
    moe_attn_layer_weights _weight_desc;
    weight_container proj_weights_helper;
    weight_container norm_exp_helper;

    void _gen_sequence(npu_sequence* seq, int L);
    void _set_rope_weights(int);

    void _send_hidden_states(npu_sequence* seq);
    void _send_rms_weights(npu_sequence* seq, size_t byte_offset);
    void _send_rope_rms_weights(npu_sequence* seq);
    void _receive_kv_cache(npu_sequence* seq, const int L);
    void _move_kv_cache(npu_sequence* seq, const size_t L);
    void _receive_attn_tap(npu_sequence* seq);
    void _read_rtp_addresses(const std::string& xclbin_path);
    void _move_weights_q80(npu_sequence* seq, const size_t Dout, const size_t Din, const size_t w_offset);
    // wait_last_round = false: the last round's tasks are pushed but not waited for (see _gen_sequence)
    void _move_weights_q4k(npu_sequence* seq, const size_t Dout, const size_t Din, const size_t w_offset, bool wait_last_round = true);
    void _wait_proj_weights(npu_sequence* seq);
    void _send_router_weights(npu_sequence* seq, const size_t NUM_expert, const size_t D, const size_t W_offset_in_byte);

    void setup_expert_q4k(npu_sequence* seq, int col, int bd_id, size_t const_w_byte_offset);
    // model specific config
    int DH;
    int D;
    int DQ;
    int DK;
    int DV;
    int RoPE_DIM;
    int num_kv_heads;
    bool has_qk_norm;
    int intermediate_size;
    int MAX_L;
    int num_experts;
    int num_activate_experts;
    bool attn_tap;
    bool last_call_completed = false;   // the last operator() call ended in COMPLETED
    int debug_stop_after;   // HY_STOP_AFTER, see _gen_sequence
    std::string debug_tap;  // HY_DEBUG_TAP, needs a DEBUG_TAP build, see _gen_sequence

    // KV cache layout: 4 blocks of MAX_L x (kv heads of the column * DH),
    //   [K col 0][K col 1][V col 0][V col 1]
    // i.e. all K, then all V at kv_cache_size / 2 (as FLM_DLL's hunyuan_npu and
    // qwen3_6_moe_npu); col 0 = heads 0, 1, col 1 = heads 2, 3, so every column
    // reads its K and V with one linear DMA. ../dense_decoding uses the same layout.
    // MAX_L 131072 (128K) = 256 MB per layer, one buffer per layer: DMA offsets
    // are 32-bit byte offsets, so one buffer for all 48 layers would overflow.
    size_t kv_cache_size;
    /// \brief index of a column's K (kv = 0) or V (kv = 1) block
    static constexpr int kv_block(int col, int kv) { return 2 * kv + col; }
    int kv_heads_per_col;
    size_t kv_col_row;    // elements per token per column, K or V
    size_t kv_col_half;   // elements of one K (or V) block of a column

    // npu configs

    static constexpr npu_tiles xyr_tile = IT2;
    static constexpr npu_tiles mvm_tiles[4] = {IT0, IT1, IT6, IT7};

    // attention: 2 columns x 2 CUs; CU i holds KV head i.
    // AIE (row, col) of the qk and kv tile of each CU, and its column's shim.
    static constexpr int attn_cus = 4;
    static constexpr int attn_cols = 2;
    static constexpr int attn_qk_rc[attn_cus][2] = {{2, 3}, {4, 3}, {2, 4}, {4, 4}};
    static constexpr int attn_kv_rc[attn_cus][2] = {{3, 3}, {5, 3}, {3, 4}, {5, 4}};
    static constexpr npu_tiles attn_shims[attn_cols] = {IT3, IT4};
    // RTP address of the context length in each qk / kv tile, from rtp_addresses.txt
    uint32_t l_qk_address[attn_cus];
    uint32_t l_kv_address[attn_cus];

    // router CT (row 4, col 5) and its fp32 expert bias, written with RTP writes each call
    static constexpr int router_rc[2] = {4, 5};
    uint32_t router_bias_address;
    uint32_t router_layer_address;   // RTP_router_layer: which layer a call is
    int layer_slot = 0;              // this object's layer (set by load_weights)
    buffer<float> router_bias;

    static constexpr npu_tiles proj_tiles[] = {CT00, CT10, CT20, CT30,
                                               CT01, CT11, CT21, CT31,
                                               CT06, CT16, CT26, CT36,
                                               CT07, CT17, CT27, CT37
                                            };
    int L_CHUNK = 16;

public:
    /// \brief constructor
    /// \param model_type the model type
    /// \param lm_config the lm config
    /// \param xclbin_mgr the xclbin manager
    /// \param context_length the context length
    decoding_layer(model_list::model_t, LM_Config&, npu_app_manager*, int context_length, bool attn_tap);

    /// \brief load weights
    /// \param q4nx_model the q4nx model
    void load_weights(Q4NX&, int layer_idx);
    void _load_moe_weights(Q4NX&, int layer_idx);

    /// \brief operator()
    /// \param input the hidden states
    /// \return the output buffer
    buffer<bf16> operator()(buffer<bf16>&);


    void reorder_cpy(u8 *dst, buffer<u8> &src, const int col, flm_dtype_t dtype, const int vertical_blocks = 2);
    void q4k_relayout_for_mmul(u8* data, size_t bytes);
    float bench(buffer<bf16>&, int calls, int ctx = 8);


    /// \brief set context length
    /// \param length the context length
    void set_context_length(int length);

    buffer<bf16> get_k_cache(int idx);
    buffer<bf16> get_v_cache(int idx);

    /// \brief long-context test: KV rows 0..n_rows-1 = row (j % src_rows) of
    ///        k_rows / v_rows ([src_rows, KV heads * DH], K after RoPE)
    void fill_kv(const bf16* k_rows, const bf16* v_rows, int src_rows, int n_rows);

    /// \brief attention output (before o_proj) of the last call; needs an ATTN_TAP=1 build
    buffer<bf16> get_attn_tap();

    /// \brief DEBUG: how many values reached each HY_DEBUG_TAP=x tap in the last call
    void report_debug_tap();

    void set_rope_frequency(buffer<float>& freqs);
};


#endif
