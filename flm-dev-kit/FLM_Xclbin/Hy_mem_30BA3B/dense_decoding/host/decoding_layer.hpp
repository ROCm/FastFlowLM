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


struct dense_attn_layer_weights{
public:
    weight_desc_t attn_k;
    weight_desc_t attn_k_norm;
    weight_desc_t input_layer_norm;
    weight_desc_t attn_q;
    weight_desc_t attn_q_norm;
    weight_desc_t attn_v;
    weight_desc_t ffn_up;
    weight_desc_t ffn_gate;
    weight_desc_t ffn_down;
    weight_desc_t post_attntion_norm;
    weight_desc_t attn_output;

    std::vector<weight_desc_t*> all() {
        return {
            &attn_k, &attn_k_norm, &input_layer_norm, &attn_q, &attn_q_norm,
            &attn_v, &ffn_up, &ffn_gate, &ffn_down,
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
    dense_attn_layer_weights _weight_desc;
    weight_container proj_weights_helper;
    weight_container norm_helper;

    void _gen_sequence(npu_sequence* seq, int L);
    void _set_rope_weights(int);

    void _send_hidden_states(npu_sequence* seq);
    void _send_rms_weights(npu_sequence* seq);
    void _send_rope_rms_weights(npu_sequence* seq);
    void _receive_kv_cache(npu_sequence* seq, const int L);
    void _move_kv_cache(npu_sequence* seq, const size_t L);
    void _read_rtp_addresses(const std::string& xclbin_path);
    void _move_weights_q4k(npu_sequence* seq, const size_t Dout, const size_t Din, const size_t w_offset);
    void _wait_proj_weights(npu_sequence* seq);

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
    bool last_call_completed = false;   // the last operator() call ended in COMPLETED
    int debug_stop_after;   // HY_STOP_AFTER, see _gen_sequence

    // KV cache layout: 4 blocks of MAX_L x (kv heads of the column * DH),
    //   [K col 0][K col 1][V col 0][V col 1]
    // i.e. all K, then all V at kv_cache_size / 2 (as FLM_DLL's hunyuan_npu and
    // qwen3_6_moe_npu); col 0 = heads 0, 1 on shim 3, col 1 = heads 2, 3 on
    // shim 4, so every column reads its K and V with one linear DMA each. The
    // same layout as ../moe_decoding, so the engine treats all 48 layers' caches alike.
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
    // AIE (row, col) of the qk and kv tile of each CU, and the column shims.
    static constexpr int attn_cus = 4;
    static constexpr int attn_cols = 2;
    static constexpr int attn_qk_rc[attn_cus][2] = {{2, 3}, {4, 3}, {2, 4}, {4, 4}};
    static constexpr int attn_kv_rc[attn_cus][2] = {{3, 3}, {5, 3}, {3, 4}, {5, 4}};
    static constexpr npu_tiles attn_shims[attn_cols] = {IT3, IT4};
    // RoPE writes the new K row through shim 3 and the new V row through shim 4
    static constexpr npu_tiles kv_write_shim[2] = {IT3, IT4};
    // RTP address of the context length in each qk / kv tile, from rtp_addresses.txt
    uint32_t l_qk_address[attn_cus];
    uint32_t l_kv_address[attn_cus];

    // The 16 proj cores and the rms core take an IS_ATTN RTP (1 = decoder layer,
    // 0 = LM head) and wait on lock 6 for it every call.
    static constexpr int proj_rc[16][2] = {{2, 0}, {3, 0}, {4, 0}, {5, 0},
                                           {2, 1}, {3, 1}, {4, 1}, {5, 1},
                                           {2, 6}, {3, 6}, {4, 6}, {5, 6},
                                           {2, 7}, {3, 7}, {4, 7}, {5, 7}};
    static constexpr int rms_rc[2] = {2, 2};
    uint32_t proj_is_attn_address[16];
    uint32_t rms_is_attn_address;
    int L_CHUNK = 16;

public:
    /// \brief constructor
    /// \param model_type the model type
    /// \param lm_config the lm config
    /// \param xclbin_mgr the xclbin manager
    /// \param context_length the context length
    decoding_layer(model_list::model_t, LM_Config&, npu_app_manager*, int context_length);

    /// \brief load weights
    /// \param q4nx_model the q4nx model
    void load_weights(Q4NX&, int layer_idx);
    void _load_mlp_weights(Q4NX&, int layer_idx);

    /// \brief operator()
    /// \param input the hidden states
    /// \return the output buffer
    buffer<bf16> operator()(buffer<bf16>&);


    void reorder_cpy(u8 *dst, buffer<u8> &src, const int col, flm_dtype_t dtype, const int vertical_blocks = 2);
    float bench(buffer<bf16>&, int calls, int ctx = 8);


    /// \brief set context length
    /// \param length the context length
    void set_context_length(int length);

    buffer<bf16> get_k_cache(int idx);
    buffer<bf16> get_v_cache(int idx);

    /// \brief long-context test: KV rows 0..n_rows-1 = row (j % src_rows) of
    ///        k_rows / v_rows ([src_rows, KV heads * DH], K after RoPE)
    void fill_kv(const bf16* k_rows, const bf16* v_rows, int src_rows, int n_rows);

    /// \brief DEBUG: with HY_STOP_AFTER set, how much of the new K/V row arrived
    void report_debug_tap();

    void set_rope_frequency(buffer<float>& freqs);
};


#endif
