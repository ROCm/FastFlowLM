// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file qwen3_5_omni_thinker_def.hpp
/// \brief qwen3_5_omni thinker text backbone config + weight descriptor.
/// \author FastFlowLM Team
/// \note Owns the config-derived parameters, per-layer weight descriptors and
///       weight loading for the thinker text backbone (mirrors qwen3_5vl_desc).
#pragma once
#include <memory>
#include "lm_config.hpp"
#include "weight_desc.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"

struct qwen3_5_omni_thinker_layer_weright_def{
    weight_desc_t input_layer_norm;
    weight_desc_t post_attntion_norm;
    weight_desc_t ffn_down;
    weight_desc_t ffn_gate;
    weight_desc_t ffn_up;
    virtual ~qwen3_5_omni_thinker_layer_weright_def() = default;
};

struct qwen3_5_omni_thinker_linear_layer : public qwen3_5_omni_thinker_layer_weright_def{
public:
    weight_desc_t linear_attn_qkv;
    weight_desc_t attn_gate;
    weight_desc_t ssm_a;
    weight_desc_t ssm_alpha;
    weight_desc_t ssm_beta;
    weight_desc_t ssm_conv1d;
    weight_desc_t ssm_dt;
    weight_desc_t ssm_norm;
    weight_desc_t ssm_out;
    weight_desc_t ssm_alpha_cpu;
    weight_desc_t ssm_beta_cpu;

    std::vector<weight_desc_t*> all() {
        return {
            &linear_attn_qkv, &attn_gate, &ssm_a, &ssm_alpha, &ssm_beta,
            &ssm_conv1d, &ssm_dt, &ssm_norm, &ssm_out, &input_layer_norm,
            &post_attntion_norm, &ffn_down,
            &ffn_gate, &ffn_up
        };
    }
};

struct qwen3_5_omni_thinker_attn_layer : public qwen3_5_omni_thinker_layer_weright_def{
public:
    weight_desc_t attn_k;
    weight_desc_t attn_k_norm;
    weight_desc_t input_layer_norm;
    weight_desc_t attn_q;
    weight_desc_t attn_q_norm;
    weight_desc_t attn_v;
    weight_desc_t post_attntion_norm;
    weight_desc_t attn_output;
    weight_desc_t attn_gate;

    std::vector<weight_desc_t*> all() {
        return {
            &attn_k, &attn_k_norm, &input_layer_norm, &attn_q, &attn_q_norm,
            &attn_v, &ffn_down,
            &ffn_gate, &ffn_up,
            &post_attntion_norm, &attn_output, &attn_gate,
        };
    }
};

/// @brief thinker text backbone (thinker_config.text_config)
struct qwen3_5_omni_thinker_desc {
    /// RoPE inverse frequencies, inv_freq[j] = rope_theta^(-2j/RoPE_DIM).
    /// Computed in build() from thinker_config.text_config.rope_theta.
    static constexpr uint32_t MAX_ROPE_DIM = 64;
    f32 inv_freq[MAX_ROPE_DIM / 2] = {};

    int full_attention_interval = 4;

    // token ids (thinker_config)
    int image_token_id = 0;
    int audio_token_id = 0;

    // dimensions derived from config
    uint32_t num_hidden_layers = 0;
    uint32_t D = 0;
    uint32_t DH = 0;
    uint32_t DQ = 0;
    uint32_t DK = 0;
    uint32_t DV = 0;
    uint32_t num_heads = 0;
    uint32_t num_kv_heads = 0;
    uint32_t Linear_DQ = 0;
    uint32_t Linear_DK = 0;
    uint32_t Linear_DV = 0;
    uint32_t Linear_DQKV = 0;
    uint32_t linear_num_qk_heads = 0;
    uint32_t linear_num_v_heads = 0;
    uint32_t linear_num_v_heads_padded = 0;
    uint32_t linear_qk_head_dim = 0;
    uint32_t linear_v_head_dim = 0;
    uint32_t conv_kernel_size = 0;
    uint32_t RoPE_DIM = 0;
    uint32_t intermediate_size = 0;
    uint32_t vocab_size = 0;
    uint32_t vocab_size_padded = 0;
    f32 rope_theta = 0.0f;
    f32 rms_norm_eps = 0.0f;

    // representative weight descriptors (all layers of a kind share identical
    // offsets, so only one linear layer and one attn layer are needed)
    qwen3_5_omni_thinker_linear_layer linear_weight_desc;
    qwen3_5_omni_thinker_attn_layer   attn_weight_desc;
    // aggregated buffer byte sizes (one representative layer of each kind)
    size_t attn_total_proj_weights_byte_size = 0;
    size_t attn_rms_weights_byte_size = 0;
    size_t linear_total_proj_weights_byte_size = 0;
    size_t linear_rms_weights_byte_size = 0;
    size_t linear_ssm_weights_byte_size = 0;

    qwen3_5_omni_thinker_desc() {}
    qwen3_5_omni_thinker_desc(LM_Config config) { build(config); }

    inline bool is_attn_layer(int layer_idx) {
        return (layer_idx % full_attention_interval) == (full_attention_interval - 1);
    }

    inline size_t get_kv_cache_size(uint32_t MAX_L) const { return (size_t)MAX_L * (DK + DV); }
    inline size_t get_linear_cache_size() { return (size_t)(conv_kernel_size - 1) * (Linear_DQ + Linear_DK + Linear_DV) * sizeof(bf16) + (size_t)linear_num_v_heads * linear_v_head_dim * linear_qk_head_dim * sizeof(float); }
    inline size_t get_lm_head_w_size() { return (size_t)vocab_size_padded * D + (size_t)vocab_size_padded * D / 32 * 2; }
    inline size_t get_conv_w_size() { return (size_t)conv_kernel_size * (Linear_DQ + Linear_DK + Linear_DV) * sizeof(bf16); }
    inline size_t get_s_offset() { return (size_t)(conv_kernel_size - 1) * (Linear_DQ + Linear_DK + Linear_DV); }

    inline uint32_t get_upgate_offset() {return linear_weight_desc.ffn_up.offset; }
    inline uint32_t get_down_offset() {return linear_weight_desc.ffn_down.offset; }

    inline uint32_t get_attn_q_offset() {return attn_weight_desc.attn_q.offset; }
    inline uint32_t get_attn_k_offset() {return attn_weight_desc.attn_k.offset; }
    inline uint32_t get_attn_v_offset() {return attn_weight_desc.attn_v.offset; }
    inline uint32_t get_attn_g_offset() {return attn_weight_desc.attn_gate.offset; }
    inline uint32_t get_attn_o_offset() {return attn_weight_desc.attn_output.offset; }

    inline uint32_t get_linear_qkv_offset() {return linear_weight_desc.linear_attn_qkv.offset; }
    inline uint32_t get_linear_g_offset() {return linear_weight_desc.attn_gate.offset; }
    inline uint32_t get_linear_o_offset() {return linear_weight_desc.ssm_out.offset; }
    inline uint32_t get_linear_ssm_norm_offset() {return linear_weight_desc.ssm_norm.offset; }
    inline uint32_t get_linear_alpha_offset() {return linear_weight_desc.ssm_alpha.offset; }
    inline uint32_t get_linear_beta_offset() {return linear_weight_desc.ssm_beta.offset; }
    inline uint32_t get_linear_A_offset() {return linear_weight_desc.ssm_a.offset; }
    inline uint32_t get_linear_dt_offset() {return linear_weight_desc.ssm_dt.offset; }

    inline void build(LM_Config& config) {
        const nlohmann::json &thinker_config = config.sub("thinker_config");
        const nlohmann::json &text_config = cfg_sub(thinker_config, "text_config");

        JSON_GET(image_token_id, thinker_config, "image_token_id", 0, int);
        JSON_GET(audio_token_id, thinker_config, "audio_token_id", 0, int);

        uint32_t head_dim = 0, num_attention_heads = 0, num_key_value_heads = 0;
        f32 partial_rotary_factor = 0.0f;
        JSON_GET(num_hidden_layers,   text_config, "num_hidden_layers",   0, uint32_t);
        JSON_GET(D,                   text_config, "hidden_size",         0, uint32_t);
        JSON_GET(head_dim,            text_config, "head_dim",            0, uint32_t);
        JSON_GET(num_attention_heads, text_config, "num_attention_heads", 0, uint32_t);
        JSON_GET(num_key_value_heads, text_config, "num_key_value_heads", 0, uint32_t);
        JSON_GET(linear_qk_head_dim,  text_config, "linear_key_head_dim",   0, uint32_t);
        JSON_GET(linear_v_head_dim,   text_config, "linear_value_head_dim", 0, uint32_t);
        JSON_GET(linear_num_qk_heads, text_config, "linear_num_key_heads",   0, uint32_t);
        JSON_GET(linear_num_v_heads,  text_config, "linear_num_value_heads", 0, uint32_t);
        JSON_GET(conv_kernel_size,    text_config, "linear_conv_kernel_dim", 0, uint32_t);
        JSON_GET(intermediate_size,   text_config, "intermediate_size",   0, uint32_t);
        JSON_GET(full_attention_interval, text_config, "full_attention_interval", 4, int);
        JSON_GET(vocab_size,          text_config, "vocab_size",          0, uint32_t);
        JSON_GET(rope_theta,          text_config, "rope_theta",          0.0f, f32);
        JSON_GET(rms_norm_eps,        text_config, "rms_norm_eps",        0.0f, f32);
        JSON_GET(partial_rotary_factor, text_config, "partial_rotary_factor", 0.25f, f32);

        DH = head_dim;
        DQ = head_dim * num_attention_heads;
        DK = head_dim * num_key_value_heads;
        DV = head_dim * num_key_value_heads;
        num_heads = num_attention_heads;
        num_kv_heads = num_key_value_heads;
        linear_num_v_heads_padded = (linear_num_v_heads + 32 - 1) / 32 * 32;
        Linear_DQ = linear_qk_head_dim * linear_num_qk_heads;
        Linear_DK = linear_qk_head_dim * linear_num_qk_heads;
        Linear_DV = linear_v_head_dim * linear_num_v_heads;
        Linear_DQKV = Linear_DQ + Linear_DK + Linear_DV;
        RoPE_DIM = head_dim * partial_rotary_factor;
        vocab_size_padded = (vocab_size + 1024 - 1) / 1024 * 1024;

        assert(RoPE_DIM <= MAX_ROPE_DIM);
        for (uint32_t j = 0; j < RoPE_DIM / 2; j++){
            inv_freq[j] = (f32)std::pow((double)rope_theta, -2.0 * (double)j / (double)RoPE_DIM);
        }

        DEBUG_BLOCK(1,
        std::cout << "============ qwen3_5_omni_thinker_desc ============" << std::endl;
        std::cout << std::left << std::setw(28) << "num_hidden_layers"        << " = " << num_hidden_layers << std::endl;
        std::cout << std::left << std::setw(28) << "D (hidden_size)"          << " = " << D << std::endl;
        std::cout << std::left << std::setw(28) << "DH (head_dim)"            << " = " << DH << std::endl;
        std::cout << std::left << std::setw(28) << "DQ"                       << " = " << DQ << std::endl;
        std::cout << std::left << std::setw(28) << "DK"                       << " = " << DK << std::endl;
        std::cout << std::left << std::setw(28) << "DV"                       << " = " << DV << std::endl;
        std::cout << std::left << std::setw(28) << "num_heads"                << " = " << num_heads << std::endl;
        std::cout << std::left << std::setw(28) << "num_kv_heads"             << " = " << num_kv_heads << std::endl;
        std::cout << std::left << std::setw(28) << "linear_qk_head_dim"       << " = " << linear_qk_head_dim << std::endl;
        std::cout << std::left << std::setw(28) << "linear_v_head_dim"        << " = " << linear_v_head_dim << std::endl;
        std::cout << std::left << std::setw(28) << "linear_num_qk_heads"      << " = " << linear_num_qk_heads << std::endl;
        std::cout << std::left << std::setw(28) << "linear_num_v_heads"       << " = " << linear_num_v_heads << std::endl;
        std::cout << std::left << std::setw(28) << "linear_num_v_heads_padded"<< " = " << linear_num_v_heads_padded << std::endl;
        std::cout << std::left << std::setw(28) << "conv_kernel_size"         << " = " << conv_kernel_size << std::endl;
        std::cout << std::left << std::setw(28) << "Linear_DQ"                << " = " << Linear_DQ << std::endl;
        std::cout << std::left << std::setw(28) << "Linear_DK"                << " = " << Linear_DK << std::endl;
        std::cout << std::left << std::setw(28) << "Linear_DV"                << " = " << Linear_DV << std::endl;
        std::cout << std::left << std::setw(28) << "intermediate_size"        << " = " << intermediate_size << std::endl;
        std::cout << std::left << std::setw(28) << "RoPE_DIM"                 << " = " << RoPE_DIM << std::endl;
        std::cout << std::left << std::setw(28) << "full_attention_interval"  << " = " << full_attention_interval << std::endl;
        std::cout << std::left << std::setw(28) << "vocab_size"               << " = " << vocab_size << std::endl;
        std::cout << std::left << std::setw(28) << "vocab_size_padded"        << " = " << vocab_size_padded << std::endl;
        std::cout << std::left << std::setw(28) << "rope_theta"               << " = " << rope_theta << std::endl;
        std::cout << std::left << std::setw(28) << "rms_norm_eps"             << " = " << rms_norm_eps << std::endl;
        std::cout << std::left << std::setw(28) << "image_token_id"           << " = " << image_token_id << std::endl;
        std::cout << std::left << std::setw(28) << "audio_token_id"           << " = " << audio_token_id << std::endl;
        std::cout << "==================================================" << std::endl;
        )
        weight_container attn_proj_weights_helper;
        weight_container attn_rms_weights_helper;

        weight_container linear_proj_weights_helper;
        weight_container linear_rms_weights_helper;
        weight_container linear_ssm_weights_helper;

        {
            qwen3_5_omni_thinker_attn_layer& layer = attn_weight_desc;
            layer.attn_k = weight_desc_t(flm_q41, {D, DK}, "model.layers.%d.self_attn.k_proj.weight");
            layer.attn_k_norm = weight_desc_t(flm_bf16, {DH}, "model.layers.%d.self_attn.k_norm.weight");
            layer.input_layer_norm = weight_desc_t(flm_bf16, {D}, "model.layers.%d.input_layernorm.weight");
            layer.attn_q = weight_desc_t(flm_q41, {D, DQ}, "model.layers.%d.self_attn.q_proj.weight");
            layer.attn_q_norm = weight_desc_t(flm_bf16, {DH}, "model.layers.%d.self_attn.q_norm.weight");
            layer.attn_v = weight_desc_t(flm_q41, {D, DV}, "model.layers.%d.self_attn.v_proj.weight");
            layer.ffn_down = weight_desc_t(flm_q41, {intermediate_size, D}, "model.layers.%d.mlp.down_proj.weight");
            layer.ffn_up = weight_desc_t(flm_q41, {D, intermediate_size}, "model.layers.%d.mlp.up_proj.weight");
            layer.ffn_gate = weight_desc_t(flm_q41, {D, intermediate_size}, "model.layers.%d.mlp.gate_proj.weight");
            layer.post_attntion_norm = weight_desc_t(flm_bf16, {D}, "model.layers.%d.post_attention_layernorm.weight");
            layer.attn_output = weight_desc_t(flm_q41, {DQ, D}, "model.layers.%d.self_attn.o_proj.weight");
            layer.attn_gate = weight_desc_t(flm_q41, {D, DQ}, "model.layers.%d.self_attn.q_proj.weight"); // special actully merged with q

            attn_proj_weights_helper.add_weight(layer.ffn_up);
            attn_proj_weights_helper.add_weight(layer.ffn_gate);
            attn_proj_weights_helper.add_weight(layer.ffn_down);
            attn_proj_weights_helper.add_weight(layer.attn_q);
            attn_proj_weights_helper.add_weight(layer.attn_k);
            attn_proj_weights_helper.add_weight(layer.attn_v);
            attn_proj_weights_helper.add_weight(layer.attn_gate);
            attn_proj_weights_helper.add_weight(layer.attn_output);

            attn_rms_weights_helper.add_weight(layer.input_layer_norm);
            attn_rms_weights_helper.add_weight(layer.post_attntion_norm);
        }

        {
            qwen3_5_omni_thinker_linear_layer& layer = linear_weight_desc;
            layer.linear_attn_qkv = weight_desc_t(flm_q41, {D, Linear_DQ + Linear_DK + Linear_DV}, "model.layers.%d.linear_attn.qkv_proj.weight");
            layer.attn_gate = weight_desc_t(flm_q41, {D, Linear_DV}, "model.layers.%d.self_attn.gate_proj.weight");
            layer.ssm_a = weight_desc_t(flm_f32, {linear_num_v_heads_padded}, "model.layers.%d.linear_attn.ssm_a");
            layer.ssm_alpha = weight_desc_t(flm_q41, {D, linear_num_v_heads_padded}, "model.layers.%d.linear_attn.ssm_alpha_proj.weight");
            layer.ssm_beta = weight_desc_t(flm_q41, {D, linear_num_v_heads_padded}, "model.layers.%d.linear_attn.ssm_beta_proj.weight");
            layer.ssm_conv1d = weight_desc_t(flm_bf16, {Linear_DV + Linear_DQ + Linear_DK, conv_kernel_size}, "model.layers.%d.linear_attn.ssm_conv1d.weight");
            layer.ssm_dt = weight_desc_t(flm_f32, {linear_num_v_heads_padded}, "model.layers.%d.linear_attn.ssm_dt.bias");
            layer.ssm_norm = weight_desc_t(flm_bf16, {linear_v_head_dim}, "model.layers.%d.linear_attn.ssm_norm.weight");
            layer.ssm_out = weight_desc_t(flm_q41, {Linear_DV, D}, "model.layers.%d.linear_attn.ssm_out_proj.weight");
            layer.ssm_alpha_cpu = weight_desc_t(flm_bf16, {D, linear_num_v_heads}, "model.layers.%d.linear_attn.ssm_alpha_proj.bf16.weight");
            layer.ssm_beta_cpu = weight_desc_t(flm_bf16, {D, linear_num_v_heads}, "model.layers.%d.linear_attn.ssm_beta_proj.bf16.weight");
            layer.post_attntion_norm = weight_desc_t(flm_bf16, {D}, "model.layers.%d.post_attention_layernorm.weight");
            layer.input_layer_norm = weight_desc_t(flm_bf16, {D}, "model.layers.%d.input_layernorm.weight");
            layer.ffn_down = weight_desc_t(flm_q41, {intermediate_size, D}, "model.layers.%d.mlp.down_proj.weight");
            layer.ffn_up = weight_desc_t(flm_q41, {D, intermediate_size}, "model.layers.%d.mlp.up_proj.weight");
            layer.ffn_gate = weight_desc_t(flm_q41, {D, intermediate_size}, "model.layers.%d.mlp.gate_proj.weight");

            linear_proj_weights_helper.add_weight(layer.ffn_up);
            linear_proj_weights_helper.add_weight(layer.ffn_gate);
            linear_proj_weights_helper.add_weight(layer.ffn_down);
            linear_proj_weights_helper.add_weight(layer.linear_attn_qkv);
            linear_proj_weights_helper.add_weight(layer.attn_gate);
            linear_proj_weights_helper.add_weight(layer.ssm_out);

            linear_rms_weights_helper.add_weight(layer.input_layer_norm);
            linear_rms_weights_helper.add_weight(layer.post_attntion_norm);

            linear_ssm_weights_helper.add_weight(layer.ssm_conv1d);
            linear_ssm_weights_helper.add_weight(layer.ssm_norm);
            linear_ssm_weights_helper.add_weight(layer.ssm_a);
            linear_ssm_weights_helper.add_weight(layer.ssm_dt);
            linear_ssm_weights_helper.add_weight(layer.ssm_alpha);
            linear_ssm_weights_helper.add_weight(layer.ssm_beta);
        }

        attn_total_proj_weights_byte_size = attn_proj_weights_helper.get_size();
        attn_rms_weights_byte_size = attn_rms_weights_helper.get_size();

        linear_total_proj_weights_byte_size = linear_proj_weights_helper.get_size();
        linear_rms_weights_byte_size = linear_rms_weights_helper.get_size();
        linear_ssm_weights_byte_size = linear_ssm_weights_helper.get_size();
    }

    void reorder_cpy(u8 *dst, buffer<u8> &src, const int col, flm_dtype_t dtype, const int vertical_blocks = 2)
    {
        assert(is_quantize(dtype));
        const int a_block_size = get_quantization_byte_size(QXNX_ROW_BLOCK_SIZE * QXNX_COL_BLOCK_SIZE, dtype);

        const int blocks_per_row = col / QXNX_COL_BLOCK_SIZE;
        const int rows = src.size() / a_block_size / blocks_per_row;

        // Step 1: vertical block reorder at a_block_size granularity
        u8 *dst_ptr = dst;
        std::vector<u8 *> src_ptr(vertical_blocks);
        for (int i = 0; i < vertical_blocks; i++)
            src_ptr[i] = src.data() + i * a_block_size * blocks_per_row;

        for (int r = 0; r < rows; r += vertical_blocks)
        {
            for (int c = 0; c < blocks_per_row; c++)
            {
                for (int i = 0; i < vertical_blocks; i++)
                {
                    memcpy(dst_ptr, src_ptr[i], a_block_size);
                    dst_ptr += a_block_size;
                    src_ptr[i] += a_block_size;
                }
            }
            for (int i = 0; i < vertical_blocks; i++)
            {
                src_ptr[i] += (vertical_blocks - 1) * a_block_size * blocks_per_row;
                if (src_ptr[i] + a_block_size * blocks_per_row > src.end())
                    src_ptr[i] = src.data(); // useless padding
            }
        }
    }

    void load_linear_weights(int layer_idx, Q4NX& q4nx, buffer<u8>& proj_weights, buffer<bf16>& rms_weights, buffer<bf16>& ssm_weights, buffer<bf16>& alpha_proj_cpu, buffer<bf16>& beta_proj_cpu){
        constexpr int GLU_SLICE = 1024;
        constexpr int UP_GATE_OUTDIM_REORDEER = 512;
        qwen3_5_omni_thinker_linear_layer& _weight_desc = this->linear_weight_desc;
        //constexpr int GLU_OUTDIM = 512;
        u8* w_ptr_begin = proj_weights.data();
        {
            LOG_VERBOSE(1, "Loading qkv weights");
            weight_desc_t& desc = _weight_desc.linear_attn_qkv;
            u8* w_ptr = w_ptr_begin + desc.offset;
            buffer<u8> w;
            q4nx.load_weights(w, desc.format_name(layer_idx));
            LOG_VERBOSE(1, "Copying qkv weights to buffer");
            reorder_cpy(w_ptr, w, D, desc.dtype);
            desc.load();
        }
        // read gate weights
        {
            LOG_VERBOSE(1, "Loading gate weights");
            weight_desc_t& desc = _weight_desc.attn_gate;
            u8* w_ptr = w_ptr_begin + desc.offset;
            buffer<u8> w;
            q4nx.load_weights(w, desc.format_name(layer_idx));
            LOG_VERBOSE(1, "Copying qkv weights to buffer");
            reorder_cpy(w_ptr, w, D, desc.dtype);
            desc.load();
        }
        // read o weights
        {
            LOG_VERBOSE(1, "Loading o weights");
            weight_desc_t& desc = _weight_desc.ssm_out;
            u8* w_ptr = w_ptr_begin + desc.offset;
            buffer<u8> w;
            q4nx.load_weights(w, desc.format_name(layer_idx));
            assert(w_ptr + w.size() <= proj_weights.end());
            reorder_cpy(w_ptr, w, Linear_DV, desc.dtype);
            desc.load();
        }
        // read up weights
        // read up/gate weights
        {
            LOG_VERBOSE(1, "Loading shared up/gate weights");
            weight_desc_t& desc_up = _weight_desc.ffn_up;
            weight_desc_t& desc_gate = _weight_desc.ffn_gate;
            buffer<u8> w_up;
            buffer<u8> w_gate;
            q4nx.load_weights(w_up, desc_up.format_name(layer_idx));
            q4nx.load_weights(w_gate, desc_gate.format_name(layer_idx));

            u8* w_ptr = w_ptr_begin + desc_up.offset;

            assert(desc_up.shape[1] % UP_GATE_OUTDIM_REORDEER == 0);
            assert(desc_gate.shape[1] % UP_GATE_OUTDIM_REORDEER == 0);

            int slices = desc_up.shape[1] / UP_GATE_OUTDIM_REORDEER;
            size_t slice_size = desc_up.get_size() / slices;
            for (int i = 0; i < slices; i++) {
                buffer<u8> up_slice = buffer<u8>(w_up.data() + i * slice_size, slice_size);
                buffer<u8> gate_slice = buffer<u8>(w_gate.data() + i * slice_size, slice_size);
                reorder_cpy(w_ptr, up_slice, D, _weight_desc.ffn_up.dtype);
                w_ptr += slice_size;
                reorder_cpy(w_ptr, gate_slice, D, _weight_desc.ffn_gate.dtype);
                w_ptr += slice_size;
            }
            desc_up.load();
            desc_gate.load();
        }
        // read down weights
        {
            LOG_VERBOSE(1, "Loading down weights");
            weight_desc_t& desc = _weight_desc.ffn_down;
            u8* w_ptr = w_ptr_begin + desc.offset;
            buffer<u8> w;
            q4nx.load_weights(w, desc.format_name(layer_idx));
            reorder_cpy(w_ptr, w, intermediate_size, desc.dtype);
            desc.load();
        }

        // read rms weights
        {
            LOG_VERBOSE(1, "Loading rms weights");
            buffer<bf16> w_rms_pre;
            buffer<bf16> w_rms_post;

            u8* rms_weights_begin = (u8*)rms_weights.data();
            q4nx.load_weights(w_rms_pre, _weight_desc.input_layer_norm.format_name(layer_idx));
            q4nx.load_weights(w_rms_post, _weight_desc.post_attntion_norm.format_name(layer_idx));

            // w_rms_post.memset(1);
            memcpy(rms_weights_begin + _weight_desc.input_layer_norm.offset, w_rms_pre.data(), D * sizeof(bf16));
            memcpy(rms_weights_begin + _weight_desc.post_attntion_norm.offset, w_rms_post.data(), D * sizeof(bf16));
            _weight_desc.input_layer_norm.load();
            _weight_desc.post_attntion_norm.load();
        }

        u8* other_weights_begin = (u8*)ssm_weights.data();
        {
            // load conv_weights
            LOG_VERBOSE(1, "Loading ssm conv weights");
            weight_desc_t& desc = _weight_desc.ssm_conv1d;
            u8* w_ptr = other_weights_begin + desc.offset;
            buffer<bf16> conv_weights;
            q4nx.load_weights(conv_weights, desc.format_name(layer_idx));
            memcpy(w_ptr, conv_weights.data(), conv_weights.size() * sizeof(bf16));
            desc.load();
        }

        {
            // load ssm_norm_weights
            LOG_VERBOSE(1, "Loading ssm norm weights");
            weight_desc_t& desc = _weight_desc.ssm_norm;
            u8* w_ptr = other_weights_begin + desc.offset;
            buffer<bf16> norm_weights;
            q4nx.load_weights(norm_weights, desc.format_name(layer_idx));
            memcpy(w_ptr, norm_weights.data(), linear_v_head_dim * sizeof(bf16));
            desc.load();
        }
        {
            // load ssm a, dt
            LOG_VERBOSE(1, "Loading ssm a, dt weights");
            weight_desc_t& desc_a = _weight_desc.ssm_a;
            weight_desc_t& desc_dt = _weight_desc.ssm_dt;
            buffer<float> a;
            buffer<float> dt;

            q4nx.load_weights(a, desc_a.format_name(layer_idx));
            q4nx.load_weights(dt, desc_dt.format_name(layer_idx));
            memcpy(other_weights_begin + desc_a.offset, a.data(), linear_num_v_heads * sizeof(float));
            memcpy(other_weights_begin + desc_dt.offset, dt.data(), linear_num_v_heads * sizeof(float));

            desc_a.load();
            desc_dt.load();
        }
        {
            // load ssm alpha, beta
            LOG_VERBOSE(1, "Loading ssm alpha, beta weights");
            weight_desc_t& desc_alpha = _weight_desc.ssm_alpha;
            weight_desc_t& desc_beta = _weight_desc.ssm_beta;
            buffer<bf16> alpha;
            buffer<bf16> beta;

            q4nx.load_weights(alpha, desc_alpha.format_name(layer_idx));
            q4nx.load_weights(beta, desc_beta.format_name(layer_idx));

            memcpy(other_weights_begin + desc_alpha.offset, alpha.data(), desc_alpha.get_size());
            memcpy(other_weights_begin + desc_beta.offset, beta.data(), desc_beta.get_size());

            desc_alpha.load();
            desc_beta.load();
        }
        {
            // load cpu (bf16) ssm alpha, beta for host-side computation
            LOG_VERBOSE(1, "Loading cpu ssm alpha, beta weights");
            weight_desc_t& desc_alpha_cpu = _weight_desc.ssm_alpha_cpu;
            weight_desc_t& desc_beta_cpu = _weight_desc.ssm_beta_cpu;
            buffer<bf16> alpha;
            buffer<bf16> beta;

            q4nx.load_weights(alpha, desc_alpha_cpu.format_name(layer_idx));
            q4nx.load_weights(beta, desc_beta_cpu.format_name(layer_idx));

            memcpy(alpha_proj_cpu.data(), alpha.data(), desc_alpha_cpu.get_size());
            memcpy(beta_proj_cpu.data(), beta.data(), desc_beta_cpu.get_size());

            desc_alpha_cpu.load();
            desc_beta_cpu.load();
        }

        proj_weights.sync_to_device();
        rms_weights.sync_to_device();
        ssm_weights.sync_to_device();
    }

    void load_attn_weights(int layer_idx, Q4NX& q4nx, buffer<u8>& proj_weights, buffer<bf16>& rms_weights, buffer<bf16>& rope_weights){
        constexpr int GLU_SLICE = 1024;
        constexpr int UP_GATE_OUTDIM_REORDEER = 512;
        qwen3_5_omni_thinker_attn_layer& _weight_desc = this->attn_weight_desc;
        u8* w_ptr_begin = proj_weights.data();
        {
            LOG_VERBOSE(1, "Loading qkv weights");
            weight_desc_t& desc_q = _weight_desc.attn_q;
            weight_desc_t& desc_k = _weight_desc.attn_k;
            weight_desc_t& desc_v = _weight_desc.attn_v;
            weight_desc_t& desc_gate = _weight_desc.attn_gate;
            u8* w_ptr = w_ptr_begin + desc_q.offset;
            buffer<u8> w_q;
            buffer<u8> w_k;
            buffer<u8> w_v;
            q4nx.load_weights(w_q, desc_q.format_name(layer_idx));
            q4nx.load_weights(w_k, desc_k.format_name(layer_idx));
            q4nx.load_weights(w_v, desc_v.format_name(layer_idx));
            LOG_VERBOSE(1, "Copying qkv weights to buffer");
            size_t w_q_gate_size = w_q.size();
            buffer<u8> w_q_q(w_q.data(),w_q_gate_size / 2);
            buffer<u8> w_gate(w_q.data() + w_q_gate_size / 2, w_q_gate_size / 2);
            reorder_cpy(w_ptr, w_q_q, D, desc_q.dtype);
            w_ptr += desc_q.get_size();
            reorder_cpy(w_ptr, w_k, D, desc_k.dtype);
            w_ptr += desc_k.get_size();
            reorder_cpy(w_ptr, w_v, D, desc_v.dtype);
            w_ptr += desc_v.get_size();
            reorder_cpy(w_ptr, w_gate, D, desc_gate.dtype);
            desc_q.load();
            desc_k.load();
            desc_v.load();
            desc_gate.load();
        }
        // read o weights
        {
            LOG_VERBOSE(1, "Loading o weights");
            weight_desc_t& desc = _weight_desc.attn_output;
            u8* w_ptr = w_ptr_begin + desc.offset;
            buffer<u8> w_o;
            q4nx.load_weights(w_o, desc.format_name(layer_idx));
            reorder_cpy(w_ptr, w_o, DQ, desc.dtype);
            desc.load();
        }
        // read up/gate weights
        {
            LOG_VERBOSE(1, "Loading shared up/gate weights");
            weight_desc_t& desc_up = _weight_desc.ffn_up;
            weight_desc_t& desc_gate = _weight_desc.ffn_gate;
            u8* w_ptr = w_ptr_begin + desc_up.offset;
            buffer<u8> w_up;
            buffer<u8> w_gate;
            q4nx.load_weights(w_up, desc_up.format_name(layer_idx));
            q4nx.load_weights(w_gate, desc_gate.format_name(layer_idx));

            assert(desc_up.shape[1] % UP_GATE_OUTDIM_REORDEER == 0);
            assert(desc_gate.shape[1] % UP_GATE_OUTDIM_REORDEER == 0);

            int slices = desc_up.shape[1] / UP_GATE_OUTDIM_REORDEER;

            size_t slice_size = desc_up.get_size() / slices;
            for (int i = 0; i < slices; i++) {
                buffer<u8> up_slice = buffer<u8>(w_up.data() + i * slice_size, slice_size);
                buffer<u8> gate_slice = buffer<u8>(w_gate.data() + i * slice_size, slice_size);
                reorder_cpy(w_ptr, up_slice, D, desc_up.dtype);
                w_ptr += slice_size;
                reorder_cpy(w_ptr, gate_slice, D, desc_gate.dtype);
                w_ptr += slice_size;
            }
            desc_up.load();
            desc_gate.load();
        }
        // read down weights
        {
            LOG_VERBOSE(1, "Loading shared down weights");
            weight_desc_t& desc = _weight_desc.ffn_down;
            u8* w_ptr = w_ptr_begin + desc.offset;
            buffer<u8> w_down;
            q4nx.load_weights(w_down, desc.format_name(layer_idx));
            reorder_cpy(w_ptr, w_down, intermediate_size, desc.dtype);
            desc.load();
        }
        // read rms weights
        {
            LOG_VERBOSE(1, "Loading rms weights");
            weight_desc_t& desc_pre = _weight_desc.input_layer_norm;
            weight_desc_t& desc_post = _weight_desc.post_attntion_norm;
            buffer<bf16> w_rms_pre;
            buffer<bf16> w_rms_post;

            u8* rms_weights_begin = (u8*)rms_weights.data();
            q4nx.load_weights(w_rms_pre, desc_pre.format_name(layer_idx));
            q4nx.load_weights(w_rms_post, desc_post.format_name(layer_idx));

            // w_rms_post.memset(1);
            memcpy(rms_weights_begin + desc_pre.offset, w_rms_pre.data(), D * sizeof(bf16));
            memcpy(rms_weights_begin + desc_post.offset, w_rms_post.data(), D * sizeof(bf16));
            desc_pre.load();
            desc_post.load();
        }
        {
            LOG_VERBOSE(1, "Loading attention norm weights");
            weight_desc_t& desc_q_norm = _weight_desc.attn_q_norm;
            weight_desc_t& desc_k_norm = _weight_desc.attn_k_norm;
            buffer<bf16> q_attn_norm;
            buffer<bf16> k_attn_norm;
            q4nx.load_weights(q_attn_norm, desc_q_norm.format_name(layer_idx));
            q4nx.load_weights(k_attn_norm, desc_k_norm.format_name(layer_idx));
            memcpy(rope_weights.data() + RoPE_DIM, q_attn_norm.data(), DH * sizeof(bf16));
            memcpy(rope_weights.data() + RoPE_DIM + DH, k_attn_norm.data(), DH * sizeof(bf16));
            desc_q_norm.load();
            desc_k_norm.load();
        }

        proj_weights.sync_to_device();
        rms_weights.sync_to_device();
        rope_weights.sync_to_device();
    }
};
