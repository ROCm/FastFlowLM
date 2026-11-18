// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file hunyuan_model_def.hpp
/// \brief weight-shape definition for the `hunyuan-dense` architecture
/// \author FastFlowLM Team
/// \date 2026-09-01
/// \note Hy-MT2-1.8B is a plain Llama-shaped GQA decoder with exactly two
///       deviations (see the HunYuan V1 dense model card / HF modeling code):
///         * the per-head Q/K RMS norms run *after* RoPE, not before;
///         * the DynamicNTKAlpha rope base rescale is already folded into the
///           checkpoint's rope_theta, so build() only has to honour an explicit
///           `alpha` when one is still present in config.json.
///       Neither is visible here -- this file only lays weights out -- but the
///       host kernels in cpu_functions.hpp depend on both.
#ifndef __HUNYUAN_MODEL_DEF_HPP__
#define __HUNYUAN_MODEL_DEF_HPP__
#include <cmath>
#include <memory>
#include "lm_config.hpp"
#include "weight_desc.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"

/// @brief per-layer weight descriptors. Unlike the hybrid models there is only
///        one kind of layer here, so a single descriptor set describes every
///        one of them: format_name() takes the layer index at load time and the
///        device offsets are identical layer to layer.
struct hunyuan_layer_weight_def{
    weight_desc_t input_layer_norm;
    weight_desc_t post_attntion_norm;
    weight_desc_t attn_q;
    weight_desc_t attn_k;
    weight_desc_t attn_v;
    weight_desc_t attn_output;
    weight_desc_t attn_q_norm;
    weight_desc_t attn_k_norm;
    weight_desc_t ffn_up;
    weight_desc_t ffn_gate;
    weight_desc_t ffn_down;

    std::vector<weight_desc_t*> all() {
        return {
            &attn_q, &attn_k, &attn_v, &attn_output,
            &attn_q_norm, &attn_k_norm,
            &input_layer_norm, &post_attntion_norm,
            &ffn_up, &ffn_gate, &ffn_down
        };
    }
};

/// @brief  Owns the weight-shape definition for the hunyuan-dense model: the
///         dimensions derived from LM_Config, the per-layer weight descriptors,
///         and the aggregated buffer byte sizes.
struct hunyuan_desc {
    /// @brief the up/gate output slab the two projections are interleaved at.
    ///        Also the slice _exp_glu walks, so the two must agree.
    static constexpr int UP_GATE_OUTDIM_REORDEER = 512;

    // dimensions derived from config
    uint32_t num_hidden_layers = 0;
    uint32_t D = 0;
    uint32_t DH = 0;
    uint32_t DQ = 0;
    uint32_t DK = 0;
    uint32_t DV = 0;
    uint32_t num_heads = 0;
    uint32_t num_kv_heads = 0;
    uint32_t num_kv_groups = 0;
    uint32_t RoPE_DIM = 0;
    uint32_t intermediate_size = 0;
    uint32_t vocab_size = 0;
    uint32_t vocab_size_padded = 0;
    f32 rms_norm_eps = 1e-5f;
    f32 attn_scale = 0.0f;
    f32 rope_base = 0.0f;

    /// @brief 1 / base^(2j/DH), j in [0, DH/2). Built once in build(): the
    ///        "dynamic" rope_type this architecture declares is a static
    ///        NTK-style base rescale, never recomputed as the sequence grows.
    std::vector<f32> inv_freq;

    /// @brief one descriptor set, shared by all layers
    hunyuan_layer_weight_def layer_desc;
    /// @brief the output projection, alone in its own buffer (offset 0)
    weight_desc_t lm_head;

    // aggregated buffer byte sizes
    size_t total_proj_weights_byte_size = 0;
    size_t rms_weights_byte_size = 0;

    hunyuan_desc() {}
    hunyuan_desc(LM_Config config) { build(config); }

    inline size_t get_kv_cache_size(uint32_t MAX_L) const { return (size_t)MAX_L * (DK + DV); }
    /// @brief the lm_head is Q8_0, stored as two 32 x 128 halves per 32 x 256 chunk, each padded
    ///        to one q4_k chunk, so the projection cores read it through their q4_k buffers
    inline size_t lm_head_chunk_bytes() const { return 2 * get_quantization_byte_size(QXNX_ROW_BLOCK_SIZE * QXNX_COL_BLOCK_SIZE, flm_q4k); }
    inline size_t get_lm_head_w_size() const {
        return (size_t)vocab_size_padded / QXNX_ROW_BLOCK_SIZE * (D / QXNX_COL_BLOCK_SIZE) * lm_head_chunk_bytes();
    }

    inline uint32_t get_upgate_offset() { return layer_desc.ffn_up.offset; }
    inline uint32_t get_down_offset()   { return layer_desc.ffn_down.offset; }
    inline uint32_t get_attn_q_offset() { return layer_desc.attn_q.offset; }
    inline uint32_t get_attn_k_offset() { return layer_desc.attn_k.offset; }
    inline uint32_t get_attn_v_offset() { return layer_desc.attn_v.offset; }
    inline uint32_t get_attn_o_offset() { return layer_desc.attn_output.offset; }

    inline void build(LM_Config& config) {
        num_hidden_layers = config.get("num_hidden_layers");
        D = config.get("hidden_size");
        num_heads = config.get("num_attention_heads");
        num_kv_heads = config.get("num_key_value_heads", num_heads);
        // head_dim may be explicit and may not equal D / num_heads: do not assume.
        DH = config.get("head_dim", D / num_heads);
        DQ = DH * num_heads;
        DK = DH * num_kv_heads;
        DV = DH * num_kv_heads;
        num_kv_groups = num_heads / num_kv_heads;
        // full-head rotation, unlike the partial-rope models
        RoPE_DIM = DH;
        intermediate_size = config.get("intermediate_size");
        vocab_size = config.get("vocab_size");
        vocab_size_padded = (vocab_size + 1024 - 1) / 1024 * 1024;
        rms_norm_eps = config.get<f32>("rms_norm_eps", 1e-5f);
        attn_scale = 1.0f / std::sqrt((f32)DH);

        // --- rope base -------------------------------------------------------
        // Branch A (the Hunyuan path): rope_type "dynamic" with an explicit
        // alpha rescales the base once, at construction. Every shipped q4nx
        // checkpoint has that rescale already folded into rope_theta, so the
        // branch is a no-op there -- it is kept because a checkpoint converted
        // straight from HF still carries the raw theta and the alpha.
        f32 rope_theta = config.get<f32>("rope_theta", 10000.0f);
        const nlohmann::json& rope_params = config._json_config.contains("rope_parameters")
                                          ? config.sub("rope_parameters")
                                          : config.sub("rope_scaling");
        std::string rope_type = cfg_get<std::string>(rope_params, "rope_type", std::string(""));
        f32 alpha = cfg_get<f32>(rope_params, "alpha", 0.0f);
        rope_base = rope_theta;
        if (rope_type.find("dynamic") != std::string::npos && alpha > 0.0f) {
            rope_base = rope_theta * std::pow(alpha, (f32)DH / (f32)(DH - 2));
        }
        inv_freq.resize(DH / 2);
        for (uint32_t j = 0; j < DH / 2; j++) {
            inv_freq[j] = 1.0f / std::pow(rope_base, (f32)(2 * j) / (f32)DH);
        }

        DEBUG_BLOCK(1,
        std::cout << "================ hunyuan_desc ================" << std::endl;
        std::cout << std::left << std::setw(28) << "num_hidden_layers"  << " = " << num_hidden_layers << std::endl;
        std::cout << std::left << std::setw(28) << "D (hidden_size)"    << " = " << D << std::endl;
        std::cout << std::left << std::setw(28) << "DH (head_dim)"      << " = " << DH << std::endl;
        std::cout << std::left << std::setw(28) << "DQ"                 << " = " << DQ << std::endl;
        std::cout << std::left << std::setw(28) << "DK"                 << " = " << DK << std::endl;
        std::cout << std::left << std::setw(28) << "DV"                 << " = " << DV << std::endl;
        std::cout << std::left << std::setw(28) << "num_heads"          << " = " << num_heads << std::endl;
        std::cout << std::left << std::setw(28) << "num_kv_heads"       << " = " << num_kv_heads << std::endl;
        std::cout << std::left << std::setw(28) << "num_kv_groups"      << " = " << num_kv_groups << std::endl;
        std::cout << std::left << std::setw(28) << "intermediate_size"  << " = " << intermediate_size << std::endl;
        std::cout << std::left << std::setw(28) << "RoPE_DIM"           << " = " << RoPE_DIM << std::endl;
        std::cout << std::left << std::setw(28) << "rms_norm_eps"       << " = " << rms_norm_eps << std::endl;
        std::cout << std::left << std::setw(28) << "attn_scale"         << " = " << attn_scale << std::endl;
        std::cout << std::left << std::setw(28) << "rope_type"          << " = " << (rope_type.empty() ? "<none>" : rope_type) << std::endl;
        std::cout << std::left << std::setw(28) << "rope_theta"         << " = " << rope_theta << std::endl;
        std::cout << std::left << std::setw(28) << "rope_alpha"         << " = " << alpha << std::endl;
        std::cout << std::left << std::setw(28) << "rope_base"          << " = " << rope_base << std::endl;
        std::cout << std::left << std::setw(28) << "vocab_size"         << " = " << vocab_size << std::endl;
        std::cout << std::left << std::setw(28) << "vocab_size_padded"  << " = " << vocab_size_padded << std::endl;
        std::cout << "==============================================" << std::endl;
        )

        lm_head = weight_desc_t(flm_q80, {D, vocab_size_padded}, "lm_head.weight");
        // owns its buffer outright, so its offset is trivially 0
        lm_head.indp();

        layer_desc.attn_q            = weight_desc_t(flm_q4k, {D, DQ}, "model.layers.%d.self_attn.q_proj.weight");
        layer_desc.attn_k            = weight_desc_t(flm_q4k, {D, DK}, "model.layers.%d.self_attn.k_proj.weight");
        layer_desc.attn_v            = weight_desc_t(flm_q4k, {D, DV}, "model.layers.%d.self_attn.v_proj.weight");
        layer_desc.attn_output       = weight_desc_t(flm_q4k, {DQ, D}, "model.layers.%d.self_attn.o_proj.weight");
        layer_desc.attn_q_norm       = weight_desc_t(flm_bf16, {DH}, "model.layers.%d.self_attn.q_norm.weight");
        layer_desc.attn_k_norm       = weight_desc_t(flm_bf16, {DH}, "model.layers.%d.self_attn.k_norm.weight");
        layer_desc.input_layer_norm  = weight_desc_t(flm_bf16, {D}, "model.layers.%d.input_layernorm.weight");
        layer_desc.post_attntion_norm= weight_desc_t(flm_bf16, {D}, "model.layers.%d.post_attention_layernorm.weight");
        layer_desc.ffn_up            = weight_desc_t(flm_q4k, {D, intermediate_size}, "model.layers.%d.mlp.up_proj.weight");
        layer_desc.ffn_gate          = weight_desc_t(flm_q4k, {D, intermediate_size}, "model.layers.%d.mlp.gate_proj.weight");
        layer_desc.ffn_down          = weight_desc_t(flm_q4k, {intermediate_size, D}, "model.layers.%d.mlp.down_proj.weight");

        // The order of add_weight() calls *is* the device memory layout. up and
        // gate are adjacent because load_layer_weights interleaves them into one
        // slab; changing this order silently changes what every get_*_offset()
        // and every generated sequence reads.
        weight_container proj_weights_helper;
        proj_weights_helper.add_weight(layer_desc.ffn_up);
        proj_weights_helper.add_weight(layer_desc.ffn_gate);
        proj_weights_helper.add_weight(layer_desc.ffn_down);
        proj_weights_helper.add_weight(layer_desc.attn_k);
        proj_weights_helper.add_weight(layer_desc.attn_v);
        proj_weights_helper.add_weight(layer_desc.attn_q);
        proj_weights_helper.add_weight(layer_desc.attn_output);

        weight_container rms_weights_helper;
        rms_weights_helper.add_weight(layer_desc.input_layer_norm);
        rms_weights_helper.add_weight(layer_desc.post_attntion_norm);

        total_proj_weights_byte_size = proj_weights_helper.get_size();
        rms_weights_byte_size = rms_weights_helper.get_size();
    }

    /// @brief row-band interleave for the array's shared weight streams.
    /// @note  Source block (band b, col block c) lands at destination block
    ///        (b / V) * V * blocks_per_row + c * V + (b % V). The host matmul in
    ///        cpu_functions.hpp inverts exactly this, so the two backends read
    ///        the same bytes.
    void reorder_cpy(u8 *dst, buffer<u8> &src, const int col, flm_dtype_t dtype, const int vertical_blocks = 2)
    {
        assert(is_quantize(dtype));
        const int a_block_size = get_quantization_byte_size(QXNX_ROW_BLOCK_SIZE * QXNX_COL_BLOCK_SIZE, dtype);
        reorder_cpy_blocks(dst, src, col, a_block_size, vertical_blocks);
    }

    /// @brief the lm_head: every Q8_0 32 x 256 chunk becomes two 32 x 128 halves
    ///        ([128 bf16 scales][4096 int8] each), each padded to one q4_k chunk,
    ///        then the row-band interleave runs over the halves, so a band pair
    ///        reads [b0 h0][b1 h0][b0 h1][b1 h1]: the mem tile gives the first
    ///        4736 B of each 9472 B buffer to one core and the second to the other.
    void reorder_cpy_lm_head(u8 *dst, buffer<u8> &src)
    {
        const size_t chunk          = get_quantization_byte_size(QXNX_ROW_BLOCK_SIZE * QXNX_COL_BLOCK_SIZE, flm_q80);
        const size_t pair           = lm_head_chunk_bytes();
        const size_t padded_half    = pair / 2;
        const size_t scale_bytes    = QXNX_ROW_BLOCK_SIZE * (QXNX_COL_BLOCK_SIZE / GGML_GROUP_SIZE) * sizeof(bf16);
        const size_t half_scale     = scale_bytes / 2;
        const size_t half_q         = (chunk - scale_bytes) / 2;
        const size_t n_chunks       = src.size() / chunk;

        buffer<u8> expanded(n_chunks * pair);
        memset(expanded.data(), 0, expanded.size());
        for (size_t i = 0; i < n_chunks; i++) {
            const u8* s = src.data() + i * chunk;
            u8* d = expanded.data() + i * pair;
            for (int half = 0; half < 2; half++) {
                memcpy(d + half * padded_half, s + half * half_scale, half_scale);
                memcpy(d + half * padded_half + half_scale, s + scale_bytes + half * half_q, half_q);
            }
        }
        reorder_cpy_blocks(dst, expanded, 2 * D, (int)padded_half, 2);
    }

    void reorder_cpy_blocks(u8 *dst, buffer<u8> &src, const int col, const int a_block_size, const int vertical_blocks)
    {
        const int blocks_per_row = col / QXNX_COL_BLOCK_SIZE;
        const int rows = src.size() / a_block_size / blocks_per_row;

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

    /// @brief load one decoder layer into its device buffers
    /// @note  rope_weights holds, in order: cos[DH/2], sin[DH/2], q_norm[DH],
    ///        k_norm[DH]. The cos/sin half is filled per token by
    ///        _set_rope_rms_weights(); only the two norms are loaded here.
    void load_layer_weights(int layer_idx, Q4NX& q4nx, buffer<u8>& proj_weights, buffer<bf16>& rms_weights, buffer<bf16>& rope_weights){
        hunyuan_layer_weight_def& _weight_desc = this->layer_desc;
        u8* w_ptr_begin = proj_weights.data();
        // read q/k/v weights
        {
            LOG_VERBOSE(1, "Loading qkv weights");
            weight_desc_t& desc_q = _weight_desc.attn_q;
            weight_desc_t& desc_k = _weight_desc.attn_k;
            weight_desc_t& desc_v = _weight_desc.attn_v;
            buffer<u8> w_q;
            buffer<u8> w_k;
            buffer<u8> w_v;
            q4nx.load_weights(w_q, desc_q.format_name(layer_idx));
            q4nx.load_weights(w_k, desc_k.format_name(layer_idx));
            q4nx.load_weights(w_v, desc_v.format_name(layer_idx));
            reorder_cpy(w_ptr_begin + desc_q.offset, w_q, D, desc_q.dtype);
            reorder_cpy(w_ptr_begin + desc_k.offset, w_k, D, desc_k.dtype);
            reorder_cpy(w_ptr_begin + desc_v.offset, w_v, D, desc_v.dtype);
            desc_q.load();
            desc_k.load();
            desc_v.load();
        }
        // read o weights
        {
            LOG_VERBOSE(1, "Loading o weights");
            weight_desc_t& desc = _weight_desc.attn_output;
            buffer<u8> w_o;
            q4nx.load_weights(w_o, desc.format_name(layer_idx));
            reorder_cpy(w_ptr_begin + desc.offset, w_o, DQ, desc.dtype);
            desc.load();
        }
        // read up/gate weights, interleaved a slab at a time
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
            LOG_VERBOSE(1, "Loading down weights");
            weight_desc_t& desc = _weight_desc.ffn_down;
            buffer<u8> w_down;
            q4nx.load_weights(w_down, desc.format_name(layer_idx));
            reorder_cpy(w_ptr_begin + desc.offset, w_down, intermediate_size, desc.dtype);
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

            memcpy(rms_weights_begin + desc_pre.offset, w_rms_pre.data(), D * sizeof(bf16));
            memcpy(rms_weights_begin + desc_post.offset, w_rms_post.data(), D * sizeof(bf16));
            desc_pre.load();
            desc_post.load();
        }
        // read the per-head q/k norms, behind the cos/sin table
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

#endif // __HUNYUAN_MODEL_DEF_HPP__
