// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __QWEN3VL_MODEL_DEF_HPP__
#define __QWEN3VL_MODEL_DEF_HPP__
#include <memory>
#include <cmath>
#include <stdexcept>
#include "lm_config.hpp"
#include "weight_desc.hpp"
#include "modules/mha.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"

/// @brief Weight descriptors of one qwen3vl decoder layer.
/// @note  The declaration order in all() mirrors the physical layout inside the
///        per-layer proj_weights buffer, which the dequant sequences index by
///        absolute byte offset. Do not reorder.
struct qwen3vl_layer_weight_def {
    weight_desc_t attn_q;
    weight_desc_t attn_k;
    weight_desc_t attn_v;
    weight_desc_t attn_output;
    weight_desc_t ffn_up;
    weight_desc_t ffn_gate;
    weight_desc_t ffn_down;
    weight_desc_t input_layer_norm;
    weight_desc_t post_attntion_norm;
    /// @brief the rope cos/sin table shares the q_norm/k_norm buffer and occupies
    ///        its first DH entries; it has no name and is never loaded from a file.
    weight_desc_t rope_cache;
    weight_desc_t attn_q_norm;
    weight_desc_t attn_k_norm;

    std::vector<weight_desc_t*> all() {
        return {
            &attn_q, &attn_k, &attn_v, &attn_output,
            &ffn_up, &ffn_gate, &ffn_down,
            &input_layer_norm, &post_attntion_norm,
            &rope_cache, &attn_q_norm, &attn_k_norm
        };
    }
};

/// @brief Owns the weight-shape definition for the qwen3vl model: the model
///        dimensions derived from LM_Config, the per-layer weight descriptors,
///        the aggregated buffer byte sizes, and the weight loading/reordering.
struct qwen3vl_desc {
    static constexpr f32 inv_freq[64] = {
        1.0000e+00, 7.8583e-01, 6.1753e-01, 4.8527e-01, 3.8134e-01, 2.9967e-01,
        2.3549e-01, 1.8505e-01, 1.4542e-01, 1.1428e-01, 8.9802e-02, 7.0569e-02,
        5.5455e-02, 4.3578e-02, 3.4245e-02, 2.6911e-02, 2.1147e-02, 1.6618e-02,
        1.3059e-02, 1.0262e-02, 8.0644e-03, 6.3372e-03, 4.9800e-03, 3.9134e-03,
        3.0753e-03, 2.4167e-03, 1.8991e-03, 1.4924e-03, 1.1727e-03, 9.2157e-04,
        7.2420e-04, 5.6910e-04, 4.4721e-04, 3.5143e-04, 2.7617e-04, 2.1702e-04,
        1.7054e-04, 1.3402e-04, 1.0531e-04, 8.2759e-05, 6.5034e-05, 5.1106e-05,
        4.0161e-05, 3.1559e-05, 2.4800e-05, 1.9489e-05, 1.5315e-05, 1.2035e-05,
        9.4574e-06, 7.4319e-06, 5.8402e-06, 4.5894e-06, 3.6065e-06, 2.8341e-06,
        2.2271e-06, 1.7501e-06, 1.3753e-06, 1.0808e-06, 8.4929e-07, 6.6740e-07,
        5.2446e-07, 4.1214e-07, 3.2387e-07, 2.5451e-07
    };

    /// @brief up and gate are interleaved in the weight buffer, one slice of
    ///        this many output columns at a time.
    static constexpr int UP_GATE_OUTDIM_REORDER = 512;

    /// @brief rms epsilon, both for the layer norms and for the per-head q/k norms.
    static constexpr f32 rms_eps = 1e-6f;

    // dimensions derived from config
    uint32_t num_hidden_layers = 0;
    uint32_t D = 0;
    uint32_t DH = 0;
    uint32_t DQ = 0;
    uint32_t DK = 0;
    uint32_t DV = 0;
    uint32_t num_heads = 0;
    uint32_t num_kv_heads = 0;
    uint32_t intermediate_size = 0;
    uint32_t vocab_size = 0;
    uint32_t vocab_size_padded = 0;

    /// @brief attention kernel flavour, picked by intermediate size.
    mha_type_t mha_type = mha_d128_q4;
    /// @brief smallest context this model is willing to run with.
    uint32_t min_max_length = 2048;
    /// @brief the attn overlay is built for eight kv heads regardless of config.
    static constexpr uint32_t mha_kv_heads = 8;

    // per-layer weight descriptors
    std::vector<std::unique_ptr<qwen3vl_layer_weight_def>> weight_desc;
    weight_desc_t lm_head;
    weight_desc_t final_norm;

    // aggregated buffer byte sizes (one representative layer)
    size_t total_proj_weights_byte_size = 0;
    size_t rms_weights_byte_size = 0;
    size_t rope_rms_weights_byte_size = 0;

    qwen3vl_desc() {}
    qwen3vl_desc(LM_Config config) { build(config); }

    inline size_t get_kv_cache_size(uint32_t MAX_L) const { return (size_t)MAX_L * (DK + DV); }
    inline size_t get_lm_head_w_size() { return lm_head.get_size(); }

    inline uint32_t get_attn_q_offset() { return weight_desc[0]->attn_q.offset; }
    inline uint32_t get_attn_k_offset() { return weight_desc[0]->attn_k.offset; }
    inline uint32_t get_attn_v_offset() { return weight_desc[0]->attn_v.offset; }
    inline uint32_t get_attn_o_offset() { return weight_desc[0]->attn_output.offset; }
    inline uint32_t get_upgate_offset() { return weight_desc[0]->ffn_up.offset; }
    inline uint32_t get_down_offset()   { return weight_desc[0]->ffn_down.offset; }

    /// @brief byte offset of the kvq band, which the dequant kernel treats as one
    ///        contiguous [D, DK + DV + DQ] matrix. o is dequantized separately.
    inline uint32_t get_qkv_offset() { return get_attn_k_offset(); }
    inline uint32_t get_qkv_out_dim() { return DK + DV + DQ; }

    inline void build(LM_Config& config) {
        num_hidden_layers = config.get("num_hidden_layers");
        D = config.get("hidden_size");
        DH = config.get("head_dim");
        num_heads = config.get("num_attention_heads");
        num_kv_heads = config.get("num_key_value_heads");
        DQ = DH * num_heads;
        DK = DH * num_kv_heads;
        DV = DH * num_kv_heads;
        intermediate_size = config.get("intermediate_size");
        vocab_size = config.get("vocab_size");
        vocab_size_padded = (vocab_size + D - 1) / D * D;

        // The attention kernel and the minimum context are keyed off the
        // intermediate size, which is what distinguishes the checkpoints.
        if (intermediate_size == 3072 || intermediate_size == 6144) {
            mha_type = mha_d128_q2;
            min_max_length = 8192;
        }
        else if (intermediate_size == 9728 || intermediate_size == 12288) {
            mha_type = mha_d128_q4;
            min_max_length = 4096;
        }
        else {
            throw std::runtime_error("Unsupported intermediate size: " + std::to_string(intermediate_size));
        }

        DEBUG_BLOCK(1,
        std::cout << "================= qwen3vl_desc ===============" << std::endl;
        std::cout << std::left << std::setw(28) << "num_hidden_layers" << " = " << num_hidden_layers << std::endl;
        std::cout << std::left << std::setw(28) << "D (hidden_size)"   << " = " << D << std::endl;
        std::cout << std::left << std::setw(28) << "DH (head_dim)"     << " = " << DH << std::endl;
        std::cout << std::left << std::setw(28) << "DQ"                << " = " << DQ << std::endl;
        std::cout << std::left << std::setw(28) << "DK"                << " = " << DK << std::endl;
        std::cout << std::left << std::setw(28) << "DV"                << " = " << DV << std::endl;
        std::cout << std::left << std::setw(28) << "num_heads"         << " = " << num_heads << std::endl;
        std::cout << std::left << std::setw(28) << "num_kv_heads"      << " = " << num_kv_heads << std::endl;
        std::cout << std::left << std::setw(28) << "intermediate_size" << " = " << intermediate_size << std::endl;
        std::cout << std::left << std::setw(28) << "vocab_size"        << " = " << vocab_size << std::endl;
        std::cout << std::left << std::setw(28) << "vocab_size_padded" << " = " << vocab_size_padded << std::endl;
        std::cout << "==============================================" << std::endl;
        )

        lm_head = weight_desc_t(flm_q41, {D, vocab_size_padded}, "lm_head.weight");
        final_norm = weight_desc_t(flm_bf16, {D}, "model.norm.weight");
        // both own their buffer outright, so their offset is trivially 0
        lm_head.indp();
        final_norm.indp();

        weight_desc.resize(num_hidden_layers);

        weight_container proj_weights_helper;
        weight_container rms_weights_helper;
        weight_container rope_rms_weights_helper;
        bool is_helper_initialized = false;

        for (size_t i = 0; i < num_hidden_layers; i++) {
            auto layer = std::make_unique<qwen3vl_layer_weight_def>();
            layer->attn_q = weight_desc_t(flm_q41, {D, DQ}, "model.layers.%d.self_attn.q_proj.weight");
            layer->attn_k = weight_desc_t(flm_q41, {D, DK}, "model.layers.%d.self_attn.k_proj.weight");
            layer->attn_v = weight_desc_t(flm_q41, {D, DV}, "model.layers.%d.self_attn.v_proj.weight");
            layer->attn_output = weight_desc_t(flm_q41, {DQ, D}, "model.layers.%d.self_attn.o_proj.weight");
            layer->ffn_up = weight_desc_t(flm_q41, {D, intermediate_size}, "model.layers.%d.mlp.up_proj.weight");
            layer->ffn_gate = weight_desc_t(flm_q41, {D, intermediate_size}, "model.layers.%d.mlp.gate_proj.weight");
            layer->ffn_down = weight_desc_t(flm_q41, {intermediate_size, D}, "model.layers.%d.mlp.down_proj.weight");
            layer->input_layer_norm = weight_desc_t(flm_bf16, {D}, "model.layers.%d.input_layernorm.weight");
            layer->post_attntion_norm = weight_desc_t(flm_bf16, {D}, "model.layers.%d.post_attention_layernorm.weight");
            layer->rope_cache = weight_desc_t(flm_bf16, {DH}, "");
            layer->attn_q_norm = weight_desc_t(flm_bf16, {DH}, "model.layers.%d.self_attn.q_norm.weight");
            layer->attn_k_norm = weight_desc_t(flm_bf16, {DH}, "model.layers.%d.self_attn.k_norm.weight");

            if (!is_helper_initialized) {
                // The add order fixes the byte offsets: the dequant kernels read
                // q/k/v as one contiguous band, then o, then up|gate, then down.
                // K and V come before Q so the decode layer streams the KV half
                // of the fused projection to the RoPE tile first: the kv cache
                // write then overlaps the (much larger) Q projection instead of
                // waiting behind it. See qwen3vl_flash_sequence::_receive_kv_cache
                // and FLM_Xclbin/generic_decoding_layer/kernels/rope.cc.
                proj_weights_helper.add_weight(layer->attn_k);
                proj_weights_helper.add_weight(layer->attn_v);
                proj_weights_helper.add_weight(layer->attn_q);
                proj_weights_helper.add_weight(layer->attn_output);
                proj_weights_helper.add_weight(layer->ffn_up);
                proj_weights_helper.add_weight(layer->ffn_gate);
                proj_weights_helper.add_weight(layer->ffn_down);

                rms_weights_helper.add_weight(layer->input_layer_norm);
                rms_weights_helper.add_weight(layer->post_attntion_norm);

                // the rope table comes first, then the two per-head norms
                rope_rms_weights_helper.add_weight(layer->rope_cache);
                rope_rms_weights_helper.add_weight(layer->attn_q_norm);
                rope_rms_weights_helper.add_weight(layer->attn_k_norm);

                is_helper_initialized = true;
            }
            else {
                // every layer shares one layout; copy the offsets of layer 0
                auto& ref = *weight_desc[0];
                auto ref_all = ref.all();
                auto cur_all = layer->all();
                for (size_t w = 0; w < cur_all.size(); w++) {
                    cur_all[w]->offset = ref_all[w]->offset;
                    cur_all[w]->added = ref_all[w]->added;
                }
            }
            weight_desc[i] = std::move(layer);
        }

        total_proj_weights_byte_size = proj_weights_helper.get_size();
        rms_weights_byte_size = rms_weights_helper.get_size();
        rope_rms_weights_byte_size = rope_rms_weights_helper.get_size();
    }

    /// @brief Interleaves rows of a quantized weight in groups of `vertical_blocks`
    ///        so that each column of the xclbin reads a contiguous descriptor.
    void reorder_cpy(u8* dst, buffer<u8>& src, const int col, flm_dtype_t dtype, const int vertical_blocks = 2) {
        assert(is_quantize(dtype));
        const int a_block_size = get_quantization_byte_size(QXNX_ROW_BLOCK_SIZE * QXNX_COL_BLOCK_SIZE, dtype);

        const int blocks_per_row = col / QXNX_COL_BLOCK_SIZE;
        const int rows = src.size() / a_block_size / blocks_per_row;

        u8* dst_ptr = dst;
        std::vector<u8*> src_ptr(vertical_blocks);
        for (int i = 0; i < vertical_blocks; i++) {
            src_ptr[i] = src.data() + i * a_block_size * blocks_per_row;
        }
        for (int r = 0; r < rows; r += vertical_blocks) {
            for (int c = 0; c < blocks_per_row; c++) {
                for (int i = 0; i < vertical_blocks; i++) {
                    memcpy(dst_ptr, src_ptr[i], a_block_size);
                    dst_ptr += a_block_size;
                    src_ptr[i] += a_block_size;
                }
            }
            for (int i = 0; i < vertical_blocks; i++) {
                src_ptr[i] += (vertical_blocks - 1) * a_block_size * blocks_per_row;
                if (src_ptr[i] + a_block_size * blocks_per_row > src.end()) {
                    src_ptr[i] = src.data(); // useless padding
                }
            }
        }
    }

    /// @brief Loads one decoder layer into its device buffers.
    void load_layer_weights(int layer_idx, Q4NX& q4nx, buffer<u8>& proj_weights,
                            buffer<bf16>& rms_weights, buffer<bf16>& rope_rms_weights) {
        qwen3vl_layer_weight_def& _weight_desc = *this->weight_desc[layer_idx];
        u8* w_ptr_begin = proj_weights.data();
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
            LOG_VERBOSE(1, "Copying qkv weights to buffer");
            // each matrix goes to its own descriptor offset -- the band is
            // packed K|V|Q, not Q|K|V
            reorder_cpy(w_ptr_begin + desc_k.offset, w_k, desc_k.shape[0], desc_k.dtype);
            reorder_cpy(w_ptr_begin + desc_v.offset, w_v, desc_v.shape[0], desc_v.dtype);
            reorder_cpy(w_ptr_begin + desc_q.offset, w_q, desc_q.shape[0], desc_q.dtype);
            desc_q.load();
            desc_k.load();
            desc_v.load();
        }
        // read o weights
        {
            LOG_VERBOSE(1, "Loading o weights");
            weight_desc_t& desc = _weight_desc.attn_output;
            u8* w_ptr = w_ptr_begin + desc.offset;
            buffer<u8> w_o;
            q4nx.load_weights(w_o, desc.format_name(layer_idx));
            reorder_cpy(w_ptr, w_o, desc.shape[0], desc.dtype);
            desc.load();
        }
        // read up/gate weights, interleaved one 512-column slice at a time
        {
            LOG_VERBOSE(1, "Loading shared up/gate weights");
            weight_desc_t& desc_up = _weight_desc.ffn_up;
            weight_desc_t& desc_gate = _weight_desc.ffn_gate;
            u8* w_ptr = w_ptr_begin + desc_up.offset;
            buffer<u8> w_up;
            buffer<u8> w_gate;
            q4nx.load_weights(w_up, desc_up.format_name(layer_idx));
            q4nx.load_weights(w_gate, desc_gate.format_name(layer_idx));

            assert(desc_up.shape[1] % UP_GATE_OUTDIM_REORDER == 0);
            assert(desc_gate.shape[1] % UP_GATE_OUTDIM_REORDER == 0);

            int slices = desc_up.shape[1] / UP_GATE_OUTDIM_REORDER;
            size_t slice_size = desc_up.get_size() / slices;
            for (int i = 0; i < slices; i++) {
                buffer<u8> up_slice = buffer<u8>(w_up.data() + i * slice_size, slice_size);
                buffer<u8> gate_slice = buffer<u8>(w_gate.data() + i * slice_size, slice_size);
                reorder_cpy(w_ptr, up_slice, desc_up.shape[0], desc_up.dtype);
                w_ptr += slice_size;
                reorder_cpy(w_ptr, gate_slice, desc_gate.shape[0], desc_gate.dtype);
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
            buffer<u8> w_down;
            q4nx.load_weights(w_down, desc.format_name(layer_idx));
            assert(w_ptr + desc.get_size() <= proj_weights.end());
            reorder_cpy(w_ptr, w_down, desc.shape[0], desc.dtype);
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

            memcpy(rms_weights_begin + desc_pre.offset, w_rms_pre.data(), desc_pre.get_size());
            memcpy(rms_weights_begin + desc_post.offset, w_rms_post.data(), desc_post.get_size());
            desc_pre.load();
            desc_post.load();
        }
        // read the per-head q/k norms, which live behind the rope table
        {
            LOG_VERBOSE(1, "Loading attention norm weights");
            weight_desc_t& desc_q_norm = _weight_desc.attn_q_norm;
            weight_desc_t& desc_k_norm = _weight_desc.attn_k_norm;
            buffer<bf16> q_attn_norm;
            buffer<bf16> k_attn_norm;

            u8* rope_rms_begin = (u8*)rope_rms_weights.data();
            q4nx.load_weights(q_attn_norm, desc_q_norm.format_name(layer_idx));
            q4nx.load_weights(k_attn_norm, desc_k_norm.format_name(layer_idx));

            memcpy(rope_rms_begin + desc_q_norm.offset, q_attn_norm.data(), desc_q_norm.get_size());
            memcpy(rope_rms_begin + desc_k_norm.offset, k_attn_norm.data(), desc_k_norm.get_size());
            desc_q_norm.load();
            desc_k_norm.load();
        }

        proj_weights.sync_to_device();
        rms_weights.sync_to_device();
        rope_rms_weights.sync_to_device();
    }

    /// @brief Loads the output projection and the final norm.
    void load_head_weights(Q4NX& q4nx, buffer<u8>& lm_head_weights, buffer<bf16>& final_rms_weight) {
        {
            LOG_VERBOSE(1, "Loading final rms weights");
            q4nx.load_weights(final_rms_weight, final_norm.name);
            final_rms_weight.sync_to_device();
            final_norm.load();
        }
        {
            LOG_VERBOSE(1, "Loading lm_head weights");
            buffer<u8> w_lm_head;
            q4nx.load_weights(w_lm_head, lm_head.name);
            reorder_cpy(lm_head_weights.data(), w_lm_head, lm_head.shape[0], lm_head.dtype);
            lm_head_weights.sync_to_device();
            lm_head.load();
        }
    }
};

#endif
