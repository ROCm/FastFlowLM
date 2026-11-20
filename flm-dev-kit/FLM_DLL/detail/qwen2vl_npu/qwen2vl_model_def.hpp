// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __QWEN2VL_MODEL_DEF_HPP__
#define __QWEN2VL_MODEL_DEF_HPP__
#include <memory>
#include <cmath>
#include "lm_config.hpp"
#include "weight_desc.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"

/// @brief Weight descriptors of one qwen2vl decoder layer.
/// @note  The declaration order in all() mirrors the physical layout inside the
///        per-layer proj_weights / rms_weights buffers, which the dequant
///        sequences and the decoding layer index by absolute byte offset.
///        Do not reorder.
struct qwen2vl_layer_weight_def {
    weight_desc_t attn_q;
    weight_desc_t attn_k;
    weight_desc_t attn_v;
    weight_desc_t attn_output;
    weight_desc_t ffn_up;
    weight_desc_t ffn_gate;
    weight_desc_t ffn_down;
    weight_desc_t input_layer_norm;
    weight_desc_t post_attntion_norm;
    weight_desc_t attn_q_bias;
    weight_desc_t attn_k_bias;
    weight_desc_t attn_v_bias;

    std::vector<weight_desc_t*> all() {
        return {
            &attn_q, &attn_k, &attn_v, &attn_output,
            &ffn_up, &ffn_gate, &ffn_down,
            &input_layer_norm, &post_attntion_norm,
            &attn_q_bias, &attn_k_bias, &attn_v_bias
        };
    }
};

/// @brief Owns the weight-shape definition for the qwen2vl model: the model
///        dimensions derived from LM_Config, the per-layer weight descriptors,
///        the aggregated buffer byte sizes, and the weight loading/reordering.
struct qwen2vl_desc {
    static constexpr f32 inv_freq[64] = {
        1.0000e+00, 8.0584e-01, 6.4938e-01, 5.2330e-01, 4.2170e-01, 3.3982e-01, 2.7384e-01, 2.2067e-01, 1.7783e-01, 1.4330e-01, 1.1548e-01, 9.3057e-02, 7.4989e-02, 6.0430e-02, 4.8697e-02, 3.9242e-02,
        3.1623e-02, 2.5483e-02, 2.0535e-02, 1.6548e-02, 1.3335e-02, 1.0746e-02, 8.6596e-03, 6.9783e-03, 5.6234e-03, 4.5316e-03, 3.6517e-03, 2.9427e-03, 2.3714e-03, 1.9110e-03, 1.5399e-03, 1.2409e-03,
        1.0000e-03, 8.0584e-04, 6.4938e-04, 5.2330e-04, 4.2170e-04, 3.3982e-04, 2.7384e-04, 2.2067e-04, 1.7783e-04, 1.4330e-04, 1.1548e-04, 9.3057e-05, 7.4989e-05, 6.0430e-05, 4.8697e-05, 3.9242e-05,
        3.1623e-05, 2.5483e-05, 2.0535e-05, 1.6548e-05, 1.3335e-05, 1.0746e-05, 8.6596e-06, 6.9783e-06, 5.6234e-06, 4.5316e-06, 3.6517e-06, 2.9427e-06, 2.3714e-06, 1.9110e-06, 1.5399e-06, 1.2409e-06
    };

    /// @brief up and gate are interleaved in the weight buffer, one slice of
    ///        this many output columns at a time.
    static constexpr int UP_GATE_OUTDIM_REORDER = 512;

    /// @brief The rms_weights buffer is also handed to the projection gemms as
    ///        their scratch argument, so it carries this much slack past the
    ///        last real weight. One 32x256 block per bf16 lane.
    static constexpr size_t RMS_SCRATCH_ELEMS = QXNX_ROW_BLOCK_SIZE * QXNX_COL_BLOCK_SIZE * 2;

    // dimensions derived from config
    uint32_t num_hidden_layers = 0;
    uint32_t D = 0;
    uint32_t DH = 0;
    uint32_t DQ = 0;
    uint32_t DK = 0;
    uint32_t DV = 0;
    uint32_t num_heads = 0;
    uint32_t num_kv_heads = 0;
    /// @brief intermediate_size rounded up to a whole number of up/gate slices;
    /// every buffer, offset and sequence is sized from this padded value.
    uint32_t intermediate_size = 0;
    uint32_t vocab_size = 0;
    uint32_t vocab_size_padded = 0;

    // per-layer weight descriptors
    std::vector<std::unique_ptr<qwen2vl_layer_weight_def>> weight_desc;
    weight_desc_t lm_head;
    weight_desc_t final_norm;

    // aggregated buffer byte sizes (one representative layer)
    size_t total_proj_weights_byte_size = 0;
    size_t rms_weights_byte_size = 0;

    qwen2vl_desc() {}
    qwen2vl_desc(LM_Config config) { build(config); }

    inline size_t get_kv_cache_size(uint32_t MAX_L) const { return (size_t)MAX_L * (DK + DV); }
    inline size_t get_lm_head_w_size() { return lm_head.get_size(); }

    inline uint32_t get_attn_q_offset() { return weight_desc[0]->attn_q.offset; }
    inline uint32_t get_attn_k_offset() { return weight_desc[0]->attn_k.offset; }
    inline uint32_t get_attn_v_offset() { return weight_desc[0]->attn_v.offset; }
    inline uint32_t get_attn_o_offset() { return weight_desc[0]->attn_output.offset; }
    inline uint32_t get_upgate_offset()  { return weight_desc[0]->ffn_up.offset; }
    inline uint32_t get_down_offset()    { return weight_desc[0]->ffn_down.offset; }

    /// @brief byte offset of the qkv band, which the dequant kernel treats as
    ///        one contiguous [D, DQ + DK + DV] matrix.
    inline uint32_t get_qkv_offset() { return get_attn_q_offset(); }
    inline uint32_t get_qkv_out_dim() { return DQ + DK + DV; }

    // element offsets of the q/k/v bias rows inside the rms_weights buffer
    inline uint32_t get_q_bias_elem_offset() { return weight_desc[0]->attn_q_bias.offset / sizeof(bf16); }
    inline uint32_t get_k_bias_elem_offset() { return weight_desc[0]->attn_k_bias.offset / sizeof(bf16); }
    inline uint32_t get_v_bias_elem_offset() { return weight_desc[0]->attn_v_bias.offset / sizeof(bf16); }

    inline void build(LM_Config& config) {
        num_hidden_layers = config.get("num_hidden_layers");
        D = config.get("hidden_size");
        DH = config.get("head_dim");
        num_heads = config.get("num_attention_heads");
        num_kv_heads = config.get("num_key_value_heads");
        DQ = DH * num_heads;
        DK = DH * num_kv_heads;
        DV = DH * num_kv_heads;
        intermediate_size = (config.get("intermediate_size") + UP_GATE_OUTDIM_REORDER - 1)
                          / UP_GATE_OUTDIM_REORDER * UP_GATE_OUTDIM_REORDER;
        vocab_size = config.get("vocab_size");
        vocab_size_padded = (vocab_size + D - 1) / D * D;

        DEBUG_BLOCK(1,
        std::cout << "================= qwen2vl_desc =================" << std::endl;
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
        bool is_helper_initialized = false;

        for (size_t i = 0; i < num_hidden_layers; i++) {
            auto layer = std::make_unique<qwen2vl_layer_weight_def>();
            layer->attn_q = weight_desc_t(flm_q41, {D, DQ}, "model.layers.%d.self_attn.q_proj.weight");
            layer->attn_k = weight_desc_t(flm_q41, {D, DK}, "model.layers.%d.self_attn.k_proj.weight");
            layer->attn_v = weight_desc_t(flm_q41, {D, DV}, "model.layers.%d.self_attn.v_proj.weight");
            layer->attn_output = weight_desc_t(flm_q41, {DQ, D}, "model.layers.%d.self_attn.o_proj.weight");
            layer->ffn_up = weight_desc_t(flm_q41, {D, intermediate_size}, "model.layers.%d.mlp.up_proj.weight");
            layer->ffn_gate = weight_desc_t(flm_q41, {D, intermediate_size}, "model.layers.%d.mlp.gate_proj.weight");
            layer->ffn_down = weight_desc_t(flm_q41, {intermediate_size, D}, "model.layers.%d.mlp.down_proj.weight");
            layer->input_layer_norm = weight_desc_t(flm_bf16, {D}, "model.layers.%d.input_layernorm.weight");
            layer->post_attntion_norm = weight_desc_t(flm_bf16, {D}, "model.layers.%d.post_attention_layernorm.weight");
            layer->attn_q_bias = weight_desc_t(flm_bf16, {DQ}, "model.layers.%d.self_attn.q_proj.bias");
            layer->attn_k_bias = weight_desc_t(flm_bf16, {DK}, "model.layers.%d.self_attn.k_proj.bias");
            layer->attn_v_bias = weight_desc_t(flm_bf16, {DV}, "model.layers.%d.self_attn.v_proj.bias");

            if (!is_helper_initialized) {
                // The add order fixes the byte offsets: the dequant kernels read
                // q|k|v as one contiguous band, then o, then up|gate, then down.
                proj_weights_helper.add_weight(layer->attn_q);
                proj_weights_helper.add_weight(layer->attn_k);
                proj_weights_helper.add_weight(layer->attn_v);
                proj_weights_helper.add_weight(layer->attn_output);
                proj_weights_helper.add_weight(layer->ffn_up);
                proj_weights_helper.add_weight(layer->ffn_gate);
                proj_weights_helper.add_weight(layer->ffn_down);

                rms_weights_helper.add_weight(layer->input_layer_norm);
                rms_weights_helper.add_weight(layer->post_attntion_norm);
                rms_weights_helper.add_weight(layer->attn_q_bias);
                rms_weights_helper.add_weight(layer->attn_k_bias);
                rms_weights_helper.add_weight(layer->attn_v_bias);

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
        rms_weights_byte_size = rms_weights_helper.get_size() + RMS_SCRATCH_ELEMS * sizeof(bf16);
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
    void load_layer_weights(int layer_idx, Q4NX& q4nx, buffer<u8>& proj_weights, buffer<bf16>& rms_weights) {
        qwen2vl_layer_weight_def& _weight_desc = *this->weight_desc[layer_idx];
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
            u8* w_ptr = w_ptr_begin + desc.offset;
            buffer<u8> w_down;
            q4nx.load_weights(w_down, desc.format_name(layer_idx));
            assert(w_ptr + desc.get_size() <= proj_weights.end());
            reorder_cpy(w_ptr, w_down, intermediate_size, desc.dtype);
            desc.load();
        }
        // read rms weights and the q/k/v projection biases that follow them
        {
            LOG_VERBOSE(1, "Loading rms weights");
            u8* rms_weights_begin = (u8*)rms_weights.data();
            weight_desc_t* rms_descs[] = {
                &_weight_desc.input_layer_norm, &_weight_desc.post_attntion_norm,
                &_weight_desc.attn_q_bias, &_weight_desc.attn_k_bias, &_weight_desc.attn_v_bias
            };
            for (weight_desc_t* desc : rms_descs) {
                buffer<bf16> w;
                q4nx.load_weights(w, desc->format_name(layer_idx));
                memcpy(rms_weights_begin + desc->offset, w.data(), desc->get_size());
                desc->load();
            }
        }

        proj_weights.sync_to_device();
        rms_weights.sync_to_device();
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
            reorder_cpy(lm_head_weights.data(), w_lm_head, D, lm_head.dtype);
            lm_head_weights.sync_to_device();
            lm_head.load();
        }
    }
};

#endif
