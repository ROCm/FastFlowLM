// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __PHI4_MODEL_DEF_HPP__
#define __PHI4_MODEL_DEF_HPP__
#include <memory>
#include <cmath>
#include <stdexcept>
#include "lm_config.hpp"
#include "weight_desc.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"

/// @brief Weight descriptors of one phi4 decoder layer.
/// @note  The declaration order in all() mirrors the physical layout inside
///        the per-layer proj_weights buffer, which the dequant sequences index
///        by absolute byte offset. Do not reorder.
struct phi4_layer_weight_def {
    weight_desc_t attn_q;
    weight_desc_t attn_k;
    weight_desc_t attn_v;
    weight_desc_t attn_output;
    weight_desc_t ffn_up;
    weight_desc_t ffn_gate;
    weight_desc_t ffn_down;
    weight_desc_t input_layer_norm;
    weight_desc_t post_attntion_norm;

    std::vector<weight_desc_t*> all() {
        return {
            &attn_q, &attn_k, &attn_v, &attn_output,
            &ffn_up, &ffn_gate, &ffn_down,
            &input_layer_norm, &post_attntion_norm
        };
    }
};

/// @brief Owns the weight-shape definition for the phi4 model: the model
///        dimensions derived from LM_Config, the per-layer weight descriptors,
///        the aggregated buffer byte sizes, and the weight loading/reordering.
struct phi4_desc {
    /// @brief RoPE table used when the context fits in the short-factor regime.
    static constexpr f32 inv_freq_short[64] = {
        1.0000e+00, 8.2540e-01, 6.8129e-01, 5.6234e-01, 4.6416e-01, 3.8312e-01, 3.1623e-01, 2.6102e-01,
        2.1544e-01, 1.7783e-01, 1.4678e-01, 1.2115e-01, 1.0000e-01, 8.2540e-02, 6.8129e-02, 5.6234e-02,
        4.6416e-02, 3.8312e-02, 3.1623e-02, 2.6102e-02, 2.1544e-02, 1.7783e-02, 1.4678e-02, 1.2115e-02,
        1.0000e-02, 8.2540e-03, 6.8129e-03, 5.6234e-03, 4.6416e-03, 3.8312e-03, 3.1623e-03, 2.6102e-03,
        2.1544e-03, 1.7783e-03, 1.4678e-03, 1.2115e-03, 1.0000e-03, 8.2540e-04, 6.8129e-04, 5.6234e-04,
        4.6416e-04, 3.8312e-04, 3.1623e-04, 2.6102e-04, 2.1544e-04, 1.7783e-04, 1.4678e-04, 1.2115e-04,
        // paddings, not used.
        2.1544e-03, 1.7783e-03, 1.4678e-03, 1.2115e-03, 1.0000e-03, 8.2540e-04, 6.8129e-04, 5.6234e-04,
        4.6416e-04, 3.8312e-04, 3.1623e-04, 2.6102e-04, 2.1544e-04, 1.7783e-04, 1.4678e-04, 1.2115e-04
    };
    /// @brief RoPE table used past the short-factor context length.
    static constexpr f32 inv_freq_long[64] = {
        1.0000e+00, 7.3807e-01, 5.4475e-01, 4.0207e-01, 2.9676e-01, 2.1903e-01, 1.6166e-01, 1.1932e-01,
        8.8065e-02, 6.4998e-02, 4.7974e-02, 3.5408e-02, 2.6134e-02, 1.9289e-02, 1.4237e-02, 1.0508e-02,
        7.7554e-03, 5.7241e-03, 4.2248e-03, 3.1182e-03, 2.3015e-03, 1.6987e-03, 1.2537e-03, 9.2535e-04,
        6.8298e-04, 5.0409e-04, 3.7206e-04, 2.7461e-04, 2.0268e-04, 1.4959e-04, 1.1041e-04, 8.1491e-05,
        6.7116e-05, 5.5346e-05, 4.5541e-05, 3.7163e-05, 3.0665e-05, 2.5288e-05, 2.0860e-05, 1.7197e-05,
        1.4190e-05, 1.1634e-05, 9.5914e-06, 7.8430e-06, 6.4331e-06, 5.3083e-06, 3.3238e-06, 2.5362e-06,
        // paddings, not used.
        6.7116e-05, 5.5346e-05, 4.5541e-05, 3.7163e-05, 3.0665e-05, 2.5288e-05, 2.0860e-05, 1.7197e-05,
        1.4190e-05, 1.1634e-05, 9.5914e-06, 7.8430e-06, 6.4331e-06, 5.3083e-06, 3.3238e-06, 2.5362e-06
    };

    /// @brief phi4 folds its attention scaling into the RoPE angles.
    static constexpr f32 ROPE_SCALE = 1.1902380714238083f;
    /// @brief context length at which the long RoPE factors take over.
    static constexpr uint32_t ROPE_SHORT_MAX_LENGTH = 4096;

    /// @brief up and gate are interleaved in the weight buffer, one slice of
    ///        this many output columns at a time.
    static constexpr int UP_GATE_OUTDIM_REORDER = 512;

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

    /// @brief RoPE table in use, picked by the requested context length.
    const f32* inv_freq = nullptr;
    /// @brief smallest context this model is willing to run with.
    uint32_t min_max_length = 4096;

    // per-layer weight descriptors
    std::vector<std::unique_ptr<phi4_layer_weight_def>> weight_desc;
    weight_desc_t lm_head;
    weight_desc_t final_norm;

    // aggregated buffer byte sizes (one representative layer)
    size_t total_proj_weights_byte_size = 0;
    size_t rms_weights_byte_size = 0;

    phi4_desc() {}
    phi4_desc(LM_Config config) { build(config); }

    inline size_t get_kv_cache_size(uint32_t MAX_L) const { return (size_t)MAX_L * (DK + DV); }
    inline size_t get_lm_head_w_size() { return lm_head.get_size(); }

    inline uint32_t get_attn_q_offset() { return weight_desc[0]->attn_q.offset; }
    inline uint32_t get_attn_k_offset() { return weight_desc[0]->attn_k.offset; }
    inline uint32_t get_attn_v_offset() { return weight_desc[0]->attn_v.offset; }
    inline uint32_t get_attn_o_offset() { return weight_desc[0]->attn_output.offset; }
    inline uint32_t get_upgate_offset()  { return weight_desc[0]->ffn_up.offset; }
    inline uint32_t get_down_offset()    { return weight_desc[0]->ffn_down.offset; }

    /// @brief byte offset of the qkvo band, which the dequant kernel treats as
    ///        one contiguous [D, DK + DV + DQ + DQ] matrix.
    inline uint32_t get_qkvo_offset() { return get_attn_k_offset(); }
    inline uint32_t get_qkvo_out_dim() { return DK + DV + DQ + DQ; }

    /// @brief Picks the RoPE table for a context length.
    /// @note  Called once from the constructor. update_max_length deliberately
    ///        does not re-run it: switching tables mid-conversation would make
    ///        the already-cached keys inconsistent with the new ones.
    inline void select_rope(uint32_t MAX_L) {
        inv_freq = (MAX_L <= ROPE_SHORT_MAX_LENGTH) ? inv_freq_short : inv_freq_long;
    }

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

        DEBUG_BLOCK(1,
        std::cout << "================== phi4_desc =================" << std::endl;
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
            auto layer = std::make_unique<phi4_layer_weight_def>();
            layer->attn_q = weight_desc_t(flm_q41, {D, DQ}, "model.layers.%d.self_attn.q_proj.weight");
            layer->attn_k = weight_desc_t(flm_q41, {D, DK}, "model.layers.%d.self_attn.k_proj.weight");
            layer->attn_v = weight_desc_t(flm_q41, {D, DV}, "model.layers.%d.self_attn.v_proj.weight");
            layer->attn_output = weight_desc_t(flm_q41, {DQ, D}, "model.layers.%d.self_attn.o_proj.weight");
            layer->ffn_up = weight_desc_t(flm_q41, {D, intermediate_size}, "model.layers.%d.mlp.up_proj.weight");
            layer->ffn_gate = weight_desc_t(flm_q41, {D, intermediate_size}, "model.layers.%d.mlp.gate_proj.weight");
            layer->ffn_down = weight_desc_t(flm_q41, {intermediate_size, D}, "model.layers.%d.mlp.down_proj.weight");
            layer->input_layer_norm = weight_desc_t(flm_bf16, {D}, "model.layers.%d.input_layernorm.weight");
            layer->post_attntion_norm = weight_desc_t(flm_bf16, {D}, "model.layers.%d.post_attention_layernorm.weight");

            if (!is_helper_initialized) {
                // The add order fixes the byte offsets. The dequant kernels read
                // q/k/v/o as one contiguous band, followed by up|gate interleaved
                // and then down.
                // K and V come before Q so the decode layer streams the KV half
                // of the fused projection to the RoPE tile first: the kernel can
                // release K and V into the kv-cache write while the much larger
                // Q rotation is still running.
                proj_weights_helper.add_weight(layer->attn_k);
                proj_weights_helper.add_weight(layer->attn_v);
                proj_weights_helper.add_weight(layer->attn_q);
                proj_weights_helper.add_weight(layer->attn_output);
                proj_weights_helper.add_weight(layer->ffn_up);
                proj_weights_helper.add_weight(layer->ffn_gate);
                proj_weights_helper.add_weight(layer->ffn_down);

                rms_weights_helper.add_weight(layer->input_layer_norm);
                rms_weights_helper.add_weight(layer->post_attntion_norm);

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
    }

    /// @brief Fills the decode RoPE table for one position.
    void set_rope_weights(buffer<bf16>& rope_weights, int idx) {
        bf16* w_rope_ptr = rope_weights.data();
        for (uint32_t i = 0; i < DH / 2; i++) {
            w_rope_ptr[i] = bf16(std::cos(inv_freq[i] * idx) * ROPE_SCALE);
            w_rope_ptr[i + DH / 2] = bf16(std::sin(inv_freq[i] * idx) * ROPE_SCALE);
        }
        rope_weights.sync_to_device();
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
        phi4_layer_weight_def& _weight_desc = *this->weight_desc[layer_idx];
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
            reorder_cpy(w_ptr_begin + desc_k.offset, w_k, D, desc_k.dtype);
            reorder_cpy(w_ptr_begin + desc_v.offset, w_v, D, desc_v.dtype);
            reorder_cpy(w_ptr_begin + desc_q.offset, w_q, D, desc_q.dtype);
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
            reorder_cpy(w_ptr_begin + desc.offset, w_o, D, desc.dtype);
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

        proj_weights.sync_to_device();
        rms_weights.sync_to_device();
    }

    /// @brief Loads the output projection and the final norm.
    void load_head_weights(Q4NX& q4nx, buffer<u8>& lm_head_weights, buffer<bf16>& final_rms_weight) {
        {
            LOG_VERBOSE(1, "Loading final rms weights");
            buffer<bf16> w_rms;
            q4nx.load_weights(w_rms, final_norm.name);
            memcpy(final_rms_weight.data(), w_rms.data(), D * sizeof(bf16));
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
