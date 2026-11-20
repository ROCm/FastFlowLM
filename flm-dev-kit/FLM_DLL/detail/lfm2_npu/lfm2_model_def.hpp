// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __LFM2_MODEL_DEF_HPP__
#define __LFM2_MODEL_DEF_HPP__
#include <memory>
#include <cmath>
#include <stdexcept>
#include "lm_config.hpp"
#include "weight_desc.hpp"
#include "modules/mha.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"

/// @brief Weight descriptors of one lfm2 block.
/// @note  lfm2 interleaves two kinds of block: attention blocks and short
///        convolution blocks. Both share a single per-layer proj_weights
///        buffer whose layout is
///
///            [ slot 0: D x 3D | slot 1: D x D | up/gate interleaved | down ]
///
///        On an attention layer slot 0 holds q|k|v (which is narrower than 3D,
///        so the tail of the slot is unused padding) and slot 1 holds o_proj.
///        On a conv layer slot 0 holds shortconv.in_proj and slot 1 holds
///        shortconv.out_proj. Keeping the two kinds on the same offsets is what
///        lets one buffer size and one set of dequant sequences serve both.
struct lfm2_layer_weight_def {
    bool is_attn = false;

    // slot 0
    weight_desc_t attn_q;
    weight_desc_t attn_k;
    weight_desc_t attn_v;
    weight_desc_t conv_in;
    // slot 1
    weight_desc_t attn_output;
    weight_desc_t conv_out;
    // shared mlp
    weight_desc_t ffn_up;
    weight_desc_t ffn_gate;
    weight_desc_t ffn_down;
    // rms band
    weight_desc_t input_layer_norm;
    weight_desc_t post_attntion_norm;
    // rope/rms band: rope table + q_norm + k_norm on attention layers, the
    // transposed shortconv kernel on conv layers
    weight_desc_t rope_table;
    weight_desc_t attn_q_norm;
    weight_desc_t attn_k_norm;
    weight_desc_t conv_kernel;

    std::vector<weight_desc_t*> all() {
        return {
            &attn_q, &attn_k, &attn_v, &conv_in,
            &attn_output, &conv_out,
            &ffn_up, &ffn_gate, &ffn_down,
            &input_layer_norm, &post_attntion_norm,
            &rope_table, &attn_q_norm, &attn_k_norm, &conv_kernel
        };
    }
};

/// @brief Owns the weight-shape definition for the lfm2 model: the model
///        dimensions derived from LM_Config, the attention/conv layer split,
///        the per-layer weight descriptors, the aggregated buffer byte sizes,
///        and the weight loading/reordering.
struct lfm2_desc {
    static constexpr f32 inv_freq_1_2b[32] = {
        1.0000e+00, 6.4938e-01, 4.2170e-01, 2.7384e-01, 1.7783e-01, 1.1548e-01,
        7.4989e-02, 4.8697e-02, 3.1623e-02, 2.0535e-02, 1.3335e-02, 8.6596e-03,
        5.6234e-03, 3.6517e-03, 2.3714e-03, 1.5399e-03, 1.0000e-03, 6.4938e-04,
        4.2170e-04, 2.7384e-04, 1.7783e-04, 1.1548e-04, 7.4989e-05, 4.8697e-05,
        3.1623e-05, 2.0535e-05, 1.3335e-05, 8.6596e-06, 5.6234e-06, 3.6517e-06,
        2.3714e-06, 1.5399e-06
    };

    static constexpr f32 inv_freq_2_6b[32] = {
        1.0000e+00, 6.4938e-01, 4.2170e-01, 2.7384e-01, 1.7783e-01, 1.1548e-01,
        7.4989e-02, 4.8697e-02, 3.1623e-02, 2.0535e-02, 1.3335e-02, 8.6596e-03,
        5.6234e-03, 3.6517e-03, 2.3714e-03, 1.5399e-03, 1.0000e-03, 6.4938e-04,
        4.2170e-04, 2.7384e-04, 1.7783e-04, 1.1548e-04, 7.4989e-05, 4.8697e-05,
        3.1623e-05, 2.0535e-05, 1.3335e-05, 8.6596e-06, 5.6234e-06, 3.6517e-06,
        2.3714e-06, 1.5399e-06
    };

    static constexpr int lfm2_1_2b_attn_layer_idxs[] = {2, 5, 8, 10, 12, 14};
    static constexpr int lfm2_2_6b_attn_layer_idxs[] = {2, 5, 9, 13, 17, 21, 24, 27};

    /// @brief up and gate are interleaved in the weight buffer, one slice of
    ///        this many output columns at a time.
    static constexpr int UP_GATE_OUTDIM_REORDER = 512;
    /// @brief the short convolution has a kernel width of 3.
    static constexpr int CONV_KERNEL_SIZE = 3;

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

    /// @brief RoPE table, attention-layer positions and the minimum context,
    ///        all picked by intermediate size.
    const f32* inv_freq = nullptr;
    const int* attn_layer_idxs = nullptr;
    int total_attn_layers = 0;
    mha_type_t mha_type = mha_d64_q4;
    /// @brief smallest context this model is willing to run with.
    uint32_t min_max_length = 4096;

    // per-layer weight descriptors
    std::vector<std::unique_ptr<lfm2_layer_weight_def>> weight_desc;
    weight_desc_t lm_head;
    weight_desc_t final_norm;

    // aggregated buffer byte sizes (one representative layer)
    size_t total_proj_weights_byte_size = 0;
    size_t rms_weights_byte_size = 0;
    size_t rope_rms_weights_byte_size = 0;

    lfm2_desc() {}
    lfm2_desc(LM_Config config) { build(config); }

    /// @brief Whether block `layer_idx` is an attention block (the rest are
    ///        short convolution blocks).
    inline bool is_attn_layer(uint32_t layer_idx) const {
        for (int i = 0; i < total_attn_layers; i++) {
            if ((uint32_t)attn_layer_idxs[i] == layer_idx) return true;
        }
        return false;
    }

    inline size_t get_kv_cache_size(uint32_t MAX_L) const { return (size_t)MAX_L * (DK + DV); }
    /// @brief conv blocks keep the last two rows of the conv input instead of a kv cache.
    inline size_t get_conv_cache_size() const { return (size_t)D * 2; }
    inline size_t get_lm_head_w_size() { return lm_head.get_size(); }

    inline uint32_t get_qkv_offset()     { return weight_desc[0]->attn_q.offset; }
    inline uint32_t get_conv_in_offset() { return weight_desc[0]->conv_in.offset; }
    inline uint32_t get_attn_o_offset()  { return weight_desc[0]->attn_output.offset; }
    inline uint32_t get_upgate_offset()  { return weight_desc[0]->ffn_up.offset; }
    inline uint32_t get_down_offset()    { return weight_desc[0]->ffn_down.offset; }

    /// @brief output width of the qkv band as the dequant kernel sees it.
    inline uint32_t get_qkv_out_dim() { return DQ + DK + DV; }
    /// @brief output width of slot 0, which is what the conv in_proj fills.
    inline uint32_t get_slot0_out_dim() { return D * 3; }

    /// @brief element offsets inside the rope/rms band of one layer
    inline uint32_t get_q_norm_element_offset() { return DH; }
    inline uint32_t get_k_norm_element_offset() { return DH * 2; }

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

        // The RoPE table, the attention-layer positions and the minimum context
        // are all keyed off the intermediate size, which is what distinguishes
        // the 1.2B and 2.6B checkpoints from each other.
        if (intermediate_size == 8192) {
            inv_freq = inv_freq_1_2b;
            attn_layer_idxs = lfm2_1_2b_attn_layer_idxs;
            total_attn_layers = sizeof(lfm2_1_2b_attn_layer_idxs) / sizeof(lfm2_1_2b_attn_layer_idxs[0]);
            min_max_length = 8192;
        }
        else if (intermediate_size == 10752) {
            inv_freq = inv_freq_2_6b;
            attn_layer_idxs = lfm2_2_6b_attn_layer_idxs;
            total_attn_layers = sizeof(lfm2_2_6b_attn_layer_idxs) / sizeof(lfm2_2_6b_attn_layer_idxs[0]);
            min_max_length = 4096;
        }
        else {
            throw std::runtime_error("Unsupported intermediate size: " + std::to_string(intermediate_size));
        }

        // o_proj and shortconv.out_proj must be the same shape for them to share
        // slot 1, and q|k|v must fit inside slot 0.
        assert(DQ == D);
        assert(get_qkv_out_dim() <= get_slot0_out_dim());

        DEBUG_BLOCK(1,
        std::cout << "================== lfm2_desc =================" << std::endl;
        std::cout << std::left << std::setw(28) << "num_hidden_layers" << " = " << num_hidden_layers << std::endl;
        std::cout << std::left << std::setw(28) << "D (hidden_size)"   << " = " << D << std::endl;
        std::cout << std::left << std::setw(28) << "DH (head_dim)"     << " = " << DH << std::endl;
        std::cout << std::left << std::setw(28) << "DQ"                << " = " << DQ << std::endl;
        std::cout << std::left << std::setw(28) << "DK"                << " = " << DK << std::endl;
        std::cout << std::left << std::setw(28) << "DV"                << " = " << DV << std::endl;
        std::cout << std::left << std::setw(28) << "num_heads"         << " = " << num_heads << std::endl;
        std::cout << std::left << std::setw(28) << "num_kv_heads"      << " = " << num_kv_heads << std::endl;
        std::cout << std::left << std::setw(28) << "intermediate_size" << " = " << intermediate_size << std::endl;
        std::cout << std::left << std::setw(28) << "attention layers"  << " = " << total_attn_layers << std::endl;
        std::cout << std::left << std::setw(28) << "vocab_size"        << " = " << vocab_size << std::endl;
        std::cout << "==============================================" << std::endl;
        )

        lm_head = weight_desc_t(flm_q41, {D, vocab_size}, "lm_head.weight");
        final_norm = weight_desc_t(flm_bf16, {D}, "model.norm.weight");
        // both own their buffer outright, so their offset is trivially 0
        lm_head.indp();
        final_norm.indp();

        weight_desc.resize(num_hidden_layers);

        // Slot 0 and slot 1 of the shared layout. They are not weights of their
        // own; they only reserve the space that the two layer kinds overlay.
        weight_desc_t slot0(flm_q41, {D, (int64_t)get_slot0_out_dim()}, "<slot0>");
        weight_desc_t slot1(flm_q41, {D, D}, "<slot1>");

        weight_container proj_weights_helper;
        weight_container rms_weights_helper;
        weight_container rope_rms_weights_helper;
        bool is_helper_initialized = false;

        for (size_t i = 0; i < num_hidden_layers; i++) {
            auto layer = std::make_unique<lfm2_layer_weight_def>();
            layer->is_attn = is_attn_layer(i);

            layer->attn_q = weight_desc_t(flm_q41, {D, DQ}, "model.layers.%d.self_attn.q_proj.weight");
            layer->attn_k = weight_desc_t(flm_q41, {D, DK}, "model.layers.%d.self_attn.k_proj.weight");
            layer->attn_v = weight_desc_t(flm_q41, {D, DV}, "model.layers.%d.self_attn.v_proj.weight");
            layer->conv_in = weight_desc_t(flm_q41, {D, (int64_t)get_slot0_out_dim()}, "model.layers.%d.shortconv.in_proj.weight");
            layer->attn_output = weight_desc_t(flm_q41, {DQ, D}, "model.layers.%d.self_attn.o_proj.weight");
            layer->conv_out = weight_desc_t(flm_q41, {D, D}, "model.layers.%d.shortconv.out_proj.weight");
            layer->ffn_up = weight_desc_t(flm_q41, {D, intermediate_size}, "model.layers.%d.mlp.up_proj.weight");
            layer->ffn_gate = weight_desc_t(flm_q41, {D, intermediate_size}, "model.layers.%d.mlp.gate_proj.weight");
            layer->ffn_down = weight_desc_t(flm_q41, {intermediate_size, D}, "model.layers.%d.mlp.down_proj.weight");
            layer->input_layer_norm = weight_desc_t(flm_bf16, {D}, "model.layers.%d.input_layernorm.weight");
            layer->post_attntion_norm = weight_desc_t(flm_bf16, {D}, "model.layers.%d.post_attention_layernorm.weight");
            layer->rope_table = weight_desc_t(flm_bf16, {DH}, "<rope>");
            layer->attn_q_norm = weight_desc_t(flm_bf16, {DH}, "model.layers.%d.self_attn.q_norm.weight");
            layer->attn_k_norm = weight_desc_t(flm_bf16, {DH}, "model.layers.%d.self_attn.k_norm.weight");
            layer->conv_kernel = weight_desc_t(flm_bf16, {(int64_t)(CONV_KERNEL_SIZE * D)}, "model.layers.%d.shortconv.conv.weight");

            if (!is_helper_initialized) {
                // The add order fixes the byte offsets. The dequant kernels read
                // slot 0, then slot 1, then up|gate as one band, then down.
                proj_weights_helper.add_weight(slot0);
                proj_weights_helper.add_weight(slot1);
                proj_weights_helper.add_weight(layer->ffn_up);
                proj_weights_helper.add_weight(layer->ffn_gate);
                proj_weights_helper.add_weight(layer->ffn_down);

                // q|k|v are packed at the front of slot 0; the conv in_proj is
                // the whole slot. o_proj and out_proj both start at slot 1.
                layer->conv_in.offset = slot0.offset;
                layer->attn_q.offset = slot0.offset;
                layer->attn_k.offset = layer->attn_q.offset + layer->attn_q.get_size();
                layer->attn_v.offset = layer->attn_k.offset + layer->attn_k.get_size();
                layer->attn_output.offset = slot1.offset;
                layer->conv_out.offset = slot1.offset;
                layer->conv_in.indp();
                layer->attn_q.indp();
                layer->attn_k.indp();
                layer->attn_v.indp();
                layer->attn_output.indp();
                layer->conv_out.indp();

                rms_weights_helper.add_weight(layer->input_layer_norm);
                rms_weights_helper.add_weight(layer->post_attntion_norm);

                // The rope/rms band is a union too: an attention layer stores the
                // rope table then q_norm then k_norm, a conv layer stores the
                // transposed 3 x D convolution kernel over the whole band.
                rope_rms_weights_helper.add_weight(layer->rope_table);
                rope_rms_weights_helper.add_weight(layer->attn_q_norm);
                rope_rms_weights_helper.add_weight(layer->attn_k_norm);
                layer->conv_kernel.offset = 0;
                layer->conv_kernel.indp();

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
        // the conv kernel is the wider of the two overlays of this band
        rope_rms_weights_byte_size = std::max(rope_rms_weights_helper.get_size(),
                                              weight_desc[0]->conv_kernel.get_size());
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

    /// @brief Loads the up/gate band, interleaved one UP_GATE_OUTDIM_REORDER
    ///        slice at a time, and then the down projection.
    void load_mlp_weights(int layer_idx, Q4NX& q4nx, u8* w_ptr_begin, buffer<u8>& proj_weights) {
        lfm2_layer_weight_def& _weight_desc = *this->weight_desc[layer_idx];
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
    }

    /// @brief Loads the projections of one attention block.
    void load_attn_proj_weights(int layer_idx, Q4NX& q4nx, buffer<u8>& proj_weights) {
        lfm2_layer_weight_def& _weight_desc = *this->weight_desc[layer_idx];
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
        {
            LOG_VERBOSE(1, "Loading o weights");
            weight_desc_t& desc = _weight_desc.attn_output;
            buffer<u8> w_o;
            q4nx.load_weights(w_o, desc.format_name(layer_idx));
            reorder_cpy(w_ptr_begin + desc.offset, w_o, D, desc.dtype);
            desc.load();
        }
        load_mlp_weights(layer_idx, q4nx, w_ptr_begin, proj_weights);
        proj_weights.sync_to_device();
    }

    /// @brief Loads the projections of one short convolution block.
    void load_conv_proj_weights(int layer_idx, Q4NX& q4nx, buffer<u8>& proj_weights) {
        lfm2_layer_weight_def& _weight_desc = *this->weight_desc[layer_idx];
        u8* w_ptr_begin = proj_weights.data();
        {
            LOG_VERBOSE(1, "Loading conv in/out weights");
            weight_desc_t& desc_in = _weight_desc.conv_in;
            weight_desc_t& desc_out = _weight_desc.conv_out;
            buffer<u8> w_in;
            buffer<u8> w_out;
            q4nx.load_weights(w_in, desc_in.format_name(layer_idx));
            q4nx.load_weights(w_out, desc_out.format_name(layer_idx));
            reorder_cpy(w_ptr_begin + desc_in.offset, w_in, D, desc_in.dtype);
            reorder_cpy(w_ptr_begin + desc_out.offset, w_out, D, desc_out.dtype);
            desc_in.load();
            desc_out.load();
        }
        load_mlp_weights(layer_idx, q4nx, w_ptr_begin, proj_weights);
        proj_weights.sync_to_device();
    }

    /// @brief Loads one block: its projections, its two norms, and whichever of
    ///        the q/k norms or the convolution kernel its kind calls for.
    void load_layer_weights(int layer_idx, Q4NX& q4nx, buffer<u8>& proj_weights,
                            buffer<bf16>& rms_weights, buffer<bf16>& rope_rms_weights) {
        lfm2_layer_weight_def& _weight_desc = *this->weight_desc[layer_idx];

        if (_weight_desc.is_attn) {
            load_attn_proj_weights(layer_idx, q4nx, proj_weights);
        }
        else {
            load_conv_proj_weights(layer_idx, q4nx, proj_weights);
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

        u8* rope_rms_begin = (u8*)rope_rms_weights.data();
        if (_weight_desc.is_attn) {
            LOG_VERBOSE(1, "Loading attention norm weights");
            weight_desc_t& desc_q_norm = _weight_desc.attn_q_norm;
            weight_desc_t& desc_k_norm = _weight_desc.attn_k_norm;
            buffer<bf16> q_attn_norm;
            buffer<bf16> k_attn_norm;
            q4nx.load_weights(q_attn_norm, desc_q_norm.format_name(layer_idx));
            q4nx.load_weights(k_attn_norm, desc_k_norm.format_name(layer_idx));
            memcpy(rope_rms_begin + desc_q_norm.offset, q_attn_norm.data(), DH * sizeof(bf16));
            memcpy(rope_rms_begin + desc_k_norm.offset, k_attn_norm.data(), DH * sizeof(bf16));
            desc_q_norm.load();
            desc_k_norm.load();
        }
        else {
            LOG_VERBOSE(1, "Loading conv weights");
            weight_desc_t& desc = _weight_desc.conv_kernel;
            buffer<bf16> w_conv;
            q4nx.load_weights(w_conv, desc.format_name(layer_idx));
            // the checkpoint stores D x 3; the kernel wants 3 x D
            bf16* w_conv_ptr = (bf16*)(rope_rms_begin + desc.offset);
            for (int i = 0; i < CONV_KERNEL_SIZE; i++) {
                for (uint32_t j = 0; j < D; j++) {
                    *w_conv_ptr++ = w_conv[j * CONV_KERNEL_SIZE + i];
                }
            }
            desc.load();
        }

        proj_weights.sync_to_device();
        rms_weights.sync_to_device();
        rope_rms_weights.sync_to_device();
    }

    /// @brief Writes the rope cos/sin table for position `idx` into the rope/rms
    ///        band of every attention layer.
    void set_rope_weights(int idx, std::vector<buffer<bf16>>& rope_rms_weights) {
        buffer<bf16> cos_buf(DH / 2);
        buffer<bf16> sin_buf(DH / 2);
        for (uint32_t j = 0; j < DH / 2; j++) {
            cos_buf[j] = (bf16)cos(inv_freq[j] * idx);
            sin_buf[j] = (bf16)sin(inv_freq[j] * idx);
        }

        for (int i = 0; i < total_attn_layers; i++) {
            int layer_idx = attn_layer_idxs[i];
            weight_desc_t& desc = weight_desc[layer_idx]->rope_table;
            bf16* w_rope_ptr = (bf16*)((u8*)rope_rms_weights[layer_idx].data() + desc.offset);
            for (uint32_t j = 0; j < DH / 2; j++) {
                w_rope_ptr[j] = cos_buf[j];
                w_rope_ptr[j + DH / 2] = sin_buf[j];
            }
            rope_rms_weights[layer_idx].sync_to_device();
        }
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
