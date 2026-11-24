// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#ifndef __QWEN3_ASR_NPU_DEF_HPP__
#define __QWEN3_ASR_NPU_DEF_HPP__
#include <memory>
#include <cmath>
#include <stdexcept>
#include <filesystem>
#include <cstring>
#include "lm_config.hpp"
#include "weight_desc.hpp"
#include "modules/mha.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"

/// @brief Weight descriptors of one qwen3 decoder layer.
/// @note  The declaration order in all() mirrors the physical layout inside the
///        per-layer proj_weights buffer, which the dequant sequences index by
///        absolute byte offset. Do not reorder.
struct qwen3_asr_layer_weight_def {
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

/// @brief Owns the weight-shape definition for the qwen3 model: the model
///        dimensions derived from LM_Config, the per-layer weight descriptors,
///        the aggregated buffer byte sizes, and the weight loading/reordering.
struct qwen3_asr_desc {
    static constexpr f32 inv_freq[64] = {
        1.0000e+00, 8.0584e-01, 6.4938e-01, 5.2330e-01, 4.2170e-01, 3.3982e-01, 2.7384e-01, 2.2067e-01, 1.7783e-01, 1.4330e-01, 1.1548e-01, 9.3057e-02, 7.4989e-02, 6.0430e-02, 4.8697e-02, 3.9242e-02,
        3.1623e-02, 2.5483e-02, 2.0535e-02, 1.6548e-02, 1.3335e-02, 1.0746e-02, 8.6596e-03, 6.9783e-03, 5.6234e-03, 4.5316e-03, 3.6517e-03, 2.9427e-03, 2.3714e-03, 1.9110e-03, 1.5399e-03, 1.2409e-03,
        1.0000e-03, 8.0584e-04, 6.4938e-04, 5.2330e-04, 4.2170e-04, 3.3982e-04, 2.7384e-04, 2.2067e-04, 1.7783e-04, 1.4330e-04, 1.1548e-04, 9.3057e-05, 7.4989e-05, 6.0430e-05, 4.8697e-05, 3.9242e-05,
        3.1623e-05, 2.5483e-05, 2.0535e-05, 1.6548e-05, 1.3335e-05, 1.0746e-05, 8.6596e-06, 6.9783e-06, 5.6234e-06, 4.5316e-06, 3.6517e-06, 2.9427e-06, 2.3714e-06, 1.9110e-06, 1.5399e-06, 1.2409e-06
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
    uint32_t min_max_length = 4096;
    /// @brief the attn overlay is built for eight kv heads regardless of config.
    static constexpr uint32_t mha_kv_heads = 8;

    // per-layer weight descriptors
    std::vector<std::unique_ptr<qwen3_asr_layer_weight_def>> weight_desc;
    weight_desc_t lm_head;
    weight_desc_t final_norm;

    /// @brief mixed Q4_1 / Q8_0 projection weights for the decode xclbin built as MODEL_TYPE QWEN3_ASR_1_7B_MIX
    ///        (FLM_Xclbin/generic_decoding_layer/models/qwen3-asr-1.7b-mix.h): on when the package has MIX_FILE, which holds the
    ///        Q4_1 copies of the projections (model.q4nx keeps Q8_0 for all of them: the CPU prefill reads those). Every weight DMA is
    ///        then one MIX_UNIT-byte unit: two Q4_1 32x256 blocks side by side, or one Q8_0 block and zero padding.
    bool mixed = false;
    static constexpr size_t MIX_UNIT = 10240;
    static constexpr const char* MIX_FILE = "npu_q41.q4nx";
    /// which projections stay Q8_0 in the mixed build; must match MIX_*_Q8 in the kernel's model header
    static constexpr bool MIX_V_Q8 = true, MIX_DOWN_Q8 = true, MIX_HEAD_Q8 = false;

    // aggregated buffer byte sizes (one representative layer)
    size_t total_proj_weights_byte_size = 0;
    size_t rms_weights_byte_size = 0;
    size_t rope_rms_weights_byte_size = 0;

    qwen3_asr_desc() {}
    qwen3_asr_desc(LM_Config config) { build(config); }

    inline size_t get_kv_cache_size(uint32_t MAX_L) const { return (size_t)MAX_L * (DK + DV); }
    inline size_t get_lm_head_w_size() { return npu_bytes(lm_head); }

    /// @brief bytes of one 32-row block row of `w` (cols columns) as the NPU streams it
    size_t npu_row_bytes(const weight_desc_t& w, size_t cols) const {
        if (!mixed)
            return cols / QXNX_COL_BLOCK_SIZE * get_quantization_byte_size(QXNX_ROW_BLOCK_SIZE * QXNX_COL_BLOCK_SIZE, w.dtype);
        return cols / (w.dtype == flm_q41 ? 2 * QXNX_COL_BLOCK_SIZE : QXNX_COL_BLOCK_SIZE) * MIX_UNIT;
    }
    /// @brief bytes of `w` in its NPU buffer (shape {in, out}; the out dim is rounded up to whole 32-row blocks)
    size_t npu_bytes(weight_desc_t& w) {
        if (!mixed)
            return w.get_size();
        return (w.shape[1] + QXNX_ROW_BLOCK_SIZE - 1) / QXNX_ROW_BLOCK_SIZE * npu_row_bytes(w, w.shape[0]);
    }

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
        mixed = std::filesystem::exists(utils::path_join(config.model_path, MIX_FILE));
        const flm_dtype_t q4 = mixed ? flm_q41 : flm_q80;              // q, k, o, up, gate
        const flm_dtype_t v_t = (mixed && !MIX_V_Q8) ? flm_q41 : flm_q80;
        const flm_dtype_t down_t = (mixed && !MIX_DOWN_Q8) ? flm_q41 : flm_q80;

        // The attention kernel and the minimum context are keyed off the
        // intermediate size, which is what distinguishes the checkpoints.
        if (intermediate_size == 3072 || intermediate_size == 6144) {
            mha_type = mha_d128_q2;
            min_max_length = 2048;   // a request's context is at most a ~65 s piece (~850 audio rows) and its text
        }
        else if (intermediate_size == 9728 || intermediate_size == 12288) {
            mha_type = mha_d128_q4;
            min_max_length = 4096;
        }
        else {
            throw std::runtime_error("Unsupported intermediate size: " + std::to_string(intermediate_size));
        }

        DEBUG_BLOCK(1,
        std::cout << "================= qwen3_asr_desc =================" << std::endl;
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

        lm_head = weight_desc_t((mixed && !MIX_HEAD_Q8) ? flm_q41 : flm_q80, {D, vocab_size_padded}, "lm_head.weight");
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
            auto layer = std::make_unique<qwen3_asr_layer_weight_def>();
            layer->attn_q = weight_desc_t(q4, {D, DQ}, "model.layers.%d.self_attn.q_proj.weight");
            layer->attn_k = weight_desc_t(q4, {D, DK}, "model.layers.%d.self_attn.k_proj.weight");
            layer->attn_v = weight_desc_t(v_t, {D, DV}, "model.layers.%d.self_attn.v_proj.weight");
            layer->attn_output = weight_desc_t(q4, {DQ, D}, "model.layers.%d.self_attn.o_proj.weight");
            layer->ffn_up = weight_desc_t(q4, {D, intermediate_size}, "model.layers.%d.mlp.up_proj.weight");
            layer->ffn_gate = weight_desc_t(q4, {D, intermediate_size}, "model.layers.%d.mlp.gate_proj.weight");
            layer->ffn_down = weight_desc_t(down_t, {intermediate_size, D}, "model.layers.%d.mlp.down_proj.weight");
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
                // waiting behind it. See qwen3_asr_npu_sequence::_receive_kv_cache
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
        if (mixed) {
            // the same K|V|Q, o, up|gate, down order, in DMA units
            size_t off = 0;
            auto& l0 = *weight_desc[0];
            for (weight_desc_t* w : {&l0.attn_k, &l0.attn_v, &l0.attn_q, &l0.attn_output, &l0.ffn_up, &l0.ffn_gate, &l0.ffn_down}) {
                w->offset = off;
                off += npu_bytes(*w);
            }
            for (size_t i = 1; i < num_hidden_layers; i++) {
                auto ref_all = l0.all();
                auto cur_all = weight_desc[i]->all();
                for (size_t w = 0; w < cur_all.size(); w++) cur_all[w]->offset = ref_all[w]->offset;
            }
            total_proj_weights_byte_size = off;
        }
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

    /// @brief The mixed layout's units, in the same 2-row-block interleave as reorder_cpy: per pair of 32-row blocks and per unit,
    ///        the unit of the first row block then that of the second. A Q4_1 unit is two consecutive 32x256 blocks of one row
    ///        block; a Q8_0 unit one block and zeros. dst_rows (row blocks) may exceed the source's: those units are zero.
    void reorder_units(u8* dst, buffer<u8>& src, const int col, flm_dtype_t dtype, int dst_rows = -1) {
        const size_t block = get_quantization_byte_size(QXNX_ROW_BLOCK_SIZE * QXNX_COL_BLOCK_SIZE, dtype);
        const int blocks_per_row = col / QXNX_COL_BLOCK_SIZE;
        const int per_unit = dtype == flm_q41 ? 2 : 1;
        const int units_per_row = blocks_per_row / per_unit;
        const int src_rows = (int)(src.size() / block / blocks_per_row);
        if (dst_rows < 0) dst_rows = src_rows;
        assert(blocks_per_row % per_unit == 0 && dst_rows % 2 == 0);
        u8* d = dst;
        for (int r = 0; r < dst_rows; r += 2)
            for (int u = 0; u < units_per_row; u++)
                for (int i = 0; i < 2; i++, d += MIX_UNIT) {
                    if (r + i >= src_rows) {
                        std::memset(d, 0, MIX_UNIT);
                        continue;
                    }
                    const u8* s = src.data() + ((size_t)(r + i) * blocks_per_row + (size_t)u * per_unit) * block;
                    std::memcpy(d, s, per_unit * block);
                    std::memset(d + per_unit * block, 0, MIX_UNIT - per_unit * block);
                }
    }
    /// @brief copy one weight into its NPU buffer layout
    void npu_cpy(u8* dst, buffer<u8>& src, const int col, flm_dtype_t dtype, int dst_rows = -1) {
        if (mixed) reorder_units(dst, src, col, dtype, dst_rows);
        else reorder_cpy(dst, src, col, dtype);
    }
    /// @brief the file a weight comes from: the Q4_1 copies are in MIX_FILE
    static Q4NX& source(Q4NX& q4nx, Q4NX* npu, const weight_desc_t& w) {
        return (npu && w.dtype == flm_q41) ? *npu : q4nx;
    }

    /// @brief Loads one decoder layer into its device buffers.
    void load_layer_weights(int layer_idx, Q4NX& q4nx, Q4NX* npu, buffer<u8>& proj_weights,
                            buffer<bf16>& rms_weights, buffer<bf16>& rope_rms_weights) {
        qwen3_asr_layer_weight_def& _weight_desc = *this->weight_desc[layer_idx];
        u8* w_ptr_begin = proj_weights.data();
        {
            LOG_VERBOSE(1, "Loading qkv weights");
            weight_desc_t& desc_q = _weight_desc.attn_q;
            weight_desc_t& desc_k = _weight_desc.attn_k;
            weight_desc_t& desc_v = _weight_desc.attn_v;
            buffer<u8> w_q;
            buffer<u8> w_k;
            buffer<u8> w_v;
            source(q4nx, npu, desc_q).load_weights(w_q, desc_q.format_name(layer_idx));
            source(q4nx, npu, desc_k).load_weights(w_k, desc_k.format_name(layer_idx));
            source(q4nx, npu, desc_v).load_weights(w_v, desc_v.format_name(layer_idx));
            LOG_VERBOSE(1, "Copying qkv weights to buffer");
            // each matrix goes to its own descriptor offset -- the band is
            // packed K|V|Q, not Q|K|V
            npu_cpy(w_ptr_begin + desc_k.offset, w_k, desc_k.shape[0], desc_k.dtype);
            npu_cpy(w_ptr_begin + desc_v.offset, w_v, desc_v.shape[0], desc_v.dtype);
            npu_cpy(w_ptr_begin + desc_q.offset, w_q, desc_q.shape[0], desc_q.dtype);
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
            source(q4nx, npu, desc).load_weights(w_o, desc.format_name(layer_idx));
            npu_cpy(w_ptr, w_o, desc.shape[0], desc.dtype);
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
            source(q4nx, npu, desc_up).load_weights(w_up, desc_up.format_name(layer_idx));
            source(q4nx, npu, desc_gate).load_weights(w_gate, desc_gate.format_name(layer_idx));
            // the slices below are the same bytes in the source and on the NPU: Q4_1 units carry no padding
            assert(!mixed || (desc_up.dtype == flm_q41 && desc_gate.dtype == flm_q41));

            assert(desc_up.shape[1] % UP_GATE_OUTDIM_REORDER == 0);
            assert(desc_gate.shape[1] % UP_GATE_OUTDIM_REORDER == 0);

            int slices = desc_up.shape[1] / UP_GATE_OUTDIM_REORDER;
            size_t slice_size = desc_up.get_size() / slices;
            for (int i = 0; i < slices; i++) {
                buffer<u8> up_slice = buffer<u8>(w_up.data() + i * slice_size, slice_size);
                buffer<u8> gate_slice = buffer<u8>(w_gate.data() + i * slice_size, slice_size);
                npu_cpy(w_ptr, up_slice, desc_up.shape[0], desc_up.dtype);
                w_ptr += slice_size;
                npu_cpy(w_ptr, gate_slice, desc_gate.shape[0], desc_gate.dtype);
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
            source(q4nx, npu, desc).load_weights(w_down, desc.format_name(layer_idx));
            assert(w_ptr + npu_bytes(desc) <= proj_weights.end());
            npu_cpy(w_ptr, w_down, desc.shape[0], desc.dtype);
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
    void load_head_weights(Q4NX& q4nx, Q4NX* npu, buffer<u8>& lm_head_weights, buffer<bf16>& final_rms_weight) {
        {
            LOG_VERBOSE(1, "Loading final rms weights");
            q4nx.load_weights(final_rms_weight, final_norm.name);
            final_rms_weight.sync_to_device();
            final_norm.load();
        }
        {
            LOG_VERBOSE(1, "Loading lm_head weights");
            buffer<u8> w_lm_head;
            source(q4nx, npu, lm_head).load_weights(w_lm_head, lm_head.name);
            npu_cpy(lm_head_weights.data(), w_lm_head, lm_head.shape[0], lm_head.dtype, (int)(vocab_size_padded / QXNX_ROW_BLOCK_SIZE));
            lm_head_weights.sync_to_device();
            lm_head.load();
        }
    }
};

#endif
