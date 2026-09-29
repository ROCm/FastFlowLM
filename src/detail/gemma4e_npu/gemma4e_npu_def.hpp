/// \file gemma4e_npu_def.hpp
/// \brief Gemma4e text backbone config + weight descriptors.
/// \author FastFlowLM Team
/// \note Owns everything that describes the model on disk and in device memory:
///       the config-derived dimensions, the per-layer weight descriptors (and
///       therefore the buffer layout the kernels read), and the weight loading
///       itself. Mirrors llama_desc / gemma4_12b_desc.
/// \note Gemma4e is a hybrid model in two independent ways:
///       - attention is sliding-window (SWA) except every `global_layer_period`-th
///         layer, which is full (global) attention; the two differ in head_dim.
///       - the last `num_kv_shared_layers` layers ("skip" layers) reuse the KV
///         cache of an earlier layer, so they carry no k/v projection and (when
///         `use_double_wide_mlp`) a twice-as-wide MLP.
///       That gives four distinct buffer layouts, indexed by gemma4e_layer_type_t:
///       [0]=swa, [1]=global, [2]=swa_skip, [3]=global_skip.
#pragma once
#include <cmath>
#include <iomanip>
#include <vector>
#include "lm_config.hpp"
#include "weight_desc.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"
#include "models/gemma4e/gemma4e_npu.hpp"

/// minimum tail padding (in bf16 elements) of the rope/rms buffer, kept in sync
/// with gemma4e_npu_sequence.hpp.
#ifndef MIN_BF16_PAD
#define MIN_BF16_PAD 32
#endif

/// \brief Descriptors of every weight of one decoder layer.
/// \note The order in which these are handed to the weight_container in
///       gemma4e_desc::_build_layer() IS the on-device layout, and both the
///       sequence generator (gemma4e_npu_sequence::gen_layer_seq) and the
///       dequant sequences depend on it: q/k/v contiguous, then o, then the
///       interleaved up/gate block, then down, then the three bf16 PLI
///       projections.
struct gemma4e_layer_weight_def {
    // ---- projections, quantized, live in the per-layer proj buffer ----
    weight_desc_t attn_q;
    weight_desc_t attn_k;                 //!< absent on skip layers, see has_kv
    weight_desc_t attn_v;                 //!< absent on skip layers, see has_kv
    weight_desc_t attn_output;
    weight_desc_t ffn_up_gate;            //!< up and gate interleaved, see UP_GATE_OUTDIM_REORDER
    weight_desc_t ffn_down;

    // ---- per-layer-input projections, bf16, same buffer, after the quantized block ----
    weight_desc_t pli_down_proj;
    weight_desc_t pli_gate_proj;
    weight_desc_t pli_up_proj;

    // ---- rms norms, bf16, live in the per-layer rms buffer ----
    weight_desc_t input_layer_norm;
    weight_desc_t post_attention_norm;
    weight_desc_t pre_feedforward_norm;
    weight_desc_t post_feedforward_norm;

    // ---- rope/rms buffer, bf16: [cos|sin, q_norm, k_norm, pli_embed, pli_norm, post_pli_norm, scale, pad] ----
    weight_desc_t rope_cos_sin;           //!< not a checkpoint tensor: host-computed per position
    weight_desc_t attn_q_norm;
    weight_desc_t attn_k_norm;
    weight_desc_t pli_embed;              //!< scratch slot the PLI path writes into, not a checkpoint tensor
    weight_desc_t pli_norm;               //!< model.per_layer_proj_norm.weight, shared by every layer
    weight_desc_t post_pli_norm;
    weight_desc_t layer_output_scale;

    // ---- fused / aliased views, never registered in a container ----
    // These carry no storage of their own: they name a span that already exists so
    // that the code moving it around can be handed a descriptor instead of an
    // offset + two dimensions. Their offsets are copied from the real descriptor
    // they alias once it has been placed.
    weight_desc_t attn_qkv;               //!< q(|k|v) as one block, what _move_weights ships
    weight_desc_t ffn_up;                 //!< the up half of ffn_up_gate, what the dequant kernel picks out
    weight_desc_t ffn_gate;               //!< the gate half of ffn_up_gate

    bool has_kv = true;                   //!< false on skip layers (they share an earlier layer's cache)

    /// \brief The projection weights, in device-buffer order.
    std::vector<weight_desc_t*> proj_all() {
        std::vector<weight_desc_t*> out = {&attn_q};
        if (has_kv) { out.push_back(&attn_k); out.push_back(&attn_v); }
        out.push_back(&attn_output);
        out.push_back(&ffn_up_gate);
        out.push_back(&ffn_down);
        out.push_back(&pli_down_proj);
        out.push_back(&pli_gate_proj);
        out.push_back(&pli_up_proj);
        return out;
    }

    /// \brief The four layer norms, in device-buffer order.
    std::vector<weight_desc_t*> rms_all() {
        return {&input_layer_norm, &post_attention_norm, &pre_feedforward_norm, &post_feedforward_norm};
    }

    /// \brief The rope/rms buffer entries, in device-buffer order.
    std::vector<weight_desc_t*> rope_all() {
        return {&rope_cos_sin, &attn_q_norm, &attn_k_norm,
                &pli_embed, &pli_norm, &post_pli_norm, &layer_output_scale};
    }

    /// \brief Every descriptor of the layer that owns storage.
    std::vector<weight_desc_t*> all() {
        std::vector<weight_desc_t*> out = proj_all();
        for (weight_desc_t* w : rms_all())  out.push_back(w);
        for (weight_desc_t* w : rope_all()) out.push_back(w);
        return out;
    }
};

/// \brief Gemma4e text backbone description (config.json + model.q4nx).
struct gemma4e_desc {
    /// up/gate are interleaved in slices of this many output rows
    static constexpr int UP_GATE_OUTDIM_REORDER = 512;
    /// \note Switching the body to another 4-bit dtype is a one-line change here:
    ///       every size, offset and DMA length below is derived from it.
    static constexpr flm_dtype_t PROJ_DTYPE    = flm_q41;
    static constexpr flm_dtype_t LM_HEAD_DTYPE = flm_q41;
    static constexpr flm_dtype_t PLI_DTYPE     = flm_bf16;

    /// gemma4e moves weights in 32x256 hardware blocks (unlike the 16x256 of gemma4-12b).
    static constexpr int QXNX_M = QXNX_ROW_BLOCK_SIZE;
    static constexpr int QXNX_K = QXNX_COL_BLOCK_SIZE;

    /// \brief Bytes of one 32x256 hardware block of a quantized dtype.
    static size_t block_bytes(flm_dtype_t dtype) {
        return get_quantization_byte_size((size_t)QXNX_M * QXNX_K, dtype);
    }

    // ---- dimensions (config.json) ----
    uint32_t num_hidden_layers = 0;
    uint32_t D = 0;                       //!< hidden_size
    uint32_t INTERMEDIATE_SIZE = 0;
    uint32_t PLI_D = 0;                   //!< hidden_size_per_layer_input
    uint32_t num_attention_heads = 0;
    uint32_t num_kv_heads = 0;
    uint32_t DH = 0;                      //!< global_head_dim
    uint32_t DQ = 0;
    uint32_t DK = 0;
    uint32_t DV = 0;
    uint32_t SWA_DH = 0;                  //!< head_dim
    uint32_t SWA_DQ = 0;
    uint32_t SWA_DK = 0;
    uint32_t SWA_DV = 0;
    uint32_t SLIDING_LENGTH = 0;
    uint32_t vocab_size = 0;
    uint32_t vocab_size_padded = 0;
    uint32_t num_kv_shared_layers = 0;
    uint32_t non_skip_layers = 0;
    uint32_t global_layer_period = 0;
    bool enable_double_wide_mlp = false;
    f32 final_logit_softcapping = 0.0f;

    std::vector<gemma4e_layer_type_t> layer_types;

    // ---- weight descriptors ----
    /// one representative layout per layer kind, indexed by gemma4e_layer_type_t
    gemma4e_layer_weight_def layer_defs[4];
    weight_desc_t final_norm;
    weight_desc_t lm_head;

    // ---- aggregated buffer byte sizes, indexed by gemma4e_layer_type_t ----
    size_t proj_weights_byte_size[4] = {0, 0, 0, 0};
    size_t rms_weights_byte_size[4]  = {0, 0, 0, 0};
    size_t rope_rms_byte_size[4]     = {0, 0, 0, 0};  //!< payload only, without MIN_BF16_PAD

    gemma4e_desc() {}
    gemma4e_desc(LM_Config config) { build(config); }

    /// \brief Whether layer `layer_idx` uses full (global) attention.
    inline bool is_global_layer_idx(int layer_idx) const {
        return ((layer_idx + 1) % (int)global_layer_period) == 0;
    }

    /// \brief The representative descriptor of a layer kind.
    inline gemma4e_layer_weight_def& weight_desc(gemma4e_layer_type_t type) {
        return layer_defs[int(type)];
    }
    /// \brief The representative descriptor of a layer, by index.
    inline gemma4e_layer_weight_def& weight_desc_of(int layer_idx) {
        return layer_defs[int(layer_types[layer_idx])];
    }

    // ---- layer-kind-specific dimensions ----
    inline uint32_t get_DH(gemma4e_layer_type_t t) const { return is_global_layer(t) ? DH : SWA_DH; }
    inline uint32_t get_DQ(gemma4e_layer_type_t t) const { return is_global_layer(t) ? DQ : SWA_DQ; }
    inline uint32_t get_DK(gemma4e_layer_type_t t) const { return is_global_layer(t) ? DK : SWA_DK; }
    inline uint32_t get_DV(gemma4e_layer_type_t t) const { return is_global_layer(t) ? DV : SWA_DV; }
    /// \brief MLP width of a layer kind: skip layers are twice as wide when enabled.
    inline uint32_t get_intermediate_size(gemma4e_layer_type_t t) const {
        return (is_skip_layer(t) && enable_double_wide_mlp) ? INTERMEDIATE_SIZE * 2 : INTERMEDIATE_SIZE;
    }
    /// \brief MLP width of the widest layer kind, which sizes the dequant buffers.
    inline uint32_t get_max_intermediate_size() const {
        return enable_double_wide_mlp ? INTERMEDIATE_SIZE * 2 : INTERMEDIATE_SIZE;
    }

    // ---- buffer sizes ----
    inline size_t get_proj_weights_byte_size(gemma4e_layer_type_t t) const { return proj_weights_byte_size[int(t)]; }
    /// \brief rms buffer size, in bf16 elements (input | post attn | pre ffn | post ffn).
    inline size_t get_rms_elems(gemma4e_layer_type_t t) const { return rms_weights_byte_size[int(t)] / sizeof(bf16); }
    /// \brief rope/rms buffer size, in bf16 elements, padding included.
    /// \note The layer scale is the last entry, and MIN_BF16_PAD elements are kept
    ///       behind it so that the kernel never reads past the buffer.
    inline size_t get_rope_rms_elems(gemma4e_layer_type_t t) {
        return rope_rms_byte_size[int(t)] / sizeof(bf16) + MIN_BF16_PAD;
    }
    /// \brief kv cache size of one layer, in bf16 elements.
    /// \note Sliding layers only ever keep `sliding_window` positions.
    inline size_t get_kv_cache_size(gemma4e_layer_type_t t, uint32_t MAX_L) const {
        return is_global_layer(t) ? (size_t)MAX_L * (DK + DV)
                                  : (size_t)SLIDING_LENGTH * (SWA_DK + SWA_DV);
    }
    inline size_t get_lm_head_w_size() { return lm_head.get_size(); }

    // ---- rope/rms buffer offsets, in bf16 elements ----
    inline uint32_t rope_elem_offset(weight_desc_t& w) const { return (uint32_t)(w.offset / sizeof(bf16)); }
    inline uint32_t get_q_norm_offset(gemma4e_layer_type_t t)     { return rope_elem_offset(weight_desc(t).attn_q_norm); }
    inline uint32_t get_k_norm_offset(gemma4e_layer_type_t t)     { return rope_elem_offset(weight_desc(t).attn_k_norm); }
    inline uint32_t get_pli_embed_offset(gemma4e_layer_type_t t)  { return rope_elem_offset(weight_desc(t).pli_embed); }
    inline uint32_t get_pli_norm_offset(gemma4e_layer_type_t t)   { return rope_elem_offset(weight_desc(t).pli_norm); }
    inline uint32_t get_post_pli_norm_offset(gemma4e_layer_type_t t) { return rope_elem_offset(weight_desc(t).post_pli_norm); }
    inline uint32_t get_layer_scale_offset(gemma4e_layer_type_t t){ return rope_elem_offset(weight_desc(t).layer_output_scale); }

    /// \brief Parse config.json and lay out every weight.
    inline void build(LM_Config& config) {
        const nlohmann::json& jc = config._json_config;

        uint32_t head_dim = 0, global_head_dim = 0;
        JSON_GET(num_hidden_layers,   jc, "num_hidden_layers",   0, uint32_t);
        JSON_GET(D,                   jc, "hidden_size",         0, uint32_t);
        JSON_GET(INTERMEDIATE_SIZE,   jc, "intermediate_size",   0, uint32_t);
        JSON_GET(num_attention_heads, jc, "num_attention_heads", 0, uint32_t);
        JSON_GET(num_kv_heads,        jc, "num_key_value_heads", 0, uint32_t);
        JSON_GET(head_dim,            jc, "head_dim",            0, uint32_t);
        JSON_GET(global_head_dim,     jc, "global_head_dim",     0, uint32_t);
        JSON_GET(SLIDING_LENGTH,      jc, "sliding_window",      0, uint32_t);
        JSON_GET(PLI_D,               jc, "hidden_size_per_layer_input", 0, uint32_t);
        JSON_GET(vocab_size,          jc, "vocab_size",          0, uint32_t);
        JSON_GET(num_kv_shared_layers, jc, "num_kv_shared_layers", 0, uint32_t);
        JSON_GET(final_logit_softcapping, jc, "final_logit_softcapping", 0.0f, f32);
        JSON_GET(enable_double_wide_mlp,  jc, "use_double_wide_mlp", false, bool);

        DH  = global_head_dim;
        DQ  = DH * num_attention_heads;
        DK  = DH * num_kv_heads;
        DV  = DK;
        SWA_DH = head_dim;
        SWA_DQ = SWA_DH * num_attention_heads;
        SWA_DK = SWA_DH * num_kv_heads;
        SWA_DV = SWA_DK;
        vocab_size_padded = (vocab_size + 1024 - 1) / 1024 * 1024;
        non_skip_layers = num_hidden_layers - num_kv_shared_layers;

        // The global-layer period is not in config.json; it follows from the model size.
        if (D == 1536)      global_layer_period = 5;   // E2B
        else if (D == 2560) global_layer_period = 6;   // E4B
        else throw std::runtime_error("gemma4e_desc: unsupported hidden size " + std::to_string(D));

        layer_types.resize(num_hidden_layers);
        for (uint32_t i = 0; i < num_hidden_layers; i++) {
            int type = 0;
            if (is_global_layer_idx(i))  type |= GEMMA4E_IS_GLOBAL_MASK;
            if (i >= non_skip_layers)    type |= GEMMA4E_IS_SKIP_MASK;
            layer_types[i] = static_cast<gemma4e_layer_type_t>(type);
        }

        DEBUG_BLOCK(1,
        std::cout << "================ gemma4e_desc ================" << std::endl;
        std::cout << std::left << std::setw(28) << "num_hidden_layers"     << " = " << num_hidden_layers << std::endl;
        std::cout << std::left << std::setw(28) << "D (hidden_size)"       << " = " << D << std::endl;
        std::cout << std::left << std::setw(28) << "PLI_D"                 << " = " << PLI_D << std::endl;
        std::cout << std::left << std::setw(28) << "INTERMEDIATE_SIZE"     << " = " << INTERMEDIATE_SIZE << std::endl;
        std::cout << std::left << std::setw(28) << "DH / DQ / DK / DV"     << " = " << DH << " / " << DQ << " / " << DK << " / " << DV << std::endl;
        std::cout << std::left << std::setw(28) << "SWA DH / DQ / DK / DV" << " = " << SWA_DH << " / " << SWA_DQ << " / " << SWA_DK << " / " << SWA_DV << std::endl;
        std::cout << std::left << std::setw(28) << "SLIDING_LENGTH"        << " = " << SLIDING_LENGTH << std::endl;
        std::cout << std::left << std::setw(28) << "global_layer_period"   << " = " << global_layer_period << std::endl;
        std::cout << std::left << std::setw(28) << "num_kv_shared_layers"  << " = " << num_kv_shared_layers << std::endl;
        std::cout << std::left << std::setw(28) << "use_double_wide_mlp"   << " = " << enable_double_wide_mlp << std::endl;
        std::cout << std::left << std::setw(28) << "final_logit_softcapping" << " = " << final_logit_softcapping << std::endl;
        std::cout << std::left << std::setw(28) << "vocab_size"            << " = " << vocab_size << " (" << vocab_size_padded << " padded)" << std::endl;
        std::cout << "=============================================" << std::endl;
        )

        final_norm = weight_desc_t(flm_bf16, {(int64_t)D}, "model.norm.weight");
        lm_head    = weight_desc_t(LM_HEAD_DTYPE, {(int64_t)D, (int64_t)vocab_size_padded}, "lm_head.weight");
        // Neither shares a buffer with anything, so they never go through a
        // weight_container and would never get `added` set. Mark them registered
        // here; their offset inside their own buffer is 0 by construction.
        final_norm.indp();
        lm_head.indp();

        for (int t = 0; t < 4; t++) _build_layer(static_cast<gemma4e_layer_type_t>(t));
    }

    /// \brief Copy a quantized weight into the device buffer in NPU block order.
    /// \param dst destination inside the per-layer projection buffer
    /// \param src the weight as stored in the q4nx file
    /// \param col the input dimension of the weight
    /// \param dtype the quantized dtype of the weight
    /// \param vertical_blocks how many 32-row bands are interleaved
    void reorder_cpy(u8* dst, buffer<u8>& src, const int col, flm_dtype_t dtype,
                     const int vertical_blocks = 2)
    {
        assert(is_quantize(dtype));
        const size_t a_block_size  = block_bytes(dtype);
        const int    blocks_per_row = col / QXNX_K;
        const int    rows = src.size() / a_block_size / blocks_per_row;

        u8* dst_ptr = dst;
        std::vector<u8*> src_ptr(vertical_blocks);
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

    /// \brief Load one decoder layer's weights from the checkpoint into its device buffers.
    /// \param layer_idx which layer to load
    /// \param q4nx the opened checkpoint
    /// \param proj_buffer the layer's projection buffer (quantized block + bf16 PLI block)
    /// \param rms_buffer the layer's four layer norms
    /// \param rope_rms_buffer the layer's rope/norm/scale buffer
    /// \param pli_gate_up_buffer the prefill-shaped copies of the PLI gate/up projections
    /// \param per_layer_norm model.per_layer_proj_norm.weight, shared by every layer
    /// \param layer_scale_out receives the layer output scale as a float
    /// \note Every destination is addressed through its descriptor's offset, so the
    ///       physical layout lives in _build_layer() alone.
    void load_layer_weights(int layer_idx, Q4NX& q4nx,
                            buffer<u8>& proj_buffer,
                            buffer<bf16>& rms_buffer,
                            buffer<bf16>& rope_rms_buffer,
                            buffer<bf16>& pli_gate_up_buffer,
                            buffer<bf16>& per_layer_norm,
                            float& layer_scale_out)
    {
        const gemma4e_layer_type_t type = layer_types[layer_idx];
        gemma4e_layer_weight_def& L = weight_desc(type);
        u8* base = proj_buffer.data();

        // ---- q (k, v) ----
        {
            buffer<u8> w;
            q4nx.load_weights(w, L.attn_q.format_name(layer_idx));
            reorder_cpy(base + L.attn_q.offset, w, D, L.attn_q.dtype);
            L.attn_q.load();
        }
        if (L.has_kv) {
            for (weight_desc_t* desc : {&L.attn_k, &L.attn_v}) {
                buffer<u8> w;
                q4nx.load_weights(w, desc->format_name(layer_idx));
                reorder_cpy(base + desc->offset, w, D, desc->dtype);
                desc->load();
            }
        }
        // ---- o ----
        {
            buffer<u8> w;
            q4nx.load_weights(w, L.attn_output.format_name(layer_idx));
            reorder_cpy(base + L.attn_output.offset, w, get_DQ(type), L.attn_output.dtype);
            L.attn_output.load();
        }
        // ---- up / gate, interleaved in slices of UP_GATE_OUTDIM_REORDER output rows ----
        {
            buffer<u8> w_up, w_gate;
            q4nx.load_weights(w_up,   L.ffn_up.format_name(layer_idx));
            q4nx.load_weights(w_gate, L.ffn_gate.format_name(layer_idx));

            const size_t chunk_size = get_quantization_byte_size((size_t)UP_GATE_OUTDIM_REORDER * D,
                                                                 L.ffn_up_gate.dtype);
            const size_t half_size  = L.ffn_up_gate.get_size() / 2;
            const size_t phases     = half_size / chunk_size;
            assert(phases * chunk_size == half_size && "up/gate must split into whole slices");

            tensor_2d<u8> up_tensor(w_up, chunk_size, 0);
            tensor_2d<u8> gate_tensor(w_gate, chunk_size, 0);
            u8* w_ptr = base + L.ffn_up_gate.offset;
            for (size_t i = 0; i < phases; i++) {
                LOG_VERBOSE(1, "Copying up and gate weights to buffer, phase " << i + 1 << "/" << phases);
                reorder_cpy(w_ptr, up_tensor[i], D, L.ffn_up_gate.dtype);
                w_ptr += chunk_size;
                reorder_cpy(w_ptr, gate_tensor[i], D, L.ffn_up_gate.dtype);
                w_ptr += chunk_size;
            }
            L.ffn_up_gate.load();
            L.ffn_up.load();
            L.ffn_gate.load();
        }
        // ---- down ----
        {
            buffer<u8> w;
            q4nx.load_weights(w, L.ffn_down.format_name(layer_idx));
            reorder_cpy(base + L.ffn_down.offset, w, get_intermediate_size(type), L.ffn_down.dtype);
            L.ffn_down.load();
        }
        // ---- per-layer-input projections, plain bf16 ----
        for (weight_desc_t* desc : {&L.pli_down_proj, &L.pli_gate_proj, &L.pli_up_proj}) {
            buffer<bf16> w;
            q4nx.load_weights(w, desc->format_name(layer_idx));
            memcpy(base + desc->offset, w.data(), desc->get_size());
            desc->load();
        }

        // ---- the four layer norms ----
        for (weight_desc_t* desc : L.rms_all()) {
            buffer<bf16> w;
            q4nx.load_weights(w, desc->format_name(layer_idx));
            memcpy(desc->locate_myself<bf16>(rms_buffer).data(), w.data(), desc->get_size());
            desc->load();
        }

        // ---- rope/rms buffer ----
        // rope_cos_sin and pli_embed are scratch slots the runtime fills per position,
        // so only the four checkpoint-backed entries are read here.
        for (weight_desc_t* desc : {&L.attn_q_norm, &L.attn_k_norm, &L.post_pli_norm}) {
            buffer<bf16> w;
            q4nx.load_weights(w, desc->format_name(layer_idx));
            memcpy(desc->locate_myself<bf16>(rope_rms_buffer).data(), w.data(), desc->get_size());
            desc->load();
        }
        memcpy(L.pli_norm.locate_myself<bf16>(rope_rms_buffer).data(),
               per_layer_norm.data(), L.pli_norm.get_size());
        L.pli_norm.load();
        {
            buffer<bf16> w;
            q4nx.load_weights(w, L.layer_output_scale.format_name(layer_idx));
            layer_scale_out = (float)w[0];
            // The kernel broadcasts the scale, but it still reads a full vector's
            // worth behind it, so zero the padding that follows.
            bf16* scale_ptr = L.layer_output_scale.locate_myself<bf16>(rope_rms_buffer).data();
            memset(scale_ptr, 0, (1 + MIN_BF16_PAD) * sizeof(bf16));
            scale_ptr[0] = w[0];
            L.layer_output_scale.load();
        }

        // ---- prefill-shaped copies of the PLI gate/up projections ----
        {
            buffer<bf16> w_gate, w_up;
            q4nx.load_weights(w_gate, "model.layers." + std::to_string(layer_idx) + ".inp_gate.weight_prefill");
            q4nx.load_weights(w_up,   "model.layers." + std::to_string(layer_idx) + ".per_layer_projection.weight_prefill");
            memcpy(pli_gate_up_buffer.data(),                     w_gate.data(), (size_t)PLI_D * D * sizeof(bf16));
            memcpy(pli_gate_up_buffer.data() + (size_t)PLI_D * D, w_up.data(),   (size_t)PLI_D * D * sizeof(bf16));
        }
    }

    /// \brief Load the weights that do not belong to any single layer.
    /// \param q4nx the opened checkpoint
    /// \param lm_head_buffer destination of the (reordered) lm head
    /// \param final_norm_dst destination of model.norm.weight
    /// \param pli_down_buffer destination of the prefill-shaped per-layer down projection
    void load_head_weights(Q4NX& q4nx,
                           buffer<u8>& lm_head_buffer,
                           bf16* final_norm_dst,
                           buffer<bf16>& pli_down_buffer)
    {
        {
            buffer<bf16> w;
            q4nx.load_weights(w, final_norm.name);
            memcpy(final_norm_dst, w.data(), final_norm.get_size());
            final_norm.load();
        }
        {
            buffer<u8> w;
            q4nx.load_weights(w, lm_head.name);
            // The lm head spans four columns of cores rather than two.
            reorder_cpy(lm_head_buffer.data(), w, D, lm_head.dtype, 4);
            lm_head.load();
        }
        {
            buffer<bf16> w;
            q4nx.load_weights(w, "model.per_layer_model_proj.weight_prefill");
            memcpy(pli_down_buffer.data(), w.data(),
                   (size_t)num_hidden_layers * D * PLI_D * sizeof(bf16));
        }
    }

private:
    /// \brief Lay out one layer kind's three buffers and name every tensor.
    /// \param type the layer kind whose layout is being built
    /// \note The add_weight() call order is the physical layout; see the note on
    ///       gemma4e_layer_weight_def.
    inline void _build_layer(gemma4e_layer_type_t type) {
        gemma4e_layer_weight_def& L = layer_defs[int(type)];

        const int64_t _DH = get_DH(type);
        const int64_t _DQ = get_DQ(type);
        const int64_t _DK = get_DK(type);
        const int64_t _DV = get_DV(type);
        const int64_t _IS = get_intermediate_size(type);
        const int64_t d   = D;
        const int64_t pli = PLI_D;

        L.has_kv = !is_skip_layer(type);

        L.attn_q      = weight_desc_t(PROJ_DTYPE, {d, _DQ}, "model.layers.%d.self_attn.q_proj.weight");
        L.attn_k      = weight_desc_t(PROJ_DTYPE, {d, _DK}, "model.layers.%d.self_attn.k_proj.weight");
        L.attn_v      = weight_desc_t(PROJ_DTYPE, {d, _DV}, "model.layers.%d.self_attn.v_proj.weight");
        L.attn_output = weight_desc_t(PROJ_DTYPE, {_DQ, d}, "model.layers.%d.self_attn.o_proj.weight");
        L.ffn_up_gate = weight_desc_t(PROJ_DTYPE, {d, 2 * _IS}, "model.layers.%d.mlp.{up,gate}_proj.weight");
        L.ffn_down    = weight_desc_t(PROJ_DTYPE, {_IS, d}, "model.layers.%d.mlp.down_proj.weight");
        L.pli_down_proj = weight_desc_t(PLI_DTYPE, {pli, d}, "model.per_layer_model_proj.weight_layer%d");
        L.pli_gate_proj = weight_desc_t(PLI_DTYPE, {pli, d}, "model.layers.%d.inp_gate.weight");
        L.pli_up_proj   = weight_desc_t(PLI_DTYPE, {pli, d}, "model.layers.%d.per_layer_projection.weight");

        L.input_layer_norm      = weight_desc_t(flm_bf16, {d}, "model.layers.%d.input_layernorm.weight");
        L.post_attention_norm   = weight_desc_t(flm_bf16, {d}, "model.layers.%d.post_attention_layernorm.weight");
        L.pre_feedforward_norm  = weight_desc_t(flm_bf16, {d}, "model.layers.%d.pre_feedforward_layernorm.weight");
        L.post_feedforward_norm = weight_desc_t(flm_bf16, {d}, "model.layers.%d.post_feedforward_layernorm.weight");

        L.rope_cos_sin       = weight_desc_t(flm_bf16, {_DH}, "<rope cos|sin>");
        L.attn_q_norm        = weight_desc_t(flm_bf16, {_DH}, "model.layers.%d.self_attn.q_norm.weight");
        L.attn_k_norm        = weight_desc_t(flm_bf16, {_DH}, "model.layers.%d.self_attn.k_norm.weight");
        L.pli_embed          = weight_desc_t(flm_bf16, {pli}, "<per layer input embedding>");
        L.pli_norm           = weight_desc_t(flm_bf16, {pli}, "model.per_layer_proj_norm.weight");
        L.post_pli_norm      = weight_desc_t(flm_bf16, {d},   "model.layers.%d.post_layernorm.weight");
        L.layer_output_scale = weight_desc_t(flm_bf16, {1},   "model.layers.%d.layer_output_scale.weight");

        weight_container proj, rms, rope;
        for (weight_desc_t* w : L.proj_all()) proj.add_weight(*w);
        for (weight_desc_t* w : L.rms_all())  rms.add_weight(*w);
        for (weight_desc_t* w : L.rope_all()) rope.add_weight(*w);

        proj_weights_byte_size[int(type)] = proj.get_size();
        rms_weights_byte_size[int(type)]  = rms.get_size();
        rope_rms_byte_size[int(type)]     = rope.get_size();

        // Aliased views over regions that are already placed above. q/k/v are
        // contiguous and are shipped and dequantized as one block; up/gate share
        // one interleaved region that the dequant kernel splits by output mode.
        L.attn_qkv = weight_desc_t(PROJ_DTYPE, {d, L.has_kv ? (_DQ + _DK + _DV) : _DQ},
                                   "<fused qkv projection>");
        L.attn_qkv.offset = L.attn_q.offset;
        L.attn_qkv.indp();

        L.ffn_up   = weight_desc_t(PROJ_DTYPE, {d, _IS}, "model.layers.%d.mlp.up_proj.weight");
        L.ffn_gate = weight_desc_t(PROJ_DTYPE, {d, _IS}, "model.layers.%d.mlp.gate_proj.weight");
        L.ffn_up.offset = L.ffn_gate.offset = L.ffn_up_gate.offset;
        L.ffn_up.indp();
        L.ffn_gate.indp();
    }
};
