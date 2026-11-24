// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file gemma4_12b_npu_def.hpp
/// \brief Gemma4-12B text backbone config + weight descriptors.
/// \author FastFlowLM Team
/// \note Owns everything that describes the model on disk and in device memory:
///       the config-derived dimensions, the per-layer weight descriptors (and
///       therefore the buffer layout the decode kernel reads), and the weight
///       loading itself. Mirrors qwen3_5_omni_thinker_desc.
/// \note Gemma4-12B is a hybrid model: every layer is a sliding-window (SWA)
///       layer except every `sliding_window_pattern`-th one, which is a full
///       (global) attention layer. The two kinds differ in head_dim, kv head
///       count and RoPE parameters, and the global layers do not store a
///       v_proj at all (attention_k_eq_v: K is reused as V).
#pragma once
#include <cmath>
#include <iomanip>
#include <vector>
#include "lm_config.hpp"
#include "weight_desc.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"
#include "models/gemma4_12b/gemma4_12b_npu.hpp"

/// minimum tail padding (in bf16 elements) of the rope/rms buffer, kept in sync
/// with gemma4_12b_npu_sequence.hpp.
#ifndef MIN_BF16_PAD
#define MIN_BF16_PAD 32
#endif

/// \brief Descriptors of every weight of one decoder layer.
/// \note The order in which these are handed to the weight_container in
///       gemma4_12b_desc::build() IS the on-device layout, and the decode
///       sequence generator (gemma4_12b_npu_sequence::gen_layer_seq) depends
///       on it: q/k/(v) must be contiguous, then o, then the interleaved
///       up/gate block, then down.
struct gemma4_12b_layer_weight_def {
    // ---- projections, q4_0, live in the per-layer proj buffer ----
    weight_desc_t attn_q;
    weight_desc_t attn_k;
    weight_desc_t attn_v;                 //!< only on SWA layers, see has_v_proj
    weight_desc_t attn_output;
    weight_desc_t ffn_up;                 //!< up and gate share one region, interleaved
    weight_desc_t ffn_gate;               //!< in slices of UP_GATE_OUTDIM_REORDER rows
    weight_desc_t ffn_down;

    // ---- rms norms, bf16, live in the per-layer rms buffer ----
    weight_desc_t input_layer_norm;
    weight_desc_t post_attention_norm;
    weight_desc_t pre_feedforward_norm;
    weight_desc_t post_feedforward_norm;

    // ---- rope/rms buffer, bf16: [cos|sin, q_norm, k_norm, layer_scale, pad] ----
    weight_desc_t rope_cos_sin;           //!< not a checkpoint tensor: host-computed per position
    weight_desc_t attn_q_norm;
    weight_desc_t attn_k_norm;
    weight_desc_t layer_output_scale;

    bool has_v_proj = true;               //!< false on global layers (attention_k_eq_v)

    /// \brief The projection weights, in device-buffer order.
    std::vector<weight_desc_t*> proj_all() {
        if (has_v_proj) {
            return {&attn_q, &attn_k, &attn_v, &attn_output, &ffn_up, &ffn_gate, &ffn_down};
        }
        return {&attn_q, &attn_k, &attn_output, &ffn_up, &ffn_gate, &ffn_down};
    }

    /// \brief The four layer norms, in device-buffer order.
    std::vector<weight_desc_t*> rms_all() {
        return {&input_layer_norm, &post_attention_norm, &pre_feedforward_norm, &post_feedforward_norm};
    }

    /// \brief The rope/rms buffer entries, in device-buffer order.
    std::vector<weight_desc_t*> rope_all() {
        return {&rope_cos_sin, &attn_q_norm, &attn_k_norm, &layer_output_scale};
    }

    /// \brief Every descriptor of the layer.
    std::vector<weight_desc_t*> all() {
        std::vector<weight_desc_t*> out = proj_all();
        for (weight_desc_t* w : rms_all())  out.push_back(w);
        for (weight_desc_t* w : rope_all()) out.push_back(w);
        return out;
    }
};

/// \brief Gemma4-12B text backbone description (config.json + model.q4nx).
struct gemma4_12b_desc {
    /// up/gate are interleaved in slices of this many output rows
    static constexpr int UP_GATE_OUTDIM_REORDER = 512;
    static constexpr flm_dtype_t PROJ_DTYPE = flm_q40;
    static constexpr flm_dtype_t LM_HEAD_DTYPE = flm_q80;

    static constexpr int Q4NX_M = 16;
    static constexpr int Q4NX_K = QXNX_COL_BLOCK_SIZE;
    static constexpr int Q4NX_BLOCK_BYTES = Q4NX_M * Q4NX_K * 9 / 16;  //!< 2304 B, q4_0 only
    static constexpr int Q8NX_BLOCK_BYTES = Q4NX_M * Q4NX_K * 17 / 16; //!< 4352 B, q8_0 only

    /// \brief Bytes of one 16x256 hardware block of a quantized dtype.
    static constexpr int block_bytes(flm_dtype_t dtype) {
        return (dtype & IS_Q8_MASK) ? Q8NX_BLOCK_BYTES : Q4NX_BLOCK_BYTES;
    }

    // ---- dimensions (config.json) ----
    uint32_t num_hidden_layers = 0;
    uint32_t D = 0;                       //!< hidden_size
    uint32_t intermediate_size = 0;
    uint32_t num_attention_heads = 0;
    uint32_t num_kv_heads = 0;            //!< global layers (num_global_key_value_heads)
    uint32_t swa_num_kv_heads = 0;        //!< sliding-window layers (num_key_value_heads)
    uint32_t DH = 0;                      //!< global_head_dim
    uint32_t DQ = 0;
    uint32_t DK = 0;
    uint32_t DV = 0;                      //!< == DK, V is K on global layers
    uint32_t SWA_DH = 0;                  //!< head_dim
    uint32_t SWA_DQ = 0;
    uint32_t SWA_DK = 0;
    uint32_t SWA_DV = 0;
    uint32_t sliding_window = 0;
    uint32_t sliding_window_pattern = 0;  //!< one global layer every N layers
    uint32_t vocab_size = 0;
    uint32_t vocab_size_padded = 0;
    uint32_t num_kv_shared_layers = 0;
    bool attention_k_eq_v = false;
    bool tie_word_embeddings = false;
    f32 rms_norm_eps = 0.0f;
    f32 final_logit_softcapping = 0.0f;

    // ---- RoPE (config.json rope_parameters) ----
    f32 rope_theta_global = 0.0f;
    f32 rope_theta_swa = 0.0f;
    f32 partial_rotary_factor_global = 1.0f;
    f32 partial_rotary_factor_swa = 1.0f;
    uint32_t RoPE_DIM = 0;                //!< rotated dims of a global head
    uint32_t SWA_RoPE_DIM = 0;            //!< rotated dims of a sliding-window head
    /// inv_freq[j] for j < _DH/2; zero past the rotated part.
    /// \note The frequencies are laid out over the FULL head_dim ("proportional"
    ///       rope_type): inv_freq[j] = theta^(-2j/_DH), truncated to the first
    ///       RoPE_DIM/2 entries. This reproduces rope_freqs.weight of the
    ///       checkpoint, which is 1.0 for the kept entries and 1e30 (-> 0) for
    ///       the rest.
    std::vector<f32> inv_freq_global;
    std::vector<f32> inv_freq_swa;

    // ---- multi modal token ids ----
    int image_token_id = 0;
    int audio_token_id = 0;
    int video_token_id = 0;
    int boi_token_id = 0;
    int eoi_token_id = 0;
    int boa_token_id = 0;
    int eoa_token_id = 0;

    // ---- weight descriptors ----
    // all layers of a kind share the same layout, so one representative each
    gemma4_12b_layer_weight_def swa_weight_desc;
    gemma4_12b_layer_weight_def global_weight_desc;
    weight_desc_t embed_tokens;
    weight_desc_t final_norm;
    weight_desc_t lm_head;

    // ---- aggregated buffer byte sizes ----
    size_t swa_proj_weights_byte_size = 0;
    size_t global_proj_weights_byte_size = 0;
    size_t rms_weights_byte_size = 0;         //!< identical for both layer kinds
    size_t swa_rope_rms_byte_size = 0;        //!< payload only, without MIN_BF16_PAD
    size_t global_rope_rms_byte_size = 0;

    gemma4_12b_desc() {}
    gemma4_12b_desc(LM_Config config) { build(config); }

    /// \brief Whether layer `layer_idx` uses full (global) attention.
    inline bool is_global_layer_idx(int layer_idx) const {
        return (layer_idx % (int)sliding_window_pattern) == (int)(sliding_window_pattern - 1);
    }

    /// \brief The layer kind of layer `layer_idx`.
    inline gemma4_12b_layer_type_t layer_type(int layer_idx) const {
        return is_global_layer_idx(layer_idx) ? gemma4_12b_global_layer : gemma4_12b_swa_layer;
    }

    /// \brief The representative descriptor of a layer kind.
    inline gemma4_12b_layer_weight_def& weight_desc(gemma4_12b_layer_type_t type) {
        return is_global_layer(type) ? global_weight_desc : swa_weight_desc;
    }

    // ---- buffer sizes ----
    inline size_t get_proj_weights_byte_size(gemma4_12b_layer_type_t type) const {
        return is_global_layer(type) ? global_proj_weights_byte_size : swa_proj_weights_byte_size;
    }
    /// \brief rms buffer size, in bf16 elements (input | post attn | pre ffn | post ffn).
    inline size_t get_rms_elems() const { return rms_weights_byte_size / sizeof(bf16); }
    /// \brief rope/rms buffer size, in bf16 elements, padding included.
    /// \note The layer scale is the last entry, and MIN_BF16_PAD elements are kept
    ///       behind it so that the kernel never reads past the buffer.
    inline size_t get_rope_rms_elems(gemma4_12b_layer_type_t type) {
        return get_layer_scale_offset(type) + MIN_BF16_PAD;
    }
    /// \brief kv cache size of one layer, in bf16 elements.
    /// \note Sliding layers only ever keep `sliding_window` positions.
    inline size_t get_kv_cache_size(gemma4_12b_layer_type_t type, uint32_t MAX_L) const {
        return is_global_layer(type) ? (size_t)MAX_L * (DK + DV)
                                     : (size_t)sliding_window * (SWA_DK + SWA_DV);
    }
    inline size_t get_lm_head_w_size() {
        return get_quantization_byte_size((size_t)vocab_size_padded * D, lm_head.dtype);
    }

    // ---- proj buffer offsets, in bf16 elements (what the sequence generator wants) ----
    inline uint32_t get_qkv_offset(gemma4_12b_layer_type_t type) {
        return (uint32_t)(weight_desc(type).attn_q.offset / sizeof(bf16));
    }
    inline uint32_t get_o_offset(gemma4_12b_layer_type_t type) {
        return (uint32_t)(weight_desc(type).attn_output.offset / sizeof(bf16));
    }
    inline uint32_t get_upgate_offset(gemma4_12b_layer_type_t type) {
        return (uint32_t)(weight_desc(type).ffn_up.offset / sizeof(bf16));
    }
    inline uint32_t get_down_offset(gemma4_12b_layer_type_t type) {
        return (uint32_t)(weight_desc(type).ffn_down.offset / sizeof(bf16));
    }

    // ---- rope/rms buffer offsets, in bf16 elements ----
    inline uint32_t get_rope_cos_offset(gemma4_12b_layer_type_t type) {
        return (uint32_t)(weight_desc(type).rope_cos_sin.offset / sizeof(bf16));
    }
    inline uint32_t get_rope_sin_offset(gemma4_12b_layer_type_t type) {
        return get_rope_cos_offset(type) + (is_global_layer(type) ? DH : SWA_DH) / 2;
    }
    inline uint32_t get_q_norm_offset(gemma4_12b_layer_type_t type) {
        return (uint32_t)(weight_desc(type).attn_q_norm.offset / sizeof(bf16));
    }
    inline uint32_t get_k_norm_offset(gemma4_12b_layer_type_t type) {
        return (uint32_t)(weight_desc(type).attn_k_norm.offset / sizeof(bf16));
    }
    inline uint32_t get_layer_scale_offset(gemma4_12b_layer_type_t type) {
        return (uint32_t)(weight_desc(type).layer_output_scale.offset / sizeof(bf16));
    }

    /// \brief The RoPE frequencies of a layer kind (_DH/2 entries).
    inline const std::vector<f32>& inv_freq(gemma4_12b_layer_type_t type) const {
        return is_global_layer(type) ? inv_freq_global : inv_freq_swa;
    }

    /// \brief Parse config.json and lay out every weight.
    inline void build(LM_Config& config) {
        const nlohmann::json& jc = config._json_config;

        uint32_t head_dim = 0, global_head_dim = 0;
        JSON_GET(num_hidden_layers,   jc, "num_hidden_layers",       0, uint32_t);
        JSON_GET(D,                   jc, "hidden_size",             0, uint32_t);
        JSON_GET(intermediate_size,   jc, "intermediate_size",       0, uint32_t);
        JSON_GET(num_attention_heads, jc, "num_attention_heads",     0, uint32_t);
        JSON_GET(swa_num_kv_heads,    jc, "num_key_value_heads",     0, uint32_t);
        JSON_GET(num_kv_heads,        jc, "num_global_key_value_heads", 0, uint32_t);
        JSON_GET(head_dim,            jc, "head_dim",                0, uint32_t);
        JSON_GET(global_head_dim,     jc, "global_head_dim",         0, uint32_t);
        JSON_GET(sliding_window,      jc, "sliding_window",          0, uint32_t);
        JSON_GET(sliding_window_pattern, jc, "sliding_window_pattern", 6, uint32_t);
        JSON_GET(vocab_size,          jc, "vocab_size",              0, uint32_t);
        JSON_GET(num_kv_shared_layers, jc, "num_kv_shared_layers",   0, uint32_t);
        JSON_GET(rms_norm_eps,        jc, "rms_norm_eps",            0.0f, f32);
        JSON_GET(final_logit_softcapping, jc, "final_logit_softcapping", 0.0f, f32);
        JSON_GET(attention_k_eq_v,    jc, "attention_k_eq_v",        false, bool);
        JSON_GET(tie_word_embeddings, jc, "tie_word_embeddings",     false, bool);

        JSON_GET(image_token_id, jc, "image_token_id", 0, int);
        JSON_GET(audio_token_id, jc, "audio_token_id", 0, int);
        JSON_GET(video_token_id, jc, "video_token_id", 0, int);
        JSON_GET(boi_token_id,   jc, "boi_token_id",   0, int);
        JSON_GET(eoi_token_id,   jc, "eoi_token_id",   0, int);
        JSON_GET(boa_token_id,   jc, "boa_token_id",   0, int);
        JSON_GET(eoa_token_id,   jc, "eoa_token_id",   0, int);

        const nlohmann::json& rope_params = cfg_sub(jc, "rope_parameters");
        const nlohmann::json& rope_full   = cfg_sub(rope_params, "full_attention");
        const nlohmann::json& rope_swa    = cfg_sub(rope_params, "sliding_attention");
        JSON_GET(rope_theta_global, rope_full, "rope_theta", 1000000.0f, f32);
        JSON_GET(rope_theta_swa,    rope_swa,  "rope_theta", 10000.0f, f32);
        JSON_GET(partial_rotary_factor_global, rope_full, "partial_rotary_factor", 1.0f, f32);
        JSON_GET(partial_rotary_factor_swa,    rope_swa,  "partial_rotary_factor", 1.0f, f32);

        DH  = global_head_dim;
        DQ  = DH * num_attention_heads;
        DK  = DH * num_kv_heads;
        DV  = DK;                          // K is reused as V on global layers
        SWA_DH = head_dim;
        SWA_DQ = SWA_DH * num_attention_heads;
        SWA_DK = SWA_DH * swa_num_kv_heads;
        SWA_DV = SWA_DK;
        RoPE_DIM     = (uint32_t)(DH * partial_rotary_factor_global);
        SWA_RoPE_DIM = (uint32_t)(SWA_DH * partial_rotary_factor_swa);
        vocab_size_padded = (vocab_size + 1024 - 1) / 1024 * 1024;

        _build_inv_freq(inv_freq_global, DH, RoPE_DIM, rope_theta_global);
        _build_inv_freq(inv_freq_swa, SWA_DH, SWA_RoPE_DIM, rope_theta_swa);

        DEBUG_BLOCK(1,
        std::cout << "============== gemma4_12b_desc ==============" << std::endl;
        std::cout << std::left << std::setw(28) << "num_hidden_layers"       << " = " << num_hidden_layers << std::endl;
        std::cout << std::left << std::setw(28) << "D (hidden_size)"         << " = " << D << std::endl;
        std::cout << std::left << std::setw(28) << "intermediate_size"       << " = " << intermediate_size << std::endl;
        std::cout << std::left << std::setw(28) << "num_attention_heads"     << " = " << num_attention_heads << std::endl;
        std::cout << std::left << std::setw(28) << "DH / DQ / DK / DV"       << " = " << DH << " / " << DQ << " / " << DK << " / " << DV << std::endl;
        std::cout << std::left << std::setw(28) << "SWA DH / DQ / DK / DV"   << " = " << SWA_DH << " / " << SWA_DQ << " / " << SWA_DK << " / " << SWA_DV << std::endl;
        std::cout << std::left << std::setw(28) << "num_kv_heads"            << " = " << num_kv_heads << std::endl;
        std::cout << std::left << std::setw(28) << "swa_num_kv_heads"        << " = " << swa_num_kv_heads << std::endl;
        std::cout << std::left << std::setw(28) << "sliding_window"          << " = " << sliding_window << std::endl;
        std::cout << std::left << std::setw(28) << "sliding_window_pattern"  << " = " << sliding_window_pattern << std::endl;
        std::cout << std::left << std::setw(28) << "RoPE_DIM / SWA_RoPE_DIM" << " = " << RoPE_DIM << " / " << SWA_RoPE_DIM << std::endl;
        std::cout << std::left << std::setw(28) << "rope_theta (full / swa)" << " = " << rope_theta_global << " / " << rope_theta_swa << std::endl;
        std::cout << std::left << std::setw(28) << "rms_norm_eps"            << " = " << rms_norm_eps << std::endl;
        std::cout << std::left << std::setw(28) << "final_logit_softcapping" << " = " << final_logit_softcapping << std::endl;
        std::cout << std::left << std::setw(28) << "attention_k_eq_v"        << " = " << attention_k_eq_v << std::endl;
        std::cout << std::left << std::setw(28) << "tie_word_embeddings"     << " = " << tie_word_embeddings << std::endl;
        std::cout << std::left << std::setw(28) << "vocab_size"              << " = " << vocab_size << " (" << vocab_size_padded << " padded)" << std::endl;
        std::cout << "=============================================" << std::endl;
        )

        embed_tokens = weight_desc_t(flm_bf16, {(int64_t)vocab_size, (int64_t)D}, "model.embed_tokens.weight");
        final_norm   = weight_desc_t(flm_bf16, {(int64_t)D}, "model.norm.weight");
        lm_head      = weight_desc_t(LM_HEAD_DTYPE, {(int64_t)D, (int64_t)vocab_size_padded}, "lm_head.weight");
        // These three do not share a buffer with anything, so they never go through a
        // weight_container and would never get `added` set. Mark them registered here,
        // their offset inside their own buffer is 0 by construction.
        embed_tokens.indp();
        final_norm.indp();
        lm_head.indp();

        _build_layer(swa_weight_desc, gemma4_12b_swa_layer);
        _build_layer(global_weight_desc, gemma4_12b_global_layer);
    }

    /// \brief Copy a quantized weight into the device buffer in NPU block order.
    /// \param dst destination inside the per-layer projection buffer
    /// \param src the weight as stored in the q4nx file
    /// \param col the input dimension of the weight
    /// \param dtype the quantized dtype of the weight
    /// \param vertical_blocks how many 16-row bands are interleaved
    void reorder_cpy(u8* dst, buffer<u8>& src, const int col, flm_dtype_t dtype,
                     const int vertical_blocks = 2)
    {
        assert(is_quantize(dtype));
        // The body of the model is q4_0 (2304 B blocks) and lm_head.weight is q8_0
        // (4352 B); those are the only two block sizes block_bytes() knows about, and a
        // wrong one straddles hardware blocks silently, so refuse anything else outright.
        assert((dtype == flm_q40 || dtype == flm_q80) &&
               "gemma4-12b stores q4_0 (body) or q8_0 (lm_head) only");

        const int a_block_size = block_bytes(dtype);
        const int blocks_per_row = col / Q4NX_K;
        const int rows = src.size() / a_block_size / blocks_per_row;

        // Vertical block reorder at a_block_size granularity
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

    /// \brief Copy the lm head into its device buffer in NPU block order.
    /// \param dst destination lm head buffer
    /// \param src the weight as stored in the q4nx file
    /// \param col the input dimension of the weight (D)
    /// \param dtype the quantized dtype of the weight (q8_0)
    /// \param vertical_blocks how many 16-row bands one column of cores covers (4)
    /// \note The lm head does not use the generic reorder_cpy above because a
    ///       column's weights arrive on two shim MM2S channels, one per pair of
    ///       cores, instead of one. Each column-round therefore has to be two
    ///       contiguous halves in DDR rather than one run with all four cores'
    ///       blocks interleaved: rows r+0/r+1 of the group for the stream that
    ///       feeds cores 0 and 1, rows r+2/r+3 for the stream that feeds cores 2
    ///       and 3. Within a half the order is still k-major, one block per core,
    ///       because that is exactly one memtile slot. Must stay in sync with
    ///       lm_head_memtile in FLM_Xclbin/Gemma4_12B_QAT/lm_head_npu_bin/iron/q4k_mm.py
    ///       and with gen_lm_head_seq's W_PER_STREAM.
    void reorder_cpy_lm_head(u8* dst, buffer<u8>& src, const int col, flm_dtype_t dtype,
                             const int vertical_blocks = 4)
    {
        assert(is_quantize(dtype));
        assert(vertical_blocks % 2 == 0 && "the two weight streams split the band evenly");
        assert((dtype == flm_q40 || dtype == flm_q80) &&
               "gemma4-12b stores q4_0 (body) or q8_0 (lm_head) only");

        const int a_block_size = block_bytes(dtype);
        const int blocks_per_row = col / Q4NX_K;
        const int rows = src.size() / a_block_size / blocks_per_row;
        const int rows_per_stream = vertical_blocks / 2;

        u8* dst_ptr = dst;
        for (int r = 0; r < rows; r += vertical_blocks)
        {
            for (int half = 0; half < 2; half++)
            {
                for (int c = 0; c < blocks_per_row; c++)
                {
                    for (int i = 0; i < rows_per_stream; i++)
                    {
                        const int row = r + half * rows_per_stream + i;
                        const u8* src_ptr = (row < rows)
                            ? src.data() + (size_t)(row * blocks_per_row + c) * a_block_size
                            : src.data(); // useless padding
                        memcpy(dst_ptr, src_ptr, a_block_size);
                        dst_ptr += a_block_size;
                    }
                }
            }
        }
    }

    /// \brief Load every weight of one decoder layer into its device buffers.
    /// \param layer_idx the layer index
    /// \param type the kind of the layer
    /// \param q4nx the opened checkpoint
    /// \param proj_weights per-layer projection buffer
    /// \param rms_weights per-layer rms buffer
    /// \param rope_rms_weights per-layer rope/rms buffer
    /// \return the layer output scale, which the host also needs for prefill
    bf16 load_layer_weights(int layer_idx, gemma4_12b_layer_type_t type, Q4NX& q4nx,
                            buffer<u8>& proj_weights, buffer<bf16>& rms_weights,
                            buffer<bf16>& rope_rms_weights)
    {
        gemma4_12b_layer_weight_def& _weight_desc = weight_desc(type);
        const uint32_t _DH = is_global_layer(type) ? DH : SWA_DH;
        const uint32_t _DQ = is_global_layer(type) ? DQ : SWA_DQ;

        u8* w_ptr_begin = proj_weights.data();
        // read q / k / v weights, which the kernel streams as one contiguous block
        {
            LOG_VERBOSE(1, "Loading qkv weights");
            std::vector<weight_desc_t*> qkv = {&_weight_desc.attn_q, &_weight_desc.attn_k};
            if (_weight_desc.has_v_proj) qkv.push_back(&_weight_desc.attn_v);
            for (weight_desc_t* desc : qkv) {
                buffer<u8> w;
                q4nx.load_weights(w, desc->format_name(layer_idx));
                assert(w.size() == desc->get_size());
                reorder_cpy(w_ptr_begin + desc->offset, w, D, desc->dtype);
                desc->load();
            }
        }
        // read o weights
        {
            LOG_VERBOSE(1, "Loading o weights");
            weight_desc_t& desc = _weight_desc.attn_output;
            buffer<u8> w_o;
            q4nx.load_weights(w_o, desc.format_name(layer_idx));
            assert(w_ptr_begin + desc.offset + w_o.size() <= proj_weights.end());
            reorder_cpy(w_ptr_begin + desc.offset, w_o, _DQ, desc.dtype);
            desc.load();
        }
        // read up/gate weights, interleaved every UP_GATE_OUTDIM_REORDER output rows
        {
            LOG_VERBOSE(1, "Loading shared up/gate weights");
            weight_desc_t& desc_up = _weight_desc.ffn_up;
            weight_desc_t& desc_gate = _weight_desc.ffn_gate;
            buffer<u8> w_up;
            buffer<u8> w_gate;
            q4nx.load_weights(w_up, desc_up.format_name(layer_idx));
            q4nx.load_weights(w_gate, desc_gate.format_name(layer_idx));

            assert(desc_up.shape[1] % UP_GATE_OUTDIM_REORDER == 0);
            assert(desc_gate.shape[1] % UP_GATE_OUTDIM_REORDER == 0);

            const int slices = (int)(desc_up.shape[1] / UP_GATE_OUTDIM_REORDER);
            const size_t slice_size = desc_up.get_size() / slices;
            u8* w_ptr = w_ptr_begin + desc_up.offset;
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
        // read the four rms norms
        {
            LOG_VERBOSE(1, "Loading rms weights");
            u8* rms_weights_begin = (u8*)rms_weights.data();
            for (weight_desc_t* desc : _weight_desc.rms_all()) {
                buffer<bf16> w;
                q4nx.load_weights(w, desc->format_name(layer_idx));
                memcpy(rms_weights_begin + desc->offset, w.data(), D * sizeof(bf16));
                desc->load();
            }
        }
        // read the q/k norms into the rope buffer, behind the cos/sin table
        {
            LOG_VERBOSE(1, "Loading attention norm weights");
            weight_desc_t& desc_q_norm = _weight_desc.attn_q_norm;
            weight_desc_t& desc_k_norm = _weight_desc.attn_k_norm;
            buffer<bf16> q_attn_norm;
            buffer<bf16> k_attn_norm;
            q4nx.load_weights(q_attn_norm, desc_q_norm.format_name(layer_idx));
            q4nx.load_weights(k_attn_norm, desc_k_norm.format_name(layer_idx));
            memcpy((u8*)rope_rms_weights.data() + desc_q_norm.offset, q_attn_norm.data(), _DH * sizeof(bf16));
            memcpy((u8*)rope_rms_weights.data() + desc_k_norm.offset, k_attn_norm.data(), _DH * sizeof(bf16));
            desc_q_norm.load();
            desc_k_norm.load();
        }
        // read the per layer output scale
        bf16 layer_scale;
        {
            LOG_VERBOSE(1, "Loading layer output scale");
            weight_desc_t& desc = _weight_desc.layer_output_scale;
            buffer<bf16> layer_scale_buffer;
            q4nx.load_weights(layer_scale_buffer, desc.format_name(layer_idx));
            layer_scale = layer_scale_buffer[0];
            // the tail of the rope buffer is padding, keep it zeroed
            memset(rope_rms_weights.data() + desc.offset / sizeof(bf16), 0, MIN_BF16_PAD * sizeof(bf16));
            rope_rms_weights[desc.offset / sizeof(bf16)] = layer_scale;
            desc.load();
        }

        proj_weights.sync_to_device();
        rms_weights.sync_to_device();
        rope_rms_weights.sync_to_device();
        return layer_scale;
    }

    /// \brief Load the final rms norm into `dst` (D bf16 elements).
    void load_final_norm(Q4NX& q4nx, bf16* dst) {
        buffer<bf16> w;
        q4nx.load_weights(w, final_norm.name);
        memcpy(dst, w.data(), D * sizeof(bf16));
        final_norm.load();
    }

    /// \brief Load and reorder the lm head into its device buffer.
    void load_lm_head_weights(Q4NX& q4nx, buffer<u8>& lm_head_weights) {
        buffer<u8> w;
        q4nx.load_weights(w, lm_head.name);
        assert(w.size() <= lm_head_weights.size());
        // reorder_cpy only writes the blocks that exist in the checkpoint; when
        // vocab_size_padded > vocab_size the tail is never touched, and the padded rows
        // would produce garbage logits. Zero it first (a no-op on gemma4-12b, whose
        // 262144-entry vocab is already a multiple of the 1024 padding).
        if (w.size() < lm_head_weights.size()) {
            memset(lm_head_weights.data() + w.size(), 0, lm_head_weights.size() - w.size());
        }
        // 4 vertical blocks: the lm head is spread over 4 rows of cores per column,
        // split into two halves of two so each of the column's two weight streams
        // gets a contiguous run
        reorder_cpy_lm_head(lm_head_weights.data(), w, D, lm_head.dtype, 4);
        lm_head.load();
        lm_head_weights.sync_to_device();
    }

private:
    /// \brief inv_freq[j] = theta^(-2j/head_dim) over the rotated part, 0 elsewhere.
    static void _build_inv_freq(std::vector<f32>& inv_freq, uint32_t head_dim, uint32_t rope_dim, f32 theta) {
        inv_freq.assign(head_dim / 2, 0.0f);
        for (uint32_t j = 0; j < rope_dim / 2; j++) {
            inv_freq[j] = (f32)std::pow((double)theta, -2.0 * (double)j / (double)head_dim);
        }
    }

    /// \brief Fill in one layer kind's descriptors and hand them to the containers.
    void _build_layer(gemma4_12b_layer_weight_def& layer, gemma4_12b_layer_type_t type) {
        const int64_t _D  = (int64_t)D;
        const int64_t _DH = is_global_layer(type) ? (int64_t)DH : (int64_t)SWA_DH;
        const int64_t _DQ = is_global_layer(type) ? (int64_t)DQ : (int64_t)SWA_DQ;
        const int64_t _DK = is_global_layer(type) ? (int64_t)DK : (int64_t)SWA_DK;
        const int64_t _DV = is_global_layer(type) ? (int64_t)DV : (int64_t)SWA_DV;
        const int64_t _FF = (int64_t)intermediate_size;

        // global layers reuse K as V (attention_k_eq_v), so no v_proj is stored
        layer.has_v_proj = !(is_global_layer(type) && attention_k_eq_v);

        layer.attn_q      = weight_desc_t(PROJ_DTYPE, {_D, _DQ}, "model.layers.%d.self_attn.q_proj.weight");
        layer.attn_k      = weight_desc_t(PROJ_DTYPE, {_D, _DK}, "model.layers.%d.self_attn.k_proj.weight");
        layer.attn_v      = weight_desc_t(PROJ_DTYPE, {_D, _DV}, "model.layers.%d.self_attn.v_proj.weight");
        layer.attn_output = weight_desc_t(PROJ_DTYPE, {_DQ, _D}, "model.layers.%d.self_attn.o_proj.weight");
        layer.ffn_up      = weight_desc_t(PROJ_DTYPE, {_D, _FF}, "model.layers.%d.mlp.up_proj.weight");
        layer.ffn_gate    = weight_desc_t(PROJ_DTYPE, {_D, _FF}, "model.layers.%d.mlp.gate_proj.weight");
        layer.ffn_down    = weight_desc_t(PROJ_DTYPE, {_FF, _D}, "model.layers.%d.mlp.down_proj.weight");

        layer.input_layer_norm      = weight_desc_t(flm_bf16, {_D}, "model.layers.%d.input_layernorm.weight");
        layer.post_attention_norm   = weight_desc_t(flm_bf16, {_D}, "model.layers.%d.post_attention_layernorm.weight");
        layer.pre_feedforward_norm  = weight_desc_t(flm_bf16, {_D}, "model.layers.%d.pre_feedforward_layernorm.weight");
        layer.post_feedforward_norm = weight_desc_t(flm_bf16, {_D}, "model.layers.%d.post_feedforward_layernorm.weight");

        layer.rope_cos_sin        = weight_desc_t(flm_bf16, {_DH}, "");  // cos (_DH/2) | sin (_DH/2)
        layer.attn_q_norm         = weight_desc_t(flm_bf16, {_DH}, "model.layers.%d.self_attn.q_norm.weight");
        layer.attn_k_norm         = weight_desc_t(flm_bf16, {_DH}, "model.layers.%d.self_attn.k_norm.weight");
        layer.layer_output_scale  = weight_desc_t(flm_bf16, {1}, "model.layers.%d.layer_output_scale.weight");

        weight_container proj_helper;
        weight_container rms_helper;
        weight_container rope_helper;
        for (weight_desc_t* w : layer.proj_all()) proj_helper.add_weight(*w);
        for (weight_desc_t* w : layer.rms_all())  rms_helper.add_weight(*w);
        for (weight_desc_t* w : layer.rope_all()) rope_helper.add_weight(*w);

        if (is_global_layer(type)) {
            global_proj_weights_byte_size = proj_helper.get_size();
            global_rope_rms_byte_size = rope_helper.get_size();
        }
        else {
            swa_proj_weights_byte_size = proj_helper.get_size();
            swa_rope_rms_byte_size = rope_helper.get_size();
        }
        rms_weights_byte_size = rms_helper.get_size();
    }
};
