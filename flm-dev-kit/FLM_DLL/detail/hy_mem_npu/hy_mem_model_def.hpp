// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file hy_mem_model_def.hpp
/// \brief weight-shape definition for Hy-Memory-30B-A3B (hy_v3)
/// \author FastFlowLM Team
/// \date 2026-10-26
/// \note Two layer kinds, each on its own decode overlay:
///         * dense (layer 0, first_k_dense_replace = 1): GQA attention + SwiGLU MLP,
///           FLM_Xclbin/Hy_mem_30BA3B/dense_decoding
///         * moe (layers 1..47): the same attention + 128 experts, top 8, plus one
///           shared expert, all 768 wide (expert_hidden_dim), with a sigmoid router,
///           FLM_Xclbin/Hy_mem_30BA3B/moe_decoding
///       Every projection is Q4_K (4736-byte 32x256 blocks), the lm_head is Q8_0.
///       The layouts and loaders below are ported from those two folders'
///       host/decoding_layer.cpp, which are checked against the Q4_K reference;
///       keep them in step with the kernels, not with other engines.
#ifndef __HY_MEM_MODEL_DEF_HPP__
#define __HY_MEM_MODEL_DEF_HPP__
#include <cmath>
#include <iomanip>
#include <stdexcept>
#include "lm_config.hpp"
#include "weight_desc.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"

/// @brief the two kinds of layer, one decode overlay each
enum class hy_mem_layer_kind { dense, moe };

/// @brief layer 0: attention + dense SwiGLU MLP
/// @note  proj buffer order = device order: k, v, q (one matrix to the proj engine,
///        RoPE takes K/V first), o, then up|gate (interleaved per round), down.
struct hy_mem_dense_layer_def {
    weight_desc_t attn_k, attn_v, attn_q, attn_output;
    weight_desc_t ffn_up, ffn_gate, ffn_down;
    weight_desc_t input_layer_norm, post_attntion_norm;
    weight_desc_t attn_q_norm, attn_k_norm;

    std::vector<weight_desc_t*> proj_all() {
        return {&attn_k, &attn_v, &attn_q, &attn_output, &ffn_up, &ffn_gate, &ffn_down};
    }
    std::vector<weight_desc_t*> rms_all() { return {&input_layer_norm, &post_attntion_norm}; }
    std::vector<weight_desc_t*> all() {
        std::vector<weight_desc_t*> v = proj_all();
        for (weight_desc_t* w : rms_all()) v.push_back(w);
        v.push_back(&attn_q_norm);
        v.push_back(&attn_k_norm);
        return v;
    }
};

/// @brief layers 1..47: attention + MoE
/// @note  proj buffer order = device order: routed up, gate, down (one region,
///        re-cut per expert, see load_moe_layer), k, v, q, o, shared up|gate, shared down.
///        rms buffer: [input_layernorm | post_attention_layernorm | router weight].
struct hy_mem_moe_layer_def {
    weight_desc_t ffn_up_exps, ffn_gate_exps, ffn_down_exps;
    weight_desc_t attn_k, attn_v, attn_q, attn_output;
    weight_desc_t ffn_up_shexp, ffn_gate_shexp, ffn_down_shexp;
    weight_desc_t input_layer_norm, post_attntion_norm, ffn_gate_inp;
    weight_desc_t attn_q_norm, attn_k_norm;

    std::vector<weight_desc_t*> proj_all() {
        return {&ffn_up_exps, &ffn_gate_exps, &ffn_down_exps,
                &attn_k, &attn_v, &attn_q, &attn_output,
                &ffn_up_shexp, &ffn_gate_shexp, &ffn_down_shexp};
    }
    std::vector<weight_desc_t*> rms_all() { return {&input_layer_norm, &post_attntion_norm, &ffn_gate_inp}; }
    std::vector<weight_desc_t*> all() {
        std::vector<weight_desc_t*> v = proj_all();
        for (weight_desc_t* w : rms_all()) v.push_back(w);
        v.push_back(&attn_q_norm);
        v.push_back(&attn_k_norm);
        return v;
    }
};

struct hy_mem_desc {
    // ---- values baked into the xclbins (FLM_Xclbin/Hy_mem_30BA3B/*/models/*.h) ----
    // build() refuses a config.json that disagrees: the kernels would not notice.
    static constexpr uint32_t KERNEL_D = 2048;
    static constexpr uint32_t KERNEL_DH = 128;
    static constexpr uint32_t KERNEL_HEADS = 32;
    static constexpr uint32_t KERNEL_KV_HEADS = 4;
    static constexpr uint32_t KERNEL_DENSE_WIDTH = 6912;
    static constexpr uint32_t KERNEL_EXPERT_WIDTH = 768;
    static constexpr uint32_t KERNEL_EXPERTS = 128;
    static constexpr uint32_t KERNEL_TOP_K = 8;
    static constexpr uint32_t KERNEL_MAX_LAYERS = 48;     // the router's per-layer address table
    static constexpr uint32_t KERNEL_VOCAB = 120832;      // lm_head.xclbin

    // up/gate rows per proj round (one round = 16 cores x 32 rows = 512 values):
    // dense and shared expert [up 256 | gate 256], routed experts [up 64 | gate 64] per column
    static constexpr int DENSE_ROWS_PER_ROUND = 256;
    static constexpr int SHARED_ROWS_PER_ROUND = 256;
    static constexpr int EXPERT_ROWS_PER_ROUND = 64;
    // router weight blocks: 32 experts x 256 inputs (router_weight_row/col_block)
    static constexpr int ROUTER_ROW_BLOCK = 32;
    static constexpr int ROUTER_COL_BLOCK = 256;
    // attention: 2 array columns, each holding 2 of the 4 KV heads
    static constexpr int ATTN_COLS = 2;
    // the attention reads K/V in 16-token chunks, so MAX_L must be whole chunks
    static constexpr int KV_CHUNK = 16;
    // largest KV cache the attention was checked at (README: 128K, no drift)
    static constexpr uint32_t MAX_CONTEXT = 131072;

    static constexpr size_t Q4K_BLOCK_BYTES = 4736;       // one 32 x 256 Q4_K block
    static constexpr size_t Q80_BLOCK_BYTES = 8704;       // one 32 x 256 Q8_0 block

    uint32_t num_hidden_layers;
    uint32_t D, DH, DQ, DK, DV, RoPE_DIM;
    uint32_t num_heads, num_kv_heads;
    uint32_t dense_width;          // intermediate_size, layer 0's MLP
    uint32_t expert_width;         // expert_hidden_dim (NOT moe_intermediate_size)
    uint32_t num_experts, top_k;
    uint32_t first_k_dense;
    uint32_t vocab_size, vocab_size_padded;
    uint32_t eos_token_id, eod_token_id;
    f32 rms_norm_eps;
    f32 rope_theta;
    f32 router_scaling;            // router_scaling_factor: weights = score / sum * this
    std::vector<f32> inv_freq;     // RoPE_DIM / 2, fp32 like transformers

    hy_mem_dense_layer_def dense;
    hy_mem_moe_layer_def moe;
    size_t dense_proj_bytes, dense_rms_bytes;
    size_t moe_proj_bytes, moe_rms_bytes;

    inline hy_mem_layer_kind layer_kind(int idx) const {
        return idx < (int)first_k_dense ? hy_mem_layer_kind::dense : hy_mem_layer_kind::moe;
    }

    /// @brief one layer's KV cache in elements: [K col 0][K col 1][V col 0][V col 1],
    ///        each MAX_L x (2 KV heads x DH): all K, then all V at half the size
    ///        (as hunyuan_npu). Same for both layer kinds.
    inline size_t get_kv_cache_elems(uint32_t MAX_L) const { return (size_t)MAX_L * (DK + DV); }
    /// @brief zero rows after the last block (V col 1): the prefill attention reads
    ///        whole 128-row blocks up to ~380 rows past MAX_L for a chunk that ends
    ///        near it (the other blocks' overrun lands in the next block: finite,
    ///        and masked)
    static constexpr uint32_t KV_TAIL_ROWS = 512;
    /// @brief elements to allocate per layer: the cache plus its zero tail
    inline size_t get_kv_cache_alloc_elems(uint32_t MAX_L) const {
        return get_kv_cache_elems(MAX_L) + (size_t)KV_TAIL_ROWS * kv_col_row();
    }
    /// @brief elements per token of one column's K (or V) block
    inline size_t kv_col_row() const { return (size_t)(num_kv_heads / ATTN_COLS) * DH; }
    /// @brief element offset of a column's K (kv = 0) or V (kv = 1) block
    ///        (kv_cache_view::row in hy_mem_prefill_ops.hpp has the same rule)
    inline size_t kv_block_offset(int col, int kv, uint32_t MAX_L) const {
        return (size_t)(2 * kv + col) * MAX_L * kv_col_row();
    }
    /// @brief rope buffer: [cos RoPE_DIM/2 | sin RoPE_DIM/2 | q_norm DH | k_norm DH]
    inline size_t get_rope_rms_elems() const { return RoPE_DIM + 2 * DH; }
    /// @brief lm_head buffer bytes: Q8_0, vocab padded to 1024 rows
    inline size_t get_lm_head_w_size() const {
        return (size_t)vocab_size_padded / 32 * (D / 256) * Q80_BLOCK_BYTES;
    }

    inline void build(LM_Config& config) {
        num_hidden_layers = config.get("num_hidden_layers");
        D = config.get("hidden_size");
        num_heads = config.get("num_attention_heads");
        num_kv_heads = config.get("num_key_value_heads", num_heads);
        DH = config.get("head_dim", D / num_heads);
        DQ = DH * num_heads;
        DK = DH * num_kv_heads;
        DV = DH * num_kv_heads;
        RoPE_DIM = DH;                                   // full-head rotation
        dense_width = config.get("intermediate_size");
        // config.json carries moe_intermediate_size = 1536 too, but the checkpoint's
        // experts (and shared expert) are expert_hidden_dim = 768 wide
        expert_width = config.get("expert_hidden_dim", config.get("moe_intermediate_size"));
        num_experts = config.get("num_experts");
        top_k = config.get("num_experts_per_tok");
        first_k_dense = config.get("first_k_dense_replace", 1);
        vocab_size = config.get("vocab_size");
        vocab_size_padded = (vocab_size + 1024 - 1) / 1024 * 1024;
        eos_token_id = config.get("eos_token_id");
        eod_token_id = config.get("eod_token_id", eos_token_id);
        rms_norm_eps = config.get<f32>("rms_norm_eps", 1e-5f);
        router_scaling = config.get<f32>("router_scaling_factor", 1.0f);
        // hy_v3 keeps theta under rope_parameters (rope_type "default")
        rope_theta = cfg_get<f32>(config.sub("rope_parameters"), "rope_theta",
                                  config.get<f32>("rope_theta", 10000.0f));
        inv_freq.resize(RoPE_DIM / 2);
        for (uint32_t i = 0; i < RoPE_DIM / 2; i++) {
            // fp32, as transformers builds it: theta^(-2i / dim)
            inv_freq[i] = (f32)(1.0 / std::pow((double)rope_theta, (2.0 * i) / RoPE_DIM));
        }

        DEBUG_BLOCK(1,
        std::cout << "================ hy_mem_desc ================" << std::endl;
        std::cout << std::left << std::setw(28) << "num_hidden_layers" << " = " << num_hidden_layers << std::endl;
        std::cout << std::left << std::setw(28) << "first_k_dense"     << " = " << first_k_dense << std::endl;
        std::cout << std::left << std::setw(28) << "D (hidden_size)"   << " = " << D << std::endl;
        std::cout << std::left << std::setw(28) << "DH (head_dim)"     << " = " << DH << std::endl;
        std::cout << std::left << std::setw(28) << "DQ / DK / DV"      << " = " << DQ << " / " << DK << " / " << DV << std::endl;
        std::cout << std::left << std::setw(28) << "num_heads / kv"    << " = " << num_heads << " / " << num_kv_heads << std::endl;
        std::cout << std::left << std::setw(28) << "dense_width"       << " = " << dense_width << std::endl;
        std::cout << std::left << std::setw(28) << "expert_width"      << " = " << expert_width << std::endl;
        std::cout << std::left << std::setw(28) << "experts / top_k"   << " = " << num_experts << " / " << top_k << std::endl;
        std::cout << std::left << std::setw(28) << "rms_norm_eps"      << " = " << rms_norm_eps << std::endl;
        std::cout << std::left << std::setw(28) << "rope_theta"        << " = " << rope_theta << std::endl;
        std::cout << std::left << std::setw(28) << "router_scaling"    << " = " << router_scaling << std::endl;
        std::cout << std::left << std::setw(28) << "vocab (padded)"    << " = " << vocab_size << " (" << vocab_size_padded << ")" << std::endl;
        std::cout << std::left << std::setw(28) << "eos / eod"         << " = " << eos_token_id << " / " << eod_token_id << std::endl;
        std::cout << "==============================================" << std::endl;
        )

        _check_against_kernels();
        _build_dense_layer();
        _build_moe_layer();
    }

    /// @brief dense layer weights; returns nothing, every value is in the buffers
    /// @param proj Q4_K projections in proj engine order (dense.proj_all())
    /// @param rms [input_layernorm | post_attention_layernorm]
    /// @param rope_rms the rope buffer; q_norm / k_norm go behind the cos/sin table
    void load_dense_layer(int layer_idx, Q4NX& q4nx, buffer<u8>& proj, buffer<bf16>& rms, buffer<bf16>& rope_rms);

    /// @brief moe layer weights
    /// @param rms [input_layernorm | post_attention_layernorm | router weight in read order]
    /// @param router_bias out: the fp32 expert bias (moe_expert_bias), sent with RTP writes
    void load_moe_layer(int layer_idx, Q4NX& q4nx, buffer<u8>& proj, buffer<bf16>& rms, buffer<bf16>& rope_rms,
                        std::vector<float>& router_bias);

    /// @brief a dense layer's projections relaid for the PREFILL overlay's dequant_mm
    /// @param dst dense_proj_bytes, filled from `proj`; sync_to_device() is the caller's
    /// @note The prefill overlay's Q4_K reader takes the int8-mmul block order, because
    ///       the 47 MoE layers are stored that way and it reads them in place (17 GB,
    ///       no second copy). A dense layer's decode kernel needs the plain order, so
    ///       its one buffer cannot be shared -- but there is only layer 0, ~35 MB, so
    ///       it gets a relaid copy instead of the overlay carrying a second decoder.
    ///       Same q4k_relayout_for_mmul the MoE layers go through, so the two orders
    ///       can never drift apart.
    void make_prefill_proj(buffer<u8>& dst, const buffer<u8>& proj);

    /// @brief lm_head.weight (Q8_0) in lm_head.xclbin's read order
    void load_lm_head(Q4NX& q4nx, buffer<u8>& dst);

private:
    inline void _check_against_kernels() {
        struct { const char* name; uint32_t got, want; } checks[] = {
            {"hidden_size", D, KERNEL_D},
            {"head_dim", DH, KERNEL_DH},
            {"num_attention_heads", num_heads, KERNEL_HEADS},
            {"num_key_value_heads", num_kv_heads, KERNEL_KV_HEADS},
            {"intermediate_size", dense_width, KERNEL_DENSE_WIDTH},
            {"expert_hidden_dim", expert_width, KERNEL_EXPERT_WIDTH},
            {"num_experts", num_experts, KERNEL_EXPERTS},
            {"num_experts_per_tok", top_k, KERNEL_TOP_K},
            {"vocab_size (padded)", vocab_size_padded, KERNEL_VOCAB},
        };
        for (auto& c : checks) {
            if (c.got != c.want) {
                throw std::runtime_error(std::string("hy_mem_npu: config ") + c.name + " = " + std::to_string(c.got) +
                                         ", but the xclbins are built for " + std::to_string(c.want));
            }
        }
        if (num_hidden_layers > KERNEL_MAX_LAYERS) {
            throw std::runtime_error("hy_mem_npu: the MoE router keeps addresses for " +
                                     std::to_string(KERNEL_MAX_LAYERS) + " layers only");
        }
    }

    inline void _build_dense_layer() {
        dense.attn_k = weight_desc_t(flm_q4k, {D, DK}, "model.layer.%d.self_attn.k_proj.weight");
        dense.attn_v = weight_desc_t(flm_q4k, {D, DV}, "model.layer.%d.self_attn.v_proj.weight");
        dense.attn_q = weight_desc_t(flm_q4k, {D, DQ}, "model.layer.%d.self_attn.q_proj.weight");
        dense.attn_output = weight_desc_t(flm_q4k, {DQ, D}, "model.layer.%d.self_attn.o_proj.weight");
        dense.ffn_up = weight_desc_t(flm_q4k, {D, dense_width}, "model.layer.%d.mlp.up_proj.weight");
        dense.ffn_gate = weight_desc_t(flm_q4k, {D, dense_width}, "model.layer.%d.mlp.gate_proj.weight");
        dense.ffn_down = weight_desc_t(flm_q4k, {dense_width, D}, "model.layer.%d.mlp.down_proj.weight");
        dense.input_layer_norm = weight_desc_t(flm_bf16, {D}, "model.layer.%d.input_layernorm.weight");
        dense.post_attntion_norm = weight_desc_t(flm_bf16, {D}, "model.layer.%d.post_attention_layernorm.weight");
        dense.attn_q_norm = weight_desc_t(flm_bf16, {DH}, "model.layer.%d.self_attn.q_norm.weight");
        dense.attn_k_norm = weight_desc_t(flm_bf16, {DH}, "model.layer.%d.self_attn.k_norm.weight");

        weight_container proj_c, rms_c;
        for (weight_desc_t* w : dense.proj_all()) proj_c.add_weight(*w);
        for (weight_desc_t* w : dense.rms_all()) rms_c.add_weight(*w);
        dense.attn_q_norm.indp();   // live in the rope buffer, placed by hand
        dense.attn_k_norm.indp();
        dense_proj_bytes = proj_c.get_size();
        dense_rms_bytes = rms_c.get_size();
    }

    inline void _build_moe_layer() {
        moe.ffn_up_exps = weight_desc_t(flm_q4k, {D, expert_width, num_experts}, "model.layer.%d.mlp.up_exps_proj.weight");
        moe.ffn_gate_exps = weight_desc_t(flm_q4k, {D, expert_width, num_experts}, "model.layer.%d.mlp.gate_exps_proj.weight");
        moe.ffn_down_exps = weight_desc_t(flm_q4k, {expert_width, D, num_experts}, "model.layer.%d.mlp.down_exps_proj.weight");
        moe.attn_k = weight_desc_t(flm_q4k, {D, DK}, "model.layer.%d.self_attn.k_proj.weight");
        moe.attn_v = weight_desc_t(flm_q4k, {D, DV}, "model.layer.%d.self_attn.v_proj.weight");
        moe.attn_q = weight_desc_t(flm_q4k, {D, DQ}, "model.layer.%d.self_attn.q_proj.weight");
        moe.attn_output = weight_desc_t(flm_q4k, {DQ, D}, "model.layer.%d.self_attn.o_proj.weight");
        moe.ffn_up_shexp = weight_desc_t(flm_q4k, {D, expert_width}, "model.layer.%d.mlp.share_up_exps_proj.weight");
        moe.ffn_gate_shexp = weight_desc_t(flm_q4k, {D, expert_width}, "model.layer.%d.mlp.share_gate_exps_proj.weight");
        moe.ffn_down_shexp = weight_desc_t(flm_q4k, {expert_width, D}, "model.layer.%d.mlp.share_down_exps_proj.weight");
        moe.input_layer_norm = weight_desc_t(flm_bf16, {D}, "model.layer.%d.input_layernorm.weight");
        moe.post_attntion_norm = weight_desc_t(flm_bf16, {D}, "model.layer.%d.post_attention_layernorm.weight");
        moe.ffn_gate_inp = weight_desc_t(flm_bf16, {D, num_experts}, "model.layer.%d.moe_router.weight");
        moe.attn_q_norm = weight_desc_t(flm_bf16, {DH}, "model.layer.%d.self_attn.q_norm.weight");
        moe.attn_k_norm = weight_desc_t(flm_bf16, {DH}, "model.layer.%d.self_attn.k_norm.weight");

        weight_container proj_c, rms_c;
        for (weight_desc_t* w : moe.proj_all()) proj_c.add_weight(*w);
        for (weight_desc_t* w : moe.rms_all()) rms_c.add_weight(*w);
        moe.attn_q_norm.indp();
        moe.attn_k_norm.indp();
        moe_proj_bytes = proj_c.get_size();
        moe_rms_bytes = rms_c.get_size();
        // shim DMA offsets are 32-bit, and one buffer per layer keeps them small
        if (moe_proj_bytes >= 512ull * 1024 * 1024) {
            throw std::runtime_error("hy_mem_npu: one MoE layer's weights do not fit 512 MB");
        }
    }
};

#endif
