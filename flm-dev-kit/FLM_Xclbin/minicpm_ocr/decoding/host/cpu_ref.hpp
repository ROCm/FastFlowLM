// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file cpu_ref.hpp
/// \brief fp32 CPU reference for one Qwen3.5 decode step (GDN layer, attention layer), weights from model.q4nx
/// \note See README.md in this folder.
#pragma once
#include <string>
#include <vector>
#include "tensor_utils/q4_npu_eXpress.hpp"
#include "qwen3_5vl_model_def.hpp"

namespace cpu_ref {

using vec = std::vector<float>;

struct matrix_t {
    int rows = 0;
    int cols = 0;
    vec w;
};

struct gdn_weights_t {
    vec rms_pre, rms_post, conv, norm, a, dt;
    matrix_t qkv, z, out, alpha, beta, up, gate, down;
};

struct attn_weights_t {
    vec rms_pre, rms_post, q_norm, k_norm;
    matrix_t q, k, v, o, up, gate, down;
};

struct gdn_state_t {
    vec conv;
    vec S;
};

struct attn_step_t {
    vec k_new;
    vec v_new;
    vec core_out;
};

enum class kv_quant_t { none, int8, fp8 };

const char* kv_quant_name(kv_quant_t q);
void quantize_rows(float* data, size_t rows, int heads, int head_dim, kv_quant_t q);
vec embed_token(const buffer<u8>& lm_head_q80, int token, int D);
vec lm_head_logits(const buffer<u8>& lm_head_q80, const vec& x, const vec& norm, int rows);

gdn_weights_t load_gdn_weights(Q4NX& q4nx, const qwen3_5vl_desc& desc, int layer);
attn_weights_t load_attn_weights(Q4NX& q4nx, const qwen3_5vl_desc& desc, int layer);

vec gdn_layer(const qwen3_5vl_desc& desc, const gdn_weights_t& w, gdn_state_t& state, const vec& x);
vec attn_layer(const qwen3_5vl_desc& desc, const attn_weights_t& w, const float* k_cache, const float* v_cache,
               int context_len, int position, const vec& x, attn_step_t& step, kv_quant_t quant = kv_quant_t::none);

}  // namespace cpu_ref
