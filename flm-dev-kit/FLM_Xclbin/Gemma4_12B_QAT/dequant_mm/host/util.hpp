// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include <bits/stdc++.h>
#include <boost/program_options.hpp>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdfloat>
#include "utils/debug_utils.hpp"
#include <omp.h>

#include "typedef.hpp"
#include "npu_utils/npu_utils.hpp"
#include "utils/utils.hpp"
#include "utils/vm_args.hpp"
#include <math.h>
#include "tensor_utils/q4_npu_eXpress.hpp"
#include "tensor_2d.hpp"
#include "weights_desc.hpp"

// q8_0 packs 32 elements per block plus a 2-byte (bf16) scale per block.
inline size_t q8_0_size(size_t elems) { return elems + elems / 32 * 2; }

// void reorder_cpy(u8 *dst, buffer<u8> &src, const int col, flm_dtype_t dtype, const int vertical_blocks, int block_for_512_reorder);
void reorder_cpy(u8 *dst, buffer<u8> &src, const int col, const int vertical_blocks = 2);
struct gated_attn_layer{
public:
    // norms (bf16)
    weight_desc_t input_layer_norm;
    weight_desc_t post_attn_norm;
    weight_desc_t attn_q_norm;
    weight_desc_t attn_k_norm;
    // attention projections (q4_1)
    weight_desc_t attn_q;
    weight_desc_t attn_gate;   // merged into q_proj (second half), shares its data
    weight_desc_t attn_k;
    weight_desc_t attn_v;
    weight_desc_t attn_output;
    // mlp projections (q4_1)
    weight_desc_t mlp_up;
    weight_desc_t mlp_gate;
    weight_desc_t mlp_down;

    std::vector<weight_desc_t*> all() {
        return {
            &input_layer_norm, &post_attn_norm, &attn_q_norm, &attn_k_norm,
            &attn_q, &attn_gate, &attn_k, &attn_v, &attn_output,
            &mlp_up, &mlp_gate, &mlp_down,
        };
    }
};

gated_attn_layer _weight_desc;

struct gated_deltanet_layer{
public:
    // norms (bf16)
    weight_desc_t input_layer_norm;
    weight_desc_t post_attn_norm;
    // linear-attention (gated DeltaNet) projections
    weight_desc_t linear_qkv;      // q4_1, merged q|k|v: out = Linear_DQ+Linear_DK+Linear_DV
    weight_desc_t linear_gate;     // q4_1, output gate (self_attn.gate_proj)
    weight_desc_t linear_output;   // q8_0, ssm_out_proj
    weight_desc_t linear_alpha;    // q4_1, ssm_alpha_proj (decay/forget projection)
    weight_desc_t linear_beta;     // q4_1, ssm_beta_proj (delta-rule update projection)
    // linear-attention auxiliary weights (bf16 / f32)
    weight_desc_t linear_conv1d;   // bf16, short causal conv over q|k|v
    weight_desc_t linear_norm;     // bf16, ssm_norm (per v-head-dim RMS gain)
    weight_desc_t linear_a;        // f32, ssm_a (per-head state decay)
    weight_desc_t linear_dt_bias;  // f32, ssm_dt.bias (per-head step bias)
    // mlp projections (q4_1)
    weight_desc_t mlp_up;
    weight_desc_t mlp_gate;
    weight_desc_t mlp_down;

    std::vector<weight_desc_t*> all() {
        return {
            &input_layer_norm, &post_attn_norm,
            &linear_qkv, &linear_gate, &linear_output, &linear_alpha, &linear_beta,
            &linear_conv1d, &linear_norm, &linear_a, &linear_dt_bias,
            &mlp_up, &mlp_gate, &mlp_down,
        };
    }
};

gated_deltanet_layer _deltanet_weight_desc;

// Mirrors load_all_proj_weights but for the Qwen3.5 gated DeltaNet (linear attention)
// layer. Weight names, dtypes, and tensor layout follow the qwen3_5vl NPU runtime.
//   LINEAR_DQ / LINEAR_DK / LINEAR_DV : fused q/k/v widths out of qkv_proj
//   LINEAR_NV                         : number of v-heads (ssm_a / ssm_dt.bias length)
//   LINEAR_VH                         : v-head dim (ssm_norm length)
//   CONV_KERNEL                       : ssm_conv1d kernel size
void load_all_deltanet_proj_weights(buffer<u8>& w, buffer<bf16>& w_rms, Q4NX& q4nx, int layer_idx,
                                     int LINEAR_DQ, int LINEAR_DK, int LINEAR_DV,
                                     int LINEAR_NV, int LINEAR_VH, int CONV_KERNEL,
                                     int D, int INTERMEDIATE_SIZE) {
    const int LINEAR_QKV = LINEAR_DQ + LINEAR_DK + LINEAR_DV;
    u8* w_ptr = w.data();
    u8* w_ptr_end = w.end();
    bf16* w_rms_ptr = w_rms.data();
    bf16* w_rms_ptr_end = w_rms.end();

    // norms
    _deltanet_weight_desc.input_layer_norm = weight_desc_t(flm_bf16, {D}, "model.layers.%d.input_layernorm.weight");
    _deltanet_weight_desc.post_attn_norm   = weight_desc_t(flm_bf16, {D}, "model.layers.%d.post_attention_layernorm.weight");
    // linear-attention projections. qkv_proj on disk holds q|k|v fused on the out dim.
    _deltanet_weight_desc.linear_qkv    = weight_desc_t(flm_q41, {D, LINEAR_QKV}, "model.layers.%d.linear_attn.qkv_proj.weight");
    _deltanet_weight_desc.linear_gate   = weight_desc_t(flm_q41, {D, LINEAR_DV},  "model.layers.%d.self_attn.gate_proj.weight");
    _deltanet_weight_desc.linear_output = weight_desc_t(flm_q80, {LINEAR_DV, D},  "model.layers.%d.linear_attn.ssm_out_proj.weight");
    _deltanet_weight_desc.linear_alpha  = weight_desc_t(flm_q41, {D, LINEAR_NV},  "model.layers.%d.linear_attn.ssm_alpha_proj.weight");
    _deltanet_weight_desc.linear_beta   = weight_desc_t(flm_q41, {D, LINEAR_NV},  "model.layers.%d.linear_attn.ssm_beta_proj.weight");
    // linear-attention auxiliary weights
    _deltanet_weight_desc.linear_conv1d  = weight_desc_t(flm_bf16, {CONV_KERNEL, LINEAR_QKV}, "model.layers.%d.linear_attn.ssm_conv1d.weight");
    _deltanet_weight_desc.linear_norm    = weight_desc_t(flm_bf16, {LINEAR_VH},               "model.layers.%d.linear_attn.ssm_norm.weight");
    _deltanet_weight_desc.linear_a       = weight_desc_t(flm_f32,  {LINEAR_NV},               "model.layers.%d.linear_attn.ssm_a");
    _deltanet_weight_desc.linear_dt_bias = weight_desc_t(flm_f32,  {LINEAR_NV},               "model.layers.%d.linear_attn.ssm_dt.bias");
    // mlp projections
    _deltanet_weight_desc.mlp_up   = weight_desc_t(flm_q41, {D, INTERMEDIATE_SIZE}, "model.layers.%d.mlp.up_proj.weight");
    _deltanet_weight_desc.mlp_gate = weight_desc_t(flm_q41, {D, INTERMEDIATE_SIZE}, "model.layers.%d.mlp.gate_proj.weight");
    _deltanet_weight_desc.mlp_down = weight_desc_t(flm_q41, {INTERMEDIATE_SIZE, D}, "model.layers.%d.mlp.down_proj.weight");

    // offsets are managed manually via w_ptr below, so mark every descriptor as
    // independently managed (sets `added`); load() is called once its bytes are copied.
    for (weight_desc_t* wd : _deltanet_weight_desc.all()) wd->indp();

    // read fused qkv projection
    {
        LOG_VERBOSE(1, "Loading linear qkv weights");
        assert(w_ptr < w_ptr_end);
        buffer<u8> w_qkv;
        q4nx.load_weights(w_qkv, _deltanet_weight_desc.linear_qkv.format_name(layer_idx));
        reorder_cpy(w_ptr, w_qkv, D);
        w_ptr += _deltanet_weight_desc.linear_qkv.get_size();
        _deltanet_weight_desc.linear_qkv.load();
    }
    // read output gate
    {
        LOG_VERBOSE(1, "Loading linear gate weights");
        assert(w_ptr < w_ptr_end);
        buffer<u8> w_gate;
        q4nx.load_weights(w_gate, _deltanet_weight_desc.linear_gate.format_name(layer_idx));
        reorder_cpy(w_ptr, w_gate, D);
        w_ptr += _deltanet_weight_desc.linear_gate.get_size();
        _deltanet_weight_desc.linear_gate.load();
    }
    // read ssm output projection (q8_0)
    {
        LOG_VERBOSE(1, "Loading linear output weights");
        assert(w_ptr < w_ptr_end);
        buffer<u8> w_o;
        q4nx.load_weights(w_o, _deltanet_weight_desc.linear_output.format_name(layer_idx));
        reorder_cpy(w_ptr, w_o, LINEAR_DV);
        w_ptr += _deltanet_weight_desc.linear_output.get_size();
        _deltanet_weight_desc.linear_output.load();
    }
    // read up/gate weights, interleaved per GLU chunk
    {
        LOG_VERBOSE(1, "Loading up/gate weights");
        assert(w_ptr < w_ptr_end);
        buffer<u8> w_up;
        buffer<u8> w_gate;
        q4nx.load_weights(w_up, _deltanet_weight_desc.mlp_up.format_name(layer_idx));
        q4nx.load_weights(w_gate, _deltanet_weight_desc.mlp_gate.format_name(layer_idx));
        const int blocks = 1024 / 2 / 32 * (D / 256);
        const int chunk_size = blocks * 32 * 256 * 5 / 8;
        const int total_size = INTERMEDIATE_SIZE * D * 5 / 8;
        const int phases = total_size / chunk_size;
        tensor_2d<u8> w_up_tensor(w_up, chunk_size, 0);
        tensor_2d<u8> w_gate_tensor(w_gate, chunk_size, 0);
        for (int i = 0; i < phases; i++)
        {
            LOG_VERBOSE(1, "Copying up and gate weights to buffer, phase " << i + 1 << "/" << phases);
            reorder_cpy(w_ptr, w_up_tensor[i], D);
            w_ptr += chunk_size;
            reorder_cpy(w_ptr, w_gate_tensor[i], D);
            w_ptr += chunk_size;
        }
        _deltanet_weight_desc.mlp_up.load();
        _deltanet_weight_desc.mlp_gate.load();
    }
    // read down weights
    {
        LOG_VERBOSE(1, "Loading down weights");
        assert(w_ptr < w_ptr_end);
        buffer<u8> w_down;
        q4nx.load_weights(w_down, _deltanet_weight_desc.mlp_down.format_name(layer_idx));
        reorder_cpy(w_ptr, w_down, INTERMEDIATE_SIZE);
        w_ptr += _deltanet_weight_desc.mlp_down.get_size();
        _deltanet_weight_desc.mlp_down.load();
    }
    // read alpha/beta projections (copied raw, not block-reordered)
    {
        LOG_VERBOSE(1, "Loading alpha/beta weights");
        assert(w_ptr < w_ptr_end);
        buffer<u8> w_alpha;
        buffer<u8> w_beta;
        q4nx.load_weights(w_alpha, _deltanet_weight_desc.linear_alpha.format_name(layer_idx));
        q4nx.load_weights(w_beta, _deltanet_weight_desc.linear_beta.format_name(layer_idx));
        memcpy(w_ptr, w_alpha.data(), w_alpha.size());
        w_ptr += w_alpha.size();
        memcpy(w_ptr, w_beta.data(), w_beta.size());
        w_ptr += w_beta.size();
        _deltanet_weight_desc.linear_alpha.load();
        _deltanet_weight_desc.linear_beta.load();
    }
    assert(w_ptr == w_ptr_end);
    // read rms (pre/post attention) weights
    {
        LOG_VERBOSE(1, "Loading rms weights");
        buffer<bf16> w_rms_pre;
        buffer<bf16> w_rms_post;
        assert(w_rms_ptr < w_rms_ptr_end);
        q4nx.load_weights(w_rms_pre, _deltanet_weight_desc.input_layer_norm.format_name(layer_idx));
        q4nx.load_weights(w_rms_post, _deltanet_weight_desc.post_attn_norm.format_name(layer_idx));
        memcpy((unsigned char*)w_rms_ptr, w_rms_pre.data(), D * sizeof(bf16));
        w_rms_ptr += D;
        memcpy((unsigned char*)w_rms_ptr, w_rms_post.data(), D * sizeof(bf16));
        w_rms_ptr += D;
        _deltanet_weight_desc.input_layer_norm.load();
        _deltanet_weight_desc.post_attn_norm.load();
    }
    // read short conv1d weights
    {
        LOG_VERBOSE(1, "Loading conv1d weights");
        buffer<bf16> conv_weights;
        q4nx.load_weights(conv_weights, _deltanet_weight_desc.linear_conv1d.format_name(layer_idx));
        memcpy((unsigned char*)w_rms_ptr, conv_weights.data(), conv_weights.size() * sizeof(bf16));
        w_rms_ptr += conv_weights.size();
        _deltanet_weight_desc.linear_conv1d.load();
    }
    // read ssm_norm weights (per v-head-dim)
    {
        LOG_VERBOSE(1, "Loading ssm_norm weights");
        buffer<bf16> norm_weights;
        q4nx.load_weights(norm_weights, _deltanet_weight_desc.linear_norm.format_name(layer_idx));
        memcpy((unsigned char*)w_rms_ptr, norm_weights.data(), LINEAR_VH * sizeof(bf16));
        w_rms_ptr += LINEAR_VH;
        _deltanet_weight_desc.linear_norm.load();
    }
    // read ssm_a and ssm_dt.bias (f32, stored after the bf16 block)
    {
        LOG_VERBOSE(1, "Loading ssm a/dt weights");
        buffer<float> a;
        buffer<float> dt;
        q4nx.load_weights(a, _deltanet_weight_desc.linear_a.format_name(layer_idx));
        q4nx.load_weights(dt, _deltanet_weight_desc.linear_dt_bias.format_name(layer_idx));
        memcpy((unsigned char*)w_rms_ptr, a.data(), LINEAR_NV * sizeof(float));
        w_rms_ptr += LINEAR_NV * sizeof(float) / sizeof(bf16);
        memcpy((unsigned char*)w_rms_ptr, dt.data(), LINEAR_NV * sizeof(float));
        w_rms_ptr += LINEAR_NV * sizeof(float) / sizeof(bf16);
        _deltanet_weight_desc.linear_a.load();
        _deltanet_weight_desc.linear_dt_bias.load();
    }
    assert(w_rms_ptr == w_rms_ptr_end);
    bool all_ready = true;
    for (weight_desc_t* wd : _deltanet_weight_desc.all()) {
        if (!wd->ready()) {
            header_print("error", "Weight not ready: " << wd->name);
            all_ready = false;
        }
    }
    if (all_ready) {
        header_print("info", "DeltaNet weights ready!");
    } else {
        header_print("error", "Some DeltaNet weights were not ready!");
    }
}

void reorder_cpy(u8 *dst, buffer<u8> &src, const int col, const int vertical_blocks)
{
    // memcpy(dst, src.data(), src.size());
    // return;
    const int a_block_size = 16 * 256 * 9 / 16;
    const int blocks_per_row = col / 256;

    const int rows = src.size() / a_block_size / blocks_per_row;

    u8 *dst_ptr = dst;
    std::vector<u8 *> src_ptr(vertical_blocks);
    for (int i = 0; i < vertical_blocks; i++)
    {
        src_ptr[i] = src.data() + i * a_block_size * blocks_per_row;
    }
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
            {
                src_ptr[i] = src.data(); // useless padding
            }
        }
    }
}

void load_all_proj_weights(buffer<u8>& w, buffer<bf16>& w_rms, Q4NX& q4nx, int layer_idx, int DQ, int DK, int DV, int D, int INTERMEDIATE_SIZE, int DH) {
    constexpr int GLU_SLICE = 1024;
    constexpr int GLU_OUTDIM = 512;
    u8* w_ptr = w.data();
    u8* w_ptr_end = w.end();
    bf16* w_rms_ptr = w_rms.data();
    bf16* w_rms_ptr_end = w_rms.end();

    // norms
    _weight_desc.input_layer_norm = weight_desc_t(flm_bf16, {D},  "model.layers.%d.input_layernorm.weight");
    _weight_desc.post_attn_norm   = weight_desc_t(flm_bf16, {D},  "model.layers.%d.post_attention_layernorm.weight");
    _weight_desc.attn_q_norm      = weight_desc_t(flm_bf16, {DH}, "model.layers.%d.self_attn.q_norm.weight");
    _weight_desc.attn_k_norm      = weight_desc_t(flm_bf16, {DH}, "model.layers.%d.self_attn.k_norm.weight");
    // attention projections. q_proj on disk holds q and the attention gate merged
    // (out dim = 2*DQ); it is split in half below, so attn_gate is a separate
    // DQ-wide descriptor over the same q_proj tensor.
    _weight_desc.attn_q      = weight_desc_t(flm_q41, {D, DQ}, "model.layers.%d.self_attn.q_proj.weight");
    _weight_desc.attn_gate   = weight_desc_t(flm_q41, {D, DQ}, "model.layers.%d.self_attn.q_proj.weight");
    _weight_desc.attn_k      = weight_desc_t(flm_q41, {D, DK}, "model.layers.%d.self_attn.k_proj.weight");
    _weight_desc.attn_v      = weight_desc_t(flm_q41, {D, DV}, "model.layers.%d.self_attn.v_proj.weight");
    _weight_desc.attn_output = weight_desc_t(flm_q41, {DQ, D}, "model.layers.%d.self_attn.o_proj.weight");
    // mlp projections
    _weight_desc.mlp_up   = weight_desc_t(flm_q41, {D, INTERMEDIATE_SIZE}, "model.layers.%d.mlp.up_proj.weight");
    _weight_desc.mlp_gate = weight_desc_t(flm_q41, {D, INTERMEDIATE_SIZE}, "model.layers.%d.mlp.gate_proj.weight");
    _weight_desc.mlp_down = weight_desc_t(flm_q41, {INTERMEDIATE_SIZE, D}, "model.layers.%d.mlp.down_proj.weight");

    // offsets are managed manually via w_ptr below, so mark every descriptor as
    // independently managed (sets `added`); load() is called once its bytes are copied.
    for (weight_desc_t* wd : _weight_desc.all()) wd->indp();

    // read qkv + attention gate weights
    {
        assert(w_ptr < w_ptr_end);
        buffer<u8> w_q;
        buffer<u8> w_k;
        buffer<u8> w_v;
        q4nx.load_weights(w_q, _weight_desc.attn_q.format_name(layer_idx));
        q4nx.load_weights(w_k, _weight_desc.attn_k.format_name(layer_idx));
        q4nx.load_weights(w_v, _weight_desc.attn_v.format_name(layer_idx));
        LOG_VERBOSE(1, "Copying qkv weights to buffer");
        size_t w_q_gate_size = w_q.size();
        buffer<u8> w_q_q(w_q.data(),w_q_gate_size / 2);
        buffer<u8> w_gate(w_q.data() + w_q_gate_size / 2, w_q_gate_size / 2);
        reorder_cpy(w_ptr, w_q_q, D);
        w_ptr += _weight_desc.attn_q.get_size();
        reorder_cpy(w_ptr, w_k, D);
        w_ptr += _weight_desc.attn_k.get_size();
        reorder_cpy(w_ptr, w_v, D);
        w_ptr += _weight_desc.attn_v.get_size();
        reorder_cpy(w_ptr, w_gate, D);
        w_ptr += _weight_desc.attn_gate.get_size();
        _weight_desc.attn_q.load();
        _weight_desc.attn_k.load();
        _weight_desc.attn_v.load();
        _weight_desc.attn_gate.load();
    }
    // read o weights
    {
        LOG_VERBOSE(1, "Loading o weights");
        assert(w_ptr < w_ptr_end);
        buffer<u8> w_o;
        q4nx.load_weights(w_o, _weight_desc.attn_output.format_name(layer_idx));
        reorder_cpy(w_ptr, w_o, DQ);
        w_ptr += _weight_desc.attn_output.get_size();
        _weight_desc.attn_output.load();
    }
    // read up/gate weights, interleaved per GLU chunk
    {
        LOG_VERBOSE(1, "Loading up/gate weights");
        assert(w_ptr < w_ptr_end);
        buffer<u8> w_up;
        buffer<u8> w_gate;
        q4nx.load_weights(w_up, _weight_desc.mlp_up.format_name(layer_idx));
        q4nx.load_weights(w_gate, _weight_desc.mlp_gate.format_name(layer_idx));
        // For every D rows, half is up, half is gate
        const int blocks = 1024 / 2 / 32 * (D / 256);
        const int chunk_size = blocks * 32 * 256 * 5 / 8;
        const int total_size = INTERMEDIATE_SIZE * D * 5 / 8;
        const int phases = total_size / chunk_size;
        tensor_2d<u8> w_up_tensor(w_up, chunk_size, 0);
        tensor_2d<u8> w_gate_tensor(w_gate, chunk_size, 0);
        for (int i = 0; i < phases; i++)
        {
            LOG_VERBOSE(1, "Copying up and gate weights to buffer, phase " << i + 1 << "/" << phases);
            reorder_cpy(w_ptr, w_up_tensor[i], D);
            w_ptr += chunk_size;
            reorder_cpy(w_ptr, w_gate_tensor[i], D);
            w_ptr += chunk_size;
        }
        _weight_desc.mlp_up.load();
        _weight_desc.mlp_gate.load();
    }
    // read down weights
    {
        LOG_VERBOSE(1, "Loading down weights");
        assert(w_ptr < w_ptr_end);
        buffer<u8> w_down;
        q4nx.load_weights(w_down, _weight_desc.mlp_down.format_name(layer_idx));
        reorder_cpy(w_ptr, w_down, INTERMEDIATE_SIZE);
        w_ptr += _weight_desc.mlp_down.get_size();
        _weight_desc.mlp_down.load();
    }
    assert(w_ptr == w_ptr_end);
    // read rms weights
    {
        LOG_VERBOSE(1, "Loading rms weights");
        buffer<bf16> w_rms_pre;
        buffer<bf16> w_rms_post;
        assert(w_rms_ptr < w_rms_ptr_end);
        q4nx.load_weights(w_rms_pre, _weight_desc.input_layer_norm.format_name(layer_idx));
        q4nx.load_weights(w_rms_post, _weight_desc.post_attn_norm.format_name(layer_idx));

        // w_rms_post.memset(1);
        memcpy((unsigned char *)w_rms_ptr, w_rms_pre.data(), D * sizeof(bf16));
        w_rms_ptr += D;
        memcpy((unsigned char *)w_rms_ptr, w_rms_post.data(), D * sizeof(bf16));
        w_rms_ptr += D;
        _weight_desc.input_layer_norm.load();
        _weight_desc.post_attn_norm.load();
    }
    // read attention norm weights
    {
        LOG_VERBOSE(1, "Loading attention norm weights");
        buffer<bf16> q_attn_norm;
        buffer<bf16> k_attn_norm;
        q4nx.load_weights(q_attn_norm, _weight_desc.attn_q_norm.format_name(layer_idx));
        q4nx.load_weights(k_attn_norm, _weight_desc.attn_k_norm.format_name(layer_idx));
        memcpy((unsigned char *)w_rms_ptr, q_attn_norm.data(), DH * sizeof(bf16));
        w_rms_ptr += DH;
        memcpy((unsigned char *)w_rms_ptr, k_attn_norm.data(), DH * sizeof(bf16));
        w_rms_ptr += DH;
        _weight_desc.attn_q_norm.load();
        _weight_desc.attn_k_norm.load();
    }
    assert(w_rms_ptr == w_rms_ptr_end);
    bool all_ready = true;
    for (weight_desc_t* w : _weight_desc.all()) {
        if (!w->ready()) {
            header_print("error", "Weight not ready: " << w->name);
            all_ready = false;
        }
    }
    if (all_ready) {
        header_print("info", "Weights ready!");
    } else {
        header_print("error", "Some weights were not ready!");
    }
}

void rms_norm(buffer<bf16>& y, buffer<bf16>& x, buffer<bf16>& rms_weight, float eps) {
    int total_elements = x.size();
    int hidden_size = rms_weight.size();

    if (hidden_size == 0 || total_elements % hidden_size != 0) {
        // Simple error handling for mismatch
        printf("Error: RMS Norm input size %d not divisible by hidden size %d\n", total_elements, hidden_size);
        return;
    }

    int num_tokens = total_elements / hidden_size;
    const int simd_width = 16;

    for (int t = 0; t < num_tokens; ++t) {
        int offset = t * hidden_size;
        bf16* x_ptr = &x[offset];
        bf16* y_ptr = &y[offset];
        
        __m512 sum_xx_vec = _mm512_setzero_ps();
        
        // Step 1: Compute sum of squares using AVX-512
        int i = 0;
        for (; i <= hidden_size - simd_width; i += simd_width) {
            __m256i bf16_vals = _mm256_loadu_si256((__m256i*)&x_ptr[i]);

            // Convert BF16 to FP32
            __m512 fp32_vals = bf16o_fp32_512(bf16_vals);
            sum_xx_vec = _mm512_fmadd_ps(fp32_vals, fp32_vals, sum_xx_vec);  // sum_xx += x[i] * x[i];
        }

        // Horizontal sum of the AVX-512 register
        f32 sum_xx = _mm512_reduce_add_ps(sum_xx_vec);

        sum_xx /= (f32)hidden_size;
        f32 rms_x = sqrtf(sum_xx + eps);

        // Step 2: Compute element-wise operation y[i] = w[i] * x[i] / rms_x
        __m512 rms_x_vec = _mm512_set1_ps(rms_x);
        for (i = 0; i <= hidden_size - simd_width; i += simd_width) {
            __m256i bf16_vals = _mm256_loadu_si256((__m256i*)&x_ptr[i]);
            __m512 fp32_vals = bf16o_fp32_512(bf16_vals);
            __m256i bf16_vals_w = _mm256_loadu_si256((__m256i*)&rms_weight[i]);
            __m512 fp32_vals_w = bf16o_fp32_512(bf16_vals_w);
            __m512 y_vec = _mm512_div_ps(_mm512_mul_ps(fp32_vals_w, fp32_vals), rms_x_vec);
            __m256i result_bf16 = f32o_bf16_512(y_vec);
            _mm256_storeu_si256((__m256i*)&y_ptr[i], result_bf16);
        }
    }
}
