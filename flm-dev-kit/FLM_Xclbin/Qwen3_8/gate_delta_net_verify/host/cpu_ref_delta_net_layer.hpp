// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file cpu_ref_delta_net_layer.hpp
/// \brief CPU golden reference for one gated-delta-net block.
///
/// Reads the same model.q4nx the NPU does and runs the block in fp32, so the
/// host can diff an NPU layer against it directly instead of against a
/// safetensors dump captured from transformers. The two differ in what they
/// catch: the dump is the true ground truth but only exists for the one
/// prompt it was captured with, while this reference runs on whatever tokens
/// the host feeds the NPU and isolates *NPU* error from quantisation error,
/// since both sides start from the same quantised weights.
///
/// Math (transformers Qwen3_5GatedDeltaNet, checked line by line, verified
/// against the reference dumps):
///
///     xn        = rmsnorm(x, input_layernorm)
///     qkv       = W_qkv xn                       [conv_dim]
///     z         = W_z   xn                       [dv]
///     a, b      = W_a xn, W_b xn                 [v_heads] each
///     qkv       = silu(depthwise_conv1d_k4(qkv)) causal, state carried
///     q,k,v     = split(qkv, dq | dk | dv)
///     q,k       = l2norm(.), q *= 1/sqrt(k_head_dim), repeat_interleave(expand)
///     beta      = sigmoid(b)
///     g         = ssm_a * softplus(a + dt_bias)     (ssm_a is already -exp(A_log))
///     S_t       = e^{g_t} S_{t-1};  S_t += k_t (beta_t (v_t - S_t^T k_t))^T
///     o_t       = S_t^T q_t
///     o         = rmsnorm_nogain(o) * ssm_norm * silu(z)
///     mixer_out = W_out o
///     h         = x + mixer_out
///     y         = h + down(silu(gate(rmsnorm(h))) * up(rmsnorm(h)))
///
/// Everything is fp32 and the geometry is runtime, so one build covers the
/// whole Qwen3.5 / Qwen3.8 family. This is a reference for the *math*, not a
/// bit-accurate model of the AIE datapath.
#pragma once

#include "cpu_ref_common.hpp"

namespace cpu_ref {

// ==========================================================================
// geometry
// ==========================================================================
/// Everything about a delta-net block that varies across the model family.
/// Defaults are Qwen3.8-27B; build one from an LM_Config with
/// cpu_ref::geometry_from() in cpu_ref_bridge.hpp.
struct delta_net_geometry {
    int   model_dim        = q38::MODEL_DIM;
    int   intermediate_size = q38::INTERMEDIATE_SIZE;
    int   qk_heads         = q38::L_QK_HEADS;
    int   v_heads          = q38::L_V_HEADS;
    int   k_head_dim       = q38::L_K_HEAD_DIM;
    int   v_head_dim       = q38::L_V_HEAD_DIM;
    int   conv_k           = q38::CONV_K;
    float rms_eps          = q38::RMS_EPS;
    float l2_eps           = q38::L2N_EPS;

    /// repeat_interleave factor: how many v heads share one q/k head.
    int   expand()   const { return v_heads / qk_heads; }
    int   dq()       const { return qk_heads * k_head_dim; }
    int   dk()       const { return dq(); }
    int   dv()       const { return v_heads * v_head_dim; }
    /// qkv_proj row count, and the depthwise conv's channel count.
    int   conv_dim() const { return dq() + dk() + dv(); }
    float qk_scale() const { return 1.0f / std::sqrt(static_cast<float>(k_head_dim)); }

    /// Throw if the numbers cannot describe a delta-net block.
    void validate() const;
    std::string str() const;
};

// ==========================================================================
// weights
// ==========================================================================
/// Views into the mmap'd .q4nx (quantised) plus fp32 copies of the small
/// dense tensors. Cheap to construct; holds no bulk data of its own.
struct delta_net_weights {
    q4nx_tensor qkv;        // [conv_dim][D]  linear_attn.qkv_proj
    q4nx_tensor z;          // [dv][D]        self_attn.gate_proj
    q4nx_tensor out;        // [D][dv]        linear_attn.ssm_out_proj
    q4nx_tensor mlp_gate;   // [intermediate][D]
    q4nx_tensor mlp_up;     // [intermediate][D]
    q4nx_tensor mlp_down;   // [D][intermediate]

    std::vector<float> in_ln;      // [D]
    std::vector<float> post_ln;    // [D]
    std::vector<float> conv_w;     // [conv_k][conv_dim], tap-major
    std::vector<float> ssm_norm;   // [v_head_dim]
    std::vector<float> ssm_a;      // [v_heads], already negative
    std::vector<float> dt_bias;    // [v_heads]
    std::vector<float> alpha_w;    // [v_heads][D]  in_proj_a (bf16 copy)
    std::vector<float> beta_w;     // [v_heads][D]  in_proj_b (bf16 copy)

    void load(const safetensors& q4nx, const delta_net_geometry& g, int layer);
};

// ==========================================================================
// the layer
// ==========================================================================
class delta_net_layer {
public:
    /// \param q4nx  an open model.q4nx
    /// \param geo   block geometry; must match the checkpoint
    /// \param layer layer index, matching linear_decoding_layer::load_weights
    delta_net_layer(const safetensors& q4nx, const delta_net_geometry& geo, int layer);

    /// Zero the conv window and the recurrent state (start of a sequence).
    void reset();

    /// x, y: [L][model_dim] row-major fp32. Advances conv + recurrent state by L,
    /// so calling it once with L tokens and calling it L times with one token
    /// each give the same answer -- which is what lets the host compare a
    /// batched CPU run against the NPU's token-at-a-time loop.
    void forward(const float* x, int L, float* y);

    const delta_net_geometry& geometry()    const { return g_; }
    int                       layer()       const { return layer_; }
    int                       tokens_seen() const { return seen_; }

    /// [v_heads][k_head_dim][v_head_dim], to diff against get_S_cache().
    const std::vector<float>& recurrent_state() const { return state_; }
    /// Row `head` of the above.
    const float* head_state(int head) const {
        return state_.data() + static_cast<size_t>(head) * g_.k_head_dim * g_.v_head_dim;
    }
    /// [conv_k][conv_dim], newest tap last, to diff against get_conv_cache().
    const std::vector<float>& conv_window() const { return conv_state_; }

private:
    void mlp_residual(const float* h1, int L, float* y);

    delta_net_geometry g_;
    int                layer_;
    delta_net_weights  w_;
    std::vector<float> conv_state_;   // [conv_k][conv_dim]
    std::vector<float> state_;        // [v_heads][k_head_dim][v_head_dim]
    int                seen_ = 0;
};

} // namespace cpu_ref
