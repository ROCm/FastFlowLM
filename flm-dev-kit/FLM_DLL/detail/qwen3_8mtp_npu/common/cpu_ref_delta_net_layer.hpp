// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file common/cpu_ref_delta_net_layer.hpp
/// \brief CPU golden reference for one gated-delta-net block.
///
/// Reads the same model.q4nx the NPU does, so a diff against it isolates NPU
/// error from quantisation error -- both sides start from the same bytes.
///
/// Math (transformers Qwen3_5GatedDeltaNet, checked line by line):
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
/// Geometry is runtime, so one build covers the whole Qwen3.5 / Qwen3.8 family.
#pragma once

#include "common/cpu_ref_common.hpp"

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

    // ======================================================================
    // the stages of forward(), by name
    // ======================================================================
    /// forward() is re-expressed over exactly these, so a caller that wants to
    /// run the six matmuls somewhere else can drive the rest of the layer
    /// without reimplementing any of it. Splitting it is what makes the NPU
    /// projection path (prefill/npu/delta_proj.hpp) a *rearrangement* of this
    /// file rather than a second copy of the math that has to be kept in step.
    ///
    /// Two of them mutate; the rest are const. That split is the whole
    /// contract, and it is why the recurrence can stay here while the
    /// projections leave:
    ///
    ///   conv_silu   advances conv_state_ by L
    ///   delta_rule  advances state_ by L
    ///
    /// Everything else is a pure function of its inputs and the weights. A
    /// caller must run those two exactly once per token, in order, or the
    /// layer's history diverges silently -- there is no inverse to the fold.

    /// rmsnorm by input_layernorm. \param xn [L][D] out, resized here
    void in_norm(const float* x, int L, std::vector<float>& xn) const;

    /// The two dense bf16 gate projections. \param aa,bb [L][v_heads], resized
    void alpha_beta(const float* xn, int L,
                    std::vector<float>& aa, std::vector<float>& bb) const;

    /// Causal depthwise conv1d(k=4) then silu, in place on [L][conv_dim].
    /// **Advances conv_state_ by L.**
    void conv_silu(float* qkv, int L);

    /// beta = sigmoid(bb); alpha = exp(ssm_a * softplus(aa + dt_bias)).
    /// \param alpha,beta [L][v_heads] out, resized here
    void gates(const float* aa, const float* bb, int L,
               std::vector<float>& alpha, std::vector<float>& beta) const;

    /// The gated delta rule. **Advances state_ by L.**
    /// \param qkv [L][conv_dim] post-conv \param core [L][dv] out, resized here
    void delta_rule(const float* qkv, const float* alpha, const float* beta,
                    int L, std::vector<float>& core);

    /// Per-head RMSNorm, then * ssm_norm * silu(z). In place on core [L][dv].
    void gated_rmsnorm(float* core, const float* zz, int L) const;

    /// rmsnorm by post_attention_layernorm. \param hn [L][D] out, resized here
    void post_norm(const float* h1, int L, std::vector<float>& hn) const;

    /// g = silu(gate) * up, elementwise over [L][intermediate_size].
    /// The reference the fused up|gate dispatch is scored against.
    void swiglu(const float* gate, const float* up, int L,
                std::vector<float>& g) const;

    /// Bump the token counter. The caller of the stages owes exactly one of
    /// these per window, after conv_silu and delta_rule have run.
    void advance(int L) { seen_ += L; }

    const delta_net_geometry& geometry()    const { return g_; }
    const delta_net_weights&  weights()     const { return w_; }
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

    /// The same two, writable, for a caller that advanced the history somewhere
    /// else -- today `prefill/npu/delta_net_op.hpp`, which runs conv_silu and
    /// delta_rule on two bitstreams and hands the advanced state back.
    ///
    /// Such a caller owns the advance *instead of*, never as well as, those two
    /// stages: the fold is destructive and has no inverse, so running both
    /// paths over the same tokens silently doubles the history.
    /// `advance(L)` is still owed exactly
    /// once. Purely additive -- forward()'s arithmetic does not read these.
    std::vector<float>& recurrent_state_mut() { return state_; }
    std::vector<float>& conv_window_mut()     { return conv_state_; }

    /// Byte size of one snapshot: state_ + conv_state_ + the token counter.
    size_t state_bytes() const {
        return state_.size() * sizeof(float)
             + conv_state_.size() * sizeof(float)
             + sizeof(int);
    }

    /// Scratch-copy-and-commit, for speculative decode.
    ///
    /// S_t = e^{g_t} S_{t-1} + k_t (v_t - S_t^T k_t)^T is a destructive fold
    /// with no inverse, so a rejected draft token must never have touched the
    /// committed state -- the only way back is a saved copy. Both are additive
    /// and do not touch forward()'s verified arithmetic.
    void save_state(void* dst) const {
        uint8_t* p = static_cast<uint8_t*>(dst);
        const size_t sn = state_.size()      * sizeof(float);
        const size_t cn = conv_state_.size() * sizeof(float);
        std::memcpy(p, state_.data(),      sn); p += sn;
        std::memcpy(p, conv_state_.data(), cn); p += cn;
        std::memcpy(p, &seen_, sizeof seen_);
    }
    void load_state(const void* src) {
        const uint8_t* p = static_cast<const uint8_t*>(src);
        const size_t sn = state_.size()      * sizeof(float);
        const size_t cn = conv_state_.size() * sizeof(float);
        std::memcpy(state_.data(),      p, sn); p += sn;
        std::memcpy(conv_state_.data(), p, cn); p += cn;
        std::memcpy(&seen_, p, sizeof seen_);
    }

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
