// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file cpu_ref_delta_net_layer.cpp
/// \brief Implementation of the CPU gated-delta-net reference; see the header
///        for the math and the layout notes.

#include "cpu_ref_delta_net_layer.hpp"

#include <sstream>

namespace cpu_ref {

// ==========================================================================
// geometry
// ==========================================================================
void delta_net_geometry::validate() const {
    auto need = [](bool ok, const char* what) {
        if (!ok) throw std::runtime_error(std::string("delta_net_geometry: ") + what);
    };
    need(model_dim > 0 && model_dim % Q4NX_COL_BLOCK == 0,
         "model_dim must be a positive multiple of 256");
    need(intermediate_size > 0, "intermediate_size must be positive");
    need(qk_heads > 0 && v_heads > 0, "head counts must be positive");
    need(v_heads % qk_heads == 0, "v_heads must be a multiple of qk_heads");
    need(k_head_dim > 0 && v_head_dim > 0, "head dims must be positive");
    need(conv_k > 0, "conv_k must be positive");
}

std::string delta_net_geometry::str() const {
    std::ostringstream s;
    s << "D=" << model_dim << " ffn=" << intermediate_size
      << " qk_heads=" << qk_heads << " v_heads=" << v_heads
      << " (expand " << expand() << ")"
      << " dk=" << k_head_dim << " dv=" << v_head_dim
      << " conv_k=" << conv_k
      << " | DQ=" << dq() << " DK=" << dk() << " DV=" << dv()
      << " conv_dim=" << conv_dim();
    return s.str();
}

// ==========================================================================
// weights
// ==========================================================================
void delta_net_weights::load(const safetensors& f, const delta_net_geometry& g, int layer) {
    const std::string p = "model.layers." + std::to_string(layer) + ".";
    const std::string l = p + "linear_attn.";

    qkv      = get_q4nx(f, l + "qkv_proj.weight");
    z        = get_q4nx(f, p + "self_attn.gate_proj.weight");
    out      = get_q4nx(f, l + "ssm_out_proj.weight");
    mlp_gate = get_q4nx(f, p + "mlp.gate_proj.weight");
    mlp_up   = get_q4nx(f, p + "mlp.up_proj.weight");
    mlp_down = get_q4nx(f, p + "mlp.down_proj.weight");

    in_ln    = f.f32(p + "input_layernorm.weight");
    post_ln  = f.f32(p + "post_attention_layernorm.weight");
    conv_w   = f.f32(l + "ssm_conv1d.weight");
    ssm_norm = f.f32(l + "ssm_norm.weight");
    ssm_a    = f.f32(l + "ssm_a");
    dt_bias  = f.f32(l + "ssm_dt.bias");
    // The quantized alpha/beta copies are Q8_0 chunks that only the decode
    // kernel consumes; the converter also emits a dense bf16 copy in the same
    // (group-major) row order, which is what the prefill path -- and this
    // reference -- uses. So a diff against the NPU decode path carries the
    // Q8_0-vs-bf16 difference on these two small projections.
    alpha_w  = f.f32(l + "ssm_alpha_proj.bf16.weight");
    beta_w   = f.f32(l + "ssm_beta_proj.bf16.weight");

    auto check = [&](bool ok, const std::string& what) {
        if (!ok) throw std::runtime_error("layer " + std::to_string(layer) + ": " + what);
    };
    check(qkv.rows == g.conv_dim() && qkv.cols == g.model_dim, "qkv_proj geometry mismatch");
    check(z.rows == g.dv() && z.cols == g.model_dim,           "gate_proj (z) geometry mismatch");
    check(out.rows == g.model_dim && out.cols == g.dv(),       "ssm_out_proj geometry mismatch");
    check(conv_w.size() == static_cast<size_t>(g.conv_k) * g.conv_dim(),
          "ssm_conv1d geometry mismatch");
    check(ssm_a.size() >= static_cast<size_t>(g.v_heads),   "ssm_a too short");
    check(dt_bias.size() >= static_cast<size_t>(g.v_heads), "ssm_dt.bias too short");
    check(ssm_norm.size() >= static_cast<size_t>(g.v_head_dim), "ssm_norm too short");
    check(alpha_w.size() >= static_cast<size_t>(g.v_heads) * g.model_dim, "alpha proj too short");
    check(beta_w.size()  >= static_cast<size_t>(g.v_heads) * g.model_dim, "beta proj too short");
}

// ==========================================================================
// the layer
// ==========================================================================
delta_net_layer::delta_net_layer(const safetensors& q4nx, const delta_net_geometry& geo,
                                 int layer)
    : g_(geo), layer_(layer) {
    g_.validate();
    w_.load(q4nx, g_, layer);
    reset();
}

void delta_net_layer::reset() {
    conv_state_.assign(static_cast<size_t>(g_.conv_k) * g_.conv_dim(), 0.0f);
    state_.assign(static_cast<size_t>(g_.v_heads) * g_.k_head_dim * g_.v_head_dim, 0.0f);
    seen_ = 0;
}

void delta_net_layer::forward(const float* x, int L, float* y) {
    const size_t LD       = static_cast<size_t>(L);
    const int    D        = g_.model_dim;
    const int    CONV_DIM = g_.conv_dim();
    const int    DQ       = g_.dq();
    const int    DK       = g_.dk();
    const int    DV       = g_.dv();
    const int    NV       = g_.v_heads;
    const int    DKH      = g_.k_head_dim;
    const int    DVH      = g_.v_head_dim;
    const int    EXPAND   = g_.expand();

    std::vector<float> xn(LD * D);
    for (int t = 0; t < L; t++)
        rms_norm(x + t * static_cast<size_t>(D), w_.in_ln.data(), D,
                 xn.data() + t * static_cast<size_t>(D), g_.rms_eps);

    // ---- projections -----------------------------------------------------
    std::vector<float> qkv(LD * CONV_DIM);
    std::vector<float> zz (LD * DV);
    std::vector<float> aa (LD * NV), bb(LD * NV);
    q4nx_matmul(w_.qkv, xn.data(), L, qkv.data());
    q4nx_matmul(w_.z,   xn.data(), L, zz.data());
    dense_matmul(w_.alpha_w.data(), NV, D, xn.data(), L, aa.data());
    dense_matmul(w_.beta_w .data(), NV, D, xn.data(), L, bb.data());

    // ---- causal depthwise conv + silu ------------------------------------
    // conv_state_ holds the last conv_k *pre-conv* inputs, newest in the last
    // slot; that is exactly transformers' conv_states cache layout.
    {
        const float* cw = w_.conv_w.data();   // [tap][channel]
        const int    K  = g_.conv_k;
#pragma omp parallel for schedule(static)
        for (int c = 0; c < CONV_DIM; c++) {
            std::vector<float> win(K);
            for (int k = 0; k < K; k++) win[k] = conv_state_[static_cast<size_t>(k) * CONV_DIM + c];
            for (int t = 0; t < L; t++) {
                for (int k = 0; k + 1 < K; k++) win[k] = win[k + 1];
                win[K - 1] = qkv[static_cast<size_t>(t) * CONV_DIM + c];
                float acc = 0.0f;
                for (int k = 0; k < K; k++) acc += win[k] * cw[static_cast<size_t>(k) * CONV_DIM + c];
                qkv[static_cast<size_t>(t) * CONV_DIM + c] = siluf_(acc);
            }
            for (int k = 0; k < K; k++) conv_state_[static_cast<size_t>(k) * CONV_DIM + c] = win[k];
        }
    }

    // ---- per-head gates --------------------------------------------------
    std::vector<float> beta(LD * NV), alpha(LD * NV);
    for (int t = 0; t < L; t++)
        for (int h = 0; h < NV; h++) {
            const size_t i = static_cast<size_t>(t) * NV + h;
            beta[i]  = sigmoidf_(bb[i]);
            alpha[i] = std::exp(w_.ssm_a[h] * softplusf_(aa[i] + w_.dt_bias[h]));
        }

    // ---- gated delta rule ------------------------------------------------
    std::vector<float> core(LD * DV);
#pragma omp parallel for schedule(static)
    for (int h = 0; h < NV; h++) {
        const int gq = h / EXPAND;                       // repeat_interleave(expand)
        float* S = state_.data() + static_cast<size_t>(h) * DKH * DVH;   // [dk][dv]
        std::vector<float> qn(DKH), kn(DKH), kv(DVH), dl(DVH);
        for (int t = 0; t < L; t++) {
            const float* base = qkv.data() + static_cast<size_t>(t) * CONV_DIM;
            l2_norm(base + gq * DKH,        DKH, qn.data(), g_.qk_scale(), g_.l2_eps);
            l2_norm(base + DQ + gq * DKH,   DKH, kn.data(), 1.0f,          g_.l2_eps);
            const float* vt = base + DQ + DK + h * DVH;
            const float a = alpha[static_cast<size_t>(t) * NV + h];
            const float b = beta [static_cast<size_t>(t) * NV + h];

            for (size_t i = 0; i < static_cast<size_t>(DKH) * DVH; i++) S[i] *= a;

            for (int dv = 0; dv < DVH; dv++) kv[dv] = 0.0f;
            for (int dk = 0; dk < DKH; dk++) {
                const float kk = kn[dk];
                if (kk == 0.0f) continue;
                const float* Sr = S + static_cast<size_t>(dk) * DVH;
                for (int dv = 0; dv < DVH; dv++) kv[dv] += Sr[dv] * kk;
            }
            for (int dv = 0; dv < DVH; dv++) dl[dv] = (vt[dv] - kv[dv]) * b;

            float* o = core.data() + static_cast<size_t>(t) * DV + h * DVH;
            for (int dv = 0; dv < DVH; dv++) o[dv] = 0.0f;
            for (int dk = 0; dk < DKH; dk++) {
                float* Sr = S + static_cast<size_t>(dk) * DVH;
                const float kk = kn[dk], qq = qn[dk];
                for (int dv = 0; dv < DVH; dv++) {
                    Sr[dv] += kk * dl[dv];
                    o[dv]  += Sr[dv] * qq;
                }
            }
        }
    }

    // ---- gated RMSNorm (norm first, then silu gate) ----------------------
    for (int t = 0; t < L; t++)
        for (int h = 0; h < NV; h++) {
            float* o       = core.data() + static_cast<size_t>(t) * DV + h * DVH;
            const float* g = zz.data()   + static_cast<size_t>(t) * DV + h * DVH;
            float acc = 0.0f;
            for (int i = 0; i < DVH; i++) acc += o[i] * o[i];
            const float r = 1.0f / std::sqrt(acc / DVH + g_.rms_eps);
            for (int i = 0; i < DVH; i++)
                o[i] = o[i] * r * w_.ssm_norm[i] * siluf_(g[i]);
        }

    // ---- out_proj + residual ---------------------------------------------
    std::vector<float> h1(LD * D);
    q4nx_matmul(w_.out, core.data(), L, h1.data());
    for (size_t i = 0; i < LD * D; i++) h1[i] += x[i];

    // ---- MLP + residual ---------------------------------------------------
    mlp_residual(h1.data(), L, y);
    seen_ += L;
}

void delta_net_layer::mlp_residual(const float* h1, int L, float* y) {
    const size_t LD = static_cast<size_t>(L);
    const int    D  = g_.model_dim;
    const int    FF = g_.intermediate_size;

    std::vector<float> hn(LD * D);
    for (int t = 0; t < L; t++)
        rms_norm(h1 + t * static_cast<size_t>(D), w_.post_ln.data(), D,
                 hn.data() + t * static_cast<size_t>(D), g_.rms_eps);

    std::vector<float> g(LD * FF), u(LD * FF);
    q4nx_matmul(w_.mlp_gate, hn.data(), L, g.data());
    q4nx_matmul(w_.mlp_up,   hn.data(), L, u.data());
    for (size_t i = 0; i < LD * FF; i++) g[i] = siluf_(g[i]) * u[i];
    q4nx_matmul(w_.mlp_down, g.data(), L, y);
    for (size_t i = 0; i < LD * D; i++) y[i] += h1[i];
}

} // namespace cpu_ref
