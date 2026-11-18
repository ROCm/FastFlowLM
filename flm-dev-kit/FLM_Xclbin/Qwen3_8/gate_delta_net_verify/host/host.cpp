// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file host.cpp
/// \brief Single-layer NPU test bench for the Qwen3.8-27B delta-net layer.
///
/// The layer is end to end now, and two things are scored: its output
///
///     h2 = h1 + mlp.down_proj( silu(gate) * up ),   h1 = x + ssm_out_proj(...)
///
/// and the recurrent state. Everything between -- the input norm, the qkv / z /
/// a / b projections, the causal conv1d, the gated delta rule, the gated
/// RMSNorm, ssm_out_proj, the post-attention norm, the fused up | gate
/// projection and the SwiGLU -- is computed on the CPU too, because the golden
/// depends on all of it, but none of it is compared: the array keeps those
/// intermediates on chip or in scratch DDR and the host no longer reads them.
///
/// The reference is the CPU golden in cpu_ref_*, not the transformers dump. The
/// dump is the true ground truth but only exists for the one prompt it was
/// captured with, and it folds NPU error together with quantisation error. The
/// CPU reference starts from the same quantised model.q4nx the NPU does, so a
/// disagreement is an NPU-side bug and nothing else. The dump supplies the
/// inputs (`verify.inputs_embeds`, 8 rows) and nothing more.

#include "typedef.hpp"
#include "buffer.hpp"
#include "linear_decoding_layer.hpp"
#include "npu_utils/npu_utils.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"
#include "utils/debug_utils.hpp"
#include "utils/vm_args.hpp"
#include "utils/utils.hpp"
#include "models.h"
#include "metrices.hpp"
#include "tensor_2d.hpp"
#include "lm_config.hpp"

#include "cpu_ref_bridge.hpp"

#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

/// linear_decoding_layer::load_weights() pins itself to layer 0, so the CPU
/// reference has to load the same one.
constexpr int kDeltaNetLayer = 0;

// ==========================================================================
// shared setup
// ==========================================================================
/// Everything the test needs that is model- rather than layer-specific.
struct test_env {
    model_list::model_t model;
    std::string         model_path;
    LM_Config&          lm_config;
    Q4NX&               q4nx;
    npu_app_manager*    layer_app_mgr;
    buffer<bf16>&       embed;        ///< model.embed_tokens.weight
    int                 D;
};

/// Gather the rows of the embedding table named by `ids` and score them against
/// the dump's own copy, so a token-id or vocab mismatch is caught here rather
/// than showing up as a mystery layer error. The layer is still fed the dump's
/// embeddings, so this is a check and not a data path.
void check_embeddings(const test_env& env, buffer<int32_t>& ids, buffer<bf16>& ref_embed) {
    header_print("info", "input ids:");
    utils::print_matrix(ids, ids.size());

    // Quantised embedding table: main() left `embed` empty, nothing to gather.
    if (env.embed.size() == 0) return;

    buffer<bf16> embeddings(ref_embed.size());
    for (size_t i = 0; i < ids.size(); i++) {
        memcpy(embeddings.data() + i * env.D,
               env.embed.data() + static_cast<size_t>(ids[i]) * env.D,
               env.D * sizeof(bf16));
    }
    print_error_metrics(get_error_metrics(embeddings, ref_embed),
                        "Model Embedding Error Metrics");
}

/// Per-token cosine plus the aggregate, the shape the test reports in.
void report(const std::string& title, tensor_2d<bf16>& got, tensor_2d<bf16>& ref, int steps) {
    header_print("info", title);
    for (int i = 0; i < steps; i++) {
        std::cout << "\tStep " << i << " Cosim: "
                  << get_error_metrics(got[i], ref[i]).CosineSimilarity << std::endl;
    }
}

// ==========================================================================
// the whole layer, scored at h2 and at the recurrent state
// ==========================================================================
void run_layer_test(const test_env& env, const std::string& ref_path) {
    const int D = env.D;
    const int L = L_TOKENS;

    const cpu_ref::delta_net_geometry g = cpu_ref::geometry_from(env.lm_config);
    const int DQ  = g.dq();
    const int DK  = g.dk();
    const int DV  = g.dv();
    const int DKH = g.k_head_dim;
    const int DVH = g.v_head_dim;
    const int NV  = g.v_heads;
    const int EXPAND   = g.expand();
    const int CONV_DIM = g.conv_dim();
    const int K = g.conv_k;

    header_print("info", "=== delta-net layer " +
                             std::to_string(kDeltaNetLayer) + ", " + std::to_string(L) +
                             " rows, inputs from " + ref_path + " ===");
    std::unique_ptr<SafeTensors> references = std::make_unique<SafeTensors>(ref_path);

    buffer<bf16>    ref_embed;
    buffer<int32_t> ids;
    references->load_weights(ref_embed, "verify.inputs_embeds");
    references->load_weights(ids, "verify.input_ids");
    check_embeddings(env, ids, ref_embed);

    if (ref_embed.size() < static_cast<size_t>(L) * D) {
        std::cerr << "dump has only " << ref_embed.size() / D << " rows, need " << L << std::endl;
        return;
    }

    buffer<bf16> input(static_cast<size_t>(L) * D);
    memcpy(input.data(), ref_embed.data(), static_cast<size_t>(L) * D * sizeof(bf16));
    tensor_2d<bf16> inputs(input, D);

    // ======================================================================
    // CPU golden. Every stage below feeds the next; only the last one is
    // scored. The bf16 rounds are not cosmetic -- they mark the points where
    // the array actually narrows, and dropping one would charge this test for
    // precision the hardware never had.
    // ======================================================================

    // ---- input_layernorm ---------------------------------------------------
    cpu_ref::safetensors q4nx_file(cpu_ref::q4nx_path_of(env.model_path));
    const std::string prefix = "model.layers." + std::to_string(kDeltaNetLayer) + ".";

    std::vector<float> ln_w = q4nx_file.f32(prefix + "input_layernorm.weight");
    if (static_cast<int>(ln_w.size()) != D) {
        std::cerr << prefix << "input_layernorm.weight has " << ln_w.size()
                  << " elements, expected " << D << std::endl;
        return;
    }
    std::vector<float> normed(static_cast<size_t>(L) * D);
    for (int t = 0; t < L; t++) {
        std::vector<float> x_t = cpu_ref::to_f32(inputs[t], D);
        cpu_ref::rms_norm(x_t.data(), ln_w.data(), D,
                          normed.data() + static_cast<size_t>(t) * D,
                          env.lm_config.rms_norm_eps);
    }
    for (float& v : normed) v = static_cast<float>(static_cast<bf16>(v));

    // ---- qkv, conv + SiLU, per-head l2_norm --------------------------------
    cpu_ref::q4nx_tensor W_qkv = cpu_ref::get_q4nx(q4nx_file, prefix + "linear_attn.qkv_proj.weight");
    if (W_qkv.cols != D || W_qkv.rows != CONV_DIM) {
        std::cerr << "qkv_proj is " << W_qkv.rows << " x " << W_qkv.cols << ", expected "
                  << CONV_DIM << " x " << D << std::endl;
        return;
    }
    std::vector<float> qkv(static_cast<size_t>(L) * CONV_DIM);
    cpu_ref::q4nx_matmul(W_qkv, normed.data(), L, qkv.data());
    for (float& v : qkv) v = static_cast<float>(static_cast<bf16>(v));

    std::vector<float> conv_w = q4nx_file.f32(prefix + "linear_attn.ssm_conv1d.weight");
    if (conv_w.size() != static_cast<size_t>(K) * CONV_DIM) {
        std::cerr << prefix << "linear_attn.ssm_conv1d.weight has " << conv_w.size()
                  << " elements, expected " << static_cast<size_t>(K) * CONV_DIM << std::endl;
        return;
    }
    const float* cw = conv_w.data();
    // Every accept length's conv cache. Snapshotted here because the loop below
    // overwrites qkv in place with the post-conv, post-SiLU values.
    //
    // Speculative decode does not know how many of the L drafted tokens will
    // survive, so the pass emits a window per accept length. Accepting j of
    // them leaves the next pass starting at token j, whose window is the K - 1
    // *pre-conv* rows before it: token indices j - (K - 1) .. j - 1. For
    // j < K - 1 that reaches back past token 0 into this pass's incoming cache,
    // which the constructor zeroes. cpu_conv_out[j - 1] is accept-j, so the
    // last entry is the old accept-everything reference.
    std::vector<std::vector<float>> cpu_conv_out(
        L, std::vector<float>(static_cast<size_t>(K - 1) * CONV_DIM, 0.0f));
    for (int j = 1; j <= L; j++)
        for (int k = 0; k < K - 1; k++) {
            const int t = j - (K - 1) + k;
            if (t < 0) continue;              // the zeroed incoming cache
            memcpy(cpu_conv_out[j - 1].data() + static_cast<size_t>(k) * CONV_DIM,
                   qkv.data() + static_cast<size_t>(t) * CONV_DIM,
                   CONV_DIM * sizeof(float));
        }
    for (int c = 0; c < CONV_DIM; c++) {
        std::vector<float> win(K, 0.0f);          // zeroed conv cache, as uploaded
        for (int t = 0; t < L; t++) {
            for (int k = 0; k + 1 < K; k++) win[k] = win[k + 1];
            win[K - 1] = qkv[static_cast<size_t>(t) * CONV_DIM + c];
            float acc = 0.0f;
            for (int k = 0; k < K; k++) acc += win[k] * cw[static_cast<size_t>(k) * CONV_DIM + c];
            qkv[static_cast<size_t>(t) * CONV_DIM + c] = cpu_ref::siluf_(acc);
        }
    }
    std::vector<float> qn_row(DKH), kn_row(DKH);
    for (int t = 0; t < L; t++) {
        float* base = qkv.data() + static_cast<size_t>(t) * CONV_DIM;
        for (int h = 0; h < g.qk_heads; h++) {
            cpu_ref::l2_norm(base + h * DKH, DKH, qn_row.data(), g.qk_scale(), g.l2_eps);
            memcpy(base + h * DKH, qn_row.data(), DKH * sizeof(float));
            cpu_ref::l2_norm(base + DQ + h * DKH, DKH, kn_row.data(), 1.0f, g.l2_eps);
            memcpy(base + DQ + h * DKH, kn_row.data(), DKH * sizeof(float));
        }
    }
    // What the arena holds, and therefore what the delta cores see.
    for (float& v : qkv) v = static_cast<float>(static_cast<bf16>(v));

    // ---- alpha / beta ------------------------------------------------------
    cpu_ref::q4nx_tensor W_a = cpu_ref::get_q4nx(q4nx_file, prefix + "linear_attn.ssm_alpha_proj.weight");
    cpu_ref::q4nx_tensor W_b = cpu_ref::get_q4nx(q4nx_file, prefix + "linear_attn.ssm_beta_proj.weight");
    if (W_a.cols != D || W_b.cols != D || W_a.rows != W_b.rows || W_a.rows < NV) {
        std::cerr << "a/b weights are " << W_a.rows << " x " << W_a.cols << " and " << W_b.rows
                  << " x " << W_b.cols << ", expected two equal-height " << D
                  << "-column matrices of at least " << NV << " rows" << std::endl;
        return;
    }
    const int NH = W_a.rows;
    std::vector<float> ssm_a   = q4nx_file.f32(prefix + "linear_attn.ssm_a");
    std::vector<float> dt_bias = q4nx_file.f32(prefix + "linear_attn.ssm_dt.bias");
    if (static_cast<int>(ssm_a.size()) != NV || dt_bias.size() != ssm_a.size()) {
        std::cerr << "ssm_a has " << ssm_a.size() << " entries and ssm_dt.bias " << dt_bias.size()
                  << ", expected a matching pair of " << NV << std::endl;
        return;
    }
    std::vector<float> aa(static_cast<size_t>(L) * NH), bb(static_cast<size_t>(L) * NH);
    cpu_ref::q4nx_matmul(W_a, normed.data(), L, aa.data());
    cpu_ref::q4nx_matmul(W_b, normed.data(), L, bb.data());

    std::vector<float> alpha(static_cast<size_t>(L) * NV), beta(static_cast<size_t>(L) * NV);
    for (int t = 0; t < L; t++)
        for (int h = 0; h < NV; h++) {
            const size_t src = static_cast<size_t>(t) * NH + h;
            const size_t dst = static_cast<size_t>(t) * NV + h;
            alpha[dst] = std::exp(ssm_a[h] * cpu_ref::softplusf_(aa[src] + dt_bias[h]));
            beta[dst]  = cpu_ref::sigmoidf_(bb[src]);
        }

    // ---- z, the gated norm's silu gate -------------------------------------
    // The engine's second mvm, off the same resident x, produced long before the
    // delta rule finishes -- which is why MT[3] has to hold it. The silu is not
    // applied here: the gate tile does it, so the reference applies it below,
    // to the bf16 the tile actually receives.
    cpu_ref::q4nx_tensor W_z = cpu_ref::get_q4nx(q4nx_file, prefix + "self_attn.gate_proj.weight");
    if (W_z.cols != D || W_z.rows != DV) {
        std::cerr << "gate_proj is " << W_z.rows << " x " << W_z.cols << ", expected "
                  << DV << " x " << D << std::endl;
        return;
    }
    std::vector<float> zz(static_cast<size_t>(L) * DV);
    cpu_ref::q4nx_matmul(W_z, normed.data(), L, zz.data());
    for (float& v : zz) v = static_cast<float>(static_cast<bf16>(v));
    buffer<bf16> cpu_z_buffer = cpu_ref::to_bf16(zz);

    // ---- the delta rule ----------------------------------------------------
    // Transcribed from cpu_ref_delta_net_layer.cpp, with the state local and
    // zeroed rather than carried in the layer object, to match the zeroed S
    // region the layer's constructor uploads.
    std::vector<float> core(static_cast<size_t>(L) * DV);
    std::vector<float> state(static_cast<size_t>(NV) * DKH * DVH, 0.0f);
#pragma omp parallel for schedule(static)
    for (int h = 0; h < NV; h++) {
        const int gq = h / EXPAND;
        float* S = state.data() + static_cast<size_t>(h) * DKH * DVH;
        std::vector<float> kv(DVH), dl(DVH);
        for (int t = 0; t < L; t++) {
            const float* base = qkv.data() + static_cast<size_t>(t) * CONV_DIM;
            const float* qq_v = base + gq * DKH;
            const float* kn_v = base + DQ + gq * DKH;
            const float* vt   = base + DQ + DK + h * DVH;
            const float a = alpha[static_cast<size_t>(t) * NV + h];
            const float b = beta [static_cast<size_t>(t) * NV + h];

            for (size_t i = 0; i < static_cast<size_t>(DKH) * DVH; i++) S[i] *= a;

            for (int dv = 0; dv < DVH; dv++) kv[dv] = 0.0f;
            for (int dk = 0; dk < DKH; dk++) {
                const float kk = kn_v[dk];
                if (kk == 0.0f) continue;
                const float* Sr = S + static_cast<size_t>(dk) * DVH;
                for (int dv = 0; dv < DVH; dv++) kv[dv] += Sr[dv] * kk;
            }
            for (int dv = 0; dv < DVH; dv++) dl[dv] = (vt[dv] - kv[dv]) * b;

            float* o = core.data() + static_cast<size_t>(t) * DV + h * DVH;
            for (int dv = 0; dv < DVH; dv++) o[dv] = 0.0f;
            for (int dk = 0; dk < DKH; dk++) {
                float* Sr = S + static_cast<size_t>(dk) * DVH;
                const float kk = kn_v[dk], q = qq_v[dk];
                for (int dv = 0; dv < DVH; dv++) {
                    Sr[dv] += kk * dl[dv];
                    o[dv]  += Sr[dv] * q;
                }
            }
        }
    }
    buffer<bf16> cpu_o_buffer = cpu_ref::to_bf16(core);

    // ---- the gated RMSNorm -------------------------------------------------
    // Transcribed from cpu_ref_delta_net_layer.cpp. Fed from the bf16-rounded o
    // and z rather than from `core` and `zz` directly, because that is what
    // MT[3] hands the norm and gate tiles.
    std::vector<float> ssm_norm = q4nx_file.f32(prefix + "linear_attn.ssm_norm.weight");
    if (static_cast<int>(ssm_norm.size()) != DVH) {
        std::cerr << prefix << "linear_attn.ssm_norm.weight has " << ssm_norm.size()
                  << " elements, expected " << DVH << std::endl;
        return;
    }
    std::vector<float> gated(static_cast<size_t>(L) * DV);
    for (int t = 0; t < L; t++)
        for (int h = 0; h < NV; h++) {
            const size_t off = static_cast<size_t>(t) * DV + static_cast<size_t>(h) * DVH;
            float acc = 0.0f;
            for (int i = 0; i < DVH; i++) {
                const float o = static_cast<float>(cpu_o_buffer[off + i]);
                acc += o * o;
            }
            const float r = 1.0f / std::sqrt(acc / DVH + g.rms_eps);
            for (int i = 0; i < DVH; i++)
                gated[off + i] = static_cast<float>(cpu_o_buffer[off + i]) * r * ssm_norm[i] *
                                 cpu_ref::siluf_(static_cast<float>(cpu_z_buffer[off + i]));
        }
    buffer<bf16> cpu_gated_buffer = cpu_ref::to_bf16(gated);

    // ---- ssm_out_proj ------------------------------------------------------
    // Fed from the bf16-rounded gated rows, not from `gated`: that is what MT[2]
    // replays to the projection cores. The only projection in the layer whose K
    // is not D.
    cpu_ref::q4nx_tensor W_o = cpu_ref::get_q4nx(q4nx_file, prefix + "linear_attn.ssm_out_proj.weight");
    if (W_o.cols != DV || W_o.rows != D) {
        std::cerr << "ssm_out_proj is " << W_o.rows << " x " << W_o.cols << ", expected "
                  << D << " x " << DV << std::endl;
        return;
    }
    std::vector<float> o_proj(static_cast<size_t>(L) * D);
    {
        std::vector<float> gated_bf16 = cpu_ref::to_f32(cpu_gated_buffer);
        cpu_ref::q4nx_matmul(W_o, gated_bf16.data(), L, o_proj.data());
    }
    for (float& v : o_proj) v = static_cast<float>(static_cast<bf16>(v));
    buffer<bf16> cpu_o_proj_buffer = cpu_ref::to_bf16(o_proj);

    // ---- residual + post_attention_layernorm -------------------------------
    // The residual branch is the *unnormalised* layer input, and the sum is the
    // one thing here that is not re-rounded before use: the rms tile adds in
    // bf16 in place and norms the result in the same invocation.
    std::vector<float> post_w = q4nx_file.f32(prefix + "post_attention_layernorm.weight");
    if (static_cast<int>(post_w.size()) != D) {
        std::cerr << prefix << "post_attention_layernorm.weight has " << post_w.size()
                  << " elements, expected " << D << std::endl;
        return;
    }
    std::vector<float> post_norm(static_cast<size_t>(L) * D);
    // h1 is kept: it is the resident hidden state the rms tile carries from
    // phase 2 to phase 3, and therefore the residual branch of the MLP as well.
    // Stored as it is on the array -- bf16, the sum re-rounded, not the f32
    // accumulator -- because that is what the second residual adds down into.
    std::vector<float> h1(static_cast<size_t>(L) * D);
    for (int t = 0; t < L; t++) {
        std::vector<float> sum = cpu_ref::to_f32(inputs[t], D);
        for (int i = 0; i < D; i++)
            sum[i] = static_cast<float>(static_cast<bf16>(
                sum[i] + static_cast<float>(cpu_o_proj_buffer[static_cast<size_t>(t) * D + i])));
        memcpy(h1.data() + static_cast<size_t>(t) * D, sum.data(), D * sizeof(float));
        cpu_ref::rms_norm(sum.data(), post_w.data(), D,
                          post_norm.data() + static_cast<size_t>(t) * D,
                          env.lm_config.rms_norm_eps);
    }
    buffer<bf16> cpu_post_buffer = cpu_ref::to_bf16(post_norm);

    // ======================================================================
    // The MLP. Only cpu_post_buffer and h1 above feed it, and only its last
    // value -- cpu_layer_out_rows -- is scored.
    // ======================================================================

    // ---- the MLP's up | gate projection ------------------------------------
    // Fed from the bf16-rounded post-norm rows, not from `post_norm`: that is
    // what the rms tile emits and MT[2] replays 68 times to the 16 proj cores.
    //
    // The two matrices are *woven*, not concatenated. load_weights() interleaves
    // them 512 output rows at a time -- the granularity the GLU tile will want,
    // one row block per core per phase -- so engine column e is row (e % 512) of
    // chunk (e / 512), taken from up on even chunks and from gate on odd ones.
    // The reference is assembled in that order rather than the NPU result being
    // un-woven, so what gets scored is exactly the bytes the array wrote.
    const int I = static_cast<int>(env.lm_config.intermediate_size);
    constexpr int WEAVE = 512;
    cpu_ref::q4nx_tensor W_up   = cpu_ref::get_q4nx(q4nx_file, prefix + "mlp.up_proj.weight");
    cpu_ref::q4nx_tensor W_gate = cpu_ref::get_q4nx(q4nx_file, prefix + "mlp.gate_proj.weight");
    if (W_up.cols != D || W_up.rows != I || W_gate.cols != D || W_gate.rows != I) {
        std::cerr << "mlp.up_proj is " << W_up.rows << " x " << W_up.cols << " and mlp.gate_proj "
                  << W_gate.rows << " x " << W_gate.cols << ", expected two " << I << " x " << D
                  << " matrices" << std::endl;
        return;
    }
    if (I % WEAVE != 0) {
        std::cerr << "intermediate_size " << I << " is not a multiple of the " << WEAVE
                  << "-row weave" << std::endl;
        return;
    }
    std::vector<float> up(static_cast<size_t>(L) * I), gate(static_cast<size_t>(L) * I);
    {
        std::vector<float> post_bf16 = cpu_ref::to_f32(cpu_post_buffer);
        cpu_ref::q4nx_matmul(W_up,   post_bf16.data(), L, up.data());
        cpu_ref::q4nx_matmul(W_gate, post_bf16.data(), L, gate.data());
    }
    std::vector<float> up_gate(static_cast<size_t>(L) * 2 * I);
    for (int t = 0; t < L; t++)
        for (int c = 0; c < 2 * (I / WEAVE); c++) {
            const std::vector<float>& src = (c % 2 == 0) ? up : gate;
            memcpy(up_gate.data() + static_cast<size_t>(t) * 2 * I + static_cast<size_t>(c) * WEAVE,
                   src.data() + static_cast<size_t>(t) * I + static_cast<size_t>(c / 2) * WEAVE,
                   WEAVE * sizeof(float));
        }
    buffer<bf16> cpu_up_gate_buffer = cpu_ref::to_bf16(up_gate);

    // ---- SwiGLU ------------------------------------------------------------
    // hid = activation(gate) * up, elementwise, intermediate_size wide. Fed from
    // the *bf16-rounded* woven rows rather than from `up` and `gate` directly:
    // that is what the projection engine hands the GLU tile, and the rounding is
    // a real part of the datapath.
    //
    // Column i of token t is at woven column (i / WEAVE) * 2 * WEAVE + i % WEAVE
    // for up, and WEAVE past that for gate -- the inverse of the weave above.
    // The activation is the array's getActivationBf16, which is SiLU here
    // (A_FUNC == A_SILU in models/qwen3.8-27b.h), the same one the conv tap and
    // the z gate already score against.
    std::vector<float> hid(static_cast<size_t>(L) * I);
    for (int t = 0; t < L; t++) {
        const bf16* row = cpu_up_gate_buffer.data() + static_cast<size_t>(t) * 2 * I;
        for (int i = 0; i < I; i++) {
            const int base = (i / WEAVE) * 2 * WEAVE + (i % WEAVE);
            const float u = static_cast<float>(row[base]);
            const float g = static_cast<float>(row[base + WEAVE]);
            hid[static_cast<size_t>(t) * I + i] = cpu_ref::siluf_(g) * u;
        }
    }
    buffer<bf16> cpu_glu_buffer = cpu_ref::to_bf16(hid);

    // ---- down projection + the MLP residual --------------------------------
    // h2 = h1 + mlp.down_proj(hid), the layer's output and the next layer's
    // input. No norm after it: the model normalises at the *top* of a layer, so
    // the next input_layernorm is what h2 meets next.
    //
    // Fed from the bf16-rounded hid for the same reason as everything above --
    // that is the buffer MT[2] replays -- and added into the bf16 h1, which is
    // what the rms tile has been holding in its neighbours' memory since phase
    // 2. Neither operand is the f32 accumulator.
    cpu_ref::q4nx_tensor W_down = cpu_ref::get_q4nx(q4nx_file, prefix + "mlp.down_proj.weight");
    if (W_down.cols != I || W_down.rows != D) {
        std::cerr << "mlp.down_proj is " << W_down.rows << " x " << W_down.cols
                  << ", expected " << D << " x " << I << std::endl;
        return;
    }
    std::vector<float> down(static_cast<size_t>(L) * D);
    {
        std::vector<float> hid_bf16 = cpu_ref::to_f32(cpu_glu_buffer);
        cpu_ref::q4nx_matmul(W_down, hid_bf16.data(), L, down.data());
    }
    std::vector<float> layer_out(static_cast<size_t>(L) * D);
    for (size_t i = 0; i < layer_out.size(); i++)
        layer_out[i] = h1[i] + down[i];   // h1 is already bf16-rounded, see above
    buffer<bf16>    cpu_layer_out_buffer = cpu_ref::to_bf16(layer_out);
    tensor_2d<bf16> cpu_layer_out_rows(cpu_layer_out_buffer, D);

    // ======================================================================
    // NPU
    // ======================================================================
    linear_decoding_layer npu_layer(env.model, env.lm_config, env.layer_app_mgr);
    npu_layer.load_weights(env.q4nx);

    // FLM_PROBE_SEQ="a,b,..." dispatches a prefix of the sequence, in order,
    // from a freshly loaded xclbin -- so it controls how much state the *first*
    // pass leaves behind, not just where the second one stops. That is the
    // discriminator FLM_PROBE_STAGES cannot give: "8,9" makes pass one stop
    // before the conv cache ever reaches MT[5], so if step 9 still parks on
    // pass two the stall is not a memtile credit the cache send consumed.
    if (const char* ps = getenv("FLM_PROBE_SEQ")) {
        std::string spec(ps);
        size_t pos = 0;
        while (pos <= spec.size()) {
            size_t comma = spec.find(',', pos);
            if (comma == std::string::npos) comma = spec.size();
            int n = atoi(spec.substr(pos, comma - pos).c_str());
            if (!npu_layer.probe_prefix(input, n)) break;
            pos = comma + 1;
        }
        return;
    }

    time_utils::time_point start = time_utils::now();
    buffer<bf16> npu_layer_out_buffer = npu_layer(input);

    time_utils::time_point end = time_utils::now();
    const double pass_ms = time_utils::duration_ms(start, end).first;
    header_print("info", "Latency: " + std::to_string(pass_ms) + " ms for " +
                             std::to_string(L) + " rows");

    // ---- a second dispatch, against the same inputs ------------------------
    // Nothing the first pass ran changed an input -- S_in, the conv window in,
    // x and every weight region are all untouched by a pass -- so pass two has
    // to reproduce pass one bit for bit. It is worth its own check because the
    // failure it guards is not a numerical one: this design used to be
    // single-dispatch, the a/b tile's weight channel opening with a one-shot
    // prologue BD for the activation bias that nothing chained back to
    // (iron/AB_tile.py). Pass two's bias landed in w_buffer_0, every block
    // after it was shifted, and the host parked on a DMA wait until XRT's 60 s
    // timeout. The bias rides in as the first block of each token's weight
    // stream now, so the chain is a uniform ping/pong and every pass puts the
    // same block count on it.
    //
    // The engine dispatches this bitstream 48 times per verify pass against one
    // hardware context, so a one-shot design is not usable there at all. Time
    // it too: a park shows up as seconds, not as a wrong number.
    // FLM_PROBE_STAGES bisects a wedged array: dispatch 2 is rebuilt with only
    // the first N steps of the sequence, so the shortest N that times out names
    // the transfer the array is stuck on. Destructive -- the array is left
    // mid-pass -- so it exits rather than going on to score anything.
    if (const char* pg = getenv("FLM_PROBE_STAGES")) {
        npu_layer.probe_prefix(input, atoi(pg));
        return;
    }

    // FLM_PROBE_N raises the repeat count. The engine puts 48 of these on one
    // hardware context per verify pass, so 2 is the cheap default and 48 is the
    // number that matters; anything that only wedges after k passes shows up as
    // the k-th line here.
    const int npass = getenv("FLM_PROBE_N") ? atoi(getenv("FLM_PROBE_N")) : 2;
    for (int p = 2; p <= npass; p++)
        if (!npu_layer.probe_pass(input, "dispatch " + std::to_string(p))) {
            header_print("error", "dispatch " + std::to_string(p) + " did not reproduce the one before it");
            break;
        }

    // ---- can the state replay hide inside the dispatch? --------------------
    // The replay reads only layer i's arena and S_0, and a dispatch does not
    // touch host memory until it writes back, so in principle the whole thing
    // fits in the array's shadow. Measured here rather than asserted.
    {
        buffer<float> S_replay;
        npu_layer.get_all_S_at(L, S_replay);   // warm: fault in the output

        time_utils::time_point r0 = time_utils::now();
        for (int r = 0; r < 20; r++) npu_layer.get_all_S_at(L, S_replay);
        time_utils::time_point r1 = time_utils::now();
        const double replay_ms = time_utils::duration_ms(r0, r1).first / 20.0;

        // Attribution: the replay reads its inputs out of XRT BOs, which are
        // mapped for the device rather than the CPU. get_all_S_at() stages them
        // into heap first for exactly that reason -- reading them in place cost
        // 9.5 ms against 2.4 ms staged, while the BO sync below is ~0.05 ms, so
        // it is the reads and not the transfer.
        time_utils::time_point c0 = time_utils::now();
        for (int r = 0; r < 20; r++) npu_layer.sync_state_in();
        time_utils::time_point c1 = time_utils::now();
        const double sync_ms = time_utils::duration_ms(c0, c1).first / 20.0;

        header_print("info", "state replay, all " + std::to_string(NV) + " heads: " +
                                 std::to_string(replay_ms) + " ms (BO sync " +
                                 std::to_string(sync_ms) + " ms of it)");
        header_print("info", "  vs the " + std::to_string(pass_ms) +
                                 " ms pass above: the replay is " +
                                 std::to_string(100.0 * replay_ms / pass_ms) +
                                 "% of a dispatch, so serial costs ~" +
                                 std::to_string(pass_ms + replay_ms) +
                                 " ms and overlapped ~" + std::to_string(pass_ms) + " ms");
    }

    // ---- h2, the layer's output --------------------------------------------
    // Downstream of every stage above, so a mismatch here does not say where.
    // ARCHITECTURE.md sec 8 has what tapping an intermediate takes.
    tensor_2d<bf16> npu_layer_out_rows(npu_layer_out_buffer, D);

    report("layer output h2 = h1 + down_proj(hid): NPU vs CPU reference",
           npu_layer_out_rows, cpu_layer_out_rows, L);
    print_error_metrics(get_error_metrics(npu_layer_out_buffer, cpu_layer_out_buffer),
                        "layer output h2, all " + std::to_string(L) + " rows");

    // ---- S -----------------------------------------------------------------
    // The layer's other real output, not a diagnostic dump: it is what step t+1
    // consumes, and it is the only check that catches an off-by-one in the alpha
    // decay -- a state that drifted near-orthogonally to q would still leave the
    // output row scoring 1.0. Scored in f32 on both sides, since the state
    // never leaves f32 on the core.
    //
    // Host-replayed from the triplet arena, not read off the array. There used
    // to be a second check here -- replay-to-L against the device's own
    // write-back, which had to be near-exact because both sides consume the
    // identical bf16 k / v / alpha / beta. It passed at cos 1.0, NMSE 6e-11, and
    // that is what licensed trusting the shorter prefixes below. The write-back
    // has since been removed (it could only ever report the accept-all state,
    // which is the one case speculation cannot use), so that cross-check is no
    // longer available in-tree. What still covers the device's scan is h2 above,
    // which depends on S through o.
    const size_t head_words = static_cast<size_t>(DKH) * DVH;
    buffer<float> npu_s(static_cast<size_t>(NV) * head_words);
    buffer<float> cpu_s(static_cast<size_t>(NV) * head_words);
    npu_layer.get_all_S_at(L, npu_s);
    memcpy(cpu_s.data(), state.data(),
           static_cast<size_t>(NV) * head_words * sizeof(float));
    print_error_metrics(get_error_metrics(npu_s, cpu_s),
                        "recurrent state after " + std::to_string(L) + " steps, all " +
                            std::to_string(NV) + " heads");

    // ---- S at every accept length ------------------------------------------
    // Speculation does not know how many drafted tokens survive, so the layer
    // has to be able to produce S after any prefix. The cores never write a
    // state back -- they keep S resident for all L steps by design -- so every
    // prefix is replayed on the host from the arena, which holds the same bf16
    // k / v / alpha / beta the cores consumed.

    // Now every accept length against the CPU reference. The reference has to
    // be re-run per prefix rather than snapshotted, because the delta rule is a
    // destructive fold: S_j is not recoverable from S_L.
    buffer<float> prev_cpu;
    for (int j = 1; j <= L; j++) {
        buffer<float> npu_j(static_cast<size_t>(NV) * head_words);
        buffer<float> cpu_j(static_cast<size_t>(NV) * head_words);
#pragma omp parallel for schedule(static)
        for (int h = 0; h < NV; h++) {
            buffer<float> one = npu_layer.get_S_at(h, j);
            memcpy(npu_j.data() + static_cast<size_t>(h) * head_words, one.data(),
                   head_words * sizeof(float));

            // The same scan as the reference above, stopped at j.
            const int gq = h / EXPAND;
            float* S = cpu_j.data() + static_cast<size_t>(h) * head_words;
            memset(S, 0, head_words * sizeof(float));
            std::vector<float> kv(DVH), dl(DVH);
            for (int t = 0; t < j; t++) {
                const float* base = qkv.data() + static_cast<size_t>(t) * CONV_DIM;
                const float* kn_v = base + DQ + gq * DKH;
                const float* vt   = base + DQ + DK + h * DVH;
                const float a = alpha[static_cast<size_t>(t) * NV + h];
                const float b = beta [static_cast<size_t>(t) * NV + h];
                for (size_t i = 0; i < head_words; i++) S[i] *= a;
                for (int dv = 0; dv < DVH; dv++) kv[dv] = 0.0f;
                for (int dk = 0; dk < DKH; dk++) {
                    const float kk = kn_v[dk];
                    if (kk == 0.0f) continue;
                    const float* Sr = S + static_cast<size_t>(dk) * DVH;
                    for (int dv = 0; dv < DVH; dv++) kv[dv] += Sr[dv] * kk;
                }
                for (int dv = 0; dv < DVH; dv++) dl[dv] = (vt[dv] - kv[dv]) * b;
                for (int dk = 0; dk < DKH; dk++) {
                    float* Sr = S + static_cast<size_t>(dk) * DVH;
                    const float kk = kn_v[dk];
                    for (int dv = 0; dv < DVH; dv++) Sr[dv] += kk * dl[dv];
                }
            }
        }
        print_error_metrics(get_error_metrics(npu_j, cpu_j),
                            "recurrent state, accept " + std::to_string(j) + " of " +
                                std::to_string(L) + ", all " + std::to_string(NV) + " heads");
        // Control: this state against the *previous* prefix's reference. alpha
        // is near 1 and one step is a rank-1 update to a 128x128 state, so
        // consecutive states are inherently similar -- if they were similar
        // enough, every score above would pass no matter which prefix was
        // replayed. This is the number that says how much of the match is the
        // prefix being right rather than the states being interchangeable.
        if (j > 1) {
            print_error_metrics(get_error_metrics(npu_j, prev_cpu),
                                "  control: same state vs the accept-" + std::to_string(j - 1) +
                                    " reference (must be worse)");
        }
        prev_cpu = cpu_j;
    }

    // ---- the conv window ---------------------------------------------------
    // The pass's third real output, and the one h2 and S cannot vouch for: both
    // of those are computed *before* the writeback, so a writeback that wrote
    // nothing at all would leave them scoring exactly the same. Only reading it
    // back separates "ran" from "no-op".
    //
    // Both sides are K - 1 pre-conv rows, [tap][channel] oldest first. One per
    // accept length, since the pass now emits all L of them.
    //
    // Each is also scored against its *neighbour's* reference. The windows are
    // consecutive slices of one run, so a mis-indexed read would land on a
    // plausible-looking window and still score well against the right answer by
    // overlap alone -- K - 1 = 3 taps means two of three rows are shared. The
    // shifted control is what separates "read the right window" from "read a
    // window": it has to be far worse, or the match means nothing.
    for (int j = 1; j <= L; j++) {
        buffer<bf16> npu_conv = npu_layer.get_conv_out(j);
        buffer<bf16> cpu_conv = cpu_ref::to_bf16(cpu_conv_out[j - 1]);
        print_error_metrics(get_error_metrics(npu_conv, cpu_conv),
                            "conv window out, accept " + std::to_string(j) + " of " +
                                std::to_string(L) + ", " + std::to_string(K - 1) +
                                " taps x " + std::to_string(CONV_DIM) + " channels");
        const int ctrl = (j == L) ? j - 1 : j + 1;
        buffer<bf16> cpu_ctrl = cpu_ref::to_bf16(cpu_conv_out[ctrl - 1]);
        print_error_metrics(get_error_metrics(npu_conv, cpu_ctrl),
                            "  control: same window vs the accept-" + std::to_string(ctrl) +
                                " reference (must be worse)");
    }
}

} // namespace

// ==========================================================================
// entry point
// ==========================================================================
int main(int argc, char** argv) {
    arg_utils::po::options_description desc("Allowed options");
    arg_utils::po::variables_map vm;
    arg_utils::add_default_options(desc);
    desc.add_options()
        ("model,m", arg_utils::po::value<std::string>()->default_value("QWEN3_8_27B"),
            "Model type, e.g., QWEN3_8_27B")
        ("delta-ref", arg_utils::po::value<std::string>()
            ->default_value(""),
            "Reference dump, relative to the model path. Empty picks the model's "
            "own reference file from models.h.");

    arg_utils::parse_options(argc, argv, desc, vm);

    const std::string model_type = vm["model"].as<std::string>();
    const std::string delta_ref  = vm["delta-ref"].as<std::string>();

    header_print("info", "Testing model: " + model_type);
    model_list::model_t model = model_list::get_model_type(model_type);
    if (model == model_list::UNKNOWN_MODEL) {
        std::cerr << "Unknown model type: " << model_type << std::endl;
        return 1;
    }

    const std::string model_path = model_list::get_model_path(model);
    header_print("info", "Model path: " + model_path);

    LM_Config lm_config;
    lm_config.from_pretrained(model_path);
    lm_config.layer_xclbin_name = model_list::get_layer_xclbin_path(model);
    header_print("info", "Layer xclbin path: " + lm_config.layer_xclbin_name);
    header_print("info", "Hidden size (D): " + std::to_string(lm_config.hidden_size));

    std::unique_ptr<Q4NX> q4nx_model = std::make_unique<Q4NX>(model_path);

    // The embedding table is only readable as bf16 when the converter left it
    // unquantised. Qwen3.8-27B quantises it (I8, [7760, 20, 8704] q4nx blocks),
    // so reading it into a buffer<bf16> yields packed nibbles, not weights.
    // Leave `embed` empty in that case; check_embeddings() skips itself and the
    // test takes its inputs from the reference dump either way.
    buffer<bf16> embed;
    const std::string embed_dtype =
        q4nx_model->get_tensor_metadata("model.embed_tokens.weight").dtype;
    if (embed_dtype == "BF16" || embed_dtype == "F16") {
        q4nx_model->load_weights(embed, "model.embed_tokens.weight");
        header_print("info", "Embedding weights:");
        utils::print_matrix(embed, lm_config.hidden_size, 10);
    } else {
        header_print("info", "Embedding table is " + embed_dtype +
                                 " (quantised); skipping the embedding check.");
    }

    header_print("info", "Initializing NPU manager...");
    xrt::device npu_inst = xrt::device(0);
    npu_xclbin_manager xclbin_mgr(npu_device::device_npu2, &npu_inst, false);
    npu_app_manager* layer_app_mgr = xclbin_mgr.register_xclbin(lm_config.layer_xclbin_name);

    test_env env{
        model,
        model_path,
        lm_config,
        *q4nx_model,
        layer_app_mgr,
        embed,
        static_cast<int>(lm_config.hidden_size),
    };

    const std::string ref_path = delta_ref.empty() ? model_list::get_reference_path(model)
                                                   : model_path + "/" + delta_ref;

    run_layer_test(env, ref_path);

    return 0;
}
