// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#include "typedef.hpp"
#include "buffer.hpp"
#include "decoding_layer.hpp"
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

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

/// decoding_layer::load_weights() pins itself to layer 3, so the CPU reference
/// has to load the same one. It is also the first layer for which
/// cpu_ref::q38::is_full_attention() holds.
constexpr int kAttentionLayer = 3;

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
/// than showing up as a mystery layer error. The layer is fed layer 2's output,
/// not the embeddings, so this is a check and not a data path.
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

/// Round a whole fp32 vector through bf16, in place. Marks a point where the
/// array narrows; see the stage comments below.
void round_bf16(std::vector<float>& v) {
    for (float& x : v) x = static_cast<float>(static_cast<bf16>(x));
}

// ==========================================================================
// the attention half of the layer, scored at post_attention_layernorm
// ==========================================================================
void run_post_norm_test(const test_env& env, const std::string& ref_path) {
    using namespace cpu_ref::q38;
    const int D = env.D;
    const int L = L_TOKENS;

    // The CPU golden runs LC positions; the NPU is handed only the last
    // L_TOKENS of them, with the first CTX rows of k and v pre-seeded into its
    // cache. That is the decode case -- a pass over a few new tokens against a
    // context that is already there -- and it is the only way to exercise the
    // replay's chunk walk, the per-token key counts and the append offset. At
    // CTX = 0 the context rounds up to a single chunk and all three collapse to
    // their trivial form.
    //
    // CTX is deliberately not a multiple of L_CHUNK: the append writes at the
    // exact offset while the replay starts at row 0 and rounds its length up, so
    // a ragged context is the normal case, not a special one.
    // LC is overridable so one binary can cover the interesting shapes: 8 (no
    // context), 14 (ragged, one chunk), 30 and up (several chunks, which is
    // what walks the replay's chunk axis). All of 8/14/30/40/56 score ~0.9997.
    // ATTN_POISON below is the matching negative control.
    const int LC  = std::getenv("ATTN_LC") ? std::atoi(std::getenv("ATTN_LC")) : 30;
    const int CTX = LC - L;

    header_print("info", "=== post_attention_layernorm, layer " +
                             std::to_string(kAttentionLayer) + ", " + std::to_string(L) +
                             " rows at context offset " + std::to_string(CTX) + " of " +
                             std::to_string(LC) + ", inputs from " + ref_path + " ===");

    if (D != MODEL_DIM) {
        std::cerr << "lm_config hidden_size is " << D << ", but the CPU reference is compiled "
                  << "for " << MODEL_DIM << std::endl;
        return;
    }
    if (!is_full_attention(kAttentionLayer)) {
        std::cerr << "layer " << kAttentionLayer << " is delta net, not full attention; "
                  << "that one belongs to ../../gate_delta_net_verify" << std::endl;
        return;
    }

    // The embedding cross-check still reads through the XRT-side SafeTensors,
    // because check_embeddings() works in buffer<bf16>. Everything the golden
    // consumes goes through cpu_ref::safetensors instead, which converts on
    // read and so does not care whether the dump stored a tensor as bf16 or f32.
    std::unique_ptr<SafeTensors> references = std::make_unique<SafeTensors>(ref_path);
    buffer<bf16>    ref_embed;
    buffer<int32_t> ids;
    references->load_weights(ref_embed, "verify.inputs_embeds");
    references->load_weights(ids, "verify.input_ids");
    check_embeddings(env, ids, ref_embed);

    cpu_ref::safetensors ref_file(ref_path);
    cpu_ref::safetensors q4nx_file(cpu_ref::q4nx_path_of(env.model_path));
    const std::string prefix = "model.layers." + std::to_string(kAttentionLayer) + ".";

    char key[64];
    std::snprintf(key, sizeof key, "prefill.layer_%02d_out", kAttentionLayer - 1);
    std::vector<float> x_in;
    try {
        x_in = ref_file.f32(key);
    } catch (const std::exception& e) {
        std::cerr << "reference dump " << ref_path << " is missing a tensor: " << e.what()
                  << std::endl;
        return;
    }
    if (x_in.size() < static_cast<size_t>(LC) * D) {
        std::cerr << "dump's " << key << " has only " << x_in.size() / D << " rows, need " << LC
                  << std::endl;
        return;
    }
    x_in.resize(static_cast<size_t>(LC) * D);

    const double kRopeTheta = 10000000.0;
    std::vector<float> cos_in(static_cast<size_t>(LC) * ROPE_DIM);
    std::vector<float> sin_in(static_cast<size_t>(LC) * ROPE_DIM);
    for (int t = 0; t < LC; t++) {
        for (int j = 0; j < ROPE_DIM / 2; j++) {
            const double inv = 1.0 / std::pow(kRopeTheta, (2.0 * j) / ROPE_DIM);
            const double a   = static_cast<double>(t) * inv;
            const size_t r   = static_cast<size_t>(t) * ROPE_DIM;
            cos_in[r + j] = cos_in[r + j + ROPE_DIM / 2] = static_cast<float>(std::cos(a));
            sin_in[r + j] = sin_in[r + j + ROPE_DIM / 2] = static_cast<float>(std::sin(a));
        }
    }

    // The whole LC-row input, rounded to what the array would see. The NPU is
    // handed the last L rows of it.
    buffer<bf16>    full_input = cpu_ref::to_bf16(x_in);
    for (size_t i = 0; i < x_in.size(); i++) x_in[i] = static_cast<float>(full_input[i]);

    buffer<bf16> input(static_cast<size_t>(L) * D);
    memcpy(input.data(), full_input.data() + static_cast<size_t>(CTX) * D,
           static_cast<size_t>(L) * D * sizeof(bf16));
    tensor_2d<bf16> inputs(input, D);

    // ======================================================================
    // CPU golden. Every stage below feeds the next; only the last one is
    // scored. The bf16 rounds are not cosmetic -- they mark the points where
    // the array actually narrows, and dropping one would charge this test for
    // precision the hardware never had.
    // ======================================================================

    // ---- input_layernorm ---------------------------------------------------
    std::vector<float> ln_w = q4nx_file.f32(prefix + "input_layernorm.weight");
    if (static_cast<int>(ln_w.size()) != D) {
        std::cerr << prefix << "input_layernorm.weight has " << ln_w.size()
                  << " elements, expected " << D << std::endl;
        return;
    }
    std::vector<float> normed(static_cast<size_t>(LC) * D);
    for (int t = 0; t < LC; t++)
        cpu_ref::rms_norm(x_in.data() + static_cast<size_t>(t) * D, ln_w.data(), D,
                          normed.data() + static_cast<size_t>(t) * D,
                          env.lm_config.rms_norm_eps);
    round_bf16(normed);

    // ---- q | z, k, v -- the stage under test -------------------------------
    // self_attn.q_proj.weight is 2 * DQ rows: q in [0, DQ), the output gate z in
    // [DQ, 2 * DQ). HF interleaves the two per head; the q4nx converter
    // de-interleaves them, which is what lets the array split the matrix between
    // two mvm() calls -- rows [0, DQ) ride along with k and v on pkt_id_to_rope,
    // rows [DQ, 2 * DQ) go out on pkt_id_to_aux and park in MT[2] for the whole
    // attention latency. One matmul here covers both.
    //
    // Both halves are on the array now, and every stage below them is too; the
    // golden runs end to end because each stage feeds the next.
    cpu_ref::q4nx_tensor W_q = cpu_ref::get_q4nx(q4nx_file, prefix + "self_attn.q_proj.weight");
    cpu_ref::q4nx_tensor W_k = cpu_ref::get_q4nx(q4nx_file, prefix + "self_attn.k_proj.weight");
    cpu_ref::q4nx_tensor W_v = cpu_ref::get_q4nx(q4nx_file, prefix + "self_attn.v_proj.weight");
    if (W_q.cols != D || W_q.rows != 2 * DQ || W_k.cols != D || W_k.rows != DKV ||
        W_v.cols != D || W_v.rows != DKV) {
        std::cerr << "projection geometry is q " << W_q.rows << " x " << W_q.cols << ", k "
                  << W_k.rows << " x " << W_k.cols << ", v " << W_v.rows << " x " << W_v.cols
                  << ", expected " << 2 * DQ << " / " << DKV << " / " << DKV << " rows of " << D
                  << std::endl;
        return;
    }
    std::vector<float> qz(static_cast<size_t>(LC) * 2 * DQ);
    std::vector<float> kk(static_cast<size_t>(LC) * DKV), vv(static_cast<size_t>(LC) * DKV);
    cpu_ref::q4nx_matmul(W_q, normed.data(), LC, qz.data());
    cpu_ref::q4nx_matmul(W_k, normed.data(), LC, kk.data());
    cpu_ref::q4nx_matmul(W_v, normed.data(), LC, vv.data());
    round_bf16(qz);
    round_bf16(kk);
    round_bf16(vv);

    // What the projection engine emits, kept before the RoPE tile touches it:
    // the NPU's qkv rows are exactly this, and k is normed in place just below.
    // (q is read straight out of qz, whose first DQ columns per row stay put.)
    const std::vector<float> kk_raw = kk;
    const std::vector<float> vv_raw = vv;

    // ---- per-head q/k RMSNorm, then partial RoPE ---------------------------
    // Both happen on the RoPE tile, in that order: Qwen3.8 norms before rotating
    // (the reverse of HunYuan). q is pulled out of the woven q | z rows here, so
    // everything downstream indexes it DQ-wide.
    std::vector<float> q_norm_w = q4nx_file.f32(prefix + "self_attn.q_norm.weight");
    std::vector<float> k_norm_w = q4nx_file.f32(prefix + "self_attn.k_norm.weight");
    if (static_cast<int>(q_norm_w.size()) != HEAD_DIM ||
        static_cast<int>(k_norm_w.size()) != HEAD_DIM) {
        std::cerr << "q_norm / k_norm are " << q_norm_w.size() << " and " << k_norm_w.size()
                  << " elements, expected " << HEAD_DIM << " each" << std::endl;
        return;
    }
    std::vector<float> qq(static_cast<size_t>(LC) * DQ);
    for (int t = 0; t < LC; t++) {
        for (int h = 0; h < NUM_ATTN_HEADS; h++)
            cpu_ref::rms_norm(qz.data() + static_cast<size_t>(t) * 2 * DQ + h * HEAD_DIM,
                              q_norm_w.data(), HEAD_DIM,
                              qq.data() + static_cast<size_t>(t) * DQ + h * HEAD_DIM,
                              env.lm_config.rms_norm_eps);
        for (int h = 0; h < NUM_KV_HEADS; h++) {
            float  tmp[HEAD_DIM];
            float* dst = kk.data() + static_cast<size_t>(t) * DKV + h * HEAD_DIM;
            cpu_ref::rms_norm(dst, k_norm_w.data(), HEAD_DIM, tmp, env.lm_config.rms_norm_eps);
            memcpy(dst, tmp, sizeof tmp);
        }
    }

    // Partial rotary over dims [0, 64) only, rotate_half with halves 0..31 /
    // 32..63. cos/sin come from the dump, row t for position t, and the NPU is
    // given rows [CTX, LC) of the same table so the two agree position by
    // position. Only relative position matters to q . k, but the seeded context
    // and the new rows have to be rotated on the same origin for that to hold.
    auto rope = [&](float* v, int t) {
        const float* c = cos_in.data() + static_cast<size_t>(t) * ROPE_DIM;
        const float* s = sin_in.data() + static_cast<size_t>(t) * ROPE_DIM;
        float r[ROPE_DIM];
        const int H = ROPE_DIM / 2;
        for (int j = 0; j < H; j++)        r[j] = v[j] * c[j] - v[j + H] * s[j];
        for (int j = H; j < ROPE_DIM; j++) r[j] = v[j] * c[j] + v[j - H] * s[j];
        memcpy(v, r, sizeof r);
    };
    for (int t = 0; t < LC; t++) {
        for (int h = 0; h < NUM_ATTN_HEADS; h++)
            rope(qq.data() + static_cast<size_t>(t) * DQ + h * HEAD_DIM, t);
        for (int h = 0; h < NUM_KV_HEADS; h++)
            rope(kk.data() + static_cast<size_t>(t) * DKV + h * HEAD_DIM, t);
    }
    // What leaves the RoPE tile: q to the qk cores, k | v to DDR and back into
    // the bf16 KV cache. The cache is bf16, so these rounds are the cache.
    round_bf16(qq);
    round_bf16(kk);
    round_bf16(vv);

    // ---- causal GQA softmax attention --------------------------------------
    // Token t sees keys [0, t]: rows [0, CTX) are the seeded context and rows
    // [CTX, LC) the ones the NPU computes, so the scored rows attend across the
    // seam. 6 q heads per kv head; the scale is 1/sqrt(256).
    std::vector<float> attn(static_cast<size_t>(LC) * DQ);
#pragma omp parallel for collapse(2) schedule(static)
    for (int t = 0; t < LC; t++) {
        for (int h = 0; h < NUM_ATTN_HEADS; h++) {
            const int    kvh  = h / GQA_GROUP;
            const int    nkey = t + 1;
            const float* qv   = qq.data() + static_cast<size_t>(t) * DQ + h * HEAD_DIM;
            std::vector<float> s(nkey);
            float smax = -INFINITY;
            for (int j = 0; j < nkey; j++) {
                const float* kv = kk.data() + static_cast<size_t>(j) * DKV + kvh * HEAD_DIM;
                float acc = 0.0f;
                for (int d = 0; d < HEAD_DIM; d++) acc += qv[d] * kv[d];
                s[j] = acc * ATTN_SCALE;
                smax = std::max(smax, s[j]);
            }
            float sum = 0.0f;
            for (int j = 0; j < nkey; j++) { s[j] = std::exp(s[j] - smax); sum += s[j]; }
            const float inv = 1.0f / sum;
            float* out = attn.data() + static_cast<size_t>(t) * DQ + h * HEAD_DIM;
            for (int d = 0; d < HEAD_DIM; d++) out[d] = 0.0f;
            for (int j = 0; j < nkey; j++) {
                const float p = s[j] * inv;
                if (p == 0.0f) continue;
                const float* vj = vv.data() + static_cast<size_t>(j) * DKV + kvh * HEAD_DIM;
                for (int d = 0; d < HEAD_DIM; d++) out[d] += p * vj[d];
            }
        }
    }
    round_bf16(attn);

    // Everything past this point concerns only the rows the NPU computes, so
    // the golden narrows to the last L of the LC positions here. The seeded
    // context has served its purpose -- it is in the keys and values the
    // softmax above just read across.
    const size_t ctx_q  = static_cast<size_t>(CTX) * 2 * DQ;
    const size_t ctx_dq = static_cast<size_t>(CTX) * DQ;

    // ---- the output gate ---------------------------------------------------
    // The gate tile on CT[2][2]: sigmoid the parked z, multiply it into the
    // attention output. Both operands are the bf16 the tile actually receives --
    // z came off the proj cores rounded and has been sitting in MT[2] since.
    std::vector<float> gated(static_cast<size_t>(L) * DQ);
    for (int t = 0; t < L; t++) {
        const float* z = qz.data() + ctx_q + static_cast<size_t>(t) * 2 * DQ + DQ;
        const float* a = attn.data() + ctx_dq + static_cast<size_t>(t) * DQ;
        float*       g = gated.data() + static_cast<size_t>(t) * DQ;
        for (int i = 0; i < DQ; i++) g[i] = a[i] * cpu_ref::sigmoidf_(z[i]);
    }
    round_bf16(gated);

    // ---- o_proj, the stage under test --------------------------------------
    // Fed from the bf16-rounded gated rows, which is what MT[2] replays to the
    // projection cores. The only projection in the layer whose K is not D.
    cpu_ref::q4nx_tensor W_o = cpu_ref::get_q4nx(q4nx_file, prefix + "self_attn.o_proj.weight");
    if (W_o.cols != DQ || W_o.rows != D) {
        std::cerr << "o_proj is " << W_o.rows << " x " << W_o.cols << ", expected " << D << " x "
                  << DQ << std::endl;
        return;
    }
    std::vector<float> o_proj(static_cast<size_t>(L) * D);
    cpu_ref::q4nx_matmul(W_o, gated.data(), L, o_proj.data());
    round_bf16(o_proj);

    // ---- residual + post_attention_layernorm, the stage under test ---------
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
    for (int t = 0; t < L; t++) {
        std::vector<float> sum(D);
        for (int i = 0; i < D; i++) {
            const size_t j = static_cast<size_t>(t) * D + i;
            sum[i] = static_cast<float>(
                static_cast<bf16>(x_in[static_cast<size_t>(CTX) * D + j] + o_proj[j]));
        }
        cpu_ref::rms_norm(sum.data(), post_w.data(), D,
                          post_norm.data() + static_cast<size_t>(t) * D,
                          env.lm_config.rms_norm_eps);
    }
    buffer<bf16>    cpu_post_buffer = cpu_ref::to_bf16(post_norm);
    tensor_2d<bf16> cpu_post_rows(cpu_post_buffer, D);

    // h1, the attention residual, kept for the MLP residual below. This is the
    // bf16 row the rms tile holds in its neighbours' memory after phase 2 --
    // not the f32 accumulator -- so phase 3 adds into exactly this.
    std::vector<float> h1(static_cast<size_t>(L) * D);
    for (int t = 0; t < L; t++)
        for (int i = 0; i < D; i++) {
            const size_t j = static_cast<size_t>(t) * D + i;
            h1[j] = static_cast<float>(static_cast<bf16>(
                x_in[static_cast<size_t>(CTX) * D + j] + o_proj[j]));
        }

    // ---- the MLP: up | gate, woven ------------------------------------------
    // The two matrices are interleaved WEAVE output rows at a time rather than
    // concatenated, because the GLU tile takes one engine round at a time and a
    // round has to be all up or all gate. WEAVE = M_PER_ROUND = 16 cores x 32
    // rows, which is what load_weights() lays down on the host side.
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
    // what the rms tile has been holding since phase 2.
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
    round_bf16(down);
    std::vector<float> layer_out(static_cast<size_t>(L) * D);
    for (size_t i = 0; i < layer_out.size(); i++)
        layer_out[i] = h1[i] + down[i];   // h1 is already bf16-rounded, see above
    buffer<bf16>    cpu_down_buffer = cpu_ref::to_bf16(down);
    buffer<bf16>    cpu_h2_buffer = cpu_ref::to_bf16(layer_out);
    tensor_2d<bf16> cpu_down_rows(cpu_down_buffer, D);
    tensor_2d<bf16> cpu_h2_rows(cpu_h2_buffer, D);

    // ======================================================================
    // NPU -- the whole attention half of the layer, from input_layernorm through
    // to post_attention_layernorm.
    //
    // The scored tensor is the post-norm row, L x D, against `post_norm` above.
    // It is the end of everything the array computes, so a disagreement is an
    // NPU-side bug in any stage from the first norm down -- which is why k, v,
    // the ungated attention, the gated rows and o are all pulled back and scored
    // first. Each is the direct control for the stage after it: if `o` still
    // scores 0.9998 and the post-norm row does not, the fault is in the residual
    // or the second norm, and not upstream of o_proj.
    //
    // q is not read back. It goes straight from the RoPE tile to the four
    // attention CUs as four packets, and its correctness is implied by the
    // output. An earlier build scored it at 0.999966 with the GQA padding
    // verified zero, which is the layout the CUs index against.
    //
    // Nothing in the attention half is golden-only any more; the MLP is absent
    // from both sides.
    // ======================================================================
    (void)kk_raw;

    decoding_layer npu_layer(env.model, env.lm_config, env.layer_app_mgr);
    npu_layer.load_weights(env.q4nx);

    // The RoPE table the tile walks is the L rows for *these* positions, so the
    // slice starts at CTX. Feeding it from row 0 would rotate the new tokens as
    // though they were the first in the sequence and put every score against the
    // seeded context at the wrong relative position.
    {
        std::vector<float> cos_slice(cos_in.begin() + static_cast<size_t>(CTX) * ROPE_DIM,
                                     cos_in.begin() + static_cast<size_t>(CTX + L) * ROPE_DIM);
        std::vector<float> sin_slice(sin_in.begin() + static_cast<size_t>(CTX) * ROPE_DIM,
                                     sin_in.begin() + static_cast<size_t>(CTX + L) * ROPE_DIM);
        npu_layer.set_rope_table(cos_slice, sin_slice);
    }

    // The context the NPU is told it already has: the golden's first CTX rows of
    // k and v, written straight into the cache. These are post-norm, post-RoPE,
    // which is what the array would have left there on an earlier pass.
    {
        // ATTN_POISON=<row>: corrupt one seeded row on the way in. A negative
        // control for the replay -- if poisoning a row past the first chunk
        // does not move the score, that chunk is not being read and a passing
        // result means nothing.
        std::vector<float> k_seed = kk, v_seed = vv;
        if (const char* p = std::getenv("ATTN_POISON")) {
            const int row = std::atoi(p);
            if (row >= 0 && row < CTX) {
                for (int i = 0; i < DKV; i++) {
                    k_seed[static_cast<size_t>(row) * DKV + i] = 3.0f;
                    v_seed[static_cast<size_t>(row) * DKV + i] = 3.0f;
                }
                header_print("info", "poisoning seeded row " << row << " (chunk "
                             << row / 16 << ")");
            }
        }
        int seed_rows = CTX;

        // ATTN_TAIL_POISON=<value>: fill the cache rows from ATTN_TAIL_FROM
        // (default CTX) to the chunk round-up with <value>. The opposite
        // control to ATTN_POISON -- these are rows the replay *does* send, since
        // the last chunk is only partly real, and every token has to mask them
        // off. Rows [CTX, CTX + L) are covered too, and the pass's own append
        // has to overwrite them.
        //
        // **A masked row still has to be of a sane magnitude.** The mask is
        // exact -- a masked lane contributes an exact zero to the score and to
        // the softmax denominator -- but k is the B operand of an 8x8x8 bf16
        // mmul, and that truncates to bfp16: eight rows share one exponent. A
        // masked row with an absurd magnitude therefore costs the *real* rows
        // of its own 8-row block their mantissa. So poisoning [49, 64) at 1000
        // at CTX = 41 moves token 7 alone, which is the token whose last real
        // key (row 48) shares block [48, 56) with the poison; poisoning the
        // block-aligned [56, 64) at the same value is inert to the last bit.
        // That is an operand-quantisation artifact of a deliberately absurd
        // input, not a mask leak.
        if (const char* p = std::getenv("ATTN_TAIL_POISON")) {
            const float val = static_cast<float>(std::atof(p));
            const int rows_up = (CTX + L + 15) / 16 * 16;
            const int from = std::getenv("ATTN_TAIL_FROM")
                                 ? std::atoi(std::getenv("ATTN_TAIL_FROM")) : CTX;
            // Below `from`: the golden where it exists (kk/vv only run to LC),
            // and zero in the gap -- the value an untouched cache row holds, so
            // the gap is not itself a poison.
            k_seed.assign(static_cast<size_t>(rows_up) * DKV, 0.0f);
            v_seed.assign(static_cast<size_t>(rows_up) * DKV, 0.0f);
            const int keep = std::min(from, CTX + L);
            std::copy(kk.begin(), kk.begin() + static_cast<size_t>(keep) * DKV, k_seed.begin());
            std::copy(vv.begin(), vv.begin() + static_cast<size_t>(keep) * DKV, v_seed.begin());
            std::fill(k_seed.begin() + static_cast<size_t>(from) * DKV, k_seed.end(), val);
            std::fill(v_seed.begin() + static_cast<size_t>(from) * DKV, v_seed.end(), val);
            seed_rows = rows_up;
            header_print("info", "tail-poisoning cache rows [" << from << ", " << rows_up
                         << ") with " << val << " -- " << std::max(0, rows_up - std::max(from, CTX + L))
                         << " of them past every token's keys");
        }
        npu_layer.seed_kv_cache(k_seed, v_seed, seed_rows);
    }
    npu_layer.set_context_offset(CTX);

    time_utils::time_point start = time_utils::now();
    buffer<bf16> npu_h2_buffer = npu_layer(input);
    time_utils::time_point end = time_utils::now();
    header_print("info", "Latency: " +
                             std::to_string(time_utils::duration_ms(start, end).first) + " ms for " +
                             std::to_string(L) + " rows");

    // TEMPORARY probe: does one app survive more than one dispatch? The scored
    // pass above is the only invocation the test itself needs, so without this
    // the bench cannot see dispatch state at all -- a design that hangs or
    // corrupts on pass 2 looks perfect here. ATTN_REPEAT=n dispatches n extra
    // times and leaves the scoring below untouched.
    //
    // Repeat 1 reuses the same cursor and the same input. The append then
    // rewrites the same cache rows with the same values, so the pass is
    // idempotent and the output has to come back **bit-identical**; that, not
    // merely returning, is the check. Later repeats advance the cursor by L,
    // which rebuilds the sequence (set_context_offset does) and grows the
    // replay's chunk walk -- those are scored only for completion, since the
    // RoPE table still holds the slice for the original positions.
    if (const char* rep = std::getenv("ATTN_REPEAT")) {
        for (int i = 0; i < std::atoi(rep); i++) {
            const int cur = CTX + i * L;
            if (cur + L > npu_layer.max_context()) break;
            npu_layer.set_context_offset(cur);
            buffer<bf16> again = npu_layer(input);
            size_t diff = 0;
            for (size_t j = 0; j < again.size(); j++)
                if ((float)again[j] != (float)npu_h2_buffer[j]) diff++;
            header_print("info", "repeat dispatch " << (i + 1) << " at cursor " << cur
                         << ": ok, h2[0] = " << (float)again[0] << ", "
                         << diff << " of " << again.size() << " words differ from pass 1"
                         << (i == 0 ? " (must be 0 -- same cursor, same input)" : ""));
        }
        npu_layer.set_context_offset(CTX);
    }

    buffer<bf16> npu_k_buffer = npu_layer.get_k();
    buffer<bf16> npu_v_buffer = npu_layer.get_v();

    // The golden holds LC rows; the NPU computed the last L. Score like against
    // like.
    auto tail = [](const std::vector<float>& v, int rows, int width) {
        return std::vector<float>(v.end() - static_cast<size_t>(rows) * width, v.end());
    };
    buffer<bf16> cpu_k_buffer = cpu_ref::to_bf16(tail(kk, L, DKV));
    buffer<bf16> cpu_v_buffer = cpu_ref::to_bf16(tail(vv, L, DKV));

    tensor_2d<bf16> npu_k_rows(npu_k_buffer, DKV), cpu_k_rows(cpu_k_buffer, DKV);
    tensor_2d<bf16> npu_v_rows(npu_v_buffer, DKV), cpu_v_rows(cpu_v_buffer, DKV);

    // ---- the inputs to the attention, as diagnostics ----------------------
    // These went through the cache the CUs read from, so if they are wrong the
    // output cannot be right and the fault is upstream of the softmax.
    print_error_metrics(get_error_metrics(npu_k_buffer, cpu_k_buffer),
                        "k cache");
    print_error_metrics(get_error_metrics(npu_v_buffer, cpu_v_buffer),
                        "v cache");

    // // ---- o_proj, the new stage under test ---------------------------------
    // // D wide, not DQ: this is the projection back to the model dimension, and
    // // the first mvm in the layer whose K is not D either -- it reduces over the
    // // gated rows above. A failure here with `gated` still at 0.999 is in the
    // // o_proj weight round, its replay phase or the de-permute on the way back,
    // // not in anything upstream.
    // buffer<bf16>    npu_o_buffer = npu_layer.get_o();
    // buffer<bf16>    cpu_o_buffer = cpu_ref::to_bf16(o_proj);
    // tensor_2d<bf16> npu_o_rows(npu_o_buffer, D), cpu_o_rows(cpu_o_buffer, D);

    // header_print("info", "o_proj NPU row 0, first 32:");
    // utils::print_matrix(npu_o_rows[0], 32, 1);
    // header_print("info", "o_proj CPU row 0, first 32:");
    // utils::print_matrix(cpu_o_rows[0], 32, 1);

    // report("o_proj: NPU vs CPU reference", npu_o_rows, cpu_o_rows, L);
    // print_error_metrics(get_error_metrics(npu_o_buffer, cpu_o_buffer),
    //                     "o_proj, all " + std::to_string(L) + " rows");

    // // ---- down_proj, the MLP's output ---------------------------------------
    // // **post_attention_layernorm is no longer separately observable.** It is
    // // up|gate's x now and goes west to the projection engine; rms_core.Y_out
    // // reaches at most two destination sets and both norms share the westward
    // // one, so its DDR tap had to go when the MLP landed. What replaces it as
    // // the check on that norm is `down` -- which reduces over it through two
    // // more projections, so a bad norm cannot leave this clean.
    // buffer<bf16>    npu_down_buffer = npu_layer.get_down();
    // tensor_2d<bf16> npu_down_rows(npu_down_buffer, D);

    // header_print("info", "down_proj NPU row 0, first 32:");
    // utils::print_matrix(npu_down_rows[0], 32, 1);
    // header_print("info", "down_proj CPU row 0, first 32:");
    // utils::print_matrix(cpu_down_rows[0], 32, 1);

    // report("down_proj (after up|gate and the SwiGLU): NPU vs CPU reference",
    //        npu_down_rows, cpu_down_rows, L);
    // print_error_metrics(get_error_metrics(npu_down_buffer, cpu_down_buffer),
    //                     "down_proj, all " + std::to_string(L) + " rows");

    // ---- h2, the layer's output and the scored stage -----------------------
    // h1 + down_proj, where h1 is the attention residual the rms tile has been
    // holding in its neighbours' memory since phase 2. The same 80 KiB has been
    // h, then h1, then h2 -- one allocation for the layer's whole residual
    // chain, never touching DDR.
    tensor_2d<bf16> npu_h2_rows(npu_h2_buffer, D);

    header_print("info", "h2 NPU row 0, first 32:");
    utils::print_matrix(npu_h2_rows[0], 32, 1);
    header_print("info", "h2 CPU row 0, first 32:");
    utils::print_matrix(cpu_h2_rows[0], 32, 1);

    report("layer output h2 = h1 + down_proj(SwiGLU): NPU vs CPU reference",
           npu_h2_rows, cpu_h2_rows, L);
    print_error_metrics(get_error_metrics(npu_h2_buffer, cpu_h2_buffer),
                        "layer output h2, all " + std::to_string(L) + " rows");

    // // **h2 against the NPU's own down**, the direct control for the MLP
    // // residual, in the same spirit as the two above: add the array's own down
    // // rows into the golden's h1 rather than the golden's down. A split between
    // // this and the end-to-end score puts the fault in phase 3; agreement puts
    // // it upstream.
    // {
    //     std::vector<float> h2_of_npu_down(static_cast<size_t>(L) * D);
    //     for (size_t i = 0; i < h2_of_npu_down.size(); i++)
    //         h2_of_npu_down[i] = static_cast<float>(static_cast<bf16>(
    //             h1[i] + static_cast<float>(npu_down_buffer.data()[i])));
    //     buffer<bf16> ref_buffer = cpu_ref::to_bf16(h2_of_npu_down);
    //     print_error_metrics(get_error_metrics(npu_h2_buffer, ref_buffer),
    //                         "h2 vs h1 + the NPU's own down");
    // }

    // // **Is the MLP residual actually being added?** Same trap as the attention
    // // residual: score the same NPU rows against a golden that omits h1. Unlike
    // // phase 2 there is no norm here to flatter a no-op, but the control is
    // // cheap and it is the only thing that sees `h` on this path -- if the
    // // resident row were stale or misaddressed, every stage above would still
    // // score clean and only this would move.
    // {
    //     buffer<bf16> ref_buffer = cpu_ref::to_bf16(down);
    //     print_error_metrics(get_error_metrics(npu_h2_buffer, ref_buffer),
    //                         "NEGATIVE CONTROL: h2 vs down alone (no residual) -- "
    //                         "must be far below the score above");
    // }
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
        ("attn-ref", arg_utils::po::value<std::string>()
            ->default_value(""),
            "Reference dump, relative to the model path. Empty picks the model's "
            "own reference file from models.h.");

    arg_utils::parse_options(argc, argv, desc, vm);

    const std::string model_type = vm["model"].as<std::string>();
    const std::string attn_ref   = vm["attn-ref"].as<std::string>();

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
    // test takes its input from the reference dump either way.
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

    const std::string ref_path = attn_ref.empty() ? model_list::get_reference_path(model)
                                                  : model_path + "/" + attn_ref;

    run_post_norm_test(env, ref_path);

    return 0;
}
