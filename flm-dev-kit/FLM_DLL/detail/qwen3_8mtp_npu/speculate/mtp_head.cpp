// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file speculate/mtp_head.cpp
/// \brief mtp_head implementation, carried across from
///        MTP_decoding/host/cpu_ref_mtp_head.cpp.
#include "speculate/mtp_head.hpp"
#include "common/host_par.hpp"   // q38npu::host_par_min, for the per-row stages

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace cpu_ref {

using namespace q38;

// ==========================================================================
// weights
// ==========================================================================
void mtp_weights::load(const safetensors& f) {
    const std::string p = "model.mtp.";
    fc       = get_q4nx(f, p + "fc.weight");
    q        = get_q4nx(f, p + "layers.0.self_attn.q_proj.weight");
    k        = get_q4nx(f, p + "layers.0.self_attn.k_proj.weight");
    v        = get_q4nx(f, p + "layers.0.self_attn.v_proj.weight");
    o        = get_q4nx(f, p + "layers.0.self_attn.o_proj.weight");
    mlp_gate = get_q4nx(f, p + "layers.0.mlp.gate_proj.weight");
    mlp_up   = get_q4nx(f, p + "layers.0.mlp.up_proj.weight");
    mlp_down = get_q4nx(f, p + "layers.0.mlp.down_proj.weight");

    pre_fc_norm_embedding = f.f32(p + "pre_fc_norm_embedding.weight");
    pre_fc_norm_hidden    = f.f32(p + "pre_fc_norm_hidden.weight");
    in_ln    = f.f32(p + "layers.0.input_layernorm.weight");
    post_ln  = f.f32(p + "layers.0.post_attention_layernorm.weight");
    out_norm = f.f32(p + "norm.weight");
    q_norm   = f.f32(p + "layers.0.self_attn.q_norm.weight");
    k_norm   = f.f32(p + "layers.0.self_attn.k_norm.weight");

    // q_proj is 2*DQ (attn_output_gate emits a query and a gate per head),
    // o_proj takes DQ (the gate is consumed before it), fc takes 2*D (its
    // input is the concat of two normalised 5120-vectors).
    if (q.rows != 2 * DQ || k.rows != DKV || v.rows != DKV)
        throw std::runtime_error("mtp: qkv projection geometry mismatch");
    if (o.cols != DQ || o.rows != MODEL_DIM)
        throw std::runtime_error("mtp: o_proj geometry mismatch");
    if (fc.rows != MODEL_DIM || fc.cols != mtp::FC_IN)
        throw std::runtime_error("mtp: fc geometry mismatch");
}

// ==========================================================================
// the head
// ==========================================================================
mtp_head::mtp_head(const safetensors& q4nx, const embedding_table* embed, int max_len)
    : max_len_(max_len), embed_(embed) {
    if (!embed_)
        throw std::runtime_error("mtp_head: needs the base model's embedding table "
                                 "(mtp_use_dedicated_embeddings is false)");
    w_.load(q4nx);
    cache_k_.assign((size_t)max_len_ * DKV, 0.0f);
    cache_v_.assign((size_t)max_len_ * DKV, 0.0f);
}

void mtp_head::step(const int32_t* ids, const int32_t* pos_thw,
                    const float* prev_hidden, int T, mtp_taps& tp,
                    const float* cos_in, const float* sin_in) {
    if (cache_len_ + T > max_len_) throw std::runtime_error("MTP KV cache overflow");
    const size_t TD = (size_t)T * MODEL_DIM;

    std::vector<float> cos_buf, sin_buf;
    const float *cos = cos_in, *sin = sin_in;
    if (!cos) {
        cos_buf.resize((size_t)T * ROPE_DIM);
        sin_buf.resize((size_t)T * ROPE_DIM);
        mrope_table(pos_thw, T, cos_buf.data(), sin_buf.data());
        cos = cos_buf.data(); sin = sin_buf.data();
    }

    // ---- fusion: embed | hidden -> fc -------------------------------------
    fuse(ids, prev_hidden, T, tp);

    // ---- the decoder layer -------------------------------------------------
    tp.layer_out.assign(TD, 0.0f);
    decoder_layer(tp.fc_out.data(), T, cos, sin, tp.layer_out.data());

    // ---- mtp.norm ----------------------------------------------------------
    tp.norm_out.assign(TD, 0.0f);
    for (int t = 0; t < T; t++)
        rms_norm(tp.layer_out.data() + (size_t)t * MODEL_DIM,
                 w_.out_norm.data(), MODEL_DIM,
                 tp.norm_out.data() + (size_t)t * MODEL_DIM);
}

/// The fusion half of a step: embedding lookup, the two pre-fc RMSNorms, the
/// concatenation and `model.mtp.fc`. Fills `tp` up to and including `fc_out`,
/// which is the decoder layer's input.
///
/// Touches no cache and no position, so the hybrid step 0 in speculate_npu runs
/// it unchanged.
void mtp_head::fuse(const int32_t* ids, const float* prev_hidden, int T,
                    mtp_taps& tp) {
    std::vector<float> fc_in;
    fuse_pre(ids, prev_hidden, T, tp, fc_in);
    q4nx_matmul(w_.fc, fc_in.data(), T, tp.fc_out.data());
}

/// Everything `fuse` does except the matmul. Split out so the NPU path runs
/// `model.mtp.fc` on dequant_mm and this verbatim.
void mtp_head::fuse_pre(const int32_t* ids, const float* prev_hidden, int T,
                        mtp_taps& tp, std::vector<float>& fc_in) {
    const size_t TD = (size_t)T * MODEL_DIM;

    // `resize` for the three this function fully writes -- `embed_rows` covers
    // every row of `inputs_embeds` and the loop below covers both norms.
    // `fc_out` keeps its `assign`: nothing here writes it, the caller's fusion
    // matmul does, and this is the one buffer of the four where a partial
    // writer downstream would go unnoticed.
    tp.inputs_embeds.resize(TD);
    tp.norm_embed.resize(TD);
    tp.norm_hidden.resize(TD);
    tp.fc_out.assign(TD, 0.0f);

    embed_->embed_rows(ids, T, tp.inputs_embeds.data());

    // Threaded on the shared gate, like every per-row stage in this file: the
    // iterations write disjoint rows and share only read-only weights, so the
    // arithmetic and therefore the reference are unchanged. See
    // common/host_par.hpp for why the gate is conditional and why it keeps
    // these serial at the draft loop's L = 1.
#pragma omp parallel for schedule(static) \
    if (TD >= q38npu::host_par_min())
    for (int t = 0; t < T; t++) {
        rms_norm(tp.inputs_embeds.data() + (size_t)t * MODEL_DIM,
                 w_.pre_fc_norm_embedding.data(), MODEL_DIM,
                 tp.norm_embed.data() + (size_t)t * MODEL_DIM);
        rms_norm(prev_hidden + (size_t)t * MODEL_DIM,
                 w_.pre_fc_norm_hidden.data(), MODEL_DIM,
                 tp.norm_hidden.data() + (size_t)t * MODEL_DIM);
    }

    // cat([norm_embed, norm_hidden]) -- embeddings first. Swapping the two
    // halves yields plausible-looking garbage, so this order is load-bearing.
    fc_in.resize((size_t)T * mtp::FC_IN);   // both memcpys cover every row
#pragma omp parallel for schedule(static) \
    if ((size_t)T * mtp::FC_IN >= q38npu::host_par_min())
    for (int t = 0; t < T; t++) {
        float* d = fc_in.data() + (size_t)t * mtp::FC_IN;
        std::memcpy(d, tp.norm_embed.data() + (size_t)t * MODEL_DIM,
                    MODEL_DIM * sizeof(float));
        std::memcpy(d + MODEL_DIM, tp.norm_hidden.data() + (size_t)t * MODEL_DIM,
                    MODEL_DIM * sizeof(float));
    }
}

/// One Qwen3_5DecoderLayer, full attention, gated. Identical to the main
/// model's full-attention layers; see cpu_ref_attention_layer.hpp.
///
/// The body is three helpers plus the cache append. Split for the same reason
/// attention_layer::forward() is: `speculate_npu`'s step 0 replaces the middle
/// one with an `attn.xclbin` dispatch and keeps the other two verbatim, so the
/// reference it is scored against has to BE the same code rather than a copy of
/// it. Calling them in this order from here is what keeps that true.
void mtp_head::decoder_layer(const float* x, int L, const float* cos,
                             const float* sin, float* y) {
    std::vector<float> qg, kk, vv, qh;
    project_qkv_rope(x, L, cos, sin, qg, kk, vv, qh);

    // ---- append to the head's own KV cache ---------------------------------
    // Stays here rather than in a helper: the hybrid path appends into the NPU
    // cache BO instead, and that is precisely the substitution being made.
    const int base = cache_len_;
    for (int t = 0; t < L; t++) {
        std::memcpy(cache_k_.data() + (size_t)(base + t) * DKV,
                    kk.data() + (size_t)t * DKV, DKV * sizeof(float));
        std::memcpy(cache_v_.data() + (size_t)(base + t) * DKV,
                    vv.data() + (size_t)t * DKV, DKV * sizeof(float));
    }
    cache_len_ += L;

    std::vector<float> attn;
    attn_core(qh.data(), L, base, attn);

    gate_out_mlp(qg.data(), attn.data(), x, L, y);
}

/// input_layernorm, q/k/v, per-head q/k RMSNorm, partial RoPE.
///
/// Touches no cache. `kk`/`vv` come back post-norm post-RoPE, ready to be
/// appended by whoever owns the cache; `qh` is [L][DQ] head-major, which is
/// exactly what attn.xclbin's q buffer wants.
void mtp_head::project_qkv_rope(const float* x, int L, const float* cos,
                                const float* sin, std::vector<float>& qg,
                                std::vector<float>& kk, std::vector<float>& vv,
                                std::vector<float>& qh) {
    const size_t LD = (size_t)L;
    std::vector<float> xn;
    in_norm(x, L, xn);

    // ---- projections -------------------------------------------------------
    qg.assign(LD * 2 * DQ, 0.0f);              // [q 6144 | gate 6144]
    kk.assign(LD * DKV, 0.0f);
    vv.assign(LD * DKV, 0.0f);
    q4nx_matmul(w_.q, xn.data(), L, qg.data());
    q4nx_matmul(w_.k, xn.data(), L, kk.data());
    q4nx_matmul(w_.v, xn.data(), L, vv.data());

    qkv_post_rope(qg.data(), kk.data(), L, cos, sin, qh);
}

/// input_layernorm alone, the part of `project_qkv_rope` before its matmuls.
void mtp_head::in_norm(const float* x, int L, std::vector<float>& xn) {
    xn.resize((size_t)L * MODEL_DIM);       // the loop writes all of it
#pragma omp parallel for schedule(static) \
    if ((size_t)L * MODEL_DIM >= q38npu::host_par_min())
    for (int t = 0; t < L; t++)
        rms_norm(x + (size_t)t * MODEL_DIM, w_.in_ln.data(), MODEL_DIM,
                 xn.data() + (size_t)t * MODEL_DIM);
}

/// The part of `project_qkv_rope` after its matmuls.
///
/// v never reaches here: neither the q/k norm nor RoPE touches it, so it comes
/// out of its projection ready to append and is not a parameter.
void mtp_head::qkv_post_rope(const float* qg, float* kk, int L,
                             const float* cos, const float* sin,
                             std::vector<float>& qh) {
    const size_t LD = (size_t)L;

    // ---- per-head q/k RMSNorm, then partial RoPE ---------------------------
    // q/k norm is on the 256 head dim and runs BEFORE RoPE -- same order as
    // the other Qwen ports, unlike the HunYuan one.
    qh.resize(LD * DQ);                     // the head loops cover all of it
    // `kk` is normed and rotated in place, but strictly within row `t`, so no
    // iteration reads another's row.
#pragma omp parallel for schedule(static) \
    if (LD * DQ >= q38npu::host_par_min())
    for (int t = 0; t < L; t++) {
        for (int h = 0; h < NUM_ATTN_HEADS; h++) {
            int qo, go; q_gate_off(h, qo, go);
            rms_norm(qg + (size_t)t * 2 * DQ + qo,
                     w_.q_norm.data(), HEAD_DIM,
                     qh.data() + (size_t)t * DQ + h * HEAD_DIM);
        }
        for (int h = 0; h < NUM_KV_HEADS; h++) {
            float tmp[HEAD_DIM];
            float* dst = kk + (size_t)t * DKV + h * HEAD_DIM;
            rms_norm(dst, w_.k_norm.data(), HEAD_DIM, tmp);
            std::memcpy(dst, tmp, sizeof tmp);
        }
        for (int h = 0; h < NUM_ATTN_HEADS; h++)
            rope(qh.data() + (size_t)t * DQ + h * HEAD_DIM,
                 cos + (size_t)t * ROPE_DIM, sin + (size_t)t * ROPE_DIM);
        for (int h = 0; h < NUM_KV_HEADS; h++)
            rope(kk + (size_t)t * DKV + h * HEAD_DIM,
                 cos + (size_t)t * ROPE_DIM, sin + (size_t)t * ROPE_DIM);
    }
}

/// Causal GQA softmax attention over the head's own cache.
///
/// \param base rows already in the cache BEFORE this window was appended, so
///             query `t` attends to keys `0 .. base + t`. The caller appends
///             first and passes the pre-append cursor -- getting that backwards
///             lets every query see the whole window including its own future.
///
/// This is the one helper `speculate_npu`'s step 0 does NOT call: it is what
/// attn.xclbin replaces.
void mtp_head::attn_core(const float* qh, int L, int base,
                         std::vector<float>& attn) const {
    const size_t LD = (size_t)L;

    // The mask is index-based and RoPE is value-based; they are deliberately
    // decoupled. Step 0 is a PxP lower triangle, the T=1 steps are one row of
    // all-ones over the whole cache.
    attn.assign(LD * DQ, 0.0f);
    // Hand-flattened rather than collapse(2): MSVC's /openmp is OpenMP 2.0,
    // which has no collapse clause. Same iteration space, same results.
    const int TH = L * NUM_ATTN_HEADS;
#pragma omp parallel for schedule(static)
    for (int th = 0; th < TH; th++) {
        const int t = th / NUM_ATTN_HEADS;
        const int h = th % NUM_ATTN_HEADS;
        {   // the old inner `for (h)` body, brace kept so the diff is small
            const int    kvh  = h / GQA_GROUP;
            const int    nkey = base + t + 1;
            const float* qv   = qh + (size_t)t * DQ + h * HEAD_DIM;
            std::vector<float> s(nkey);
            float smax = -INFINITY;
            for (int j = 0; j < nkey; j++) {
                const float* kv = cache_k_.data() + (size_t)j * DKV + kvh * HEAD_DIM;
                float acc = 0.0f;
                for (int d = 0; d < HEAD_DIM; d++) acc += qv[d] * kv[d];
                s[j] = acc * ATTN_SCALE;
                smax = std::max(smax, s[j]);
            }
            float sum = 0.0f;
            for (int j = 0; j < nkey; j++) { s[j] = std::exp(s[j] - smax); sum += s[j]; }
            const float inv = 1.0f / sum;
            float* out = attn.data() + (size_t)t * DQ + h * HEAD_DIM;
            for (int d = 0; d < HEAD_DIM; d++) out[d] = 0.0f;
            for (int j = 0; j < nkey; j++) {
                const float p = s[j] * inv;
                if (p == 0.0f) continue;
                const float* vj = cache_v_.data() + (size_t)j * DKV + kvh * HEAD_DIM;
                for (int d = 0; d < HEAD_DIM; d++) out[d] += p * vj[d];
            }
        }
    }
}

/// Output gate, o_proj, the attention residual, then post_attention_layernorm
/// and the whole MLP with its own residual. Everything after the attention
/// core.
///
/// \param attn the core's output, [L][DQ] head-major. Modified in place -- the
///             gate is applied to it -- so the caller's buffer is consumed.
/// \param x    the layer input, for the attention residual.
void mtp_head::gate_out_mlp(const float* qg, float* attn, const float* x,
                            int L, float* y) {
    const size_t LD = (size_t)L;

    apply_gate(qg, attn, L);

    std::vector<float> h1(LD * MODEL_DIM);
    q4nx_matmul(w_.o, attn, L, h1.data());
    for (size_t i = 0; i < LD * MODEL_DIM; i++) h1[i] += x[i];

    mlp_residual(h1.data(), L, y);
}

/// The sigmoid output gate, the part of `gate_out_mlp` before o_proj.
///
/// gate.reshape stays head-major and matches attn_output head for head, so the
/// multiply is per head under either layout.
void mtp_head::apply_gate(const float* qg, float* attn, int L) const {
#pragma omp parallel for schedule(static) \
    if ((size_t)L * DQ >= q38npu::host_par_min())
    for (int t = 0; t < L; t++) {
        const float* gb = qg + (size_t)t * 2 * DQ;
        float* a = attn + (size_t)t * DQ;
        for (int h = 0; h < NUM_ATTN_HEADS; h++) {
            int qo, go; q_gate_off(h, qo, go);
            float* ah = a + h * HEAD_DIM;
            for (int d = 0; d < HEAD_DIM; d++) ah[d] *= sigmoidf_(gb[go + d]);
        }
    }
}

void mtp_head::rope(float* v, const float* cos, const float* sin) {
    float r[ROPE_DIM];
    const int H = ROPE_DIM / 2;
    for (int j = 0; j < H; j++)         r[j] = v[j] * cos[j] - v[j + H] * sin[j];
    for (int j = H; j < ROPE_DIM; j++)  r[j] = v[j] * cos[j] + v[j - H] * sin[j];
    std::memcpy(v, r, sizeof r);
}

void mtp_head::mlp_residual(const float* h1, int L, float* y) {
    const size_t LD = (size_t)L;
    std::vector<float> hn;
    post_norm(h1, L, hn);
    std::vector<float> g(LD * INTERMEDIATE_SIZE), u(LD * INTERMEDIATE_SIZE);
    q4nx_matmul(w_.mlp_gate, hn.data(), L, g.data());
    q4nx_matmul(w_.mlp_up,   hn.data(), L, u.data());
    swiglu(g.data(), u.data(), L, g.data());
    q4nx_matmul(w_.mlp_down, g.data(), L, y);
    for (size_t i = 0; i < LD * MODEL_DIM; i++) y[i] += h1[i];
}

/// post_attention_layernorm alone, the part of `mlp_residual` before the MLP.
void mtp_head::post_norm(const float* h1, int L, std::vector<float>& hn) {
    hn.resize((size_t)L * MODEL_DIM);       // the loop writes all of it
#pragma omp parallel for schedule(static) \
    if ((size_t)L * MODEL_DIM >= q38npu::host_par_min())
    for (int t = 0; t < L; t++)
        rms_norm(h1 + (size_t)t * MODEL_DIM, w_.post_ln.data(), MODEL_DIM,
                 hn.data() + (size_t)t * MODEL_DIM);
}

/// silu(gate) * up. Elementwise, so `g` may alias `gate` -- mlp_residual
/// relies on that to avoid a third intermediate_size-wide buffer.
///
/// The order is not interchangeable: `silu(up) * gate` is just as plausible a
/// reading of a SwiGLU and scores nearly as well, which is why the driver runs
/// the swapped version as a control rather than trusting the cosine.
/// (memory: score-swiglu-by-swapping-up-and-gate)
void mtp_head::swiglu(const float* gate, const float* up, int L, float* g) const {
    // A row at a time so the induction variable is an `int` and each iteration
    // owns a disjoint span; the permitted aliasing is `g == gate`, which is
    // elementwise and stays inside one row.
    constexpr size_t DI = (size_t)INTERMEDIATE_SIZE;
#pragma omp parallel for schedule(static) \
    if ((size_t)L * DI >= q38npu::host_par_min())
    for (int t = 0; t < L; t++) {
        const size_t o = (size_t)t * DI;
        for (size_t i = 0; i < DI; i++) g[o + i] = siluf_(gate[o + i]) * up[o + i];
    }
}

} // namespace cpu_ref
