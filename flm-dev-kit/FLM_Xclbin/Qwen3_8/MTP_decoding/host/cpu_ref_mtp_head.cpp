// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file cpu_ref_mtp_head.cpp
/// \brief CPU golden reference for the Qwen3.8-27B MTP (multi-token-prediction)
///        draft head -- the T=1 chained decode steps and the T=221 priming step.
///
/// Target: models_simple/docs/QWEN3_8_MTP_DATAFLOW.html. One speculative cycle
/// is seven draft steps through a single decoder layer that owns its own KV
/// cache:
///
///     step 0   T = 221   x_1..x_P     prev_hidden = prefill final_norm_out
///     step s   T = 1     d_s          prev_hidden = the head's OWN norm_out
///
/// Step 0 exists only to prime the head's KV cache; steps 1..6 are the decode
/// shape this project's xclbin implements, and are what the NPU is scored on.
///
/// Math (transformers MtpLayer + Qwen3_5DecoderLayer forced to full attention):
///
///     e       = embed_tokens(step_ids)              -- raw ids, image slots
///                                                      get the placeholder id
///     fc_in   = cat[ rms(e, pre_fc_norm_embedding)      -- cols 0:5120
///                  , rms(h, pre_fc_norm_hidden)    ]    -- cols 5120:10240
///     fc_out  = W_fc fc_in                          -- 10240 -> 5120
///     ... one Qwen3_5DecoderLayer, layer_idx 0, full attention ...
///     norm_out = rms(layer_out, mtp.norm)
///     logits   = lm_head(norm_out[-1]) ; draft = argmax
///
/// and the layer itself is the same gated attention as the main model's full
/// layers, so this mirrors cpu_ref_attention_layer.cpp line for line:
///
///     xn      = rmsnorm(x, input_layernorm)
///     q|gate  = W_q xn          -- 12288 rows, q [0:6144] then gate [6144:12288]
///     k, v    = W_k xn, W_v xn                  [1024] = 4 kv heads x 256
///     q, k    = q_norm(q), k_norm(k)            per 256-dim head, BEFORE rope
///     q, k    = rope(q, k)                      first 64 dims only, rotate_half
///     cache  += k, v
///     o       = softmax(q K^T / 16 + causal) V  GQA, kv head = q head / 6
///     o      *= sigmoid(gate)
///     h       = x + W_o o
///     y       = h + down(silu(gate(rmsnorm(h))) * up(rmsnorm(h)))
///
/// Weights are the 15 `model.mtp.*` tensors in model.q4nx, Q4_K in 32x256
/// blocks (4736 B/block), plus the shared Q8_0 embed_tokens and lm_head.
///
/// Everything is fp32. This is a golden reference for the *math*, not a
/// bit-accurate model of the AIE datapath, so the expected agreement with
/// ref.safetensors is "Q4_K's own error floor", not 1e-6.
///
/// Build (standalone; without -DCPU_REF_STANDALONE this file defines no main
/// and is harmless in the run.exe glob):
///   g++ -O3 -march=native -fopenmp -std=c++17 -DCPU_REF_STANDALONE
///       -Ihost host/cpu_ref_mtp_head.cpp -o cpu_ref_mtp_head
/// Run:
///   ./cpu_ref_mtp_head [model.q4nx] [ref.safetensors] [--flags]

#include "cpu_ref_common.hpp"

#include <chrono>
#include <algorithm>

namespace cpu_ref {

using namespace q38;

namespace mtp {
constexpr int FC_IN      = 2 * MODEL_DIM;   // 10240, cat of two normed 5120s
constexpr int VOCAB      = 248320;
constexpr int MTP_STEPS  = 7;               // draft depth k
constexpr float ROPE_THETA = 1e7f;
} // namespace mtp

// ==========================================================================
// weights
// ==========================================================================
struct mtp_weights {
    // the head proper -- all under model.mtp.
    q4k_tensor fc;         // [ 5120][10240]  cols 0:5120 embed, 5120:10240 hidden
    q4k_tensor q;          // [12288][ 5120]  q rows then gate rows
    q4k_tensor k;          // [ 1024][ 5120]
    q4k_tensor v;          // [ 1024][ 5120]
    q4k_tensor o;          // [ 5120][ 6144]
    q4k_tensor mlp_gate;   // [17408][ 5120]
    q4k_tensor mlp_up;     // [17408][ 5120]
    q4k_tensor mlp_down;   // [ 5120][17408]

    // shared with the main model, Q8_0
    q4k_tensor embed;      // [248320][5120]
    q4k_tensor lm_head;    // [248320][5120]

    std::vector<float> pre_fc_norm_embedding;  // [5120]
    std::vector<float> pre_fc_norm_hidden;     // [5120]
    std::vector<float> in_ln;                  // [5120]
    std::vector<float> post_ln;                // [5120]
    std::vector<float> out_norm;               // [5120]  mtp.norm
    std::vector<float> q_norm;                 // [256]
    std::vector<float> k_norm;                 // [256]

    void load(const safetensors& f, bool with_lm_head = true) {
        const std::string p = "model.mtp.";
        fc       = get_q4k(f, p + "fc.weight");
        q        = get_q4k(f, p + "layers.0.self_attn.q_proj.weight");
        k        = get_q4k(f, p + "layers.0.self_attn.k_proj.weight");
        v        = get_q4k(f, p + "layers.0.self_attn.v_proj.weight");
        o        = get_q4k(f, p + "layers.0.self_attn.o_proj.weight");
        mlp_gate = get_q4k(f, p + "layers.0.mlp.gate_proj.weight");
        mlp_up   = get_q4k(f, p + "layers.0.mlp.up_proj.weight");
        mlp_down = get_q4k(f, p + "layers.0.mlp.down_proj.weight");

        embed = get_q4k(f, "model.embed_tokens.weight");
        if (with_lm_head) lm_head = get_q4k(f, "lm_head.weight");

        pre_fc_norm_embedding = f.f32(p + "pre_fc_norm_embedding.weight");
        pre_fc_norm_hidden    = f.f32(p + "pre_fc_norm_hidden.weight");
        in_ln    = f.f32(p + "layers.0.input_layernorm.weight");
        post_ln  = f.f32(p + "layers.0.post_attention_layernorm.weight");
        out_norm = f.f32(p + "norm.weight");
        q_norm   = f.f32(p + "layers.0.self_attn.q_norm.weight");
        k_norm   = f.f32(p + "layers.0.self_attn.k_norm.weight");

        // The three dimensions that surprise people (doc section 3): q_proj is
        // 2*DQ because attn_output_gate emits a query *and* a gate per head,
        // o_proj takes DQ because the gate is consumed before it, and fc takes
        // 2*D because its input is the concat of two normalised 5120-vectors.
        if (q.rows != 2 * DQ || k.rows != DKV || v.rows != DKV)
            throw std::runtime_error("mtp: qkv projection geometry mismatch");
        if (o.cols != DQ || o.rows != MODEL_DIM)
            throw std::runtime_error("mtp: o_proj geometry mismatch");
        if (fc.rows != MODEL_DIM || fc.cols != mtp::FC_IN)
            throw std::runtime_error("mtp: fc geometry mismatch");
        if (embed.cols != MODEL_DIM || embed.rows < mtp::VOCAB)
            throw std::runtime_error("mtp: embed_tokens geometry mismatch");
    }
};

// ==========================================================================
// M-RoPE table
//
// partial_rotary_factor 0.25 -> rotary_dim 64 -> 32 frequency slots, and
// mrope_interleaved assigns slot i to channel i%3 (t,h,w). For pure-text
// drafts all three channels carry the same value and this degenerates to
// plain 1-D RoPE; only step 0's window touches the image plateau, where the
// three channels genuinely differ.
//
// transformers computes this in fp32 and returns cos.to(bf16), so the
// reference consumed bf16 tables even though the fp32 twin is also dumped.
// ==========================================================================
inline void mrope_table(const int32_t* pos_thw, int T, float* cos, float* sin,
                        bool round_to_bf16 = true) {
    const int H = ROPE_DIM / 2;                   // 32 frequency slots
    for (int t = 0; t < T; t++) {
        for (int j = 0; j < H; j++) {
            const int   ch  = j % 3;              // 0 = t, 1 = h, 2 = w
            const float pos = static_cast<float>(pos_thw[ch * T + t]);
            const double inv =
                std::pow(static_cast<double>(mtp::ROPE_THETA),
                         -static_cast<double>(2 * j) / static_cast<double>(ROPE_DIM));
            const double ang = static_cast<double>(pos) * inv;
            float c = static_cast<float>(std::cos(ang));
            float s = static_cast<float>(std::sin(ang));
            if (round_to_bf16) { c = round_bf16(c); s = round_bf16(s); }
            // the 64-wide table is the 32 slots duplicated, matching rotate_half
            cos[(size_t)t * ROPE_DIM + j]     = c;
            cos[(size_t)t * ROPE_DIM + j + H] = c;
            sin[(size_t)t * ROPE_DIM + j]     = s;
            sin[(size_t)t * ROPE_DIM + j + H] = s;
        }
    }
}

// ==========================================================================
// per-step intermediates, so the driver can score every tap the dump carries
// ==========================================================================
struct mtp_taps {
    std::vector<float> inputs_embeds;   // [T][5120]
    std::vector<float> norm_embed;      // [T][5120]
    std::vector<float> norm_hidden;     // [T][5120]
    std::vector<float> fc_out;          // [T][5120]
    std::vector<float> layer_out;       // [T][5120]
    std::vector<float> norm_out;        // [T][5120]
};

// ==========================================================================
// the head
// ==========================================================================
class mtp_head {
public:
    mtp_head(const safetensors& q4nx, int max_len = 4096, bool with_lm_head = true)
        : max_len_(max_len) {
        w_.load(q4nx, with_lm_head);
        cache_k_.assign((size_t)max_len_ * DKV, 0.0f);
        cache_v_.assign((size_t)max_len_ * DKV, 0.0f);
    }

    void reset() { cache_len_ = 0; }
    int  cache_len() const { return cache_len_; }

    /// Crop the head's cache back to `len`. Section 10.3: at a cycle boundary
    /// this is always k-1, regardless of how many drafts were accepted,
    /// because every chained slot was fused against the head's own output
    /// rather than the main model's.
    void crop(int len) { cache_len_ = std::min(cache_len_, std::max(0, len)); }

    /// embed_tokens row lookup. The table is Q8_0, so a row costs one block
    /// column sweep; image slots resolve to the placeholder id's row, exactly
    /// as MtpModel.forward does (doc section 8).
    void embed_row(int32_t token, float* out) const {
        const int rb = token / Q4NX_ROW_BLOCK;
        const int rl = token % Q4NX_ROW_BLOCK;
        std::vector<float> tile(Q4NX_COL_BLOCK * Q4NX_ROW_BLOCK);
        for (int cb = 0; cb < w_.embed.ncb; cb++) {
            w_.embed.dequant_block(rb, cb, tile.data());
            float* dst = out + (size_t)cb * Q4NX_COL_BLOCK;
            for (int c = 0; c < Q4NX_COL_BLOCK; c++)
                dst[c] = tile[(size_t)c * Q4NX_ROW_BLOCK + rl];
        }
    }

    /// One MTP step. `ids`/`pos_thw` are the window contents (T tokens,
    /// positions laid out (3,T) as t,h,w), `prev_hidden` is [T][5120].
    /// Appends T rows to the head's KV cache and returns norm_out in `tp`.
    ///
    /// If `cos_in`/`sin_in` are null the M-RoPE table is generated here;
    /// pass the dumped tables to isolate a table mismatch from a math one.
    ///
    /// `fc_in_override`, if given, replaces the computed fc_out -- feeding the
    /// dump's fc_out isolates the decoder layer from the fusion in front of it.
    void step(const int32_t* ids, const int32_t* pos_thw, const float* prev_hidden,
              int T, mtp_taps& tp,
              const float* cos_in = nullptr, const float* sin_in = nullptr,
              const float* fc_in_override = nullptr) {
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

        // ---- fusion: embed | hidden -> fc ---------------------------------
        tp.inputs_embeds.assign(TD, 0.0f);
        tp.norm_embed.assign(TD, 0.0f);
        tp.norm_hidden.assign(TD, 0.0f);
        tp.fc_out.assign(TD, 0.0f);

#pragma omp parallel for schedule(static)
        for (int t = 0; t < T; t++)
            embed_row(ids[t], tp.inputs_embeds.data() + (size_t)t * MODEL_DIM);

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
        std::vector<float> fc_in((size_t)T * mtp::FC_IN);
        for (int t = 0; t < T; t++) {
            float* d = fc_in.data() + (size_t)t * mtp::FC_IN;
            std::memcpy(d, tp.norm_embed.data() + (size_t)t * MODEL_DIM,
                        MODEL_DIM * sizeof(float));
            std::memcpy(d + MODEL_DIM, tp.norm_hidden.data() + (size_t)t * MODEL_DIM,
                        MODEL_DIM * sizeof(float));
        }
        q4k_matmul(w_.fc, fc_in.data(), T, tp.fc_out.data());
        if (fc_in_override) std::memcpy(tp.fc_out.data(), fc_in_override, TD * sizeof(float));

        // ---- the decoder layer --------------------------------------------
        tp.layer_out.assign(TD, 0.0f);
        decoder_layer(tp.fc_out.data(), T, cos, sin, tp.layer_out.data());

        // ---- mtp.norm ------------------------------------------------------
        tp.norm_out.assign(TD, 0.0f);
        for (int t = 0; t < T; t++)
            rms_norm(tp.layer_out.data() + (size_t)t * MODEL_DIM,
                     w_.out_norm.data(), MODEL_DIM,
                     tp.norm_out.data() + (size_t)t * MODEL_DIM);
    }

    /// lm_head on a single 5120-vector. 2.37 GiB of Q8_0 and 75 % of the
    /// draft-step cost, which is why it is separable from step().
    void logits(const float* norm_out_row, float* out) const {
        if (!w_.lm_head.blocks) throw std::runtime_error("lm_head not loaded");
        q4k_matmul(w_.lm_head, norm_out_row, 1, out);
    }

    static int argmax(const float* v, int n) {
        int best = 0;
        for (int i = 1; i < n; i++) if (v[i] > v[best]) best = i;
        return best;
    }

private:
    /// One Qwen3_5DecoderLayer, full attention, gated. Identical to the main
    /// model's full-attention layers; see cpu_ref_attention_layer.cpp.
    void decoder_layer(const float* x, int L, const float* cos, const float* sin,
                       float* y) {
        const size_t LD = (size_t)L;
        std::vector<float> xn(LD * MODEL_DIM);
        for (int t = 0; t < L; t++)
            rms_norm(x + (size_t)t * MODEL_DIM, w_.in_ln.data(), MODEL_DIM,
                     xn.data() + (size_t)t * MODEL_DIM);

        // ---- projections ---------------------------------------------------
        std::vector<float> qg(LD * 2 * DQ);        // [q 6144 | gate 6144]
        std::vector<float> kk(LD * DKV), vv(LD * DKV);
        q4k_matmul(w_.q, xn.data(), L, qg.data());
        q4k_matmul(w_.k, xn.data(), L, kk.data());
        q4k_matmul(w_.v, xn.data(), L, vv.data());

        // ---- per-head q/k RMSNorm, then partial RoPE ------------------------
        // q/k norm is on the 256 head dim and runs BEFORE RoPE -- same order as
        // the other Qwen ports, unlike the HunYuan one.
        std::vector<float> qh(LD * DQ);
        for (int t = 0; t < L; t++) {
            for (int h = 0; h < NUM_ATTN_HEADS; h++) {
                int qo, go; q_gate_off(h, qo, go);
                rms_norm(qg.data() + (size_t)t * 2 * DQ + qo,
                         w_.q_norm.data(), HEAD_DIM,
                         qh.data() + (size_t)t * DQ + h * HEAD_DIM);
            }
            for (int h = 0; h < NUM_KV_HEADS; h++) {
                float tmp[HEAD_DIM];
                float* dst = kk.data() + (size_t)t * DKV + h * HEAD_DIM;
                rms_norm(dst, w_.k_norm.data(), HEAD_DIM, tmp);
                std::memcpy(dst, tmp, sizeof tmp);
            }
            for (int h = 0; h < NUM_ATTN_HEADS; h++)
                rope(qh.data() + (size_t)t * DQ + h * HEAD_DIM,
                     cos + (size_t)t * ROPE_DIM, sin + (size_t)t * ROPE_DIM);
            for (int h = 0; h < NUM_KV_HEADS; h++)
                rope(kk.data() + (size_t)t * DKV + h * HEAD_DIM,
                     cos + (size_t)t * ROPE_DIM, sin + (size_t)t * ROPE_DIM);
        }

        // ---- append to the head's own KV cache ------------------------------
        const int base = cache_len_;
        for (int t = 0; t < L; t++) {
            std::memcpy(cache_k_.data() + (size_t)(base + t) * DKV,
                        kk.data() + (size_t)t * DKV, DKV * sizeof(float));
            std::memcpy(cache_v_.data() + (size_t)(base + t) * DKV,
                        vv.data() + (size_t)t * DKV, DKV * sizeof(float));
        }
        cache_len_ += L;

        // ---- causal GQA softmax attention -----------------------------------
        // The mask is index-based and RoPE is value-based; they are deliberately
        // decoupled (doc section 6.1). Step 0 is a 221x221 lower triangle, the
        // T=1 steps are one row of all-ones over the whole cache.
        std::vector<float> attn(LD * DQ);
#pragma omp parallel for collapse(2) schedule(static)
        for (int t = 0; t < L; t++) {
            for (int h = 0; h < NUM_ATTN_HEADS; h++) {
                const int    kvh  = h / GQA_GROUP;
                const int    nkey = base + t + 1;
                const float* qv   = qh.data() + (size_t)t * DQ + h * HEAD_DIM;
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

        // ---- output gate, o_proj, residual -----------------------------------
        // gate.reshape stays head-major and matches attn_output head for head,
        // so the multiply is per head under either layout.
        for (int t = 0; t < L; t++) {
            const float* gb = qg.data() + (size_t)t * 2 * DQ;
            float* a = attn.data() + (size_t)t * DQ;
            for (int h = 0; h < NUM_ATTN_HEADS; h++) {
                int qo, go; q_gate_off(h, qo, go);
                float* ah = a + h * HEAD_DIM;
                for (int d = 0; d < HEAD_DIM; d++) ah[d] *= sigmoidf_(gb[go + d]);
            }
        }
        std::vector<float> h1(LD * MODEL_DIM);
        q4k_matmul(w_.o, attn.data(), L, h1.data());
        for (size_t i = 0; i < LD * MODEL_DIM; i++) h1[i] += x[i];

        mlp_residual(h1.data(), L, y);
    }

    /// Partial rotary: dims [0,64) only, rotate_half with halves 0..31 / 32..63.
    /// Dims [64,256) pass through unrotated.
    static void rope(float* v, const float* cos, const float* sin) {
        float r[ROPE_DIM];
        const int H = ROPE_DIM / 2;
        for (int j = 0; j < H; j++)         r[j] = v[j] * cos[j] - v[j + H] * sin[j];
        for (int j = H; j < ROPE_DIM; j++)  r[j] = v[j] * cos[j] + v[j - H] * sin[j];
        std::memcpy(v, r, sizeof r);
    }

    void mlp_residual(const float* h1, int L, float* y) {
        const size_t LD = (size_t)L;
        std::vector<float> hn(LD * MODEL_DIM);
        for (int t = 0; t < L; t++)
            rms_norm(h1 + (size_t)t * MODEL_DIM, w_.post_ln.data(), MODEL_DIM,
                     hn.data() + (size_t)t * MODEL_DIM);
        std::vector<float> g(LD * INTERMEDIATE_SIZE), u(LD * INTERMEDIATE_SIZE);
        q4k_matmul(w_.mlp_gate, hn.data(), L, g.data());
        q4k_matmul(w_.mlp_up,   hn.data(), L, u.data());
        for (size_t i = 0; i < LD * INTERMEDIATE_SIZE; i++) g[i] = siluf_(g[i]) * u[i];
        q4k_matmul(w_.mlp_down, g.data(), L, y);
        for (size_t i = 0; i < LD * MODEL_DIM; i++) y[i] += h1[i];
    }

public:
    /// HF stores q_proj head-interleaved (row h*512+0..255 = head h query,
    /// +256..511 = head h gate); the q4nx converter de-interleaves the main
    /// model's layers into [all q | all gate]. Which of the two the MTP head's
    /// copy got is a property of the checkpoint, so it is selectable.
    bool q_interleaved = false;

private:
    /// Offsets of head h's query and gate rows under the active layout.
    void q_gate_off(int h, int& qo, int& go) const {
        if (q_interleaved) { qo = h * 2 * HEAD_DIM; go = qo + HEAD_DIM; }
        else               { qo = h * HEAD_DIM;     go = DQ + qo;       }
    }

    int                max_len_, cache_len_ = 0;
    mtp_weights        w_;
    std::vector<float> cache_k_, cache_v_;   // [max_len][1024]
};

} // namespace cpu_ref

// ==========================================================================
// driver
//
// Replays one full speculative cycle -- step 0 (T=221, cache priming) then the
// six chained T=1 decode steps -- and scores every tap against the transformers
// dump in ref.safetensors.
//
// Two modes:
//   chained (default)  prev_hidden and step_ids come from our own previous
//                      step, so errors compound exactly as they will on the
//                      NPU. This is the number that matters.
//   --teacher-forced   prev_hidden and step_ids come from the dump each step,
//                      isolating per-step error from accumulation. Use this
//                      first when a chained run diverges, to find which step
//                      actually broke.
//
// Build:
//   g++ -O3 -march=native -fopenmp -std=c++17 -DCPU_REF_STANDALONE -Ihost \
//       host/cpu_ref_mtp_head.cpp -o cpu_ref_mtp_head
// ==========================================================================
#ifdef CPU_REF_STANDALONE

#include <cstring>

namespace {

using namespace cpu_ref;

void score(const char* tag, const std::vector<float>& got,
           const safetensors& ref, const std::string& key) {
    if (!ref.has(key)) { std::printf("  %-24s (no tap)\n", tag); return; }
    std::vector<float> want = ref.f32(key);
    const size_t n = std::min(got.size(), want.size());
    print_err(tag, error_matrix(got.data(), want.data(), n));
}

} // namespace

int main(int argc, char** argv) {
    using namespace cpu_ref;
    using namespace cpu_ref::q38;
    using clk = std::chrono::steady_clock;

    std::string q4nx_path = "/path/to/models/Qwen3.8-27B-NPU2/model.q4nx";
    std::string ref_path  = "/path/to/models/Qwen3.8-27B-NPU2/ref.safetensors";
    bool teacher_forced = false;   // feed the dump's prev_hidden/ids each step
    bool rope_from_ref  = false;   // use the dumped cos/sin instead of generating
    bool no_logits      = false;   // skip lm_head (fast iteration on the head math)
    bool fc_from_ref    = false;   // feed the dump's fc_out into the decoder layer
    bool q_interleaved  = false;   // q_proj rows are per-head (q,gate) pairs
    bool per_token      = false;   // per-row error, to localise a structural fault
    int  steps          = mtp::MTP_STEPS;

    std::vector<std::string> pos;
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        if      (a == "--teacher-forced") teacher_forced = true;
        else if (a == "--rope-from-ref")  rope_from_ref  = true;
        else if (a == "--no-logits")      no_logits      = true;
        else if (a == "--fc-from-ref")    fc_from_ref    = true;
        else if (a == "--q-interleaved")  q_interleaved  = true;
        else if (a == "--per-token")      per_token      = true;
        else if (a == "--steps" && i + 1 < argc) steps = std::atoi(argv[++i]);
        else if (a.rfind("--", 0) == 0) {
            std::fprintf(stderr,
                "usage: %s [model.q4nx] [ref.safetensors]"
                " [--teacher-forced] [--rope-from-ref] [--no-logits] [--steps N]\n",
                argv[0]);
            return 1;
        }
        else pos.push_back(a);
    }
    if (pos.size() > 0) q4nx_path = pos[0];
    if (pos.size() > 1) ref_path  = pos[1];
    steps = std::max(1, std::min(steps, mtp::MTP_STEPS));

    safetensors q4nx(q4nx_path);
    safetensors ref(ref_path);

    const int P = static_cast<int>(ref.info("mtp.step0.input_ids").shape[1]);

    std::printf("== Qwen3.8-27B MTP draft head ==\n");
    std::printf("   step 0 primes the cache with %d tokens, then %d chained T=1 decode steps\n",
                P, steps - 1);
    std::printf("   weights Q4_K 32x256 (model.mtp.*), embed/lm_head Q8_0\n");
    std::printf("   mode: %s, rope: %s, lm_head: %s\n\n",
                teacher_forced ? "teacher-forced" : "chained",
                rope_from_ref ? "from dump" : "generated",
                no_logits ? "skipped" : "on");

    auto tload = clk::now();
    mtp_head head(q4nx, P + mtp::MTP_STEPS + 64, !no_logits);
    head.q_interleaved = q_interleaved;
    std::printf("weights mapped in %.2f s\n\n",
                std::chrono::duration<double>(clk::now() - tload).count());

    std::vector<int32_t> ref_drafts = ref.i32("mtp.draft_tokens");

    // chained state: what the previous step produced
    std::vector<float>   prev_hidden;      // [T][5120]
    std::vector<int32_t> step_ids;
    int32_t              last_draft = 0;
    int                  n_match = 0, n_scored = 0;

    for (int s = 0; s < steps; s++) {
        const std::string pre = "mtp.step" + std::to_string(s) + ".";
        const int T = static_cast<int>(ref.info(pre + "input_ids").shape[1]);

        std::vector<int32_t> ids  = ref.i32(pre + "input_ids");
        std::vector<int32_t> pids = ref.i32(pre + "position_ids");   // (3,1,T)

        // Step 0 always takes the dump's prev_hidden -- it is the main model's
        // prefill final_norm_out, which this project does not compute. Steps
        // 1..6 chain on our own norm_out unless teacher-forced.
        std::vector<float> hid;
        if (s == 0 || teacher_forced) {
            hid = ref.f32(pre + "prev_hidden");
            if (s > 0) ids = ref.i32(pre + "input_ids");
        } else {
            hid = prev_hidden;                 // our own norm_out's last row
            ids.assign(1, last_draft);         // the token we just emitted
        }

        std::vector<float> cosb, sinb;
        const float *cosp = nullptr, *sinp = nullptr;
        if (rope_from_ref) {
            cosb = ref.f32(pre + "cos"); sinb = ref.f32(pre + "sin");
            cosp = cosb.data(); sinp = sinb.data();
        }

        std::vector<float> fcb;
        const float* fcp = nullptr;
        if (fc_from_ref) { fcb = ref.f32(pre + "fc_out"); fcp = fcb.data(); }

        mtp_taps tp;
        auto t0 = clk::now();
        head.step(ids.data(), pids.data(), hid.data(), T, tp, cosp, sinp, fcp);
        const double dt = std::chrono::duration<double>(clk::now() - t0).count();

        std::printf("---- step %d : T = %d, positions %d..%d, cache -> %d ----\n",
                    s, T, pids[0], pids[T - 1], head.cache_len());
        std::printf("  %.2f s\n", dt);

        // Only the taps whose inputs we actually reproduced are meaningful in
        // chained mode -- inputs_embeds/norm_embed depend on our own token.
        score("inputs_embeds", tp.inputs_embeds, ref, pre + "inputs_embeds");
        score("norm_embed",    tp.norm_embed,    ref, pre + "norm_embed");
        score("norm_hidden",   tp.norm_hidden,   ref, pre + "norm_hidden");
        score("fc_out",        tp.fc_out,        ref, pre + "fc_out");
        score("layer_out",     tp.layer_out,     ref, pre + "layer_out");
        score("norm_out",      tp.norm_out,      ref, pre + "norm_out");

        // layer_out carries fc_out straight through on the residual path, so the
        // raw cosine above is flattered by an identity term. Subtracting the
        // layer's input leaves attention+MLP alone -- the number that actually
        // moves when a head order or a mask is wrong.
        if (ref.has(pre + "layer_out") && ref.has(pre + "fc_out")) {
            std::vector<float> lref = ref.f32(pre + "layer_out");
            std::vector<float> fin  = ref.f32(pre + "fc_out");
            const size_t n = std::min(tp.layer_out.size(), lref.size());
            std::vector<float> a(n), b(n);
            for (size_t i = 0; i < n; i++) {
                a[i] = tp.layer_out[i] - fin[i];
                b[i] = lref[i]         - fin[i];
            }
            print_err("layer_out - residual", error_matrix(a.data(), b.data(), n));

            // Per-token, to separate a uniform quantisation floor from a
            // structural fault: a mask or position bug lands on specific rows
            // (the image plateau at slots 45..109, or the first/last token)
            // long before it moves the aggregate.
            if (per_token) {
                std::printf("  per-token (residual-free) cos:\n");
                for (int t = 0; t < T; t++) {
                    err_stats st = error_matrix(a.data() + (size_t)t * MODEL_DIM,
                                                b.data() + (size_t)t * MODEL_DIM,
                                                MODEL_DIM);
                    if (T <= 16 || t < 4 || (t >= 44 && t <= 48) ||
                        (t >= 108 && t <= 112) || t >= T - 4)
                        std::printf("    t %3d  cos %.6f  relL1 %.5f\n",
                                    t, st.cosine, st.rel_l1);
                }
            }
        }

        // carry the last row forward: prev_hidden for step s+1 is our norm_out
        prev_hidden.assign(tp.norm_out.end() - MODEL_DIM, tp.norm_out.end());

        if (!no_logits) {
            std::vector<float> lg(mtp::VOCAB);
            t0 = clk::now();
            head.logits(prev_hidden.data(), lg.data());
            const double dtl = std::chrono::duration<double>(clk::now() - t0).count();
            last_draft = mtp_head::argmax(lg.data(), mtp::VOCAB);

            score("logits_last", lg, ref, pre + "logits_last");
            const int32_t want = ref.i32(pre + "draft_token")[0];
            const bool ok = (last_draft == want);
            n_scored++; n_match += ok;
            std::printf("  %-24s got %6d  ref %6d  %s   (lm_head %.2f s)\n",
                        "draft_token", last_draft, want, ok ? "OK" : "MISMATCH", dtl);
        } else {
            // without lm_head we cannot chain tokens; fall back to the dump's
            last_draft = (s < (int)ref_drafts.size()) ? ref_drafts[s] : 0;
        }
        std::printf("\n");
    }

    if (!no_logits)
        std::printf("draft tokens matched: %d / %d\n", n_match, n_scored);
    std::printf("final MTP cache length: %d  (expected P + steps - 1 = %d)\n",
                head.cache_len(), P + steps - 1);

    return 0;
}
#endif // CPU_REF_STANDALONE
