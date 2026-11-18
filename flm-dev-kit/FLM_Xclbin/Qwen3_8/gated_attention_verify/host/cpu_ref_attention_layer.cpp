// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file cpu_ref_attention_layer.cpp
/// \brief CPU golden reference for one Qwen3.8-27B gated full-attention block.
///
/// Layer i is full attention iff (i+1) % 4 == 0, so 3, 7, ... 63. Same target
/// as the delta-net reference: the MTP verify step, 8 query tokens against a
/// KV cache that already holds the 221 prefill tokens.
///
/// Weights come straight out of model.q4nx in Q4_K (32x256 chunks,
/// parallel_size 32).
///
/// Math (transformers Qwen3_5Attention, checked line by line):
///
///     xn      = rmsnorm(x, input_layernorm)
///     q|gate  = W_q xn           -- 12288 rows; q4nx stores q [0:6144] then
///                                   gate [6144:12288], both 24 heads x 256
///                                   (the converter de-interleaves HF's
///                                    per-head (q,gate) pairs)
///     k, v    = W_k xn, W_v xn                  [1024] = 4 kv heads x 256
///     q, k    = q_norm(q), k_norm(k)            per 256-dim head, BEFORE rope
///     q, k    = rope(q, k)                      first 64 dims only, rotate_half
///     cache  += k, v
///     o       = softmax(q K^T / 16 + causal) V  GQA, kv head = q head / 6
///     o      *= sigmoid(gate)
///     h       = x + W_o o
///     y       = h + down(silu(gate(rmsnorm(h))) * up(rmsnorm(h)))
///
/// Build (see the driver at the bottom of the file):
///   g++ -O3 -march=native -fopenmp -std=c++17 -DCPU_REF_STANDALONE -Ihost
///       host/cpu_ref_attention_layer.cpp -o cpu_ref_attention_layer
/// Run:
///   ./cpu_ref_attention_layer [model.q4nx] [ref.safetensors] [layer]

#include "cpu_ref_common.hpp"

#include <chrono>
#include <algorithm>

namespace cpu_ref {

using namespace q38;

// ==========================================================================
// weights
// ==========================================================================
struct attention_weights {
    q4k_tensor q;          // [12288][5120]  q rows then gate rows
    q4k_tensor k;          // [ 1024][5120]
    q4k_tensor v;          // [ 1024][5120]
    q4k_tensor o;          // [ 5120][6144]
    q4k_tensor mlp_gate;   // [17408][5120]
    q4k_tensor mlp_up;     // [17408][5120]
    q4k_tensor mlp_down;   // [ 5120][17408]

    std::vector<float> in_ln;    // [5120]
    std::vector<float> post_ln;  // [5120]
    std::vector<float> q_norm;   // [256]
    std::vector<float> k_norm;   // [256]

    void load(const safetensors& f, int layer) {
        const std::string p = "model.layers." + std::to_string(layer) + ".";
        q        = get_q4k(f, p + "self_attn.q_proj.weight");
        k        = get_q4k(f, p + "self_attn.k_proj.weight");
        v        = get_q4k(f, p + "self_attn.v_proj.weight");
        o        = get_q4k(f, p + "self_attn.o_proj.weight");
        mlp_gate = get_q4k(f, p + "mlp.gate_proj.weight");
        mlp_up   = get_q4k(f, p + "mlp.up_proj.weight");
        mlp_down = get_q4k(f, p + "mlp.down_proj.weight");

        in_ln   = f.f32(p + "input_layernorm.weight");
        post_ln = f.f32(p + "post_attention_layernorm.weight");
        q_norm  = f.f32(p + "self_attn.q_norm.weight");
        k_norm  = f.f32(p + "self_attn.k_norm.weight");

        if (q.rows != 2 * DQ || k.rows != DKV || o.cols != DQ)
            throw std::runtime_error("attention projection geometry mismatch");
    }
};

// ==========================================================================
// the layer
// ==========================================================================
class attention_layer {
public:
    attention_layer(const safetensors& q4nx, int layer, int max_len = 4096)
        : layer_(layer), max_len_(max_len) {
        w_.load(q4nx, layer);
        cache_k_.assign((size_t)max_len_ * DKV, 0.0f);
        cache_v_.assign((size_t)max_len_ * DKV, 0.0f);
    }

    void reset() { cache_len_ = 0; }
    int  cache_len() const { return cache_len_; }
    const std::vector<float>& cache_k() const { return cache_k_; }
    const std::vector<float>& cache_v() const { return cache_v_; }

    /// x, y: [L][5120]. cos/sin: [L][64] for the L new positions.
    /// Appends L rows to the KV cache; queries see keys [0, cache_len+t].
    void forward(const float* x, int L, const float* cos, const float* sin, float* y) {
        if (cache_len_ + L > max_len_) throw std::runtime_error("KV cache overflow");
        const size_t LD = static_cast<size_t>(L);

        std::vector<float> xn(LD * MODEL_DIM);
        for (int t = 0; t < L; t++)
            rms_norm(x + t * (size_t)MODEL_DIM, w_.in_ln.data(), MODEL_DIM,
                     xn.data() + t * (size_t)MODEL_DIM);

        // ---- projections -------------------------------------------------
        std::vector<float> qg(LD * 2 * DQ);          // [q 6144 | gate 6144]
        std::vector<float> kk(LD * DKV), vv(LD * DKV);
        q4k_matmul(w_.q, xn.data(), L, qg.data());
        q4k_matmul(w_.k, xn.data(), L, kk.data());
        q4k_matmul(w_.v, xn.data(), L, vv.data());

        // ---- per-head q/k RMSNorm, then partial RoPE ----------------------
        std::vector<float> qh(LD * DQ);
        for (int t = 0; t < L; t++) {
            for (int h = 0; h < NUM_ATTN_HEADS; h++)
                rms_norm(qg.data() + (size_t)t * 2 * DQ + h * HEAD_DIM,
                         w_.q_norm.data(), HEAD_DIM,
                         qh.data() + (size_t)t * DQ + h * HEAD_DIM);
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

        // ---- append to the KV cache ---------------------------------------
        const int base = cache_len_;
        for (int t = 0; t < L; t++) {
            std::memcpy(cache_k_.data() + (size_t)(base + t) * DKV,
                        kk.data() + (size_t)t * DKV, DKV * sizeof(float));
            std::memcpy(cache_v_.data() + (size_t)(base + t) * DKV,
                        vv.data() + (size_t)t * DKV, DKV * sizeof(float));
        }
        cache_len_ += L;

        // ---- causal GQA softmax attention ---------------------------------
        std::vector<float> attn(LD * DQ);
#pragma omp parallel for collapse(2) schedule(static)
        for (int t = 0; t < L; t++) {
            for (int h = 0; h < NUM_ATTN_HEADS; h++) {
                const int    kvh  = h / GQA_GROUP;
                const int    nkey = base + t + 1;          // causal horizon
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
                    const float* vv2 = cache_v_.data() + (size_t)j * DKV + kvh * HEAD_DIM;
                    for (int d = 0; d < HEAD_DIM; d++) out[d] += p * vv2[d];
                }
            }
        }

        // ---- output gate, o_proj, residual --------------------------------
        for (int t = 0; t < L; t++) {
            const float* g = qg.data() + (size_t)t * 2 * DQ + DQ;
            float* a = attn.data() + (size_t)t * DQ;
            for (int i = 0; i < DQ; i++) a[i] *= sigmoidf_(g[i]);
        }
        std::vector<float> h1(LD * MODEL_DIM);
        q4k_matmul(w_.o, attn.data(), L, h1.data());
        for (size_t i = 0; i < LD * MODEL_DIM; i++) h1[i] += x[i];

        mlp_residual(h1.data(), L, y);
    }

private:
    /// Partial rotary: dims [0,64) only, rotate_half with halves 0..31 / 32..63.
    /// cos/sin are the dumped 64-wide vectors whose two halves are identical.
    static void rope(float* v, const float* cos, const float* sin) {
        float r[ROPE_DIM];
        const int H = ROPE_DIM / 2;
        for (int j = 0; j < H; j++)     r[j]     = v[j]     * cos[j]     - v[j + H] * sin[j];
        for (int j = H; j < ROPE_DIM; j++) r[j]  = v[j]     * cos[j]     + v[j - H] * sin[j];
        std::memcpy(v, r, sizeof r);
    }

    void mlp_residual(const float* h1, int L, float* y) {
        const size_t LD = static_cast<size_t>(L);
        std::vector<float> hn(LD * MODEL_DIM);
        for (int t = 0; t < L; t++)
            rms_norm(h1 + t * (size_t)MODEL_DIM, w_.post_ln.data(), MODEL_DIM,
                     hn.data() + t * (size_t)MODEL_DIM);
        std::vector<float> g(LD * INTERMEDIATE_SIZE), u(LD * INTERMEDIATE_SIZE);
        q4k_matmul(w_.mlp_gate, hn.data(), L, g.data());
        q4k_matmul(w_.mlp_up,   hn.data(), L, u.data());
        for (size_t i = 0; i < LD * INTERMEDIATE_SIZE; i++) g[i] = siluf_(g[i]) * u[i];
        q4k_matmul(w_.mlp_down, g.data(), L, y);
        for (size_t i = 0; i < LD * MODEL_DIM; i++) y[i] += h1[i];
    }

    int                layer_, max_len_, cache_len_ = 0;
    attention_weights  w_;
    std::vector<float> cache_k_, cache_v_;   // [max_len][1024]
};

} // namespace cpu_ref

// ==========================================================================
// driver
//
// The host Makefile globs every host/*.cpp into run.exe, so this standalone
// entry point is opt-in. Build it on its own with:
//   g++ -O3 -march=native -fopenmp -std=c++17 -DCPU_REF_STANDALONE -Ihost
//       host/cpu_ref_attention_layer.cpp -o cpu_ref_attention_layer
// It covers the whole layer, MLP included, and scores against the transformers
// dump; the bench in host.cpp is a different thing -- the attention half only,
// scored against the NPU. The delta-net reference lives in the other xclbin
// project, ../../gate_delta_net_verify/host/cpu_ref_delta_net_layer.cpp.
// ==========================================================================
#ifdef CPU_REF_STANDALONE
int main(int argc, char** argv) {
    using namespace cpu_ref;
    using namespace cpu_ref::q38;
    using clk = std::chrono::steady_clock;

    const std::string q4nx_path = argc > 1 ? argv[1]
        : "/path/to/models/Qwen3.8-27B-NPU2/model.q4nx";
    const std::string ref_path  = argc > 2 ? argv[2]
        : "/path/to/models/Qwen3.8-27B-NPU2/ref.safetensors";
    const int layer = argc > 3 ? std::atoi(argv[3]) : 3;

    if (!is_full_attention(layer)) {
        std::fprintf(stderr, "layer %d is delta-net; use ../../gate_delta_net_verify\n", layer);
        return 1;
    }

    safetensors q4nx(q4nx_path);
    safetensors ref(ref_path);

    char key[64];
    std::snprintf(key, sizeof key, "prefill.layer_%02d_out", layer);
    const std::string pref_out_key = key;
    std::snprintf(key, sizeof key, "verify.layer_%02d_out", layer);
    const std::string ver_out_key = key;
    std::snprintf(key, sizeof key, "prefill.layer_%02d_out", layer - 1);
    const std::string pref_in_key = layer == 0 ? "prefill.inputs_embeds" : key;
    std::snprintf(key, sizeof key, "verify.layer_%02d_out", layer - 1);
    const std::string ver_in_key = layer == 0 ? "verify.inputs_embeds" : key;

    const int P = static_cast<int>(ref.info(pref_in_key).shape[1]);
    const int V = static_cast<int>(ref.info(ver_in_key).shape[1]);

    std::printf("== Qwen3.8-27B full-attention layer %d ==\n", layer);
    std::printf("   prefill %d tokens, verify %d tokens, weights Q4_K 32x256\n\n", P, V);

    std::vector<float> px  = ref.f32(pref_in_key);
    std::vector<float> pgt = ref.f32(pref_out_key);
    std::vector<float> vx  = ref.f32(ver_in_key);
    std::vector<float> vgt = ref.f32(ver_out_key);
    std::vector<float> pcos = ref.f32("prefill.cos"), psin = ref.f32("prefill.sin");
    std::vector<float> vcos = ref.f32("verify.cos"),  vsin = ref.f32("verify.sin");

    attention_layer L(q4nx, layer, P + V + 64);

    // ---- 1. prefill: fills KV rows [0, 221) ------------------------------
    std::vector<float> py(static_cast<size_t>(P) * MODEL_DIM);
    auto t0 = clk::now();
    L.forward(px.data(), P, pcos.data(), psin.data(), py.data());
    auto t1 = clk::now();
    std::printf("prefill %d tokens in %.2f s\n", P,
                std::chrono::duration<double>(t1 - t0).count());
    print_err("prefill.layer_out", error_matrix(py.data(), pgt.data(),
                                                (size_t)P * MODEL_DIM));

    // ---- 2. verify: 8 tokens against the 221-row cache -------------------
    std::printf("\nKV cache length before verify: %d rows\n", L.cache_len());
    std::vector<float> vy(static_cast<size_t>(V) * MODEL_DIM);
    t0 = clk::now();
    L.forward(vx.data(), V, vcos.data(), vsin.data(), vy.data());
    t1 = clk::now();
    std::printf("verify %d tokens in %.2f s (cache now %d rows)\n\n", V,
                std::chrono::duration<double>(t1 - t0).count(), L.cache_len());
    compare_rows("verify.layer_out per token", vy.data(), vgt.data(), V, MODEL_DIM);
    std::printf("\n");
    compare_residual_free("verify.layer_out minus residual (attn + MLP only)",
                          vy.data(), vgt.data(), vx.data(), V, MODEL_DIM);

    return 0;
}
#endif // CPU_REF_STANDALONE
