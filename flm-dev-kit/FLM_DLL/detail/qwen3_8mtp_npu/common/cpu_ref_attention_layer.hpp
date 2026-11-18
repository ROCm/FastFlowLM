// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file common/cpu_ref_attention_layer.hpp
/// \brief CPU golden reference for one gated full-attention block, layer i
///        where (i+1) % 4 == 0. Byte-identical to the scored reference.
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
#pragma once

#include "common/cpu_ref_common.hpp"
#include "common/host_par.hpp"   // q38npu::host_par_min, for the per-row stages

#include <chrono>
#include <algorithm>

namespace cpu_ref {

using namespace q38;

// ==========================================================================
// weights
// ==========================================================================
struct attention_weights {
    q4nx_tensor q;          // [12288][5120]  q rows then gate rows
    q4nx_tensor k;          // [ 1024][5120]
    q4nx_tensor v;          // [ 1024][5120]
    q4nx_tensor o;          // [ 5120][6144]
    q4nx_tensor mlp_gate;   // [17408][5120]
    q4nx_tensor mlp_up;     // [17408][5120]
    q4nx_tensor mlp_down;   // [ 5120][17408]

    std::vector<float> in_ln;    // [5120]
    std::vector<float> post_ln;  // [5120]
    std::vector<float> q_norm;   // [256]
    std::vector<float> k_norm;   // [256]

    void load(const safetensors& f, int layer) {
        const std::string p = "model.layers." + std::to_string(layer) + ".";
        q        = get_q4nx(f, p + "self_attn.q_proj.weight");
        k        = get_q4nx(f, p + "self_attn.k_proj.weight");
        v        = get_q4nx(f, p + "self_attn.v_proj.weight");
        o        = get_q4nx(f, p + "self_attn.o_proj.weight");
        mlp_gate = get_q4nx(f, p + "mlp.gate_proj.weight");
        mlp_up   = get_q4nx(f, p + "mlp.up_proj.weight");
        mlp_down = get_q4nx(f, p + "mlp.down_proj.weight");

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

    /// Drop every cached row past `n`. A KV cache rolls back in O(1) -- rows
    /// above the cursor are never read again, so there is nothing to clear.
    /// Added for speculative decode; not part of the scored reference.
    void truncate(int n) {
        if (n < 0 || n > cache_len_)
            throw std::runtime_error("attention_layer::truncate out of range");
        cache_len_ = n;
    }

    /// Overwrite rows [base, base+n) of the cache without moving the cursor.
    ///
    /// Install `n` rows of pre-computed k and v as this layer's cache.
    ///
    /// The counterpart of npu_attn_layer::mirror_from_cpu, and its reason for
    /// existing is the same: a test that wants both implementations to start
    /// from one identical context, including a deliberately perturbed one that
    /// neither would produce on its own. The rows are taken post-RoPE and
    /// post-norm -- cache_k() layout, not raw projections.
    ///
    /// Not part of the scored reference; forward() never calls it.
    void seed_cache(const std::vector<float>& k, const std::vector<float>& v, int n) {
        if (n < 0 || n > max_len_)
            throw std::runtime_error("attention_layer::seed_cache out of range");
        const size_t want = (size_t)n * DKV;
        if (k.size() < want || v.size() < want)
            throw std::runtime_error("attention_layer::seed_cache: short k or v");
        std::memcpy(cache_k_.data(), k.data(), want * sizeof(float));
        std::memcpy(cache_v_.data(), v.data(), want * sizeof(float));
        cache_len_ = n;
    }

    /// x, y: [L][5120]. cos/sin: [L][64] for the L new positions.
    /// Appends L rows to the KV cache; queries see keys [0, cache_len+t].
    ///
    /// The body is three helpers with the cache append between them. They are
    /// public because the prefill path replaces the MIDDLE one -- the causal
    /// GQA softmax -- with `attn.xclbin`, and calls the other two unchanged.
    /// Splitting rather than copying is the point: the reference a hybrid path
    /// is scored against must be the same code, not a second transcription of
    /// it that can drift.
    void forward(const float* x, int L, const float* cos, const float* sin, float* y) {
        if (cache_len_ + L > max_len_) throw std::runtime_error("KV cache overflow");

        std::vector<float> qg, kk, vv, qh;
        project_qkv_rope(x, L, cos, sin, qg, kk, vv, qh);

        // ---- append to the KV cache ---------------------------------------
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

    /// RMSNorm, the q|gate / k / v projections, the per-head q/k RMSNorm and
    /// partial RoPE. Everything before the cache append.
    ///
    /// \param qg  out, [L][2*DQ]: q then the output gate, pre-head-norm
    /// \param kk  out, [L][DKV]: post-norm, post-RoPE -- cache rows
    /// \param vv  out, [L][DKV]: the projection, untouched
    /// \param qh  out, [L][DQ]: post-norm, post-RoPE queries
    void project_qkv_rope(const float* x, int L, const float* cos, const float* sin,
                          std::vector<float>& qg, std::vector<float>& kk,
                          std::vector<float>& vv, std::vector<float>& qh) const {
        const size_t LD = static_cast<size_t>(L);

        std::vector<float> xn;
        in_norm(x, L, xn);

        // ---- projections -------------------------------------------------
        qg.assign(LD * 2 * DQ, 0.0f);                // [q 6144 | gate 6144]
        kk.assign(LD * DKV, 0.0f);
        vv.assign(LD * DKV, 0.0f);
        q4nx_matmul(w_.q, xn.data(), L, qg.data());
        q4nx_matmul(w_.k, xn.data(), L, kk.data());
        q4nx_matmul(w_.v, xn.data(), L, vv.data());

        // ---- per-head q/k RMSNorm, then partial RoPE ----------------------
        qkv_post_rope(qg.data(), kk.data(), L, cos, sin, qh);
    }

    /// The causal GQA softmax: `o = softmax(q K^T * ATTN_SCALE + mask) V`.
    ///
    /// **This is the block `attn.xclbin` replaces.** It reads the cache, which
    /// the caller has already appended this window's rows to, so `base` is the
    /// cursor BEFORE the append: query `t` attends to keys [0, base + t].
    ///
    /// \param qh    [L][DQ], post-norm post-RoPE queries
    /// \param base  the pre-append cursor
    /// \param attn  out, [L][DQ], ungated
    void attn_core(const float* qh, int L, int base, std::vector<float>& attn) const {
        const size_t LD = static_cast<size_t>(L);
        attn.assign(LD * DQ, 0.0f);
        // Hand-flattened rather than collapse(2): MSVC's /openmp is OpenMP 2.0,
        // which has no collapse clause. Same iteration space -- every (t, h) is
        // independent -- so the results are unchanged.
        const int TH = L * NUM_ATTN_HEADS;
#pragma omp parallel for schedule(static)
        for (int th = 0; th < TH; th++) {
            const int t = th / NUM_ATTN_HEADS;
            const int h = th % NUM_ATTN_HEADS;
            {   // the old inner `for (h)` body, brace kept so the diff is small
                const int    kvh  = h / GQA_GROUP;
                const int    nkey = base + t + 1;          // causal horizon
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
                    const float* vv2 = cache_v_.data() + (size_t)j * DKV + kvh * HEAD_DIM;
                    for (int d = 0; d < HEAD_DIM; d++) out[d] += p * vv2[d];
                }
            }
        }
    }

    /// The output gate, o_proj with its residual, and the MLP block.
    /// Everything after the attention core.
    ///
    /// `attn` is modified in place by the gate -- it is scratch by the time
    /// this is called, and the caller owns it.
    ///
    /// \param qg    [L][2*DQ]; only the gate half, at offset DQ, is read
    /// \param attn  [L][DQ], ungated in, gated out
    /// \param x     [L][MODEL_DIM], the block input, for the residual
    void gate_out_mlp(const float* qg, float* attn, const float* x, int L, float* y) const {
        const size_t LD = static_cast<size_t>(L);
        apply_gate(qg, attn, L);

        std::vector<float> h1(LD * MODEL_DIM);
        q4nx_matmul(w_.o, attn, L, h1.data());
        for (size_t i = 0; i < LD * MODEL_DIM; i++) h1[i] += x[i];

        mlp_residual(h1.data(), L, y);
    }

    // ---- the same three helpers, split once more along every matmul --------
    /// The three above, split at each `q4nx_matmul` boundary, so that a backend
    /// can replace the *projections* and keep the elementwise work
    /// byte-identical. Same contract and same reason as the split above: the
    /// three are re-expressed in terms of these, so every existing caller gets
    /// the same code and the NPU path cannot drift from the reference.
    ///
    /// The seven matmuls they bracket are q|z, k, v, o, up, gate and down; on
    /// `dequant_mm.xclbin` those become four dispatches, because the staged
    /// q|k|v|z is one fused matrix and the woven up|gate another. See
    /// prefill/npu/prefill_proj.hpp.
    ///
    /// Mirrors cpu_ref::mtp_head's own finer split, function for function.

    /// `input_layernorm` alone. \param xn [L][MODEL_DIM] out, resized here
    ///
    /// `resize`, not `assign`, here and in the two stages below: each writes all
    /// of its output before anything reads it, so the fill is a memset that is
    /// overwritten byte for byte. Across a prefill window the three are
    /// 16 layers x L x (5120 + 6144 + 5120) floats -- 1.1 GB at L = 1024 -- and
    /// after the first layer the `resize` is a no-op, since the vector is
    /// already long enough.
    ///
    /// ## Threading, here and in the three stages below
    ///
    /// Every one of them is a loop over `t` whose iterations touch disjoint
    /// output rows and share only read-only weights, so the `parallel for` is a
    /// pure annotation -- no reduction, no ordering, and the fp32 arithmetic per
    /// element is unchanged, which is what keeps the reference bit-identical to
    /// what it was. A check that compares against dumps would
    /// see a single changed ulp.
    ///
    /// They are threaded because, unlike the marshalling loops in
    /// `common/cpu_simd.hpp`, there was no vectorisation left to win: gcc
    /// already does the inner `MODEL_DIM` loops, so one core is near its
    /// ceiling and the other 31 were idle. At L = 1400 these four stages are
    /// 16 layers x 1400 x (5120 + 6144 + 6144 + 5120) rows-worth of elements.
    ///
    /// The gate is `q38npu::host_par_min()`, the same one the dispatch
    /// marshalling uses, for the same two reasons: a spinning libgomp barrier
    /// makes threading a wash on this path, and the draft head calls these at
    /// L = 1..4 where a fork costs more than the loop.
    void in_norm(const float* x, int L, std::vector<float>& xn) const {
        xn.resize((size_t)L * MODEL_DIM);
#pragma omp parallel for schedule(static) \
    if ((size_t)L * MODEL_DIM >= q38npu::host_par_min())
        for (int t = 0; t < L; t++)
            rms_norm(x + t * (size_t)MODEL_DIM, w_.in_ln.data(), MODEL_DIM,
                     xn.data() + t * (size_t)MODEL_DIM);
    }

    /// `project_qkv_rope` after its three matmuls: per-head q/k RMSNorm then
    /// partial RoPE. v is untouched by both, so it is not a parameter.
    ///
    /// \param qg  [L][2*DQ] q then gate, read only
    /// \param kk  [L][DKV] normed and rotated IN PLACE -- cache rows on the way out
    /// \param qh  [L][DQ] out, head-major, resized here
    void qkv_post_rope(const float* qg, float* kk, int L,
                       const float* cos, const float* sin,
                       std::vector<float>& qh) const {
        // DQ == NUM_ATTN_HEADS * HEAD_DIM, so the per-head rms_norm below
        // covers every element of every row.
        qh.resize((size_t)L * DQ);
        // `kk` is normed and rotated in place, but strictly within row `t` --
        // no iteration reads another's row, so this is as independent as the
        // three stages that write a fresh output.
#pragma omp parallel for schedule(static) \
    if ((size_t)L * DQ >= q38npu::host_par_min())
        for (int t = 0; t < L; t++) {
            for (int h = 0; h < NUM_ATTN_HEADS; h++)
                rms_norm(qg + (size_t)t * 2 * DQ + h * HEAD_DIM,
                         w_.q_norm.data(), HEAD_DIM,
                         qh.data() + (size_t)t * DQ + h * HEAD_DIM);
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

    /// The sigmoid output gate. \param attn [L][DQ] gated in place
    void apply_gate(const float* qg, float* attn, int L) const {
#pragma omp parallel for schedule(static) \
    if ((size_t)L * DQ >= q38npu::host_par_min())
        for (int t = 0; t < L; t++) {
            const float* g = qg + (size_t)t * 2 * DQ + DQ;
            float* a = attn + (size_t)t * DQ;
            for (int i = 0; i < DQ; i++) a[i] *= sigmoidf_(g[i]);
        }
    }

    /// `post_attention_layernorm` alone. \param hn [L][MODEL_DIM] out, resized here
    void post_norm(const float* h1, int L, std::vector<float>& hn) const {
        hn.resize((size_t)L * MODEL_DIM);
#pragma omp parallel for schedule(static) \
    if ((size_t)L * MODEL_DIM >= q38npu::host_par_min())
        for (int t = 0; t < L; t++)
            rms_norm(h1 + t * (size_t)MODEL_DIM, w_.post_ln.data(), MODEL_DIM,
                     hn.data() + t * (size_t)MODEL_DIM);
    }

    /// `silu(gate) * up`, elementwise. May alias `gate`.
    /// \param g [L][INTERMEDIATE_SIZE] out
    ///
    /// Driven a row at a time so the `parallel for` has an `int` induction
    /// variable and each iteration owns a disjoint span; the aliasing the
    /// contract permits is `g == gate`, which is elementwise and therefore
    /// stays inside one row.
    void swiglu(const float* gate, const float* up, int L, float* g) const {
        constexpr size_t DI = (size_t)INTERMEDIATE_SIZE;
#pragma omp parallel for schedule(static) \
    if ((size_t)L * DI >= q38npu::host_par_min())
        for (int t = 0; t < L; t++) {
            const size_t o = (size_t)t * DI;
            for (size_t i = 0; i < DI; i++) g[o + i] = siluf_(gate[o + i]) * up[o + i];
        }
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

    void mlp_residual(const float* h1, int L, float* y) const {
        const size_t LD = static_cast<size_t>(L);
        std::vector<float> hn;
        post_norm(h1, L, hn);
        std::vector<float> g(LD * INTERMEDIATE_SIZE), u(LD * INTERMEDIATE_SIZE);
        q4nx_matmul(w_.mlp_gate, hn.data(), L, g.data());
        q4nx_matmul(w_.mlp_up,   hn.data(), L, u.data());
        swiglu(g.data(), u.data(), L, g.data());
        q4nx_matmul(w_.mlp_down, g.data(), L, y);
        for (size_t i = 0; i < LD * MODEL_DIM; i++) y[i] += h1[i];
    }

    int                layer_, max_len_, cache_len_ = 0;
    attention_weights  w_;
    std::vector<float> cache_k_, cache_v_;   // [max_len][1024]
};

} // namespace cpu_ref
