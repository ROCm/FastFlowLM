// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file qwen3_asr_audio_encoder.cpp
/// \brief Host implementation of the Qwen3-ASR / R2T2 audio encoder (see the header for the numerics contract).
#include "qwen3_asr_audio_encoder.hpp"

#include <immintrin.h>
#include <omp.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <unordered_map>

#include "nlohmann/json.hpp"
#include "../file_map.hpp"
#include "../omp_threads.hpp"

// ----------------------------------------------------------------------------------------------------------------
// bf16 helpers (activations are fp32 values that are exactly bf16-representable)
// ----------------------------------------------------------------------------------------------------------------
static inline float rbf(float x) {          // round to nearest even bf16, return as float
    uint32_t u;
    std::memcpy(&u, &x, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    u &= 0xffff0000u;
    std::memcpy(&x, &u, 4);
    return x;
}
static inline float bf2f(asr_bf16_bits b) {
    uint32_t u = (uint32_t)b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
static inline asr_bf16_bits f2bf(float x) {  // x must already be bf16-representable or will be rounded RNE
    uint32_t u;
    std::memcpy(&u, &x, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (asr_bf16_bits)(u >> 16);
}
static inline float gelu_erf(float x) { return 0.5f * x * (1.0f + std::erf(x * 0.70710678118654752440f)); }

static inline int round_up(int x, int m) { return (x + m - 1) / m * m; }

int qwen3_asr_audio_tokens(int L) {
    int leave = L % 100;
    int tail = 0;
    if (leave > 0) {
        int feat = (leave - 1) / 2 + 1;
        tail = ((feat - 1) / 2 + 1 - 1) / 2 + 1;
    }
    return tail + (L / 100) * 13;
}

// ----------------------------------------------------------------------------------------------------------------
// safetensors reader (BF16 only needed here)
// ----------------------------------------------------------------------------------------------------------------
namespace {
struct st_tensor {
    std::vector<int64_t> shape;
    const asr_bf16_bits* data;
};

struct safetensors_file {
    qwen3_asr_cpu::file_map file;
    std::unordered_map<std::string, st_tensor> tensors;

    explicit safetensors_file(const std::string& path) : file(path) {
        const char* map = (const char*)file.data();
        uint64_t n;
        std::memcpy(&n, map, 8);
        auto hdr = nlohmann::json::parse(std::string(map + 8, n));
        const char* base = map + 8 + n;
        for (auto it = hdr.begin(); it != hdr.end(); ++it) {
            if (it.key() == "__metadata__") continue;
            if (it.value()["dtype"] != "BF16") throw std::runtime_error("expected BF16 tensor: " + it.key());
            st_tensor t;
            t.shape = it.value()["shape"].get<std::vector<int64_t>>();
            auto off = it.value()["data_offsets"].get<std::vector<size_t>>();
            t.data = (const asr_bf16_bits*)(base + off[0]);
            tensors[it.key()] = t;
        }
    }
    const st_tensor& get(const std::string& name) const {
        auto it = tensors.find(name);
        if (it == tensors.end()) throw std::runtime_error("missing tensor " + name);
        return it->second;
    }
};

std::vector<float> to_f32(const st_tensor& t) {
    size_t n = 1;
    for (auto d : t.shape) n *= (size_t)d;
    std::vector<float> v(n);
    for (size_t i = 0; i < n; i++) v[i] = bf2f(t.data[i]);
    return v;
}

/// Invert FLM_Q4NX_Converter's _multi_modal_mm_weight_rearrange(K=512, N=64): stored [Np/64][Kp/512][64*512] with
/// element (n, k) of the (zero padded to a multiple of 512) matrix at ((n/64)*(Kp/512) + k/512)*32768 + (n%64)*512 + k%512.
/// Returns the plain row-major [N][K] fp32 matrix (padding dropped).
std::vector<float> unpermute_matrix(const st_tensor& t, int N, int K) {
    const int mm_n = 64, mm_k = 512;
    if (t.shape.size() != 3 || t.shape[2] != mm_n * mm_k) throw std::runtime_error("unexpected MM-tile weight shape");
    const int kb = (int)t.shape[1];
    if (round_up(N, 512) / mm_n != t.shape[0] || round_up(K, 512) / mm_k != kb) throw std::runtime_error("MM-tile weight shape mismatch");
    std::vector<float> w((size_t)N * K);
#pragma omp parallel for num_threads(qwen3_asr_cpu::asr_threads())
    for (int n = 0; n < N; n++) {
        const asr_bf16_bits* blk_row = t.data + (size_t)(n / mm_n) * kb * (mm_n * mm_k) + (size_t)(n % mm_n) * mm_k;
        for (int k = 0; k < K; k++) w[(size_t)n * K + k] = bf2f(blk_row[(size_t)(k / mm_k) * (mm_n * mm_k) + (k % mm_k)]);
    }
    return w;
}

// ----------------------------------------------------------------------------------------------------------------
// dense kernels
// ----------------------------------------------------------------------------------------------------------------

/// A weight matrix W[N][K] (bf16 values) packed for vdpbf16ps: [N/16][K/2][16 outputs][2 k], so one 64-byte load holds
/// the k, k+1 pair of 16 outputs and a broadcast pair of the activations multiplies all of them at once.
struct packed_w {
    std::vector<asr_bf16_bits> d;
    int N = 0, K = 0;
};

packed_w pack_bf16(const std::vector<float>& W, int N, int K) {
    assert(N % 32 == 0 && K % 2 == 0);
    packed_w p;
    p.N = N;
    p.K = K;
    p.d.resize((size_t)N * K);
#pragma omp parallel for num_threads(qwen3_asr_cpu::asr_threads())
    for (int nb = 0; nb < N / 16; nb++)
        for (int kk = 0; kk < K / 2; kk++)
            for (int lane = 0; lane < 16; lane++)
                for (int j = 0; j < 2; j++)
                    p.d[(((size_t)nb * (K / 2) + kk) * 16 + lane) * 2 + j] = f2bf(W[(size_t)(nb * 16 + lane) * K + 2 * kk + j]);
    return p;
}

/// C[M][N] = rbf(A[M][K] . W[N][K]^T + bias[N]) (optionally followed by rbf(GELU)). A holds bf16 values (every activation of the
/// encoder is rounded to bf16, as in the reference), so the conversion is exact; products are exact in fp32, accumulation is fp32.
void gemm_bias(const float* A, int M, int K, const packed_w& W, const float* bias, int N, float* C, bool gelu) {
    assert(K == W.K && N == W.N);
    std::vector<asr_bf16_bits> Ab((size_t)M * K);
#pragma omp parallel for num_threads(qwen3_asr_cpu::asr_threads())
    for (int i = 0; i < M * K; i++) Ab[i] = f2bf(A[i]);

    constexpr int MT = 6;                              // 6 rows x 2 vectors of 16 outputs = 12 accumulators
    const int tiles = N / 32;
#pragma omp parallel for num_threads(qwen3_asr_cpu::asr_threads()) schedule(dynamic)
    for (int tile = 0; tile < tiles; tile++) {
        const int n0 = tile * 32;
        const asr_bf16_bits* w0 = W.d.data() + (size_t)(n0 / 16) * K * 16;
        const asr_bf16_bits* w1 = w0 + (size_t)K * 16;
        for (int m = 0; m < M; m += MT) {
            const int T = std::min(MT, M - m);
            const asr_bf16_bits* a[MT];
            for (int i = 0; i < MT; i++) a[i] = Ab.data() + (size_t)std::min(m + i, M - 1) * K;
            __m512 acc0[MT], acc1[MT];
            for (int i = 0; i < MT; i++) acc0[i] = acc1[i] = _mm512_setzero_ps();
            for (int kk = 0; kk < K / 2; kk++) {
                const __m512bh wv0 = (__m512bh)_mm512_loadu_si512(w0 + (size_t)kk * 32);
                const __m512bh wv1 = (__m512bh)_mm512_loadu_si512(w1 + (size_t)kk * 32);
                for (int i = 0; i < MT; i++) {
                    int32_t pair;
                    std::memcpy(&pair, a[i] + 2 * kk, 4);
                    const __m512bh av = (__m512bh)_mm512_set1_epi32(pair);
                    acc0[i] = _mm512_dpbf16_ps(acc0[i], av, wv0);
                    acc1[i] = _mm512_dpbf16_ps(acc1[i], av, wv1);
                }
            }
            for (int i = 0; i < T; i++) {
                float out[32];
                _mm512_storeu_ps(out, acc0[i]);
                _mm512_storeu_ps(out + 16, acc1[i]);
                float* c = C + (size_t)(m + i) * N + n0;
                for (int j = 0; j < 32; j++) {
                    float v = rbf(out[j] + (bias ? bias[n0 + j] : 0.f));
                    if (gelu) v = rbf(gelu_erf(v));
                    c[j] = v;
                }
            }
        }
    }
}

void layer_norm(const float* x, int rows, int D, const float* w, const float* b, float eps, float* y) {
#pragma omp parallel for num_threads(qwen3_asr_cpu::asr_threads())
    for (int r = 0; r < rows; r++) {
        const float* xr = x + (size_t)r * D;
        float* yr = y + (size_t)r * D;
        double s = 0, s2 = 0;
        for (int i = 0; i < D; i++) s += xr[i];
        const float mean = (float)(s / D);
        for (int i = 0; i < D; i++) { float d = xr[i] - mean; s2 += (double)d * d; }
        const float rstd = 1.0f / std::sqrt((float)(s2 / D) + eps);
        for (int i = 0; i < D; i++) yr[i] = rbf((xr[i] - mean) * rstd * w[i] + b[i]);
    }
}

struct layer_w {
    packed_w q, k, v, o, fc1, fc2;                          // [out][in]
    std::vector<float> qb, kb, vb, ob, fc1b, fc2b;
    std::vector<float> ln1w, ln1b, ln2w, ln2b;              // self_attn_layer_norm, final_layer_norm
};
}  // namespace

struct qwen3_asr_audio_encoder::Impl {
    // conv stack: weights [480][Kpad] in (kh, kw, cin) order, so an im2col row is a run of cin-contiguous copies
    packed_w conv_w[3];
    std::vector<float> conv_b[3];
    int conv_cin[3] = {1, 480, 480};
    packed_w conv_out_w;                         // [1024][7680]
    std::vector<float> pos;                      // [chunk_tokens][d_model], bf16 values
    std::vector<layer_w> layers;
    std::vector<float> ln_post_w, ln_post_b, proj1_b, proj2_b;
    packed_w proj1_w, proj2_w;
    // attention numerics switches (see README): scores / probabilities rounded to bf16 like the eager reference
    bool round_scores = false, round_probs = false;
};

namespace {

/// im2col for Conv2d(3x3, stride 2, pad 1) on an NHWC activation [H][W][cin]; out [Hout*Wout][kpad], row = (kh, kw, cin).
void im2col_s2(const float* in, int H, int W, int cin, int kpad, float* out) {
    const int Ho = (H + 1) / 2, Wo = (W + 1) / 2;
#pragma omp parallel for num_threads(qwen3_asr_cpu::asr_threads())
    for (int p = 0; p < Ho * Wo; p++) {
        const int ho = p / Wo, wo = p % Wo;
        float* row = out + (size_t)p * kpad;
        for (int kh = 0; kh < 3; kh++) {
            for (int kw = 0; kw < 3; kw++) {
                const int h = 2 * ho - 1 + kh, w = 2 * wo - 1 + kw;
                float* dst = row + (kh * 3 + kw) * cin;
                if (h < 0 || h >= H || w < 0 || w >= W) std::memset(dst, 0, sizeof(float) * cin);
                else std::memcpy(dst, in + ((size_t)h * W + w) * cin, sizeof(float) * cin);
            }
        }
        for (int i = 9 * cin; i < kpad; i++) row[i] = 0.f;
    }
}

}  // namespace

qwen3_asr_audio_encoder::qwen3_asr_audio_encoder(const std::string& path, const qwen3_asr_audio_config& cfg) : cfg_(cfg) {
    impl_ = new Impl();
    Impl& I = *impl_;
    if (const char* e = std::getenv("R2T2_ROUND_SCORES")) I.round_scores = std::atoi(e) != 0;
    if (const char* e = std::getenv("R2T2_ROUND_PROBS")) I.round_probs = std::atoi(e) != 0;
    safetensors_file f(path);
    const int D = cfg_.d_model, C = cfg_.downsample_hidden;

    // conv weights [Cout][Cin][3][3] -> [Cout][kpad] with k = (kh*3+kw)*Cin + cin
    for (int l = 0; l < 3; l++) {
        const auto& t = f.get("model.audio.conv2d" + std::to_string(l + 1) + ".weight");
        const int cin = I.conv_cin[l], kpad = round_up(9 * cin, 16);
        std::vector<float> w((size_t)C * kpad, 0.f);
        for (int co = 0; co < C; co++)
            for (int ci = 0; ci < cin; ci++)
                for (int k = 0; k < 9; k++) w[(size_t)co * kpad + (size_t)k * cin + ci] = bf2f(t.data[((size_t)co * cin + ci) * 9 + k]);
        I.conv_w[l] = pack_bf16(w, C, kpad);
        I.conv_b[l] = to_f32(f.get("model.audio.conv2d" + std::to_string(l + 1) + ".bias"));
    }
    const int conv_k = C * (cfg_.num_mel_bins / 8);   // 480 * 16 = 7680
    I.conv_out_w = pack_bf16(unpermute_matrix(f.get("model.audio.conv_out.weight"), D, conv_k), D, conv_k);

    // sinusoidal position table (built like SinusoidsPositionEmbedding, then cast to bf16 by the reference)
    I.pos.resize((size_t)cfg_.chunk_tokens() * D);
    {
        const double inc = std::log(10000.0) / (D / 2 - 1);
        for (int t = 0; t < cfg_.chunk_tokens(); t++)
            for (int i = 0; i < D / 2; i++) {
                const float inv = (float)std::exp(-inc * i);
                const float a = (float)t * inv;
                I.pos[(size_t)t * D + i] = rbf(std::sin(a));
                I.pos[(size_t)t * D + D / 2 + i] = rbf(std::cos(a));
            }
    }

    I.layers.resize(cfg_.layers);
    for (int l = 0; l < cfg_.layers; l++) {
        const std::string p = "model.audio.layers." + std::to_string(l) + ".";
        layer_w& L = I.layers[l];
        L.q = pack_bf16(unpermute_matrix(f.get(p + "self_attn.q_proj.weight"), D, D), D, D);
        L.k = pack_bf16(unpermute_matrix(f.get(p + "self_attn.k_proj.weight"), D, D), D, D);
        L.v = pack_bf16(unpermute_matrix(f.get(p + "self_attn.v_proj.weight"), D, D), D, D);
        L.o = pack_bf16(unpermute_matrix(f.get(p + "self_attn.out_proj.weight"), D, D), D, D);
        L.fc1 = pack_bf16(unpermute_matrix(f.get(p + "fc1.weight"), cfg_.ffn_dim, D), cfg_.ffn_dim, D);
        L.fc2 = pack_bf16(unpermute_matrix(f.get(p + "fc2.weight"), D, cfg_.ffn_dim), D, cfg_.ffn_dim);
        L.qb = to_f32(f.get(p + "self_attn.q_proj.bias"));
        L.kb = to_f32(f.get(p + "self_attn.k_proj.bias"));
        L.vb = to_f32(f.get(p + "self_attn.v_proj.bias"));
        L.ob = to_f32(f.get(p + "self_attn.out_proj.bias"));
        L.fc1b = to_f32(f.get(p + "fc1.bias"));
        L.fc2b = to_f32(f.get(p + "fc2.bias"));
        L.ln1w = to_f32(f.get(p + "self_attn_layer_norm.weight"));
        L.ln1b = to_f32(f.get(p + "self_attn_layer_norm.bias"));
        L.ln2w = to_f32(f.get(p + "final_layer_norm.weight"));
        L.ln2b = to_f32(f.get(p + "final_layer_norm.bias"));
    }
    I.ln_post_w = to_f32(f.get("model.audio.ln_post.weight"));
    I.ln_post_b = to_f32(f.get("model.audio.ln_post.bias"));
    I.proj1_w = pack_bf16(unpermute_matrix(f.get("model.audio.tower.proj1.weight"), D, D), D, D);
    I.proj1_b = to_f32(f.get("model.audio.tower.proj1.bias"));
    I.proj2_w = pack_bf16(unpermute_matrix(f.get("model.audio.tower.proj2.weight"), cfg_.output_dim, D), cfg_.output_dim, D);
    I.proj2_b = to_f32(f.get("model.audio.tower.proj2.bias"));
}

qwen3_asr_audio_encoder::~qwen3_asr_audio_encoder() { delete impl_; }

void qwen3_asr_audio_encoder::conv_front(const float* mel_chunk, int valid_frames, float* out_tokens) {
    Impl& I = *impl_;
    const int C = cfg_.downsample_hidden, F = cfg_.num_mel_bins, W0 = cfg_.chunk_frames();
    // conv1 input as NHWC [H=128 (mel bin)][W=100 (time)][1]; frames beyond valid_frames are zero (reference pads the chunk)
    std::vector<float> in0((size_t)F * W0, 0.f);
    for (int t = 0; t < std::min(valid_frames, W0); t++)
        for (int h = 0; h < F; h++) in0[(size_t)h * W0 + t] = rbf(mel_chunk[(size_t)t * F + h]);   // mel is cast to bf16 first

    int H = F, W = W0;
    std::vector<float> cur, patches, nxt;
    const float* src = in0.data();
    for (int l = 0; l < 3; l++) {
        const int cin = I.conv_cin[l], kpad = round_up(9 * cin, 16), Ho = (H + 1) / 2, Wo = (W + 1) / 2;
        patches.resize((size_t)Ho * Wo * kpad);
        im2col_s2(src, H, W, cin, kpad, patches.data());
        nxt.resize((size_t)Ho * Wo * C);
        gemm_bias(patches.data(), Ho * Wo, kpad, I.conv_w[l], I.conv_b[l].data(), C, nxt.data(), /*gelu=*/true);
        cur.swap(nxt);
        src = cur.data();
        H = Ho;
        W = Wo;
    }
    // cur: [H=16 (f)][W=13 (t)][C]; token t = vector over (c, f) in c-major order: index c*16 + f
    const int Fq = H, T = W, K = C * Fq;
    std::vector<float> tok((size_t)T * K);
    for (int t = 0; t < T; t++)
        for (int c = 0; c < C; c++)
            for (int fq = 0; fq < Fq; fq++) tok[(size_t)t * K + (size_t)c * Fq + fq] = cur[((size_t)fq * T + t) * C + c];
    const int n_valid = qwen3_asr_audio_tokens(valid_frames);
    std::vector<float> full((size_t)T * cfg_.d_model);
    gemm_bias(tok.data(), T, K, I.conv_out_w, nullptr, cfg_.d_model, full.data(), false);
    std::memcpy(out_tokens, full.data(), sizeof(float) * (size_t)n_valid * cfg_.d_model);
}

void qwen3_asr_audio_encoder::add_pos(float* x, int tokens) {
    const int D = cfg_.d_model;
    for (int t = 0; t < tokens; t++)
        for (int i = 0; i < D; i++) x[(size_t)t * D + i] = rbf(x[(size_t)t * D + i] + impl_->pos[(size_t)t * D + i]);
}

void qwen3_asr_audio_encoder::encoder_layer(int layer, float* x, int n) {
    Impl& I = *impl_;
    const layer_w& L = I.layers[layer];
    const int D = cfg_.d_model, H = cfg_.heads, hd = D / H, FF = cfg_.ffn_dim, WT = cfg_.window_tokens();
    std::vector<float> a((size_t)n * D), q((size_t)n * D), k((size_t)n * D), v((size_t)n * D), o((size_t)n * D), t1((size_t)n * D);

    layer_norm(x, n, D, L.ln1w.data(), L.ln1b.data(), cfg_.ln_eps, a.data());
    gemm_bias(a.data(), n, D, L.q, L.qb.data(), D, q.data(), false);
    gemm_bias(a.data(), n, D, L.k, L.kb.data(), D, k.data(), false);
    gemm_bias(a.data(), n, D, L.v, L.vb.data(), D, v.data(), false);

    // window attention: tokens [w0, w1) only see each other
    const float scale = 1.0f / std::sqrt((float)hd);
    const int nwin = (n + WT - 1) / WT;
#pragma omp parallel for num_threads(qwen3_asr_cpu::asr_threads()) schedule(dynamic)
    for (int wh = 0; wh < nwin * H; wh++) {           // one (window, head) pair per iteration
        {
            const int w = wh / H, h = wh % H;
            const int w0 = w * WT, w1 = std::min(n, w0 + WT), m = w1 - w0;
            std::vector<float> s((size_t)m * m);
            for (int i = 0; i < m; i++) {
                const float* qi = q.data() + (size_t)(w0 + i) * D + h * hd;
                for (int j = 0; j < m; j++) {
                    const float* kj = k.data() + (size_t)(w0 + j) * D + h * hd;
                    float acc = 0.f;
                    for (int d = 0; d < hd; d++) acc += qi[d] * kj[d];
                    if (I.round_scores) acc = rbf(rbf(acc) * scale); else acc *= scale;
                    s[(size_t)i * m + j] = acc;
                }
                float mx = -INFINITY;
                for (int j = 0; j < m; j++) mx = std::max(mx, s[(size_t)i * m + j]);
                float sum = 0.f;
                for (int j = 0; j < m; j++) { float e = std::exp(s[(size_t)i * m + j] - mx); s[(size_t)i * m + j] = e; sum += e; }
                for (int j = 0; j < m; j++) { float p = s[(size_t)i * m + j] / sum; s[(size_t)i * m + j] = I.round_probs ? rbf(p) : p; }
                float* oi = o.data() + (size_t)(w0 + i) * D + h * hd;
                for (int d = 0; d < hd; d++) oi[d] = 0.f;
                for (int j = 0; j < m; j++) {
                    const float p = s[(size_t)i * m + j];
                    const float* vj = v.data() + (size_t)(w0 + j) * D + h * hd;
                    for (int d = 0; d < hd; d++) oi[d] += p * vj[d];
                }
                for (int d = 0; d < hd; d++) oi[d] = rbf(oi[d]);
            }
        }
    }
    gemm_bias(o.data(), n, D, L.o, L.ob.data(), D, t1.data(), false);
    for (size_t i = 0; i < (size_t)n * D; i++) x[i] = rbf(x[i] + t1[i]);

    layer_norm(x, n, D, L.ln2w.data(), L.ln2b.data(), cfg_.ln_eps, a.data());
    std::vector<float> ff((size_t)n * FF);
    gemm_bias(a.data(), n, D, L.fc1, L.fc1b.data(), FF, ff.data(), /*gelu=*/true);
    gemm_bias(ff.data(), n, FF, L.fc2, L.fc2b.data(), D, t1.data(), false);
    for (size_t i = 0; i < (size_t)n * D; i++) x[i] = rbf(x[i] + t1[i]);
}

void qwen3_asr_audio_encoder::post(const float* x, int n, float* out) {
    Impl& I = *impl_;
    const int D = cfg_.d_model;
    std::vector<float> a((size_t)n * D), b((size_t)n * D);
    layer_norm(x, n, D, I.ln_post_w.data(), I.ln_post_b.data(), cfg_.ln_eps, a.data());
    gemm_bias(a.data(), n, D, I.proj1_w, I.proj1_b.data(), D, b.data(), /*gelu=*/true);
    gemm_bias(b.data(), n, D, I.proj2_w, I.proj2_b.data(), cfg_.output_dim, out, false);
}

std::vector<asr_bf16_bits> qwen3_asr_audio_encoder::encode(const float* mel, int frames, conv_cache* cache) {
    const int D = cfg_.d_model, CF = cfg_.chunk_frames();
    const int n_tokens = qwen3_asr_audio_tokens(frames);
    const int nchunks = (frames + CF - 1) / CF;
    std::vector<float> x((size_t)n_tokens * D);
    int off = 0;
    for (int c = 0; c < nchunks; c++) {
        const int valid = std::min(CF, frames - c * CF);
        const int nt = qwen3_asr_audio_tokens(valid);
        float* rows = x.data() + (size_t)off * D;
        const float* chunk_mel = mel + (size_t)c * CF * cfg_.num_mel_bins;
        const size_t mel_floats = (size_t)CF * cfg_.num_mel_bins;
        if (cache && c < cache->chunks && std::memcmp(cache->mel.data() + (size_t)c * mel_floats, chunk_mel, sizeof(float) * mel_floats) != 0)
            cache->truncate(c, cfg_.chunk_tokens() * D, (int)mel_floats);   // its mel changed (the clip's clamp moved): redo it and the rest
        if (cache && c < cache->chunks) {   // a finished chunk seen by an earlier call
            std::memcpy(rows, cache->rows.data() + (size_t)c * cfg_.chunk_tokens() * D, sizeof(float) * (size_t)nt * D);
        } else {
            conv_front(chunk_mel, valid, rows);
            add_pos(rows, nt);
            if (cache && c == cache->chunks && valid == CF) {
                cache->rows.insert(cache->rows.end(), rows, rows + (size_t)nt * D);
                cache->mel.insert(cache->mel.end(), chunk_mel, chunk_mel + mel_floats);
                cache->chunks++;
            }
        }
        off += nt;
    }
    for (int l = 0; l < cfg_.layers; l++) encoder_layer(l, x.data(), n_tokens);
    std::vector<float> y((size_t)n_tokens * cfg_.output_dim);
    post(x.data(), n_tokens, y.data());
    std::vector<asr_bf16_bits> out(y.size());
    for (size_t i = 0; i < y.size(); i++) out[i] = f2bf(y[i]);
    return out;
}
