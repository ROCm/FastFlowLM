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
#include <mutex>
#include <stdexcept>
#include <unordered_map>

#include "nlohmann/json.hpp"
#include "../file_map.hpp"
#include "../omp_threads.hpp"
#include "qwen3_asr_audio_simd.hpp"

namespace simd = qwen3_asr_audio_simd;

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
                float* c = C + (size_t)(m + i) * N + n0;
                for (int h = 0; h < 2; h++) {
                    __m512 v = h ? acc1[i] : acc0[i];
                    if (bias) v = _mm512_add_ps(v, _mm512_loadu_ps(bias + n0 + 16 * h));
                    v = simd::round_bf16(v);
                    if (gelu) v = simd::round_bf16(simd::gelu_erf(v));
                    _mm512_storeu_ps(c + 16 * h, v);
                }
            }
        }
    }
}

/// D: a multiple of 16
void layer_norm(const float* x, int rows, int D, const float* w, const float* b, float eps, float* y) {
    assert(D % 16 == 0);
#pragma omp parallel for num_threads(qwen3_asr_cpu::asr_threads())
    for (int r = 0; r < rows; r++) {
        const float* xr = x + (size_t)r * D;
        float* yr = y + (size_t)r * D;
        // statistics in double: on quiet input a row is a small signal over a large offset, and fp32 sums lose it
        __m512d s0 = _mm512_setzero_pd(), s1 = _mm512_setzero_pd();
        for (int i = 0; i < D; i += 16) {
            s0 = _mm512_add_pd(s0, _mm512_cvtps_pd(_mm256_loadu_ps(xr + i)));
            s1 = _mm512_add_pd(s1, _mm512_cvtps_pd(_mm256_loadu_ps(xr + i + 8)));
        }
        const __m512 mean = _mm512_set1_ps((float)(_mm512_reduce_add_pd(_mm512_add_pd(s0, s1)) / D));
        s0 = s1 = _mm512_setzero_pd();
        for (int i = 0; i < D; i += 16) {
            const __m512 d = _mm512_sub_ps(_mm512_loadu_ps(xr + i), mean);
            const __m512d d0 = _mm512_cvtps_pd(_mm512_castps512_ps256(d)), d1 = _mm512_cvtps_pd(_mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(d), 1)));
            s0 = _mm512_fmadd_pd(d0, d0, s0);
            s1 = _mm512_fmadd_pd(d1, d1, s1);
        }
        const __m512 rstd = _mm512_set1_ps(1.0f / std::sqrt((float)(_mm512_reduce_add_pd(_mm512_add_pd(s0, s1)) / D) + eps));
        for (int i = 0; i < D; i += 16) {
            const __m512 d = _mm512_mul_ps(_mm512_sub_ps(_mm512_loadu_ps(xr + i), mean), rstd);
            _mm512_storeu_ps(yr + i, simd::round_bf16(_mm512_fmadd_ps(d, _mm512_loadu_ps(w + i), _mm512_loadu_ps(b + i))));
        }
    }
}

/// x = rbf(x + t), n a multiple of 16
void add_residual(float* x, const float* t, size_t n) {
#pragma omp parallel for num_threads(qwen3_asr_cpu::asr_threads())
    for (size_t i = 0; i < n; i += 16) _mm512_storeu_ps(x + i, simd::round_bf16(_mm512_add_ps(_mm512_loadu_ps(x + i), _mm512_loadu_ps(t + i))));
}

/// one (window, head) of attention: o[i] = rbf(softmax_j(q[i] . k[j] * scale) v[j]) for i, j < m. q, k, v rows have stride rs,
/// o rows stride os; hd a multiple of 16. K is transposed first (zero past m), so a row of scores is hd broadcast FMAs per 16 keys.
void window_attention(const float* q, const float* k, const float* v, size_t rs, int m, int hd, float scale, bool round_scores,
                      bool round_probs, float* o, size_t os, std::vector<float>& kt, std::vector<float>& s) {
    constexpr int G = 8;   // key vectors per pass (registers)
    const int mv = (m + 15) / 16, ld = mv * 16;
    kt.assign((size_t)hd * ld, 0.f);
    s.resize(ld);
    for (int j = 0; j < m; j++)
        for (int d = 0; d < hd; d++) kt[(size_t)d * ld + j] = k[(size_t)j * rs + d];
    const __mmask16 tail = (__mmask16)((1u << (m - (mv - 1) * 16)) - 1);
    const __m512 vscale = _mm512_set1_ps(scale), ninf = _mm512_set1_ps(-INFINITY);
    for (int i = 0; i < m; i++) {
        const float* qi = q + (size_t)i * rs;
        __m512 mx = ninf;
        for (int g0 = 0; g0 < mv; g0 += G) {
            const int gn = std::min(G, mv - g0);
            __m512 acc[G];
            for (int t = 0; t < G; t++) acc[t] = _mm512_setzero_ps();
            for (int d = 0; d < hd; d++) {
                const __m512 qd = _mm512_set1_ps(qi[d]);
                const float* kd = kt.data() + (size_t)d * ld + g0 * 16;
                for (int t = 0; t < gn; t++) acc[t] = _mm512_fmadd_ps(qd, _mm512_loadu_ps(kd + t * 16), acc[t]);
            }
            for (int t = 0; t < gn; t++) {
                __m512 a = round_scores ? simd::round_bf16(_mm512_mul_ps(simd::round_bf16(acc[t]), vscale)) : _mm512_mul_ps(acc[t], vscale);
                if (g0 + t == mv - 1) a = _mm512_mask_mov_ps(ninf, tail, a);
                _mm512_storeu_ps(s.data() + (g0 + t) * 16, a);
                mx = _mm512_max_ps(mx, a);
            }
        }
        const __m512 vmx = _mm512_set1_ps(_mm512_reduce_max_ps(mx));
        __m512 sum = _mm512_setzero_ps();
        for (int jv = 0; jv < mv; jv++) {
            __m512 e = simd::exp(_mm512_sub_ps(_mm512_loadu_ps(s.data() + jv * 16), vmx));
            if (jv == mv - 1) e = _mm512_maskz_mov_ps(tail, e);
            _mm512_storeu_ps(s.data() + jv * 16, e);
            sum = _mm512_add_ps(sum, e);
        }
        const __m512 vsum = _mm512_set1_ps(_mm512_reduce_add_ps(sum));
        for (int jv = 0; jv < mv; jv++) {
            __m512 p = _mm512_div_ps(_mm512_loadu_ps(s.data() + jv * 16), vsum);
            _mm512_storeu_ps(s.data() + jv * 16, round_probs ? simd::round_bf16(p) : p);
        }
        float* oi = o + (size_t)i * os;
        for (int d0 = 0; d0 < hd; d0 += 64) {
            const int dn = std::min(4, (hd - d0) / 16);
            __m512 acc[4] = {_mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps()};
            for (int j = 0; j < m; j++) {
                const __m512 p = _mm512_set1_ps(s[j]);
                const float* vj = v + (size_t)j * rs + d0;
                for (int t = 0; t < dn; t++) acc[t] = _mm512_fmadd_ps(p, _mm512_loadu_ps(vj + t * 16), acc[t]);
            }
            for (int t = 0; t < dn; t++) _mm512_storeu_ps(oi + d0 + t * 16, simd::round_bf16(acc[t]));
        }
    }
}

/// One projection: host-packed weights, or the id of the weights the gemm backend holds
struct linear {
    packed_w w;
    std::vector<float> b;
    int id = -1;
    int N = 0, K = 0;
};

struct layer_w {
    linear qkv, o, fc1, fc2;                                // [out][in]; qkv = q, k, v stacked (3 d_model outputs)
    std::vector<float> ln1w, ln1b, ln2w, ln2b;              // self_attn_layer_norm, final_layer_norm
};
}  // namespace

struct qwen3_asr_audio_encoder::Impl {
    qwen3_asr_audio_gemm* gemm = nullptr;
    // conv stack: weights [480][Kpad] in (kh, kw, cin) order, so an im2col row is a run of cin-contiguous copies
    linear conv[3];
    int conv_cin[3] = {1, 480, 480};
    linear conv_out;                             // [1024][7680]
    std::vector<float> pos;                      // [chunk_tokens][d_model], bf16 values
    std::vector<layer_w> layers;
    std::vector<float> ln_post_w, ln_post_b;
    linear proj1, proj2;
    // attention numerics switches (see README): scores / probabilities rounded to bf16 like the eager reference
    bool round_scores = false, round_probs = false;

    // conv front end scratch, kept between calls (conv_mu)
    std::mutex conv_mu;
    std::vector<float> in0, act[3], full, rows_buf;

    /// W [N][K]; on the backend when there is one and `host` is false. The backend keeps A and W at bf16 precision
    /// (exact_a, exact_w of qwen3_asr_audio_gemm::add_weight): plain bfp16 products drift the conv stack on quiet input.
    linear make(const std::vector<float>& W, int N, int K, std::vector<float> b, bool host = false) {
        linear L;
        L.N = N;
        L.K = K;
        if (gemm && !host) L.id = gemm->add_weight(W.data(), N, K, b.empty() ? nullptr : b.data(), true, true);
        else {
            L.w = pack_bf16(W, N, K);
            L.b = std::move(b);
        }
        return L;
    }
    void mm(const linear& L, const float* A, int M, float* C, bool gelu) {
        if (L.id >= 0) gemm->run(L.id, [A, K = L.K](int r, float*) { return A + (size_t)r * K; }, M, C, gelu);
        else gemm_bias(A, M, L.K, L.w, L.b.empty() ? nullptr : L.b.data(), L.N, C, gelu);
    }
    /// the same with A given row by row (the backend packs each row as it comes; the host gathers A first, into rows_buf)
    void mm_rows(const linear& L, const qwen3_asr_audio_gemm::rows_fn& A, int M, float* C, bool gelu) {
        if (L.id >= 0) return gemm->run(L.id, A, M, C, gelu);
        rows_buf.resize((size_t)M * L.K);
#pragma omp parallel for num_threads(qwen3_asr_cpu::asr_threads())
        for (int r = 0; r < M; r++) {
            float* dst = rows_buf.data() + (size_t)r * L.K;
            const float* p = A(r, dst);
            if (p != dst) std::memcpy(dst, p, sizeof(float) * L.K);
        }
        gemm_bias(rows_buf.data(), M, L.K, L.w, L.b.empty() ? nullptr : L.b.data(), L.N, C, gelu);
    }
    /// conv front end of `count` chunks at once (one product per conv over all of them): see conv_front()
    void conv_chunks(const qwen3_asr_audio_config& cfg, const float* const* mel, const int* valid, int count, float* const* out);
};

namespace {

/// im2col row p (= ho * Wout + wo) of Conv2d(3x3, stride 2, pad 1) on an NHWC activation [H][W][cin]: kpad values, (kh, kw, cin).
void im2col_row(const float* in, int H, int W, int cin, int kpad, int p, float* row) {
    const int Wo = (W + 1) / 2, ho = p / Wo, wo = p % Wo;
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

}  // namespace

void qwen3_asr_audio_gemm::run_conv(int weight, const float* in, int count, int H, int W, int cin, int K, float* C, bool gelu) {
    const int rows = (H + 1) / 2 * ((W + 1) / 2);   // per image
    run(weight, [=](int r, float* tmp) {
            im2col_row(in + (size_t)(r / rows) * H * W * cin, H, W, cin, K, r % rows, tmp);
            return (const float*)tmp;
        }, count * rows, C, gelu);
}

qwen3_asr_audio_encoder::qwen3_asr_audio_encoder(const std::string& path, const qwen3_asr_audio_config& cfg, qwen3_asr_audio_gemm* gemm)
    : cfg_(cfg) {
    impl_ = new Impl();
    Impl& I = *impl_;
    I.gemm = gemm;
    if (const char* e = std::getenv("R2T2_ROUND_SCORES")) I.round_scores = std::atoi(e) != 0;
    if (const char* e = std::getenv("R2T2_ROUND_PROBS")) I.round_probs = std::atoi(e) != 0;
    safetensors_file f(path);
    const int D = cfg_.d_model, C = cfg_.downsample_hidden;

    // conv weights [Cout][Cin][3][3] -> [Cout][kpad] with k = (kh*3+kw)*Cin + cin. conv1 (K = 9) stays on the host.
    for (int l = 0; l < 3; l++) {
        const auto& t = f.get("model.audio.conv2d" + std::to_string(l + 1) + ".weight");
        const int cin = I.conv_cin[l], kpad = round_up(9 * cin, 16);
        std::vector<float> w((size_t)C * kpad, 0.f);
        for (int co = 0; co < C; co++)
            for (int ci = 0; ci < cin; ci++)
                for (int k = 0; k < 9; k++) w[(size_t)co * kpad + (size_t)k * cin + ci] = bf2f(t.data[((size_t)co * cin + ci) * 9 + k]);
        I.conv[l] = I.make(w, C, kpad, to_f32(f.get("model.audio.conv2d" + std::to_string(l + 1) + ".bias")), /*host=*/l == 0);
    }
    const int conv_k = C * (cfg_.num_mel_bins / 8);   // 480 * 16 = 7680
    I.conv_out = I.make(unpermute_matrix(f.get("model.audio.conv_out.weight"), D, conv_k), D, conv_k, {});

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
        std::vector<float> qkv_w, qkv_b;
        for (const char* m : {"q", "k", "v"}) {
            const auto w = unpermute_matrix(f.get(p + "self_attn." + m + "_proj.weight"), D, D);
            const auto b = to_f32(f.get(p + "self_attn." + m + "_proj.bias"));
            qkv_w.insert(qkv_w.end(), w.begin(), w.end());
            qkv_b.insert(qkv_b.end(), b.begin(), b.end());
        }
        L.qkv = I.make(qkv_w, 3 * D, D, std::move(qkv_b));
        L.o = I.make(unpermute_matrix(f.get(p + "self_attn.out_proj.weight"), D, D), D, D, to_f32(f.get(p + "self_attn.out_proj.bias")));
        L.fc1 = I.make(unpermute_matrix(f.get(p + "fc1.weight"), cfg_.ffn_dim, D), cfg_.ffn_dim, D, to_f32(f.get(p + "fc1.bias")));
        L.fc2 = I.make(unpermute_matrix(f.get(p + "fc2.weight"), D, cfg_.ffn_dim), D, cfg_.ffn_dim, to_f32(f.get(p + "fc2.bias")));
        L.ln1w = to_f32(f.get(p + "self_attn_layer_norm.weight"));
        L.ln1b = to_f32(f.get(p + "self_attn_layer_norm.bias"));
        L.ln2w = to_f32(f.get(p + "final_layer_norm.weight"));
        L.ln2b = to_f32(f.get(p + "final_layer_norm.bias"));
    }
    I.ln_post_w = to_f32(f.get("model.audio.ln_post.weight"));
    I.ln_post_b = to_f32(f.get("model.audio.ln_post.bias"));
    I.proj1 = I.make(unpermute_matrix(f.get("model.audio.tower.proj1.weight"), D, D), D, D, to_f32(f.get("model.audio.tower.proj1.bias")));
    I.proj2 = I.make(unpermute_matrix(f.get("model.audio.tower.proj2.weight"), cfg_.output_dim, D), cfg_.output_dim, D,
                     to_f32(f.get("model.audio.tower.proj2.bias")));
}

qwen3_asr_audio_encoder::~qwen3_asr_audio_encoder() { delete impl_; }

void qwen3_asr_audio_encoder::Impl::conv_chunks(const qwen3_asr_audio_config& cfg, const float* const* mel, const int* valid, int count,
                                                float* const* out) {
    const int C = cfg.downsample_hidden, F = cfg.num_mel_bins, W0 = cfg.chunk_frames();
    std::lock_guard<std::mutex> lock(conv_mu);
    // conv1 input as NHWC [H=128 (mel bin)][W=100 (time)][1] per chunk; frames beyond valid are zero (reference pads the chunk)
    in0.assign((size_t)count * F * W0, 0.f);
    for (int c = 0; c < count; c++)
        for (int t = 0; t < std::min(valid[c], W0); t++)
            for (int h = 0; h < F; h++) in0[((size_t)c * F + h) * W0 + t] = rbf(mel[c][(size_t)t * F + h]);   // mel is cast to bf16 first

    // each conv product takes its rows (im2col rows) straight from the previous activation
    int H = F, W = W0;
    const float* src = in0.data();
    for (int l = 0; l < 3; l++) {
        const int cin = conv_cin[l], kpad = round_up(9 * cin, 16), Ho = (H + 1) / 2, Wo = (W + 1) / 2;
        const int rows = Ho * Wo;   // per chunk
        act[l].resize((size_t)count * rows * C);
        if (conv[l].id >= 0)
            gemm->run_conv(conv[l].id, src, count, H, W, cin, kpad, act[l].data(), /*gelu=*/true);
        else {
            auto row = [=](int r, float* tmp) {
                im2col_row(src + (size_t)(r / rows) * H * W * cin, H, W, cin, kpad, r % rows, tmp);
                return (const float*)tmp;
            };
            mm_rows(conv[l], row, count * rows, act[l].data(), /*gelu=*/true);
        }
        src = act[l].data();
        H = Ho;
        W = Wo;
    }
    // act[2]: per chunk [H=16 (f)][W=13 (t)][C]; token t = vector over (c, f) in c-major order: index c*16 + f
    const int Fq = H, T = W, D = cfg.d_model;
    auto token = [=](int ct, float* tmp) {
        const float* cc = src + (size_t)(ct / T) * Fq * T * C + (size_t)(ct % T) * C;
        for (int fq = 0; fq < Fq; fq++)
            for (int ch = 0; ch < C; ch++) tmp[(size_t)ch * Fq + fq] = cc[(size_t)fq * T * C + ch];
        return (const float*)tmp;
    };
    full.resize((size_t)count * T * D);
    mm_rows(conv_out, token, count * T, full.data(), false);
    for (int c = 0; c < count; c++)
        std::memcpy(out[c], full.data() + (size_t)c * T * D, sizeof(float) * (size_t)qwen3_asr_audio_tokens(valid[c]) * D);
}

void qwen3_asr_audio_encoder::conv_front(const float* mel_chunk, int valid_frames, float* out_tokens) {
    impl_->conv_chunks(cfg_, &mel_chunk, &valid_frames, 1, &out_tokens);
}

void qwen3_asr_audio_encoder::add_pos(float* x, int tokens) {
    const int D = cfg_.d_model;
    for (int t = 0; t < tokens; t++)
        for (int i = 0; i < D; i++) x[(size_t)t * D + i] = rbf(x[(size_t)t * D + i] + impl_->pos[(size_t)t * D + i]);
}

namespace {
/// an encoder layer's activations; encode() keeps one set for all its layers (freed when it returns)
struct layer_scratch {
    std::vector<float> a, qkv, o, t1, ff;
};
thread_local layer_scratch* tl_scratch = nullptr;
struct scratch_scope {
    layer_scratch s;
    scratch_scope() { tl_scratch = &s; }
    ~scratch_scope() { tl_scratch = nullptr; }
};
}  // namespace

void qwen3_asr_audio_encoder::encoder_layer(int layer, float* x, int n) {
    Impl& I = *impl_;
    const layer_w& L = I.layers[layer];
    const int D = cfg_.d_model, H = cfg_.heads, hd = D / H, FF = cfg_.ffn_dim, WT = cfg_.window_tokens();
    layer_scratch local;
    layer_scratch& S = tl_scratch ? *tl_scratch : local;
    for (auto* v : {&S.a, &S.o, &S.t1}) v->resize((size_t)n * D);
    S.qkv.resize((size_t)n * 3 * D);
    S.ff.resize((size_t)n * FF);
    std::vector<float>&a = S.a, &qkv = S.qkv, &o = S.o, &t1 = S.t1, &ff = S.ff;

    layer_norm(x, n, D, L.ln1w.data(), L.ln1b.data(), cfg_.ln_eps, a.data());
    I.mm(L.qkv, a.data(), n, qkv.data(), false);   // row t = q[t], k[t], v[t]
    const float* q = qkv.data();
    const float* k = qkv.data() + D;
    const float* v = qkv.data() + 2 * D;
    const size_t RS = (size_t)3 * D;

    // window attention: tokens [w0, w1) only see each other
    const float scale = 1.0f / std::sqrt((float)hd);
    const int nwin = (n + WT - 1) / WT;
#pragma omp parallel for num_threads(qwen3_asr_cpu::asr_threads()) schedule(dynamic)
    for (int wh = 0; wh < nwin * H; wh++) {           // one (window, head) pair per iteration
        thread_local std::vector<float> kt, s;
        const int w = wh / H, h = wh % H;
        const int w0 = w * WT, m = std::min(n, w0 + WT) - w0;
        const size_t off = (size_t)w0 * RS + h * hd;
        window_attention(q + off, k + off, v + off, RS, m, hd, scale, I.round_scores, I.round_probs, o.data() + (size_t)w0 * D + h * hd, D,
                         kt, s);
    }
    I.mm(L.o, o.data(), n, t1.data(), false);
    add_residual(x, t1.data(), (size_t)n * D);

    layer_norm(x, n, D, L.ln2w.data(), L.ln2b.data(), cfg_.ln_eps, a.data());
    I.mm(L.fc1, a.data(), n, ff.data(), /*gelu=*/true);
    I.mm(L.fc2, ff.data(), n, t1.data(), false);
    add_residual(x, t1.data(), (size_t)n * D);
}

void qwen3_asr_audio_encoder::post(const float* x, int n, float* out) {
    Impl& I = *impl_;
    const int D = cfg_.d_model;
    std::vector<float> a((size_t)n * D), b((size_t)n * D);
    layer_norm(x, n, D, I.ln_post_w.data(), I.ln_post_b.data(), cfg_.ln_eps, a.data());
    I.mm(I.proj1, a.data(), n, b.data(), /*gelu=*/true);
    I.mm(I.proj2, b.data(), n, out, false);
}

std::vector<asr_bf16_bits> qwen3_asr_audio_encoder::encode(const float* mel, int frames, conv_cache* cache) {
    const int D = cfg_.d_model, CF = cfg_.chunk_frames();
    const int n_tokens = qwen3_asr_audio_tokens(frames);
    const int nchunks = (frames + CF - 1) / CF;
    std::vector<float> x((size_t)n_tokens * D);
    const size_t mel_floats = (size_t)CF * cfg_.num_mel_bins, row_floats = (size_t)cfg_.chunk_tokens() * D;
    auto chunk_mel = [&](int c) { return mel + (size_t)c * mel_floats; };
    auto chunk_rows = [&](int c) { return x.data() + (size_t)c * row_floats; };   // every chunk but the last has 13 tokens

    // finished chunks seen by an earlier call, while their mel is the same (the clip's clamp can move: then redo it and the rest)
    int cached = 0;
    if (cache) {
        for (int c = 0; c < std::min(cache->chunks, nchunks); c++)
            if (std::memcmp(cache->mel.data() + (size_t)c * mel_floats, chunk_mel(c), sizeof(float) * mel_floats) != 0) {
                cache->truncate(c, (int)row_floats, (int)mel_floats);
                break;
            }
        cached = std::min(cache->chunks, nchunks);
        for (int c = 0; c < cached; c++)
            std::memcpy(chunk_rows(c), cache->rows.data() + (size_t)c * row_floats,
                        sizeof(float) * (size_t)qwen3_asr_audio_tokens(std::min(CF, frames - c * CF)) * D);
    }
    // the rest, a group of chunks per conv product
    constexpr int GROUP = 8;
    for (int c0 = cached; c0 < nchunks; c0 += GROUP) {
        const int n = std::min(GROUP, nchunks - c0);
        const float* mels[GROUP];
        int valid[GROUP];
        float* outs[GROUP];
        for (int i = 0; i < n; i++) {
            mels[i] = chunk_mel(c0 + i);
            valid[i] = std::min(CF, frames - (c0 + i) * CF);
            outs[i] = chunk_rows(c0 + i);
        }
        impl_->conv_chunks(cfg_, mels, valid, n, outs);
        for (int i = 0; i < n; i++) {
            const int c = c0 + i, nt = qwen3_asr_audio_tokens(valid[i]);
            add_pos(outs[i], nt);
            if (cache && c == cache->chunks && valid[i] == CF) {
                cache->rows.insert(cache->rows.end(), outs[i], outs[i] + (size_t)nt * D);
                cache->mel.insert(cache->mel.end(), mels[i], mels[i] + mel_floats);
                cache->chunks++;
            }
        }
    }
    {
        scratch_scope scope;
        for (int l = 0; l < cfg_.layers; l++) encoder_layer(l, x.data(), n_tokens);
    }
    std::vector<float> y((size_t)n_tokens * cfg_.output_dim);
    post(x.data(), n_tokens, y.data());
    std::vector<asr_bf16_bits> out(y.size());
    for (size_t i = 0; i < y.size(); i++) out[i] = f2bf(y[i]);
    return out;
}
