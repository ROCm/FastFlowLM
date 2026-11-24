// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file inkling2_vision.cpp
/// \brief the Inkling-2 image tower on the host (see inkling2_vision.hpp)
#include "inkling2_vision.hpp"

#include <immintrin.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "inkling2_q4nx_file.hpp"

namespace inkling2_vision {

using inkling2_host::q4nx_file;

namespace {

constexpr int MM_K = 512, MM_N = 64;            // the matmul kernel's tile: 64 rows x 512 columns
constexpr int TILE = MM_K * MM_N;
constexpr float LN_EPS = 1e-5f;
constexpr float ROPE_THETA = 10000.f;

inline int round_up(int v, int m) { return (v + m - 1) / m * m; }

inline float bf2f(u16 v) {
    const uint32_t u = (uint32_t)v << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

/// \brief fp32 -> bf16, round to nearest even (finite values)
inline u16 f2bf(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    u += 0x7FFF + ((u >> 16) & 1);
    return (u16)(u >> 16);
}

std::vector<float> to_f32(const u16* p, size_t n) {
    std::vector<float> v(n);
    for (size_t i = 0; i < n; i++) v[i] = bf2f(p[i]);
    return v;
}

/// \brief y[t][n] = x[t][K] W^T + bias for a matrix in the kernel's tile layout:
///        W[n][k] = tiles[((n / 64) * Kp / 512 + k / 512) * 32768 + (n % 64) * 512 + k % 512]; N and K are the real sizes
struct linear {
    const u16* tiles = nullptr;
    int N = 0, K = 0;
    std::vector<float> bias;   // N values, empty for none

    int np() const { return round_up(N, MM_K); }
    int kp() const { return round_up(K, MM_K); }
};

linear make_linear(const q4nx_file& f, const std::string& name, int N, int K, bool bias) {
    linear l;
    l.N = N;
    l.K = K;
    const auto& t = f.get(name + ".weight", "BF16", {l.np() / MM_N, l.kp() / MM_K, TILE});
    l.tiles = reinterpret_cast<const u16*>(t.data);
    if (bias) l.bias = to_f32(f.bf16(name + ".bias", {N}), N);
    return l;
}

/// \brief the dot product of n (a multiple of 16) fp32 pairs
inline float dot16(const float* a, const float* b, int n) {
    __m512 s0 = _mm512_setzero_ps(), s1 = _mm512_setzero_ps();
    int k = 0;
    for (; k + 32 <= n; k += 32) {
        s0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + k), _mm512_loadu_ps(b + k), s0);
        s1 = _mm512_fmadd_ps(_mm512_loadu_ps(a + k + 16), _mm512_loadu_ps(b + k + 16), s1);
    }
    if (k < n) s0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + k), _mm512_loadu_ps(b + k), s0);
    return _mm512_reduce_add_ps(_mm512_add_ps(s0, s1));
}

/// \brief y[T][N] = x[T][K] W^T + bias (rows ldx and ldy floats apart); the weight tiles are read once per
///        64-token chunk
void linear_fwd(const linear& l, const float* x, int ldx, int T, float* y, int ldy) {
    const int nb = (l.N + MM_N - 1) / MM_N, kb = l.kp() / MM_K, TB = 64, nt = (T + TB - 1) / TB;
#pragma omp parallel
    {
        alignas(64) float wf[MM_K];
#pragma omp for collapse(2) schedule(dynamic)
        for (int ib = 0; ib < nb; ib++)
            for (int it = 0; it < nt; it++) {
                const int t0 = it * TB, t1 = std::min(T, t0 + TB), r1 = std::min(MM_N, l.N - ib * MM_N);
                for (int r = 0; r < r1; r++) {
                    const int n = ib * MM_N + r;
                    const float b = l.bias.empty() ? 0.f : l.bias[n];
                    for (int t = t0; t < t1; t++) y[(size_t)t * ldy + n] = b;
                    for (int k = 0; k < kb; k++) {
                        const int kn = std::min(MM_K, l.K - k * MM_K);
                        if (kn <= 0) break;
                        const u16* w = l.tiles + ((size_t)ib * kb + k) * TILE + (size_t)r * MM_K;
                        inkling2_host::bf16_to_f32(wf, w, kn);
                        for (int t = t0; t < t1; t++) y[(size_t)t * ldy + n] += dot16(x + (size_t)t * ldx + k * MM_K, wf, kn);
                    }
                }
            }
    }
}

void layer_norm(const float* x, const float* w, const float* b, int rows, float* out) {
#pragma omp parallel for schedule(static)
    for (int t = 0; t < rows; t++) {
        const float* xr = x + (size_t)t * DIM;
        float* o = out + (size_t)t * DIM;
        double s = 0;
        for (int i = 0; i < DIM; i++) s += xr[i];
        const float mean = (float)(s / DIM);
        double v = 0;
        for (int i = 0; i < DIM; i++) v += (double)(xr[i] - mean) * (xr[i] - mean);
        const float inv = 1.f / std::sqrt((float)(v / DIM) + LN_EPS);
        for (int i = 0; i < DIM; i++) o[i] = (xr[i] - mean) * inv * w[i] + b[i];
    }
}

inline float gelu(float v) { return 0.5f * v * (1.f + std::erf(v * 0.70710678118654752f)); }

void gelu_rows(float* x, int rows, int cols) {
#pragma omp parallel for schedule(static)
    for (int t = 0; t < rows; t++) {
        float* r = x + (size_t)t * cols;
        for (int i = 0; i < cols; i++) r[i] = gelu(r[i]);
    }
}

/// \brief the taps and weights of one axis of the bilinear resample of the 32-entry position grid to n entries
///        (half-pixel centres, edge taps clamped)
void axis_taps(int i, int n, int tap[2], float w[2]) {
    const float src = ((float)i + 0.5f) * POS_SIDE / (float)n - 0.5f;
    const float fl = std::floor(src);
    for (int o = 0; o < 2; o++) {
        tap[o] = std::min(std::max((int)fl + o, 0), POS_SIDE - 1);
        w[o] = std::max(1.f - std::fabs(src - fl - (float)o), 0.f);
    }
}

}  // namespace

grid grid_for(int height, int width) {
    if (height <= 0 || width <= 0) throw std::runtime_error("inkling2 vision: empty image");
    return {(height + CELL - 1) / CELL, (width + CELL - 1) / CELL};
}

void patchify(const uint8_t* rgb, int height, int width, float* pixels) {
    const grid g = grid_for(height, width);
    const float scale = (float)(1.0 / 255.0);
    auto norm = [&](float v) { return (v * scale - 0.5f) / 0.5f; };
    const float pad = norm(-1.f);
#pragma omp parallel for schedule(static)
    for (int tok = 0; tok < g.tokens(); tok++) {
        const int r = tok / g.cols, c = tok % g.cols;
        float* o = pixels + (size_t)tok * CELL * CELL * 3;
        for (int yy = 0; yy < CELL; yy++) {
            const int y = r * CELL + yy;
            for (int xx = 0; xx < CELL; xx++, o += 3) {
                const int x = c * CELL + xx;
                if (y < height && x < width) {
                    const uint8_t* p = rgb + ((size_t)y * width + x) * 3;
                    o[0] = norm(p[0]);
                    o[1] = norm(p[1]);
                    o[2] = norm(p[2]);
                } else {
                    o[0] = o[1] = o[2] = pad;
                }
            }
        }
    }
}

struct tower::impl {
    q4nx_file file;

    struct layer {
        std::vector<float> ln1_w, ln1_b, ln2_w, ln2_b, ls1, ls2;
        linear qkv, out, fc1, fc2;
    };
    std::vector<layer> layers;
    linear patch, proj_in, proj_out;
    std::vector<float> cls;      // class embedding + its position embedding
    const u16* pos = nullptr;    // [32 * 32][DIM] bf16
    std::vector<float> ln_pre_w, ln_pre_b;
    std::vector<float> inv_freq;  // [HDIM / 4]
    std::unique_ptr<accel> dev;

    void set_accel(std::unique_ptr<accel> a) {
        dev = std::move(a);
        if (!dev) return;
        auto m = [](const linear& l) { return accel::mat{l.tiles, l.N, l.K, l.bias.empty() ? nullptr : l.bias.data()}; };
        std::vector<accel::layer_mats> lm(layers.size());
        for (size_t i = 0; i < layers.size(); i++) lm[i] = {m(layers[i].qkv), m(layers[i].out), m(layers[i].fc1), m(layers[i].fc2)};
        dev->bind(m(patch), lm, m(proj_in), m(proj_out));
    }

    void load(const std::string& path) {
        file.open(path);
        const std::string B = "model.visual.backbone.";
        patch = make_linear(file, B + "embeddings.patch_embedding", DIM, PATCH_K, false);
        const auto c = to_f32(file.bf16(B + "embeddings.class_embedding", {DIM}), DIM);
        const auto cp = to_f32(file.bf16(B + "embeddings.cls_position_embedding", {DIM}), DIM);
        cls.resize(DIM);
        for (int i = 0; i < DIM; i++) cls[i] = c[i] + cp[i];
        pos = file.bf16(B + "embeddings.position_embedding", {POS_SIDE * POS_SIDE, DIM});
        ln_pre_w = to_f32(file.bf16(B + "ln_pre.weight", {DIM}), DIM);
        ln_pre_b = to_f32(file.bf16(B + "ln_pre.bias", {DIM}), DIM);
        layers.resize(NLAYER);
        for (int i = 0; i < NLAYER; i++) {
            const std::string L = B + "layers." + std::to_string(i) + ".";
            auto& l = layers[i];
            auto vec = [&](const char* n) { return to_f32(file.bf16(L + n, {DIM}), DIM); };
            l.ln1_w = vec("ln_1.weight");
            l.ln1_b = vec("ln_1.bias");
            l.ln2_w = vec("ln_2.weight");
            l.ln2_b = vec("ln_2.bias");
            l.ls1 = vec("ls_1.gamma");
            l.ls2 = vec("ls_2.gamma");
            l.qkv = make_linear(file, L + "self_attn.qkv_proj", 3 * DIM, DIM, true);
            l.out = make_linear(file, L + "self_attn.out_proj", DIM, DIM, true);
            l.fc1 = make_linear(file, L + "mlp.c_fc", FFN, DIM, true);
            l.fc2 = make_linear(file, L + "mlp.c_proj", DIM, FFN, true);
        }
        proj_in = make_linear(file, "model.visual.proj_in", MERGED, MERGED, false);
        proj_out = make_linear(file, "model.visual.proj_out", OUT, MERGED, false);
        inv_freq.resize(HDIM / 4);
        for (int i = 0; i < HDIM / 4; i++) inv_freq[i] = 1.f / std::pow(ROPE_THETA, (float)(2 * i) / (float)(HDIM / 2));
    }

    /// \brief rotary angles of every row: cos/sin [L][HDIM / 2], pair i of dims (2i, 2i + 1); pairs 0-15 turn with
    ///        the column, 16-31 with the row; positions are 1-based, the class token (row 0) is at (0, 0)
    void rope_tables(const grid& g, std::vector<float>& cs, std::vector<float>& sn) const {
        const int L = g.seq(), P = HDIM / 2;
        cs.assign((size_t)L * P, 1.f);
        sn.assign((size_t)L * P, 0.f);
        for (int tok = 0; tok < g.tokens(); tok++)
            for (int m = 0; m < MERGE * MERGE; m++) {
                const int row = 1 + tok * MERGE * MERGE + m;
                const int y = (tok / g.cols) * MERGE + m / MERGE, x = (tok % g.cols) * MERGE + m % MERGE;
                for (int i = 0; i < P / 2; i++) {
                    const float aw = (float)(x + 1) * inv_freq[i], ah = (float)(y + 1) * inv_freq[i];
                    cs[(size_t)row * P + i] = std::cos(aw);
                    sn[(size_t)row * P + i] = std::sin(aw);
                    cs[(size_t)row * P + P / 2 + i] = std::cos(ah);
                    sn[(size_t)row * P + P / 2 + i] = std::sin(ah);
                }
            }
    }

    void encode(const float* pixels, const grid& g, u16* out_bf16, float* out_f32,
                const std::function<void(const char*, const float*, int, int)>& trace) const;
};

tower::tower() : p_(new impl) {}
tower::~tower() = default;

void tower::load(const std::string& vision_q4nx) { p_->load(vision_q4nx); }
void tower::set_accel(std::unique_ptr<accel> a) { p_->set_accel(std::move(a)); }

void tower::encode(const float* pixels, const grid& g, u16* out_bf16, float* out_f32) const {
    p_->encode(pixels, g, out_bf16, out_f32, trace);
}

void tower::impl::encode(const float* pixels, const grid& g, u16* out_bf16, float* out_f32,
                         const std::function<void(const char*, const float*, int, int)>& trace) const {
    const int N = g.tokens(), L = g.seq();
    auto tap = [&](const char* tag, const std::vector<float>& x) {
        if (trace) trace(tag, x.data(), L, DIM);
    };

    // ---- the patches of every cell, [py][px][c] each, as the rows of the patch embedding's input
    std::vector<float> patches((size_t)(L - 1) * PATCH_K);
#pragma omp parallel for schedule(static)
    for (int tok = 0; tok < N; tok++)
        for (int m = 0; m < MERGE * MERGE; m++) {
            float* d = patches.data() + ((size_t)tok * MERGE * MERGE + m) * PATCH_K;
            const float* s = pixels + (size_t)tok * CELL * CELL * 3 + (size_t)(m / MERGE) * PATCH * CELL * 3 + (m % MERGE) * PATCH * 3;
            for (int py = 0; py < PATCH; py++) memcpy(d + py * PATCH * 3, s + (size_t)py * CELL * 3, PATCH * 3 * sizeof(float));
        }

    // ---- embeddings: class token, then patch + position embedding (bilinear of the 32 x 32 table)
    std::vector<float> x((size_t)L * DIM), h((size_t)L * DIM);
    memcpy(x.data(), cls.data(), DIM * sizeof(float));
    if (dev) dev->embed(patches.data(), L - 1, x.data() + DIM);
    else linear_fwd(patch, patches.data(), PATCH_K, L - 1, x.data() + DIM, DIM);
    {
        const int H = g.rows * MERGE, W = g.cols * MERGE;
#pragma omp parallel for schedule(static)
        for (int tok = 0; tok < N; tok++)
            for (int m = 0; m < MERGE * MERGE; m++) {
                const int y = (tok / g.cols) * MERGE + m / MERGE, xx = (tok % g.cols) * MERGE + m % MERGE;
                int ty[2], tx[2];
                float wy[2], wx[2];
                axis_taps(y, H, ty, wy);
                axis_taps(xx, W, tx, wx);
                float* r = x.data() + (size_t)(1 + tok * MERGE * MERGE + m) * DIM;
                for (int a = 0; a < 2; a++)
                    for (int b = 0; b < 2; b++) {
                        const u16* p = pos + (size_t)(ty[a] * POS_SIDE + tx[b]) * DIM;
                        const float wt = wy[a] * wx[b];
                        for (int i = 0; i < DIM; i++) r[i] += wt * bf2f(p[i]);
                    }
            }
    }
    tap("embed", x);
    layer_norm(x.data(), ln_pre_w.data(), ln_pre_b.data(), L, x.data());
    tap("ln_pre", x);

    std::vector<float> cs, sn;
    rope_tables(g, cs, sn);
    const int Lp = round_up(L, 16);
    std::vector<float> qkv, att, ff;
    if (!dev) { qkv.resize((size_t)L * 3 * DIM); att.resize((size_t)L * DIM); ff.resize((size_t)L * FFN); }
    std::vector<float> kt, vv;
    if (!dev) { kt.resize((size_t)NHEAD * HDIM * Lp); vv.resize((size_t)NHEAD * L * HDIM); }
    const float scale = 1.f / std::sqrt((float)HDIM);

    for (int il = 0; il < NLAYER; il++) {
        const layer& ly = layers[il];

        // ---- attention
        layer_norm(x.data(), ly.ln1_w.data(), ly.ln1_b.data(), L, h.data());
        if (dev) {
            dev->attention(il, h.data(), L, cs.data(), sn.data(), h.data());
        } else {
            linear_fwd(ly.qkv, h.data(), DIM, L, qkv.data(), 3 * DIM);
            // rotate q and k by pairs, then lay k out [head][d][position] and v [head][position][d]
#pragma omp parallel for schedule(static)
            for (int t = 0; t < L; t++) {
                float* q = qkv.data() + (size_t)t * 3 * DIM;
                const float* c = cs.data() + (size_t)t * (HDIM / 2);
                const float* s = sn.data() + (size_t)t * (HDIM / 2);
                for (int hd = 0; hd < NHEAD; hd++)
                    for (int part = 0; part < 2; part++) {
                        float* v = q + part * DIM + hd * HDIM;
                        for (int i = 0; i < HDIM / 2; i++) {
                            const float a = v[2 * i], b = v[2 * i + 1];
                            v[2 * i] = a * c[i] - b * s[i];
                            v[2 * i + 1] = b * c[i] + a * s[i];
                        }
                    }
                const float* k = q + DIM;
                const float* vsrc = q + 2 * DIM;
                for (int hd = 0; hd < NHEAD; hd++) {
                    for (int d = 0; d < HDIM; d++) kt[((size_t)hd * HDIM + d) * Lp + t] = k[hd * HDIM + d];
                    memcpy(vv.data() + ((size_t)hd * L + t) * HDIM, vsrc + hd * HDIM, HDIM * sizeof(float));
                }
            }
            for (int hd = 0; hd < NHEAD; hd++)   // the padded keys score -inf
                for (int d = 0; d < HDIM; d++)
                    for (int t = L; t < Lp; t++) kt[((size_t)hd * HDIM + d) * Lp + t] = 0.f;
#pragma omp parallel
            {
                std::vector<float> sc(Lp);
#pragma omp for collapse(2) schedule(dynamic, 8)
                for (int hd = 0; hd < NHEAD; hd++)
                    for (int t = 0; t < L; t++) {
                        const float* q = qkv.data() + (size_t)t * 3 * DIM + hd * HDIM;
                        const float* K = kt.data() + (size_t)hd * HDIM * Lp;
                        std::fill(sc.begin(), sc.end(), 0.f);
                        for (int d = 0; d < HDIM; d++) {
                            const __m512 qd = _mm512_set1_ps(q[d] * scale);
                            for (int j = 0; j < Lp; j += 16)
                                _mm512_storeu_ps(&sc[j], _mm512_fmadd_ps(qd, _mm512_loadu_ps(K + (size_t)d * Lp + j), _mm512_loadu_ps(&sc[j])));
                        }
                        float mx = -INFINITY;
                        for (int j = 0; j < L; j++) mx = std::max(mx, sc[j]);
                        float den = 0.f;
                        for (int j = 0; j < L; j++) {
                            sc[j] = std::exp(sc[j] - mx);
                            den += sc[j];
                        }
                        const float* V = vv.data() + (size_t)hd * L * HDIM;
                        __m512 a0 = _mm512_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
                        for (int j = 0; j < L; j++) {
                            const __m512 p = _mm512_set1_ps(sc[j]);
                            const float* v = V + (size_t)j * HDIM;
                            a0 = _mm512_fmadd_ps(p, _mm512_loadu_ps(v), a0);
                            a1 = _mm512_fmadd_ps(p, _mm512_loadu_ps(v + 16), a1);
                            a2 = _mm512_fmadd_ps(p, _mm512_loadu_ps(v + 32), a2);
                            a3 = _mm512_fmadd_ps(p, _mm512_loadu_ps(v + 48), a3);
                        }
                        const __m512 inv = _mm512_set1_ps(1.f / den);
                        float* o = att.data() + (size_t)t * DIM + hd * HDIM;
                        _mm512_storeu_ps(o, _mm512_mul_ps(a0, inv));
                        _mm512_storeu_ps(o + 16, _mm512_mul_ps(a1, inv));
                        _mm512_storeu_ps(o + 32, _mm512_mul_ps(a2, inv));
                        _mm512_storeu_ps(o + 48, _mm512_mul_ps(a3, inv));
                    }
            }
            linear_fwd(ly.out, att.data(), DIM, L, h.data(), DIM);
        }
#pragma omp parallel for schedule(static)
        for (int t = 0; t < L; t++)
            for (int i = 0; i < DIM; i++) x[(size_t)t * DIM + i] += ly.ls1[i] * h[(size_t)t * DIM + i];

        // ---- MLP
        layer_norm(x.data(), ly.ln2_w.data(), ly.ln2_b.data(), L, h.data());
        if (dev) {
            dev->mlp(il, h.data(), L, h.data());
        } else {
            linear_fwd(ly.fc1, h.data(), DIM, L, ff.data(), FFN);
            gelu_rows(ff.data(), L, FFN);
            linear_fwd(ly.fc2, ff.data(), FFN, L, h.data(), DIM);
        }
#pragma omp parallel for schedule(static)
        for (int t = 0; t < L; t++)
            for (int i = 0; i < DIM; i++) x[(size_t)t * DIM + i] += ly.ls2[i] * h[(size_t)t * DIM + i];
        if (trace) {
            const std::string tag = "layer" + std::to_string(il);
            trace(tag.c_str(), x.data(), L, DIM);
        }
    }

    // ---- merger: the four patches of a cell are four consecutive rows, so the rows after the class token already
    //      are [tokens][4 * DIM]
    std::vector<float> m1((size_t)N * MERGED), o((size_t)N * OUT);
    if (dev) {
        dev->merge(x.data() + DIM, N, o.data());
    } else {
        linear_fwd(proj_in, x.data() + DIM, MERGED, N, m1.data(), MERGED);
        if (trace) trace("merger_fc1", m1.data(), N, MERGED);
        gelu_rows(m1.data(), N, MERGED);
        linear_fwd(proj_out, m1.data(), MERGED, N, o.data(), OUT);
    }
    if (out_f32) memcpy(out_f32, o.data(), o.size() * sizeof(float));
    if (out_bf16)
        for (size_t i = 0; i < o.size(); i++) out_bf16[i] = f2bf(o[i]);
}

}  // namespace inkling2_vision
