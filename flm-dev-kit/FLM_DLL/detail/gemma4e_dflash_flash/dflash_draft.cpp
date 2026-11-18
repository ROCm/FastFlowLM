// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

// Host (AVX-512) DFlash drafter. See dflash_draft.hpp and README.md.
//
// Numerics: activations, attention and the K/V ring are fp32; weights are dequantized in registers from the Q4NX Q4_1
// blocks (w = q * d + m, d/m bf16 as stored) or read as bf16 for tensors the converter kept BF16. The weight matrices are
// repacked once at load so that the GEMM streams linearly: rows are grouped in blocks of 16, K in tiles of 32 groups (1024
// columns), and one (row block, tile) is contiguous:  group index of (n, g) = nb*16*G + (g/32)*512 + (n%16)*gt + g%32
// where nb = n/16, G = K/32 and gt = groups in that tile (32, or fewer in the last tile)
//     q : [group][16 B]  (byte j = q[g*32+j] | q[g*32+16+j] << 4)
//     dm: [group]        (d, m as the stored bf16 pair in one 32-bit word: 4 bytes instead of 8, the GEMM is memory bound)
// GEMM(M<=8 tokens): tiled over K so the x tile (8 x 1024 fp32 = 32 KB) stays in L1 while the weights stream once.
#ifdef _WIN32
#include <malloc.h>
#endif
#include "dflash_draft.hpp"
#include "nlohmann/json.hpp"

#include <immintrin.h>
#include <omp.h>
#include <initializer_list>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <fstream>
#include <map>
#include <memory>
#include <stdexcept>

namespace {

using f32 = float;
constexpr int RING = 2048;   // positions kept in the ctx K/V ring (D12)

static inline f32 bf16_to_f32(uint16_t v) {
    uint32_t u = (uint32_t)v << 16;
    f32 f;
    std::memcpy(&f, &u, 4);
    return f;
}
static inline uint16_t f32_to_bf16(f32 f) {   // round to nearest even
    uint32_t u;
    std::memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x40);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

// std::aligned_alloc does not exist on MSVC
inline void* aligned_malloc64(size_t bytes) {
#ifdef _WIN32
    return _aligned_malloc(bytes, 64);
#else
    return std::aligned_alloc(64, bytes);
#endif
}
struct aligned_free {
    void operator()(void* p) const {
#ifdef _WIN32
        _aligned_free(p);
#else
        std::free(p);
#endif
    }
};
template <typename T>
struct abuf {
    std::unique_ptr<T, aligned_free> p;
    size_t n = 0;
    void alloc(size_t cnt) {
        n = cnt;
        size_t bytes = ((cnt * sizeof(T) + 63) / 64) * 64;
        void* m = aligned_malloc64(bytes ? bytes : 64);
        if (!m) throw std::bad_alloc();
        std::memset(m, 0, bytes);
        p.reset((T*)m);
    }
    T* data() { return p.get(); }
    const T* data() const { return p.get(); }
    T& operator[](size_t i) { return p.get()[i]; }
    const T& operator[](size_t i) const { return p.get()[i]; }
};

// ---- safetensors (the .q4nx container) --------------------------------------------------------------------------------
struct st_tensor { std::string dtype; std::vector<int64_t> shape; size_t b, e; };
struct st_file {
    std::vector<uint8_t> data;   // whole file
    size_t base = 0;
    std::map<std::string, st_tensor> t;
    bool open(const std::string& path) {
        std::ifstream f(path, std::ios::binary | std::ios::ate);
        if (!f) return false;
        size_t sz = (size_t)f.tellg();
        f.seekg(0);
        data.resize(sz);
        f.read((char*)data.data(), (std::streamsize)sz);
        if (sz < 8) return false;
        uint64_t hl;
        std::memcpy(&hl, data.data(), 8);
        if (8 + hl > sz) return false;
        auto j = nlohmann::json::parse(std::string((const char*)data.data() + 8, (size_t)hl));
        base = 8 + (size_t)hl;
        for (auto it = j.begin(); it != j.end(); ++it) {
            if (it.key() == "__metadata__") continue;
            st_tensor s;
            s.dtype = it.value()["dtype"].get<std::string>();
            for (auto& d : it.value()["shape"]) s.shape.push_back(d.get<int64_t>());
            s.b = it.value()["data_offsets"][0].get<size_t>();
            s.e = it.value()["data_offsets"][1].get<size_t>();
            t[it.key()] = s;
        }
        return true;
    }
    const uint8_t* ptr(const std::string& n) const { return data.data() + base + t.at(n).b; }
    bool has(const std::string& n) const { return t.count(n) != 0; }
};

// ---- weight matrices -------------------------------------------------------------------------------------------------
struct Mat {
    int N = 0, K = 0;
    bool bf = false;
    abuf<uint8_t> q;     // Q4: [N][K/32][16]
    abuf<uint32_t> dm;   // Q4: [N][K/32] (d | m << 16) bf16 pair
    abuf<uint16_t> w;    // BF16: [N][K]
};

constexpr int RB = 32, CB = 256, PS = 16;
constexpr int KT_GROUPS = 32;   // K tile = 32 groups = 1024 columns

// group slot (see the layout note above)
static inline size_t grp_slot(int n, int g, int G) {
    const int t = g / KT_GROUPS, gi = g % KT_GROUPS;
    const int gt = std::min(KT_GROUPS, G - t * KT_GROUPS);
    return (size_t)(n / 16) * 16 * G + (size_t)t * 16 * KT_GROUPS + (size_t)(n % 16) * gt + gi;
}

// q4nx chunk blob -> repacked Mat.  chunk index = rb * (K/256) + cb; layout per chunk (see FLM_Q4NX_Converter/check_dflash_q4nx.py)
static void load_q4(Mat& m, const uint8_t* blob, int N, int K) {
    m.N = N; m.K = K; m.bf = false;
    const int G = K / 32, ncb = K / CB;
    m.q.alloc((size_t)N * G * 16);
    m.dm.alloc((size_t)N * G);
    const size_t chunk_bytes = (size_t)RB * CB * 5 / 8;   // 5120
    const int ng = CB / 32;
    #pragma omp parallel for schedule(static)
    for (int rb = 0; rb < N / RB; rb++) {
        for (int cb = 0; cb < ncb; cb++) {
            const uint8_t* c = blob + ((size_t)rb * ncb + cb) * chunk_bytes;
            const uint16_t* d = (const uint16_t*)c;                 // [ng][RB]
            const uint16_t* mn = (const uint16_t*)(c + ng * RB * 2);   // [ng][RB]
            const uint8_t* qs = c + 2 * ng * RB * 2;                // [RB/PS][CB][PS/2]
            for (int R = 0; R < RB; R++) {
                const int n = rb * RB + R;
                const int g0 = R / PS, r = (R % PS) / 2, b = R % 2;
                for (int j = 0; j < ng; j++) {
                    const int G_idx = cb * ng + j;
                    const size_t slot = grp_slot(n, G_idx, G);
                    uint8_t* dst = &m.q[slot * 16];
                    for (int e = 0; e < 16; e++) {
                        const int c_lo = j * 32 + e, c_hi = j * 32 + 16 + e;
                        uint8_t lo = (qs[((size_t)g0 * CB + c_lo) * (PS / 2) + r] >> (4 * b)) & 0xF;
                        uint8_t hi = (qs[((size_t)g0 * CB + c_hi) * (PS / 2) + r] >> (4 * b)) & 0xF;
                        dst[e] = (uint8_t)(lo | (hi << 4));
                    }
                    m.dm[slot] = (uint32_t)d[j * RB + R] | ((uint32_t)mn[j * RB + R] << 16);
                }
            }
        }
    }
}

// BF16 kept tensors: [N/32][K][32] (each 32-row block column-major) -> [N][K]
static void load_bf16_blocks(Mat& m, const uint16_t* src, int N, int K) {
    m.N = N; m.K = K; m.bf = true;
    m.w.alloc((size_t)N * K);
    for (int rb = 0; rb < N / 32; rb++)
        for (int k = 0; k < K; k++)
            for (int r = 0; r < 32; r++) m.w[(size_t)(rb * 32 + r) * K + k] = src[((size_t)rb * K + k) * 32 + r];
}

static inline __m512 bf16x16_to_ps(const uint16_t* p) {
    return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i*)p)), 16));
}


// exp(x) for 16 lanes: x = n*ln2 + r, |r| <= ln2/2, degree-6 polynomial (rel. error ~1e-7), 2^n via scalef
static inline __m512 exp512(__m512 x) {
    x = _mm512_max_ps(x, _mm512_set1_ps(-87.0f));
    x = _mm512_min_ps(x, _mm512_set1_ps(88.0f));
    const __m512 n = _mm512_roundscale_ps(_mm512_mul_ps(x, _mm512_set1_ps(1.44269504088896341f)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m512 r = _mm512_fnmadd_ps(n, _mm512_set1_ps(0.693359375f), x);
    r = _mm512_fnmadd_ps(n, _mm512_set1_ps(-2.12194440e-4f), r);
    __m512 p = _mm512_set1_ps(1.9875691500e-4f);
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.3981999507e-3f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(8.3334519073e-3f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(4.1665795894e-2f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.6666665459e-1f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(5.0000001201e-1f));
    p = _mm512_fmadd_ps(_mm512_mul_ps(p, r), r, _mm512_add_ps(r, _mm512_set1_ps(1.0f)));
    return _mm512_scalef_ps(p, n);
}

// ---- GEMM: Y[t][n] = sum_k X[t][k] * W[n][k], X f32 [MT][ldx], Y f32 [MT][ldy] ---------------------------------------
// One 16-row weight block x one K tile (groups [g0, g1)) x MT tokens, accumulated into part[16][MT][16] (lane partial sums).
template <int MT, bool BF>
static void gemm_tile(const Mat& m, const f32* X, int ldx, f32 (*part)[8][16], int nb, int nrows, int g0, int g1) {
    const int G = m.K / 32;
    const int gt = g1 - g0;
    const size_t tile_base = (size_t)(nb / 16) * 16 * G + (size_t)(g0 / KT_GROUPS) * 16 * KT_GROUPS;
    for (int rp = 0; rp < nrows; rp += 2) {
        __m512 acc0[MT], acc1[MT];
        for (int t = 0; t < MT; t++) { acc0[t] = _mm512_load_ps(part[rp][t]); acc1[t] = _mm512_load_ps(part[rp + 1][t]); }
        const int n0 = nb + rp, n1 = n0 + 1;
        const size_t s0 = tile_base + (size_t)rp * gt, s1 = s0 + gt;   // slots of (row rp, group g0) and (row rp+1, group g0)
        for (int g = g0; g < g1; g++) {
            __m512 w0lo, w0hi, w1lo, w1hi;
            if constexpr (!BF) {
                const __m128i v0 = _mm_loadu_si128((const __m128i*)&m.q[(s0 + (g - g0)) * 16]);
                const __m128i v1 = _mm_loadu_si128((const __m128i*)&m.q[(s1 + (g - g0)) * 16]);
                const __m128i mask = _mm_set1_epi8(0x0F);
                const __m512 q0l = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_and_si128(v0, mask)));
                const __m512 q0h = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_and_si128(_mm_srli_epi16(v0, 4), mask)));
                const __m512 q1l = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_and_si128(v1, mask)));
                const __m512 q1h = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_and_si128(_mm_srli_epi16(v1, 4), mask)));
                const uint32_t p0 = m.dm[s0 + (g - g0)], p1 = m.dm[s1 + (g - g0)];
                const __m512 sc0 = _mm512_castsi512_ps(_mm512_set1_epi32((int)(p0 << 16))), z0 = _mm512_castsi512_ps(_mm512_set1_epi32((int)(p0 & 0xFFFF0000u)));
                const __m512 sc1 = _mm512_castsi512_ps(_mm512_set1_epi32((int)(p1 << 16))), z1 = _mm512_castsi512_ps(_mm512_set1_epi32((int)(p1 & 0xFFFF0000u)));
                w0lo = _mm512_fmadd_ps(q0l, sc0, z0); w0hi = _mm512_fmadd_ps(q0h, sc0, z0);
                w1lo = _mm512_fmadd_ps(q1l, sc1, z1); w1hi = _mm512_fmadd_ps(q1h, sc1, z1);
            } else {
                w0lo = bf16x16_to_ps(&m.w[(size_t)n0 * m.K + g * 32]);
                w0hi = bf16x16_to_ps(&m.w[(size_t)n0 * m.K + g * 32 + 16]);
                w1lo = bf16x16_to_ps(&m.w[(size_t)n1 * m.K + g * 32]);
                w1hi = bf16x16_to_ps(&m.w[(size_t)n1 * m.K + g * 32 + 16]);
            }
            for (int t = 0; t < MT; t++) {
                const __m512 xl = _mm512_loadu_ps(X + (size_t)t * ldx + g * 32);
                const __m512 xh = _mm512_loadu_ps(X + (size_t)t * ldx + g * 32 + 16);
                acc0[t] = _mm512_fmadd_ps(w0lo, xl, acc0[t]);
                acc0[t] = _mm512_fmadd_ps(w0hi, xh, acc0[t]);
                acc1[t] = _mm512_fmadd_ps(w1lo, xl, acc1[t]);
                acc1[t] = _mm512_fmadd_ps(w1hi, xh, acc1[t]);
            }
        }
        for (int t = 0; t < MT; t++) { _mm512_store_ps(part[rp][t], acc0[t]); _mm512_store_ps(part[rp + 1][t], acc1[t]); }
    }
}

template <bool BF>
static void gemm_tile_dispatch(int mt, const Mat& m, const f32* X, int ldx, f32 (*part)[8][16], int nb, int nrows, int g0, int g1) {
    switch (mt) {
        case 1: gemm_tile<1, BF>(m, X, ldx, part, nb, nrows, g0, g1); break;
        case 2: gemm_tile<2, BF>(m, X, ldx, part, nb, nrows, g0, g1); break;
        case 3: gemm_tile<3, BF>(m, X, ldx, part, nb, nrows, g0, g1); break;
        case 4: gemm_tile<4, BF>(m, X, ldx, part, nb, nrows, g0, g1); break;
        case 5: gemm_tile<5, BF>(m, X, ldx, part, nb, nrows, g0, g1); break;
        case 6: gemm_tile<6, BF>(m, X, ldx, part, nb, nrows, g0, g1); break;
        case 7: gemm_tile<7, BF>(m, X, ldx, part, nb, nrows, g0, g1); break;
        default: gemm_tile<8, BF>(m, X, ldx, part, nb, nrows, g0, g1); break;
    }
}

// rows [nb, nb+16) of Y for all M tokens: K tiles outer, 8-token tiles inner, so a weight K tile (16 x 1024, 12 KB) is fetched once and
// reused from L1 by every token tile, and the X K tile of all tokens (M x 4 KB) stays in L2
static constexpr int MAXTT = 8;     // up to 64 tokens per call
static void gemm_rows(const Mat& m, const f32* X, int ldx, int M, f32* Y, int ldy, int nb) {
    const int G = m.K / 32;
    const int nrows = std::min(16, m.N - nb);
    const int ntt = (M + 7) / 8;
    alignas(64) f32 part[MAXTT][16][8][16];
    std::memset(part, 0, sizeof(f32) * 16 * 8 * 16 * ntt);
    for (int g0 = 0; g0 < G; g0 += KT_GROUPS) {
        const int g1 = std::min(G, g0 + KT_GROUPS);
        for (int tt = 0; tt < ntt; tt++) {
            const int mt = std::min(8, M - tt * 8);
            if (m.bf) gemm_tile_dispatch<true>(mt, m, X + (size_t)tt * 8 * ldx, ldx, part[tt], nb, nrows, g0, g1);
            else gemm_tile_dispatch<false>(mt, m, X + (size_t)tt * 8 * ldx, ldx, part[tt], nb, nrows, g0, g1);
        }
    }
    for (int tt = 0; tt < ntt; tt++) {
        const int mt = std::min(8, M - tt * 8);
        for (int r = 0; r < nrows; r++)
            for (int t = 0; t < mt; t++) Y[(size_t)(tt * 8 + t) * ldy + nb + r] = _mm512_reduce_add_ps(_mm512_load_ps(part[tt][r][t]));
    }
}

}  // namespace

// =====================================================================================================================
struct dflash_draft_host::Impl {
    // FLM_DFLASH_PROFILE=1: per-phase wall time of draft() accumulated over calls, printed by the destructor
    bool prof = false;
    double t_gemm = 0, t_attn = 0, t_glue = 0;
    double t_inj = 0, t_inj_gemm = 0; long long n_inj_rows = 0; bool in_inject = false;
    int n_draft = 0;
    bool in_draft = false;
    static double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
    config c;
    bool loaded = false;
    int nthreads = 8;

    Mat fc;
    std::vector<f32> hidden_norm, out_norm_f;
    abuf<uint16_t> out_norm_bf;
    struct Layer {
        Mat q, k, v, o, gate, up, down;
        std::vector<f32> attn_norm, ffn_norm, q_norm, k_norm;
    };
    std::vector<Layer> L;

    // ctx K/V ring: [layer][RING][n_kv*head_dim] fp32
    std::vector<abuf<f32>> Kc, Vc;
    int ctx_begin = 0, ctx_end = 0;   // positions [ctx_begin, ctx_end) held (at most RING)
    std::vector<f32> inv_freq;        // [head_dim/2]

    void gemm(const Mat& m, const f32* X, int ldx, int M, f32* Y, int ldy) {
        const double tp0 = prof ? now_ms() : 0;
        const int nblk = m.N / 16;
        // one parallel region; each thread walks its 16-row weight blocks and, per block, every 8-token tile: the block
        // (16 x K, <= 150 KB) is read from DRAM once and re-read from L2 for the other tiles (prompt priming injects up to 1024 rows)
        for (int m0 = 0; m0 < M; m0 += 8 * MAXTT) {
            const int mm = std::min(8 * MAXTT, M - m0);
            #pragma omp parallel for schedule(dynamic, 2) num_threads(nthreads)     // dynamic: a descheduled thread (busy host) does not stall the rest
            for (int b = 0; b < nblk; b++) gemm_rows(m, X + (size_t)m0 * ldx, ldx, mm, Y + (size_t)m0 * ldy, ldy, b * 16);
        }
        if (prof && in_draft) t_gemm += now_ms() - tp0;
        if (prof && in_inject) t_inj_gemm += now_ms() - tp0;
    }

    // several matrices sharing the input X (q|k|v, gate|up): one parallel region over all their 16-row blocks, so the small
    // matrices do not leave threads idle and there is one fork/join instead of one per matrix
    struct Out { const Mat* m; f32* y; int ldy; };
    void gemm_multi(std::initializer_list<Out> outs, const f32* X, int ldx, int M) {
        const double tp0 = prof ? now_ms() : 0;
        Out o[4]; int nb[5] = {0}; int no = 0;
        for (const Out& x : outs) { o[no] = x; nb[no + 1] = nb[no] + x.m->N / 16; no++; }
        const int total = nb[no];
        #pragma omp parallel for schedule(dynamic, 2) num_threads(nthreads)
        for (int gb = 0; gb < total; gb++) {
            int i = 0;
            while (gb >= nb[i + 1]) i++;
            const int b = gb - nb[i];
            const Mat& m = *o[i].m;
            for (int m0 = 0; m0 < M; m0 += 8 * MAXTT)
                gemm_rows(m, X + (size_t)m0 * ldx, ldx, std::min(8 * MAXTT, M - m0), o[i].y + (size_t)m0 * o[i].ldy, o[i].ldy, b * 16);
        }
        if (prof && in_draft) t_gemm += now_ms() - tp0;
        if (prof && in_inject) t_inj_gemm += now_ms() - tp0;
    }

    // y[i] = x[i] * rsqrt(mean(x^2)+eps) (* w)
    static void rms_rows(const f32* x, f32* y, int rows, int n, const f32* w, f32 eps) {
        for (int r = 0; r < rows; r++) {
            const f32* xr = x + (size_t)r * n;
            f32* yr = y + (size_t)r * n;
            f32 ss = 0;
            for (int i = 0; i < n; i++) ss += xr[i] * xr[i];
            const f32 s = 1.0f / std::sqrt(ss / n + eps);
            if (w) for (int i = 0; i < n; i++) yr[i] = xr[i] * s * w[i];
            else for (int i = 0; i < n; i++) yr[i] = xr[i] * s;
        }
    }

    // per head rms norm (weight shared across heads) then NEOX rope at position pos
    void head_norm_rope(f32* v, int heads, const f32* w, int pos) const {
        const int dh = c.head_dim, half = dh / 2;
        f32 cs[128], sn[128];
        for (int i = 0; i < half; i++) { f32 a = (f32)pos * inv_freq[i]; cs[i] = std::cos(a); sn[i] = std::sin(a); }
        for (int h = 0; h < heads; h++) {
            f32* x = v + (size_t)h * dh;
            f32 ss = 0;
            for (int i = 0; i < dh; i++) ss += x[i] * x[i];
            const f32 s = 1.0f / std::sqrt(ss / dh + c.eps);
            for (int i = 0; i < dh; i++) x[i] = x[i] * s * w[i];
            for (int i = 0; i < half; i++) {
                const f32 a = x[i], b = x[i + half];
                x[i] = a * cs[i] - b * sn[i];
                x[i + half] = b * cs[i] + a * sn[i];
            }
        }
    }

    bool load(const std::string& dir);
    void reset() { ctx_begin = ctx_end = 0; }
    void inject(const bf16* taps_b, int n, int pos0);
    void draft(int pos0, const bf16* emb_b, bf16* hidden);
};

static std::string slurp(const std::string& p) {
    std::ifstream f(p);
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

bool dflash_draft_host::Impl::load(const std::string& dir) {
    std::string wfile = "dflash_weight.q4nx";
    nlohmann::json cj;
    {
        std::string s = slurp(dir + "/config.json");
        if (!s.empty()) {
            try { cj = nlohmann::json::parse(s); } catch (...) { std::fprintf(stderr, "dflash_draft: bad config.json\n"); return false; }
            if (cj.contains("draft_weight")) wfile = cj["draft_weight"].get<std::string>();
        }
    }
    st_file st;
    if (!st.open(dir + "/" + wfile)) { std::fprintf(stderr, "dflash_draft: cannot read %s/%s\n", dir.c_str(), wfile.c_str()); return false; }

    // metadata: file tensors first (always present), config.json overrides
    auto geti = [&](const char* n, int d) { return st.has(n) ? *(const int32_t*)st.ptr(n) : d; };
    auto getf = [&](const char* n, f32 d) { return st.has(n) ? *(const f32*)st.ptr(n) : d; };
    c.block = geti("dflash.block_size", c.block);
    c.window = geti("dflash.sliding_window", c.window);
    c.eps = getf("dflash.rms_eps", c.eps);
    c.rope_theta = getf("dflash.rope_theta", c.rope_theta);
    c.embed_scale = getf("dflash.embedding_scale", c.embed_scale);
    if (st.has("dflash.target_layer_ids")) {
        const int32_t* t = (const int32_t*)st.ptr("dflash.target_layer_ids");
        for (int i = 0; i < 5; i++) c.taps[i] = t[i];
    }
    if (!cj.is_null()) {
        auto ov = [&](const char* k, int& v) { if (cj.contains(k)) v = cj[k].get<int>(); };
        auto ovf = [&](const char* k, f32& v) { if (cj.contains(k)) v = cj[k].get<f32>(); };
        ov("block_size", c.block); ov("mask_token_id", c.mask_id); ov("num_hidden_layers", c.layers); ov("hidden_size", c.D);
        ov("intermediate_size", c.ffn); ov("num_attention_heads", c.n_q); ov("num_key_value_heads", c.n_kv); ov("head_dim", c.head_dim);
        ov("sliding_window", c.window);
        ovf("rms_norm_eps", c.eps); ovf("rope_theta", c.rope_theta); ovf("embedding_scale", c.embed_scale);
        if (cj.contains("target_layer_ids")) { int i = 0; for (auto& v : cj["target_layer_ids"]) if (i < 5) c.taps[i++] = v.get<int>(); }
    }
    if (c.D % 256 || c.ffn % 256 || c.head_dim != 128 || c.layers < 1 || c.block != 8 || c.n_q % c.n_kv) {
        std::fprintf(stderr, "dflash_draft: unsupported draft shape\n");
        return false;
    }

    auto mat = [&](Mat& m, const std::string& name, int N, int K) {
        if (!st.has(name)) throw std::runtime_error("missing tensor " + name);
        const st_tensor& t = st.t.at(name);
        if (t.dtype == "BF16" && t.shape.size() == 3) load_bf16_blocks(m, (const uint16_t*)st.ptr(name), N, K);
        else load_q4(m, st.ptr(name), N, K);
    };
    auto vec = [&](std::vector<f32>& v, const std::string& name, int n) {
        if (!st.has(name)) throw std::runtime_error("missing tensor " + name);
        const uint16_t* p = (const uint16_t*)st.ptr(name);
        v.resize(n);
        for (int i = 0; i < n; i++) v[i] = bf16_to_f32(p[i]);
    };
    try {
        const int D = c.D, QD = c.n_q * c.head_dim, KD = c.n_kv * c.head_dim;
        mat(fc, "model.fc.weight", D, D * c.n_taps);
        vec(hidden_norm, "model.hidden_norm.weight", D);
        vec(out_norm_f, "model.norm.weight", D);
        out_norm_bf.alloc(D);
        for (int i = 0; i < D; i++) out_norm_bf[i] = f32_to_bf16(out_norm_f[i]);
        L.resize(c.layers);
        for (int l = 0; l < c.layers; l++) {
            const std::string p = "model.layers." + std::to_string(l) + ".";
            mat(L[l].q, p + "self_attn.q_proj.weight", QD, D);
            mat(L[l].k, p + "self_attn.k_proj.weight", KD, D);
            mat(L[l].v, p + "self_attn.v_proj.weight", KD, D);
            mat(L[l].o, p + "self_attn.o_proj.weight", D, QD);
            mat(L[l].gate, p + "mlp.gate_proj.weight", c.ffn, D);
            mat(L[l].up, p + "mlp.up_proj.weight", c.ffn, D);
            mat(L[l].down, p + "mlp.down_proj.weight", D, c.ffn);
            vec(L[l].attn_norm, p + "input_layernorm.weight", D);
            vec(L[l].ffn_norm, p + "post_attention_layernorm.weight", D);
            vec(L[l].q_norm, p + "self_attn.q_norm.weight", c.head_dim);
            vec(L[l].k_norm, p + "self_attn.k_norm.weight", c.head_dim);
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "dflash_draft: %s\n", e.what());
        return false;
    }
    Kc.resize(c.layers); Vc.resize(c.layers);
    for (int l = 0; l < c.layers; l++) { Kc[l].alloc((size_t)RING * c.n_kv * c.head_dim); Vc[l].alloc((size_t)RING * c.n_kv * c.head_dim); }
    inv_freq.resize(c.head_dim / 2);
    for (int i = 0; i < c.head_dim / 2; i++) inv_freq[i] = std::pow(c.rope_theta, -(f32)i / (f32)(c.head_dim / 2));
    nthreads = std::max(1, std::min(16, omp_get_num_procs() / 2));   // physical cores (SMT siblings add nothing to these streaming GEMMs)
    prof = std::getenv("FLM_DFLASH_PROFILE") != nullptr;
    if (const char* e = std::getenv("FLM_DFLASH_THREADS")) nthreads = std::max(1, std::min(32, std::atoi(e)));
    reset();
    loaded = true;
    return true;
}

void dflash_draft_host::Impl::inject(const bf16* taps_b, int n, int pos0) {
    if (!loaded || n <= 0) return;
    if (ctx_end == ctx_begin || pos0 != ctx_end) { ctx_begin = ctx_end = pos0; }   // empty or non-contiguous: start a fresh context here
    const double ti0 = prof ? now_ms() : 0;
    in_inject = prof;
    const int D = c.D, KD = c.n_kv * c.head_dim, TD = D * c.n_taps;
    const uint16_t* tb = (const uint16_t*)taps_b;
    const int CH = 64;     // rows per pass: the weights stream from DRAM once per 64 rows (see gemm())
    std::vector<f32> x((size_t)CH * TD), cc((size_t)CH * D), cn((size_t)CH * D), kk((size_t)CH * KD), vv((size_t)CH * KD);
    for (int r0 = 0; r0 < n; r0 += CH) {
        const int m = std::min(CH, n - r0);
        for (size_t i = 0; i < (size_t)m * TD; i++) x[i] = bf16_to_f32(tb[(size_t)r0 * TD + i]);
        gemm(fc, x.data(), TD, m, cc.data(), D);
        rms_rows(cc.data(), cn.data(), m, D, hidden_norm.data(), c.eps);
        for (int l = 0; l < c.layers; l++) {
            gemm_multi({{&L[l].k, kk.data(), KD}, {&L[l].v, vv.data(), KD}}, cn.data(), D, m);
            for (int r = 0; r < m; r++) {
                const int pos = pos0 + r0 + r;
                head_norm_rope(kk.data() + (size_t)r * KD, c.n_kv, L[l].k_norm.data(), pos);
                const size_t slot = (size_t)(pos % RING) * KD;
                std::memcpy(&Kc[l][slot], kk.data() + (size_t)r * KD, KD * sizeof(f32));
                std::memcpy(&Vc[l][slot], vv.data() + (size_t)r * KD, KD * sizeof(f32));
            }
        }
    }
    ctx_end = pos0 + n;
    if (ctx_end - ctx_begin > RING) ctx_begin = ctx_end - RING;
    if (prof) { t_inj += now_ms() - ti0; n_inj_rows += n; in_inject = false; }
}

void dflash_draft_host::Impl::draft(int pos0, const bf16* emb_b, bf16* hidden) {
    const int D = c.D, B = c.block, QD = c.n_q * c.head_dim, KD = c.n_kv * c.head_dim, dh = c.head_dim, F = c.ffn;
    const int grp = c.n_q / c.n_kv;
    const uint16_t* eb = (const uint16_t*)emb_b;
    std::vector<f32> X((size_t)B * D), H((size_t)B * D), Q((size_t)B * QD), Kb((size_t)B * KD), Vb((size_t)B * KD), A((size_t)B * QD),
        O((size_t)B * D), G((size_t)B * F), U((size_t)B * F);
    for (size_t i = 0; i < (size_t)B * D; i++) X[i] = bf16_to_f32(eb[i]);
    const double tdraft0 = prof ? now_ms() : 0;
    in_draft = prof;
    const int ctx_hi = std::min(ctx_end, pos0);   // ctx positions are < pos0 (anything else would be a stale future position)
    const f32 scale = 1.0f / std::sqrt((f32)dh);

    for (int l = 0; l < c.layers; l++) {
        Layer& ly = L[l];
        rms_rows(X.data(), H.data(), B, D, ly.attn_norm.data(), c.eps);
        gemm_multi({{&ly.q, Q.data(), QD}, {&ly.k, Kb.data(), KD}, {&ly.v, Vb.data(), KD}}, H.data(), D, B);
        for (int r = 0; r < B; r++) {
            head_norm_rope(Q.data() + (size_t)r * QD, c.n_q, ly.q_norm.data(), pos0 + r);
            head_norm_rope(Kb.data() + (size_t)r * KD, c.n_kv, ly.k_norm.data(), pos0 + r);
        }
        const double ta0 = prof ? now_ms() : 0;
        // attention: task = (kv head, query row); keys = ctx ring (window-limited per row) + the whole block (non-causal)
        #pragma omp parallel for schedule(dynamic, 1) num_threads(nthreads)
        for (int task = 0; task < c.n_kv * B; task++) {
            const int kvh = task / B, i = task % B;
            const int qpos = pos0 + i;
            const int lo = std::max(ctx_begin, qpos - c.window + 1);   // k_pos > q_pos - window
            const int nctx = std::max(0, ctx_hi - lo);
            const int nk = nctx + B;
            std::vector<f32> sc((size_t)grp * nk);
            const f32* qh[8];
            for (int g = 0; g < grp; g++) qh[g] = Q.data() + (size_t)i * QD + (size_t)(kvh * grp + g) * dh;
            for (int j = 0; j < nk; j++) {
                const f32* k = (j < nctx) ? &Kc[l][(size_t)((lo + j) % RING) * KD + (size_t)kvh * dh]
                                          : Kb.data() + (size_t)(j - nctx) * KD + (size_t)kvh * dh;
                __m512 kv[8];
                for (int d = 0; d < 8; d++) kv[d] = _mm512_loadu_ps(k + d * 16);
                for (int g = 0; g < grp; g++) {
                    __m512 a0 = _mm512_mul_ps(_mm512_loadu_ps(qh[g]), kv[0]), a1 = _mm512_mul_ps(_mm512_loadu_ps(qh[g] + 16), kv[1]);
                    for (int d = 2; d < 8; d += 2) {
                        a0 = _mm512_fmadd_ps(_mm512_loadu_ps(qh[g] + d * 16), kv[d], a0);
                        a1 = _mm512_fmadd_ps(_mm512_loadu_ps(qh[g] + (d + 1) * 16), kv[d + 1], a1);
                    }
                    sc[(size_t)g * nk + j] = _mm512_reduce_add_ps(_mm512_add_ps(a0, a1)) * scale;
                }
            }
            for (int g = 0; g < grp; g++) {
                f32* sp = &sc[(size_t)g * nk];
                f32 mx = sp[0];
                for (int j = 1; j < nk; j++) mx = std::max(mx, sp[j]);
                const __m512 vmx = _mm512_set1_ps(mx);
                __m512 vsum = _mm512_setzero_ps();
                for (int j = 0; j < nk; j += 16) {
                    const __mmask16 mk = (nk - j >= 16) ? (__mmask16)0xFFFF : (__mmask16)((1u << (nk - j)) - 1);
                    const __m512 e = exp512(_mm512_sub_ps(_mm512_maskz_loadu_ps(mk, sp + j), vmx));
                    _mm512_mask_storeu_ps(sp + j, mk, e);
                    vsum = _mm512_mask_add_ps(vsum, mk, vsum, e);
                }
                const __m512 vinv = _mm512_set1_ps(1.0f / _mm512_reduce_add_ps(vsum));
                for (int j = 0; j < nk; j += 16) {
                    const __mmask16 mk = (nk - j >= 16) ? (__mmask16)0xFFFF : (__mmask16)((1u << (nk - j)) - 1);
                    _mm512_mask_storeu_ps(sp + j, mk, _mm512_mul_ps(_mm512_maskz_loadu_ps(mk, sp + j), vinv));
                }
            }
            __m512 acc[8][8];   // [group][dh/16]
            for (int g = 0; g < grp; g++) for (int d = 0; d < 8; d++) acc[g][d] = _mm512_setzero_ps();
            for (int j = 0; j < nk; j++) {
                const f32* v = (j < nctx) ? &Vc[l][(size_t)((lo + j) % RING) * KD + (size_t)kvh * dh]
                                          : Vb.data() + (size_t)(j - nctx) * KD + (size_t)kvh * dh;
                __m512 vv[8];
                for (int d = 0; d < 8; d++) vv[d] = _mm512_loadu_ps(v + d * 16);
                for (int g = 0; g < grp; g++) {
                    const __m512 p = _mm512_set1_ps(sc[(size_t)g * nk + j]);
                    for (int d = 0; d < 8; d++) acc[g][d] = _mm512_fmadd_ps(p, vv[d], acc[g][d]);
                }
            }
            for (int g = 0; g < grp; g++) {
                f32* out = A.data() + (size_t)i * QD + (size_t)(kvh * grp + g) * dh;
                for (int d = 0; d < 8; d++) _mm512_storeu_ps(out + d * 16, acc[g][d]);
            }
        }
        if (prof) t_attn += now_ms() - ta0;
        gemm(ly.o, A.data(), QD, B, O.data(), D);
        for (size_t i = 0; i < (size_t)B * D; i++) X[i] += O[i];
        rms_rows(X.data(), H.data(), B, D, ly.ffn_norm.data(), c.eps);
        gemm_multi({{&ly.gate, G.data(), F}, {&ly.up, U.data(), F}}, H.data(), D, B);
        #pragma omp parallel for schedule(static) num_threads(nthreads)
        for (int i = 0; i < B * F; i += 16) {
            const __m512 g = _mm512_loadu_ps(&G[i]);
            const __m512 e = exp512(_mm512_sub_ps(_mm512_setzero_ps(), g));
            _mm512_storeu_ps(&G[i], _mm512_mul_ps(_mm512_div_ps(g, _mm512_add_ps(_mm512_set1_ps(1.0f), e)), _mm512_loadu_ps(&U[i])));
        }
        gemm(ly.down, G.data(), F, B, O.data(), D);
        for (size_t i = 0; i < (size_t)B * D; i++) X[i] += O[i];
    }
    if (prof) { t_glue += now_ms() - tdraft0; n_draft++; in_draft = false; }
    uint16_t* ho = (uint16_t*)hidden;
    for (int r = 1; r < B; r++)
        for (int i = 0; i < D; i++) ho[(size_t)(r - 1) * D + i] = f32_to_bf16(X[(size_t)r * D + i]);
}

// ---- public wrappers ---------------------------------------------------------------------------------------------------
dflash_draft_host::dflash_draft_host() : impl_(new Impl()) {}
dflash_draft_host::~dflash_draft_host() {
    if (impl_->prof && impl_->n_draft)
        std::fprintf(stderr, "dflash_draft profile over %d draft() calls: total %.2f ms, gemm %.2f, attention %.2f, other %.2f ms per call\n", impl_->n_draft,
                     impl_->t_glue / impl_->n_draft, impl_->t_gemm / impl_->n_draft, impl_->t_attn / impl_->n_draft,
                     (impl_->t_glue - impl_->t_gemm - impl_->t_attn) / impl_->n_draft);
    if (impl_->prof && impl_->n_inj_rows)
        std::fprintf(stderr, "dflash_draft inject profile over %lld rows: %.3f ms/row, gemm %.3f ms/row\n", impl_->n_inj_rows,
                     impl_->t_inj / impl_->n_inj_rows, impl_->t_inj_gemm / impl_->n_inj_rows);
    delete impl_;
}
bool dflash_draft_host::load(const std::string& dir) { return impl_->load(dir); }
bool dflash_draft_host::loaded() const { return impl_->loaded; }
const dflash_draft_host::config& dflash_draft_host::cfg() const { return impl_->c; }
void dflash_draft_host::reset() { impl_->reset(); }
void dflash_draft_host::truncate(int end) {
    if (end >= impl_->ctx_end) return;
    if (end <= impl_->ctx_begin) { impl_->reset(); return; }
    impl_->ctx_end = end;
}
int dflash_draft_host::ctx_end() const { return impl_->ctx_end; }
void dflash_draft_host::inject(const bf16* taps, int n, int pos0) { impl_->inject(taps, n, pos0); }
void dflash_draft_host::draft(int pos0, const bf16* block_emb, bf16* hidden) { if (impl_->loaded) impl_->draft(pos0, block_emb, hidden); }
const bf16* dflash_draft_host::output_norm() const { return (const bf16*)impl_->out_norm_bf.data(); }
void dflash_draft_host::set_fp32_reference(bool) { /* activations, K/V ring and attention are always fp32; nothing to switch */ }
