// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

// CPU-only self-test of the host kernels the d3 engine's prefix_bidir mode and the 27B shapes rely on:
//
//  1. d3_cpu::attention called twice (state x state, then question x all, causal = false) equals a dense
//     fp64 reference with the block mask (state rows see the state only, question rows see every row),
//     for GQA 4 and GQA 6 (and 8, 1), odd and boundary lengths (S = 1, 127, 128, 129; L - S = 1, ...).
//  2. a state row's output does not depend on the question rows (perturbing question K / V leaves state
//     outputs bit-identical), and a prompt without a split (S = 0 or S >= L) is one full bidirectional call.
//  3. the host gated delta rule against a naive fp64 version for value / key head ratios 1, 2 and 3
//     (d3-turbo: 16 key, 48 value heads), and cpu_mm with 48 outputs (the alpha / beta projections).
//
//   make test_cpu && ../../build/d3_npu/bin/d3_test_prefix_bidir_cpu
//
// Needs no device. Exit code 0 = all passed.
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "d3_prefill_cpu.hpp"

int d3_physical_cores() { return 0; }   // d3_npu.cpp's, not linked here

static int g_fail = 0, g_checks = 0;
#define CHECK(cond, ...)                                                    \
    do {                                                                    \
        g_checks++;                                                         \
        if (!(cond)) {                                                      \
            g_fail++;                                                       \
            std::printf("FAIL %s:%d: ", __FILE__, __LINE__);                \
            std::printf(__VA_ARGS__);                                       \
            std::printf("\n");                                              \
        }                                                                   \
    } while (0)

static std::vector<bf16> rand_bf16(size_t n, std::mt19937& rng, float scale) {
    std::normal_distribution<float> nd(0.0f, scale);
    std::vector<bf16> v(n);
    for (auto& x : v) x = bf16(nd(rng));
    return v;
}

// dense fp64 reference with the prefix block mask: rows < S see keys [0, S), rows >= S see [0, L)
static void ref_attention(std::vector<double>& out, const std::vector<bf16>& q, const std::vector<bf16>& k,
                          const std::vector<bf16>& v, int L, int S, int H, int NKV, int DH) {
    const int group = H / NKV;
    const double scale = 1.0 / std::sqrt((double)DH);
    out.assign((size_t)L * H * DH, 0.0);
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i = 0; i < L; i++) {
        for (int h = 0; h < H; h++) {
            const int hk = h / group;
            const int lim = (S > 0 && S < L && i < S) ? S : L;
            std::vector<double> p(lim);
            double m = -1e300;
            for (int j = 0; j < lim; j++) {
                double a = 0;
                for (int d = 0; d < DH; d++)
                    a += (double)(float)q[((size_t)i * H + h) * DH + d] * (double)(float)k[((size_t)j * NKV + hk) * DH + d];
                p[j] = a * scale;
                m = std::max(m, p[j]);
            }
            double sum = 0;
            for (int j = 0; j < lim; j++) { p[j] = std::exp(p[j] - m); sum += p[j]; }
            for (int d = 0; d < DH; d++) {
                double a = 0;
                for (int j = 0; j < lim; j++) a += p[j] * (double)(float)v[((size_t)j * NKV + hk) * DH + d];
                out[((size_t)i * H + h) * DH + d] = a / sum;
            }
        }
    }
}

// the engine's host path: one call without a split, else state x state then question x all
static void host_attention(std::vector<bf16>& out, const std::vector<bf16>& q, const std::vector<bf16>& k,
                           const std::vector<bf16>& v, int L, int S, int H, int NKV, int DH, std::vector<float>& kt) {
    out.assign((size_t)L * H * DH, bf16(0.0f));
    const float scale = 1.0f / std::sqrt((float)DH);
    const int DQ = H * DH, DK = NKV * DH;
    if (S > 0 && S < L) {
        d3_cpu::attention(out.data(), q.data(), k.data(), v.data(), S, S, H, NKV, DH, 0, 0, DK, DK, scale, false, kt);
        d3_cpu::attention(out.data() + (size_t)S * DQ, q.data() + (size_t)S * DQ, k.data(), v.data(), L - S, L, H, NKV, DH,
                          0, 0, DK, DK, scale, false, kt);
    } else {
        d3_cpu::attention(out.data(), q.data(), k.data(), v.data(), L, L, H, NKV, DH, 0, 0, DK, DK, scale, false, kt);
    }
}

static void test_attention() {
    std::mt19937 rng(1234);
    std::vector<float> kt;
    struct Shape { int H, NKV; };
    const Shape shapes[] = {{8, 2}, {24, 4}, {12, 2}, {4, 4}, {6, 1}};   // GQA 4, 6, 6, 1, 6
    const int DH = 256;
    struct Case { int S, L; };
    const Case cases[] = {{1, 2}, {1, 130}, {127, 128}, {127, 300}, {128, 129}, {128, 256}, {129, 130}, {129, 400},
                          {3, 4}, {200, 201}, {64, 177}, {0, 100}, {100, 100}, {100, 150}};
    double worst_rel = 0;
    for (const auto& sh : shapes) {
        for (const auto& c : cases) {
            if (sh.H * c.L > 24 * 400) continue;
            auto q = rand_bf16((size_t)c.L * sh.H * DH, rng, 1.0f);
            auto k = rand_bf16((size_t)c.L * sh.NKV * DH, rng, 1.0f);
            auto v = rand_bf16((size_t)c.L * sh.NKV * DH, rng, 1.0f);
            std::vector<bf16> got;
            std::vector<double> ref;
            host_attention(got, q, k, v, c.L, c.S, sh.H, sh.NKV, DH, kt);
            ref_attention(ref, q, k, v, c.L, c.S, sh.H, sh.NKV, DH);
            // fp32 accumulation + bf16 rounding of the output: half an ulp (2^-9) relative, plus slack
            size_t bad = 0;
            double max_rel = 0;
            for (size_t i = 0; i < ref.size(); i++) {
                const double g = (double)(float)got[i], e = std::fabs(g - ref[i]);
                const double tol = 4.2e-3 * std::fabs(ref[i]) + 6e-3; // + bf16 P and V operands (2^-9 each, averaged over keys)
                if (e > tol) bad++;
                max_rel = std::max(max_rel, e / (std::fabs(ref[i]) + 1e-3));
            }
            worst_rel = std::max(worst_rel, max_rel);
            CHECK(bad == 0, "attention H=%d NKV=%d (GQA %d) S=%d L=%d: %zu of %zu outputs beyond tolerance",
                  sh.H, sh.NKV, sh.H / sh.NKV, c.S, c.L, bad, ref.size());

            if (c.S > 0 && c.S < c.L) {
                // state rows must not see question rows: perturb question K / V, state outputs stay bit-identical
                auto k2 = k, v2 = v;
                for (size_t r = (size_t)c.S; r < (size_t)c.L; r++)
                    for (int d = 0; d < sh.NKV * DH; d++) {
                        k2[r * sh.NKV * DH + d] = bf16(3.0f * (float)k2[r * sh.NKV * DH + d] + 0.5f);
                        v2[r * sh.NKV * DH + d] = bf16(-2.0f * (float)v2[r * sh.NKV * DH + d] + 1.0f);
                    }
                std::vector<bf16> got2;
                host_attention(got2, q, k2, v2, c.L, c.S, sh.H, sh.NKV, DH, kt);
                const size_t state_elems = (size_t)c.S * sh.H * DH;
                CHECK(std::memcmp(got.data(), got2.data(), state_elems * sizeof(bf16)) == 0,
                      "state rows depend on question rows (S=%d L=%d GQA %d)", c.S, c.L, sh.H / sh.NKV);
                CHECK(std::memcmp(got.data() + state_elems, got2.data() + state_elems, (got.size() - state_elems) * sizeof(bf16)) != 0,
                      "question rows ignore question keys (S=%d L=%d)", c.S, c.L);
            }
        }
    }
    std::printf("attention: worst |err| / (|ref| + 1e-3) = %.3g\n", worst_rel);

    // no split == one full bidirectional call, bit for bit (S = 0 and S >= L)
    {
        const int H = 24, NKV = 4, L = 150;
        auto q = rand_bf16((size_t)L * H * DH, rng, 1.0f);
        auto k = rand_bf16((size_t)L * NKV * DH, rng, 1.0f);
        auto v = rand_bf16((size_t)L * NKV * DH, rng, 1.0f);
        std::vector<bf16> a, b, c;
        host_attention(a, q, k, v, L, 0, H, NKV, DH, kt);
        host_attention(b, q, k, v, L, L, H, NKV, DH, kt);
        host_attention(c, q, k, v, L, L + 7, H, NKV, DH, kt);
        CHECK(std::memcmp(a.data(), b.data(), a.size() * 2) == 0 && std::memcmp(a.data(), c.data(), a.size() * 2) == 0,
              "S=0 / S>=L differ from the unsplit call");
    }

    // a query's arithmetic does not depend on how the work is cut: the question call alone on a suffix
    // gives the same rows as the same rows inside a longer call (task shape differs, results must not)
    {
        const int H = 24, NKV = 4, L = 211, S = 77;
        auto q = rand_bf16((size_t)L * H * DH, rng, 1.0f);
        auto k = rand_bf16((size_t)L * NKV * DH, rng, 1.0f);
        auto v = rand_bf16((size_t)L * NKV * DH, rng, 1.0f);
        const int DQ = H * DH, DK = NKV * DH;
        const float scale = 1.0f / std::sqrt((float)DH);
        std::vector<bf16> full((size_t)L * DQ), part((size_t)(L - S) * DQ), one(DQ);
        d3_cpu::attention(full.data(), q.data(), k.data(), v.data(), L, L, H, NKV, DH, 0, 0, DK, DK, scale, false, kt);
        d3_cpu::attention(part.data(), q.data() + (size_t)S * DQ, k.data(), v.data(), L - S, L, H, NKV, DH, 0, 0, DK, DK, scale, false, kt);
        CHECK(std::memcmp(part.data(), full.data() + (size_t)S * DQ, part.size() * sizeof(bf16)) == 0, "task shape changes a query's result");
        d3_cpu::attention(one.data(), q.data() + (size_t)(L - 1) * DQ, k.data(), v.data(), 1, L, H, NKV, DH, 0, 0, DK, DK, scale, false, kt);
        CHECK(std::memcmp(one.data(), full.data() + (size_t)(L - 1) * DQ, DQ * sizeof(bf16)) == 0, "single query row differs");
    }

    // bad shapes are refused at run time (release builds define NDEBUG, where an assert is gone)
    {
        std::vector<bf16> z(4096);
        bool threw = false;
        try {
            d3_cpu::attention(z.data(), z.data(), z.data(), z.data(), 1, 1, 6, 4, 16, 0, 0, 64, 64, 1.0f, false, kt);
        } catch (const std::invalid_argument&) { threw = true; }
        CHECK(threw, "heads not a multiple of kv heads was accepted");
    }
}

static void test_gdn() {
    std::mt19937 rng(77);
    std::uniform_real_distribution<float> ud(0.0f, 1.0f);
    const int DK = 128, DV = 128, L = 37;
    for (int Hqk : {2, 4, 16}) {
        for (int ratio : {1, 2, 3}) {
            const int Hv = Hqk * ratio;
            if (Hv > 48) continue;
            const int ld = 2 * Hqk * DK + Hv * DV;
            auto qkv = rand_bf16((size_t)L * ld, rng, 0.5f);
            // q and k are L2-normalised per head by the engine before the rule; do it here too (as the engine feeds it)
            for (int l = 0; l < L; l++)
                for (int part = 0; part < 2; part++)
                    for (int h = 0; h < Hqk; h++) {
                        bf16* p = qkv.data() + (size_t)l * ld + (size_t)part * Hqk * DK + (size_t)h * DK;
                        double n = 0;
                        for (int i = 0; i < DK; i++) n += (double)(float)p[i] * (double)(float)p[i];
                        n = std::sqrt(n) + 1e-6;
                        for (int i = 0; i < DK; i++) p[i] = bf16((float)((double)(float)p[i] / n));
                    }
            std::vector<float> alpha((size_t)L * Hv), beta((size_t)L * Hv);
            for (auto& a : alpha) a = 0.85f + 0.14f * ud(rng);
            for (auto& b : beta) b = ud(rng);
            std::vector<bf16> out((size_t)L * Hv * DV);
            std::vector<float> S((size_t)Hv * DK * DV, 0.0f);
            d3_cpu::gated_delta_net(out.data(), S.data(), qkv.data(), ld, 0, Hqk * DK, 2 * Hqk * DK, alpha.data(), beta.data(),
                                    L, Hqk, Hv, DK, DV);
            // naive fp64: value head hv reads key head hv / ratio (the HF order)
            double worst = 0, scale_ref = 0;
            std::vector<double> ref((size_t)L * Hv * DV);
            for (int hv = 0; hv < Hv; hv++) {
                const int hq = hv / ratio;
                std::vector<double> St((size_t)DK * DV, 0.0);
                for (int l = 0; l < L; l++) {
                    const bf16* row = qkv.data() + (size_t)l * ld;
                    std::vector<double> kv(DV, 0.0), delta(DV);
                    for (int i = 0; i < DK; i++)
                        for (int j = 0; j < DV; j++) {
                            St[(size_t)i * DV + j] *= alpha[(size_t)l * Hv + hv];
                            kv[j] += St[(size_t)i * DV + j] * (double)(float)row[Hqk * DK + hq * DK + i];
                        }
                    for (int j = 0; j < DV; j++)
                        delta[j] = ((double)(float)row[2 * Hqk * DK + hv * DV + j] - kv[j]) * beta[(size_t)l * Hv + hv];
                    for (int j = 0; j < DV; j++) {
                        double o = 0;
                        for (int i = 0; i < DK; i++) {
                            St[(size_t)i * DV + j] += (double)(float)row[Hqk * DK + hq * DK + i] * delta[j];
                            o += St[(size_t)i * DV + j] * (double)(float)row[hq * DK + i] / std::sqrt((double)DK);
                        }
                        ref[((size_t)l * Hv + hv) * DV + j] = o;
                    }
                }
            }
            for (size_t i = 0; i < ref.size(); i++) scale_ref = std::max(scale_ref, std::fabs(ref[i]));
            for (size_t i = 0; i < ref.size(); i++) worst = std::max(worst, std::fabs((double)(float)out[i] - ref[i]));
            CHECK(worst <= 6e-3 * scale_ref + 1e-6, "gated_delta_net Hqk=%d Hv=%d: max abs error %.3g (max |ref| %.3g)", Hqk, Hv, worst, scale_ref);
            std::printf("gated_delta_net Hqk=%2d Hv=%2d: max abs error %.3g of max |ref| %.3g\n", Hqk, Hv, worst, scale_ref);
        }
    }
    // alpha / beta projection with 48 outputs (d3-turbo's value heads): fast path == naive
    for (int N : {32, 48, 16, 6}) {
        const int M = 19, K = 5120 / 8 + 3;   // odd K: the scalar tail
        auto x = rand_bf16((size_t)M * K, rng, 1.0f);
        auto w = rand_bf16((size_t)N * K, rng, 0.05f);
        std::vector<float> y((size_t)M * N);
        cpu_func::_cpu_mm<bf16>(y.data(), w.data(), x.data(), M, K, N);
        double worst = 0;
        for (int m = 0; m < M; m++)
            for (int n = 0; n < N; n++) {
                double a = 0;
                for (int kk = 0; kk < K; kk++) a += (double)(float)x[(size_t)m * K + kk] * (double)(float)w[(size_t)n * K + kk];
                worst = std::max(worst, std::fabs(a - y[(size_t)m * N + n]));
            }
        CHECK(worst < 1e-3, "cpu_mm N=%d: max abs error %.3g", N, worst);
    }
}

int main() {
    test_attention();
    test_gdn();
    std::printf("%d checks, %d failed\n", g_checks, g_fail);
    return g_fail ? 1 : 0;
}
