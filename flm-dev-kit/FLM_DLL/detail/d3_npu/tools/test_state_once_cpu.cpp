// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

// CPU-only self-test of the host kernels of the d3 engine's state-once pass (encode_state_once):
//
//  1. d3_cpu::attention_prefix (a question block over [cached state K,V | its own K,V], bidirectional)
//     is BIT-IDENTICAL to the question rows of d3_cpu::attention over the joint key list [state; own]
//     (what the per-row prefix_bidir path computes), for GQA 1 / 4 / 6, S in {0, 1, 15, 16, 17, 127, 128, 129,
//     4096}, Lq in {1, 16, 17, 110}; and within fp32 + bf16 rounding of a dense fp64 reference (S <= 129).
//  2. the same block's result does not depend on the other blocks sharing the state (one shared state K^T,
//     one call per block, any order), nor on repetition.
//  3. d3_cpu::conv1d_silu_carry continuing from the state's conv tail equals the question rows of one conv
//     over [state | question] (bit for bit), for states shorter than the kernel (S = 1, 2) too; the tail it
//     hands on is the last KS - 1 rows of the conv input.
//  4. d3_cpu::gated_delta_net_carry continuing from the state's recurrent state equals the question rows of
//     one pass over [state | question] (bit for bit), and so does the state after it, for 16 key / 48 value
//     heads (d3-turbo), 16 / 32 (d3-mini) and 2 / 4 heads.
//
//   make state_once_test && ../../build/d3_npu/bin/d3_test_state_once_cpu
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

// dense fp64 reference: every query row sees all S + Lq keys
static void ref_question_attention(std::vector<double>& out, const std::vector<bf16>& q, const std::vector<bf16>& k,
                                   const std::vector<bf16>& v, int Lq, int J, int H, int NKV, int DH) {
    const int group = H / NKV;
    const double scale = 1.0 / std::sqrt((double)DH);
    out.assign((size_t)Lq * H * DH, 0.0);
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i = 0; i < Lq; i++) {
        for (int h = 0; h < H; h++) {
            const int hk = h / group;
            std::vector<double> p(J);
            double m = -1e300;
            for (int j = 0; j < J; j++) {
                double a = 0;
                for (int d = 0; d < DH; d++)
                    a += (double)(float)q[((size_t)i * H + h) * DH + d] * (double)(float)k[((size_t)j * NKV + hk) * DH + d];
                p[j] = a * scale;
                m = std::max(m, p[j]);
            }
            double sum = 0;
            for (int j = 0; j < J; j++) { p[j] = std::exp(p[j] - m); sum += p[j]; }
            for (int d = 0; d < DH; d++) {
                double a = 0;
                for (int j = 0; j < J; j++) a += p[j] * (double)(float)v[((size_t)j * NKV + hk) * DH + d];
                out[((size_t)i * H + h) * DH + d] = a / sum;
            }
        }
    }
}

static void test_attention_prefix() {
    std::mt19937 rng(4321);
    const int DH = 256;
    struct Shape { int H, NKV; };
    const Shape shapes[] = {{4, 4}, {16, 4}, {24, 4}, {12, 2}};   // GQA 1, 4, 6, 6
    const int Ss[] = {0, 1, 15, 16, 17, 127, 128, 129, 4096};
    const int Lqs[] = {1, 16, 17, 110};
    std::vector<float> kt_ref, scratch, st_kt;
    double worst_rel = 0;
    for (const auto& sh : shapes) {
        const int DQ = sh.H * DH, DK = sh.NKV * DH;
        const float scale = 1.0f / std::sqrt((float)DH);
        for (int S : Ss) {
            if (S == 4096 && sh.H != 24 && sh.H != 16) continue;   // two big shapes are enough
            // one state and a pool of question blocks that all continue from it
            auto sk = rand_bf16((size_t)S * DK, rng, 1.0f), sv = rand_bf16((size_t)S * DK, rng, 1.0f);
            st_kt.resize(d3_cpu::keys_t_floats(std::max(S, 1), sh.NKV, DH));
            if (S > 0) d3_cpu::transpose_keys(st_kt.data(), sk.data(), S, DK, sh.NKV, DH);
            std::vector<std::vector<bf16>> firsts;
            for (int Lq : Lqs) {
                auto q = rand_bf16((size_t)Lq * DQ, rng, 1.0f);
                auto ok = rand_bf16((size_t)Lq * DK, rng, 1.0f), ov = rand_bf16((size_t)Lq * DK, rng, 1.0f);
                std::vector<bf16> got((size_t)Lq * DQ), want((size_t)Lq * DQ), again((size_t)Lq * DQ);
                d3_cpu::attention_prefix(got.data(), q.data(), ok.data(), ov.data(), Lq, st_kt.data(), sv.data(), S,
                                         sh.H, sh.NKV, DH, DK, DK, DK, scale, scratch);
                // the per-row path: the question rows over the joint key list [state; own]
                std::vector<bf16> k((size_t)(S + Lq) * DK), v((size_t)(S + Lq) * DK);
                memcpy(k.data(), sk.data(), (size_t)S * DK * 2); memcpy(k.data() + (size_t)S * DK, ok.data(), (size_t)Lq * DK * 2);
                memcpy(v.data(), sv.data(), (size_t)S * DK * 2); memcpy(v.data() + (size_t)S * DK, ov.data(), (size_t)Lq * DK * 2);
                d3_cpu::attention(want.data(), q.data(), k.data(), v.data(), Lq, S + Lq, sh.H, sh.NKV, DH, 0, 0, DK, DK, scale, false, kt_ref);
                CHECK(std::memcmp(got.data(), want.data(), got.size() * sizeof(bf16)) == 0,
                      "attention_prefix != joint attention (GQA %d S=%d Lq=%d)", sh.H / sh.NKV, S, Lq);
                // repeat with a different scratch history: same bits
                d3_cpu::attention_prefix(again.data(), q.data(), ok.data(), ov.data(), Lq, st_kt.data(), sv.data(), S,
                                         sh.H, sh.NKV, DH, DK, DK, DK, scale, scratch);
                CHECK(std::memcmp(got.data(), again.data(), got.size() * sizeof(bf16)) == 0, "attention_prefix not repeatable (S=%d Lq=%d)", S, Lq);
                if (S <= 129 && Lq <= 17 + 100) {
                    std::vector<double> ref;
                    ref_question_attention(ref, q, k, v, Lq, S + Lq, sh.H, sh.NKV, DH);
                    size_t bad = 0;
                    for (size_t i = 0; i < ref.size(); i++) {
                        const double g = (double)(float)got[i], e = std::fabs(g - ref[i]);
                        if (e > 4.2e-3 * std::fabs(ref[i]) + 6e-3) bad++;
                        worst_rel = std::max(worst_rel, e / (std::fabs(ref[i]) + 1e-3));
                    }
                    CHECK(bad == 0, "attention_prefix GQA %d S=%d Lq=%d: %zu of %zu outputs beyond the fp64 tolerance", sh.H / sh.NKV, S, Lq, bad, ref.size());
                }
                if (Lq == 17 || Lq == 110) firsts.push_back(std::move(got));
            }
            // blocks sharing a state do not change each other: run Lq = 17 again after the 110 block (scratch grown)
            {
                const int Lq = 17;
                auto q = rand_bf16((size_t)Lq * DQ, rng, 1.0f);
                auto ok = rand_bf16((size_t)Lq * DK, rng, 1.0f), ov = rand_bf16((size_t)Lq * DK, rng, 1.0f);
                std::vector<bf16> a((size_t)Lq * DQ), b((size_t)Lq * DQ);
                std::vector<float> fresh;
                d3_cpu::attention_prefix(a.data(), q.data(), ok.data(), ov.data(), Lq, st_kt.data(), sv.data(), S, sh.H, sh.NKV, DH, DK, DK, DK, scale, scratch);
                d3_cpu::attention_prefix(b.data(), q.data(), ok.data(), ov.data(), Lq, st_kt.data(), sv.data(), S, sh.H, sh.NKV, DH, DK, DK, DK, scale, fresh);
                CHECK(std::memcmp(a.data(), b.data(), a.size() * sizeof(bf16)) == 0, "scratch history changes a block (S=%d)", S);
            }
        }
    }
    std::printf("attention_prefix: worst |err| / (|ref| + 1e-3) vs fp64 = %.3g\n", worst_rel);

    // a wrong shape is refused at run time
    {
        std::vector<bf16> z(4096);
        std::vector<float> s;
        bool threw = false;
        try {
            d3_cpu::attention_prefix(z.data(), z.data(), z.data(), z.data(), 1, nullptr, nullptr, 0, 6, 4, 16, 64, 64, 64, 1.0f, s);
        } catch (const std::invalid_argument&) { threw = true; }
        CHECK(threw, "heads not a multiple of kv heads was accepted");
        threw = false;
        try {
            d3_cpu::attention_prefix(z.data(), z.data(), z.data(), z.data(), 1, nullptr, z.data(), 5, 4, 4, 16, 64, 64, 64, 1.0f, s);
        } catch (const std::invalid_argument&) { threw = true; }
        CHECK(threw, "a state without K^T was accepted");
    }
}

// l2-normalise q and k heads as the engine does before the delta rule
static void l2norm_qk(std::vector<bf16>& qkv, int rows, int ld, int Hqk, int DK) {
    for (int l = 0; l < rows; l++)
        for (int part = 0; part < 2; part++)
            for (int h = 0; h < Hqk; h++) {
                bf16* p = qkv.data() + (size_t)l * ld + (size_t)part * Hqk * DK + (size_t)h * DK;
                double n = 0;
                for (int i = 0; i < DK; i++) n += (double)(float)p[i] * (double)(float)p[i];
                n = std::sqrt(n) + 1e-6;
                for (int i = 0; i < DK; i++) p[i] = bf16((float)((double)(float)p[i] / n));
            }
}

static void test_conv_carry() {
    std::mt19937 rng(99);
    const int KS = 4;
    for (int C : {64, 10240 / 8 + 5}) {   // a multiple of 16 and one with a scalar tail
        auto w = rand_bf16((size_t)KS * C, rng, 0.5f);
        for (int S : {1, 2, 3, 4, 17, 130}) {
            for (int Lq : {1, 5, 40}) {
                auto rows = rand_bf16((size_t)(S + Lq) * C, rng, 1.0f);
                std::vector<bf16> joint_in((size_t)(S + Lq + KS - 1) * C, bf16(0.0f)), joint((size_t)(S + Lq) * C);
                memcpy(joint_in.data() + (size_t)(KS - 1) * C, rows.data(), rows.size() * 2);
                d3_cpu::conv1d_silu(joint.data(), joint_in.data(), w.data(), S + Lq, C, KS);
                std::vector<bf16> conv_in, ys((size_t)S * C), yq((size_t)Lq * C), tail((size_t)(KS - 1) * C);
                d3_cpu::conv1d_silu_carry(ys.data(), rows.data(), nullptr, w.data(), S, C, KS, conv_in, tail.data());
                CHECK(std::memcmp(ys.data(), joint.data(), ys.size() * 2) == 0, "state conv differs (S=%d C=%d)", S, C);
                // the tail is the last KS - 1 conv inputs of the state
                CHECK(std::memcmp(tail.data(), joint_in.data() + (size_t)S * C, tail.size() * 2) == 0, "conv tail wrong (S=%d)", S);
                d3_cpu::conv1d_silu_carry(yq.data(), rows.data() + (size_t)S * C, tail.data(), w.data(), Lq, C, KS, conv_in);
                CHECK(std::memcmp(yq.data(), joint.data() + (size_t)S * C, yq.size() * 2) == 0, "question conv differs (S=%d Lq=%d C=%d)", S, Lq, C);
            }
        }
    }
}

static void test_gdn_carry() {
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> ud(0.0f, 1.0f);
    const int DK = 128, DV = 128;
    struct Cfg { int Hqk, Hv; };
    for (Cfg c : {Cfg{16, 48}, Cfg{16, 32}, Cfg{2, 4}, Cfg{4, 4}}) {
        const int Hqk = c.Hqk, Hv = c.Hv, ld = 2 * Hqk * DK + Hv * DV;
        for (int S : {1, 2, 17, 130, 600}) {
            for (int Lq : {1, 9, 110}) {
                const int L = S + Lq;
                auto qkv = rand_bf16((size_t)L * ld, rng, 0.5f);
                l2norm_qk(qkv, L, ld, Hqk, DK);
                std::vector<float> alpha((size_t)L * Hv), beta((size_t)L * Hv);
                for (auto& a : alpha) a = 0.85f + 0.14f * ud(rng);
                for (auto& b : beta) b = ud(rng);
                const size_t sn = (size_t)Hv * DK * DV;
                // one pass over [state | question]
                std::vector<bf16> joint((size_t)L * Hv * DV);
                std::vector<float> Sj(sn, 0.0f);
                d3_cpu::gated_delta_net(joint.data(), Sj.data(), qkv.data(), ld, 0, Hqk * DK, 2 * Hqk * DK, alpha.data(), beta.data(),
                                        L, Hqk, Hv, DK, DV);
                // the state alone, its recurrent state saved; then the question block from the cache
                std::vector<bf16> os((size_t)S * Hv * DV), oq((size_t)Lq * Hv * DV);
                std::vector<float> Sw(sn), Scache(sn);
                d3_cpu::gated_delta_net_carry(os.data(), Sw.data(), nullptr, qkv.data(), ld, 0, Hqk * DK, 2 * Hqk * DK, alpha.data(), beta.data(),
                                              S, Hqk, Hv, DK, DV);
                Scache = Sw;
                // two blocks continuing from one cache must not disturb it (the cache is read only)
                for (int rep = 0; rep < 2; rep++) {
                    d3_cpu::gated_delta_net_carry(oq.data(), Sw.data(), Scache.data(), qkv.data() + (size_t)S * ld, ld, 0, Hqk * DK, 2 * Hqk * DK,
                                                  alpha.data() + (size_t)S * Hv, beta.data() + (size_t)S * Hv, Lq, Hqk, Hv, DK, DV);
                    CHECK(std::memcmp(oq.data(), joint.data() + (size_t)S * Hv * DV, oq.size() * 2) == 0,
                          "question delta rule differs from the joint pass (Hqk=%d Hv=%d S=%d Lq=%d rep %d)", Hqk, Hv, S, Lq, rep);
                    CHECK(std::memcmp(Sw.data(), Sj.data(), sn * sizeof(float)) == 0, "final recurrent state differs (S=%d Lq=%d)", S, Lq);
                }
                CHECK(std::memcmp(os.data(), joint.data(), os.size() * 2) == 0, "state rows differ (S=%d)", S);
            }
        }
    }
}

int main() {
    test_attention_prefix();
    test_conv_carry();
    test_gdn_carry();
    std::printf("%d checks, %d failed\n", g_checks, g_fail);
    return g_fail ? 1 : 0;
}
