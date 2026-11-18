// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file tools/verify_cpu_simd.cpp
/// \brief `common/cpu_simd.hpp` against the scalar loops it replaced.
///
/// No device and no model -- this is a pure CPU check, and it is the one that
/// licenses the claim in cpu_simd.hpp's header: the vectorised marshalling is
/// **bit-identical** to what module 1 used to run, so swapping it in cannot
/// move a cosine or an argmax.
///
/// Two halves:
///
///  1. **Exhaustive.** All 2^32 float bit patterns through `f32_to_bf16`,
///     compared against `biovault::bfloat16_t(float)` one at a time. Not a
///     sample: the disagreements this is guarding against live in the corners
///     (denormals, the ties that round-to-even decides, NaN payloads), and
///     those are exactly the ones a random draw over plausible activations
///     never produces. ~30 s on 32 threads.
///
///  2. **The wrappers.** Every entry point over a pseudo-random buffer at
///     lengths 0..129 and at the real ones, `memcmp`ed against the scalar body
///     written out longhand. This covers the 16-lane tail, which the exhaustive
///     half does not touch, and the reads at `n`-not-a-multiple-of-16 that a
///     ragged band would produce.
///
/// **Build it with the module's own flags.** The exhaustive half is sensitive
/// to `-ffast-math`: `-ffinite-math-only` folds the constructor's `FP_NAN` and
/// `FP_INFINITE` arms away, so which answer is "correct" for a signaling NaN
/// depends on how the library was compiled. Compiling this at different flags
/// than `libqwen3_8mtp_npu.so` tests a conversion nobody runs.
///
///     source ~/setup.sh
///     g++-13 -std=c++20 -O3 -fopenmp \
///       -mavx512f -mavx512vl -mavx512bw -mavx512dq -mfma -march=native \
///       -ffast-math -I. -I../../include -I../q4nx \
///       tools/verify_cpu_simd.cpp -o /tmp/verify_cpu_simd
///     /tmp/verify_cpu_simd
#include "common/cpu_simd.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

int failures = 0;

void report(const char* what, long long bad, long long total) {
    std::printf("  %-34s %s  (%lld / %lld differ)\n", what,
                bad ? "[FAIL]" : "[ ok ]", bad, total);
    if (bad) failures++;
}

// The scalar bodies, longhand, exactly as they stood before the port.
void ref_f32_to_bf16(bf16* d, const float* s, size_t n) {
    for (size_t i = 0; i < n; i++) d[i] = (bf16)s[i];
}
void ref_bf16_to_f32(float* d, const bf16* s, size_t n) {
    for (size_t i = 0; i < n; i++) d[i] = (float)s[i];
}
void ref_bf16_to_f32_add(float* d, const bf16* s, const float* r, size_t n) {
    for (size_t i = 0; i < n; i++) d[i] = (float)s[i] + r[i];
}
void ref_bf16_mul_to_bf16(bf16* d, const bf16* a, const bf16* b, size_t n) {
    for (size_t i = 0; i < n; i++) d[i] = (bf16)((float)a[i] * (float)b[i]);
}
void ref_bf16_mul_to_f32(float* d, const bf16* a, const bf16* b, size_t n) {
    for (size_t i = 0; i < n; i++) d[i] = (float)a[i] * (float)b[i];
}
int ref_argmax_f32(const float* v, int n) {
    int best = 0;
    for (int i = 1; i < n; i++)
        if (v[i] > v[best]) best = i;
    return best;
}
int ref_argmax_bf16(const bf16* v, int n) {
    int   best = 0;
    float bv   = (float)v[0];
    for (int i = 1; i < n; i++) {
        const float x = (float)v[i];
        if (x > bv) { bv = x; best = i; }
    }
    return best;
}

} // namespace

int main() {
#ifdef QWEN3_8MTP_CPU_SIMD
    std::puts("cpu_simd: AVX-512 path compiled in");
#else
    std::puts("cpu_simd: NO SIMD PATH -- built without AVX-512F/DQ, so this "
              "run only tests the scalar fallback against itself");
#endif

    // ---- 1. exhaustive ----------------------------------------------------
    std::puts("\n-- f32 -> bf16 over all 2^32 bit patterns --");
    long long bad = 0;
#pragma omp parallel for schedule(static) reduction(+ : bad)
    for (long long blk = 0; blk < (1LL << 22); blk++) {
        // 1024 floats a block, so the wrapper's own loop (and its tail path at
        // the very end of the block) is what runs, not a hand-written kernel.
        constexpr int N = 1024;
        float  in[N];
        bf16   got[N], want[N];
        const uint32_t base = (uint32_t)(blk * N);
        for (int i = 0; i < N; i++) {
            const uint32_t u = base + (uint32_t)i;
            std::memcpy(&in[i], &u, 4);
        }
        q38simd::f32_to_bf16(got, in, N);
        ref_f32_to_bf16(want, in, N);
        for (int i = 0; i < N; i++) {
            uint16_t g, w;
            std::memcpy(&g, &got[i], 2);
            std::memcpy(&w, &want[i], 2);
            if (g != w) {
                if (bad < 6) {
                    const uint32_t u = base + (uint32_t)i;
                    std::printf("    u=%08x  scalar=%04x  simd=%04x\n", u, w, g);
                }
                bad++;
            }
        }
    }
    report("f32_to_bf16, exhaustive", bad, 1LL << 32);

    // ---- 2. the wrappers, including the ragged tail -----------------------
    std::puts("\n-- every entry point, lengths 0..129 and the real ones --");
    std::mt19937 rng(20260922);
    // Draw over the exponent range activations actually occupy, then widen the
    // draw with occasional zeros and signs so the masked arm gets exercised.
    std::uniform_real_distribution<float> uni(-8.0f, 8.0f);

    constexpr size_t MAX = 17408;
    std::vector<float> fa(MAX), fb(MAX), o1(MAX), o2(MAX);
    std::vector<bf16>  ba(MAX), bb(MAX), c1(MAX), c2(MAX);
    for (size_t i = 0; i < MAX; i++) {
        fa[i] = (i % 97 == 0) ? 0.0f : uni(rng);
        fb[i] = uni(rng);
        ba[i] = (bf16)fa[i];
        bb[i] = (bf16)fb[i];
    }

    std::vector<size_t> lens;
    for (size_t n = 0; n <= 129; n++) lens.push_back(n);
    for (size_t n : {512u, 1024u, 5120u, 6144u, 10240u, 17408u})
        lens.push_back(n);

    long long b1 = 0, b2 = 0, b3 = 0, b4 = 0, b5 = 0, total = 0;
    for (size_t n : lens) {
        total += (long long)n;

        q38simd::f32_to_bf16(c1.data(), fa.data(), n);
        ref_f32_to_bf16     (c2.data(), fa.data(), n);
        b1 += std::memcmp(c1.data(), c2.data(), n * sizeof(bf16)) != 0;

        q38simd::bf16_to_f32(o1.data(), ba.data(), n);
        ref_bf16_to_f32     (o2.data(), ba.data(), n);
        b2 += std::memcmp(o1.data(), o2.data(), n * sizeof(float)) != 0;

        q38simd::bf16_to_f32_add(o1.data(), ba.data(), fb.data(), n);
        ref_bf16_to_f32_add     (o2.data(), ba.data(), fb.data(), n);
        b3 += std::memcmp(o1.data(), o2.data(), n * sizeof(float)) != 0;

        q38simd::bf16_mul_to_bf16(c1.data(), ba.data(), bb.data(), n);
        ref_bf16_mul_to_bf16     (c2.data(), ba.data(), bb.data(), n);
        b4 += std::memcmp(c1.data(), c2.data(), n * sizeof(bf16)) != 0;

        q38simd::bf16_mul_to_f32(o1.data(), ba.data(), bb.data(), n);
        ref_bf16_mul_to_f32     (o2.data(), ba.data(), bb.data(), n);
        b5 += std::memcmp(o1.data(), o2.data(), n * sizeof(float)) != 0;
    }
    const long long nl = (long long)lens.size();
    report("f32_to_bf16",       b1, nl);
    report("bf16_to_f32",       b2, nl);
    report("bf16_to_f32_add",   b3, nl);
    report("bf16_mul_to_bf16",  b4, nl);
    report("bf16_mul_to_f32",   b5, nl);
    (void)total;

    // ---- 3. argmax, and specifically its ties -------------------------------
    //
    // A value mismatch would show up anywhere; the tie rule would not. The
    // scalar loop's `>` keeps the *first* maximum, and the vectorised form has
    // two ways to lose that -- a later element in the same lane column, and a
    // later column holding an equal maximum -- so the interesting draw is one
    // with a handful of distinct values, not a smooth one. `mtp` acceptance is
    // an integer compare against the draft head's pick, so an off-by-one here
    // would not be a rounding question, it would silently stop accepting.
    std::puts("\n-- argmax, ties to the lowest index --");
    long long b6 = 0, b7 = 0;
    long long cases = 0;
    std::vector<size_t> alens = lens;
    for (size_t n : {248320u, 250000u}) alens.push_back(n);

    std::vector<float> av(250000);
    std::vector<bf16>  bv(250000);
    // levels 1 = every element identical (the maximal tie), up to a draw with
    // essentially no ties at all.
    for (int levels : {1, 2, 3, 16, 1000, 0}) {
        for (size_t n : alens) {
            if (n == 0) continue;
            for (size_t i = 0; i < n; i++) {
                av[i] = levels ? (float)(int)(std::uniform_int_distribution<int>(
                                     0, levels - 1)(rng))
                               : uni(rng);
                bv[i] = (bf16)av[i];
            }
            cases++;
            b6 += q38simd::argmax_f32 (av.data(), (int)n) !=
                  ref_argmax_f32      (av.data(), (int)n);
            b7 += q38simd::argmax_bf16(bv.data(), (int)n) !=
                  ref_argmax_bf16     (bv.data(), (int)n);
        }
    }
    report("argmax_f32",  b6, cases);
    report("argmax_bf16", b7, cases);

    std::printf("\n== %s ==\n",
                failures ? "cpu_simd is NOT bit-identical" : "bit-identical");
    return failures != 0;
}
