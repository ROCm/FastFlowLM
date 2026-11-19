// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file tools/verify_vision_bfp16.cpp
/// \brief The bit-identity gate on `image/npu/vision_bfp16.hpp`.
///
/// The vectorised pre-round replaced a scalar one that had already shipped, and
/// it was taken for speed, not for accuracy: **it must change no output.** A
/// cosine cannot see the difference between the two -- the whole operation moves
/// values by at most half a grid step -- so the gate is `memcmp`, not a score.
///
/// Three checks, each over the same adversarial corpus:
///
///   1. `bfp16_preround_range` (vector) vs `bfp16_preround_range_scalar`;
///   2. `preround_bfp16_inplace` (threaded) vs the scalar, which also proves no
///      chunk boundary lands inside a group of 8;
///   3. `narrow_preround_bfp16` (fused) vs the two passes it replaces -- a
///      scalar `(bf16)src[i]` narrow followed by the scalar pre-round.
///
/// Check 3 is the one with a real question in it. The fused kernel converts in
/// registers, so its narrow must agree with `biovault::bfloat16_t(float)` --
/// `common/cpu_simd.hpp` proves that exhaustively for `_cvt_f32_bf16`, and this
/// re-proves it in situ -- and its pre-round output goes out through plain
/// round-to-nearest-even where the scalar used `cpu_ref::f32_to_bf16`. Those two
/// differ on zero and subnormal inputs only; the argument that a pre-rounded
/// value is never subnormal is in the header, and this is the check that would
/// catch the argument being wrong.
///
/// The corpus is built to hit the branches rather than to look like a tensor:
/// blocks whose max exponent sits exactly on the `be < 7` cut and on the 0xFF
/// inf/nan cut, blocks of pure zero, blocks mixing one huge value with seven
/// tiny ones (where the clamp does the work), and uniform random bit patterns.
///
/// Build (any flags -- but run it at the module's, since `-ffast-math` is what
/// makes the scalar constructor take the RNE path for inf and nan):
///   g++-13 -O3 -ffast-math -fopenmp -std=c++20
///       -mavx512f -mavx512vl -mavx512bw -mavx512dq -mfma
///       -I. -I../../include -I../q4nx
///       tools/verify_vision_bfp16.cpp -o /tmp/verify_vision_bfp16
///
/// It touches no device and needs no checkpoint.
#include "image/npu/vision_bfp16.hpp"

#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

constexpr size_t GROUPS = 1u << 17;        // 131072 groups = 1M elements
constexpr size_t N      = GROUPS * 8;

/// A float whose exponent is drawn to land near the interesting cuts.
float draw(std::mt19937& rng) {
    std::uniform_int_distribution<int> which(0, 9);
    std::uniform_int_distribution<uint32_t> bits(0, 0xFFFFFFFFu);
    switch (which(rng)) {
        case 0: return 0.0f;
        case 1: return -0.0f;
        case 2: {   // be just under and just over the `be < 7` cut
            std::uniform_int_distribution<int> e(0, 12);
            uint32_t u = (uint32_t)(e(rng)) << 23 | (bits(rng) & 0x7FFFFFu);
            float f; std::memcpy(&f, &u, 4);
            return (bits(rng) & 1) ? -f : f;
        }
        case 3: return std::numeric_limits<float>::infinity();
        case 4: return -std::numeric_limits<float>::infinity();
        case 5: return std::numeric_limits<float>::quiet_NaN();
        case 6: {   // subnormal
            uint32_t u = bits(rng) & 0x007FFFFFu;
            float f; std::memcpy(&f, &u, 4);
            return f;
        }
        case 7: {   // a plain activation-sized value
            std::uniform_real_distribution<float> d(-4.0f, 4.0f);
            return d(rng);
        }
        case 8: {   // huge next to tiny, so the clamp is live
            std::uniform_real_distribution<float> d(-1e30f, 1e30f);
            return d(rng);
        }
        default: {  // uniform bit pattern
            uint32_t u = bits(rng);
            float f; std::memcpy(&f, &u, 4);
            return f;
        }
    }
}

int fails = 0;

void check(const char* name, const uint16_t* got, const uint16_t* want,
           size_t n) {
    size_t bad = 0, first = 0;
    for (size_t i = 0; i < n; i++)
        if (got[i] != want[i]) { if (!bad) first = i; bad++; }
    if (bad) {
        std::printf("[FAIL] %-34s %zu / %zu differ, first at %zu: "
                    "got 0x%04X want 0x%04X\n",
                    name, bad, n, first, got[first], want[first]);
        fails++;
    } else {
        std::printf("[ OK ] %-34s %zu elements bit-identical\n", name, n);
    }
}

}  // namespace

int main() {
    std::mt19937 rng(20260922);

    std::vector<float> src(N);
    for (size_t i = 0; i < N; i++) src[i] = draw(rng);

    // Some whole blocks forced to a single regime, which the elementwise draw
    // above would almost never produce: all zero, all subnormal, all inf.
    for (size_t g = 0; g < GROUPS; g += 997)
        for (int i = 0; i < 8; i++) src[g * 8 + i] = 0.0f;
    for (size_t g = 3; g < GROUPS; g += 1009)
        for (int i = 0; i < 8; i++) src[g * 8 + i] = 1e-41f;

    std::printf("== vision_bfp16, %zu elements (%zu groups of 8) ==\n\n",
                N, GROUPS);
#ifdef QWEN3_8MTP_CPU_SIMD
    std::printf("   built with AVX-512 -- the vector body is live\n\n");
#else
    std::printf("   NO AVX-512: every path below is the scalar one and this "
                "run proves nothing\n\n");
#endif

    // The common starting point: the scalar narrow, exactly as the two-pass
    // version did it.
    std::vector<bf16> narrowed(N);
    for (size_t i = 0; i < N; i++) narrowed[i] = (bf16)src[i];
    const uint16_t* base = reinterpret_cast<const uint16_t*>(narrowed.data());

    std::vector<uint16_t> want(base, base + N);
    q38npu::bfp16_preround_range_scalar(want.data(), N, 8);

    // 1. vector vs scalar, serial
    {
        std::vector<uint16_t> got(base, base + N);
        q38npu::bfp16_preround_range(got.data(), N, 8);
        check("bfp16_preround_range", got.data(), want.data(), N);
    }

    // 2. the threaded wrapper -- also a chunk-boundary check
    {
        std::vector<uint16_t> got(base, base + N);
        q38npu::preround_bfp16_inplace(got.data(), N, 8);
        check("preround_bfp16_inplace", got.data(), want.data(), N);
    }

    // 3. the fused narrow + pre-round against the two passes
    {
        std::vector<bf16> got(N);
        q38npu::narrow_preround_bfp16(got.data(), src.data(), N);
        check("narrow_preround_bfp16",
              reinterpret_cast<const uint16_t*>(got.data()), want.data(), N);
    }

    // 4. a ragged length: not a multiple of 16, so the scalar tail runs, and
    //    the last partial group must be left alone by BOTH paths.
    {
        const size_t n = N - 13;
        std::vector<uint16_t> w2(base, base + n);
        q38npu::bfp16_preround_range_scalar(w2.data(), n, 8);

        std::vector<uint16_t> g1(base, base + n);
        q38npu::bfp16_preround_range(g1.data(), n, 8);
        check("bfp16_preround_range, ragged", g1.data(), w2.data(), n);

        std::vector<bf16> g2(n);
        q38npu::narrow_preround_bfp16(g2.data(), src.data(), n);
        check("narrow_preround_bfp16, ragged",
              reinterpret_cast<const uint16_t*>(g2.data()), w2.data(), n);
    }

    // 5. the plain marshalling wrappers, against the scalar loops they replace
    {
        std::vector<bf16> got(N);
        q38npu::narrow_bf16_par(got.data(), src.data(), N);
        check("narrow_bf16_par",
              reinterpret_cast<const uint16_t*>(got.data()), base, N);

        std::vector<float> fw(N), fg(N);
        for (size_t i = 0; i < N; i++) fw[i] = (float)narrowed[i];
        q38npu::widen_f32_par(fg.data(), narrowed.data(), N);
        check("widen_f32_par",
              reinterpret_cast<const uint16_t*>(fg.data()),
              reinterpret_cast<const uint16_t*>(fw.data()), N * 2);
    }

    std::printf("\n%s -- %d check(s) failed\n", fails ? "FAILED" : "PASSED",
                fails);
    return fails ? 1 : 0;
}
