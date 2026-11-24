// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#ifndef __Q4_L8_H__
#define __Q4_L8_H__
// 8-token Q4_1 projection core (plan S6.3). Measured on hardware in a one-tile toy design (variant q4_v4, README.md "Projection core"): 1.92 us per 32x256 block
// against 1.13 us for the shipping single-token core.
//
// One call = one 32-row x 256-column weight block (q4k_block_t, the DDR layout shared with decode) against an x slice of 8 tokens.
//   x slice layout  [kt = 32][t = 8][k8 = 8]   (256 columns = 32 tiles of 8; one 8x8 A tile = 64 contiguous elements)
//   accumulators    acc[sl = 4][t = 8][n = 8]   f32, sl = rh * 2 + nh: rows rh * 16 + nh * 8 .. +7
// Weights are dequantised once (w = q * scale + min, at 128 lanes), `filter_even/odd(., 8)` then splits the 128 lanes
// [8 columns][16 rows] into the two 8x8 B tiles [k][n] (rows 0..7 and 8..15) that `mmul<8,8,8>` takes with tokens as M.
#include "aie_kernel_utils.h"
#include "model_spec.h"
#include "q4_k.h"

constexpr int L8_TOKENS = 8;

static inline void _q4k_l8_zero(float* acc) {
    AIE_LOOP_UNROLL_FULL
    for (int i = 0; i < L8_TOKENS * Q4NX_ROW_BLOCK_SIZE; i += 16) aie::store_v(acc + i, aie::zeros<float, 16>());
}

static inline void _qmm_q4k_l8(const q4k_block_t* w, const bf16* x, float* acc) {
    // One row half (16 rows) at a time, so only its two 8x8 accumulators are live: with all four live the compiler spills
    // (measured in the one-tile toy, variant v11: 1.62 us per block instead of 1.92 us, bit-identical results).
    const uint4* qs_ptr = w->qs;
    for (int rh = 0; rh < 2; rh++) {
        aie::mmul<8, 8, 8, bf16, bf16, accfloat> m0, m1;
        for (int s = 0; s < 8; s++) {
            aie::vector<bf16, 16> sc16 = aie::load_v<16>(w->scales + s * 32 + rh * 16);
            aie::vector<bf16, 16> mn16 = aie::load_v<16>(w->mins + s * 32 + rh * 16);
            aie::vector<bf16, 32> sc32 = aie::concat(sc16, sc16);
            aie::vector<bf16, 32> mn32 = aie::concat(mn16, mn16);
            aie::vector<bf16, 64> sc64 = aie::concat(sc32, sc32);
            aie::vector<bf16, 64> mn64 = aie::concat(mn32, mn32);
            aie::vector<bf16, 128> sc128 = aie::concat(sc64, sc64);
            aie::vector<bf16, 128> mn128 = aie::concat(mn64, mn64);
            AIE_LOOP_UNROLL_FULL
            for (int j = 0; j < 4; j++) {
                aie::vector<uint4, 128> q = aie::load_v<128>(qs_ptr);
                qs_ptr += 64;
                aie::accum<accfloat, 128> fa;
                fa.from_vector(aie::to_float(q, 0));
                aie::vector<bf16, 128> b = fa.template to_vector<bf16>();
                aie::accum<accfloat, 128> d = aie::mul(b, sc128);
                d = aie::mac(d, mn128, (bf16)1.0f);
                aie::vector<bf16, 128> wd = d.template to_vector<bf16>();
                aie::vector<bf16, 64> A = aie::load_v<64>(x + (s * 4 + j) * 64);
                if (s == 0 && j == 0) { m0.mul(A, aie::filter_even(wd, 8)); m1.mul(A, aie::filter_odd(wd, 8)); }
                else { m0.mac(A, aie::filter_even(wd, 8)); m1.mac(A, aie::filter_odd(wd, 8)); }
            }
        }
        aie::accum<accfloat, 64> p0; p0.from_vector(aie::load_v<64>(acc + (rh * 2 + 0) * 64));
        aie::accum<accfloat, 64> p1; p1.from_vector(aie::load_v<64>(acc + (rh * 2 + 1) * 64));
        aie::store_v(acc + (rh * 2 + 0) * 64, aie::add(p0, m0.template to_vector<float>()).template to_vector<float>());
        aie::store_v(acc + (rh * 2 + 1) * 64, aie::add(p1, m1.template to_vector<float>()).template to_vector<float>());
    }
}

/// y rows of this tile for the 8 tokens, `[t][32]` bf16 (so a combo's packet is [lower tile 8x32][upper tile 8x32])
static inline void _q4k_l8_store(bf16* y, const float* acc) {
    AIE_LOOP_UNROLL_FULL
    for (int sl = 0; sl < 4; sl++) {
        aie::accum<accfloat, 64> a;
        a.from_vector(aie::load_v<64>(acc + sl * 64));
        aie::vector<bf16, 64> v = a.template to_vector<bf16>();
        AIE_LOOP_UNROLL_FULL
        for (int t = 0; t < L8_TOKENS; t++) aie::store_v(y + t * Q4NX_ROW_BLOCK_SIZE + sl * 8, v.extract<8>(t));
    }
}

#endif
