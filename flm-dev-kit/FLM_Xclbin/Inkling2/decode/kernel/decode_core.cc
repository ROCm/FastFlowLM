// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.
//
// Stream core of the Inkling-2-Mini decode overlay. Everything arrives as a sequence of 4736-byte units (one Q4_K
// block each) on S2MM 1; a task starts with a header unit (int32 words, then the task's x at byte X_OFF), and its
// results leave as 96-float y units on MM2S 0. The core runs task after task and never returns.
//
// Tasks:
//   OP_SINK      [1] n: take n more units, emit one y unit (y[0] = n, y[1] = a byte sum)
//   OP_GEMV_Q4K  [1] groups, [2] col blocks nc, x = nc * 256 bf16: y = W x for groups of 96 rows, the weights in the
//                NPU's block-triple order ([group][cb][3 row blocks], each block in q4k_to_npu order); one y unit
//                per group, row 32 b + i of the group at y[32 b + i]. fp32 accumulation of bf16 weights x bf16 x.

#include "aie_kernel_utils.h"
#include "common.hpp"

static constexpr int UNIT = 4736;
static constexpr int RING = 4;
static constexpr int X_OFF = 128;
static constexpr int OP_SINK = 1, OP_GEMV_Q4K = 2;
static constexpr int W_PROD = 0, W_CONS = 1, Y_PROD = 2, Y_CONS = 3;
static constexpr uint8 Q4_BIASED_EXP_BYTE = 0x43;   // 0x43 << 8 | q is the bf16 128 + q
static constexpr int NIBBLE_BIAS = 128;

static inline __attribute__((always_inline)) aie::vector<bf16, 64> q4_biased(aie::vector<uint8, 64> nib, aie::vector<uint8, 64> hi) {
    return aie::vector_cast<bf16>(aie::concat(aie::interleave_zip(nib, hi, 1)));
}

// lane r * 8 + k = v[r]
static inline __attribute__((always_inline)) aie::vector<bf16, 64> splat8(aie::vector<bf16, 8> v) {
    aie::vector<bf16, 16> v16 = aie::concat(v, v);
    aie::vector<bf16, 32> v32 = aie::concat(v16, v16);
    return aie::transpose(aie::concat(v32, v32), 8, 8);
}

// lane r * 8 + k = v[k]
static inline __attribute__((always_inline)) aie::vector<bf16, 64> rep8(aie::vector<bf16, 8> v) {
    aie::vector<bf16, 16> v16 = aie::concat(v, v);
    aie::vector<bf16, 32> v32 = aie::concat(v16, v16);
    return aie::concat(v32, v32);
}

// lane r * 8 + k = v[k], fp32
static inline __attribute__((always_inline)) aie::vector<float, 64> rep8f(aie::vector<float, 8> v) {
    aie::vector<float, 16> v16 = aie::concat(v, v);
    aie::vector<float, 32> v32 = aie::concat(v16, v16);
    return aie::concat(v32, v32);
}

// The column block's x laid out for gemv_block: xr[kt] = rep8(x[8 kt .. 8 kt + 8]) (32 x 64 bf16) and
// xnr[g] = rep8 of the 8 values -128 sum_{kt of group g} x[8 kt + k] (8 x 64 fp32); xs[g] = the group's x sum.
static void prep_x(bf16* __restrict xr, float* __restrict xnr, bf16* __restrict xs, const bf16* __restrict xc) {
    for (int kt = 0; kt < 32; kt++) aie::store_v(xr + 64 * kt, rep8(aie::load_v<8>(xc + 8 * kt)));
    const bfloat16 nb = bfloat16(-NIBBLE_BIAS), one = bfloat16(1.0f);
    aie::vector<float, 16> tot = aie::zeros<float, 16>();
    for (int g = 0; g < 8; g++) {
        const bf16* xg = xr + 256 * g;
        aie::accum<accfloat, 64> t = aie::mul(aie::load_v<64>(xg), nb);
        t = aie::mac(t, aie::load_v<64>(xg + 64), nb);
        t = aie::mac(t, aie::load_v<64>(xg + 128), nb);
        t = aie::mac(t, aie::load_v<64>(xg + 192), nb);
        aie::store_v(xnr + 64 * g, t.to_vector<float>());
        aie::accum<accfloat, 16> u = aie::mul(aie::load_v<16>(xg), one);
        u = aie::mac(u, aie::load_v<16>(xg + 64), one);
        u = aie::mac(u, aie::load_v<16>(xg + 128), one);
        u = aie::mac(u, aie::load_v<16>(xg + 192), one);
        tot[g] = aie::reduce_add(u.to_vector<float>().extract<8>(0));
    }
    aie::accum<accfloat, 16> ta;
    ta.from_vector(tot);
    aie::store_v(xs, rep8(ta.to_vector<bf16>().extract<8>(0)));   // lane r * 8 + g = the group sum
}

// One Q4_K block (q4k_to_npu order) times the 256 x of its column block, into acc[hp][r 8][k 8] (hp = 2 H + par,
// row 16 H + 2 r + par of the block), fp32. Per 32-column group g the nibbles are used as the exact bf16 128 + q:
// p = sum (128 + q) x over the group, started at -128 sum x (xnr, fp32) so p = sum q x, then acc += bf16(p) * S sc.
// The min term M mn sum x goes in once per block from the group sums xs (lane r * 8 + g; the final reduce adds
// over the 8 lanes of a row whatever their meaning). ss: scratch for the scale splats (16 x 64 bf16).
static inline void gemv_block(float* __restrict acc, const uint8* __restrict blk, const float* __restrict xnr,
                              const bf16* __restrict xs, const bf16* __restrict xr, bf16* __restrict ss) {
    const aie::vector<uint8, 64> hi = aie::broadcast<uint8, 64>(Q4_BIASED_EXP_BYTE);
    const aie::vector<bf16, 64> xs8 = aie::load_v<64>(xs);
    for (int hp = 0; hp < 4; hp++) {
        const int H = hp >> 1, par = hp & 1;
        const uint8* q = blk + 512 + 2048 * H + 1024 * par;
        aie::accum<accfloat, 64> t;
        t.from_vector(aie::to_float(aie::load_v<64>(blk + 64 * hp), 0));
        const aie::vector<bf16, 64> a_sc = aie::mul(t.to_vector<bf16>(), rep8(aie::load_v<8>((const bf16*)(blk + 4608) + 8 * hp))).to_vector<bf16>();
        for (int g = 0; g < 8; g++) aie::store_v(ss + 64 * g, splat8(aie::shuffle_down(a_sc, 8 * g).extract<8>(0)));
        aie::accum<accfloat, 64> c(aie::load_v<64>(acc + 64 * hp));
        // min term: lanes r * 8 + g
        t.from_vector(aie::to_float(aie::load_v<64>(blk + 256 + 64 * hp), 0));
        c = aie::mac(c, aie::transpose(aie::mul(t.to_vector<bf16>(), rep8(aie::load_v<8>((const bf16*)(blk + 4672) + 8 * hp))).to_vector<bf16>(), 8, 8), xs8);
        AIE_LOOP_RANGE(8, 8)
        for (int g = 0; g < 8; g++) {
            aie::accum<accfloat, 64> p(aie::load_v<64>(xnr + 64 * g));
            const bf16* xg = xr + 256 * g;
            const uint8* qg = q + 128 * g;
#pragma clang loop unroll(full)
            for (int j = 0; j < 4; j += 2) {
                aie::vector<uint8, 128> u = aie::unpack(aie::vector_cast<uint4>(aie::load_v<64>(qg + 32 * j)));
                p = aie::mac(p, q4_biased(u.extract<64>(0), hi), aie::load_v<64>(xg + 64 * j));
                p = aie::mac(p, q4_biased(u.extract<64>(1), hi), aie::load_v<64>(xg + 64 * j + 64));
            }
            c = aie::mac(c, p.to_vector<bf16>(), aie::load_v<64>(ss + 64 * g));
        }
        aie::store_v(acc + 64 * hp, c.to_vector<float>());
    }
}

extern "C" void decode_core(uint8* w0, uint8* w1, uint8* w2, uint8* w3, float* y0, float* y1, bf16* x, float* acc,
                            float* xnr, bf16* xs, bf16* xr, bf16* ss) {
    aie::set_rounding(aie::rounding_mode::conv_even);
    uint8* const W[RING] = {w0, w1, w2, w3};
    float* const Y[2] = {y0, y1};
    int wi = 0, yi = 0;
    auto take = [&]() -> const uint8* { _lock_acquire(W_CONS); return W[wi]; };
    auto give = [&]() { _lock_release(W_PROD); if (++wi == RING) wi = 0; };
    auto y_get = [&]() -> float* { _lock_acquire(Y_PROD); return Y[yi]; };
    auto y_put = [&]() { _lock_release(Y_CONS); yi ^= 1; };

    for (;;) {
        const volatile int* h = (const volatile int*)take();
        const int op = h[0], a1 = h[1], a2 = h[2];
        if (op == OP_GEMV_Q4K) {
            // x: nc * 512 bytes from X_OFF on, continuing into the following units
            int left = a2 * 512, off = X_OFF;
            uint8* xd = (uint8*)x;
            const uint8* u = (const uint8*)h;
            for (;;) {
                const int n = left < UNIT - off ? left : UNIT - off;
                for (int i = 0; i < n; i += 32) aie::store_v(xd + i, aie::load_v<32>(u + off + i));
                xd += n; left -= n;
                give();
                if (left == 0) break;
                u = take(); off = 0;
            }
            for (int g = 0; g < a1; g++) {
                for (int i = 0; i < 12 * 64; i += 64) aie::store_v(acc + i, aie::zeros<float, 64>());
                for (int cb = 0; cb < a2; cb++) {
                    prep_x(xr, xnr, xs, x + 256 * cb);
                    for (int b = 0; b < 3; b++) {
                        gemv_block(acc + 256 * b, take(), xnr, xs, xr, ss);
                        give();
                    }
                }
                float* y = y_get();
                for (int s = 0; s < 12; s++) {     // s = 4 b + 2 H + par
                    const int b = s / 4, H = (s / 2) % 2, par = s % 2;
                    for (int r = 0; r < 8; r++)
                        y[32 * b + 16 * H + 2 * r + par] = aie::reduce_add(aie::load_v<8>(acc + 64 * s + 8 * r));
                }
                y_put();
            }
        } else {   // OP_SINK
            const int n = a1;
            give();
            unsigned sum = 0;
            for (int i = 0; i < n; i++) {
                const uint8* u = take();
                sum += u[0];
                give();
            }
            float* y = y_get();
            y[0] = (float)n;
            y[1] = (float)sum;
            y_put();
        }
    }
}
