// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.
//
// Stream core of the Inkling-2-Mini decode overlay. Everything arrives as a sequence of 4736-byte units (one Q4_K
// block each) on S2MM 1; a task starts with a header unit (int32 words, then the task's x at byte X_OFF), and its
// results leave as 96-float y units on MM2S 0. The core runs task after task and never returns.
//
// Tasks:
//   OP_GEMV_Q4K  [1] groups, [2] col blocks nc, x = nc * 256 bf16: y = W x for groups of 96 rows, the weights in the
//                NPU's block-triple order ([group][cb][3 row blocks], each block in q4k_to_npu order); one y unit
//                per group, row 32 b + i of the group at y[32 b + i]. fp32 accumulation of bf16 weights x bf16 x.
//   OP_EXPERT    x = 2304 bf16: one half of an expert (gate_up, situ, down), see below
//   OP_ROUTER    8 router rows (bf16 hi + lo, row-major: 2 units per row) times x hi + lo: one y unit of 8 logits
//   OP_TOPK      (core (0, 4)) the 128 logits to the hub, which routes; its ids and weights as one y unit
//   OP_PROJ      one header shared by all 16 channels: [1] nx, [2 + c] g0 | ng << 16 for channel c, [18 + m] end | xi
//                << 16 for matrix m (5 matrices; its end group, counted over all of them, and its x vector), int16
//                [46 + c] nd | xi << 8; then nx x vectors (2304 bf16 each) from X_OFF, padded to whole units. Channel
//                c runs groups g0 .. g0 + ng - 1 as OP_GEMV_Q4K (nc 9) with each group's x, then nd dense bf16 rows
//                (one unit each: 2304 bf16, then padding) times x vector xi as one more y unit (y[i] = row i); a
//                channel's groups and dense rows use at most 2 x vectors
//   OP_UP        a loop controller's up projections: [1] nk controllers; the down projection's partials P[k][h][64]
//                (fp32, h: the x / message half) from X_OFF, padded to the unit; then channel c's units 9 nk c ..
//                9 nk c + 9 nk - 1, unit u = 16 rows of controller u / 144 as [64 ranks][16 forget | 16 gain] bf16
//                times g = bf16(P[k][0] + P[k][1]) (fp32 accumulation), out tanh of each; 3 units per y unit
//   op >= OP_GLUE the partner glue core runs the task (kernel/glue.hpp) on this core's ring and y buffers: the ring
//                and y indices go to rl, GL_GO starts it, GL_DONE gives the core back

#include "aie_kernel_utils.h"
#include "common.hpp"

static constexpr int UNIT = 4736;
static constexpr int RING = 4;
static constexpr int X_OFF = 128;
static constexpr int OP_GEMV_Q4K = 2, OP_EXPERT = 3, OP_ROUTER = 4, OP_TOPK = 5, OP_PROJ = 6, OP_UP = 7, OP_GLUE = 16;
static constexpr int W_PROD = 0, W_CONS = 1, Y_PROD = 2, Y_CONS = 3;
static constexpr int HB_H2 = 4, HB_H3 = 5, HB_C2 = 6, HB_C3 = 7;   // the h exchange, locks of the row 3 tile
static constexpr int GL_GO = 8, GL_DONE = 9;                       // the glue core hand-off, own locks
static constexpr int TK_PROD = 13, TK_CONS = 14, TK_DONE = 15;      // top-k with the hub, locks of the hub tile
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

// lane-wise fp32 of a bf16 vector
template <int N>
static inline __attribute__((always_inline)) aie::vector<float, N> to_f32(aie::vector<bf16, N> v) {
    aie::accum<accfloat, N> a(v);
    return a.template to_vector<float>();
}

// keeps the compiler from moving buffer accesses across a lock call (the other core writes those buffers)
static inline __attribute__((always_inline)) void mem_barrier() { __builtin_aie2p_sched_barrier(); }

using v16 = aie::vector<float, 16>;
static inline __attribute__((always_inline)) v16 fmul(v16 a, v16 b) { return aie::mul(a, b).to_vector<float>(); }
static inline __attribute__((always_inline)) v16 fmul(v16 a, float b) { return aie::mul(a, b).to_vector<float>(); }

// tanh in fp32: [7/6] Pade of x clamped to +-4.97 (abs error < 1e-4 everywhere, ~1e-7 for |x| < 2; the bf16 tanh
// of the AIE API is ~2 % off on small arguments, which situ hits through tanh(u / 25)); 1 / den is the API inverse
// with two Newton steps
static __attribute__((noinline)) v16 tanh_f32(v16 x) {
    x = aie::max(aie::min(x, 4.97f), -4.97f);
    const v16 x2 = fmul(x, x);
    const v16 num = fmul(x, aie::add(fmul(x2, aie::add(fmul(x2, aie::add(x2, 378.0f)), 17325.0f)), 135135.0f));
    const v16 den = aie::add(fmul(x2, aie::add(fmul(x2, aie::add(fmul(x2, 28.0f), 3150.0f)), 62370.0f)), 135135.0f);
    v16 r = aie::inv(den);
    r = fmul(r, aie::sub(aie::broadcast<float, 16>(2.0f), fmul(den, r)));
    r = fmul(r, aie::sub(aie::broadcast<float, 16>(2.0f), fmul(den, r)));
    return fmul(num, r);
}

// h[k] = situ(g[k], u[k]) = 50 tanh(g / 4) (1 + tanh(g / 2)) tanh(u / 25), n a multiple of 16; fp32, h rounded to bf16
static void situ(bf16* __restrict h, const float* __restrict g, const float* __restrict u, int n) {
    AIE_LOOP_NO_UNROLL
    for (int k = 0; k < n; k += 16) {
        const v16 gv = aie::load_v<16>(g + k), uv = aie::load_v<16>(u + k);
        const v16 t1 = tanh_f32(fmul(gv, 0.25f)), t2 = tanh_f32(fmul(gv, 0.5f)), t3 = tanh_f32(fmul(uv, 0.04f));
        const v16 s = fmul(fmul(t1, t3), aie::add(t2, 1.0f));
        aie::store_v(h + k, aie::mul(s, 50.0f).to_vector<bf16>());
    }
}

// the y unit of the group's 96 rows: row 32 b + i at out[32 b + i]
static inline void group_rows(float* __restrict out, const float* __restrict acc) {
    for (int s = 0; s < 12; s++) {     // s = 4 b + 2 H + par
        const int b = s >> 2, H = (s >> 1) & 1, par = s & 1;
        for (int r = 0; r < 8; r++)
            out[32 * b + 16 * H + 2 * r + par] = aie::reduce_add(aie::load_v<8>(acc + 64 * s + 8 * r));
    }
}

extern "C" __attribute__((minsize)) void decode_core(uint8* w0, uint8* w1, uint8* w2, uint8* w3, float* y0, float* y1, bf16* x, float* acc,
                            float* xnr, bf16* xs, bf16* xr, bf16* ss, float* gu, bf16* hb, int32_t* tk, int32_t row,
                            volatile int32_t* rl) {
    aie::set_rounding(aie::rounding_mode::conv_even);
    const int ch = row;   // the shim channel feeding this core: col + 8 row
    row >>= 3;
    uint8* const W[RING] = {w0, w1, w2, w3};
    float* const Y[2] = {y0, y1};
    int wi = 0, yi = 0;
    auto take = [&]() -> const uint8* { _lock_acquire(W_CONS); return W[wi]; };
    auto give = [&]() { _lock_release(W_PROD); if (++wi == RING) wi = 0; };
    auto y_get = [&]() -> float* { _lock_acquire(Y_PROD); return Y[yi]; };
    auto y_put = [&]() { _lock_release(Y_CONS); yi ^= 1; };
    // the task's inputs after the header words: n bytes (multiples of 32) from X_OFF on, continuing into the
    // following units (d null: skip them); `cur` / `off` = where the next read starts
    const uint8* cur = nullptr;
    int off = 0;
    auto read = [&](uint8* d, int n) __attribute__((noinline)) {
        while (n > 0) {
            if (off == UNIT) { give(); cur = take(); off = 0; }
            const int k = n < UNIT - off ? n : UNIT - off;
            if (d) {
                for (int i = 0; i < k; i += 32) aie::store_v(d + i, aie::load_v<32>(cur + off + i));
                d += k;
            }
            n -= k; off += k;
        }
    };
    // y = W x for groups of 96 rows (nc column blocks); each group's rows as a y unit, or (to_gu) into gu as
    // gate / up rows of the expert half (glu order: row block 2 j = gate j, 2 j + 1 = up j)
    auto gemv = [&](int groups, int nc, bool to_gu, const bf16* xv) {
        for (int g = 0; g < groups; g++) {
            for (int i = 0; i < 12 * 64; i += 64) aie::store_v(acc + i, aie::zeros<float, 64>());
            for (int cb = 0; cb < nc; cb++) {
                prep_x(xr, xnr, xs, xv + 256 * cb);
                for (int b = 0; b < 3; b++) {
                    gemv_block(acc + 256 * b, take(), xnr, xs, xr, ss);
                    give();
                }
            }
            if (!to_gu) { float* y = y_get(); group_rows(y, acc); y_put(); continue; }
            for (int b = 0; b < 3; b++) {
                const int pb = 3 * g + b;
                float* dst = gu + (pb & 1) * 576 + 32 * (pb >> 1);
                for (int s = 4 * b; s < 4 * b + 4; s++) {
                    const int H = (s >> 1) & 1, par = s & 1;
                    for (int r = 0; r < 8; r++)
                        dst[16 * H + 2 * r + par] = aie::reduce_add(aie::load_v<8>(acc + 64 * s + 8 * r));
                }
            }
        }
    };

    for (;;) {
        cur = take();
        const volatile int* h = (const volatile int*)cur;
        const int op = h[0], a1 = h[1], a2 = h[2];
        off = X_OFF;
        if (op == OP_GEMV_Q4K) {
            read((uint8*)x, a2 * 512);
            give();
            gemv(a1, a2, false, x);
        } else if (op == OP_EXPERT) {
            // half `row` of one expert: gate_up groups 12 row .. 12 row + 11 (gate / up rows j 18 row .. 18 row + 17,
            // glu order: row block 2 j = gate j, 2 j + 1 = up j), situ, the two halves of h meet in hb (the row 2
            // tile), then down groups 12 row .. 12 row + 11 over the whole h (1280, zero padded): 12 y units
            read((uint8*)x, 2304 * 2);
            give();
            gemv(12, 9, true, x);
            if (row == 0) {
                _lock_acquire(HB_C3);
                mem_barrier();
                situ(hb, gu, gu + 576, 576);
                for (int i = 1152; i < 1280; i += 32) aie::store_v(hb + i, aie::zeros<bf16, 32>());
                mem_barrier();
                _lock_release(HB_H2);
                _lock_acquire(HB_H3);
                mem_barrier();
                for (int i = 0; i < 1280; i += 32) aie::store_v(x + i, aie::load_v<32>(hb + i));
                mem_barrier();
                _lock_release(HB_C2);
            } else {
                _down_lock_acquire(HB_C2);
                mem_barrier();
                situ(hb + 576, gu, gu + 576, 576);
                mem_barrier();
                _down_lock_release(HB_H3);
                _down_lock_acquire(HB_H2);
                mem_barrier();
                for (int i = 0; i < 1280; i += 32) aie::store_v(x + i, aie::load_v<32>(hb + i));
                mem_barrier();
                _down_lock_release(HB_C3);
            }
            gemv(12, 5, false, x);
        } else if (op == OP_PROJ) {
            const int t = h[2 + ch], g0 = t & 0xFFFF, ng = t >> 16;
            int end[5], xi[5], slot[4] = {-1, -1, -1, -1}, ns = 0;
            const int dn = ((const volatile int16_t*)h)[46 + ch], nd = dn & 0xFF, dx = dn >> 8;
            for (int m = 0; m < 5; m++) { end[m] = h[18 + m] & 0xFFFF; xi[m] = h[18 + m] >> 16; }
            for (int m = 0, b = 0; m < 5; b = end[m++])
                if (b < g0 + ng && end[m] > g0 && slot[xi[m]] < 0) slot[xi[m]] = ns++;
            if (nd && slot[dx] < 0) slot[dx] = ns++;
            bf16* const xb[2] = {x, (bf16*)gu};
            for (int i = 0; i < a1; i++) {
                read(slot[i] >= 0 ? (uint8*)xb[slot[i]] : nullptr, 2304 * 2);
            }
            give();
            for (int g = g0, m = 0; g < g0 + ng; g++) {
                while (g >= end[m]) m++;
                gemv(1, 9, false, xb[slot[xi[m]]]);
            }
            if (nd) {
                const bf16* xd = xb[slot[dx]];
                float* y = y_get();
                for (int i = 0; i < nd; i++) {
                    const bf16* w = (const bf16*)take();
                    aie::accum<accfloat, 32> a = aie::zeros<accfloat, 32>();
                    for (int k = 0; k < 2304; k += 32) a = aie::mac(a, aie::load_v<32>(w + k), aie::load_v<32>(xd + k));
                    y[i] = aie::reduce_add(a.to_vector<float>());
                    give();
                }
                y_put();
            }
        } else if (op == OP_UP) {
            const int nk = h[1];
            read((uint8*)gu, 512 * nk);
            for (int i = 0; i < 64 * nk; i += 16) {
                const float* p = gu + 2 * i - (i & 63);
                aie::store_v(ss + i, aie::accum<accfloat, 16>(aie::add(aie::load_v<16>(p), aie::load_v<16>(p + 64))).to_vector<bf16>());
            }
            give();
            float* y = nullptr;
            int u = 9 * nk * ch, b = 144, j = 0;
            const bf16* g = ss;
            while (u >= b) { g += 64; b += 144; }
            for (int i = 0; i < 9 * nk; i++, u++) {
                if (u == b) { g += 64; b += 144; }
                const bf16* w = (const bf16*)take();
                aie::accum<accfloat, 32> a = aie::zeros<accfloat, 32>();
                for (int k = 0; k < 64; k++) a = aie::mac(a, aie::load_v<32>(w + 32 * k), aie::broadcast<bf16, 32>(g[k]));
                give();
                if (j == 0) y = y_get();
                const aie::vector<float, 32> v = a.to_vector<float>();
                aie::store_v(y + 32 * j, tanh_f32(v.extract<16>(0)));
                aie::store_v(y + 32 * j + 16, tanh_f32(v.extract<16>(1)));
                if (++j == 3) { y_put(); j = 0; }
            }
        } else if (op == OP_ROUTER) {
            // 8 rows of the fp32 router as bf16 hi + lo: x hi (2304) then x lo (2304, into gu) after the header,
            // then 16 units, row r in units 2 r and 2 r + 1: 18 granule pairs (hi, lo: 64 bf16 each) of column
            // chunks kc = 18 h + k, then 128 bytes padding. logit = Whi xhi + Whi xlo + Wlo xhi, fp32 accumulation
            // in registers; one y unit, y[i] = logit of row i
            bf16* xlo = (bf16*)gu;
            read((uint8*)x, 2304 * 2);
            read((uint8*)xlo, 2304 * 2);
            give();
            alignas(aie::vector_decl_align) float lg[8];
            for (int r = 0; r < 8; r++) {
                aie::accum<accfloat, 64> c0 = aie::zeros<accfloat, 64>(), c1 = c0, c2 = c0;
                for (int h = 0; h < 2; h++) {
                    const uint8* u = take();
                    const bf16* xh = x + 1152 * h;
                    const bf16* xl = xlo + 1152 * h;
                    for (int k = 0; k < 18; k++) {
                        const aie::vector<bf16, 64> wh = aie::load_v<64>((const bf16*)(u + 256 * k)), wl = aie::load_v<64>((const bf16*)(u + 256 * k + 128));
                        const aie::vector<bf16, 64> a = aie::load_v<64>(xh + 64 * k), b = aie::load_v<64>(xl + 64 * k);
                        c0 = aie::mac(c0, wh, a);
                        c1 = aie::mac(c1, wh, b);
                        c2 = aie::mac(c2, wl, a);
                    }
                    give();
                }
                lg[r] = aie::reduce_add(aie::add(aie::add(c0.to_vector<float>(), c1.to_vector<float>()), c2.to_vector<float>()));
            }
            float* y = y_get();
            for (int i = 0; i < 8; i++) y[i] = lg[i];
            y_put();
        } else if (op == OP_TOPK) {
            // core (0, 3): bias (128 fp32) after the header, then 2 units of router y units (16 x 96 floats, the
            // logits at y[8 c .. 8 c + 8)); the hub routes (sigmoid, top 8 on p + bias, w = p / sum * 8) and
            // dispatches the experts; its ids (int32) and weights come back as one y unit
            float* logits = gu;
            float* bias = gu + 128;
            read((uint8*)bias, 512);
            give();
            int c = 0, o = 0;
            for (int k = 0; k < 2; k++) {
                const uint8* u = take();
                for (; c < 16 && o + 32 <= UNIT; c++, o += 384)
                    aie::store_v(logits + 8 * c, aie::load_v<8>((const float*)(u + o)));
                o -= UNIT;
                give();
            }
            // the hub (tile above) routes: tk = [logits 128][bias 128][ids 8][weights 8]
            _up_lock_acquire(TK_PROD);
            mem_barrier();
            for (int i = 0; i < 256; i += 16) aie::store_v((float*)tk + i, aie::load_v<16>(gu + i));
            mem_barrier();
            _up_lock_release(TK_CONS);
            _up_lock_acquire(TK_DONE);
            mem_barrier();
            float* y = y_get();
            aie::store_v(y, aie::load_v<16>((const float*)tk + 256));
            y_put();
            mem_barrier();
            _up_lock_release(TK_PROD);
        } else if (op >= OP_GLUE) {
            rl[0] = wi;
            rl[1] = yi;
            mem_barrier();
            _lock_release(GL_GO);
            _lock_acquire(GL_DONE);
            mem_barrier();
            wi = rl[0];
            yi = rl[1];
        }
    }
}
