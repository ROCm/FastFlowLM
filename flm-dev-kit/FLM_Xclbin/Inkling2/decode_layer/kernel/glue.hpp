// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.
//
// Glue core of the Inkling-2-Mini decode overlay: the non-GEMV work of a layer (norms, mixes, convolutions, clips,
// attention over the cache, gates, residuals). It has no stream of its own: it runs the tasks its partner stream core
// (the tile above for glue_up.cc, below for glue_dn.cc) hands over (op >= OP_GLUE) on the partner's input ring and y
// buffers, with the partner's locks, then hands the partner back (see decode_core.cc).
//
// Tasks (header words after the op):
//   OP_RMS   (glue_up only) [1] n (<= 2304; the mean is over 2304): x (n fp32) then g (n fp32) after the header -> y = x / rms(x) * g,
//            n / 96 y units
//   OP_ATTN  one KV head g of a decode token at position p (glue_up of column g): [1] p, [2] global layer, [3] tau
//            (fp32), [4] cache records streamed, [5] ring (SWA: record s holds the key with position = s mod 1024),
//            [6] g. Inputs: q_g, gate_g (384 + 384 fp32); the projections' A block (K 1056, V 1056, R 384 rows,
//            then MISC: sinks 24 | pad | k taps 32 | pad | v taps 32 | pad, 96 rows each); the head's static block
//            (q_clip 384, rel_clip 48, k conv taps [4][128], v conv taps [4][128]); rel_bias 1024 x 16; the conv
//            ring (4 rows of raw [k 128 | v 128], row p mod 4 = this token's); the cache records (K 128 | V 128
//            bf16). Out (6 y units): this token's raw [k | v] (the ring row), its record, O_g (384 bf16) =
//            softmax(q k / sqrt(128) + rel bias, sink) v * sigmoid(gate)
//   OP_MIX   (glue_dn only) a layer's glue on D = 2304 fp32: [1] prefix, [2] flags, [3] p, [4] S states, [5] nq
//            queries, [6] out mode, [7] norm (1: one per query). Inputs in order: the S states; the prefix (0: cur;
//            1: the top-k y unit then the 16 x 12 expert y units, cur = sum_e w_e y_e; 2: two halves, cur = sum);
//            then the application's block: [flags 1] conv [24 chunks][taps 4 | ring 4][96] (ring row p mod 4 =
//            this token's), [flags 2] the residual; per query: q, [norm] g. Out (y units, in order): [flags 1] raw cur
//            (the ring row), conv cur = (x3 + sum_k w_k x_k) / sqrt(5); [flags 4] cur (+ residual); per query the
//            mix over {states, cur} (softmax of q . x_i / rms_i / 48, sum of w_i x_i / rms_i), [norm] times
//            g / rms: out 0 fp32, 1 bf16, 2 (nq 1) bf16 hi, lo, hi, 3 fp32 then
//            bf16 of the mix / rms (no g)

//   OP_CTOUT (glue_dn only) a loop controller's output and the norm after it: [1] out mode (1 bf16, 2 bf16 hi, lo,
//            hi); x, m (2304 fp32 each), then per 48 rows [16 | 16] x 3 (fp32: tanh f, tanh a, from OP_UP), then
//            g (2304 fp32): out = x - tanh(f) x + tanh(a) m, y = out / rms(out) * g

#include "aie_kernel_utils.h"
#include "common.hpp"
#include "router_math.hpp"

#ifndef GLUE_NAME
#error "define GLUE_NAME and GLUE_UP before including glue.hpp"
#endif

#pragma clang attribute push(__attribute__((minsize)), apply_to = function)
namespace {
constexpr int UNIT = 4736;
constexpr int RING = 4;
constexpr int X_OFF = 128;
constexpr int Y_UNIT = 96;
constexpr int OP_RMS = 16, OP_ATTN = 17, OP_MIX = 18, OP_CTOUT = 19;
constexpr int W_PROD = 0, W_CONS = 1, Y_PROD = 2, Y_CONS = 3;   // the partner's locks
constexpr int GL_GO = 8, GL_DONE = 9;
constexpr float EPS = 1e-6f;
constexpr float QK_CLIP = 22.627416997969522f, REL_CLIP = 8.0f, LOG2E = 1.44269504088896341f, NEG = -1e30f;
constexpr int DH = 128, NREL = 16, REL_EXT = 1024;
// OP_ATTN's stream: units 0-3 header, q / gate, the A block; then the slot's region: static (2 units), rel bias
// (bf16 [2 halves][1024 rows (distance 1023 - row)][8], 7 units of 37 tiles), the conv ring (1 unit), the cache records (KPU per unit:
// K as [16 chunks of 8 dims][KPU records][8] bf16, then V as [KPU records][128] bf16)
constexpr int U_ST = 4, U_RB = 6, U_RING = 13, U_REC = 14, KPU = 8, TPU = UNIT / 128, RB_TILES = 256;
// OP_MIX scratch in a (floats)
constexpr int DM = 2304, M_T = 5 * DM, M_SC = M_T + 768;   // states + cur, a 768 chunk, scalars (64)
// OP_ATTN scratch in a (floats)
constexpr int A_TAB = 0;                          // [1026][4]: rel bias of distance 1023 - row per head (log2 units); 1024: 0, 1025: NEG
constexpr int A_QG = 4128;                        // q (384) then gate (384)
constexpr int A_ST = 4896;                        // static block: q_clip 384, rel_clip 48, k taps 512, v taps 512
constexpr int A_RING = 6368;                      // 4 x 256
constexpr int A_OUT = 7392;                       // the ring row 256 | record 128 | O 192 (as y units)
constexpr int A_BQ = 7968, A_BR = 8352;           // q as bf16 [3][hi, lo][128], r as bf16 [3][hi, lo][32] (16 used)
constexpr int A_R = 8544;                         // 3 x 16
constexpr int A_MISC = 8608;                      // sinks 3 | k taps 4 | v taps 4
constexpr int A_KV = 8640;                        // conv outputs k, v (256)
// OP_ATTN scratch in b (floats)
constexpr int B_CUR = 0;                          // this token's record as a unit
constexpr int B_ACC = 1024;                       // P V: [3][128]
constexpr int B_M = 1408, B_L = 1440, B_TV = 1472, B_CV = 1504;   // [8][4] each: max, sum (per head c), rel bias, rescale
constexpr int B_P = 1536;                         // P as bf16 [8][4]
constexpr int B_TMP = 1568;                       // 768

#if GLUE_UP
inline void p_acquire(int id) { _up_lock_acquire(id); }
inline void p_release(int id) { _up_lock_release(id); }
#else
inline void p_acquire(int id) { _down_lock_acquire(id); }
inline void p_release(int id) { _down_lock_release(id); }
#endif

inline __attribute__((always_inline)) void mem_barrier() { __builtin_aie2p_sched_barrier(); }

using v16 = aie::vector<float, 16>;
using v32 = aie::vector<float, 32>;
// fp32 products are emulated with bf16 ones; the low accuracy variant (3 products, ~2^-16 relative) is enough here
inline __attribute__((always_inline)) v16 fmul(v16 a, v16 b) { return v16(::mul_elem_16_accuracy_low(a, b)); }
inline __attribute__((always_inline)) v16 fmul(v16 a, float b) { return fmul(a, aie::broadcast<float, 16>(b)); }

// 1 / sqrt(ss inv_n + EPS) in fp32, all lanes (the API estimate, then two Newton steps)
__attribute__((noinline)) v16 rsqrt_mean(float ss, float inv_n) {
    v16 x = aie::add(fmul(aie::broadcast<float, 16>(ss), inv_n), EPS);
    v16 r = aie::invsqrt(x);
    const v16 h = fmul(x, 0.5f);
    for (int i = 0; i < 2; i++) r = fmul(r, aie::sub(aie::broadcast<float, 16>(1.5f), fmul(h, fmul(r, r))));
    return r;
}

// sum of a[i] b[i], n a multiple of 16
float dot_f32(const float* __restrict a, const float* __restrict b, int n) {
    aie::accum<accfloat, 16> acc = aie::zeros<accfloat, 16>();
    for (int i = 0; i < n; i += 16) acc = aie::mac(acc, aie::load_v<16>(a + i), aie::load_v<16>(b + i));
    return aie::reduce_add(acc.to_vector<float>());
}
// 2^t, t rounded to the nearest integer n, 2^(t - n) to degree 5 (|t - n| <= 1/2: error 3e-6)
__attribute__((noinline)) v16 exp2_v(v16 t) {
    t = aie::max(t, -125.0f);
    const aie::vector<int32, 16> n = aie::to_fixed<int32>(t, 0);
    const v16 f = aie::sub(t, aie::to_float<float>(n, 0));
    constexpr float c[6] = {1.0f, 0.69314718f, 0.24022651f, 0.055504109f, 0.0096181291f, 0.0013333558f};
    v16 q = aie::broadcast<float, 16>(c[5]);
    for (int k = 4; k >= 0; k--) q = aie::add(fmul(q, f), c[k]);
    return aie::add(q.cast_to<int32>(), aie::upshift(n, 23)).cast_to<float>();
}
// e^x
v16 exp_f32(v16 x) { return exp2_v(fmul(x, LOG2E)); }

// x *= min(thr / |x|, 1) scale, n a multiple of 16
__attribute__((noinline)) void l2_clip(float* x, int n, float thr, v16 scale) {
    const v16 f = fmul(aie::min(fmul(rsqrt_mean(dot_f32(x, x, n), 1.0f), thr), 1.0f), scale);
    for (int i = 0; i < n; i += 16) aie::store_v(x + i, fmul(aie::load_v<16>(x + i), f));
}

// x *= w elementwise
__attribute__((noinline)) void vmul(float* x, const float* w, int n) {
    for (int i = 0; i < n; i += 16) aie::store_v(x + i, fmul(aie::load_v<16>(x + i), aie::load_v<16>(w + i)));
}

// n floats as bf16 hi + lo (x - hi rounded), n a multiple of 16
__attribute__((noinline)) void split_bf16(bf16* hi, bf16* lo, const float* x, int n) {
    for (int i = 0; i < n; i += 16) {
        const v16 v = aie::load_v<16>(x + i);
        const aie::vector<bf16, 16> h = aie::accum<accfloat, 16>(v).to_vector<bf16>();
        aie::store_v(hi + i, h);
        aie::store_v(lo + i, aie::accum<accfloat, 16>(aie::sub(v, aie::accum<accfloat, 16>(h).to_vector<float>())).to_vector<bf16>());
    }
}

// n floats to bf16
__attribute__((noinline)) void to_bf16(bf16* d, const float* x, int n) {
    for (int i = 0; i < n; i += 16) aie::store_v(d + i, aie::accum<accfloat, 16>(aie::load_v<16>(x + i)).to_vector<bf16>());
}

// 1 / x, fp32 (the API estimate, then two Newton steps)
v16 inv_f32(v16 x) {
    v16 r = aie::inv(x);
    for (int i = 0; i < 2; i++) r = fmul(r, aie::sub(aie::broadcast<float, 16>(2.0f), fmul(x, r)));
    return r;
}

// one head's conv of this token (x3) with the 3 before it, then K: l2 clip / V: rms
//   out = (x3 + sum_k (w_k + dyn_{3 - k}) x_k) * inv   (w_k: tap k of [4][128], x_0 the oldest)
__attribute__((noinline)) void conv_head(float* out, const float* x0, const float* x1, const float* x2, const float* x3,
                                         const float* w, const float* dyn, float inv) {
    const v16 d0 = aie::broadcast<float, 16>(dyn[3]), d1 = aie::broadcast<float, 16>(dyn[2]),
              d2 = aie::broadcast<float, 16>(dyn[1]), d3 = aie::broadcast<float, 16>(dyn[0]);
    for (int c = 0; c < DH; c += 16) {
        const v16 a3 = aie::load_v<16>(x3 + c);
        aie::accum<accfloat, 16> acc(a3);
        acc = aie::mac(acc, aie::add(aie::load_v<16>(w + c), d0), aie::load_v<16>(x0 + c));
        acc = aie::mac(acc, aie::add(aie::load_v<16>(w + DH + c), d1), aie::load_v<16>(x1 + c));
        acc = aie::mac(acc, aie::add(aie::load_v<16>(w + 2 * DH + c), d2), aie::load_v<16>(x2 + c));
        acc = aie::mac(acc, aie::add(aie::load_v<16>(w + 3 * DH + c), d3), a3);
        aie::store_v(out + c, fmul(acc.to_vector<float>(), inv));
    }
}

// rel bias rows of the KPU records from s0 (s0 < 0: this token as record 0, the rest masked): one load when the
// distances run down inside 1 .. 1023 (rows up) or are all >= 1024 (0)
__attribute__((noinline)) void gather_tv(float* a, float* b, int p, int s0, int nk, bool ring) {
    const float* tab = a + A_TAB;
    float* tv = b + B_TV;
    if (s0 >= 0 && s0 + KPU <= nk) {
        int d = p - s0;
        if (ring) d &= REL_EXT - 1;
        if (d - (KPU - 1) >= REL_EXT) { aie::store_v(tv, aie::zeros<float, 32>()); return; }
        if (d - (KPU - 1) >= 1 && d < REL_EXT) { aie::store_v(tv, aie::load_unaligned_v<32>(tab + 4 * (REL_EXT - 1 - d), 4)); return; }
    }
    for (int k = 0; k < KPU; k++) {
        int row;
        if (s0 < 0) row = k == 0 ? REL_EXT - 1 : REL_EXT + 1;
        else {
            const int s = s0 + k;
            int d = p - s;
            if (ring) d &= REL_EXT - 1;
            row = s >= nk || d == 0 ? REL_EXT + 1 : d >= REL_EXT ? REL_EXT : REL_EXT - 1 - d;
        }
        for (int c = 0; c < 4; c++) tv[4 * k + c] = tab[4 * row + c];
    }
}

// sum (add) or max over the 8 records of [8][4]: every lane gets its head's value
template <bool MAX> __attribute__((always_inline)) v32 over_records(v32 x) {
    v16 v = MAX ? aie::max(x.extract<16>(0), x.extract<16>(1)) : aie::add(x.extract<16>(0), x.extract<16>(1));
    for (int r = 8; r >= 4; r >>= 1) {
        const v16 w = aie::shuffle_down_rotate(v, r);
        v = MAX ? aie::max(v, w) : aie::add(v, w);
    }
    return aie::concat(v, v);
}

// the max grew by more than 8 (log2): rescale sum and P V to m_new
__attribute__((noinline)) void rescale(float* b, v32 m_new) {
    const v32 d = aie::sub(aie::load_v<32>(b + B_M), m_new);
    const v32 corr = aie::concat(exp2_v(d.extract<16>(0)), exp2_v(d.extract<16>(1)));
    aie::store_v(b + B_CV, corr);
    const v32 l = aie::load_v<32>(b + B_L);
    aie::store_v(b + B_L, aie::concat(fmul(l.extract<16>(0), corr.extract<16>(0)), fmul(l.extract<16>(1), corr.extract<16>(1))));
    for (int c = 0; c < 3; c++) {
        const v16 f = aie::broadcast<float, 16>(b[B_CV + c]);
        for (int i = DH * c; i < DH * c + DH; i += 16) aie::store_v(b + B_ACC + i, fmul(aie::load_v<16>(b + B_ACC + i), f));
    }
    aie::store_v(b + B_M, m_new);
}

#pragma clang attribute pop

#if !GLUE_UP
// OP_MIX's hot loops, pipelined (outside minsize)
// sum of a[i] b[i] with ~fp32 products, n a multiple of 16
__attribute__((noinline)) float dot_acc(const float* __restrict a, const float* __restrict b, int n) {
    v16 acc = aie::zeros<float, 16>();
    for (int i = 0; i < n; i += 16) acc = aie::add(acc, fmul(aie::load_v<16>(a + i), aie::load_v<16>(b + i)));
    return aie::reduce_add(acc);
}
// native bf16 path: an fp32 x as bf16 hi + lo, products hi hi + hi lo + lo hi into fp32 accumulators (the same 3
// products as fmul, without splitting both operands per product)
using b32 = aie::vector<bf16, 32>;
using a32 = aie::accum<accfloat, 32>;
inline __attribute__((always_inline)) void splitv(v32 f, b32& h, b32& l) {
    h = a32(f).to_vector<bf16>();
    l = a32(aie::sub(f, a32(h).to_vector<float>())).to_vector<bf16>();
}
inline __attribute__((always_inline)) void split32(const float* x, b32& h, b32& l) { splitv(aie::load_v<32>(x), h, l); }
inline __attribute__((always_inline)) v32accfloat mac3(v32accfloat acc, b32 ah, b32 al, b32 bh, b32 bl) {
    acc = ::mac_elem_32(ah, bh, acc);
    acc = ::mac_elem_32(ah, bl, acc);
    return ::mac_elem_32(al, bh, acc);
}
// rows x_i (i < n, DM floats each) in place as [24 chunks][hi 96 | lo 96] bf16
__attribute__((noinline)) void split_rows(float* x, int n) {
    for (int i = 0; i < n * DM; i += Y_UNIT) {
        b32 h0, l0, h1, l1, h2, l2;
        split32(x + i, h0, l0);
        split32(x + i + 32, h1, l1);
        split32(x + i + 64, h2, l2);
        bf16* o = (bf16*)(x + i);
        aie::store_v(o, h0); aie::store_v(o + 32, h1); aie::store_v(o + 64, h2);
        aie::store_v(o + 96, l0); aie::store_v(o + 128, l1); aie::store_v(o + 160, l2);
    }
}
// sum of a split row's squares
__attribute__((noinline)) float self_dot(const bf16* r) {
    v32accfloat a0 = a32(aie::zeros<float, 32>()), a1 = a0, a2 = a0;
    for (int c = 0; c < DM / Y_UNIT; c++, r += 2 * Y_UNIT) {
        const b32 h0 = aie::load_v<32>(r), h1 = aie::load_v<32>(r + 32), h2 = aie::load_v<32>(r + 64);
        const b32 l0 = aie::load_v<32>(r + 96), l1 = aie::load_v<32>(r + 128), l2 = aie::load_v<32>(r + 160);
        a0 = mac3(a0, h0, l0, h0, l0);
        a1 = mac3(a1, h1, l1, h1, l1);
        a2 = mac3(a2, h2, l2, h2, l2);
    }
    return aie::reduce_add(aie::add(aie::add(a32(a0).to_vector<float>(), a32(a1).to_vector<float>()), a32(a2).to_vector<float>()));
}
// pd[32 i, 32 i + 32) += lane partials of nch chunks t[96 k, 96 k + 96) . chunk c + k of split row i (xb: row 0, chunk c)
__attribute__((noinline)) void dots32(float* __restrict pd, const float* __restrict t, const bf16* __restrict xb, int ns, int nch) {
    for (int k = 0; k < nch; k++, t += Y_UNIT, xb += 2 * Y_UNIT) {
        b32 h0, l0, h1, l1, h2, l2;
        split32(t, h0, l0);
        split32(t + 32, h1, l1);
        split32(t + 64, h2, l2);
        const bf16* x = xb;
        for (int i = 0; i < ns; i++, x += 2 * DM) {
            v32accfloat acc = a32(aie::load_v<32>(pd + 32 * i));
            acc = mac3(acc, aie::load_v<32>(x), aie::load_v<32>(x + 96), h0, l0);
            acc = mac3(acc, aie::load_v<32>(x + 32), aie::load_v<32>(x + 128), h1, l1);
            acc = mac3(acc, aie::load_v<32>(x + 64), aie::load_v<32>(x + 160), h2, l2);
            aie::store_v(pd + 32 * i, a32(acc).to_vector<float>());
        }
    }
}
// one chunk of sum_i c_i x_i (c_i = ch[i] + cl[i]) as 3 x 32 lanes
inline __attribute__((always_inline)) void mix_c(v32accfloat* a, const bf16* xb, const bf16* ch, const bf16* cl, int ns) {
    a[0] = a32(aie::zeros<float, 32>()); a[1] = a[0]; a[2] = a[0];
    for (int i = 0; i < ns; i++, xb += 2 * DM) {
        const b32 h = aie::broadcast<bf16, 32>(ch[i]), l = aie::broadcast<bf16, 32>(cl[i]);
        a[0] = mac3(a[0], aie::load_v<32>(xb), aie::load_v<32>(xb + 96), h, l);
        a[1] = mac3(a[1], aie::load_v<32>(xb + 32), aie::load_v<32>(xb + 128), h, l);
        a[2] = mac3(a[2], aie::load_v<32>(xb + 64), aie::load_v<32>(xb + 160), h, l);
    }
}
// sum over all DM of (sum_i c_i x_i)^2
__attribute__((noinline)) float mix_ss(const bf16* __restrict xb, const bf16* __restrict ch, const bf16* __restrict cl, int ns) {
    v32accfloat s = a32(aie::zeros<float, 32>());
    for (int c = 0; c < DM / Y_UNIT; c++, xb += 2 * Y_UNIT) {
        v32accfloat a[3];
        mix_c(a, xb, ch, cl, ns);
        for (int k = 0; k < 3; k++) {
            b32 h, l;
            splitv(a32(a[k]).to_vector<float>(), h, l);
            s = mac3(s, h, l, h, l);
        }
    }
    return aie::reduce_add(a32(s).to_vector<float>());
}
// y[0, 192) = bf16 of 2 chunks of the mix (times g[0, 192) if g)
__attribute__((noinline)) void mix_bf16(bf16* __restrict y, const bf16* __restrict xb, const bf16* __restrict ch, const bf16* __restrict cl, int ns, const float* __restrict g) {
    for (int hh = 0; hh < 2; hh++, xb += 2 * Y_UNIT, y += Y_UNIT) {
        v32accfloat a[3];
        mix_c(a, xb, ch, cl, ns);
        for (int k = 0; k < 3; k++) {
            v32accfloat v = a[k];
            if (g) {
                b32 ah, al, bh, bl;
                splitv(a32(v).to_vector<float>(), ah, al);
                split32(g + Y_UNIT * hh + 32 * k, bh, bl);
                v = mac3(a32(aie::zeros<float, 32>()), ah, al, bh, bl);
            }
            aie::store_v(y + 32 * k, a32(v).to_vector<bf16>());
        }
    }
}
// o[0, 96) = sum_i c_i chunk of split row i, c_i = ch[i] + cl[i]
__attribute__((noinline)) void mix32(float* __restrict o, const bf16* __restrict xb, const bf16* __restrict ch, const bf16* __restrict cl, int ns) {
    v32accfloat a[3];
    mix_c(a, xb, ch, cl, ns);
    for (int k = 0; k < 3; k++) aie::store_v(o + 32 * k, a32(a[k]).to_vector<float>());
}
// o[0, n) += w t[0, n), w = wh + wl, n a multiple of 32
__attribute__((noinline)) void axpy32(float* __restrict o, const float* __restrict t, bf16 wh, bf16 wl, int n) {
    const b32 h = aie::broadcast<bf16, 32>(wh), l = aie::broadcast<bf16, 32>(wl);
    for (int k = 0; k < n; k += 32) {
        b32 th, tl;
        split32(t + k, th, tl);
        aie::store_v(o + k, a32(mac3(a32(aie::load_v<32>(o + k)), th, tl, h, l)).to_vector<float>());
    }
}
// x[0, 96) = (x + t0 t_r0 + t1 t_r1 + t2 t_r2 + t3 x) / sqrt(5): the conv taps t[0, 4 Y_UNIT), ring rows at t + r
__attribute__((noinline)) void conv32(float* __restrict x, const float* __restrict t, int r0, int r1, int r2) {
    b32 kh, kl;
    splitv(aie::broadcast<float, 32>(0.4472135954999579f), kh, kl);
    for (int k = 0; k < Y_UNIT; k += 32) {
        b32 ah, al, bh, bl;
        const v32 x3 = aie::load_v<32>(x + k);
        v32accfloat acc = a32(x3);
        split32(t + k, ah, al); split32(t + r0 + k, bh, bl); acc = mac3(acc, ah, al, bh, bl);
        split32(t + Y_UNIT + k, ah, al); split32(t + r1 + k, bh, bl); acc = mac3(acc, ah, al, bh, bl);
        split32(t + 2 * Y_UNIT + k, ah, al); split32(t + r2 + k, bh, bl); acc = mac3(acc, ah, al, bh, bl);
        split32(t + 3 * Y_UNIT + k, ah, al); splitv(x3, bh, bl); acc = mac3(acc, ah, al, bh, bl);
        splitv(a32(acc).to_vector<float>(), ah, al);
        aie::store_v(x + k, a32(mac3(a32(aie::zeros<float, 32>()), ah, al, kh, kl)).to_vector<float>());
    }
}
// o[0, 96) = t g elementwise (o may be t)
__attribute__((noinline)) void gmul32(float* o, const float* t, const float* g) {
    for (int k = 0; k < Y_UNIT; k += 32) {
        b32 ah, al, bh, bl;
        split32(t + k, ah, al);
        split32(g + k, bh, bl);
        aie::store_v(o + k, a32(mac3(a32(aie::zeros<float, 32>()), ah, al, bh, bl)).to_vector<float>());
    }
}
// x[0, n) (n <= 16) as bf16 hi, lo
__attribute__((noinline)) void split_scalars(bf16* h, bf16* l, const float* x) {
    const v16 f = aie::load_v<16>(x);
    const aie::vector<bf16, 16> hh = aie::accum<accfloat, 16>(f).to_vector<bf16>();
    aie::store_v(h, hh);
    aie::store_v(l, aie::accum<accfloat, 16>(aie::sub(f, aie::accum<accfloat, 16>(hh).to_vector<float>())).to_vector<bf16>());
}
// n bytes, a multiple of 32
__attribute__((noinline)) void copy32(uint8* __restrict d, const uint8* __restrict s, int n) {
    for (int i = 0; i < n; i += 32) aie::store_v(d + i, aie::load_v<32>(s + i));
}
#endif
// the dots of the 8 rows of n [8 rows][8 dims] bf16 tiles t (64 lanes each) with dims 8 n0 .. 8 (n0 + n) of the 3
// heads' q (bf16 hi at q + 2 qs c, lo qs after): native 64-lane bf16 macs on q chunks broadcast to every row, then the
// sums over each row's 8 lanes -> [8 rows][4 (head, 0)]
// q chunk ch of x (8 bf16) on every row of a [8][8] tile
__attribute__((always_inline)) aie::vector<bf16, 64> q_rows(const bf16* x, int ch) {
    const aie::vector<bf16, 32> h =
        aie::vector<int32, 16>(::broadcast_elem_128(aie::load_v<32>(x + 32 * (ch >> 2)).cast_to<int32>(), ch & 3)).cast_to<bf16>();
    return aie::concat(h, h);
}
// the sums over each row's 8 lanes of [8 rows][8 dims]
__attribute__((always_inline)) aie::vector<float, 8> row_sums(v64accfloat acc) {
    const aie::vector<float, 64> x = aie::accum<accfloat, 64>(acc).to_vector<float>();
    const auto [x0, x1] = aie::interleave_unzip(x.extract<32>(0), x.extract<32>(1), 4);
    const v32 y = aie::add(x0, x1);
    const auto [y0, y1] = aie::interleave_unzip(y.extract<16>(0), y.extract<16>(1), 2);
    const v16 z = aie::add(y0, y1);
    const auto [z0, z1] = aie::interleave_unzip(z.extract<8>(0), z.extract<8>(1), 1);
    return aie::add(z0, z1);
}
__attribute__((noinline)) v32 rows_dot(const bf16* q, int qs, const bf16* t, int n0, int n) {
    v64accfloat s0 = aie::zeros<accfloat, 64>(), s1 = s0, s2 = s0;
    for (int i = 0; i < n; i++) {
        const int ch = n0 + i;
        const aie::vector<bf16, 64> kk = aie::load_v<64>(t + 64 * i);
        s0 = ::mac_elem_64(kk, q_rows(q, ch), s0);
        s1 = ::mac_elem_64(kk, q_rows(q + 2 * qs, ch), s1);
        s2 = ::mac_elem_64(kk, q_rows(q + 4 * qs, ch), s2);
        s0 = ::mac_elem_64(kk, q_rows(q + qs, ch), s0);
        s1 = ::mac_elem_64(kk, q_rows(q + 3 * qs, ch), s1);
        s2 = ::mac_elem_64(kk, q_rows(q + 5 * qs, ch), s2);
    }
    const auto [a0, a1] = aie::interleave_zip(row_sums(s0), row_sums(s1), 1);
    const auto [b0, b1] = aie::interleave_zip(row_sums(s2), aie::zeros<float, 8>(), 1);
    const auto [c0, c1] = aie::interleave_zip(aie::concat(a0, a1), aie::concat(b0, b1), 2);
    return aie::concat(c0, c1);
}

// 2^t in bf16: n = t rounded, 2^(t - n) to degree 3 (error 1e-4) with bf16 products (t - n as hi + lo in the
// last), then n added to the exponent
__attribute__((always_inline)) aie::vector<bf16, 32> exp2_p(v32 t) {
    using acc32 = aie::accum<accfloat, 32>;
    using b32 = aie::vector<bf16, 32>;
    t = aie::max(t, -120.0f);
    const aie::vector<int32, 32> n = aie::to_fixed<int32>(t, 0);
    const v32 f = aie::sub(t, aie::to_float<float>(n, 0));
    const b32 fh = acc32(f).to_vector<bf16>();
    const b32 fl = acc32(aie::sub(f, acc32(fh).to_vector<float>())).to_vector<bf16>();
    acc32 q = aie::mac(acc32(aie::broadcast<float, 32>(0.242640083f)), fh, aie::broadcast<bf16, 32>(0.0559220356f));
    q = aie::mac(acc32(aie::broadcast<float, 32>(0.693121034f)), fh, q.to_vector<bf16>());
    const b32 qb = q.to_vector<bf16>();
    q = aie::mac(aie::mac(acc32(aie::broadcast<float, 32>(0.999924481f)), fh, qb), fl, qb);
    return acc32(aie::add(q.to_vector<float>().cast_to<int32>(), aie::upshift(n, 23)).cast_to<float>()).to_vector<bf16>();
}

// online softmax over one unit of KPU records (scores in log2 units, rel bias rows in tv); P V into b's accumulators
__attribute__((noinline)) void attn_unit(const float* a, float* b, const bf16* u, int dbg) {
    const v32 sc = aie::add((dbg & 1) ? aie::zeros<float, 32>() : rows_dot((const bf16*)(a + A_BQ), DH, u, 0, DH / 8), aie::load_v<32>(b + B_TV));
    const v32 mx = aie::max(over_records<true>(sc), aie::load_v<32>(b + B_M));
    if (!aie::gt(mx, aie::add(aie::load_v<32>(b + B_M), 8.0f)).empty()) rescale(b, mx);
    // p = 2^(s - m) in bf16 (native exp2); the sum and P V use the same rounded p
    const aie::vector<bf16, 32> ph = aie::select(exp2_p(aie::sub(sc, aie::load_v<32>(b + B_M))), aie::zeros<bf16, 32>(),
                                                 aie::lt(sc, 0.5f * NEG) | aie::mask<32>::from_uint32(0x88888888u));
    aie::store_v(b + B_L, aie::add(aie::load_v<32>(b + B_L), over_records<false>(aie::accum<accfloat, 32>(ph).to_vector<float>())));
    bf16* pp = (bf16*)(b + B_P);
    aie::store_v(pp, ph);
    if (dbg & 2) return;
    // P V: dims in lanes, p broadcast
    const bf16* v = u + DH * KPU;
    for (int c = 0; c < 3; c++) {
        float* ap = b + B_ACC + DH * c;
        v64accfloat o0 = aie::accum<accfloat, 64>(aie::load_v<64>(ap)), o1 = aie::accum<accfloat, 64>(aie::load_v<64>(ap + 64));
        for (int k = 0; k < KPU; k++) {
            const aie::vector<bf16, 64> v0 = aie::load_v<64>(v + DH * k), v1 = aie::load_v<64>(v + DH * k + 64);
            const aie::vector<bf16, 64> h = aie::broadcast<bf16, 64>(pp[4 * k + c]);
            o0 = ::mac_elem_64(v0, h, o0);
            o1 = ::mac_elem_64(v1, h, o1);
        }
        aie::store_v(ap, aie::accum<accfloat, 64>(o0).to_vector<float>());
        aie::store_v(ap + 64, aie::accum<accfloat, 64>(o1).to_vector<float>());
    }
}
#pragma clang attribute push(__attribute__((minsize)), apply_to = function)
}  // namespace

extern "C" void GLUE_NAME(uint8* w0, uint8* w1, uint8* w2, uint8* w3, float* y0, float* y1, volatile int32_t* rl,
                          float* a, float* b) {
    uint8* const W[RING] = {w0, w1, w2, w3};
    float* const Y[2] = {y0, y1};
    aie::set_rounding(aie::rounding_mode::conv_even);
    p_acquire(GL_GO);
    mem_barrier();
    int wi = rl[0], yi = rl[1];
    const uint8* cur = W[wi];   // the header, taken by the partner
    int off = X_OFF, ui = 0;    // ui: units of the task taken so far
    auto give = [&]() { p_release(W_PROD); if (++wi == RING) wi = 0; };
    auto take = [&]() -> const uint8* { p_acquire(W_CONS); return W[wi]; };
    auto y_get = [&]() -> float* { p_acquire(Y_PROD); return Y[yi]; };
    auto y_put = [&]() { p_release(Y_CONS); yi ^= 1; };
    // n bytes (a multiple of 32) of the task's inputs, continuing into the following units
    auto read = [&](uint8* d, int n) {
        while (n > 0) {
            if (off == UNIT) { give(); cur = take(); off = 0; ui++; }
            const int k = n < UNIT - off ? n : UNIT - off;
#if GLUE_UP
            for (int i = 0; i < k; i += 32) aie::store_v(d + i, aie::load_v<32>(cur + off + i));
#else
            copy32(d, cur + off, k);
#endif
            d += k; n -= k; off += k;
        }
    };
#if !GLUE_UP
    // the next n bytes in place when they lie in one ring unit (valid until the next read / view), else copied to tmp;
    // consumes exactly what read() would
    auto view = [&](float* tmp, int n) -> const float* {
        if (off == UNIT) { give(); cur = take(); off = 0; ui++; }
        if (off + n > UNIT || (off & 63)) { read((uint8*)tmp, n); return tmp; }
        const float* r = (const float*)(cur + off);
        off += n;
        return r;
    };
#endif
    // to the start of unit n of the task
    auto unit = [&](int n) { while (ui < n) { give(); cur = take(); ui++; } off = 0; };
    // n floats (a multiple of 96) as y units
    auto write = [&](const float* s, int n) {
        for (int i = 0; i < n; i += Y_UNIT) {
            float* y = y_get();
            for (int k = 0; k < Y_UNIT; k += 16) aie::store_v(y + k, aie::load_v<16>(s + i + k));
            y_put();
        }
    };

#if !GLUE_UP
    // DM floats s as 36 y units: bf16 hi, lo, hi (t: 2 y units of scratch)
    auto write_hlh = [&](const float* s, float* t) {
        for (int pass = 0; pass < 3; pass++)
            for (int c = 0; c < DM; c += 2 * Y_UNIT) {
                float* y = y_get();
                split_bf16((bf16*)t, (bf16*)t + 2 * Y_UNIT, s + c, 2 * Y_UNIT);
                const bf16* src = (const bf16*)t + (pass == 1 ? 2 * Y_UNIT : 0);
                for (int k = 0; k < 2 * Y_UNIT; k += 32) aie::store_v((bf16*)y + k, aie::load_v<32>(src + k));
                y_put();
            }
    };
#endif

    const volatile int* h = (const volatile int*)cur;
    const int op = h[0], a1 = h[1];
#if GLUE_UP
    if (op == OP_RMS) {
        const int n = a1;
        read((uint8*)a, 4 * n);
        read((uint8*)b, 4 * n);
        give();
        const v16 s = rsqrt_mean(dot_f32(a, a, n), 1.0f / 2304);
        for (int i = 0; i < n; i += 16) aie::store_v(a + i, fmul(fmul(aie::load_v<16>(a + i), s), aie::load_v<16>(b + i)));
        write(a, n);
    }
    else if (op == OP_ATTN) {
        const int p = a1, glob = h[2], nk = h[4], ring = h[5], g = h[6], dbg = h[7];
        const float tau = ((const volatile float*)h)[3];
        read((uint8*)(a + A_QG), 768 * 4);
        // the A block: K, V, R, MISC
        read((uint8*)b, 2784 * 4);
        for (int c = 0; c < 256; c += 16) aie::store_v(a + A_OUT + c, aie::load_v<16>(b + (c < DH ? 128 * g + c : 1056 + 128 * g + c - DH)));
        for (int c = 0; c < 48; c += 16) aie::store_v(a + A_R + c, aie::load_v<16>(b + 2112 + 48 * g + c));
        float* misc = a + A_MISC;
        for (int i = 0; i < 3; i++) misc[i] = b[2496 + 3 * g + i];
        for (int i = 0; i < 4; i++) { misc[4 + i] = glob ? b[2496 + 96 + 4 * g + i] : 0.0f; misc[8 + i] = glob ? b[2496 + 192 + 4 * g + i] : 0.0f; }
        unit(U_ST);
        read((uint8*)(a + A_ST), 1456 * 4);
        // q: clip, log scale, 1 / sqrt(128), log2 e; r: clip, log scale, log2 e; as B tiles (bf16 hi + lo)
        float* q = a + A_QG;
        vmul(q, a + A_ST, 384);
        const v16 vtau = fmul(aie::broadcast<float, 16>(glob ? tau : 1.0f), LOG2E);
        for (int i = 0; i < 3; i++) l2_clip(q + DH * i, DH, QK_CLIP, fmul(vtau, 0.08838834764831845f));
        for (int i = 0; i < 3; i++) split_bf16((bf16*)(a + A_BQ) + 2 * DH * i, (bf16*)(a + A_BQ) + 2 * DH * i + DH, q + DH * i, DH);
        float* r = a + A_R;
        vmul(r, a + A_ST + 384, 48);
        for (int i = 0; i < 3; i++) l2_clip(r + NREL * i, NREL, REL_CLIP, vtau);
        for (int i = 0; i < 3; i++) split_bf16((bf16*)(a + A_BR) + 64 * i, (bf16*)(a + A_BR) + 64 * i + 32, r + NREL * i, NREL);
        // tab[d] = rel_bias[d] . r: the first 8 dims, then the second
        float* tab = a + A_TAB;
        for (int i = 0; i < RB_TILES; i++) {
            unit(U_RB + i / TPU);
            const v32 x = rows_dot((const bf16*)(a + A_BR), 32, (const bf16*)(cur + 128 * (i % TPU)), i >> 7, 1);
            float* tp = tab + 32 * (i & 127);
            aie::store_v(tp, i < 128 ? x : aie::add(x, aie::load_v<32>(tp)));
        }
        for (int c = 0; c < 4; c++) { tab[4 * REL_EXT + c] = 0.0f; tab[4 * REL_EXT + 4 + c] = NEG; }
        // the conv: rows p - 3 .. p - 1 of the ring, then this token's raw k, v
        unit(U_RING);
        read((uint8*)(a + A_RING), 1024 * 4);
        const float* x0 = a + A_RING + 256 * ((p - 3) & 3);
        const float* x1 = a + A_RING + 256 * ((p - 2) & 3);
        const float* x2 = a + A_RING + 256 * ((p - 1) & 3);
        const float* x3 = a + A_OUT;
        const float inv = glob ? 0.3333333333333333f : 0.4472135954999579f;
        float* kvf = a + A_KV;
        conv_head(kvf, x0, x1, x2, x3, a + A_ST + 432, misc + 4, inv);
        conv_head(kvf + DH, x0 + DH, x1 + DH, x2 + DH, x3 + DH, a + A_ST + 432 + 512, misc + 8, inv);
        l2_clip(kvf, DH, QK_CLIP, aie::broadcast<float, 16>(1.0f));
        {
            const v16 s = rsqrt_mean(dot_f32(kvf + DH, kvf + DH, DH), 1.0f / DH);
            for (int c = DH; c < 2 * DH; c += 16) aie::store_v(kvf + c, fmul(aie::load_v<16>(kvf + c), s));
        }
        bf16* rec = (bf16*)(a + A_OUT + 256);
        to_bf16(rec, kvf, 2 * DH);
        // this token's record as a unit (record 0)
        float* cu = b + B_CUR;
        for (int c = 0; c < 1024; c += 16) aie::store_v(cu + c, aie::zeros<float, 16>());
        for (int n = 0; n < 16; n++)
            for (int j = 0; j < 4; j++) ((uint32_t*)cu)[32 * n + j] = ((const uint32_t*)rec)[4 * n + j];
        for (int c = 0; c < DH; c += 16) aie::store_v((bf16*)cu + DH * KPU + c, aie::load_v<16>(rec + DH + c));
        // softmax state: the sinks (value 0) first, then this token's key, then the cache
        {
            const v16 sk = fmul(aie::load_v<16>(misc), LOG2E);
            alignas(aie::vector_decl_align) float m0[16];
            aie::store_v(m0, sk);
            for (int i = 0; i < 32; i++) b[B_M + i] = (i & 3) < 3 ? m0[i & 3] : 0.0f;
        }
        for (int c = 0; c < 32; c += 16) aie::store_v(b + B_L + c, aie::broadcast<float, 16>(1.0f));
        for (int c = 0; c < 3 * DH; c += 16) aie::store_v(b + B_ACC + c, aie::zeros<float, 16>());
        gather_tv(a, b, p, -1, 0, false);
        attn_unit(a, b, (const bf16*)cu, 0);
        for (int s0 = 0; s0 < nk; s0 += KPU) {
            unit(U_REC + s0 / KPU);
            if (!(dbg & 4)) gather_tv(a, b, p, s0, nk, ring);
            if (!(dbg & 8)) attn_unit(a, b, (const bf16*)cur, dbg);
        }
        give();
        float* acc = b + B_ACC;
        float* ml = b + B_L;
        // O = acc / l * sigmoid(gate)
        float* gt = a + A_QG + 384;
        for (int i = 0; i < 3; i++) {
            const v16 rl = aie::broadcast<float, 16>(ml[i]);
            float* ap = acc + DH * i;
            // acc / (l (1 + e^-gate))
            for (int c = 0; c < DH; c += 16) {
                const v16 e = exp_f32(aie::max(aie::min(aie::neg(aie::load_v<16>(gt + DH * i + c)), 80.0f), -80.0f));
                aie::store_v(ap + c, fmul(aie::load_v<16>(ap + c), inv_f32(fmul(aie::add(e, 1.0f), rl))));
            }
        }
        to_bf16((bf16*)(a + A_OUT + 384), acc, 384);
        write(a + A_OUT, 576);
    }
#else
    if (op == OP_MIX) {
        const int pre = a1, fl = h[2], p = h[3], S = h[4], nq = h[5], om = h[6], nn = h[7];
        float* const cu = a + S * DM;
        float* const t = a + M_T;
        float* const sc = a + M_SC;   // [0, 8) expert weights, [8, 13) 1 / rms, [16, 32) mix weights
        read((uint8*)a, 4 * S * DM);
        if (pre == 1) {
            read((uint8*)t, 4 * Y_UNIT);
            for (int e = 0; e < 8; e++) sc[e] = t[8 + e];
            alignas(aie::vector_decl_align) bf16 wh[16], wl[16];
            split_scalars(wh, wl, sc);
            for (int c = 0; c < DM; c += 16) aie::store_v(cu + c, aie::zeros<float, 16>());
            for (int s = 0; s < 16; s++) {
                float* o = cu + (s >> 3) * 1152;
                for (int u = 0; u < 12; u += 4, o += 4 * Y_UNIT) axpy32(o, view(t, 16 * Y_UNIT), wh[s & 7], wl[s & 7], 4 * Y_UNIT);
            }
        } else {
            read((uint8*)cu, 4 * DM);
            if (pre == 2)
                for (int c = 0; c < DM; c += Y_UNIT) {
                    read((uint8*)t, 4 * Y_UNIT);
                    for (int k = 0; k < Y_UNIT; k += 16) aie::store_v(cu + c + k, aie::add(aie::load_v<16>(cu + c + k), aie::load_v<16>(t + k)));
                }
        }
        if (fl & 1) {
            write(cu, DM);
            const int r0 = 4 * Y_UNIT + ((p - 3) & 3) * Y_UNIT, r1 = 4 * Y_UNIT + ((p - 2) & 3) * Y_UNIT, r2 = 4 * Y_UNIT + ((p - 1) & 3) * Y_UNIT;
            for (int c = 0; c < DM; c += Y_UNIT) {
                conv32(cu + c, view(t, 8 * 4 * Y_UNIT), r0, r1, r2);
            }
        }
        if (fl & 2)
            for (int c = 0; c < DM; c += Y_UNIT) {
                read((uint8*)t, 4 * Y_UNIT);
                for (int k = 0; k < Y_UNIT; k += 16) aie::store_v(cu + c + k, aie::add(aie::load_v<16>(cu + c + k), aie::load_v<16>(t + k)));
            }
        if (fl & 4) write(cu, DM);
        const int ns = S + 1;
        const bf16* const xb = (const bf16*)a;   // the rows split as [chunk][hi | lo] from here on (cur: until om 2 rewrites it)
        if (nq > 0) {
            split_rows(a, ns);
            for (int i = 0; i < ns; i++) sc[8 + i] = rsqrt_mean(self_dot(xb + 2 * i * DM), 1.0f / DM).get(0);
        }
        for (int j = 0; j < nq; j++) {
            // q . x_i over the streamed query
            alignas(aie::vector_decl_align) float d[16];
            alignas(64) float pd[5 * 32];
            for (int i = 0; i < 5 * 32; i += 16) aie::store_v(pd + i, aie::zeros<float, 16>());
            for (int c = 0; c < DM; c += 4 * Y_UNIT) {
                const float* q = view(t, 16 * Y_UNIT);
                dots32(pd, q, xb + 2 * c, ns, 4);
            }
            for (int i = 0; i < 16; i++) d[i] = i < ns ? aie::reduce_add(aie::load_v<32>(pd + 32 * i)) : 0.0f;
            // softmax over the live lanes (vector ops; no scalar float math)
            const auto live = aie::mask<16>::from_uint32((1u << ns) - 1);
            const v16 rs = aie::load_unaligned_v<16>(sc + 8);
            const v16 dv = aie::select(aie::broadcast<float, 16>(NEG), fmul(aie::load_v<16>(d), fmul(rs, 1.0f / 48)), live);
            const v16 e = exp_f32(aie::sub(dv, aie::broadcast<float, 16>(aie::reduce_max(dv))));
            const v16 r = inv_f32(aie::broadcast<float, 16>(aie::reduce_add(e)));
            float* cw = sc + 16;
            aie::store_v(cw, aie::select(aie::zeros<float, 16>(), fmul(fmul(e, r), rs), live));
            alignas(aie::vector_decl_align) bf16 ch[16], cl[16];
            split_scalars(ch, cl, cw);
            if (om == 0 || om == 3) {
                for (int c = 0; c < DM; c += Y_UNIT) { float* y = y_get(); mix32(y, xb + 2 * c, ch, cl, ns); y_put(); }
                if (om == 0) continue;
            }
            if (nn || om == 3) {   // 1 / rms of the mix, folded into its weights
                const float ss = mix_ss(xb, ch, cl, ns);
                aie::store_v(cw, fmul(aie::load_v<16>(cw), rsqrt_mean(ss, 1.0f / DM)));
                split_scalars(ch, cl, cw);
            }
            if (om != 2) {
                for (int c = 0; c < DM; c += 2 * Y_UNIT) {
                    float* y = y_get();
                    const float* g = nn ? view(t, 8 * Y_UNIT) : nullptr;
                    mix_bf16((bf16*)y, xb + 2 * c, ch, cl, ns, g);
                    y_put();
                }
            } else {   // in place into cur, then hi, lo, hi
                for (int c = 0; c < DM; c += Y_UNIT) {
                    mix32(t, xb + 2 * c, ch, cl, ns);
                    if (nn) gmul32(cu + c, t, view(t + Y_UNIT, 4 * Y_UNIT));
                    else copy32((uint8*)(cu + c), (const uint8*)t, 4 * Y_UNIT);
                }
                write_hlh(cu, t);
            }
        }
        give();
    } else if (op == OP_CTOUT) {
        float* const x = a;
        const float* const m = a + DM;
        float* const t = a + M_T;
        read((uint8*)a, 8 * DM);
        for (int c = 0; c < DM; c += 48) {
            const float* fa = view(t, 4 * Y_UNIT);
            for (int i = 0; i < 48; i += 16) {
                const v16 xv = aie::load_v<16>(x + c + i);
                aie::store_v(x + c + i, aie::add(aie::sub(xv, fmul(aie::load_v<16>(fa + 2 * i), xv)),
                                                 fmul(aie::load_v<16>(fa + 2 * i + 16), aie::load_v<16>(m + c + i))));
            }
        }
        const v16 r = rsqrt_mean(dot_acc(x, x, DM), 1.0f / DM);
        for (int c = 0; c < DM; c += Y_UNIT) {
            const float* g = view(t, 4 * Y_UNIT);
            for (int k = 0; k < Y_UNIT; k += 16) aie::store_v(x + c + k, fmul(fmul(aie::load_v<16>(x + c + k), r), aie::load_v<16>(g + k)));
        }
        if (a1 == 1)
            for (int c = 0; c < DM; c += 2 * Y_UNIT) { float* y = y_get(); to_bf16((bf16*)y, x + c, 2 * Y_UNIT); y_put(); }
        else write_hlh(x, t);
        give();
    }
#endif
    mem_barrier();
    rl[0] = wi;
    rl[1] = yi;
    mem_barrier();
    p_release(GL_DONE);
}
#pragma clang attribute pop
