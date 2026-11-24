// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef CONV_KERNEL_H
#define CONV_KERNEL_H
#include <aie_api/aie.hpp>
#include "aie_kernel_utils.h"
#include <math.h>

// ---------------------------------------------------------------------------
// One token of the conv1d prefill, for the 192 channels this core owns: one
// 128-wide q/k head (channels 0..127) and 64 v channels (128..191):
//     a[c] = x_0[c] w_0[c] + x_1[c] w_1[c] + x_2[c] w_2[c] + x_3[c] w_3[c]
//     y[c] = bf16(silu(a[c]))
// and an L2 norm over the head. Every core does the same work.
// Everything works on 64 lanes: the bf16 MAC unit is natively 64 wide.
//
// fp32 products use bf16 "limbs": an fp32 x is split into bf16 pieces
// x ~= l0 + l1, each multiplied on the native bf16 MAC unit, summed in the fp32
// accumulator. 2 limbs give ~16 bits, plenty for a bf16 result.
//
// SiLU, the trick: silu(a) = a * sigmoid(a), with
//     e = 2^(-|a| log2(e))  in (0, 1]       see exp2_accurate() below
//     r = 1 / (1 + e)       in [0.5, 1)     well-conditioned: 1 + e is in (1, 2]
//     sigmoid(a) = r        if a >= 0
//                = e * r    if a <  0       a product, not 1 - r: no cancellation
// Using |a| means e never overflows, so no clamping is needed. r starts from
// the fast bf16 reciprocal (aie::inv on bf16, ~8 bits) and gets one Newton step,
// r <- r + r (1 - d r), which squares the error to ~16 bits.
//
// L2 norm: y * 1/sqrt(sum(y^2) + eps), with y the bf16-rounded SiLU output, as
// the engine's host reference does. The outputs stay in registers; 1/sqrt is
// the bf16 aie::invsqrt (~8 bits) plus one Newton step, s <- s + s (1 - x s^2) / 2.
//
// Measured against the host reference (fp32 SiLU, RNE to bf16): ~97% of outputs
// are exactly the reference, the rest one bf16 step off, unbiased.
// ---------------------------------------------------------------------------

constexpr int CH = 64;                  // lanes per step
using vbf = aie::vector<bf16, CH>;
using acc = aie::accum<accfloat, CH>;

// -log2(e) = -1.44269504... as two bf16 limbs (both exact bf16 values), so
// t = -|a| log2(e) keeps ~17 bits: -1.4453125 + 0.00262451171875
constexpr bf16 NEG_LOG2E_0 = -1.4453125f;
constexpr bf16 NEG_LOG2E_1 = 0.00262451171875f;

inline vbf ones_bf() { return aie::broadcast<bf16, CH>((bf16)1.0f); }

// 2^f = 1 + f (C1 + f (C2 + f (C3 + f C4))), a degree-4 fit on f in [-0.5, 1]
// (4.2e-5 from 2^f in exact arithmetic). [-0.5, 1] rather than [-0.5, 0.5]
// because the magic-number "round(t)" below gives round(t) or floor(t)
// depending on how the accumulator add rounds, and this range is right either
// way. All values are exact bf16; C1 (~ln 2) needs two limbs because C1 f is the
// biggest term.
constexpr bf16 EXP2_C1_0 = 0.69140625f;
constexpr bf16 EXP2_C1_1 = 0.00173187255859375f;
constexpr bf16 EXP2_C2   = 0.240234375f;
constexpr bf16 EXP2_C3   = 0.0556640625f;
constexpr bf16 EXP2_C4   = 0.01116943359375f;
// 1.5 * 2^23: adding it to an fp32 drops the fraction (to nearest, or down)
constexpr float ROUND_MAGIC = 12582912.0f;

// x ~= l0 + l1 (bf16 limbs of an fp32 accumulator). x - l0 is exact in fp32.
inline void split2(const acc& x, vbf& l0, vbf& l1)
{
    l0 = x.template to_vector<bf16>();
    acc r = aie::msc(x, l0, ones_bf());
    l1 = r.template to_vector<bf16>();
}

// x * y for two fp32 values held as limbs: the 3 products above ~2^-16
inline acc mul2(const vbf& x0, const vbf& x1, const vbf& y0, const vbf& y1)
{
    acc p = aie::mul(x1, y0);
    p = aie::mac(p, x0, y1);
    p = aie::mac(p, x0, y0);
    return p;
}

// a for 64 channels: the 4-tap causal conv, fp32 accumulation
inline acc conv64(const bf16* __restrict x_0, const bf16* __restrict x_1,
                  const bf16* __restrict x_2, const bf16* __restrict x_3,
                  const bf16* __restrict w_0, const bf16* __restrict w_1,
                  const bf16* __restrict w_2, const bf16* __restrict w_3)
{
    acc a = aie::mul(aie::load_v<CH>(x_0), aie::load_v<CH>(w_0));
    a = aie::mac(a, aie::load_v<CH>(x_1), aie::load_v<CH>(w_1));
    a = aie::mac(a, aie::load_v<CH>(x_2), aie::load_v<CH>(w_2));
    a = aie::mac(a, aie::load_v<CH>(x_3), aie::load_v<CH>(w_3));
    return a;
}

// 2^t for t <= 0, as two bf16 limbs e0 + e1, accurate to ~1e-3 relative.
// The native exp2 is not used at all: it is a linear 2^n (1 + f) approximation,
// off by up to 6% (2^-0.3926 came out 0.8047 instead of 0.7617), and not even
// exact at whole numbers (it came out ~0.4% low there). Instead:
//     t = r + f,  r = round(t) or floor(t) (magic-number add, exact),  f in [-0.5, 1)
//     2^t = 2^r (1 + f (C1 + f (C2 + f (C3 + f C4))))
// 2^r is built from bits: after the magic add, the low 16 bits of the fp32 hold
// r as an int16, and (r + 127) << 7 is the bf16 bit pattern of 2^r, exactly.
// 2^r is a power of two, so f * 2^r is exact in bf16 and the last Horner step
// can produce 2^t directly: 2^r + mid * (f 2^r). f and mid are kept as single
// bf16 values: each costs < 2^-10 of 2^t, and saves a split (two conversions).
__attribute__((always_inline)) inline void exp2_accurate(const acc& t, vbf& e0, vbf& e1)
{
    const vbf one = ones_bf();
    acc magic;
    magic.from_vector(aie::broadcast<float, CH>(ROUND_MAGIC));
    const acc rm = aie::add(t, magic);                         // fraction dropped
    const acc r = aie::sub(rm, magic);                         // whole number, exact
    const acc f = aie::sub(t, r);                              // t - r, exact
    const vbf f0 = f.template to_vector<bf16>();

    // 2^r from bits: the even int16 halves of rm are its low 16 bits = r.
    // r below -127 is clamped there, which makes 2^r exactly 0.
    const aie::vector<int16, CH> r_i16 =
        aie::filter_even(rm.template to_vector<float>().template cast_to<int16>(), 1);
    const aie::vector<int16, CH> biased =
        aie::add(aie::max(r_i16, aie::broadcast<int16, CH>(-127)), aie::broadcast<int16, CH>(127));
    const vbf pow2 = aie::upshift(biased, 7).template cast_to<bf16>();

    acc i1 = aie::mul(f0, aie::broadcast<bf16, CH>(EXP2_C4));
    i1 = aie::mac(i1, aie::broadcast<bf16, CH>(EXP2_C3), one);
    acc inner = aie::mul(i1.template to_vector<bf16>(), f0);
    inner = aie::mac(inner, aie::broadcast<bf16, CH>(EXP2_C2), one);
    const vbf inner_b = inner.template to_vector<bf16>();
    acc mid = aie::mul(inner_b, f0);
    mid = aie::mac(mid, aie::broadcast<bf16, CH>(EXP2_C1_1), one);
    mid = aie::mac(mid, aie::broadcast<bf16, CH>(EXP2_C1_0), one);
    const vbf mid_b = mid.template to_vector<bf16>();

    const vbf f_scaled = aie::mul(f0, pow2).template to_vector<bf16>();   // exact
    acc e = aie::mul(pow2, one);
    e = aie::mac(e, mid_b, f_scaled);
    split2(e, e0, e1);
}

// silu(a), fp32 accumulator in, fp32 accumulator out (not yet rounded).
__attribute__((always_inline)) inline acc silu64(const acc& a)
{
    const vbf one = ones_bf();
    vbf a0, a1;
    split2(a, a0, a1);                       // a's limbs: used for t and for the result

    // sign from the big limb (it has a's sign), then t = -|a| log2(e) as
    // a * (+-log2(e)): the constant's sign is picked per lane, no abs needed
    const auto negative = aie::lt(a0, aie::zeros<bf16, CH>());
    const vbf c0 = aie::select(aie::broadcast<bf16, CH>(NEG_LOG2E_0), aie::broadcast<bf16, CH>(-NEG_LOG2E_0), negative);
    const vbf c1 = aie::select(aie::broadcast<bf16, CH>(NEG_LOG2E_1), aie::broadcast<bf16, CH>(-NEG_LOG2E_1), negative);
    acc t = aie::mul(a1, c0);
    t = aie::mac(t, a0, c1);
    t = aie::mac(t, a0, c0);

    // e = 2^t in (0, 1] as two limbs
    vbf e0, e1;
    exp2_accurate(t, e0, e1);

    // r = 1/(1 + e): bf16 guess from bf16(1 + e), then one Newton step
    // r + r (1 - (1 + e) r), with the exact e limbs in the error term. The step's
    // correction r0 * err is small, so its bf16 rounding is the second limb of r.
    acc d = aie::mul(one, one);
    d = aie::mac(d, e0, one);
    const vbf r0 = aie::inv(d.template to_vector<bf16>());
    acc err = aie::mul(one, one);
    err = aie::msc(err, r0, one);
    err = aie::msc(err, e1, r0);
    err = aie::msc(err, e0, r0);
    const vbf r1 = aie::mul(r0, err.template to_vector<bf16>()).template to_vector<bf16>();

    // sigmoid = m r with m = e (a < 0) or 1 (a >= 0), m picked per limb
    const vbf m0 = aie::select(one, e0, negative);
    const vbf m1 = aie::select(aie::zeros<bf16, CH>(), e1, negative);
    const acc q = mul2(m0, m1, r0, r1);

    // a * sigmoid
    vbf q0, q1;
    split2(q, q0, q1);
    return mul2(a0, a1, q0, q1);
}

// The three SiLU blocks of a token in one body. Each is one long chain of
// dependent steps, so on its own the core mostly waits; with three chains in the
// same basic block the scheduler fills some of one chain's waits with the
// others. It is a call, not inlined into the caller: inlined, the register
// spills went past what chess can address on the stack (a build error).
__attribute__((noinline)) void silu64_triple(const acc& a, const acc& b, const acc& c, vbf& ya, vbf& yb, vbf& yc)
{
    ya = silu64(a).template to_vector<bf16>();
    yb = silu64(b).template to_vector<bf16>();
    yc = silu64(c).template to_vector<bf16>();
}

// 1 / sqrt(x) for x > 0 in all lanes (fp32): bf16 guess + one Newton step
inline acc rsqrt64(const acc& x)
{
    vbf x0, x1;
    split2(x, x0, x1);
    const vbf s = aie::invsqrt(x0);
    // err = 1 - x s^2; s^2 is exact in fp32 (bf16 x bf16)
    const acc s2 = aie::mul(s, s);
    vbf q0, q1;
    split2(s2, q0, q1);
    acc err;
    err.from_vector(aie::broadcast<float, CH>(1.0f));
    err = aie::sub(err, mul2(x0, x1, q0, q1));
    // s + s * err / 2
    acc half_err = aie::mul(err.template to_vector<bf16>(), aie::broadcast<bf16, CH>((bf16)0.5f));
    acc out = aie::mul(s, ones_bf());
    out = aie::mac(out, s, half_err.template to_vector<bf16>());
    return out;
}

// conv + SiLU for the 192 channels of one token, then the L2 norm of the head
// (channels 0..127). The head's two output vectors stay in registers.
inline void conv_silu_norm_192(
    const bf16* __restrict x_0, const bf16* __restrict x_1, const bf16* __restrict x_2, const bf16* __restrict x_3,
    const bf16* __restrict w_0, const bf16* __restrict w_1, const bf16* __restrict w_2, const bf16* __restrict w_3,
    bf16* __restrict y)
{
    const float epsilon = 1e-6f;

    vbf y0, y1, yv;                // the head (two halves), then the v channels
    silu64_triple(conv64(x_0,       x_1,       x_2,       x_3,       w_0,       w_1,       w_2,       w_3),
                  conv64(x_0 + 64,  x_1 + 64,  x_2 + 64,  x_3 + 64,  w_0 + 64,  w_1 + 64,  w_2 + 64,  w_3 + 64),
                  conv64(x_0 + 128, x_1 + 128, x_2 + 128, x_3 + 128, w_0 + 128, w_1 + 128, w_2 + 128, w_3 + 128),
                  y0, y1, yv);
    aie::store_v(y + 128, yv);

    // sum of squares of the rounded head outputs, as the host reference does
    acc ss = aie::mul(y0, y0);
    ss = aie::mac(ss, y1, y1);
    const float sum = aie::reduce_add(ss.template to_vector<float>());
    acc x;
    x.from_vector(aie::broadcast<float, CH>(sum));
    x = aie::add(x, aie::broadcast<float, CH>(epsilon));

    // y * 1/sqrt(sum + eps): y is bf16 (exact), the scale has 2 limbs
    vbf k0, k1;
    split2(rsqrt64(x), k0, k1);
    acc o = aie::mul(y0, k1);
    o = aie::mac(o, y0, k0);
    aie::store_v(y, o.template to_vector<bf16>());
    o = aie::mul(y1, k1);
    o = aie::mac(o, y1, k0);
    aie::store_v(y + 64, o.template to_vector<bf16>());
}

template <typename T_in, typename T_out, int N>
void passThrough_aie(T_out *restrict out, T_in *restrict in) {
    const int vec_factor = 64;

    aie::vector<T_out, vec_factor> Out;

    constexpr int F = N / vec_factor;

    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_RANGE(F, F)
    for (int i = 0; i < F; i++){
        aie::accum<accfloat, vec_factor> c_acc_in;
        c_acc_in.from_vector(aie::load_v<vec_factor>(in));
        aie::store_v(out, c_acc_in.template to_vector<T_out>());
        in += vec_factor;
        out += vec_factor;
    }
}

#endif
