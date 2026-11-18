// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __BF16_LIMB_MATH_HPP__
#define __BF16_LIMB_MATH_HPP__

// ---------------------------------------------------------------------------
// fp32 x bf16 products on the native bf16 MMUL datapath.
//
// For a one-off fp32 x fp32 product, use aie::mul / aie::mac directly. Those
// lower to the vendor's emulation (mul_elem_N_accuracy_safe in
// $AIETOOLS/data/aie2p/lib/me_vmult_float_emulated.h, via
// detail/aie2p/mul_acc32_fp.hpp), which splits BOTH operands into 3 bf16 limbs
// and issues all 9 cross products for an fp32-exact result. The cheaper
// mul_elem_N_accuracy_fast (6 MACs) and _low (3 MACs) are always defined too,
// and are callable by name per call site -- the AIE2_FP32_EMULATION_ACCURACY_*
// macros only pick which one aie::mul binds to, so you do not need a
// translation-unit-wide flag to reach them.
//
// Two things the vendor cannot express, which is what this header is for:
//
// 1. Asymmetric fp32 x bf16: one operand is fp32, the other is already exactly
//    representable in bf16 (it came from a bf16 buffer). Only the fp32 side
//    needs splitting, so the product costs one MAC per limb -- 2 or 3, not 9.
//    The vendor has no asymmetric entry point at all; the cheapest thing it can
//    do here is _low's 3 full products.
//
// 2. Operands you already hold as limbs. split_f32_3() once, then feed the
//    mac_limbs3_* forms -- useful when the same limbs serve several products,
//    when you also want l0 as the plain bf16 rounding of x, or when the split
//    is invariant in a way the compiler cannot prove (behind a pointer it
//    cannot alias-analyse, or a call it did not inline).
//
// What is NOT a reason to use this header: saving a redundant split. The
// scheduler already hoists a loop-invariant split out of a loop body, including
// one buried inside the vendor's emulation -- measured, an aie::mul with one
// invariant operand leaves a single split (3 VCONV + 2 VMSC) in the body, not
// two. The win over the vendor default is the product set and its reduction
// shape, not the split.
//
// Accuracy, measured against a float64 reference over 4e5 operand pairs drawn
// across a ~2^12 dynamic range, worst-case relative error:
//
//     op                       MACs   max rel err   ~mantissa bits
//     plain bf16 x bf16           1      3.9e-03           8
//     mac_f32_bf16_low            2      7.6e-06          17
//     mac_f32_bf16_safe           3      6.0e-08          24   (fp32-exact)
//     mac_limbs3_tri (f32xf32)    6      6.0e-08          24   (fp32-exact)
//     aie::mac on 2x fp32         9      6.0e-08          24   (vendor, f32xf32)
//
// fp32 carries 24 mantissa bits, so the 3-limb form is exact for practical
// purposes and matches the vendor's 9-MAC safe tier at a third of the cost.
// The 2-limb form drops the last limb (~2^-24 of the operand) for one fewer
// MAC; it is the right pick when the product feeds a bf16 output anyway, and
// the wrong pick for anything that compounds over a sequence, like a decay gate.
//
// The 3 products the triangular fp32 x fp32 form drops (l1*l2, l2*l1, l2*l2)
// are all <= 2^-24 of the result, i.e. below where an fp32 mantissa can hold
// them, so 6 MACs is not an approximation of the 9-MAC form -- it is the same
// answer with the provably-zero terms removed.
//
// Note on rounding: to_vector<bfloat16>() rounds using the core's current
// rounding mode. The numbers above assume the default round-to-nearest-even.
// ---------------------------------------------------------------------------

#include <aie_api/aie.hpp>
#include <utility>

namespace bf16_limb {

// A bf16 vector of ones, used to subtract a limb back out inside a single msc
// rather than converting to fp32 and doing a separate aie::sub.
template <unsigned N>
inline aie::vector<bfloat16, N> ones()
{
    return aie::broadcast<bfloat16, N>((bfloat16)1.0f);
}

// ---------------------------------------------------------------------------
// Splitting
// ---------------------------------------------------------------------------

// Split x into 2 bf16 limbs with x ~= l0 + l1, carrying ~16 of fp32's 24
// mantissa bits. Each residual is exact in fp32: l0 is within a factor of two
// of x, so x - l0 cancels without rounding.
template <unsigned N>
inline std::pair<aie::vector<bfloat16, N>, aie::vector<bfloat16, N>>
split_f32(aie::vector<float, N> x)
{
    const aie::vector<bfloat16, N> one = ones<N>();

    aie::accum<accfloat, N> r0;
    r0.from_vector(x);
    aie::vector<bfloat16, N> l0 = r0.template to_vector<bfloat16>();

    aie::accum<accfloat, N> r1 = aie::msc(r0, l0, one);   // r1 = x - l0
    aie::vector<bfloat16, N> l1 = r1.template to_vector<bfloat16>();

    return {l0, l1};
}

// Split x into 3 bf16 limbs with x ~= l0 + l1 + l2. 3 x 8 = 24 bits covers the
// whole fp32 mantissa, so the triple is exact up to the exponent range.
template <unsigned N>
struct limbs3 {
    aie::vector<bfloat16, N> l0, l1, l2;
};

template <unsigned N>
inline limbs3<N> split_f32_3(aie::vector<float, N> x)
{
    const aie::vector<bfloat16, N> one = ones<N>();
    limbs3<N> out;

    aie::accum<accfloat, N> r0;
    r0.from_vector(x);
    out.l0 = r0.template to_vector<bfloat16>();

    aie::accum<accfloat, N> r1 = aie::msc(r0, out.l0, one);   // x - l0
    out.l1 = r1.template to_vector<bfloat16>();

    aie::accum<accfloat, N> r2 = aie::msc(r1, out.l1, one);   // x - l0 - l1
    out.l2 = r2.template to_vector<bfloat16>();

    return out;
}

// ---------------------------------------------------------------------------
// acc += a * b, where a is fp32 and b is already exact in bf16
//
// Limbs are accumulated smallest-first, matching the vendor's ordering in
// mac_elem_N_accuracy_safe, so the small terms combine before meeting the
// large one instead of being rounded off against it.
// ---------------------------------------------------------------------------

// 2 MACs, ~17 bits.
template <unsigned N>
inline aie::accum<accfloat, N> mac_f32_bf16_low(
    aie::accum<accfloat, N> acc,
    aie::vector<float, N>   a,
    aie::vector<bfloat16, N> b)
{
    auto [l0, l1] = split_f32(a);
    acc = aie::mac(acc, b, l1);
    acc = aie::mac(acc, b, l0);
    return acc;
}

// 3 MACs, ~24 bits -- fp32-exact, same accuracy as the vendor's 9-MAC path.
template <unsigned N>
inline aie::accum<accfloat, N> mac_f32_bf16_safe(
    aie::accum<accfloat, N> acc,
    aie::vector<float, N>   a,
    aie::vector<bfloat16, N> b)
{
    limbs3<N> l = split_f32_3(a);
    acc = aie::mac(acc, b, l.l2);
    acc = aie::mac(acc, b, l.l1);
    acc = aie::mac(acc, b, l.l0);
    return acc;
}

// ---------------------------------------------------------------------------
// Pre-split operands
//
// Take the limbs directly when you already have them -- reused across several
// products, wanted as a bf16 value in their own right (l0 IS the RNE bf16 of x),
// or split outside a loop the compiler cannot prove invariance across.
// ---------------------------------------------------------------------------

template <unsigned N>
inline aie::accum<accfloat, N> mac_limbs_bf16(
    aie::accum<accfloat, N> acc,
    aie::vector<bfloat16, N> l0,
    aie::vector<bfloat16, N> l1,
    aie::vector<bfloat16, N> b)
{
    acc = aie::mac(acc, b, l1);
    acc = aie::mac(acc, b, l0);
    return acc;
}

template <unsigned N>
inline aie::accum<accfloat, N> mac_limbs_bf16(
    aie::accum<accfloat, N> acc,
    limbs3<N>               l,
    aie::vector<bfloat16, N> b)
{
    acc = aie::mac(acc, b, l.l2);
    acc = aie::mac(acc, b, l.l1);
    acc = aie::mac(acc, b, l.l0);
    return acc;
}

// acc += a * b, both fp32, both pre-split into 3 limbs.
//
// Only the 6 products with i+j <= 2 are issued. Limb i carries magnitude
// ~2^-8i of its operand, so a dropped term l_i*l_j is ~2^-8(i+j) <= 2^-24 of
// the result -- at or below the fp32 mantissa floor. That is why this matches
// the vendor's 9-product safe tier bit-for-bit in practice rather than
// approximating it.
//
// Against the vendor default this is 6 MACs through an accumulator chain in
// place of 9 VMULs into an 8-deep serial VADD tree -- 26 bundles vs 71 at
// N=32. mul_elem_N_accuracy_fast issues the same 6 products and is callable by
// name; reach for this form when you want the limbs themselves, or when the
// other operand is bf16-exact and belongs in mac_limbs_bf16 instead.
template <unsigned N>
inline aie::accum<accfloat, N> mac_limbs3_tri(
    aie::accum<accfloat, N> acc,
    const limbs3<N>&        a,
    const limbs3<N>&        b)
{
    acc = aie::mac(acc, a.l2, b.l0);
    acc = aie::mac(acc, a.l1, b.l1);
    acc = aie::mac(acc, a.l0, b.l2);
    acc = aie::mac(acc, a.l1, b.l0);
    acc = aie::mac(acc, a.l0, b.l1);
    acc = aie::mac(acc, a.l0, b.l0);
    return acc;
}

// Same, with the second operand still packed as fp32. Splits b in-loop.
template <unsigned N>
inline aie::accum<accfloat, N> mac_limbs3_f32_tri(
    aie::accum<accfloat, N> acc,
    const limbs3<N>&        a,
    aie::vector<float, N>   b)
{
    return mac_limbs3_tri(acc, a, split_f32_3(b));
}

// acc -= a * b, the subtracting counterparts of the two above.
template <unsigned N>
inline aie::accum<accfloat, N> msc_limbs3_tri(
    aie::accum<accfloat, N> acc,
    const limbs3<N>&        a,
    const limbs3<N>&        b)
{
    acc = aie::msc(acc, a.l2, b.l0);
    acc = aie::msc(acc, a.l1, b.l1);
    acc = aie::msc(acc, a.l0, b.l2);
    acc = aie::msc(acc, a.l1, b.l0);
    acc = aie::msc(acc, a.l0, b.l1);
    acc = aie::msc(acc, a.l0, b.l0);
    return acc;
}

template <unsigned N>
inline aie::accum<accfloat, N> msc_limbs3_f32_tri(
    aie::accum<accfloat, N> acc,
    const limbs3<N>&        a,
    aie::vector<float, N>   b)
{
    return msc_limbs3_tri(acc, a, split_f32_3(b));
}

// ---------------------------------------------------------------------------
// Split-accumulator forms, for reduction loops
//
// In a loop that accumulates over many terms, chaining both limb MACs into one
// accumulator makes the loop-carried dependency two MACs deep and sets the
// initiation interval accordingly. Giving each limb its own accumulator keeps
// each chain one MAC deep; the caller folds them together once, after the loop,
// with reduce().
// ---------------------------------------------------------------------------

// 2 MACs into 2 independent accumulators.
template <unsigned N>
inline void mac_f32_bf16_low(
    aie::accum<accfloat, N>& acc0,
    aie::accum<accfloat, N>& acc1,
    aie::vector<float, N>    a,
    aie::vector<bfloat16, N> b)
{
    auto [l0, l1] = split_f32(a);
    acc0 = aie::mac(acc0, b, l0);
    acc1 = aie::mac(acc1, b, l1);
}

// 3 MACs into 3 independent accumulators.
template <unsigned N>
inline void mac_f32_bf16_safe(
    aie::accum<accfloat, N>& acc0,
    aie::accum<accfloat, N>& acc1,
    aie::accum<accfloat, N>& acc2,
    aie::vector<float, N>    a,
    aie::vector<bfloat16, N> b)
{
    limbs3<N> l = split_f32_3(a);
    acc0 = aie::mac(acc0, b, l.l0);
    acc1 = aie::mac(acc1, b, l.l1);
    acc2 = aie::mac(acc2, b, l.l2);
}

// Fold split accumulators back together, smallest-first.
template <unsigned N>
inline aie::accum<accfloat, N> reduce(
    aie::accum<accfloat, N> acc0,
    aie::accum<accfloat, N> acc1)
{
    aie::accum<accfloat, N> out;
    out.from_vector(aie::add(acc1.template to_vector<float>(),
                             acc0.template to_vector<float>()));
    return out;
}

template <unsigned N>
inline aie::accum<accfloat, N> reduce(
    aie::accum<accfloat, N> acc0,
    aie::accum<accfloat, N> acc1,
    aie::accum<accfloat, N> acc2)
{
    return reduce(acc0, reduce(acc1, acc2));
}

} // namespace bf16_limb

#endif // __BF16_LIMB_MATH_HPP__
