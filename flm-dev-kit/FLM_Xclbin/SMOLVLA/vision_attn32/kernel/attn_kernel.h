// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include <aie_api/aie.hpp>
#include "common.h"
#include <math.h>
// For the AIE_* scheduling macros. Included here rather than relying on the
// translation unit's include order: attn.cc pulls this header in first, so the
// macros would otherwise be undefined while these loops are parsed. The header
// is guarded, so including it twice is free.
#include "aie_kernel_utils.h"

// Copy Q into the core-local buffer, folding in the 1/sqrt(dh) attention scale.
// The copy already round-tripped through an accumulator, so the multiply rides
// along for free. Q is staged once per round, whereas _attn_qk runs its inner
// loop 256 times per round, so this applies the scale ~16x less often than
// scaling the scores did.
//
// Exact, not approximate: scale is a power of two (0.125 == 2^-3), so it only
// decrements the exponent and leaves the bf16 mantissa untouched. Scaling Q
// ahead of the mmul is therefore bit-identical to scaling S after it, including
// under the bfp16 mmul emulation, where a uniform power-of-two shift moves the
// block's shared exponent and leaves its mantissas alone. This relies on
// scale being a power of two -- a head dim whose 1/sqrt(dh) is not (dh != 64)
// would make the fold lossy and it should stay on the scores.
template <typename T_in, typename T_out, int N>
void scale_copy_aie(T_in *restrict in0, T_out *restrict out) {
    const int vec_factor = 64;

    const int F = N / vec_factor;
    for (int i = 0; i < F; i++){
        aie::accum<accfloat, vec_factor> c_acc_in =
            aie::mul(aie::load_v<vec_factor>(in0), (T_in)scale);
        aie::store_v(out, c_acc_in.template to_vector<T_out>());
        in0 += vec_factor;
        out += vec_factor;
    }
}

// Row-wise max of a 16 x 16 score block held as the four 8 x 8 mmul C tiles
// (C00 C01 / C10 C11), folded into the running maxima at m[0..15].
// Pairwise tree: the column-pair max collapses the two tiles of a block-row
// first, then each step halves the column count while keeping the vectors at
// least 32 wide, so 16 rows reduce in 7 vector max ops with no scalar extracts.
inline void accumulate_row_max16(bf16* m,
                                 const aie::vector<bf16, 64>& C00,
                                 const aie::vector<bf16, 64>& C01,
                                 const aie::vector<bf16, 64>& C10,
                                 const aie::vector<bf16, 64>& C11) {
    // 16 columns -> 8, folding each row's right tile onto its left tile.
    // W0 holds rows 0-7, W1 holds rows 8-15, each 8 rows x 8 columns.
    aie::vector<bf16, 64> W0 = aie::max(C00, C01);
    aie::vector<bf16, 64> W1 = aie::max(C10, C11);

    // 8 columns -> 4, 8 rows at a time.
    aie::vector<bf16, 32> a01 = aie::max(aie::filter_even(W0, 4), aie::filter_odd(W0, 4));
    aie::vector<bf16, 32> a23 = aie::max(aie::filter_even(W1, 4), aie::filter_odd(W1, 4));

    // 4 columns -> 1, all 16 rows at once.
    aie::vector<bf16, 64> A = aie::concat(a01, a23);
    aie::vector<bf16, 32> b = aie::max(aie::filter_even(A, 2), aie::filter_odd(A, 2));
    aie::vector<bf16, 16> rmax = aie::max(aie::filter_even(b, 1), aie::filter_odd(b, 1));

    aie::store_v(m, aie::max(rmax, aie::load_v<16>(m)));
}

template <typename T_in, typename T_out, unsigned rowA, unsigned colA, unsigned colB, unsigned r, unsigned s, unsigned t>
void _attn_qk(T_out *__restrict pS, T_in *__restrict pQ, T_in *__restrict pK, bf16 *__restrict m) {
    using MMUL = aie::mmul<r, s, t, T_in, T_in, accauto>;

    // The fused row-max folds into m[0..15] unconditionally, so every z
    // iteration would target the same 16 maxima. Only rowA == 2 (a single
    // z step covering exactly 16 rows) is sound; re-tiling needs the fold
    // hoisted out and indexed by z.
    static_assert(rowA == 2, "fused row-max assumes a single 16-row z step");
    static_assert(r == 8 && t == 8, "fused row-max assumes 8x8 C tiles");

    for (unsigned z = 0; z < rowA; z += 2){
        T_out *__restrict pS1 = pS + (z * colB + 0) * MMUL::size_C;
        T_out *__restrict pS2 = pS + ((z + 1) * colB + 0) * MMUL::size_C;

        for (unsigned j = 0; j < colB; j += 2){
            const T_in *__restrict pQ1 = pQ + (z * colA + 0) * MMUL::size_A;
            const T_in *__restrict pQ2 = pQ + ((z + 1) * colA + 0) * MMUL::size_A;
            const T_in *__restrict pK1 = pK + (j + 0) * MMUL::size_B;
            const T_in *__restrict pK2 = pK + (j + 1) * MMUL::size_B;

            aie::vector<T_in, MMUL::size_A> Q0 = aie::load_v<MMUL::size_A>(pQ1);
            pQ1 += MMUL::size_A;
            aie::vector<T_in, MMUL::size_A> Q1 = aie::load_v<MMUL::size_A>(pQ2);
            pQ2 += MMUL::size_A;
            aie::vector<T_in, MMUL::size_B> K00 = aie::load_v<MMUL::size_B>(pK1);
            aie::vector<T_in, MMUL::size_B> K0 = aie::transpose(K00, 8, 8);
            pK1 += MMUL::size_B * colB;
            aie::vector<T_in, MMUL::size_B> K01 = aie::load_v<MMUL::size_B>(pK2);
            aie::vector<T_in, MMUL::size_B> K1 = aie::transpose(K01, 8, 8);
            pK2 += MMUL::size_B * colB;

            aie::vector<T_out, MMUL::size_C> acc_C00 = aie::zeros<T_out, MMUL::size_C>();
            aie::vector<T_out, MMUL::size_C> acc_C01 = aie::zeros<T_out, MMUL::size_C>();
            aie::vector<T_out, MMUL::size_C> acc_C10 = aie::zeros<T_out, MMUL::size_C>();
            aie::vector<T_out, MMUL::size_C> acc_C11 = aie::zeros<T_out, MMUL::size_C>();

            MMUL C00(acc_C00);
            MMUL C01(acc_C01);
            MMUL C10(acc_C10);
            MMUL C11(acc_C11);

            C00.mac(Q0, K0);
            C01.mac(Q0, K1);
            C10.mac(Q1, K0);
            C11.mac(Q1, K1);

            // The hot loop of the whole kernel: 4 MACs per iteration, run
            // colA-1 = 7 times per 16x16 block, 32 blocks per softmax pass.
            // The trip count is compile-time, so tell the scheduler rather than
            // making it guess and insert a conservative prologue guard.
            AIE_PREPARE_FOR_PIPELINING
            AIE_LOOP_RANGE(colA - 1, colA - 1)
            for (unsigned i = 1; i < colA; ++i){
                Q0 = aie::load_v<MMUL::size_A>(pQ1);
                pQ1 += MMUL::size_A;
                Q1 = aie::load_v<MMUL::size_A>(pQ2);
                pQ2 += MMUL::size_A;
                K00 = aie::load_v<MMUL::size_B>(pK1);
                K0 = aie::transpose(K00, 8, 8);
                pK1 += MMUL::size_B * colB;
                K01 = aie::load_v<MMUL::size_B>(pK2);
                K1 = aie::transpose(K01, 8, 8);
                pK2 += MMUL::size_B * colB;

                C00.mac(Q0, K0);
                C01.mac(Q0, K1);
                C10.mac(Q1, K0);
                C11.mac(Q1, K1);
            }

            // Store the raw 8x8 C tiles. The old interleave_zip reshape is gone:
            // it was permuting into a layout that _attn_fv and the row-sum path
            // already read directly (C00 C01 / C10 C11 at +0/+64/+128/+192), so
            // the shuffle and the matching reorder_s pass were pure overhead.
            //
            // No scaling here: 1/sqrt(dh) is folded into Q once at staging time
            // (see scale_copy_aie), which is exact for a power-of-two scale.
            auto mout0 = C00.template to_vector<T_out>();
            auto mout1 = C01.template to_vector<T_out>();
            auto mout2 = C10.template to_vector<T_out>();
            auto mout3 = C11.template to_vector<T_out>();

            aie::store_v(pS1, mout0);
            pS1 += MMUL::size_C;
            aie::store_v(pS1, mout1);
            pS1 += MMUL::size_C;
            aie::store_v(pS2, mout2);
            pS2 += MMUL::size_C;
            aie::store_v(pS2, mout3);
            pS2 += MMUL::size_C;

            // Fold the running row maxima out of the registers we just stored,
            // rather than reloading the whole 16 x 16 block afterwards.
            accumulate_row_max16(m, mout0, mout1, mout2, mout3);
        }
    }
}

// correct = exp(m_prev - m_new) for all 16 rows at once.
// The 16 corrections are independent elementwise ops, so they map directly onto
// the 16 lanes. The previous version looped 16 times and, per iteration,
// broadcast one scalar across a vector, clamped it, ran a full 16-lane exp2 and
// then kept lane 0 -- doing 16x the exp2 work and paying 16 scalar extracts.
// Same arithmetic per lane, so results are bit-identical.
void calculate_c(float* c, bf16* prev_m_local, bf16* new_m_local) {
    // correct = exp(m_left - max)
    aie::vector<bf16, 16> correct_vec =
        aie::sub(aie::load_v<16>(prev_m_local), aie::load_v<16>(new_m_local));
    // clamp
    constexpr bf16 min_clamp = -87.0f;
    constexpr bf16 max_clamp = 88.0f;
    aie::vector<bf16, 16> min_value = aie::broadcast<bf16, 16>(min_clamp);
    aie::vector<bf16, 16> max_value = aie::broadcast<bf16, 16>(max_clamp); // max of float
    correct_vec = aie::clamp(correct_vec, min_value, max_value);
    aie::accum<accfloat, 16> correct_acc = aie::mul(correct_vec, exp_scale);
    correct_vec = aie::exp2<bf16>(correct_acc.template to_vector<float>());

    aie::accum<accfloat, 16> c_acc;
    c_acc.from_vector(correct_vec);
    aie::store_v(c, c_acc.template to_vector<float>());
}

// Broadcast 8 consecutive row maxima into a 64-wide vector matching one 8 x 8
// mmul C tile: [m0 x8, m1 x8, ... m7 x8], so lane e carries the maximum of
// row e / 8.
inline aie::vector<bf16, 64> row_max_tile(bf16* m) {
    return aie::concat(aie::broadcast<bf16, 8>(m[0]), aie::broadcast<bf16, 8>(m[1]),
                       aie::broadcast<bf16, 8>(m[2]), aie::broadcast<bf16, 8>(m[3]),
                       aie::broadcast<bf16, 8>(m[4]), aie::broadcast<bf16, 8>(m[5]),
                       aie::broadcast<bf16, 8>(m[6]), aie::broadcast<bf16, 8>(m[7]));
}

// exp(S - M) elementwise, at the 16-lane width the rest of the kernel uses.
inline aie::vector<bf16, 64> exp_sub(const aie::vector<bf16, 64>& S,
                                     const aie::vector<bf16, 64>& M) {
    aie::vector<bf16, 64> Vec = aie::sub(S, M);
    aie::accum<accfloat, 64> Vec_acc = aie::mul(Vec, exp_scale);
    return aie::exp2<bf16>(Vec_acc.template to_vector<float>());
}

// Collapse a 64-lane accumulator holding 8 rows x 8 columns down to the 8
// per-row sums, in the low 8 lanes of each half. Mirrors the tree in
// accumulate_row_max16: halve the column count per step while keeping the
// vectors at least 32 wide, so no scalar extracts are needed.
inline aie::vector<float, 16> reduce_rows8x2(const aie::accum<accfloat, 64>& A0,
                                             const aie::accum<accfloat, 64>& A1) {
    aie::vector<float, 64> v0 = A0.template to_vector<float>();
    aie::vector<float, 64> v1 = A1.template to_vector<float>();

    // 8 columns -> 4, 8 rows at a time.
    aie::vector<float, 32> a0 = aie::add(aie::filter_even(v0, 4), aie::filter_odd(v0, 4));
    aie::vector<float, 32> a1 = aie::add(aie::filter_even(v1, 4), aie::filter_odd(v1, 4));

    // 4 columns -> 1, all 16 rows at once.
    aie::vector<float, 64> A = aie::concat(a0, a1);
    aie::vector<float, 32> b = aie::add(aie::filter_even(A, 2), aie::filter_odd(A, 2));
    return aie::add(aie::filter_even(b, 1), aie::filter_odd(b, 1));
}

// Softmax numerator in place, returning the 16 row sums of the result.
//
// _attn_qk now stores raw 8 x 8 mmul C tiles, which is already the layout
// _attn_fv consumes, so this is a pure exp pass with no reshuffling. The row
// sums are accumulated here while the exponentials are still in registers,
// which removes calculate_l_and_y's separate ones-matmul pass over s entirely
// (a full 16 KB re-read per call, plus 8x redundant MACs).
//
// Summation order differs from the old ones-matmul, so l can differ in the last
// fp32 ulp; the accumulation is still fp32 throughout, and pairwise tile adds
// are if anything better conditioned than the mmul's sequential chain.
//
// The two row-max broadcasts are loop-invariant: tiles 0 and 1 hold rows 0-7,
// tiles 2 and 3 hold rows 8-15, for every block.
aie::vector<float, 16> softmax(bf16*__restrict pS, bf16* new_m_local) {
    const int N = lk_mt / lk;

    const aie::vector<bf16, 64> M0 = row_max_tile(new_m_local + 0);
    const aie::vector<bf16, 64> M1 = row_max_tile(new_m_local + 8);

    // A0 accumulates rows 0-7, A1 rows 8-15, both still 8 columns wide.
    aie::accum<accfloat, 64> A0 = aie::zeros<accfloat, 64>();
    aie::accum<accfloat, 64> A1 = aie::zeros<accfloat, 64>();

    // Longest trip count in the kernel (32). Iterations are independent apart
    // from the two accumulators, whose adds are the only loop-carried
    // dependence, so there is plenty for the scheduler to overlap.
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_RANGE(lk_mt / lk, lk_mt / lk)
    for (int i = 0; i < N; i++){
        bf16 *pS1 = pS + i * 256;

        aie::vector<bf16, 64> P0 = exp_sub(aie::load_v<64>(pS1),       M0);
        aie::vector<bf16, 64> P1 = exp_sub(aie::load_v<64>(pS1 + 64),  M0);
        aie::vector<bf16, 64> P2 = exp_sub(aie::load_v<64>(pS1 + 128), M1);
        aie::vector<bf16, 64> P3 = exp_sub(aie::load_v<64>(pS1 + 192), M1);

        aie::store_v(pS1,       P0);
        aie::store_v(pS1 + 64,  P1);
        aie::store_v(pS1 + 128, P2);
        aie::store_v(pS1 + 192, P3);

        // Rows 0-7 live in tiles 0 and 1, rows 8-15 in tiles 2 and 3.
        A0 = aie::add(aie::add(A0, P0), P1);
        A1 = aie::add(aie::add(A1, P2), P3);
    }

    return reduce_rows8x2(A0, A1);
}

// The row sums of P now come in from softmax(), which accumulates them while it
// already has each exp tile in registers. The old ones-matmul pass over pS is
// gone: it re-read the whole 16 KB s buffer and computed every row sum eight
// times over (one per output column of the mmul) only to discard seven of them.
//
// first_chunk lets the caller skip the y correction on the opening chunk of a
// round, where it provably does nothing (see below).
void calculate_l_and_y(float* y, float* l, float* c,
                       const aie::vector<float, 16>& rowsum,
                       bool first_chunk) {
    // calculate l
    using MMUL = aie::mmul<8, 8, 8, bf16, bf16, accauto>;

    aie::accum<accfloat, 16> sum_acc0;
    sum_acc0.from_vector(rowsum);

    aie::vector<float, 16> c0_float32 = aie::load_v<16>(c);
    aie::vector<float, 16> l0_float32 = aie::load_v<16>(l);
    aie::accum<accfloat, 16> l_out0;
    l_out0 = aie::mac(sum_acc0, l0_float32, c0_float32);
    aie::store_v(l, l_out0.template to_vector<float>());

    // calculate y
    //
    // On the first chunk of a round y is still all zeros -- attn.cc clears it
    // with zero_vectorized right before the j loop, and nothing has written to
    // it yet, since _attn_fv for this chunk runs after this call. Rescaling is
    // therefore a read-multiply-write over 4 KB that stores back the zeros it
    // just loaded. Note the guard is NOT "c == 1": on the first chunk prev_m is
    // -inf, so calculate_c clamps to exp(-87) ~ 1.6e-38 rather than 1. The skip
    // is exact regardless, because 0 * c == 0 for every finite c, and c is
    // clamped finite by construction. With chunks == 2 this halves the rescale
    // traffic; l still accumulates normally above, where c does matter.
    if (first_chunk) {
        return;
    }

    for (int i = 0; i < 2; i++){
        aie::vector<float, 8> c0 = aie::broadcast<float, 8>(c[i * 8]);
        aie::vector<float, 8> c1 = aie::broadcast<float, 8>(c[i * 8 + 1]);
        aie::vector<float, 8> c2 = aie::broadcast<float, 8>(c[i * 8 + 2]);
        aie::vector<float, 8> c3 = aie::broadcast<float, 8>(c[i * 8 + 3]);
        aie::vector<float, 8> c4 = aie::broadcast<float, 8>(c[i * 8 + 4]);
        aie::vector<float, 8> c5 = aie::broadcast<float, 8>(c[i * 8 + 5]);
        aie::vector<float, 8> c6 = aie::broadcast<float, 8>(c[i * 8 + 6]);
        aie::vector<float, 8> c7 = aie::broadcast<float, 8>(c[i * 8 + 7]);
        aie::vector<float, 64> CORRECT = aie::concat(c0, c1, c2, c3, c4, c5, c6, c7);

        float* pY = y + i * 8 * dh;
        AIE_PREPARE_FOR_PIPELINING
        AIE_LOOP_RANGE(dh / 8, dh / 8)
        for (unsigned j = 0; j < dh / 8; j += 1){
            aie::vector<float, MMUL::size_C> Y = aie::load_v<MMUL::size_C>(pY);
            aie::accum<accfloat, MMUL::size_C> ACC_Y = aie::mul(CORRECT, Y);
            aie::store_v(pY, ACC_Y.template to_vector<float>());
            pY += MMUL::size_C;
        }
    }
}

template <typename T_in, typename T_out, unsigned r, unsigned s, unsigned t>
void _attn_fv(T_out *__restrict pY, T_in *__restrict pS, T_in *__restrict pV) {
    using MMUL = aie::mmul<r, s, t, T_in, T_in, accauto>;

    for (int it = 0; it < 2; it += 2){
        T_out *__restrict pY1 = pY + (it + 0) * 8 * dh;
        T_out *__restrict pY2 = pY + (it + 1) * 8 * dh;

        // dh/8 = 8 columns stepped 2 at a time -> 4 trips, each 8 MACs.
        AIE_PREPARE_FOR_PIPELINING
        AIE_LOOP_RANGE(dh / 16, dh / 16)
        for (unsigned j = 0; j < dh / 8; j += 2){
            T_in *__restrict pV1 = pV + (j + 0) * 8 * lk;
            T_in *__restrict pV2 = pV + (j + 1) * 8 * lk;
            T_in *__restrict pS1 = pS + (it + 0) * 8 * lk;
            T_in *__restrict pS2 = pS + (it + 1) * 8 * lk;

            aie::vector<T_in, MMUL::size_A> S0 = aie::load_v<MMUL::size_A>(pS1);
            pS1 += MMUL::size_A;
            aie::vector<T_in, MMUL::size_A> S1 = aie::load_v<MMUL::size_A>(pS2);
            pS2 += MMUL::size_A;
            aie::vector<T_in, MMUL::size_B> V0 = aie::load_v<MMUL::size_B>(pV1);
            pV1 += MMUL::size_B;
            aie::vector<T_in, MMUL::size_B> V1 = aie::load_v<MMUL::size_B>(pV2);
            pV2 += MMUL::size_B;

            aie::vector<T_out, MMUL::size_C> acc_Y00 = aie::load_v<MMUL::size_C>(pY1);
            aie::vector<T_out, MMUL::size_C> acc_Y01 = aie::load_v<MMUL::size_C>(pY1 + MMUL::size_C);
            aie::vector<T_out, MMUL::size_C> acc_Y10 = aie::load_v<MMUL::size_C>(pY2);
            aie::vector<T_out, MMUL::size_C> acc_Y11 = aie::load_v<MMUL::size_C>(pY2 + MMUL::size_C);

            MMUL Y00(acc_Y00);
            MMUL Y01(acc_Y01);
            MMUL Y10(acc_Y10);
            MMUL Y11(acc_Y11);

            Y00.mac(S0, V0);
            Y01.mac(S0, V1);
            Y10.mac(S1, V0);
            Y11.mac(S1, V1);

            // lk/8 = 2, so this runs exactly once. Unrolling it away lets the
            // enclosing j loop pipeline across a straight-line body instead of
            // scheduling around a nested loop boundary.
            AIE_LOOP_UNROLL_FULL
            for (int i = 1; i < lk / 8; i++){
                S0 = aie::load_v<MMUL::size_A>(pS1);
                pS1 += MMUL::size_A;
                S1 = aie::load_v<MMUL::size_A>(pS2);
                pS2 += MMUL::size_A;
                V0 = aie::load_v<MMUL::size_B>(pV1);
                pV1 += MMUL::size_B;
                V1 = aie::load_v<MMUL::size_B>(pV2);
                pV2 += MMUL::size_B;

                Y00.mac(S0, V0);
                Y01.mac(S0, V1);
                Y10.mac(S1, V0);
                Y11.mac(S1, V1);
            }

            aie::store_v(pY1, Y00.template to_vector<T_out>());
            pY1 += MMUL::size_C;
            aie::store_v(pY1, Y01.template to_vector<T_out>());
            pY1 += MMUL::size_C;
            aie::store_v(pY2, Y10.template to_vector<T_out>());
            pY2 += MMUL::size_C;
            aie::store_v(pY2, Y11.template to_vector<T_out>());
            pY2 += MMUL::size_C;
        }
    }
}

void calculate_l(float *l) {
    for (int i = 0; i < lq / 32; i++){
        aie::vector<float, 32> l_vec = aie::load_v<32>(l);
        l_vec = aie::inv(l_vec);
        aie::store_v(l, l_vec);
        l += 32;
    }
}

void scale_div_aie(bf16* o, float *l, float *y) {
    constexpr int vec_factor = 64;

    aie::vector<float, 8> L0 = aie::broadcast<float, 8>(l[0]);
    aie::vector<float, 8> L1 = aie::broadcast<float, 8>(l[1]);
    aie::vector<float, 8> L2 = aie::broadcast<float, 8>(l[2]);
    aie::vector<float, 8> L3 = aie::broadcast<float, 8>(l[3]);
    aie::vector<float, 8> L4 = aie::broadcast<float, 8>(l[4]);
    aie::vector<float, 8> L5 = aie::broadcast<float, 8>(l[5]);
    aie::vector<float, 8> L6 = aie::broadcast<float, 8>(l[6]);
    aie::vector<float, 8> L7 = aie::broadcast<float, 8>(l[7]);

    auto LL00 = aie::concat(L0, L1, L2, L3, L4, L5, L6, L7);

    aie::vector<float, vec_factor> Y00 = aie::load_v<vec_factor>(y);
    aie::accum<accfloat, vec_factor> Y00_acc;
    Y00_acc.from_vector(Y00);
    aie::accum<accfloat, vec_factor> AL00 = aie::mul(Y00_acc.template to_vector<float>(), LL00);
    aie::store_v(o, AL00.template to_vector<bf16>());
}
