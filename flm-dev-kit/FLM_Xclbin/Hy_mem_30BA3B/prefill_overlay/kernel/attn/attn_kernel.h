// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef ATTN_KERNEL_H
#define ATTN_KERNEL_H
#include <aie_api/aie.hpp>
#include "common.h"
#include <math.h>

// new_m_local[i0 .. i0+3], each broadcast across a row of lk, as one 64-lane
// vector -- the row-wise subtrahend for a 4-row group of S.
static inline aie::vector<bf16, 64> _row_broadcast(bf16*__restrict new_m_local, int i0) {
    return aie::concat(aie::broadcast<bf16, lk>(*(new_m_local + i0 + 0)),
                       aie::broadcast<bf16, lk>(*(new_m_local + i0 + 1)),
                       aie::broadcast<bf16, lk>(*(new_m_local + i0 + 2)),
                       aie::broadcast<bf16, lk>(*(new_m_local + i0 + 3)));
}

// f = exp(s - max), then the 16x16 -> 8x8x4 shuffle that _attn_fv and
// calculate_l expect, in one pass.
//
// The row max used to be a scalar broadcast per row, which pinned the whole
// chain to 16 lanes and left it with no independent work to hide the load /
// sub / exp2 latencies.  Broadcasting it into a 4-row vector once instead lets
// four exp2 chains run in parallel per iteration, and the reorder's operands
// are already in registers by then, so its loads and stores disappear.
void apply_softmax_reorder(
    bf16*__restrict pS,
    bf16*__restrict new_m_local
){
    for (int h = 0; h < 2; h++){
        // rows 8h..8h+3 feed A, rows 8h+4..8h+7 feed B
        const aie::vector<bf16, 64> MB_A = _row_broadcast(new_m_local, 8 * h + 0);
        const aie::vector<bf16, 64> MB_B = _row_broadcast(new_m_local, 8 * h + 4);

        bf16 *pS1 = pS + h * 128;
        for (int b = 0; b < 8; b++)
        chess_prepare_for_pipelining chess_loop_range(8, 8) {
            aie::vector<bf16, 64> A = aie::sub(aie::load_v<64>(pS1), MB_A);
            aie::vector<bf16, 64> B = aie::sub(aie::load_v<64>(pS1 + 64), MB_B);
            aie::accum<accfloat, 64> A_acc = aie::mul(A, exp_scale);
            aie::accum<accfloat, 64> B_acc = aie::mul(B, exp_scale);
            A = aie::exp2<bf16>(A_acc.template to_vector<float>());
            B = aie::exp2<bf16>(B_acc.template to_vector<float>());

            aie::store_v(pS1,      aie::concat(aie::filter_even(A, 8), aie::filter_even(B, 8)));
            aie::store_v(pS1 + 64, aie::concat(aie::filter_odd(A, 8),  aie::filter_odd(B, 8)));
            pS1 += 256;
        }
    }
}

// S is [query][key] row-major, and _attn_qk emits it four rows at a time.  Both
// the causal bound and the sliding window depend only on (key - query), so
// tabulate that offset for one 4-row x lk group: d_offset[p] = p % lk - p / lk.
// Row group g just shifts the row index by 4*g, so every group reuses this one
// table against thresholds moved by 4*g.
alignas(aie::vector_decl_align) int16 d_offset[64] = {
     0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
    -1,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14,
    -2, -1,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13,
    -3, -2, -1,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12,
};

// Mask one 4-row group of S and fold it into the running elementwise max, with
// S still in registers from the mmul.  Keeps S[i][j] iff lo < (j - i) <= hi.
template <typename T_out>
static inline void _qk_mask_max(
    T_out *__restrict pS,
    T_out *__restrict pM,
    const aie::vector<int16, 64> &D,
    aie::vector<T_out, 64> S,
    int hi,
    int lo
){
    aie::mask<64> keep = aie::le(D, (int16)hi) & aie::gt(D, (int16)lo);
    S = aie::select((T_out)(-0x1.FEp127f), S, keep);
    aie::vector<T_out, 64> M = aie::load_v<64>(pM);
    aie::store_v(pM, aie::max(M, S));
    aie::store_v(pS, S);
}

// One 4-row x lk group of M reduced to 4 row maxima, held as 4 x 2 lanes so
// every intermediate stays at a native vector width.
static inline aie::vector<bf16, 8> _row_max_pairs(const bf16*__restrict pM) {
    aie::vector<bf16, 64> v0 = aie::load_v<64>(pM);
    aie::vector<bf16, 32> v1 = aie::max(aie::filter_even(v0, 8), aie::filter_odd(v0, 8));
    aie::vector<bf16, 16> v2 = aie::max(aie::filter_even(v1, 4), aie::filter_odd(v1, 4));
    return aie::max(aie::filter_even(v2, 2), aie::filter_odd(v2, 2));
}

// Row-wise max of the whole lq x lk block.  Four independent reduction trees
// instead of lq serial aie::reduce_max calls.
static inline aie::vector<bf16, lq> _block_row_max(const bf16*__restrict pM) {
    aie::vector<bf16, 32> p = aie::concat(_row_max_pairs(pM),       _row_max_pairs(pM + 64),
                                          _row_max_pairs(pM + 128), _row_max_pairs(pM + 192));
    return aie::max(aie::filter_even(p, 1), aie::filter_odd(p, 1));
}

void calculate_c(bf16* c, bf16* prev_m_local, bf16* new_m_local) {
    // correct = exp(m_left - max), all lq rows in one pass
    aie::vector<bf16, lq> prev_m_vec = aie::load_v<lq>(prev_m_local);
    aie::vector<bf16, lq> new_m_vec = aie::load_v<lq>(new_m_local);
    aie::vector<bf16, lq> inner_correct = aie::sub(prev_m_vec, new_m_vec);
    aie::accum<accfloat, lq> correct_acc = aie::mul(inner_correct, exp_scale);
    aie::store_v(c, aie::exp2<bf16>(correct_acc.template to_vector<float>()));
}

void calculate_l(bf16* l, bf16*__restrict c, bf16*__restrict pS) {
    // calculate l
    using MMUL = aie::mmul<8, 8, 8, bf16, bf16, accauto>;

    // calculte l = correct * l_left + sum(f)
    aie::vector<bf16, 64> Ones = aie::broadcast<bf16, 64>(1.0f);
    aie::vector<bf16, MMUL::size_C> acc_C0 = aie::zeros<bf16, MMUL::size_C>(); 
    aie::vector<bf16, MMUL::size_C> acc_C1 = aie::zeros<bf16, MMUL::size_C>(); 
    MMUL C0(acc_C0);
    MMUL C1(acc_C1);

    for (int b = 0; b < 8; b++){
        bf16 *pS1 = pS + b * 256;
        aie::vector<bf16, 64> S0 = aie::load_v<64>(pS1);
        aie::vector<bf16, 64> S1 = aie::load_v<64>(pS1 + 64);
        aie::vector<bf16, 64> S2 = aie::load_v<64>(pS1 + 128);
        aie::vector<bf16, 64> S3 = aie::load_v<64>(pS1 + 192);

        C0.mac(S0, Ones);
        C1.mac(S2, Ones);

        C0.mac(S1, Ones);
        C1.mac(S3, Ones);
    }

    auto sum0 = aie::filter_even(C0.template to_vector<float>(), 4);
    auto sum00 = aie::filter_even(sum0, 2);
    auto sum01 = aie::filter_even(sum00, 1);
    auto sum1 = aie::filter_even(C1.template to_vector<float>(), 4);
    auto sum10 = aie::filter_even(sum1, 2);
    auto sum11 = aie::filter_even(sum10, 1);
    auto sum2 = aie::concat(sum01, sum11);
    aie::accum<accfloat, 16> sum;
    sum.from_vector(sum2);

    aie::vector<bf16, 16> c_float32 = aie::load_v<16>(c);
    aie::vector<bf16, 16> l_float32 = aie::load_v<16>(l);

    aie::vector<bf16, 16> l_out;
    l_out = aie::mac(sum, l_float32, c_float32);
    aie::store_v(l, l_out);
}

void calculate_y(float* y, bf16* c) {
    // calculate y
    for (int i = 0; i < lq / 8; i++) {
        // whole 8-row group unscaled -> nothing to do
        if (aie::equal(aie::load_v<8>(c + i * 8), (bf16)1.0f))
            continue;

        aie::vector<bf16, 8> c0 = aie::broadcast<bf16, 8>((bf16)c[i * 8]);
        aie::vector<bf16, 8> c1 = aie::broadcast<bf16, 8>((bf16)c[i * 8 + 1]);
        aie::vector<bf16, 8> c2 = aie::broadcast<bf16, 8>((bf16)c[i * 8 + 2]);
        aie::vector<bf16, 8> c3 = aie::broadcast<bf16, 8>((bf16)c[i * 8 + 3]);
        aie::vector<bf16, 8> c4 = aie::broadcast<bf16, 8>((bf16)c[i * 8 + 4]);
        aie::vector<bf16, 8> c5 = aie::broadcast<bf16, 8>((bf16)c[i * 8 + 5]);
        aie::vector<bf16, 8> c6 = aie::broadcast<bf16, 8>((bf16)c[i * 8 + 6]);
        aie::vector<bf16, 8> c7 = aie::broadcast<bf16, 8>((bf16)c[i * 8 + 7]);
        aie::vector<bf16, 64> CORRECT = aie::concat(c0, c1, c2, c3, c4, c5, c6, c7);

        // two 8x8 blocks per iteration: the hi/lo split is a long serial chain,
        // so a second independent copy fills the latency slots
        float* pY = y + i * 8 * dh;
        for (unsigned j = 0; j < dh / 16; j += 1)
        chess_prepare_for_pipelining chess_loop_range(8, 8) {
            aie::accum<accfloat, 64> A_acc, B_acc;
            A_acc.from_vector(aie::load_v<64>(pY));
            B_acc.from_vector(aie::load_v<64>(pY + 64));
            aie::vector<bf16, 64> A_hi = A_acc.template to_vector<bf16>();
            aie::vector<bf16, 64> B_hi = B_acc.template to_vector<bf16>();
            aie::vector<bf16, 64> A_lo = aie::sub(A_acc, A_hi).template to_vector<bf16>();
            aie::vector<bf16, 64> B_lo = aie::sub(B_acc, B_hi).template to_vector<bf16>();
            aie::accum<accfloat, 64> A_out = aie::mul(A_hi, CORRECT);
            aie::accum<accfloat, 64> B_out = aie::mul(B_hi, CORRECT);
            A_out = aie::mac(A_out, A_lo, CORRECT);
            B_out = aie::mac(B_out, B_lo, CORRECT);
            aie::store_v(pY,      A_out.template to_vector<float>());
            aie::store_v(pY + 64, B_out.template to_vector<float>());
            pY += 128;
        }
    }
}

template <typename T_in, typename T_out, unsigned rowA, unsigned colA, unsigned colB, unsigned r, unsigned s, unsigned t>
void _attn_fv(T_out *__restrict pY, T_in *__restrict pS, T_in *__restrict pV) {
    using MMUL = aie::mmul<r, s, t, T_in, T_in, accauto>;

    // load S and reorder s from 16 x 16 to 8 x 8 x 4
    aie::vector<T_in, 64> S0 = aie::load_v<64>(pS);
    aie::vector<T_in, 64> S1 = aie::load_v<64>(pS + 64);
    aie::vector<T_in, 64> S2 = aie::load_v<64>(pS + 128);
    aie::vector<T_in, 64> S3 = aie::load_v<64>(pS + 192);

    T_out *__restrict pY1 = pY;
    T_out *__restrict pY2 = pY + colB * MMUL::size_C;

    for (unsigned j = 0; j < colB; j += 2)
    {
        T_in *__restrict pV1 = pV + j * MMUL::size_B * colA;
        T_in *__restrict pV2 = pV + (j + 1) * MMUL::size_B * colA;

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
        Y10.mac(S2, V0);
        Y11.mac(S2, V1);

        V0 = aie::load_v<MMUL::size_B>(pV1);
        pV1 += MMUL::size_B;
        V1 = aie::load_v<MMUL::size_B>(pV2);
        pV2 += MMUL::size_B;

        Y00.mac(S1, V0);
        Y01.mac(S1, V1);
        Y10.mac(S3, V0);
        Y11.mac(S3, V1);

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

template <typename T_in, typename T_out, unsigned rowA, unsigned colA, unsigned colB, unsigned r, unsigned s, unsigned t>
void _attn_qk(T_out *__restrict pS, T_in *__restrict pQ, T_in *__restrict pK,
              T_out *__restrict pM, int l_base, int q_base, int k_base) {
    using MMUL = aie::mmul<r, s, t, T_in, T_in, accauto>;

    // keep S[i][j] iff lo < (j - i) <= hi; (j - i) spans [-15, 15] over the tile
    int hi = q_base - k_base;
    int lo = l_base - k_base;
    if (hi < -15 || lo >= 15) {
        // no key in this block is visible to any query: skip the mmul entirely.
        // -inf (not zero) so the later exp2(s - m) still turns these into zeros.
        const aie::vector<T_out, 64> neg_inf = aie::broadcast<T_out, 64>((T_out)(-0x1.FEp127f));
        for (unsigned g = 0; g < rowA * colB; g++)
            aie::store_v(pS + g * 64, neg_inf);
        return;
    }
    hi = hi >  64 ?  64 : hi;   // clamp so hi/lo + 4*g stay in int16 range
    lo = lo < -64 ? -64 : lo;
    const aie::vector<int16, 64> D = aie::load_v<64>(d_offset);

    for (unsigned z = 0; z < rowA; z += 2){
        T_out *__restrict pS1 = pS + (z * colB + 0) * MMUL::size_C;
        T_out *__restrict pS2 = pS + ((z + 1) * colB + 0) * MMUL::size_C;

        for (unsigned j = 0; j < colB; j += 2){
            const T_in *__restrict pQ1 = pQ + (z * colA + 0) * MMUL::size_A;
            const T_in *__restrict pQ2 = pQ + ((z + 1) * colA + 0) * MMUL::size_A;
            const T_in *__restrict pK1 = pK + (0 * colB + j) * MMUL::size_B;
            const T_in *__restrict pK2 = pK + (0 * colB + (j + 1)) * MMUL::size_B;

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
            auto mout0 = aie::interleave_zip(C00.template to_vector<T_out>(), C01.template to_vector<T_out>(), 8);
            auto mout1 = aie::interleave_zip(C10.template to_vector<T_out>(), C11.template to_vector<T_out>(), 8);

            auto mout2 = aie::mul(mout0.first, (T_out)scale).template to_vector<T_out>(0);
            auto mout3 = aie::mul(mout0.second, (T_out)scale).template to_vector<T_out>(0);
            auto mout4 = aie::mul(mout1.first, (T_out)scale).template to_vector<T_out>(0);
            auto mout5 = aie::mul(mout1.second, (T_out)scale).template to_vector<T_out>(0);

            const int g0 = z * 2;
            _qk_mask_max<T_out>(pS1, pM + (g0 + 0) * 64, D, mout2, hi + 4 * (g0 + 0), lo + 4 * (g0 + 0));
            pS1 += MMUL::size_C;
            _qk_mask_max<T_out>(pS1, pM + (g0 + 1) * 64, D, mout3, hi + 4 * (g0 + 1), lo + 4 * (g0 + 1));
            pS1 += MMUL::size_C;
            _qk_mask_max<T_out>(pS2, pM + (g0 + 2) * 64, D, mout4, hi + 4 * (g0 + 2), lo + 4 * (g0 + 2));
            pS2 += MMUL::size_C;
            _qk_mask_max<T_out>(pS2, pM + (g0 + 3) * 64, D, mout5, hi + 4 * (g0 + 3), lo + 4 * (g0 + 3));
            pS2 += MMUL::size_C;
        }
    }
}

void scale_div_aie(bf16* o, bf16 *l, float *y) {
    constexpr int vec_factor = 64;

    // organize l
    aie::vector<bf16, 8> L0 = aie::broadcast<bf16, 8>(l[0]);
    aie::vector<bf16, 8> L1 = aie::broadcast<bf16, 8>(l[1]);
    aie::vector<bf16, 8> L2 = aie::broadcast<bf16, 8>(l[2]);
    aie::vector<bf16, 8> L3 = aie::broadcast<bf16, 8>(l[3]);
    aie::vector<bf16, 8> L4 = aie::broadcast<bf16, 8>(l[4]);
    aie::vector<bf16, 8> L5 = aie::broadcast<bf16, 8>(l[5]);
    aie::vector<bf16, 8> L6 = aie::broadcast<bf16, 8>(l[6]);
    aie::vector<bf16, 8> L7 = aie::broadcast<bf16, 8>(l[7]);

    auto LL00 = aie::concat(L0, L1, L2, L3, L4, L5, L6, L7);

    aie::vector<float, vec_factor> Y00 = aie::load_v<vec_factor>(y);
    aie::accum<accfloat, vec_factor> Y00_acc;
    Y00_acc.from_vector(Y00);
    aie::accum<accfloat, vec_factor> AL00 = aie::mul(Y00_acc.template to_vector<bf16>(), LL00);
    aie::store_v(o, AL00.template to_vector<bf16>());
}
#endif