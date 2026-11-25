// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef ATTN_KERNEL_H
#define ATTN_KERNEL_H
#include <aie_api/aie.hpp>
#include "common.h"
#include <math.h>

// S (8 tiles of 16 q x 16 k, rows of 16) = exp(S - m): mb the reference of each row broadcast to 16 lanes
void apply_softmax(bf16*__restrict pS, const bf16*__restrict mb) {
    for (int k = 0; k < 8 * lq / 2; k++) {
        aie::accum<accfloat, 32> a = aie::mul(aie::load_v<32>(pS), exp_scale);
        a = aie::msc(a, aie::load_v<32>(mb + (k % (lq / 2)) * 32), exp_scale);
        aie::store_v(pS, aie::exp2<bf16>(a.template to_vector<float>()));
        pS += 32;
    }
}

// One 16 q x 16 k tile of S: add the relative bias (BIAS: row i adds pB[i * (win - 1) .. + 15]), mask the
// keys outside the window, and fold the rows into the lane maxima m (16 x 16). dq = q - k of row 0, key 0;
// a key is visible when 0 <= q - k < wd. didx: q - k offsets of a row pair (lane l: l / 16 - l % 16).
template <bool BIAS>
void mask_bias_max(bf16*__restrict pS, bf16*__restrict m, int dq, int wd, const int*__restrict didx,
                   const bf16*__restrict pB) {
    if (dq - 15 >= 0 && dq + 15 < wd) {
        // the whole tile is visible
        for (int i = 0; i < lq; i += 2) {
            aie::vector<bf16, 32> v = aie::load_v<32>(pS);
            if constexpr (BIAS) {
                aie::accum<accfloat, 32> acc;
                acc.from_vector(v);
                v = aie::add(acc, aie::concat(aie::load_unaligned_v<16>(pB), aie::load_unaligned_v<16>(pB + win - 1)))
                        .template to_vector<bf16>();
                aie::store_v(pS, v);
                pB += 2 * (win - 1);
            }
            aie::store_v(m, aie::max(aie::load_v<32>(m), v));
            pS += 32;
            m += 32;
        }
        return;
    }
    const aie::vector<int, 32> d0 = aie::add(aie::load_v<32>(didx), dq);
    const aie::vector<bf16, 32> neg = aie::broadcast<bf16, 32>(-0x1.FEp127f);
    for (int i = 0; i < lq; i += 2) {
        const aie::vector<int, 32> d = aie::add(d0, i);
        const aie::mask<32> vis = aie::ge(d, 0) & aie::lt(d, wd);
        aie::vector<bf16, 32> v = aie::load_v<32>(pS);
        if constexpr (BIAS) {
            aie::accum<accfloat, 32> acc;
            acc.from_vector(v);
            v = aie::add(acc, aie::concat(aie::load_unaligned_v<16>(pB), aie::load_unaligned_v<16>(pB + win - 1)))
                    .template to_vector<bf16>();
            pB += 2 * (win - 1);
        }
        v = aie::select(neg, v, vis);
        aie::store_v(pS, v);
        aie::store_v(m, aie::max(aie::load_v<32>(m), v));
        pS += 32;
        m += 32;
    }
}

inline aie::vector<bf16, 32> pair_max(aie::vector<bf16, 64> v) {
    return aie::max(aie::filter_even(v, 1), aie::filter_odd(v, 1));
}

// out[i] = max of row i of m (16 x 16)
void row_max(bf16*__restrict out, const bf16*__restrict m) {
    const auto a0 = pair_max(aie::load_v<64>(m)), a1 = pair_max(aie::load_v<64>(m + 64));
    const auto a2 = pair_max(aie::load_v<64>(m + 128)), a3 = pair_max(aie::load_v<64>(m + 192));
    const auto c = pair_max(aie::concat(pair_max(aie::concat(a0, a1)), pair_max(aie::concat(a2, a3))));
    aie::store_v(out, aie::max(aie::filter_even(c, 1), aie::filter_odd(c, 1)));
}

// P[16][win] = R . W^T: R the 16 x nr rel features of the core (8x8 tiles), W one window [2][win][8]
void compute_p(bf16*__restrict pP, const bf16*__restrict pR, const bf16*__restrict pW) {
    using MMUL = aie::mmul<8, 8, 8, bf16, bf16, accauto>;
    const aie::vector<bf16, 64> R00 = aie::load_v<64>(pR);
    const aie::vector<bf16, 64> R01 = aie::load_v<64>(pR + 64);
    const aie::vector<bf16, 64> R10 = aie::load_v<64>(pR + 128);
    const aie::vector<bf16, 64> R11 = aie::load_v<64>(pR + 192);
    for (int xb = 0; xb < win / 8; xb += 2) {
        aie::vector<bf16, 64> B00 = aie::transpose(aie::load_v<64>(pW + xb * 64), 8, 8);
        aie::vector<bf16, 64> B01 = aie::transpose(aie::load_v<64>(pW + win * 8 + xb * 64), 8, 8);
        aie::vector<bf16, 64> B10 = aie::transpose(aie::load_v<64>(pW + (xb + 1) * 64), 8, 8);
        aie::vector<bf16, 64> B11 = aie::transpose(aie::load_v<64>(pW + win * 8 + (xb + 1) * 64), 8, 8);
        MMUL C00, C01, C10, C11;
        C00.mul(R00, B00); C00.mac(R01, B01);
        C01.mul(R00, B10); C01.mac(R01, B11);
        C10.mul(R10, B00); C10.mac(R11, B01);
        C11.mul(R10, B10); C11.mac(R11, B11);
        auto top = aie::interleave_zip(C00.template to_vector<bf16>(), C01.template to_vector<bf16>(), 8);
        auto bot = aie::interleave_zip(C10.template to_vector<bf16>(), C11.template to_vector<bf16>(), 8);
        bf16* o = pP + xb * 8;
        for (int r = 0; r < 4; r++) {
            aie::store_v(o + r * win, top.first.template extract<16>(r));
            aie::store_v(o + (r + 4) * win, top.second.template extract<16>(r));
            aie::store_v(o + (r + 8) * win, bot.first.template extract<16>(r));
            aie::store_v(o + (r + 12) * win, bot.second.template extract<16>(r));
        }
    }
}

// c = exp(prev - next) per row
void calculate_c(float*__restrict c, const bf16*__restrict prev, const bf16*__restrict next) {
    aie::accum<accfloat, lq> a = aie::mul(aie::load_v<lq>(prev), exp_scale);
    a = aie::msc(a, aie::load_v<lq>(next), exp_scale);
    aie::accum<accfloat, lq> e;
    e.from_vector(aie::exp2<bf16>(a.template to_vector<float>()));
    aie::store_v(c, e.template to_vector<float>());
}

void reorder_s(bf16*__restrict pS) {
    for (int b = 0; b < 8; b++){
        bf16 *pS1 = pS + b * 256;
        aie::vector<bf16, 64> S0 = aie::load_v<64>(pS1);
        aie::vector<bf16, 32> S00 = aie::filter_even(S0, 8);
        aie::vector<bf16, 32> S01 = aie::filter_odd(S0, 8);
        aie::vector<bf16, 64> S1 = aie::load_v<64>(pS1 + 64);
        aie::vector<bf16, 32> S10 = aie::filter_even(S1, 8);
        aie::vector<bf16, 32> S11 = aie::filter_odd(S1, 8);
        aie::vector<bf16, 64> S2 = aie::load_v<64>(pS1 + 128);
        aie::vector<bf16, 32> S20 = aie::filter_even(S2, 8);
        aie::vector<bf16, 32> S21 = aie::filter_odd(S2, 8);
        aie::vector<bf16, 64> S3 = aie::load_v<64>(pS1 + 192);
        aie::vector<bf16, 32> S30 = aie::filter_even(S3, 8);
        aie::vector<bf16, 32> S31 = aie::filter_odd(S3, 8);

        S0 = aie::concat(S00, S10);
        S1 = aie::concat(S01, S11);
        S2 = aie::concat(S20, S30);
        S3 = aie::concat(S21, S31);

        aie::store_v(pS1, S0);
        aie::store_v(pS1 + 64, S1);
        aie::store_v(pS1 + 128, S2);
        aie::store_v(pS1 + 192, S3);
    }
}

void calculate_l(float* l, float*__restrict c, bf16*__restrict pS) {
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

    aie::vector<float, 16> c_float32 = aie::load_v<16>(c);
    aie::vector<float, 16> l_float32 = aie::load_v<16>(l);

    aie::accum<accfloat, 16> l_out;
    l_out = mac_elem_16_accuracy_safe(l_float32, c_float32, sum, 0, 0, 0);
    aie::store_v(l, l_out.template to_vector<float>());
}

void calculate_y(float* y, float* c) {
    // calculate y
    for (int i = 0; i < lq / 8; i++)
    chess_prepare_for_pipelining chess_loop_range(2, ) {
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
        for (unsigned j = 0; j < dh / 8; j += 1){
            aie::vector<float, 64> Y = aie::load_v<64>(pY);
            aie::accum<accfloat, 64> ACC_Y = aie::mul(CORRECT, Y);
            aie::store_v(pY, ACC_Y.template to_vector<float>());
            pY += 64;
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
void _attn_qk(T_out *__restrict pS, T_in *__restrict pQ, T_in *__restrict pK) {
    using MMUL = aie::mmul<r, s, t, T_in, T_in, accauto>;

    for (unsigned z = 0; z < rowA; z += 2){
        T_out *__restrict pS1 = pS + (z * colB + 0) * MMUL::size_C;
        T_out *__restrict pS2 = pS + ((z + 1) * colB + 0) * MMUL::size_C;

        for (unsigned j = 0; j < colB; j += 2){
            const T_in *__restrict pQ1 = pQ + (z * colA + 0) * MMUL::size_A;
            const T_in *__restrict pQ2 = pQ + ((z + 1) * colA + 0) * MMUL::size_A;
            const T_in *__restrict pK1 = pK + (0 * colB + j) * MMUL::size_B;
            const T_in *__restrict pK2 = pK + (0 * colB + (j + 1)) * MMUL::size_B;

#if defined(__chess__)
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
#else
            // Peano-only: run the 4 accumulators as two K-passes instead of one
            // 4-accumulator loop. Xilinx/llvm-aie#1066: this loop is
            // register-footprint-bound (~9 of 12 X regs live), which is what stops
            // the post-pipeliner from overlapping iterations. Each pass keeps only
            // Q0/Q1 and ONE K operand live. Every accumulator still sees exactly the
            // same MAC sequence in the same order, so the result is bit-identical.
            aie::vector<T_out, MMUL::size_C> zc = aie::zeros<T_out, MMUL::size_C>();
            MMUL C00(zc); MMUL C01(zc); MMUL C10(zc); MMUL C11(zc);
            {   // pass 1: C00, C10  <- K0
                const T_in *__restrict qa = pQ1;
                const T_in *__restrict qb = pQ2;
                const T_in *__restrict ka = pK1;
                for (unsigned i = 0; i < colA; ++i){
                    aie::vector<T_in, MMUL::size_A> A0 = aie::load_v<MMUL::size_A>(qa); qa += MMUL::size_A;
                    aie::vector<T_in, MMUL::size_A> A1 = aie::load_v<MMUL::size_A>(qb); qb += MMUL::size_A;
                    aie::vector<T_in, MMUL::size_B> B0 = aie::transpose(aie::load_v<MMUL::size_B>(ka), 8, 8); ka += MMUL::size_B * colB;
                    C00.mac(A0, B0);
                    C10.mac(A1, B0);
                }
            }
            {   // pass 2: C01, C11  <- K1
                const T_in *__restrict qa = pQ1;
                const T_in *__restrict qb = pQ2;
                const T_in *__restrict kb = pK2;
                for (unsigned i = 0; i < colA; ++i){
                    aie::vector<T_in, MMUL::size_A> A0 = aie::load_v<MMUL::size_A>(qa); qa += MMUL::size_A;
                    aie::vector<T_in, MMUL::size_A> A1 = aie::load_v<MMUL::size_A>(qb); qb += MMUL::size_A;
                    aie::vector<T_in, MMUL::size_B> B1 = aie::transpose(aie::load_v<MMUL::size_B>(kb), 8, 8); kb += MMUL::size_B * colB;
                    C01.mac(A0, B1);
                    C11.mac(A1, B1);
                }
            }
#endif
            auto mout0 = aie::interleave_zip(C00.template to_vector<T_out>(), C01.template to_vector<T_out>(), 8);
            auto mout1 = aie::interleave_zip(C10.template to_vector<T_out>(), C11.template to_vector<T_out>(), 8);

            auto mout2 = aie::mul(mout0.first, (T_out)scale).template to_vector<T_out>(0);
            auto mout3 = aie::mul(mout0.second, (T_out)scale).template to_vector<T_out>(0);
            auto mout4 = aie::mul(mout1.first, (T_out)scale).template to_vector<T_out>(0);
            auto mout5 = aie::mul(mout1.second, (T_out)scale).template to_vector<T_out>(0);

            aie::store_v(pS1, mout2);
            pS1 += MMUL::size_C;
            aie::store_v(pS1, mout3);
            pS1 += MMUL::size_C;
            aie::store_v(pS2, mout4);
            pS2 += MMUL::size_C;
            aie::store_v(pS2, mout5);
            pS2 += MMUL::size_C;
        }
    }
}

// bfp16 tiles (72 bytes: 8 blocks of a shared exponent byte and 8 int8 mantissas) as the native bfp16 MAC takes
// them: mac_8x8_8x8T(A, B) = A B^T with A and B both row-major 8x8, one block per row.

// this core's 16 queries (bf16 tiles of the pack, [q8 0, 1][d8 16]) -> bfp16 tiles [d8 16][q8 0, 1], once per round
void convert_q(uint8*__restrict pQb, const bf16*__restrict pQ) {
    aie::block_vector_output_buffer_stream<bfp16ebs8, 64> qo((bfp16ebs8 *)pQb);
    for (int i = 0; i < dh / 8; i++) {
        qo << ::to_v64bfp16ebs8(aie::accum<accfloat, 64>(aie::load_v<64>(pQ + i * 64)));
        qo << ::to_v64bfp16ebs8(aie::accum<accfloat, 64>(aie::load_v<64>(pQ + (dh / 8 + i) * 64)));
    }
}

// S (16 q x 16 k: [q8 0, 1] rows of 16, as before) = scale Q K^T; pQ bfp16 [d8 16][q8 0, 1], pK one 16-key
// tile from the host, bfp16 [d8 16][k8 0, 1] (K rows: one block per key)
void attn_qk_bfp16(bf16*__restrict pS, const uint8*__restrict pQ, const uint8*__restrict pK) {
    aie::block_vector_input_buffer_stream<bfp16ebs8, 64> qs((const bfp16ebs8 *)pQ);
    aie::block_vector_input_buffer_stream<bfp16ebs8, 64> ks((const bfp16ebs8 *)pK);
    aie::accum<accfloat, 64> c00 = aie::zeros<accfloat, 64>(), c01 = aie::zeros<accfloat, 64>();
    aie::accum<accfloat, 64> c10 = aie::zeros<accfloat, 64>(), c11 = aie::zeros<accfloat, 64>();
    chess_prepare_for_pipelining
    for (int i = 0; i < dh / 8; i++) {
        auto a0 = qs.pop();
        auto a1 = qs.pop();
        auto b0 = ks.pop();
        auto b1 = ks.pop();
        c00 = mac_8x8_8x8T(a0, b0, c00);
        c01 = mac_8x8_8x8T(a0, b1, c01);
        c10 = mac_8x8_8x8T(a1, b0, c10);
        c11 = mac_8x8_8x8T(a1, b1, c11);
    }
    auto mout0 = aie::interleave_zip(c00.template to_vector<bf16>(), c01.template to_vector<bf16>(), 8);
    auto mout1 = aie::interleave_zip(c10.template to_vector<bf16>(), c11.template to_vector<bf16>(), 8);
    aie::store_v(pS, aie::mul(mout0.first, (bf16)scale).template to_vector<bf16>(0));
    aie::store_v(pS + 64, aie::mul(mout0.second, (bf16)scale).template to_vector<bf16>(0));
    aie::store_v(pS + 128, aie::mul(mout1.first, (bf16)scale).template to_vector<bf16>(0));
    aie::store_v(pS + 192, aie::mul(mout1.second, (bf16)scale).template to_vector<bf16>(0));
}

// Y (16 q x dh, fp32 8x8 tiles [q8 0, 1][d8 16]) += P V; pS the reordered 8x8 tiles of P [q8][k8], pV one 16-key
// tile from the host, bfp16 V^T tiles [d8 pair 8][k8 0, 1][d8 of the pair] (one block per d)
void attn_fv_bfp16(float*__restrict pY, const bf16*__restrict pS, const uint8*__restrict pV) {
    const auto S0 = ::to_v64bfp16ebs8(aie::accum<accfloat, 64>(aie::load_v<64>(pS)));
    const auto S1 = ::to_v64bfp16ebs8(aie::accum<accfloat, 64>(aie::load_v<64>(pS + 64)));
    const auto S2 = ::to_v64bfp16ebs8(aie::accum<accfloat, 64>(aie::load_v<64>(pS + 128)));
    const auto S3 = ::to_v64bfp16ebs8(aie::accum<accfloat, 64>(aie::load_v<64>(pS + 192)));
    aie::block_vector_input_buffer_stream<bfp16ebs8, 64> vs((const bfp16ebs8 *)pV);
    float *__restrict pY1 = pY;
    float *__restrict pY2 = pY + (dh / 8) * 64;
    chess_prepare_for_pipelining
    for (int j = 0; j < dh / 8; j += 2) {
        auto v00 = vs.pop();   // d8 j, keys 0-7
        auto v10 = vs.pop();   // d8 j + 1, keys 0-7
        auto v01 = vs.pop();   // d8 j, keys 8-15
        auto v11 = vs.pop();   // d8 j + 1, keys 8-15
        aie::accum<accfloat, 64> y00(aie::load_v<64>(pY1)), y01(aie::load_v<64>(pY1 + 64));
        aie::accum<accfloat, 64> y10(aie::load_v<64>(pY2)), y11(aie::load_v<64>(pY2 + 64));
        y00 = mac_8x8_8x8T(S0, v00, y00);
        y01 = mac_8x8_8x8T(S0, v10, y01);
        y10 = mac_8x8_8x8T(S2, v00, y10);
        y11 = mac_8x8_8x8T(S2, v10, y11);
        y00 = mac_8x8_8x8T(S1, v01, y00);
        y01 = mac_8x8_8x8T(S1, v11, y01);
        y10 = mac_8x8_8x8T(S3, v01, y10);
        y11 = mac_8x8_8x8T(S3, v11, y11);
        aie::store_v(pY1, y00.template to_vector<float>());
        aie::store_v(pY1 + 64, y01.template to_vector<float>());
        aie::store_v(pY2, y10.template to_vector<float>());
        aie::store_v(pY2 + 64, y11.template to_vector<float>());
        pY1 += 128;
        pY2 += 128;
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