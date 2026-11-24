// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include <type_traits>

#include <aie_api/aie.hpp>
#include "common.h"
#include "lut_based_ops.h"
#include "zero.h"
#include "gptossParam.hpp"


void reorder_s(bf16 *__restrict pS) {
    aie::vector<bf16, 64> S0 = aie::load_v<64>(pS);
    aie::vector<bf16, 64> S1 = aie::load_v<64>(pS + 64);
    aie::vector<bf16, 64> S2 = aie::load_v<64>(pS + 128);
    aie::vector<bf16, 64> S3 = aie::load_v<64>(pS + 192);

    aie::vector<bf16, 32> S00 = aie::filter_even(S0, 8);
    aie::vector<bf16, 32> S01 = aie::filter_odd(S0, 8);
    aie::vector<bf16, 32> S10 = aie::filter_even(S1, 8);
    aie::vector<bf16, 32> S11 = aie::filter_odd(S1, 8);
    aie::vector<bf16, 32> S20 = aie::filter_even(S2, 8);
    aie::vector<bf16, 32> S21 = aie::filter_odd(S2, 8);
    aie::vector<bf16, 32> S30 = aie::filter_even(S3, 8);
    aie::vector<bf16, 32> S31 = aie::filter_odd(S3, 8);

    S0 = aie::concat(S00, S10);
    S1 = aie::concat(S01, S11);
    S2 = aie::concat(S20, S30);
    S3 = aie::concat(S21, S31);

    aie::store_v(pS, S0);
    aie::store_v(pS + 64, S1);
    aie::store_v(pS + 128, S2);
    aie::store_v(pS + 192, S3);
}

void calculate_l(bf16 *__restrict pS, float *__restrict c, float *__restrict l) {
    using MMUL = aie::mmul<8, 8, 8, bf16, bf16, accfloat>;
    bf16 *__restrict pS1 = pS;

    aie::vector<bf16, 64> S0 = aie::load_v<64>(pS1);
    pS1 += 64;
    aie::vector<bf16, 64> S1 = aie::load_v<64>(pS1);
    pS1 += 64;
    aie::vector<bf16, 64> S2 = aie::load_v<64>(pS1);
    pS1 += 64;
    aie::vector<bf16, 64> S3 = aie::load_v<64>(pS1);

    aie::vector<bf16, MMUL::size_B> ones = aie::broadcast<bf16, MMUL::size_B>(1);

    aie::vector<bf16, MMUL::size_C> acc_C0 = aie::zeros<bf16, MMUL::size_C>(); 
    MMUL C0(acc_C0);
    C0.mac(S0, ones);
    C0.mac(S1, ones);
    aie::vector<float, MMUL::size_C> sumf = C0.template to_vector<float>();
    auto sum03 = aie::filter_even(sumf, 4);
    auto sum02 = aie::filter_even(sum03, 2);
    auto sum01 = aie::filter_even(sum02, 1);

    aie::vector<bf16, MMUL::size_C> acc_C1 = aie::zeros<bf16, MMUL::size_C>();
    MMUL C1(acc_C1);
    C1.mac(S2, ones);
    C1.mac(S3, ones);
    aie::vector<float, MMUL::size_C> sumf1 = C1.template to_vector<float>();
    auto sum13 = aie::filter_even(sumf1, 4);
    auto sum12 = aie::filter_even(sum13, 2);
    auto sum11 = aie::filter_even(sum12, 1);

    auto sum = aie::concat(sum01, sum11);
    aie::accum<accfloat, 16> sum_acc;
    sum_acc.from_vector(sum);

    aie::vector<float, 16> c_float = aie::load_v<16>(c);
    aie::vector<float, 16> l_float = aie::load_v<16>(l);

    aie::accum<accfloat, 16> l_acc;
    l_acc = mac_elem_16_accuracy_safe(l_float, c_float, sum_acc, 0, 0, 0);
    aie::store_v(l, l_acc.template to_vector<float>());
}

#if defined(__chess__)

template <unsigned colQ, unsigned r, unsigned s, unsigned t>
void attn_fv(bf16 *__restrict pS, bf16 *__restrict pV, float *__restrict pY, float *__restrict c) {

    using MMUL = aie::mmul<r, s, t, bf16, bf16, accfloat>;

    event0();

    bf16 *__restrict pS1 = pS;
   
    aie::vector<bf16, 64> S0 = aie::load_v<64>(pS1);
    pS1 += 64;
    aie::vector<bf16, 64> S1 = aie::load_v<64>(pS1);
    pS1 += 64;
    aie::vector<bf16, 64> S2 = aie::load_v<64>(pS1);
    pS1 += 64;
    aie::vector<bf16, 64> S3 = aie::load_v<64>(pS1);
    
    {
        aie::vector<float, 8> c0 = aie::broadcast<float, 8>(c[0]);
        aie::vector<float, 8> c1 = aie::broadcast<float, 8>(c[1]);
        aie::vector<float, 8> c2 = aie::broadcast<float, 8>(c[2]);
        aie::vector<float, 8> c3 = aie::broadcast<float, 8>(c[3]);   
        aie::vector<float, 8> c4 = aie::broadcast<float, 8>(c[4]);
        aie::vector<float, 8> c5 = aie::broadcast<float, 8>(c[5]);
        aie::vector<float, 8> c6 = aie::broadcast<float, 8>(c[6]);
        aie::vector<float, 8> c7 = aie::broadcast<float, 8>(c[7]);
        aie::vector<float, 64> CORRECT = aie::concat(c0, c1, c2, c3, c4, c5, c6, c7);

        bf16 *__restrict pV1 = pV;
        float *__restrict pY1 = pY;

        for (unsigned j = 0; j < colQ; j += 4)
        chess_prepare_for_pipelining chess_loop_range(2, ) {
            //v0
            bf16 *__restrict pV01 = pV1 + (j + 0) * MMUL::size_B;
            bf16 *__restrict pV02 = pV1 + (j + 1) * MMUL::size_B;
            bf16 *__restrict pV03 = pV1 + (j + 2) * MMUL::size_B;
            bf16 *__restrict pV04 = pV1 + (j + 3) * MMUL::size_B;

            aie::vector<bf16, MMUL::size_B> V00 = aie::load_v<MMUL::size_B>(pV01);
            pV01 += MMUL::size_B * colQ * 2;
            aie::vector<bf16, MMUL::size_B> V01 = aie::load_v<MMUL::size_B>(pV02);
            pV02 += MMUL::size_B * colQ * 2;
            aie::vector<bf16, MMUL::size_B> V02 = aie::load_v<MMUL::size_B>(pV03);
            pV03 += MMUL::size_B * colQ * 2;
            aie::vector<bf16, MMUL::size_B> V03 = aie::load_v<MMUL::size_B>(pV04);
            pV04 += MMUL::size_B * colQ * 2;

            aie::vector<float, MMUL::size_C> acc_Y00 = aie::load_v<MMUL::size_C>(pY1);
            aie::vector<float, MMUL::size_C> acc_Y01 = aie::load_v<MMUL::size_C>(pY1 + MMUL::size_C);
            aie::vector<float, MMUL::size_C> acc_Y02 = aie::load_v<MMUL::size_C>(pY1 + MMUL::size_C * 2);
            aie::vector<float, MMUL::size_C> acc_Y03 = aie::load_v<MMUL::size_C>(pY1 + MMUL::size_C * 3);

            aie::accum<accfloat, MMUL::size_C> ACC_Y00;
            aie::accum<accfloat, MMUL::size_C> ACC_Y01;
            aie::accum<accfloat, MMUL::size_C> ACC_Y02;
            aie::accum<accfloat, MMUL::size_C> ACC_Y03;
            ACC_Y00 = aie::mul(CORRECT, acc_Y00);
            ACC_Y01 = aie::mul(CORRECT, acc_Y01);
            ACC_Y02 = aie::mul(CORRECT, acc_Y02);
            ACC_Y03 = aie::mul(CORRECT, acc_Y03);
            
            MMUL Y00(ACC_Y00);
            MMUL Y01(ACC_Y01);
            MMUL Y02(ACC_Y02);
            MMUL Y03(ACC_Y03);


            Y00.mac(S0, V00);
            Y01.mac(S0, V01);
            Y02.mac(S0, V02);
            Y03.mac(S0, V03);

            V00 = aie::load_v<MMUL::size_B>(pV01);
            V01 = aie::load_v<MMUL::size_B>(pV02);
            V02 = aie::load_v<MMUL::size_B>(pV03);
            V03 = aie::load_v<MMUL::size_B>(pV04);

            Y00.mac(S1, V00);
            Y01.mac(S1, V01);
            Y02.mac(S1, V02);
            Y03.mac(S1, V03);

            aie::store_v(pY1, Y00.template to_vector<float>());
            pY1 += MMUL::size_C;
            aie::store_v(pY1, Y01.template to_vector<float>());
            pY1 += MMUL::size_C;
            aie::store_v(pY1, Y02.template to_vector<float>());
            pY1 += MMUL::size_C;
            aie::store_v(pY1, Y03.template to_vector<float>());
            pY1 += MMUL::size_C;
        }
    }

    {
        aie::vector<float, 8> c0 = aie::broadcast<float, 8>(c[8]);
        aie::vector<float, 8> c1 = aie::broadcast<float, 8>(c[9]);
        aie::vector<float, 8> c2 = aie::broadcast<float, 8>(c[10]);
        aie::vector<float, 8> c3 = aie::broadcast<float, 8>(c[11]);   
        aie::vector<float, 8> c4 = aie::broadcast<float, 8>(c[12]);
        aie::vector<float, 8> c5 = aie::broadcast<float, 8>(c[13]);
        aie::vector<float, 8> c6 = aie::broadcast<float, 8>(c[14]);
        aie::vector<float, 8> c7 = aie::broadcast<float, 8>(c[15]);
        aie::vector<float, 64> CORRECT = aie::concat(c0, c1, c2, c3, c4, c5, c6, c7);

        bf16 *__restrict pV2 = pV + MMUL::size_B * colQ;
        float *__restrict pY2 = pY + MMUL::size_C * colQ;

        for (unsigned j = 0; j < colQ; j += 4)
        chess_prepare_for_pipelining chess_loop_range(2, ) {
            //v1
            bf16 *__restrict pV11 = pV2 + (j + 0) * MMUL::size_B;
            bf16 *__restrict pV12 = pV2 + (j + 1) * MMUL::size_B;
            bf16 *__restrict pV13 = pV2 + (j + 2) * MMUL::size_B;
            bf16 *__restrict pV14 = pV2 + (j + 3) * MMUL::size_B;

            aie::vector<bf16, MMUL::size_B> V10 = aie::load_v<MMUL::size_B>(pV11);
            pV11 += MMUL::size_B * colQ * 2;
            aie::vector<bf16, MMUL::size_B> V11 = aie::load_v<MMUL::size_B>(pV12);
            pV12 += MMUL::size_B * colQ * 2;
            aie::vector<bf16, MMUL::size_B> V12 = aie::load_v<MMUL::size_B>(pV13);
            pV13 += MMUL::size_B * colQ * 2;
            aie::vector<bf16, MMUL::size_B> V13 = aie::load_v<MMUL::size_B>(pV14);
            pV14 += MMUL::size_B * colQ * 2;

            aie::vector<float, MMUL::size_C> acc_Y10 = aie::load_v<MMUL::size_C>(pY2);
            aie::vector<float, MMUL::size_C> acc_Y11 = aie::load_v<MMUL::size_C>(pY2 + MMUL::size_C);
            aie::vector<float, MMUL::size_C> acc_Y12 = aie::load_v<MMUL::size_C>(pY2 + MMUL::size_C * 2);
            aie::vector<float, MMUL::size_C> acc_Y13 = aie::load_v<MMUL::size_C>(pY2 + MMUL::size_C * 3);

            aie::accum<accfloat, MMUL::size_C> ACC_Y10;
            aie::accum<accfloat, MMUL::size_C> ACC_Y11;
            aie::accum<accfloat, MMUL::size_C> ACC_Y12;
            aie::accum<accfloat, MMUL::size_C> ACC_Y13;
            ACC_Y10 = aie::mul(CORRECT, acc_Y10);
            ACC_Y11 = aie::mul(CORRECT, acc_Y11);
            ACC_Y12 = aie::mul(CORRECT, acc_Y12);
            ACC_Y13 = aie::mul(CORRECT, acc_Y13);

            MMUL Y10(ACC_Y10);
            MMUL Y11(ACC_Y11);
            MMUL Y12(ACC_Y12);
            MMUL Y13(ACC_Y13);

            Y10.mac(S2, V10);
            Y11.mac(S2, V11);
            Y12.mac(S2, V12);
            Y13.mac(S2, V13);

            V10 = aie::load_v<MMUL::size_B>(pV11);
            V11 = aie::load_v<MMUL::size_B>(pV12);
            V12 = aie::load_v<MMUL::size_B>(pV13);
            V13 = aie::load_v<MMUL::size_B>(pV14);

            Y10.mac(S3, V10);
            Y11.mac(S3, V11);
            Y12.mac(S3, V12);
            Y13.mac(S3, V13);

            aie::store_v(pY2, Y10.template to_vector<float>());
            pY2 += MMUL::size_C;
            aie::store_v(pY2, Y11.template to_vector<float>());
            pY2 += MMUL::size_C;
            aie::store_v(pY2, Y12.template to_vector<float>());
            pY2 += MMUL::size_C;
            aie::store_v(pY2, Y13.template to_vector<float>());
            pY2 += MMUL::size_C;
        }
    }
    event1();
}

#else  // Peano

// Peano: one V-plane per call. With both planes in one frame and the BFP16
// mmul path enabled, Peano has to spill BFP16 matrix registers and fails to
// encode the spills. A non-inlined per-plane function halves the live state.
template <unsigned colQ, unsigned r, unsigned s, unsigned t>
__attribute__((noinline))
void attn_fv_plane(bf16 *__restrict pS, bf16 *__restrict pV1, float *__restrict pY1,
                   const float *__restrict c) {

    using MMUL = aie::mmul<r, s, t, bf16, bf16, accfloat>;

    aie::vector<bf16, 64> SA = aie::load_v<64>(pS);
    aie::vector<bf16, 64> SB = aie::load_v<64>(pS + 64);

    aie::vector<float, 8> c0 = aie::broadcast<float, 8>(c[0]);
    aie::vector<float, 8> c1 = aie::broadcast<float, 8>(c[1]);
    aie::vector<float, 8> c2 = aie::broadcast<float, 8>(c[2]);
    aie::vector<float, 8> c3 = aie::broadcast<float, 8>(c[3]);
    aie::vector<float, 8> c4 = aie::broadcast<float, 8>(c[4]);
    aie::vector<float, 8> c5 = aie::broadcast<float, 8>(c[5]);
    aie::vector<float, 8> c6 = aie::broadcast<float, 8>(c[6]);
    aie::vector<float, 8> c7 = aie::broadcast<float, 8>(c[7]);
    aie::vector<float, 64> CORRECT = aie::concat(c0, c1, c2, c3, c4, c5, c6, c7);

    for (unsigned j = 0; j < colQ; j += 4)
    chess_prepare_for_pipelining chess_loop_range(2, ) {
        bf16 *__restrict pV01 = pV1 + (j + 0) * MMUL::size_B;
        bf16 *__restrict pV02 = pV1 + (j + 1) * MMUL::size_B;
        bf16 *__restrict pV03 = pV1 + (j + 2) * MMUL::size_B;
        bf16 *__restrict pV04 = pV1 + (j + 3) * MMUL::size_B;

        aie::vector<bf16, MMUL::size_B> V00 = aie::load_v<MMUL::size_B>(pV01);
        pV01 += MMUL::size_B * colQ * 2;
        aie::vector<bf16, MMUL::size_B> V01 = aie::load_v<MMUL::size_B>(pV02);
        pV02 += MMUL::size_B * colQ * 2;
        aie::vector<bf16, MMUL::size_B> V02 = aie::load_v<MMUL::size_B>(pV03);
        pV03 += MMUL::size_B * colQ * 2;
        aie::vector<bf16, MMUL::size_B> V03 = aie::load_v<MMUL::size_B>(pV04);
        pV04 += MMUL::size_B * colQ * 2;

        aie::vector<float, MMUL::size_C> acc_Y00 = aie::load_v<MMUL::size_C>(pY1);
        aie::vector<float, MMUL::size_C> acc_Y01 = aie::load_v<MMUL::size_C>(pY1 + MMUL::size_C);
        aie::vector<float, MMUL::size_C> acc_Y02 = aie::load_v<MMUL::size_C>(pY1 + MMUL::size_C * 2);
        aie::vector<float, MMUL::size_C> acc_Y03 = aie::load_v<MMUL::size_C>(pY1 + MMUL::size_C * 3);

        aie::accum<accfloat, MMUL::size_C> ACC_Y00;
        aie::accum<accfloat, MMUL::size_C> ACC_Y01;
        aie::accum<accfloat, MMUL::size_C> ACC_Y02;
        aie::accum<accfloat, MMUL::size_C> ACC_Y03;
        ACC_Y00 = aie::mul(CORRECT, acc_Y00);
        ACC_Y01 = aie::mul(CORRECT, acc_Y01);
        ACC_Y02 = aie::mul(CORRECT, acc_Y02);
        ACC_Y03 = aie::mul(CORRECT, acc_Y03);

        MMUL Y00(ACC_Y00);
        MMUL Y01(ACC_Y01);
        MMUL Y02(ACC_Y02);
        MMUL Y03(ACC_Y03);

        Y00.mac(SA, V00);
        Y01.mac(SA, V01);
        Y02.mac(SA, V02);
        Y03.mac(SA, V03);

        V00 = aie::load_v<MMUL::size_B>(pV01);
        V01 = aie::load_v<MMUL::size_B>(pV02);
        V02 = aie::load_v<MMUL::size_B>(pV03);
        V03 = aie::load_v<MMUL::size_B>(pV04);

        Y00.mac(SB, V00);
        Y01.mac(SB, V01);
        Y02.mac(SB, V02);
        Y03.mac(SB, V03);

        aie::store_v(pY1, Y00.template to_vector<float>());
        pY1 += MMUL::size_C;
        aie::store_v(pY1, Y01.template to_vector<float>());
        pY1 += MMUL::size_C;
        aie::store_v(pY1, Y02.template to_vector<float>());
        pY1 += MMUL::size_C;
        aie::store_v(pY1, Y03.template to_vector<float>());
        pY1 += MMUL::size_C;
    }
}

template <unsigned colQ, unsigned r, unsigned s, unsigned t>
void attn_fv(bf16 *__restrict pS, bf16 *__restrict pV, float *__restrict pY, float *__restrict c) {

    using MMUL = aie::mmul<r, s, t, bf16, bf16, accfloat>;

    event0();

    // plane 0: S0/S1 (pS, pS+64), V rows [0, colQ), c[0..7]
    attn_fv_plane<colQ, r, s, t>(pS, pV, pY, c);
    // plane 1: S2/S3 (pS+128, pS+192), V rows [colQ, 2*colQ), c[8..15]
    attn_fv_plane<colQ, r, s, t>(pS + 128, pV + MMUL::size_B * colQ,
                                 pY + MMUL::size_C * colQ, c + 8);

    event1();
}

#endif  // __chess__

template <unsigned dv>
void scale_div_aie(bf16 *o, float * a, float *l, bf16 * sink, float * m) {
    event0();
    constexpr int vec_factor = 64;
    const int F = 8 * dv / vec_factor;

    aie::vector<bf16, 16> sink_vec = aie::load_v<16>(sink);
    aie::accum<accfloat, 16> sink_acc;
    sink_acc.from_vector(sink_vec);
    aie::vector<float, 16> sink_float = sink_acc.template to_vector<float>();
    aie::vector<float, 16> m_vec = aie::load_v<16>(m);
    aie::vector<float, 16> sink_sub_vec = aie::sub(sink_float, m_vec);

    constexpr int min_clamp = -87.0f;
    constexpr int max_clamp = 88.0f;
    aie::vector<float, 16> min_value = aie::broadcast<float, 16>(min_clamp);
    aie::vector<float, 16> max_value = aie::broadcast<float, 16>(max_clamp);

    sink_sub_vec = aie::clamp(sink_sub_vec, min_value, max_value);
    aie::accum<accfloat, 16> sink_sub_acc;
    sink_sub_acc.from_vector(sink_sub_vec);
    sink_sub_acc = getExpBf16(sink_sub_acc.template to_vector<bf16>());
    aie::vector<float, 16> sink_sub_float = sink_sub_acc.template to_vector<float>();

    {

        float* pA1 = a;
        bf16* pO1 = o;
        bf16 l00 = (bf16)getInvBf16(l[0] + sink_sub_float.get(0));
        aie::vector<bf16, 8> L0 = aie::broadcast<bf16, 8>(l00);
        bf16 l01 = (bf16)getInvBf16(l[1] + sink_sub_float.get(1));
        aie::vector<bf16, 8> L1 = aie::broadcast<bf16, 8>(l01);
        bf16 l02 = (bf16)getInvBf16(l[2] + sink_sub_float.get(2));
        aie::vector<bf16, 8> L2 = aie::broadcast<bf16, 8>(l02);
        bf16 l03 = (bf16)getInvBf16(l[3] + sink_sub_float.get(3));
        aie::vector<bf16, 8> L3 = aie::broadcast<bf16, 8>(l03);
        bf16 l04 = (bf16)getInvBf16(l[4] + sink_sub_float.get(4));
        aie::vector<bf16, 8> L4 = aie::broadcast<bf16, 8>(l04);
        bf16 l05 = (bf16)getInvBf16(l[5] + sink_sub_float.get(5));
        aie::vector<bf16, 8> L5 = aie::broadcast<bf16, 8>(l05);
        bf16 l06 = (bf16)getInvBf16(l[6] + sink_sub_float.get(6));
        aie::vector<bf16, 8> L6 = aie::broadcast<bf16, 8>(l06);
        bf16 l07 = (bf16)getInvBf16(l[7] + sink_sub_float.get(7));
        aie::vector<bf16, 8> L7 = aie::broadcast<bf16, 8>(l07);
        auto L = aie::concat(L0, L1, L2, L3, L4, L5, L6, L7);

        for (int i = 0; i < F ; i++) {
            aie::vector<float, vec_factor> A0 = aie::load_v<vec_factor>(pA1);
            aie::accum<accfloat, vec_factor> A0_acc;
            A0_acc.from_vector(A0);
            aie::accum<accfloat, vec_factor> AL = aie::mul(A0_acc.template to_vector<bf16>(), L);
            aie::vector<bf16, vec_factor> AL_bf16 = AL.template to_vector<bf16>();
            aie::store_v(pO1, AL_bf16);
            pO1 += vec_factor;
            pA1 += vec_factor;
        }
    }

    {
        float* pA1 = a + 8 * dv;
        bf16* pO1 = o + 8 * dv;
        bf16 l00 = (bf16)getInvBf16(l[8] + sink_sub_float.get(8));
        aie::vector<bf16, 8> L0 = aie::broadcast<bf16, 8>(l00);
        bf16 l01 = (bf16)getInvBf16(l[9] + sink_sub_float.get(9));
        aie::vector<bf16, 8> L1 = aie::broadcast<bf16, 8>(l01);
        bf16 l02 = (bf16)getInvBf16(l[10] + sink_sub_float.get(10));
        aie::vector<bf16, 8> L2 = aie::broadcast<bf16, 8>(l02);
        bf16 l03 = (bf16)getInvBf16(l[11] + sink_sub_float.get(11));
        aie::vector<bf16, 8> L3 = aie::broadcast<bf16, 8>(l03);
        bf16 l04 = (bf16)getInvBf16(l[12] + sink_sub_float.get(12));
        aie::vector<bf16, 8> L4 = aie::broadcast<bf16, 8>(l04);
        bf16 l05 = (bf16)getInvBf16(l[13] + sink_sub_float.get(13));
        aie::vector<bf16, 8> L5 = aie::broadcast<bf16, 8>(l05);
        bf16 l06 = (bf16)getInvBf16(l[14] + sink_sub_float.get(14));
        aie::vector<bf16, 8> L6 = aie::broadcast<bf16, 8>(l06);
        bf16 l07 = (bf16)getInvBf16(l[15] + sink_sub_float.get(15));
        aie::vector<bf16, 8> L7 = aie::broadcast<bf16, 8>(l07);
        auto L = aie::concat(L0, L1, L2, L3, L4, L5, L6, L7);
        
        for (int i = 0; i < F ; i++) {
            aie::vector<float, vec_factor> A0 = aie::load_v<vec_factor>(pA1);
            aie::accum<accfloat, vec_factor> A0_acc;
            A0_acc.from_vector(A0);
            aie::accum<accfloat, vec_factor> AL = aie::mul(A0_acc.template to_vector<bf16>(), L);
            aie::vector<bf16, vec_factor> AL_bf16 = AL.template to_vector<bf16>();
            aie::store_v(pO1, AL_bf16);
            pO1 += vec_factor;
            pA1 += vec_factor;
        }
    }

    event1();
}
constexpr int s_prod_lock = 4;
constexpr int s_cons_lock = 5;
constexpr int v_prod_lock = 2;
constexpr int v_cons_lock = 3;
constexpr int o_prod_lock = 0;
constexpr int o_cons_lock = 1;  
constexpr int l_cons_lock = 7;

extern "C"{

void attn_kv(
    bf16 *__restrict s_ping,
    bf16 *__restrict s_pong,
    bf16 *__restrict v_ping,
    bf16 *__restrict v_pong,
    bf16 *__restrict o_ping,
    bf16 *__restrict o_pong,
    float * y,
    bf16 * sink,
    int* L
)
{
    alignas(aie::vector_decl_align) static float l[16] = {};
    alignas(aie::vector_decl_align) static float m_local[16] = {};
    const aie::vector<float, 16> zero = aie::broadcast<float, 16>(0);

    // Peano's CDO loader does not zero .bss; keep ping/pong selectors in .data.
#if !defined(__chess__)
    __attribute__((section(".data")))
#endif
    static bool is_svc_ping = false;
#if !defined(__chess__)
    __attribute__((section(".data")))
#endif
    static bool is_lo_ping = false;
    // Peano miscompiles read-modify-write on persisted .data bools across the loop;
    // keep ping/pong state in stack locals and sync back once at the end.
    // chess binds plain references so its generated code is unchanged.
#if defined(__chess__)
    bool &svc = is_svc_ping;
    bool &lo = is_lo_ping;
#else
    bool svc = is_svc_ping;
    bool lo = is_lo_ping;
#endif
    zero_vectorized<float, 16*DH>(y);
    aie::store_v(l, zero);
    _down_lock_acquire(l_cons_lock);
    AIE_LOCK_LOOP_CTR i = L[0];
    do {
        _down_lock_acquire(s_cons_lock);
        _lock_acquire(v_cons_lock);
        svc = !svc;
        bf16 *s = svc ? s_ping : s_pong;
        bf16 *v = svc ? v_ping : v_pong;
        float *c = (float *)(s + 16 * 16);
        float *m = (float *)(s + 16 * 16 + 32);
        reorder_s(s);
        calculate_l(s, c, l);
        attn_fv<DH / 8, 8, 8, 8>(s, v, y, c);
        aie::vector<float, 16> m_vec = aie::load_v<16>(m);
        aie::store_v(m_local, m_vec);
        _down_lock_release(s_prod_lock);
        _lock_release(v_prod_lock);
        i -= 16;
    } while (i > 0);
    lo = !lo;
    bf16 *o = lo ? o_ping : o_pong;
#if !defined(__chess__)
    is_svc_ping = svc;
    is_lo_ping = lo;
#endif
    _lock_acquire(o_prod_lock);
    scale_div_aie<DH>(o, y, l, sink, m_local);
    _lock_release(o_cons_lock);
}
}
