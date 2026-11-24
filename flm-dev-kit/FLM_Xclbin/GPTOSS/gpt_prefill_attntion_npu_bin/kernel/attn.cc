// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include <aie_api/aie.hpp>
#include "common.h"
#include "lut_based_ops.h"
#include <math.h>

alignas(aie::vector_decl_align) int idx_array[32] = {31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
template<uint32_t LQ>
void apply_mask_softmax(bf16*__restrict Inout, bf16* m, bf16* c, int llk, int lq, int l_current) {
    // aie::mask<NPU_N_TILE> submask;
    aie::vector<int, 32> idx_vec = aie::load_v<32>(idx_array);
    aie::vector<int, LK> l_vec = aie::broadcast<int, LK>(llk);
    l_vec = aie::add(l_vec, idx_vec);
    aie::vector<int, LK> r_vec = aie::broadcast<int, LK>(lq);
    r_vec = aie::add(r_vec, idx_vec);
    for (int i = 0; i < LQ; i++)
    chess_prepare_for_pipelining chess_loop_range(32, ){
        int current_idx = l_current + 31 - i;
        aie::mask<LK> mask_l = aie::gt(current_idx, l_vec);
        aie::mask<LK> mask_r = aie::le(current_idx, r_vec);
        aie::vector<bf16, LK> In_vec = aie::load_v<LK>(Inout);
        aie::vector<bf16, LK> s_vec = aie::select(bf16(-0x1.FEp127f), In_vec, mask_r);
        s_vec = aie::select(bf16(-0x1.FEp127f), s_vec, mask_l);
        bf16 vm = aie::reduce_max(s_vec);
        bf16 vmax = aie::max(vm, *(m + i));
        aie::vector<bf16, LK> Vec = aie::sub(s_vec, vmax);
        aie::accum<accfloat, LK> Vec_acc = aie::mul(Vec, exp_scale);
        Vec = aie::exp2<bf16>(Vec_acc.template to_vector<float>()); // f = exp(s - max)
        Vec = aie::select(bf16(0.0f), Vec, mask_l);
        Vec = aie::select(bf16(0.0f), Vec, mask_r);
        bf16 inner_correct = aie::sub(*(m + i), vmax); // correct = exp(m_left - max)
        inner_correct = aie::mul(inner_correct, exp_scale);
        aie::vector<bf16, LK> correct_vec = aie::broadcast<bf16, LK>(inner_correct);
        aie::accum<accfloat, LK> correct_acc;
        correct_acc.from_vector(correct_vec);
        correct_vec = aie::exp2<bf16>(correct_acc.template to_vector<float>());
        *(m + i) = (bf16)vmax;
        *(c + i) = (bf16)correct_vec.get(0);
        aie::store_v(Inout, Vec);
        Inout += LK;
    }
}

template<uint32_t LQ>
void reorder_s(bf16*__restrict pS) {
    for (int i = 0; i < LQ / 8; i++)
    chess_prepare_for_pipelining chess_loop_range(4, ){
        aie::vector<bf16, 64> S0 = aie::load_v<64>(pS);
        aie::vector<bf16, 32> S00 = aie::filter_even(S0, 16);
        aie::vector<bf16, 32> S01 = aie::filter_odd(S0, 16);
        aie::vector<bf16, 64> S1 = aie::load_v<64>(pS + 64);
        aie::vector<bf16, 32> S10 = aie::filter_even(S1, 16);
        aie::vector<bf16, 32> S11 = aie::filter_odd(S1, 16);
        aie::vector<bf16, 64> S2 = aie::load_v<64>(pS + 128);
        aie::vector<bf16, 32> S20 = aie::filter_even(S2, 16);
        aie::vector<bf16, 32> S21 = aie::filter_odd(S2, 16);
        aie::vector<bf16, 64> S3 = aie::load_v<64>(pS + 192);
        aie::vector<bf16, 32> S30 = aie::filter_even(S3, 16);
        aie::vector<bf16, 32> S31 = aie::filter_odd(S3, 16);

        aie::vector<bf16, 16> C0 = aie::filter_even(S00, 8);
        aie::vector<bf16, 16> C1 = aie::filter_odd(S00, 8);
        aie::vector<bf16, 16> C2 = aie::filter_even(S01, 8);
        aie::vector<bf16, 16> C3 = aie::filter_odd(S01, 8);
        aie::vector<bf16, 16> C4 = aie::filter_even(S10, 8);
        aie::vector<bf16, 16> C5 = aie::filter_odd(S10, 8);
        aie::vector<bf16, 16> C6 = aie::filter_even(S11, 8);
        aie::vector<bf16, 16> C7 = aie::filter_odd(S11, 8);
        aie::vector<bf16, 16> C8 = aie::filter_even(S20, 8);
        aie::vector<bf16, 16> C9 = aie::filter_odd(S20, 8);
        aie::vector<bf16, 16> C10 = aie::filter_even(S21, 8);
        aie::vector<bf16, 16> C11 = aie::filter_odd(S21, 8);
        aie::vector<bf16, 16> C12 = aie::filter_even(S30, 8);
        aie::vector<bf16, 16> C13 = aie::filter_odd(S30, 8);
        aie::vector<bf16, 16> C14 = aie::filter_even(S31, 8);
        aie::vector<bf16, 16> C15 = aie::filter_odd(S31, 8);

        S0 = aie::concat(C0, C4, C8, C12);
        S1 = aie::concat(C1, C5, C9, C13);
        S2 = aie::concat(C2, C6, C10, C14);
        S3 = aie::concat(C3, C7, C11, C15);

        aie::store_v(pS, S0);
        pS += 64;
        aie::store_v(pS, S1);
        pS += 64;
        aie::store_v(pS, S2);
        pS += 64;
        aie::store_v(pS, S3);
        pS += 64;
    }
}

template<int LQ, int DH>
void calculate_l_and_y(float* y, float* l, bf16*__restrict c, bf16*__restrict pS) {
    // calculate l
    using MMUL = aie::mmul<8, 8, 4, bf16, bf16, accauto>;

    aie::vector<bf16, 32> Ones = aie::broadcast<bf16, 32>(1.0f);
    for (int i = 0; i < LQ / 16; i++){
        aie::vector<bf16, 64> S0 = aie::load_v<64>(pS);
        pS += 64;
        aie::vector<bf16, 64> S1 = aie::load_v<64>(pS);
        pS += 64;
        aie::vector<bf16, 64> S2 = aie::load_v<64>(pS);
        pS += 64;
        aie::vector<bf16, 64> S3 = aie::load_v<64>(pS);
        pS += 64;
        aie::vector<bf16, 64> S4 = aie::load_v<64>(pS);
        pS += 64;
        aie::vector<bf16, 64> S5 = aie::load_v<64>(pS);
        pS += 64;
        aie::vector<bf16, 64> S6 = aie::load_v<64>(pS);
        pS += 64;
        aie::vector<bf16, 64> S7 = aie::load_v<64>(pS);
        pS += 64;

        // calculte l = correct * l_left + sum(f)
        aie::vector<bf16, MMUL::size_C> acc_C0 = aie::zeros<bf16, MMUL::size_C>(); 
        aie::vector<bf16, MMUL::size_C> acc_C1 = aie::zeros<bf16, MMUL::size_C>(); 

        MMUL C0(acc_C0);
        MMUL C1(acc_C1);

        C0.mac(S0, Ones);
        C0.mac(S1, Ones);
        C0.mac(S2, Ones);
        C0.mac(S3, Ones);
        C1.mac(S4, Ones);
        C1.mac(S5, Ones);
        C1.mac(S6, Ones);
        C1.mac(S7, Ones);

        auto sum0 = aie::filter_even(C0.template to_vector<float>(), 2);
        auto sum01 = aie::filter_even(sum0, 1);
        auto sum1 = aie::filter_even(C1.template to_vector<float>(), 2);
        auto sum11 = aie::filter_even(sum1, 1);
        auto sum2 = aie::concat(sum01, sum11);
        aie::accum<accfloat, 16> sum;
        sum.from_vector(sum2);

        aie::vector<bf16, 16> c_bf16 = aie::load_v<16>(c + i * 16);
        aie::accum<accfloat, 16> c_accf;
        c_accf.from_vector(c_bf16);
        aie::vector<float, 16> c_float32 = c_accf.template to_vector<float>();

        aie::vector<float, 16> l_float32 = aie::load_v<16>(l + i * 16);

        aie::accum<accfloat, 16> l_out;
        l_out = mac_elem_16_accuracy_safe(l_float32, c_float32, sum, 0, 0, 0);
        aie::store_v(l + i * 16, l_out.template to_vector<float>());
    }

    for (int i = 0; i < LQ / 8; i++)
    chess_prepare_for_pipelining chess_loop_range(4, ){
        aie::vector<float, 4> c0 = aie::broadcast<float, 4>(c[i * 8]);
        aie::vector<float, 4> c1 = aie::broadcast<float, 4>(c[i * 8 + 1]);
        aie::vector<float, 4> c2 = aie::broadcast<float, 4>(c[i * 8 + 2]);
        aie::vector<float, 4> c3 = aie::broadcast<float, 4>(c[i * 8 + 3]);   
        aie::vector<float, 4> c4 = aie::broadcast<float, 4>(c[i * 8 + 4]);
        aie::vector<float, 4> c5 = aie::broadcast<float, 4>(c[i * 8 + 5]);
        aie::vector<float, 4> c6 = aie::broadcast<float, 4>(c[i * 8 + 6]);
        aie::vector<float, 4> c7 = aie::broadcast<float, 4>(c[i * 8 + 7]);

        aie::vector<float, 32> CORRECT = aie::concat(c0, c1, c2, c3, c4, c5, c6, c7);

        float* pY1 = y + i * 8 * DH;

        for (unsigned j = 0; j < DH / 4; j += 1){
            aie::vector<float, MMUL::size_C> Y = aie::load_v<MMUL::size_C>(pY1);
            aie::accum<accfloat, MMUL::size_C> ACC_Y;
            ACC_Y = aie::mul(CORRECT, Y);
            aie::store_v(pY1, ACC_Y.template to_vector<float>());
            pY1 += MMUL::size_C;
        }
    }
}

template <unsigned int LQ>
void offset_l(float *l, bf16* m, bf16& sink_val){
    aie::vector<bf16, LQ> sinc_vec = aie::broadcast<bf16, LQ>(sink_val);
    aie::vector<bf16, LQ> f_vec = aie::sub(sinc_vec, aie::load_v<LQ>(m));
    aie::accum<accfloat, LQ> Vec_acc = aie::mul(f_vec, exp_scale);
    aie::vector<bf16, LQ> Vec = aie::exp2<bf16>(Vec_acc.template to_vector<float>()); // f = exp(s - max)
    aie::vector<float, LQ> l_vec = aie::load_v<LQ>(l);
    aie::accum<accfloat, LQ> l_acc_vec;
    l_acc_vec.from_vector(l_vec);
    l_acc_vec = aie::mac(l_acc_vec, Vec, aie::broadcast<bf16, LQ>(1.0f));
    l_vec = aie::inv(l_acc_vec.template to_vector<float>());
    aie::store_v(l, l_vec);
}

template <typename T_in, typename T_out, unsigned rowA, unsigned colA, unsigned colB, unsigned r, unsigned s, unsigned t>
void _attn_fv(T_out *__restrict pY, T_in *__restrict pS, T_in *__restrict pV) {
    using MMUL = aie::mmul<r, s, t, T_in, T_in, accauto>;

    // load S and reorder s from 16 x 16 to 8 x 8 x 4
    for (int i = 0; i < rowA; i += 2)
    chess_prepare_for_pipelining chess_loop_range(2, ){
        T_out *__restrict pY1 = pY + i * colB * MMUL::size_C;
        T_out *__restrict pY2 = pY + (i + 1) * colB * MMUL::size_C;

        for (unsigned j = 0; j < colB; j += 2)
        chess_prepare_for_pipelining chess_loop_range(8, ){
            T_in *__restrict pS1 = pS + i * MMUL::size_A * colA;
            T_in *__restrict pS2 = pS + (i + 1) * MMUL::size_A * colA;
            T_in *__restrict pV1 = pV + j * MMUL::size_B * colA;
            T_in *__restrict pV2 = pV + (j + 1) * MMUL::size_B * colA;

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

            for (unsigned k = 1; k < colA; k++){
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

#if defined(__chess__)
template <typename T_in, typename T_out, unsigned rowA, unsigned colA, unsigned colB, unsigned r, unsigned s, unsigned t>
void _attn_qk(T_out *__restrict pS, T_in *__restrict pQ, T_in *__restrict pK) {
    using MMUL = aie::mmul<r, s, t, T_in, T_in, accauto>;

    for (unsigned z = 0; z < rowA; z += 1)
    chess_prepare_for_pipelining chess_loop_range(4, ) {
        T_out *__restrict pS1 = pS + (z * colB + 0) * MMUL::size_C;

        for (unsigned j = 0; j < colB; j += 8)
        {
            const T_in *__restrict pQ1 = pQ + (z * colA + 0) * MMUL::size_A;
            const T_in *__restrict pK1 = pK + (0 * colB + j) * MMUL::size_B;
            const T_in *__restrict pK2 = pK + (0 * colB + (j + 1)) * MMUL::size_B;
            const T_in *__restrict pK3 = pK + (0 * colB + (j + 2)) * MMUL::size_B;
            const T_in *__restrict pK4 = pK + (0 * colB + (j + 3)) * MMUL::size_B;
            const T_in *__restrict pK5 = pK + (0 * colB + (j + 4)) * MMUL::size_B;
            const T_in *__restrict pK6 = pK + (0 * colB + (j + 5)) * MMUL::size_B;
            const T_in *__restrict pK7 = pK + (0 * colB + (j + 6)) * MMUL::size_B;
            const T_in *__restrict pK8 = pK + (0 * colB + (j + 7)) * MMUL::size_B;

            aie::vector<T_in, MMUL::size_A> Q0 = aie::load_v<MMUL::size_A>(pQ1);
            pQ1 += MMUL::size_A;
            aie::vector<T_in, MMUL::size_B> K00 = aie::load_v<MMUL::size_B>(pK1);
            aie::vector<T_in, MMUL::size_B> K0 = aie::transpose(K00, 4, 8);
            pK1 += MMUL::size_B * colB;
            aie::vector<T_in, MMUL::size_B> K01 = aie::load_v<MMUL::size_B>(pK2);
            aie::vector<T_in, MMUL::size_B> K1 = aie::transpose(K01, 4, 8);
            pK2 += MMUL::size_B * colB;
            aie::vector<T_in, MMUL::size_B> K02 = aie::load_v<MMUL::size_B>(pK3);
            aie::vector<T_in, MMUL::size_B> K2 = aie::transpose(K02, 4, 8);
            pK3 += MMUL::size_B * colB;
            aie::vector<T_in, MMUL::size_B> K03 = aie::load_v<MMUL::size_B>(pK4);
            aie::vector<T_in, MMUL::size_B> K3 = aie::transpose(K03, 4, 8);
            pK4 += MMUL::size_B * colB;
            aie::vector<T_in, MMUL::size_B> K04 = aie::load_v<MMUL::size_B>(pK5);
            aie::vector<T_in, MMUL::size_B> K4 = aie::transpose(K04, 4, 8);
            pK5 += MMUL::size_B * colB;
            aie::vector<T_in, MMUL::size_B> K05 = aie::load_v<MMUL::size_B>(pK6);
            aie::vector<T_in, MMUL::size_B> K5 = aie::transpose(K05, 4, 8);
            pK6 += MMUL::size_B * colB;
            aie::vector<T_in, MMUL::size_B> K06 = aie::load_v<MMUL::size_B>(pK7);
            aie::vector<T_in, MMUL::size_B> K6 = aie::transpose(K06, 4, 8);
            pK7 += MMUL::size_B * colB;
            aie::vector<T_in, MMUL::size_B> K07 = aie::load_v<MMUL::size_B>(pK8);
            aie::vector<T_in, MMUL::size_B> K7 = aie::transpose(K07, 4, 8);
            pK8 += MMUL::size_B * colB;

            aie::vector<T_out, MMUL::size_C> acc_C00 = aie::zeros<T_out, MMUL::size_C>(); 
            aie::vector<T_out, MMUL::size_C> acc_C01 = aie::zeros<T_out, MMUL::size_C>();
            aie::vector<T_out, MMUL::size_C> acc_C02 = aie::zeros<T_out, MMUL::size_C>();
            aie::vector<T_out, MMUL::size_C> acc_C03 = aie::zeros<T_out, MMUL::size_C>();
            aie::vector<T_out, MMUL::size_C> acc_C04 = aie::zeros<T_out, MMUL::size_C>();
            aie::vector<T_out, MMUL::size_C> acc_C05 = aie::zeros<T_out, MMUL::size_C>();
            aie::vector<T_out, MMUL::size_C> acc_C06 = aie::zeros<T_out, MMUL::size_C>();
            aie::vector<T_out, MMUL::size_C> acc_C07 = aie::zeros<T_out, MMUL::size_C>();

            MMUL C00(acc_C00);
            MMUL C01(acc_C01);
            MMUL C02(acc_C02);
            MMUL C03(acc_C03);
            MMUL C04(acc_C04);
            MMUL C05(acc_C05);
            MMUL C06(acc_C06);
            MMUL C07(acc_C07);

            C00.mac(Q0, K0);
            C01.mac(Q0, K1);
            C02.mac(Q0, K2);
            C03.mac(Q0, K3);
            C04.mac(Q0, K4);
            C05.mac(Q0, K5);
            C06.mac(Q0, K6);
            C07.mac(Q0, K7);

            for (unsigned i = 1; i < colA; ++i)
            {
                Q0 = aie::load_v<MMUL::size_A>(pQ1);
                pQ1 += MMUL::size_A;
                K00 = aie::load_v<MMUL::size_B>(pK1);
                K0 = aie::transpose(K00, 4, 8);
                pK1 += MMUL::size_B * colB;
                K01 = aie::load_v<MMUL::size_B>(pK2);
                K1 = aie::transpose(K01, 4, 8);
                pK2 += MMUL::size_B * colB;
                K02 = aie::load_v<MMUL::size_B>(pK3);
                K2 = aie::transpose(K02, 4, 8);
                pK3 += MMUL::size_B * colB;
                K03 = aie::load_v<MMUL::size_B>(pK4);
                K3 = aie::transpose(K03, 4, 8);
                pK4 += MMUL::size_B * colB;
                K04 = aie::load_v<MMUL::size_B>(pK5);
                K4 = aie::transpose(K04, 4, 8);
                pK5 += MMUL::size_B * colB;
                K05 = aie::load_v<MMUL::size_B>(pK6);
                K5 = aie::transpose(K05, 4, 8);
                pK6 += MMUL::size_B * colB;
                K06 = aie::load_v<MMUL::size_B>(pK7);
                K6 = aie::transpose(K06, 4, 8);
                pK7 += MMUL::size_B * colB;
                K07 = aie::load_v<MMUL::size_B>(pK8);
                K7 = aie::transpose(K07, 4, 8);
                pK8 += MMUL::size_B * colB;

                C00.mac(Q0, K0);
                C01.mac(Q0, K1);
                C02.mac(Q0, K2);
                C03.mac(Q0, K3);
                C04.mac(Q0, K4);
                C05.mac(Q0, K5);
                C06.mac(Q0, K6);
                C07.mac(Q0, K7);
            }

            auto mout0 = aie::interleave_zip(C00.template to_vector<T_out>(), C01.template to_vector<T_out>(), 4);
            auto mout1 = aie::interleave_zip(C02.template to_vector<T_out>(), C03.template to_vector<T_out>(), 4);
            auto mout2= aie::interleave_zip(C04.template to_vector<T_out>(), C05.template to_vector<T_out>(), 4);
            auto mout3= aie::interleave_zip(C06.template to_vector<T_out>(), C07.template to_vector<T_out>(), 4);

            auto mout00 = aie::interleave_zip(mout0.first, mout1.first, 8);
            auto mout01 = aie::interleave_zip(mout0.second, mout1.second, 8);
            auto mout10 = aie::interleave_zip(mout2.first, mout3.first, 8);
            auto mout11 = aie::interleave_zip(mout2.second, mout3.second, 8);
            
            auto mout20 = aie::interleave_zip(mout00.first, mout10.first, 16);
            auto mout21 = aie::interleave_zip(mout00.second, mout10.second, 16);
            auto mout30 = aie::interleave_zip(mout01.first, mout11.first, 16);
            auto mout31 = aie::interleave_zip(mout01.second, mout11.second, 16);

            auto out0 = aie::mul(mout20.first, (T_out)scale).template to_vector<T_out>(0);
            auto out1 = aie::mul(mout20.second, (T_out)scale).template to_vector<T_out>(0);
            auto out2 = aie::mul(mout21.first, (T_out)scale).template to_vector<T_out>(0);
            auto out3 = aie::mul(mout21.second, (T_out)scale).template to_vector<T_out>(0);
            auto out4 = aie::mul(mout30.first, (T_out)scale).template to_vector<T_out>(0);
            auto out5 = aie::mul(mout30.second, (T_out)scale).template to_vector<T_out>(0);
            auto out6 = aie::mul(mout31.first, (T_out)scale).template to_vector<T_out>(0);
            auto out7 = aie::mul(mout31.second, (T_out)scale).template to_vector<T_out>(0);

            aie::store_v(pS1, out0);
            pS1 += MMUL::size_C;
            aie::store_v(pS1, out1);
            pS1 += MMUL::size_C;
            aie::store_v(pS1, out2);
            pS1 += MMUL::size_C;
            aie::store_v(pS1, out3);
            pS1 += MMUL::size_C;
            aie::store_v(pS1, out4);
            pS1 += MMUL::size_C;
            aie::store_v(pS1, out5);
            pS1 += MMUL::size_C;
            aie::store_v(pS1, out6);
            pS1 += MMUL::size_C;
            aie::store_v(pS1, out7);
            pS1 += MMUL::size_C;
        }
    }
}
#else
template <typename T_in, typename T_out, unsigned rowA, unsigned colA, unsigned colB, unsigned r, unsigned s, unsigned t>
__attribute__((noinline))
void _attn_qk(T_out *__restrict pS, T_in *__restrict pQ, T_in *__restrict pK) {
    using MMUL = aie::mmul<r, s, t, T_in, T_in, accauto>;
    // Peano: not inlined and no peeled first k-step, so the call chain fits the
    // stack. The macs are the same as in the chess version, in the same order.

    for (unsigned z = 0; z < rowA; z += 1)
    chess_prepare_for_pipelining chess_loop_range(4, ) {
        T_out *__restrict pS1 = pS + (z * colB + 0) * MMUL::size_C;

        for (unsigned j = 0; j < colB; j += 8)
        {
            const T_in *__restrict pQ1 = pQ + (z * colA + 0) * MMUL::size_A;
            const T_in *__restrict pK1 = pK + (0 * colB + j) * MMUL::size_B;
            const T_in *__restrict pK2 = pK + (0 * colB + (j + 1)) * MMUL::size_B;
            const T_in *__restrict pK3 = pK + (0 * colB + (j + 2)) * MMUL::size_B;
            const T_in *__restrict pK4 = pK + (0 * colB + (j + 3)) * MMUL::size_B;
            const T_in *__restrict pK5 = pK + (0 * colB + (j + 4)) * MMUL::size_B;
            const T_in *__restrict pK6 = pK + (0 * colB + (j + 5)) * MMUL::size_B;
            const T_in *__restrict pK7 = pK + (0 * colB + (j + 6)) * MMUL::size_B;
            const T_in *__restrict pK8 = pK + (0 * colB + (j + 7)) * MMUL::size_B;

            aie::vector<T_out, MMUL::size_C> acc_C00 = aie::zeros<T_out, MMUL::size_C>(); 
            aie::vector<T_out, MMUL::size_C> acc_C01 = aie::zeros<T_out, MMUL::size_C>();
            aie::vector<T_out, MMUL::size_C> acc_C02 = aie::zeros<T_out, MMUL::size_C>();
            aie::vector<T_out, MMUL::size_C> acc_C03 = aie::zeros<T_out, MMUL::size_C>();
            aie::vector<T_out, MMUL::size_C> acc_C04 = aie::zeros<T_out, MMUL::size_C>();
            aie::vector<T_out, MMUL::size_C> acc_C05 = aie::zeros<T_out, MMUL::size_C>();
            aie::vector<T_out, MMUL::size_C> acc_C06 = aie::zeros<T_out, MMUL::size_C>();
            aie::vector<T_out, MMUL::size_C> acc_C07 = aie::zeros<T_out, MMUL::size_C>();

            MMUL C00(acc_C00);
            MMUL C01(acc_C01);
            MMUL C02(acc_C02);
            MMUL C03(acc_C03);
            MMUL C04(acc_C04);
            MMUL C05(acc_C05);
            MMUL C06(acc_C06);
            MMUL C07(acc_C07);

            for (unsigned i = 0; i < colA; ++i)
            {
                aie::vector<T_in, MMUL::size_A> Q0 = aie::load_v<MMUL::size_A>(pQ1);
                pQ1 += MMUL::size_A;
                aie::vector<T_in, MMUL::size_B> K00 = aie::load_v<MMUL::size_B>(pK1);
                aie::vector<T_in, MMUL::size_B> K0 = aie::transpose(K00, 4, 8);
                pK1 += MMUL::size_B * colB;
                aie::vector<T_in, MMUL::size_B> K01 = aie::load_v<MMUL::size_B>(pK2);
                aie::vector<T_in, MMUL::size_B> K1 = aie::transpose(K01, 4, 8);
                pK2 += MMUL::size_B * colB;
                aie::vector<T_in, MMUL::size_B> K02 = aie::load_v<MMUL::size_B>(pK3);
                aie::vector<T_in, MMUL::size_B> K2 = aie::transpose(K02, 4, 8);
                pK3 += MMUL::size_B * colB;
                aie::vector<T_in, MMUL::size_B> K03 = aie::load_v<MMUL::size_B>(pK4);
                aie::vector<T_in, MMUL::size_B> K3 = aie::transpose(K03, 4, 8);
                pK4 += MMUL::size_B * colB;
                aie::vector<T_in, MMUL::size_B> K04 = aie::load_v<MMUL::size_B>(pK5);
                aie::vector<T_in, MMUL::size_B> K4 = aie::transpose(K04, 4, 8);
                pK5 += MMUL::size_B * colB;
                aie::vector<T_in, MMUL::size_B> K05 = aie::load_v<MMUL::size_B>(pK6);
                aie::vector<T_in, MMUL::size_B> K5 = aie::transpose(K05, 4, 8);
                pK6 += MMUL::size_B * colB;
                aie::vector<T_in, MMUL::size_B> K06 = aie::load_v<MMUL::size_B>(pK7);
                aie::vector<T_in, MMUL::size_B> K6 = aie::transpose(K06, 4, 8);
                pK7 += MMUL::size_B * colB;
                aie::vector<T_in, MMUL::size_B> K07 = aie::load_v<MMUL::size_B>(pK8);
                aie::vector<T_in, MMUL::size_B> K7 = aie::transpose(K07, 4, 8);
                pK8 += MMUL::size_B * colB;

                C00.mac(Q0, K0);
                C01.mac(Q0, K1);
                C02.mac(Q0, K2);
                C03.mac(Q0, K3);
                C04.mac(Q0, K4);
                C05.mac(Q0, K5);
                C06.mac(Q0, K6);
                C07.mac(Q0, K7);
            }

            auto mout0 = aie::interleave_zip(C00.template to_vector<T_out>(), C01.template to_vector<T_out>(), 4);
            auto mout1 = aie::interleave_zip(C02.template to_vector<T_out>(), C03.template to_vector<T_out>(), 4);
            auto mout2= aie::interleave_zip(C04.template to_vector<T_out>(), C05.template to_vector<T_out>(), 4);
            auto mout3= aie::interleave_zip(C06.template to_vector<T_out>(), C07.template to_vector<T_out>(), 4);

            auto mout00 = aie::interleave_zip(mout0.first, mout1.first, 8);
            auto mout01 = aie::interleave_zip(mout0.second, mout1.second, 8);
            auto mout10 = aie::interleave_zip(mout2.first, mout3.first, 8);
            auto mout11 = aie::interleave_zip(mout2.second, mout3.second, 8);
            
            auto mout20 = aie::interleave_zip(mout00.first, mout10.first, 16);
            auto mout21 = aie::interleave_zip(mout00.second, mout10.second, 16);
            auto mout30 = aie::interleave_zip(mout01.first, mout11.first, 16);
            auto mout31 = aie::interleave_zip(mout01.second, mout11.second, 16);

            auto out0 = aie::mul(mout20.first, (T_out)scale).template to_vector<T_out>(0);
            auto out1 = aie::mul(mout20.second, (T_out)scale).template to_vector<T_out>(0);
            auto out2 = aie::mul(mout21.first, (T_out)scale).template to_vector<T_out>(0);
            auto out3 = aie::mul(mout21.second, (T_out)scale).template to_vector<T_out>(0);
            auto out4 = aie::mul(mout30.first, (T_out)scale).template to_vector<T_out>(0);
            auto out5 = aie::mul(mout30.second, (T_out)scale).template to_vector<T_out>(0);
            auto out6 = aie::mul(mout31.first, (T_out)scale).template to_vector<T_out>(0);
            auto out7 = aie::mul(mout31.second, (T_out)scale).template to_vector<T_out>(0);

            aie::store_v(pS1, out0);
            pS1 += MMUL::size_C;
            aie::store_v(pS1, out1);
            pS1 += MMUL::size_C;
            aie::store_v(pS1, out2);
            pS1 += MMUL::size_C;
            aie::store_v(pS1, out3);
            pS1 += MMUL::size_C;
            aie::store_v(pS1, out4);
            pS1 += MMUL::size_C;
            aie::store_v(pS1, out5);
            pS1 += MMUL::size_C;
            aie::store_v(pS1, out6);
            pS1 += MMUL::size_C;
            aie::store_v(pS1, out7);
            pS1 += MMUL::size_C;
        }
    }
}
#endif

template <unsigned DH, unsigned LQ>
void scale_div_aie(bf16* o, float *l, float *y) {
    constexpr int vec_factor = 32;

    for (int i = 0; i < LQ / 8; i++){
        // organize l
        aie::vector<float, 4> L0 = aie::broadcast<float, 4>(l[i * 8]);
        aie::vector<float, 4> L1 = aie::broadcast<float, 4>(l[i * 8 + 1]);
        aie::vector<float, 4> L2 = aie::broadcast<float, 4>(l[i * 8 + 2]);
        aie::vector<float, 4> L3 = aie::broadcast<float, 4>(l[i * 8 + 3]);
        aie::vector<float, 4> L4 = aie::broadcast<float, 4>(l[i * 8 + 4]);
        aie::vector<float, 4> L5 = aie::broadcast<float, 4>(l[i * 8 + 5]);
        aie::vector<float, 4> L6 = aie::broadcast<float, 4>(l[i * 8 + 6]);
        aie::vector<float, 4> L7 = aie::broadcast<float, 4>(l[i * 8 + 7]);

        auto LL00 = aie::concat(L0, L1, L2, L3, L4, L5, L6, L7);

        bf16* pO1 = o + i * 8 * DH;
        float* pY1 = y + i * 8 * DH;

        for (int j = 0; j < DH / 4; j += 1) {
            aie::vector<float, vec_factor> Y00 = aie::load_v<vec_factor>(pY1);
            pY1 += vec_factor;
            aie::accum<accfloat, vec_factor> Y00_acc;
            Y00_acc.from_vector(Y00);
            aie::accum<accfloat, vec_factor> AL00 = aie::mul(Y00_acc.template to_vector<float>(), LL00);
            aie::store_v(pO1, AL00.template to_vector<bf16>());
            pO1 += vec_factor;
        }
    }
}

constexpr int q_prod_lock = 0;
constexpr int q_cons_lock = 1;
constexpr int in_prod_lock = 2;
constexpr int in_cons_lock = 3;
constexpr int o_prod_lock = 4;
constexpr int o_cons_lock = 5;

extern "C" {
    void attn(
        bf16 *__restrict o_ping,
        bf16 *__restrict o_pong,
        bf16 *__restrict q_ping,
        bf16 *__restrict q_pong,
        bf16 *__restrict in_ping,
        bf16 *__restrict in_pong,
        bf16 *__restrict q,
        bf16 * s,
        float * y,
        int * L_begin_buffer,
        int * L_end_buffer,
        float * sinks,
        int * window_size_buffer,
        const int row
    ){
        AIE_PERSIST_ZERO static bool is_kv_ping = false;
        AIE_PERSIST_ZERO static bool is_q_ping = false;
        AIE_PERSIST_ZERO static bool is_o_ping = false;

        alignas(aie::vector_decl_align) bf16 m[LQ];
        alignas(aie::vector_decl_align) bf16 c[LQ];
        alignas(aie::vector_decl_align) float l[LQ];

        AIE_TRANSIENT const aie::vector<bf16, LQ> one = aie::broadcast<bf16, LQ>(1.0);

        int i = 0;
        int offset;
        int L_div_l;
        int lq = 0;
        int lk = 0;
        int llk = 0;
        int block_location = 0;
        int l_current = 0;
        bf16 sink_value;
        do{
            aie::store_v(c, one);
            zero_vectorized<float, LQ>(l);
            zero_vectorized<float, LQ * DH>(y);
            for (int q_row = 0; q_row < LQ / 8; q_row++){
                for (int ll = 0; ll < 8; ll++){
                    _lock_acquire(q_cons_lock);
                    is_q_ping = !is_q_ping;
                    bf16 *q_in = is_q_ping ? (q_ping + row * DH) : (q_pong + row * DH);
                    bf16 *q_out = q + q_row * 8 * DH + ll * 8;
                    for (int lc = 0; lc < DH / 8; lc++){
                        aie::vector<bf16, 8> q_slice = aie::load_v<8>(q_in);
                        aie::store_v(q_out, q_slice);
                        q_out += 64;
                        q_in += 8;
                    }
                    _lock_release(q_prod_lock);
                }
            }

            sink_value = (bf16)sinks[0];
            aie::vector<bf16, LQ> sink_vector = aie::broadcast<bf16, LQ>(sink_value);
            aie::store_v(m, sink_vector);

            offset = (L_begin_buffer[0] >> 5);
            // determine lq
            lq = L_begin_buffer[0] + i * 32;

            // determine lk
            lk = lq - window_size_buffer[0];
            lk = lk > 0 ? lk : 0;

            // determine llk
            llk = lq - window_size_buffer[0];
            llk = llk > -32 ? llk : -32;

            // apply softmax
            for (int j = 0; j < ((lq - lk) / 32 + 1); j++){
                block_location = j;
                l_current = lk + block_location * 32;
                _lock_acquire(in_cons_lock);
                is_kv_ping = !is_kv_ping;
                bf16 *k = is_kv_ping ? in_ping : in_pong;
                _attn_qk<bf16, bf16, LQ/8, DH/8, LK/4, 8, 8, 4>(s, q, k);
                _lock_release(in_prod_lock);
                apply_mask_softmax<LQ>(s, m, c, llk, lq, l_current);
                reorder_s<LQ>(s);
                calculate_l_and_y<LQ, DH>(y, l, c, s);
                _lock_acquire(in_cons_lock);
                is_kv_ping = !is_kv_ping;
                bf16 *v = is_kv_ping ? in_ping : in_pong;
                _attn_fv<bf16, float, LQ/8, LK/8, DH/4, 8, 8, 4>(y, s, v);
                _lock_release(in_prod_lock);
            }

            _lock_acquire(o_prod_lock);
            is_o_ping = !is_o_ping;
            bf16 *o = is_o_ping ? o_ping : o_pong;
            offset_l<LQ>(l, m, sink_value);
            scale_div_aie<DH, LQ>(o, l, y);
            _lock_release(o_cons_lock);
            i++;
            L_div_l = L_end_buffer[0] >> 5;
            L_div_l -= offset;
        } while (i < L_div_l);
    }
}