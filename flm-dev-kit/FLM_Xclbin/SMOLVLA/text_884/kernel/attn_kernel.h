// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef ATTN_KERNEL_H
#define ATTN_KERNEL_H
#include <aie_api/aie.hpp>
#include "common.h"
#include <math.h>

template <typename T_in, typename T_out>
void passThrough_aie(T_in *restrict in, T_out *restrict out) {
    const int vec_factor = 64;

    aie::accum<accfloat, vec_factor> c_acc_in;
    c_acc_in.from_vector(aie::load_v<vec_factor>(in));
    aie::store_v(out, c_acc_in.template to_vector<T_out>());
}

void get_max(bf16*__restrict new_m, bf16*__restrict S, bf16*__restrict pre_m){
    for (int i = 0; i < lq / 8; i++){
        // vertical: element-wise max across this query-tile's key-tiles
        aie::vector<bf16, 64> run = aie::load_v<64>(S + i * 8 * lk);
        for (int j = 1; j < lk / 8; j++){
            aie::vector<bf16, 64> ns_vec = aie::load_v<64>(S + i * 8 * lk + j * 64);
            run = aie::max(run, ns_vec);
        }
        // horizontal: reduce each row's 8 lanes, fold in pre_m, broadcast into m
        for (int k = 0; k < 8; k++){
            int q = i * 8 + k;
            aie::vector<bf16,8> r8 = run.extract<8>(k);
            float mx = (float)pre_m[q];
            for (int e = 0; e < 8; e++) mx = (float)r8.get(e) > mx ? (float)r8.get(e) : mx;
            new_m[q] = (bf16)mx;
        }
    }
}

void apply_softmax(
    bf16*__restrict S,
    bf16*__restrict m
){
    const aie::vector<bf16, 64> zero = aie::broadcast<bf16, 64>(bf16(0.0f));
    for (int qt = 0; qt < lq / 8; qt++){
        for (int kt = 0; kt < lk / 8; kt++){
            for (int k = 0; k < 8; k++){   
                bf16 * pS = S + qt * 512 + kt * 64 + k * 8;
                aie::vector<bf16, 8> s_vec = aie::load_v<8>(pS);
                aie::vector<bf16, 8> Vec = aie::sub(s_vec, *(m + qt * 8 + k));
                aie::accum<accfloat, 8> Vec_acc = aie::mul(Vec, exp_scale);
                Vec = aie::exp2<bf16>(Vec_acc.template to_vector<float>()); // f = exp(s - max)
                aie::store_v(pS, Vec);
            }
        }
    }
}

void calculate_c(float* c, bf16* prev_m_local, bf16* new_m_local) {
    for (int i = 0; i < lq; i++){
        // correct = exp(m_left - max)
        bf16 inner_correct = aie::sub(*(prev_m_local + i), *(new_m_local + i)); // correct = exp(m_left - max)
        inner_correct = aie::mul(inner_correct, exp_scale);
        aie::vector<bf16, lk> correct_vec = aie::broadcast<bf16, lk>(inner_correct);
        aie::accum<accfloat, lk> correct_acc;
        correct_acc.from_vector(correct_vec);
        correct_vec = aie::exp2<bf16>(correct_acc.template to_vector<float>());
        *(c + i) = (float)correct_vec.get(0);
    }
}

void calculate_l(float* l, float*__restrict c, bf16*__restrict pS) {
    // calculate l
    using MMUL = aie::mmul<8, 8, 8, bf16, bf16, accauto>;

    // calculte l = correct * l_left + sum(f)
    aie::vector<bf16, 64> Ones = aie::broadcast<bf16, 64>(1.0f);
    aie::vector<bf16, MMUL::size_C> acc_C0 = aie::zeros<bf16, MMUL::size_C>(); 
    aie::vector<bf16, MMUL::size_C> acc_C1 = aie::zeros<bf16, MMUL::size_C>(); 
    aie::vector<bf16, MMUL::size_C> acc_C2 = aie::zeros<bf16, MMUL::size_C>(); 
    aie::vector<bf16, MMUL::size_C> acc_C3 = aie::zeros<bf16, MMUL::size_C>(); 
    aie::vector<bf16, MMUL::size_C> acc_C4 = aie::zeros<bf16, MMUL::size_C>(); 
    aie::vector<bf16, MMUL::size_C> acc_C5 = aie::zeros<bf16, MMUL::size_C>();
    aie::vector<bf16, MMUL::size_C> acc_C6 = aie::zeros<bf16, MMUL::size_C>();  
    aie::vector<bf16, MMUL::size_C> acc_C7 = aie::zeros<bf16, MMUL::size_C>(); 
    MMUL C0(acc_C0);
    MMUL C1(acc_C1);
    MMUL C2(acc_C2);
    MMUL C3(acc_C3);
    MMUL C4(acc_C4);
    MMUL C5(acc_C5);
    MMUL C6(acc_C6);
    MMUL C7(acc_C7);

    bf16 * pS1 = pS;
    bf16 * pS2 = pS + 8 * lk;
    bf16 * pS3 = pS + 2 * 8 * lk;
    bf16 * pS4 = pS + 3 * 8 * lk;
    bf16 * pS5 = pS + 4 * 8 * lk;
    bf16 * pS6 = pS + 5 * 8 * lk;
    bf16 * pS7 = pS + 6 * 8 * lk;
    bf16 * pS8 = pS + 7 * 8 * lk;

    for (int i = 0; i < lk / 8; i++){
        aie::vector<bf16, 64> S0 = aie::load_v<64>(pS1);
        aie::vector<bf16, 64> S1 = aie::load_v<64>(pS2);
        aie::vector<bf16, 64> S2 = aie::load_v<64>(pS3);
        aie::vector<bf16, 64> S3 = aie::load_v<64>(pS4);
        aie::vector<bf16, 64> S4 = aie::load_v<64>(pS5);
        aie::vector<bf16, 64> S5 = aie::load_v<64>(pS6);
        aie::vector<bf16, 64> S6 = aie::load_v<64>(pS7);
        aie::vector<bf16, 64> S7 = aie::load_v<64>(pS8);

        C0.mac(S0, Ones);
        C1.mac(S1, Ones);
        C2.mac(S2, Ones);
        C3.mac(S3, Ones);
        C4.mac(S4, Ones);
        C5.mac(S5, Ones);
        C6.mac(S6, Ones);
        C7.mac(S7, Ones);

        pS1 += 64;
        pS2 += 64;
        pS3 += 64;
        pS4 += 64;
        pS5 += 64;
        pS6 += 64;
        pS7 += 64;
        pS8 += 64;
    }

    auto sum0 = aie::filter_even(C0.template to_vector<float>(), 4);
    auto sum00 = aie::filter_even(sum0, 2);
    auto sum01 = aie::filter_even(sum00, 1);
    auto sum1 = aie::filter_even(C1.template to_vector<float>(), 4);
    auto sum10 = aie::filter_even(sum1, 2);
    auto sum11 = aie::filter_even(sum10, 1);
    auto sum2 = aie::filter_even(C2.template to_vector<float>(), 4);
    auto sum20 = aie::filter_even(sum2, 2);
    auto sum21 = aie::filter_even(sum20, 1);
    auto sum3 = aie::filter_even(C3.template to_vector<float>(), 4);
    auto sum30 = aie::filter_even(sum3, 2);
    auto sum31 = aie::filter_even(sum30, 1);
    auto sum4 = aie::filter_even(C4.template to_vector<float>(), 4);
    auto sum40 = aie::filter_even(sum4, 2);
    auto sum41 = aie::filter_even(sum40, 1);
    auto sum5 = aie::filter_even(C5.template to_vector<float>(), 4);
    auto sum50 = aie::filter_even(sum5, 2);
    auto sum51 = aie::filter_even(sum50, 1);
    auto sum6 = aie::filter_even(C6.template to_vector<float>(), 4);
    auto sum60 = aie::filter_even(sum6, 2);
    auto sum61 = aie::filter_even(sum60, 1);
    auto sum7 = aie::filter_even(C7.template to_vector<float>(), 4);
    auto sum70 = aie::filter_even(sum7, 2);
    auto sum71 = aie::filter_even(sum70, 1);
    auto sum_vec0 = aie::concat(sum01, sum11, sum21, sum31);
    auto sum_vec1 = aie::concat(sum41, sum51, sum61, sum71);

    aie::accum<accfloat, 32> sum_acc0;
    sum_acc0.from_vector(sum_vec0);
    aie::accum<accfloat, 32> sum_acc1;
    sum_acc1.from_vector(sum_vec1);

    aie::vector<float, 32> c_vec0 = aie::load_v<32>(c);
    aie::vector<float, 32> c_vec1 = aie::load_v<32>(c + 32);
    aie::vector<float, 32> l_vec0 = aie::load_v<32>(l);
    aie::vector<float, 32> l_vec1 = aie::load_v<32>(l + 32);

    l_vec0 = aie::mac(sum_acc0, l_vec0, c_vec0);
    l_vec1 = aie::mac(sum_acc1, l_vec1, c_vec1);
    aie::store_v(l, l_vec0);
    aie::store_v(l + 32, l_vec1);
}

void calculate_y(float* y, float* c) {
    // calculate y
    for (int i = 0; i < lq / 8; i++)
    chess_prepare_for_pipelining chess_loop_range(4, ) {
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

template <typename T_in, typename T_out, unsigned r, unsigned s, unsigned t>
void _attn_fv(T_out *__restrict pY, T_in *__restrict pS, T_in *__restrict pV) {
    using MMUL = aie::mmul<r, s, t, T_in, T_in, accauto>;

    // Process query-tiles in groups of 4 (rows 0-31, then 32-63). The inner body
    // handles one group of 4 query-tiles across all output features and key-tiles.
    for (unsigned qg = 0; qg < lq / 8; qg += 4){
        T_out *__restrict pY1 = pY + qg * 8 * dh;
        T_out *__restrict pY2 = pY + (qg + 1) * 8 * dh;
        T_out *__restrict pY3 = pY + (qg + 2) * 8 * dh;
        T_out *__restrict pY4 = pY + (qg + 3) * 8 * dh;


        for (unsigned j = 0; j < dh / 8; j += 2)
        {
            T_in *__restrict pS1 = pS + qg * 8 * lk;
            T_in *__restrict pS2 = pS + (qg + 1) * 8 * lk;
            T_in *__restrict pS3 = pS + (qg + 2) * 8 * lk;
            T_in *__restrict pS4 = pS + (qg + 3) * 8 * lk;
            T_in *__restrict pV1 = pV + j * 8 * lk;
            T_in *__restrict pV2 = pV + (j + 1) * 8 * lk;

            aie::vector<T_in, MMUL::size_A> S0 = aie::load_v<MMUL::size_A>(pS1);
            pS1 += MMUL::size_A;
            aie::vector<T_in, MMUL::size_A> S1 = aie::load_v<MMUL::size_A>(pS2);
            pS2 += MMUL::size_A;
            aie::vector<T_in, MMUL::size_A> S2 = aie::load_v<MMUL::size_A>(pS3);
            pS3 += MMUL::size_A;
            aie::vector<T_in, MMUL::size_A> S3 = aie::load_v<MMUL::size_A>(pS4);
            pS4 += MMUL::size_A;

            aie::vector<T_in, MMUL::size_B> V0 = aie::load_v<MMUL::size_B>(pV1);
            pV1 += MMUL::size_B;
            aie::vector<T_in, MMUL::size_B> V1 = aie::load_v<MMUL::size_B>(pV2);
            pV2 += MMUL::size_B;

            aie::vector<T_out, MMUL::size_C> acc_Y00 = aie::load_v<MMUL::size_C>(pY1);
            aie::vector<T_out, MMUL::size_C> acc_Y01 = aie::load_v<MMUL::size_C>(pY1 + MMUL::size_C);
            aie::vector<T_out, MMUL::size_C> acc_Y10 = aie::load_v<MMUL::size_C>(pY2);
            aie::vector<T_out, MMUL::size_C> acc_Y11 = aie::load_v<MMUL::size_C>(pY2 + MMUL::size_C);
            aie::vector<T_out, MMUL::size_C> acc_Y20 = aie::load_v<MMUL::size_C>(pY3);
            aie::vector<T_out, MMUL::size_C> acc_Y21 = aie::load_v<MMUL::size_C>(pY3 + MMUL::size_C);
            aie::vector<T_out, MMUL::size_C> acc_Y30 = aie::load_v<MMUL::size_C>(pY4);
            aie::vector<T_out, MMUL::size_C> acc_Y31 = aie::load_v<MMUL::size_C>(pY4 + MMUL::size_C);

            MMUL Y00(acc_Y00);
            MMUL Y01(acc_Y01);
            MMUL Y10(acc_Y10);
            MMUL Y11(acc_Y11);
            MMUL Y20(acc_Y20);
            MMUL Y21(acc_Y21);
            MMUL Y30(acc_Y30);
            MMUL Y31(acc_Y31);

            Y00.mac(S0, V0);
            Y01.mac(S0, V1);
            Y10.mac(S1, V0);
            Y11.mac(S1, V1);
            Y20.mac(S2, V0);
            Y21.mac(S2, V1);
            Y30.mac(S3, V0);
            Y31.mac(S3, V1);

            for (unsigned i = 1; i < lk / 8; ++i){
                S0 = aie::load_v<MMUL::size_A>(pS1);
                pS1 += MMUL::size_A;
                S1 = aie::load_v<MMUL::size_A>(pS2);
                pS2 += MMUL::size_A;
                S2 = aie::load_v<MMUL::size_A>(pS3);
                pS3 += MMUL::size_A;
                S3 = aie::load_v<MMUL::size_A>(pS4);
                pS4 += MMUL::size_A;

                V0 = aie::load_v<MMUL::size_B>(pV1);
                pV1 += MMUL::size_B;
                V1 = aie::load_v<MMUL::size_B>(pV2);
                pV2 += MMUL::size_B;

                Y00.mac(S0, V0);
                Y01.mac(S0, V1);
                Y10.mac(S1, V0);
                Y11.mac(S1, V1);
                Y20.mac(S2, V0);
                Y21.mac(S2, V1);
                Y30.mac(S3, V0);
                Y31.mac(S3, V1);
            }

            aie::store_v(pY1, Y00.template to_vector<T_out>());
            pY1 += MMUL::size_C;
            aie::store_v(pY1, Y01.template to_vector<T_out>());
            pY1 += MMUL::size_C;
            aie::store_v(pY2, Y10.template to_vector<T_out>());
            pY2 += MMUL::size_C;
            aie::store_v(pY2, Y11.template to_vector<T_out>());
            pY2 += MMUL::size_C;
            aie::store_v(pY3, Y20.template to_vector<T_out>());
            pY3 += MMUL::size_C;
            aie::store_v(pY3, Y21.template to_vector<T_out>());
            pY3 += MMUL::size_C;
            aie::store_v(pY4, Y30.template to_vector<T_out>());
            pY4 += MMUL::size_C;
            aie::store_v(pY4, Y31.template to_vector<T_out>());
            pY4 += MMUL::size_C;
        }
    } // qg
}

// key-in-tile index jj for each lane p = ii*8 + jj (jj = p % 8).
// Used for key-padding mask: keep key (kt*8 + jj) < n_valid  <=>  jj < n_valid - kt*8.
#if defined(__chess__)
alignas(aie::vector_decl_align) int key_col[64] = {
     0, 1, 2, 3, 4, 5, 6, 7,
     0, 1, 2, 3, 4, 5, 6, 7,
     0, 1, 2, 3, 4, 5, 6, 7,
     0, 1, 2, 3, 4, 5, 6, 7,
     0, 1, 2, 3, 4, 5, 6, 7,
     0, 1, 2, 3, 4, 5, 6, 7,
     0, 1, 2, 3, 4, 5, 6, 7,
     0, 1, 2, 3, 4, 5, 6, 7
};
#endif

// n_valid: number of valid keys in this chunk (contiguous [0, n_valid); rest is padding).
// Keys j >= n_valid are masked to -inf on S so exp() -> 0 (keeps zero-pad K/V safe, no NaN).
// Pass n_valid = lk for a fully-populated chunk.
template <typename T_in, typename T_out, unsigned r, unsigned s, unsigned t>
void _attn_qk(T_out *__restrict pS, T_in *__restrict pQ, T_in *__restrict pK, int n_valid) {
    using MMUL = aie::mmul<r, s, t, T_in, T_in, accauto>;

#if defined(__chess__)
    const aie::vector<int, 64> kcol = aie::load_v<64>(key_col);
#else
    // Built in registers under Peano: the attn1 cores have no room for .data.
    aie::vector<int, 64> kcol;
    for (int p = 0; p < 64; p++) kcol[p] = p % 8;
#endif
    const aie::vector<bf16, 64> ninf = aie::broadcast<bf16, 64>(bf16(-0x1.FEp127f));

    for (unsigned z = 0; z < lq / 8; z += 2){
        T_out *__restrict pS1 = pS + (z + 0) * 8 * lk;
        T_out *__restrict pS2 = pS + (z + 1) * 8 * lk;

        for (unsigned j = 0; j < lk / 8; j += 4){
            const T_in *__restrict pQ1 = pQ + (z + 0) * MMUL::size_A;
            const T_in *__restrict pQ2 = pQ + (z + 1) * MMUL::size_A;
            const T_in *__restrict pK1 = pK + (j + 0) * MMUL::size_B;
            const T_in *__restrict pK2 = pK + (j + 1) * MMUL::size_B;
            const T_in *__restrict pK3 = pK + (j + 2) * MMUL::size_B;
            const T_in *__restrict pK4 = pK + (j + 3) * MMUL::size_B;

            aie::vector<T_in, MMUL::size_A> Q0 = aie::load_v<MMUL::size_A>(pQ1);
            pQ1 += 8 * lq;
            aie::vector<T_in, MMUL::size_A> Q1 = aie::load_v<MMUL::size_A>(pQ2);
            pQ2 += 8 * lq;
            aie::vector<T_in, MMUL::size_B> K00 = aie::load_v<MMUL::size_B>(pK1);
            aie::vector<T_in, MMUL::size_B> K0 = aie::transpose(K00, 8, 8);
            pK1 += 8 * lk;
            aie::vector<T_in, MMUL::size_B> K01 = aie::load_v<MMUL::size_B>(pK2);
            aie::vector<T_in, MMUL::size_B> K1 = aie::transpose(K01, 8, 8);
            pK2 += 8 * lk;
            aie::vector<T_in, MMUL::size_B> K02 = aie::load_v<MMUL::size_B>(pK3);
            aie::vector<T_in, MMUL::size_B> K2 = aie::transpose(K02, 8, 8);
            pK3 += 8 * lk;
            aie::vector<T_in, MMUL::size_B> K03 = aie::load_v<MMUL::size_B>(pK4);
            aie::vector<T_in, MMUL::size_B> K3 = aie::transpose(K03, 8, 8);
            pK4 += 8 * lk;

            aie::vector<T_out, MMUL::size_C> acc_C00 = aie::zeros<T_out, MMUL::size_C>(); 
            aie::vector<T_out, MMUL::size_C> acc_C01 = aie::zeros<T_out, MMUL::size_C>();
            aie::vector<T_out, MMUL::size_C> acc_C02 = aie::zeros<T_out, MMUL::size_C>();
            aie::vector<T_out, MMUL::size_C> acc_C03 = aie::zeros<T_out, MMUL::size_C>();
            aie::vector<T_out, MMUL::size_C> acc_C10 = aie::zeros<T_out, MMUL::size_C>();
            aie::vector<T_out, MMUL::size_C> acc_C11 = aie::zeros<T_out, MMUL::size_C>();
            aie::vector<T_out, MMUL::size_C> acc_C12 = aie::zeros<T_out, MMUL::size_C>();
            aie::vector<T_out, MMUL::size_C> acc_C13 = aie::zeros<T_out, MMUL::size_C>();

            MMUL C00(acc_C00);
            MMUL C01(acc_C01);
            MMUL C02(acc_C02);
            MMUL C03(acc_C03);
            MMUL C10(acc_C10);
            MMUL C11(acc_C11);
            MMUL C12(acc_C12);
            MMUL C13(acc_C13);

            C00.mac(Q0, K0);
            C01.mac(Q0, K1);
            C02.mac(Q0, K2);
            C03.mac(Q0, K3);
            C10.mac(Q1, K0);
            C11.mac(Q1, K1);
            C12.mac(Q1, K2);
            C13.mac(Q1, K3);

            for (unsigned i = 1; i < dh / 8; i++){
                Q0 = aie::load_v<MMUL::size_A>(pQ1);
                pQ1 += 8 * lq;
                Q1 = aie::load_v<MMUL::size_A>(pQ2);
                pQ2 += 8 * lq;
                K00 = aie::load_v<MMUL::size_B>(pK1);
                K0 = aie::transpose(K00, 8, 8);
                pK1 += 8 * lk;
                K01 = aie::load_v<MMUL::size_B>(pK2);
                K1 = aie::transpose(K01, 8, 8);
                pK2 += 8 * lk;
                K02 = aie::load_v<MMUL::size_B>(pK3);
                K2 = aie::transpose(K02, 8, 8);
                pK3 += 8 * lk;
                K03 = aie::load_v<MMUL::size_B>(pK4);
                K3 = aie::transpose(K03, 8, 8);
                pK4 += 8 * lk;

                C00.mac(Q0, K0);
                C01.mac(Q0, K1);
                C02.mac(Q0, K2);
                C03.mac(Q0, K3);

                C10.mac(Q1, K0);
                C11.mac(Q1, K1);
                C12.mac(Q1, K2);
                C13.mac(Q1, K3);
            }

            aie::vector<bf16, 64> mout0 = aie::mul(C00.template to_vector<T_out>(), (T_out)scale).template to_vector<T_out>(0);
            aie::vector<bf16, 64> mout1 = aie::mul(C01.template to_vector<T_out>(), (T_out)scale).template to_vector<T_out>(0);
            aie::vector<bf16, 64> mout2 = aie::mul(C02.template to_vector<T_out>(), (T_out)scale).template to_vector<T_out>(0);
            aie::vector<bf16, 64> mout3 = aie::mul(C03.template to_vector<T_out>(), (T_out)scale).template to_vector<T_out>(0);
            aie::vector<bf16, 64> mout4 = aie::mul(C10.template to_vector<T_out>(), (T_out)scale).template to_vector<T_out>(0);
            aie::vector<bf16, 64> mout5 = aie::mul(C11.template to_vector<T_out>(), (T_out)scale).template to_vector<T_out>(0);
            aie::vector<bf16, 64> mout6 = aie::mul(C12.template to_vector<T_out>(), (T_out)scale).template to_vector<T_out>(0);
            aie::vector<bf16, 64> mout7 = aie::mul(C13.template to_vector<T_out>(), (T_out)scale).template to_vector<T_out>(0);

            if (n_valid < (int)lk){
                // key-padding mask: keep lane where key (kt*8 + jj) < n_valid,
                // i.e. jj(=kcol) < n_valid - kt*8. kt = j+col for this tile.
                aie::vector<int, 64> b0 = aie::broadcast<int, 64>(n_valid - (int)(j + 0) * 8);
                aie::vector<int, 64> b1 = aie::broadcast<int, 64>(n_valid - (int)(j + 1) * 8);
                aie::vector<int, 64> b2 = aie::broadcast<int, 64>(n_valid - (int)(j + 2) * 8);
                aie::vector<int, 64> b3 = aie::broadcast<int, 64>(n_valid - (int)(j + 3) * 8);
                mout0 = aie::select(ninf, mout0, aie::lt(kcol, b0));
                mout1 = aie::select(ninf, mout1, aie::lt(kcol, b1));
                mout2 = aie::select(ninf, mout2, aie::lt(kcol, b2));
                mout3 = aie::select(ninf, mout3, aie::lt(kcol, b3));
                mout4 = aie::select(ninf, mout4, aie::lt(kcol, b0));
                mout5 = aie::select(ninf, mout5, aie::lt(kcol, b1));
                mout6 = aie::select(ninf, mout6, aie::lt(kcol, b2));
                mout7 = aie::select(ninf, mout7, aie::lt(kcol, b3));
            }

            aie::store_v(pS1, mout0);
            pS1 += MMUL::size_C;
            aie::store_v(pS1, mout1);
            pS1 += MMUL::size_C;
            aie::store_v(pS1, mout2);
            pS1 += MMUL::size_C;
            aie::store_v(pS1, mout3);
            pS1 += MMUL::size_C;

            aie::store_v(pS2, mout4);
            pS2 += MMUL::size_C;
            aie::store_v(pS2, mout5);
            pS2 += MMUL::size_C;
            aie::store_v(pS2, mout6);
            pS2 += MMUL::size_C;
            aie::store_v(pS2, mout7);
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