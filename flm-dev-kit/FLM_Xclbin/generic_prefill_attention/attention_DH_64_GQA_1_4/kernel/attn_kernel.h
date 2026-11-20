// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef ATTN_KERNEL_H
#define ATTN_KERNEL_H
#include <aie_api/aie.hpp>
#include "common.h"
#include <math.h>

void apply_softmax(
    bf16*__restrict pS, 
    bf16*__restrict new_m_local
){
    for (int b = 0; b < 8; b++){
        for (int q = 0; q < lq; q++){
            aie::vector<bf16, 32> s_vec = aie::load_v<32>(pS);
            aie::vector<bf16, 32> Vec = aie::sub(s_vec, *(new_m_local + q));
            aie::accum<accfloat, 32> Vec_acc = aie::mul(Vec, exp_scale);
            Vec = aie::exp2<bf16>(Vec_acc.template to_vector<float>()); // f = exp(s - max)
            aie::store_v(pS, Vec);
            pS += 32;
        }
    }
}

void apply_mask_and_get_max(
    bf16*__restrict pS,
    bf16*__restrict m,
    int pointer_block_inner_k,
    int pointer_block_inner_q,
    int pointer_block_inner_k_current
){
    // Build the descending index {31..0} at runtime instead of from a
    // statically-initialized global. Under Peano (llvm-aie) the global's init
    // data lands in a combined .data/.bss PT_LOAD with FileSiz<MemSiz that the
    // aie-rt CDO loader does not initialize correctly (reads as garbage);
    // chess loads it fine. A runtime-built index has no static-init dependency.
    alignas(aie::vector_decl_align) int idx_buf[32];
    for (int z = 0; z < 32; z++) idx_buf[z] = 31 - z;
    aie::vector<int, 32> idx_vec = aie::load_v<32>(idx_buf);
    aie::vector<int, lk> l_vec = aie::broadcast<int, lk>(pointer_block_inner_k);
    l_vec = aie::add(l_vec, idx_vec);
    aie::vector<int, lk> r_vec = aie::broadcast<int, lk>(pointer_block_inner_q);
    r_vec = aie::add(r_vec, idx_vec);

    for (int i = 0; i < lq; i++){
        int current_idx = pointer_block_inner_k_current + 31 - i;
        aie::mask<lk> mask_l = aie::gt(current_idx, l_vec);
        aie::mask<lk> mask_r = aie::le(current_idx, r_vec);
        aie::vector<bf16, lk> In_vec = aie::load_v<lk>(pS);
        aie::vector<bf16, lk> s_vec = aie::select(bf16(-0x1.FEp127f), In_vec, mask_r);
        s_vec = aie::select(bf16(-0x1.FEp127f), s_vec, mask_l);

        aie::vector<bf16, lk> m_vec = aie::load_v<lk>(m);
        aie::mask<lk> mask_m = aie::gt(s_vec, m_vec);
        aie::vector<bf16, lk> m_vec_new = aie::select(m_vec, s_vec, mask_m);
        aie::store_v(pS, s_vec);
        aie::store_v(m, m_vec_new);
        pS += lk;
        m += lk;
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

void reorder_s(bf16*__restrict pS) {
    for (int b = 0; b < 8; b++){
        bf16 *pS1 = pS + b * 1024;

        for (int i = 0; i < lq / 8; i++){
            aie::vector<bf16, 64> S0 = aie::load_v<64>(pS1);
            aie::vector<bf16, 32> S00 = aie::filter_even(S0, 16);
            aie::vector<bf16, 32> S01 = aie::filter_odd(S0, 16);
            aie::vector<bf16, 16> S000 = aie::filter_even(S00, 8);
            aie::vector<bf16, 16> S001 = aie::filter_odd(S00, 8);
            aie::vector<bf16, 16> S010 = aie::filter_even(S01, 8);
            aie::vector<bf16, 16> S011 = aie::filter_odd(S01, 8);
            
            aie::vector<bf16, 64> S1 = aie::load_v<64>(pS1 + 64);
            aie::vector<bf16, 32> S10 = aie::filter_even(S1, 16);
            aie::vector<bf16, 32> S11 = aie::filter_odd(S1, 16);
            aie::vector<bf16, 16> S100 = aie::filter_even(S10, 8);
            aie::vector<bf16, 16> S101 = aie::filter_odd(S10, 8);
            aie::vector<bf16, 16> S110 = aie::filter_even(S11, 8);
            aie::vector<bf16, 16> S111 = aie::filter_odd(S11, 8);

            aie::vector<bf16, 64> S2 = aie::load_v<64>(pS1 + 128);
            aie::vector<bf16, 32> S20 = aie::filter_even(S2, 16);
            aie::vector<bf16, 32> S21 = aie::filter_odd(S2, 16);
            aie::vector<bf16, 16> S200 = aie::filter_even(S20, 8);
            aie::vector<bf16, 16> S201 = aie::filter_odd(S20, 8);
            aie::vector<bf16, 16> S210 = aie::filter_even(S21, 8);
            aie::vector<bf16, 16> S211 = aie::filter_odd(S21, 8);

            aie::vector<bf16, 64> S3 = aie::load_v<64>(pS1 + 192);
            aie::vector<bf16, 32> S30 = aie::filter_even(S3, 16);
            aie::vector<bf16, 32> S31 = aie::filter_odd(S3, 16);
            aie::vector<bf16, 16> S300 = aie::filter_even(S30, 8);
            aie::vector<bf16, 16> S301 = aie::filter_odd(S30, 8);
            aie::vector<bf16, 16> S310 = aie::filter_even(S31, 8);
            aie::vector<bf16, 16> S311 = aie::filter_odd(S31, 8);

            S0 = aie::concat(S000, S100, S200, S300);
            S1 = aie::concat(S001, S101, S201, S301);
            S2 = aie::concat(S010, S110, S210, S310);
            S3 = aie::concat(S011, S111, S211, S311);

            aie::store_v(pS1, S0);
            pS1 += 64;
            aie::store_v(pS1, S1);
            pS1 += 64;
            aie::store_v(pS1, S2);
            pS1 += 64;
            aie::store_v(pS1, S3);
            pS1 += 64;
        }
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
    MMUL C0(acc_C0);
    MMUL C1(acc_C1);
    MMUL C2(acc_C2);
    MMUL C3(acc_C3);

    for (int b = 0; b < 8; b++){
        bf16 *pS0 = pS + b * 1024;
        bf16 *pS1 = pS0;
        bf16 *pS2 = pS0 + 8 * lk;
        bf16 *pS3 = pS0 + 16 * lk;
        bf16 *pS4 = pS0 + 24 * lk;
        for (int i = 0; i < lk / 8; i++){
            aie::vector<bf16, 64> S0 = aie::load_v<64>(pS1);
            aie::vector<bf16, 64> S1 = aie::load_v<64>(pS2);
            aie::vector<bf16, 64> S2 = aie::load_v<64>(pS3);
            aie::vector<bf16, 64> S3 = aie::load_v<64>(pS4);

            C0.mac(S0, Ones);
            C1.mac(S1, Ones);
            C2.mac(S2, Ones);
            C3.mac(S3, Ones);

            pS1 += 64;
            pS2 += 64;
            pS3 += 64;
            pS4 += 64;
        }
    }

    auto sum0 = aie::filter_even(C0.template to_vector<float>(), 4);
    auto sum00 = aie::filter_even(sum0, 2);
    auto sum01 = aie::filter_even(sum00, 1);
    auto sum1 = aie::filter_even(C1.template to_vector<float>(), 4);
    auto sum10 = aie::filter_even(sum1, 2);
    auto sum11 = aie::filter_even(sum10, 1);
    auto sums0 = aie::concat(sum01, sum11);
    auto sum2 = aie::filter_even(C2.template to_vector<float>(), 4);
    auto sum20 = aie::filter_even(sum2, 2);
    auto sum21 = aie::filter_even(sum20, 1);
    auto sum3 = aie::filter_even(C3.template to_vector<float>(), 4);
    auto sum30 = aie::filter_even(sum3, 2);
    auto sum31 = aie::filter_even(sum30, 1);
    auto sums1 = aie::concat(sum21, sum31);

    aie::accum<accfloat, 16> sum_acc0;
    sum_acc0.from_vector(sums0);
    aie::accum<accfloat, 16> sum_acc1;
    sum_acc1.from_vector(sums1);

    aie::vector<float, 16> c0 = aie::load_v<16>(c);
    aie::vector<float, 16> c1 = aie::load_v<16>(c + 16);
    aie::vector<float, 16> l0 = aie::load_v<16>(l);
    aie::vector<float, 16> l1 = aie::load_v<16>(l + 16);

    aie::accum<accfloat, 16> l_out0;
    aie::accum<accfloat, 16> l_out1;
    l_out0 = mac_elem_16_accuracy_safe(l0, c0, sum_acc0, 0, 0, 0);
    l_out1 = mac_elem_16_accuracy_safe(l1, c1, sum_acc1, 0, 0, 0);
    aie::store_v(l, l_out0.template to_vector<float>());
    aie::store_v(l + 16, l_out1.template to_vector<float>());
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

    T_out *__restrict pY1 = pY;
    T_out *__restrict pY2 = pY + 8 * dh;
    T_out *__restrict pY3 = pY + 8 * dh * 2;
    T_out *__restrict pY4 = pY + 8 * dh * 3;


    for (unsigned j = 0; j < dh / 8; j += 2)
    {
        T_in *__restrict pS1 = pS;
        T_in *__restrict pS2 = pS + 8 * lk;
        T_in *__restrict pS3 = pS + 8 * lk * 2;
        T_in *__restrict pS4 = pS + 8 * lk * 3;
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
}

template <typename T_in, typename T_out, unsigned r, unsigned s, unsigned t>
void _attn_qk(T_out *__restrict pS, T_in *__restrict pQ, T_in *__restrict pK) {
    using MMUL = aie::mmul<r, s, t, T_in, T_in, accauto>;

    for (unsigned z = 0; z < lq / 8; z += 2){
        T_out *__restrict pS1 = pS + (z + 0) * 8 * lk;
        T_out *__restrict pS2 = pS + (z + 1) * 8 * lk;

        for (unsigned j = 0; j < lk / 8; j += 4){
            const T_in *__restrict pQ1 = pQ + (z + 0) * 8 * dh;
            const T_in *__restrict pQ2 = pQ + (z + 1) * 8 * dh;
            const T_in *__restrict pK1 = pK + (j + 0) * MMUL::size_B;
            const T_in *__restrict pK2 = pK + (j + 1) * MMUL::size_B;
            const T_in *__restrict pK3 = pK + (j + 2) * MMUL::size_B;
            const T_in *__restrict pK4 = pK + (j + 3) * MMUL::size_B;

            aie::vector<T_in, MMUL::size_A> Q0 = aie::load_v<MMUL::size_A>(pQ1);
            pQ1 += MMUL::size_A;
            aie::vector<T_in, MMUL::size_A> Q1 = aie::load_v<MMUL::size_A>(pQ2);
            pQ2 += MMUL::size_A;
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

            for (unsigned i = 1; i < dh / 8; ++i){
                Q0 = aie::load_v<MMUL::size_A>(pQ1);
                pQ1 += MMUL::size_A;
                Q1 = aie::load_v<MMUL::size_A>(pQ2);
                pQ2 += MMUL::size_A;
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
            auto mout00 = aie::interleave_zip(C00.template to_vector<T_out>(), C01.template to_vector<T_out>(), 8);
            auto mout01 = aie::interleave_zip(C02.template to_vector<T_out>(), C03.template to_vector<T_out>(), 8);
            auto mout10 = aie::interleave_zip(C10.template to_vector<T_out>(), C11.template to_vector<T_out>(), 8);
            auto mout11 = aie::interleave_zip(C12.template to_vector<T_out>(), C13.template to_vector<T_out>(), 8);

            auto mout0 = aie::interleave_zip(mout00.first, mout01.first, 16);
            auto mout1 = aie::interleave_zip(mout00.second, mout01.second, 16);
            auto mout2 = aie::interleave_zip(mout10.first, mout11.first, 16);
            auto mout3 = aie::interleave_zip(mout10.second, mout11.second, 16);

            auto mout4 = aie::mul(mout0.first, (T_out)scale).template to_vector<T_out>(0);
            auto mout5 = aie::mul(mout0.second, (T_out)scale).template to_vector<T_out>(0);
            auto mout6 = aie::mul(mout1.first, (T_out)scale).template to_vector<T_out>(0);
            auto mout7 = aie::mul(mout1.second, (T_out)scale).template to_vector<T_out>(0);
            auto mout14 = aie::mul(mout2.first, (T_out)scale).template to_vector<T_out>(0);
            auto mout15 = aie::mul(mout2.second, (T_out)scale).template to_vector<T_out>(0);
            auto mout16 = aie::mul(mout3.first, (T_out)scale).template to_vector<T_out>(0);
            auto mout17 = aie::mul(mout3.second, (T_out)scale).template to_vector<T_out>(0);

            aie::store_v(pS1, mout4);
            pS1 += MMUL::size_C;
            aie::store_v(pS1, mout5);
            pS1 += MMUL::size_C;
            aie::store_v(pS1, mout6);
            pS1 += MMUL::size_C;
            aie::store_v(pS1, mout7);
            pS1 += MMUL::size_C;

            aie::store_v(pS2, mout14);
            pS2 += MMUL::size_C;
            aie::store_v(pS2, mout15);
            pS2 += MMUL::size_C;
            aie::store_v(pS2, mout16);
            pS2 += MMUL::size_C;
            aie::store_v(pS2, mout17);
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