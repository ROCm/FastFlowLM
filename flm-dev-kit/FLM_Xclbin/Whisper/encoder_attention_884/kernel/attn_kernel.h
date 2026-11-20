// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include <aie_api/aie.hpp>
#include "common.h"
#include <math.h>

inline aie::vector<bf16, 16> update(bf16* new_m_local, aie::vector<bf16, 16>& out){
    // s = exp(s - max)
    aie::vector<bf16, 16> Vec = aie::sub(out, *new_m_local);
    //clamp vec to -87 , 88
    constexpr bf16 min_clamp = -87.0f;
    constexpr bf16 max_clamp = 88.0f;
    aie::vector<bf16, 16> min_value = aie::broadcast<bf16, 16>(min_clamp);
    aie::vector<bf16, 16> max_value = aie::broadcast<bf16, 16>(max_clamp); // max of float
    Vec = aie::clamp(Vec, min_value, max_value);
    // aie::accum<accfloat, 16> Vec_acc = getExpBf16(Vec);
    // Vec = Vec_acc.template to_vector<bf16>();
    aie::accum<accfloat, 16> Vec_acc = aie::mul(Vec, exp_scale);
    Vec = aie::exp2<bf16>(Vec_acc.template to_vector<float>());
    return Vec;
}

template <typename T_in, typename T_out, unsigned rowA, unsigned colA, unsigned colB, unsigned r, unsigned s, unsigned t>
void _attn_qk(T_out *__restrict pS, T_in *__restrict pQ, T_in *__restrict pK) {
    using MMUL = aie::mmul<r, s, t, T_in, T_in, accauto>;

    for (unsigned z = 0; z < rowA; z += 2){
        T_out *__restrict pS1 = pS + (z * colB + 0) * MMUL::size_C;
        T_out *__restrict pS2 = pS + ((z + 1) * colB + 0) * MMUL::size_C;

        for (unsigned j = 0; j < colB; j += 4){
            const T_in *__restrict pQ1 = pQ + (z * colA + 0) * MMUL::size_A;
            const T_in *__restrict pQ2 = pQ + ((z + 1) * colA + 0) * MMUL::size_A;
            const T_in *__restrict pK1 = pK + (j + 0) * MMUL::size_B;
            const T_in *__restrict pK2 = pK + (j + 1) * MMUL::size_B;
            const T_in *__restrict pK3 = pK + (j + 2) * MMUL::size_B;
            const T_in *__restrict pK4 = pK + (j + 3) * MMUL::size_B;

            aie::vector<T_in, MMUL::size_A> Q0 = aie::load_v<MMUL::size_A>(pQ1);
            pQ1 += MMUL::size_A;
            aie::vector<T_in, MMUL::size_A> Q1 = aie::load_v<MMUL::size_A>(pQ2);
            pQ2 += MMUL::size_A;
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

            for (unsigned i = 1; i < colA; ++i){
                Q0 = aie::load_v<MMUL::size_A>(pQ1);
                pQ1 += MMUL::size_A;
                Q1 = aie::load_v<MMUL::size_A>(pQ2);
                pQ2 += MMUL::size_A;
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

                C00.mac(Q0, K0);
                C01.mac(Q0, K1);
                C02.mac(Q0, K2);
                C03.mac(Q0, K3);
                C10.mac(Q1, K0);
                C11.mac(Q1, K1);
                C12.mac(Q1, K2);
                C13.mac(Q1, K3);
            }
            auto mout0 = aie::interleave_zip(C00.template to_vector<T_out>(), C01.template to_vector<T_out>(), 4);
            auto mout1 = aie::interleave_zip(C02.template to_vector<T_out>(), C03.template to_vector<T_out>(), 4);
            auto mout2 = aie::interleave_zip(C10.template to_vector<T_out>(), C11.template to_vector<T_out>(), 4);
            auto mout3 = aie::interleave_zip(C12.template to_vector<T_out>(), C13.template to_vector<T_out>(), 4);

            auto mout4 = aie::interleave_zip(mout0.first, mout1.first, 8);
            auto mout5 = aie::interleave_zip(mout0.second, mout1.second, 8);
            auto mout6 = aie::interleave_zip(mout2.first, mout3.first, 8);
            auto mout7 = aie::interleave_zip(mout2.second, mout3.second, 8);
            
            auto mout8 = aie::mul(mout4.first, (T_out)scale).template to_vector<T_out>(0);
            auto mout9 = aie::mul(mout4.second, (T_out)scale).template to_vector<T_out>(0);
            auto mout10 = aie::mul(mout5.first, (T_out)scale).template to_vector<T_out>(0);
            auto mout11 = aie::mul(mout5.second, (T_out)scale).template to_vector<T_out>(0);
            auto mout12 = aie::mul(mout6.first, (T_out)scale).template to_vector<T_out>(0);
            auto mout13 = aie::mul(mout6.second, (T_out)scale).template to_vector<T_out>(0);
            auto mout14 = aie::mul(mout7.first, (T_out)scale).template to_vector<T_out>(0);
            auto mout15 = aie::mul(mout7.second, (T_out)scale).template to_vector<T_out>(0);

            aie::store_v(pS1, mout8);
            pS1 += MMUL::size_C;
            aie::store_v(pS1, mout9);
            pS1 += MMUL::size_C;
            aie::store_v(pS1, mout10);
            pS1 += MMUL::size_C;
            aie::store_v(pS1, mout11);
            pS1 += MMUL::size_C;
            aie::store_v(pS2, mout12);
            pS2 += MMUL::size_C;
            aie::store_v(pS2, mout13);
            pS2 += MMUL::size_C;
            aie::store_v(pS2, mout14);
            pS2 += MMUL::size_C;
            aie::store_v(pS2, mout15);
            pS2 += MMUL::size_C;
        }
    }
}

void get_max(bf16*__restrict pS, bf16* m) {

    for (int t = 0; t < 16; t++){
        aie::vector<bf16, 16> S0 = aie::load_v<16>(pS);

        bf16 vm0 = aie::reduce_max(S0);
        bf16 vmax0 = aie::max(vm0, *(m + t));

        *(m + t) = vmax0;
        pS += 16;
    }
}

void calculate_c(float* c, bf16* prev_m_local, bf16* new_m_local) {
    for (int i = 0; i < 16; i++){
        // correct = exp(m_left - max)
        bf16 correct = aie::sub(*(prev_m_local + i), *(new_m_local + i));
        aie::vector<bf16, 16> correct_vec = aie::broadcast<bf16, 16>(correct);
        // clamp
        constexpr bf16 min_clamp = -87.0f;
        constexpr bf16 max_clamp = 88.0f;
        aie::vector<bf16, 16> min_value = aie::broadcast<bf16, 16>(min_clamp);
        aie::vector<bf16, 16> max_value = aie::broadcast<bf16, 16>(max_clamp); // max of float
        correct_vec = aie::clamp(correct_vec, min_value, max_value);
        aie::accum<accfloat, 16> correct_acc = aie::mul(correct_vec, exp_scale);
        correct_vec = aie::exp2<bf16>(correct_acc.template to_vector<float>());
        // aie::accum<accfloat, 16> correct_acc = getExpBf16(correct_vec);
        // aie::vector<float, 16> correct_float = correct_acc.template to_vector<float>();
        *(c + i) = (float)(correct_vec.get(0));
    }
}

void softmax(bf16*__restrict pS, bf16* new_m_local) {
    int N = lk_mt / lk;

    for (int i = 0; i < N; i++){
        bf16* pS1 = pS + i * 256;
        for (int t = 0; t < 16; t++){

            aie::vector<bf16, 16> S0 = aie::load_v<16>(pS1);

            S0 = update(new_m_local + t, S0);

            aie::store_v(pS1, S0);
            pS1 += 16;
        }
    }
}

void reorder_s(bf16*__restrict pS) {
    int N = lk_mt / lk;

    for (int i = 0; i < N; i++){
        bf16 *pS1 = pS + i * 256;
        for (int t = 0; t < 2; t++){
            aie::vector<bf16, 64> S0 = aie::load_v<64>(pS1);
            aie::vector<bf16, 32> S00 = aie::filter_even(S0, 8);
            aie::vector<bf16, 32> S01 = aie::filter_odd(S0, 8);
            aie::vector<bf16, 64> S1 = aie::load_v<64>(pS1 + 64);
            aie::vector<bf16, 32> S10 = aie::filter_even(S1, 8);
            aie::vector<bf16, 32> S11 = aie::filter_odd(S1, 8);

            S0 = aie::concat(S00, S10);
            S1 = aie::concat(S01, S11);
            aie::store_v(pS1, S0);
            pS1 += 64;
            aie::store_v(pS1, S1);
            pS1 += 64;
        }
    }
}

void calculate_l_and_y(float* y, float* l, float* c, bf16*__restrict pS) {
    // calculate l
    using MMUL = aie::mmul<8, 8, 4, bf16, bf16, accauto>;

    int N = lk_mt / lk;

    aie::vector<bf16, 32> Ones = aie::broadcast<bf16, 32>(1.0f);
    aie::vector<bf16, MMUL::size_C> acc_C0 = aie::zeros<bf16, MMUL::size_C>(); 
    aie::vector<bf16, MMUL::size_C> acc_C1 = aie::zeros<bf16, MMUL::size_C>(); 
    MMUL C0(acc_C0);
    MMUL C1(acc_C1);

    for (int b = 0; b < N; b++){
        bf16 *pS0 = pS + b * 256;
        bf16 *pS1 = pS0;
        bf16 *pS2 = pS0 + 8 * lk;
        for (int i = 0; i < lk / 8; i++){
            aie::vector<bf16, 64> S0 = aie::load_v<64>(pS1);
            aie::vector<bf16, 64> S1 = aie::load_v<64>(pS2);

            C0.mac(S0, Ones);
            C1.mac(S1, Ones);

            pS1 += 64;
            pS2 += 64;
        }
    }

    auto sum0 = aie::filter_even(C0.template to_vector<float>(), 2);
    auto sum00 = aie::filter_even(sum0, 1);
    auto sum1 = aie::filter_even(C1.template to_vector<float>(), 2);
    auto sum10 = aie::filter_even(sum1, 1);
    auto sums0 = aie::concat(sum00, sum10);
    aie::accum<accfloat, 16> sum_acc0;
    sum_acc0.from_vector(sums0);

    aie::vector<float, 16> c0_float32 = aie::load_v<16>(c); 
    aie::vector<float, 16> l0_float32 = aie::load_v<16>(l);
    aie::accum<accfloat, 16> l_out0;
    l_out0 = mac_elem_16_accuracy_safe(l0_float32, c0_float32, sum_acc0, 0, 0, 0);
    aie::store_v(l, l_out0.template to_vector<float>());

    // calculate y
    for (int i = 0; i < 2; i++){
        aie::vector<float, 4> c0 = aie::broadcast<float, 4>(c[i * 8]);
        aie::vector<float, 4> c1 = aie::broadcast<float, 4>(c[i * 8 + 1]);
        aie::vector<float, 4> c2 = aie::broadcast<float, 4>(c[i * 8 + 2]);
        aie::vector<float, 4> c3 = aie::broadcast<float, 4>(c[i * 8 + 3]);  
        aie::vector<float, 4> c4 = aie::broadcast<float, 4>(c[i * 8 + 4]);  
        aie::vector<float, 4> c5 = aie::broadcast<float, 4>(c[i * 8 + 5]);  
        aie::vector<float, 4> c6 = aie::broadcast<float, 4>(c[i * 8 + 6]);  
        aie::vector<float, 4> c7 = aie::broadcast<float, 4>(c[i * 8 + 7]);  
        aie::vector<float, 32> CORRECT = aie::concat(c0, c1, c2, c3, c4, c5, c6, c7);

        float* pY = y + i * 8 * dh;
        for (unsigned j = 0; j < dh / 4; j += 1){
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

        for (unsigned j = 0; j < dh / 4; j += 2){
            T_in *__restrict pV1 = pV + (j + 0) * 4 * lk;
            T_in *__restrict pV2 = pV + (j + 1) * 4 * lk;
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
    constexpr int vec_factor = 32;

    aie::vector<float, 4> L0 = aie::broadcast<float, 4>(l[0]);
    aie::vector<float, 4> L1 = aie::broadcast<float, 4>(l[1]);
    aie::vector<float, 4> L2 = aie::broadcast<float, 4>(l[2]);
    aie::vector<float, 4> L3 = aie::broadcast<float, 4>(l[3]);
    aie::vector<float, 4> L4 = aie::broadcast<float, 4>(l[4]);
    aie::vector<float, 4> L5 = aie::broadcast<float, 4>(l[5]);
    aie::vector<float, 4> L6 = aie::broadcast<float, 4>(l[6]);
    aie::vector<float, 4> L7 = aie::broadcast<float, 4>(l[7]);

    auto LL00 = aie::concat(L0, L1, L2, L3, L4, L5, L6, L7);

    aie::vector<float, vec_factor> Y00 = aie::load_v<vec_factor>(y);
    aie::accum<accfloat, vec_factor> Y00_acc;
    Y00_acc.from_vector(Y00);
    aie::accum<accfloat, vec_factor> AL00 = aie::mul(Y00_acc.template to_vector<float>(), LL00);
    aie::store_v(o, AL00.template to_vector<bf16>());
}