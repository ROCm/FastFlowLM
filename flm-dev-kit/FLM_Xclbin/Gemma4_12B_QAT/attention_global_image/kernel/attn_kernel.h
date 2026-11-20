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
    for (int b = 0; b < 16; b++){
        for (int q = 0; q < lq; q++){
            aie::vector<bf16, 8> s_vec = aie::load_v<8>(pS);
            aie::vector<bf16, 8> Vec = aie::sub(s_vec, *(new_m_local + q));
            aie::accum<accfloat, 8> Vec_acc = aie::mul(Vec, exp_scale);
            Vec = aie::exp2<bf16>(Vec_acc.template to_vector<float>()); // f = exp(s - max)
            aie::store_v(pS, Vec);
            pS += 8;
        }
    }
}

#if defined(__chess__)
alignas(aie::vector_decl_align) int idx_array[8] = {7, 6, 5, 4, 3, 2, 1, 0};
#endif
void get_max(
    bf16*__restrict pS,
    bf16*__restrict m
){
    for (int i = 0; i < lq; i++){
        aie::vector<bf16, lk> s_vec = aie::load_v<lk>(pS);

        aie::vector<bf16, lk> m_vec = aie::load_v<lk>(m);
        aie::mask<lk> mask_m = aie::gt(s_vec, m_vec);
        aie::vector<bf16, lk> m_vec_new = aie::select(m_vec, s_vec, mask_m);
        aie::store_v(m, m_vec_new);
        pS += lk;
        m += lk;
    }
}

void apply_mask_and_get_max(
    bf16*__restrict pS,
    bf16*__restrict m,
    int hi,
    int k_base
){
#ifndef __chess__
    // Built on each call: Peano's loader does not initialize this global.
    alignas(aie::vector_decl_align) int idx_array[8];
    for (int z = 0; z < 8; z++) idx_array[z] = 7 - z;
#endif
    aie::vector<int, lk> r_vec = aie::load_v<lk>(idx_array);
    aie::mask<lk> mask_r = aie::le(k_base + 7 - hi, r_vec);

    for (int i = 0; i < lq; i++){
        aie::vector<bf16, lk> In_vec = aie::load_v<lk>(pS);
        aie::vector<bf16, lk> s_vec = aie::select(bf16(-0x1.FEp127f), In_vec, mask_r);

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

void calculate_l(float* l, float*__restrict c, bf16*__restrict pS) {
    // calculate l
    using MMUL = aie::mmul<8, 8, 8, bf16, bf16, accauto>;

    // calculte l = correct * l_left + sum(f)
    aie::vector<bf16, 64> Ones = aie::broadcast<bf16, 64>(1.0f);
    aie::vector<bf16, MMUL::size_C> acc_C0 = aie::zeros<bf16, MMUL::size_C>(); 
    MMUL C0(acc_C0);

    for (int b = 0; b < 16; b++){
        bf16 *pS1 = pS + b * 64;
        aie::vector<bf16, 64> S0 = aie::load_v<64>(pS1);

        C0.mac(S0, Ones);
    }

    auto sum0 = aie::filter_even(C0.template to_vector<float>(), 4);
    auto sum00 = aie::filter_even(sum0, 2);
    auto sum01 = aie::filter_even(sum00, 1);
    aie::accum<accfloat, 8> sum;
    sum.from_vector(sum01);

    aie::vector<float, 8> c_float32 = aie::load_v<8>(c);
    aie::vector<float, 8> l_float32 = aie::load_v<8>(l);
    sum = aie::mac(sum, c_float32, l_float32);
    aie::store_v(l, sum.template to_vector<float>());
}

void calculate_y(float* y, float* c) {
    // calculate y
    aie::vector<float, 8> c0 = aie::broadcast<float, 8>(c[0]);
    aie::vector<float, 8> c1 = aie::broadcast<float, 8>(c[1]);
    aie::vector<float, 8> c2 = aie::broadcast<float, 8>(c[2]);
    aie::vector<float, 8> c3 = aie::broadcast<float, 8>(c[3]);  
    aie::vector<float, 8> c4 = aie::broadcast<float, 8>(c[4]);  
    aie::vector<float, 8> c5 = aie::broadcast<float, 8>(c[5]);  
    aie::vector<float, 8> c6 = aie::broadcast<float, 8>(c[6]);  
    aie::vector<float, 8> c7 = aie::broadcast<float, 8>(c[7]);  
    aie::vector<float, 64> CORRECT = aie::concat(c0, c1, c2, c3, c4, c5, c6, c7);

    float* pY = y;
    for (unsigned j = 0; j < dh / 8; j += 1){
        aie::vector<float, 64> Y = aie::load_v<64>(pY);
        aie::accum<accfloat, 64> ACC_Y = aie::mul(CORRECT, Y);
        aie::store_v(pY, ACC_Y.template to_vector<float>());
        pY += 64;
    }
}

template <typename T_in, typename T_out, unsigned rowA, unsigned colA, unsigned colB, unsigned r, unsigned s, unsigned t>
void _attn_fv(T_out *__restrict pY, T_in *__restrict pS, T_in *__restrict pV) {
    using MMUL = aie::mmul<r, s, t, T_in, T_in, accauto>;

    // load S and reorder s from 16 x 16 to 8 x 8 x 4
    aie::vector<T_in, 64> S0 = aie::load_v<64>(pS);

    T_out *__restrict pY1 = pY;

    for (unsigned j = 0; j < colB; j += 2)
    {
        T_in *__restrict pV1 = pV + j * MMUL::size_B * colA;
        T_in *__restrict pV2 = pV + (j + 1) * MMUL::size_B * colA;

        aie::vector<T_in, MMUL::size_B> V0 = aie::load_v<MMUL::size_B>(pV1);
        aie::vector<T_in, MMUL::size_B> V1 = aie::load_v<MMUL::size_B>(pV2);

        aie::vector<T_out, MMUL::size_C> acc_Y00 = aie::load_v<MMUL::size_C>(pY1);
        aie::vector<T_out, MMUL::size_C> acc_Y01 = aie::load_v<MMUL::size_C>(pY1 + MMUL::size_C);

        MMUL Y00(acc_Y00);
        MMUL Y01(acc_Y01);

        Y00.mac(S0, V0);
        Y01.mac(S0, V1);

        aie::store_v(pY1, Y00.template to_vector<T_out>());
        pY1 += MMUL::size_C;
        aie::store_v(pY1, Y01.template to_vector<T_out>());
        pY1 += MMUL::size_C;
    }
}

template <typename T_in, typename T_out, unsigned rowA, unsigned colA, unsigned colB, unsigned r, unsigned s, unsigned t>
void _attn_qk(T_out *__restrict pS, T_in *__restrict pQ, T_in *__restrict pK) {
    using MMUL = aie::mmul<r, s, t, T_in, T_in, accauto>;

    for (unsigned z = 0; z < rowA; z += 1){
        T_out *__restrict pS1 = pS + (z * colB + 0) * MMUL::size_C;

        for (unsigned j = 0; j < colB; j += 1){
            const T_in *__restrict pQ1 = pQ + (z * colA + 0) * MMUL::size_A;
            const T_in *__restrict pK1 = pK + (0 * colB + j) * MMUL::size_B;

            aie::vector<T_in, MMUL::size_A> Q0 = aie::load_v<MMUL::size_A>(pQ1);
            pQ1 += MMUL::size_A;
            aie::vector<T_in, MMUL::size_B> K00 = aie::load_v<MMUL::size_B>(pK1);
            aie::vector<T_in, MMUL::size_B> K0 = aie::transpose(K00, 8, 8);
            pK1 += MMUL::size_B * colB;

            aie::vector<T_out, MMUL::size_C> acc_C00 = aie::zeros<T_out, MMUL::size_C>(); 

            MMUL C00(acc_C00);

            C00.mac(Q0, K0);

            for (unsigned i = 1; i < colA; ++i){
                Q0 = aie::load_v<MMUL::size_A>(pQ1);
                pQ1 += MMUL::size_A;
                K00 = aie::load_v<MMUL::size_B>(pK1);
                K0 = aie::transpose(K00, 8, 8);
                pK1 += MMUL::size_B * colB;

                C00.mac(Q0, K0);
            }

            aie::store_v(pS1, C00.template to_vector<T_out>());
            pS1 += MMUL::size_C;
        }
    }
}

void scale_div_aie(bf16* o, float *l, float *y) {
    constexpr int vec_factor = 64;

    // organize l
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
    aie::accum<accfloat, vec_factor> AL00 = aie::mul(Y00, LL00);
    aie::store_v(o, AL00.template to_vector<bf16>());
}
#endif