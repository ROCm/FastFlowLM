// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

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

// rowmax[t] = elementwise max(rowmax[t], row t of the 16x16 block pS): vertical maxes only,
// the horizontal reduction happens once per chunk in fold_row_max.
inline void accumulate_row_max(const bf16*__restrict pS, bf16*__restrict rowmax) {
    for (int t = 0; t < 16; t++){
        aie::vector<bf16, 16> S0 = aie::load_v<16>(pS + t * 16);
        aie::vector<bf16, 16> M0 = aie::load_v<16>(rowmax + t * 16);
        aie::store_v(rowmax + t * 16, aie::max(S0, M0));
    }
}

// m[t] = max(m[t], max over row t of rowmax)
inline void fold_row_max(const bf16*__restrict rowmax, bf16* m) {
    for (int t = 0; t < 16; t++){
        bf16 vm0 = aie::reduce_max(aie::load_v<16>(rowmax + t * 16));
        m[t] = aie::max(vm0, m[t]);
    }
}

// ---- score tiles in 8x8 block layout: a 16x16 tile (16 query rows x 16 keys) is stored as the four
// mmul outputs [C00 rows 0-7 keys 0-7][C01 rows 0-7 keys 8-15][C10 rows 8-15 keys 0-7][C11 rows 8-15
// keys 8-15], each row-major 8x8. That is the layout P*V and calculate_l_and_y read.

// QK^T of 16 query rows x 16 keys, stored in block layout (one (z, j) step of _attn_qk).
// q arrives already scaled by 1/sqrt(dh) (exact: a power of two).
inline void attn_qk_blk(bf16 *__restrict pS, const bf16 *__restrict pQ, const bf16 *__restrict pK) {
    using MMUL = aie::mmul<8, 8, 8, bf16, bf16, accauto>;
    constexpr int colA = dh / 8;   // 8 steps over the head dim
    constexpr int colB = 2;        // 16 keys = 2 blocks of 8
    const bf16 *__restrict pQ1 = pQ;
    const bf16 *__restrict pQ2 = pQ + colA * MMUL::size_A;
    const bf16 *__restrict pK1 = pK;
    const bf16 *__restrict pK2 = pK + MMUL::size_B;
    MMUL C00(aie::zeros<bf16, MMUL::size_C>());
    MMUL C01(aie::zeros<bf16, MMUL::size_C>());
    MMUL C10(aie::zeros<bf16, MMUL::size_C>());
    MMUL C11(aie::zeros<bf16, MMUL::size_C>());
    for (unsigned i = 0; i < colA; ++i){
        aie::vector<bf16, MMUL::size_A> Q0 = aie::load_v<MMUL::size_A>(pQ1);
        pQ1 += MMUL::size_A;
        aie::vector<bf16, MMUL::size_A> Q1 = aie::load_v<MMUL::size_A>(pQ2);
        pQ2 += MMUL::size_A;
        aie::vector<bf16, MMUL::size_B> K0 = aie::transpose(aie::load_v<MMUL::size_B>(pK1), 8, 8);
        pK1 += MMUL::size_B * colB;
        aie::vector<bf16, MMUL::size_B> K1 = aie::transpose(aie::load_v<MMUL::size_B>(pK2), 8, 8);
        pK2 += MMUL::size_B * colB;
        C00.mac(Q0, K0);
        C01.mac(Q0, K1);
        C10.mac(Q1, K0);
        C11.mac(Q1, K1);
    }
    aie::store_v(pS,       C00.template to_vector<bf16>());
    aie::store_v(pS + 64,  C01.template to_vector<bf16>());
    aie::store_v(pS + 128, C10.template to_vector<bf16>());
    aie::store_v(pS + 192, C11.template to_vector<bf16>());
}

// index of (row, key) inside a 16x16 tile in block layout
inline int blk_index(int row, int key) {
    return ((row >> 3) * 2 + (key >> 3)) * 64 + (row & 7) * 8 + (key & 7);
}

// rowmax[0..63]: rows 0-7, rowmax[64..127]: rows 8-15, element-wise max over the tile's two key halves
inline void accumulate_row_max_blk(const bf16*__restrict pS, bf16*__restrict rowmax) {
    aie::vector<bf16, 64> top = aie::max(aie::load_v<64>(pS), aie::load_v<64>(pS + 64));
    aie::vector<bf16, 64> bot = aie::max(aie::load_v<64>(pS + 128), aie::load_v<64>(pS + 192));
    aie::store_v(rowmax, aie::max(top, aie::load_v<64>(rowmax)));
    aie::store_v(rowmax + 64, aie::max(bot, aie::load_v<64>(rowmax + 64)));
}

inline void fold_row_max_blk(const bf16*__restrict rowmax, bf16* m) {
    for (int t = 0; t < 16; t++){
        bf16 vm0 = aie::reduce_max(aie::load_v<8>(rowmax + t * 8));
        m[t] = aie::max(vm0, m[t]);
    }
}

// exp((s - m_row) ) of every score, block layout; N = number of 16x16 tiles
void softmax_blk(bf16*__restrict pS, bf16* new_m_local, int N) {
    constexpr bf16 min_clamp = -87.0f;
    constexpr bf16 max_clamp = 88.0f;
    const aie::vector<bf16, 32> min_value = aie::broadcast<bf16, 32>(min_clamp);
    const aie::vector<bf16, 32> max_value = aie::broadcast<bf16, 32>(max_clamp);
    // m4[q]: rows 4q .. 4q+3, each row's max repeated over its 8 keys (32 lanes = half an 8x8 block)
    aie::vector<bf16, 32> m4[4];
    for (int q = 0; q < 4; q++){
        m4[q] = aie::concat(aie::broadcast<bf16, 8>(new_m_local[4 * q]), aie::broadcast<bf16, 8>(new_m_local[4 * q + 1]),
                            aie::broadcast<bf16, 8>(new_m_local[4 * q + 2]), aie::broadcast<bf16, 8>(new_m_local[4 * q + 3]));
    }
    for (int i = 0; i < N; i++){
        bf16* pS1 = pS + i * 256;
        for (int b = 0; b < 4; b++){            // C00, C01: rows 0-7; C10, C11: rows 8-15
            for (int half = 0; half < 2; half++){
                aie::vector<bf16, 32> Vec = aie::sub(aie::load_v<32>(pS1), m4[(b >> 1) * 2 + half]);
                Vec = aie::clamp(Vec, min_value, max_value);
                aie::accum<accfloat, 32> Vec_acc = aie::mul(Vec, exp_scale);
                aie::store_v(pS1, aie::exp2<bf16>(Vec_acc.template to_vector<float>()));
                pS1 += 32;
            }
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
    constexpr bf16 min_clamp = -87.0f;
    constexpr bf16 max_clamp = 88.0f;
    const aie::vector<bf16, 32> min_value = aie::broadcast<bf16, 32>(min_clamp);
    const aie::vector<bf16, 32> max_value = aie::broadcast<bf16, 32>(max_clamp);
    // rows t and t+1 of a 16x16 block are adjacent, so one 32-lane vector covers two rows;
    // m_pair[p] holds row 2p's max in lanes 0..15 and row 2p+1's in lanes 16..31
    aie::vector<bf16, 32> m_pair[8];
    for (int p = 0; p < 8; p++){
        m_pair[p] = aie::concat(aie::broadcast<bf16, 16>(new_m_local[2 * p]), aie::broadcast<bf16, 16>(new_m_local[2 * p + 1]));
    }
    for (int i = 0; i < N; i++){
        bf16* pS1 = pS + i * 256;
        for (int p = 0; p < 8; p++){
            aie::vector<bf16, 32> S0 = aie::load_v<32>(pS1);
            aie::vector<bf16, 32> Vec = aie::sub(S0, m_pair[p]);
            Vec = aie::clamp(Vec, min_value, max_value);
            aie::accum<accfloat, 32> Vec_acc = aie::mul(Vec, exp_scale);
            aie::store_v(pS1, aie::exp2<bf16>(Vec_acc.template to_vector<float>()));
            pS1 += 32;
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
    using MMUL = aie::mmul<8, 8, 8, bf16, bf16, accauto>;

    int N = lk_mt / lk;

    aie::vector<bf16, 64> Ones = aie::broadcast<bf16, 64>(1.0f);
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

    auto sum0 = aie::filter_even(C0.template to_vector<float>(), 4);
    auto sum00 = aie::filter_even(sum0, 2);
    auto sum01 = aie::filter_even(sum00, 1);
    auto sum1 = aie::filter_even(C1.template to_vector<float>(), 4);
    auto sum10 = aie::filter_even(sum1, 2);
    auto sum11 = aie::filter_even(sum10, 1);
    auto sums0 = aie::concat(sum01, sum11);
    aie::accum<accfloat, 16> sum_acc0;
    sum_acc0.from_vector(sums0);

    aie::vector<float, 16> c0_float32 = aie::load_v<16>(c); 
    aie::vector<float, 16> l0_float32 = aie::load_v<16>(l);
    aie::accum<accfloat, 16> l_out0;
    l_out0 = mac_elem_16_accuracy_safe(l0_float32, c0_float32, sum_acc0, 0, 0, 0);
    aie::store_v(l, l_out0.template to_vector<float>());

    // calculate y
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