// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#ifndef ATTN_KERNEL_H
#define ATTN_KERNEL_H
#include <aie_api/aie.hpp>
#include "common.h"
#include <math.h>

// exp(s - max) fused with the 16x16 -> 8x8x4 reordering that _attn_fv/calculate_l
// expect, so S is read and written once instead of twice.
// S is [8 K sub-chunks][lq q rows][lk], and _attn_qk emits it as NG stacked
// copies of the same 256-element pattern, so we walk it 256 elements at a time.
void apply_softmax_and_reorder(
    bf16*__restrict pS,
    bf16*__restrict new_m_local
){
    constexpr int NG = lq / 16;  // 256-element patterns per K sub-chunk

    for (int g = 0; g < NG; g++){
        bf16 *pm = new_m_local + g * 16;
        // One 64-lane broadcast of the row maxima per group of 4 q rows,
        // hoisted out of the sub-chunk loop.
        aie::vector<bf16, 64> M0 = aie::concat(aie::broadcast<bf16, 16>(pm[0]), aie::broadcast<bf16, 16>(pm[1]),
                                               aie::broadcast<bf16, 16>(pm[2]), aie::broadcast<bf16, 16>(pm[3]));
        aie::vector<bf16, 64> M1 = aie::concat(aie::broadcast<bf16, 16>(pm[4]), aie::broadcast<bf16, 16>(pm[5]),
                                               aie::broadcast<bf16, 16>(pm[6]), aie::broadcast<bf16, 16>(pm[7]));
        aie::vector<bf16, 64> M2 = aie::concat(aie::broadcast<bf16, 16>(pm[8]), aie::broadcast<bf16, 16>(pm[9]),
                                               aie::broadcast<bf16, 16>(pm[10]), aie::broadcast<bf16, 16>(pm[11]));
        aie::vector<bf16, 64> M3 = aie::concat(aie::broadcast<bf16, 16>(pm[12]), aie::broadcast<bf16, 16>(pm[13]),
                                               aie::broadcast<bf16, 16>(pm[14]), aie::broadcast<bf16, 16>(pm[15]));

        for (int j = 0; j < 8; j++){
            bf16 *pS1 = pS + (j * NG + g) * 256;

            aie::vector<bf16, 64> S0 = aie::load_v<64>(pS1);
            aie::vector<bf16, 64> S1 = aie::load_v<64>(pS1 + 64);
            aie::vector<bf16, 64> S2 = aie::load_v<64>(pS1 + 128);
            aie::vector<bf16, 64> S3 = aie::load_v<64>(pS1 + 192);

            aie::accum<accfloat, 64> A0 = aie::mul(aie::sub(S0, M0), exp_scale);
            aie::accum<accfloat, 64> A1 = aie::mul(aie::sub(S1, M1), exp_scale);
            aie::accum<accfloat, 64> A2 = aie::mul(aie::sub(S2, M2), exp_scale);
            aie::accum<accfloat, 64> A3 = aie::mul(aie::sub(S3, M3), exp_scale);

            S0 = aie::exp2<bf16>(A0.template to_vector<float>());
            S1 = aie::exp2<bf16>(A1.template to_vector<float>());
            S2 = aie::exp2<bf16>(A2.template to_vector<float>());
            S3 = aie::exp2<bf16>(A3.template to_vector<float>());

            aie::store_v(pS1,       aie::concat(aie::filter_even(S0, 8), aie::filter_even(S1, 8)));
            aie::store_v(pS1 + 64,  aie::concat(aie::filter_odd(S0, 8),  aie::filter_odd(S1, 8)));
            aie::store_v(pS1 + 128, aie::concat(aie::filter_even(S2, 8), aie::filter_even(S3, 8)));
            aie::store_v(pS1 + 192, aie::concat(aie::filter_odd(S2, 8),  aie::filter_odd(S3, 8)));
        }
    }
}

#if defined(__chess__)
alignas(aie::vector_decl_align) int idx_array[32] = {15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                                                     15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
#endif
// Two q rows per iteration: 32 bf16 lanes is the native ALU width, 16 was half of it.
void apply_mask_and_get_max(
    bf16*__restrict pS,
    bf16*__restrict m,
    int pointer_block_inner_k,
    int pointer_block_inner_q,
    int pointer_block_inner_k_current
){
#ifndef __chess__
    // Under Peano the index is built here: an initialized global does not come
    // up with its initial values there.
    alignas(aie::vector_decl_align) int idx_buf[32];
    for (int z = 0; z < 32; z++) idx_buf[z] = 15 - (z & 15);
    aie::vector<int, 32> idx_vec = aie::load_v<32>(idx_buf);
#else
    aie::vector<int, 32> idx_vec = aie::load_v<32>(idx_array);
#endif
    aie::vector<int, 32> l_vec = aie::add(aie::broadcast<int, 32>(pointer_block_inner_k), idx_vec);
    aie::vector<int, 32> r_vec = aie::add(aie::broadcast<int, 32>(pointer_block_inner_q), idx_vec);

    for (int i = 0; i < lq; i += 2){
        aie::vector<int, 32> cur_vec =
            aie::concat(aie::broadcast<int, 16>(pointer_block_inner_k_current + 15 - i),
                        aie::broadcast<int, 16>(pointer_block_inner_k_current + 14 - i));
        aie::mask<32> mask_l = aie::gt(cur_vec, l_vec);
        aie::mask<32> mask_r = aie::le(cur_vec, r_vec);

        aie::vector<bf16, 32> In_vec = aie::load_v<32>(pS);
        aie::vector<bf16, 32> s_vec = aie::select(bf16(-0x1.FEp127f), In_vec, mask_r);
        s_vec = aie::select(bf16(-0x1.FEp127f), s_vec, mask_l);

        aie::vector<bf16, 32> m_vec = aie::load_v<32>(m);
        aie::vector<bf16, 32> m_vec_new = aie::max(m_vec, s_vec);
        aie::store_v(pS, s_vec);
        aie::store_v(m, m_vec_new);
        pS += 2 * lk;
        m += 2 * lk;
    }
}

void calculate_c(float* c, bf16* prev_m_local, bf16* new_m_local) {
    // correct = exp(m_left - max), one lq-wide exp2 instead of lq scalar ones
    aie::vector<bf16, lq> prev_vec = aie::load_v<lq>(prev_m_local);
    aie::vector<bf16, lq> new_vec = aie::load_v<lq>(new_m_local);
    aie::accum<accfloat, lq> diff_acc = aie::mul(aie::sub(prev_vec, new_vec), exp_scale);
    aie::vector<bf16, lq> correct_vec = aie::exp2<bf16>(diff_acc.template to_vector<float>());
    aie::accum<accfloat, lq> correct_acc;
    correct_acc.from_vector(correct_vec);
    aie::store_v(c, correct_acc.template to_vector<float>());
}

void calculate_l(float* l, float*__restrict c, bf16*__restrict pS) {
    // l = correct * l_left + sum(f)
    using MMUL = aie::mmul<8, 8, 8, bf16, bf16, accauto>;
    using MMUL_C = mmul_c_t<MMUL, bf16, 8, 8, 8>;
    constexpr int NG = lq / 16;

    aie::vector<bf16, 64> Ones = aie::broadcast<bf16, 64>(1.0f);

    for (int g = 0; g < NG; g++){
        aie::vector<bf16, MMUL::size_C> acc_C0 = aie::zeros<bf16, MMUL::size_C>();
        aie::vector<bf16, MMUL::size_C> acc_C1 = aie::zeros<bf16, MMUL::size_C>();
        MMUL_C C0(acc_C0);
        MMUL_C C1(acc_C1);

        for (int b = 0; b < 8; b++){
            bf16 *pS1 = pS + (b * NG + g) * 256;
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

        aie::vector<float, 16> c_float32 = aie::load_v<16>(c + g * 16);
        aie::vector<float, 16> l_float32 = aie::load_v<16>(l + g * 16);

        aie::accum<accfloat, 16> l_out;
        l_out = mac_elem_16_accuracy_safe(l_float32, c_float32, sum, 0, 0, 0);
        aie::store_v(l + g * 16, l_out.template to_vector<float>());
    }
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
    using MMUL_C = mmul_c_t<MMUL, T_in, r, s, t>;

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

        MMUL_C Y00(acc_Y00);
        MMUL_C Y01(acc_Y01);
        MMUL_C Y10(acc_Y10);
        MMUL_C Y11(acc_Y11);

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
    using MMUL_C = mmul_c_t<MMUL, T_in, r, s, t>;

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
            // Two K passes of two accumulators each: with all four in one loop
            // too many operands are live for Peano to pipeline it. Each
            // accumulator gets the same macs in the same order.
            aie::vector<T_out, MMUL::size_C> zc = aie::zeros<T_out, MMUL::size_C>();
            MMUL_C C00(zc); MMUL_C C01(zc); MMUL_C C10(zc); MMUL_C C11(zc);
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