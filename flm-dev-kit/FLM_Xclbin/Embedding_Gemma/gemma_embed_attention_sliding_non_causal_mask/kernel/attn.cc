// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include <aie_api/aie.hpp>
#include "common.h"
#include "lut_based_ops.h"
#include <math.h>

#if defined(__chess__)
alignas(aie::vector_decl_align) int idx_array[16] = {15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
#endif
template<uint32_t M = 16>
void apply_mask_softmax(bf16*__restrict Inout, bf16* m, bf16* c, int k_pos, int k_begin, int k_end, int L) {
    static_assert(M == 16, "M must be 16");
#if defined(__chess__)
    aie::vector<int, 16> idx_vec = aie::load_v<16>(idx_array);
#else
    // A local vector under Peano: this design's buffers leave no room for .data.
    aie::vector<int, 16> idx_vec = aie::vector<int, 16>(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
#endif


    aie::vector<int, NPU_N_TILE> l_vec = aie::broadcast<int, NPU_N_TILE>(L);
    l_vec = aie::add(l_vec, idx_vec);
    aie::mask<NPU_N_TILE> cutoff_mask = aie::le(k_pos + 16, l_vec);


    l_vec = aie::broadcast<int, NPU_N_TILE>(k_begin);
    l_vec = aie::add(l_vec, idx_vec);
    aie::vector<int, NPU_N_TILE> r_vec = aie::broadcast<int, NPU_N_TILE>(k_end);
    r_vec = aie::add(r_vec, idx_vec);

    for (int row = 0; row < M; row++)
    {
        int current_idx = k_pos + 15 - row;
        aie::mask<NPU_N_TILE> mask_l = aie::ge(current_idx, l_vec);
        aie::mask<NPU_N_TILE> mask_r = aie::le(current_idx, r_vec);
        aie::vector<bf16, NPU_N_TILE> In_vec = aie::load_v<NPU_N_TILE>(Inout);
        aie::vector<bf16, NPU_N_TILE> s_vec = aie::select(bf16(-0x1.FEp127f), In_vec, mask_r);
        s_vec = aie::select(bf16(-0x1.FEp127f), s_vec, mask_l);
        s_vec = aie::select(bf16(-0x1.FEp127f), s_vec, cutoff_mask);
        bf16 vm = aie::reduce_max(s_vec);
        bf16 vmax = aie::max(vm, *(m + row));
        aie::vector<bf16, NPU_N_TILE> Vec = aie::sub(s_vec, vmax);
        aie::accum<accfloat, NPU_N_TILE> Vec_acc = aie::mul(Vec, exp_scale);
        Vec = aie::exp2<bf16>(Vec_acc.template to_vector<float>()); // f = exp(s - max)
        Vec = aie::select(bf16(0.0f), Vec, mask_l);
        Vec = aie::select(bf16(0.0f), Vec, mask_r);
        Vec = aie::select(bf16(0.0f), Vec, cutoff_mask);
        bf16 inner_correct = aie::sub(*(m + row), vmax); // correct = exp(m_left - max)
        inner_correct = aie::mul(inner_correct, exp_scale);
        aie::vector<bf16, NPU_N_TILE> correct_vec = aie::broadcast<bf16, NPU_N_TILE>(inner_correct);
        aie::accum<accfloat, NPU_N_TILE> correct_acc;
        correct_acc.from_vector(correct_vec);
        correct_vec = aie::exp2<bf16>(correct_acc.template to_vector<float>());
        *(m + row) = (bf16)vmax;
        *(c + row) = (bf16)correct_vec.get(0);
        aie::store_v(Inout, Vec);
        Inout += NPU_N_TILE;
    }
}

template<int colB>
void calculate_l_and_y(float* y, float* l, bf16*__restrict c, bf16*__restrict pS) {
    // calculate l
    using MMUL = aie::mmul<8, 8, 8, bf16, bf16, accauto>;
    aie::vector<bf16, 64> S0 = aie::load_v<64>(pS);
    aie::vector<bf16, 32> S00 = aie::filter_even(S0, 8);
    aie::vector<bf16, 32> S01 = aie::filter_odd(S0, 8);
    aie::vector<bf16, 64> S1 = aie::load_v<64>(pS + 64);
    aie::vector<bf16, 32> S10 = aie::filter_even(S1, 8);
    aie::vector<bf16, 32> S11 = aie::filter_odd(S1, 8);
    aie::vector<bf16, 64> S2 = aie::load_v<64>(pS + 128);
    aie::vector<bf16, 32> S20 = aie::filter_even(S2, 8);
    aie::vector<bf16, 32> S21 = aie::filter_odd(S2, 8);
    aie::vector<bf16, 64> S3 = aie::load_v<64>(pS + 192);
    aie::vector<bf16, 32> S30 = aie::filter_even(S3, 8);
    aie::vector<bf16, 32> S31 = aie::filter_odd(S3, 8);

    S0 = aie::concat(S00, S10);
    S1 = aie::concat(S01, S11);
    S2 = aie::concat(S20, S30);
    S3 = aie::concat(S21, S31);

    // calculte l = correct * l_left + sum(f)
    aie::vector<bf16, 64> Ones = aie::broadcast<bf16, 64>(1.0f);
    aie::vector<bf16, MMUL::size_C> acc_C0 = aie::zeros<bf16, MMUL::size_C>(); 
    aie::vector<bf16, MMUL::size_C> acc_C1 = aie::zeros<bf16, MMUL::size_C>(); 

    MMUL C0(acc_C0);
    MMUL C1(acc_C1);

    C0.mac(S0, Ones);
    C1.mac(S2, Ones);

    C0.mac(S1, Ones);
    C1.mac(S3, Ones);

    auto sum0 = aie::filter_even(C0.template to_vector<float>(), 4);
    auto sum00 = aie::filter_even(sum0, 2);
    auto sum01 = aie::filter_even(sum00, 1);
    auto sum1 = aie::filter_even(C1.template to_vector<float>(), 4);
    auto sum10 = aie::filter_even(sum1, 2);
    auto sum11 = aie::filter_even(sum10, 1);
    auto sum2 = aie::concat(sum01, sum11);
    aie::accum<accfloat, 16> sum;
    sum.from_vector(sum2);

    aie::vector<bf16, 16> c_bf16 = aie::load_v<16>(c);
    aie::accum<accfloat, 16> c_accf;
    c_accf.from_vector(c_bf16);
    aie::vector<float, 16> c_float32 = c_accf.template to_vector<float>();

    aie::vector<float, 16> l_float32 = aie::load_v<16>(l);

    aie::accum<accfloat, 16> l_out;
    l_out = mac_elem_16_accuracy_safe(l_float32, c_float32, sum, 0, 0, 0);
    aie::store_v(l, l_out.template to_vector<float>());

    aie::vector<float, 8> c0 = aie::broadcast<float, 8>(c[0]);
    aie::vector<float, 8> c1 = aie::broadcast<float, 8>(c[1]);
    aie::vector<float, 8> c2 = aie::broadcast<float, 8>(c[2]);
    aie::vector<float, 8> c3 = aie::broadcast<float, 8>(c[3]);   
    aie::vector<float, 8> c4 = aie::broadcast<float, 8>(c[4]);
    aie::vector<float, 8> c5 = aie::broadcast<float, 8>(c[5]);
    aie::vector<float, 8> c6 = aie::broadcast<float, 8>(c[6]);
    aie::vector<float, 8> c7 = aie::broadcast<float, 8>(c[7]);
    aie::vector<float, 8> c8 = aie::broadcast<float, 8>(c[8]);
    aie::vector<float, 8> c9 = aie::broadcast<float, 8>(c[9]);
    aie::vector<float, 8> c10 = aie::broadcast<float, 8>(c[10]);
    aie::vector<float, 8> c11 = aie::broadcast<float, 8>(c[11]);
    aie::vector<float, 8> c12 = aie::broadcast<float, 8>(c[12]);
    aie::vector<float, 8> c13 = aie::broadcast<float, 8>(c[13]);
    aie::vector<float, 8> c14 = aie::broadcast<float, 8>(c[14]);
    aie::vector<float, 8> c15 = aie::broadcast<float, 8>(c[15]);

    aie::vector<float, 64> CORRECT00 = aie::concat(c0, c1, c2, c3, c4, c5, c6, c7);
    aie::vector<float, 64> CORRECT01 = aie::concat(c8, c9, c10, c11, c12, c13, c14, c15);

    float* pY1 = y;
    float* pY2 = y + colB * MMUL::size_C;

    for (unsigned j = 0; j < colB; j += 2){
        aie::vector<float, MMUL::size_C> Y00 = aie::load_v<MMUL::size_C>(pY1);
        aie::vector<float, MMUL::size_C> Y01 = aie::load_v<MMUL::size_C>(pY1 + MMUL::size_C);
        aie::vector<float, MMUL::size_C> Y10 = aie::load_v<MMUL::size_C>(pY2);
        aie::vector<float, MMUL::size_C> Y11 = aie::load_v<MMUL::size_C>(pY2 + MMUL::size_C);

        aie::accum<accfloat, MMUL::size_C> ACC_Y00;
        aie::accum<accfloat, MMUL::size_C> ACC_Y01;
        aie::accum<accfloat, MMUL::size_C> ACC_Y10;
        aie::accum<accfloat, MMUL::size_C> ACC_Y11;

        ACC_Y00 = aie::mul(CORRECT00, Y00);
        ACC_Y01 = aie::mul(CORRECT00, Y01);
        ACC_Y10 = aie::mul(CORRECT01, Y10);
        ACC_Y11 = aie::mul(CORRECT01, Y11);
        
        aie::store_v(pY1, ACC_Y00.template to_vector<float>());
        pY1 += MMUL::size_C;
        aie::store_v(pY1, ACC_Y01.template to_vector<float>());
        pY1 += MMUL::size_C;
        aie::store_v(pY2, ACC_Y10.template to_vector<float>());
        pY2 += MMUL::size_C;
        aie::store_v(pY2, ACC_Y11.template to_vector<float>());
        pY2 += MMUL::size_C;
    }
}

template <typename T_in, typename T_out, unsigned rowA, unsigned colA, unsigned colB, unsigned r, unsigned s, unsigned t>
void _attn_fv(T_out *__restrict pY, T_in *__restrict pS, T_in *__restrict pV) {
    using MMUL = aie::mmul<r, s, t, T_in, T_in, accauto>;

    // load S and reorder s from 16 x 16 to 8 x 8 x 4
    aie::vector<T_in, 64> S0 = aie::load_v<64>(pS);
    aie::vector<T_in, 32> S00 = aie::filter_even(S0, 8);
    aie::vector<T_in, 32> S01 = aie::filter_odd(S0, 8);
    aie::vector<T_in, 64> S1 = aie::load_v<64>(pS + 64);
    aie::vector<T_in, 32> S10 = aie::filter_even(S1, 8);
    aie::vector<T_in, 32> S11 = aie::filter_odd(S1, 8);
    aie::vector<T_in, 64> S2 = aie::load_v<64>(pS + 128);
    aie::vector<T_in, 32> S20 = aie::filter_even(S2, 8);
    aie::vector<T_in, 32> S21 = aie::filter_odd(S2, 8);
    aie::vector<T_in, 64> S3 = aie::load_v<64>(pS + 192);
    aie::vector<T_in, 32> S30 = aie::filter_even(S3, 8);
    aie::vector<T_in, 32> S31 = aie::filter_odd(S3, 8);

    S0 = aie::concat(S00, S10);
    S1 = aie::concat(S01, S11);
    S2 = aie::concat(S20, S30);
    S3 = aie::concat(S21, S31);
    
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

    for (unsigned z = 0; z < rowA; z += 2)
    chess_prepare_for_pipelining chess_loop_range(2, ) {
        T_out *__restrict pS1 = pS + (z * colB + 0) * MMUL::size_C;
        T_out *__restrict pS2 = pS + ((z + 1) * colB + 0) * MMUL::size_C;

        for (unsigned j = 0; j < colB; j += 2)
        {
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

            for (unsigned i = 1; i < colA; ++i)
            {
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

template <unsigned N>
void scale_div_aie(bf16* o, float *l, float *y) {
    constexpr int vec_factor = 64;

    // organize l
    bf16 l00 = (bf16)getInvBf16(l[0]);
    aie::vector<bf16, 8> L0 = aie::broadcast<bf16, 8>(l00);
    bf16 l01 = (bf16)getInvBf16(l[1]);
    aie::vector<bf16, 8> L1 = aie::broadcast<bf16, 8>(l01);
    bf16 l02 = (bf16)getInvBf16(l[2]);
    aie::vector<bf16, 8> L2 = aie::broadcast<bf16, 8>(l02);
    bf16 l03 = (bf16)getInvBf16(l[3]);
    aie::vector<bf16, 8> L3 = aie::broadcast<bf16, 8>(l03);
    bf16 l04 = (bf16)getInvBf16(l[4]);
    aie::vector<bf16, 8> L4 = aie::broadcast<bf16, 8>(l04);
    bf16 l05 = (bf16)getInvBf16(l[5]);
    aie::vector<bf16, 8> L5 = aie::broadcast<bf16, 8>(l05);
    bf16 l06 = (bf16)getInvBf16(l[6]);
    aie::vector<bf16, 8> L6 = aie::broadcast<bf16, 8>(l06);
    bf16 l07 = (bf16)getInvBf16(l[7]);
    aie::vector<bf16, 8> L7 = aie::broadcast<bf16, 8>(l07);
    bf16 l08 = (bf16)getInvBf16(l[8]);
    aie::vector<bf16, 8> L8 = aie::broadcast<bf16, 8>(l08);
    bf16 l09 = (bf16)getInvBf16(l[9]);
    aie::vector<bf16, 8> L9 = aie::broadcast<bf16, 8>(l09);
    bf16 l10 = (bf16)getInvBf16(l[10]);
    aie::vector<bf16, 8> L10 = aie::broadcast<bf16, 8>(l10);
    bf16 l11 = (bf16)getInvBf16(l[11]);
    aie::vector<bf16, 8> L11 = aie::broadcast<bf16, 8>(l11);
    bf16 l12 = (bf16)getInvBf16(l[12]);
    aie::vector<bf16, 8> L12 = aie::broadcast<bf16, 8>(l12);
    bf16 l13 = (bf16)getInvBf16(l[13]);
    aie::vector<bf16, 8> L13 = aie::broadcast<bf16, 8>(l13);
    bf16 l14 = (bf16)getInvBf16(l[14]);
    aie::vector<bf16, 8> L14 = aie::broadcast<bf16, 8>(l14);
    bf16 l15 = (bf16)getInvBf16(l[15]);
    aie::vector<bf16, 8> L15 = aie::broadcast<bf16, 8>(l15);
    auto LL00 = aie::concat(L0, L1, L2, L3, L4, L5, L6, L7);
    auto LL10 = aie::concat(L8, L9, L10, L11, L12, L13, L14, L15);

    bf16* pO1 = o;
    bf16* pO2 = o + 8 * NPU_K_TILE;
    float* pY1 = y;
    float* pY2 = y + 8 * NPU_K_TILE;

    for (int i = 0; i < N; i += 4) {
        aie::vector<float, vec_factor> Y00 = aie::load_v<vec_factor>(pY1);
        pY1 += vec_factor;
        aie::vector<float, vec_factor> Y01 = aie::load_v<vec_factor>(pY1);
        pY1 += vec_factor;
        aie::vector<float, vec_factor> Y02 = aie::load_v<vec_factor>(pY1);
        pY1 += vec_factor;
        aie::vector<float, vec_factor> Y03 = aie::load_v<vec_factor>(pY1);
        pY1 += vec_factor;
        aie::vector<float, vec_factor> Y10 = aie::load_v<vec_factor>(pY2);
        pY2 += vec_factor;
        aie::vector<float, vec_factor> Y11 = aie::load_v<vec_factor>(pY2);
        pY2 += vec_factor;
        aie::vector<float, vec_factor> Y12 = aie::load_v<vec_factor>(pY2);
        pY2 += vec_factor;
        aie::vector<float, vec_factor> Y13 = aie::load_v<vec_factor>(pY2);
        pY2 += vec_factor;
        
        aie::accum<accfloat, vec_factor> Y00_acc;
        aie::accum<accfloat, vec_factor> Y01_acc;
        aie::accum<accfloat, vec_factor> Y02_acc;
        aie::accum<accfloat, vec_factor> Y03_acc;
        aie::accum<accfloat, vec_factor> Y10_acc;
        aie::accum<accfloat, vec_factor> Y11_acc;
        aie::accum<accfloat, vec_factor> Y12_acc;
        aie::accum<accfloat, vec_factor> Y13_acc;

        Y00_acc.from_vector(Y00);
        Y01_acc.from_vector(Y01);
        Y02_acc.from_vector(Y02);
        Y03_acc.from_vector(Y03);
        Y10_acc.from_vector(Y10);
        Y11_acc.from_vector(Y11);
        Y12_acc.from_vector(Y12);
        Y13_acc.from_vector(Y13);

        aie::accum<accfloat, vec_factor> AL00 = aie::mul(Y00_acc.template to_vector<bf16>(), LL00);
        aie::accum<accfloat, vec_factor> AL01 = aie::mul(Y01_acc.template to_vector<bf16>(), LL00);
        aie::accum<accfloat, vec_factor> AL02 = aie::mul(Y02_acc.template to_vector<bf16>(), LL00);
        aie::accum<accfloat, vec_factor> AL03 = aie::mul(Y03_acc.template to_vector<bf16>(), LL00);
        aie::accum<accfloat, vec_factor> AL10 = aie::mul(Y10_acc.template to_vector<bf16>(), LL10);
        aie::accum<accfloat, vec_factor> AL11 = aie::mul(Y11_acc.template to_vector<bf16>(), LL10);
        aie::accum<accfloat, vec_factor> AL12 = aie::mul(Y12_acc.template to_vector<bf16>(), LL10);
        aie::accum<accfloat, vec_factor> AL13 = aie::mul(Y13_acc.template to_vector<bf16>(), LL10);

        aie::store_v(pO1, AL00.template to_vector<bf16>());
        pO1 += vec_factor;
        aie::store_v(pO1, AL01.template to_vector<bf16>());
        pO1 += vec_factor;
        aie::store_v(pO1, AL02.template to_vector<bf16>());
        pO1 += vec_factor;
        aie::store_v(pO1, AL03.template to_vector<bf16>());
        pO1 += vec_factor;
        aie::store_v(pO2, AL10.template to_vector<bf16>());
        pO2 += vec_factor;
        aie::store_v(pO2, AL11.template to_vector<bf16>());
        pO2 += vec_factor;
        aie::store_v(pO2, AL12.template to_vector<bf16>());
        pO2 += vec_factor;
        aie::store_v(pO2, AL13.template to_vector<bf16>());
        pO2 += vec_factor;
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
        int * L_buffer,
        int * window_size_buffer,
        const int row, 
        const int col
    ){
#if defined(__chess__)
        static bool is_kv_ping = false;
        static bool is_q_ping = false;
        static bool is_o_ping = false;
#else
        // This design's buffers leave no room for .data under Peano. is_kv_ping and
        // is_q_ping flip an even number of times per call, so they start each call
        // false. is_o_ping flips once per chunk and is kept in window_size_buffer[1..2]
        // (the host writes only word 0); word 1 holds a marker, so the first call
        // after load starts from false.
        constexpr int o_sel_marker = 0x5a5a17e5;
        bool is_kv_ping = false;
        bool is_q_ping = false;
        bool is_o_ping = window_size_buffer[1] == o_sel_marker && window_size_buffer[2] != 0;
#endif
        constexpr int CHUNK_SIZE = 128;
        constexpr int BLOCK_SIZE = 16;
        constexpr int COLUMNS = 8;

        alignas(aie::vector_decl_align) bf16 m[NPU_M_TILE];
        alignas(aie::vector_decl_align) bf16 c[NPU_M_TILE];
        alignas(aie::vector_decl_align) float l[NPU_M_TILE];

        AIE_TRANSIENT const aie::vector<bf16, 16> neg_inf = aie::broadcast<bf16, 16>(-0x1.FEp127f);
        AIE_TRANSIENT const aie::vector<bf16, NPU_M_TILE> one = aie::broadcast<bf16, NPU_M_TILE>(1.0);

        int chunk_idx = 0;
        int total_chunks = 0;

        do{
            aie::store_v(m, neg_inf);
            aie::store_v(c, one);
            zero_vectorized<float, NPU_M_TILE>(l);
            zero_vectorized<float, NPU_M_TILE * NPU_K_TILE>(y);
            for (int q_row = 0; q_row < 2; q_row++){
                for (int ll = 0; ll < 8; ll++){
                    _lock_acquire(q_cons_lock);
                    is_q_ping = !is_q_ping;
                    bf16 *q_in = is_q_ping ? (q_ping + row * NPU_K_TILE) : (q_pong + row * NPU_K_TILE);
                    bf16 *q_out = q + q_row * 8 * NPU_K_TILE + ll * 8;
                    for (int lc = 0; lc < NPU_K_TILE / 8; lc++){
                        aie::vector<bf16, 8> q_slice = aie::load_v<8>(q_in);
                        aie::store_v(q_out, q_slice);
                        q_out += 64;
                        q_in += 8;
                    }
                    _lock_release(q_prod_lock);
                }
            }

            int L = L_buffer[0];
            int W = window_size_buffer[0];
            int L_PADDED = (L + CHUNK_SIZE - 1) / CHUNK_SIZE * CHUNK_SIZE;
            total_chunks = L_PADDED / CHUNK_SIZE;
            int chunk_q_begin = chunk_idx * CHUNK_SIZE;
            int chunk_q_end = chunk_q_begin + CHUNK_SIZE - 1;
            int chunk_k_begin = chunk_q_begin - W + 1;
            int chunk_k_end = chunk_q_end + W;
            chunk_k_begin = chunk_k_begin < 0 ? 0 : chunk_k_begin;
            chunk_k_end = chunk_k_end > L ? L : chunk_k_end;
            int k_begin = chunk_k_begin / CHUNK_SIZE * CHUNK_SIZE;
            int k_end = (chunk_k_end + CHUNK_SIZE - 1) / CHUNK_SIZE * CHUNK_SIZE;
            int k_chunks = (k_end - k_begin) / CHUNK_SIZE;

            int ct_q_begin = chunk_q_begin + col * BLOCK_SIZE;
            int ct_q_end = ct_q_begin + BLOCK_SIZE - 1;
            int ct_k_begin = ct_q_begin - W + 1;
            int ct_k_end = ct_q_end + W;
            ct_k_begin = ct_k_begin > -BLOCK_SIZE ? ct_k_begin : -BLOCK_SIZE;
            ct_k_end = ct_k_end > (L_PADDED + W) ? (L_PADDED + W) : ct_k_end;
            ct_k_end -= BLOCK_SIZE;

            // apply softmax
            for (int k_chunk_idx = 0; k_chunk_idx < k_chunks; k_chunk_idx++){
                for (int k_block_idx = 0; k_block_idx < COLUMNS; k_block_idx++){
                    int k_pos = k_begin + k_chunk_idx * CHUNK_SIZE + k_block_idx * BLOCK_SIZE;
                    _lock_acquire(in_cons_lock);
                    is_kv_ping = !is_kv_ping;
                    bf16 *k = is_kv_ping ? in_ping : in_pong;
                    _attn_qk<bf16, bf16, NPU_M_TILE/8, NPU_K_TILE/8, NPU_N_TILE/8, 8, 8, 8>(s, q, k);
                    _lock_release(in_prod_lock);
                    apply_mask_softmax<NPU_M_TILE>(s, m, c, k_pos, ct_k_begin, ct_k_end, L);
                    calculate_l_and_y<NPU_K_TILE/8>(y, l, c, s);
                    _lock_acquire(in_cons_lock);
                    is_kv_ping = !is_kv_ping;
                    bf16 *v = is_kv_ping ? in_ping : in_pong;
                    _attn_fv<bf16, float, NPU_M_TILE/8, NPU_N_TILE/8, NPU_K_TILE/8, 8, 8, 8>(y, s, v);
                    _lock_release(in_prod_lock);
                }
            }
            _lock_acquire(o_prod_lock);
            is_o_ping = !is_o_ping;
            bf16 *o = is_o_ping ? o_ping : o_pong;
            scale_div_aie<NPU_K_TILE / 8>(o, l, y);
            _lock_release(o_cons_lock);
            chunk_idx++;
        } while (chunk_idx < total_chunks);
#if !defined(__chess__)
        window_size_buffer[1] = o_sel_marker;
        window_size_buffer[2] = is_o_ping;
#endif
    }
}