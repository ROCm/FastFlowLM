// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include <aie_api/aie.hpp>
#include "common.h"
#include "lut_based_ops.h"
#include <math.h>

inline aie::vector<bf16, 16> update(bf16* new_m_local, aie::vector<bf16, 16>& out, aie::mask<16>& mask){
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
    Vec = aie::select((bf16)0, Vec, mask);
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
            chess_prepare_for_pipelining chess_loop_range(31, ) {
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

            auto mout4 = aie::mul(mout0.first, (T_out)scale).template to_vector<T_out>(0);
            auto mout5 = aie::mul(mout0.second, (T_out)scale).template to_vector<T_out>(0);
            auto mout6 = aie::mul(mout1.first, (T_out)scale).template to_vector<T_out>(0);
            auto mout7 = aie::mul(mout1.second, (T_out)scale).template to_vector<T_out>(0);

            aie::store_v(pS1, mout4);
            pS1 += MMUL::size_C;
            aie::store_v(pS1, mout5);
            pS1 += MMUL::size_C;
            aie::store_v(pS2, mout6);
            pS2 += MMUL::size_C;
            aie::store_v(pS2, mout7);
            pS2 += MMUL::size_C;
        }
    }
}

template<int LK, int lq>
void get_max(bf16*__restrict pS, bf16* prev_m_local, bf16* new_m_local) {
    bf16* pS1 = pS;
#if defined(__chess__)
    bf16 prev_16_m_local[lq];
#else
    // Peano needs the vector alignment for the store_v below.
    alignas(aie::vector_decl_align) bf16 prev_16_m_local[lq];
#endif
    int N = LK / 16;

    aie::vector<bf16, lq> prev_16_m_local_vec = aie::load_v<lq>(prev_m_local);
    aie::store_v(prev_16_m_local, prev_16_m_local_vec);

    for (int t = 0; t < lq / 8; t++){
        for (int i = 0; i < N; i++){
            for (int j = 0; j < 8; j++){ // row 8
                aie::vector<bf16, 16> S0 = aie::load_v<16>(pS1);

                bf16 vm0 = aie::reduce_max(S0);
                bf16 vmax0 = aie::max(vm0, *(prev_16_m_local + j + t * 8));

                *(new_m_local + j + t * 8) = vmax0;
                *(prev_16_m_local + j + t * 8) = vmax0;
                pS1 += 16;
            }
        }
    }
}

template<int lq>
void calculate_c(float* c, bf16* prev_m_local, bf16* new_m_local) {
    for (int i = 0; i < lq; i++){
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

template<int LK, int lq>
void softmax(bf16*__restrict pS, bf16* new_m_local, aie::mask<16>& mask) {
    bf16* pS1 = pS;
    int N = LK / 16;

    for (int t = 0; t < lq / 8; t++){
        for (int i = 0; i < N; i++){
            for (int j = 0; j < 8; j++){ // row 8
                aie::vector<bf16, 16> S0 = aie::load_v<16>(pS1);

                S0 = update(new_m_local + j + t * 8, S0, mask);

                aie::store_v(pS1, S0);
                pS1 += 16;
            }
        }
    }
}

template<int LK, int lq>
void reorder_s(bf16*__restrict pS) {
    bf16 *pS1 = pS;
    int N = LK / 16;

    for (int t = 0; t < lq / 8; t++){
        for (int i = 0; i < N; i++){
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

template<int LK, int DH, int lq>
void calculate_l_and_y(float* y, float* l, float* c, bf16*__restrict pS) {
    // calculate l
    using MMUL = aie::mmul<8, 8, 8, bf16, bf16, accauto>;

    int N = LK / 8;

    for (int t = 0; t < lq / 8; t += 2){
        bf16 *pS1 = pS + (t + 0) * 8 * LK;
        bf16 *pS2 = pS + (t + 1) * 8 * LK;
        aie::vector<bf16, 64> S0 = aie::load_v<64>(pS1);
        pS1 += 64;
        aie::vector<bf16, 64> S2 = aie::load_v<64>(pS2);
        pS2 += 64;
        aie::vector<bf16, 64> Ones = aie::broadcast<bf16, 64>(1.0f);
        aie::vector<bf16, MMUL::size_C> acc_C0 = aie::zeros<bf16, MMUL::size_C>(); 
        aie::vector<bf16, MMUL::size_C> acc_C1 = aie::zeros<bf16, MMUL::size_C>();

        MMUL C0(acc_C0);
        MMUL C1(acc_C1);

        C0.mac(S0, Ones);
        C1.mac(S2, Ones);

        for (int i = 1; i < N; i++){
            S0 = aie::load_v<64>(pS1);
            pS1 += 64;
            S2 = aie::load_v<64>(pS2);
            pS2 += 64;

            C0.mac(S0, Ones);
            C1.mac(S2, Ones);
        }

        auto sum0 = aie::filter_even(C0.template to_vector<float>(), 4);
        auto sum00 = aie::filter_even(sum0, 2);
        auto sum01 = aie::filter_even(sum00, 1);
        auto sum1 = aie::filter_even(C1.template to_vector<float>(), 4);
        auto sum10 = aie::filter_even(sum1, 2);
        auto sum11 = aie::filter_even(sum10, 1);

        auto sum0_vec = aie::concat(sum01, sum11);
        aie::accum<accfloat, 16> sum0_acc;
        sum0_acc.from_vector(sum0_vec);

        aie::vector<float, 16> c0_float32 = aie::load_v<16>(c + t * 8); 
        aie::vector<float, 16> l0_float32 = aie::load_v<16>(l + t * 8);
        aie::accum<accfloat, 16> l_out0;
        l_out0 = mac_elem_16_accuracy_safe(l0_float32, c0_float32, sum0_acc, 0, 0, 0);
        aie::store_v(l + t * 8, l_out0.template to_vector<float>());
    }

    // calculate y
    for (int i = 0; i < lq / 8; i++){
        aie::vector<float, 8> c0 = aie::broadcast<float, 8>(c[i * 8]);
        aie::vector<float, 8> c1 = aie::broadcast<float, 8>(c[i * 8 + 1]);
        aie::vector<float, 8> c2 = aie::broadcast<float, 8>(c[i * 8 + 2]);
        aie::vector<float, 8> c3 = aie::broadcast<float, 8>(c[i * 8 + 3]);  
        aie::vector<float, 8> c4 = aie::broadcast<float, 8>(c[i * 8 + 4]);  
        aie::vector<float, 8> c5 = aie::broadcast<float, 8>(c[i * 8 + 5]);  
        aie::vector<float, 8> c6 = aie::broadcast<float, 8>(c[i * 8 + 6]);  
        aie::vector<float, 8> c7 = aie::broadcast<float, 8>(c[i * 8 + 7]);  
        aie::vector<float, 64> CORRECT = aie::concat(c0, c1, c2, c3, c4, c5, c6, c7);

        float* pY = y + i * 8 * DH;
        for (unsigned j = 0; j < DH / 8; j += 1){
            aie::vector<float, MMUL::size_C> Y = aie::load_v<MMUL::size_C>(pY);
            aie::accum<accfloat, MMUL::size_C> ACC_Y = aie::mul(CORRECT, Y);
            aie::store_v(pY, ACC_Y.template to_vector<float>());
            pY += MMUL::size_C;
        }
    }
}

template <typename T_in, typename T_out, unsigned lq, unsigned LK, unsigned DH, unsigned r, unsigned s, unsigned t>
void _attn_fv(T_out *__restrict pY, T_in *__restrict pS, T_in *__restrict pV) {
    using MMUL = aie::mmul<r, s, t, T_in, T_in, accauto>;

    for (int it = 0; it < lq / 8; it += 2){
        T_out *__restrict pY1 = pY + (it + 0) * 8 * DH;
        T_out *__restrict pY2 = pY + (it + 1) * 8 * DH;

        for (unsigned j = 0; j < DH / 8; j += 2){
            T_in *__restrict pV1 = pV + j * 8 * LK;
            T_in *__restrict pV2 = pV + (j + 1) * 8 * LK;
            T_in *__restrict pS1 = pS + (it + 0) * 8 * LK;
            T_in *__restrict pS2 = pS + (it + 1) * 8 * LK;

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

            for (int i = 1; i < LK / 8; i++){
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

template <unsigned lq>
void calculate_l(float *l, bf16 *l_bf16) {
    aie::vector<float, lq> l_vec = aie::load_v<lq>(l);
    l_vec = aie::inv(l_vec);
    aie::accum<accfloat, lq> l_acc;
    l_acc.from_vector(l_vec);
    aie::store_v(l_bf16, l_acc.template to_vector<bf16>());
}

void scale_div_aie(bf16* o, bf16 *l, float *y) {
    constexpr int vec_factor = 64;

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

constexpr int q_prod_lock = 0;
constexpr int q_cons_lock = 1;
constexpr int in_prod_lock = 2;
constexpr int in_cons_lock = 3;
constexpr int o_prod_lock = 4;
constexpr int o_cons_lock = 5;

extern "C" {
    void attn(
        bf16 *__restrict q_ping,
        bf16 *__restrict q_pong,
        bf16 *__restrict in_ping,
        bf16 *__restrict in_pong,
        bf16 *__restrict o_ping,
        bf16 *__restrict o_pong,
        bf16 *__restrict q,
        bf16 *__restrict s,
        float * y,
        int * L_end_buffer,
        int row
    ){
        AIE_PERSIST_ZERO static bool is_in_ping = false;
        AIE_PERSIST_ZERO static bool is_q_ping = false;
        AIE_PERSIST_ZERO static bool is_o_ping = false;

        alignas(aie::vector_decl_align) static bf16 prev_m_local[lq];
        alignas(aie::vector_decl_align) static bf16 new_m_local[lq];
        alignas(aie::vector_decl_align) static float c_local[lq] = {};
        alignas(aie::vector_decl_align) static float l_local[lq] = {};
        alignas(aie::vector_decl_align) static bf16 l_bf16_local[lq] = {};
        AIE_TRANSIENT const aie::vector<bf16, lq> neg_inf = aie::broadcast<bf16, lq>(-0x1.FEp127f);
        AIE_TRANSIENT const aie::vector<int, 16> idx = aie::vector<int, 16>(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);

        zero_vectorized<float, lq>(l_local);
        zero_vectorized<float, lq * d>(y);

        for (int i = 0; i < lq / 8; i++){
            for (int ll = 0; ll < 8; ll++){
                _lock_acquire(q_cons_lock);
                is_q_ping = !is_q_ping;
                bf16 *q_in = is_q_ping ? (q_ping + row * d) : (q_pong + row * d);
                bf16 *q_out = q + i * 8 * d + ll * 8;
                for (int lc = 0; lc < d / 8; lc++){
                    aie::vector<bf16, 8> q_slice = aie::load_v<8>(q_in);
                    aie::store_v(q_out, q_slice);
                    q_out += 64;
                    q_in += 8;
                } // end of lc loop
                _lock_release(q_prod_lock);
            } // end of ll loop: implement 8 rows of q
        } // end of i loop: implement 2 of 8 rows of q

        aie::store_v(prev_m_local, neg_inf);
        aie::store_v(new_m_local, neg_inf);
        zero_vectorized<float, lq>(c_local);

        int i = L_end_buffer[0];
        do {
            aie::mask<16> mask = aie::le(idx, (i < 16) ? i : 16);
            _lock_acquire(in_cons_lock);
            is_in_ping = !is_in_ping;
            bf16 *k_in = is_in_ping ? in_ping : in_pong;
            _attn_qk<bf16, bf16, lq/8, d/8, lk/8, 8, 8, 8>(s, q, k_in);
            _lock_release(in_prod_lock);
            get_max<lk, lq>(s, prev_m_local, new_m_local);
            softmax<lk, lq>(s, new_m_local, mask);
            calculate_c<lq>(c_local, prev_m_local, new_m_local);
            reorder_s<lk, lq>(s);
            calculate_l_and_y<lk, d, lq>(y, l_local, c_local, s);
            _lock_acquire(in_cons_lock);
            is_in_ping = !is_in_ping;
            bf16 *v_in = is_in_ping ? in_ping : in_pong;
            _attn_fv<bf16, float, lq, lk, d, 8, 8, 8>(y, s, v_in);
            _lock_release(in_prod_lock);
            aie::vector<bf16, lq> prev_m_local_vec = aie::load_v<lq>(new_m_local);
            aie::store_v(prev_m_local, prev_m_local_vec);
            i -= 16;
        } while (i > 0);

        calculate_l<lq>(l_local, l_bf16_local);
        for (int i = 0; i < lq / 8; i++){
            for (int j = 0; j < d / 8; j++){
                _lock_acquire(o_prod_lock);
                is_o_ping = !is_o_ping;
                bf16 *o = is_o_ping ? o_ping : o_pong;
                scale_div_aie(o, l_bf16_local + i * 8, y + i * 8 * d + j * 64);
                _lock_release(o_cons_lock);
            }
        }
    }
}