// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include "aie_kernel_utils.h"
#include "model_spec.h"
#include "typedef.hpp"

typedef float y_acc_dtype;

// A operand of the S @ V mmul: an r x s tile of bf16.
using s_tile_t = aie::vector<bf16, GQA_R * GQA_S>;

void calculate_l(bf16 *__restrict pS, float *__restrict c, float *__restrict l);
void calculate_y(float *__restrict y, float *__restrict c);
template <unsigned colQ, unsigned r, unsigned s, unsigned t>
void attn_fv(const s_tile_t& S0, const s_tile_t& S1, bf16 *__restrict pV, y_acc_dtype *__restrict pY, int index);
template <unsigned N>
void scale_div_aie(float *a, bf16* o, float *l);


constexpr int s_prod_lock_0 = 4;
constexpr int s_cons_lock_0 = 5;
constexpr int s_prod_lock_1 = 6;
constexpr int s_cons_lock_1 = 7;
constexpr int v_prod_lock = 2;
constexpr int v_cons_lock = 3;
constexpr int o_prod_lock = 0;
constexpr int o_cons_lock = 1; 
constexpr int l_prod_lock = 8;

inline void reorder_s(bf16 *__restrict pS, s_tile_t& S0, s_tile_t& S1) {
    aie::vector<bf16, 64> S_up   = aie::load_v<64>(pS);
    aie::vector<bf16, 64> S_down = aie::load_v<64>(pS + 64);

    S0 = aie::concat(aie::filter_even(S_up, 8), aie::filter_even(S_down, 8));
    S1 = aie::concat(aie::filter_odd(S_up, 8),  aie::filter_odd(S_down, 8));
}

extern "C"{

void attn_kv(
    bf16 *__restrict s_ping,
    bf16 *__restrict s_pong,
    bf16 *__restrict v_ping,
    bf16 *__restrict v_pong,
    float*__restrict y,
    bf16 *__restrict o,
    int* L
)
{
    alignas(aie::vector_decl_align) AIE_TRANSIENT float l[8] = {};
    const aie::vector<float, 8> zero = aie::broadcast<float, 8>(0);

    AIE_PERSIST_ZERO static bool is_v_ping = false;
    AIE_PERSIST_ZERO static bool is_sc_ping = false;
    zero_vectorized<y_acc_dtype, 8 * DH>(y);
    aie::store_v(l, zero);
    _lock_acquire(l_prod_lock); 
    int i = L[0];
    do {
        // s
        is_sc_ping = !is_sc_ping;
        if(is_sc_ping){
            _down_lock_acquire(s_cons_lock_0);
        } else {
            _down_lock_acquire(s_cons_lock_1);
        }
        bf16 *s = is_sc_ping ? s_ping : s_pong;
        float *c = (float *)(s + Q_HEADS_PADDED_PER_CU * 16);
        calculate_l(s, c, l);
        calculate_y(y, c);
        s_tile_t S0, S1;
        reorder_s(s, S0, S1);
        for (int j = 0; j < 4; j++){
            _lock_acquire(v_cons_lock);
            is_v_ping = !is_v_ping;
            bf16 *v = is_v_ping ? v_ping : v_pong;
            attn_fv<DH / 8, GQA_R, GQA_S, GQA_T>(S0, S1, v, y, j);
            _lock_release(v_prod_lock);
        }
        if(is_sc_ping){
            _down_lock_release(s_prod_lock_0);
        } else {
            _down_lock_release(s_prod_lock_1);
        }
        i -= 16;
    } while (i > 0);
    _lock_acquire(o_prod_lock, 1);
    scale_div_aie<8 * DH>(y, o, l);
    _lock_release(o_cons_lock, 1);
}
}

#if ATTN_IMPL == ATTN_IMPL_2x4x1
// specialized for gemma, DH is 512
void calculate_l(bf16 *__restrict pS, float *__restrict c, float *__restrict l) {
    bf16 *__restrict pS1 = pS;
    for (int i = 0; i < 8; i++) {
        aie::vector<bf16, 16> s_vec = aie::load_v<16>(pS1);
        float sum = aie::reduce_add(s_vec);
        l[i] = sum + l[i] * c[i];
        pS1 += 16;
    }
}

void calculate_y(float *__restrict y, float *__restrict c) {
    aie::vector<float, 8> c0 = aie::broadcast<float, 8>(c[0]);
    aie::vector<float, 8> c1 = aie::broadcast<float, 8>(c[1]);
    aie::vector<float, 8> c2 = aie::broadcast<float, 8>(c[2]);
    aie::vector<float, 8> c3 = aie::broadcast<float, 8>(c[3]);   
    aie::vector<float, 8> c4 = aie::broadcast<float, 8>(c[4]);
    aie::vector<float, 8> c5 = aie::broadcast<float, 8>(c[5]);
    aie::vector<float, 8> c6 = aie::broadcast<float, 8>(c[6]);
    aie::vector<float, 8> c7 = aie::broadcast<float, 8>(c[7]);
    aie::vector<float, 64> CORRECT = aie::concat(c0, c1, c2, c3, c4, c5, c6, c7);

    constexpr int vec_factor = 64;
    for (int i = 0; i < DH / 8; i++) {
        aie::vector<y_acc_dtype, vec_factor> y_vec = aie::load_v<vec_factor>(y);
        aie::accum<accfloat, vec_factor> y_acc = aie::mul(y_vec, CORRECT);
        aie::store_v(y, y_acc.template to_vector<y_acc_dtype>());
        y += vec_factor;
    }
}

template <unsigned colQ, unsigned r, unsigned s, unsigned t>
void attn_fv(bf16 *__restrict pS, bf16 *__restrict pV, y_acc_dtype *__restrict pY, int index) {

    using MMUL = aie::mmul<r, s, t, bf16, bf16, accfloat>;

    y_acc_dtype *__restrict pY1 = pY;
    bf16 *__restrict pS1 = pS;
   
    aie::vector<bf16, 64> S_up = aie::load_v<64>(pS1);
    pS1 += 64;
    aie::vector<bf16, 64> S_down = aie::load_v<64>(pS1);

    aie::vector<bf16, 32> S10 = aie::filter_even(S_up, 8);
    aie::vector<bf16, 32> S11 = aie::filter_odd(S_up, 8);
    aie::vector<bf16, 32> S20 = aie::filter_even(S_down, 8);
    aie::vector<bf16, 32> S21 = aie::filter_odd(S_down, 8);
    
    aie::vector<bf16, MMUL::size_A> S0 = aie::concat(S10, S20);
    aie::vector<bf16, MMUL::size_A> S1 = aie::concat(S11, S21);

    bf16 *__restrict pV1 = pV;

	for (unsigned j = 0; j < colQ; j += 4){
        //v0
        bf16 *__restrict pV01 = pV1 + (j + 0) * MMUL::size_B * 2;
        bf16 *__restrict pV02 = pV1 + (j + 1) * MMUL::size_B * 2;
        bf16 *__restrict pV03 = pV1 + (j + 2) * MMUL::size_B * 2;
        bf16 *__restrict pV04 = pV1 + (j + 3) * MMUL::size_B * 2;

        aie::vector<bf16, MMUL::size_B> V00 = aie::load_v<MMUL::size_B>(pV01);
        pV01 += MMUL::size_B;
        aie::vector<bf16, MMUL::size_B> V01 = aie::load_v<MMUL::size_B>(pV02);
        pV02 += MMUL::size_B;
        aie::vector<bf16, MMUL::size_B> V02 = aie::load_v<MMUL::size_B>(pV03);
        pV03 += MMUL::size_B;
        aie::vector<bf16, MMUL::size_B> V03 = aie::load_v<MMUL::size_B>(pV04);
        pV04 += MMUL::size_B;

        aie::vector<y_acc_dtype, MMUL::size_C> acc_Y00 = aie::load_v<MMUL::size_C>(pY1);
        aie::vector<y_acc_dtype, MMUL::size_C> acc_Y01 = aie::load_v<MMUL::size_C>(pY1 + MMUL::size_C);
        aie::vector<y_acc_dtype, MMUL::size_C> acc_Y02 = aie::load_v<MMUL::size_C>(pY1 + MMUL::size_C * 2);
        aie::vector<y_acc_dtype, MMUL::size_C> acc_Y03 = aie::load_v<MMUL::size_C>(pY1 + MMUL::size_C * 3);

        MMUL Y00(acc_Y00);
        MMUL Y01(acc_Y01);
        MMUL Y02(acc_Y02);
        MMUL Y03(acc_Y03);

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

        if(index == 0) {
            auto Y20 = aie::filter_even(Y00.template to_vector<y_acc_dtype>(), 32);
            auto Y21 = aie::filter_even(Y01.template to_vector<y_acc_dtype>(), 32);
            auto Y22 = aie::filter_even(Y02.template to_vector<y_acc_dtype>(), 32);
            auto Y23 = aie::filter_even(Y03.template to_vector<y_acc_dtype>(), 32);

            aie::store_v(pY1, Y20);
            pY1 += MMUL::size_C;
            aie::store_v(pY1, Y21);
            pY1 += MMUL::size_C;
            aie::store_v(pY1, Y22);
            pY1 += MMUL::size_C;
            aie::store_v(pY1, Y23);
            pY1 += MMUL::size_C;
        } else {
            auto Y20 = aie::filter_odd(Y00.template to_vector<y_acc_dtype>(), 32);
            auto Y21 = aie::filter_odd(Y01.template to_vector<y_acc_dtype>(), 32);
            auto Y22 = aie::filter_odd(Y02.template to_vector<y_acc_dtype>(), 32);
            auto Y23 = aie::filter_odd(Y03.template to_vector<y_acc_dtype>(), 32);

            aie::store_v(pY1 + 32, Y20);
            pY1 += MMUL::size_C;
            aie::store_v(pY1 + 32, Y21);
            pY1 += MMUL::size_C;
            aie::store_v(pY1 + 32, Y22);
            pY1 += MMUL::size_C;
            aie::store_v(pY1 + 32, Y23);
            pY1 += MMUL::size_C;
        }
    }
}

template <unsigned N>
void scale_div_aie(float *a, bf16* o, float *l) {

    constexpr int vec_factor = 64;
    const int F = N / vec_factor;

    float* pA1 = a;
    bf16* pO1 = o;

    aie::vector<float, 8> L0 = aie::broadcast<float, 8>(1.0f / l[0]);
    aie::vector<float, 8> L1 = aie::broadcast<float, 8>(1.0f / l[1]);
    aie::vector<float, 8> L2 = aie::broadcast<float, 8>(1.0f / l[2]);
    aie::vector<float, 8> L3 = aie::broadcast<float, 8>(1.0f / l[3]);
    aie::vector<float, 8> L4 = aie::broadcast<float, 8>(1.0f / l[4]);
    aie::vector<float, 8> L5 = aie::broadcast<float, 8>(1.0f / l[5]);
    aie::vector<float, 8> L6 = aie::broadcast<float, 8>(1.0f / l[6]);
    aie::vector<float, 8> L7 = aie::broadcast<float, 8>(1.0f / l[7]);

    auto L = aie::concat(L0, L1, L2, L3, L4, L5, L6, L7);

    for (int d = 0; d < N / vec_factor; d++){
        aie::vector<float, vec_factor> A0 = aie::load_v<vec_factor>(pA1);
        aie::accum<accfloat, vec_factor> AL = aie::mul(A0, L);
        aie::vector<bf16, vec_factor> AL_bf16 = AL.template to_vector<bf16>();
        aie::store_v(pO1, AL_bf16);
        pO1 += vec_factor;
        pA1 += vec_factor;
    }

}

#elif ATTN_IMPL == ATTN_IMPL_4x2x1
void calculate_l(bf16 *__restrict pS, float *__restrict c, float *__restrict l) {
    bf16 *__restrict pS1 = pS;
    for (int i = 0; i < 8; i++) {
        aie::vector<bf16, 16> s_vec = aie::load_v<16>(pS1);
        float sum = aie::reduce_add(s_vec);
        l[i] = sum + l[i] * c[i];
        pS1 += 16;
    }
}

void calculate_y(float *__restrict y, float *__restrict c) {
    aie::vector<float, 8> c0 = aie::broadcast<float, 8>(c[0]);
    aie::vector<float, 8> c1 = aie::broadcast<float, 8>(c[1]);
    aie::vector<float, 8> c2 = aie::broadcast<float, 8>(c[2]);
    aie::vector<float, 8> c3 = aie::broadcast<float, 8>(c[3]);   
    aie::vector<float, 8> c4 = aie::broadcast<float, 8>(c[4]);
    aie::vector<float, 8> c5 = aie::broadcast<float, 8>(c[5]);
    aie::vector<float, 8> c6 = aie::broadcast<float, 8>(c[6]);
    aie::vector<float, 8> c7 = aie::broadcast<float, 8>(c[7]);
    aie::vector<float, 64> CORRECT = aie::concat(c0, c1, c2, c3, c4, c5, c6, c7);

    constexpr int vec_factor = 64;
    for (int i = 0; i < DH / 8; i++) {
        aie::vector<y_acc_dtype, vec_factor> y_vec = aie::load_v<vec_factor>(y);
        aie::accum<accfloat, vec_factor> y_acc = aie::mul(y_vec, CORRECT);
        aie::store_v(y, y_acc.template to_vector<y_acc_dtype>());
        y += vec_factor;
    }
}

template <unsigned colQ, unsigned r, unsigned s, unsigned t>
void attn_fv(const s_tile_t& S0, const s_tile_t& S1, bf16 *__restrict pV, y_acc_dtype *__restrict pY, int index) {

    using MMUL = aie::mmul<r, s, t, bf16, bf16, accfloat>;

    constexpr unsigned Y_SLICE = Q_HEADS_PER_GROUP_PADDED * t;

    y_acc_dtype *__restrict pY1 = pY;
    bf16 *__restrict pV1 = pV;

	for (unsigned j = 0; j < colQ; j += 4){
        //v0
        bf16 *__restrict pV01 = pV1 + (j + 0) * MMUL::size_B * 2;
        bf16 *__restrict pV02 = pV1 + (j + 1) * MMUL::size_B * 2;
        bf16 *__restrict pV03 = pV1 + (j + 2) * MMUL::size_B * 2;
        bf16 *__restrict pV04 = pV1 + (j + 3) * MMUL::size_B * 2;

        aie::vector<bf16, MMUL::size_B> V00 = aie::load_v<MMUL::size_B>(pV01);
        pV01 += MMUL::size_B;
        aie::vector<bf16, MMUL::size_B> V01 = aie::load_v<MMUL::size_B>(pV02);
        pV02 += MMUL::size_B;
        aie::vector<bf16, MMUL::size_B> V02 = aie::load_v<MMUL::size_B>(pV03);
        pV03 += MMUL::size_B;
        aie::vector<bf16, MMUL::size_B> V03 = aie::load_v<MMUL::size_B>(pV04);
        pV04 += MMUL::size_B;

        aie::vector<y_acc_dtype, MMUL::size_C> acc_Y00 = aie::load_v<MMUL::size_C>(pY1);
        aie::vector<y_acc_dtype, MMUL::size_C> acc_Y01 = aie::load_v<MMUL::size_C>(pY1 + MMUL::size_C);
        aie::vector<y_acc_dtype, MMUL::size_C> acc_Y02 = aie::load_v<MMUL::size_C>(pY1 + MMUL::size_C * 2);
        aie::vector<y_acc_dtype, MMUL::size_C> acc_Y03 = aie::load_v<MMUL::size_C>(pY1 + MMUL::size_C * 3);

        MMUL Y00(acc_Y00);
        MMUL Y01(acc_Y01);
        MMUL Y02(acc_Y02);
        MMUL Y03(acc_Y03);

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

        aie::store_v(pY1 + index * Y_SLICE, Y00.template to_vector<y_acc_dtype>().template extract<Y_SLICE>(index));
        pY1 += MMUL::size_C;
        aie::store_v(pY1 + index * Y_SLICE, Y01.template to_vector<y_acc_dtype>().template extract<Y_SLICE>(index));
        pY1 += MMUL::size_C;
        aie::store_v(pY1 + index * Y_SLICE, Y02.template to_vector<y_acc_dtype>().template extract<Y_SLICE>(index));
        pY1 += MMUL::size_C;
        aie::store_v(pY1 + index * Y_SLICE, Y03.template to_vector<y_acc_dtype>().template extract<Y_SLICE>(index));
        pY1 += MMUL::size_C;
    }
}

template <unsigned N>
void scale_div_aie(float *a, bf16* o, float *l) {

    constexpr int vec_factor = 64;
    const int F = N / vec_factor;

    float* pA1 = a;
    bf16* pO1 = o;

    aie::vector<float, 8> L0 = aie::broadcast<float, 8>(1.0f / l[0]);
    aie::vector<float, 8> L1 = aie::broadcast<float, 8>(1.0f / l[1]);
    aie::vector<float, 8> L2 = aie::broadcast<float, 8>(1.0f / l[2]);
    aie::vector<float, 8> L3 = aie::broadcast<float, 8>(1.0f / l[3]);
    aie::vector<float, 8> L4 = aie::broadcast<float, 8>(1.0f / l[4]);
    aie::vector<float, 8> L5 = aie::broadcast<float, 8>(1.0f / l[5]);
    aie::vector<float, 8> L6 = aie::broadcast<float, 8>(1.0f / l[6]);
    aie::vector<float, 8> L7 = aie::broadcast<float, 8>(1.0f / l[7]);

    auto L = aie::concat(L0, L1, L2, L3, L4, L5, L6, L7);

    for (int d = 0; d < N / vec_factor; d++){
        aie::vector<float, vec_factor> A0 = aie::load_v<vec_factor>(pA1);
        aie::accum<accfloat, vec_factor> AL = aie::mul(A0, L);
        aie::vector<bf16, vec_factor> AL_bf16 = AL.template to_vector<bf16>();
        aie::store_v(pO1, AL_bf16);
        pO1 += vec_factor;
        pA1 += vec_factor;
    }

}
#elif ATTN_IMPL == ATTN_IMPL_1x8x1

void calculate_l(bf16 * pS, float * c, float * l) {
    bf16 *__restrict pS1 = pS;
    for (int i = 0; i < 8; i++) {
        aie::vector<bf16, 16> s_vec = aie::load_v<16>(pS1);
        float sum = aie::reduce_add(s_vec);
        l[i] = sum + l[i] * c[i];
        pS1 += 16;
    }
}

typedef float y_acc_dtype;
template <unsigned colQ, unsigned r, unsigned s, unsigned t>
void attn_fv(bf16 *__restrict pS, bf16 *__restrict pV, y_acc_dtype *__restrict pY, float *__restrict c, int index) {

    using MMUL = aie::mmul<r, s, t, bf16, bf16, accfloat>;

    y_acc_dtype *__restrict pY1 = pY;
    bf16 *__restrict pS1 = pS;
    
    aie::vector<bf16, 64> S_up = aie::load_v<64>(pS1);
    pS1 += 64;
    aie::vector<bf16, 64> S_down = aie::load_v<64>(pS1);
    pS1 += 64;

    aie::vector<bf16, 32> S10 = aie::filter_even(S_up, 8);
    aie::vector<bf16, 32> S11 = aie::filter_odd(S_up, 8);
    aie::vector<bf16, 32> S20 = aie::filter_even(S_down, 8);
    aie::vector<bf16, 32> S21 = aie::filter_odd(S_down, 8);
    
    aie::vector<bf16, MMUL::size_A> S0 = aie::concat(S10, S20);
    aie::vector<bf16, MMUL::size_A> S1 = aie::concat(S11, S21);
    
    aie::vector<y_acc_dtype, 8> c0 = aie::broadcast<float, 8>(c[0]);
    aie::vector<y_acc_dtype, 8> c1 = aie::broadcast<float, 8>(c[1]);
    aie::vector<y_acc_dtype, 8> c2 = aie::broadcast<float, 8>(c[2]);
    aie::vector<y_acc_dtype, 8> c3 = aie::broadcast<float, 8>(c[3]); 
    aie::vector<y_acc_dtype, 8> c4 = aie::broadcast<float, 8>(c[4]);
    aie::vector<y_acc_dtype, 8> c5 = aie::broadcast<float, 8>(c[5]);
    aie::vector<y_acc_dtype, 8> c6 = aie::broadcast<float, 8>(c[6]);
    aie::vector<y_acc_dtype, 8> c7 = aie::broadcast<float, 8>(c[7]);
    aie::vector<y_acc_dtype, 64> CORRECT = aie::concat(c0, c1, c2, c3, c4, c5, c6, c7);

    bf16 *__restrict pV1 = pV;

	for (unsigned j = 0; j < colQ; j += 4) {
        //v0
        bf16 *__restrict pV01 = pV1 + (j + 0) * MMUL::size_B;
        bf16 *__restrict pV02 = pV1 + (j + 1) * MMUL::size_B;
        bf16 *__restrict pV03 = pV1 + (j + 2) * MMUL::size_B;
        bf16 *__restrict pV04 = pV1 + (j + 3) * MMUL::size_B;

        aie::vector<bf16, MMUL::size_B> V00 = aie::load_v<MMUL::size_B>(pV01);
        pV01 += MMUL::size_B * colQ;
        aie::vector<bf16, MMUL::size_B> V01 = aie::load_v<MMUL::size_B>(pV02);
        pV02 += MMUL::size_B * colQ;
        aie::vector<bf16, MMUL::size_B> V02 = aie::load_v<MMUL::size_B>(pV03);
        pV03 += MMUL::size_B * colQ;
        aie::vector<bf16, MMUL::size_B> V03 = aie::load_v<MMUL::size_B>(pV04);
        pV04 += MMUL::size_B * colQ;

        aie::vector<y_acc_dtype, MMUL::size_C> acc_y00 = aie::load_v<MMUL::size_C>(pY1);
        aie::vector<y_acc_dtype, MMUL::size_C> acc_y01 = aie::load_v<MMUL::size_C>(pY1 + MMUL::size_C);
        aie::vector<y_acc_dtype, MMUL::size_C> acc_y02 = aie::load_v<MMUL::size_C>(pY1 + MMUL::size_C * 2);
        aie::vector<y_acc_dtype, MMUL::size_C> acc_y03 = aie::load_v<MMUL::size_C>(pY1 + MMUL::size_C * 3);

        aie::accum<accfloat, MMUL::size_C> ACC_Y00;
        aie::accum<accfloat, MMUL::size_C> ACC_Y01;
        aie::accum<accfloat, MMUL::size_C> ACC_Y02;
        aie::accum<accfloat, MMUL::size_C> ACC_Y03;
        ACC_Y00 = aie::mul(CORRECT, acc_y00);
        ACC_Y01 = aie::mul(CORRECT, acc_y01);
        ACC_Y02 = aie::mul(CORRECT, acc_y02);
        ACC_Y03 = aie::mul(CORRECT, acc_y03);
        
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

        aie::store_v(pY1, Y00.template to_vector<y_acc_dtype>());
        pY1 += MMUL::size_C;
        aie::store_v(pY1, Y01.template to_vector<y_acc_dtype>());
        pY1 += MMUL::size_C;
        aie::store_v(pY1, Y02.template to_vector<y_acc_dtype>());
        pY1 += MMUL::size_C;
        aie::store_v(pY1, Y03.template to_vector<y_acc_dtype>());
        pY1 += MMUL::size_C;
    }
    
}

template <unsigned N>
void scale_div_aie(float *a, bf16* o, float *l) {

    constexpr int vec_factor = 64;
    const int F = N / vec_factor;

    float* pA1 = a;
    bf16* pO1 = o;

    aie::vector<float, 8> L0 = aie::broadcast<float, 8>(1.0f / l[0]);
    aie::vector<float, 8> L1 = aie::broadcast<float, 8>(1.0f / l[1]);
    aie::vector<float, 8> L2 = aie::broadcast<float, 8>(1.0f / l[2]);
    aie::vector<float, 8> L3 = aie::broadcast<float, 8>(1.0f / l[3]);
    aie::vector<float, 8> L4 = aie::broadcast<float, 8>(1.0f / l[4]);
    aie::vector<float, 8> L5 = aie::broadcast<float, 8>(1.0f / l[5]);
    aie::vector<float, 8> L6 = aie::broadcast<float, 8>(1.0f / l[6]);
    aie::vector<float, 8> L7 = aie::broadcast<float, 8>(1.0f / l[7]);

    auto L = aie::concat(L0, L1, L2, L3, L4, L5, L6, L7);

    for (int d = 0; d < N / vec_factor; d++){
        aie::vector<float, vec_factor> A0 = aie::load_v<vec_factor>(pA1);
        aie::accum<accfloat, vec_factor> AL = aie::mul(A0, L);
        aie::vector<bf16, vec_factor> AL_bf16 = AL.template to_vector<bf16>();
        aie::store_v(pO1, AL_bf16);
        pO1 += vec_factor;
        pA1 += vec_factor;
    }
    
}

#endif