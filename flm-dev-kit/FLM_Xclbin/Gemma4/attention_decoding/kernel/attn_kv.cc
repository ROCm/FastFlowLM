// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include "aie_kernel_utils.h"
#include "model_spec.h"
#include "typedef.hpp"

typedef float y_acc_dtype;
void calculate_l(bf16 *__restrict pS, float *__restrict c, float *__restrict l);
void calculate_y(float *__restrict y, float *__restrict c);
template <unsigned colQ, unsigned r, unsigned s, unsigned t>
void attn_fv(bf16 *__restrict pS, bf16 *__restrict pV, y_acc_dtype *__restrict pY, bool is_up);
template <unsigned N>
void scale_div_aie(bf16 *a, bf16* o, float *l);
template <typename T_in, typename T_out, int N>
void passThrough_aie(T_in *restrict in0, T_out *restrict out);


constexpr int s_prod_lock_0 = 4;
constexpr int s_cons_lock_0 = 5;
constexpr int s_prod_lock_1 = 6;
constexpr int s_cons_lock_1 = 7;
constexpr int v_prod_lock = 2;
constexpr int v_cons_lock = 3;
constexpr int o_prod_lock = 0;
constexpr int o_cons_lock = 1; 
constexpr int l_prod_lock = 8;

extern "C"{

alignas(aie::vector_decl_align) unsigned char m_inv_lut[128] = {
    0,   126, 124, 122, 120, 118, 117, 115, 113, 111, 109, 108, 106, 104, 103,
    101, 100, 98,  96,  95,  93,  92,  90,  89,  88,  86,  85,  83,  82,  81,
    79,  78,  77,  76,  74,  73,  72,  71,  69,  68,  67,  66,  65,  64,  63,
    61,  60,  59,  58,  57,  56,  55,  54,  53,  52,  51,  50,  49,  48,  47,
    46,  45,  44,  44,  43,  42,  41,  40,  39,  38,  37,  37,  36,  35,  34,
    33,  33,  32,  31,  30,  30,  29,  28,  27,  27,  26,  25,  24,  24,  23,
    22,  22,  21,  20,  20,  19,  18,  18,  17,  16,  16,  15,  14,  14,  13,
    13,  12,  11,  11,  10,  10,  9,   9,   8,   7,   7,   6,   6,   5,   5,
    4,   4,   3,   3,   2,   2,   1,   1};

__attribute__((always_inline)) bfloat16 getInvBf16(float x) {
  unsigned int *B_x;
  unsigned int exp_mask = 0x7F800000;
  unsigned int mantissa_mask = 0x007FFFFF;
  unsigned int mantissa_Q = 0x00008000;
  unsigned char exponent, mantissa;
  unsigned inv_exponent;
  unsigned short inv_x_val;
  unsigned int B_Q;
  bfloat16 *inv_x;
  B_x = (unsigned int *)&x;
  B_Q = *B_x + mantissa_Q;
  exponent = (B_Q & exp_mask) >> 23;
  mantissa = (B_Q & mantissa_mask) >> 16;
  inv_exponent = (mantissa == 0) + (253 - exponent);
  inv_x_val = (inv_exponent << 7) + m_inv_lut[mantissa];
  inv_x = (bfloat16 *)&inv_x_val;
  return *inv_x;
}

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
    // NOT static: l is fully overwritten (store_v(l, zero) below) before use, so
    // it needn't persist; non-static keeps it off .bss (Peano CDO loader won't
    // zero it) and off the tight .data region.
    alignas(aie::vector_decl_align) AIE_TRANSIENT float l[8] = {};
    const aie::vector<float, 8> zero = aie::broadcast<float, 8>(0);

    // Persistent ping-pong toggles: force to .data (file-backed) so their initial
    // value survives load under Peano. (chess is fine.)
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
        _lock_acquire(v_cons_lock);
        is_v_ping = !is_v_ping;
        bf16 *v_up = is_v_ping ? v_ping : v_pong;
        attn_fv<DH / 8, GQA_R, GQA_S, GQA_T>(s, v_up, y, true);
        _lock_release(v_prod_lock);
        _lock_acquire(v_cons_lock);
        is_v_ping = !is_v_ping;
        bf16 *v_down = is_v_ping ? v_ping : v_pong;
        attn_fv<DH / 8, GQA_R, GQA_S, GQA_T>(s, v_down, y, false);
        _lock_release(v_prod_lock);
        if(is_sc_ping){
            _down_lock_release(s_prod_lock_0);
        } else {
            _down_lock_release(s_prod_lock_1);
        }
        i -= 16;
    } while (i > 0);
    passThrough_aie<y_acc_dtype, bf16, 8 * DH>(y, (bf16*)y);
    _lock_acquire(o_prod_lock, 1);
    scale_div_aie<8 * DH>((bf16*)y, o, l);
    // passThrough_aie<bf16, bf16, 8 * DH>((bf16*)y, o);
    _lock_release(o_cons_lock, 1);
}
}


template <typename T_in, typename T_out, int N>
void passThrough_aie(T_in *restrict in0, T_out *restrict out) {
    
    const int vec_factor = 16;

    aie::vector<T_out, vec_factor> Out;

    const int F = N / vec_factor;
    for (int i = 0; i < F; i++) {
        aie::accum<accfloat, vec_factor> c_acc_in;
        c_acc_in.from_vector(aie::load_v<vec_factor>(in0));
        aie::store_v(out, c_acc_in.template to_vector<T_out>());
        in0 += vec_factor;
        out += vec_factor;
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
void attn_fv(bf16 *__restrict pS, bf16 *__restrict pV, y_acc_dtype *__restrict pY, bool is_up) {

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

        if(is_up) {
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
void scale_div_aie(bf16 *a, bf16* o, float *l) {
    
    constexpr int vec_factor = 64;
    const int F = N / vec_factor;

    bf16* pA1 = a;
    bf16* pO1 = o;
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

    auto L = aie::concat(L0, L1, L2, L3, L4, L5, L6, L7);

    for (int d = 0; d < N / vec_factor; d++){
        aie::vector<bf16, vec_factor> A0 = aie::load_v<vec_factor>(pA1);
        aie::accum<accfloat, vec_factor> AL = aie::mul(A0, L);
        aie::vector<bf16, vec_factor> AL_bf16 = AL.template to_vector<bf16>();
        // for (int g = 0; g < KV_HEADS_PER_CU; g++) chess_flatten_loop{
        //     int v_offset = g * 4;
        //     for (int h = 0; h < Q_HEADS_PER_GROUP; h++) {
        //         aie::vector<bf16, 8> temp = AL_bf16.extract<8>(v_offset + h);
        //         aie::store_v(pO1, temp);
        //         pO1 += 8;
        //     }
        // }
        aie::store_v(pO1, AL_bf16);
        pO1 += vec_factor;
        pA1 += vec_factor;
    }

}

#elif ATTN_IMPL == ATTN_IMPL_1x8x1

void calculate_l(bf16 * pS, float * c, float * l) {
    using MMUL = aie::mmul<GQA_R, GQA_S, GQA_T, bf16, bf16, accfloat>;
    bf16 *__restrict pS1 = pS;
   
    aie::vector<bf16, 64> S_up = aie::load_v<64>(pS1);
    pS1 += 64;
    aie::vector<bf16, 64> S_down = aie::load_v<64>(pS1);
    pS1 += 64;

    aie::vector<bf16, 32> S10 = aie::filter_even(S_up, 8);
    aie::vector<bf16, 32> S11 = aie::filter_odd(S_up, 8);
    aie::vector<bf16, 32> S20 = aie::filter_even(S_down, 8);
    aie::vector<bf16, 32> S21 = aie::filter_odd(S_down, 8);
    
    S_up = aie::concat(S10, S20);
    S_down = aie::concat(S11, S21);

    aie::vector<bf16, MMUL::size_B> ones = aie::broadcast<bf16, MMUL::size_B>(1.0f);

    aie::vector<bf16, MMUL::size_C> acc_C00 = aie::zeros<bf16, MMUL::size_C>(); 

    MMUL C00(acc_C00);

    C00.mac(S_up, ones);
    C00.mac(S_down, ones);

    aie::vector<float, MMUL::size_C> sumf = C00.template to_vector<float>();
    auto sum03 = aie::filter_even(sumf, 4);
    auto sum02 = aie::filter_even(sum03, 2);
    auto sum01 = aie::filter_even(sum02, 1);
    aie::accum<accfloat, 8> sum;
    sum.from_vector(sum01);

    aie::vector<float, 8> c_float32 = aie::load_v<8>(c);
    aie::vector<float, 8> l_float32 = aie::load_v<8>(l);
    sum = aie::mac(sum, c_float32, l_float32);
    aie::store_v(l, sum.template to_vector<float>());
    // bf16 *__restrict pS1 = pS;
    // for (int i = 0; i < 8; i++) {
    //     aie::vector<bf16, 16> s_vec = aie::load_v<16>(pS1);
    //     float sum = aie::reduce_add(s_vec);
    //     l[i] = sum + l[i] * c[i];
    //     pS1 += 16;
    // }
}

typedef float y_acc_dtype;
template <unsigned colQ, unsigned r, unsigned s, unsigned t>
void attn_fv(bf16 *__restrict pS, bf16 *__restrict pV, y_acc_dtype *__restrict pY, float *__restrict c) {

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
void scale_div_aie(bf16 *a, bf16* o, float *l) {
    
    constexpr int vec_factor = 64;
    const int F = N / vec_factor;

    bf16* pA1 = a;
    bf16* pO1 = o;
    aie::vector<bf16, 8> L_chunks[8];
    for (int i = 0; i < 8; i++) {
        bf16 l_i = (bf16)getInvBf16(l[i]);
        L_chunks[i] = aie::broadcast<bf16, 8>(l_i);
    }
    auto L = aie::concat(L_chunks[0], L_chunks[1], L_chunks[2], L_chunks[3], L_chunks[4], L_chunks[5], L_chunks[6], L_chunks[7]);

    for (int d = 0; d < N / vec_factor; d++){
        aie::vector<bf16, vec_factor> A0 = aie::load_v<vec_factor>(pA1);
        aie::accum<accfloat, vec_factor> AL = aie::mul(A0, L);
        aie::vector<bf16, vec_factor> AL_bf16 = AL.template to_vector<bf16>();
        // aie::vector<bf16, vec_factor> AL_bf16 = aie::broadcast<bf16, vec_factor>((bf16)1.0f);
        aie::store_v(pO1, AL_bf16);
        pO1 += vec_factor;
        pA1 += vec_factor;
    }
    
}

#endif