// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include "lut_based_ops.h"
#include "aie_kernel_utils.h"
#include "aie_array_layout.h"

typedef float y_acc_dtype;
void calculate_l(bf16 *__restrict pS, float *__restrict c, float *__restrict l);
template <unsigned colQ, unsigned r, unsigned s, unsigned t>
void attn_fv(bf16 *__restrict pS, bf16 *__restrict pV, y_acc_dtype *__restrict pY, float *__restrict c);
template <unsigned N>
void scale_div_aie(bf16 *a, bf16* o, float *l);
template <typename T_in, typename T_out, int N>
void passThrough_aie(T_in *restrict in0, T_out *restrict out);


constexpr int s_prod_lock = 4;
constexpr int s_cons_lock = 5;
constexpr int v_prod_lock = 2;
constexpr int v_cons_lock = 3;
constexpr int o_prod_lock = 0;
constexpr int o_cons_lock = 1;  
// constexpr int l_prod_lock = 6;
constexpr int l_cons_lock = 8;

extern "C"{

void swa_attn_kv(
    bf16 *__restrict s_ping,
    bf16 *__restrict s_pong,
    bf16 *__restrict v_ping,
    bf16 *__restrict v_pong,
    float*__restrict y,
    bf16 *__restrict o,
    int* L
)
{
    alignas(aie::vector_decl_align) static float l[16] = {};
    const aie::vector<float, 16> zero = aie::broadcast<float, 16>(0);

    AIE_PERSIST_ZERO static bool is_svc_ping = false;
    zero_vectorized<y_acc_dtype, 8*SWA_DH>(y);
    aie::store_v(l, zero);
    _down_lock_acquire(l_cons_lock);
    AIE_LOCK_LOOP_CTR i = L[0];
    do {
        _down_lock_acquire(s_cons_lock);
        _lock_acquire(v_cons_lock);
        is_svc_ping = !is_svc_ping;
        bf16 *s = is_svc_ping ? s_ping : s_pong;
        bf16 *v = is_svc_ping ? v_ping : v_pong;
        float *c = (float *)(s + Q_HEADS_PADDED_PER_CU * 16);
        calculate_l(s, c, l);
        attn_fv<SWA_DH / 8, GQA_R, GQA_S, GQA_T>(s, v, y, c);
        _down_lock_release(s_prod_lock);
        _lock_release(v_prod_lock);
        i -= 16;
    } while (i > 0);
    passThrough_aie<y_acc_dtype, bf16, 8*SWA_DH>(y, (bf16*)y);
    _lock_acquire(o_prod_lock, O_DOWN_REPEATS);
    scale_div_aie<8 * SWA_DH>((bf16*)y, o, l);
    _lock_release(o_cons_lock, O_DOWN_REPEATS);
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
void calculate_l(bf16 *__restrict pS, float *__restrict c, float *__restrict l) {
        bf16 *__restrict pS1 = pS;
    for (int i = 0; i < 8; i++) {
        aie::vector<bf16, 16> s_vec = aie::load_v<16>(pS1);
        float sum = aie::reduce_add(s_vec);
        #if defined(__chess__)
        l[i] = sum + l[i] * c[i];
        #else
        l[i] = sum + vmulf(l[i], c[i]);
        #endif
        pS1 += 16;
    }
}

#if defined(__chess__)

template <unsigned colQ, unsigned r, unsigned s, unsigned t>
void attn_fv(bf16 *__restrict pS, bf16 *__restrict pV, y_acc_dtype *__restrict pY, float *__restrict c) {

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
    bf16 *__restrict pV2 = pV + MMUL::size_B * colQ;

	for (unsigned j = 0; j < colQ; j += 4)
    chess_prepare_for_pipelining chess_loop_range(2, ) {
        //v0
        bf16 *__restrict pV01 = pV1 + (j + 0) * MMUL::size_B;
        bf16 *__restrict pV02 = pV1 + (j + 1) * MMUL::size_B;

        aie::vector<bf16, MMUL::size_B> V00 = aie::load_v<MMUL::size_B>(pV01);
        pV01 += MMUL::size_B * colQ * 2;
        aie::vector<bf16, MMUL::size_B> V01 = aie::load_v<MMUL::size_B>(pV02);
        pV02 += MMUL::size_B * colQ * 2;

        aie::vector<y_acc_dtype, MMUL::size_C> acc_Y00 = aie::load_v<MMUL::size_C>(pY1);
        aie::vector<y_acc_dtype, MMUL::size_C> acc_Y01 = aie::load_v<MMUL::size_C>(pY1 + MMUL::size_C);

        aie::accum<accfloat, MMUL::size_C> ACC_Y00;
        aie::accum<accfloat, MMUL::size_C> ACC_Y01;
        ACC_Y00 = aie::mul(CORRECT, acc_Y00);
        ACC_Y01 = aie::mul(CORRECT, acc_Y01);
        
        MMUL Y00(ACC_Y00);
        MMUL Y01(ACC_Y01);
        MMUL Y10(ACC_Y00);
        MMUL Y11(ACC_Y01);


        Y00.mac(S0, V00);
        Y01.mac(S0, V01);

        V00 = aie::load_v<MMUL::size_B>(pV01);
        V01 = aie::load_v<MMUL::size_B>(pV02);

        Y00.mac(S1, V00);
        Y01.mac(S1, V01);

        //v1
        bf16 *__restrict pV11 = pV2 + (j + 0) * MMUL::size_B;
        bf16 *__restrict pV12 = pV2 + (j + 1) * MMUL::size_B;

        aie::vector<bf16, MMUL::size_B> V10 = aie::load_v<MMUL::size_B>(pV11);
        pV11 += MMUL::size_B * colQ * 2;
        aie::vector<bf16, MMUL::size_B> V11 = aie::load_v<MMUL::size_B>(pV12);
        pV12 += MMUL::size_B * colQ * 2;

        Y10.mac(S0, V10);
        Y11.mac(S0, V11);

        V10 = aie::load_v<MMUL::size_B>(pV11);
        V11 = aie::load_v<MMUL::size_B>(pV12);

        Y10.mac(S1, V10);
        Y11.mac(S1, V11);

        auto Y20 = aie::filter_even(Y00.template to_vector<y_acc_dtype>(), 32);
        auto Y22 = aie::filter_even(Y01.template to_vector<y_acc_dtype>(), 32);

        auto Y31 = aie::filter_odd (Y10.template to_vector<y_acc_dtype>(), 32);
        auto Y33 = aie::filter_odd (Y11.template to_vector<y_acc_dtype>(), 32);

        auto Y = aie::concat(Y20, Y31); //, Y22, Y33, Y24, Y35, Y26, Y37);
        aie::store_v(pY1, Y);
        pY1 += MMUL::size_C;

        Y = aie::concat(Y22, Y33);
        aie::store_v(pY1, Y);
        pY1 += MMUL::size_C;

        bf16 *__restrict pV03 = pV1 + (j + 2) * MMUL::size_B;
        bf16 *__restrict pV04 = pV1 + (j + 3) * MMUL::size_B;

        aie::vector<bf16, MMUL::size_B> V02 = aie::load_v<MMUL::size_B>(pV03);
        pV03 += MMUL::size_B * colQ * 2;
        aie::vector<bf16, MMUL::size_B> V03 = aie::load_v<MMUL::size_B>(pV04);
        pV04 += MMUL::size_B * colQ * 2;

        aie::vector<y_acc_dtype, MMUL::size_C> acc_Y02 = aie::load_v<MMUL::size_C>(pY1);
        aie::vector<y_acc_dtype, MMUL::size_C> acc_Y03 = aie::load_v<MMUL::size_C>(pY1 + MMUL::size_C);

        aie::accum<accfloat, MMUL::size_C> ACC_Y02;
        aie::accum<accfloat, MMUL::size_C> ACC_Y03;
        ACC_Y02 = aie::mul(CORRECT, acc_Y02);
        ACC_Y03 = aie::mul(CORRECT, acc_Y03);
    
        
        MMUL Y02(ACC_Y02);
        MMUL Y03(ACC_Y03);
        MMUL Y12(ACC_Y02);
        MMUL Y13(ACC_Y03);


        Y02.mac(S0, V02);
        Y03.mac(S0, V03);

        V02 = aie::load_v<MMUL::size_B>(pV03);
        V03 = aie::load_v<MMUL::size_B>(pV04);

        Y02.mac(S1, V02);
        Y03.mac(S1, V03);

        //v1
        bf16 *__restrict pV13 = pV2 + (j + 2) * MMUL::size_B;
        bf16 *__restrict pV14 = pV2 + (j + 3) * MMUL::size_B;

        aie::vector<bf16, MMUL::size_B> V12 = aie::load_v<MMUL::size_B>(pV13);
        pV13 += MMUL::size_B * colQ * 2;
        aie::vector<bf16, MMUL::size_B> V13 = aie::load_v<MMUL::size_B>(pV14);
        pV14 += MMUL::size_B * colQ * 2;


        Y12.mac(S0, V12);
        Y13.mac(S0, V13);

        V12 = aie::load_v<MMUL::size_B>(pV13);
        V13 = aie::load_v<MMUL::size_B>(pV14);

        Y12.mac(S1, V12);
        Y13.mac(S1, V13);

        auto Y24 = aie::filter_even(Y02.template to_vector<y_acc_dtype>(), 32);
        auto Y26 = aie::filter_even(Y03.template to_vector<y_acc_dtype>(), 32);

        auto Y35 = aie::filter_odd (Y12.template to_vector<y_acc_dtype>(), 32);
        auto Y37 = aie::filter_odd (Y13.template to_vector<y_acc_dtype>(), 32);

        Y = aie::concat(Y24, Y35); //, Y22, Y33, Y24, Y35, Y26, Y37);
        aie::store_v(pY1, Y);
        pY1 += MMUL::size_C;

        Y = aie::concat(Y26, Y37);
        aie::store_v(pY1, Y);
        pY1 += MMUL::size_C;
    }
    
}

#else  // Peano

// The pinned Peano (llvm-aie 21.0.0.2026061101+50bd3557) cannot encode the code
// this function generates for GEMMA4_E4B (attn_fv<32,8,8,8>) when
// -DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16 turns the aie::mmul into the BFP16
// path: eight live MMUL accumulators force BFP16 matrix registers to be spilled
// and the AIE2P code emitter aborts with "Register not in mBMs"
// (AIE2PMCCodeEmitterGen).  Moving one column-pair into a separate non-inlined
// function halves the live state and keeps the fast BFP16 mmul path.
// chess (ENABLE_CHESSCC=1) is unaffected -- see the #if above.
template <unsigned colQ, unsigned r, unsigned s, unsigned t>
__attribute__((noinline))
void attn_fv_pair(bf16 *__restrict pVa, bf16 *__restrict pVb,
                  y_acc_dtype *__restrict pY1,
                  const aie::vector<bf16, aie::mmul<r, s, t, bf16, bf16, accfloat>::size_A> &S0,
                  const aie::vector<bf16, aie::mmul<r, s, t, bf16, bf16, accfloat>::size_A> &S1,
                  const aie::vector<y_acc_dtype, 64> &CORRECT) {

    using MMUL = aie::mmul<r, s, t, bf16, bf16, accfloat>;

    //v0
    bf16 *__restrict pV01 = pVa;
    bf16 *__restrict pV02 = pVa + MMUL::size_B;

    aie::vector<bf16, MMUL::size_B> V00 = aie::load_v<MMUL::size_B>(pV01);
    pV01 += MMUL::size_B * colQ * 2;
    aie::vector<bf16, MMUL::size_B> V01 = aie::load_v<MMUL::size_B>(pV02);
    pV02 += MMUL::size_B * colQ * 2;

    aie::vector<y_acc_dtype, MMUL::size_C> acc_Y00 = aie::load_v<MMUL::size_C>(pY1);
    aie::vector<y_acc_dtype, MMUL::size_C> acc_Y01 = aie::load_v<MMUL::size_C>(pY1 + MMUL::size_C);

    aie::accum<accfloat, MMUL::size_C> ACC_Y00;
    aie::accum<accfloat, MMUL::size_C> ACC_Y01;
    ACC_Y00 = aie::mul(CORRECT, acc_Y00);
    ACC_Y01 = aie::mul(CORRECT, acc_Y01);

    MMUL Y00(ACC_Y00);
    MMUL Y01(ACC_Y01);
    MMUL Y10(ACC_Y00);
    MMUL Y11(ACC_Y01);

    Y00.mac(S0, V00);
    Y01.mac(S0, V01);

    V00 = aie::load_v<MMUL::size_B>(pV01);
    V01 = aie::load_v<MMUL::size_B>(pV02);

    Y00.mac(S1, V00);
    Y01.mac(S1, V01);

    //v1
    bf16 *__restrict pV11 = pVb;
    bf16 *__restrict pV12 = pVb + MMUL::size_B;

    aie::vector<bf16, MMUL::size_B> V10 = aie::load_v<MMUL::size_B>(pV11);
    pV11 += MMUL::size_B * colQ * 2;
    aie::vector<bf16, MMUL::size_B> V11 = aie::load_v<MMUL::size_B>(pV12);
    pV12 += MMUL::size_B * colQ * 2;

    Y10.mac(S0, V10);
    Y11.mac(S0, V11);

    V10 = aie::load_v<MMUL::size_B>(pV11);
    V11 = aie::load_v<MMUL::size_B>(pV12);

    Y10.mac(S1, V10);
    Y11.mac(S1, V11);

    auto Y20 = aie::filter_even(Y00.template to_vector<y_acc_dtype>(), 32);
    auto Y22 = aie::filter_even(Y01.template to_vector<y_acc_dtype>(), 32);

    auto Y31 = aie::filter_odd (Y10.template to_vector<y_acc_dtype>(), 32);
    auto Y33 = aie::filter_odd (Y11.template to_vector<y_acc_dtype>(), 32);

    auto Y = aie::concat(Y20, Y31);
    aie::store_v(pY1, Y);
    pY1 += MMUL::size_C;

    Y = aie::concat(Y22, Y33);
    aie::store_v(pY1, Y);
}

template <unsigned colQ, unsigned r, unsigned s, unsigned t>
void attn_fv(bf16 *__restrict pS, bf16 *__restrict pV, y_acc_dtype *__restrict pY, float *__restrict c) {

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
    bf16 *__restrict pV2 = pV + MMUL::size_B * colQ;

    for (unsigned j = 0; j < colQ; j += 4)
    chess_prepare_for_pipelining chess_loop_range(2, ) {
        attn_fv_pair<colQ, r, s, t>(pV1 + (j + 0) * MMUL::size_B,
                                    pV2 + (j + 0) * MMUL::size_B, pY1, S0, S1, CORRECT);
        pY1 += 2 * MMUL::size_C;
        attn_fv_pair<colQ, r, s, t>(pV1 + (j + 2) * MMUL::size_B,
                                    pV2 + (j + 2) * MMUL::size_B, pY1, S0, S1, CORRECT);
        pY1 += 2 * MMUL::size_C;
    }
}

#endif  // __chess__

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
        aie::store_v(pO1, AL_bf16);
        pO1 += vec_factor;
        pA1 += vec_factor;
    }

}


#elif ATTN_IMPL == ATTN_IMPL_1x8x1

void calculate_l(bf16 *__restrict pS, float *__restrict c, float *__restrict l) {
    bf16 *__restrict pS1 = pS;
    for (int i = 0; i < 8; i++) {
        aie::vector<bf16, 16> s_vec = aie::load_v<16>(pS1);
        float sum = aie::reduce_add(s_vec);
        #if defined(__chess__)
        l[i] = sum + l[i] * c[i];
        #else
        l[i] = sum + vmulf(l[i], c[i]);
        #endif
        pS1 += 16;
    }
    // using MMUL = aie::mmul<GQA_R, GQA_S, GQA_T, bf16, bf16, accfloat>;
    // bf16 *__restrict pS1 = pS;
   
    // aie::vector<bf16, 64> S_up = aie::load_v<64>(pS1);
    // pS1 += 64;
    // aie::vector<bf16, 64> S_down = aie::load_v<64>(pS1);
    // pS1 += 64;

    // aie::vector<bf16, 32> S10 = aie::filter_even(S_up, 8);
    // aie::vector<bf16, 32> S11 = aie::filter_odd(S_up, 8);
    // aie::vector<bf16, 32> S20 = aie::filter_even(S_down, 8);
    // aie::vector<bf16, 32> S21 = aie::filter_odd(S_down, 8);
    
    // aie::vector<bf16, MMUL::size_A> S0 = aie::concat(S10, S20);
    // aie::vector<bf16, MMUL::size_A> S1 = aie::concat(S11, S21);
    // aie::vector<bf16, MMUL::size_B> ones = aie::broadcast<bf16, MMUL::size_B>(1);

    // aie::vector<bf16, MMUL::size_C> acc_C00 = aie::zeros<bf16, MMUL::size_C>(); 

    // MMUL C00(acc_C00);

    // C00.mac(S0, ones);
    // C00.mac(S1, ones);

    // aie::vector<float, MMUL::size_C> sumf = C00.template to_vector<float>();
    // auto sum03 = aie::filter_even(sumf, 4);
    // auto sum02 = aie::filter_even(sum03, 2);
    // auto sum01 = aie::filter_even(sum02, 1);
    // auto sum00 = aie::concat(sum01, sum01);
    // aie::accum<accfloat, 16> sum;
    // sum.from_vector(sum00);

    // aie::vector<float, 8> c_float32 = aie::load_v<8>(c);
    // aie::vector<float, 16> c_in = aie::concat(c_float32, c_float32);
    
    // aie::vector<float, 8> l_float32 = aie::load_v<8>(l);
    // aie::vector<float, 16> l_in = aie::concat(l_float32, l_float32);

    // aie::accum<accfloat, 16> l_out;
    // l_out = mac_elem_16_accuracy_safe(l_in, c_in, sum, 0, 0, 0);
    // aie::store_v(l, l_out.template to_vector<float>());
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

#ifndef __chess__
    // Peano: process ONE V-column per iteration (one live MMUL accumulator).
    // The 4-wide chess version keeps 4 BFP16 matrix accumulators live, which peano
    // spills to a ~3KB stack frame; a residual defect in that cross-regfile spill/
    // reload (same AIESpillSlotOptimization path as the encoder crash) corrupts the
    // accumulator for high-pressure data and yields nan. Low pressure avoids spilling.
    for (unsigned j = 0; j < colQ; j += 1) {
        bf16 *__restrict pVa = pV1 + j * MMUL::size_B;
        aie::vector<bf16, MMUL::size_B> Va = aie::load_v<MMUL::size_B>(pVa);
        aie::vector<y_acc_dtype, MMUL::size_C> acc_ya = aie::load_v<MMUL::size_C>(pY1);
        aie::accum<accfloat, MMUL::size_C> ACC_Ya = aie::mul(CORRECT, acc_ya);
        MMUL Ya(ACC_Ya);
        Ya.mac(S0, Va);
        Va = aie::load_v<MMUL::size_B>(pVa + MMUL::size_B * colQ);
        Ya.mac(S1, Va);
        aie::store_v(pY1, Ya.template to_vector<y_acc_dtype>());
        pY1 += MMUL::size_C;
    }
#else
	for (unsigned j = 0; j < colQ; j += 4){
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
#endif

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
        aie::store_v(pO1, AL_bf16);
        pO1 += vec_factor;
        pA1 += vec_factor;
    }
    
}

#endif