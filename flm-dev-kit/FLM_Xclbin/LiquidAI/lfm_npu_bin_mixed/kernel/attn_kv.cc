// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#include <type_traits>

#include <aie_api/aie.hpp>
#include "common.h"
#include "lut_based_ops.h"

#if !defined(__chess__)
__attribute__((noinline))
#endif
void calculate_l(bf16 *__restrict pS, float *__restrict c, float *__restrict l) {
    using MMUL = aie::mmul<8, 8, 8, bf16, bf16, accfloat>;
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
    aie::vector<bf16, MMUL::size_B> ones = aie::broadcast<bf16, MMUL::size_B>(1);

    aie::vector<bf16, MMUL::size_C> acc_C00 = aie::zeros<bf16, MMUL::size_C>(); 

    MMUL C00(acc_C00);

    C00.mac(S0, ones);
    C00.mac(S1, ones);

    aie::vector<float, MMUL::size_C> sumf = C00.template to_vector<float>();
    auto sum03 = aie::filter_even(sumf, 4);
    auto sum02 = aie::filter_even(sum03, 2);
    auto sum01 = aie::filter_even(sum02, 1);
    auto sum00 = aie::concat(sum01, sum01);
    aie::accum<accfloat, 16> sum;
    sum.from_vector(sum00);

    aie::vector<float, 8> c_float32 = aie::load_v<8>(c);
    aie::vector<float, 16> c_in = aie::concat(c_float32, c_float32);
    
    aie::vector<float, 8> l_float32 = aie::load_v<8>(l);
    aie::vector<float, 16> l_in = aie::concat(l_float32, l_float32);

    aie::accum<accfloat, 16> l_out;
    l_out = mac_elem_16_accuracy_safe(l_in, c_in, sum, 0, 0, 0);
    aie::store_v(l, l_out.template to_vector<float>());
}

typedef float y_acc_dtype;
template <unsigned colQ, unsigned r, unsigned s, unsigned t>
#if !defined(__chess__)
__attribute__((noinline))
#endif
void attn_fv(bf16 *__restrict pS, bf16 *__restrict pV, y_acc_dtype *__restrict pY, float *__restrict c) {

    using MMUL = aie::mmul<r, s, t, bf16, bf16, accfloat>;

    event0();

#if !defined(__chess__)
    {
    bf16 *__restrict pV1 = pV;
    bf16 *__restrict pV2 = pV + MMUL::size_B * colQ;
    // Peano: the natural one-loop-two-planes form keeps four BFP16 accumulators live
    // AND the 64-wide CORRECT vector live across the whole loop; peano spills them and
    // a residual cross-regfile spill/reload defect corrupts iterations (bug B) ->
    // NaN/garbage. Restructure so only ONE zero-pressure mmul accumulator is ever live:
    //   (1) apply the online-softmax correction y *= CORRECT in a pure-vector pass
    //       (CORRECT is dead afterwards, freeing registers);
    //   (2) plane-1 pass: one mmul seeded from the corrected acc, store its lower half;
    //   (3) plane-2 pass: same, store its upper half (plane-1 only wrote the lower half).
    {
        aie::vector<bf16, 64> S_up = aie::load_v<64>(pS);
        aie::vector<bf16, 64> S_down = aie::load_v<64>(pS + 64);
        aie::vector<bf16, MMUL::size_A> S0 = aie::concat(aie::filter_even(S_up, 8), aie::filter_even(S_down, 8));
        aie::vector<bf16, MMUL::size_A> S1 = aie::concat(aie::filter_odd(S_up, 8), aie::filter_odd(S_down, 8));
        // (1) correction pass
        {
            aie::vector<y_acc_dtype, 64> CORRECT = aie::concat(
                aie::broadcast<float, 8>(c[0]), aie::broadcast<float, 8>(c[1]),
                aie::broadcast<float, 8>(c[2]), aie::broadcast<float, 8>(c[3]),
                aie::broadcast<float, 8>(c[4]), aie::broadcast<float, 8>(c[5]),
                aie::broadcast<float, 8>(c[6]), aie::broadcast<float, 8>(c[7]));
            for (unsigned j = 0; j < colQ; j += 1) {
                y_acc_dtype *__restrict pYc = pY + j * MMUL::size_C;
                aie::store_v(pYc, aie::mul(CORRECT, aie::load_v<MMUL::size_C>(pYc)).template to_vector<y_acc_dtype>());
            }
        }
        // (2) plane 1 -> lower 32 lanes
        for (unsigned j = 0; j < colQ; j += 1) {
            y_acc_dtype *__restrict pYc = pY + j * MMUL::size_C;
            bf16 *__restrict pVp = pV1 + j * MMUL::size_B;
            aie::vector<bf16, MMUL::size_B> Vp = aie::load_v<MMUL::size_B>(pVp);
            MMUL Ya(aie::load_v<MMUL::size_C>(pYc));
            Ya.mac(S0, Vp);
            Vp = aie::load_v<MMUL::size_B>(pVp + MMUL::size_B * colQ * 2);
            Ya.mac(S1, Vp);
            aie::store_v(pYc, Ya.template to_vector<y_acc_dtype>().template extract<32>(0));
        }
        // (3) plane 2 -> upper 32 lanes
        for (unsigned j = 0; j < colQ; j += 1) {
            y_acc_dtype *__restrict pYc = pY + j * MMUL::size_C;
            bf16 *__restrict pVp = pV2 + j * MMUL::size_B;
            aie::vector<bf16, MMUL::size_B> Vp = aie::load_v<MMUL::size_B>(pVp);
            MMUL Yb(aie::load_v<MMUL::size_C>(pYc));
            Yb.mac(S0, Vp);
            Vp = aie::load_v<MMUL::size_B>(pVp + MMUL::size_B * colQ * 2);
            Yb.mac(S1, Vp);
            aie::store_v(pYc + 32, Yb.template to_vector<y_acc_dtype>().template extract<32>(1));
        }
    }
    event1();
    return;
    }
#endif

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
    event1();
}

template <unsigned N>
#if !defined(__chess__)
__attribute__((noinline))
#endif
void scale_div_aie(bf16 *a, bf16* o, float *l) {
    event0();
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

    #if MODEL_TYPE == LLAMA_3B
    for (int d = 0; d < N / vec_factor; d++){
        aie::vector<bf16, vec_factor> A0 = aie::load_v<vec_factor>(pA1);
        aie::accum<accfloat, vec_factor> AL = aie::mul(A0, L);
        aie::vector<bf16, vec_factor> AL_bf16 = AL.template to_vector<bf16>();
        aie::vector<bf16, 8> temp = AL_bf16.extract<8>(0);
        aie::store_v(pO1, temp);
        pO1 += 8;
        temp = AL_bf16.extract<8>(1);
        aie::store_v(pO1, temp);
        pO1 += 8;
        temp = AL_bf16.extract<8>(2);
        aie::store_v(pO1, temp);
        pO1 += 8;
        temp = AL_bf16.extract<8>(4);
        aie::store_v(pO1, temp);
        pO1 += 8;
        temp = AL_bf16.extract<8>(5);
        aie::store_v(pO1, temp);
        pO1 += 8;
        temp = AL_bf16.extract<8>(6);
        aie::store_v(pO1, temp);
        pO1 += 8;
        pA1 += vec_factor;
    }
    #else
    for (int i = 0; i < F ; i++) {
        aie::vector<bf16, vec_factor> A0 = aie::load_v<vec_factor>(pA1);
        aie::accum<accfloat, vec_factor> AL = aie::mul(A0, L);
        aie::vector<bf16, vec_factor> AL_bf16 = AL.template to_vector<bf16>();
        aie::store_v(pO1, AL_bf16);
        pO1 += vec_factor;
        pA1 += vec_factor;
    }
    #endif

    event1();
}
constexpr int s_prod_lock = 4;
constexpr int s_cons_lock = 5;
constexpr int v_prod_lock = 2;
constexpr int v_cons_lock = 3;
constexpr int o_prod_lock = 0;
constexpr int o_cons_lock = 1;  
// constexpr int l_prod_lock = 6;
constexpr int l_cons_lock = 7;

extern "C"{

void attn_kv(
    bf16 *__restrict s_ping,
    bf16 *__restrict s_pong,
    bf16 *__restrict v_ping,
    bf16 *__restrict v_pong,
    bf16 *__restrict o_ping,
    bf16 *__restrict o_pong,
    int* L
)
{
    // Peano's CDO loader does not zero .bss; keep zero-init scratch and ping/pong
    // selectors in .data so they are actually zeroed at load.
#if !defined(__chess__)
    alignas(aie::vector_decl_align) static __attribute__((section(".data"))) y_acc_dtype y[8 * DH] = {};
    alignas(aie::vector_decl_align) static __attribute__((section(".data"))) bf16 y_bf16[8 * DH] = {};
    alignas(aie::vector_decl_align) static __attribute__((section(".data"))) float l[16] = {};
#else
    alignas(aie::vector_decl_align) static y_acc_dtype y[8 * DH] = {};
    alignas(aie::vector_decl_align) static bf16 y_bf16[8 * DH] = {};
    alignas(aie::vector_decl_align) static float l[16] = {};
#endif
    const aie::vector<float, 16> zero = aie::broadcast<float, 16>(0);

#if !defined(__chess__)
    __attribute__((section(".data")))
#endif
    static bool is_svc_ping = false;
#if !defined(__chess__)
    __attribute__((section(".data")))
#endif
    static bool is_lo_ping = false;

#if defined(__chess__)
    bool &svc = is_svc_ping;
    bool &lo = is_lo_ping;
#else
    // Peano miscompiles RMW on persisted .data bools; keep ping/pong state in
    // stack locals, sync to .data once at entry/exit.
    bool svc = is_svc_ping;
    bool lo = is_lo_ping;
#endif
    zero_vectorized<y_acc_dtype, 8*DH>(y);
    aie::store_v(l, zero);
    _down_lock_acquire(l_cons_lock);
    AIE_LOCK_LOOP_CTR i = L[0];
    do {
        _down_lock_acquire(s_cons_lock);
        _lock_acquire(v_cons_lock);
        svc = !svc;
        bf16 *s = svc ? s_ping : s_pong;
        bf16 *v = svc ? v_ping : v_pong;
        float *c = (float *)(s + 8 * 16);
        calculate_l(s, c, l);
        attn_fv<DH / 8, 8, 8, 8>(s, v, y, c);
        _down_lock_release(s_prod_lock);
        _lock_release(v_prod_lock);
        i -= 16;
    } while (i > 0);
    passThrough_aie<y_acc_dtype, bf16, 8*DH>(y, y_bf16);
    lo = !lo;
    bf16 *o = lo ? o_ping : o_pong;
    _lock_acquire(o_prod_lock);
    scale_div_aie<8*DH>(y_bf16, o, l);
    _lock_release(o_cons_lock);
#if !defined(__chess__)
    is_svc_ping = svc;
    is_lo_ping = lo;
#endif
}
}
