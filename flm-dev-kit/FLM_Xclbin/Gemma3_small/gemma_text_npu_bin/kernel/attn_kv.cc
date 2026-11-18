// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include <type_traits>

#include <aie_api/aie.hpp>
#include "common.h"
#include "lut_based_ops.h"

#if !defined(__chess__)
// Not inlined under Peano: inlined, its spills stay in attn_kv's frame while attn_fv
// runs, and the call chain then needs more stack than this core has (the V buffer
// sits right above the stack).
__attribute__((noinline))
#endif
void calculate_l(bf16 *__restrict pS, float *__restrict c, float *__restrict l) {
    using MMUL = aie::mmul<4, 8, 8, bf16, bf16, accfloat>;
    bf16 *__restrict pS1 = pS;
   
    aie::vector<bf16, 64> S_up = aie::load_v<64>(pS1);
    pS1 += 64;

    aie::vector<bf16, MMUL::size_A> S0 = aie::filter_even(S_up, 8);
    aie::vector<bf16, MMUL::size_A> S1 = aie::filter_odd(S_up, 8);
    aie::vector<bf16, MMUL::size_B> ones = aie::broadcast<bf16, MMUL::size_B>(1);

    aie::vector<bf16, MMUL::size_C> acc_C00 = aie::zeros<bf16, MMUL::size_C>(); 

    MMUL C00(acc_C00);

    C00.mac(S0, ones);
    C00.mac(S1, ones);

    aie::vector<float, MMUL::size_C> sumf = C00.template to_vector<float>();
    auto sum03 = aie::filter_even(sumf, 4);
    auto sum02 = aie::filter_even(sum03, 2);
    auto sum01 = aie::filter_even(sum02, 1);

    float sum0 = sum01[0];
    float sum1 = sum01[1];
    float sum2 = sum01[2];
    float sum3 = sum01[3];

    l[0] = l[0] * c[0] + sum0;
    l[1] = l[1] * c[1] + sum1;
    l[2] = l[2] * c[2] + sum2;
    l[3] = l[3] * c[3] + sum3;
}

typedef float y_acc_dtype;
template <unsigned colQ, unsigned r, unsigned s, unsigned t>
void attn_fv(bf16 *__restrict pS, bf16 *__restrict pV, y_acc_dtype *__restrict pY, float *__restrict c) {

    using MMUL = aie::mmul<r, s, t, bf16, bf16, accfloat>;

    event0();

    y_acc_dtype *__restrict pY1 = pY;
    bf16 *__restrict pS1 = pS;
   
    aie::vector<bf16, 64> S_up = aie::load_v<64>(pS1);
    pS1 += 64;

    aie::vector<bf16, MMUL::size_A> S0 = aie::filter_even(S_up, 8);
    aie::vector<bf16, MMUL::size_A> S1 = aie::filter_odd(S_up, 8);
    
    aie::vector<y_acc_dtype, 8> c0 = aie::broadcast<float, 8>(c[0]);
    aie::vector<y_acc_dtype, 8> c1 = aie::broadcast<float, 8>(c[1]);
    aie::vector<y_acc_dtype, 8> c2 = aie::broadcast<float, 8>(c[2]);
    aie::vector<y_acc_dtype, 8> c3 = aie::broadcast<float, 8>(c[3]); 
    aie::vector<y_acc_dtype, 32> CORRECT = aie::concat(c0, c1, c2, c3);

    bf16 *__restrict pV1 = pV;

	for (unsigned j = 0; j < colQ; j += 4)
    chess_prepare_for_pipelining chess_loop_range(2, ) {
        //v0
        bf16 *__restrict pV01 = pV1 + (j + 0) * MMUL::size_B;
        bf16 *__restrict pV02 = pV1 + (j + 1) * MMUL::size_B;

        aie::vector<bf16, MMUL::size_B> V00 = aie::load_v<MMUL::size_B>(pV01);
        pV01 += MMUL::size_B * colQ;
        aie::vector<bf16, MMUL::size_B> V01 = aie::load_v<MMUL::size_B>(pV02);
        pV02 += MMUL::size_B * colQ;

        aie::vector<y_acc_dtype, MMUL::size_C> acc_Y00 = aie::load_v<MMUL::size_C>(pY1);
        aie::vector<y_acc_dtype, MMUL::size_C> acc_Y01 = aie::load_v<MMUL::size_C>(pY1 + MMUL::size_C);

        aie::accum<accfloat, MMUL::size_C> ACC_Y00;
        aie::accum<accfloat, MMUL::size_C> ACC_Y01;
        ACC_Y00 = aie::mul(CORRECT, acc_Y00);
        ACC_Y01 = aie::mul(CORRECT, acc_Y01);
        
        MMUL Y00(ACC_Y00);
        MMUL Y01(ACC_Y01);


        Y00.mac(S0, V00);
        Y01.mac(S0, V01);

        V00 = aie::load_v<MMUL::size_B>(pV01);
        V01 = aie::load_v<MMUL::size_B>(pV02);

        Y00.mac(S1, V00);
        Y01.mac(S1, V01);

        aie::store_v(pY1, Y00.template to_vector<y_acc_dtype>());
        pY1 += MMUL::size_C;

        aie::store_v(pY1, Y01.template to_vector<y_acc_dtype>());
        pY1 += MMUL::size_C;

        bf16 *__restrict pV03 = pV1 + (j + 2) * MMUL::size_B;
        bf16 *__restrict pV04 = pV1 + (j + 3) * MMUL::size_B;

        aie::vector<bf16, MMUL::size_B> V02 = aie::load_v<MMUL::size_B>(pV03);
        pV03 += MMUL::size_B * colQ;
        aie::vector<bf16, MMUL::size_B> V03 = aie::load_v<MMUL::size_B>(pV04);
        pV04 += MMUL::size_B * colQ;

        aie::vector<y_acc_dtype, MMUL::size_C> acc_Y02 = aie::load_v<MMUL::size_C>(pY1);
        aie::vector<y_acc_dtype, MMUL::size_C> acc_Y03 = aie::load_v<MMUL::size_C>(pY1 + MMUL::size_C);

        aie::accum<accfloat, MMUL::size_C> ACC_Y02;
        aie::accum<accfloat, MMUL::size_C> ACC_Y03;
        ACC_Y02 = aie::mul(CORRECT, acc_Y02);
        ACC_Y03 = aie::mul(CORRECT, acc_Y03);
    
        
        MMUL Y02(ACC_Y02);
        MMUL Y03(ACC_Y03);

        Y02.mac(S0, V02);
        Y03.mac(S0, V03);

        V02 = aie::load_v<MMUL::size_B>(pV03);
        V03 = aie::load_v<MMUL::size_B>(pV04);

        Y02.mac(S1, V02);
        Y03.mac(S1, V03);

        aie::store_v(pY1, Y02.template to_vector<y_acc_dtype>());
        pY1 += MMUL::size_C;

        aie::store_v(pY1, Y03.template to_vector<y_acc_dtype>());
        pY1 += MMUL::size_C;
    }
    event1();
}

template <unsigned N>
void scale_div_aie(float *a, bf16* o, float *l) {
    event0();
    constexpr int vec_factor = 32;
    const int F = N / vec_factor;

    float* pA1 = a;
    bf16* pO1 = o;
    bf16 l00 = (bf16)getInvBf16(l[0]);
    aie::vector<bf16, 8> L0 = aie::broadcast<bf16, 8>(l00);
    bf16 l01 = (bf16)getInvBf16(l[1]);
    aie::vector<bf16, 8> L1 = aie::broadcast<bf16, 8>(l01);
    bf16 l02 = (bf16)getInvBf16(l[2]);
    aie::vector<bf16, 8> L2 = aie::broadcast<bf16, 8>(l02);
    bf16 l03 = (bf16)getInvBf16(l[3]);
    aie::vector<bf16, 8> L3 = aie::broadcast<bf16, 8>(l03);

    auto L = aie::concat(L0, L1, L2, L3);

    for (int d = 0; d < N / vec_factor; d++){
        aie::vector<float, vec_factor> A0 = aie::load_v<vec_factor>(pA1);
        aie::accum<accfloat, vec_factor> A0_acc;
        A0_acc.from_vector(A0);
        aie::accum<accfloat, vec_factor> AL = aie::mul(A0_acc.template to_vector<bf16>(), L);
        aie::vector<bf16, vec_factor> AL_bf16 = AL.template to_vector<bf16>();
        aie::store_v(pO1, AL_bf16);
        pO1 += vec_factor;
        pA1 += vec_factor;
    }
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
    alignas(aie::vector_decl_align) static y_acc_dtype y[4 * DH] = {};
    alignas(aie::vector_decl_align) static float l[8] = {};
    const aie::vector<float, 8> zero = aie::broadcast<float, 8>(0);

    static AIE_PING_FLAG is_svc_ping = false;
    static AIE_PING_FLAG is_lo_ping = false;
#if defined(__chess__)
    bool &svc = is_svc_ping;
    bool &lo = is_lo_ping;
#else
    // The flags are kept in locals for the call and written back at the end.
    bool svc = is_svc_ping;
    bool lo = is_lo_ping;
#endif
    zero_vectorized<y_acc_dtype, 4*DH>(y);
    aie::store_v(l, zero);
    _down_lock_acquire(l_cons_lock);
    AIE_LOCK_LOOP_CTR i = L[0];
    do {
        _down_lock_acquire(s_cons_lock);
        _lock_acquire(v_cons_lock);
        svc = !svc;
        bf16 *s = svc ? s_ping : s_pong;
        bf16 *v = svc ? v_ping : v_pong;
        float *c = (float *)(s + 4 * 16);
        calculate_l(s, c, l);
        attn_fv<DH / 8, 4, 8, 8>(s, v, y, c);
        _down_lock_release(s_prod_lock);
        _lock_release(v_prod_lock);
        i -= 16;
    } while (i > 0);

    _lock_acquire(o_prod_lock);
    lo = !lo;
    bf16 *o = lo ? o_ping : o_pong;
    scale_div_aie<4*DH>(y, o, l);
    _lock_release(o_cons_lock);
#if !defined(__chess__)
    is_svc_ping = svc;
    is_lo_ping = lo;
#endif
}
}
