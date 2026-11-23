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
    using MMUL = aie::mmul<4, 8, 8, bf16, bf16, accfloat>;
    bf16 *__restrict pS1 = pS;
   
    aie::vector<bf16, 32> S_up = aie::load_v<32>(pS1);
    pS1 += 32;

    aie::vector<bf16, 16> S10 = aie::filter_even(S_up, 8);
    aie::vector<bf16, 16> S11 = aie::filter_odd(S_up, 8);
    aie::vector<bf16, 16> zeros = aie::zeros<bf16, 16>();
    
    aie::vector<bf16, MMUL::size_A> S0 = aie::concat(S10, zeros);
    aie::vector<bf16, MMUL::size_A> S1 = aie::concat(S11, zeros);
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

    l[0] = l[0] * c[0] + sum0;
    l[1] = l[1] * c[1] + sum1;
}

typedef float y_acc_dtype;
template <unsigned colQ, unsigned r, unsigned s, unsigned t>
#if !defined(__chess__)
__attribute__((noinline))
#endif
void attn_fv(bf16 *__restrict pS, bf16 *__restrict pV, y_acc_dtype *__restrict pY, float *__restrict c) {

    using MMUL = aie::mmul<r, s, t, bf16, bf16, accfloat>;

    event0();

    bf16 *__restrict pV1 = pV;

#if !defined(__chess__)
    // Peano: the natural j+=4 form keeps FOUR BFP16 mmul accumulators (Y00..Y03)
    // AND the 32-wide CORRECT vector live across the loop; peano spills the
    // accumulators and a cross-regfile spill/reload defect corrupts iterations
    // (bug B, AIESpillSlotOptimization family). Restructure so nothing spills:
    //   (1) online-softmax correction y *= CORRECT in a pure-vector pass (CORRECT
    //       dead afterwards); the valid plane is 16-wide so CORRECT16 = (c0,c1);
    //   (2) mac pass: exactly ONE zero-pressure mmul accumulator seeded from the
    //       corrected acc (concat zeros to fill the dead upper plane), output the
    //       lower 16 lanes via filter_even radix 16 (same layout as chess).
    {
        aie::vector<bf16, 32> S_up = aie::load_v<32>(pS);
        aie::vector<bf16, 16> S10 = aie::filter_even(S_up, 8);
        aie::vector<bf16, 16> S11 = aie::filter_odd(S_up, 8);
        aie::vector<bf16, 16> zeros_vec = aie::zeros<bf16, 16>();
        aie::vector<bf16, MMUL::size_A> S0 = aie::concat(S10, zeros_vec);
        aie::vector<bf16, MMUL::size_A> S1 = aie::concat(S11, zeros_vec);
        aie::vector<y_acc_dtype, 16> zeros_float = aie::zeros<y_acc_dtype, 16>();
        // (1) correction pass
        {
            aie::vector<y_acc_dtype, 16> CORRECT16 = aie::concat(
                aie::broadcast<float, 8>(c[0]), aie::broadcast<float, 8>(c[1]));
            for (unsigned j = 0; j < colQ; j += 1) {
                y_acc_dtype *__restrict pYc = pY + j * 16;
                aie::store_v(pYc, aie::mul(CORRECT16, aie::load_v<16>(pYc)).template to_vector<y_acc_dtype>());
            }
        }
        // (2) mac pass, single accumulator
        for (unsigned j = 0; j < colQ; j += 1) {
            y_acc_dtype *__restrict pYc = pY + j * 16;
            bf16 *__restrict pVp = pV1 + j * MMUL::size_B;
            MMUL Y(aie::concat(aie::load_v<16>(pYc), zeros_float));
            aie::vector<bf16, MMUL::size_B> Vp = aie::load_v<MMUL::size_B>(pVp);
            Y.mac(S0, Vp);
            Vp = aie::load_v<MMUL::size_B>(pVp + MMUL::size_B * colQ);
            Y.mac(S1, Vp);
            aie::store_v(pYc, aie::filter_even(Y.template to_vector<y_acc_dtype>(), 16));
        }
    }
#else
    y_acc_dtype *__restrict pY1 = pY;
    bf16 *__restrict pS1 = pS;

    aie::vector<bf16, 32> S_up = aie::load_v<32>(pS1);
    pS1 += 32;

    aie::vector<bf16, 16> S10 = aie::filter_even(S_up, 8);
    aie::vector<bf16, 16> S11 = aie::filter_odd(S_up, 8);
    aie::vector<bf16, 16> zeros_vec = aie::zeros<bf16, 16>();

    aie::vector<bf16, MMUL::size_A> S0 = aie::concat(S10, zeros_vec);
    aie::vector<bf16, MMUL::size_A> S1 = aie::concat(S11, zeros_vec);

    aie::vector<y_acc_dtype, 8> c0 = aie::broadcast<float, 8>(c[0]);
    aie::vector<y_acc_dtype, 8> c1 = aie::broadcast<float, 8>(c[1]);
    aie::vector<y_acc_dtype, 8> c2 = aie::broadcast<float, 8>(c[2]);
    aie::vector<y_acc_dtype, 8> c3 = aie::broadcast<float, 8>(c[3]);
    aie::vector<y_acc_dtype, 32> CORRECT = aie::concat(c0, c1, c2, c3);

    aie::vector<float, 16> zeros_float = aie::zeros<float, 16>();

	for (unsigned j = 0; j < colQ; j += 4)
    chess_prepare_for_pipelining chess_loop_range(2, ) {
        //v0
        bf16 *__restrict pV01 = pV1 + (j + 0) * MMUL::size_B;
        bf16 *__restrict pV02 = pV1 + (j + 1) * MMUL::size_B;

        aie::vector<bf16, MMUL::size_B> V00 = aie::load_v<MMUL::size_B>(pV01);
        pV01 += MMUL::size_B * colQ;
        aie::vector<bf16, MMUL::size_B> V01 = aie::load_v<MMUL::size_B>(pV02);
        pV02 += MMUL::size_B * colQ;

        aie::vector<y_acc_dtype, 16> acc_y00 = aie::load_v<16>(pY1);
        aie::vector<y_acc_dtype, 16> acc_y01 = aie::load_v<16>(pY1 + 16);
        aie::vector<y_acc_dtype, MMUL::size_C> acc_Y00 = aie::concat(acc_y00, zeros_float);
        aie::vector<y_acc_dtype, MMUL::size_C> acc_Y01 = aie::concat(acc_y01, zeros_float);

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

        auto Y0 = aie::filter_even(Y00.template to_vector<y_acc_dtype>(), 16);
        auto Y1 = aie::filter_even (Y01.template to_vector<y_acc_dtype>(), 16);

        aie::store_v(pY1, Y0);
        pY1 += 16;

        aie::store_v(pY1, Y1);
        pY1 += 16;

        bf16 *__restrict pV03 = pV1 + (j + 2) * MMUL::size_B;
        bf16 *__restrict pV04 = pV1 + (j + 3) * MMUL::size_B;

        aie::vector<bf16, MMUL::size_B> V02 = aie::load_v<MMUL::size_B>(pV03);
        pV03 += MMUL::size_B * colQ;
        aie::vector<bf16, MMUL::size_B> V03 = aie::load_v<MMUL::size_B>(pV04);
        pV04 += MMUL::size_B * colQ;

        aie::vector<y_acc_dtype, 16> acc_y02 = aie::load_v<16>(pY1);
        aie::vector<y_acc_dtype, 16> acc_y03 = aie::load_v<16>(pY1 + 16);
        aie::vector<y_acc_dtype, MMUL::size_C> acc_Y02 = aie::concat(acc_y02, zeros_float);
        aie::vector<y_acc_dtype, MMUL::size_C> acc_Y03 = aie::concat(acc_y03, zeros_float);

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

        auto Y2 = aie::filter_even(Y02.template to_vector<y_acc_dtype>(), 16);
        auto Y3 = aie::filter_even(Y03.template to_vector<y_acc_dtype>(), 16);

        aie::store_v(pY1, Y2);
        pY1 += 16;

        aie::store_v(pY1, Y3);
        pY1 += 16;
    }
#endif
    event1();
}

template <unsigned N>
#if !defined(__chess__)
__attribute__((noinline))
#endif
void scale_div_aie(bf16 *a, bf16* o, float *l) {
    event0();
    constexpr int vec_factor = 16;
    const int F = N / vec_factor;

    bf16* pA1 = a;
    bf16* pO1 = o;
    bf16 l00 = (bf16)getInvBf16(l[0]);
    aie::vector<bf16, 8> L0 = aie::broadcast<bf16, 8>(l00);
    bf16 l01 = (bf16)getInvBf16(l[1]);
    aie::vector<bf16, 8> L1 = aie::broadcast<bf16, 8>(l01);

    auto L = aie::concat(L0, L1);

    for (int d = 0; d < N / vec_factor; d++){
        aie::vector<bf16, vec_factor> A0 = aie::load_v<vec_factor>(pA1);
        aie::accum<accfloat, vec_factor> AL = aie::mul(A0, L);
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
    alignas(aie::vector_decl_align) static y_acc_dtype y[2 * DH] = {};
    alignas(aie::vector_decl_align) static bf16 y_bf16[2 * DH] = {};
    alignas(aie::vector_decl_align) static float l[8] = {};
    const aie::vector<float, 8> zero = aie::broadcast<float, 8>(0);

    // Peano's CDO loader does not zero .bss; keep ping/pong selectors in .data.
#if !defined(__chess__)
    __attribute__((section(".data")))
#endif
    static bool is_svc_ping = false;
#if !defined(__chess__)
    __attribute__((section(".data")))
#endif
    static bool is_lo_ping = false;
    // Peano miscompiles read-modify-write on persisted .data bools across the loop;
    // keep ping/pong state in stack locals, sync to .data once at entry/exit.
    bool svc = is_svc_ping;
    bool lo = is_lo_ping;
    zero_vectorized<y_acc_dtype, 2*DH>(y);
    aie::store_v(l, zero);
    _down_lock_acquire(l_cons_lock);
    AIE_LOCK_LOOP_CTR i = L[0];
    do {
        _down_lock_acquire(s_cons_lock);
        _lock_acquire(v_cons_lock);
        svc = !svc;
        bf16 *s = svc ? s_ping : s_pong;
        bf16 *v = svc ? v_ping : v_pong;
        float *c = (float *)(s + 2 * 16);
        calculate_l(s, c, l);
        attn_fv<DH / 8, 4, 8, 8>(s, v, y, c);
        _down_lock_release(s_prod_lock);
        _lock_release(v_prod_lock);
        i -= 16;
    } while (i > 0);
    passThrough_aie<y_acc_dtype, bf16, 2*DH>(y, y_bf16);
    lo = !lo;
    bf16 *o = lo ? o_ping : o_pong;
    is_svc_ping = svc;
    is_lo_ping = lo;
    _lock_acquire(o_prod_lock);
    scale_div_aie<2*DH>(y_bf16, o, l);
    _lock_release(o_cons_lock);
}
}
