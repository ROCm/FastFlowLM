// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#include <type_traits>

#include <aie_api/aie.hpp>
#include "common.h"
#include "lut_based_ops.h"

// inline aie::vector<bf16, 16> update(bf16* m, float* c, aie::vector<bf16, 16>& out, aie::mask<16>& mask, bool& is_first){
//     bf16 vm = aie::reduce_max(out);
//     bf16 vmax = aie::max(vm, *m);
//     aie::vector<bf16, 16> Vecsub = aie::sub(out, vmax);
//     v16bfloat16 v0 = Vecsub;
//     aie::accum<accfloat, 16> Outvec = getExpBf16_clamped(v0);
//     aie::vector<bf16, 16> Vec = Outvec.template to_vector<bf16>(0);
//     Vec = aie::select((bf16)0, Vec, mask);
//     bf16 correct = is_first ? (bf16)1.0f : (bf16)aie::sub(*m, vmax);
//     v32bfloat16 v10 = broadcast_to_v32bfloat16(correct);
//     v16bfloat16 v20 = extract_v16bfloat16(v10, 0);
//     v16accfloat v30 = getExpBf16_clamped(v20);
//     v16float vlcorrect = v16float(v30);
//     float lcorrect = extract_elem(vlcorrect, 0U);
//     *m = (bf16)vmax;
//     *c = lcorrect;
//     return Vec;
// }

#if defined(__chess__)
inline
#else
// Not inlined under Peano: inlined four times into _attn_qk, its spills make the call
// chain need more stack than this core has, and the K buffer sits right above the stack.
__attribute__((noinline))
#endif
aie::vector<bf16, 16> update(bf16* m, float* c, aie::vector<bf16, 16>& out, aie::mask<16>& mask, bool& is_first){
    bf16 vm = aie::reduce_max(out);
    bf16 vmax = aie::max(vm, *m);
    aie::vector<bf16, 16> Vecsub = aie::sub(out, vmax);
    aie::accum<accfloat, 16> Vec_acc = aie::mul(Vecsub, exp_scale);
    aie::vector<bf16, 16> Vec = aie::exp2<bf16>(Vec_acc.template to_vector<float>()); // f = exp(s - max)
    Vec = aie::select((bf16)0, Vec, mask);
    bf16 correct = is_first ? (bf16)1.0f : (bf16)aie::sub(*m, vmax);
    v32bfloat16 v10 = broadcast_to_v32bfloat16(correct);
    aie::vector<bf16, 16> v20 = extract_v16bfloat16(v10, 0);
    aie::accum<accfloat, 16> v30 = aie::mul(v20, exp_scale);
    // v16accfloat v30 = getExpBf16_clamped(v20);
    aie::vector<bf16, 16> vlcorrect = aie::exp2<bf16>(v30.template to_vector<float>());

    bf16 lcorrect = vlcorrect.get(0);
    *m = (bf16)vmax;
    *c = (float)lcorrect;
    return Vec;
}

template <unsigned colQ, unsigned r, unsigned s, unsigned t>
void _attn_qk(bf16 *__restrict pQ, bf16 *__restrict pK, bfloat16 *__restrict pY, bf16 *__restrict m, float *__restrict c, aie::mask<16>& mask, bool& is_first) {

    using MMUL = aie::mmul<r, s, t, bf16, bf16, accfloat>;

    event0();

    bfloat16 *__restrict pY1 = pY;
    // g times K
    bf16 *__restrict pK1 = pK; 
    bf16 *__restrict pK2 = pK + 1 * MMUL::size_B;

    {
        bf16 *__restrict pQ1 = pQ;
        aie::vector<bf16, MMUL::size_A> A0 = aie::load_v<MMUL::size_A>(pQ1);
        pQ1 += MMUL::size_A;
        aie::vector<bf16, MMUL::size_B> B00 = aie::load_v<MMUL::size_B>(pK1);
        aie::vector<bf16, MMUL::size_B> B0 = aie::transpose(B00, 8, 8);
        pK1 += MMUL::size_B * 2;
        aie::vector<bf16, MMUL::size_B> B01 = aie::load_v<MMUL::size_B>(pK2);
        aie::vector<bf16, MMUL::size_B> B1 = aie::transpose(B01, 8, 8);
        pK2 += MMUL::size_B * 2;

        aie::vector<bf16, MMUL::size_C> acc_C00 = aie::zeros<bf16, MMUL::size_C>(); 
        aie::vector<bf16, MMUL::size_C> acc_C01 = aie::zeros<bf16, MMUL::size_C>(); 

        MMUL C00(acc_C00);
        MMUL C01(acc_C01);

        C00.mac(A0, B0);
        C01.mac(A0, B1);

        for (unsigned i = 1; i < colQ; ++i)
        chess_prepare_for_pipelining chess_loop_range(7, ) {
            A0 = aie::load_v<MMUL::size_A>(pQ1);
            pQ1 += MMUL::size_A;

            B00 = aie::load_v<MMUL::size_B>(pK1);
            B0 = aie::transpose(B00, 8, 8);
            pK1 += MMUL::size_B * 2;
            B01 = aie::load_v<MMUL::size_B>(pK2);
            B1 = aie::transpose(B01, 8, 8);
            pK2 += MMUL::size_B * 2;

            C00.mac(A0, B0);
            C01.mac(A0, B1);
        }
        auto mout0 = aie::interleave_zip(C00.template to_vector<bf16>(), C01.template to_vector<bf16>(), 8);

        aie::vector<bf16, 16> mout2, mout3, mout4, mout5;

        mout2 = aie::filter_even(mout0.first, 16);
        mout3 = aie::filter_odd (mout0.first, 16);
        mout4 = aie::filter_even(mout0.second, 16);
        mout5 = aie::filter_odd (mout0.second, 16);

        mout2 = aie::mul(mout2, (bf16)scale).template to_vector<bf16>(0);
        mout3 = aie::mul(mout3, (bf16)scale).template to_vector<bf16>(0);
        mout4 = aie::mul(mout4, (bf16)scale).template to_vector<bf16>(0);
        mout5 = aie::mul(mout5, (bf16)scale).template to_vector<bf16>(0);

        aie::vector<bf16, 16> vec = update(m, c, mout2, mask, is_first);
        aie::store_v(pY1, vec);
        pY1 += 16;
        vec = update(m + 1, c + 1, mout3, mask, is_first);
        aie::store_v(pY1, vec);
        pY1 += 16;
        vec = update(m + 2, c + 2, mout4, mask, is_first);
        aie::store_v(pY1, vec);
        pY1 += 16;
        vec = update(m + 3, c + 3, mout5, mask, is_first);
        aie::store_v(pY1, vec);
        pY1 += 16;
    }
    event1();
}

constexpr int q_prod_lock = 0;
constexpr int q_cons_lock = 1;
constexpr int k_prod_lock = 2;
constexpr int k_cons_lock = 3;
constexpr int s_prod_lock = 4;
constexpr int s_cons_lock = 5;
// constexpr int l_prod_lock = 6;
constexpr int l_cons_lock = 7;

extern "C"{

void attn_qk(
    bf16 *__restrict q_ping,
    bf16 *__restrict q_pong,
    bf16 *__restrict k_ping,
    bf16 *__restrict k_pong,
    bf16 *__restrict s_ping,
    bf16 *__restrict s_pong,
    int* L
)
{
    // Peano's CDO loader does not zero .bss; keep ping/pong selectors in .data.
#if !defined(__chess__)
    __attribute__((section(".data")))
#endif
    static bool is_ql_ping = false;
#if !defined(__chess__)
    __attribute__((section(".data")))
#endif
    static bool is_ksc_ping = false;
    alignas(aie::vector_decl_align) static  bf16 m[16] = {};
    alignas(aie::vector_decl_align) static float c_local[8] = {};
#if defined(__chess__)
    static const aie::vector<bf16, 16> neg_inf = aie::broadcast<bf16, 16>(-0x1.FEp127f);
    static const aie::vector<int, 16> idx = aie::vector<int, 16>(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
#else
    // Not static under Peano: the guard of a function-local static initializer lives
    // in .bss, which is not zeroed at load, so the initializer could be skipped.
    const aie::vector<bf16, 16> neg_inf = aie::broadcast<bf16, 16>(-0x1.FEp127f);
    const aie::vector<int, 16> idx = aie::vector<int, 16>(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
#endif
    // const aie::vector<bf16, 16> neg_inf = aie::broadcast<bf16, 16>(-0x1.FEp127f);
    // const aie::vector<float, 8> zero = aie::broadcast<float, 8>(0);

    _lock_acquire(q_cons_lock);
    is_ql_ping = !is_ql_ping;
    // bf16 *l = is_ql_ping ? l_ping : l_pong;
    bf16 *q = is_ql_ping ? q_ping : q_pong;
    // aie::store_v(l_local, zero);
    aie::store_v(m, neg_inf);
    int rounds = (L[0] + 15) / 16;
    _lock_release(l_cons_lock);
    int i = L[0];
    do {
        _lock_acquire(k_cons_lock);
        _lock_acquire(s_prod_lock);
        is_ksc_ping = !is_ksc_ping;
        bf16 *k = is_ksc_ping ? k_ping : k_pong;
        bf16 *s = is_ksc_ping ? s_ping : s_pong;
        float *c = (float *)(s + 4 * 16);
        bool is_first = (i == L[0]);
        aie::mask<16> mask = aie::le(idx, (i < 16) ? i : 16);
        _attn_qk<DH / 8, 4, 8, 8>(q, k, s, m, c_local, mask, is_first);
        aie::vector<float, 8> c_vec = aie::load_v<8>(c_local);
        aie::store_v(c, c_vec);
		_lock_release(k_prod_lock);
		_lock_release(s_cons_lock);
        i -= 16;
    } while (i > 0);
	_lock_release(q_prod_lock);
}

}
