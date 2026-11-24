// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include <type_traits>

#include <aie_api/aie.hpp>
#include "common.h"
#include "lut_based_ops.h"
#include "gptossParam.hpp"

inline aie::vector<bf16, 16> update(float* m, float* c, aie::vector<bf16, 16>& out, aie::mask<16>& mask){
    bf16 vm = aie::reduce_max(out);
    bf16 vmax = aie::max(vm, (bf16) *m);
    aie::vector<bf16, 16> Vecsub = aie::sub(out, vmax);

    constexpr int min_clamp = -87.0f;
    constexpr int max_clamp = 88.0f;
    aie::vector<bf16, 16> min_value = aie::broadcast<bf16, 16>(min_clamp);
    aie::vector<bf16, 16> max_value = aie::broadcast<bf16, 16>(max_clamp);

    aie::vector<bf16, 16> Vec = aie::clamp(Vecsub, min_value, max_value);
    aie::accum<accfloat, 16> Outvec = getExpBf16(Vec);
    Vec = Outvec.template to_vector<bf16>(0);
    Vec = aie::select((bf16)0, Vec, mask);

    bf16 correct = aie::sub((bf16) *m, vmax);
    aie::vector<bf16, 16> correct_vec = aie::broadcast<bf16, 16>(correct);
    correct_vec = aie::clamp(correct_vec, min_value, max_value);
    aie::accum<accfloat, 16> correct_acc = getExpBf16(correct_vec);
    aie::vector<float, 16> correct_float = correct_acc.template to_vector<float>();

    *m = (float)vmax;
    *c = (float)correct_float.get(0);
    return Vec;
}

template <unsigned colQ, unsigned r, unsigned s, unsigned t>
void _attn_qk(bf16 *__restrict pQ, bf16 *__restrict pK, bfloat16 *__restrict pY, float *__restrict m, float *__restrict c, aie::mask<16>& mask) {

    using MMUL = aie::mmul<r, s, t, bf16, bf16, accfloat>;

    event0();

    {
        bf16 *__restrict pQ1 = pQ;
        bf16 *__restrict pY1 = pY;
        bf16 *__restrict pK1 = pK; 
        bf16 *__restrict pK2 = pK + 1 * MMUL::size_B;
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

        aie::vector<bf16, 32> mout2, mout3, mout4, mout5;

        mout2 = aie::filter_even(mout0.first, 32);
        mout3 = aie::filter_odd (mout0.first, 32);
        mout4 = aie::filter_even(mout0.second, 32);
        mout5 = aie::filter_odd(mout0.second, 32);

        mout2 = aie::mul(mout2, (bf16)scale).template to_vector<bf16>(0);
        mout3 = aie::mul(mout3, (bf16)scale).template to_vector<bf16>(0);
        mout4 = aie::mul(mout4, (bf16)scale).template to_vector<bf16>(0);
        mout5 = aie::mul(mout5, (bf16)scale).template to_vector<bf16>(0);

        auto out0 = aie::filter_even(mout2, 16);
        auto out1 = aie::filter_odd(mout2, 16);
        auto out2 = aie::filter_even(mout3, 16);
        auto out3 = aie::filter_odd(mout3, 16);
        auto out4 = aie::filter_even(mout4, 16);
        auto out5 = aie::filter_odd(mout4, 16);
        auto out6 = aie::filter_even(mout5, 16);
        auto out7 = aie::filter_odd(mout5, 16);

        aie::vector<bf16, 16> vec = update(m, c, out0, mask);
        aie::store_v(pY1, vec);
        pY1 += 16;
        vec = update(m + 1, c + 1, out1, mask);
        aie::store_v(pY1, vec);
        pY1 += 16;
        vec = update(m + 2, c + 2, out2, mask);
        aie::store_v(pY1, vec);
        pY1 += 16;
        vec = update(m + 3, c + 3, out3, mask);
        aie::store_v(pY1, vec);
        pY1 += 16;
        vec = update(m + 4, c + 4, out4, mask);
        aie::store_v(pY1, vec);
        pY1 += 16;
        vec = update(m + 5, c + 5, out5, mask);
        aie::store_v(pY1, vec);
        pY1 += 16;
        vec = update(m + 6, c + 6, out6, mask);
        aie::store_v(pY1, vec);
        pY1 += 16;
        vec = update(m + 7, c + 7, out7, mask);
        aie::store_v(pY1, vec);
        pY1 += 16;
    }
    {
        bf16 *__restrict pQ1 = pQ + 8 * MMUL::size_A;
        bfloat16 *__restrict pY1 = pY + 8 * 16;
        bf16 *__restrict pK1 = pK + 8 * MMUL::size_B * 2;
        bf16 *__restrict pK2 = pK + 8 * MMUL::size_B * 2 + 1 * MMUL::size_B;
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

        aie::vector<bf16, 32> mout2, mout3, mout4, mout5;
        mout2 = aie::filter_even(mout0.first, 32);
        mout3 = aie::filter_odd(mout0.first, 32);
        mout4 = aie::filter_even(mout0.second, 32);
        mout5 = aie::filter_odd(mout0.second, 32);

        mout2 = aie::mul(mout2, (bf16)scale).template to_vector<bf16>(0);
        mout3 = aie::mul(mout3, (bf16)scale).template to_vector<bf16>(0);
        mout4 = aie::mul(mout4, (bf16)scale).template to_vector<bf16>(0);
        mout5 = aie::mul(mout5, (bf16)scale).template to_vector<bf16>(0);

        auto out0 = aie::filter_even(mout2, 16);
        auto out1 = aie::filter_odd(mout2, 16);
        auto out2 = aie::filter_even(mout3, 16);
        auto out3 = aie::filter_odd(mout3, 16);
        auto out4 = aie::filter_even(mout4, 16);
        auto out5 = aie::filter_odd(mout4, 16);
        auto out6 = aie::filter_even(mout5, 16);
        auto out7 = aie::filter_odd(mout5, 16);

        aie::vector<bf16, 16> vec = update(m + 8, c + 8, out0, mask);
        aie::store_v(pY1, vec);
        pY1 += 16;
        vec = update(m + 9, c + 9, out1, mask);
        aie::store_v(pY1, vec);
        pY1 += 16;
        vec = update(m + 10, c + 10, out2, mask);
        aie::store_v(pY1, vec);
        pY1 += 16;
        vec = update(m + 11, c + 11, out3, mask);
        aie::store_v(pY1, vec);
        pY1 += 16;
        vec = update(m + 12, c + 12, out4, mask);
        aie::store_v(pY1, vec);
        pY1 += 16;
        vec = update(m + 13, c + 13, out5, mask);
        aie::store_v(pY1, vec);
        pY1 += 16;
        vec = update(m + 14, c + 14, out6, mask);
        aie::store_v(pY1, vec);
        pY1 += 16;
        vec = update(m + 15, c + 15, out7, mask);
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
constexpr int l_cons_lock = 7;

extern "C"{

void attn_qk(
    bf16 *__restrict q_ping,
    bf16 *__restrict q_pong,
    bf16 *__restrict k_ping,
    bf16 *__restrict k_pong,
    bf16 *__restrict s_ping,
    bf16 *__restrict s_pong,
    bf16 *__restrict sink,
    int * L
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
    alignas(aie::vector_decl_align) static float m_local[16] = {};
    alignas(aie::vector_decl_align) static float c_local[16] = {};
#if defined(__chess__)
    static const aie::vector<int, 16> idx = aie::vector<int, 16>(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
#else
    // Peano: build idx at runtime. A function-local `static const` with a
    // non-trivial constructor needs a guard variable in .bss, and Peano's CDO
    // loader does not zero .bss, so the guard can already be set on entry and
    // the constructor never runs, leaving idx undefined.
    aie::vector<int, 16> idx;
    for (int _i = 0; _i < 16; _i++) idx[_i] = _i + 1;
#endif

    // Peano miscompiles read-modify-write on persisted .data bools across the loop;
    // keep ping/pong state in stack locals and sync back once at the end.
    // chess binds plain references so its generated code is unchanged.
#if defined(__chess__)
    bool &ql = is_ql_ping;
    bool &ksc = is_ksc_ping;
#else
    bool ql = is_ql_ping;
    bool ksc = is_ksc_ping;
#endif
    _lock_acquire(q_cons_lock);
    ql = !ql;
    bf16 *q = ql ? q_ping : q_pong;

    AIE_LOCK_LOOP_CTR i = L[0];
    aie::vector<bf16, 16> sink_vec = aie::load_v<16>(q +  16*DH ); // Each process 16 DH, so has 16 sink value(one per head)
    aie::store_v(sink,sink_vec ); // write to sink buffer for k_v
    aie::accum<accfloat, 16> sink_acc;
    sink_acc.from_vector(sink_vec);
    aie::vector<float, 16> sink_float = sink_acc.template to_vector<float>();
    aie::store_v(m_local, sink_float);

    _lock_release(l_cons_lock); // release here, so that k_v core will not acrquire the sink value earlier as it gaurded by _down_acquire_lock
    do {
        _lock_acquire(k_cons_lock);
        _lock_acquire(s_prod_lock);
        ksc = !ksc;
        bf16 *k = ksc ? k_ping : k_pong;
        bf16 *s = ksc ? s_ping : s_pong;
        float *c = (float *)(s + 16 * 16);
        float *m = (float *)(s + 16 * 16 + 32);
        aie::mask<16> mask = aie::le(idx, (i < 16) ? i : 16);
        _attn_qk<DH / 8, 8, 8, 8>(q, k, s, m_local, c_local, mask);
        aie::vector<float, 16> c_vec = aie::load_v<16>(c_local);
        aie::store_v(c, c_vec);
        aie::vector<float, 16> m_vec = aie::load_v<16>(m_local);
        aie::store_v(m, m_vec);
		_lock_release(k_prod_lock);
		_lock_release(s_cons_lock);
        i -= 16;
    } while (i > 0);
#if !defined(__chess__)
    is_ql_ping = ql;
    is_ksc_ping = ksc;
#endif
	_lock_release(q_prod_lock);
}

}
