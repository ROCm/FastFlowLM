// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include "lut_based_ops.h"
#include "aie_kernel_utils.h"
#include "aie_array_layout.h"

inline aie::vector<bf16, 16> update(bf16* m, float* c, aie::vector<bf16, 16>& out, aie::mask<16>& mask, bool& is_first);
template <unsigned colQ, unsigned r, unsigned s, unsigned t>
void _attn_qk(bf16 *__restrict pQ, bf16 *__restrict pK, bf16 *__restrict pY, bf16 *__restrict m, float *__restrict c, aie::mask<16>& mask, bool& is_first);



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
    // Peano's CDO loader does not zero .bss; keep ping/pong selectors and
    // zero-init scratch in .data so they are actually zeroed at load.
#if !defined(__chess__)
    __attribute__((section(".data")))
#endif
    static bool is_ql_ping = false;
#if !defined(__chess__)
    __attribute__((section(".data")))
#endif
    static bool is_ksc_ping = false;
#if !defined(__chess__)
    alignas(aie::vector_decl_align) static __attribute__((section(".data"))) bf16 m[16] = {};
    alignas(aie::vector_decl_align) static __attribute__((section(".data"))) float c_local[8] = {};
#else
    alignas(aie::vector_decl_align) static  bf16 m[16] = {};
    alignas(aie::vector_decl_align) static float c_local[8] = {};
#endif
    // Peano: non-static so the broadcast/init runs each call (peano's non-atomic
    // static-init guard lives in unzeroed .bss and may skip the constructor).
    const aie::vector<bf16, 16> neg_inf = aie::broadcast<bf16, 16>(-0x1.FEp127f);
    const aie::vector<int, 16> idx = aie::vector<int, 16>(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);

    // Peano miscompiles read-modify-write on persisted .data bools; keep ping/pong
    // state in stack locals, sync to .data once at entry/exit.
    bool ql = is_ql_ping;
    bool ksc = is_ksc_ping;
    _lock_acquire(q_cons_lock);
    ql = !ql;
    // bf16 *l = ql ? l_ping : l_pong;
    bf16 *q = ql ? q_ping : q_pong;
    // aie::store_v(l_local, zero);
    aie::store_v(m, neg_inf);
    int rounds = (L[0] + 15) / 16;
    _lock_release(l_cons_lock);
    AIE_LOCK_LOOP_CTR i = L[0];
    do {
        _lock_acquire(k_cons_lock);
        _lock_acquire(s_prod_lock);
        ksc = !ksc;
        bf16 *k = ksc ? k_ping : k_pong;
        bf16 *s = ksc ? s_ping : s_pong;
        float *c = (float *)(s + Q_HEADS_PADDED_PER_CU * 16);
        bool is_first = (i == L[0]);
        aie::mask<16> mask = aie::le(idx, (i < 16) ? i : 16);
        _attn_qk<DH / 8, GQA_R, GQA_S, GQA_T>(q, k, s, m, c_local, mask, is_first);
        aie::vector<float, 8> c_vec = aie::load_v<8>(c_local);
        aie::store_v(c, c_vec);
		_lock_release(k_prod_lock);
		_lock_release(s_cons_lock);
        i -= 16;
    } while (i > 0);
	_lock_release(q_prod_lock);
    is_ql_ping = ql;
    is_ksc_ping = ksc;
}

}


#if !defined(__chess__)
__attribute__((noinline))
#else
inline
#endif
aie::vector<bf16, 16> update(bf16* m, float* c, aie::vector<bf16, 16>& out, aie::mask<16>& mask, bool& is_first){
    aie::vector<bf16, 16> masked_vm = aie::select((bf16)(-0x1.FEp127f), out, mask);
    bf16 vm = aie::reduce_max(masked_vm);
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

#if ATTN_IMPL == ATTN_IMPL_2x4x1

template <unsigned colQ, unsigned r, unsigned s, unsigned t>
void _attn_qk(bf16 *__restrict pQ, bf16 *__restrict pK, bfloat16 *__restrict pY, bf16 *__restrict m, float *__restrict c, aie::mask<16>& mask, bool& is_first) {

    using MMUL = aie::mmul<r, s, t, bf16, bf16, accfloat>;

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

        aie::vector<bf16, 32> mout2, mout3;

        mout2 = aie::filter_even(mout0.first, 32);
        mout3 = aie::filter_odd (mout0.first, 32);

        mout2 = aie::mul(mout2, (bf16)ATTN_SCALE).template to_vector<bf16>(0);
        mout3 = aie::mul(mout3, (bf16)ATTN_SCALE).template to_vector<bf16>(0);

        
        aie::vector<bf16, 16> out[4];
        out[0] = aie::filter_even(mout2, 16);
        out[1] = aie::filter_odd(mout2, 16);
        out[2] = aie::filter_even(mout3, 16);
        out[3] = aie::filter_odd(mout3, 16);

        AIE_LOOP_UNROLL_FULL
        for (int h = 0; h < Q_HEADS_PER_GROUP; h++) {
            aie::vector<bf16, 16> vec = update(m + h, c + h, out[h], mask, is_first);
            aie::store_v(pY1, vec);
            pY1 += 16;
        }
        pY1 += 16 * ATTN_GROUPS_PADDING;
    }
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

        aie::vector<bf16, 32> mout2, mout3;
        mout2 = aie::filter_even(mout0.second, 32);
        mout3 = aie::filter_odd(mout0.second, 32);
        mout2 = aie::mul(mout2, (bf16)ATTN_SCALE).template to_vector<bf16>(0);
        mout3 = aie::mul(mout3, (bf16)ATTN_SCALE).template to_vector<bf16>(0);

        aie::vector<bf16, 16> out[4];
        out[0] = aie::filter_even(mout2, 16);
        out[1] = aie::filter_odd(mout2, 16);
        out[2] = aie::filter_even(mout3, 16);
        out[3] = aie::filter_odd(mout3, 16);
        AIE_LOOP_UNROLL_FULL
        for (int h = 0; h < Q_HEADS_PER_GROUP; h++) {
            aie::vector<bf16, 16> vec = update(m + 4 + h, c + 4 + h, out[h], mask, is_first);
            aie::store_v(pY1, vec);
            pY1 += 16;
        }
        pY1 += 16 * ATTN_GROUPS_PADDING;
    }
}

#elif ATTN_IMPL == ATTN_IMPL_1x4x1


template <unsigned colQ, unsigned r, unsigned s, unsigned t>
void _attn_qk(bf16 *__restrict pQ, bf16 *__restrict pK, bfloat16 *__restrict pY, bf16 *__restrict m, float *__restrict c, aie::mask<16>& mask, bool& is_first) {

    using MMUL = aie::mmul<r, s, t, bf16, bf16, accfloat>;

    

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

        aie::vector<bf16, 16> mout2, mout3;

        mout2 = aie::filter_even(mout0.first, 16);
        mout3 = aie::filter_odd (mout0.first, 16);

        mout2 = aie::mul(mout2, (bf16)ATTN_SCALE).template to_vector<bf16>(0);
        mout3 = aie::mul(mout3, (bf16)ATTN_SCALE).template to_vector<bf16>(0);

        aie::vector<bf16, 16> vec = update(m, c, mout2, mask, is_first);
        aie::store_v(pY1, vec);
        pY1 += 16;
        vec = update(m + 1, c + 1, mout3, mask, is_first);
        aie::store_v(pY1, vec);
        pY1 += 16;
    }
    
}

#elif ATTN_IMPL == ATTN_IMPL_1x8x1


template <unsigned colQ, unsigned r, unsigned s, unsigned t>
#if !defined(__chess__)
__attribute__((noinline))
#endif
void _attn_qk(bf16 *__restrict pQ, bf16 *__restrict pK, bfloat16 *__restrict pY, bf16 *__restrict m, float *__restrict c, aie::mask<16>& mask, bool& is_first) {

    using MMUL = aie::mmul<r, s, t, bf16, bf16, accfloat>;

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

        aie::vector<bf16, 64> mout2 = aie::mul(mout0.first, (bf16)ATTN_SCALE).template to_vector<bf16>(0);
        aie::vector<bf16, 64> mout3 = aie::mul(mout0.second, (bf16)ATTN_SCALE).template to_vector<bf16>(0);

        aie::vector<bf16, 16> out0 = mout2.extract<16>(0);
        aie::vector<bf16, 16> out1 = mout2.extract<16>(1);
        aie::vector<bf16, 16> out2 = mout2.extract<16>(2);
        aie::vector<bf16, 16> out3 = mout2.extract<16>(3);
        aie::vector<bf16, 16> out4 = mout3.extract<16>(0);
        aie::vector<bf16, 16> out5 = mout3.extract<16>(1);
        aie::vector<bf16, 16> out6 = mout3.extract<16>(2);
        aie::vector<bf16, 16> out7 = mout3.extract<16>(3);

        aie::vector<bf16, 16> vec = update(m, c, out0, mask, is_first);
        aie::store_v(pY1, vec);
        pY1 += 16;
        vec = update(m + 1, c + 1, out1, mask, is_first);
        aie::store_v(pY1, vec);
        pY1 += 16;
        vec = update(m + 2, c + 2, out2, mask, is_first);
        aie::store_v(pY1, vec);
        pY1 += 16;
        vec = update(m + 3, c + 3, out3, mask, is_first);
        aie::store_v(pY1, vec);
        pY1 += 16;
        vec = update(m + 4, c + 4, out4, mask, is_first);
        aie::store_v(pY1, vec);
        pY1 += 16;
        vec = update(m + 5, c + 5, out5, mask, is_first);
        aie::store_v(pY1, vec);
        pY1 += 16;
        vec = update(m + 6, c + 6, out6, mask, is_first);
        aie::store_v(pY1, vec);
        pY1 += 16;
        vec = update(m + 7, c + 7, out7, mask, is_first);
        aie::store_v(pY1, vec);
        pY1 += 16;
    }
    
}

#endif
