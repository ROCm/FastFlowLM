// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#include <type_traits>

#include <aie_api/aie.hpp>
#include "common.h"
#include "lut_based_ops.h"

#if !defined(__chess__)
__attribute__((noinline))
#else
inline
#endif
aie::vector<bf16, 16> update(bf16* m, float* c, aie::vector<bf16, 16>& out, aie::mask<16>& mask, bool& is_first){
    bf16 vm = aie::reduce_max(out);
    bf16 vmax = aie::max(vm, *m);
    aie::vector<bf16, 16> Vecsub = aie::sub(out, vmax);
    v16bfloat16 v0 = Vecsub;
    aie::accum<accfloat, 16> Outvec = getExpBf16(v0);
    aie::vector<bf16, 16> Vec = Outvec.template to_vector<bf16>(0);
    Vec = aie::select((bf16)0, Vec, mask);
    bf16 correct = is_first ? (bf16)1.0f : (bf16)aie::sub(*m, vmax);
    v32bfloat16 v10 = broadcast_to_v32bfloat16(correct);
    v16bfloat16 v20 = extract_v16bfloat16(v10, 0);
    v16accfloat v30 = getExpBf16(v20);
    v16float vlcorrect = v16float(v30);
    float lcorrect = extract_elem(vlcorrect, 0U);
    *m = (bf16)vmax;
    *c = lcorrect;
    return Vec;
}

#if !defined(__chess__)
// Peano: computing C00 and C01 in one interleaved loop keeps TWO bf16 mmul
// accumulators live across the colQ loop -> spill/bug-B -> corrupt scores. Compute
// one score matrix with a single live accumulator; the caller runs it twice.
template <unsigned colQ, unsigned r, unsigned s, unsigned t>
__attribute__((noinline))
static aie::vector<bf16, r * t> qk_one(bf16 *__restrict pQ, bf16 *__restrict pK) {
    using MMUL = aie::mmul<r, s, t, bf16, bf16, accfloat>;
    bf16 *__restrict pQ1 = pQ;
    bf16 *__restrict pK1 = pK;
    aie::vector<bf16, MMUL::size_A> A0 = aie::load_v<MMUL::size_A>(pQ1); pQ1 += MMUL::size_A;
    aie::vector<bf16, MMUL::size_B> B0 = aie::transpose(aie::load_v<MMUL::size_B>(pK1), 8, 8); pK1 += MMUL::size_B * 2;
    aie::vector<bf16, MMUL::size_C> acc = aie::zeros<bf16, MMUL::size_C>();
    MMUL C(acc);
    C.mac(A0, B0);
    for (unsigned i = 1; i < colQ; ++i) {
        A0 = aie::load_v<MMUL::size_A>(pQ1); pQ1 += MMUL::size_A;
        B0 = aie::transpose(aie::load_v<MMUL::size_B>(pK1), 8, 8); pK1 += MMUL::size_B * 2;
        C.mac(A0, B0);
    }
    return C.template to_vector<bf16>();
}
#endif

template <unsigned colQ, unsigned r, unsigned s, unsigned t>
#if !defined(__chess__)
__attribute__((noinline))
#endif
void _attn_qk(bf16 *__restrict pQ, bf16 *__restrict pK, bfloat16 *__restrict pY, bf16 *__restrict m, float *__restrict c, aie::mask<16>& mask, bool& is_first) {

    using MMUL = aie::mmul<r, s, t, bf16, bf16, accfloat>;

    event0();

    // bf16 c_local[8];
    // bf16 l_local[8];

    // aie::vector<bf16, 8> l_vec = aie::load_v<8>(l);
    // aie::store_v(l_local, l_vec);
    bfloat16 *__restrict pY1 = pY;
    // g times K
    bf16 *__restrict pK1 = pK; 
    bf16 *__restrict pK2 = pK + 1 * MMUL::size_B;

    {
#if !defined(__chess__)
        auto c00v = qk_one<colQ, r, s, t>(pQ, pK1);
        auto c01v = qk_one<colQ, r, s, t>(pQ, pK2);
        pK1 += MMUL::size_B * 2 * colQ;
        pK2 += MMUL::size_B * 2 * colQ;
        auto mout0 = aie::interleave_zip(c00v, c01v, 8);
#else
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
#endif

        aie::vector<bf16, 32> mout2, mout3;

        // Peano miscompiles aie::filter_even/odd on bf16 at radix 32/16; at these
        // radices the filter is a contiguous half, so use extract<> (chess-exact).
#if defined(__chess__)
        mout2 = aie::filter_even(mout0.first, 32);
        mout3 = aie::filter_odd (mout0.first, 32);
#else
        mout2 = mout0.first.template extract<32>(0);
        mout3 = mout0.first.template extract<32>(1);
#endif

        mout2 = aie::mul(mout2, (bf16)scale).template to_vector<bf16>(0);
        mout3 = aie::mul(mout3, (bf16)scale).template to_vector<bf16>(0);

#if defined(__chess__)
        aie::vector<bf16, 16> out0 = aie::filter_even(mout2, 16);
        aie::vector<bf16, 16> out1 = aie::filter_odd(mout2, 16);
        aie::vector<bf16, 16> out2 = aie::filter_even(mout3, 16);
        aie::vector<bf16, 16> out3 = aie::filter_odd(mout3, 16);
#else
        aie::vector<bf16, 16> out0 = mout2.extract<16>(0);
        aie::vector<bf16, 16> out1 = mout2.extract<16>(1);
        aie::vector<bf16, 16> out2 = mout3.extract<16>(0);
        aie::vector<bf16, 16> out3 = mout3.extract<16>(1);
#endif
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
    }
    {
#if !defined(__chess__)
        auto c00v = qk_one<colQ, r, s, t>(pQ, pK1);
        auto c01v = qk_one<colQ, r, s, t>(pQ, pK2);
        pK1 += MMUL::size_B * 2 * colQ;
        pK2 += MMUL::size_B * 2 * colQ;
        auto mout0 = aie::interleave_zip(c00v, c01v, 8);
#else
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
#endif

        aie::vector<bf16, 32> mout2, mout3;
#if defined(__chess__)
        mout2 = aie::filter_even(mout0.second, 32);
        mout3 = aie::filter_odd(mout0.second, 32);
#else
        mout2 = mout0.second.template extract<32>(0);
        mout3 = mout0.second.template extract<32>(1);
#endif
        mout2 = aie::mul(mout2, (bf16)scale).template to_vector<bf16>(0);
        mout3 = aie::mul(mout3, (bf16)scale).template to_vector<bf16>(0);

#if defined(__chess__)
        aie::vector<bf16, 16> out0 = aie::filter_even(mout2, 16);
        aie::vector<bf16, 16> out1 = aie::filter_odd(mout2, 16);
        aie::vector<bf16, 16> out2 = aie::filter_even(mout3, 16);
        aie::vector<bf16, 16> out3 = aie::filter_odd(mout3, 16);
#else
        aie::vector<bf16, 16> out0 = mout2.extract<16>(0);
        aie::vector<bf16, 16> out1 = mout2.extract<16>(1);
        aie::vector<bf16, 16> out2 = mout3.extract<16>(0);
        aie::vector<bf16, 16> out3 = mout3.extract<16>(1);
#endif

        aie::vector<bf16, 16> vec = update(m + 4, c + 4, out0, mask, is_first);
        aie::store_v(pY1, vec);
        pY1 += 16;
        vec = update(m + 5, c + 5, out1, mask, is_first);
        aie::store_v(pY1, vec);
        pY1 += 16;
        vec = update(m + 6, c + 6, out2, mask, is_first);
        aie::store_v(pY1, vec);
        pY1 += 16;
        vec = update(m + 7, c + 7, out3, mask, is_first);
        aie::store_v(pY1, vec);
        pY1 += 16;
    }
    // l_vec = aie::load_v<8>(l_local);
    // aie::vector<bf16, 8> c_vec = aie::load_v<8>(c_local);
    // aie::store_v(l, l_vec);
    // aie::store_v(c, c_vec);
    // lower part
    event1();
}


#if MODEL_TYPE == LFM3_1_7B
// LFM3-1.7B: one KV head of DH = 128 per core. The Q tile is 8 rows x 128 with only rows 0-3 real (the 4 query heads of
// the KV head); rows 4-7 are zero padding. K tile layout (from the memtile): [colblock of 8 cols (colQ = DH/8 = 16)]
// [2 token blocks of 8 tokens][8 tokens][8 cols], i.e. block (cb, tb) at cb * 2 * size_B + tb * size_B.
// Output: scores of rows 0-3 as softmax numerators P (16 tokens per row), rows 4-7 written as zeros.
template <unsigned colQ>
__attribute__((noinline))
void _attn_qk_h128(bf16 *__restrict pQ, bf16 *__restrict pK, bf16 *__restrict pY, bf16 *__restrict m, float *__restrict c,
                   aie::mask<16>& mask, bool& is_first) {
    using MMUL = aie::mmul<8, 8, 8, bf16, bf16, accfloat>;
    event0();
    bf16 *__restrict pY1 = pY;
    auto c00v = qk_one<colQ, 8, 8, 8>(pQ, pK);                       // tokens 0-7
    auto c01v = qk_one<colQ, 8, 8, 8>(pQ, pK + MMUL::size_B);        // tokens 8-15
    auto mout0 = aie::interleave_zip(c00v, c01v, 8);                 // .first: rows 0-3, 16 tokens each
    aie::vector<bf16, 32> mout2 = mout0.first.template extract<32>(0);
    aie::vector<bf16, 32> mout3 = mout0.first.template extract<32>(1);
    mout2 = aie::mul(mout2, (bf16)scale).template to_vector<bf16>(0);
    mout3 = aie::mul(mout3, (bf16)scale).template to_vector<bf16>(0);
    aie::vector<bf16, 16> out0 = mout2.extract<16>(0);
    aie::vector<bf16, 16> out1 = mout2.extract<16>(1);
    aie::vector<bf16, 16> out2 = mout3.extract<16>(0);
    aie::vector<bf16, 16> out3 = mout3.extract<16>(1);
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
    // padded rows: no probability mass, correction 1
    const aie::vector<bf16, 16> zeros16 = aie::zeros<bf16, 16>();
    for (int r = 4; r < 8; r++) {
        aie::store_v(pY1, zeros16);
        pY1 += 16;
        c[r] = 1.0f;
    }
    event1();
}
#endif

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
    bf16 *q = ql ? q_ping : q_pong;
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
        float *c = (float *)(s + 8 * 16);
#if MODEL_TYPE == LFM3_1_7B
        // gates of the 4 query heads of this core ride in row 4 of the Q tile (first 8x8 block: row 4 = elements 32..35)
        s[GATE_OFFSET + 0] = q[4 * 8 + 0];
        s[GATE_OFFSET + 1] = q[4 * 8 + 1];
        s[GATE_OFFSET + 2] = q[4 * 8 + 2];
        s[GATE_OFFSET + 3] = q[4 * 8 + 3];
#endif
        bool is_first = (i == L[0]);
        aie::mask<16> mask = aie::le(idx, (i < 16) ? i : 16);
#if MODEL_TYPE == LFM3_1_7B
        _attn_qk_h128<DH / 8>(q, k, s, m, c_local, mask, is_first);
#else
        _attn_qk<DH / 8, 8, 8, 8>(q, k, s, m, c_local, mask, is_first);
#endif
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
