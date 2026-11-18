// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include "aie_kernel_utils.h"
#include "model_spec.h"
#include "typedef.hpp"

inline aie::vector<bf16, 16> update(bf16* m, float* c, const aie::vector<bf16, 16>& out, aie::mask<16>& mask);
template <unsigned colQ, unsigned r, unsigned s, unsigned t>
void _attn_qk(bf16 *__restrict pQ, bf16 *__restrict pK, bf16 *__restrict pY, bf16 *__restrict m, float *__restrict c, aie::mask<16>& mask, int index);

constexpr int q_prod_lock = 0;
constexpr int q_cons_lock = 1;
constexpr int k_prod_lock = 2;
constexpr int k_cons_lock = 3;
constexpr int s_prod_lock_0 = 4;
constexpr int s_cons_lock_0 = 5;
constexpr int s_prod_lock_1 = 6;
constexpr int s_cons_lock_1 = 7;
constexpr int l_prod_lock = 8;

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
#if defined(__chess__)
    static bool is_ql_ping = false;
    static bool is_k_ping = false;
    static bool is_sc_ping = false;
#else
    // This design's buffers leave no room for .data under Peano, so the selectors
    // are kept in L[2..4] between calls (the host writes only L[0]); L[1] holds
    // a marker, so the first call after load starts from false.
    constexpr int sel_marker = 0x5a5a17e5;
    const bool sel_valid = L[1] == sel_marker;
    bool is_ql_ping = sel_valid && L[2] != 0;
    bool is_k_ping = sel_valid && L[3] != 0;
    bool is_sc_ping = sel_valid && L[4] != 0;
#endif
    alignas(aie::vector_decl_align) AIE_TRANSIENT bf16 m[16] = {};
    alignas(aie::vector_decl_align) AIE_TRANSIENT float c_local[8] = {};
    AIE_TRANSIENT const aie::vector<bf16, 16> neg_inf = aie::broadcast<bf16, 16>(-0x1.FEp127f);
    AIE_TRANSIENT const aie::vector<int, 16> idx = aie::vector<int, 16>(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);

    _lock_acquire(q_cons_lock);
    is_ql_ping = !is_ql_ping;
    bf16 *q = is_ql_ping ? q_ping : q_pong;
    aie::store_v(m, neg_inf);
    _lock_acquire(l_prod_lock);
    int i = L[0];
    do {
        is_sc_ping = !is_sc_ping;
        if(is_sc_ping){
            _lock_acquire(s_prod_lock_0);
        } else {
            _lock_acquire(s_prod_lock_1);
        }
        bf16 *s = is_sc_ping ? s_ping : s_pong;
        float *c = (float *)(s + Q_HEADS_PADDED_PER_CU * 16);
        aie::mask<16> mask = aie::le(idx, (i < 16) ? i : 16);
        for (int j = 0; j < 8 / Q_HEADS_PER_GROUP_PADDED; j ++){
            _lock_acquire(k_cons_lock);
            is_k_ping = !is_k_ping;
            bf16 *k = is_k_ping ? k_ping : k_pong;
            _attn_qk<DH / 8, GQA_R, GQA_S, GQA_T>(q, k,
                                                    s + j * Q_HEADS_PER_GROUP_PADDED * 16,
                                                    m + j * Q_HEADS_PER_GROUP_PADDED,
                                                    c_local + j * Q_HEADS_PER_GROUP_PADDED, mask, j);
            _lock_release(k_prod_lock);
        }
        aie::vector<float, 8> c_vec = aie::load_v<8>(c_local);
        aie::store_v(c, c_vec);
        if(is_sc_ping){
            _lock_release(s_cons_lock_0);
        } else {
            _lock_release(s_cons_lock_1);
        }
        i -= 16;
    } while (i > 0);
	_lock_release(q_prod_lock);
#if !defined(__chess__)
    L[1] = sel_marker;
    L[2] = is_ql_ping;
    L[3] = is_k_ping;
    L[4] = is_sc_ping;
#endif
}
}

inline aie::vector<bf16, 16> update(bf16* m, float* c, const aie::vector<bf16, 16>& out, aie::mask<16>& mask){
    aie::vector<bf16, 16> masked_vm = aie::select((bf16)(-0x1.FEp127f), out, mask);
    bf16 vm = aie::reduce_max(masked_vm);
    bf16 vmax = aie::max(vm, (bf16) *m);
    aie::vector<bf16, 16> Vecsub = aie::sub(out, vmax);

    // Clamp is still expressed in the natural-log domain; exp_scale = log2(e)
    // rebases it for the hardware exp2 below.
    constexpr float min_clamp = -87.0f;
    constexpr float max_clamp = 88.0f;
    aie::vector<bf16, 16> min_value = aie::broadcast<bf16, 16>((bf16)min_clamp);
    aie::vector<bf16, 16> max_value = aie::broadcast<bf16, 16>((bf16)max_clamp);

    aie::vector<bf16, 16> Vec = aie::clamp(Vecsub, min_value, max_value);
    aie::accum<accfloat, 16> Vec_acc = aie::mul(Vec, exp_scale);
    Vec = aie::exp2<bf16>(Vec_acc.template to_vector<float>()); // f = exp(s - max)
    Vec = aie::select((bf16)0, Vec, mask);

    bf16 correct = aie::sub((bf16) *m, vmax);
    aie::vector<bf16, 16> correct_vec = aie::broadcast<bf16, 16>(correct);
    correct_vec = aie::clamp(correct_vec, min_value, max_value);
    aie::accum<accfloat, 16> correct_acc = aie::mul(correct_vec, exp_scale);
    correct_vec = aie::exp2<bf16>(correct_acc.template to_vector<float>());

    *m = vmax;
    *c = (float)correct_vec.get(0);
    return Vec;
}

#if ATTN_IMPL == ATTN_IMPL_2x4x1

// this code is specifically for gemma model, which head dimension is DH
template <unsigned colQ, unsigned r, unsigned s, unsigned t>
void _attn_qk(bf16 *__restrict pQ, bf16 *__restrict pK, bfloat16 *__restrict pY, bf16 *__restrict m, float *__restrict c, aie::mask<16>& mask, int index) {

    using MMUL = aie::mmul<r, s, t, bf16, bf16, accfloat>;

    bfloat16 *__restrict pY1 = pY;
    // g times K
    bf16 *__restrict pK1 = pK; 
    bf16 *__restrict pK2 = pK + 1 * MMUL::size_B;

    bf16 *__restrict pQ1 = pQ;
    aie::vector<bf16, MMUL::size_A> A0 = aie::load_v<MMUL::size_A>(pQ1);
    pQ1 += MMUL::size_A;
    aie::vector<bf16, MMUL::size_B> B00 = aie::load_v<MMUL::size_B>(pK1);
    aie::vector<bf16, MMUL::size_B> B0 = aie::transpose(B00, 8, 8);
    pK1 += MMUL::size_B * 2;
    aie::vector<bf16, MMUL::size_B> B01 = aie::load_v<MMUL::size_B>(pK2);
    aie::vector<bf16, MMUL::size_B> B1 = aie::transpose(B01, 8, 8);
    pK2 += MMUL::size_B * 2;

    MMUL C00;
    MMUL C01;

    C00.mul(A0, B0);
    C01.mul(A0, B1);

    for (unsigned i = 1; i < colQ; ++i)
    chess_prepare_for_pipelining chess_loop_range(colQ - 1, colQ - 1) {
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

    aie::vector<bf16, 32> mout2;
    aie::vector<bf16, 32> mout3;

    if (index == 0) {
        mout2 = aie::filter_even(mout0.first, 32);
        mout3 = aie::filter_odd (mout0.first, 32);
    } else {
        mout2 = aie::filter_even(mout0.second, 32);
        mout3 = aie::filter_odd(mout0.second, 32);
    }

    aie::vector<bf16, 16> out[4];
    out[0] = aie::filter_even(mout2, 16);
    out[1] = aie::filter_odd(mout2, 16);
    out[2] = aie::filter_even(mout3, 16);
    out[3] = aie::filter_odd(mout3, 16);

    AIE_LOOP_UNROLL_FULL
    for (int h = 0; h < Q_HEADS_PER_GROUP; h++) {
        aie::vector<bf16, 16> vec = update(m + h, c + h, out[h], mask);
        aie::store_v(pY1, vec);
        pY1 += 16;
    }
    pY1 += 16 * ATTN_GROUPS_PADDING;
}


#elif ATTN_IMPL == ATTN_IMPL_1x8x1
template <unsigned colQ, unsigned r, unsigned s, unsigned t>
void _attn_qk(bf16 *__restrict pQ, bf16 *__restrict pK, bfloat16 *__restrict pY, bf16 *__restrict m, float *__restrict c, aie::mask<16>& mask, int index) {

    using MMUL = aie::mmul<r, s, t, bf16, bf16, accfloat>;

    bfloat16 *__restrict pY1 = pY;
    // g times K
    bf16 *__restrict pK1 = pK; 
    bf16 *__restrict pK2 = pK + 1 * MMUL::size_B;
    
    bf16 *__restrict pQ1 = pQ;
    aie::vector<bf16, MMUL::size_A> A0 = aie::load_v<MMUL::size_A>(pQ1);
    pQ1 += MMUL::size_A;
    aie::vector<bf16, MMUL::size_B> B00 = aie::load_v<MMUL::size_B>(pK1);
    aie::vector<bf16, MMUL::size_B> B0 = aie::transpose(B00, 8, 8);
    pK1 += MMUL::size_B * 2;
    aie::vector<bf16, MMUL::size_B> B01 = aie::load_v<MMUL::size_B>(pK2);
    aie::vector<bf16, MMUL::size_B> B1 = aie::transpose(B01, 8, 8);
    pK2 += MMUL::size_B * 2;

    MMUL C00;
    MMUL C01;

    C00.mul(A0, B0);
    C01.mul(A0, B1);

    for (unsigned i = 1; i < colQ; ++i){
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

    aie::vector<bf16, 64> mout2 = mout0.first;
    aie::vector<bf16, 64> mout3 = mout0.second;

    aie::vector<bf16, 16> out[8];

    out[0] = mout2.extract<16>(0);
    out[1] = mout2.extract<16>(1);
    out[2] = mout2.extract<16>(2);
    out[3] = mout2.extract<16>(3);
    out[4] = mout3.extract<16>(0);
    out[5] = mout3.extract<16>(1);
    out[6] = mout3.extract<16>(2);
    out[7] = mout3.extract<16>(3);

    for (int h = 0; h < Q_HEADS_PER_CU; h++) {
        aie::vector<bf16, 16> vec = update(m + h, c + h, out[h], mask);
        aie::store_v(pY1, vec);
        pY1 += 16;
    }
    pY1 += 16 * ATTN_GROUPS_PADDING;
}



#elif ATTN_IMPL == ATTN_IMPL_4x2x1
// this code is specifically for gemma model, which head dimension is DH (256)
template <unsigned colQ, unsigned r, unsigned s, unsigned t>
void _attn_qk(bf16 *__restrict pQ, bf16 *__restrict pK, bfloat16 *__restrict pY, bf16 *__restrict m, float *__restrict c, aie::mask<16>& mask, int index) {

    using MMUL = aie::mmul<r, s, t, bf16, bf16, accfloat>;

    bfloat16 *__restrict pY1 = pY;
    // g times K
    bf16 *__restrict pK1 = pK; 
    bf16 *__restrict pK2 = pK + 1 * MMUL::size_B;

    bf16 *__restrict pQ1 = pQ;
    aie::vector<bf16, MMUL::size_A> A0 = aie::load_v<MMUL::size_A>(pQ1);
    pQ1 += MMUL::size_A;
    aie::vector<bf16, MMUL::size_B> B00 = aie::load_v<MMUL::size_B>(pK1);
    aie::vector<bf16, MMUL::size_B> B0 = aie::transpose(B00, 8, 8);
    pK1 += MMUL::size_B * 2;
    aie::vector<bf16, MMUL::size_B> B01 = aie::load_v<MMUL::size_B>(pK2);
    aie::vector<bf16, MMUL::size_B> B1 = aie::transpose(B01, 8, 8);
    pK2 += MMUL::size_B * 2;

    MMUL C00;
    MMUL C01;

    C00.mul(A0, B0);
    C01.mul(A0, B1);

    for (unsigned i = 1; i < colQ; ++i)
    chess_prepare_for_pipelining chess_loop_range(colQ - 1, colQ - 1) {
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

    aie::vector<bf16, 32> mout2, mout3 = aie::zeros<bf16, 32>();
    aie::vector<bf16, 16> out[2];
    if ((index >> 1) == 0) {
        mout2 = aie::filter_even(mout0.first, 32);
        mout3 = aie::filter_odd (mout0.first, 32);
    } else {
        mout2 = aie::filter_even(mout0.second, 32);
        mout3 = aie::filter_odd(mout0.second, 32);
    }

    if((index & 0x1) == 0){
        out[0] = aie::filter_even(mout2, 16);
        out[1] = aie::filter_odd(mout2, 16);
    }
    else{
        out[0] = aie::filter_even(mout3, 16);
        out[1] = aie::filter_odd(mout3, 16);
    }

    AIE_LOOP_UNROLL_FULL
    for (unsigned h = 0; h < Q_HEADS_PER_GROUP; h++) {
        aie::vector<bf16, 16> vec = update(m + h, c + h, out[h], mask);
        aie::store_v(pY1, vec);
        pY1 += 16;
    }
    pY1 += 16 * ATTN_GROUPS_PADDING;
}

#elif ATTN_IMPL == ATTN_IMPL_1x16x1
template <unsigned colQ, unsigned r, unsigned s, unsigned t>
void _attn_qk(bf16 *__restrict pQ, bf16 *__restrict pK, bfloat16 *__restrict pY, bf16 *__restrict m, float *__restrict c, aie::mask<16>& mask, int index) {

    using MMUL = aie::mmul<r, s, t, bf16, bf16, accfloat>;

    bfloat16 *__restrict pY1 = pY;
    bfloat16 *__restrict pY2 = pY + 8 * 16;
    // g times K
    bf16 *__restrict pK1 = pK; 
    bf16 *__restrict pK2 = pK + 1 * MMUL::size_B;
    bf16 *__restrict pQ1 = pQ;
    bf16 *__restrict pQ2 = pQ + 8 * DH;

    aie::vector<bf16, MMUL::size_A> A0 = aie::load_v<MMUL::size_A>(pQ1);
    pQ1 += MMUL::size_A;
    aie::vector<bf16, MMUL::size_A> A1 = aie::load_v<MMUL::size_A>(pQ2);
    pQ2 += MMUL::size_A;
    aie::vector<bf16, MMUL::size_B> B00 = aie::load_v<MMUL::size_B>(pK1);
    aie::vector<bf16, MMUL::size_B> B0 = aie::transpose(B00, 8, 8);
    pK1 += MMUL::size_B * 2;
    aie::vector<bf16, MMUL::size_B> B01 = aie::load_v<MMUL::size_B>(pK2);
    aie::vector<bf16, MMUL::size_B> B1 = aie::transpose(B01, 8, 8);
    pK2 += MMUL::size_B * 2;

    MMUL C00;
    MMUL C01;
    MMUL C10;
    MMUL C11;

    C00.mul(A0, B0);
    C01.mul(A0, B1);
    C10.mul(A1, B0);
    C11.mul(A1, B1);

    for (unsigned i = 1; i < colQ; ++i){
        A0 = aie::load_v<MMUL::size_A>(pQ1);
        pQ1 += MMUL::size_A;
        A1 = aie::load_v<MMUL::size_A>(pQ2);
        pQ2 += MMUL::size_A;

        B00 = aie::load_v<MMUL::size_B>(pK1);
        B0 = aie::transpose(B00, 8, 8);
        pK1 += MMUL::size_B * 2;
        B01 = aie::load_v<MMUL::size_B>(pK2);
        B1 = aie::transpose(B01, 8, 8);
        pK2 += MMUL::size_B * 2;

        C00.mac(A0, B0);
        C01.mac(A0, B1);
        C10.mac(A1, B0);
        C11.mac(A1, B1);
    }
    auto mout0 = aie::interleave_zip(C00.template to_vector<bf16>(), C01.template to_vector<bf16>(), 8);
    auto mout1 = aie::interleave_zip(C10.template to_vector<bf16>(), C11.template to_vector<bf16>(), 8);

    aie::vector<bf16, 64> mout2 = mout0.first;
    aie::vector<bf16, 64> mout3 = mout0.second;
    aie::vector<bf16, 64> mout4 = mout1.first;
    aie::vector<bf16, 64> mout5 = mout1.second;

    aie::vector<bf16, 16> out[16];

    out[0] = mout2.extract<16>(0);
    out[1] = mout2.extract<16>(1);
    out[2] = mout2.extract<16>(2);
    out[3] = mout2.extract<16>(3);
    out[4] = mout3.extract<16>(0);
    out[5] = mout3.extract<16>(1);
    out[6] = mout3.extract<16>(2);
    out[7] = mout3.extract<16>(3);
    out[8] = mout4.extract<16>(0);
    out[9] = mout4.extract<16>(1);
    out[10] = mout4.extract<16>(2);
    out[11] = mout4.extract<16>(3);
    out[12] = mout5.extract<16>(0);
    out[13] = mout5.extract<16>(1);
    out[14] = mout5.extract<16>(2);
    out[15] = mout5.extract<16>(3);

    for (int h = 0; h < Q_HEADS_PER_CU; h++) {
        aie::vector<bf16, 16> vec = update(m + h, c + h, out[h], mask);
        aie::store_v(pY1, vec);
        pY1 += 16;
    }
    pY1 += 16 * ATTN_GROUPS_PADDING;
}


#endif