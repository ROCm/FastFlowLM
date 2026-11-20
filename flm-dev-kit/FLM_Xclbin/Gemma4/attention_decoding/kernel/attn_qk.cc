// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include "lut_based_ops.h"
#include "aie_kernel_utils.h"
#include "model_spec.h"
#include "typedef.hpp"

inline aie::vector<bf16, 16> update(bf16* m, float* c, aie::vector<bf16, 16>& out, aie::mask<16>& mask, bool& is_first);
template <unsigned colQ, unsigned r, unsigned s, unsigned t>
void _attn_qk(bf16 *__restrict pQ, bf16 *__restrict pK, bf16 *__restrict pY, bf16 *__restrict m, float *__restrict c, aie::mask<16>& mask, bool& is_first, bool& is_up);

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
    // Force persistent statics into .data (file-backed): under Peano (llvm-aie)
    // the aie-rt CDO loader does not reliably zero a .bss tail, leaving these as
    // garbage on load. (chess is fine.) Applies to the ping-pong toggles and the
    // zero-init scratch buffers.
    AIE_PERSIST_ZERO static bool is_ql_ping = false;
    AIE_PERSIST_ZERO static bool is_k_ping = false;
    AIE_PERSIST_ZERO static bool is_sc_ping = false;
    // NOT static: these are fully written before use each call (m at store_v
    // below; c_local initialized on is_first), so they don't need to persist.
    // Making them non-static keeps them off both .bss (Peano CDO loader won't
    // zero it) and .data (tile data region is tight).
    alignas(aie::vector_decl_align) AIE_TRANSIENT bf16 m[16] = {};
    alignas(aie::vector_decl_align) AIE_TRANSIENT float c_local[8] = {};
    // NOT static: function-local static const is lazily initialized via a .bss
    // guard byte Peano's loader may not zero (init skipped -> garbage). Construct
    // each call; cheap and removes the static-init dependency.
    AIE_TRANSIENT const aie::vector<bf16, 16> neg_inf = aie::broadcast<bf16, 16>(-0x1.FEp127f);
    AIE_TRANSIENT const aie::vector<int, 16> idx = aie::vector<int, 16>(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);

    _lock_acquire(q_cons_lock);
    is_ql_ping = !is_ql_ping;
    bf16 *q = is_ql_ping ? q_ping : q_pong;
    aie::store_v(m, neg_inf);
    int rounds = (L[0] + 15) / 16;
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
        bool is_first = (i == L[0]);
        aie::mask<16> mask = aie::le(idx, (i < 16) ? i : 16);
        bool is_up = true;
        for (int j = 0; j < 2; j++){
            _lock_acquire(k_cons_lock);
            is_k_ping = !is_k_ping;
            bf16 *k = is_k_ping ? k_ping : k_pong;
            _attn_qk<DH / 8, GQA_R, GQA_S, GQA_T>(q, k, s + j * 4 * 16, m + j * 4, c_local + j * 4, mask, is_first, is_up);
            is_up = !is_up;
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
}
}

inline aie::vector<bf16, 16> update(bf16* m, float* c, aie::vector<bf16, 16>& out, aie::mask<16>& mask, bool& is_first){
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

// this code is specifically for gemma model, which head dimension is 512
template <unsigned colQ, unsigned r, unsigned s, unsigned t>
void _attn_qk(bf16 *__restrict pQ, bf16 *__restrict pK, bfloat16 *__restrict pY, bf16 *__restrict m, float *__restrict c, aie::mask<16>& mask, bool& is_first, bool& is_up) {

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

    aie::vector<bf16, 32> mout2, mout3 = aie::zeros<bf16, 32>();

    if (is_up) {
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
        aie::vector<bf16, 16> vec = update(m + h, c + h, out[h], mask, is_first);
        aie::store_v(pY1, vec);
        pY1 += 16;
    }
    pY1 += 16 * ATTN_GROUPS_PADDING;
}

#elif ATTN_IMPL == ATTN_IMPL_1x8x1

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
            aie::vector<bf16, 16> vec = update(m + h, c + h, out[h], mask, is_first);
            aie::store_v(pY1, vec);
            pY1 += 16;
        }
        pY1 += ATTN_GROUPS_PADDING * 16;
    }
}

#endif