// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

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

#ifndef KV_BFP16
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
    AIE_PERSIST_ZERO static bool is_ql_ping = false;
    AIE_PERSIST_ZERO static bool is_ksc_ping = false;
    alignas(aie::vector_decl_align) static  bf16 m[16] = {};
    alignas(aie::vector_decl_align) AIE_PERSIST_ZERO static float c_local[8] = {};
#if defined(__chess__)
    static const aie::vector<bf16, 16> neg_inf = aie::broadcast<bf16, 16>(-0x1.FEp127f);
    static const aie::vector<int, 16> idx = aie::vector<int, 16>(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
#else
    // Peano: build these at runtime. A function-local `static const` with a
    // non-trivial constructor needs a guard variable in .bss, and Peano's CDO
    // loader does not zero .bss, so the guard can already be set on entry and
    // the constructor never runs, leaving both vectors undefined.
    aie::vector<bf16, 16> neg_inf = aie::broadcast<bf16, 16>(-0x1.FEp127f);
    aie::vector<int, 16> idx;
    for (int _i = 0; _i < 16; _i++) idx[_i] = _i + 1;
#endif

    _lock_acquire(q_cons_lock);
    is_ql_ping = !is_ql_ping;
    // bf16 *l = is_ql_ping ? l_ping : l_pong;
    bf16 *q = is_ql_ping ? q_ping : q_pong;
    // aie::store_v(l_local, zero);
    aie::store_v(m, neg_inf);
    int rounds = (L[0] + 15) / 16;
    _lock_release(l_cons_lock);
    AIE_LOCK_LOOP_CTR i = L[0];
    do {
        _lock_acquire(k_cons_lock);
        _lock_acquire(s_prod_lock);
        is_ksc_ping = !is_ksc_ping;
        bf16 *k = is_ksc_ping ? k_ping : k_pong;
        bf16 *s = is_ksc_ping ? s_ping : s_pong;
        float *c = (float *)(s + Q_HEADS_PADDED_PER_CU * 16);
        bool is_first = (i == L[0]);
        aie::mask<16> mask = aie::le(idx, (i < 16) ? i : 16);
#ifndef PROBE_QK_SKIP
        _attn_qk<DH / 8, GQA_R, GQA_S, GQA_T>(q, k, s, m, c_local, mask, is_first);
#endif
        aie::vector<float, 8> c_vec = aie::load_v<8>(c_local);
        aie::store_v(c, c_vec);
		_lock_release(k_prod_lock);
		_lock_release(s_cons_lock);
        i -= 16;
    } while (i > 0);
	_lock_release(q_prod_lock);
}

}

#else  // KV_BFP16: finished chunks arrive as host-packed bfp16 K, the newest chunk as bf16 rows (README "bfp16 K and V")

constexpr int CHUNK_KEYS = 16;
constexpr int HEAD_BYTES = CHUNK_KEYS * DH / 8 * 9;   // one KV head of one chunk as bfp16: 4,608 B
constexpr int TAIL_LEAD_ROWS = 2;                     // tail units start 2 rows early when older rows exist

template <unsigned colQ>
void _attn_qk_bfp16(bf16 *__restrict pQ, const uint8_t *__restrict pK, bf16 *__restrict pY, bf16 *__restrict m, float *__restrict c, aie::mask<16>& mask, bool& is_first);

__attribute__((noinline)) static void qk_chunk(bf16 *__restrict q, const uint8_t *__restrict k, bf16 *__restrict s,
                                               bf16 *__restrict m, float *__restrict c_local,
                                               const aie::vector<int, 16> &idx, int remaining, bool is_first) {
    float *c = (float *)(s + Q_HEADS_PADDED_PER_CU * 16);
    aie::mask<16> mask = aie::le(idx, (remaining < CHUNK_KEYS) ? remaining : CHUNK_KEYS);
#ifndef PROBE_QK_SKIP
    _attn_qk_bfp16<DH / 8>(q, k, s, m, c_local, mask, is_first);
#endif
    aie::vector<float, 8> c_vec = aie::load_v<8>(c_local);
    aie::store_v(c, c_vec);
}

__attribute__((noinline)) static void pack_k_tail(const bf16 *__restrict rows, uint8_t *__restrict packed) {
#ifndef PROBE_QK_SKIP
    aie::block_vector_output_buffer_stream<bfp16ebs8, 64> out((bfp16ebs8 *)packed);
    for (int c = 0; c < DH / 8; c++) {
        for (int kb = 0; kb < 2; kb++) {
            const bf16 *__restrict p = rows + kb * 8 * DH + c * 8;
            aie::vector<bf16, 64> tile = aie::concat(
                aie::load_v<8>(p), aie::load_v<8>(p + DH), aie::load_v<8>(p + 2 * DH), aie::load_v<8>(p + 3 * DH),
                aie::load_v<8>(p + 4 * DH), aie::load_v<8>(p + 5 * DH), aie::load_v<8>(p + 6 * DH), aie::load_v<8>(p + 7 * DH));
            aie::accum<accfloat, 64> acc(tile);
            out << acc.template to_vector<bfp16ebs8>();
        }
    }
#endif
}

extern "C"{

void attn_qk(
    bf16 *__restrict q_ping,
    bf16 *__restrict q_pong,
    bf16 *__restrict k_ping,
    bf16 *__restrict k_pong,
    bf16 *__restrict k_tail,
    bf16 *__restrict s_ping,
    bf16 *__restrict s_pong,
    int* L
)
{
    uint8_t *k_tail_packed = (uint8_t *)k_tail;
    AIE_PERSIST_ZERO static bool is_ql_ping = false;
    AIE_PERSIST_ZERO static bool is_k_ping = false;
    AIE_PERSIST_ZERO static bool is_s_ping = false;
    alignas(aie::vector_decl_align) static  bf16 m[16] = {};
    alignas(aie::vector_decl_align) AIE_PERSIST_ZERO static float c_local[8] = {};
    aie::vector<bf16, 16> neg_inf = aie::broadcast<bf16, 16>(-0x1.FEp127f);
    aie::vector<int, 16> idx;
    for (int _i = 0; _i < 16; _i++) idx[_i] = _i + 1;

    _lock_acquire(q_cons_lock);
    is_ql_ping = !is_ql_ping;
    bf16 *q = is_ql_ping ? q_ping : q_pong;
    aie::store_v(m, neg_inf);
    _lock_release(l_cons_lock);
    const int total = L[0];
    const int n_chunks = L[1];
    AIE_LOCK_LOOP_CTR i = total;
    AIE_LOCK_LOOP_CTR n = n_chunks;
    while (n > 0) {
        _lock_acquire(k_cons_lock);
        _lock_acquire(s_prod_lock);
        is_k_ping = !is_k_ping;
        is_s_ping = !is_s_ping;
        const uint8_t *k = (const uint8_t *)(is_k_ping ? k_ping : k_pong);
        bf16 *s = is_s_ping ? s_ping : s_pong;
        qk_chunk(q, k, s, m, c_local, idx, i, i == total);
        _lock_release(k_prod_lock);
        _lock_release(s_cons_lock);
        i -= CHUNK_KEYS;
        n -= 1;
    }

    const int lead = n_chunks > 0 ? TAIL_LEAD_ROWS * DH : 0;
    _lock_acquire(k_cons_lock, 2);
    is_k_ping = !is_k_ping;
    const bf16 *tail0 = (is_k_ping ? k_ping : k_pong) + lead;
    is_k_ping = !is_k_ping;
    const bf16 *tail1 = (is_k_ping ? k_ping : k_pong) + lead;
    pack_k_tail(tail0, k_tail_packed);
    pack_k_tail(tail1, k_tail_packed + HEAD_BYTES);
    _lock_release(k_prod_lock, 2);
    _lock_acquire(s_prod_lock);
    is_s_ping = !is_s_ping;
    bf16 *s = is_s_ping ? s_ping : s_pong;
    qk_chunk(q, k_tail_packed, s, m, c_local, idx, i, i == total);
    _lock_release(s_cons_lock);
    _lock_release(q_prod_lock);
}

}

#endif // KV_BFP16


inline aie::vector<bf16, 16> update(bf16* m, float* c, aie::vector<bf16, 16>& out, aie::mask<16>& mask, bool& is_first){
#ifdef PROBE_QK_STUB_UPDATE
    *c = 1.0f;
    return aie::select((bf16)0, out, mask);
#endif
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

        mout2 = AIE_SPLIT_HALF_LO(mout0.first, 32);
        mout3 = AIE_SPLIT_HALF_HI(mout0.first, 32);

        mout2 = aie::mul(mout2, (bf16)ATTN_SCALE).template to_vector<bf16>(0);
        mout3 = aie::mul(mout3, (bf16)ATTN_SCALE).template to_vector<bf16>(0);


        aie::vector<bf16, 16> out[4];
        out[0] = AIE_SPLIT_HALF_LO(mout2, 16);
        out[1] = AIE_SPLIT_HALF_HI(mout2, 16);
        out[2] = AIE_SPLIT_HALF_LO(mout3, 16);
        out[3] = AIE_SPLIT_HALF_HI(mout3, 16);

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
        mout2 = AIE_SPLIT_HALF_LO(mout0.second, 32);
        mout3 = AIE_SPLIT_HALF_HI(mout0.second, 32);
        mout2 = aie::mul(mout2, (bf16)ATTN_SCALE).template to_vector<bf16>(0);
        mout3 = aie::mul(mout3, (bf16)ATTN_SCALE).template to_vector<bf16>(0);

        aie::vector<bf16, 16> out[4];
        out[0] = AIE_SPLIT_HALF_LO(mout2, 16);
        out[1] = AIE_SPLIT_HALF_HI(mout2, 16);
        out[2] = AIE_SPLIT_HALF_LO(mout3, 16);
        out[3] = AIE_SPLIT_HALF_HI(mout3, 16);
        AIE_LOOP_UNROLL_FULL
        for (int h = 0; h < Q_HEADS_PER_GROUP; h++) {
            aie::vector<bf16, 16> vec = update(m + 4 + h, c + 4 + h, out[h], mask, is_first);
            aie::store_v(pY1, vec);
            pY1 += 16;
        }
        pY1 += 16 * ATTN_GROUPS_PADDING;
    }
}

#ifdef KV_BFP16
// Same math as _attn_qk above on host-packed K: K is [KV head][DH / 8][2 key blocks][72 B] (one block = one key's 8
// dh values, the blocks the emulated bf16 mmul makes from K), Q is converted per dh block as the emulation does, and
// the native mac_8x8_8x8T_conf replaces the emulated aie::mmul. Bit-identical.
template <unsigned colQ>
void _attn_qk_bfp16(bf16 *__restrict pQ, const uint8_t *__restrict pK, bfloat16 *__restrict pY, bf16 *__restrict m, float *__restrict c, aie::mask<16>& mask, bool& is_first) {
    using ACC = aie::accum<accfloat, 64>;
    bfloat16 *__restrict pY1 = pY;
    AIE_LOOP_UNROLL_FULL
    for (int g = 0; g < KV_HEADS_PER_CU; g++) {
        aie::block_vector_input_buffer_stream<bfp16ebs8, 64> k_in((const bfp16ebs8 *)(pK + g * HEAD_BYTES));
        bf16 *__restrict pQ1 = pQ;
        ACC C00 = aie::zeros<accfloat, 64>();
        ACC C01 = aie::zeros<accfloat, 64>();
        AIE_PREPARE_FOR_PIPELINING AIE_LOOP_RANGE(8)
        for (unsigned i = 0; i < colQ; ++i) {
            aie::vector<bf16, 64> A0 = aie::load_v<64>(pQ1);
            pQ1 += 64;
            const v64bfp16ebs8 A0b = ::to_v64bfp16ebs8(ACC(A0));
            aie::block_vector<bfp16ebs8, 64> B0 = k_in.pop();
            aie::block_vector<bfp16ebs8, 64> B1 = k_in.pop();
            C00 = ::mac_8x8_8x8T_conf(A0b, B0, C00, 0, 0, 0);
            C01 = ::mac_8x8_8x8T_conf(A0b, B1, C01, 0, 0, 0);
        }
        auto mout0 = aie::interleave_zip(C00.template to_vector<bf16>(), C01.template to_vector<bf16>(), 8);
        aie::vector<bf16, 64> mine = g == 0 ? mout0.first : mout0.second;

        aie::vector<bf16, 32> mout2 = AIE_SPLIT_HALF_LO(mine, 32);
        aie::vector<bf16, 32> mout3 = AIE_SPLIT_HALF_HI(mine, 32);
        mout2 = aie::mul(mout2, (bf16)ATTN_SCALE).template to_vector<bf16>(0);
        mout3 = aie::mul(mout3, (bf16)ATTN_SCALE).template to_vector<bf16>(0);

        aie::vector<bf16, 16> out[4];
        out[0] = AIE_SPLIT_HALF_LO(mout2, 16);
        out[1] = AIE_SPLIT_HALF_HI(mout2, 16);
        out[2] = AIE_SPLIT_HALF_LO(mout3, 16);
        out[3] = AIE_SPLIT_HALF_HI(mout3, 16);
        AIE_LOOP_UNROLL_FULL
        for (int h = 0; h < Q_HEADS_PER_GROUP; h++) {
            aie::vector<bf16, 16> vec = update(m + g * Q_HEADS_PER_GROUP + h, c + g * Q_HEADS_PER_GROUP + h, out[h], mask, is_first);
            aie::store_v(pY1, vec);
            pY1 += 16;
        }
        pY1 += 16 * ATTN_GROUPS_PADDING;
    }
}
#endif

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
