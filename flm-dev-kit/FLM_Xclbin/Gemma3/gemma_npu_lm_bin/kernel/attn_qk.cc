// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#include <type_traits>

#include <aie_api/aie.hpp>
#include "common.h"
#include "lut_based_ops.h"

inline aie::vector<bf16, 16> update(bf16* m, float* c, aie::vector<bf16, 16>& out, aie::mask<16>& mask, bool& is_first){
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

template <unsigned colQ, unsigned r, unsigned s, unsigned t>
void _attn_qk(bf16 *__restrict pQ, bf16 *__restrict pK, bfloat16 *__restrict pY, bf16 *__restrict m, float *__restrict c, aie::mask<16>& mask, bool& is_first) {

    using MMUL = aie::mmul<r, s, t, bf16, bf16, accfloat>;

    event0();

    bfloat16 *__restrict pY1 = pY;
    // g times K
    bf16 *__restrict pK1 = pK; 
    bf16 *__restrict pK2 = pK + 1 * MMUL::size_B;
    aie::vector<bf16, 16> zeros_vec = aie::zeros<bf16, 16>();

    {
        bf16 *__restrict pQ1 = pQ;
        aie::vector<bf16, 16> Q0 = aie::load_v<16>(pQ1);
        pQ1 += 16;
        aie::vector<bf16, MMUL::size_A> A0 = aie::concat(Q0, zeros_vec);
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
            Q0 = aie::load_v<16>(pQ1);
            pQ1 += 16;
            A0 = aie::concat(Q0, zeros_vec);

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

        mout2 = aie::mul(mout2, (bf16)scale).template to_vector<bf16>(0);
        mout3 = aie::mul(mout3, (bf16)scale).template to_vector<bf16>(0);

        aie::vector<bf16, 16> vec = update(m, c, mout2, mask, is_first);
        aie::store_v(pY1, vec);
        pY1 += 16;
        vec = update(m + 1, c + 1, mout3, mask, is_first);
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
    static const aie::vector<bf16, 16> neg_inf = aie::broadcast<bf16, 16>(-0x1.FEp127f);
    static const aie::vector<int, 16> idx = aie::vector<int, 16>(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
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
    AIE_LOCK_LOOP_CTR i = L[0];
    do {
        _lock_acquire(k_cons_lock);
        _lock_acquire(s_prod_lock);
        is_ksc_ping = !is_ksc_ping;
        bf16 *k = is_ksc_ping ? k_ping : k_pong;
        bf16 *s = is_ksc_ping ? s_ping : s_pong;
        float *c = (float *)(s + 2 * 16);
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
#else  // KV_BFP16: finished chunks arrive as host-packed bfp16 K, the newest chunk as bf16 rows (README "bfp16 KV")

constexpr int CHUNK_KEYS = 16;
constexpr int HEAD_BYTES = CHUNK_KEYS * DH / 8 * 9;   // one KV head of one chunk as bfp16: 4,608 B

// _attn_qk's math on packed K: K is [DH / 8][2 key blocks][72 B] (one block = one key's 8 dh values, the blocks the
// emulated bf16 mmul makes from K), Q is converted per dh block as the emulation does, and the native
// mac_8x8_8x8T_conf replaces the emulated aie::mmul<4, 8, 8>. Rows 4-7 of the 8x8 result are unused.
template <unsigned colQ>
__attribute__((noinline))
void _attn_qk_bfp16(bf16 *__restrict pQ, const uint8_t *__restrict pK, bfloat16 *__restrict pY, bf16 *__restrict m, float *__restrict c, aie::mask<16>& mask, bool& is_first) {
    using ACC = aie::accum<accfloat, 64>;
    aie::block_vector_input_buffer_stream<bfp16ebs8, 64> k_in((const bfp16ebs8 *)pK);
    const aie::vector<bf16, 16> zeros_vec = aie::zeros<bf16, 16>();
    bf16 *__restrict pQ1 = pQ;
    ACC C00 = aie::zeros<accfloat, 64>();
    ACC C01 = aie::zeros<accfloat, 64>();
    for (unsigned i = 0; i < colQ; ++i) {
        aie::vector<bf16, 16> Q0 = aie::load_v<16>(pQ1);
        pQ1 += 16;
        const v64bfp16ebs8 A0b = ::to_v64bfp16ebs8(ACC(aie::concat(Q0, zeros_vec, zeros_vec, zeros_vec)));
        aie::block_vector<bfp16ebs8, 64> B0 = k_in.pop();
        aie::block_vector<bfp16ebs8, 64> B1 = k_in.pop();
        C00 = ::mac_8x8_8x8T_conf(A0b, B0, C00, 0, 0, 0);
        C01 = ::mac_8x8_8x8T_conf(A0b, B1, C01, 0, 0, 0);
    }
    aie::vector<bf16, 32> c00 = C00.template to_vector<bf16>().template extract<32>(0);
    aie::vector<bf16, 32> c01 = C01.template to_vector<bf16>().template extract<32>(0);
    auto mout0 = aie::interleave_zip(c00, c01, 8);

    aie::vector<bf16, 16> mout2 = aie::filter_even(mout0.first, 16);
    aie::vector<bf16, 16> mout3 = aie::filter_odd(mout0.first, 16);
    mout2 = aie::mul(mout2, (bf16)scale).template to_vector<bf16>(0);
    mout3 = aie::mul(mout3, (bf16)scale).template to_vector<bf16>(0);

    bfloat16 *__restrict pY1 = pY;
    aie::vector<bf16, 16> vec = update(m, c, mout2, mask, is_first);
    aie::store_v(pY1, vec);
    pY1 += 16;
    vec = update(m + 1, c + 1, mout3, mask, is_first);
    aie::store_v(pY1, vec);
}

// The newest chunk: this head's keys 0-7 and 8-15, as bf16 rows (stride DH) in the two K buffers -> the packed
// layout above, converted on the core exactly as the emulation converts K.
__attribute__((noinline)) static void pack_k_tail(const bf16 *__restrict keys_0_7, const bf16 *__restrict keys_8_15, uint8_t *__restrict packed) {
#ifndef FLM_STUB
    aie::block_vector_output_buffer_stream<bfp16ebs8, 64> out((bfp16ebs8 *)packed);
    for (int c = 0; c < DH / 8; c++) {
        for (int kb = 0; kb < 2; kb++) {
            const bf16 *__restrict p = (kb == 0 ? keys_0_7 : keys_8_15) + c * 8;
            aie::vector<bf16, 64> tile = aie::concat(
                aie::load_v<8>(p), aie::load_v<8>(p + DH), aie::load_v<8>(p + 2 * DH), aie::load_v<8>(p + 3 * DH),
                aie::load_v<8>(p + 4 * DH), aie::load_v<8>(p + 5 * DH), aie::load_v<8>(p + 6 * DH), aie::load_v<8>(p + 7 * DH));
            aie::accum<accfloat, 64> acc(tile);
            out << acc.template to_vector<bfp16ebs8>();
        }
    }
#endif
}

__attribute__((noinline)) static void qk_chunk(bf16 *__restrict q, const uint8_t *__restrict k, bf16 *__restrict s,
                                               bf16 *__restrict m, float *__restrict c_local,
                                               const aie::vector<int, 16> &idx, int remaining, bool is_first) {
    float *c = (float *)(s + 2 * 16);
    aie::mask<16> mask = aie::le(idx, (remaining < CHUNK_KEYS) ? remaining : CHUNK_KEYS);
#ifndef FLM_STUB
    _attn_qk_bfp16<DH / 8>(q, k, s, m, c_local, mask, is_first);
#endif
    aie::vector<float, 8> c_vec = aie::load_v<8>(c_local);
    aie::store_v(c, c_vec);
}

extern "C"{

// L[0]: keys, L[1]: finished chunks that arrive packed; then the newest chunk arrives as bf16 rows 0-8 and 8-16 of
// the cache in the two K buffers (rows 8 and 16 unused).
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
    aie::set_rounding(aie::rounding_mode::floor);
    // Peano's CDO loader does not zero .bss; keep ping/pong selectors in .data, work on stack copies.
    __attribute__((section(".data"))) static bool is_ql_ping = false;
    __attribute__((section(".data"))) static bool is_k_ping = false;
    __attribute__((section(".data"))) static bool is_s_ping = false;
    alignas(aie::vector_decl_align) static bf16 m[16] = {};
    alignas(aie::vector_decl_align) static float c_local[8] = {};
    static const aie::vector<bf16, 16> neg_inf = aie::broadcast<bf16, 16>(-0x1.FEp127f);
    static const aie::vector<int, 16> idx = aie::vector<int, 16>(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
    bool k_sel = is_k_ping;
    bool s_sel = is_s_ping;

    _lock_acquire(q_cons_lock);
    is_ql_ping = !is_ql_ping;
    bf16 *q = is_ql_ping ? q_ping : q_pong;
    aie::store_v(m, neg_inf);
    _lock_release(l_cons_lock);
    const int total = L[0];
    AIE_LOCK_LOOP_CTR i = total;
    AIE_LOCK_LOOP_CTR n = L[1];
    while (n > 0) {
        _lock_acquire(k_cons_lock);
        _lock_acquire(s_prod_lock);
        k_sel = !k_sel;
        s_sel = !s_sel;
        const uint8_t *k = (const uint8_t *)(k_sel ? k_ping : k_pong);
        bf16 *s = s_sel ? s_ping : s_pong;
        qk_chunk(q, k, s, m, c_local, idx, i, i == total);
        _lock_release(k_prod_lock);
        _lock_release(s_cons_lock);
        i -= CHUNK_KEYS;
        n -= 1;
    }

    _lock_acquire(k_cons_lock, 2);
    k_sel = !k_sel;
    const bf16 *keys_0_7 = k_sel ? k_ping : k_pong;
    k_sel = !k_sel;
    const bf16 *keys_8_15 = k_sel ? k_ping : k_pong;
    pack_k_tail(keys_0_7, keys_8_15, (uint8_t *)k_tail);
    _lock_release(k_prod_lock, 2);
    _lock_acquire(s_prod_lock);
    s_sel = !s_sel;
    bf16 *s = s_sel ? s_ping : s_pong;
    qk_chunk(q, (const uint8_t *)k_tail, s, m, c_local, idx, i, i == total);
    _lock_release(s_cons_lock);
    is_k_ping = k_sel;
    is_s_ping = s_sel;
    _lock_release(q_prod_lock);
}

}
#endif // KV_BFP16
