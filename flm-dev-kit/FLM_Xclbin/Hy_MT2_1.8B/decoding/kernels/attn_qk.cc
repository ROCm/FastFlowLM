// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#include "lut_based_ops.h"
#include "aie_kernel_utils.h"
#include "aie_array_layout.h"


inline aie::vector<bf16, 16> update(bf16* m, float* c, aie::vector<bf16, 16>& out, aie::mask<16>& mask, bool& is_first);
template <unsigned colQ>
void _attn_qk(bf16 *__restrict pQ, const uint8_t *__restrict pK, bf16 *__restrict pY, bf16 *__restrict m, float *__restrict c, aie::mask<16>& mask, bool& is_first);



constexpr int q_prod_lock = 0;
constexpr int q_cons_lock = 1;
constexpr int k_prod_lock = 2;
constexpr int k_cons_lock = 3;
constexpr int s_prod_lock = 4;
constexpr int s_cons_lock = 5;
// constexpr int l_prod_lock = 6;
constexpr int l_cons_lock = 7;
constexpr int tail_prod_lock = 6;   // on the P·V tile above
constexpr int tail_cons_lock = 7;

constexpr int CHUNK_KEYS = 16;
constexpr int K_CHUNK_BYTES = CHUNK_KEYS * DH / 8 * 9;   // one head, bfp16

/// one 16-key chunk: S = Q K^T into s, running max / correction into m / c_local
__attribute__((noinline)) static void qk_chunk(bf16 *__restrict q, const uint8_t *__restrict k, bf16 *__restrict s,
                                               bf16 *__restrict m, float *__restrict c_local,
                                               const aie::vector<int, 16> &idx, int remaining, bool is_first) {
    float *c = (float *)(s + Q_HEADS_PADDED_PER_CU * 16);
    aie::mask<16> mask = aie::le(idx, (remaining < CHUNK_KEYS) ? remaining : CHUNK_KEYS);
#if !defined(DECODE_STUB_ATTN) && !defined(DECODE_STUB_QK)
    _attn_qk<DH / 8>(q, k, s, m, c_local, mask, is_first);
#endif
    aie::vector<float, 8> c_vec = aie::load_v<8>(c_local);
    aie::store_v(c, c_vec);
}

/// the bf16 tail chunk (8-key x 8-dh tiles, [dh block][key block]) -> the packed bfp16 layout the
/// host writes: the same conversion the emulated bf16 mmul does on every call
__attribute__((noinline)) static void pack_k_tail(const bf16 *__restrict k, uint8_t *__restrict packed) {
    aie::block_vector_output_buffer_stream<bfp16ebs8, 64> out((bfp16ebs8 *)packed);
    for (int t = 0; t < CHUNK_KEYS * DH / 64; t++) {
        aie::accum<accfloat, 64> tile(aie::load_v<64>(k + t * 64));
        out << tile.template to_vector<bfp16ebs8>();
    }
}

extern "C"{

void attn_qk(
    bf16 *__restrict q_ping,
    bf16 *__restrict q_pong,
    bf16 *__restrict k_ping,
    bf16 *__restrict k_pong,
    bf16 *__restrict kv_tail,
    bf16 *__restrict s_ping,
    bf16 *__restrict s_pong,
    int* L
)
{
    AIE_PERSIST_ZERO static bool is_ql_ping = false;
    AIE_PERSIST_ZERO static bool is_k_ping = false;
    AIE_PERSIST_ZERO static bool is_s_ping = false;
    alignas(aie::vector_decl_align) static  bf16 m[16] = {};
    alignas(aie::vector_decl_align) AIE_PERSIST_ZERO static float c_local[8] = {};
    alignas(aie::vector_decl_align) static uint8_t k_tail_packed[K_CHUNK_BYTES];
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
    bf16 *q = is_ql_ping ? q_ping : q_pong;
    aie::store_v(m, neg_inf);
    _lock_release(l_cons_lock);
    AIE_LOCK_LOOP_CTR i = L[0];
    AIE_LOCK_LOOP_CTR n_bfp16 = L[1];
    while (n_bfp16 > 0) {
        _lock_acquire(k_cons_lock);
        _lock_acquire(s_prod_lock);
        is_k_ping = !is_k_ping;
        is_s_ping = !is_s_ping;
        const uint8_t *k = (const uint8_t *)(is_k_ping ? k_ping : k_pong);
        bf16 *s = is_s_ping ? s_ping : s_pong;
        qk_chunk(q, k, s, m, c_local, idx, i, i == L[0]);
        _lock_release(k_prod_lock);
        _lock_release(s_cons_lock);
        i -= CHUNK_KEYS;
        n_bfp16 -= 1;
    }

    _up_lock_acquire(tail_cons_lock);
    pack_k_tail(kv_tail, k_tail_packed);
    _up_lock_release(tail_prod_lock);
    _lock_acquire(s_prod_lock);
    is_s_ping = !is_s_ping;
    bf16 *s = is_s_ping ? s_ping : s_pong;
    qk_chunk(q, k_tail_packed, s, m, c_local, idx, i, i == L[0]);
    _lock_release(s_cons_lock);
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
    // the row max did not move: the correction is exactly 1 (the LUT exp is not),
    // which lets attn_kv skip its rescale. See "Decode attention" in ../README.md.
    const uint16_t correct_bits = *(const uint16_t*)&correct;
    *c = (correct_bits & 0x7FFF) == 0 ? 1.0f : (float)correct_float.get(0);
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

        AIE_PREPARE_FOR_PIPELINING AIE_LOOP_RANGE(7)
        for (unsigned i = 1; i < colQ; ++i) {
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

        AIE_PREPARE_FOR_PIPELINING AIE_LOOP_RANGE(7)
        for (unsigned i = 1; i < colQ; ++i) {
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

        AIE_PREPARE_FOR_PIPELINING AIE_LOOP_RANGE(7)
        for (unsigned i = 1; i < colQ; ++i) {
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

/// S = Q K^T for one chunk with K as packed bfp16 tiles [DH / 8][2 key blocks][72 B]: native
/// mac_8x8_8x8T_conf, bit-identical to the emulated aie::mmul<8,8,8,bf16> on bf16 K. See "bfp16 K"
/// in ../README.md.
template <unsigned colQ>
void _attn_qk(bf16 *__restrict pQ, const uint8_t *__restrict pK, bfloat16 *__restrict pY, bf16 *__restrict m, float *__restrict c, aie::mask<16>& mask, bool& is_first) {

    using ACC = aie::accum<accfloat, 64>;

    bfloat16 *__restrict pY1 = pY;
    bf16 *__restrict pQ1 = pQ;
    aie::block_vector_input_buffer_stream<bfp16ebs8, 64> k_in((const bfp16ebs8 *)pK);
    {
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

