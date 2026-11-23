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

// Finished chunks arrive as host-packed bfp16 V^T, the newest chunk as bf16 rows (README "bfp16 KV").

constexpr int CHUNK_KEYS = 16;

// attn_fv's math (Peano path) on packed V^T: V is [DH / 8][2 key blocks][72 B] (one block = one dh column's 8 keys,
// the blocks the emulated bf16 mmul makes from V), P is converted to bfp16 once per chunk (the emulation converts it
// per call, to the same blocks), and the native mac_8x8_8x8T_conf replaces the emulated aie::mmul<4, 8, 8>.
template <unsigned colQ>
__attribute__((noinline))
void attn_fv_bfp16(bf16 *__restrict pS, const uint8_t *__restrict pV, y_acc_dtype *__restrict pY, float *__restrict c) {
    using ACC = aie::accum<accfloat, 64>;
    aie::vector<bf16, 32> S_up = aie::load_v<32>(pS);
    aie::vector<bf16, 16> S10 = aie::filter_even(S_up, 8);
    aie::vector<bf16, 16> S11 = aie::filter_odd(S_up, 8);
    const aie::vector<bf16, 16> zeros_vec = aie::zeros<bf16, 16>();
    const v64bfp16ebs8 S0b = ::to_v64bfp16ebs8(ACC(aie::concat(S10, zeros_vec, zeros_vec, zeros_vec)));
    const v64bfp16ebs8 S1b = ::to_v64bfp16ebs8(ACC(aie::concat(S11, zeros_vec, zeros_vec, zeros_vec)));
    const aie::vector<y_acc_dtype, 16> zeros_float = aie::zeros<y_acc_dtype, 16>();
    {
        aie::vector<y_acc_dtype, 16> CORRECT16 = aie::concat(
            aie::broadcast<float, 8>(c[0]), aie::broadcast<float, 8>(c[1]));
        for (unsigned j = 0; j < colQ; j += 1) {
            y_acc_dtype *__restrict pYc = pY + j * 16;
            aie::store_v(pYc, aie::mul(CORRECT16, aie::load_v<16>(pYc)).template to_vector<y_acc_dtype>());
        }
    }
    aie::block_vector_input_buffer_stream<bfp16ebs8, 64> v_in((const bfp16ebs8 *)pV);
    for (unsigned j = 0; j < colQ; j += 1) {
        y_acc_dtype *__restrict pYc = pY + j * 16;
        ACC Y(aie::concat(aie::load_v<16>(pYc), zeros_float, zeros_float, zeros_float));
        Y = ::mac_8x8_8x8T_conf(S0b, v_in.pop(), Y, 0, 0, 0);
        Y = ::mac_8x8_8x8T_conf(S1b, v_in.pop(), Y, 0, 0, 0);
        aie::store_v(pYc, Y.template to_vector<y_acc_dtype>().template extract<16>(0));
    }
}

// The newest chunk: this head's keys 0-7 and 8-15, as bf16 rows (stride DH) in the two V buffers -> the packed V^T
// layout above, converted on the core exactly as the emulation converts V.
__attribute__((noinline)) static void pack_vt_tail(const bf16 *__restrict keys_0_7, const bf16 *__restrict keys_8_15, uint8_t *__restrict packed) {
#ifndef FLM_STUB
    aie::block_vector_output_buffer_stream<bfp16ebs8, 64> out((bfp16ebs8 *)packed);
    for (int j = 0; j < DH / 8; j++) {
        for (int kb = 0; kb < 2; kb++) {
            const bf16 *__restrict p = (kb == 0 ? keys_0_7 : keys_8_15) + j * 8;
            aie::vector<bf16, 64> tile = aie::concat(
                aie::load_v<8>(p), aie::load_v<8>(p + DH), aie::load_v<8>(p + 2 * DH), aie::load_v<8>(p + 3 * DH),
                aie::load_v<8>(p + 4 * DH), aie::load_v<8>(p + 5 * DH), aie::load_v<8>(p + 6 * DH), aie::load_v<8>(p + 7 * DH));
            aie::accum<accfloat, 64> acc(aie::transpose(tile, 8, 8));
            out << acc.template to_vector<bfp16ebs8>();
        }
    }
#endif
}

__attribute__((noinline)) static void kv_chunk(bf16 *__restrict s, const uint8_t *__restrict v, y_acc_dtype *__restrict y, float *__restrict l) {
    float *c = (float *)(s + 2 * 16);
#ifndef FLM_STUB
    calculate_l(s, c, l);
    attn_fv_bfp16<DH / 8>(s, v, y, c);
#endif
}

extern "C"{

// L[1]: finished chunks that arrive packed; then the newest chunk arrives as bf16 rows 0-8 and 8-16 of the cache in
// the two V buffers (rows 8 and 16 unused).
void attn_kv(
    bf16 *__restrict s_ping,
    bf16 *__restrict s_pong,
    bf16 *__restrict v_ping,
    bf16 *__restrict v_pong,
    bf16 *__restrict v_tail,
    bf16 *__restrict o_ping,
    bf16 *__restrict o_pong,
    int* L
)
{
    aie::set_rounding(aie::rounding_mode::floor);
    alignas(aie::vector_decl_align) static y_acc_dtype y[2 * DH] = {};
    alignas(aie::vector_decl_align) static bf16 y_bf16[2 * DH] = {};
    alignas(aie::vector_decl_align) static float l[8] = {};
    const aie::vector<float, 8> zero = aie::broadcast<float, 8>(0);
    // Peano's CDO loader does not zero .bss, and it miscompiles read-modify-write on persisted .data bools across
    // the loop: keep them in .data, work on stack copies, write back once.
    __attribute__((section(".data"))) static bool is_s_ping = false;
    __attribute__((section(".data"))) static bool is_v_ping = false;
    __attribute__((section(".data"))) static bool is_lo_ping = false;
    bool s_sel = is_s_ping;
    bool v_sel = is_v_ping;
    bool lo = is_lo_ping;
    zero_vectorized<y_acc_dtype, 2*DH>(y);
    aie::store_v(l, zero);
    _down_lock_acquire(l_cons_lock);
    AIE_LOCK_LOOP_CTR n = L[1];
    while (n > 0) {
        _down_lock_acquire(s_cons_lock);
        _lock_acquire(v_cons_lock);
        s_sel = !s_sel;
        v_sel = !v_sel;
        bf16 *s = s_sel ? s_ping : s_pong;
        const uint8_t *v = (const uint8_t *)(v_sel ? v_ping : v_pong);
        kv_chunk(s, v, y, l);
        _down_lock_release(s_prod_lock);
        _lock_release(v_prod_lock);
        n -= 1;
    }

    _down_lock_acquire(s_cons_lock);
    _lock_acquire(v_cons_lock, 2);
    v_sel = !v_sel;
    const bf16 *keys_0_7 = v_sel ? v_ping : v_pong;
    v_sel = !v_sel;
    const bf16 *keys_8_15 = v_sel ? v_ping : v_pong;
    pack_vt_tail(keys_0_7, keys_8_15, (uint8_t *)v_tail);
    _lock_release(v_prod_lock, 2);
    s_sel = !s_sel;
    bf16 *s = s_sel ? s_ping : s_pong;
    kv_chunk(s, (const uint8_t *)v_tail, y, l);
    _down_lock_release(s_prod_lock);

    passThrough_aie<y_acc_dtype, bf16, 2*DH>(y, y_bf16);
    lo = !lo;
    bf16 *o = lo ? o_ping : o_pong;
    is_s_ping = s_sel;
    is_v_ping = v_sel;
    is_lo_ping = lo;
    _lock_acquire(o_prod_lock);
    scale_div_aie<2*DH>(y_bf16, o, l);
    _lock_release(o_cons_lock);
}

}
