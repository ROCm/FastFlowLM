// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#include "lut_based_ops.h"
#include "aie_kernel_utils.h"
#include "aie_array_layout.h"

typedef float y_acc_dtype;
void calculate_l(bf16 *__restrict pS, float *__restrict c, float *__restrict l);
template <unsigned colQ, unsigned r, unsigned s, unsigned t>
void attn_fv(bf16 *__restrict pS, bf16 *__restrict pV, y_acc_dtype *__restrict pY, float *__restrict c);
template <unsigned N>
void scale_div_aie(bf16 *a, bf16* o, float *l);
template <typename T_in, typename T_out, int N>
void passThrough_aie(T_in *restrict in0, T_out *restrict out);


constexpr int s_prod_lock = 4;
constexpr int s_cons_lock = 5;
constexpr int v_prod_lock = 2;
constexpr int v_cons_lock = 3;
constexpr int o_prod_lock = 0;
constexpr int o_cons_lock = 1;  
// constexpr int l_prod_lock = 6;
constexpr int l_cons_lock = 7;

#if !defined(__chess__)
// Peano: a separate static y_bf16 does not fit next to y in the attn_kv tile's
// data region. y is dead once converted, so the bf16 copy goes into its front
// half instead. Each store lands below every float a later iteration reads, and
// nothing here is __restrict, so the conversion is safe in place.
template <int N>
__attribute__((noinline)) bf16 *to_bf16_in_place(y_acc_dtype *y) {
    bf16 *out = reinterpret_cast<bf16 *>(y);
    for (int i = 0; i < N; i += 16) {
        aie::accum<accfloat, 16> acc;
        acc.from_vector(aie::load_v<16>(y + i));
        aie::store_v(out + i, acc.template to_vector<bf16>());
    }
    return out;
}
#endif

#if !defined(__chess__)
// What the emulated fp32 multiply by exactly 1.0f returns. That is not always
// y: the 3-way bf16 split and the accumulator adds can round. With a 1.0f
// operand, six of the nine partial products are zero and the other three are
// exact, so this is the same sequence without the zero terms.
__attribute__((always_inline)) inline aie::accum<accfloat, 32> mul_by_one(aie::vector<float, 32> y) {
    v32bfloat16 one = broadcast_one_to_v32bfloat16();
    v32accfloat yacc = (v32accfloat)(v32float)y;
    v32bfloat16 d = to_v32bfloat16(yacc);
    v32accfloat r = msc_elem_32(d, one, yacc);
    v32bfloat16 e = to_v32bfloat16(r);
    v32bfloat16 f = to_v32bfloat16(msc_elem_32(e, one, r));
    v32accfloat acc = mul_elem_32_conf(one, f, 0);
    acc = add(acc, mul_elem_32_conf(one, e, 0));
    return add(acc, mul_elem_32_conf(one, d, 0));
}
#endif

extern "C"{

void attn_kv(
    bf16 *__restrict s_ping,
    bf16 *__restrict s_pong,
    bf16 *__restrict v_ping,
    bf16 *__restrict v_pong,
    bf16 *__restrict o_ping,
    bf16 *__restrict o_pong,
    int* L                      // L[0]: the pass's largest key count
)
{
    alignas(aie::vector_decl_align) static y_acc_dtype y[8 * DH] = {};
#if defined(__chess__)
    alignas(aie::vector_decl_align) static bf16 y_bf16[8 * DH] = {};
#endif
    alignas(aie::vector_decl_align) static float l[16] = {};
    const aie::vector<float, 16> zero = aie::broadcast<float, 16>(0);

    // Peano's CDO loader does not zero .bss; keep ping/pong selectors in .data.
    AIE_PERSIST_ZERO static bool is_svc_ping = false;
    AIE_PERSIST_ZERO static bool is_lo_ping = false;

    zero_vectorized<y_acc_dtype, 8*DH>(y);
    aie::store_v(l, zero);
    _down_lock_acquire(l_cons_lock);
    // The pass's chunk count, identical for every token -- the same derivation
    // from the same single RTP word as attn_qk, and for the same reason: the
    // host sends every token the whole context rounded up to L_CHUNK, so
    // consuming a per-token count instead would leave the surplus queued for
    // the next token. See the long comment in attn_qk.cc.
    //
    // Nothing here is per token, so there is no token counter. The causal mask
    // is entirely the qk core's: a chunk past a token's keys arrives with
    // s == 0 and c == 1, so calculate_l adds nothing to l and attn_fv rescales
    // y by one and accumulates zero.
    const int n_chunks = (L[0] + 15) >> 4;
    for (AIE_LOCK_LOOP_CTR j = 0; j < n_chunks; j++)
    chess_loop_range(1, ) {
        _down_lock_acquire(s_cons_lock);
        _lock_acquire(v_cons_lock);
        is_svc_ping = !is_svc_ping;
        bf16 *s = is_svc_ping ? s_ping : s_pong;
        bf16 *v = is_svc_ping ? v_ping : v_pong;
        float *c = (float *)(s + Q_HEADS_PADDED_PER_CU * 16);
        calculate_l(s, c, l);
        attn_fv<DH / 8, GQA_R, GQA_S, GQA_T>(s, v, y, c);
        _down_lock_release(s_prod_lock);
        _lock_release(v_prod_lock);
    }
#if defined(__chess__)
    passThrough_aie<y_acc_dtype, bf16, 8*DH>(y, y_bf16);
#else
    bf16 *y_bf16 = to_bf16_in_place<8 * DH>(y);
#endif
    is_lo_ping = !is_lo_ping;
    bf16 *o = is_lo_ping ? o_ping : o_pong;
    _lock_acquire(o_prod_lock);
    scale_div_aie<8 * DH>(y_bf16, o, l);
    _lock_release(o_cons_lock);
}
}


template <typename T_in, typename T_out, int N>
void passThrough_aie(T_in *restrict in0, T_out *restrict out) {
    
    const int vec_factor = 16;

    aie::vector<T_out, vec_factor> Out;

    const int F = N / vec_factor;
    for (int i = 0; i < F; i++)
    chess_flatten_loop{
            aie::accum<accfloat, vec_factor> c_acc_in;
            c_acc_in.from_vector(aie::load_v<vec_factor>(in0));
            aie::store_v(out, c_acc_in.template to_vector<T_out>());
            in0 += vec_factor;
            out += vec_factor;
        }
}

#if !defined(__chess__)
// Not inlined under Peano: inlined, its spills stay in attn_kv's frame while
// attn_fv runs, and the call chain needs more stack than this core has.
__attribute__((noinline))
#endif
void calculate_l(bf16 *__restrict pS, float *__restrict c, float *__restrict l) {
    using MMUL = aie::mmul<GQA_R, GQA_S, GQA_T, bf16, bf16, accfloat>;
    bf16 *__restrict pS1 = pS;
   
    aie::vector<bf16, 64> S_up = aie::load_v<64>(pS1);
    pS1 += 64;
    aie::vector<bf16, 64> S_down = aie::load_v<64>(pS1);
    pS1 += 64;

    aie::vector<bf16, 32> S10 = aie::filter_even(S_up, 8);
    aie::vector<bf16, 32> S11 = aie::filter_odd(S_up, 8);
    aie::vector<bf16, 32> S20 = aie::filter_even(S_down, 8);
    aie::vector<bf16, 32> S21 = aie::filter_odd(S_down, 8);
    
    aie::vector<bf16, MMUL::size_A> S0 = aie::concat(S10, S20);
    aie::vector<bf16, MMUL::size_A> S1 = aie::concat(S11, S21);
    aie::vector<bf16, MMUL::size_B> ones = aie::broadcast<bf16, MMUL::size_B>(1);

    aie::vector<bf16, MMUL::size_C> acc_C00 = aie::zeros<bf16, MMUL::size_C>(); 

    MMUL C00(acc_C00);

    C00.mac(S0, ones);
    C00.mac(S1, ones);

    aie::vector<float, MMUL::size_C> sumf = C00.template to_vector<float>();
    auto sum03 = aie::filter_even(sumf, 4);
    auto sum02 = aie::filter_even(sum03, 2);
    auto sum01 = aie::filter_even(sum02, 1);
    auto sum00 = aie::concat(sum01, sum01);
    aie::accum<accfloat, 16> sum;
    sum.from_vector(sum00);

    aie::vector<float, 8> c_float32 = aie::load_v<8>(c);
    aie::vector<float, 16> c_in = aie::concat(c_float32, c_float32);
    
    aie::vector<float, 8> l_float32 = aie::load_v<8>(l);
    aie::vector<float, 16> l_in = aie::concat(l_float32, l_float32);

    aie::accum<accfloat, 16> l_out;
    l_out = mac_elem_16_accuracy_safe(l_in, c_in, sum, 0, 0, 0);
    aie::store_v(l, l_out.template to_vector<float>());
}

typedef float y_acc_dtype;
template <unsigned colQ, unsigned r, unsigned s, unsigned t>
void attn_fv(bf16 *__restrict pS, bf16 *__restrict pV, y_acc_dtype *__restrict pY, float *__restrict c) {
    using MMUL = aie::mmul<r, s, t, bf16, bf16, accfloat>;

#if !defined(__chess__)
    // The chess form keeps the 64-wide CORRECT vector and four accumulators live
    // across one loop, which Peano spills past this core's stack. Rescale y in its
    // own pass instead, then run one accumulator at a time.
    aie::vector<bf16, 64> S_up = aie::load_v<64>(pS);
    aie::vector<bf16, 64> S_down = aie::load_v<64>(pS + 64);
    aie::vector<bf16, MMUL::size_A> S0 = aie::concat(aie::filter_even(S_up, 8), aie::filter_even(S_down, 8));
    aie::vector<bf16, MMUL::size_A> S1 = aie::concat(aie::filter_odd(S_up, 8), aie::filter_odd(S_down, 8));
    // c[h] is exactly 1.0f whenever head h's running max did not move.
    bool need_correct = false;
    for (int h = 0; h < 8; h++)
        need_correct |= c[h] != 1.0f;
    if (need_correct) {
        aie::vector<y_acc_dtype, 64> CORRECT = aie::concat(
            aie::broadcast<float, 8>(c[0]), aie::broadcast<float, 8>(c[1]),
            aie::broadcast<float, 8>(c[2]), aie::broadcast<float, 8>(c[3]),
            aie::broadcast<float, 8>(c[4]), aie::broadcast<float, 8>(c[5]),
            aie::broadcast<float, 8>(c[6]), aie::broadcast<float, 8>(c[7]));
        for (unsigned j = 0; j < colQ; j += 1) {
            y_acc_dtype *__restrict pYc = pY + j * MMUL::size_C;
            aie::store_v(pYc, aie::mul(CORRECT, aie::load_v<MMUL::size_C>(pYc)).template to_vector<y_acc_dtype>());
        }
    }
    for (unsigned j = 0; j < colQ; j += 1) {
        y_acc_dtype *__restrict pYc = pY + j * MMUL::size_C;
        bf16 *__restrict pVc = pV + j * MMUL::size_B;
        aie::vector<y_acc_dtype, MMUL::size_C> Yv = aie::load_v<MMUL::size_C>(pYc);
        MMUL Y(need_correct ? aie::accum<accfloat, MMUL::size_C>(Yv)
                            : aie::concat(mul_by_one(Yv.template extract<32>(0)), mul_by_one(Yv.template extract<32>(1))));
        Y.mac(S0, aie::load_v<MMUL::size_B>(pVc));
        Y.mac(S1, aie::load_v<MMUL::size_B>(pVc + MMUL::size_B * colQ));
        aie::store_v(pYc, Y.template to_vector<y_acc_dtype>());
    }
#else


    y_acc_dtype *__restrict pY1 = pY;
    bf16 *__restrict pS1 = pS;
    
    aie::vector<bf16, 64> S_up = aie::load_v<64>(pS1);
    pS1 += 64;
    aie::vector<bf16, 64> S_down = aie::load_v<64>(pS1);
    pS1 += 64;

    aie::vector<bf16, 32> S10 = aie::filter_even(S_up, 8);
    aie::vector<bf16, 32> S11 = aie::filter_odd(S_up, 8);
    aie::vector<bf16, 32> S20 = aie::filter_even(S_down, 8);
    aie::vector<bf16, 32> S21 = aie::filter_odd(S_down, 8);
    
    aie::vector<bf16, MMUL::size_A> S0 = aie::concat(S10, S20);
    aie::vector<bf16, MMUL::size_A> S1 = aie::concat(S11, S21);
    
    aie::vector<y_acc_dtype, 8> c0 = aie::broadcast<float, 8>(c[0]);
    aie::vector<y_acc_dtype, 8> c1 = aie::broadcast<float, 8>(c[1]);
    aie::vector<y_acc_dtype, 8> c2 = aie::broadcast<float, 8>(c[2]);
    aie::vector<y_acc_dtype, 8> c3 = aie::broadcast<float, 8>(c[3]); 
    aie::vector<y_acc_dtype, 8> c4 = aie::broadcast<float, 8>(c[4]);
    aie::vector<y_acc_dtype, 8> c5 = aie::broadcast<float, 8>(c[5]);
    aie::vector<y_acc_dtype, 8> c6 = aie::broadcast<float, 8>(c[6]);
    aie::vector<y_acc_dtype, 8> c7 = aie::broadcast<float, 8>(c[7]);
    aie::vector<y_acc_dtype, 64> CORRECT = aie::concat(c0, c1, c2, c3, c4, c5, c6, c7);

    bf16 *__restrict pV1 = pV;

	for (unsigned j = 0; j < colQ; j += 4)
    chess_prepare_for_pipelining chess_loop_range(2, ) {
        //v0
        bf16 *__restrict pV01 = pV1 + (j + 0) * MMUL::size_B;
        bf16 *__restrict pV02 = pV1 + (j + 1) * MMUL::size_B;
        bf16 *__restrict pV03 = pV1 + (j + 2) * MMUL::size_B;
        bf16 *__restrict pV04 = pV1 + (j + 3) * MMUL::size_B;

        aie::vector<bf16, MMUL::size_B> V00 = aie::load_v<MMUL::size_B>(pV01);
        pV01 += MMUL::size_B * colQ;
        aie::vector<bf16, MMUL::size_B> V01 = aie::load_v<MMUL::size_B>(pV02);
        pV02 += MMUL::size_B * colQ;
        aie::vector<bf16, MMUL::size_B> V02 = aie::load_v<MMUL::size_B>(pV03);
        pV03 += MMUL::size_B * colQ;
        aie::vector<bf16, MMUL::size_B> V03 = aie::load_v<MMUL::size_B>(pV04);
        pV04 += MMUL::size_B * colQ;

        aie::vector<y_acc_dtype, MMUL::size_C> acc_y00 = aie::load_v<MMUL::size_C>(pY1);
        aie::vector<y_acc_dtype, MMUL::size_C> acc_y01 = aie::load_v<MMUL::size_C>(pY1 + MMUL::size_C);
        aie::vector<y_acc_dtype, MMUL::size_C> acc_y02 = aie::load_v<MMUL::size_C>(pY1 + MMUL::size_C * 2);
        aie::vector<y_acc_dtype, MMUL::size_C> acc_y03 = aie::load_v<MMUL::size_C>(pY1 + MMUL::size_C * 3);

        aie::accum<accfloat, MMUL::size_C> ACC_Y00;
        aie::accum<accfloat, MMUL::size_C> ACC_Y01;
        aie::accum<accfloat, MMUL::size_C> ACC_Y02;
        aie::accum<accfloat, MMUL::size_C> ACC_Y03;
        ACC_Y00 = aie::mul(CORRECT, acc_y00);
        ACC_Y01 = aie::mul(CORRECT, acc_y01);
        ACC_Y02 = aie::mul(CORRECT, acc_y02);
        ACC_Y03 = aie::mul(CORRECT, acc_y03);
        
        MMUL Y00(ACC_Y00);
        MMUL Y01(ACC_Y01);
        MMUL Y02(ACC_Y02);
        MMUL Y03(ACC_Y03);

        Y00.mac(S0, V00);
        Y01.mac(S0, V01);
        Y02.mac(S0, V02);
        Y03.mac(S0, V03);

        V00 = aie::load_v<MMUL::size_B>(pV01);
        V01 = aie::load_v<MMUL::size_B>(pV02);
        V02 = aie::load_v<MMUL::size_B>(pV03);
        V03 = aie::load_v<MMUL::size_B>(pV04);

        Y00.mac(S1, V00);
        Y01.mac(S1, V01);
        Y02.mac(S1, V02);
        Y03.mac(S1, V03);

        aie::store_v(pY1, Y00.template to_vector<y_acc_dtype>());
        pY1 += MMUL::size_C;
        aie::store_v(pY1, Y01.template to_vector<y_acc_dtype>());
        pY1 += MMUL::size_C;
        aie::store_v(pY1, Y02.template to_vector<y_acc_dtype>());
        pY1 += MMUL::size_C;
        aie::store_v(pY1, Y03.template to_vector<y_acc_dtype>());
        pY1 += MMUL::size_C;
    }
    
#endif
}

#if !defined(__chess__)
// Stores the first Q 8-lane head slices of v to o, two heads per 16-lane store.
template <int Q>
inline void store_head_pairs(bf16 *o, const aie::vector<bf16, 64> &v) {
    static_assert(Q % 2 == 0 && Q <= 8);
    if constexpr (Q >= 2) {
        store_head_pairs<Q - 2>(o, v);
        aie::store_v(o + 8 * (Q - 2), v.template extract<16>((Q - 2) / 2));
    }
}
#endif

template <unsigned N>
void scale_div_aie(bf16 *a, bf16* o, float *l) {
    
    constexpr int vec_factor = 64;
    const int F = N / vec_factor;

    bf16* pA1 = a;
    bf16* pO1 = o;
    bf16 l00 = (bf16)getInvBf16(l[0]);
    aie::vector<bf16, 8> L0 = aie::broadcast<bf16, 8>(l00);
    bf16 l01 = (bf16)getInvBf16(l[1]);
    aie::vector<bf16, 8> L1 = aie::broadcast<bf16, 8>(l01);
    bf16 l02 = (bf16)getInvBf16(l[2]);
    aie::vector<bf16, 8> L2 = aie::broadcast<bf16, 8>(l02);
    bf16 l03 = (bf16)getInvBf16(l[3]);
    aie::vector<bf16, 8> L3 = aie::broadcast<bf16, 8>(l03);
    bf16 l04 = (bf16)getInvBf16(l[4]);
    aie::vector<bf16, 8> L4 = aie::broadcast<bf16, 8>(l04);
    bf16 l05 = (bf16)getInvBf16(l[5]);
    aie::vector<bf16, 8> L5 = aie::broadcast<bf16, 8>(l05);
    bf16 l06 = (bf16)getInvBf16(l[6]);
    aie::vector<bf16, 8> L6 = aie::broadcast<bf16, 8>(l06);
    bf16 l07 = (bf16)getInvBf16(l[7]);
    aie::vector<bf16, 8> L7 = aie::broadcast<bf16, 8>(l07);
    auto L = aie::concat(L0, L1, L2, L3, L4, L5, L6, L7);

    for (int d = 0; d < N / vec_factor; d++){
        aie::vector<bf16, vec_factor> A0 = aie::load_v<vec_factor>(pA1);
        aie::accum<accfloat, vec_factor> AL = aie::mul(A0, L);
        aie::vector<bf16, vec_factor> AL_bf16 = AL.template to_vector<bf16>();
#if defined(__chess__)
        for (int h = 0; h < Q_HEADS_PER_GROUP; h++) {
            aie::vector<bf16, 8> temp = AL_bf16.extract<8>(h);
            aie::store_v(pO1, temp);
            pO1 += 8;
        }
#else
        // Peano miscompiles extract<8> with a runtime index. The heads' slices
        // go out back to back, so store them two at a time with constant indices.
        store_head_pairs<Q_HEADS_PER_GROUP>(pO1, AL_bf16);
        pO1 += 8 * Q_HEADS_PER_GROUP;
#endif
        pA1 += vec_factor;
    }
    
}
