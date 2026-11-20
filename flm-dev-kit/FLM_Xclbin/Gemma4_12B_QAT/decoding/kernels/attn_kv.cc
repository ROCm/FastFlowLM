// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include "aie_kernel_utils.h"
#include "aie_array_layout.h"
#include "bf16_limb_math.hpp"

typedef float y_acc_dtype;
// A operand of the S @ V mmul: an r x s tile of bf16.
using s_tile_t = aie::vector<bf16, GQA_R * GQA_S>;

#if ROLE_IMPL == ATTN_IMPL_1x8x1
template <unsigned colQ, unsigned r, unsigned s, unsigned t>
void attn_fv(const s_tile_t& S0, const s_tile_t& S1, bf16 *__restrict pV, y_acc_dtype *__restrict pY, float *__restrict c, int index);
#else
template <unsigned colQ, unsigned r, unsigned s, unsigned t>
void attn_fv(const s_tile_t& S0, const s_tile_t& S1, bf16 *__restrict pV, y_acc_dtype *__restrict pY, int index);
#endif

// ---------------------------------------------------------------------------
// Softmax rescale factors
//
// c[] is fp32 in memory, but every value it holds is exactly representable in
// bf16: attn_qk's update() forms the rescale with aie::exp2<bf16> and widens a
// single bf16 lane on the way out (`*c = (float)correct_vec.get(0)`). Reading
// it back as bf16 is therefore lossless -- and it is what turns every `* c`
// below from an fp32 x fp32 product into an asymmetric fp32 x bf16 one.
//
// That matters because AIE2 has no fp32 multiplier. aie::mul on two fp32
// vectors lowers to the vendor's emulation, which defaults to the `safe` tier:
// split both operands into 3 bf16 limbs and issue all 9 cross products into an
// 8-deep serial add tree. With one operand already bf16-exact only the other
// needs splitting, so the same fp32-exact result costs 3 MACs. See
// bf16_limb_math.hpp for the measured accuracy tiers.
// ---------------------------------------------------------------------------

/// @brief The eight per-head rescale factors as bf16, one per lane.
inline aie::vector<bf16, 8> correction_bf16(const float *__restrict c) {
    aie::accum<accfloat, 8> acc;
    acc.from_vector(aie::load_v<8>(c));
    return acc.template to_vector<bf16>();
}

/// @brief The same eight factors, broadcast eight lanes wide each to match the
/// head-major layout of a 64-lane slice of the y accumulator.
inline aie::vector<bf16, 64> correction_bf16_x8(const float *__restrict c) {
    return aie::concat(aie::broadcast<bf16, 8>((bf16)c[0]),
                       aie::broadcast<bf16, 8>((bf16)c[1]),
                       aie::broadcast<bf16, 8>((bf16)c[2]),
                       aie::broadcast<bf16, 8>((bf16)c[3]),
                       aie::broadcast<bf16, 8>((bf16)c[4]),
                       aie::broadcast<bf16, 8>((bf16)c[5]),
                       aie::broadcast<bf16, 8>((bf16)c[6]),
                       aie::broadcast<bf16, 8>((bf16)c[7]));
}

/// @brief Fold this tile's S row sums into the running softmax denominators:
///        l = rowsum(S) + l * c, one lane per Q head.
///
/// Both halves used to be scalar. The row sums came out of reduce_add on a bf16
/// vector, which returns the vector's element type, so each tile's partial sum
/// was rounded to 8 mantissa bits before it ever reached l; widening the tile to
/// fp32 first keeps the reduction in the accumulator domain and costs nothing
/// on top of the conversion reduce_add would have done internally anyway.
///
/// The `l * c` update was eight scalar fp32 multiply-adds. AIE2's scalar unit
/// has no FPU, so those were software-emulated; the vector form below is one
/// three-MAC fp32 x bf16 product covering all eight heads.
void calculate_l(bf16 *__restrict pS, float *__restrict c, float *__restrict l) {
    aie::vector<float, 8> sums;
    AIE_LOOP_UNROLL_FULL
    for (int i = 0; i < 8; i++) {
        aie::accum<accfloat, 16> s_acc;
        s_acc.from_vector(aie::load_v<16>(pS + i * 16));
        sums.set(aie::reduce_add(s_acc.template to_vector<float>()), i);
    }

    aie::accum<accfloat, 8> acc;
    acc.from_vector(sums);
    acc = bf16_limb::mac_f32_bf16_safe(acc, aie::load_v<8>(l), correction_bf16(c));
    aie::store_v(l, acc.template to_vector<float>());
}

#if ROLE_IMPL != ATTN_IMPL_1x8x1
/// @brief Rescale the whole y accumulator by this tile's per-head factors.
///
/// The 1x8x1 flavor does not need this: its attn_fv folds the same rescale into
/// the accumulator it is about to mac into, so the tile is never written back
/// to memory just to be read again.
void calculate_y(float *__restrict y, float *__restrict c) {
    const aie::vector<bf16, 64> CORRECT = correction_bf16_x8(c);

    constexpr int vec_factor = 64;
    for (int i = 0; i < HEAD_DIM / 8; i++) {
        aie::vector<y_acc_dtype, vec_factor> y_vec = aie::load_v<vec_factor>(y);
        aie::accum<accfloat, vec_factor> y_acc =
            bf16_limb::mul_f32_bf16_safe(y_vec, CORRECT);
        aie::store_v(y, y_acc.template to_vector<y_acc_dtype>());
        y += vec_factor;
    }
}
#endif

/// @brief The flash-attention epilogue: o = a / l, a fp32 and o bf16.
///
/// Unlike the per-tile rescale, 1/l is a genuine fp32 value rather than a
/// widened bf16, so this stays an fp32 x fp32 product. It is still cheaper than
/// aie::mul: the divisor is loop-invariant, so its limbs are split once up
/// here, and only the six limb products that land above fp32's mantissa floor
/// are issued -- 6 MACs against the default emulation's 9 VMULs and 8-deep add
/// tree, at the same accuracy.
template <unsigned N>
void scale_div_aie(float *a, bf16* o, float *l) {

    constexpr int vec_factor = 64;

    float* pA1 = a;
    bf16* pO1 = o;

    aie::vector<float, 8> L0 = aie::broadcast<float, 8>(1.0f / l[0]);
    aie::vector<float, 8> L1 = aie::broadcast<float, 8>(1.0f / l[1]);
    aie::vector<float, 8> L2 = aie::broadcast<float, 8>(1.0f / l[2]);
    aie::vector<float, 8> L3 = aie::broadcast<float, 8>(1.0f / l[3]);
    aie::vector<float, 8> L4 = aie::broadcast<float, 8>(1.0f / l[4]);
    aie::vector<float, 8> L5 = aie::broadcast<float, 8>(1.0f / l[5]);
    aie::vector<float, 8> L6 = aie::broadcast<float, 8>(1.0f / l[6]);
    aie::vector<float, 8> L7 = aie::broadcast<float, 8>(1.0f / l[7]);

    const bf16_limb::limbs3<vec_factor> L =
        bf16_limb::split_f32_3(aie::concat(L0, L1, L2, L3, L4, L5, L6, L7));

    for (int d = 0; d < N / vec_factor; d++){
        aie::vector<float, vec_factor> A0 = aie::load_v<vec_factor>(pA1);
        aie::accum<accfloat, vec_factor> AL = bf16_limb::mul_limbs3_f32_tri(L, A0);
        aie::store_v(pO1, AL.template to_vector<bf16>());
        pO1 += vec_factor;
        pA1 += vec_factor;
    }
}

inline void reorder_s(bf16 *__restrict pS, s_tile_t& S0, s_tile_t& S1) {
    aie::vector<bf16, 64> S_up   = aie::load_v<64>(pS);
    aie::vector<bf16, 64> S_down = aie::load_v<64>(pS + 64);

    S0 = aie::concat(aie::filter_even(S_up, 8), aie::filter_even(S_down, 8));
    S1 = aie::concat(aie::filter_odd(S_up, 8),  aie::filter_odd(S_down, 8));
}

constexpr int s_prod_lock = 4;
constexpr int s_cons_lock = 5;
constexpr int v_prod_lock = 2;
constexpr int v_cons_lock = 3;
constexpr int o_prod_lock = 0;
constexpr int o_cons_lock = 1;  
// constexpr int l_prod_lock = 6;
constexpr int l_cons_lock = 8;

extern "C"{

// attn_kv for ATTN_ROLE_GLOBAL, swa_attn_kv for ATTN_ROLE_SWA; see attn_role.h.
void KV_ENTRY(
    bf16 *__restrict s_ping,
    bf16 *__restrict s_pong,
    bf16 *__restrict v_ping,
    bf16 *__restrict v_pong,
    float*__restrict y,
    bf16 *__restrict o,
    int* L
#if !defined(__chess__)
    , int32_t* kv_state
#endif
)
{
#if defined(__chess__)
    alignas(aie::vector_decl_align) static float l[16] = {};
    static bool is_sc_ping = false;
    static bool is_v_ping = false;
#else
    // Peano links a core's statics above its highest buffer, which on the global
    // attention tile leaves no room. l is cleared on every call anyway, and the
    // ping/pong selectors live in a tile buffer (kv_state, see iron/attn.py).
    alignas(aie::vector_decl_align) float l[16];
    int32_t &is_sc_ping = kv_state[0];
    int32_t &is_v_ping = kv_state[1];
#endif
    const aie::vector<float, 16> zero = aie::broadcast<float, 16>(0);

    zero_vectorized<y_acc_dtype, 8 * HEAD_DIM>(y);
    aie::store_v(l, zero);
    _down_lock_acquire(l_cons_lock);
    AIE_LOCK_LOOP_CTR i = L[0];
    do {
        _down_lock_acquire(s_cons_lock);
        is_sc_ping = !is_sc_ping;
        bf16 *s = is_sc_ping ? s_ping : s_pong;
        float *c = (float *)(s + Q_HEADS_PADDED_PER_CU * 16);
        calculate_l(s, c, l);
#if ROLE_IMPL != ATTN_IMPL_1x8x1
        calculate_y(y, c);
#endif
        s_tile_t S0, S1;
        reorder_s(s, S0, S1);
        for (int j = 0; j < 8 / Q_HEADS_PER_GROUP_PADDED; j++){
            _lock_acquire(v_cons_lock);
            is_v_ping = !is_v_ping;
            bf16 *v = is_v_ping ? v_ping : v_pong;
#if ROLE_IMPL == ATTN_IMPL_1x8x1
            attn_fv<HEAD_DIM / 8, GQA_R, GQA_S, GQA_T>(S0, S1, v, y, c, j);
#else
            attn_fv<HEAD_DIM / 8, GQA_R, GQA_S, GQA_T>(S0, S1, v, y, j);
#endif
            _lock_release(v_prod_lock);
        }
        _down_lock_release(s_prod_lock);
        i -= 16;
    } while (i > 0);
    _lock_acquire(o_prod_lock, O_DOWN_REPEATS);
    scale_div_aie<8 * HEAD_DIM>(y, o, l);
    _lock_release(o_cons_lock, O_DOWN_REPEATS);
}
}

#if ROLE_IMPL == ATTN_IMPL_2x4x1

template <unsigned colQ, unsigned r, unsigned s, unsigned t>
void attn_fv(const s_tile_t& S0, const s_tile_t& S1, bf16 *__restrict pV, y_acc_dtype *__restrict pY, int index) {

    using MMUL = aie::mmul<r, s, t, bf16, bf16, accfloat>;

    y_acc_dtype *__restrict pY1 = pY;

    bf16 *__restrict pV1 = pV;

	for (unsigned j = 0; j < colQ; j += 4)
    chess_prepare_for_pipelining chess_loop_range(2, ) {
        //v0
        bf16 *__restrict pV01 = pV1 + (j + 0) * MMUL::size_B * 2;
        bf16 *__restrict pV02 = pV1 + (j + 1) * MMUL::size_B * 2;
        bf16 *__restrict pV03 = pV1 + (j + 2) * MMUL::size_B * 2;
        bf16 *__restrict pV04 = pV1 + (j + 3) * MMUL::size_B * 2;

        aie::vector<bf16, MMUL::size_B> V00 = aie::load_v<MMUL::size_B>(pV01);
        pV01 += MMUL::size_B * colQ;
        aie::vector<bf16, MMUL::size_B> V01 = aie::load_v<MMUL::size_B>(pV02);
        pV02 += MMUL::size_B * colQ;
        aie::vector<bf16, MMUL::size_B> V02 = aie::load_v<MMUL::size_B>(pV03);
        pV03 += MMUL::size_B;
        aie::vector<bf16, MMUL::size_B> V03 = aie::load_v<MMUL::size_B>(pV04);
        pV04 += MMUL::size_B;

        aie::vector<y_acc_dtype, MMUL::size_C> acc_Y00 = aie::load_v<MMUL::size_C>(pY1);
        aie::vector<y_acc_dtype, MMUL::size_C> acc_Y01 = aie::load_v<MMUL::size_C>(pY1 + MMUL::size_C);
        aie::vector<y_acc_dtype, MMUL::size_C> acc_Y02 = aie::load_v<MMUL::size_C>(pY1 + MMUL::size_C * 2);
        aie::vector<y_acc_dtype, MMUL::size_C> acc_Y03 = aie::load_v<MMUL::size_C>(pY1 + MMUL::size_C * 3);

        aie::accum<accfloat, MMUL::size_C> ACC_Y00;
        aie::accum<accfloat, MMUL::size_C> ACC_Y01;
        MMUL Y02(acc_Y02);
        MMUL Y03(acc_Y03);

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

        if(index == 0) {
            auto Y20 = aie::filter_even(Y00.template to_vector<y_acc_dtype>(), 32);
            auto Y21 = aie::filter_even(Y01.template to_vector<y_acc_dtype>(), 32);
            auto Y22 = aie::filter_even(Y02.template to_vector<y_acc_dtype>(), 32);
            auto Y23 = aie::filter_even(Y03.template to_vector<y_acc_dtype>(), 32);

            aie::store_v(pY1, Y20);
            pY1 += MMUL::size_C;
            aie::store_v(pY1, Y21);
            pY1 += MMUL::size_C;
            aie::store_v(pY1, Y22);
            pY1 += MMUL::size_C;
            aie::store_v(pY1, Y23);
            pY1 += MMUL::size_C;
        } else {
            auto Y20 = aie::filter_odd(Y00.template to_vector<y_acc_dtype>(), 32);
            auto Y21 = aie::filter_odd(Y01.template to_vector<y_acc_dtype>(), 32);
            auto Y22 = aie::filter_odd(Y02.template to_vector<y_acc_dtype>(), 32);
            auto Y23 = aie::filter_odd(Y03.template to_vector<y_acc_dtype>(), 32);

            aie::store_v(pY1 + 32, Y20);
            pY1 += MMUL::size_C;
            aie::store_v(pY1 + 32, Y21);
            pY1 += MMUL::size_C;
            aie::store_v(pY1 + 32, Y22);
            pY1 += MMUL::size_C;
            aie::store_v(pY1 + 32, Y23);
            pY1 += MMUL::size_C;
        }
    }
}

#elif ROLE_IMPL == ATTN_IMPL_1x8x1

template <unsigned colQ, unsigned r, unsigned s, unsigned t>
void attn_fv(const s_tile_t& S0, const s_tile_t& S1, bf16 *__restrict pV, y_acc_dtype *__restrict pY, float *__restrict c, int index) {

    using MMUL = aie::mmul<r, s, t, bf16, bf16, accfloat>;

    y_acc_dtype *__restrict pY1 = pY;

    static_assert(MMUL::size_C == 64, "correction_bf16_x8 lays out 8 heads x 8 lanes");
    const aie::vector<bf16, 64> CORRECT = correction_bf16_x8(c);

    bf16 *__restrict pV1 = pV;

	for (unsigned j = 0; j < colQ; j += 4){
        //v0
        bf16 *__restrict pV01 = pV1 + (j + 0) * MMUL::size_B * 2;
        bf16 *__restrict pV02 = pV1 + (j + 1) * MMUL::size_B * 2;
        bf16 *__restrict pV03 = pV1 + (j + 2) * MMUL::size_B * 2;
        bf16 *__restrict pV04 = pV1 + (j + 3) * MMUL::size_B * 2;

        aie::vector<bf16, MMUL::size_B> V00 = aie::load_v<MMUL::size_B>(pV01);
        pV01 += MMUL::size_B;
        aie::vector<bf16, MMUL::size_B> V01 = aie::load_v<MMUL::size_B>(pV02);
        pV02 += MMUL::size_B;
        aie::vector<bf16, MMUL::size_B> V02 = aie::load_v<MMUL::size_B>(pV03);
        pV03 += MMUL::size_B;
        aie::vector<bf16, MMUL::size_B> V03 = aie::load_v<MMUL::size_B>(pV04);
        pV04 += MMUL::size_B;

        aie::vector<y_acc_dtype, MMUL::size_C> acc_y00 = aie::load_v<MMUL::size_C>(pY1);
        aie::vector<y_acc_dtype, MMUL::size_C> acc_y01 = aie::load_v<MMUL::size_C>(pY1 + MMUL::size_C);
        aie::vector<y_acc_dtype, MMUL::size_C> acc_y02 = aie::load_v<MMUL::size_C>(pY1 + MMUL::size_C * 2);
        aie::vector<y_acc_dtype, MMUL::size_C> acc_y03 = aie::load_v<MMUL::size_C>(pY1 + MMUL::size_C * 3);

        // Rescale the running accumulator by this tile's softmax correction on
        // the way into the mmul. CORRECT is bf16-exact, so this is an fp32 x
        // bf16 product: 3 MACs, fp32-exact, in place of the 9-VMUL emulation
        // aie::mul would have issued for two fp32 operands.
        aie::accum<accfloat, MMUL::size_C> ACC_Y00 = bf16_limb::mul_f32_bf16_safe(acc_y00, CORRECT);
        aie::accum<accfloat, MMUL::size_C> ACC_Y01 = bf16_limb::mul_f32_bf16_safe(acc_y01, CORRECT);
        aie::accum<accfloat, MMUL::size_C> ACC_Y02 = bf16_limb::mul_f32_bf16_safe(acc_y02, CORRECT);
        aie::accum<accfloat, MMUL::size_C> ACC_Y03 = bf16_limb::mul_f32_bf16_safe(acc_y03, CORRECT);

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
    
}

#elif ROLE_IMPL == ATTN_IMPL_4x2x1
/// @brief Recursion for store_y_slice(); see there.
template <unsigned Y_SLICE, unsigned N, unsigned I = 0>
struct store_y_slice_impl {
    template <typename V>
    static inline void run(y_acc_dtype *__restrict p, const V& y, int index) {
        // The last slice is the fall-through case, so the chain has one
        // comparison per slice rather than one per slice plus a dead tail.
        if (I + 1 == N || index == (int)I)
            aie::store_v(p + I * Y_SLICE, y.template extract<Y_SLICE>(I));
        else
            store_y_slice_impl<Y_SLICE, N, I + 1>::run(p, y, index);
    }
};

template <unsigned Y_SLICE, unsigned N>
struct store_y_slice_impl<Y_SLICE, N, N> {
    template <typename V>
    static inline void run(y_acc_dtype *, const V&, int) {}
};

/// @brief Store slice @p index of an mmul output tile at the matching offset.
///
/// Written as a branch chain over compile-time slice indices instead of the
/// obvious `extract<Y_SLICE>(index)`: a runtime vector index is not a register
/// shuffle, so the backend would spill the whole accumulator to a stack slot
/// and reload the slice from it. Four of those per loop body is what makes the
/// stack-coloring pass merge slots and drop its alias info ("merging of stack
/// variables requires dropping alias analysis info"). Every extract here is
/// static, so the tile never leaves the register file.
template <unsigned Y_SLICE, typename V>
inline void store_y_slice(y_acc_dtype *__restrict p, const V& y, int index) {
    store_y_slice_impl<Y_SLICE, V::size() / Y_SLICE>::run(p, y, index);
}

template <unsigned colQ, unsigned r, unsigned s, unsigned t>
void attn_fv(const s_tile_t& S0, const s_tile_t& S1, bf16 *__restrict pV, y_acc_dtype *__restrict pY, int index) {

    using MMUL = aie::mmul<r, s, t, bf16, bf16, accfloat>;

    constexpr unsigned Y_SLICE = Q_HEADS_PER_GROUP_PADDED * t;

    y_acc_dtype *__restrict pY1 = pY;
    bf16 *__restrict pV1 = pV;

	for (unsigned j = 0; j < colQ; j += 4){
        //v0
        bf16 *__restrict pV01 = pV1 + (j + 0) * MMUL::size_B * 2;
        bf16 *__restrict pV02 = pV1 + (j + 1) * MMUL::size_B * 2;
        bf16 *__restrict pV03 = pV1 + (j + 2) * MMUL::size_B * 2;
        bf16 *__restrict pV04 = pV1 + (j + 3) * MMUL::size_B * 2;

        aie::vector<bf16, MMUL::size_B> V00 = aie::load_v<MMUL::size_B>(pV01);
        pV01 += MMUL::size_B;
        aie::vector<bf16, MMUL::size_B> V01 = aie::load_v<MMUL::size_B>(pV02);
        pV02 += MMUL::size_B;
        aie::vector<bf16, MMUL::size_B> V02 = aie::load_v<MMUL::size_B>(pV03);
        pV03 += MMUL::size_B;
        aie::vector<bf16, MMUL::size_B> V03 = aie::load_v<MMUL::size_B>(pV04);
        pV04 += MMUL::size_B;

        aie::vector<y_acc_dtype, MMUL::size_C> acc_Y00 = aie::load_v<MMUL::size_C>(pY1);
        aie::vector<y_acc_dtype, MMUL::size_C> acc_Y01 = aie::load_v<MMUL::size_C>(pY1 + MMUL::size_C);
        aie::vector<y_acc_dtype, MMUL::size_C> acc_Y02 = aie::load_v<MMUL::size_C>(pY1 + MMUL::size_C * 2);
        aie::vector<y_acc_dtype, MMUL::size_C> acc_Y03 = aie::load_v<MMUL::size_C>(pY1 + MMUL::size_C * 3);

        MMUL Y00(acc_Y00);
        MMUL Y01(acc_Y01);
        MMUL Y02(acc_Y02);
        MMUL Y03(acc_Y03);

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

        store_y_slice<Y_SLICE>(pY1, Y00.template to_vector<y_acc_dtype>(), index);
        pY1 += MMUL::size_C;
        store_y_slice<Y_SLICE>(pY1, Y01.template to_vector<y_acc_dtype>(), index);
        pY1 += MMUL::size_C;
        store_y_slice<Y_SLICE>(pY1, Y02.template to_vector<y_acc_dtype>(), index);
        pY1 += MMUL::size_C;
        store_y_slice<Y_SLICE>(pY1, Y03.template to_vector<y_acc_dtype>(), index);
        pY1 += MMUL::size_C;
    }
}
#endif