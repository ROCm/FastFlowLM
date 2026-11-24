// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

#include "common_external.hpp"
#include "lock_util.hpp"
#include "zero.h"
#include "passThroughCommon.hpp"
//TODO: align with python or pass in
#include "aie_kernel_utils.h" //TODO: a better method?
#include "bf16_limb_math.hpp"

// ---------------------------------------------------------------------------
// Gated delta rule, token-serial. Each core owns a DK x V_chunk (128 x 32) slab
// of one head's fp32 state S. The DMA delivers and collects it column by column
// (column c = 32 floats, one per V lane, at S + c*V_chunk); while a head is
// being processed it is kept in the interleaved pair layout of trick 4.
//
// The maths per token t:
//     n_t = S_{t-1} k_t
//     u_t = beta_t * (v_t - alpha_t * n_t)
//     S_t = alpha_t * S_{t-1} + u_t k_t^T
//     o_t = S_t q_t / sqrt(DK)
//
// Trick 1 -- lazy decay. Keep S = A * S' with ONE scalar A. The decay then only
// touches A, never the 4096 numbers of S':
//     A_true = A * alpha_t                  (so alpha_t * S_{t-1} = A_true * S'_{t-1})
//     S'_t   = S'_{t-1} + (u_t / A_true) k_t^T,   and A becomes A_true
// The old kernel spent 6 MACs per state element per token on alpha * S; this
// spends 0. A only shrinks, so S' and u/A grow. When A drops below 2^-64 we
// "fold": multiply S' by the power-of-two part of A_true (exact, no rounding)
// and keep only its mantissa, in [1, 2), as the new A.
//
// Trick 2 -- one pass over S' per token, reading only the OLD S'_{t-1}:
//     S_t q_t     = A_true * (S'_{t-1} q_t)     + u_t * (k_t . q_t)
//     S_t k_{t+1} = A_true * (S'_{t-1} k_{t+1}) + u_t * (k_t . k_{t+1})
// so the output o_t and the next token's n_{t+1} come out of the same pass that
// writes S'_t, plus two 128-long dot products per token (done for a whole CT
// chunk at once, in chunk_dots). S' is read and written once per token instead
// of twice. Reading the old column (not the new one)
// splits the per-column work into two chains that do not wait on each other:
//     load -> update -> store,   and   load -> limbs -> o and n.
// The first token of each CT chunk has no previous pass to get n from, so it
// gets a short matvec pass of its own.
//
// Trick 3 (no software floats per token) and trick 4 (interleaved column
// pairs) are explained where they are used, further down.
//
// fp32 products use bf16 limbs (see bf16_limb_math.hpp): an fp32 x is split
// into bf16 pieces x ~= l0 + l1 (+ l2), each multiplied on the native bf16 MAC
// unit and summed in the fp32 accumulator.
// ---------------------------------------------------------------------------

template<typename T>
T* determine_ping_pong_buffer(
    T* ping_buffer, T* pong_buffer,
    int* ping_pong_flag
){
    if(*ping_pong_flag == 0){
        *ping_pong_flag=1;
        return ping_buffer;
    }
    else{
        *ping_pong_flag=0;
        return pong_buffer;
    }

}

constexpr float csqrt(float x)
{
    float r = x;
    for (int i = 0; i < 10; ++i)
        r = 0.5f * (r + x / r);
    return r;
}

// The core has no scalar FPU: float compares and multiplies are software calls
// (~50-100 cycles). Tests on the raw bits are plain integer ops, so all the
// decisions below look at bits.
union f32_bits { float f; int32_t i; };
inline int32_t bits_of(float x)    { f32_bits b; b.f = x; return b.i; }
inline float   float_of(int32_t i) { f32_bits b; b.i = i; return b.f; }

constexpr int32_t F32_EXP_MASK  = 0x7F800000;
constexpr int32_t F32_MANT_MASK = 0x007FFFFF;
constexpr int32_t F32_ONE_BITS  = 0x3F800000;     // 1.0f
// fold when A < 2^-64. A >= 0, so comparing the bits as integers is the same as
// comparing the floats. 2^-64 keeps S' and u/A below 2^64 times their true
// size, far from fp32's 2^127 ceiling.
constexpr int32_t FOLD_BELOW_BITS = (127 - 64) << 23;

// bf16 vector with every lane equal to the float whose bits are f32_bits. Only
// used for powers of two (and 0), whose low 16 bits are zero, so it is exact.
template <int N>
inline aie::vector<bfloat16, N> bf16_bcast_from_bits(int32_t f32_bits)
{
    return aie::broadcast<int16, N>((int16)(f32_bits >> 16)).template cast_to<bfloat16>();
}

// PROFILE=1 builds read the core cycle counter around the parts of each head and
// write the counts over the first 4 words of the head's output state (so S is
// wrong in those builds). Start times sit in statics, not registers: a timer
// value kept live across a call has hung chess-built cores before.
#ifdef PROFILE
static uint64_t prof_head_t0, prof_t0;
static uint32_t prof_pass, prof_prep;
inline uint64_t prof_now() { return aie::tile::current().cycles(); }
#define PROF_START()        (prof_t0 = prof_now())
#define PROF_ADD(counter)   (counter += (uint32_t)(prof_now() - prof_t0))
#else
#define PROF_START()        ((void)0)
#define PROF_ADD(counter)   ((void)0)
#endif

// ---------------------------------------------------------------------------
// Column pairs. The bf16 MAC unit is natively 64 lanes wide (mac_elem_64); a
// 32-lane MAC uses half of it and still occupies a whole accumulator register.
// So the passes below work on two S' columns at once, as one 64-lane vector.
//
// Trick 4 -- interleaved pairs. While a head is being processed, S' is kept
// with each column pair interleaved: for columns c, c+1 the 64 floats are
//     S'[c][0], S'[c+1][0], S'[c][1], S'[c+1][1], ...
// so lane 2i is column c and lane 2i+1 is column c+1. The per-column scalars
// k[c], k[c+1] are adjacent bf16 in memory, i.e. ONE 32-bit word, and
// broadcasting that word gives exactly the multiplier this layout needs:
// k[c], k[c+1], k[c], k[c+1], ... One scalar load and one broadcast per pair
// instead of two of each. The layout is changed once at the start of a head
// (to_pair_layout) and changed back once at the end (from_pair_layout).
// Running sums (o, m) are 64 lanes too; their even and odd lanes are added once
// at the end of the pass.
// ---------------------------------------------------------------------------
template <int V_chunk>
inline aie::vector<bfloat16, 2 * V_chunk> pair_bcast(const bfloat16* __restrict__ a, int c)
{
    const int32_t two = *reinterpret_cast<const int32_t*>(a + c);     // a[c], a[c+1]
    return aie::broadcast<int32, V_chunk>(two).template cast_to<bfloat16>();
}

// x[0], x[0], x[1], x[1], ...: a per-row value, the same for both columns
template <int V_chunk>
inline aie::vector<bfloat16, 2 * V_chunk> pair_dup(aie::vector<bfloat16, V_chunk> x)
{
    return aie::concat(aie::interleave_zip(x, x, 1));
}

// even lanes + odd lanes: the pair sum back to one 32-lane result
template <int V_chunk>
inline aie::vector<float, V_chunk> add_halves(aie::accum<accfloat, 2 * V_chunk> acc)
{
    const aie::vector<float, 2 * V_chunk> v = acc.template to_vector<float>();
    auto [even, odd] = aie::interleave_unzip(v.template extract<V_chunk>(0), v.template extract<V_chunk>(1), 1);
    return aie::add(even, odd);
}

// column-major (as the DMA delivers it) -> interleaved pairs, in place
template <int DK, int V_chunk>
inline void to_pair_layout(float* __restrict__ S)
{
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_RANGE(DK / 2, DK / 2)
    for (int col = 0; col < DK; col += 2) {
        const aie::vector<float, V_chunk> a = aie::load_v<V_chunk>(S + col * V_chunk);
        const aie::vector<float, V_chunk> b = aie::load_v<V_chunk>(S + (col + 1) * V_chunk);
        auto [lo, hi] = aie::interleave_zip(a, b, 1);
        aie::store_v(S + col * V_chunk, lo);
        aie::store_v(S + (col + 1) * V_chunk, hi);
    }
}

// interleaved pairs -> column-major, in place
template <int DK, int V_chunk>
inline void from_pair_layout(float* __restrict__ S)
{
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_RANGE(DK / 2, DK / 2)
    for (int col = 0; col < DK; col += 2) {
        const aie::vector<float, V_chunk> lo = aie::load_v<V_chunk>(S + col * V_chunk);
        const aie::vector<float, V_chunk> hi = aie::load_v<V_chunk>(S + (col + 1) * V_chunk);
        auto [a, b] = aie::interleave_unzip(lo, hi, 1);
        aie::store_v(S + col * V_chunk, a);
        aie::store_v(S + (col + 1) * V_chunk, b);
    }
}

// ---------------------------------------------------------------------------
// m = S' k, the matvec for the first token of a chunk, in the pair layout.
// S' is split into 2 limbs, each with its own accumulator (m0 for l0, m1 for
// l1) so no accumulator is updated twice per step.
// ---------------------------------------------------------------------------
template <int DK, int V_chunk>
inline aie::vector<float, V_chunk> matvec_pass(
    const float* __restrict__ S,
    const bfloat16* __restrict__ k)
{
    constexpr int P = 2 * V_chunk;
    aie::accum<accfloat, P> m0 = aie::zeros<accfloat, P>();
    aie::accum<accfloat, P> m1 = aie::zeros<accfloat, P>();
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_RANGE(DK / 2, DK / 2)
    for (int col = 0; col < DK; col += 2) {
        aie::vector<bfloat16, P> k_pair = pair_bcast<V_chunk>(k, col);
        aie::vector<float, P> s = aie::load_v<P>(S + col * V_chunk);
        auto [s_l0, s_l1] = bf16_limb::split_f32(s);
        m0 = aie::mac(m0, s_l0, k_pair);
        m1 = aie::mac(m1, s_l1, k_pair);
    }
    return aie::add(add_halves<V_chunk>(m0), add_halves<V_chunk>(m1));
}

// ---------------------------------------------------------------------------
// The fused pass for one token (trick 2). Per column c, with old = S'_{t-1}[c]:
//     o   += bf16(old) * q_s[c]               q_s = q / sqrt(DK), in bf16
//     m   += limbs(old) * k_next[c]           only if there is a next token
//     S'[c] = old + w * k[c]                  w = u / A, pre-split into 2 limbs
//             (FOLD: fold_scale * old + w * k[c])
// done for two columns per step (see column pairs above).
// FOLD and HAS_NEXT are template flags, so each of the 4 versions is a plain
// loop with no branch inside.
// S_rd and S_wr point at the same S'. They are two restrict pointers so the
// compiler does not make each load wait for the previous step's store.
// ---------------------------------------------------------------------------
template <bool FOLD, bool HAS_NEXT, int DK, int V_chunk>
__attribute__((noinline)) void update_pass(
    const float* __restrict__ S_rd,
    float* __restrict__ S_wr,
    const bfloat16* __restrict__ k,
    const bfloat16* __restrict__ q_s,
    const bfloat16* __restrict__ k_next,
    aie::vector<bfloat16, V_chunk> w_l0, aie::vector<bfloat16, V_chunk> w_l1,
    aie::vector<bfloat16, V_chunk> fold_scale,     // a power of two (or 0); FOLD only
    aie::vector<float, V_chunk>& o_out,            // S'_{t-1} q_s
    aie::vector<float, V_chunk>& m_out)            // S'_{t-1} k_next, if HAS_NEXT
{
    constexpr int P = 2 * V_chunk;
    const aie::vector<bfloat16, P> one = bf16_limb::ones<P>();
    const aie::vector<bfloat16, P> w_l0_pair = pair_dup<V_chunk>(w_l0);
    const aie::vector<bfloat16, P> w_l1_pair = pair_dup<V_chunk>(w_l1);
    const aie::vector<bfloat16, P> fold_pair = pair_dup<V_chunk>(fold_scale);
    aie::accum<accfloat, P> o = aie::zeros<accfloat, P>();
    aie::accum<accfloat, P> m = aie::zeros<accfloat, P>();

#if !defined(__chess__)
    const int32_t* __restrict__ k_it  = reinterpret_cast<const int32_t*>(k);
    const int32_t* __restrict__ q_it  = reinterpret_cast<const int32_t*>(q_s);
    const int32_t* __restrict__ kn_it = reinterpret_cast<const int32_t*>(k_next);
    const float*   __restrict__ rd    = S_rd;
    float*         __restrict__ wr    = S_wr;
#endif

    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_RANGE(DK / 2, DK / 2)
    AIE_TRY_INITIATION_INTERVAL(4)
    for (int col = 0; col < DK; col += 2) {
#if defined(__chess__)
        aie::vector<bfloat16, P> k_pair = pair_bcast<V_chunk>(k, col);
        aie::vector<bfloat16, P> q_pair = pair_bcast<V_chunk>(q_s, col);

        // fp32 loads straight into the accumulator: no conversion, no MAC
        aie::accum<accfloat, P> old;
        old.from_vector(aie::load_v<P>(S_rd + col * V_chunk));
#else
        aie::vector<bfloat16, P> k_pair = aie::broadcast<int32, V_chunk>(*k_it++).template cast_to<bfloat16>();
        aie::vector<bfloat16, P> q_pair = aie::broadcast<int32, V_chunk>(*q_it++).template cast_to<bfloat16>();
        aie::accum<accfloat, P> old;
        old.from_vector(aie::load_v<P>(rd));
        rd += P;
#endif

        // chain 1: limbs of the old columns -> o and m
        aie::vector<bfloat16, P> l0 = old.template to_vector<bfloat16>();
        o = aie::mac(o, l0, q_pair);

        aie::accum<accfloat, P> s_new;
        if constexpr (FOLD || HAS_NEXT) {
            aie::accum<accfloat, P> r1 = aie::msc(old, l0, one);     // old - l0, exact
            aie::vector<bfloat16, P> l1 = r1.template to_vector<bfloat16>();
            if constexpr (HAS_NEXT) {
#if defined(__chess__)
                aie::vector<bfloat16, P> kn_pair = pair_bcast<V_chunk>(k_next, col);
#else
                aie::vector<bfloat16, P> kn_pair = aie::broadcast<int32, V_chunk>(*kn_it++).template cast_to<bfloat16>();
#endif
                // l0 first: it is ready earlier than l1
                m = aie::mac(m, l0, kn_pair);
                m = aie::mac(m, l1, kn_pair);
            }
            if constexpr (FOLD) {
                // times a power of two is exact per limb, so 3 limbs give c * old exactly
                aie::accum<accfloat, P> r2 = aie::msc(r1, l1, one);
                aie::vector<bfloat16, P> l2 = r2.template to_vector<bfloat16>();
                s_new = aie::mul(l2, fold_pair);
                s_new = aie::mac(s_new, l1, fold_pair);
                s_new = aie::mac(s_new, l0, fold_pair);
                s_new = aie::mac(s_new, k_pair, w_l1_pair);
            }
        }

        // chain 2: rank-1 update, small limb first
        if constexpr (!FOLD) {
            // in place: old is not needed once l0 (and l1) are made
            old = aie::mac(old, k_pair, w_l1_pair);
            old = aie::mac(old, k_pair, w_l0_pair);
#if defined(__chess__)
            aie::store_v(S_wr + col * V_chunk, old.template to_vector<float>());
#else
            aie::store_v(wr, old.template to_vector<float>());
#endif
        } else {
            s_new = aie::mac(s_new, k_pair, w_l0_pair);
#if defined(__chess__)
            aie::store_v(S_wr + col * V_chunk, s_new.template to_vector<float>());
#else
            aie::store_v(wr, s_new.template to_vector<float>());
#endif
        }
#if !defined(__chess__)
        wr += P;
#endif
    }

    o_out = add_halves<V_chunk>(o);
    if constexpr (HAS_NEXT) {
        m_out = add_halves<V_chunk>(m);
    }
}

// S = A * S' in place, at the end of a head, so the state leaves as the true S.
// Element by element, so it works in either layout.
// fp32 x fp32 via 3 limbs each (6 MACs): this S feeds the next prefill chunk
// and decode, so it keeps full precision.
template <int DK, int V_chunk>
inline void unfold_state(const float* __restrict__ S_rd, float* __restrict__ S_wr, float A)
{
    if (bits_of(A) == F32_ONE_BITS) return;
    const bf16_limb::limbs3<V_chunk> a_l = bf16_limb::split_f32_3(aie::broadcast<float, V_chunk>(A));
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_RANGE(DK, DK)
    for (int col = 0; col < DK; col++) {
        aie::vector<float, V_chunk> s = aie::load_v<V_chunk>(S_rd + col * V_chunk);
        aie::accum<accfloat, V_chunk> acc = aie::zeros<accfloat, V_chunk>();
        acc = bf16_limb::mac_limbs3_f32_tri(acc, a_l, s);
        aie::store_v(S_wr + col * V_chunk, acc.template to_vector<float>());
    }
}

// ---------------------------------------------------------------------------
// Trick 3 -- no software floats per token. A_true = A * alpha and
// w_scale = beta / A_after were software calls, ~350 cycles per token. They
// are replaced by one vector computation per CT chunk (up to 16 tokens, one
// per lane), using alpha = mantissa * 2^exponent:
//   - the mantissas (in [1, 2)) are multiplied up in a prefix product across
//     the lanes, so lane t holds mantissa(A) * m_0 * ... * m_t (< 2^17, no
//     underflow is possible);
//   - the exponents are plain integers, summed in the per-token loop. Fold
//     decisions only look at the exponent, so they are integer tests;
//   - 1 / mantissa is Newton's method on all 16 lanes at once.
// A zero (or denormal) alpha gets exponent -100000: the fold test then fires
// with a fold factor of 0, which wipes S' exactly as alpha = 0 should, and
// the exponent sum restarts from there.
// ---------------------------------------------------------------------------

// x * y for fp32 vectors: 3 limbs each, the 6 products that matter
template <unsigned N>
inline aie::vector<float, N> mul_f32(aie::vector<float, N> x, aie::vector<float, N> y)
{
    return bf16_limb::mac_limbs3_tri(aie::zeros<accfloat, N>(),
                                     bf16_limb::split_f32_3(x),
                                     bf16_limb::split_f32_3(y)).template to_vector<float>();
}

// acc + x * y with 2 limbs each (3 MACs, ~16 bits): enough when the result is
// rounded to bf16, or when x itself is only 2-limb accurate
template <unsigned N>
inline aie::accum<accfloat, N> mac_f32_low(aie::accum<accfloat, N> acc, aie::vector<float, N> x, aie::vector<float, N> y)
{
    auto [x0, x1] = bf16_limb::split_f32(x);
    auto [y0, y1] = bf16_limb::split_f32(y);
    acc = aie::mac(acc, x1, y0);
    acc = aie::mac(acc, x0, y1);
    acc = aie::mac(acc, x0, y0);
    return acc;
}

// two scalars as one 64-lane vector: a in lanes 0..31, b in lanes 32..63
template <int V_chunk>
inline aie::vector<float, 2 * V_chunk> pair_of(float a, float b)
{
    return aie::concat(aie::broadcast<float, V_chunk>(a), aie::broadcast<float, V_chunk>(b));
}

// 1/m for m in [1, 2). Newton's method x <- x * (2 - m x) from the straight-line
// guess 24/17 - 8/17 m (the usual 48/17 - 32/17 m is for m in [0.5, 1); this is
// it rescaled to [1, 2)). The guess is off by at most 1/17, and each step
// squares the error: 1/17 -> 3.5e-3 -> 1.2e-5 -> 1.5e-10, past fp32's 6e-8
// after 3 steps.
template <unsigned N>
inline aie::vector<float, N> recip_1_to_2(aie::vector<float, N> m)
{
    aie::vector<float, N> x = aie::sub(aie::broadcast<float, N>(24.0f / 17.0f),
                                       mul_f32(m, aie::broadcast<float, N>(8.0f / 17.0f)));
    AIE_LOOP_UNROLL_FULL
    for (int i = 0; i < 3; i++) {
        x = mul_f32(x, aie::sub(aie::broadcast<float, N>(2.0f), mul_f32(m, x)));
    }
    return x;
}

// Per-chunk scalars (trick 3). In: A (the lazy decay carried in), the alphas
// and betas of the chunk's n_tokens rows. Out, per token t:
//   A_true_bits[t]  : A * alpha_t, the factor for S'_{t-1} in o_t and n_{t+1}
//   fold_bits[t]    : the power of two S'_{t-1} is multiplied by (1.0 = no fold)
//   w_scale_bits[t] : beta_t / A_after, so that w = w_scale * (v - alpha n)
// and A updated to the value after the chunk's last token.
template <int CT_L, int DK, int DV>
__attribute__((noinline)) void chunk_scalars(
    const bfloat16* __restrict__ K, const bfloat16* __restrict__ V,
    int n_tokens, int row_stride, float& A,
    int32_t* __restrict__ A_true_bits, int32_t* __restrict__ fold_bits, int32_t* __restrict__ w_scale_bits)
{
    static_assert(CT_L == 16, "one token per lane of a 16-lane vector");
    alignas(64) int32_t alpha_bits[CT_L];
    alignas(64) int32_t beta_bits[CT_L];
    alignas(64) int32_t expo[CT_L];
    alignas(64) int32_t prod_bits[CT_L];
    alignas(64) int32_t br_bits[CT_L];

    // gather the alpha and beta bits of the chunk. Rows past n_tokens act as
    // alpha = 1, beta = 0 (selects, not branches, so the loop pipelines).
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_RANGE(CT_L, CT_L)
    for (int t = 0; t < CT_L; t++) {
        const bool    live = t < n_tokens;
        const int32_t ab = *reinterpret_cast<const int32_t*>(K + t * row_stride + DK);
        const int32_t bb = *reinterpret_cast<const int32_t*>(V + t * row_stride + DV);
        alpha_bits[t] = live ? ab : F32_ONE_BITS;
        beta_bits[t]  = live ? bb : 0;
    }

    // alpha = mantissa * 2^exponent, all 16 lanes at once with integer vector ops.
    // A zero or denormal alpha (exponent field 0) gets exponent -100000.
    using ivec = aie::vector<int32, CT_L>;
    const ivec ab_v   = aie::load_v<CT_L>(alpha_bits);
    // mask BEFORE shifting: a vector downshift goes through the accumulator's
    // shift-and-round, and with round-to-nearest set, the mantissa bits below
    // would round the exponent up. With them zeroed there is nothing to round.
    const ivec e_field = aie::downshift(aie::bit_and(ab_v, aie::broadcast<int32, CT_L>(F32_EXP_MASK)), 23);
    const ivec e_v    = aie::select(aie::sub(e_field, aie::broadcast<int32, CT_L>(127)),
                                    aie::broadcast<int32, CT_L>(-100000),
                                    aie::eq(e_field, aie::broadcast<int32, CT_L>(0)));
    aie::store_v(expo, e_v);
    const ivec mant_b = aie::bit_or(aie::bit_and(ab_v, aie::broadcast<int32, CT_L>(F32_MANT_MASK)),
                                    aie::broadcast<int32, CT_L>(F32_ONE_BITS));

    // prefix product of the mantissas, with A's mantissa folded into lane 0:
    // after the steps with shifts 1, 2, 4, 8, lane t holds mant(A) * m_0 * ... * m_t
    // (lanes shifted in from below are 1.0). Always < 2^17, so no underflow.
    const int32_t A_in = bits_of(A);
    const aie::vector<float, CT_L> ones = aie::broadcast<float, CT_L>(1.0f);
    aie::vector<float, CT_L> first = ones;
    first[0] = float_of((A_in & F32_MANT_MASK) | F32_ONE_BITS);
    aie::vector<float, CT_L> P = mul_f32(mant_b.template cast_to<float>(), first);
    AIE_LOOP_UNROLL_FULL
    for (int s = 1; s < CT_L; s *= 2) {
        P = mul_f32(P, aie::shuffle_up_fill(P, ones, s));
    }

    // the mantissa of each product, and beta / that mantissa
    const ivec P_b  = P.template cast_to<int32>();
    const ivec Pm_b = aie::bit_or(aie::bit_and(P_b, aie::broadcast<int32, CT_L>(F32_MANT_MASK)),
                                  aie::broadcast<int32, CT_L>(F32_ONE_BITS));
    const aie::vector<float, CT_L> BR =
        mul_f32(aie::load_v<CT_L>(beta_bits).template cast_to<float>(), recip_1_to_2(Pm_b.template cast_to<float>()));
    aie::store_v(prod_bits, P_b);
    aie::store_v(br_bits, BR.template cast_to<int32>());

    // per token, integers only, no branches. A_true = mantissa(P_t) * 2^Y.
    // Only E_run carries from one token to the next.
    int32_t E_run = ((A_in >> 23) & 0xFF) - 127;
    int32_t A_after = A_in;
    AIE_LOOP_RANGE(1, CT_L)
    for (int t = 0; t < n_tokens; t++) {
        E_run += expo[t];
        const int32_t pb      = prod_bits[t];
        const int32_t Y       = ((pb >> 23) & 0xFF) - 127 + E_run;
        const int32_t m1      = (pb & F32_MANT_MASK) | F32_ONE_BITS;   // mantissa, exponent 0
        const bool    normal  = Y >= -126;
        const bool    fold    = Y < -64;     // fold: S' takes 2^Y, A keeps the mantissa
        const int32_t a_true  = normal ? m1 + (Y << 23) : 0;
        const int32_t y_after = fold ? 0 : Y;
        fold_bits[t]   = fold ? (normal ? (127 + Y) << 23 : 0) : F32_ONE_BITS;
        A_true_bits[t] = a_true;
        A_after        = fold ? m1 : a_true;
        E_run         -= fold ? Y : 0;
        // beta / A_after = (beta / mantissa) * 2^-y_after: an exponent shift
        const int32_t br = br_bits[t];
        w_scale_bits[t] = ((br & F32_EXP_MASK) == 0) ? 0 : br - (y_after << 23);
    }
    A = float_of(A_after);
}

// Per-chunk dot products of trick 2, and q / sqrt(DK) in place:
//     Q row t   <- bf16(q_t / sqrt(DK))
//     kq[t]     =  k_t . q_t/sqrt(DK)
//     kk[t]     =  k_t . k_{t+1}          (t + 1 < CT_L)
// One loop over the chunk, so the per-token chain does not wait on them. The Q
// buffer belongs to this core until its lock is released; the DMA refills it.
template <int CT_L, int DQ, int DK, bfloat16 inv_sqrt_DH>
__attribute__((noinline)) void chunk_dots(
    const bfloat16* __restrict__ Q_rd, bfloat16* __restrict__ Q_wr,   // the same Q, split so
    const bfloat16* __restrict__ K, int row_stride,                    // loads need not wait on stores
    float* __restrict__ kq, float* __restrict__ kk)
{
    static_assert(DQ == DK, "q and k rows are dotted together");
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_RANGE(CT_L, CT_L)
    for (int t = 0; t < CT_L; t++) {
        const bfloat16* q = Q_rd + t * row_stride;
        bfloat16* q_out = Q_wr + t * row_stride;
        const bfloat16* k = K + t * row_stride;
        // the last row's k_{t+1} would be past the buffer: dot it with itself, unused
        const bfloat16* kn = (t + 1 < CT_L) ? k + row_stride : k;
        aie::accum<accfloat, 32> qk = aie::zeros<accfloat, 32>();
        aie::accum<accfloat, 32> kk_acc = aie::zeros<accfloat, 32>();
        AIE_LOOP_UNROLL_FULL
        for (int i = 0; i < DK / 32; i++) {
            const aie::vector<bfloat16, 32> k_v = aie::load_v<32>(k + i * 32);
            const aie::vector<bfloat16, 32> q_v =
                aie::mul(aie::load_v<32>(q + i * 32), inv_sqrt_DH).template to_vector<bfloat16>();
            aie::store_v(q_out + i * 32, q_v);
            qk     = aie::mac(qk, k_v, q_v);
            kk_acc = aie::mac(kk_acc, k_v, aie::load_v<32>(kn + i * 32));
        }
        kq[t] = aie::reduce_add(qk.template to_vector<float>());
        kk[t] = aie::reduce_add(kk_acc.template to_vector<float>());
    }
}

// ---------------------------------------------------------------------------
// All tokens of one CT chunk (up to CT_L rows of Q/K/V).
// Rows are row_stride bf16 apart. The fp32 alpha sits in the 2 bf16 slots after
// K[DK], beta after V[DV]. A carries the lazy decay across chunks.
// ---------------------------------------------------------------------------
template <int CT_L, int DQ, int DK, int DV, int V_chunk, bfloat16 inv_sqrt_DH>
inline void process_chunk(
    float* __restrict__ S,
    bfloat16* __restrict__ Q, const bfloat16* __restrict__ K, const bfloat16* __restrict__ V,
    bfloat16* __restrict__ out,
    int n_tokens, int row_stride, int chunk_idx,
    float& A)
{
    constexpr int P = 2 * V_chunk;
    int32_t A_true_bits[CT_L], fold_bits[CT_L], w_scale_bits[CT_L];
    float kq_all[CT_L], kk_all[CT_L];

    // n = S k_0 = A * (S' k_0) for the chunk's first token. Uses A from before
    // the chunk, so it runs before chunk_scalars() moves A on.
    PROF_START();
    aie::vector<float, V_chunk> n =
        mul_f32(matvec_pass<DK, V_chunk>(S, K), aie::broadcast<float, V_chunk>(A));
    chunk_scalars<CT_L, DK, DV>(K, V, n_tokens, row_stride, A, A_true_bits, fold_bits, w_scale_bits);
    chunk_dots<CT_L, DQ, DK, inv_sqrt_DH>(Q, Q, K, row_stride, kq_all, kk_all);   // also scales Q in place
    PROF_ADD(prof_prep);

    const aie::vector<bfloat16, V_chunk> one = bf16_limb::ones<V_chunk>();
    aie::vector<float, V_chunk> o_p, m_p;    // S'_{t-1} q_s and S'_{t-1} k_{t+1}

    for (int t = 0; t < n_tokens; t++) {
        const bfloat16* k = K + t * row_stride;
        const bfloat16* q_s = Q + t * row_stride;   // already q / sqrt(DK), by chunk_dots
        const bfloat16* v = V + t * row_stride + chunk_idx * V_chunk;
        const bfloat16* k_next = k + row_stride;  // only read when t + 1 < n_tokens
        const bool has_next = (t + 1 < n_tokens);

        const float alpha   = *reinterpret_cast<const float*>(k + DK);
        const float beta    = *reinterpret_cast<const float*>(V + t * row_stride + DV);
        const float A_true  = float_of(A_true_bits[t]);
        const float w_scale = float_of(w_scale_bits[t]);
        const bool  fold    = fold_bits[t] != F32_ONE_BITS;

        // ---- d = v - alpha * n, then [u | w] = [d | d] * [beta | w_scale] ----
        aie::accum<accfloat, V_chunk> d = aie::mul(aie::load_v<V_chunk>(v), one);   // exact
        d = bf16_limb::msc_limbs3_f32_tri(
                d, bf16_limb::split_f32_3(aie::broadcast<float, V_chunk>(alpha)), n);
        const aie::vector<float, V_chunk> d_f = d.template to_vector<float>();
        const aie::vector<float, P> uw = mul_f32(aie::concat(d_f, d_f), pair_of<V_chunk>(beta, w_scale));
        const aie::vector<float, V_chunk> u = uw.template extract<V_chunk>(0);
        auto [w_l0, w_l1] = bf16_limb::split_f32(uw.template extract<V_chunk>(1));

        const float kq = kq_all[t];
        const float kk = kk_all[t];

        // ---- the pass over S' ----
        const aie::vector<bfloat16, V_chunk> fold_scale = bf16_bcast_from_bits<V_chunk>(fold_bits[t]);
        PROF_START();
        if (fold) {
            if (has_next) update_pass<true,  true,  DK, V_chunk>(S, S, k, q_s, k_next, w_l0, w_l1, fold_scale, o_p, m_p);
            else          update_pass<true,  false, DK, V_chunk>(S, S, k, q_s, k_next, w_l0, w_l1, fold_scale, o_p, m_p);
        } else {
            if (has_next) update_pass<false, true,  DK, V_chunk>(S, S, k, q_s, k_next, w_l0, w_l1, fold_scale, o_p, m_p);
            else          update_pass<false, false, DK, V_chunk>(S, S, k, q_s, k_next, w_l0, w_l1, fold_scale, o_p, m_p);
        }
        PROF_ADD(prof_pass);

        // ---- [o_t | n_{t+1}] = A_true * [S'_{t-1} q_s | S'_{t-1} k_{t+1}] + [u | u] * [k.q_s | k.k_{t+1}] ----
        aie::accum<accfloat, P> on = aie::zeros<accfloat, P>();
        on = mac_f32_low(on, aie::concat(o_p, m_p), aie::broadcast<float, P>(A_true));
        on = mac_f32_low(on, aie::concat(u, u), pair_of<V_chunk>(kq, kk));
        aie::store_v(out + t * V_chunk, on.template extract<V_chunk>(0).template to_vector<bfloat16>());
        n = on.template extract<V_chunk>(1).template to_vector<float>();
    }
}

extern "C" {

    //TODO: for now
    constexpr int DK = 128;
    constexpr int DV = 128;
    constexpr int DQ = 128;
    constexpr int V_chunk = 32;
    constexpr int CT_L_size = 16;
    constexpr int MT_L_size = 128;

    static constexpr bfloat16 inv_sqrt_DH = 1.0f / csqrt(static_cast<float>(DQ));

    ///TODO: have the lock id sync with python version
    // dq, dk, dv, chunk_size
    void gate_delta_net_prefill_128_128_128_32_16_128(
        float * S_in_chunk, // size of V_chunk x DK  in col-major
        bfloat16* Q_ping, bfloat16* Q_pong, // size of CT_L_sizex padded(DQ +2), 2 is padding in this case        Due to ptr alignment issuye, the padded(DQ+2) is being padded to padded_QKV_size
        bfloat16* K_ping, bfloat16* K_pong, // size of CT_L_sizex padded(DK +2), 2 for a scalar of float of alpha Due to ptr alignment issuye, the padded(DK+2) is being padded to padded_QKV_size
        bfloat16* V_ping, bfloat16* V_pong, // size of CT_L_sizex padded(DV +2), 2 for a scalar of float of beta  Due to ptr alignment issuye, the padded(DV+2) is being padded to padded_QKV_size
        bfloat16* out_ping, bfloat16* out_pong,// size of CT_L_size x V_chunk
        int padded_QKV_size,

        int chunk_idx, // which chunk of V to process, assume chunk_idx <= DV/V_chunk
        int* ping_pong_flag,// size of 32 ints, 0 for ping, 1 for pong
        int* rtp_buffer,
        int CT_S_IN_prod_lock_id, int CT_S_IN_con_lock_id,
        int CT_Q_prod_lock_id, int CT_Q_con_lock_id,
        int CT_K_prod_lock_id, int CT_K_con_lock_id,
        int CT_V_prod_lock_id, int CT_V_con_lock_id,
        int CT_output_prod_lock_id, int CT_output_con_lock_id
    ){
        // the bf16 roundings (output, limb splits) are round-to-nearest-even, not
        // the default truncation, so they are unbiased
        aie::set_rounding(aie::rounding_mode::conv_even);

        // the out_ping/pong buffer is used to send both output and final S_out
        static_assert(  V_chunk* DK*sizeof(float) %   CT_L_size*V_chunk*sizeof(bfloat16) == 0);
        constexpr int iter_for_send_S =   (V_chunk* DK*sizeof(float) )/ ( CT_L_size*V_chunk*sizeof(bfloat16));

        // rtp_buffer[0] tell the number_of_head(S) this CT will process
        // rtp_buffer[1] tell the L size per number_of_head(S), L is not padded

        for(int head_idx = 0; head_idx < rtp_buffer[0]; head_idx++){

            // acquire the S for this head
            acquire_greater_equal(CT_S_IN_con_lock_id, 1 );
            // note: rtp_buffer[1] is not necessarily divisible by CT_L_size

            float A = 1.0f;     // lazy decay: S = A * S_in_chunk, see trick 1 above
#ifndef DMA_ONLY
            to_pair_layout<DK, V_chunk>(S_in_chunk);    // trick 4
#endif
#ifdef PROFILE
            prof_pass = 0;
            prof_prep = 0;
            prof_head_t0 = prof_now();
#endif

            for(int MT_L_idx = 0; MT_L_idx< rtp_buffer[1]; MT_L_idx+= MT_L_size){

                // because data in MT is transaction in size of MT_L_size, thus we need to loop over the MT_L_size
                //

                for(int ct_l_chunk_idx = 0; ct_l_chunk_idx < MT_L_size; ct_l_chunk_idx+= CT_L_size){

                    int L_idx = MT_L_idx + ct_l_chunk_idx;

                    bfloat16* cur_Q_buffer = determine_ping_pong_buffer(Q_ping, Q_pong, &ping_pong_flag[1]);
                    bfloat16* cur_K_buffer = determine_ping_pong_buffer(K_ping, K_pong, &ping_pong_flag[2]);
                    bfloat16* cur_V_buffer = determine_ping_pong_buffer(V_ping, V_pong, &ping_pong_flag[3]);
                    bfloat16* cur_out_buffer = determine_ping_pong_buffer(out_ping, out_pong, &ping_pong_flag[4]);

                    int cur_iteration_size = (L_idx + CT_L_size <= rtp_buffer[1])? CT_L_size : (rtp_buffer[1] - L_idx);

                    // acquire q, k, v lock
                    // acquire o lock
                    acquire_greater_equal(CT_Q_con_lock_id, 1);
                    acquire_greater_equal(CT_K_con_lock_id, 1);
                    acquire_greater_equal(CT_V_con_lock_id, 1);
                    acquire_greater_equal(CT_output_prod_lock_id, 1);
#ifndef DMA_ONLY
                    // chunks past the end of L still cycle the locks, with no tokens
                    if (cur_iteration_size > 0) {
                        process_chunk<CT_L_size, DQ, DK, DV, V_chunk, inv_sqrt_DH>(
                            S_in_chunk,
                            cur_Q_buffer, cur_K_buffer, cur_V_buffer,
                            cur_out_buffer,
                            cur_iteration_size, padded_QKV_size, chunk_idx,
                            A
                        );
                    }
#else
                    // DMA_ONLY: no math. Every lock and buffer swap still happens,
                    // so the run time is the data-movement floor.
                    (void)cur_iteration_size;
#endif
                    // release q, k, v lock
                    //release o lock
                    release(CT_output_con_lock_id, 1);
                    release(CT_V_prod_lock_id,1);
                    release(CT_K_prod_lock_id,1);
                    release(CT_Q_prod_lock_id, 1);

                }

            }

#ifndef DMA_ONLY
            unfold_state<DK, V_chunk>(S_in_chunk, S_in_chunk, A);
            from_pair_layout<DK, V_chunk>(S_in_chunk);
#ifdef PROFILE
            {
                uint32_t* prof_out = reinterpret_cast<uint32_t*>(S_in_chunk);
                prof_out[0] = (uint32_t)(prof_now() - prof_head_t0);   // whole head
                prof_out[1] = prof_pass;                                // inside update_pass calls
                prof_out[2] = prof_prep;                                // matvec + chunk_scalars + chunk_dots
                prof_out[3] = (uint32_t)rtp_buffer[1];                  // tokens
            }
#endif
#endif

            // for loop of acquire-release o lock for sending out the S for this head
            for(int i = 0; i < iter_for_send_S; i++){
                bfloat16* cur_out_buffer = determine_ping_pong_buffer(out_ping, out_pong, &ping_pong_flag[4]);
                constexpr int S_elements_per_iteration = (CT_L_size*V_chunk*sizeof(bfloat16)) / sizeof(float);
                acquire_greater_equal(CT_output_prod_lock_id, 1);
                copy_float_to_float< S_elements_per_iteration, 32>(
                    (float*)cur_out_buffer, S_in_chunk+ i*S_elements_per_iteration
                );
                release(CT_output_con_lock_id, 1);
            }

            // release the S lock for this head,
            release(CT_S_IN_prod_lock_id, 1);
        }

    }
} // ex
