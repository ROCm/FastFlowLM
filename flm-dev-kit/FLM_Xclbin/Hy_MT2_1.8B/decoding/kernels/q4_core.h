// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __Q4_CORE_H__
#define __Q4_CORE_H__
#include "aie_kernel_utils.h"
#include "model_spec.h"
#include "aie_array_layout.h"
#include "bf16_limb_math.hpp"

/// @brief y += W x for one packed weight block, x bf16 / y float32.
///
/// One implementation per QUANT_TYPE -- Q4_1, Q4_K -- behind a single
/// four-call interface so proj_main.cc does not branch on the format:
///
///     _QMM_zero(acc, y_acc)                             once per output row block
///     _QMM_x_bf16<m, k>(acc, y_acc, x, w, b_reduce)     once per block of the K loop
///     _QMM_finalize(y_acc, acc)                         once the K loop is done
///
/// Both share the same datapath shape, "reduce first, scale after": the
/// whole GROUP_SIZE-value group is reduced against x in one float32
/// accumulator and only then meets the group's scale. That is what makes every
/// multiplier operand bf16-exact -- q is a uint4/int4, Q4_K's s'/m' are uint8
/// (bf16 holds every integer up to 256), and every bf16 scale is *stored* bf16
/// -- so a rounding can only ever land on an accumulator.
///
/// The one rounding left is the group sum on its way into the scale MAC, and it
/// is the largest error the kernel introduces: ~2e-03 relative, once per group,
/// correlated across the groups that share a scale, and it does not average away
/// over the K loop -- |y| ends up far smaller than the group sums that build it,
/// so the error is measured against the small number. QMM_LIMB_PRECISION below
/// buys it back with two-bf16-limb splits, at a cost in cycles; it is off by
/// default. The min path follows the same switch -- sum_i x_i arrives from
/// proj_main pre-split as two bf16 limbs either way, and at 0 only the high one
/// is read.
///
/// Layout differs by format and is described on the structs in model_spec.h.
/// PARALLEL_ROWS -- the packing stride, from the model header -- says how many
/// rows one vector operand covers, and a block's rows take ROW_PASSES passes of
/// that width: two half-width passes for Q4_1 weights packed with the
/// parallel-strided re-order, one full-width pass for plain column major ones.
/// Q4_K is only ever packed plain column major, so it is always the latter.

// QMM_LIMB_PRECISION is documented in qmm_core.h; default it here too so this
// header still compiles if included on its own.
#ifndef QMM_LIMB_PRECISION
#define QMM_LIMB_PRECISION 0
#endif

/// @brief One 512-bit uint4/int4 load is 128 nibbles, i.e. this many whole
///        columns of the block.
static constexpr int NIBBLES_PER_LOAD = 128;
static constexpr int COLS_PER_LOAD    = NIBBLES_PER_LOAD / PARALLEL_ROWS;
static constexpr int LOADS_PER_GROUP  = Q4NX_GROUP_SIZE / COLS_PER_LOAD;

/// @brief Independent MAC chains through the group's inner loop.
///
/// A full-width (32-row) MAC already covers its own accumulator dependency, so
/// there is nothing to interleave against; a half-width one does not, and two
/// chains keep back-to-back MACs from serialising. The caller folds them
/// together in the accumulator domain, so this is free of arithmetic effect.
static constexpr int ACC_CHAINS = (PARALLEL_ROWS >= 32) ? 1 : 2;

static_assert(NIBBLES_PER_LOAD % PARALLEL_ROWS == 0);
static_assert(Q4NX_GROUP_SIZE % COLS_PER_LOAD == 0);
static_assert(COLS_PER_LOAD % ACC_CHAINS == 0);

/// @brief acc += scale * sum_i q_i x_i over one group, PARALLEL_ROWS rows at once.
///
/// Reduce first, scale after: the group is reduced against x in ACC_CHAINS
/// float32 accumulators and only then meets the group's scale. Everything is
/// written against accumulators held by reference and each chain opens with a
/// mul rather than a mac into a zeroed accumulator -- both to keep the scheduler
/// from spilling accumulators to VMOVs, which is what the shape of this loop
/// costs if it is written the obvious way.
///
/// @param acc       running result for these rows, updated in place
/// @param it_quant  quant cursor, advanced past the group
/// @param b_col     the group's activations
/// @param scale     the group's scale, bf16-exact
template <typename quant_t>
__attribute__((always_inline))
inline void _group_mac(
    aie::accum<accfloat, PARALLEL_ROWS> & acc,
    const quant_t * & it_quant,
    const aie::vector<bf16, Q4NX_GROUP_SIZE> & b_col,
    const aie::vector<bf16, PARALLEL_ROWS> & scale
) {
    aie::accum<accfloat, PARALLEL_ROWS> t0;
    aie::accum<accfloat, PARALLEL_ROWS> t1;

    AIE_LOOP_UNROLL_FULL
    for (int l = 0; l < LOADS_PER_GROUP; l++) {
        aie::vector<quant_t, NIBBLES_PER_LOAD> q_raw =
            aie::load_v<NIBBLES_PER_LOAD>(it_quant);
        it_quant += NIBBLES_PER_LOAD / 2; // a uint4*/int4* steps two nibbles

        // uint4 / int4 -> bf16 is exact
        aie::accum<accfloat, NIBBLES_PER_LOAD> q_acc;
        q_acc.from_vector(aie::to_float(q_raw, 0));
        aie::vector<bf16, NIBBLES_PER_LOAD> q_bf16 =
            q_acc.template to_vector<bf16>();

        AIE_LOOP_UNROLL_FULL
        for (int k = 0; k < COLS_PER_LOAD; k += ACC_CHAINS) {
            const int col = l * COLS_PER_LOAD + k;
            auto q0 = q_bf16.template extract<PARALLEL_ROWS>(k);
            if (col == 0) t0 = aie::mul(q0, b_col.get(col));
            else          t0 = aie::mac(t0, q0, b_col.get(col));
            if constexpr (ACC_CHAINS == 2) {
                auto q1 = q_bf16.template extract<PARALLEL_ROWS>(k + 1);
                if (col == 0) t1 = aie::mul(q1, b_col.get(col + 1));
                else          t1 = aie::mac(t1, q1, b_col.get(col + 1));
            }
        }
    }

#if QMM_LIMB_PRECISION
    // Fold the chains in the accumulator domain -- one full-precision sum to
    // split instead of two half sums -- then split it into two bf16 limbs
    // rather than throwing 16 of its 24 mantissa bits away.
    aie::accum<accfloat, PARALLEL_ROWS> t = t0;
    if constexpr (ACC_CHAINS == 2) t = aie::add(t, t1);
    acc = bf16_limb::mac_f32_bf16_low(acc, t.template to_vector<float>(), scale);
#else
    // One MAC per chain, each rounded to bf16 on the way in. Folding the chains
    // first would look cheaper and is not: an accumulator add is not one op.
    acc = aie::mac(acc, t0.template to_vector<bf16>(), scale);
    if constexpr (ACC_CHAINS == 2)
        acc = aie::mac(acc, t1.template to_vector<bf16>(), scale);
#endif
}

#if QUANT_TYPE == __Q4_1__

/// @brief Nothing to carry: Q4_1 walks the block in ROW_PASSES half-width
///        passes, so both passes' accumulators would have to stay live across
///        the whole K loop. The float32 y_acc buffer carries the row instead --
///        it is float32 end to end, so the round trip costs bandwidth, not bits.
struct qmm_accum_t {};

inline void _QMM_zero(qmm_accum_t &, float * c) {
    zero_vectorized<float, Q4NX_ROW_BLOCK_SIZE>(c);
}

inline void _QMM_finalize(float *, const qmm_accum_t &) {}

/// @param c              float32 running result, Q4NX_ROW_BLOCK_SIZE of them
/// @param x              activations, bf16, N of them
/// @param w              the weight block
/// @param b_col_reduce   sum_i x_i per group as two interleaved bf16
///                       limbs (hi, lo), shared across all row passes
template <int M, int N>
void _QMM_x_bf16(
    qmm_accum_t &,
    float * c,
    const bf16 * x,
    const QUANT_BLOCK * w,
    const bf16 * b_col_reduce
) {
    static_assert(M == Q4NX_ROW_BLOCK_SIZE, "one call is one weight block");
    static_assert(N == Q4NX_COL_BLOCK_SIZE, "one call is one weight block");
    static constexpr int NUM_GROUPS = N / Q4NX_GROUP_SIZE;

    const uint4 * it_quant = w->qs;

    // The passes share one quant cursor: pass 0 walks the first 16-row plane,
    // pass 1 picks up where it left off.
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_RANGE( ROW_PASSES, ROW_PASSES )
    for (int row = 0; row < M; row += PARALLEL_ROWS) {
        aie::accum<accfloat, PARALLEL_ROWS> c_accum;
        c_accum.from_vector(aie::load_v<PARALLEL_ROWS>(c + row));

        const bf16 * it_x      = x;
        const bf16 * it_scale  = w->scales + row;
#if QUANT_HAS_MIN
        const bf16 * it_min    = w->mins + row;
        const bf16 * it_reduce = b_col_reduce;
#endif

        AIE_PREPARE_FOR_PIPELINING
        AIE_LOOP_RANGE( NUM_GROUPS, NUM_GROUPS )
        for (int g = 0; g < NUM_GROUPS; g++) AIE_LOOP_FLATTEN {
            aie::vector<bf16, Q4NX_GROUP_SIZE> b_col =
                aie::load_v<Q4NX_GROUP_SIZE>(it_x);
            it_x += Q4NX_GROUP_SIZE;

            // a_scales is bf16-exact, so the only rounding in here is on the
            // group sum -- which is what QMM_LIMB_PRECISION trades cycles for.
            aie::vector<bf16, PARALLEL_ROWS> a_scales =
                aie::load_v<PARALLEL_ROWS>(it_scale);
            it_scale += Q4NX_ROW_BLOCK_SIZE;
            _group_mac(c_accum, it_quant, b_col, a_scales);

#if QUANT_HAS_MIN
            // sum_i x_i arrives pre-split as two bf16 limbs, low one first so
            // the small term joins c_accum before it meets the large one; at
            // QMM_LIMB_PRECISION 0 only the high limb is used, which is the
            // plain bf16 column sum the kernel always had.
            aie::vector<bf16, PARALLEL_ROWS> a_mins =
                aie::load_v<PARALLEL_ROWS>(it_min);
            it_min += Q4NX_ROW_BLOCK_SIZE;
#if QMM_LIMB_PRECISION
            c_accum = aie::mac(c_accum, a_mins, *(it_reduce + 1));
#endif
            c_accum = aie::mac(c_accum, a_mins, *it_reduce);
            it_reduce += 2;
#endif
        }

        aie::store_v(c + row, c_accum.template to_vector<float>());
    }
}

#elif QUANT_TYPE == __Q4_K__

/// @brief The scale path and the min path of a whole output row.
///
/// Q4_K dequantizes as w = S' s'_j q - M' m'_j, so the two paths only need to
/// meet once. Keeping them in two float32 accumulators that the caller carries
/// across the whole K loop means the min term is never folded in inside the
/// inner loop, nothing round-trips through memory between blocks, and neither
/// path depends on the other. _QMM_finalize() does the single combining add.
struct qmm_accum_t {
    aie::accum<accfloat, PARALLEL_ROWS> s;
    aie::accum<accfloat, PARALLEL_ROWS> m;
};

inline void _QMM_zero(qmm_accum_t & acc, float *) {
    acc.s.from_vector(aie::zeros<bf16, PARALLEL_ROWS>());
    acc.m.from_vector(aie::zeros<bf16, PARALLEL_ROWS>());
}

/// @brief Collapse the accumulator pair into the final result, once per row block.
inline void _QMM_finalize(float * c, const qmm_accum_t & acc) {
    aie::accum<accfloat, PARALLEL_ROWS> out = aie::add(acc.s, acc.m);
    aie::store_v(c, out.template to_vector<float>());
}

/// @param acc            the two running paths, updated in place
/// @param x              activations, bf16, N of them
/// @param w              the weight block, which is exactly one super-block
/// @param b_col_reduce   sum_i x_i per group as two interleaved bf16 limbs
template <int M, int N>
void _QMM_x_bf16(
    qmm_accum_t & acc,
    float *,
    const bf16 * x,
    const QUANT_BLOCK * w,
    const bf16 * b_col_reduce
) {
    static constexpr int SUPER_BLOCK_SIZE = 256;
    static constexpr int GROUPS_PER_SB    = SUPER_BLOCK_SIZE / Q4NX_GROUP_SIZE;

    static_assert(M == Q4NX_ROW_BLOCK_SIZE, "one call is one weight block");
    static_assert(PARALLEL_ROWS == Q4NX_ROW_BLOCK_SIZE, "Q4_K is plain column major");
    // one weight block is exactly one super-block, so there is no super-block loop
    static_assert(N == SUPER_BLOCK_SIZE, "a Q4_K block must be one super-block");
    static_assert(ROW_PASSES == 1);

    const uint4 * it_quant  = w->qs;
    const uint8 * it_scale  = w->scales;
    const uint8 * it_min    = w->mins;
    const bf16  * it_reduce = b_col_reduce;
    const bf16  * it_x      = x;

    // sum_j s'_j * (sum_i q_i x_i) and sum_j m'_j * (sum_i x_i), both carried in
    // float32 for the whole super-block and both fed through a two-limb split,
    // so neither reduction passes through a bf16 on its way in.
    aie::accum<accfloat, PARALLEL_ROWS> acc_s;
    aie::accum<accfloat, PARALLEL_ROWS> acc_m;
    acc_s.from_vector(aie::zeros<bf16, PARALLEL_ROWS>());
    acc_m.from_vector(aie::zeros<bf16, PARALLEL_ROWS>());

    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_RANGE( GROUPS_PER_SB, GROUPS_PER_SB )
    for (int g = 0; g < GROUPS_PER_SB; g++) {
        aie::vector<bf16, Q4NX_GROUP_SIZE> b_col =
            aie::load_v<Q4NX_GROUP_SIZE>(it_x);
        it_x += Q4NX_GROUP_SIZE;

        // uint8 s' / m' -> bf16, exact for every value up to 255
        aie::accum<accfloat, PARALLEL_ROWS> s_acc;
        s_acc.from_vector(aie::to_float(aie::load_v<PARALLEL_ROWS>(it_scale), 0));
        _group_mac(acc_s, it_quant, b_col, s_acc.template to_vector<bf16>());

        aie::accum<accfloat, PARALLEL_ROWS> m_acc;
        m_acc.from_vector(aie::to_float(aie::load_v<PARALLEL_ROWS>(it_min), 0));
        aie::vector<bf16, PARALLEL_ROWS> a_mins = m_acc.template to_vector<bf16>();
#if QMM_LIMB_PRECISION
        acc_m = aie::mac(acc_m, a_mins, *(it_reduce + 1));
#endif
        acc_m = aie::mac(acc_m, a_mins, *it_reduce);

        it_reduce += 2;
        it_scale  += PARALLEL_ROWS;
        it_min    += PARALLEL_ROWS;
    }

    // S' * acc_s and M' * acc_m go to their own running accumulators. Both
    // multiplicands are bf16-exact (S'/M' are *stored* bf16), so the only
    // rounding here would be on the accumulators -- and it would land before the
    // scale/min cancellation in _QMM_finalize, where a relative error is
    // amplified ~2.3x (quant.md, "cancellation ratio"). Same reasoning as the
    // group sum above, one level up and twice: split each into two bf16 limbs,
    // for 2 extra MACs and 2 msc per block against the ~800 vector ops above.
    aie::vector<bf16, PARALLEL_ROWS> a_S = aie::load_v<PARALLEL_ROWS>(w->S);
    aie::vector<bf16, PARALLEL_ROWS> a_M = aie::load_v<PARALLEL_ROWS>(w->M);
    acc.s = bf16_limb::mac_f32_bf16_low(acc.s, acc_s.template to_vector<float>(), a_S);
    acc.m = bf16_limb::mac_f32_bf16_low(acc.m, acc_m.template to_vector<float>(), a_M);
}

#endif

#endif // __Q4_CORE_H__
