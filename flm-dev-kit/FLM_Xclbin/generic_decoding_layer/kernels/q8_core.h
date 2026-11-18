// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __Q8_CORE_H__
#define __Q8_CORE_H__
#include "aie_kernel_utils.h"
#include "model_spec.h"
#include "aie_array_layout.h"
#include "bf16_limb_math.hpp"

// documented in qmm_core.h
#ifndef QMM_LIMB_PRECISION
#define QMM_LIMB_PRECISION 0
#endif

/// @brief y += W x for one packed Q8_0 weight block, x bf16 / y float32.
///
/// Same four-call interface as the Q4 kernels in q4_core.h, so proj_main.cc does
/// not branch on the format:
///
///     _QMM_zero(acc, y_acc)                             once per output row block
///     _QMM_x_bf16<m, k>(acc, y_acc, x, w, b_reduce)     once per block of the K loop
///     _QMM_finalize(y_acc, acc)                         once the K loop is done
///
/// Q8_0 is symmetric -- w = s q, one bf16 scale per 32-value group, no min -- so
/// there is a single accumulator and b_reduce is unused. The datapath is the
/// same "reduce first, scale after" shape as the Q4 kernels: the whole group is
/// reduced against x in one float32 accumulator and only then meets the group's
/// scale, which keeps every multiplier operand bf16-exact (an int8 quant is an
/// integer below 256, and the scale is *stored* bf16) and leaves the accumulator
/// as the only place a rounding could land. That one is handled with a two-bf16-
/// limb split; see the "Why limbs" note in q4_core.h.
///
/// Q8_0 is only ever packed plain column major over the whole row span
/// (FLM_Q4NX_Converter's _pack_q8nx), so PARALLEL_ROWS is the whole row block,
/// one pass covers it, and the running row can stay in a register across the
/// whole K loop rather than round-tripping through y_acc.

/// @brief One 512-bit int8 load is 64 quants, i.e. this many whole columns of
///        the block.
static constexpr int INT8_PER_LOAD = 64;
static constexpr int Q8_COLS_PER_LOAD = INT8_PER_LOAD / PARALLEL_ROWS;
static constexpr int Q8_LOADS_PER_GROUP = Q4NX_GROUP_SIZE / Q8_COLS_PER_LOAD;

static_assert(INT8_PER_LOAD % PARALLEL_ROWS == 0);
static_assert(Q4NX_GROUP_SIZE % Q8_COLS_PER_LOAD == 0);

/// @brief The running output row, carried in a register for the whole K loop.
///
/// One full-width pass covers the block's rows, so unlike Q4_0/Q4_1 there is no
/// second pass whose accumulator would have to stay live alongside it.
/// @brief y += W x for one Q8_0 32x256 block, the running row in c_accum.
///
/// The body of _QMM_x_bf16 below, as a function of its own so the mixed Q4_1 / Q8_0 build (mix_core.h) can call it too.
inline void _q80_block(aie::accum<accfloat, PARALLEL_ROWS> & acc_c, const bf16 * x, const q80_block_t * w) {
    static_assert(PARALLEL_ROWS == Q4NX_ROW_BLOCK_SIZE, "Q8_0 is plain column major");
    static_assert(ROW_PASSES == 1);
    static constexpr int N = Q4NX_COL_BLOCK_SIZE;
    static constexpr int NUM_GROUPS = N / Q4NX_GROUP_SIZE;

    const int8 * it_quant = w->qs;
    const bf16 * it_scale = w->scales;
    const bf16 * it_x     = x;

    aie::accum<accfloat, PARALLEL_ROWS> c_accum = acc_c;

    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_RANGE( NUM_GROUPS, NUM_GROUPS )
    for (int g = 0; g < NUM_GROUPS; g++) AIE_LOOP_FLATTEN {
        aie::vector<bf16, Q4NX_GROUP_SIZE> b_col =
            aie::load_v<Q4NX_GROUP_SIZE>(it_x);
        it_x += Q4NX_GROUP_SIZE;

        aie::accum<accfloat, PARALLEL_ROWS> t;
        t.from_vector(aie::zeros<bf16, PARALLEL_ROWS>());

        // Unrolled by 4, not fully: a group is Q8_LOADS_PER_GROUP = 16 loads
        // here (an int8 load covers 2 columns, against a nibble load's 8), and
        // unrolling all of them costs ~9 KB of the core's 16 KB program memory
        // for no extra throughput -- the loop is load and convert bound, not
        // accumulator-dependency bound.
        AIE_LOOP_RANGE( Q8_LOADS_PER_GROUP, Q8_LOADS_PER_GROUP )
        AIE_LOOP_UNROLL( 4 )
        for (int l = 0; l < Q8_LOADS_PER_GROUP; l++) {
            aie::vector<int8, INT8_PER_LOAD> q_raw =
                aie::load_v<INT8_PER_LOAD>(it_quant);
            it_quant += INT8_PER_LOAD;

            // int8 -> bf16 is exact
            aie::accum<accfloat, INT8_PER_LOAD> q_acc;
            q_acc.from_vector(aie::to_float(q_raw, 0));
            aie::vector<bf16, INT8_PER_LOAD> q_bf16 =
                q_acc.template to_vector<bf16>();

            AIE_LOOP_UNROLL_FULL
            for (int k = 0; k < Q8_COLS_PER_LOAD; k++) {
                t = aie::mac(t, q_bf16.template extract<PARALLEL_ROWS>(k),
                             b_col.get(l * Q8_COLS_PER_LOAD + k));
            }
        }

        // a_scales is bf16-exact, so the only rounding this MAC could introduce
        // is on the group sum. Split it into two bf16 limbs rather than throwing
        // 16 of its 24 mantissa bits away.
        aie::vector<bf16, PARALLEL_ROWS> a_scales =
            aie::load_v<PARALLEL_ROWS>(it_scale);
        it_scale += PARALLEL_ROWS;
#if QMM_LIMB_PRECISION
        c_accum = bf16_limb::mac_f32_bf16_low(
            c_accum, t.template to_vector<float>(), a_scales);
#else
        c_accum = aie::mac(c_accum, t.template to_vector<bf16>(), a_scales);
#endif
    }

    acc_c = c_accum;
}

#if QUANT_TYPE == __Q8_0__
struct qmm_accum_t {
    aie::accum<accfloat, PARALLEL_ROWS> c;
};

inline void _QMM_zero(qmm_accum_t & acc, float *) {
    acc.c.from_vector(aie::zeros<bf16, PARALLEL_ROWS>());
}

inline void _QMM_finalize(float * c, const qmm_accum_t & acc) {
    aie::store_v(c, acc.c.template to_vector<float>());
}

/// @param acc  the running row, updated in place
/// @param x    activations, bf16, N of them
/// @param w    the weight block
template <int M, int N>
void _QMM_x_bf16(
    qmm_accum_t & acc,
    float *,
    const bf16 * x,
    const QUANT_BLOCK * w,
    const bf16 *
) {
    static_assert(M == Q4NX_ROW_BLOCK_SIZE, "one call is one weight block");
    static_assert(N == Q4NX_COL_BLOCK_SIZE, "one call is one weight block");
    _q80_block(acc.c, x, w);
}
#endif

#endif // __Q8_CORE_H__
