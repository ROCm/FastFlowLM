// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#ifndef __MIX_CORE_H__
#define __MIX_CORE_H__
#include "aie_kernel_utils.h"
#include "model_spec.h"
#include "aie_array_layout.h"
#include "bf16_limb_math.hpp"
#include "q4_core.h"   // _group_mac (its per-format _QMM_* are not compiled for the mixed type)
#include "q8_core.h"   // _q80_block

/// @brief The mixed Q4_1 / Q8_0 build (__Q41_Q80_MIX__): both formats are plain column major over the whole 32-row block, so
///        each takes one full-width pass and the running row stays in one register for the whole K loop, whatever the format.
///        proj_main.cc picks the format per row block and calls _q41_block / _q80_block directly.
static_assert(PARALLEL_ROWS == Q4NX_ROW_BLOCK_SIZE, "the mixed build packs both formats plain column major");

struct qmm_accum_t {
    aie::accum<accfloat, PARALLEL_ROWS> c;
};

inline void _QMM_zero(qmm_accum_t & acc, float *) {
    acc.c.from_vector(aie::zeros<bf16, PARALLEL_ROWS>());
}

inline void _QMM_finalize(float * c, const qmm_accum_t & acc) {
    aie::store_v(c, acc.c.template to_vector<float>());
}

/// @brief y += W x for one Q4_1 32x256 block (w = s q + m), one full-width pass; the Q4_1 path of q4_core.h with the running
///        row in a register instead of y_acc.
/// @param b_col_reduce sum_i x_i per group as two interleaved bf16 limbs (hi, lo)
inline void _q41_block(aie::accum<accfloat, PARALLEL_ROWS> & acc_c, const bf16 * x, const q41_block_t * w, const bf16 * b_col_reduce) {
    static constexpr int NUM_GROUPS = Q4NX_COL_BLOCK_SIZE / Q4NX_GROUP_SIZE;
    const uint4 * it_quant  = w->qs;
    const bf16  * it_x      = x;
    const bf16  * it_scale  = w->scales;
    const bf16  * it_min    = w->mins;
    const bf16  * it_reduce = b_col_reduce;
    aie::accum<accfloat, PARALLEL_ROWS> c_accum = acc_c;

    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_RANGE( NUM_GROUPS, NUM_GROUPS )
    for (int g = 0; g < NUM_GROUPS; g++) AIE_LOOP_FLATTEN {
        aie::vector<bf16, Q4NX_GROUP_SIZE> b_col = aie::load_v<Q4NX_GROUP_SIZE>(it_x);
        it_x += Q4NX_GROUP_SIZE;
        aie::vector<bf16, PARALLEL_ROWS> a_scales = aie::load_v<PARALLEL_ROWS>(it_scale);
        it_scale += Q4NX_ROW_BLOCK_SIZE;
        _group_mac(c_accum, it_quant, b_col, a_scales);

        aie::vector<bf16, PARALLEL_ROWS> a_mins = aie::load_v<PARALLEL_ROWS>(it_min);
        it_min += Q4NX_ROW_BLOCK_SIZE;
#if QMM_LIMB_PRECISION
        c_accum = aie::mac(c_accum, a_mins, *(it_reduce + 1));
#endif
        c_accum = aie::mac(c_accum, a_mins, *it_reduce);
        it_reduce += 2;
    }
    acc_c = c_accum;
}

#endif // __MIX_CORE_H__
