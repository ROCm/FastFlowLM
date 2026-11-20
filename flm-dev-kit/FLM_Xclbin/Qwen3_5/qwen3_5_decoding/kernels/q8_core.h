// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __Q8_CORE_H__
#define __Q8_CORE_H__
#include "aie_kernel_utils.h"
#include "model_spec.h"
#include "aie_array_layout.h"
#include "bf16_limb_math.hpp"

/// @brief y += W x for one Q8_0 weight block, x bf16 / y float32.
///
/// "Reduce first, scale after": a whole 32-value group is reduced against x in
/// one float32 accumulator and only then meets the group's bf16 scale.  Every
/// multiplier operand is bf16-exact -- q is an int8 (bf16 holds every integer up
/// to 256) and the scale is *stored* bf16 -- so the group accumulator is the one
/// place a rounding could creep in, and the limb split below keeps it out.
void _Q80_x_bf16(
    float * c,
    bf16 * x,
    q80_block_t* w_buffer
) {
    // The whole row span of the block is resident at once, so there is no row
    // loop: one pass over the column groups, one scale entry per group.
    static constexpr int num_groups_per_row = QXNX_COL_BLOCK_SIZE / GGML_GROUP_SIZE;

    static_assert(PARALLEL_ROWS == QXNX_ROW_BLOCK_SIZE);

    int8_t* it_q = w_buffer->qs;
    bf16* it_scale = w_buffer->scales;
    bf16* it_x = x;

    aie::accum<accfloat, PARALLEL_ROWS> c_accum;
    c_accum.from_vector(aie::load_v<PARALLEL_ROWS>(c));

    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_RANGE( num_groups_per_row, num_groups_per_row )
    for (int group = 0; group < num_groups_per_row; group++) {
        aie::vector<bf16, PARALLEL_ROWS> a_scales = aie::load_v<PARALLEL_ROWS>(it_scale);

        // inlined Q8_core
        aie::vector<bf16, GGML_GROUP_SIZE> vec_col = aie::load_v<GGML_GROUP_SIZE>(it_x);
        aie::accum<accfloat, PARALLEL_ROWS> temp_acc;
        temp_acc.from_vector(aie::broadcast<bf16, PARALLEL_ROWS>(bf16(0.0f)));

        // Unrolled by 4 to amortize the per-iteration load and int8->bf16
        // conversion overhead, which dominates this loop rather than the
        // accumulator dependency.
        static constexpr int T_UNROLL = 4;
        AIE_PREPARE_FOR_PIPELINING
        AIE_LOOP_RANGE( GGML_GROUP_SIZE, GGML_GROUP_SIZE )
        AIE_LOOP_UNROLL( T_UNROLL )
        for(int t = 0; t < GGML_GROUP_SIZE; t++ ) {
            aie::vector<int8, PARALLEL_ROWS> a_cc_0 = aie::load_v<PARALLEL_ROWS>(it_q);
            it_q += PARALLEL_ROWS;

            aie::vector<float, PARALLEL_ROWS> a_col_float = aie::to_float(a_cc_0, 0);
            // convert to bfloat16
            aie::accum<accfloat, PARALLEL_ROWS> a_col_accum;
            a_col_accum.from_vector(a_col_float);

            aie::vector<bfloat16, PARALLEL_ROWS> a_col_bf16 = a_col_accum.template to_vector<bfloat16>();

            temp_acc = aie::mac(temp_acc, a_col_bf16, vec_col.get(t));
        }

        // a_scales is bf16-exact, so the only rounding this MAC could introduce
        // is on temp_acc.  A plain to_vector<bf16>() would throw away 16 of the
        // group sum's 24 mantissa bits, once per group and per block, and those
        // errors add up over the whole K loop.  Split the accumulator into two
        // bf16 limbs instead: ~17 mantissa bits rather than 8, for 1 extra MAC
        // per group against the 32 of the inner loop above.
        c_accum = bf16_limb::mac_f32_bf16_low(
            c_accum, temp_acc.template to_vector<float>(), a_scales);
        it_x += GGML_GROUP_SIZE;
        it_scale += PARALLEL_ROWS;
    }

    aie::vector<float, PARALLEL_ROWS> c_accum_f32 = c_accum.template to_vector<float>();
    aie::store_v(c, c_accum_f32);
}
#endif // __Q8_CORE_H__
