// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __Q4_CORE_H__
#define __Q4_CORE_H__
#include "aie_kernel_utils.h"
#include "model_spec.h"
#include "bf16_limb_math.hpp"

/// @brief y += W x for one Q4_0 weight block, x bf16 and y float32.
///
/// Format: a group is GGML_GROUP_SIZE weights of one row sharing a bf16 scale d.
/// GGUF's Q4_0 is w = d * (q - 8) with q a uint4; the converter subtracts the 8
/// at pack time (unpack_q4_0 in FLM_Q4NX_Converter/q4nx/gguf_tensor.py), so the
/// block stores a signed int4 and there is no zero-point / min path to carry --
/// unlike Q4_K, which needs a per-group sum_i x_i for its min term.
///
/// The datapath keeps "reduce first, scale after": the whole GGML_GROUP_SIZE
/// group is reduced against x in float32 accumulators and only then meets d.
/// Every multiplier operand is bf16-exact -- the int4 q converts to bf16 without
/// loss and d is *stored* bf16 -- so the only place a rounding can land is on an
/// accumulator, and the one accumulator that reaches the output does not get
/// rounded to bf16 on its way in: it goes through a two-limb split instead.
///
/// That split is the whole point of this version. Rounding the group sum to bf16
/// before it meets d costs ~4e-03 relative, taken once per group and correlated
/// across the groups of a row, so it does not average away over the K loop the
/// way an independent error would. Two bf16 limbs carry ~17 mantissa bits
/// instead of 8, for one extra MAC and one split per group against the 32 MACs
/// of the reduction above it (see bf16_limb_math.hpp for the measured tiers).
///
/// Layout, all column major over the QXNX_ROW_BLOCK_SIZE x QXNX_COL_BLOCK_SIZE
/// block, PARALLEL_ROWS rows resident at a time:
///   scales : one group = QXNX_ROW_BLOCK_SIZE bf16, groups ascending
///   qs     : one column = PARALLEL_ROWS nibbles, columns ascending, the whole
///            row block's groups before the next row block's
///
/// @param c         float32 result, QXNX_ROW_BLOCK_SIZE wide, accumulated in place
/// @param x         activations, bf16, QXNX_COL_BLOCK_SIZE of them
/// @param w_buffer  the weight block
void _Q40_x_bf16(
    float * c,
    bf16 * x,
    q40_block_t* w_buffer
) {
    // one iteration per GGML quantization group: each pass consumes GGML_GROUP_SIZE
    // columns of x and one scale entry per row.
    static constexpr int NUM_GROUPS_PER_ROW = QXNX_COL_BLOCK_SIZE / GGML_GROUP_SIZE;
    // one 512-bit int4 load is 128 nibbles, i.e. this many whole columns
    static constexpr int NIBBLES_PER_LOAD   = 128;
    static constexpr int COLS_PER_LOAD      = NIBBLES_PER_LOAD / PARALLEL_ROWS;
    static constexpr int LOADS_PER_GROUP    = GGML_GROUP_SIZE / COLS_PER_LOAD;

    static_assert(QXNX_ROW_BLOCK_SIZE % PARALLEL_ROWS == 0);
    static_assert(NIBBLES_PER_LOAD % PARALLEL_ROWS == 0);
    static_assert(GGML_GROUP_SIZE % COLS_PER_LOAD == 0);

    int4* it_quant = w_buffer->qs;
    // walk the whole block, PARALLEL_ROWS rows at a time
    for (int row = 0; row < QXNX_ROW_BLOCK_SIZE; row += PARALLEL_ROWS){
        aie::accum<accfloat, PARALLEL_ROWS> c_accum;
        c_accum.from_vector(aie::load_v<PARALLEL_ROWS>(c + row));

        bf16* it_scale = w_buffer->scales + row;
        bf16* it_x = x;

        // The former col(4) x sub_chunk(2) nest carried no index into the body,
        // so it is one flat loop over the column groups.
        AIE_PREPARE_FOR_PIPELINING
        AIE_LOOP_RANGE(NUM_GROUPS_PER_ROW, NUM_GROUPS_PER_ROW )
        // Two groups per body: the int4 unpack of one group (VCONV/VUPS) then
        // overlaps the MACs of the other, which a single group cannot cover on
        // its own. Measured on the darts listing: 155 -> 143 bundles per group.
        // Unrolling by 4 loses it again (169) and doubles the object.
        AIE_LOOP_UNROLL(2)
        for (int chunk = 0; chunk < NUM_GROUPS_PER_ROW; chunk++) {
            aie::vector<bf16, GGML_GROUP_SIZE> b_col = aie::load_v<GGML_GROUP_SIZE>(it_x);

            // Two independent float32 accumulators for the group, so back-to-back
            // MACs do not serialize on a single accumulator dependency -- a
            // PARALLEL_ROWS-wide MAC is too narrow to cover the latency on its own.
            aie::accum<accfloat, PARALLEL_ROWS> temp_acc_0;
            aie::accum<accfloat, PARALLEL_ROWS> temp_acc_1;
            temp_acc_0.from_vector(aie::zeros<bf16, PARALLEL_ROWS>());
            temp_acc_1.from_vector(aie::zeros<bf16, PARALLEL_ROWS>());

            AIE_LOOP_UNROLL_FULL
            for (int l = 0; l < LOADS_PER_GROUP; l++) {
                aie::vector<int4, NIBBLES_PER_LOAD> q_raw =
                    aie::load_v<NIBBLES_PER_LOAD>(it_quant);
                it_quant += NIBBLES_PER_LOAD / 2; // int4* steps two nibbles

                // int4 -> bf16 is exact
                aie::accum<accfloat, NIBBLES_PER_LOAD> q_acc;
                q_acc.from_vector(aie::to_float(q_raw, 0));
                aie::vector<bf16, NIBBLES_PER_LOAD> q_bf16 =
                    q_acc.template to_vector<bf16>();

                AIE_LOOP_UNROLL_FULL
                for (int k = 0; k < COLS_PER_LOAD; k += 2) {
                    aie::vector<bf16, PARALLEL_ROWS> a_col_0 =
                        q_bf16.template extract<PARALLEL_ROWS>(k);
                    aie::vector<bf16, PARALLEL_ROWS> a_col_1 =
                        q_bf16.template extract<PARALLEL_ROWS>(k + 1);
                    bfloat16 scalar_0 = b_col.get(l * COLS_PER_LOAD + k);
                    bfloat16 scalar_1 = b_col.get(l * COLS_PER_LOAD + k + 1);
                    temp_acc_0 = aie::mac(temp_acc_0, a_col_0, scalar_0);
                    temp_acc_1 = aie::mac(temp_acc_1, a_col_1, scalar_1);
                }
            }

            aie::vector<bf16, PARALLEL_ROWS> a_scales = aie::load_v<PARALLEL_ROWS>(it_scale);

            // Combine the two partials and fold the group sum into c_accum
            // through a two-limb split. Rounding it to bf16 here instead -- what
            // this kernel used to do, once per partial -- is the largest error
            // term the datapath can introduce, and the only one that is
            // correlated across the groups sharing a row rather than averaging
            // out over the K loop. d is bf16-exact, so only the sum needs
            // splitting: 2 MACs, ~17 mantissa bits, against 8 for the bf16 fold.
            // Both the combining add and the split stay in the accumulator
            // domain, so the group sum never round-trips through an fp32 vector.
            aie::accum<accfloat, PARALLEL_ROWS> group_sum =
                aie::add(temp_acc_0, temp_acc_1);
            c_accum = bf16_limb::mac_acc_bf16_low(c_accum, group_sum, a_scales);

            it_x += GGML_GROUP_SIZE;
            it_scale += QXNX_ROW_BLOCK_SIZE;
        }
        aie::vector<float, PARALLEL_ROWS> c_accum_f32 = c_accum.template to_vector<float>();
        aie::store_v(c + row, c_accum_f32);

    }
}
#endif // __Q4_CORE_H__
