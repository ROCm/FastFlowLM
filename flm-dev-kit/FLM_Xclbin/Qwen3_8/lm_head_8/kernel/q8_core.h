// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#ifndef __Q8_CORE_H__
#define __Q8_CORE_H__
#include "aie_kernel_utils.h"
#include "bf16_limb_math.hpp"

#define QXNX_ROW_BLOCK_SIZE 32
#define QXNX_COL_BLOCK_SIZE 256
#define GGML_GROUP_SIZE 32

// The core holds a whole block's rows at once: one column of a block is exactly
// PARALLEL_ROWS elements, so this is a layout constant, not a tuning knob.
#define PARALLEL_ROWS QXNX_ROW_BLOCK_SIZE

///@brief Q80 block
///@param scales: scales and mins, bf16
///@param mins: scales and mins, bf16
///@param qs: 4-bit quantized values
typedef struct {
    bf16 scales[QXNX_ROW_BLOCK_SIZE * QXNX_COL_BLOCK_SIZE / GGML_GROUP_SIZE]; // per-group scales and mins
    int8_t qs[QXNX_ROW_BLOCK_SIZE * QXNX_COL_BLOCK_SIZE];                      // 8--bit quants
} q80_block_t;

/// @brief y += W x for one Q8_0 weight block over L token rows.
///
/// "Reduce first, scale after": a whole 32-value group is reduced against x in
/// one float32 accumulator and only then meets the group's bf16 scale.  Every
/// multiplier operand is bf16-exact -- q is an int8 (bf16 holds every integer up
/// to 256) and the scale is *stored* bf16 -- so the group accumulator is the one
/// place a rounding could creep in, and the limb split below keeps it out.
///
/// The L token rows sit *inside* the column loop, not around it.  Converting a
/// weight column from int8 to bf16 is the expensive part of this kernel -- it is
/// conversion-bound, not MAC-bound -- and the converted column does not depend
/// on the token row.  Converting once and MACing against all L rows is the whole
/// reason an L=8 pass costs far less than 8 separate passes.
///
/// x is [lane][QXNX_COL_BLOCK_SIZE] within a column block, L + 1 lanes: the L
/// token rows followed by this column block's slice of `model.norm.weight`.
/// c is [token][32], the L accumulators for this one 32-row weight block.
///
/// ## The norm weight is folded here, and the reciprocal-RMS is not
///
/// RMSNorm is `x_hat = x * s * nw` with `s = 1/sqrt(mean(x^2) + eps)`. `s` is a
/// per-token scalar, so it commutes straight through the matmul and is applied
/// once to the finished accumulator in copy_y_scaled -- which is what lets this
/// kernel normalise at all, since `s` is not known until the k loop that first
/// consumes x has already finished.  `nw` is per *column* and does not commute,
/// so it has to meet x elementwise.
///
/// The cheap place for that is right here, where x is loaded anyway: one extra
/// multiply and one extra bf16 round per (group, token), against the 32
/// iterations of the conversion loop below.  Folding it into the Q8_0 table
/// instead is not possible -- a scale covers 32 consecutive columns of one
/// output row and `nw` varies within that span.
template <int L>
void _Q80_x_bf16_L(
    float * c,
    bf16 * x,
    q80_block_t* w_buffer
) {
    // The whole row span of the block is resident at once, so there is no weight
    // row loop: one pass over the column groups, one scale entry per group.
    static constexpr int num_groups_per_row = QXNX_COL_BLOCK_SIZE / GGML_GROUP_SIZE;

    int8_t* it_q = w_buffer->qs;
    bf16* it_scale = w_buffer->scales;
    bf16* it_x = x;

    aie::accum<accfloat, PARALLEL_ROWS> c_accum[L];
    AIE_LOOP_UNROLL_FULL
    for (int t = 0; t < L; t++) {
        c_accum[t].from_vector(aie::load_v<PARALLEL_ROWS>(c + t * PARALLEL_ROWS));
    }

    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_RANGE( num_groups_per_row, num_groups_per_row )
    for (int group = 0; group < num_groups_per_row; group++) {
        aie::vector<bf16, PARALLEL_ROWS> a_scales = aie::load_v<PARALLEL_ROWS>(it_scale);

        // inlined Q8_core
        //
        // Lane L of the chunk is model.norm.weight, so it strides with it_x for
        // free. The product is rounded back to bf16 because the MAC below wants
        // a bf16 scalar operand; that is one rounding, the same count the host
        // path spent when it rounded its normalised row on the way into the BO.
        aie::vector<bf16, GGML_GROUP_SIZE> nw_col =
            aie::load_v<GGML_GROUP_SIZE>(it_x + L * QXNX_COL_BLOCK_SIZE);

        aie::vector<bf16, GGML_GROUP_SIZE> vec_col[L];
        aie::accum<accfloat, PARALLEL_ROWS> temp_acc[L];
        AIE_LOOP_UNROLL_FULL
        for (int t = 0; t < L; t++) {
            aie::accum<accfloat, GGML_GROUP_SIZE> xw = aie::mul(
                aie::load_v<GGML_GROUP_SIZE>(it_x + t * QXNX_COL_BLOCK_SIZE), nw_col);
            vec_col[t] = xw.template to_vector<bf16>();
            temp_acc[t].from_vector(aie::broadcast<bf16, PARALLEL_ROWS>(bf16(0.0f)));
        }

#if defined(__chess__)
        // No T_UNROLL: the body already issues L independent MACs per loaded
        // weight column, which covers the load and conversion latency that the
        // unroll-by-4 existed to hide at L=1.
        AIE_PREPARE_FOR_PIPELINING
        AIE_LOOP_RANGE( GGML_GROUP_SIZE, GGML_GROUP_SIZE )
        for(int k = 0; k < GGML_GROUP_SIZE; k++ ) {
            aie::vector<int8, PARALLEL_ROWS> a_cc_0 = aie::load_v<PARALLEL_ROWS>(it_q);
            it_q += PARALLEL_ROWS;

            aie::vector<float, PARALLEL_ROWS> a_col_float = aie::to_float(a_cc_0, 0);
            // convert to bfloat16
            aie::accum<accfloat, PARALLEL_ROWS> a_col_accum;
            a_col_accum.from_vector(a_col_float);

            aie::vector<bfloat16, PARALLEL_ROWS> a_col_bf16 = a_col_accum.template to_vector<bfloat16>();

            // one conversion above, L MACs here -- this is the L>1 win
            AIE_LOOP_UNROLL_FULL
            for (int t = 0; t < L; t++) {
                temp_acc[t] = aie::mac(temp_acc[t], a_col_bf16, vec_col[t].get(k));
            }
        }

#else
        // Peano cannot keep all L accumulators in registers across this loop and
        // spills them on every column. Two passes over the group's weight columns,
        // each with half the tokens, keep them resident.
        static_assert(L % 2 == 0);
        AIE_LOOP_UNROLL_FULL
        for (int h = 0; h < 2; h++) {
            int8_t* it_qh = it_q;
            for(int k = 0; k < GGML_GROUP_SIZE; k++ ) {
                aie::vector<int8, PARALLEL_ROWS> a_cc_0 = aie::load_v<PARALLEL_ROWS>(it_qh);
                it_qh += PARALLEL_ROWS;
                aie::accum<accfloat, PARALLEL_ROWS> a_col_accum;
                a_col_accum.from_vector(aie::to_float(a_cc_0, 0));
                aie::vector<bfloat16, PARALLEL_ROWS> a_col_bf16 = a_col_accum.template to_vector<bfloat16>();
                AIE_LOOP_UNROLL_FULL
                for (int t = h * (L / 2); t < (h + 1) * (L / 2); t++) {
                    temp_acc[t] = aie::mac(temp_acc[t], a_col_bf16, vec_col[t].get(k));
                }
            }
        }
        it_q += GGML_GROUP_SIZE * PARALLEL_ROWS;
#endif
        // a_scales is bf16-exact, so the only rounding this MAC could introduce
        // is on temp_acc.  A plain to_vector<bf16>() would throw away 16 of the
        // group sum's 24 mantissa bits, once per group and per block, and those
        // errors add up over the whole K loop.  Split the accumulator into two
        // bf16 limbs instead: ~17 mantissa bits rather than 8, for 1 extra MAC
        // per group against the 32 of the inner loop above.
        AIE_LOOP_UNROLL_FULL
        for (int t = 0; t < L; t++) {
            c_accum[t] = bf16_limb::mac_f32_bf16_low(
                c_accum[t], temp_acc[t].template to_vector<float>(), a_scales);
        }
        it_x += GGML_GROUP_SIZE;
        it_scale += PARALLEL_ROWS;
    }

    AIE_LOOP_UNROLL_FULL
    for (int t = 0; t < L; t++) {
        aie::store_v(c + t * PARALLEL_ROWS, c_accum[t].template to_vector<float>());
    }
}
#endif // __Q8_CORE_H__