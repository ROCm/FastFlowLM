// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __Q4_CORE_H__
#define __Q4_CORE_H__
#include "aie_kernel_utils.h"
#include "model_spec.h"
#include "aie_array_layout.h"
#include "bf16_limb_math.hpp"

/// @brief y += W x for one Q4_K weight block, x and y bf16 / float32.
///
/// Format (see FLM_Q4NX_Converter/quant.md): a super-block is 256 weights of
/// one row, split into 8 groups of GGML_GROUP_SIZE. Per group it stores a uint8
/// scale s' and a uint8 min m'; per super-block a bf16 S' and a bf16 M'. The
/// dequantized weight is
///
///     w = S' s'_j q  +  M' m'_j
///
/// The datapath keeps "reduce first, scale after": the whole 32-value group is
/// reduced against x in one float32 accumulator and only then meets s'_j; S' is
/// applied once per super-block. Every multiplier operand is bf16-exact -- q is
/// a uint4, s'/m' are uint8 (bf16 holds every integer up to 256), S'/M' are
/// stored bf16 -- so a rounding can only ever land on an accumulator, and none
/// of the three that reach the output are rounded to bf16 on their way in: the
/// group sum and both super-block accumulators go through a two-limb split
/// instead. Emulating the whole datapath against an exact float64 reference
/// (FLM_Q4NX_Converter/npu_q4k_emulate.py) puts the result at ~1e-05 relative,
/// two orders under the ~4e-03 of the Q4_K format itself.
///
/// Layout, all column major over the QXNX_ROW_BLOCK_SIZE x QXNX_COL_BLOCK_SIZE
/// block, whose whole row span is resident at once (PARALLEL_ROWS):
///   qs      : one column = PARALLEL_ROWS nibbles, columns ascending
///   scales  : one group  = PARALLEL_ROWS uint8, groups ascending
///   mins    : same as scales
///   S / M   : PARALLEL_ROWS bf16; a block is exactly one super-block, so there
///             is one entry per row and no super-block loop
///
/// The scale path and the min path live in two separate float32 accumulators
/// that the caller carries across the whole K loop, so the min term never has to
/// be folded in inside the inner loop and nothing round-trips through memory
/// between blocks. Call _Q4K_finalize() once the K loop is done.
///
/// @param c_accum_s            running sum_b S'_b * (scale path), updated in place
/// @param c_accum_m            running sum_b M'_b * (min path), updated in place
/// @param x                    activations, bf16, QXNX_COL_BLOCK_SIZE of them
/// @param w_buffer             the weight block
/// @param b_col_reduce_buffer  sum_i x_i per group as two interleaved bf16
///                             limbs (hi, lo), shared across all rows
using q4k_accum_t = aie::accum<accfloat, PARALLEL_ROWS>;

void _Q4K_x_bf16(
    q4k_accum_t & c_accum_s,
    q4k_accum_t & c_accum_m,
    bf16 * x,
    q4k_block_t* w_buffer,
    bf16* b_col_reduce_buffer
) {
    static constexpr int SUPER_BLOCK_SIZE  = 256;
    static constexpr int GROUPS_PER_SB     = SUPER_BLOCK_SIZE / GGML_GROUP_SIZE;
    // one 512-bit uint4 load is 128 nibbles, i.e. this many whole columns
    static constexpr int NIBBLES_PER_LOAD  = 128;
    static constexpr int COLS_PER_LOAD     = NIBBLES_PER_LOAD / PARALLEL_ROWS;
    static constexpr int LOADS_PER_GROUP   = GGML_GROUP_SIZE / COLS_PER_LOAD;

    static_assert(PARALLEL_ROWS == QXNX_ROW_BLOCK_SIZE);
    // one weight block is exactly one super-block, so there is no super-block loop
    static_assert(QXNX_COL_BLOCK_SIZE == SUPER_BLOCK_SIZE);
    static_assert(NIBBLES_PER_LOAD % PARALLEL_ROWS == 0);
    static_assert(GGML_GROUP_SIZE % COLS_PER_LOAD == 0);

    uint4* it_quant = w_buffer->qs;
    uint8* it_scale = w_buffer->scales;
    uint8* it_min   = w_buffer->mins;

    bf16* it_reduce_b = b_col_reduce_buffer;
    bf16* it_x = x;

    // sum_j s'_j * (sum_i q_i x_i)  and  sum_j m'_j * (sum_i x_i), both carried
    // in float32 for the whole super-block and both fed through a two-limb
    // split, so neither reduction passes through a bf16 on its way in.
    aie::accum<accfloat, PARALLEL_ROWS> acc_s;
    aie::accum<accfloat, PARALLEL_ROWS> acc_m;
    acc_s.from_vector(aie::zeros<bf16, PARALLEL_ROWS>());
    acc_m.from_vector(aie::zeros<bf16, PARALLEL_ROWS>());

    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_RANGE( GROUPS_PER_SB, GROUPS_PER_SB )
    for (int g = 0; g < GROUPS_PER_SB; g++) {
        aie::vector<bf16, GGML_GROUP_SIZE> b_col = aie::load_v<GGML_GROUP_SIZE>(it_x);

        // One float32 accumulator for the whole 32-value group: a
        // PARALLEL_ROWS-wide MAC already covers the accumulator dependency,
        // so there is nothing to interleave against.
        aie::accum<accfloat, PARALLEL_ROWS> temp_acc;
        temp_acc.from_vector(aie::zeros<bf16, PARALLEL_ROWS>());

        AIE_LOOP_UNROLL_FULL
        for (int l = 0; l < LOADS_PER_GROUP; l++) {
            aie::vector<uint4, NIBBLES_PER_LOAD> q_raw =
                aie::load_v<NIBBLES_PER_LOAD>(it_quant);
            it_quant += NIBBLES_PER_LOAD / 2; // uint4* steps two nibbles

            // uint4 -> bf16 is exact
            aie::accum<accfloat, NIBBLES_PER_LOAD> q_acc;
            q_acc.from_vector(aie::to_float(q_raw, 0));
            aie::vector<bf16, NIBBLES_PER_LOAD> q_bf16 =
                q_acc.template to_vector<bf16>();

            AIE_LOOP_UNROLL_FULL
            for (int k = 0; k < COLS_PER_LOAD; k++) {
                aie::vector<bf16, PARALLEL_ROWS> a_col =
                    q_bf16.template extract<PARALLEL_ROWS>(k);
                bfloat16 scalar = b_col.get(l * COLS_PER_LOAD + k);
                temp_acc = aie::mac(temp_acc, a_col, scalar);
            }
        }

        // uint8 s' / m' -> bf16, exact for every value up to 255
        aie::accum<accfloat, PARALLEL_ROWS> s_acc;
        s_acc.from_vector(aie::to_float(aie::load_v<PARALLEL_ROWS>(it_scale), 0));

        // Rounding the group sum to bf16 here is by far the largest error term
        // the kernel can introduce: 2e-03 relative, taken once per group,
        // correlated across the 8 groups that share an S', and then amplified
        // ~2.3x by the scale/min cancellation in _Q4K_finalize. It does not
        // average away over the K loop the way an independent error would --
        // |y| ends up far smaller than the group sums that build it, so the
        // error is measured against the small number. Split into two bf16 limbs
        // instead: 1 extra MAC and a split per group, against the 32 MACs of the
        // loop above.
        acc_s = bf16_limb::mac_f32_bf16_low(
            acc_s, temp_acc.template to_vector<float>(),
            s_acc.template to_vector<bf16>());

        aie::accum<accfloat, PARALLEL_ROWS> m_acc;
        m_acc.from_vector(aie::to_float(aie::load_v<PARALLEL_ROWS>(it_min), 0));
        // sum_i x_i arrives pre-split as two bf16 limbs; smallest first, so the
        // small term joins acc_m before it meets the large one.
        aie::vector<bf16, PARALLEL_ROWS> a_mins = m_acc.template to_vector<bf16>();
        acc_m = aie::mac(acc_m, a_mins, *(it_reduce_b + 1));
        acc_m = aie::mac(acc_m, a_mins, *it_reduce_b);

        it_reduce_b += 2;
        it_x     += GGML_GROUP_SIZE;
        it_scale += PARALLEL_ROWS;
        it_min   += PARALLEL_ROWS;
    }

    // S' * acc_s and M' * acc_m go to their own running accumulators; the two
    // paths never meet inside the K loop, so neither depends on the other and
    // the combining add stays off the critical path.
    //
    // Both multiplicands are bf16-exact (S'/M' are *stored* bf16), so the only
    // rounding here would be on the accumulators -- and it would land before
    // the scale/min cancellation in _Q4K_finalize, where a relative error is
    // amplified ~2.3x (quant.md, "cancellation ratio").  Same reasoning as the
    // group sum above, one level up and twice.  Split each accumulator into two
    // bf16 limbs: ~17 mantissa bits rather than 8, for 2 extra MACs and 2 msc
    // per 32x256 block -- amortized over the ~800 vector ops of the group loop
    // above, and in its drain phase.
    aie::vector<bf16, PARALLEL_ROWS> a_S = aie::load_v<PARALLEL_ROWS>(w_buffer->S);
    aie::vector<bf16, PARALLEL_ROWS> a_M = aie::load_v<PARALLEL_ROWS>(w_buffer->M);
    c_accum_s = bf16_limb::mac_f32_bf16_low(
        c_accum_s, acc_s.template to_vector<float>(), a_S);
    c_accum_m = bf16_limb::mac_f32_bf16_low(
        c_accum_m, acc_m.template to_vector<float>(), a_M);
}

/// @brief Zero the accumulator pair, once per output row block.
void _Q4K_zero(
    q4k_accum_t & c_accum_s,
    q4k_accum_t & c_accum_m
) {
    c_accum_s.from_vector(aie::zeros<bf16, PARALLEL_ROWS>());
    c_accum_m.from_vector(aie::zeros<bf16, PARALLEL_ROWS>());
}

/// @brief Collapse the accumulator pair of _Q4K_x_bf16 into the final result.
///
/// Call once, after the whole K loop: the scale path and the min path have each
/// been summed over every block of the row, and the single combining add
/// c = scale + min happens here, in the accumulator domain.
///
/// @param c  float32 result, PARALLEL_ROWS wide
void _Q4K_finalize(
    float * c,
    const q4k_accum_t & c_accum_s,
    const q4k_accum_t & c_accum_m
) {
    q4k_accum_t c_accum = aie::add(c_accum_s, c_accum_m);
    aie::store_v(c, c_accum.template to_vector<float>());
}

#endif // __Q4_CORE_H__