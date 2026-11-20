// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __Q4_CORE_H__
#define __Q4_CORE_H__
#include "aie_kernel_utils.h"
#include "model_spec.h"
#include "aie_array_layout.h"
#include "bf16_limb_math.hpp"

/// @brief uint4 -> bf16 holding 128 + q, by placing bytes rather than converting.
///
/// aie::to_float on a uint4 vector lowers to a chain -- unpack the nibbles
/// wide, add the int-to-float magic constant, subtract it back in float, round
/// down to bf16 -- roughly four vector ops to prepare one operand for one MAC.
/// Worse, the chain runs through the accumulator register files, so every
/// converted vector then has to be moved back out to the vector registers the
/// MAC reads; it showed up as 146 VMOVs in a body that issued 32 useful MACs.
///
/// So do not convert. Set a bf16 exponent field to 134: the binade is
/// [128, 256) and its ulp is exactly 1, so over that one window bf16 is a plain
/// integer counter and the integer it counts is sitting in the mantissa field
/// in binary. For every v in [0, 15],
///
///     0x4300 | v  ==  128 + v      exactly, as a bf16 bit pattern
///
/// which makes building the operand byte placement: unpack the nibbles to
/// bytes, zip them against a broadcast 0x43, reinterpret. Two vector ops per
/// load, nothing entering an accumulator file, nothing rounding -- the result
/// is bit-identical to what a correct conversion would have produced.
///
/// This is a property of the window, not of the format. A 4-bit payload fits
/// inside a 7-bit mantissa with three bits to spare, and Q4_K is named after
/// exactly that; the uint8 scales and mins in the same block are still
/// converted the honest way, because no binade in bf16 lands all 256 byte
/// values on a shifted grid.
///
/// The operand is 128 + q, so the MAC loop below leaves a bias behind. Nothing
/// in the group loop pays it: see NIBBLE_BIAS and the bias accumulator.
static constexpr uint8 Q4_BIASED_EXP_BYTE = 0x43;

/// @brief The +128 that _q4_biased_bf16 leaves in the answer, settled once.
///
/// The MAC operand is 128 + q, so each group contributes
///
///     sum_i (128 + q_i) x_i  =  sum_i q_i x_i  +  128 sum_i x_i
///
/// and over a whole row the excess is
///
///     128 * sum_b S'_b * sum_j s'_j * sum_i x_i
///
/// -- a scale-path term of exactly the min path's shape, a per-group constant
/// times sum_i x_i. So it does not have to be cancelled where it is created.
/// It rides its own accumulator pair, acc_b inside a super-block and c_accum_b
/// across the whole K loop, in the same two MACs per group the min path costs,
/// and the 128 itself is applied once per 32-row block in _Q4K_finalize --
/// never inside the loop, where it would be 32x256 times more expensive than
/// the thing it corrects.
///
/// The one thing deferral does cost: everything downstream of the MAC loop is
/// carrying a number ~15x the answer until finalize cancels it, so the two
/// splits it passes through -- the group sum into acc_s, and acc_s into
/// c_accum_s -- must measure their 2^-b against the large number. Both go to
/// the 3-limb mac_f32_bf16_safe form; at 2 limbs the residual alone is 1.2e-04
/// on the datapath.  (FLM_Q4NX_Converter/npu_q4k_emulate.py, "NIBBLE BIAS: row
/// acc" rows.)
static constexpr int NIBBLE_BIAS = 128;

/// @brief bf16 limbs per group in b_col_reduce_buffer, smallest last.
///
/// Two. The min path needs no more -- m'_j sum_i x_i is the same size as the
/// answer -- and the bias path reads the same two, which is what makes the
/// deferred form worth having: the third limb is O(K) preprocessing paid on
/// every mvm(), and dropping it takes the activation-reduction loop back to
/// its pre-nibble-bias size.
///
/// It is the accuracy floor of the scheme, though, not a free choice. The bias
/// term is 128 sum_i x_i, ~15x the group sum it corrects, so b bits of
/// sum_i x_i leave ~15 * 2^-b behind: 4.8e-05 on the datapath at two limbs,
/// 7.4e-06 at three, against 1.0e-05 for the kernel with no nibble bias at all.
/// All three sit two orders under the ~3.3e-03 the Q4_K refit itself carries --
/// end to end vs llama.cpp the choice moves 3.472e-03 to 3.474e-03 -- so two is
/// the right default and three is one constant away if a caller disagrees.
static constexpr int REDUCE_LIMBS = 2;

template <unsigned N>
static inline __attribute__((always_inline)) aie::vector<bf16, N> _q4_biased_bf16(
    aie::vector<uint4, N> q_raw,
    aie::vector<uint8, N> bias_bytes
) {
    // little endian: low byte is the nibble, high byte the 0x43 exponent field
    return aie::vector_cast<bf16>(
        aie::concat(aie::interleave_zip(aie::unpack(q_raw), bias_bytes, 1)));
}

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
/// @param c_accum_b            running sum_b S'_b * (bias path), updated in
///                             place; _Q4K_finalize scales it by -NIBBLE_BIAS
/// @param x                    activations, bf16, QXNX_COL_BLOCK_SIZE of them
/// @param w_buffer             the weight block
/// @param b_col_reduce_buffer  sum_i x_i per group as REDUCE_LIMBS interleaved
///                             bf16 limbs, largest first, shared across all rows
using q4k_accum_t = aie::accum<accfloat, PARALLEL_ROWS>;

void _Q4K_x_bf16(
    q4k_accum_t & c_accum_s,
    q4k_accum_t & c_accum_m,
    q4k_accum_t & c_accum_b,
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

    // the exponent half of every bf16 lane, built once and live in a register
    // for the whole loop, so the per-load cost is one unpack and one zip
    const aie::vector<uint8, NIBBLES_PER_LOAD> bias_bytes =
        aie::broadcast<uint8, NIBBLES_PER_LOAD>(Q4_BIASED_EXP_BYTE);

    // sum_j s'_j * (sum_i q_i x_i)  and  sum_j m'_j * (sum_i x_i), both carried
    // in float32 for the whole super-block and both fed through a two-limb
    // split, so neither reduction passes through a bf16 on its way in.
    aie::accum<accfloat, PARALLEL_ROWS> acc_s;
    aie::accum<accfloat, PARALLEL_ROWS> acc_m;
    aie::accum<accfloat, PARALLEL_ROWS> acc_b;
    acc_s.from_vector(aie::zeros<bf16, PARALLEL_ROWS>());
    acc_m.from_vector(aie::zeros<bf16, PARALLEL_ROWS>());
    acc_b.from_vector(aie::zeros<bf16, PARALLEL_ROWS>());

    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_RANGE( GROUPS_PER_SB, GROUPS_PER_SB )
    for (int g = 0; g < GROUPS_PER_SB; g++) {
        aie::vector<bf16, GGML_GROUP_SIZE> b_col = aie::load_v<GGML_GROUP_SIZE>(it_x);

        // One float32 accumulator for the whole 32-value group: a
        // PARALLEL_ROWS-wide MAC already covers the accumulator dependency, so
        // there is nothing to interleave against. It opens at zero and closes
        // holding sum_i q_i x_i + 128 sum_i x_i; nothing here corrects that,
        // see NIBBLE_BIAS.
        aie::accum<accfloat, PARALLEL_ROWS> temp_acc;
        temp_acc.from_vector(aie::zeros<bf16, PARALLEL_ROWS>());

        AIE_LOOP_UNROLL_FULL
        for (int l = 0; l < LOADS_PER_GROUP; l++) {
            aie::vector<uint4, NIBBLES_PER_LOAD> q_raw =
                aie::load_v<NIBBLES_PER_LOAD>(it_quant);
            it_quant += NIBBLES_PER_LOAD / 2; // uint4* steps two nibbles

            // uint4 -> bf16 by placing bytes: exact, and carrying a +128 that
            // the group reduction pays back below
            aie::vector<bf16, NIBBLES_PER_LOAD> q_bf16 =
                _q4_biased_bf16(q_raw, bias_bytes);

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
        aie::vector<bf16, PARALLEL_ROWS> a_scales = s_acc.template to_vector<bf16>();

        // Rounding the group sum to bf16 here is by far the largest error term
        // the kernel can introduce: 2e-03 relative, taken once per group,
        // correlated across the 8 groups that share an S', and then amplified
        // ~2.3x by the scale/min cancellation in _Q4K_finalize. It does not
        // average away over the K loop the way an independent error would --
        // |y| ends up far smaller than the group sums that build it, so the
        // error is measured against the small number. Split into bf16 limbs
        // instead, against the 32 MACs of the loop above.
        //
        // Three limbs, not two: temp_acc is still carrying the +128 that
        // _Q4K_finalize will cancel, so it is ~15x the group sum and a 2-limb
        // split would measure its 2^-17 against the large number. See
        // NIBBLE_BIAS.
        acc_s = bf16_limb::mac_f32_bf16_safe(
            acc_s, temp_acc.template to_vector<float>(), a_scales);

        // The bias path: sum_j s'_j sum_i x_i, the same two MACs against the
        // same pre-split sum_i x_i that the min path costs, and with no 128 in
        // sight -- that arrives once, in _Q4K_finalize.
        acc_b = aie::mac(acc_b, a_scales, *(it_reduce_b + 1));
        acc_b = aie::mac(acc_b, a_scales, *it_reduce_b);

        aie::accum<accfloat, PARALLEL_ROWS> m_acc;
        m_acc.from_vector(aie::to_float(aie::load_v<PARALLEL_ROWS>(it_min), 0));
        // sum_i x_i arrives pre-split; the min path needs only the top two
        // limbs, smallest first so the small term joins acc_m before the large.
        aie::vector<bf16, PARALLEL_ROWS> a_mins = m_acc.template to_vector<bf16>();
        acc_m = aie::mac(acc_m, a_mins, *(it_reduce_b + 1));
        acc_m = aie::mac(acc_m, a_mins, *it_reduce_b);

        it_reduce_b += REDUCE_LIMBS;
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
    //
    // acc_s and acc_b both hold the inflated, still-biased quantity, so both
    // take the 3-limb form for the same reason the group split does; acc_m does
    // not and stays on 2. Three MACs each, once per 256 columns.
    c_accum_s = bf16_limb::mac_f32_bf16_safe(
        c_accum_s, acc_s.template to_vector<float>(), a_S);
    c_accum_b = bf16_limb::mac_f32_bf16_safe(
        c_accum_b, acc_b.template to_vector<float>(), a_S);
    c_accum_m = bf16_limb::mac_f32_bf16_low(
        c_accum_m, acc_m.template to_vector<float>(), a_M);
}

/// @brief Zero the accumulator pair, once per output row block.
void _Q4K_zero(
    q4k_accum_t & c_accum_s,
    q4k_accum_t & c_accum_m,
    q4k_accum_t & c_accum_b
) {
    c_accum_s.from_vector(aie::zeros<bf16, PARALLEL_ROWS>());
    c_accum_m.from_vector(aie::zeros<bf16, PARALLEL_ROWS>());
    c_accum_b.from_vector(aie::zeros<bf16, PARALLEL_ROWS>());
}

/// @brief Collapse the accumulator pair of _Q4K_x_bf16 into the final result.
///
/// Call once, after the whole K loop: the three paths have each been summed
/// over every block of the row, and the combining
///
///     c = scale + min - 128 * bias
///
/// happens here, in the accumulator domain. This is the only place NIBBLE_BIAS
/// appears in the arithmetic: 32 rows x K columns of biased MACs are settled by
/// one msc. Both operands of that msc are bf16-exact -- 128 is a power of two
/// and c_accum_b is split to 3 limbs -- so the cancellation itself is clean and
/// what is left is the 2-limb truncation of sum_i x_i, see REDUCE_LIMBS.
///
/// @param c  float32 result, PARALLEL_ROWS wide
void _Q4K_finalize(
    float * c,
    const q4k_accum_t & c_accum_s,
    const q4k_accum_t & c_accum_m,
    const q4k_accum_t & c_accum_b
) {
    q4k_accum_t c_accum = aie::add(c_accum_s, c_accum_m);
    c_accum = bf16_limb::mac_f32_bf16_safe(
        c_accum, c_accum_b.template to_vector<float>(),
        aie::broadcast<bf16, PARALLEL_ROWS>(bfloat16(-NIBBLE_BIAS)));
    aie::store_v(c, c_accum.template to_vector<float>());
}

/// @brief y += W x for one Q4_1 weight block, x bf16 / y float32.
///
/// Format: a group is GGML_GROUP_SIZE weights of one row with a bf16 scale s
/// and a bf16 min m, dequantizing as
///
///     w = s q + m
///
/// Same datapath shape as _Q4K_x_bf16 -- "reduce first, scale after", the whole
/// group reduced against x in one float32 accumulator before it meets s -- but
/// one level shallower: Q4_1 has no super-block, so s and m are the final
/// scales and the scale path and the min path can both land in the running
/// float32 result directly. There is nothing to carry across the K loop beyond
/// that result, hence no accumulator struct and no finalize; the caller zeroes
/// the row with zero_vectorized<float, m>() and reads it back when the K loop
/// is done.
///
/// Every multiplier operand is bf16-exact -- q is a uint4, s and m are *stored*
/// bf16 -- so a rounding can only ever land on an accumulator. The one that
/// matters is the group sum on its way into the scale MAC, and it is bought
/// back with a two-limb split for the same reason as in _Q4K_x_bf16: |y| ends
/// up far smaller than the group sums that build it, so the error is measured
/// against the small number and does not average away over the K loop.
///
/// Layout, all column major over the QXNX_ROW_BLOCK_SIZE x QXNX_COL_BLOCK_SIZE
/// block, whose whole row span is resident at once (PARALLEL_ROWS):
///   qs      : one column = PARALLEL_ROWS nibbles, columns ascending
///   scales  : one group  = PARALLEL_ROWS bf16, groups ascending
///   mins    : same as scales
///
/// @param c                    float32 running result, PARALLEL_ROWS of them,
///                             updated in place
/// @param c_bias               float32 running sum_g s_g sum_i x_i, PARALLEL_ROWS
///                             of them, updated in place; the caller settles it
///                             with _Q41_finalize once the K loop is done
/// @param x                    activations, bf16, QXNX_COL_BLOCK_SIZE of them
/// @param w_buffer             the weight block
/// @param b_col_reduce_buffer  sum_i x_i per group as REDUCE_LIMBS interleaved
///                             bf16 limbs, largest first, shared across all rows
void _Q41_x_bf16(
    float * c,
    float * c_bias,
    bf16 * x,
    q41_block_t* w_buffer,
    bf16* b_col_reduce_buffer
) {
    // one 512-bit uint4 load is 128 nibbles, i.e. this many whole columns
    static constexpr int NIBBLES_PER_LOAD = 128;
    static constexpr int COLS_PER_LOAD    = NIBBLES_PER_LOAD / PARALLEL_ROWS;
    static constexpr int LOADS_PER_GROUP  = GGML_GROUP_SIZE / COLS_PER_LOAD;
    static constexpr int NUM_GROUPS       = QXNX_COL_BLOCK_SIZE / GGML_GROUP_SIZE;

    static_assert(PARALLEL_ROWS == QXNX_ROW_BLOCK_SIZE);
    static_assert(NIBBLES_PER_LOAD % PARALLEL_ROWS == 0);
    static_assert(GGML_GROUP_SIZE % COLS_PER_LOAD == 0);

    uint4* it_quant = w_buffer->qs;
    bf16* it_scale  = w_buffer->scales;
    bf16* it_min    = w_buffer->mins;

    bf16* it_reduce_b = b_col_reduce_buffer;
    bf16* it_x = x;

    // see _q4_biased_bf16(): the exponent half of every lane, and the
    // multiplier that pays the bias back, both hoisted
    const aie::vector<uint8, NIBBLES_PER_LOAD> bias_bytes =
        aie::broadcast<uint8, NIBBLES_PER_LOAD>(Q4_BIASED_EXP_BYTE);

    aie::accum<accfloat, PARALLEL_ROWS> c_accum;
    aie::accum<accfloat, PARALLEL_ROWS> c_bias_accum;
    c_accum.from_vector(aie::load_v<PARALLEL_ROWS>(c));
    c_bias_accum.from_vector(aie::load_v<PARALLEL_ROWS>(c_bias));

    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_RANGE( NUM_GROUPS, NUM_GROUPS )
    for (int g = 0; g < NUM_GROUPS; g++) {
        aie::vector<bf16, GGML_GROUP_SIZE> b_col = aie::load_v<GGML_GROUP_SIZE>(it_x);

        // One float32 accumulator for the whole group: a PARALLEL_ROWS-wide MAC
        // already covers the accumulator dependency, so there is nothing to
        // interleave against. It opens at zero and closes holding
        // sum_i q_i x_i + 128 sum_i x_i; see NIBBLE_BIAS for where that second
        // term goes.
        aie::accum<accfloat, PARALLEL_ROWS> temp_acc;
        temp_acc.from_vector(aie::zeros<bf16, PARALLEL_ROWS>());

        AIE_LOOP_UNROLL_FULL
        for (int l = 0; l < LOADS_PER_GROUP; l++) {
            aie::vector<uint4, NIBBLES_PER_LOAD> q_raw =
                aie::load_v<NIBBLES_PER_LOAD>(it_quant);
            it_quant += NIBBLES_PER_LOAD / 2; // uint4* steps two nibbles

            // uint4 -> bf16 by placing bytes; exact, and biased by +128
            aie::vector<bf16, NIBBLES_PER_LOAD> q_bf16 =
                _q4_biased_bf16(q_raw, bias_bytes);

            AIE_LOOP_UNROLL_FULL
            for (int k = 0; k < COLS_PER_LOAD; k++) {
                aie::vector<bf16, PARALLEL_ROWS> a_col =
                    q_bf16.template extract<PARALLEL_ROWS>(k);
                bfloat16 scalar = b_col.get(l * COLS_PER_LOAD + k);
                temp_acc = aie::mac(temp_acc, a_col, scalar);
            }
        }

        // s is stored bf16, so the only rounding on this path would be the group
        // sum; split it into bf16 limbs instead of throwing 16 of its 24
        // mantissa bits away. Three of them, because temp_acc still carries the
        // +128 and so runs ~15x the group sum -- a 2-limb split would measure
        // its 2^-17 against the large number. See NIBBLE_BIAS.
        aie::vector<bf16, PARALLEL_ROWS> a_scales = aie::load_v<PARALLEL_ROWS>(it_scale);
        c_accum = bf16_limb::mac_f32_bf16_safe(
            c_accum, temp_acc.template to_vector<float>(), a_scales);

        // The bias path: sum_g s_g sum_i x_i, two MACs against the same
        // pre-split sum_i x_i the min path reads. Q4_1 has no super-block, so
        // this accumulator is already the row-level one and _Q41_finalize
        // applies the 128 after the whole K loop.
        c_bias_accum = aie::mac(c_bias_accum, a_scales, *(it_reduce_b + 1));
        c_bias_accum = aie::mac(c_bias_accum, a_scales, *it_reduce_b);

        // sum_i x_i arrives pre-split; the min path needs only the top two
        // limbs, smallest first so the small term joins c_accum before the large.
        aie::vector<bf16, PARALLEL_ROWS> a_mins = aie::load_v<PARALLEL_ROWS>(it_min);
        c_accum = aie::mac(c_accum, a_mins, *(it_reduce_b + 1));
        c_accum = aie::mac(c_accum, a_mins, *it_reduce_b);

        it_reduce_b += REDUCE_LIMBS;
        it_x     += GGML_GROUP_SIZE;
        it_scale += PARALLEL_ROWS;
        it_min   += PARALLEL_ROWS;
    }

    aie::store_v(c, c_accum.template to_vector<float>());
    aie::store_v(c_bias, c_bias_accum.template to_vector<float>());
}

/// @brief Settle the Q4_1 bias path, once the K loop is done.
///
/// c -= 128 * c_bias, the Q4_1 counterpart of _Q4K_finalize: one msc per 32-row
/// block for every biased MAC in the row. c_bias is split to 3 limbs and 128 is
/// a power of two, so the cancellation is exact and what is left is the 2-limb
/// truncation of sum_i x_i, see REDUCE_LIMBS.
void _Q41_finalize(float * c, const float * c_bias) {
    aie::accum<accfloat, PARALLEL_ROWS> c_accum;
    c_accum.from_vector(aie::load_v<PARALLEL_ROWS>(c));
    c_accum = bf16_limb::mac_f32_bf16_safe(
        c_accum, aie::load_v<PARALLEL_ROWS>(c_bias),
        aie::broadcast<bf16, PARALLEL_ROWS>(bfloat16(-NIBBLE_BIAS)));
    aie::store_v(c, c_accum.template to_vector<float>());
}

#endif // __Q4_CORE_H__