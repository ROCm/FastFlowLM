// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

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

/// @brief y[t] += W x[t] for one Q4_K weight block and L activation rows.
///
/// Same datapath and the same rounding points as _Q4K_x_bf16 above -- group sum
/// in float32, two-limb split before s', a second two-limb split before S', min
/// path in its own accumulator -- with two changes forced by having L rows
/// instead of one:
///
///  1. The accumulator pair no longer fits in registers. At L = 8 the running
///     scale and min paths are 2 x L x PARALLEL_ROWS floats; AIE2 has four
///     1024-bit AM registers, i.e. room for about four aie::accum<accfloat,32>.
///     So they live in memory and are loaded / stored around each fold. Only
///     the group sum and the per-block sub-total stay in registers, which is
///     where the 32 MACs of the inner loop are.
///
///  2. The nibbles are unpacked once per group rather than once per (group,
///     row). One 32-column group is PARALLEL_ROWS x GGML_GROUP_SIZE nibbles;
///     expanded to bf16 that is 2 KiB of scratch, read back by every one of the
///     L rows. Without this the uint4 -> bf16 conversion would be redone L
///     times for the same weights and would dominate the MACs.
///
/// The min path is hoisted out of the group loop entirely: it needs no weights
/// beyond m'_g, so all 8 groups of one row reduce in one register accumulator
/// and touch memory once per (row, block) instead of once per (row, group).
///
/// @param c_accum_s   [L][PARALLEL_ROWS] float32, running scale path, in place
/// @param c_accum_m   [L][PARALLEL_ROWS] float32, running min path, in place
/// @param acc_s_group [L][PARALLEL_ROWS] float32 scratch, one weight block wide
/// @param x           [L][QXNX_COL_BLOCK_SIZE] bf16, row major
/// @param w_buffer    the weight block
/// @param b_col_reduce_buffer [L][2 * GROUPS_PER_SB] bf16, (hi, lo) limbs of
///                    sum_i x_i per group, per row
/// @param q_scratch   [GGML_GROUP_SIZE][PARALLEL_ROWS] bf16 scratch
#ifdef Q4K_MAC_LEGACY
template <int L>
void _Q4K_x_bf16_L(
    float * __restrict c_accum_s,
    float * __restrict c_accum_m,
    float * __restrict acc_s_group,
    const bf16 * __restrict x,
    q4k_block_t* w_buffer,
    const bf16 * __restrict b_col_reduce_buffer,
    bf16 * __restrict q_scratch
) {
    static constexpr int SUPER_BLOCK_SIZE  = 256;
    static constexpr int GROUPS_PER_SB     = SUPER_BLOCK_SIZE / GGML_GROUP_SIZE;
    static constexpr int NIBBLES_PER_LOAD  = 128;
    static constexpr int COLS_PER_LOAD     = NIBBLES_PER_LOAD / PARALLEL_ROWS;
    static constexpr int LOADS_PER_GROUP   = GGML_GROUP_SIZE / COLS_PER_LOAD;

    static_assert(PARALLEL_ROWS == QXNX_ROW_BLOCK_SIZE);
    static_assert(QXNX_COL_BLOCK_SIZE == SUPER_BLOCK_SIZE);
    static_assert(NIBBLES_PER_LOAD % PARALLEL_ROWS == 0);
    static_assert(GGML_GROUP_SIZE % COLS_PER_LOAD == 0);

    uint4* it_quant = w_buffer->qs;
    uint8* it_scale = w_buffer->scales;

    const aie::vector<bf16, PARALLEL_ROWS> a_S = aie::load_v<PARALLEL_ROWS>(w_buffer->S);
    const aie::vector<bf16, PARALLEL_ROWS> a_M = aie::load_v<PARALLEL_ROWS>(w_buffer->M);

    // ---- scale path -------------------------------------------------------
    for (int g = 0; g < GROUPS_PER_SB; g++) {
        // qs is column major and a group is 32 consecutive columns, so the
        // group's nibbles are contiguous and it_quant just walks forward.
        AIE_LOOP_UNROLL_FULL
        for (int l = 0; l < LOADS_PER_GROUP; l++) {
            aie::vector<uint4, NIBBLES_PER_LOAD> q_raw =
                aie::load_v<NIBBLES_PER_LOAD>(it_quant);
            it_quant += NIBBLES_PER_LOAD / 2; // uint4* steps two nibbles

            // uint4 -> bf16 is exact
            aie::accum<accfloat, NIBBLES_PER_LOAD> q_acc;
            q_acc.from_vector(aie::to_float(q_raw, 0));
            aie::store_v(q_scratch + l * NIBBLES_PER_LOAD,
                         q_acc.template to_vector<bf16>());
        }

        // uint8 s' -> bf16, exact for every value up to 255
        aie::accum<accfloat, PARALLEL_ROWS> s_acc;
        s_acc.from_vector(aie::to_float(aie::load_v<PARALLEL_ROWS>(it_scale), 0));
        const aie::vector<bf16, PARALLEL_ROWS> a_scale =
            s_acc.template to_vector<bf16>();

        for (int t = 0; t < L; t++) {
            aie::vector<bf16, GGML_GROUP_SIZE> b_col =
                aie::load_v<GGML_GROUP_SIZE>(x + t * SUPER_BLOCK_SIZE + g * GGML_GROUP_SIZE);

            aie::accum<accfloat, PARALLEL_ROWS> temp_acc;
            temp_acc.from_vector(aie::zeros<bf16, PARALLEL_ROWS>());

            // Load/extract in the same 128-wide shape the single-row kernel
            // used. Loading the 32 columns one 32-lane vector at a time instead
            // keeps all 32 live across the unrolled body and spills 2 KiB to a
            // stack the core does not have.
            AIE_LOOP_UNROLL_FULL
            for (int l = 0; l < LOADS_PER_GROUP; l++) {
                aie::vector<bf16, NIBBLES_PER_LOAD> q_bf16 =
                    aie::load_v<NIBBLES_PER_LOAD>(q_scratch + l * NIBBLES_PER_LOAD);
                AIE_LOOP_UNROLL_FULL
                for (int c = 0; c < COLS_PER_LOAD; c++) {
                    temp_acc = aie::mac(temp_acc,
                                        q_bf16.template extract<PARALLEL_ROWS>(c),
                                        b_col.get(l * COLS_PER_LOAD + c));
                }
            }

            // Same two-limb split as the single-row kernel, and for the same
            // reason: rounding the group sum to bf16 here is the largest error
            // the kernel can introduce.
            aie::accum<accfloat, PARALLEL_ROWS> ls;
            if (g == 0) {
                ls.from_vector(aie::zeros<bf16, PARALLEL_ROWS>());
            } else {
                ls.from_vector(aie::load_v<PARALLEL_ROWS>(acc_s_group + t * PARALLEL_ROWS));
            }
            ls = bf16_limb::mac_f32_bf16_low(
                ls, temp_acc.template to_vector<float>(), a_scale);
            aie::store_v(acc_s_group + t * PARALLEL_ROWS,
                         ls.template to_vector<float>());
        }
        it_scale += PARALLEL_ROWS;
    }

    // S' applied once per super-block per row, exactly as in _Q4K_x_bf16.
    for (int t = 0; t < L; t++) {
        aie::accum<accfloat, PARALLEL_ROWS> cs;
        cs.from_vector(aie::load_v<PARALLEL_ROWS>(c_accum_s + t * PARALLEL_ROWS));
        cs = bf16_limb::mac_f32_bf16_low(
            cs, aie::load_v<PARALLEL_ROWS>(acc_s_group + t * PARALLEL_ROWS), a_S);
        aie::store_v(c_accum_s + t * PARALLEL_ROWS, cs.template to_vector<float>());
    }

    // ---- min path ---------------------------------------------------------
    for (int t = 0; t < L; t++) {
        const bf16* it_reduce_b = b_col_reduce_buffer + t * 2 * GROUPS_PER_SB;
        uint8* it_min = w_buffer->mins;

        aie::accum<accfloat, PARALLEL_ROWS> mm;
        mm.from_vector(aie::zeros<bf16, PARALLEL_ROWS>());

        for (int g = 0; g < GROUPS_PER_SB; g++) {
            aie::accum<accfloat, PARALLEL_ROWS> m_acc;
            m_acc.from_vector(aie::to_float(aie::load_v<PARALLEL_ROWS>(it_min), 0));
            aie::vector<bf16, PARALLEL_ROWS> a_mins = m_acc.template to_vector<bf16>();
            // sum_i x_i arrives pre-split, smallest limb first
            mm = aie::mac(mm, a_mins, *(it_reduce_b + 1));
            mm = aie::mac(mm, a_mins, *it_reduce_b);
            it_reduce_b += 2;
            it_min += PARALLEL_ROWS;
        }

        aie::accum<accfloat, PARALLEL_ROWS> cm;
        cm.from_vector(aie::load_v<PARALLEL_ROWS>(c_accum_m + t * PARALLEL_ROWS));
        cm = bf16_limb::mac_f32_bf16_low(cm, mm.template to_vector<float>(), a_M);
        aie::store_v(c_accum_m + t * PARALLEL_ROWS, cm.template to_vector<float>());
    }
}

/// @brief Zero the L-row accumulator pair, once per output row block.
template <int L>
void _Q4K_zero_L(float * c_accum_s, float * c_accum_m) {
    zero_vectorized<float, L * PARALLEL_ROWS>(c_accum_s);
    zero_vectorized<float, L * PARALLEL_ROWS>(c_accum_m);
}

/// @brief Collapse the L-row accumulator pair into [L][PARALLEL_ROWS] float32.
template <int L>
void _Q4K_finalize_L(
    float * c,
    const float * c_accum_s,
    const float * c_accum_m
) {
    AIE_LOOP_UNROLL_FULL
    for (int t = 0; t < L; t++) {
        aie::store_v(c + t * PARALLEL_ROWS,
                     aie::add(aie::load_v<PARALLEL_ROWS>(c_accum_s + t * PARALLEL_ROWS),
                              aie::load_v<PARALLEL_ROWS>(c_accum_m + t * PARALLEL_ROWS)));
    }
}

#else
// ---------------------------------------------------------------------------
// L = 8 rows through one Q4_K weight block on the matrix unit.
//
// The vector x scalar formulation this replaces issued one 32-lane
// aie::mac per (column, token): 2048 of them per 32 x 256 block, and measured
// at ~62% of a layer dispatch (14.7 of 23.7 ms for the delta-net layer, see
// the notes on the proj engine). Here the same block is 128 aie::mmul<8,8,8>
// issues plus operand shuffling:
//
//   C[t][i] (8 tokens x 8 rows of a row tile) = sum_k A[t][k] * B[k][i]
//
//   B = the unpacked weights, 8 columns x 8 rows, built from the nibbles with
//       the same interleave_unzip as the prefill dequant_mm kernel
//   A = the activations, 8 tokens x 8 columns (8 loads + concat)
//
// Numerics -- read this before changing anything.
//   * The repo builds kernels with -DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16,
//     so a bf16 mmul quantises BOTH operands to bfp16ebs8 (8 values share an
//     exponent, 8-bit mantissa). The weights are integers 0..15, which are
//     exact. The activations are not: x is rounded to ~2^-8 of the largest
//     element in its 8-column block.
//   * The weight is w = S' s'_g q + M' m'_g, and q (0..15) is far from zero
//     mean, so the scale path and the min path each carry a large common-mode
//     term that cancels in the sum. The min path used to take sum_c x_c
//     exactly; against a scale path that now sees the quantised x the two no
//     longer cancel and the layer output went to cos ~0.5.  So the group sum
//     is taken through the SAME quantising datapath: mmul(A, ones) gives
//     sum_c x_q[t][c], and the min path multiplies that.  Both paths then see
//     identical activations and only w . (x - x_q) remains, ~1% relative on a
//     delta-net layer, ~2% on an attention layer, cos >= 0.9997 against the
//     previous kernel, and no change in MTP acceptance over six prompts.
//   * the per-group scale s'_g still multiplies a float32 group sum with a
//     two-limb bf16 split (bf16_limb::mac_f32_bf16_low), as before.
//
// Accumulator layout (c_accum_s, c_accum_m, acc_s_group, mm_group): 4 row
// tiles x 64 floats, lane = t * 8 + i  <->  token t, row 8 * rt + i.
// _Q4K_finalize_L un-tiles into y[t][32] once per row block.
//
// Build with -DQ4K_MAC_LEGACY to get the old vector x scalar kernel back.
// ---------------------------------------------------------------------------
#ifndef __Q4_MMUL_CORE_H__
#define __Q4_MMUL_CORE_H__

#if !defined(__chess__) && defined(AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16)
// bf16 8x8x8 mmul with the BFP16 emulation, for Peano. It issues the same
// intrinsic as aie_api's emulated mac but converts B with the accumulator
// constructor rather than a multiply by one, which would take a multiplier slot
// from the macs. The conversion is exact, so results are unchanged.
struct q4k_mmul_888 {
    aie::accum<accfloat, 64> acc;
    q4k_mmul_888() { acc.from_vector(aie::zeros<float, 64>()); }
    __attribute__((always_inline)) void mac(const aie::vector<bfloat16, 64> &a, const aie::vector<bfloat16, 64> &b) {
        aie::accum<accfloat, 64> acc_a(a);
        aie::accum<accfloat, 64> acc_b(aie::transpose(b, 8, 8));
        acc = ::mac_8x8_8x8T_conf(::to_v64bfp16ebs8(acc_a), ::to_v64bfp16ebs8(acc_b), acc, 0, 0, 0);
    }
    __attribute__((always_inline)) void mul(const aie::vector<bfloat16, 64> &a, const aie::vector<bfloat16, 64> &b) { mac(a, b); }
    template <typename T> aie::vector<T, 64> to_vector() const { return acc.template to_vector<T>(); }
};
#define Q4K_MMUL q4k_mmul_888
#else
#define Q4K_MMUL MMUL
#endif

alignas(aie::vector_decl_align) static float mm_group[L_TOKENS * QXNX_ROW_BLOCK_SIZE];
alignas(aie::vector_decl_align) static bf16 sc_buf[QXNX_ROW_BLOCK_SIZE];
alignas(aie::vector_decl_align) static bf16 mn_buf[QXNX_ROW_BLOCK_SIZE];

template <int L>
void _Q4K_x_bf16_L(
    float * __restrict c_accum_s,
    float * __restrict c_accum_m,
    float * __restrict acc_s_group,
    const bf16 * __restrict x,
    q4k_block_t* w_buffer,
    const bf16 * __restrict /*b_col_reduce_buffer, unused: group sums come from the mmul*/,
    bf16 * __restrict q_scratch
) {
    static_assert(L == 8, "mmul tiling is written for 8 tokens");
    static_assert(QXNX_ROW_BLOCK_SIZE == 32 && QXNX_COL_BLOCK_SIZE == 256 && GGML_GROUP_SIZE == 32);
    static constexpr int GROUPS_PER_SB = 8;
    static constexpr int RT = 4;           // row tiles of 8 rows
    static constexpr int KS = 4;           // k steps of 8 columns per group
    static constexpr int TILE = 64;        // floats per accumulator tile
    using MMUL = aie::mmul<8, 8, 8, bf16, bf16, accfloat>;

    uint4* it_quant = w_buffer->qs;
    uint8* it_scale = w_buffer->scales;
    uint8* it_min   = w_buffer->mins;

    AIE_LOOP_NO_UNROLL
    for (int g = 0; g < GROUPS_PER_SB; g++) {
        // ---- weights of this group -> B tiles q_scratch[rt][s][k8][n8] -----------
#if defined(__chess__)
        AIE_LOOP_NO_UNROLL
#else
        // Unrolled under Peano, with the loop inside it and the row-tile loop
        // below: Peano schedules this loop nest badly when it stays rolled.
        AIE_LOOP_UNROLL_FULL
#endif
        for (int s = 0; s < KS; s++) {
#if !defined(__chess__)
            AIE_LOOP_UNROLL_FULL
#endif
            for (int half = 0; half < 2; half++) {
                // 128 nibbles = 4 columns x 32 rows; uint4 -> bf16 is exact
                aie::vector<uint4, 128> q_raw = aie::load_v<128>(it_quant);
                it_quant += 64;
                aie::vector<bf16, 128> d = aie::to_float<bf16>(q_raw, 0);
                // chunks of 8 rows, sequence index = col * 4 + rt; two unzips
                // regroup them rt-major
                auto u = aie::interleave_unzip(d.template extract<64>(0),
                                               d.template extract<64>(1), 8);
                auto ev = aie::interleave_unzip(u.first.template extract<32>(0),
                                                u.first.template extract<32>(1), 8);
                auto od = aie::interleave_unzip(u.second.template extract<32>(0),
                                                u.second.template extract<32>(1), 8);
                bf16* qp = q_scratch + s * 64 + half * 32;
                aie::store_v(qp + 0 * KS * 64, ev.first);     // rt 0
                aie::store_v(qp + 1 * KS * 64, od.first);     // rt 1
                aie::store_v(qp + 2 * KS * 64, ev.second);    // rt 2
                aie::store_v(qp + 3 * KS * 64, od.second);    // rt 3
            }
        }

        // per-row s'_g and m'_g as bf16 (exact for uint8), parked for the splats
        {
            aie::accum<accfloat, 32> s_acc;
            s_acc.from_vector(aie::to_float(aie::load_v<32>(it_scale), 0));
            aie::store_v(sc_buf, s_acc.template to_vector<bf16>());
            aie::accum<accfloat, 32> m_acc;
            m_acc.from_vector(aie::to_float(aie::load_v<32>(it_min), 0));
            aie::store_v(mn_buf, m_acc.template to_vector<bf16>());
        }

        // ---- activation tiles A[s]: 8 tokens x 8 columns ---------------------------
        aie::vector<bf16, 64> A[KS];
        for (int s = 0; s < KS; s++) {
            const bf16* p = x + g * 32 + s * 8;
            A[s] = aie::concat(aie::load_v<8>(p),
                               aie::load_v<8>(p + 256),
                               aie::load_v<8>(p + 512),
                               aie::load_v<8>(p + 768),
                               aie::load_v<8>(p + 1024),
                               aie::load_v<8>(p + 1280),
                               aie::load_v<8>(p + 1536),
                               aie::load_v<8>(p + 1792));
        }

        // group sum of the quantised activations, split into two bf16 limbs
        aie::vector<bf16, 64> xs_hi, xs_lo;
        {
            const aie::vector<bf16, 64> ones64 = aie::broadcast<bf16, 64>((bf16)1.0f);
            Q4K_MMUL C1;
            C1.mul(A[0], ones64);
            C1.mac(A[1], ones64);
            C1.mac(A[2], ones64);
            C1.mac(A[3], ones64);
            auto lim = bf16_limb::split_f32<64>(C1.template to_vector<float>());
            xs_hi = lim.first;
            xs_lo = lim.second;
        }

#if defined(__chess__)
        AIE_LOOP_NO_UNROLL
#else
        AIE_LOOP_UNROLL_FULL
#endif
        for (int rt = 0; rt < RT; rt++) {
            // scale path: group sum of q * x_q over the group's 32 columns
            Q4K_MMUL C;
            C.mul(A[0], aie::load_v<64>(q_scratch + (rt * KS + 0) * 64));
            C.mac(A[1], aie::load_v<64>(q_scratch + (rt * KS + 1) * 64));
            C.mac(A[2], aie::load_v<64>(q_scratch + (rt * KS + 2) * 64));
            C.mac(A[3], aie::load_v<64>(q_scratch + (rt * KS + 3) * 64));
            aie::vector<float, 64> G = C.template to_vector<float>();

            const aie::vector<bf16, 64> splat =
                aie::load_v<8>(sc_buf + 8 * rt).template grow_replicate<64>();
            float* lp = acc_s_group + rt * TILE;
            aie::accum<accfloat, 64> ls;
            if (g == 0) ls.from_vector(aie::zeros<float, 64>());
            else        ls.from_vector(aie::load_v<64>(lp));
            ls = bf16_limb::mac_f32_bf16_low(ls, G, splat);
            aie::store_v(lp, ls.template to_vector<float>());

            // min path: m'_g * sum x_q
            const aie::vector<bf16, 64> msplat =
                aie::load_v<8>(mn_buf + 8 * rt).template grow_replicate<64>();
            float* mp = mm_group + rt * TILE;
            aie::accum<accfloat, 64> mm;
            if (g == 0) mm.from_vector(aie::zeros<float, 64>());
            else        mm.from_vector(aie::load_v<64>(mp));
            mm = aie::mac(mm, msplat, xs_lo);     // smallest limb first
            mm = aie::mac(mm, msplat, xs_hi);
            aie::store_v(mp, mm.template to_vector<float>());
        }

        it_scale += 32;
        it_min   += 32;
    }

    // S' and M' once per block, per row tile
    AIE_LOOP_NO_UNROLL
    for (int rt = 0; rt < RT; rt++) {
        const aie::vector<bf16, 64> sS = aie::load_v<8>(w_buffer->S + 8 * rt).template grow_replicate<64>();
        const aie::vector<bf16, 64> sM = aie::load_v<8>(w_buffer->M + 8 * rt).template grow_replicate<64>();
        const int off = rt * TILE;
        aie::accum<accfloat, 64> cs;
        cs.from_vector(aie::load_v<64>(c_accum_s + off));
        cs = bf16_limb::mac_f32_bf16_low(cs, aie::load_v<64>(acc_s_group + off), sS);
        aie::store_v(c_accum_s + off, cs.template to_vector<float>());

        aie::accum<accfloat, 64> cm;
        cm.from_vector(aie::load_v<64>(c_accum_m + off));
        cm = bf16_limb::mac_f32_bf16_low(cm, aie::load_v<64>(mm_group + off), sM);
        aie::store_v(c_accum_m + off, cm.template to_vector<float>());
    }
}

/// @brief Zero the L-row accumulator pair, once per output row block.
template <int L>
void _Q4K_zero_L(float * c_accum_s, float * c_accum_m) {
    zero_vectorized<float, L * PARALLEL_ROWS>(c_accum_s);
    zero_vectorized<float, L * PARALLEL_ROWS>(c_accum_m);
}

/// @brief Collapse the tiled accumulator pair into y [L][PARALLEL_ROWS] float32.
template <int L>
void _Q4K_finalize_L(
    float * c,
    const float * c_accum_s,
    const float * c_accum_m
) {
    AIE_LOOP_NO_UNROLL
    for (int rt = 0; rt < 4; rt++) {
        aie::vector<float, 64> v = aie::add(aie::load_v<64>(c_accum_s + rt * 64),
                                            aie::load_v<64>(c_accum_m + rt * 64));
        for (int t = 0; t < 8; t++)
            aie::store_v(c + t * PARALLEL_ROWS + 8 * rt, v.template extract<8>(t));
    }
}

#endif // __Q4_MMUL_CORE_H__
#endif // Q4K_MAC_LEGACY

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