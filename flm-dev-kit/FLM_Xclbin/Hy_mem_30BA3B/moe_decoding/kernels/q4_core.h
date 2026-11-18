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

// ===========================================================================
// int8 x / uint4 W path (Q4K_INT8_MMUL = 1, the default)
//
// The group sums sum_i q_i x_i run on the matrix unit, on the native AIE2P
// instruction mac_4x8_8x16: A int8 4 x 8, B int8 8 x 16, C int32 4 x 16. B is the
// 4-bit weights of 8 columns x 16 rows, unpacked to int8 by one native unpack.
// (aie::mmul<4,16,16, int8, uint4> exists but is built from two of these plus a
// shuffle of A on every call: ~34 cycles each, measured.)
//
//   x -> int8: per 32-value group g, x8 = round(x / 2^e_g), e_g picked so the
//   largest |x| lands in 64..127. The scale is a power of two, so undoing it
//   later is exact. Done once per x slice (the same x serves every row block).
//
//   A trick: the 4 A rows hold 4 different groups. For the 8 columns of step s
//   (group g = s / 4), only row g % 4 of A is filled, so row a of the result
//   collects group a's sum. One accumulator holds 4 groups (groups 0..3 and
//   4..7 in two of them), per 16 output rows (t = 0, 1): no reset per group.
//   The 32 A matrices of an x slice (1 KB) are made once with the int8 x.
//
//   Host layout (decoding_layer: q4k_relayout_for_mmul), per 32 x 256 block:
//     qs     : [16-column step 16][t 2][16 columns][8 bytes]: 128 B; its two
//              64 B halves are the B matrices of 8 columns
//     scales : [h 2][t 2][a 4][16 rows]  (group g = 4 h + a, row r = 16 t + n)
//     mins, S, M : unchanged
//
// The int32 group sums are exact; times 2^e_g they are exact in fp32. From there
// the scale / min paths are the same as _Q4K_x_bf16's. The one new error is x's
// rounding to 8 bits per group (bf16 itself carries 8).
// ===========================================================================
#ifndef Q4K_INT8_MMUL
#define Q4K_INT8_MMUL 1
#endif

constexpr int Q4K_STEPS = QXNX_COL_BLOCK_SIZE / 8;            // 32 steps of 8 columns
constexpr int Q4K_A_BYTES = Q4K_STEPS * 32;                   // A matrices of one x slice (4 x 8 int8 each)
constexpr int Q4K_E_ELEMS = 2 * 64;                           // 2^e_g per lane, both halves

/// @brief One x slice (QXNX_COL_BLOCK_SIZE bf16) -> the 32 A matrices (int8 x),
/// the per-lane group scales 2^e_g, and the min path's per-group sums. Once per
/// x slice per mvm().
///
/// The sums are of the ROUNDED x (sum x8 * 2^e, exact in two bf16 limbs), not of
/// x: then the block computes exactly W x^ (x^ = the rounded x) and the error is
/// W (x^ - x). With the exact sum of x in the min path instead, the error would be
/// S' s' q (x^ - x), and q is never negative (mean ~7.5), so the rounding errors
/// would add up instead of cancelling against the min term.
inline void q4k_prepare_x(const bf16* x, int8* a_tiles, bf16* e64, bf16* reduce_out)
{
    constexpr int GROUPS = QXNX_COL_BLOCK_SIZE / GGML_GROUP_SIZE;   // 8
    static_assert(GGML_GROUP_SIZE == 32 && GROUPS == 8);
    aie::set_rounding(aie::rounding_mode::conv_even);
    aie::set_saturation(aie::saturation_mode::saturate);

    // A rows that are never written must be 0. Every x slice writes the same
    // positions (row g % 4 of steps 4g .. 4g+3), so zero the cache once.
    // No scalar float below: this core has no scalar FPU (softfloat, slow).
    alignas(64) int32 sum8[16] = {};                    // per group: sum of x8
    alignas(64) uint16_t p2[16] = {};                   // per group: 2^e as bf16 bits
    for (int g = 0; g < GROUPS; g++) {
        aie::vector<bf16, 32> xv = aie::load_v<32>(x + g * 32);

        // max |x| from the bf16 bits: clear the sign; for non-negative bf16 the
        // bit pattern orders like the value. Its exponent field gives e.
        aie::vector<int16, 32> ax = aie::bit_and(xv.template cast_to<int16>(), aie::broadcast<int16, 32>(0x7FFF));
        const int max_bits = aie::reduce_max(ax);
        // m = f * 2^k, f in [1, 2): e = k - 6 puts m / 2^e in [64, 128)
        // (a value that rounds to 128 saturates to 127); clamp keeps 2^+-e normal
        int e = max_bits == 0 ? 0 : (max_bits >> 7) - 127 - 6;
        e = e < -120 ? -120 : (e > 120 ? 120 : e);

        // x * 2^-e: exact (a power of two only moves the exponent)
        union { uint16_t u; bfloat16 b; } inv;
        inv.u = (uint16_t)((127 - e) << 7);
        aie::accum<accfloat, 32> scaled = aie::mul(xv, inv.b);
        // round to the nearest integer with the fp32 "magic number": adding
        // 1.5 * 2^23 leaves the integer in the low mantissa bits (fp32 add rounds
        // to nearest even); |v| < 128 is far below 2^22. The bf16 -> int
        // conversion is not used: it is software, slow, and rounds down.
        constexpr float MAGIC = 12582912.0f;            // 1.5 * 2^23, bits 0x4B400000
        scaled = aie::add(scaled, aie::broadcast<float, 32>(MAGIC));
        aie::vector<int32, 32> xi = aie::sub(scaled.template to_vector<float>().template cast_to<int32>(),
                                             aie::broadcast<int32, 32>(0x4B400000));
        aie::accum<acc32, 32> xi_acc;
        xi_acc.from_vector(xi);
        aie::vector<int8, 32> x8 = xi_acc.template to_vector<int8>(0);   // saturating (128 -> 127)

        // group g = steps 4g .. 4g+3 (8 columns each), in A row g % 4
        // (8 bytes at a time: there is no 8-lane int8 vector, so two 32-bit words)
        const int row = g % 4;
        alignas(32) int8 x8_mem[32];
        aie::store_v(x8_mem, x8);
        AIE_LOOP_UNROLL_FULL
        for (int s = 0; s < 4; s++) {
            uint32_t* dst = (uint32_t*)(a_tiles + (4 * g + s) * 32 + row * 8);
            const uint32_t* src = (const uint32_t*)(x8_mem + s * 8);
            dst[0] = src[0];
            dst[1] = src[1];
        }

        // 2^e as bf16 (exponent field only), on group g's 16 lanes of half g / 4
        union { uint16_t u; bfloat16 b; } p2e;
        p2e.u = (uint16_t)((e + 127) << 7);
        aie::store_v(e64 + (g / 4) * 64 + row * 16, aie::broadcast<bfloat16, 16>(p2e.b));
        p2[g] = p2e.u;

        // sum of the (saturated) int8 values, for the min path
        aie::accum<acc32, 32> x8_acc;
        x8_acc.from_vector(x8);
        sum8[g] = aie::reduce_add(x8_acc.template to_vector<int32>(0));
    }

    // min path sums of all 8 groups at once, in vector ops: sum8 * 2^e as two bf16
    // limbs (hi, lo). |sum8| <= 32 * 127 needs 12 bits, so the limbs hold it
    // exactly, and times a power of two they stay exact.
    aie::vector<float, 16> sf = aie::to_float<float>(aie::load_v<16>(sum8), 0);
    auto [l0, l1] = bf16_limb::split_f32<16>(sf);
    aie::vector<bf16, 16> p2v = aie::load_v<16>((bf16*)p2);
    aie::vector<bf16, 16> hi = aie::mul(l0, p2v).template to_vector<bf16>();
    aie::vector<bf16, 16> lo = aie::mul(l1, p2v).template to_vector<bf16>();
    aie::store_v(reduce_out, aie::interleave_zip(hi, lo, 1).first);   // hi0 lo0 hi1 lo1 ...
}

// One step of 8 columns: A from the cache, B = the (step % 2) half of the 128 B
// tile of 16-column step step / 2, half t = 0 / 1 (uint4 pointers step one byte).
#define Q4K_STEP_OPERANDS(step)                                                          \
    aie::vector<int8, 64> a = aie::concat(aie::load_v<32>(a_tiles + (step) * 32),       \
                                          aie::zeros<int8, 32>());                      \
    const uint4* tile = q + ((step) / 2) * 256 + ((step) % 2) * 64;                     \
    v128int8 b0 = (v128int8)::unpack((v128uint4)aie::load_v<128>(tile), 0);             \
    v128int8 b1 = (v128int8)::unpack((v128uint4)aie::load_v<128>(tile + 128), 0);

/// @brief The group sums of groups 4H .. 4H+3 (steps 16H .. 16H+15) for both row
/// halves: acc_t0 rows 0..15, acc_t1 rows 16..31; row a of C = group 4H + a.
/// Not inlined, loop not unrolled: one big basic block made the scheduler give up.
template <int H>
__attribute__((noinline)) void q4k_group_sums(aie::accum<acc32, 64>& acc_t0, aie::accum<acc32, 64>& acc_t1,
                                              const int8* a_tiles, const uint4* q)
{
    v64acc32 c0, c1;
    {   // first step: mul (A signed = 1, B unsigned = 0)
        Q4K_STEP_OPERANDS(16 * H)
        c0 = ::mul_4x8_8x16(a, 1, b0, 0);
        c1 = ::mul_4x8_8x16(a, 1, b1, 0);
    }
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_NO_UNROLL
    for (int s = 1; s < 16; s++) {
        Q4K_STEP_OPERANDS(16 * H + s)
        c0 = ::mac_4x8_8x16(a, 1, b0, 0, c0);
        c1 = ::mac_4x8_8x16(a, 1, b1, 0, c1);
    }
    acc_t0 = aie::accum<acc32, 64>(c0);
    acc_t1 = aie::accum<acc32, 64>(c1);
}

/// @brief Scale path of one row half t: sum over the 8 groups of s'_g 2^e_g G_g
/// for its 16 rows (fp32). acc_h0 / acc_h1: the group sums of groups 0..3 / 4..7.
__attribute__((noinline)) aie::vector<float, 16> q4k_scale_rows(
    const aie::accum<acc32, 64>& acc_h0, const aie::accum<acc32, 64>& acc_h1,
    const uint8* scales_t0h0, const uint8* scales_t0h1, const bf16* e64)
{
    aie::accum<accfloat, 64> s64;
    s64.from_vector(aie::zeros<bf16, 64>());
    const aie::accum<acc32, 64>* accs[2] = {&acc_h0, &acc_h1};
    const uint8* scales[2] = {scales_t0h0, scales_t0h1};
    AIE_LOOP_NO_UNROLL
    for (int h = 0; h < 2; h++) {
        // int32 sums are exact; to_float is exact below 2^24 (|G| <= 32*15*128)
        aie::vector<float, 64> g = aie::to_float<float>(accs[h]->template to_vector<int32>(0), 0);
        // s' (uint8) * 2^e (power of two) is exact in bf16
        aie::accum<accfloat, 64> s_acc;
        s_acc.from_vector(aie::to_float(aie::load_v<64>(scales[h]), 0));
        aie::accum<accfloat, 64> sd_acc = aie::mul(s_acc.template to_vector<bf16>(), aie::load_v<64>(e64 + h * 64));
        s64 = bf16_limb::mac_f32_bf16_low(s64, g, sd_acc.template to_vector<bf16>());
    }
    // lanes are (group a, row n): add the 4 groups -> 16 rows
    aie::vector<float, 64> v = s64.template to_vector<float>();
    aie::accum<accfloat, 32> v32;
    v32.from_vector(v.template extract<32>(0));
    v32 = aie::add(v32, v.template extract<32>(1));
    aie::vector<float, 32> w32 = v32.template to_vector<float>();
    aie::accum<accfloat, 16> v16;
    v16.from_vector(w32.template extract<16>(0));
    v16 = aie::add(v16, w32.template extract<16>(1));
    return v16.template to_vector<float>();
}

/// @brief Min path of one block: sum over groups of m'_g * (sum of x_g), then
/// times M', into c_accum_m. As _Q4K_x_bf16's.
__attribute__((noinline)) void q4k_min_path(q4k_accum_t& c_accum_m, q4k_block_t* w_buffer, const bf16* reduce)
{
    aie::accum<accfloat, PARALLEL_ROWS> acc_m;
    acc_m.from_vector(aie::zeros<bf16, PARALLEL_ROWS>());
    const uint8* it_min = w_buffer->mins;
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_RANGE(QXNX_COL_BLOCK_SIZE / GGML_GROUP_SIZE, QXNX_COL_BLOCK_SIZE / GGML_GROUP_SIZE)
    for (int g = 0; g < QXNX_COL_BLOCK_SIZE / GGML_GROUP_SIZE; g++) {
        aie::accum<accfloat, PARALLEL_ROWS> m_acc;
        m_acc.from_vector(aie::to_float(aie::load_v<PARALLEL_ROWS>(it_min), 0));
        aie::vector<bf16, PARALLEL_ROWS> a_mins = m_acc.template to_vector<bf16>();
        acc_m = aie::mac(acc_m, a_mins, reduce[2 * g + 1]);
        acc_m = aie::mac(acc_m, a_mins, reduce[2 * g]);
        it_min += PARALLEL_ROWS;
    }
    aie::vector<bf16, PARALLEL_ROWS> a_M = aie::load_v<PARALLEL_ROWS>(w_buffer->M);
    c_accum_m = bf16_limb::mac_f32_bf16_low(c_accum_m, acc_m.template to_vector<float>(), a_M);
}

/// @brief y += W x for one Q4_K block, as _Q4K_x_bf16, with x given as the
/// A matrices and group scales of q4k_prepare_x.
void _Q4K_x_int8(
    q4k_accum_t & c_accum_s,
    q4k_accum_t & c_accum_m,
    const int8* a_tiles,
    const bf16* e64,
    q4k_block_t* w_buffer,
    bf16* b_col_reduce_buffer
) {
    static_assert(PARALLEL_ROWS == 32 && QXNX_COL_BLOCK_SIZE == 256);
    const uint4* q = w_buffer->qs;

    // ---- group sums: acc[t][h], row a = group 4 h + a, 16 output rows each ----
    aie::accum<acc32, 64> acc[2][2];
    q4k_group_sums<0>(acc[0][0], acc[1][0], a_tiles, q);
    q4k_group_sums<1>(acc[0][1], acc[1][1], a_tiles, q);

    // ---- scale path: per row sum_g s'_g 2^e_g G_g, then times S' ----
    // scales layout [h][t][a][16 rows]: (h, t) at (h * 2 + t) * 64
    aie::vector<float, PARALLEL_ROWS> row_sums;
    row_sums.insert(0, q4k_scale_rows(acc[0][0], acc[0][1], w_buffer->scales + 0 * 64, w_buffer->scales + 2 * 64, e64));
    row_sums.insert(1, q4k_scale_rows(acc[1][0], acc[1][1], w_buffer->scales + 1 * 64, w_buffer->scales + 3 * 64, e64));
    aie::vector<bf16, PARALLEL_ROWS> a_S = aie::load_v<PARALLEL_ROWS>(w_buffer->S);
    c_accum_s = bf16_limb::mac_f32_bf16_low(c_accum_s, row_sums, a_S);

    // ---- min path ----
    q4k_min_path(c_accum_m, w_buffer, b_col_reduce_buffer);
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