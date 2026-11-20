// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#ifndef __Q4_CORE_H__
#define __Q4_CORE_H__
#include "aie_kernel_utils.h"
#include "common.hpp"

/// bf16 has an 8 bit exponent and a 7 bit mantissa, so the [128, 256) binade has
/// an ulp of exactly 1: the bit pattern 0x4300 | v *is* the number 128 + v, for
/// every v in [0, 15].  A uint4 -> bf16 conversion is therefore just byte
/// placement -- unpack the nibbles to bytes and zip 0x43 in as the high byte --
/// instead of aie::to_float's convert chain through the accumulator files.  The
/// weights come out biased by 128, which the caller pays back for free.
static constexpr uint8 Q4_BIASED_EXP_BYTE = 0x43;
static constexpr int   NIBBLE_BIAS        = 128;

template <unsigned N>
static inline __attribute__((always_inline)) aie::vector<bf16, N> _q4_biased_bf16(
    aie::vector<uint8, N> nibble_bytes,
    aie::vector<uint8, N> bias_bytes
) {
    // little endian: low byte is the nibble, high byte the 0x43 exponent field
    return aie::vector_cast<bf16>(
        aie::concat(aie::interleave_zip(nibble_bytes, bias_bytes, 1)));
}

/// @brief Dequantize one Q4_K weight block into the bf16 mmul B tile.
///
/// Format (see FLM_Q4NX_Converter/quant.md): a super-block is 256 weights of one
/// row -- here exactly one block row -- split into 8 groups of GGML_GROUP_SIZE.
/// Per group it stores a uint8 scale s' and a uint8 min m'; per super-block a
/// bf16 S' and a bf16 M'. The converter stores M' already negated, so the
/// dequantized weight is
///
///     w = S' s'_j q  +  M' m'_j
///
/// The block is laid out in q4k_block_t field order, everything column major over
/// the QXNX_ROW_BLOCK_SIZE x QXNX_COL_BLOCK_SIZE block:
///     scales  uint8 [QXNX_COL_BLOCK_SIZE / GGML_GROUP_SIZE][QXNX_ROW_BLOCK_SIZE]
///     mins    uint8 [QXNX_COL_BLOCK_SIZE / GGML_GROUP_SIZE][QXNX_ROW_BLOCK_SIZE]
///     qs      uint4 [QXNX_COL_BLOCK_SIZE][QXNX_ROW_BLOCK_SIZE]
///     S       bf16  [QXNX_ROW_BLOCK_SIZE]
///     M       bf16  [QXNX_ROW_BLOCK_SIZE]
///
/// The whole row span of a column is packed together (PARALLEL_ROWS ==
/// QXNX_ROW_BLOCK_SIZE): one packed column is PARALLEL_ROWS nibbles, so row r of
/// a column sits at nibble r and one column step is PARALLEL_ROWS / 2 bytes.
///
/// The two cores of a row pair split the block by row -- `row` selects the half.
/// They still read the *same* 128-nibble chunks and separate their halves with
/// the first unzip below rather than with a pointer offset: a half-column offset
/// is PARALLEL_ROWS / 4 bytes, which would misalign every 512 bit load.
///
/// S' and M' are constant over the whole call, so the per-group products
/// S' s'_j and M' m'_j are formed once per group and everything downstream is
/// the same shape as the old Q4_1 path.
///
/// @brief One 8 x 8 B tile, k major, as the bfp16 tile mac_8x8_8x8T multiplies.
static inline __attribute__((always_inline)) aie::block_vector<bfp16ebs8, 64> _bfp16_tile_t(
    aie::vector<bf16, 64> tile
) {
    aie::accum<accfloat, 64> acc(aie::transpose(tile, 8, 8));
    return acc.template to_vector<bfp16ebs8>();
}

/// @param w_bfp              bfp16 output, transposed B tiles in order [n / 16][k / 8][n / 8 % 2]
/// @param qw_buffer          the packed block
/// @tparam ROW               first row this core owns (0 or 16); it writes CORE_ROWS of them. A template
///                           parameter so the half choice in load_cols is not a branch inside the loop.
/// @param first_group        first GGML group (32 k) to dequantize
/// @param num_groups         how many groups
/// @param first_n_block      n block of the block's row 0 in w_bfp (even)
/// @param tiles_per_n_block  k tiles of one n block
template <int ROW>
void _Q4_K_bfp16(
    bfp16ebs8* w_bfp,
    uint8* qw_buffer,
    int first_group,
    int num_groups,
    int first_n_block,
    int tiles_per_n_block
) {
    static constexpr int GROUPS_PER_ROW   = QXNX_COL_BLOCK_SIZE / GGML_GROUP_SIZE;
    static constexpr int SCALES_BYTES     = QXNX_ROW_BLOCK_SIZE * GROUPS_PER_ROW;
    static constexpr int MINS_BYTES       = SCALES_BYTES;
    static constexpr int QS_BYTES         = QXNX_ROW_BLOCK_SIZE * QXNX_COL_BLOCK_SIZE / 2;

    // the row pair splits the block in half, so a core owns this many rows
    static constexpr int CORE_ROWS        = QXNX_ROW_BLOCK_SIZE / 2;
    // aie::mmul<8, 8, 8> B tile: 8 k rows x 8 n columns, n fastest
    static constexpr int MM_BLK           = 8;
    static constexpr int LANES            = MM_BLK * MM_BLK;
    // one 512 bit load is 128 nibbles, i.e. this many whole columns
    static constexpr int NIBBLES_PER_LOAD = 128;
    static constexpr int COLS_PER_LOAD    = NIBBLES_PER_LOAD / QXNX_ROW_BLOCK_SIZE;
    static constexpr int LOADS_PER_TILE   = MM_BLK / COLS_PER_LOAD;
    static constexpr int TILES_PER_GROUP  = GGML_GROUP_SIZE / MM_BLK;
    static constexpr int QS_GROUP_BYTES   = QXNX_ROW_BLOCK_SIZE * GGML_GROUP_SIZE / 2;

    // a block is exactly one super-block, so there is one S'/M' per row
    static_assert(QXNX_COL_BLOCK_SIZE == 256);
    static_assert(PARALLEL_ROWS == QXNX_ROW_BLOCK_SIZE);
    static_assert(NIBBLES_PER_LOAD % QXNX_ROW_BLOCK_SIZE == 0);
    // the core's two output n blocks go out as one pair stream, lo then hi tile
    static_assert(CORE_ROWS == 2 * MM_BLK);
    static_assert(LOADS_PER_TILE == 2);

    uint8* it_scale = qw_buffer;
    uint8* it_min   = qw_buffer + SCALES_BYTES;
    uint4* it_quant = (uint4*)(qw_buffer + SCALES_BYTES + MINS_BYTES + first_group * QS_GROUP_BYTES);
    bf16*  p_S      = (bf16*)(qw_buffer + SCALES_BYTES + MINS_BYTES + QS_BYTES);
    bf16*  p_M      = p_S + QXNX_ROW_BLOCK_SIZE;

    constexpr int row = ROW;
    constexpr bool upper_half = (row >= CORE_ROWS);

    aie::vector<bf16, CORE_ROWS> a_S = aie::load_v<CORE_ROWS>(p_S + row);
    aie::vector<bf16, CORE_ROWS> a_M = aie::load_v<CORE_ROWS>(p_M + row);

    aie::block_vector_output_buffer_stream<bfp16ebs8, LANES> out(w_bfp);
    out.seek((first_n_block + row / MM_BLK) * tiles_per_n_block);

    const aie::vector<uint8, LANES / 2> bias_bytes =
        aie::broadcast<uint8, LANES / 2>(Q4_BIASED_EXP_BYTE);

    // One aligned load, unpacked into this core's two output n blocks.  The
    // first unzip separates the row pair's two halves of every column, the
    // second splits this core's half into its low and high n block.
    auto load_cols = [&]() {
        aie::vector<uint4, NIBBLES_PER_LOAD> q_raw = aie::load_v<NIBBLES_PER_LOAD>(it_quant);
        it_quant += NIBBLES_PER_LOAD / 2; // uint4* steps two nibbles

        // Both unzips are element permutations, so run them in the byte domain
        // and only widen to bf16 at the very end: on uint8 the same shuffle
        // moves half as many bits, so it costs half as many ops.
        aie::vector<uint8, NIBBLES_PER_LOAD> q_bytes = aie::unpack(q_raw);
        auto halves = aie::interleave_unzip(q_bytes.template extract<NIBBLES_PER_LOAD / 2>(0),
                                            q_bytes.template extract<NIBBLES_PER_LOAD / 2>(1),
                                            CORE_ROWS);
        aie::vector<uint8, NIBBLES_PER_LOAD / 2> mine = upper_half ? halves.second : halves.first;
        auto blks = aie::interleave_unzip(mine.template extract<LANES / 2>(0),
                                          mine.template extract<LANES / 2>(1),
                                          MM_BLK);
        return std::make_pair(_q4_biased_bf16(blks.first,  bias_bytes),
                              _q4_biased_bf16(blks.second, bias_bytes));
    };

    for (int sub_chunk = first_group; sub_chunk < first_group + num_groups; sub_chunk++) {
        // uint8 -> bf16 is exact for every value up to 255
        aie::accum<accfloat, CORE_ROWS> s_acc;
        s_acc.from_vector(aie::to_float(
            aie::load_v<CORE_ROWS>(it_scale + sub_chunk * QXNX_ROW_BLOCK_SIZE + row), 0));
        aie::accum<accfloat, CORE_ROWS> m_acc;
        m_acc.from_vector(aie::to_float(
            aie::load_v<CORE_ROWS>(it_min + sub_chunk * QXNX_ROW_BLOCK_SIZE + row), 0));

        // the effective per-group scale and min: S' s'_j and M' m'_j
        aie::vector<bf16, CORE_ROWS> a_scales =
            aie::mul(s_acc.template to_vector<bf16>(), a_S).template to_vector<bf16>();
        aie::vector<bf16, CORE_ROWS> a_mins =
            aie::mul(m_acc.template to_vector<bf16>(), a_M).template to_vector<bf16>();

        // lane n of an output tile is row (row + n % 8), so each half of the
        // metadata is splatted MM_BLK times across the tile
        auto splat_rows = [](aie::vector<bf16, CORE_ROWS> v) {
            aie::vector<bf16, 2 * CORE_ROWS> v32 = aie::concat(v, v);
            aie::vector<bf16, LANES> v64 = aie::concat(v32, v32);
            return aie::interleave_unzip(v64, v64, MM_BLK);
        };
        auto scale_vec = splat_rows(a_scales);

        auto min = splat_rows(a_mins);
        aie::accum<accfloat, LANES> base_lo;
        aie::accum<accfloat, LANES> base_hi;
        base_lo.from_vector(min.first);
        base_hi.from_vector(min.second);
        // the MAC operand is 128 + q, so every lane carries an extra
        // 128 S' s'_j; subtract it once per group, out of the tile loop.  Both
        // operands are bf16 exact and 128 is a power of two, so this is exact.
        base_lo = aie::msc(base_lo, scale_vec.first,  bfloat16(NIBBLE_BIAS));
        base_hi = aie::msc(base_hi, scale_vec.second, bfloat16(NIBBLE_BIAS));
        const aie::vector<bf16, LANES>& scale_lo = scale_vec.first;
        const aie::vector<bf16, LANES>& scale_hi = scale_vec.second;

        AIE_PREPARE_FOR_PIPELINING
        AIE_LOOP_RANGE( TILES_PER_GROUP, TILES_PER_GROUP )
        for (int tile = 0; tile < TILES_PER_GROUP; tile++) {
            // columns 0..3 then 4..7 of this k block, so the concat stays k major
            auto qa = load_cols();
            auto qb = load_cols();
            aie::vector<bf16, LANES> lo = aie::concat(qa.first,  qb.first);
            aie::vector<bf16, LANES> hi = aie::concat(qa.second, qb.second);

            // w = S' s'_j (128 + q) + (M' m'_j - 128 S' s'_j)
            out << _bfp16_tile_t(aie::mac(base_lo, lo, scale_lo).to_vector<bf16>());
            out << _bfp16_tile_t(aie::mac(base_hi, hi, scale_hi).to_vector<bf16>());
        }
    } // end of sub_chunk
}

#endif // __Q4_CORE_H__
