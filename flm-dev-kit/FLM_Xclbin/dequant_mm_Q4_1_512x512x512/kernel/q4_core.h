// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

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

/// @brief Dequantize one Q4_1 weight block into the bf16 mmul B tile.
///
/// Format (see FLM_Q4NX_Converter's _pack_q4nx and q41_block_t in
/// generic_decoding_layer/kernels/model_spec.h): a block is
/// QXNX_ROW_BLOCK_SIZE x QXNX_COL_BLOCK_SIZE weights carrying, per row and per
/// group of GGML_GROUP_SIZE columns, a *stored bf16* scale d and a stored bf16
/// min m.  The dequantized weight is
///
///     w = d q + m
///
/// which is the same shape as Q4_K's w = S' s'_j q + M' m'_j, one level
/// shallower: there is no super-block S'/M' and no uint8 -> bf16 widening, so
/// both multiplicands come straight off the block.  The min is *added* -- llama
/// stores Q4_1's m already signed, unlike Q4_K's magnitude-and-subtract.
///
/// The block is laid out in q41_block_t field order:
///     scales  bf16  [QXNX_COL_BLOCK_SIZE / GGML_GROUP_SIZE][QXNX_ROW_BLOCK_SIZE]
///     mins    bf16  [QXNX_COL_BLOCK_SIZE / GGML_GROUP_SIZE][QXNX_ROW_BLOCK_SIZE]
///     qs      uint4 [QXNX_ROW_BLOCK_SIZE / PARALLEL_ROWS]
///                   [QXNX_COL_BLOCK_SIZE][PARALLEL_ROWS]
///
/// Scales and mins are group major over the whole row block -- indexed
/// `scales + row` and stepped QXNX_ROW_BLOCK_SIZE per group -- so only the quant
/// plane is strided.  A packed column is PARALLEL_ROWS nibbles of *one* plane,
/// so the two cores of a row pair own one plane each and read disjoint,
/// contiguous, 512 bit aligned halves of qs.  `row` selects the plane, and one
/// 512 bit load is then exactly 8 whole columns: the single unzip below splits
/// those 8 columns' 16 rows into this core's two output n blocks, and no second
/// unzip is needed to separate the row pair (contrast Q4_K, which packs plain
/// column major and makes both cores read the same nibbles).
///
/// @param w16_buffer  bf16 output, mmul B tile order [n / 8][k / 8][k % 8][n % 8]
/// @param qw_buffer   the packed block
/// @param row         first row this core owns; it writes CORE_ROWS of them
void _Q4_1_bf16(
    bf16* w16_buffer,
    uint8* qw_buffer,
    int row
) {
    static constexpr int GROUPS_PER_ROW   = QXNX_COL_BLOCK_SIZE / GGML_GROUP_SIZE;
    static constexpr int SCALES_BYTES     = QXNX_ROW_BLOCK_SIZE * GROUPS_PER_ROW * sizeof(bf16);
    static constexpr int MINS_BYTES       = SCALES_BYTES;
    // one PARALLEL_ROWS-row plane of the quant field, in bytes
    static constexpr int PLANE_BYTES      = PARALLEL_ROWS * QXNX_COL_BLOCK_SIZE / 2;

    // the row pair splits the block in half, so a core owns this many rows
    static constexpr int CORE_ROWS        = QXNX_ROW_BLOCK_SIZE / 2;
    // aie::mmul<8, 8, 8> B tile: 8 k rows x 8 n columns, n fastest
    static constexpr int MM_BLK           = 8;
    static constexpr int LANES            = MM_BLK * MM_BLK;
    // one 512 bit load is 128 nibbles, i.e. this many whole columns
    static constexpr int NIBBLES_PER_LOAD = 128;
    static constexpr int COLS_PER_LOAD    = NIBBLES_PER_LOAD / PARALLEL_ROWS;
    static constexpr int TILES_PER_GROUP  = GGML_GROUP_SIZE / MM_BLK;

    // a core owns exactly one plane, so the two halves never share a load
    static_assert(PARALLEL_ROWS == CORE_ROWS);
    // ... and one load is exactly one k block, filling both output n blocks
    static_assert(COLS_PER_LOAD == MM_BLK);
    // the two w_ptr_iter below are the core's two output n blocks
    static_assert(CORE_ROWS == 2 * MM_BLK);

    bf16*  p_scale  = (bf16*)qw_buffer;
    bf16*  p_min    = (bf16*)(qw_buffer + SCALES_BYTES);
    uint4* it_quant = (uint4*)(qw_buffer + SCALES_BYTES + MINS_BYTES
                               + (row / PARALLEL_ROWS) * PLANE_BYTES);

    bf16* w_ptr_iter_0 = w16_buffer + row * QXNX_COL_BLOCK_SIZE;
    bf16* w_ptr_iter_1 = w16_buffer + (row + MM_BLK) * QXNX_COL_BLOCK_SIZE;

    const aie::vector<uint8, LANES> bias_bytes =
        aie::broadcast<uint8, LANES>(Q4_BIASED_EXP_BYTE);

    for (int sub_chunk = 0; sub_chunk < GROUPS_PER_ROW; sub_chunk++) {
        // d and m are stored bf16, so both are exact as loaded -- the only
        // rounding in the whole block is the mac's accumulator on its way out.
        aie::vector<bf16, CORE_ROWS> a_scales =
            aie::load_v<CORE_ROWS>(p_scale + sub_chunk * QXNX_ROW_BLOCK_SIZE + row);
        aie::vector<bf16, CORE_ROWS> a_mins =
            aie::load_v<CORE_ROWS>(p_min + sub_chunk * QXNX_ROW_BLOCK_SIZE + row);

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
        // the MAC operand is 128 + q, so every lane carries an extra 128 d;
        // subtract it once per group, out of the tile loop.  Both operands are
        // bf16 exact and 128 is a power of two, so this is exact.
        base_lo = aie::msc(base_lo, scale_vec.first,  bfloat16(NIBBLE_BIAS));
        base_hi = aie::msc(base_hi, scale_vec.second, bfloat16(NIBBLE_BIAS));
        const aie::vector<bf16, LANES>& scale_lo = scale_vec.first;
        const aie::vector<bf16, LANES>& scale_hi = scale_vec.second;

        AIE_PREPARE_FOR_PIPELINING
        AIE_LOOP_RANGE( TILES_PER_GROUP, TILES_PER_GROUP )
        for (int tile = 0; tile < TILES_PER_GROUP; tile++) {
            // One aligned load is COLS_PER_LOAD == MM_BLK whole columns of this
            // core's plane, lane c * PARALLEL_ROWS + p holding row p of column
            // c.  Unzipping at MM_BLK splits every column's PARALLEL_ROWS rows
            // into the low and the high output n block, and each side comes out
            // k major with n fastest -- a complete B tile.
            aie::vector<uint4, NIBBLES_PER_LOAD> q_raw = aie::load_v<NIBBLES_PER_LOAD>(it_quant);
            it_quant += NIBBLES_PER_LOAD / 2; // uint4* steps two nibbles

            // The unzip is an element permutation, so run it in the byte domain
            // and only widen to bf16 at the very end: on uint8 the same shuffle
            // moves half as many bits, so it costs half as many ops.
            aie::vector<uint8, NIBBLES_PER_LOAD> q_bytes = aie::unpack(q_raw);
            auto rows = aie::interleave_unzip(q_bytes.template extract<LANES>(0),
                                              q_bytes.template extract<LANES>(1),
                                              MM_BLK);
            aie::vector<bf16, LANES> lo = _q4_biased_bf16(rows.first,  bias_bytes);
            aie::vector<bf16, LANES> hi = _q4_biased_bf16(rows.second, bias_bytes);

            // w = d (128 + q) + (m - 128 d)
            aie::store_v(w_ptr_iter_0, aie::mac(base_lo, lo, scale_lo).to_vector<bf16>());
            aie::store_v(w_ptr_iter_1, aie::mac(base_hi, hi, scale_hi).to_vector<bf16>());
            w_ptr_iter_0 += LANES;
            w_ptr_iter_1 += LANES;
        }
    } // end of sub_chunk
}

#endif // __Q4_CORE_H__
