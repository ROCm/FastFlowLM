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

// aie::mmul<8, 8, 8> B tile: 8 k rows x 8 n columns, n fastest
static constexpr int MM_BLK    = 8;
static constexpr int LANES     = MM_BLK * MM_BLK;
// the row pair splits a 32-row block in half, so a core owns this many rows
static constexpr int CORE_ROWS = QXNX_ROW_BLOCK_SIZE / 2;
// k tiles in one n block of the output: the whole 256 wide column block
static constexpr int K_TILES   = QXNX_COL_BLOCK_SIZE / MM_BLK;
static constexpr int GROUPS    = QXNX_COL_BLOCK_SIZE / GGML_GROUP_SIZE;
static constexpr int TILES_PER_GROUP = GGML_GROUP_SIZE / MM_BLK;

/// lane n of an output tile is row (n % 8) of an 8-row half, so each half of a core's 16 per-row values is splatted
/// MM_BLK times across a tile: .first serves rows 0..7, .second rows 8..15
static inline __attribute__((always_inline)) auto _splat_rows(aie::vector<bf16, CORE_ROWS> v) {
    aie::vector<bf16, 2 * CORE_ROWS> v32 = aie::concat(v, v);
    aie::vector<bf16, LANES> v64 = aie::concat(v32, v32);
    return aie::interleave_unzip(v64, v64, MM_BLK);
}

/// turns two bf16 B tiles (k major, n fastest) the way the multiply wants them (k fastest) and writes them as block float
static inline __attribute__((always_inline)) void _push_tiles(
    aie::block_vector_output_buffer_stream<bfp16ebs8, LANES>& out,
    aie::vector<bf16, LANES> w_lo,
    aie::vector<bf16, LANES> w_hi
) {
    aie::accum<accfloat, LANES> t_lo;
    aie::accum<accfloat, LANES> t_hi;
    t_lo.from_vector(aie::transpose(w_lo, MM_BLK, MM_BLK));
    t_hi.from_vector(aie::transpose(w_hi, MM_BLK, MM_BLK));
    out.push(t_lo.template to_vector<bfp16ebs8>());
    out.push(t_hi.template to_vector<bfp16ebs8>());
}

/// @brief Dequantize one Q4_1 block of the mixed-precision decode layout into bfp16 B tiles.
///
/// The block is q41_block_t of generic_decoding_layer/kernels/model_spec.h as the mixed models pack it
/// (PARALLEL_ROWS = 32): w = d q + m with
///     scales  bf16  [GROUPS][32 rows]   at +0
///     mins    bf16  [GROUPS][32 rows]   at +512
///     qs      uint4 [256 cols][32 rows] at +1024, plain column major: a column is 16 B, byte j holding rows 2j (low
///                                       nibble) and 2j + 1 (high nibble)
/// A k tile (8 columns) is 128 B.  An 8-byte unzip of its two 64 B halves gives rows 0..15 (.first) and rows 16..31
/// (.second) of the 8 columns, each in the order the 16-row plane of the prefill Q4_1 layout has: from there on the
/// conversion is the one of _Q4_1_bfp16 in dequant_mm_bfp16.
///
/// @tparam H        the half of the block this core owns: 0 = rows 0..15 (even core), 1 = rows 16..31 (odd core)
/// @param w_tiles   bfp16ebs8 output, tiles [32-row block p][half H][k tile][low n block, high n block]
/// @param qw        the block
/// @param p         which 32-row block of the 64-row slab this is (0 or 1)
template <int H>
void _Q4_1_mix_bfp16(bfp16ebs8* w_tiles, const uint8* qw, int p) {
    const bf16*  p_scale = (const bf16*)qw;
    const bf16*  p_min   = (const bf16*)(qw + 512);
    const uint8* p_quant = qw + 1024;

    aie::block_vector_output_buffer_stream<bfp16ebs8, LANES> out(w_tiles);
    out.seek(p * 4 * K_TILES + H * 2 * K_TILES);

    const aie::vector<uint8, LANES> bias_bytes = aie::broadcast<uint8, LANES>(Q4_BIASED_EXP_BYTE);

    for (int g = 0; g < GROUPS; g++) {
        auto scale_vec = _splat_rows(aie::load_v<CORE_ROWS>(p_scale + g * QXNX_ROW_BLOCK_SIZE + H * CORE_ROWS));
        auto min = _splat_rows(aie::load_v<CORE_ROWS>(p_min + g * QXNX_ROW_BLOCK_SIZE + H * CORE_ROWS));
        aie::accum<accfloat, LANES> base_lo;
        aie::accum<accfloat, LANES> base_hi;
        base_lo.from_vector(min.first);
        base_hi.from_vector(min.second);
        // the MAC operand is 128 + q: pay the extra 128 d back once per group (exact, 128 is a power of two)
        base_lo = aie::msc(base_lo, scale_vec.first,  bfloat16(NIBBLE_BIAS));
        base_hi = aie::msc(base_hi, scale_vec.second, bfloat16(NIBBLE_BIAS));

        AIE_PREPARE_FOR_PIPELINING
        AIE_LOOP_RANGE(TILES_PER_GROUP, TILES_PER_GROUP)
        for (int tile = 0; tile < TILES_PER_GROUP; tile++) {
            const uint8* q = p_quant + (g * TILES_PER_GROUP + tile) * 128;
            auto halves = aie::interleave_unzip(aie::load_v<64>(q), aie::load_v<64>(q + 64), 8);
            aie::vector<uint4, 128> q_raw = aie::vector_cast<uint4>(H == 0 ? halves.first : halves.second);
            // lane c * 16 + r is row r of column c: unzip at 8 splits each column's 16 rows into the two n blocks
            aie::vector<uint8, 128> q_bytes = aie::unpack(q_raw);
            auto rows = aie::interleave_unzip(q_bytes.template extract<LANES>(0),
                                              q_bytes.template extract<LANES>(1),
                                              MM_BLK);
            aie::vector<bf16, LANES> lo = _q4_biased_bf16(rows.first,  bias_bytes);
            aie::vector<bf16, LANES> hi = _q4_biased_bf16(rows.second, bias_bytes);
            // w = d (128 + q) + (m - 128 d), rounded to bf16, then block float
            _push_tiles(out, aie::mac(base_lo, lo, scale_vec.first).template to_vector<bf16>(),
                             aie::mac(base_hi, hi, scale_vec.second).template to_vector<bf16>());
        }
    }
}

/// k tiles of a Q8_0 block in the first 5120 B half of its 10240 B unit (512 B of scales, then 32 B per column)
static constexpr int Q8_H0_TILES = (5120 - 512) / (MM_BLK * QXNX_ROW_BLOCK_SIZE);

/// @brief Dequantize one half of a Q8_0 block of the mixed-precision decode layout into bfp16 B tiles.
///
/// The block is q80_block_t (w = d q): scales bf16 [GROUPS][32 rows] (512 B), then qs int8 [256 cols][32 rows] (8 KB),
/// padded to a 10240 B unit.  The DMA hands the unit over as two 5120 B chunks: half 0 holds the scales and k tiles
/// 0..17, half 1 k tiles 18..31 (and the padding).  Half 0 keeps this core's scales in `scales` ([GROUPS][16]) for half 1.
///
/// A k tile is 256 B, 2 columns per 64 B load: a 16-byte unzip of two loads picks this core's 16 rows of 4 columns.
///
/// @tparam H        0 = rows 0..15 (even core), 1 = rows 16..31 (odd core)
/// @param w_tiles   bfp16ebs8 output, as _Q4_1_mix_bfp16
/// @param qw        the chunk
/// @param p         which 32-row block of the 64-row slab this is (0 or 1)
/// @param half      0 or 1, the chunk of the unit
/// @param scales    this core's scales of the block, written by half 0 and read by half 1
template <int H>
void _Q8_0_mix_bfp16(bfp16ebs8* w_tiles, const uint8* qw, int p, int half, bf16* scales) {
    const int t0 = half == 0 ? 0 : Q8_H0_TILES;
    const int t1 = half == 0 ? Q8_H0_TILES : K_TILES;
    const int8* p_quant = (const int8*)(half == 0 ? qw + 512 : qw);
    if (half == 0) {
        for (int g = 0; g < GROUPS; g++)
            aie::store_v(scales + g * CORE_ROWS,
                         aie::load_v<CORE_ROWS>((const bf16*)qw + g * QXNX_ROW_BLOCK_SIZE + H * CORE_ROWS));
    }

    aie::block_vector_output_buffer_stream<bfp16ebs8, LANES> out(w_tiles);
    out.seek(p * 4 * K_TILES + H * 2 * K_TILES + 2 * t0);

    for (int t = t0; t < t1; t++) {
        auto scale_vec = _splat_rows(aie::load_v<CORE_ROWS>(scales + (t / TILES_PER_GROUP) * CORE_ROWS));
        const int8* q = p_quant + (t - t0) * MM_BLK * QXNX_ROW_BLOCK_SIZE;
        auto c03 = aie::interleave_unzip(aie::load_v<64>(q),       aie::load_v<64>(q + 64),  CORE_ROWS);
        auto c47 = aie::interleave_unzip(aie::load_v<64>(q + 128), aie::load_v<64>(q + 192), CORE_ROWS);
        // lane c * 16 + r is row r of column c (columns 0..3, then 4..7): unzip at 8 gives the two n blocks
        auto rows = aie::interleave_unzip(H == 0 ? c03.first : c03.second,
                                          H == 0 ? c47.first : c47.second, MM_BLK);
        // int8 -> bf16 is exact
        aie::accum<accfloat, LANES> q_lo;
        aie::accum<accfloat, LANES> q_hi;
        q_lo.from_vector(aie::to_float(rows.first, 0));
        q_hi.from_vector(aie::to_float(rows.second, 0));
        aie::vector<bf16, LANES> lo = q_lo.template to_vector<bf16>();
        aie::vector<bf16, LANES> hi = q_hi.template to_vector<bf16>();
        // w = d q, rounded to bf16, then block float
        _push_tiles(out, aie::mul(lo, scale_vec.first).template to_vector<bf16>(),
                         aie::mul(hi, scale_vec.second).template to_vector<bf16>());
    }
}

#endif // __Q4_CORE_H__
