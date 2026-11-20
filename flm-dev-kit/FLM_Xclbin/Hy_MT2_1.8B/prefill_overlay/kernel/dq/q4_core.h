// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#ifndef __Q4_CORE_H__
#define __Q4_CORE_H__
#include "aie_kernel_utils.h"
#include "common.hpp"

// 0x43 as the high byte of a bf16 puts a nibble v at 128 + v (ulp 1 in [128, 256)),
// so nibble -> bf16 is two byte shuffles; the extra 128 is taken out in `base`.
static constexpr uint8 Q4_BIASED_EXP_BYTE = 0x43;
static constexpr int   NIBBLE_BIAS        = 128;

template <unsigned N>
static inline __attribute__((always_inline)) aie::vector<bf16, N> _q4_biased_bf16(
    aie::vector<uint8, N> nibble_bytes,
    aie::vector<uint8, N> bias_bytes
) {
    return aie::vector_cast<bf16>(
        aie::concat(aie::interleave_zip(nibble_bytes, bias_bytes, 1)));
}

/// \brief Dequantise one K-half of a q4_k block into the shared bf16 B-tile buffer.
///
/// The block covers 32 output rows x 256 input columns in 4736 B, column major:
///   [0, 256)      uint8 s[8 groups][32 rows]
///   [256, 512)    uint8 m[8 groups][32 rows]
///   [512, 4608)   nibbles: row r of column c is nibble c * 32 + r (even r low), unsigned
///   [4608, 4672)  bf16 S[32 rows]
///   [4672, 4736)  bf16 M[32 rows], stored negated
/// w[r][c] = S[r] s[c / 32][r] q + M[r] m[c / 32][r].
///
/// `k_half` 0 takes input columns 0-127 (groups 0-3), 1 takes 128-255; each core
/// writes all 32 output rows. `w16_buffer` layout as before: element (row, col) at
/// (row/8)*8*256 + (col/8)*64 + (col%8)*8 + (row%8).
void _Q4_K_bf16(
    bf16* w16_buffer,
    uint8* qw_buffer,
    int k_half
) {
    static constexpr int GROUPS        = QXNX_COL_BLOCK_SIZE / GGML_GROUP_SIZE;   // 8
    static constexpr int HALF_COLS     = QXNX_COL_BLOCK_SIZE / 2;                 // 128 columns
    static constexpr int SUB_CHUNKS    = HALF_COLS / GGML_GROUP_SIZE;             // 4 groups
    static constexpr int COL_BYTES     = QXNX_ROW_BLOCK_SIZE / 2;                 // 16 B per column
    static constexpr int META_BYTES    = QXNX_ROW_BLOCK_SIZE * GROUPS;            // 256 B of s, of m
    static constexpr int QS_BYTES      = QXNX_COL_BLOCK_SIZE * COL_BYTES;         // 4096 B
    static constexpr int LOAD          = 128;                                     // nibbles per load
    static constexpr int COLS_PER_LOAD = LOAD / QXNX_ROW_BLOCK_SIZE;              // 4 columns
    static constexpr int PAIRS         = GGML_GROUP_SIZE / (2 * COLS_PER_LOAD);   // loads per group / 2
    static constexpr int TILE          = 64;
    static_assert(PARALLEL_ROWS == QXNX_ROW_BLOCK_SIZE);

    const uint8* it_scale = qw_buffer + k_half * SUB_CHUNKS * QXNX_ROW_BLOCK_SIZE;
    const uint8* it_min   = it_scale + META_BYTES;
    uint4* it_quant = (uint4*)(qw_buffer + 2 * META_BYTES + k_half * HALF_COLS * COL_BYTES);
    const bf16* p_S = (const bf16*)(qw_buffer + 2 * META_BYTES + QS_BYTES);
    const bf16* p_M = p_S + QXNX_ROW_BLOCK_SIZE;

    // a load is 4 columns x 32 rows, lane = column * 32 + row: two copies of the
    // per-row values cover each 64-lane half
    const aie::vector<bf16, PARALLEL_ROWS> a_S = aie::load_v<PARALLEL_ROWS>(p_S);
    const aie::vector<bf16, PARALLEL_ROWS> a_M = aie::load_v<PARALLEL_ROWS>(p_M);
    const aie::vector<bf16, TILE> S_tile = aie::concat(a_S, a_S);
    const aie::vector<bf16, TILE> M_tile = aie::concat(a_M, a_M);
    const aie::vector<uint8, TILE> bias_bytes = aie::broadcast<uint8, TILE>(Q4_BIASED_EXP_BYTE);

    bf16* w_ptr_0 = w16_buffer + k_half * HALF_COLS * 8;
    bf16* w_ptr_1 = w_ptr_0 + 8 * QXNX_COL_BLOCK_SIZE;
    bf16* w_ptr_2 = w_ptr_1 + 8 * QXNX_COL_BLOCK_SIZE;
    bf16* w_ptr_3 = w_ptr_2 + 8 * QXNX_COL_BLOCK_SIZE;

    auto u8_bf16 = [](const uint8* p) {
        aie::accum<accfloat, PARALLEL_ROWS> a;
        a.from_vector(aie::to_float(aie::load_v<PARALLEL_ROWS>(p), 0));
        return a.template to_vector<bf16>();
    };

    for (int sub_chunk = 0; sub_chunk < SUB_CHUNKS; sub_chunk++) {
        const aie::vector<bf16, PARALLEL_ROWS> s32 = u8_bf16(it_scale + sub_chunk * QXNX_ROW_BLOCK_SIZE);
        const aie::vector<bf16, PARALLEL_ROWS> m32 = u8_bf16(it_min + sub_chunk * QXNX_ROW_BLOCK_SIZE);

        // S s rounded to bf16 (the MAC operand); M m stays exact in the accumulator
        const aie::vector<bf16, TILE> scale_tile =
            aie::mul(aie::concat(s32, s32), S_tile).template to_vector<bf16>();
        aie::accum<accfloat, TILE> base = aie::mul(aie::concat(m32, m32), M_tile);
        base = aie::msc(base, scale_tile, bfloat16(NIBBLE_BIAS));

        // w = S s (128 + q) + (M m - 128 S s)
        auto dequant = [&](const aie::vector<uint4, LOAD>& q) {
            const aie::vector<uint8, LOAD> q_bytes = aie::unpack(q);
            aie::vector<bf16, LOAD> d;
            d.insert(0, aie::mac(base, _q4_biased_bf16(q_bytes.template extract<TILE>(0), bias_bytes), scale_tile).template to_vector<bf16>());
            d.insert(1, aie::mac(base, _q4_biased_bf16(q_bytes.template extract<TILE>(1), bias_bytes), scale_tile).template to_vector<bf16>());
            return d;
        };
        auto emit_group = [&](const aie::vector<bf16, LOAD>& h, bf16* p_lo, bf16* p_hi) {
            auto rows = aie::interleave_unzip(h.extract<TILE>(0), h.extract<TILE>(1), 8);
            aie::store_v(p_lo, rows.first);
            aie::store_v(p_hi, rows.second);
        };

        AIE_PREPARE_FOR_PIPELINING
        AIE_LOOP_RANGE( PAIRS, PAIRS )
        for (int pair = 0; pair < PAIRS; pair++) {
            aie::vector<uint4, LOAD> qa = aie::load_v<LOAD>(it_quant);
            it_quant += LOAD / 2;
            aie::vector<uint4, LOAD> qb = aie::load_v<LOAD>(it_quant);
            it_quant += LOAD / 2;
            auto halves = aie::interleave_unzip(dequant(qa), dequant(qb), QXNX_ROW_BLOCK_SIZE / 2);
            emit_group(halves.first,  w_ptr_0, w_ptr_1);
            emit_group(halves.second, w_ptr_2, w_ptr_3);
            w_ptr_0 += TILE;
            w_ptr_1 += TILE;
            w_ptr_2 += TILE;
            w_ptr_3 += TILE;
        }
    }
}

#endif // __Q4_CORE_H__
