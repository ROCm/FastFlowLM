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

/// lane j * 8 + r of the result is v[j]: one per-row value spread over the 8 k lanes of its tile row
static inline __attribute__((always_inline)) aie::vector<bf16, 64> _splat8(aie::vector<bf16, 8> v) {
    aie::vector<bf16, 16> v16 = aie::concat(v, v);
    aie::vector<bf16, 32> v32 = aie::concat(v16, v16);
    return aie::transpose(aie::concat(v32, v32), 8, 8);
}

/// the 8 lanes of v repeated 8 times: lane g * 8 + r is v[r]
static inline __attribute__((always_inline)) aie::vector<bf16, 64> _rep8(aie::vector<bf16, 8> v) {
    aie::vector<bf16, 16> v16 = aie::concat(v, v);
    aie::vector<bf16, 32> v32 = aie::concat(v16, v16);
    return aie::concat(v32, v32);
}

/// @brief Dequantize half of one Q4_K weight block into bfp16 mmul B tiles (n major, ready to multiply).
///
/// Format (see FLM_Q4NX_Converter/quant.md): a super-block is 256 weights of one row -- here exactly one block
/// row -- split into 8 groups of GGML_GROUP_SIZE. Per group it stores a uint8 scale s' and a uint8 min m'; per
/// super-block a bf16 S' and a bf16 M' (stored negated), so w = S' s'_j q + M' m'_j.
///
/// The two cores of a row pair split the block by row: H = 0 takes rows 0-15, H = 1 rows 16-31, each as its
/// 8 even rows (par 0) and 8 odd rows (par 1), row = 16 H + 2 r + par. The host stores every block in that
/// order at load time (q4k_to_npu), so a core's data is contiguous and already n major:
///     scales  uint8 [H 2][par 2][group 8][r 8]
///     mins    uint8 [H 2][par 2][group 8][r 8]
///     qs      uint4 [H 2][par 2][kt 32][r 8][k 8]     (column 8 kt + k, low nibble first)
///     S       bf16  [H 2][par 2][r 8]
///     M       bf16  [H 2][par 2][r 8]
/// One vector op scales all 8 groups of a par, and a 64 byte load is two 8 n x 8 k tiles: no permutes. The
/// multiply loop does not care which rows a tile holds; the epilogue (unpermute_chunk) restores the row order.
///
/// The bf16 operand is 128 + q by byte placement (_q4_biased_bf16), so w = S' s'_j (128 + q) + base with the
/// base taken once per group; the fp32 result is rounded once, by the bfp16 conversion of the store.
///
/// The two cores of a row pair split each half block by k: KH = 0 takes groups 0-3 (kt 0-15), KH = 1 groups 4-7
/// (kt 16-31), so the region tiles of KH start at tile 32 KH and its nibbles at 512 KH bytes of each par.
///
/// @param w_region    output tiles, region[kt * 2 + par] (the whole region; this call writes its k half)
/// @param qw_buffer   the packed block
/// @param H           the half of the block (rows 16 H .. 16 H + 15)
template <int KH>
void _Q4_K_bfp16_kh(bfp16ebs8* __restrict w_region, const uint8* __restrict qw_buffer, int H) {
    static constexpr int GROUPS_PER_ROW = QXNX_COL_BLOCK_SIZE / GGML_GROUP_SIZE;
    static constexpr int SCALES_BYTES = QXNX_ROW_BLOCK_SIZE * GROUPS_PER_ROW;
    static constexpr int QS_BYTES = QXNX_ROW_BLOCK_SIZE * QXNX_COL_BLOCK_SIZE / 2;
    static constexpr int CORE_ROWS = 8;                            // per par
    static constexpr int PAR_SCALES = GROUPS_PER_ROW * CORE_ROWS;  // 64
    static constexpr int MM_BLK = 8;
    static constexpr int TILES_PER_GROUP = GGML_GROUP_SIZE / MM_BLK;
    static constexpr int PAR_QS = QS_BYTES / 4;                    // one par's nibbles of one half
    static_assert(QXNX_COL_BLOCK_SIZE == 256 && QXNX_ROW_BLOCK_SIZE == 32 && PAR_SCALES == 64);

    const uint8* sc = qw_buffer + 2 * PAR_SCALES * H;
    const uint8* mn = qw_buffer + SCALES_BYTES + 2 * PAR_SCALES * H;
    static constexpr int G0 = KH * GROUPS_PER_ROW / 2;               // first group of this k half
    const uint8* q = qw_buffer + 2 * SCALES_BYTES + QS_BYTES / 2 * H + G0 * TILES_PER_GROUP / 2 * 64;
    const bf16* S = (const bf16*)(qw_buffer + 2 * SCALES_BYTES + QS_BYTES) + 2 * CORE_ROWS * H;
    const bf16* M = S + QXNX_ROW_BLOCK_SIZE;
    const aie::vector<uint8, 64> bias_bytes = aie::broadcast<uint8, 64>(Q4_BIASED_EXP_BYTE);

    // per par: uint8 -> bf16 is exact for every value up to 255; lane g * 8 + r
    aie::accum<accfloat, 64> t;
    t.from_vector(aie::to_float(aie::load_v<64>(sc), 0));
    const aie::vector<bf16, 64> a_sc0 = aie::mul(t.template to_vector<bf16>(), _rep8(aie::load_v<8>(S))).template to_vector<bf16>();
    t.from_vector(aie::to_float(aie::load_v<64>(sc + PAR_SCALES), 0));
    const aie::vector<bf16, 64> a_sc1 = aie::mul(t.template to_vector<bf16>(), _rep8(aie::load_v<8>(S + CORE_ROWS))).template to_vector<bf16>();
    t.from_vector(aie::to_float(aie::load_v<64>(mn), 0));
    const aie::vector<bf16, 64> a_mn0 = aie::mul(t.template to_vector<bf16>(), _rep8(aie::load_v<8>(M))).template to_vector<bf16>();
    t.from_vector(aie::to_float(aie::load_v<64>(mn + PAR_SCALES), 0));
    const aie::vector<bf16, 64> a_mn1 = aie::mul(t.template to_vector<bf16>(), _rep8(aie::load_v<8>(M + CORE_ROWS))).template to_vector<bf16>();
    aie::block_vector_output_buffer_stream<bfp16ebs8, 64> w_out(
        (bfp16ebs8*)((uint8*)w_region + G0 * TILES_PER_GROUP * 2 * (MM_BLK * MM_BLK + MM_BLK)));

    AIE_LOOP_RANGE(GROUPS_PER_ROW / 2, GROUPS_PER_ROW / 2)
    for (int g = G0; g < G0 + GROUPS_PER_ROW / 2; g++) {
        const aie::vector<bf16, 64> s0 = _splat8(aie::shuffle_down(a_sc0, CORE_ROWS * g).template extract<8>(0));
        const aie::vector<bf16, 64> s1 = _splat8(aie::shuffle_down(a_sc1, CORE_ROWS * g).template extract<8>(0));
        aie::accum<accfloat, 64> b0, b1;
        b0.from_vector(_splat8(aie::shuffle_down(a_mn0, CORE_ROWS * g).template extract<8>(0)));
        b1.from_vector(_splat8(aie::shuffle_down(a_mn1, CORE_ROWS * g).template extract<8>(0)));
        // the MAC operand is 128 + q (see _q4_biased_bf16): take the 128 S' s'_j back out of the base. Both
        // operands are bf16 exact and 128 is a power of two, so this is exact.
        b0 = aie::msc(b0, s0, bfloat16(NIBBLE_BIAS));
        b1 = aie::msc(b1, s1, bfloat16(NIBBLE_BIAS));
#pragma clang loop unroll(full)
        for (int kt = 0; kt < TILES_PER_GROUP; kt += 2) {
            // per par two tiles [r 8][k 8] of nibbles, columns 8 kt .. 8 kt + 15
            aie::vector<uint8, 128> u0 = aie::unpack(aie::vector_cast<uint4>(aie::load_v<64>(q)));
            aie::vector<uint8, 128> u1 = aie::unpack(aie::vector_cast<uint4>(aie::load_v<64>(q + PAR_QS)));
            q += 64;
            w_out << ::to_v64bfp16ebs8(aie::mac(b0, _q4_biased_bf16(u0.template extract<64>(0), bias_bytes), s0));
            w_out << ::to_v64bfp16ebs8(aie::mac(b1, _q4_biased_bf16(u1.template extract<64>(0), bias_bytes), s1));
            w_out << ::to_v64bfp16ebs8(aie::mac(b0, _q4_biased_bf16(u0.template extract<64>(1), bias_bytes), s0));
            w_out << ::to_v64bfp16ebs8(aie::mac(b1, _q4_biased_bf16(u1.template extract<64>(1), bias_bytes), s1));
        }
    }
}

#endif // __Q4_CORE_H__
