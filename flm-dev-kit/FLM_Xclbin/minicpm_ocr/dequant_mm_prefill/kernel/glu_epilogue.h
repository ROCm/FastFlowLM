// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#ifndef __GLU_EPILOGUE_H__
#define __GLU_EPILOGUE_H__
#include "aie_kernel_utils.h"
#include "lut_based_ops.h"
#include "common.hpp"

// act_glu: the up/gate projection's tile holds up in col blocks 0-3 and the matching gate in 4-7 (the host pairs the
// weight rows), and goes out as hid = up * silu(gate) in the down projection's bfp16 x tile format. README.md
// "GLU epilogue" has the layout and why each piece is where it is.

constexpr int glu_band_floats = NPU_N_TILE * 8;
constexpr int glu_groups = NPU_M_TILE / 16;
constexpr int glu_k_blocks = NPU_N_TILE / 2 / 8;
constexpr int glu_group_bytes = glu_k_blocks * 2 * 72;
constexpr int glu_slot_bytes = glu_groups * glu_group_bytes;
constexpr int glu_piece_bytes = 16;
constexpr int glu_pieces = glu_slot_bytes / glu_piece_bytes;
constexpr int glu_group_pieces = glu_group_bytes / glu_piece_bytes;

/// where group g's tiles go: group 0 into the caller's group0 (its bands are still being read), group g >= 1 over
/// y_acc bytes [(g - 1) * glu_group_bytes, g * glu_group_bytes), inside bands that are already read
static inline uint8* glu_group_tiles(float* y_acc, uint8* group0, int g) {
    return g == 0 ? group0 : (uint8*)y_acc + (g - 1) * glu_group_bytes;
}

/// y_acc: 128 rows x 64 columns as mm_aie_bfp16 leaves them, [16 bands of 8 rows][8 col blocks][8 x 8] fp32. The SiLU
/// is the act 1 one (getSiluBf16_nonLUT).
static inline void glu_tiles(float* y_acc, uint8* group0) {
    const aie::rounding_mode saved = aie::get_rounding();
    for (int g = 0; g < glu_groups; g++) {
        aie::block_vector_output_buffer_stream<bfp16ebs8, 64> out((bfp16ebs8*)glu_group_tiles(y_acc, group0, g));
        for (int kb = 0; kb < glu_k_blocks; kb++) {
            for (int z = 0; z < 2; z++) {
                const float* band = y_acc + (2 * g + z) * glu_band_floats;
                aie::accum<accfloat, 64> up_acc;
                up_acc.from_vector(aie::load_v<64>(band + kb * 64));
                aie::accum<accfloat, 64> gate_acc;
                gate_acc.from_vector(aie::load_v<64>(band + (kb + glu_k_blocks) * 64));
                aie::vector<bf16, 64> up = up_acc.template to_vector<bf16>();
                aie::vector<bf16, 64> gate = gate_acc.template to_vector<bf16>();

                aie::vector<bf16, 64> silu = getSiluBf16_nonLUT<64>(gate);

                aie::set_rounding(aie::rounding_mode::conv_even);
                aie::vector<bf16, 64> hid = aie::mul(up, silu).template to_vector<bf16>();
                aie::set_rounding(saved);

                aie::accum<accfloat, 64> hid_acc(hid);
                out << hid_acc.template to_vector<bfp16ebs8>();
            }
        }
    }
}

/// read position in the tile stream, carried across the 16 y chunks of one tile
struct glu_stream {
    uint8* src;
    int piece;
    int group;
    int left_in_group;
};

static inline glu_stream glu_stream_begin(uint8* group0) {
    return glu_stream{group0, 0, 0, glu_group_pieces};
}

/// The mem tile puts chunk element cb * 64 + r * 8 + c at row t * 8 + r, column cb * 8 + c of its 128 x 64 bf16 tile
/// and streams that out row by row. So the next 64 stream pieces of 16 B go to chunk bytes cb * 128 + r * 16, piece
/// i = r * 8 + cb. After glu_pieces pieces the chunk is left as it is (filler).
static inline void glu_slot_chunk(bf16* y, float* y_acc, glu_stream& s) {
    uint8* dst = (uint8*)y;
    for (int i = 0; i < 64 && s.piece < glu_pieces; i++) {
        const int r = i >> 3;
        const int cb = i & 7;
        aie::store_v(dst + cb * 128 + r * 16, aie::load_v<glu_piece_bytes>(s.src));
        s.src += glu_piece_bytes;
        s.piece++;
        if (--s.left_in_group == 0) {
            s.group++;
            s.src = (uint8*)y_acc + (s.group - 1) * glu_group_bytes;
            s.left_in_group = glu_group_pieces;
        }
    }
}

#endif
