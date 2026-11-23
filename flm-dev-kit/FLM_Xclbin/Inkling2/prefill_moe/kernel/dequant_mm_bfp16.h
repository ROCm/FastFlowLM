// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#pragma once
#include "aie_kernel_utils.h"
#include "q4_core.h"
#include "lut_based_ops.h"
#include "common.hpp"

constexpr int x_prod_lock = 0;
constexpr int x_cons_lock = 1;
constexpr int qw_prod_lock = 2;
constexpr int qw_cons_lock = 3;
constexpr int y_prod_lock = 4;
constexpr int y_cons_lock = 5;
constexpr int sync_lock = 6;

AIE_PING_FLAG is_y_ping = false;
AIE_PING_FLAG is_qw_ping = false;
AIE_PING_FLAG is_x_ping = false;


// A core multiplies its m rows by the 48 n (3 regions of 16 n) of its pair: y_acc layout
// [m16 group][m8 row (2)][n8 tile (6)][64], region s owning n8 tiles 2s, 2s + 1 (even, odd rows of its 16 n).
// A region is [kt (32)][2 n blocks] of 72 B tiles; the even tile of a pair holds regions 0 and 1 (w_s0), the odd
// tile region 2 (the front of w_s1).
// X chunk layout (DMA re-tiled): [k8 (32)][row (16)][k % 8], i.e. x + 128 k holds the A0 (rows 0-7)
// and A1 (rows 8-15) 8x8 tiles of k block k.
static constexpr int W_TILE_BYTES = 72;
static constexpr int W_REGION_BYTES = 32 * 2 * W_TILE_BYTES; // one n pair, all 32 k blocks of a 256 k slice
static constexpr int CORE_N = 48;                              // n per core and pass: 3 regions
static constexpr int GROUP_ACC = m_slice * CORE_N;             // y_acc floats per m16 group


// bf16 x chunk -> bfp16 A tiles, [k8][A0, A1], converted once and then used by all four 2x2 blocks
static inline __attribute__((always_inline)) void
convert_x_chunk(const bf16 *__restrict px, uint8 *__restrict xc) {
    aie::block_vector_output_buffer_stream<bfp16ebs8, 64> xo((bfp16ebs8 *)xc);
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_RANGE(16, 16)
    for (int k = 0; k < 16; k++) {
        auto a0 = ::to_v64bfp16ebs8(aie::accum<accfloat, 64>(aie::load_v<64>(px)));
        auto a1 = ::to_v64bfp16ebs8(aie::accum<accfloat, 64>(aie::load_v<64>(px + 64)));
        px += 128;
        xo << a0;
        xo << a1;
    }
}

// per region 2 m8 x 2 n8 blocks over all 32 k blocks of the 256 k slice, both operands bfp16: one accumulator
// load/store per 128 MMULs. r0 .. r2 are the regions in n order (n8 tiles 0-1, 2-3, 4-5).
template <bool FIRST>
static inline __attribute__((always_inline)) void
mm_aie_bfp16(float *__restrict y_group, const uint8 *__restrict xc, const uint8 *__restrict r0, const uint8 *__restrict r1,
             const uint8 *__restrict r2) {
    for (int s = 0; s < 3; s++) {
        const uint8 *w_region = s == 0 ? r0 : s == 1 ? r1 : r2;
        float *y0 = y_group + 2 * s * 64;                      // m8 row 0
        float *y1 = y0 + (CORE_N / 8) * 64;                    // m8 row 1
        // FIRST: the first k slice of a call starts from zero instead of a zeroed y_acc (no zeroing pass)
        aie::accum<accfloat, 64> c00, c01, c10, c11;
        if constexpr (FIRST) {
            c00 = aie::zeros<accfloat, 64>(); c01 = aie::zeros<accfloat, 64>();
            c10 = aie::zeros<accfloat, 64>(); c11 = aie::zeros<accfloat, 64>();
        } else {
            c00 = aie::accum<accfloat, 64>(aie::load_v<64>(y0)); c01 = aie::accum<accfloat, 64>(aie::load_v<64>(y0 + 64));
            c10 = aie::accum<accfloat, 64>(aie::load_v<64>(y1)); c11 = aie::accum<accfloat, 64>(aie::load_v<64>(y1 + 64));
        }
        aie::block_vector_input_buffer_stream<bfp16ebs8, 64> xs((const bfp16ebs8 *)xc);
        aie::block_vector_input_buffer_stream<bfp16ebs8, 64> ws((const bfp16ebs8 *)w_region);
        AIE_PREPARE_FOR_PIPELINING
        AIE_LOOP_RANGE(16, 16)
        for (int k = 0; k < 16; k++) {
#pragma clang loop unroll(full)
            for (int u = 0; u < 2; u++) {
                auto a0 = xs.pop();
                auto a1 = xs.pop();
                auto b0 = ws.pop();
                auto b1 = ws.pop();
                c00 = mac_8x8_8x8T(a0, b0, c00);
                c01 = mac_8x8_8x8T(a0, b1, c01);
                c10 = mac_8x8_8x8T(a1, b0, c10);
                c11 = mac_8x8_8x8T(a1, b1, c11);
            }
        }
        aie::store_v(y0, c00.template to_vector<float>());
        aie::store_v(y0 + 64, c01.template to_vector<float>());
        aie::store_v(y1, c10.template to_vector<float>());
        aie::store_v(y1 + 64, c11.template to_vector<float>());
    }
}

// A y chunk is 4 n8 tiles, two tile pairs [t0, t1] of [m 8][n 8] (a region's 16 n; region boundaries fall on
// 128-element boundaries of y_acc). The dequant writes the even rows of a region's 16 n to t0 and the odd rows
// to t1 (_Q4_K_bfp16_kh); put them back in row order, in place.
static inline __attribute__((always_inline)) void unpermute_chunk(bf16 *__restrict y) {
#pragma clang loop unroll(full)
    for (int pair = 0; pair < 2; pair++) {
        bf16 *p = y + pair * 128;
        aie::vector<bf16, 64> e = aie::load_v<64>(p), o = aie::load_v<64>(p + 64);
        auto z = aie::interleave_zip(e, o, 1);                        // [m][n 16]
        auto n = aie::interleave_unzip(z.first, z.second, 8);         // [m][n 0-7], [m][n 8-15]
        aie::store_v(p, n.first);
        aie::store_v(p + 64, n.second);
    }
}

void silu_aie(bf16 *__restrict x, bf16 *__restrict o, int D){

    bf16 *x_it = x;

    constexpr int vector_size = 64;

    aie::vector<bf16, vector_size> x_vec;
    aie::vector<bf16, vector_size> o_vec;

    const int F = D / vector_size;

    for (int i = 0; i < F; i++){
        x_vec = aie::load_v<vector_size>(x_it);
        o_vec = getSiluBf16_nonLUT<vector_size>(x_vec);
        aie::store_v(o, o_vec);
        x_it += vector_size;
        o += vector_size;
    }
}

void gelu_aie(bf16 *__restrict x, bf16 *__restrict o, int D){

    bf16 *x_it = x;

    constexpr int vector_size = 64;

    aie::vector<bf16, vector_size> x_vec;
    aie::vector<bf16, vector_size> o_vec;

    const int F = D / vector_size;

    for (int i = 0; i < F; i++){
        x_vec = aie::load_v<vector_size>(x_it);
        o_vec = getGeluBf16_nonLUT<vector_size>(x_vec);
        aie::store_v(o, o_vec);
        x_it += vector_size;
        o += vector_size;
    }
}

template <typename T_in, typename T_out>
void passThrough_aie(const T_in *restrict in, T_out *restrict out) {
    const int vec_factor = 64;
    for (int c = 0; c < y_slice; c += vec_factor) {
        aie::accum<accfloat, vec_factor> c_acc_in;
        c_acc_in.from_vector(aie::load_v<vec_factor>(in + c));
        aie::store_v(out + c, c_acc_in.template to_vector<T_out>());
    }
}

