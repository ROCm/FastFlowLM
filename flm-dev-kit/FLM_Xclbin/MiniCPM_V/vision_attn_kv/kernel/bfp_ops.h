// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#ifndef BFP_OPS_H
#define BFP_OPS_H

#include <aie_api/aie.hpp>
#include "common.h"

// QK and FV on bfp16 operands converted once. With AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16 every bf16 8x8x8 mac
// converts both of its operands (A as is, B transposed) to bfp16ebs8 and runs mac_8x8_8x8T; here each operand is
// converted once -- Q per round, a K or V block once for all four q tiles, an S block once for all nine d blocks --
// from the same values, so the products and their accumulation order are the emulation's.
using bfp_vec = aie::block_vector<bfp16ebs8, 64>;
constexpr int bfp_bytes = 72;   // one 8x8 bfp16ebs8 block: 8 exponent bytes + 64 mantissas

inline bfp_vec to_bfp(aie::vector<bf16, 64> v) {
    return aie::accum<accfloat, 64>(v).template to_vector<bfp16ebs8>();
}

// K block [d8 i][key group j][8 keys x 8 d] -> bfp blocks in the same order (B of mac_8x8_8x8T is K as stored: the
// QK kernel's transpose and the emulation's cancel).
inline void pack_k(uint8_t *__restrict dst, const bf16 *__restrict k_in) {
    aie::block_vector_output_buffer_stream<bfp16ebs8, 64> out((bfp16ebs8 *)dst);
    for (int b = 0; b < d_blocks * (lk / 8); b++)
        out << to_bfp(aie::load_v<64>(k_in + b * 64));
}

// V block [d8 j][key group i][8 keys x 8 d] -> transposed (8 d x 8 keys) bfp blocks, same order.
inline void pack_v(uint8_t *__restrict dst, const bf16 *__restrict v_in) {
    aie::block_vector_output_buffer_stream<bfp16ebs8, 64> out((bfp16ebs8 *)dst);
    for (int b = 0; b < d_blocks * (lk / 8); b++)
        out << to_bfp(aie::transpose(aie::load_v<64>(v_in + b * 64), 8, 8));
}

// S[16 rows][16 keys] = scale * Q_tile . K_block^T. qb: the tile's 2 row blocks x d_blocks packed Q blocks; kb: pack_k.
inline void qk_bfp(bf16 *__restrict pS, const uint8_t *__restrict qb, const uint8_t *__restrict kb) {
    aie::block_vector_input_buffer_stream<bfp16ebs8, 64> q0((const bfp16ebs8 *)qb);
    aie::block_vector_input_buffer_stream<bfp16ebs8, 64> q1((const bfp16ebs8 *)(qb + d_blocks * bfp_bytes));
    aie::block_vector_input_buffer_stream<bfp16ebs8, 64> ks((const bfp16ebs8 *)kb);
    aie::accum<accfloat, 64> c00 = aie::zeros<accfloat, 64>(), c01 = aie::zeros<accfloat, 64>();
    aie::accum<accfloat, 64> c10 = aie::zeros<accfloat, 64>(), c11 = aie::zeros<accfloat, 64>();
    for (int i = 0; i < d_blocks; i++) {
        auto a0 = q0.pop();
        auto a1 = q1.pop();
        auto b0 = ks.pop();
        auto b1 = ks.pop();
        c00 = mac_8x8_8x8T(a0, b0, c00);
        c01 = mac_8x8_8x8T(a0, b1, c01);
        c10 = mac_8x8_8x8T(a1, b0, c10);
        c11 = mac_8x8_8x8T(a1, b1, c11);
    }
    auto m0 = aie::interleave_zip(c00.template to_vector<bf16>(), c01.template to_vector<bf16>(), 8);
    auto m1 = aie::interleave_zip(c10.template to_vector<bf16>(), c11.template to_vector<bf16>(), 8);
    aie::store_v(pS,       aie::mul(m0.first,  (bf16)scale).template to_vector<bf16>(0));
    aie::store_v(pS + 64,  aie::mul(m0.second, (bf16)scale).template to_vector<bf16>(0));
    aie::store_v(pS + 128, aie::mul(m1.first,  (bf16)scale).template to_vector<bf16>(0));
    aie::store_v(pS + 192, aie::mul(m1.second, (bf16)scale).template to_vector<bf16>(0));
}

// y[16 rows][dh] += S_block (reordered: [row block][key group][8x8]) . V_block. vb: pack_v.
inline void fv_bfp(float *__restrict pY, const bf16 *__restrict pS, const uint8_t *__restrict vb) {
    const bfp_vec s00 = to_bfp(aie::load_v<64>(pS));
    const bfp_vec s01 = to_bfp(aie::load_v<64>(pS + 64));
    const bfp_vec s10 = to_bfp(aie::load_v<64>(pS + 128));
    const bfp_vec s11 = to_bfp(aie::load_v<64>(pS + 192));
    aie::block_vector_input_buffer_stream<bfp16ebs8, 64> vs((const bfp16ebs8 *)vb);
    float *__restrict y0 = pY;
    float *__restrict y1 = pY + 8 * dh;
    for (int j = 0; j < d_blocks; j++) {
        auto b0 = vs.pop();
        auto b1 = vs.pop();
        aie::accum<accfloat, 64> a0(aie::load_v<64>(y0));
        aie::accum<accfloat, 64> a1(aie::load_v<64>(y1));
        a0 = mac_8x8_8x8T(s00, b0, a0);
        a1 = mac_8x8_8x8T(s10, b0, a1);
        a0 = mac_8x8_8x8T(s01, b1, a0);
        a1 = mac_8x8_8x8T(s11, b1, a1);
        aie::store_v(y0, a0.template to_vector<float>());
        aie::store_v(y1, a1.template to_vector<float>());
        y0 += 64;
        y1 += 64;
    }
}

#endif
