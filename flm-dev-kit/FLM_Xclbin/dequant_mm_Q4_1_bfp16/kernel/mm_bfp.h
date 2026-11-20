// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#ifndef __MM_BFP_H__
#define __MM_BFP_H__
#include <aie_api/aie.hpp>
#include "aie_kernel_utils.h"

/// @brief Accumulates a (16 x colA*8) bf16 slab of X against colB n tiles of block-float weights, into fp32 partial sums.
///
/// The same blocking as the bf16 multiply it replaces (2 x 2 mmul tiles per step), but B arrives as native bfp16ebs8
/// tiles that the dequantizer already converted, so the loop only converts the X tiles.
///
/// @param pY    fp32 partial sums, [m tile][n tile][8x8], advanced by the caller per call
/// @param pX    bf16 X, mmul A tile order [m tile][k tile][8x8] (tile strides as the bf16 multiply used)
/// @param pW    weight tiles, [n tile][k tile], ldB k tiles per n tile
/// @param w_tile0  index of the first weight tile this call reads: the slice's first tile plus 2 per k tile skipped (tiles are
///                 stored [n pair][k tile][2 n tiles])
template <unsigned rowA, unsigned colA, unsigned colB, unsigned ldB>
void mm_bfp_aie(float *__restrict pY, const bfp16ebs8 *__restrict pX, const bfp16ebs8 *__restrict pW, unsigned w_tile0) {
    using MMUL = aie::mmul<8, 8, 8, bfp16ebs8, bfp16ebs8, accauto>;

    for (unsigned z = 0; z < rowA; z += 2) {
        float *__restrict pY1 = pY + (z * colB) * MMUL::size_C * 2;
        float *__restrict pY2 = pY + ((z + 1) * colB) * MMUL::size_C * 2;

        for (unsigned j = 0; j < colB; j += 2) {
            // X tiles come as [k tile][row tile], so one stream hands out the pair (z, z + 1) tile after tile
            aie::block_vector_input_buffer_stream<bfp16ebs8, MMUL::size_A> x_stream(pX);
            x_stream.seek(z);

            // the n tiles j and j + 1 sit next to each other for every k tile (see _Q4_1_bfp16), so one stream serves both
            aie::block_vector_input_buffer_stream<bfp16ebs8, MMUL::size_B> w_stream(pW);
            w_stream.seek(w_tile0 + (j / 2) * 2 * ldB);

            MMUL Y00(aie::load_v<MMUL::size_C>(pY1));
            MMUL Y01(aie::load_v<MMUL::size_C>(pY1 + MMUL::size_C));
            MMUL Y10(aie::load_v<MMUL::size_C>(pY2));
            MMUL Y11(aie::load_v<MMUL::size_C>(pY2 + MMUL::size_C));

            AIE_PREPARE_FOR_PIPELINING
            AIE_LOOP_RANGE(colA, colA)
            AIE_TRY_INITIATION_INTERVAL(4)
            for (unsigned i = 0; i < colA; ++i) {
                aie::block_vector<bfp16ebs8, MMUL::size_A> X0 = x_stream.pop();
                aie::block_vector<bfp16ebs8, MMUL::size_A> X1 = x_stream.pop();
                aie::block_vector<bfp16ebs8, MMUL::size_B> W0 = w_stream.pop();
                aie::block_vector<bfp16ebs8, MMUL::size_B> W1 = w_stream.pop();

                Y00.mac(X0, aie::op_transpose(W0));
                Y01.mac(X0, aie::op_transpose(W1));
                Y10.mac(X1, aie::op_transpose(W0));
                Y11.mac(X1, aie::op_transpose(W1));
            }
            aie::store_v(pY1, Y00.template to_vector<float>());
            pY1 += MMUL::size_C;
            aie::store_v(pY1, Y01.template to_vector<float>());
            pY1 += MMUL::size_C;
            aie::store_v(pY2, Y10.template to_vector<float>());
            pY2 += MMUL::size_C;
            aie::store_v(pY2, Y11.template to_vector<float>());
            pY2 += MMUL::size_C;
        }
    }
}

#endif
