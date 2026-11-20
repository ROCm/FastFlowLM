// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __MVM_BF16_HPP__
#define __MVM_BF16_HPP__

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "zero.h"
#include <aie_api/aie.hpp>
#include "passThroughCommon.hpp"

#include "common_external.hpp"


// Column-major matrix-vector multiply, accumulating into result_acc_ptr.
//
// mat_in is column-major: each column is a contiguous block of TILE_ROW_STRIDE
// elements.  The outer loop walks TILE_ROW_STRIDE-row bands of the output; the
// inner loop walks all columns, processing VEC_IN_LANES columns per iteration
// alongside the matching VEC_IN_LANES input-vector elements.
//
//   mat_in  : MatrixRows x MatrixCols, column-major, column stride = TILE_ROW_STRIDE
//   vec_in  : length MatrixCols
//   result  : length MatrixRows (float accumulator; caller initialises to 0 or a partial sum)
template<unsigned int MatrixRows, unsigned int MatrixCols, unsigned int TILE_ROW_STRIDE,
    typename MatrixT, typename VectorInT, typename VectorOutT,
    unsigned int PHYSICAL_ROW_STRIDE = TILE_ROW_STRIDE
>
void matvec_vectorized_col_major(
    MatrixT *mat_in, VectorInT * vec_in, VectorOutT *result_acc_ptr
){
    static_assert(MatrixRows % TILE_ROW_STRIDE == 0);

    // Columns consumed per inner iteration.  Fixed at 16 to match the 16-way
    // aie::accumulate call below; enforced by the static_assert inside the loop.
    constexpr const int VEC_IN_LANES = 32;
    static_assert(MatrixCols % VEC_IN_LANES == 0);


    MatrixT* a_ptr = mat_in;
    VectorOutT* res_ptr = result_acc_ptr;

    // AIE_PREPARE_FOR_PIPELINING
    // AIE_LOOP_RANGE(MatrixRows/TILE_ROW_STRIDE, MatrixRows/TILE_ROW_STRIDE  )
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_RANGE(MatrixRows/TILE_ROW_STRIDE, MatrixRows/TILE_ROW_STRIDE)
    for(size_t row_idx = 0; row_idx <MatrixRows; row_idx+= TILE_ROW_STRIDE ){
        
        aie::vector<float, TILE_ROW_STRIDE> _res_vec = aie::load_v<TILE_ROW_STRIDE>(res_ptr);
        aie::accum<accfloat, TILE_ROW_STRIDE> res_acc;
        res_acc.from_vector(_res_vec);

        VectorInT* b_ptr = vec_in;
        
        AIE_PREPARE_FOR_PIPELINING
        AIE_LOOP_RANGE(MatrixCols/VEC_IN_LANES, MatrixCols/VEC_IN_LANES)
        for(int col_idx = 0; col_idx < MatrixCols; col_idx += VEC_IN_LANES){

            // Load VEC_IN_LANES consecutive input-vector elements (one column group).
            aie::vector<VectorInT, VEC_IN_LANES> b_vec = aie::load_v<VEC_IN_LANES>(b_ptr);
            aie::vector<MatrixT, VEC_IN_LANES> b_vec_in_MatrixT;
            if constexpr (std::is_same_v<VectorInT, MatrixT>) {
                b_vec_in_MatrixT = b_vec;
            } else {
                aie::accum<accfloat, VEC_IN_LANES> b_vec_acc;
                b_vec_acc.from_vector(b_vec);
                b_vec_in_MatrixT = b_vec_acc.template to_vector<MatrixT>();
            }

            b_ptr += VEC_IN_LANES;

            // Load and mac in a loop (slower compile, may reduce pipeline efficiency vs unrolled below)
            AIE_PREPARE_FOR_PIPELINING
            AIE_LOOP_UNROLL_FULL
            for (int i = 0; i < VEC_IN_LANES; i++) {
                aie::vector<MatrixT, TILE_ROW_STRIDE> a_vec = aie::load_v<TILE_ROW_STRIDE>(a_ptr);
                a_ptr += PHYSICAL_ROW_STRIDE;
                res_acc = aie::mac(res_acc, a_vec, b_vec_in_MatrixT.get(i));
            }

            // // Load the corresponding VEC_IN_LANES column slices of the matrix tile
            // // (each slice is TILE_ROW_STRIDE elements, one per output row in the band).
            // // Unrolled manually for pipeline efficiency; static_assert guards the count.
            // static_assert(VEC_IN_LANES == 16);
            // aie::vector<MatrixT, TILE_ROW_STRIDE> a_vec_0 = aie::load_v<TILE_ROW_STRIDE>(a_ptr);
            // a_ptr+= TILE_ROW_STRIDE;
            // aie::vector<MatrixT, TILE_ROW_STRIDE> a_vec_1 = aie::load_v<TILE_ROW_STRIDE>(a_ptr);
            // a_ptr+= TILE_ROW_STRIDE;
            // aie::vector<MatrixT, TILE_ROW_STRIDE> a_vec_2 = aie::load_v<TILE_ROW_STRIDE>(a_ptr);
            // a_ptr+= TILE_ROW_STRIDE;
            // aie::vector<MatrixT, TILE_ROW_STRIDE> a_vec_3 = aie::load_v<TILE_ROW_STRIDE>(a_ptr);
            // a_ptr+= TILE_ROW_STRIDE;
            // aie::vector<MatrixT, TILE_ROW_STRIDE> a_vec_4 = aie::load_v<TILE_ROW_STRIDE>(a_ptr);
            // a_ptr+= TILE_ROW_STRIDE;
            // aie::vector<MatrixT, TILE_ROW_STRIDE> a_vec_5 = aie::load_v<TILE_ROW_STRIDE>(a_ptr);
            // a_ptr+= TILE_ROW_STRIDE;
            // aie::vector<MatrixT, TILE_ROW_STRIDE> a_vec_6 = aie::load_v<TILE_ROW_STRIDE>(a_ptr);
            // a_ptr+= TILE_ROW_STRIDE;
            // aie::vector<MatrixT, TILE_ROW_STRIDE> a_vec_7 = aie::load_v<TILE_ROW_STRIDE>(a_ptr);
            // a_ptr+= TILE_ROW_STRIDE;
            // aie::vector<MatrixT, TILE_ROW_STRIDE> a_vec_8 = aie::load_v<TILE_ROW_STRIDE>(a_ptr);
            // a_ptr+= TILE_ROW_STRIDE;
            // aie::vector<MatrixT, TILE_ROW_STRIDE> a_vec_9 = aie::load_v<TILE_ROW_STRIDE>(a_ptr);
            // a_ptr+= TILE_ROW_STRIDE;
            // aie::vector<MatrixT, TILE_ROW_STRIDE> a_vec_10 = aie::load_v<TILE_ROW_STRIDE>(a_ptr);
            // a_ptr+= TILE_ROW_STRIDE;
            // aie::vector<MatrixT, TILE_ROW_STRIDE> a_vec_11 = aie::load_v<TILE_ROW_STRIDE>(a_ptr);
            // a_ptr+= TILE_ROW_STRIDE;
            // aie::vector<MatrixT, TILE_ROW_STRIDE> a_vec_12 = aie::load_v<TILE_ROW_STRIDE>(a_ptr);
            // a_ptr+= TILE_ROW_STRIDE;
            // aie::vector<MatrixT, TILE_ROW_STRIDE> a_vec_13 = aie::load_v<TILE_ROW_STRIDE>(a_ptr);
            // a_ptr+= TILE_ROW_STRIDE;
            // aie::vector<MatrixT, TILE_ROW_STRIDE> a_vec_14 = aie::load_v<TILE_ROW_STRIDE>(a_ptr);
            // a_ptr+= TILE_ROW_STRIDE;
            // aie::vector<MatrixT, TILE_ROW_STRIDE> a_vec_15 = aie::load_v<TILE_ROW_STRIDE>(a_ptr);
            // a_ptr+= TILE_ROW_STRIDE;

            // // res_acc = aie::accumulate<TILE_ROW_STRIDE>(
            // //     res_acc, b_vec, 0,
            // //     a_vec_0, a_vec_1, a_vec_2, a_vec_3,
            // //     a_vec_4, a_vec_5, a_vec_6, a_vec_7,
            // //     a_vec_8, a_vec_9, a_vec_10, a_vec_11,
            // //     a_vec_12, a_vec_13, a_vec_14, a_vec_15
            // // );

            // res_acc = aie::mac(res_acc, a_vec_0,  b_vec_in_MatrixT.get(0));
            // res_acc = aie::mac(res_acc, a_vec_1,  b_vec_in_MatrixT.get(1));
            // res_acc = aie::mac(res_acc, a_vec_2,  b_vec_in_MatrixT.get(2));
            // res_acc = aie::mac(res_acc, a_vec_3,  b_vec_in_MatrixT.get(3));
            // res_acc = aie::mac(res_acc, a_vec_4,  b_vec_in_MatrixT.get(4));
            // res_acc = aie::mac(res_acc, a_vec_5,  b_vec_in_MatrixT.get(5));
            // res_acc = aie::mac(res_acc, a_vec_6,  b_vec_in_MatrixT.get(6));
            // res_acc = aie::mac(res_acc, a_vec_7,  b_vec_in_MatrixT.get(7));
            // res_acc = aie::mac(res_acc, a_vec_8,  b_vec_in_MatrixT.get(8));
            // res_acc = aie::mac(res_acc, a_vec_9,  b_vec_in_MatrixT.get(9));
            // res_acc = aie::mac(res_acc, a_vec_10, b_vec_in_MatrixT.get(10));
            // res_acc = aie::mac(res_acc, a_vec_11, b_vec_in_MatrixT.get(11));
            // res_acc = aie::mac(res_acc, a_vec_12, b_vec_in_MatrixT.get(12));
            // res_acc = aie::mac(res_acc, a_vec_13, b_vec_in_MatrixT.get(13));
            // res_acc = aie::mac(res_acc, a_vec_14, b_vec_in_MatrixT.get(14));
            // res_acc = aie::mac(res_acc, a_vec_15, b_vec_in_MatrixT.get(15));
        }
        aie::store_v(res_ptr, res_acc.template to_vector<VectorOutT>());
        res_ptr+= TILE_ROW_STRIDE;

    }

}


#endif