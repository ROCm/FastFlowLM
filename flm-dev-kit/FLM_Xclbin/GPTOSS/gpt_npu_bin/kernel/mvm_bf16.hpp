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


// this process a column major matrix with stride of TILE_ROWS
template<unsigned int BLOCK_ROWS, unsigned int BLOCK_COLS, unsigned int TILE_ROWS> // the stride for column major
void matvec_vectorized_Bf16_col_major(
    bfloat16 *mat_in, bfloat16*vec_in, float* result_acc_ptr
){
    static_assert(BLOCK_ROWS % TILE_ROWS == 0);

    
    constexpr const int num_col_process_per_loop = 16;
    static_assert(BLOCK_COLS % num_col_process_per_loop == 0); // num_col_process_per_loop should be multiple of 16


    bfloat16* a_ptr = mat_in;
    // Use a local result pointer so we don't advance the caller's pointer across calls
    float* res_ptr = result_acc_ptr;

    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_RANGE(BLOCK_ROWS/TILE_ROWS, BLOCK_ROWS/TILE_ROWS  )
    for(size_t row_idx = 0; row_idx <BLOCK_ROWS; row_idx+= TILE_ROWS ){
        
        aie::vector<float, TILE_ROWS> _res_vec = aie::load_v<TILE_ROWS>(res_ptr);
        aie::accum<accfloat, TILE_ROWS> res_acc;
        res_acc.from_vector(_res_vec);

        bfloat16* b_ptr = vec_in;
        
        AIE_PREPARE_FOR_PIPELINING
        AIE_LOOP_RANGE(BLOCK_COLS/num_col_process_per_loop, BLOCK_COLS/num_col_process_per_loop )
        for(int col_idx = 0; col_idx < BLOCK_COLS; col_idx += num_col_process_per_loop){

            aie::vector<bfloat16, num_col_process_per_loop> b_vec= aie::load_v<num_col_process_per_loop>(b_ptr);
            b_ptr+= num_col_process_per_loop;

            // up to a_vec_15, since num_col_process_per_loop== 16
            aie::vector<bfloat16, TILE_ROWS> a_vec_0 = aie::load_v<TILE_ROWS>(a_ptr);
            a_ptr+= TILE_ROWS;
            aie::vector<bfloat16, TILE_ROWS> a_vec_1 = aie::load_v<TILE_ROWS>(a_ptr);
            a_ptr+= TILE_ROWS;
            aie::vector<bfloat16, TILE_ROWS> a_vec_2 = aie::load_v<TILE_ROWS>(a_ptr);
            a_ptr+= TILE_ROWS;
            aie::vector<bfloat16, TILE_ROWS> a_vec_3 = aie::load_v<TILE_ROWS>(a_ptr);
            a_ptr+= TILE_ROWS;
            aie::vector<bfloat16, TILE_ROWS> a_vec_4 = aie::load_v<TILE_ROWS>(a_ptr);
            a_ptr+= TILE_ROWS;
            aie::vector<bfloat16, TILE_ROWS> a_vec_5 = aie::load_v<TILE_ROWS>(a_ptr);
            a_ptr+= TILE_ROWS;
            aie::vector<bfloat16, TILE_ROWS> a_vec_6 = aie::load_v<TILE_ROWS>(a_ptr);
            a_ptr+= TILE_ROWS;
            aie::vector<bfloat16, TILE_ROWS> a_vec_7 = aie::load_v<TILE_ROWS>(a_ptr);
            a_ptr+= TILE_ROWS;
            aie::vector<bfloat16, TILE_ROWS> a_vec_8 = aie::load_v<TILE_ROWS>(a_ptr);
            a_ptr+= TILE_ROWS;
            aie::vector<bfloat16, TILE_ROWS> a_vec_9 = aie::load_v<TILE_ROWS>(a_ptr);
            a_ptr+= TILE_ROWS;
            aie::vector<bfloat16, TILE_ROWS> a_vec_10 = aie::load_v<TILE_ROWS>(a_ptr);
            a_ptr+= TILE_ROWS;      
            aie::vector<bfloat16, TILE_ROWS> a_vec_11 = aie::load_v<TILE_ROWS>(a_ptr);
            a_ptr+= TILE_ROWS;      
            aie::vector<bfloat16, TILE_ROWS> a_vec_12 = aie::load_v<TILE_ROWS>(a_ptr);
            a_ptr+= TILE_ROWS;      
            aie::vector<bfloat16, TILE_ROWS> a_vec_13 = aie::load_v<TILE_ROWS>(a_ptr);
            a_ptr+= TILE_ROWS;      
            aie::vector<bfloat16, TILE_ROWS> a_vec_14 = aie::load_v<TILE_ROWS>(a_ptr);
            a_ptr+= TILE_ROWS;      
            aie::vector<bfloat16, TILE_ROWS> a_vec_15 = aie::load_v<TILE_ROWS>(a_ptr);
            a_ptr+= TILE_ROWS;

            res_acc = aie::accumulate<TILE_ROWS>(
                res_acc, b_vec, 0,
                a_vec_0, a_vec_1, a_vec_2, a_vec_3,
                a_vec_4, a_vec_5, a_vec_6, a_vec_7,
                a_vec_8, a_vec_9, a_vec_10, a_vec_11,
                a_vec_12, a_vec_13, a_vec_14, a_vec_15
            );
        }
        // now store res_acc back to result_acc_ptr
        aie::store_v(res_ptr, res_acc.template to_vector<float>());
        res_ptr+= TILE_ROWS;

    }

}


#endif