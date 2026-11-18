// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "common.h"

#include <aie_api/aie.hpp>

#include "sort.hpp"
#include "mvm_bf16.hpp"
#include "common_external.hpp"

#include "lut_based_ops.h"
#include "gptossParam.hpp"



#include "packet_control_gen.hpp"



extern "C" {





    // This master version responsible for acquire data
    // This master version should process the first 16 value of 32 BLOCK_ROWS
    void MVM_bf16_block_32_64_16_with_bias_master(
        bfloat16* mat_in_ping, bfloat16* mat_in_pong, 
        bfloat16* vec_in,  // assume each buffer is store as 1xtotal_mat_col size already
        bfloat16* vector_out, // size of 1x BLOCK_ROWS
        float* result_acc_ptr, bfloat16* bias_ptr,

        int32_t *input_ping_pong_flag

    ){

        constexpr int BLOCK_ROWS = EXPERT_BLOCK_ROWS;
        constexpr int BLOCK_COLS = EXPERT_BLOCK_COLS;
        constexpr int TILE_ROWS = EXPERT_BLOCK_TILE_ROWS;
        
        constexpr int MAT_IN_BLOCK_SIZE = BLOCK_ROWS * BLOCK_COLS;



        bfloat16* cur_output_ptr = vector_out;
        bfloat16* cur_mat_ptr;

        _lock_acquire(M_D_CON_LOCK, 1);
        
        // At very beginning, the first few  matrix_in_ping/pong is used for loading the bias buffer 
        //Note: bias_ptr is size of 1xtotal_mat_row
        // but bias_ptr is not padded into MAT_IN_BLOCK_SIZE
        {   
            AIE_PREPARE_FOR_PIPELINING
            AIE_LOOP_RANGE(EXPERT_BIAS_BLOCK,EXPERT_BIAS_BLOCK )
            for(int i = 0; i < EXPERT_BIAS_BLOCK; i++){
                if(*input_ping_pong_flag == 0){
                    cur_mat_ptr  = mat_in_ping;
                    *input_ping_pong_flag = 1;
                }else{
                    cur_mat_ptr  = mat_in_pong;
                    *input_ping_pong_flag = 0;
                }

                _lock_acquire(M_WEIGHT_CON_LOCK, 1);
                copy_bf16_to_bf16<MAT_IN_BLOCK_SIZE>(bias_ptr + i*MAT_IN_BLOCK_SIZE, cur_mat_ptr);
                _lock_release(M_WEIGHT_PROD_LOCK, 1);
            }   

        }

        _lock_acquire(M_SCORE_PROD_LOCK, 1);

        AIE_PREPARE_FOR_PIPELINING
        AIE_LOOP_RANGE(NUM_OF_EXPERT/BLOCK_ROWS,  NUM_OF_EXPERT/BLOCK_ROWS)
        for(int row_idx = 0; row_idx< NUM_OF_EXPERT; row_idx += BLOCK_ROWS){
            // instead of loading zero, we load the bias value here
            copy_bf16_to_float<BLOCK_ROWS>(result_acc_ptr, bias_ptr + row_idx);

            static_assert(D %BLOCK_COLS == 0 );

            AIE_PREPARE_FOR_PIPELINING
            AIE_LOOP_RANGE( D/BLOCK_COLS, D/BLOCK_COLS)
            for(int col_idx = 0; col_idx < D; col_idx += BLOCK_COLS){

                if(*input_ping_pong_flag == 0){
                    cur_mat_ptr  = mat_in_ping;
                }else{
                    cur_mat_ptr  = mat_in_pong;
                }                

                _lock_acquire(M_WEIGHT_CON_LOCK, 1);
                // here, release a lock to slave to signal the slave can process
                _lock_release(M_Prod_S_Con_LOCK, 1);
                matvec_vectorized_Bf16_col_major<BLOCK_ROWS/2, BLOCK_COLS, TILE_ROWS>(
                    cur_mat_ptr, // different for slave
                    vec_in+col_idx,
                    result_acc_ptr      // different for slave
                );
                _lock_acquire(M_Con_S_Prod_LOCK, 1);
                // here wait for slave to finish
                _lock_release(M_WEIGHT_PROD_LOCK, 1);  
                
                
                // Then, change the input_ping_pong flag here
                if(*input_ping_pong_flag == 0){
                    *input_ping_pong_flag = 1;
                }else{
                    *input_ping_pong_flag = 0;
                }     

            } // end of for col_idx

            copy_float_to_bf16<BLOCK_ROWS>(cur_output_ptr + row_idx, result_acc_ptr);


        }
        _lock_release(M_SCORE_CON_LOCK, 1);

        _lock_release(M_D_PROD_LOCK, 1);


    }




}