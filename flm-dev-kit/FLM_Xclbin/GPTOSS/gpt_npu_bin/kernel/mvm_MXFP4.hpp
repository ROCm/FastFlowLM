// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.



#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <aie_api/aie.hpp>
#include "MXFP4.h"
#include "aie_kernel_utils.h"

//Note: similar to dequant_Q4NX_BLOCK_MXFP4
// the Matrix is in Q4NX format and in column major order with stride of Q4NX_COL_STRIDE
// Scalar is in column major, with stride of Q4NX_ROW

//Eachtime, we expect a vector of [q4nx_mat_col x 1] 
// Each time, we produce a [q4nx_mat_row x 1] vector as output

template<unsigned int Q4NX_ROW, unsigned int Q4NX_COL, unsigned int Q4NX_COL_STRIDE>
void mvm_MXFP4(uint8_t* scalar, uint8_t* data_block, bfloat16*vec_in,
    float* result_acc_ptr
) {
    //NOTE: for now. don't consider the bias yet


    //matrix is size of q4nx_mat_row x q4nx_mat_col
    // assume q4nx_mat_row% Q4NX_ROW == 0
    // and q4nx_mat_row% Q4NX_COL_STRIDE == 0
    // vec_in is size of q4nx_mat_colx1
    static_assert(Q4NX_ROW % Q4NX_COL_STRIDE == 0);
    static_assert(Q4NX_COL_STRIDE == 16);// the most effecitve and what we build on
    static_assert( (Q4NX_COL)%(2*32) == 0);



    uint8_t *cur_block_ptr = (uint8_t *)data_block; // each uint8 contains two 4bit data

    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_RANGE( Q4NX_ROW/Q4NX_COL_STRIDE, Q4NX_ROW/Q4NX_COL_STRIDE )
    for(size_t row_idx= 0; row_idx<  Q4NX_ROW; row_idx += Q4NX_COL_STRIDE){

        bfloat16 *cur_vec_ptr = vec_in;



        aie::vector<float, Q4NX_COL_STRIDE> _res_vec= aie::load_v<Q4NX_COL_STRIDE>(result_acc_ptr);
        aie::accum<accfloat, Q4NX_COL_STRIDE> res_acc;
        res_acc.from_vector(_res_vec);


        uint8_t *cur_scalar_ptr = (uint8_t *)scalar  + row_idx ;
        AIE_PREPARE_FOR_PIPELINING
        AIE_LOOP_RANGE(Q4NX_COL/32, Q4NX_COL/32 )
#if !defined(__chess__)
        // Unroll explicitly so Peano can schedule across iterations, as chess
        // does on its own. Iterations are independent (temp_acc is re-zeroed
        // each time, only res_acc carries), so the accumulation order is kept.
        AIE_LOOP_UNROLL(Q4NX_COL/32)
#endif
        for(size_t col_idx = 0; col_idx<Q4NX_COL; col_idx += 32 ) AIE_LOOP_FLATTEN{


            aie::vector<bfloat16, 32> in_vec_scalar = aie::load_v<32>(cur_vec_ptr);
            cur_vec_ptr+=32;
            
            // 16 q4nx_value = 8byre, 
            // uint8_t *cur_scalar_ptr = (uint8_t *)scalar + (col_idx/32) * Q4NX_ROW + row_idx ;
            aie::vector<uint8_t, Q4NX_COL_STRIDE> scalar_vec_in = aie::load_v<16>(cur_scalar_ptr);
            cur_scalar_ptr += Q4NX_ROW;

            aie::vector<bfloat16, Q4NX_COL_STRIDE> scalar_dequant = mxfp4_scalar_to_bf16(scalar_vec_in); 

            // recall the MXFP4 data is arrange: one scalar<-> 32 data value


            // do a manual l
            aie::accum<accfloat, Q4NX_COL_STRIDE> temp_acc;
            temp_acc.from_vector(aie::zeros<float, 16>());            





        {


            // --- 2. Load and dequantize all vector data ---
            // Note: cur_block_ptr is advanced 16 times
            aie::vector<uint8_t, Q4NX_COL_STRIDE> block_vec_in_0_0 = aie::load_v<Q4NX_COL_STRIDE>(cur_block_ptr); cur_block_ptr += Q4NX_COL_STRIDE;
            aie::vector<bfloat16, Q4NX_COL_STRIDE*2> data_dequant_0_0 = mxfp4_to_bf16(block_vec_in_0_0);
            
            aie::vector<uint8_t, Q4NX_COL_STRIDE> block_vec_in_0_1 = aie::load_v<Q4NX_COL_STRIDE>(cur_block_ptr); cur_block_ptr += Q4NX_COL_STRIDE;
            aie::vector<bfloat16, Q4NX_COL_STRIDE*2> data_dequant_0_1 = mxfp4_to_bf16(block_vec_in_0_1);

            aie::vector<uint8_t, Q4NX_COL_STRIDE> block_vec_in_0_2 = aie::load_v<Q4NX_COL_STRIDE>(cur_block_ptr); cur_block_ptr += Q4NX_COL_STRIDE;
            aie::vector<bfloat16, Q4NX_COL_STRIDE*2> data_dequant_0_2 = mxfp4_to_bf16(block_vec_in_0_2);

            aie::vector<uint8_t, Q4NX_COL_STRIDE> block_vec_in_0_3 = aie::load_v<Q4NX_COL_STRIDE>(cur_block_ptr); cur_block_ptr += Q4NX_COL_STRIDE;
            aie::vector<bfloat16, Q4NX_COL_STRIDE*2> data_dequant_0_3 = mxfp4_to_bf16(block_vec_in_0_3);

            aie::vector<uint8_t, Q4NX_COL_STRIDE> block_vec_in_0_4 = aie::load_v<Q4NX_COL_STRIDE>(cur_block_ptr); cur_block_ptr += Q4NX_COL_STRIDE;
            aie::vector<bfloat16, Q4NX_COL_STRIDE*2> data_dequant_0_4 = mxfp4_to_bf16(block_vec_in_0_4);

            aie::vector<uint8_t, Q4NX_COL_STRIDE> block_vec_in_0_5 = aie::load_v<Q4NX_COL_STRIDE>(cur_block_ptr); cur_block_ptr += Q4NX_COL_STRIDE;
            aie::vector<bfloat16, Q4NX_COL_STRIDE*2> data_dequant_0_5 = mxfp4_to_bf16(block_vec_in_0_5);

            aie::vector<uint8_t, Q4NX_COL_STRIDE> block_vec_in_0_6 = aie::load_v<Q4NX_COL_STRIDE>(cur_block_ptr); cur_block_ptr += Q4NX_COL_STRIDE;
            aie::vector<bfloat16, Q4NX_COL_STRIDE*2> data_dequant_0_6 = mxfp4_to_bf16(block_vec_in_0_6);

            aie::vector<uint8_t, Q4NX_COL_STRIDE> block_vec_in_0_7 = aie::load_v<Q4NX_COL_STRIDE>(cur_block_ptr); cur_block_ptr += Q4NX_COL_STRIDE;
            aie::vector<bfloat16, Q4NX_COL_STRIDE*2> data_dequant_0_7 = mxfp4_to_bf16(block_vec_in_0_7);

            aie::vector<uint8_t, Q4NX_COL_STRIDE> block_vec_in_0_8 = aie::load_v<Q4NX_COL_STRIDE>(cur_block_ptr); cur_block_ptr += Q4NX_COL_STRIDE;
            aie::vector<bfloat16, Q4NX_COL_STRIDE*2> data_dequant_0_8 = mxfp4_to_bf16(block_vec_in_0_8);

            aie::vector<uint8_t, Q4NX_COL_STRIDE> block_vec_in_0_9 = aie::load_v<Q4NX_COL_STRIDE>(cur_block_ptr); cur_block_ptr += Q4NX_COL_STRIDE;
            aie::vector<bfloat16, Q4NX_COL_STRIDE*2> data_dequant_0_9 = mxfp4_to_bf16(block_vec_in_0_9);

            aie::vector<uint8_t, Q4NX_COL_STRIDE> block_vec_in_0_10 = aie::load_v<Q4NX_COL_STRIDE>(cur_block_ptr); cur_block_ptr += Q4NX_COL_STRIDE;
            aie::vector<bfloat16, Q4NX_COL_STRIDE*2> data_dequant_0_10 = mxfp4_to_bf16(block_vec_in_0_10);

            aie::vector<uint8_t, Q4NX_COL_STRIDE> block_vec_in_0_11 = aie::load_v<Q4NX_COL_STRIDE>(cur_block_ptr); cur_block_ptr += Q4NX_COL_STRIDE;
            aie::vector<bfloat16, Q4NX_COL_STRIDE*2> data_dequant_0_11 = mxfp4_to_bf16(block_vec_in_0_11);

            aie::vector<uint8_t, Q4NX_COL_STRIDE> block_vec_in_0_12 = aie::load_v<Q4NX_COL_STRIDE>(cur_block_ptr); cur_block_ptr += Q4NX_COL_STRIDE;
            aie::vector<bfloat16, Q4NX_COL_STRIDE*2> data_dequant_0_12 = mxfp4_to_bf16(block_vec_in_0_12);

            aie::vector<uint8_t, Q4NX_COL_STRIDE> block_vec_in_0_13 = aie::load_v<Q4NX_COL_STRIDE>(cur_block_ptr); cur_block_ptr += Q4NX_COL_STRIDE;
            aie::vector<bfloat16, Q4NX_COL_STRIDE*2> data_dequant_0_13 = mxfp4_to_bf16(block_vec_in_0_13);

            aie::vector<uint8_t, Q4NX_COL_STRIDE> block_vec_in_0_14 = aie::load_v<Q4NX_COL_STRIDE>(cur_block_ptr); cur_block_ptr += Q4NX_COL_STRIDE;
            aie::vector<bfloat16, Q4NX_COL_STRIDE*2> data_dequant_0_14 = mxfp4_to_bf16(block_vec_in_0_14);

            aie::vector<uint8_t, Q4NX_COL_STRIDE> block_vec_in_0_15 = aie::load_v<Q4NX_COL_STRIDE>(cur_block_ptr); cur_block_ptr += Q4NX_COL_STRIDE;
            aie::vector<bfloat16, Q4NX_COL_STRIDE*2> data_dequant_0_15 = mxfp4_to_bf16(block_vec_in_0_15);


            // --- 3. Extract all vector parts ---
            aie::vector<bfloat16, Q4NX_COL_STRIDE> part1_0 = data_dequant_0_0.template extract<Q4NX_COL_STRIDE>(0);
            aie::vector<bfloat16, Q4NX_COL_STRIDE> part2_0 = data_dequant_0_0.template extract<Q4NX_COL_STRIDE>(1);
            aie::vector<bfloat16, Q4NX_COL_STRIDE> part1_1 = data_dequant_0_1.template extract<Q4NX_COL_STRIDE>(0);
            aie::vector<bfloat16, Q4NX_COL_STRIDE> part2_1 = data_dequant_0_1.template extract<Q4NX_COL_STRIDE>(1);
            aie::vector<bfloat16, Q4NX_COL_STRIDE> part1_2 = data_dequant_0_2.template extract<Q4NX_COL_STRIDE>(0);
            aie::vector<bfloat16, Q4NX_COL_STRIDE> part2_2 = data_dequant_0_2.template extract<Q4NX_COL_STRIDE>(1);
            aie::vector<bfloat16, Q4NX_COL_STRIDE> part1_3 = data_dequant_0_3.template extract<Q4NX_COL_STRIDE>(0);
            aie::vector<bfloat16, Q4NX_COL_STRIDE> part2_3 = data_dequant_0_3.template extract<Q4NX_COL_STRIDE>(1);
            aie::vector<bfloat16, Q4NX_COL_STRIDE> part1_4 = data_dequant_0_4.template extract<Q4NX_COL_STRIDE>(0);
            aie::vector<bfloat16, Q4NX_COL_STRIDE> part2_4 = data_dequant_0_4.template extract<Q4NX_COL_STRIDE>(1);
            aie::vector<bfloat16, Q4NX_COL_STRIDE> part1_5 = data_dequant_0_5.template extract<Q4NX_COL_STRIDE>(0);
            aie::vector<bfloat16, Q4NX_COL_STRIDE> part2_5 = data_dequant_0_5.template extract<Q4NX_COL_STRIDE>(1);
            aie::vector<bfloat16, Q4NX_COL_STRIDE> part1_6 = data_dequant_0_6.template extract<Q4NX_COL_STRIDE>(0);
            aie::vector<bfloat16, Q4NX_COL_STRIDE> part2_6 = data_dequant_0_6.template extract<Q4NX_COL_STRIDE>(1);
            aie::vector<bfloat16, Q4NX_COL_STRIDE> part1_7 = data_dequant_0_7.template extract<Q4NX_COL_STRIDE>(0);
            aie::vector<bfloat16, Q4NX_COL_STRIDE> part2_7 = data_dequant_0_7.template extract<Q4NX_COL_STRIDE>(1);
            aie::vector<bfloat16, Q4NX_COL_STRIDE> part1_8 = data_dequant_0_8.template extract<Q4NX_COL_STRIDE>(0);
            aie::vector<bfloat16, Q4NX_COL_STRIDE> part2_8 = data_dequant_0_8.template extract<Q4NX_COL_STRIDE>(1);
            aie::vector<bfloat16, Q4NX_COL_STRIDE> part1_9 = data_dequant_0_9.template extract<Q4NX_COL_STRIDE>(0);
            aie::vector<bfloat16, Q4NX_COL_STRIDE> part2_9 = data_dequant_0_9.template extract<Q4NX_COL_STRIDE>(1);
            aie::vector<bfloat16, Q4NX_COL_STRIDE> part1_10 = data_dequant_0_10.template extract<Q4NX_COL_STRIDE>(0);
            aie::vector<bfloat16, Q4NX_COL_STRIDE> part2_10 = data_dequant_0_10.template extract<Q4NX_COL_STRIDE>(1);
            aie::vector<bfloat16, Q4NX_COL_STRIDE> part1_11 = data_dequant_0_11.template extract<Q4NX_COL_STRIDE>(0);
            aie::vector<bfloat16, Q4NX_COL_STRIDE> part2_11 = data_dequant_0_11.template extract<Q4NX_COL_STRIDE>(1);
            aie::vector<bfloat16, Q4NX_COL_STRIDE> part1_12 = data_dequant_0_12.template extract<Q4NX_COL_STRIDE>(0);
            aie::vector<bfloat16, Q4NX_COL_STRIDE> part2_12 = data_dequant_0_12.template extract<Q4NX_COL_STRIDE>(1);
            aie::vector<bfloat16, Q4NX_COL_STRIDE> part1_13 = data_dequant_0_13.template extract<Q4NX_COL_STRIDE>(0);
            aie::vector<bfloat16, Q4NX_COL_STRIDE> part2_13 = data_dequant_0_13.template extract<Q4NX_COL_STRIDE>(1);
            aie::vector<bfloat16, Q4NX_COL_STRIDE> part1_14 = data_dequant_0_14.template extract<Q4NX_COL_STRIDE>(0);
            aie::vector<bfloat16, Q4NX_COL_STRIDE> part2_14 = data_dequant_0_14.template extract<Q4NX_COL_STRIDE>(1);
            aie::vector<bfloat16, Q4NX_COL_STRIDE> part1_15 = data_dequant_0_15.template extract<Q4NX_COL_STRIDE>(0);
            aie::vector<bfloat16, Q4NX_COL_STRIDE> part2_15 = data_dequant_0_15.template extract<Q4NX_COL_STRIDE>(1);



            // --- 1. Load all scalars first ---
            bfloat16 scalar_a_0 = in_vec_scalar.get(0);
            bfloat16 scalar_b_0 = in_vec_scalar.get(1);
            bfloat16 scalar_a_1 = in_vec_scalar.get(2);
            bfloat16 scalar_b_1 = in_vec_scalar.get(3);
            bfloat16 scalar_a_2 = in_vec_scalar.get(4);
            bfloat16 scalar_b_2 = in_vec_scalar.get(5);
            bfloat16 scalar_a_3 = in_vec_scalar.get(6);
            bfloat16 scalar_b_3 = in_vec_scalar.get(7);
            bfloat16 scalar_a_4 = in_vec_scalar.get(8);
            bfloat16 scalar_b_4 = in_vec_scalar.get(9);
            bfloat16 scalar_a_5 = in_vec_scalar.get(10);
            bfloat16 scalar_b_5 = in_vec_scalar.get(11);
            bfloat16 scalar_a_6 = in_vec_scalar.get(12);
            bfloat16 scalar_b_6 = in_vec_scalar.get(13);
            bfloat16 scalar_a_7 = in_vec_scalar.get(14);
            bfloat16 scalar_b_7 = in_vec_scalar.get(15);
            bfloat16 scalar_a_8 = in_vec_scalar.get(16);
            bfloat16 scalar_b_8 = in_vec_scalar.get(17);
            bfloat16 scalar_a_9 = in_vec_scalar.get(18);
            bfloat16 scalar_b_9 = in_vec_scalar.get(19);
            bfloat16 scalar_a_10 = in_vec_scalar.get(20);
            bfloat16 scalar_b_10 = in_vec_scalar.get(21);
            bfloat16 scalar_a_11 = in_vec_scalar.get(22);
            bfloat16 scalar_b_11 = in_vec_scalar.get(23);
            bfloat16 scalar_a_12 = in_vec_scalar.get(24);
            bfloat16 scalar_b_12 = in_vec_scalar.get(25);
            bfloat16 scalar_a_13 = in_vec_scalar.get(26);
            bfloat16 scalar_b_13 = in_vec_scalar.get(27);
            bfloat16 scalar_a_14 = in_vec_scalar.get(28);
            bfloat16 scalar_b_14 = in_vec_scalar.get(29);
            bfloat16 scalar_a_15 = in_vec_scalar.get(30);
            bfloat16 scalar_b_15 = in_vec_scalar.get(31);


            // --- 4. Perform all MAC operations ---
            // The accumulator 'temp_acc' is updated sequentially
            temp_acc = aie::mac(temp_acc, part1_0, scalar_a_0);
            temp_acc = aie::mac(temp_acc, part2_0, scalar_b_0);
            temp_acc = aie::mac(temp_acc, part1_1, scalar_a_1);
            temp_acc = aie::mac(temp_acc, part2_1, scalar_b_1);
            temp_acc = aie::mac(temp_acc, part1_2, scalar_a_2);
            temp_acc = aie::mac(temp_acc, part2_2, scalar_b_2);
            temp_acc = aie::mac(temp_acc, part1_3, scalar_a_3);
            temp_acc = aie::mac(temp_acc, part2_3, scalar_b_3);
            temp_acc = aie::mac(temp_acc, part1_4, scalar_a_4);
            temp_acc = aie::mac(temp_acc, part2_4, scalar_b_4);
            temp_acc = aie::mac(temp_acc, part1_5, scalar_a_5);
            temp_acc = aie::mac(temp_acc, part2_5, scalar_b_5);
            temp_acc = aie::mac(temp_acc, part1_6, scalar_a_6);
            temp_acc = aie::mac(temp_acc, part2_6, scalar_b_6);
            temp_acc = aie::mac(temp_acc, part1_7, scalar_a_7);
            temp_acc = aie::mac(temp_acc, part2_7, scalar_b_7);
            temp_acc = aie::mac(temp_acc, part1_8, scalar_a_8);
            temp_acc = aie::mac(temp_acc, part2_8, scalar_b_8);
            temp_acc = aie::mac(temp_acc, part1_9, scalar_a_9);
            temp_acc = aie::mac(temp_acc, part2_9, scalar_b_9);
            temp_acc = aie::mac(temp_acc, part1_10, scalar_a_10);
            temp_acc = aie::mac(temp_acc, part2_10, scalar_b_10);
            temp_acc = aie::mac(temp_acc, part1_11, scalar_a_11);
            temp_acc = aie::mac(temp_acc, part2_11, scalar_b_11);
            temp_acc = aie::mac(temp_acc, part1_12, scalar_a_12);
            temp_acc = aie::mac(temp_acc, part2_12, scalar_b_12);
            temp_acc = aie::mac(temp_acc, part1_13, scalar_a_13);
            temp_acc = aie::mac(temp_acc, part2_13, scalar_b_13);
            temp_acc = aie::mac(temp_acc, part1_14, scalar_a_14);
            temp_acc = aie::mac(temp_acc, part2_14, scalar_b_14);
            temp_acc = aie::mac(temp_acc, part1_15, scalar_a_15);
            temp_acc = aie::mac(temp_acc, part2_15, scalar_b_15);
        }

            // AIE_LOOP_UNROLL_FULL
            // for(int k = 0; k < 16; k++){   // repeat for 16 = (32/2) times, each times load 16 value = Q4NX_COL_STRIDEx2

            //     bfloat16 scalar_a = in_vec_scalar.get(k*2);
            //     bfloat16 scalar_b = in_vec_scalar.get(k*2+ 1);
            //     aie::vector<uint8_t, Q4NX_COL_STRIDE> block_vec_in_0 = aie::load_v<Q4NX_COL_STRIDE>(cur_block_ptr);
            //     cur_block_ptr+= Q4NX_COL_STRIDE;
            //     aie::vector<bfloat16, Q4NX_COL_STRIDE*2> data_dequant_0 = mxfp4_to_bf16(block_vec_in_0); // because 1 byte has 2 int4 data value



            //     // aie::vector<bfloat16, Q4NX_COL_STRIDE> part1 = aie::filter_even(data_dequant_0);
            //     // aie::vector<bfloat16, Q4NX_COL_STRIDE> part2 = aie::filter_odd(data_dequant_0);


            //     aie::vector<bfloat16, Q4NX_COL_STRIDE> part1  = data_dequant_0.template extract<Q4NX_COL_STRIDE>(0);
            //      aie::vector<bfloat16, Q4NX_COL_STRIDE> part2 = data_dequant_0.template extract<Q4NX_COL_STRIDE>(1);                                
            //     // aie::accum<accfloat, Q4NX_COL_STRIDE> deq_res1 = aie::mul( scalar_dequant, part1 );
            //     // aie::accum<accfloat, Q4NX_COL_STRIDE> deq_res2 = aie::mul( scalar_dequant, part2 );
            
            //     temp_acc = aie::mac(temp_acc,  part1, scalar_a);
            //     temp_acc = aie::mac(temp_acc, part2, scalar_b);
            // }

            res_acc = aie::mac(res_acc,  temp_acc.template to_vector<bfloat16>() , scalar_dequant  );
        }
        aie::store_v(result_acc_ptr, res_acc.template to_vector<float>());
        result_acc_ptr+=16;        

    }



}


