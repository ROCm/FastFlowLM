// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>
#include "MXFP4.h"
#include "aie_kernel_utils.h"


#include "common.h"

#define NPU_M_TILE 32
#define NPU_K_TILE 128

typedef struct {
    bf16 scales[NPU_M_TILE * 4]; // per-group scales and mins
    bf16 mins[NPU_M_TILE * 4]; // per-group scales and mins
    uint4 qs[NPU_M_TILE * NPU_K_TILE];           // 4-bit quantized values
}q4k_block;
template<int m, int k>
void dequant_core(
    bf16 * w_ptr,
    q4k_block * qw_ptr
) {
    constexpr int pr = 16;

    constexpr int pc = 8;
    uint4* qs_ptr = const_cast<uint4*>(qw_ptr->qs);
    
    // precompute chunk sum of B;
    for (int row = 0; row < NPU_M_TILE; row += 16) chess_unroll_loop(2)
    {
        bf16* w_ptr_iter_0 = w_ptr + row * NPU_K_TILE;
        bf16* w_ptr_iter_1 = w_ptr + (row + 8) * NPU_K_TILE;
        for (int sub_chunk = 0; sub_chunk < NPU_K_TILE / 32; sub_chunk++){
            aie::vector<bf16, pr> a_scales = aie::load_v<pr>(qw_ptr->scales + sub_chunk * NPU_M_TILE + row);
            aie::vector<bf16, pr> a_mins = aie::load_v<pr>(qw_ptr->mins + sub_chunk * NPU_M_TILE + row);
            aie::accum<accfloat, pr> a_mins_f32;
            a_mins_f32.from_vector(a_mins);
            for (int cc = 0; cc < 32; cc += pc)chess_prepare_for_pipelining chess_loop_range(4, 4){
                aie::vector<uint4, pr * 8> a_cc = aie::load_v<pr * 8>(qs_ptr);
                qs_ptr += pr * 4;
                aie::vector<float, pr * 8> a_cc_f32 = aie::to_float(a_cc, 0);

                aie::accum<accfloat, pr * 8> a_cc_f32_accum;
                a_cc_f32_accum.from_vector(a_cc_f32);
                aie::vector<bf16, pr * 8> a_cc_bf16 = a_cc_f32_accum.template to_vector<bf16>();


                aie::vector<bf16, pr> a_col_0 = a_cc_bf16.extract<pr>(0);
                aie::vector<bf16, pr> a_col_1 = a_cc_bf16.extract<pr>(1);
                aie::vector<bf16, pr> a_col_2 = a_cc_bf16.extract<pr>(2);    
                aie::vector<bf16, pr> a_col_3 = a_cc_bf16.extract<pr>(3);
                aie::vector<bf16, pr> a_col_4 = a_cc_bf16.extract<pr>(4);
                aie::vector<bf16, pr> a_col_5 = a_cc_bf16.extract<pr>(5);
                aie::vector<bf16, pr> a_col_6 = a_cc_bf16.extract<pr>(6);
                aie::vector<bf16, pr> a_col_7 = a_cc_bf16.extract<pr>(7);
              
                aie::accum<accfloat, pr> a_col_0_accum = a_mins_f32;
                aie::accum<accfloat, pr> a_col_1_accum = a_mins_f32;
                aie::accum<accfloat, pr> a_col_2_accum = a_mins_f32;
                aie::accum<accfloat, pr> a_col_3_accum = a_mins_f32;
                aie::accum<accfloat, pr> a_col_4_accum = a_mins_f32;
                aie::accum<accfloat, pr> a_col_5_accum = a_mins_f32;
                aie::accum<accfloat, pr> a_col_6_accum = a_mins_f32;
                aie::accum<accfloat, pr> a_col_7_accum = a_mins_f32;

                a_col_0_accum = aie::mac(a_col_0_accum, a_scales, a_col_0);
                a_col_1_accum = aie::mac(a_col_1_accum, a_scales, a_col_1);
                a_col_2_accum = aie::mac(a_col_2_accum, a_scales, a_col_2);
                a_col_3_accum = aie::mac(a_col_3_accum, a_scales, a_col_3);
                a_col_4_accum = aie::mac(a_col_4_accum, a_scales, a_col_4);
                a_col_5_accum = aie::mac(a_col_5_accum, a_scales, a_col_5);
                a_col_6_accum = aie::mac(a_col_6_accum, a_scales, a_col_6);
                a_col_7_accum = aie::mac(a_col_7_accum, a_scales, a_col_7);

                a_col_0 = a_col_0_accum.template to_vector<bf16>();
                a_col_1 = a_col_1_accum.template to_vector<bf16>();
                a_col_2 = a_col_2_accum.template to_vector<bf16>();
                a_col_3 = a_col_3_accum.template to_vector<bf16>();
                a_col_4 = a_col_4_accum.template to_vector<bf16>();
                a_col_5 = a_col_5_accum.template to_vector<bf16>();
                a_col_6 = a_col_6_accum.template to_vector<bf16>();
                a_col_7 = a_col_7_accum.template to_vector<bf16>();

                auto a_col_01 = aie::interleave_zip(a_col_0, a_col_1, 8);
                auto a_col_23 = aie::interleave_zip(a_col_2, a_col_3, 8);
                auto a_col_45 = aie::interleave_zip(a_col_4, a_col_5, 8);
                auto a_col_67 = aie::interleave_zip(a_col_6, a_col_7, 8);
                auto a_up = aie::concat(a_col_01.first, a_col_23.first, a_col_45.first, a_col_67.first);
                auto a_down = aie::concat(a_col_01.second, a_col_23.second, a_col_45.second, a_col_67.second);
                a_up = aie::transpose(a_up, 8, 8);
                a_down = aie::transpose(a_down, 8, 8);
                aie::store_v(w_ptr_iter_0, a_up);
                aie::store_v(w_ptr_iter_1, a_down);
                w_ptr_iter_0 += 64;
                w_ptr_iter_1 += 64;
            } // end of cc, 32 columns;
        } // end of sub_chunk, 256 columns;
    }
}

//Note: When data is store in Q4NX 
// Each Q4NX block isQ Q4NX_ROW x Q4NX_Column
// The data in Q4NX is in column_major order, with Q4NX_COL_STRIDE

//ALSO NOTEL the Q4NX is one-scalar share with 32 data blocks
//NOTE: but scalar is in column major order, with stride of Q4NX_ROW instead
template<unsigned int Q4NX_ROW, unsigned int Q4NX_COL, unsigned int Q4NX_COL_STRIDE>
void dequant_Q4NX_BLOCK_MXFP4_to_bfloat16(
     const uint8_t * data_block_ptr, const  uint8_t* scalar_ptr, bfloat16* output_ptr
){

    static_assert(Q4NX_ROW % Q4NX_COL_STRIDE == 0);
    static_assert(Q4NX_COL_STRIDE == 16);// the most effecitve and what we build on
    static_assert( (Q4NX_COL)%(2*32) == 0);
    // because each call mxfp4_to_bf16 process 16 uint8, aka 32 data blocks


    uint8_t *cur_block_ptr = (uint8_t *)data_block_ptr; // each uint8 contains two 4bit data

    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_RANGE(Q4NX_ROW/ Q4NX_COL_STRIDE, Q4NX_ROW/ Q4NX_COL_STRIDE)
    for(size_t row_idx = 0; row_idx < Q4NX_ROW; row_idx += Q4NX_COL_STRIDE){
        bfloat16* w_ptr_iter_0 = output_ptr + row_idx * Q4NX_COL;
        bfloat16* w_ptr_iter_1 = output_ptr + (row_idx + 8) * Q4NX_COL;
             
        AIE_PREPARE_FOR_PIPELINING
        AIE_LOOP_RANGE(Q4NX_COL/32,Q4NX_COL/32)
        for(size_t col_idx = 0; col_idx < Q4NX_COL; col_idx += 32){

            // 16 q4nx_value = 8byte, thus we actually look at two column of Q4NX_COL_STRIDE(16) at a time
            uint8_t *cur_scalar_ptr = (uint8_t *)scalar_ptr + (col_idx/32) * Q4NX_ROW + row_idx ;
            aie::vector<uint8_t, 16> scalar_vec_in = aie::load_v<16>(cur_scalar_ptr);
            cur_scalar_ptr+=16;
            aie::vector<bfloat16, 16> scalar_dequant = mxfp4_scalar_to_bf16(scalar_vec_in);


            AIE_PREPARE_FOR_PIPELINING
            AIE_LOOP_UNROLL_FULL
            for(int k = 0; k < 16; k+= 4){
                // repeat for 16 = (32/2) times, each times load 16 value = Q4NX_COL_STRIDEx2
                // Each loop get 8 res vector of 1xQ4NX_COL_STRIDE
                aie::vector<uint8_t, 16> block_vec_in_0 = aie::load_v<16>(cur_block_ptr);
                cur_block_ptr+= 16;
                aie::vector<uint8_t, 16> block_vec_in_1 = aie::load_v<16>(cur_block_ptr);
                cur_block_ptr+= 16;
                aie::vector<uint8_t, 16> block_vec_in_2 = aie::load_v<16>(cur_block_ptr);
                cur_block_ptr+= 16;
                aie::vector<uint8_t, 16> block_vec_in_3 = aie::load_v<16>(cur_block_ptr);
                cur_block_ptr+= 16;                

                aie::vector<bfloat16, 32> data_dequant_0 = mxfp4_to_bf16(block_vec_in_0);
                aie::vector<bfloat16, 32> data_dequant_1 = mxfp4_to_bf16(block_vec_in_1);
                aie::vector<bfloat16, 32> data_dequant_2 = mxfp4_to_bf16(block_vec_in_2);
                aie::vector<bfloat16, 32> data_dequant_3 = mxfp4_to_bf16(block_vec_in_3);

                // aie::vector<bfloat16, 16> res1_bf = aie::filter_even(data_dequant_0);
                // aie::vector<bfloat16, 16> res2_bf = aie::filter_odd(data_dequant_0);
                // aie::vector<bfloat16, 16> res3_bf = aie::filter_even(data_dequant_1);
                // aie::vector<bfloat16, 16> res4_bf = aie::filter_odd( data_dequant_1);
                // aie::vector<bfloat16, 16> res5_bf = aie::filter_even(data_dequant_2);
                // aie::vector<bfloat16, 16> res6_bf = aie::filter_odd( data_dequant_2);
                // aie::vector<bfloat16, 16> res7_bf = aie::filter_even(data_dequant_3);
                // aie::vector<bfloat16, 16> res8_bf = aie::filter_odd( data_dequant_3);


                aie::vector<bfloat16, Q4NX_COL_STRIDE> res1_bf  = data_dequant_0.template extract<Q4NX_COL_STRIDE>(0);
                aie::vector<bfloat16, Q4NX_COL_STRIDE> res2_bf =  data_dequant_0.template extract<Q4NX_COL_STRIDE>(1); 
                aie::vector<bfloat16, 16> res3_bf = data_dequant_1.template extract<Q4NX_COL_STRIDE>(0);
                aie::vector<bfloat16, 16> res4_bf = data_dequant_1.template extract<Q4NX_COL_STRIDE>(1);
                aie::vector<bfloat16, 16> res5_bf = data_dequant_2.template extract<Q4NX_COL_STRIDE>(0);
                aie::vector<bfloat16, 16> res6_bf = data_dequant_2.template extract<Q4NX_COL_STRIDE>(1);
                aie::vector<bfloat16, 16> res7_bf = data_dequant_3.template extract<Q4NX_COL_STRIDE>(0);
                aie::vector<bfloat16, 16> res8_bf = data_dequant_3.template extract<Q4NX_COL_STRIDE>(0);


                aie::accum<accfloat, 16> res1 = aie::mul( scalar_dequant, res1_bf );
                aie::accum<accfloat, 16> res2 = aie::mul( scalar_dequant, res2_bf );
                aie::accum<accfloat, 16> res3 = aie::mul( scalar_dequant, res3_bf );
                aie::accum<accfloat, 16> res4 = aie::mul( scalar_dequant, res4_bf );
                aie::accum<accfloat, 16> res5 = aie::mul( scalar_dequant, res5_bf );
                aie::accum<accfloat, 16> res6 = aie::mul( scalar_dequant, res6_bf );
                aie::accum<accfloat, 16> res7 = aie::mul( scalar_dequant, res7_bf );
                aie::accum<accfloat, 16> res8 = aie::mul( scalar_dequant, res8_bf );

                res1_bf = res1.template to_vector<bfloat16>();
                res2_bf = res2.template to_vector<bfloat16>();
                res3_bf = res3.template to_vector<bfloat16>();
                res4_bf = res4.template to_vector<bfloat16>();
                res5_bf = res5.template to_vector<bfloat16>();
                res6_bf = res6.template to_vector<bfloat16>();                
                res7_bf = res7.template to_vector<bfloat16>();
                res8_bf = res8.template to_vector<bfloat16>();
                // now do interleave stuff


                
                
                auto a_col_01 = aie::interleave_zip(res1_bf, res2_bf, 8);
                auto a_col_23 = aie::interleave_zip(res3_bf, res4_bf, 8);
                auto a_col_45 = aie::interleave_zip(res5_bf, res6_bf, 8);
                auto a_col_67 = aie::interleave_zip(res7_bf, res8_bf, 8);
                auto a_up = aie::concat(a_col_01.first, a_col_23.first, a_col_45.first, a_col_67.first);
                auto a_down = aie::concat(a_col_01.second, a_col_23.second, a_col_45.second, a_col_67.second);
                a_up = aie::transpose(a_up, 8, 8);
                a_down = aie::transpose(a_down, 8, 8);

                aie::store_v(w_ptr_iter_0, a_up);
                aie::store_v(w_ptr_iter_1, a_down);
                w_ptr_iter_0 += 8*8;
                w_ptr_iter_1 +=8*8;
            }


        }   

    }





}



extern "C"{
  

    // // input 16 uint8, which represnt 32 values (4 bit per value in MXFP4)
    // // dequant and store 32 bfloat16 data 
    // void dequant_32_MXFP4_data_to_bfloat16(uint8_t *input, 
    //     bfloat16* output
    // ){  
    //     aie::vector<uint8_t, 16> vec_in = aie::load_v<16>(input);
    //     aie::vector<bfloat16, 32> res =  mxfp4_to_bf16(vec_in);

    //     aie::store_v(output,res );
  
    // }



    // input 16 uint8, which represnt 32 values (4 bit per value in MXFP4)
    // dequant and store 32 bfloat16 data 
    void dequant_32_MXFP4_data_to_bfloat16(uint8_t *input, 
        bfloat16* output
    ){  
        aie::vector<uint8_t, 16> vec_in = aie::load_v<16>(input);
        aie::vector<bfloat16, 16> res =  mxfp4_scalar_to_bf16(vec_in); // only store 16 of it

        aie::store_v(output,res );
  
    }




    // void dequant_Q4NX_MXFP4_block_32_256_16(uint8_t* data_scalar_block, bfloat16* output){
    //     constexpr int Q4NX_ROW = 32;
    //     constexpr int Q4NX_COL = 256;
    //     constexpr int Q4NX_COL_STRIDE = 16;
        
    //     constexpr int  data_block_offset =   Q4NX_ROW * (Q4NX_COL/32) * 4;
        
    //     dequant_Q4NX_BLOCK_MXFP4_to_bfloat16<Q4NX_ROW, Q4NX_COL, Q4NX_COL_STRIDE>(
    //         data_scalar_block + data_block_offset, 
    //         data_scalar_block, output);
    //     // for(int i = 0; i < 32*256; i++){
    //     //     *(output + i) = 1.0; // for debugging
    //     // }
    
    // }
    void zero_int32_1(int32 *pt)
    {
        *pt = 0;
    }

    void dequant_Q4NX_block_32_128_16(uint8_t* data_scalar_block, bfloat16* output,
        int32_t *rtp_mode
    ){
        constexpr int Q4NX_ROW = 32;
        constexpr int Q4NX_COL = 128;
        constexpr int Q4NX_COL_STRIDE = 16;
        
        constexpr int  data_block_offset =   Q4NX_ROW * (Q4NX_COL/32) * 4;
        if(*rtp_mode == 1){
            dequant_Q4NX_BLOCK_MXFP4_to_bfloat16<Q4NX_ROW, Q4NX_COL, Q4NX_COL_STRIDE>(
                data_scalar_block + data_block_offset, 
                data_scalar_block, output);
        }else{
            dequant_core<Q4NX_ROW, Q4NX_COL>(output, (q4k_block*)data_scalar_block);
        }

        // for(int i = 0; i < 32*256; i++){
        //     *(output + i) = 1.0; // for debugging
        // }
    
    }

}





