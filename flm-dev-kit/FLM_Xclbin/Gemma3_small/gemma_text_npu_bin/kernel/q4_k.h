// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __Q4_K_H__
#define __Q4_K_H__
#include "qmm.h"
#include "aie_kernel_utils.h"
// #define REFERENCE_CODE


typedef struct {
    bf16 scales[NPU_M_TILE * 4]; // per-group scales and mins
    bf16 mins[NPU_M_TILE * 4]; // per-group scales and mins
    uint4 qs[NPU_M_TILE * NPU_K_TILE];           // 4-bit quantized values
}q4k_block;

template <int M, int N>
 void _qmm_q4k_bf16(
    const q4k_block * A,
    const bf16 * B,
    float * c,
    bf16* b_col_reduce_buffer
) { 

    constexpr int pr = 16;
    constexpr int pc = 8;
    const q4k_block * A_q4k = (q4k_block *)A;
    const uint4* qs_ptr = A_q4k->qs;
    // precompute chunk sum of B;

    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_RANGE(M/pr, M/pr)
    for (int row = 0; row < M; row += pr){
        aie::accum<accfloat, pr> c_accum;
        c_accum.from_vector(aie::load_v<pr>(c + row));
        const bf16* it_B =B;

        uint32_t scale_min_offset = row;
        bf16* cur_b_reduce_Buf = b_col_reduce_buffer;

        
        AIE_PREPARE_FOR_PIPELINING
        AIE_LOOP_RANGE( N / 32, N / 32 )
        for (int sub_chunk = 0; sub_chunk < N / 32; sub_chunk++)AIE_LOOP_FLATTEN  {

            aie::vector<bf16, 32> b_col = aie::load_v<32>(it_B);
            it_B += 32;

            
            aie::vector<bf16, pr> a_scales = aie::load_v<pr>(A_q4k->scales +scale_min_offset);
            aie::vector<bf16, pr> a_mins = aie::load_v<pr>(A_q4k->mins + scale_min_offset);

            scale_min_offset+=M;
            bf16 reduce_b = *(cur_b_reduce_Buf);
            cur_b_reduce_Buf++;

            aie::accum<accfloat, pr> temp_acc;  
                   
            {

                aie::accum<accfloat, pr * 8> a_cc_f32_accum_0;
                aie::accum<accfloat, pr * 8> a_cc_f32_accum_1;
                aie::accum<accfloat, pr * 8> a_cc_f32_accum_2;  
                aie::accum<accfloat, pr * 8> a_cc_f32_accum_3;   
                // Load all a vectors
                aie::vector<uint4, pr * 8> a_cc_0 = aie::load_v<pr * 8>(qs_ptr);
                qs_ptr += pr * 4;
                aie::vector<float, pr * 8> a_cc_f32_0 = aie::to_float(a_cc_0, 0);     

                a_cc_f32_accum_0.from_vector(a_cc_f32_0);
                aie::vector<bf16, pr * 8> a_cc_bf16_0 = a_cc_f32_accum_0.template to_vector<bf16>();

                aie::vector<uint4, pr * 8> a_cc_1 = aie::load_v<pr * 8>(qs_ptr);
                qs_ptr += pr * 4;
                aie::vector<float, pr * 8> a_cc_f32_1 = aie::to_float(a_cc_1, 0);

                a_cc_f32_accum_1.from_vector(a_cc_f32_1);
                aie::vector<bf16, pr * 8> a_cc_bf16_1 = a_cc_f32_accum_1.template to_vector<bf16>();                

                aie::vector<uint4, pr * 8> a_cc_2 = aie::load_v<pr * 8>(qs_ptr);
                qs_ptr += pr * 4;
                aie::vector<float, pr * 8> a_cc_f32_2 = aie::to_float(a_cc_2, 0);

                a_cc_f32_accum_2.from_vector(a_cc_f32_2);
                aie::vector<bf16, pr * 8> a_cc_bf16_2 = a_cc_f32_accum_2.template to_vector<bf16>();


                aie::vector<uint4, pr * 8> a_cc_3 = aie::load_v<pr * 8>(qs_ptr);
                qs_ptr += pr * 4;
                aie::vector<float, pr * 8> a_cc_f32_3 = aie::to_float(a_cc_3, 0);

                a_cc_f32_accum_3.from_vector(a_cc_f32_3);
                aie::vector<bf16, pr * 8> a_cc_bf16_3 = a_cc_f32_accum_3.template to_vector<bf16>();


  

                // Extract all columns
                aie::vector<bf16, pr> a_col_0_0 = a_cc_bf16_0.extract<pr>(0);
                aie::vector<bf16, pr> a_col_1_0 = a_cc_bf16_0.extract<pr>(1);
                aie::vector<bf16, pr> a_col_2_0 = a_cc_bf16_0.extract<pr>(2);
                aie::vector<bf16, pr> a_col_3_0 = a_cc_bf16_0.extract<pr>(3);
                aie::vector<bf16, pr> a_col_4_0 = a_cc_bf16_0.extract<pr>(4);
                aie::vector<bf16, pr> a_col_5_0 = a_cc_bf16_0.extract<pr>(5);
                aie::vector<bf16, pr> a_col_6_0 = a_cc_bf16_0.extract<pr>(6);
                aie::vector<bf16, pr> a_col_7_0 = a_cc_bf16_0.extract<pr>(7);

                aie::vector<bf16, pr> a_col_0_1 = a_cc_bf16_1.extract<pr>(0);
                aie::vector<bf16, pr> a_col_1_1 = a_cc_bf16_1.extract<pr>(1);
                aie::vector<bf16, pr> a_col_2_1 = a_cc_bf16_1.extract<pr>(2);
                aie::vector<bf16, pr> a_col_3_1 = a_cc_bf16_1.extract<pr>(3);
                aie::vector<bf16, pr> a_col_4_1 = a_cc_bf16_1.extract<pr>(4);
                aie::vector<bf16, pr> a_col_5_1 = a_cc_bf16_1.extract<pr>(5);
                aie::vector<bf16, pr> a_col_6_1 = a_cc_bf16_1.extract<pr>(6);
                aie::vector<bf16, pr> a_col_7_1 = a_cc_bf16_1.extract<pr>(7);

                aie::vector<bf16, pr> a_col_0_2 = a_cc_bf16_2.extract<pr>(0);
                aie::vector<bf16, pr> a_col_1_2 = a_cc_bf16_2.extract<pr>(1);
                aie::vector<bf16, pr> a_col_2_2 = a_cc_bf16_2.extract<pr>(2);
                aie::vector<bf16, pr> a_col_3_2 = a_cc_bf16_2.extract<pr>(3);
                aie::vector<bf16, pr> a_col_4_2 = a_cc_bf16_2.extract<pr>(4);
                aie::vector<bf16, pr> a_col_5_2 = a_cc_bf16_2.extract<pr>(5);
                aie::vector<bf16, pr> a_col_6_2 = a_cc_bf16_2.extract<pr>(6);
                aie::vector<bf16, pr> a_col_7_2 = a_cc_bf16_2.extract<pr>(7);

                aie::vector<bf16, pr> a_col_0_3 = a_cc_bf16_3.extract<pr>(0);
                aie::vector<bf16, pr> a_col_1_3 = a_cc_bf16_3.extract<pr>(1);
                aie::vector<bf16, pr> a_col_2_3 = a_cc_bf16_3.extract<pr>(2);
                aie::vector<bf16, pr> a_col_3_3 = a_cc_bf16_3.extract<pr>(3);
                aie::vector<bf16, pr> a_col_4_3 = a_cc_bf16_3.extract<pr>(4);
                aie::vector<bf16, pr> a_col_5_3 = a_cc_bf16_3.extract<pr>(5);
                aie::vector<bf16, pr> a_col_6_3 = a_cc_bf16_3.extract<pr>(6);
                aie::vector<bf16, pr> a_col_7_3 = a_cc_bf16_3.extract<pr>(7);

                // Load all b scalars
                bfloat16 scalar_0_0 = b_col.get(0);
                bfloat16 scalar_1_0 = b_col.get(1);
                bfloat16 scalar_2_0 = b_col.get(2);
                bfloat16 scalar_3_0 = b_col.get(3);
                bfloat16 scalar_4_0 = b_col.get(4);
                bfloat16 scalar_5_0 = b_col.get(5);
                bfloat16 scalar_6_0 = b_col.get(6);
                bfloat16 scalar_7_0 = b_col.get(7);

                bfloat16 scalar_0_1 = b_col.get(8);
                bfloat16 scalar_1_1 = b_col.get(9);
                bfloat16 scalar_2_1 = b_col.get(10);
                bfloat16 scalar_3_1 = b_col.get(11);
                bfloat16 scalar_4_1 = b_col.get(12);
                bfloat16 scalar_5_1 = b_col.get(13);
                bfloat16 scalar_6_1 = b_col.get(14);
                bfloat16 scalar_7_1 = b_col.get(15);

                bfloat16 scalar_0_2 = b_col.get(16);
                bfloat16 scalar_1_2 = b_col.get(17);
                bfloat16 scalar_2_2 = b_col.get(18);
                bfloat16 scalar_3_2 = b_col.get(19);
                bfloat16 scalar_4_2 = b_col.get(20);
                bfloat16 scalar_5_2 = b_col.get(21);
                bfloat16 scalar_6_2 = b_col.get(22);
                bfloat16 scalar_7_2 = b_col.get(23);

                bfloat16 scalar_0_3 = b_col.get(24);
                bfloat16 scalar_1_3 = b_col.get(25);
                bfloat16 scalar_2_3 = b_col.get(26);
                bfloat16 scalar_3_3 = b_col.get(27);
                bfloat16 scalar_4_3 = b_col.get(28);
                bfloat16 scalar_5_3 = b_col.get(29);
                bfloat16 scalar_6_3 = b_col.get(30);
                bfloat16 scalar_7_3 = b_col.get(31);

                // All MAC operations
                temp_acc = aie::mul(a_col_0_0, scalar_0_0);
                temp_acc = aie::mac(temp_acc, a_col_1_0, scalar_1_0);
                temp_acc = aie::mac(temp_acc, a_col_2_0, scalar_2_0);
                temp_acc = aie::mac(temp_acc, a_col_3_0, scalar_3_0);
                temp_acc = aie::mac(temp_acc, a_col_4_0, scalar_4_0);
                temp_acc = aie::mac(temp_acc, a_col_5_0, scalar_5_0);
                temp_acc = aie::mac(temp_acc, a_col_6_0, scalar_6_0);
                temp_acc = aie::mac(temp_acc, a_col_7_0, scalar_7_0);
                temp_acc = aie::mac(temp_acc, a_col_0_1, scalar_0_1);
                temp_acc = aie::mac(temp_acc, a_col_1_1, scalar_1_1);
                temp_acc = aie::mac(temp_acc, a_col_2_1, scalar_2_1);
                temp_acc = aie::mac(temp_acc, a_col_3_1, scalar_3_1);
                temp_acc = aie::mac(temp_acc, a_col_4_1, scalar_4_1);
                temp_acc = aie::mac(temp_acc, a_col_5_1, scalar_5_1);
                temp_acc = aie::mac(temp_acc, a_col_6_1, scalar_6_1);
                temp_acc = aie::mac(temp_acc, a_col_7_1, scalar_7_1);
                temp_acc = aie::mac(temp_acc, a_col_0_2, scalar_0_2);
                temp_acc = aie::mac(temp_acc, a_col_1_2, scalar_1_2);
                temp_acc = aie::mac(temp_acc, a_col_2_2, scalar_2_2);
                temp_acc = aie::mac(temp_acc, a_col_3_2, scalar_3_2);
                temp_acc = aie::mac(temp_acc, a_col_4_2, scalar_4_2);
                temp_acc = aie::mac(temp_acc, a_col_5_2, scalar_5_2);
                temp_acc = aie::mac(temp_acc, a_col_6_2, scalar_6_2);
                temp_acc = aie::mac(temp_acc, a_col_7_2, scalar_7_2);
                temp_acc = aie::mac(temp_acc, a_col_0_3, scalar_0_3);
                temp_acc = aie::mac(temp_acc, a_col_1_3, scalar_1_3);
                temp_acc = aie::mac(temp_acc, a_col_2_3, scalar_2_3);
                temp_acc = aie::mac(temp_acc, a_col_3_3, scalar_3_3);
                temp_acc = aie::mac(temp_acc, a_col_4_3, scalar_4_3);
                temp_acc = aie::mac(temp_acc, a_col_5_3, scalar_5_3);
                temp_acc = aie::mac(temp_acc, a_col_6_3, scalar_6_3);
                temp_acc = aie::mac(temp_acc, a_col_7_3, scalar_7_3);

            } // end of cc, 32 columns (manually unrolled);

            // c_accum = aie::mac(c_accum, c_local_accum.template to_vector<bf16>(), a_scales);
            // c_accum = aie::mac(c_accum, a_mins, (bf16)reduce_b);
            c_accum = aie::mac(c_accum, temp_acc.template to_vector<bf16>(), a_scales);
            c_accum = aie::mac(c_accum, a_mins, reduce_b);
            // 
        } // end of sub_chunk, 256 columns;
        aie::vector<float, pr> c_accum_f32 = c_accum.template to_vector<float>();
        aie::store_v(c + row, c_accum_f32);
    }
}

#endif // __Q4_K_H__
