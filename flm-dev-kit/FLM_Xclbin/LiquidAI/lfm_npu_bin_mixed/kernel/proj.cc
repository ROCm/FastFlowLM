// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include "common.h"
#include "q4_k.h"

#include "qmm.h"
// #define REFERENCE_CODE
#include "aie_kernel_utils.h"

typedef struct {
    bf16 scales[NPU_M_TILE * 8]; // per-group scales and mins
    bf16 mins[NPU_M_TILE * 8]; // per-group scales and mins
    int4 qs[NPU_M_TILE * NPU_K_TILE];           // 4-bit quantized values
}q40_block;

typedef struct {
    bf16 scales[NPU_M_TILE * 8]; // per-group scales and mins
    bf16 mins[NPU_M_TILE * 8]; // per-group scales and mins
    uint4 qs[NPU_M_TILE * NPU_K_TILE];           // 4-bit quantized values
}q41_block;


template<int m>
void zero_y(float * y) {
    zero_vectorized<float, m>(y);
}

template <int M, int N>
inline void _qmm_q40_bf16(
    const q40_block * A,
    const bf16 * B,
    float * c
) { 
    static_assert(N == 256, "N must be = 256");
    static_assert(M == 32, "M must be = 32");
    constexpr int pr = 16;
    constexpr int pc = 8;
    const q40_block * A_q40 = (q40_block *)A;
    uint8_t* qs_ptr = (uint8_t*)(A_q40->qs);
    float* c_ptr = const_cast<float*>(c);
    
    // precompute chunk sum of B;
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_RANGE(M/pr, M/pr)
    for (int row = 0; row < M; row += pr)
    {
        aie::accum<accfloat, pr> c_accum;
        c_accum.from_vector(aie::load_v<pr>(c + row));
        bf16* it_B = const_cast<bf16*>(B);
        
        bf16* scales_ptr = const_cast<bf16*>(A_q40->scales + row);

        AIE_PREPARE_FOR_PIPELINING
        AIE_LOOP_RANGE(QK_K / 32, QK_K / 32)
        for (int sub_chunk = 0; sub_chunk < QK_K / 32; sub_chunk++) {
            aie::vector<bf16, 32> b_col = aie::load_v<32>(it_B);
            it_B += 32;
       
            aie::vector<bf16, pr> a_scales = aie::load_v<pr>(scales_ptr);
            scales_ptr += M;

            aie::accum<accfloat, pr> c_local_accum;
            c_local_accum.from_vector(aie::zeros<float, pr>());

            {
                // Load all a vectors
                aie::vector<int4, pr * 8> a_cc_0 = aie::load_v<pr * 8>((int4*)qs_ptr);
                qs_ptr += pr * 4;
                aie::vector<int4, pr * 8> a_cc_1 = aie::load_v<pr * 8>((int4*)qs_ptr);
                qs_ptr += pr * 4;
                aie::vector<int4, pr * 8> a_cc_2 = aie::load_v<pr * 8>((int4*)qs_ptr);
                qs_ptr += pr * 4;
                aie::vector<int4, pr * 8> a_cc_3 = aie::load_v<pr * 8>((int4*)qs_ptr);
                qs_ptr += pr * 4;

                // Convert to float
                aie::vector<float, pr * 8> a_cc_f32_0 = aie::to_float(a_cc_0, 0);
                aie::vector<float, pr * 8> a_cc_f32_1 = aie::to_float(a_cc_1, 0);
                aie::vector<float, pr * 8> a_cc_f32_2 = aie::to_float(a_cc_2, 0);
                aie::vector<float, pr * 8> a_cc_f32_3 = aie::to_float(a_cc_3, 0);

                // Convert to bfloat16
                aie::accum<accfloat, pr * 8> a_cc_f32_accum_0;
                a_cc_f32_accum_0.from_vector(a_cc_f32_0);
                aie::vector<bf16, pr * 8> a_cc_bf16_0 = a_cc_f32_accum_0.template to_vector<bf16>();

                aie::accum<accfloat, pr * 8> a_cc_f32_accum_1;
                a_cc_f32_accum_1.from_vector(a_cc_f32_1);
                aie::vector<bf16, pr * 8> a_cc_bf16_1 = a_cc_f32_accum_1.template to_vector<bf16>();

                aie::accum<accfloat, pr * 8> a_cc_f32_accum_2;
                a_cc_f32_accum_2.from_vector(a_cc_f32_2);
                aie::vector<bf16, pr * 8> a_cc_bf16_2 = a_cc_f32_accum_2.template to_vector<bf16>();

                aie::accum<accfloat, pr * 8> a_cc_f32_accum_3;
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
                c_local_accum = aie::mac(c_local_accum, a_col_0_0, scalar_0_0);
                c_local_accum = aie::mac(c_local_accum, a_col_1_0, scalar_1_0);
                c_local_accum = aie::mac(c_local_accum, a_col_2_0, scalar_2_0);
                c_local_accum = aie::mac(c_local_accum, a_col_3_0, scalar_3_0);
                c_local_accum = aie::mac(c_local_accum, a_col_4_0, scalar_4_0);
                c_local_accum = aie::mac(c_local_accum, a_col_5_0, scalar_5_0);
                c_local_accum = aie::mac(c_local_accum, a_col_6_0, scalar_6_0);
                c_local_accum = aie::mac(c_local_accum, a_col_7_0, scalar_7_0);

                c_local_accum = aie::mac(c_local_accum, a_col_0_1, scalar_0_1);
                c_local_accum = aie::mac(c_local_accum, a_col_1_1, scalar_1_1);
                c_local_accum = aie::mac(c_local_accum, a_col_2_1, scalar_2_1);
                c_local_accum = aie::mac(c_local_accum, a_col_3_1, scalar_3_1);
                c_local_accum = aie::mac(c_local_accum, a_col_4_1, scalar_4_1);
                c_local_accum = aie::mac(c_local_accum, a_col_5_1, scalar_5_1);
                c_local_accum = aie::mac(c_local_accum, a_col_6_1, scalar_6_1);
                c_local_accum = aie::mac(c_local_accum, a_col_7_1, scalar_7_1);

                c_local_accum = aie::mac(c_local_accum, a_col_0_2, scalar_0_2);
                c_local_accum = aie::mac(c_local_accum, a_col_1_2, scalar_1_2);
                c_local_accum = aie::mac(c_local_accum, a_col_2_2, scalar_2_2);
                c_local_accum = aie::mac(c_local_accum, a_col_3_2, scalar_3_2);
                c_local_accum = aie::mac(c_local_accum, a_col_4_2, scalar_4_2);
                c_local_accum = aie::mac(c_local_accum, a_col_5_2, scalar_5_2);
                c_local_accum = aie::mac(c_local_accum, a_col_6_2, scalar_6_2);
                c_local_accum = aie::mac(c_local_accum, a_col_7_2, scalar_7_2);

                c_local_accum = aie::mac(c_local_accum, a_col_0_3, scalar_0_3);
                c_local_accum = aie::mac(c_local_accum, a_col_1_3, scalar_1_3);
                c_local_accum = aie::mac(c_local_accum, a_col_2_3, scalar_2_3);
                c_local_accum = aie::mac(c_local_accum, a_col_3_3, scalar_3_3);
                c_local_accum = aie::mac(c_local_accum, a_col_4_3, scalar_4_3);
                c_local_accum = aie::mac(c_local_accum, a_col_5_3, scalar_5_3);
                c_local_accum = aie::mac(c_local_accum, a_col_6_3, scalar_6_3);
                c_local_accum = aie::mac(c_local_accum, a_col_7_3, scalar_7_3);

            } // end of cc, 32 columns (manually unrolled);
            c_accum = aie::mac(c_accum, c_local_accum.template to_vector<bf16>(), a_scales);
        } // end of sub_chunk, 256 columns;
        aie::vector<float, pr> c_accum_f32 = c_accum.template to_vector<float>();
        aie::store_v(c_ptr, c_accum_f32);
        c_ptr += pr;
    }
}

template <int M, int N>
void _qmm_q41_bf16(
    const q41_block * A,
    const bf16 * B,
    float * c,
    bf16* b_col_reduce_buffer
) { 

    constexpr int pr = 16;
    constexpr int pc = 8;
    const q41_block * A_q41 = (q41_block *)A;
    const uint4* qs_ptr = A_q41->qs;
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

            
            aie::vector<bf16, pr> a_scales = aie::load_v<pr>(A_q41->scales +scale_min_offset);
            aie::vector<bf16, pr> a_mins = aie::load_v<pr>(A_q41->mins + scale_min_offset);

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

template<int m>
inline void copy_y(bf16 * y, const float * y_acc) {
    for (int i = 0; i < m / 16; i++){
        aie::vector<float, 16> y_vec = aie::load_v<16>(y_acc + i * 16);
        aie::accum<accfloat, 16> acc;
        acc.from_vector(y_vec);
        aie::store_v(y + i * 16, acc.template to_vector<bf16>());
    }
}
constexpr int x_prod_lock = 0;
constexpr int x_cons_lock = 1;
constexpr int w_prod_lock = 2;
constexpr int w_cons_lock = 3;
constexpr int y_prod_lock = 4;
constexpr int y_cons_lock = 5;
constexpr int rtp_available_lock = 6;
// phase 1: qkv proj, D -> (D + D // 2) to RoPE
// phase 2: o_proj, D -> D to rms_residual
// phase 3: up/gate proj, D -> D * 8 to MT_GLU
// phase 4: down proj, D * 4 -> D, to rms_residual
// phase 4 is special, do not clear y to zero

AIE_PING_FLAG is_y_ping = false;
AIE_PING_FLAG is_wx_ping = false;
float y_acc[32] = {0};

template<int M, int K, int m, int k>
void linear_proj_iD_q4_0(
    bf16 * x_ping,
    bf16 * x_pong,
    q40_block * w_ping,
    q40_block * w_pong,
    bf16 * y_ping,
    bf16 * y_pong,
    const uint32 pkt_id
) {
    aie_pkt_word* pkt_id_ping = reinterpret_cast<aie_pkt_word*>(y_ping + 14);
    aie_pkt_word* pkt_id_pong = reinterpret_cast<aie_pkt_word*>(y_pong + 14);
    // Under Peano y_acc is not aligned to the vector width, so its vector stores also
    // write the bytes of these flags; keep them in locals for the whole call.
#if defined(__chess__)
    bool &wx = is_wx_ping;
    bool &yp = is_y_ping;
#else
    bool wx = is_wx_ping;
    bool yp = is_y_ping;
#endif
    int i = 0;
    do {
        zero_y<m>(y_acc);
        int j = 0;
        do {
            wx = !wx;
            q40_block* w_using = wx ? w_ping : w_pong;
            bf16* x_using = wx ? x_ping : x_pong;
            _lock_acquire(w_cons_lock);
            _lock_acquire(x_cons_lock);
            _qmm_q40_bf16<m, k>(w_using, x_using, y_acc);
            _lock_release(w_prod_lock);
            _lock_release(x_prod_lock);
            j++;
        } while (j < K / k);
        _lock_acquire(y_prod_lock);
        yp = !yp;
        aie_pkt_word* pkt_id_using = yp ? pkt_id_ping : pkt_id_pong;
        bf16* y_using = yp ? y_ping : y_pong;
        *pkt_id_using = pkt_id;
        copy_y<m>(y_using + 16, y_acc);
        _lock_release(y_cons_lock);
        i++;
    } while (i < M / m);
#if !defined(__chess__)
    is_wx_ping = wx;
    is_y_ping = yp;
#endif
}

template<int M, int K, int m, int k>
void linear_proj_iD_q4_1(
    bf16 * x_ping,
    bf16 * x_pong,
    q41_block * w_ping,
    q41_block * w_pong,
    bf16 * y_ping,
    bf16 * y_pong,
    const uint32 pkt_id
) {
    aie_pkt_word* pkt_id_ping = reinterpret_cast<aie_pkt_word*>(y_ping + 14);
    aie_pkt_word* pkt_id_pong = reinterpret_cast<aie_pkt_word*>(y_pong + 14);
    bfloat16 b_col_reduce_add[ K/32]; 
    // Under Peano y_acc is not aligned to the vector width, so its vector stores also
    // write the bytes of these flags; keep them in locals for the whole call.
#if defined(__chess__)
    bool &wx = is_wx_ping;
    bool &yp = is_y_ping;
#else
    bool wx = is_wx_ping;
    bool yp = is_y_ping;
#endif
    int i = 0;
    do {
        zero_y<m>(y_acc);
        int j = 0;
        do {
            wx = !wx;
            q41_block* w_using = wx ? w_ping : w_pong;
            bf16* x_using = wx ? x_ping : x_pong;
            _lock_acquire(w_cons_lock);
            _lock_acquire(x_cons_lock);
            bf16* b_col_reduce_add_ptr = b_col_reduce_add + j *(k/32);
            // special logic for i==0:
            if(i == 0){
                // fill the b_col_reduce_add
                AIE_PREPARE_FOR_PIPELINING
                AIE_LOOP_UNROLL_FULL
                for(int l = 0; l < k / 32; l++)AIE_LOOP_FLATTEN{
                    *(b_col_reduce_add_ptr+ l) = bf16(aie::reduce_add(  aie::load_v<32>(x_using + 32*l) ));
                }
            }
            _qmm_q41_bf16<m, k>(w_using, x_using, y_acc, b_col_reduce_add_ptr);
            _lock_release(w_prod_lock);
            _lock_release(x_prod_lock);
            j++;
        } while (j < K / k);
        _lock_acquire(y_prod_lock);
        yp = !yp;
        aie_pkt_word* pkt_id_using = yp ? pkt_id_ping : pkt_id_pong;
        bf16* y_using = yp ? y_ping : y_pong;
        *pkt_id_using = pkt_id;
        copy_y<m>(y_using + 16, y_acc);
        _lock_release(y_cons_lock);
        i++;
    } while (i < M / m);
#if !defined(__chess__)
    is_wx_ping = wx;
    is_y_ping = yp;
#endif
}
extern "C" {
constexpr int total_cores = 16;
void proj(
    bf16 * y_ping,
    void * w_ping,
    bf16 * x_ping,
    bf16 * y_pong,
    void * w_pong,
    bf16 * x_pong,
    int* IS_ATTN
)
{
    constexpr int m = 32;
    constexpr int k = 256;
    
    _lock_acquire(rtp_available_lock);

    if (IS_ATTN[0] == 0) { // conv layer
        linear_proj_iD_q4_0<3 * D / total_cores, D, m, k>(x_ping, x_pong, (q40_block *)w_ping, (q40_block *)w_pong, y_ping, y_pong, 0);
        // // phase 2: o proj, D -> D to  RMS
        linear_proj_iD_q4_0<D / total_cores, D, m, k>(x_ping, x_pong, (q40_block *)w_ping, (q40_block *)w_pong, y_ping, y_pong, 1);
    
        // // phase 3: up/gate proj, D -> D * 8 to MT_GLU
        linear_proj_iD_q4_0<2 * HIDDEN_SIZE / total_cores, D, m, k>(x_ping, x_pong, (q40_block *)w_ping, (q40_block *)w_pong, y_ping, y_pong, 2);
    
        // // phase 4.1 down-> 4D -> D, first D -> D
        linear_proj_iD_q4_0<D / total_cores, HIDDEN_SIZE, m, k>(x_ping, x_pong, (q40_block *)w_ping, (q40_block *)w_pong, y_ping, y_pong, 1);
    }
    else if (IS_ATTN[0] == 1) {
        linear_proj_iD_q4_0<(D + DK + DV) / total_cores, D, m, k>(x_ping, x_pong, (q40_block *)w_ping, (q40_block *)w_pong, y_ping, y_pong, 0);
        // // phase 2: o proj, D -> D to  RMS
        linear_proj_iD_q4_0<D / total_cores, D, m, k>(x_ping, x_pong, (q40_block *)w_ping, (q40_block *)w_pong, y_ping, y_pong, 1);
    
        // // phase 3: up/gate proj, D -> D * 8 to MT_GLU
        linear_proj_iD_q4_0<2 * HIDDEN_SIZE / total_cores, D, m, k>(x_ping, x_pong, (q40_block *)w_ping, (q40_block *)w_pong, y_ping, y_pong, 2);
    
        // // phase 4.1 down-> 4D -> D, first D -> D
        linear_proj_iD_q4_0<D / total_cores, HIDDEN_SIZE, m, k>(x_ping, x_pong, (q40_block *)w_ping, (q40_block *)w_pong, y_ping, y_pong, 1);
    
    }
    else if (IS_ATTN[0] == 2) {
        linear_proj_iD_q4_1<65536 / total_cores, D, m, k>(x_ping, x_pong, (q41_block *)w_ping, (q41_block *)w_pong, y_ping, y_pong, 1); // send rms_core
    }

}


}