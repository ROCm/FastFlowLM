// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#ifndef __Q4_K_H__
#define __Q4_K_H__
#include "qmm.h"
// #define REFERENCE_CODE


typedef struct {
    bf16 scales[NPU_M_TILE * 8]; // per-group scales and mins
    bf16 mins[NPU_M_TILE * 8]; // per-group scales and mins
    uint4 qs[NPU_M_TILE * NPU_K_TILE];           // 4-bit quantized values
}q4k_block;


template <int M, int N>
void _qmm_q4k_q4k(
    const q4k_block * a,
    const q4k_block * b,
    float * c
) {
    float sum = 0.0f;

}
template <int M, int N>
inline void _qmm_q4k_bf16(
    const q4k_block * A,
    const bf16 * B,
    float * c
) { 
    static_assert(N == 256, "N must be = 256");
    static_assert(M == 32, "M must be = 32");
    constexpr int pr = 16;
    constexpr int pc = 8;
    const q4k_block * A_q4k = (q4k_block *)A;
    uint4* qs_ptr = const_cast<uint4*>(A_q4k->qs);
    // precompute chunk sum of B;
    for (int row = 0; row < M; row += pr) chess_unroll_loop(2)
    {
        aie::accum<accfloat, pr> c_accum;
        c_accum.from_vector(aie::load_v<pr>(c + row));
        bf16* it_B = const_cast<bf16*>(B);
        for (int sub_chunk = 0; sub_chunk < QK_K / 32; sub_chunk++){
            aie::vector<bf16, pr> a_scales = aie::load_v<pr>(A_q4k->scales + sub_chunk * M + row);
            aie::vector<bf16, pr> a_mins = aie::load_v<pr>(A_q4k->mins + sub_chunk * M + row);
            aie::accum<accfloat, pr> a_mins_f32;
            a_mins_f32.from_vector(a_mins);
            aie::vector<bf16, 32> b_col = aie::load_v<32>(it_B);
            it_B += 32;
            // float reduce_b = aie::reduce_add(b_col);

            // Peano miscompiles aie::vector::extract<N> with a RUNTIME index
            // (cc/pc below) -> wrong lanes. Pre-extract with COMPILE-TIME indices.
#if !defined(__chess__)
            const aie::vector<bf16, pc> b_sub[4] = {
                b_col.extract<pc>(0), b_col.extract<pc>(1),
                b_col.extract<pc>(2), b_col.extract<pc>(3)};
#endif

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

#if defined(__chess__)
                c_accum = aie::accumulate<pr>(c_accum, b_col.extract<pc>(cc/pc), 0, a_col_0, a_col_1, a_col_2, a_col_3, a_col_4, a_col_5, a_col_6, a_col_7);
#else
                c_accum = aie::accumulate<pr>(c_accum, b_sub[cc/pc], 0, a_col_0, a_col_1, a_col_2, a_col_3, a_col_4, a_col_5, a_col_6, a_col_7);
#endif
                // lookup;
                
            } // end of cc, 32 columns;
            // c_accum = aie::mac(c_accum, c_local_accum.template to_vector<bf16>(), a_scales);
            // c_accum = aie::mac(c_accum, a_mins, (bf16)reduce_b);

            // 
        } // end of sub_chunk, 256 columns;
        aie::vector<float, pr> c_accum_f32 = c_accum.template to_vector<float>();
        aie::store_v(c + row, c_accum_f32);
    }
}

#endif // __Q4_K_H__
