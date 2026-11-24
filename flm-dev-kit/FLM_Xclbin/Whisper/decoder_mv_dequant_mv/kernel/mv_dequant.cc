// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <aie_api/aie.hpp>
#include "common.h"

typedef struct {
    bf16 scales[Q4NX_ROW * (Q4NX_COL/32)]; // per-group scales and mins
    bf16 mins[Q4NX_ROW *  (Q4NX_COL/32)]; // per-group scales and mins
    uint8_t qs[(Q4NX_ROW * Q4NX_COL) /2]; // 4-bit quantized values

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
 void mv_dequant_core(
    float * c,
    const q4k_block * A,
    const bf16 * B
) { 

    constexpr int pr = 16;
    constexpr int pc = 8;
    const q4k_block * A_q4k = (q4k_block *)A;
    const uint8_t* qs_ptr = A_q4k->qs;
    // precompute chunk sum of B;
    for (int row = 0; row < M; row += pr)
    {
        aie::accum<accfloat, pr> c_accum;
        c_accum.from_vector(aie::load_v<pr>(c + row));
        const bf16* it_B =B;
        for (int sub_chunk = 0; sub_chunk < Q4NX_COL / 32; sub_chunk++){
            aie::vector<bf16, pr> a_scales = aie::load_v<pr>(A_q4k->scales + sub_chunk * M + row);
            aie::vector<bf16, pr> a_mins = aie::load_v<pr>(A_q4k->mins + sub_chunk * M + row);
            aie::accum<accfloat, pr> a_mins_f32;
            a_mins_f32.from_vector(a_mins);
            const bf16* b_chunk = it_B;  // x of this 32-column sub-chunk
            it_B += 32;
            // float reduce_b = aie::reduce_add(b_col);

            // AIE_PREPARE_FOR_PIPELINING
            // AIE_LOOP_RANGE( 32/pc, 32/pc )
            for (int cc = 0; cc < 32; cc += pc){
                aie::vector<uint4, pr * 8> a_cc = aie::load_v<pr * 8>( (uint4*) qs_ptr);
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

                // c_accum = aie::accumulate<pr>(c_accum, b_col.extract<pc>(cc/pc), 0, a_col_0, a_col_1, a_col_2, a_col_3, a_col_4, a_col_5, a_col_6, a_col_7);
                // lookup;
                // x[cc .. cc+7] of this 32-column chunk. Loaded directly: Peano miscompiles
                // b_col.extract<pc>(cc/pc) with a runtime index (slices 0..2 all read slice 1).
                aie::vector<bf16, pc> b8 = aie::load_v<pc>(b_chunk + cc);
                c_accum = aie::mac(c_accum, b8.get(0), a_col_0);
                c_accum = aie::mac(c_accum, b8.get(1), a_col_1);
                c_accum = aie::mac(c_accum, b8.get(2), a_col_2);
                c_accum = aie::mac(c_accum, b8.get(3), a_col_3);
                c_accum = aie::mac(c_accum, b8.get(4), a_col_4);
                c_accum = aie::mac(c_accum, b8.get(5), a_col_5);
                c_accum = aie::mac(c_accum, b8.get(6), a_col_6);
                c_accum = aie::mac(c_accum, b8.get(7), a_col_7);
            } // end of cc, 32 columns;
            // c_accum = aie::mac(c_accum, c_local_accum.template to_vector<bf16>(), a_scales);
            // c_accum = aie::mac(c_accum, a_mins, (bf16)reduce_b);

            // 
        } // end of sub_chunk, 256 columns;
        aie::vector<float, pr> c_accum_f32 = c_accum.template to_vector<float>();
        aie::store_v(c + row, c_accum_f32);
    }
}

constexpr int w_prod_lock = 0;
constexpr int w_cons_lock = 1;
constexpr int x_prod_lock = 2;
constexpr int x_cons_lock = 3;
constexpr int y_prod_lock = 4;
constexpr int y_cons_lock = 5;
constexpr int rtp_lock = 6;


AIE_PERSIST_ZERO static AIE_PING_FLAG is_wx_ping = false;
AIE_PERSIST_ZERO static AIE_PING_FLAG is_y_ping = false;

extern "C" {

void mv_dequant(
    bf16 *restrict y_ping,
    bf16 *restrict y_pong,
    const q4k_block *restrict w_ping, // matrix
    const q4k_block *restrict w_pong,
    const bf16 *restrict x_ping,
    const bf16 *restrict x_pong,
    float * y_acc,
    volatile int * M_div_m, 
    volatile int * K_div_k
){
    // The core comes out of reset rounding toward -inf. That biases every
    // conversion to bf16 or bfp16 (mmul operands, accumulator stores) low,
    // and on mostly-positive inputs (GELU, softmax) the bias adds up.
    aie::set_rounding(aie::rounding_mode::conv_even);


    int i = 0;
    do {
        zero_vectorized<float, Q4NX_ROW>(y_acc);
        int j = 0;
        do {
            _lock_acquire(x_cons_lock);
            _lock_acquire(w_cons_lock);
            is_wx_ping = !is_wx_ping;
            const bf16 *x = is_wx_ping ? x_ping : x_pong;
            const q4k_block *w = is_wx_ping ? w_ping : w_pong;
            mv_dequant_core<Q4NX_ROW, Q4NX_COL>(y_acc, w, x);
            _lock_release(x_prod_lock);
            _lock_release(w_prod_lock);
            j++;
        } while (j < K_div_k[0]);
        _lock_acquire(y_prod_lock);
        is_y_ping = !is_y_ping;
        bf16 *y = is_y_ping ? y_ping : y_pong;
        passThrough_aie<float, bf16, Q4NX_ROW>(y_acc, y);
        _lock_release(y_cons_lock);
        i++;
    } while (i < M_div_m[0]);
}

} // extern "C"