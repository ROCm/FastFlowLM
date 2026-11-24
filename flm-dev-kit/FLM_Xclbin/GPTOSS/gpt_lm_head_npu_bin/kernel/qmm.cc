// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#include <aie_api/aie.hpp>
#include "common.h"

typedef struct {
    bf16 scales[NPU_M_TILE * (NPU_K_TILE/32)]; // per-group scales and mins
    bf16 mins[NPU_M_TILE *  (NPU_K_TILE/32)]; // per-group scales and mins
    uint8_t qs[(NPU_M_TILE * NPU_K_TILE) /2]; // 4-bit quantized values

}q4k_block;

template<int m>
void copy_y(bf16 * y, const float * y_acc) {
    for (int i = 0; i < m / 16; i++){
        aie::vector<float, 16> y_vec = aie::load_v<16>(y_acc + i * 16);
        aie::accum<accfloat, 16> acc;
        acc.from_vector(y_vec);
        aie::store_v(y + i * 16, acc.template to_vector<bf16>());
    }
}

template<int m>
void zero_y(float * y) {
    zero_vectorized<float, m>(y);
}

template <int M, int N>
void _qmm_q4k_q4k(
    const q4k_block * a,
    const q4k_block * b,
    float * c
) {
    float sum = 0.0f;

}
template <int M, int N>
 void _qmm_q4k_bf16(
    const q4k_block * A,
    const bf16 * B,
    float * c
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
        for (int sub_chunk = 0; sub_chunk < NPU_K_TILE / 32; sub_chunk++){
            aie::vector<bf16, pr> a_scales = aie::load_v<pr>(A_q4k->scales + sub_chunk * M + row);
            aie::vector<bf16, pr> a_mins = aie::load_v<pr>(A_q4k->mins + sub_chunk * M + row);
            aie::accum<accfloat, pr> a_mins_f32;
            a_mins_f32.from_vector(a_mins);
            aie::vector<bf16, 32> b_col = aie::load_v<32>(it_B);
            it_B += 32;
            // float reduce_b = aie::reduce_add(b_col);
#if !defined(__chess__)
            // Peano miscompiles aie::vector::extract<N> with a runtime index
            // (b_col.extract<pc>(cc/pc) below), so extract the 4 sub-blocks
            // with constant indices up front and index the array instead.
            aie::vector<bf16, pc> b_sub[4] = {
                b_col.extract<pc>(0), b_col.extract<pc>(1),
                b_col.extract<pc>(2), b_col.extract<pc>(3)
            };
#endif

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

                // Peano miscompiles extract<N> with a runtime index, so on peano
                // the b sub-block comes from the constant-index b_sub array above.
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

constexpr int x_prod_lock = 0;
constexpr int x_cons_lock = 1;
constexpr int w_prod_lock = 2;
constexpr int w_cons_lock = 3;
constexpr int y_prod_lock = 4;
constexpr int y_cons_lock = 5;

extern "C" {
    void qmm(
        const bf16 * x_ping,
        const bf16 * x_pong,
        const q4k_block * w_ping,
        const q4k_block * w_pong,
        bf16 * y_ping,
        bf16 * y_pong,
        const int M_div_m,
        const int K_div_k
    ){
        // Peano's CDO loader does not zero .bss; keep ping/pong selectors in .data.
#if !defined(__chess__)
        __attribute__((section(".data")))
#endif
        static bool is_y_ping = false;
#if !defined(__chess__)
        __attribute__((section(".data")))
#endif
        static bool is_x_ping = false;
#if !defined(__chess__)
        __attribute__((section(".data")))
#endif
        static bool is_w_ping = false;

        float y_acc[NPU_M_TILE] = {0};

        _lock_acquire(x_cons_lock);
        is_x_ping = !is_x_ping;
        const bf16 *x = reinterpret_cast<const bf16 *>(is_x_ping ? x_ping : x_pong);

        for (int m = 0; m < M_div_m; m++) {
            zero_y<NPU_M_TILE>(y_acc);
            for (int k = 0; k < K_div_k; k++) {
                _lock_acquire(w_cons_lock);     
                is_w_ping = !is_w_ping;
                const q4k_block *w = reinterpret_cast<const q4k_block *>(is_w_ping ? w_ping : w_pong);
                _qmm_q4k_bf16<NPU_M_TILE, NPU_K_TILE>(w, x + k * NPU_K_TILE, y_acc);
                _lock_release(w_prod_lock);
            }
            _lock_acquire(y_prod_lock);
            is_y_ping = !is_y_ping;
            bf16 *y = reinterpret_cast<bf16 *>(is_y_ping ? y_ping : y_pong);
            copy_y<NPU_M_TILE>(y, y_acc);
            _lock_release(y_cons_lock);
        }
        _lock_release(x_prod_lock);
    }
}
