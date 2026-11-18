// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include <aie_api/aie.hpp>
#include "common.h"

typedef struct {
    bf16 scales[NPU_M_TILE * 4]; // per-group scales and mins
    bf16 mins[NPU_M_TILE * 4]; // per-group scales and mins
    uint4 qs[NPU_M_TILE * NPU_K_TILE];           // 4-bit quantized values
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
void _qmm_q4k_bf16(
    const q4k_block * A,
    const bf16 * B,
    float * c
) { 
    static_assert(N == 128, "N must be = 128");
    static_assert(M == 32, "M must be = 32");
    constexpr int pr = 16;
    constexpr int pc = 8;
    const q4k_block * A_q4k = (q4k_block *)A;
    uint4* qs_ptr = const_cast<uint4*>(A_q4k->qs);
    // precompute chunk sum of B;
    event0();
    for (int row = 0; row < M; row += pr){
        aie::accum<accfloat, pr> c_accum;
        c_accum.from_vector(aie::load_v<pr>(c + row));
        bf16* it_B = const_cast<bf16*>(B);
        for (int sub_chunk = 0; sub_chunk < QK_K / 32; sub_chunk++){
            aie::vector<bf16, pr> a_scales = aie::load_v<pr>(A_q4k->scales + sub_chunk * M + row);
            aie::vector<bf16, pr> a_mins = aie::load_v<pr>(A_q4k->mins + sub_chunk * M + row);
            aie::accum<accfloat, pr> a_mins_f32;
            a_mins_f32.from_vector(a_mins);
            const bf16* b_chunk = it_B;  // x of this 32-column sub-chunk
            it_B += 32;
            // float reduce_b = aie::reduce_add(b_col);

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

                // x[cc .. cc+7] of this 32-column chunk. Loaded directly: Peano miscompiles
                // b_col.extract<pc>(cc/pc) with a runtime index (slices 0..2 all read slice 1).
                aie::vector<bf16, pc> b8 = aie::load_v<pc>(b_chunk + cc);
                c_accum = aie::accumulate<pr>(c_accum, b8, 0, a_col_0, a_col_1, a_col_2, a_col_3, a_col_4, a_col_5, a_col_6, a_col_7);
            } // end of cc, 32 columns;
        
        } // end of sub_chunk, 256 columns;
        aie::vector<float, pr> c_accum_f32 = c_accum.template to_vector<float>();
        aie::store_v(c + row, c_accum_f32);
    }
    event1();
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
        // The core comes out of reset rounding toward -inf. That biases every
        // conversion to bf16 or bfp16 (mmul operands, accumulator stores) low,
        // and on mostly-positive inputs (GELU, softmax) the bias adds up.
        aie::set_rounding(aie::rounding_mode::conv_even);

        AIE_PERSIST_ZERO static AIE_PING_FLAG is_y_ping = false;
        AIE_PERSIST_ZERO static AIE_PING_FLAG is_x_ping = false;
        AIE_PERSIST_ZERO static AIE_PING_FLAG is_w_ping = false;

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
