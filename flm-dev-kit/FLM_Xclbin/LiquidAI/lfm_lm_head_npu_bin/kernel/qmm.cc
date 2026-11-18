// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include <aie_api/aie.hpp>
#include "common.h"

typedef struct {
    bf16 scales[NPU_M_TILE * 8]; // per-group scales and mins
    bf16 mins[NPU_M_TILE * 8]; // per-group scales and mins
    int4 qs[NPU_M_TILE * NPU_K_TILE];           // 4-bit quantized values
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
)
{ 
    static_assert(N == 256, "N must be = 256");
    static_assert(M == 32, "M must be = 32");
    constexpr int pr = 16;
    constexpr int pc = 8;
    const q4k_block * A_q4k = (q4k_block *)A;
    int4* qs_ptr = const_cast<int4*>(A_q4k->qs);
    float* c_ptr = const_cast<float*>(c);
    // precompute chunk sum of B;
    for (int row = 0; row < M; row += pr)
    {
        aie::accum<accfloat, pr> c_accum;
        c_accum.from_vector(aie::load_v<pr>(c + row));
        bf16* it_B = const_cast<bf16*>(B);
        
        bf16* scales_ptr = const_cast<bf16*>(A_q4k->scales + row);
        for (int sub_chunk = 0; sub_chunk < QK_K / 32; sub_chunk++) chess_unroll_loop(2) {
            aie::vector<bf16, 32> b_col = aie::load_v<32>(it_B);
            it_B += 32;
            aie::vector<int4, pr * 8> a_cc = aie::load_v<pr * 8>(qs_ptr);
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

            aie::accum<accfloat, pr> c_local_accum = aie::accumulate<pr>(b_col.extract<pc>(0), 0, a_col_0, a_col_1, a_col_2, a_col_3, a_col_4, a_col_5, a_col_6, a_col_7);

            for (int cc = pc; cc < 32; cc += pc) chess_prepare_for_pipelining chess_loop_range(3, 3) {
                aie::vector<int4, pr * 8> a_cc = aie::load_v<pr * 8>(qs_ptr);
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

#if defined(__chess__)
                c_local_accum = aie::accumulate<pr>(c_local_accum, b_col.extract<pc>(cc/pc), 0, a_col_0, a_col_1, a_col_2, a_col_3, a_col_4, a_col_5, a_col_6, a_col_7);
#else
                // The same 8 values as b_col.extract<pc>(cc/pc), loaded from memory:
                // Peano miscompiles extract with a runtime index in this loop.
                c_local_accum = aie::accumulate<pr>(c_local_accum, aie::load_v<pc>(it_B - 32 + cc), 0, a_col_0, a_col_1, a_col_2, a_col_3, a_col_4, a_col_5, a_col_6, a_col_7);
#endif
            } // end of cc, 32 columns;
            aie::vector<bf16, pr> a_scales = aie::load_v<pr>(scales_ptr);
            scales_ptr += M;
            c_accum = aie::mac(c_accum, c_local_accum.template to_vector<bf16>(), a_scales);
        } // end of sub_chunk, 256 columns;
        aie::vector<float, pr> c_accum_f32 = c_accum.template to_vector<float>();
        aie::store_v(c_ptr, c_accum_f32);
        c_ptr += pr;
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
        AIE_PERSIST_ZERO static bool is_y_ping = false;
        AIE_PERSIST_ZERO static bool is_x_ping = false;
        AIE_PERSIST_ZERO static bool is_w_ping = false;

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
