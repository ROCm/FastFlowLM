// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include "common.h"
#include "aie_kernel_utils.h"
#define NPU_M_TILE 32
#define NPU_K_TILE 128

typedef struct {
    bf16 scales[NPU_M_TILE * 4]; // per-group scales and mins
    bf16 mins[NPU_M_TILE * 4]; // per-group scales and mins
    uint4 qs[NPU_M_TILE * NPU_K_TILE];           // 4-bit quantized values
}q4k_block;

constexpr int qw_prod_lock = 0;
constexpr int qw_cons_lock = 1;
constexpr int w_prod_lock = 2;
constexpr int w_cons_lock = 3;
template<int m, int k>
void dequant_core(
    bf16 * w_ptr,
    q4k_block * qw_ptr
) {
    constexpr int pr = 16;

    constexpr int pc = 8;
    uint4* qs_ptr = const_cast<uint4*>(qw_ptr->qs);
    
    // precompute chunk sum of B;
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_RANGE(NPU_M_TILE/16, NPU_M_TILE/16)
    for (int row = 0; row < NPU_M_TILE; row += 16)
    {
        bf16* w_ptr_iter_0 = w_ptr + row * NPU_K_TILE;
        bf16* w_ptr_iter_1 = w_ptr + (row + 8) * NPU_K_TILE;

        AIE_PREPARE_FOR_PIPELINING
        AIE_LOOP_RANGE(NPU_K_TILE/32, NPU_K_TILE/32)
        for (int sub_chunk = 0; sub_chunk < NPU_K_TILE / 32; sub_chunk++){
            aie::vector<bf16, pr> a_scales = aie::load_v<pr>(qw_ptr->scales + sub_chunk * NPU_M_TILE + row);
            aie::vector<bf16, pr> a_mins = aie::load_v<pr>(qw_ptr->mins + sub_chunk * NPU_M_TILE + row);
            aie::accum<accfloat, pr> a_mins_f32;
            a_mins_f32.from_vector(a_mins);

            AIE_PREPARE_FOR_PIPELINING
            AIE_LOOP_UNROLL_FULL
            for (int cc = 0; cc < 32; cc += pc){
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
              

                aie::accum<accfloat, pr> a_col_0_accum = aie::mac(a_mins_f32, a_scales, a_col_0);
                aie::accum<accfloat, pr> a_col_1_accum = aie::mac(a_mins_f32, a_scales, a_col_1);
                aie::accum<accfloat, pr> a_col_2_accum = aie::mac(a_mins_f32, a_scales, a_col_2);
                aie::accum<accfloat, pr> a_col_3_accum = aie::mac(a_mins_f32, a_scales, a_col_3);
                aie::accum<accfloat, pr> a_col_4_accum = aie::mac(a_mins_f32, a_scales, a_col_4);
                aie::accum<accfloat, pr> a_col_5_accum = aie::mac(a_mins_f32, a_scales, a_col_5);
                aie::accum<accfloat, pr> a_col_6_accum = aie::mac(a_mins_f32, a_scales, a_col_6);
                aie::accum<accfloat, pr> a_col_7_accum = aie::mac(a_mins_f32, a_scales, a_col_7);

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
                // a_up = aie::transpose(a_up, 8, 8);
                // a_down = aie::transpose(a_down, 8, 8);
                aie::store_v(w_ptr_iter_0, a_up);
                aie::store_v(w_ptr_iter_1, a_down);
                w_ptr_iter_0 += 64;
                w_ptr_iter_1 += 64;
            } // end of cc, 32 columns;
        } // end of sub_chunk, 256 columns;
    }
}

extern "C" {
constexpr int total_cores = 16;
void dequant(
    bf16 * w_ping,
    bf16 * w_pong,
    q4k_block * qw_ping,
    q4k_block * qw_pong
)
{
    constexpr int m = 32;
    constexpr int k = 128;
    AIE_PERSIST_ZERO static bool is_ping = false;

    _lock_acquire(qw_cons_lock, 1);
    _lock_acquire(w_prod_lock, 1);
    is_ping = !is_ping;

    bf16 * w_ptr = is_ping ? w_ping : w_pong;
    q4k_block * qw_ptr = is_ping ? qw_ping : qw_pong;

    dequant_core<m, k>(w_ptr, qw_ptr);

    _lock_release(w_cons_lock, 1);
    _lock_release(qw_prod_lock, 1);

}


}