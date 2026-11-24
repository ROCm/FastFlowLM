// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __Q4_CORE_H__
#define __Q4_CORE_H__
#include "aie_kernel_utils.h"
#include "typedef.hpp"

template <bool has_zp>
void _Q4_x_bf16(
    bf16* w16_buffer,
    bf16* scale_buffer,
    bf16* zp_buffer,
    uint8* w_buffer_0,
    uint8* w_buffer_1,
    uint8* w8_buffer,
    bool& is_w_ping,
    int w_prod_lock_id,
    int w_cons_lock_id,
    int& rtp_transpose
) {
    // first load scales
    constexpr int num_slices_per_row = QXNX_COL_BLOCK_SIZE / Q4_COL_PER_SLICE / 4;
    _lock_acquire(w_cons_lock_id);
    is_w_ping = !is_w_ping;
    if (is_w_ping) {
        _memcpy<W_SLICE_SIZE>((uint8*)scale_buffer, (uint8*)w_buffer_0);
    }
    else {
        _memcpy<W_SLICE_SIZE>((uint8*)scale_buffer, (uint8*)w_buffer_1);
    }
    _lock_release(w_prod_lock_id);

    if constexpr (has_zp) {
        _lock_acquire(w_cons_lock_id);
        is_w_ping = !is_w_ping;
        if (is_w_ping) {
        _memcpy<W_SLICE_SIZE>((uint8*)zp_buffer, (uint8*)w_buffer_0);
        }
        else {
        _memcpy<W_SLICE_SIZE>((uint8*)zp_buffer, (uint8*)w_buffer_1);
        }
        _lock_release(w_prod_lock_id);
    }

    // load 8 slices, basically all 32x256 for q4
#if defined(__chess__)
    {
        uint8* w_ptr;
        uint8* w8_ptr = w8_buffer;
        _lock_acquire(w_cons_lock_id);
        is_w_ping = !is_w_ping;
        w_ptr = is_w_ping? w_buffer_0 : w_buffer_1;
        _memcpy<W_SLICE_SIZE>((uint8*)w8_ptr, (uint8*)w_ptr);
        w8_ptr += W_SLICE_SIZE;
        _lock_release(w_prod_lock_id);

        _lock_acquire(w_cons_lock_id);
        is_w_ping = !is_w_ping;
        w_ptr = is_w_ping? w_buffer_0 : w_buffer_1;
        _memcpy<W_SLICE_SIZE>((uint8*)w8_ptr, (uint8*)w_ptr);
        w8_ptr += W_SLICE_SIZE;
        _lock_release(w_prod_lock_id);
        
        _lock_acquire(w_cons_lock_id);
        is_w_ping = !is_w_ping;
        w_ptr = is_w_ping? w_buffer_0 : w_buffer_1;
        _memcpy<W_SLICE_SIZE>((uint8*)w8_ptr, (uint8*)w_ptr);
        w8_ptr += W_SLICE_SIZE;
        _lock_release(w_prod_lock_id);
        
        _lock_acquire(w_cons_lock_id);
        is_w_ping = !is_w_ping;
        w_ptr = is_w_ping? w_buffer_0 : w_buffer_1;
        _memcpy<W_SLICE_SIZE>((uint8*)w8_ptr, (uint8*)w_ptr);
        w8_ptr += W_SLICE_SIZE;
        _lock_release(w_prod_lock_id);
        
        _lock_acquire(w_cons_lock_id);
        is_w_ping = !is_w_ping;
        w_ptr = is_w_ping? w_buffer_0 : w_buffer_1;
        _memcpy<W_SLICE_SIZE>((uint8*)w8_ptr, (uint8*)w_ptr);
        w8_ptr += W_SLICE_SIZE;
        _lock_release(w_prod_lock_id);
        
        _lock_acquire(w_cons_lock_id);
        is_w_ping = !is_w_ping;
        w_ptr = is_w_ping? w_buffer_0 : w_buffer_1;
        _memcpy<W_SLICE_SIZE>((uint8*)w8_ptr, (uint8*)w_ptr);
        w8_ptr += W_SLICE_SIZE;
        _lock_release(w_prod_lock_id);
        
        _lock_acquire(w_cons_lock_id);
        is_w_ping = !is_w_ping;
        w_ptr = is_w_ping? w_buffer_0 : w_buffer_1;
        _memcpy<W_SLICE_SIZE>((uint8*)w8_ptr, (uint8*)w_ptr);
        w8_ptr += W_SLICE_SIZE;
        _lock_release(w_prod_lock_id);
        
        _lock_acquire(w_cons_lock_id);
        is_w_ping = !is_w_ping;
        w_ptr = is_w_ping? w_buffer_0 : w_buffer_1;
        _memcpy<W_SLICE_SIZE>((uint8*)w8_ptr, (uint8*)w_ptr);
        w8_ptr += W_SLICE_SIZE;
        _lock_release(w_prod_lock_id);

    }
#else
    {
        uint8* w_ptr;
        uint8* w8_ptr = w8_buffer;
        // A loop under Peano: written out, the slice loads do not fit the core's
        // program memory.
        for (int slice = 0; slice < 8; slice++) {
            _lock_acquire(w_cons_lock_id);
            is_w_ping = !is_w_ping;
            w_ptr = is_w_ping? w_buffer_0 : w_buffer_1;
            _memcpy<W_SLICE_SIZE>((uint8*)w8_ptr, (uint8*)w_ptr);
            w8_ptr += W_SLICE_SIZE;
            _lock_release(w_prod_lock_id);
        }
    }
#endif

    for (int row = 0; row < QXNX_ROW_BLOCK_SIZE; row += PARALLEL_ROWS){
        static constexpr int COLs = 4 * Q4_WEIGHTS_COUNT / PARALLEL_ROWS; // 256 is default, makes it 16 x 256 subtile
        
        bf16* w_ptr_iter_0 = w16_buffer + row * COLs;
        bf16* w_ptr_iter_1 = w16_buffer + (row + 8) * COLs;

        bf16* it_scale = scale_buffer + row;
        bf16* it_zp = zp_buffer + row;
        uint4* it_quant = (uint4*)(w8_buffer + row * COLs / 2);

        // how many columns is processed every call
        static_assert((COLs % GGML_GROUP_SIZE) == 0);
        static_assert((COLs / GGML_GROUP_SIZE) >  0);

        AIE_PREPARE_FOR_PIPELINING
        AIE_LOOP_RANGE( COLs / GGML_GROUP_SIZE, COLs / GGML_GROUP_SIZE )
        for (int sub_chunk = 0; sub_chunk < COLs / GGML_GROUP_SIZE; sub_chunk++) {

            aie::vector<bf16, PARALLEL_ROWS> a_scales = aie::load_v<PARALLEL_ROWS>(it_scale);
            aie::vector<bf16, PARALLEL_ROWS> a_mins = aie::load_v<PARALLEL_ROWS>(it_zp);

            static constexpr int GROUP_SIZE = GGML_GROUP_SIZE;
            aie::accum<accfloat, PARALLEL_ROWS * 8> a_cc_f32_accum_0;
            aie::accum<accfloat, PARALLEL_ROWS * 8> a_cc_f32_accum_1;
            aie::accum<accfloat, PARALLEL_ROWS * 8> a_cc_f32_accum_2;
            aie::accum<accfloat, PARALLEL_ROWS * 8> a_cc_f32_accum_3;
            // Load all a vectors
            aie::vector<uint4, PARALLEL_ROWS * 8> a_cc_0 = aie::load_v<PARALLEL_ROWS * 8>(it_quant);
            it_quant += PARALLEL_ROWS * 4;
            aie::vector<float, PARALLEL_ROWS * 8> a_cc_f32_0 = aie::to_float(a_cc_0, 0);
            a_cc_f32_accum_0.from_vector(a_cc_f32_0);
            aie::vector<bf16, PARALLEL_ROWS * 8> a_cc_bf16_0 = a_cc_f32_accum_0.template to_vector<bf16>();

            aie::vector<uint4, PARALLEL_ROWS * 8> a_cc_1 = aie::load_v<PARALLEL_ROWS * 8>(it_quant);
            it_quant += PARALLEL_ROWS * 4;
            aie::vector<float, PARALLEL_ROWS * 8> a_cc_f32_1 = aie::to_float(a_cc_1, 0);
            a_cc_f32_accum_1.from_vector(a_cc_f32_1);
            aie::vector<bf16, PARALLEL_ROWS * 8> a_cc_bf16_1 = a_cc_f32_accum_1.template to_vector<bf16>();

            aie::vector<uint4, PARALLEL_ROWS * 8> a_cc_2 = aie::load_v<PARALLEL_ROWS * 8>(it_quant);
            it_quant += PARALLEL_ROWS * 4;
            aie::vector<float, PARALLEL_ROWS * 8> a_cc_f32_2 = aie::to_float(a_cc_2, 0);
            a_cc_f32_accum_2.from_vector(a_cc_f32_2);
            aie::vector<bf16, PARALLEL_ROWS * 8> a_cc_bf16_2 = a_cc_f32_accum_2.template to_vector<bf16>();

            aie::vector<uint4, PARALLEL_ROWS * 8> a_cc_3 = aie::load_v<PARALLEL_ROWS * 8>(it_quant);
            it_quant += PARALLEL_ROWS * 4;
            aie::vector<float, PARALLEL_ROWS * 8> a_cc_f32_3 = aie::to_float(a_cc_3, 0);
            a_cc_f32_accum_3.from_vector(a_cc_f32_3);
            aie::vector<bf16, PARALLEL_ROWS * 8> a_cc_bf16_3 = a_cc_f32_accum_3.template to_vector<bf16>();

            // Extract all columns
            aie::vector<bf16, PARALLEL_ROWS> a_col_00 = a_cc_bf16_0.extract<PARALLEL_ROWS>(0);
            aie::vector<bf16, PARALLEL_ROWS> a_col_01 = a_cc_bf16_0.extract<PARALLEL_ROWS>(1);
            aie::vector<bf16, PARALLEL_ROWS> a_col_02 = a_cc_bf16_0.extract<PARALLEL_ROWS>(2);
            aie::vector<bf16, PARALLEL_ROWS> a_col_03 = a_cc_bf16_0.extract<PARALLEL_ROWS>(3);
            aie::vector<bf16, PARALLEL_ROWS> a_col_04 = a_cc_bf16_0.extract<PARALLEL_ROWS>(4);
            aie::vector<bf16, PARALLEL_ROWS> a_col_05 = a_cc_bf16_0.extract<PARALLEL_ROWS>(5);
            aie::vector<bf16, PARALLEL_ROWS> a_col_06 = a_cc_bf16_0.extract<PARALLEL_ROWS>(6);
            aie::vector<bf16, PARALLEL_ROWS> a_col_07 = a_cc_bf16_0.extract<PARALLEL_ROWS>(7);

            aie::vector<bf16, PARALLEL_ROWS> a_col_08 = a_cc_bf16_1.extract<PARALLEL_ROWS>(0);
            aie::vector<bf16, PARALLEL_ROWS> a_col_09 = a_cc_bf16_1.extract<PARALLEL_ROWS>(1);
            aie::vector<bf16, PARALLEL_ROWS> a_col_10 = a_cc_bf16_1.extract<PARALLEL_ROWS>(2);
            aie::vector<bf16, PARALLEL_ROWS> a_col_11 = a_cc_bf16_1.extract<PARALLEL_ROWS>(3);
            aie::vector<bf16, PARALLEL_ROWS> a_col_12 = a_cc_bf16_1.extract<PARALLEL_ROWS>(4);
            aie::vector<bf16, PARALLEL_ROWS> a_col_13 = a_cc_bf16_1.extract<PARALLEL_ROWS>(5);
            aie::vector<bf16, PARALLEL_ROWS> a_col_14 = a_cc_bf16_1.extract<PARALLEL_ROWS>(6);
            aie::vector<bf16, PARALLEL_ROWS> a_col_15 = a_cc_bf16_1.extract<PARALLEL_ROWS>(7);

            aie::vector<bf16, PARALLEL_ROWS> a_col_16 = a_cc_bf16_2.extract<PARALLEL_ROWS>(0);
            aie::vector<bf16, PARALLEL_ROWS> a_col_17 = a_cc_bf16_2.extract<PARALLEL_ROWS>(1);
            aie::vector<bf16, PARALLEL_ROWS> a_col_18 = a_cc_bf16_2.extract<PARALLEL_ROWS>(2);
            aie::vector<bf16, PARALLEL_ROWS> a_col_19 = a_cc_bf16_2.extract<PARALLEL_ROWS>(3);
            aie::vector<bf16, PARALLEL_ROWS> a_col_20 = a_cc_bf16_2.extract<PARALLEL_ROWS>(4);
            aie::vector<bf16, PARALLEL_ROWS> a_col_21 = a_cc_bf16_2.extract<PARALLEL_ROWS>(5);
            aie::vector<bf16, PARALLEL_ROWS> a_col_22 = a_cc_bf16_2.extract<PARALLEL_ROWS>(6);
            aie::vector<bf16, PARALLEL_ROWS> a_col_23 = a_cc_bf16_2.extract<PARALLEL_ROWS>(7);

            aie::vector<bf16, PARALLEL_ROWS> a_col_24 = a_cc_bf16_3.extract<PARALLEL_ROWS>(0);
            aie::vector<bf16, PARALLEL_ROWS> a_col_25 = a_cc_bf16_3.extract<PARALLEL_ROWS>(1);
            aie::vector<bf16, PARALLEL_ROWS> a_col_26 = a_cc_bf16_3.extract<PARALLEL_ROWS>(2);
            aie::vector<bf16, PARALLEL_ROWS> a_col_27 = a_cc_bf16_3.extract<PARALLEL_ROWS>(3);
            aie::vector<bf16, PARALLEL_ROWS> a_col_28 = a_cc_bf16_3.extract<PARALLEL_ROWS>(4);
            aie::vector<bf16, PARALLEL_ROWS> a_col_29 = a_cc_bf16_3.extract<PARALLEL_ROWS>(5);
            aie::vector<bf16, PARALLEL_ROWS> a_col_30 = a_cc_bf16_3.extract<PARALLEL_ROWS>(6);
            aie::vector<bf16, PARALLEL_ROWS> a_col_31 = a_cc_bf16_3.extract<PARALLEL_ROWS>(7);

            // All MAC operations
            a_col_00 = aie::mul(a_col_00, a_scales);
            a_col_01 = aie::mul(a_col_01, a_scales);
            a_col_02 = aie::mul(a_col_02, a_scales);
            a_col_03 = aie::mul(a_col_03, a_scales);
            a_col_04 = aie::mul(a_col_04, a_scales);
            a_col_05 = aie::mul(a_col_05, a_scales);
            a_col_06 = aie::mul(a_col_06, a_scales);
            a_col_07 = aie::mul(a_col_07, a_scales);
            a_col_08 = aie::mul(a_col_08, a_scales);
            a_col_09 = aie::mul(a_col_09, a_scales);
            a_col_10 = aie::mul(a_col_10, a_scales);
            a_col_11 = aie::mul(a_col_11, a_scales);
            a_col_12 = aie::mul(a_col_12, a_scales);
            a_col_13 = aie::mul(a_col_13, a_scales);
            a_col_14 = aie::mul(a_col_14, a_scales);
            a_col_15 = aie::mul(a_col_15, a_scales);
            a_col_16 = aie::mul(a_col_16, a_scales);
            a_col_17 = aie::mul(a_col_17, a_scales);
            a_col_18 = aie::mul(a_col_18, a_scales);
            a_col_19 = aie::mul(a_col_19, a_scales);
            a_col_20 = aie::mul(a_col_20, a_scales);
            a_col_21 = aie::mul(a_col_21, a_scales);
            a_col_22 = aie::mul(a_col_22, a_scales);
            a_col_23 = aie::mul(a_col_23, a_scales);
            a_col_24 = aie::mul(a_col_24, a_scales);
            a_col_25 = aie::mul(a_col_25, a_scales);
            a_col_26 = aie::mul(a_col_26, a_scales);
            a_col_27 = aie::mul(a_col_27, a_scales);
            a_col_28 = aie::mul(a_col_28, a_scales);
            a_col_29 = aie::mul(a_col_29, a_scales);
            a_col_30 = aie::mul(a_col_30, a_scales);
            a_col_31 = aie::mul(a_col_31, a_scales);

            if constexpr (has_zp){
                a_col_00 = aie::add(a_col_00, a_mins);
                a_col_01 = aie::add(a_col_01, a_mins);
                a_col_02 = aie::add(a_col_02, a_mins);
                a_col_03 = aie::add(a_col_03, a_mins);
                a_col_04 = aie::add(a_col_04, a_mins);
                a_col_05 = aie::add(a_col_05, a_mins);
                a_col_06 = aie::add(a_col_06, a_mins);
                a_col_07 = aie::add(a_col_07, a_mins);
                a_col_08 = aie::add(a_col_08, a_mins);
                a_col_09 = aie::add(a_col_09, a_mins);
                a_col_10 = aie::add(a_col_10, a_mins);
                a_col_11 = aie::add(a_col_11, a_mins);
                a_col_12 = aie::add(a_col_12, a_mins);
                a_col_13 = aie::add(a_col_13, a_mins);
                a_col_14 = aie::add(a_col_14, a_mins);
                a_col_15 = aie::add(a_col_15, a_mins);
                a_col_16 = aie::add(a_col_16, a_mins);
                a_col_17 = aie::add(a_col_17, a_mins);
                a_col_18 = aie::add(a_col_18, a_mins);
                a_col_19 = aie::add(a_col_19, a_mins);
                a_col_20 = aie::add(a_col_20, a_mins);
                a_col_21 = aie::add(a_col_21, a_mins);
                a_col_22 = aie::add(a_col_22, a_mins);
                a_col_23 = aie::add(a_col_23, a_mins);
                a_col_24 = aie::add(a_col_24, a_mins);
                a_col_25 = aie::add(a_col_25, a_mins);
                a_col_26 = aie::add(a_col_26, a_mins);
                a_col_27 = aie::add(a_col_27, a_mins);
                a_col_28 = aie::add(a_col_28, a_mins);
                a_col_29 = aie::add(a_col_29, a_mins);
                a_col_30 = aie::add(a_col_30, a_mins);
                a_col_31 = aie::add(a_col_31, a_mins);
                it_zp += QXNX_ROW_BLOCK_SIZE;
            }
            it_scale += QXNX_ROW_BLOCK_SIZE;

            auto a_01 = aie::interleave_zip(a_col_00, a_col_01, 8);
            auto a_23 = aie::interleave_zip(a_col_02, a_col_03, 8);
            auto a_45 = aie::interleave_zip(a_col_04, a_col_05, 8);
            auto a_67 = aie::interleave_zip(a_col_06, a_col_07, 8);
            auto a_up = aie::concat(a_01.first, a_23.first, a_45.first, a_67.first);
            auto a_down = aie::concat(a_01.second, a_23.second, a_45.second, a_67.second);
            if (rtp_transpose == 1){
                a_up = aie::transpose(a_up, 8, 8);
                a_down = aie::transpose(a_down, 8, 8);
            }
            aie::store_v(w_ptr_iter_0, a_up);
            aie::store_v(w_ptr_iter_1, a_down);
            w_ptr_iter_0 += 64;
            w_ptr_iter_1 += 64;

            a_01 = aie::interleave_zip(a_col_08, a_col_09, 8);
            a_23 = aie::interleave_zip(a_col_10, a_col_11, 8);
            a_45 = aie::interleave_zip(a_col_12, a_col_13, 8);
            a_67 = aie::interleave_zip(a_col_14, a_col_15, 8);
            a_up = aie::concat(a_01.first, a_23.first, a_45.first, a_67.first);
            a_down = aie::concat(a_01.second, a_23.second, a_45.second, a_67.second);
            if (rtp_transpose == 1){
                a_up = aie::transpose(a_up, 8, 8);
                a_down = aie::transpose(a_down, 8, 8);
            }
            aie::store_v(w_ptr_iter_0, a_up);
            aie::store_v(w_ptr_iter_1, a_down);
            w_ptr_iter_0 += 64;
            w_ptr_iter_1 += 64;

            a_01 = aie::interleave_zip(a_col_16, a_col_17, 8);
            a_23 = aie::interleave_zip(a_col_18, a_col_19, 8);
            a_45 = aie::interleave_zip(a_col_20, a_col_21, 8);
            a_67 = aie::interleave_zip(a_col_22, a_col_23, 8);
            a_up = aie::concat(a_01.first, a_23.first, a_45.first, a_67.first);
            a_down = aie::concat(a_01.second, a_23.second, a_45.second, a_67.second);
            if (rtp_transpose == 1){
                a_up = aie::transpose(a_up, 8, 8);
                a_down = aie::transpose(a_down, 8, 8);
            }
            aie::store_v(w_ptr_iter_0, a_up);
            aie::store_v(w_ptr_iter_1, a_down);
            w_ptr_iter_0 += 64;
            w_ptr_iter_1 += 64;

            a_01 = aie::interleave_zip(a_col_24, a_col_25, 8);
            a_23 = aie::interleave_zip(a_col_26, a_col_27, 8);
            a_45 = aie::interleave_zip(a_col_28, a_col_29, 8);
            a_67 = aie::interleave_zip(a_col_30, a_col_31, 8);
            a_up = aie::concat(a_01.first, a_23.first, a_45.first, a_67.first);
            a_down = aie::concat(a_01.second, a_23.second, a_45.second, a_67.second);
            if (rtp_transpose == 1){
                a_up = aie::transpose(a_up, 8, 8);
                a_down = aie::transpose(a_down, 8, 8);
            }
            aie::store_v(w_ptr_iter_0, a_up);
            aie::store_v(w_ptr_iter_1, a_down);
            w_ptr_iter_0 += 64;
            w_ptr_iter_1 += 64;
        } // end of sub_chunk
    } 
}

#endif // __Q4_CORE_H__