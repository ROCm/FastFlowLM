// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __Q8_CORE_H__
#define __Q8_CORE_H__
#include "aie_kernel_utils.h"
#include "typedef.hpp"


template <bool has_zp>
void _Q8_x_bf16(
    bf16* w16_buffer,
    bf16* scale_buffer,
    bf16* zp_buffer,
    int8* w_buffer_0,
    int8* w_buffer_1,
    int8* w8_buffer,
    bool& is_w_ping,
    int w_prod_lock_id,
    int w_cons_lock_id,
    int& rtp_transpose
) {
    // first load scales
    constexpr int num_slices_per_row = QXNX_COL_BLOCK_SIZE / Q8_COL_PER_SLICE; 
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

    for (int row = 0; row < QXNX_ROW_BLOCK_SIZE; row += PARALLEL_ROWS){
        
        bf16* w_ptr_iter_0 = w16_buffer + row * QXNX_COL_BLOCK_SIZE;
        bf16* w_ptr_iter_1 = w16_buffer + (row + 8) * QXNX_COL_BLOCK_SIZE;
        
#if defined(__chess__)
        {
            int8* w_ptr;
            int8* w8_ptr = w8_buffer;
        
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
            int8* w_ptr;
            int8* w8_ptr = w8_buffer;
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

        int8* it_q = w8_buffer; 

        bf16* it_scale = scale_buffer + row;
        bf16* it_zp = zp_buffer + row;

        AIE_PREPARE_FOR_PIPELINING
        AIE_LOOP_RANGE( num_slices_per_row, num_slices_per_row )
        for (int slice = 0; slice < num_slices_per_row; slice++) {
          
            aie::vector<bf16, PARALLEL_ROWS> a_scales = aie::load_v<PARALLEL_ROWS>(it_scale);
            aie::vector<bf16, PARALLEL_ROWS> a_mins = aie::load_v<PARALLEL_ROWS>(it_zp);

            AIE_PREPARE_FOR_PIPELINING
            AIE_LOOP_RANGE( GGML_GROUP_SIZE / 8, GGML_GROUP_SIZE / 8)
            for(int t = 0; t < GGML_GROUP_SIZE / 8; t++ ) {

                aie::vector<int8, PARALLEL_ROWS * 8> a_cc_0 = aie::load_v<PARALLEL_ROWS * 8>(it_q);
                it_q += PARALLEL_ROWS * 8;
                aie::vector<float, PARALLEL_ROWS * 8> a_col_float_0 = aie::to_float(a_cc_0, 0);
                // convert to bfloat16
                aie::accum<accfloat, PARALLEL_ROWS * 8> a_col_accum_0;
                a_col_accum_0.from_vector(a_col_float_0);
                aie::vector<bfloat16, PARALLEL_ROWS * 8> a_col_bf16_0 = a_col_accum_0.template to_vector<bfloat16>();

                // Extract all columns
                aie::vector<bf16, PARALLEL_ROWS> a_col_00 = a_col_bf16_0.extract<PARALLEL_ROWS>(0);
                aie::vector<bf16, PARALLEL_ROWS> a_col_01 = a_col_bf16_0.extract<PARALLEL_ROWS>(1);
                aie::vector<bf16, PARALLEL_ROWS> a_col_02 = a_col_bf16_0.extract<PARALLEL_ROWS>(2);
                aie::vector<bf16, PARALLEL_ROWS> a_col_03 = a_col_bf16_0.extract<PARALLEL_ROWS>(3);
                aie::vector<bf16, PARALLEL_ROWS> a_col_04 = a_col_bf16_0.extract<PARALLEL_ROWS>(4);
                aie::vector<bf16, PARALLEL_ROWS> a_col_05 = a_col_bf16_0.extract<PARALLEL_ROWS>(5);
                aie::vector<bf16, PARALLEL_ROWS> a_col_06 = a_col_bf16_0.extract<PARALLEL_ROWS>(6);
                aie::vector<bf16, PARALLEL_ROWS> a_col_07 = a_col_bf16_0.extract<PARALLEL_ROWS>(7);

                a_col_00 = aie::mul(a_col_00, a_scales);
                a_col_01 = aie::mul(a_col_01, a_scales);
                a_col_02 = aie::mul(a_col_02, a_scales);
                a_col_03 = aie::mul(a_col_03, a_scales);
                a_col_04 = aie::mul(a_col_04, a_scales);
                a_col_05 = aie::mul(a_col_05, a_scales);
                a_col_06 = aie::mul(a_col_06, a_scales);
                a_col_07 = aie::mul(a_col_07, a_scales);

                if constexpr (has_zp){
                    a_col_00 = aie::add(a_col_00, a_mins);
                    a_col_01 = aie::add(a_col_01, a_mins);
                    a_col_02 = aie::add(a_col_02, a_mins);
                    a_col_03 = aie::add(a_col_03, a_mins);
                    a_col_04 = aie::add(a_col_04, a_mins);
                    a_col_05 = aie::add(a_col_05, a_mins);
                    a_col_06 = aie::add(a_col_06, a_mins);
                    a_col_07 = aie::add(a_col_07, a_mins);
                }
            
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
            }
            if constexpr (has_zp){
                it_zp += QXNX_ROW_BLOCK_SIZE;
            }
            it_scale += QXNX_ROW_BLOCK_SIZE;
        }
    }
}
#endif // __Q8_CORE_H__