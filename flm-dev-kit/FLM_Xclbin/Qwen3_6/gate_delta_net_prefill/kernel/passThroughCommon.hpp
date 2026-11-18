// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef PASS_THROUGH_COMMON_HPP
#define PASS_THROUGH_COMMON_HPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <aie_api/aie.hpp>

#include "aie_kernel_utils.h" //TODO: a better method?














template<int m>
void copy_float_to_bf16(bfloat16 * y, const float * y_acc) {
    static_assert(m%16 == 0);
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_UNROLL_FULL
    for (int i = 0; i < m / 16; i++){
        aie::vector<float, 16> y_vec = aie::load_v<16>(y_acc + i * 16);
        aie::accum<accfloat, 16> acc;
        acc.from_vector(y_vec);
        aie::store_v(y + i * 16, acc.template to_vector<bfloat16>());
    }
}

template<int m>
void copy_bf16_to_float(float * y, const bfloat16 * y_bf16) {
    static_assert(m%16 == 0);
    AIE_PREPARE_FOR_PIPELINING
    for (int i = 0; i < m / 16; i++){
        aie::vector<bfloat16, 16> y_vec = aie::load_v<16>(y_bf16 + i * 16);
        aie::accum<accfloat, 16> acc;
        acc.from_vector(y_vec);
        aie::store_v(y + i * 16, acc.template to_vector<float>());
    }
}


template<int m>
void copy_bf16_to_bf16( bfloat16 *y, const bfloat16 * y_bf16) {
    static_assert(m%16 == 0);
    AIE_PREPARE_FOR_PIPELINING
    for (int i = 0; i < m / 16; i++){
        aie::vector<bfloat16, 16> y_vec = aie::load_v<16>(y_bf16 + i * 16);
        aie::store_v(y + i * 16, y_vec);
    }

}
template<int m, int vec_size=16> 
void copy_float_to_float(float*y, const float* y_in){

    static_assert(m%vec_size == 0);
    AIE_PREPARE_FOR_PIPELINING
    for (int i = 0; i < m / vec_size; i++){
        aie::vector<float, vec_size> y_vec = aie::load_v<vec_size>(y_in + i * vec_size);
        aie::store_v(y + i * vec_size, y_vec);
    }


}








#endif // PASS_THROUGH_COMMON_HPP
