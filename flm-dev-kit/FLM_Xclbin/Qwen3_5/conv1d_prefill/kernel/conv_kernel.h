// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef CONV_KERNEL_H
#define CONV_KERNEL_H
#include <aie_api/aie.hpp>
#include "aie_kernel_utils.h"
#include <math.h>

void _conv_kernel(
    bf16* w_0,
    bf16* w_1,
    bf16* w_2,
    bf16* w_3,
    bf16* x_0,
    bf16* x_1,
    bf16* x_2,
    bf16* x_3,
    bf16* y
){
    constexpr int vec_size = 32;
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_RANGE(256/vec_size, 256/vec_size)
    for (int i = 0; i < 256 / vec_size; i++) AIE_LOOP_FLATTEN{
        aie::vector<bf16, vec_size> x_vec_0 = aie::load_v<vec_size>(x_0);
        aie::vector<bf16, vec_size> w_vec_0 = aie::load_v<vec_size>(w_0);
        aie::vector<bf16, vec_size> x_vec_1 = aie::load_v<vec_size>(x_1);
        aie::vector<bf16, vec_size> w_vec_1 = aie::load_v<vec_size>(w_1);
        aie::vector<bf16, vec_size> x_vec_2 = aie::load_v<vec_size>(x_2);
        aie::vector<bf16, vec_size> w_vec_2 = aie::load_v<vec_size>(w_2);
        aie::vector<bf16, vec_size> x_vec_3 = aie::load_v<vec_size>(x_3);
        aie::vector<bf16, vec_size> w_vec_3 = aie::load_v<vec_size>(w_3);

        aie::accum<accfloat, vec_size> acc = aie::mul(x_vec_0, w_vec_0);
        acc = aie::mac(acc, x_vec_1, w_vec_1);
        acc = aie::mac(acc, x_vec_2, w_vec_2);
        acc = aie::mac(acc, x_vec_3, w_vec_3);

        aie::vector<float, vec_size> y_vec_f32 = acc.template to_vector<float>();
        aie::vector<bf16, vec_size> y_vec      = acc.template to_vector<bf16>();
        aie::vector<float, vec_size> y_scaled = aie::negmul(y_vec, exp_scale);
        aie::vector<bf16, vec_size> y_exp = aie::exp2(y_scaled);
        aie::accum<accfloat, vec_size> y_exp_acc;
        y_exp_acc.from_vector(y_exp);
        y_exp_acc = aie::add(y_exp_acc, (bf16)1.0f);
        aie::vector<float, vec_size> inv_y = aie::inv(y_exp_acc.template to_vector<float>());
        aie::accum<accfloat, vec_size> silu_y_f32 = aie::mul(inv_y, y_vec_f32);

        aie::store_v(y + i * vec_size, silu_y_f32.template to_vector<bf16>());

        x_0 += vec_size;
        w_0 += vec_size;
        x_1 += vec_size;
        w_1 += vec_size;
        x_2 += vec_size;
        w_2 += vec_size;
        x_3 += vec_size;
        w_3 += vec_size;
    }
}

void l2_norm(
    bf16 *restrict y
){
    bf16 *it_y = const_cast<bf16*>(y);
    bf16 *it_x = const_cast<bf16*>(y);
    const float epsilon = 1e-6;

    constexpr int vector_size = 32;

    aie::accum<accfloat, vector_size> sum_squares = aie::zeros<accfloat, vector_size>();

    constexpr int F = 128 / vector_size;
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_UNROLL_FULL
    for (int i = 0; i < F; i++){
        auto x_vec = aie::load_v<vector_size>(it_x);
        sum_squares = aie::mac_square(sum_squares, x_vec);
        it_x += vector_size;
    }

    it_x = const_cast<bf16*>(y);

    auto mx_vec = sum_squares.template to_vector<float>();
    float sum = aie::reduce_add(mx_vec);
#if defined(__chess__)
    sum = sum + epsilon;
#else
    sum = AIE_FADD(sum, epsilon);  // round like chess
#endif

    aie::vector<float, vector_size> v_sum = aie::broadcast<float, vector_size>(sum);
    // Use vector inverse square root for better performance and avoidance of scalar latency
    aie::vector<float, vector_size> v_divrms = aie::inv(aie::sqrt(v_sum));

    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_UNROLL_FULL
    for (int i = 0; i < F; i++) AIE_LOOP_FLATTEN{
        aie::vector<bf16, vector_size> x_vec = aie::load_v<vector_size>(it_x);
        aie::accum<accfloat, vector_size> x_acc;
        x_acc.from_vector(x_vec);
        aie::accum<accfloat, vector_size> o_acc = aie::mul(x_acc.template to_vector<float>(), v_divrms);
        aie::store_v(it_y, o_acc.template to_vector<bf16>());
        it_x += vector_size;
        it_y += vector_size;
    }
}

template <typename T_in, typename T_out, int N>
void passThrough_aie(T_out *restrict out, T_in *restrict in) {
    const int vec_factor = 64;

    aie::vector<T_out, vec_factor> Out;

    constexpr int F = N / vec_factor;
    
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_RANGE(F, F)
    for (int i = 0; i < F; i++){
        aie::accum<accfloat, vec_factor> c_acc_in;
        c_acc_in.from_vector(aie::load_v<vec_factor>(in));
        aie::store_v(out, c_acc_in.template to_vector<T_out>());
        in += vec_factor;
        out += vec_factor;
    }
}

#endif