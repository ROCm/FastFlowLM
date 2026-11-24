// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include "aie_array_layout.h"
#include "aie_kernel_utils.h"
#include "typedef.hpp"
#include "bf16_proj.h"
#include "lut_based_ops.h"
#include "flm_rsqrt.h"

///@brief  rms norm: y = nrom(x) * w;
///@brief                    x_buf = x
///@param y: output
///@param x_buf: copy of x
///@param x: input
///@param w: weight
void _rms_norm(
    bf16 * y,
    bf16 * x,
    bf16 * w
){
    bf16 *it_y = const_cast<bf16*>(y);
    bf16 *it_x = const_cast<bf16*>(x);
    bf16 *it_w = const_cast<bf16*>(w);
    const float epsilon = 1e-6;

    constexpr int vector_size = 16;
    constexpr float one_over_D = 1.0f / (float)MODEL_DIM;

    aie::accum<accfloat, vector_size> sum_squares = aie::zeros<accfloat>();
    aie::vector<float, vector_size> mx_vec;

    const int F = MODEL_DIM / vector_size;
    for (int i = 0; i < F; i++){
        auto x_vec = aie::load_v<vector_size>(it_x);
        sum_squares = aie::mac_square(sum_squares, x_vec);
        it_x += vector_size;
    }

    it_x = const_cast<bf16*>(x);

    mx_vec = sum_squares.template to_vector<float>();
    float sum = aie::reduce_add(mx_vec);
    #if defined(__chess__)
    sum = sum * one_over_D;
    #else
    sum = vmulf(sum, one_over_D);
    #endif
    sum = sum + epsilon;
    const float divrms = flm_rsqrt(sum);

    for (int i = 0; i < F; i++){
        aie::vector<bf16, vector_size> x_vec = aie::load_v<vector_size>(it_x);
        aie::vector<bf16, vector_size> w_vec = aie::load_v<vector_size>(it_w);
        aie::vector<float, vector_size> wx_vec = aie::mul(x_vec, w_vec);
        aie::vector<bf16, vector_size> o_vec = aie::mul(wx_vec, divrms);
        aie::store_v(it_y, o_vec);
        it_x += vector_size;
        it_y += vector_size;
        it_w += vector_size;
    }
    event1();
}
///@brief
void residual_add(
    bf16 * y,
    const bf16 * x_buf,
    const bf16 * x
){
    bf16 *it_x = const_cast<bf16*>(x);
    bf16 *it_y = const_cast<bf16*>(y);
    bf16 *it_x_buf = const_cast<bf16*>(x_buf);

    for (int i = 0; i < MODEL_DIM / 16; i++) {
        aie::vector<bf16, 16> x_vec = aie::load_v<16>(it_x);
        aie::vector<bf16, 16> x_buf_vec = aie::load_v<16>(it_x_buf);
        auto out_vec = aie::add(x_vec, x_buf_vec);
        aie::store_v(it_y, out_vec);
        aie::store_v(it_x_buf, out_vec);
        it_x += 16;
        it_y += 16;
        it_x_buf += 16;
    }
}

void _apply_gate(
    bf16 * y,
    const bf16 * gate
){
    for (int i = 0; i < PLI_D / 16; i++) {
        aie::vector<bf16, 16> y_vec = aie::load_v<16>(y);
        aie::vector<bf16, 16> gate_vec = aie::load_v<16>(gate);
        aie::vector<bf16, 16> out_vec = aie::mul(y_vec, gate_vec);
        aie::store_v(y, out_vec);
        y += 16;
        gate += 16;
    }
}

///@brief
template<int M>
void _scale(
    bf16 * y,
    const bf16 scale
){
    for (int i = 0; i < M / 16; i++) {
        aie::vector<bf16, 16> y_vec = aie::load_v<16>(y);
        aie::vector<bf16, 16> out_vec = aie::mul(y_vec, aie::broadcast<bf16, 16>(scale));
        aie::store_v(y, out_vec);
        y += 16;
    }
}
extern "C" {

void per_layer_up(
    bf16 * x,
    bf16 * proj_w_ping,
    bf16 * proj_w_pong,
    bf16 * y
){

   
    constexpr int x_prod_lock = 0;
    constexpr int x_cons_lock = 1;
    constexpr int proj_w_prod_lock = 2;
    constexpr int proj_w_cons_lock = 3;
    constexpr int y_prod_lock = 4;
    constexpr int y_cons_lock = 5;

    AIE_PERSIST_ZERO static bool is_proj_w_ping = false;

    alignas(aie::vector_decl_align) float  y_acc[BF16_PROJ_M_BLOCK];
    bf16* w_post_per_layer_input_norm = x;
    bf16* layer_scale = x + MODEL_DIM;
    bf16* per_layer_input = x + MODEL_DIM + 32;
    bf16* residual_x = x + MODEL_DIM + 32 + PLI_D;
    bf16* gate = x + MODEL_DIM + 32 + PLI_D + MODEL_DIM;

    _lock_acquire(x_cons_lock);
    _lock_acquire(y_prod_lock);

    _apply_gate(per_layer_input, gate);

    linear_proj<MODEL_DIM, PLI_D>(
        y,
        proj_w_ping,
        proj_w_pong,
        per_layer_input,
        y_acc,
        &is_proj_w_ping,
        proj_w_prod_lock,
        proj_w_cons_lock
    );

    _rms_norm(y, y, w_post_per_layer_input_norm);
    residual_add(y, residual_x, y);
    _scale<MODEL_DIM>(y, *layer_scale);
    
    _lock_release(y_cons_lock);
    _lock_release(x_prod_lock);

}

}