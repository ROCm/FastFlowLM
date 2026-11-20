// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include "aie_array_layout.h"
#include "aie_kernel_utils.h"
#include "typedef.hpp"
#include "bf16_proj.h"
#include "flm_rsqrt.h"


///@brief  rms norm: y = nrom(x) * w;
///@brief                    x_buf = x
///@param y: output
///@param x_buf: copy of x
///@param x: input
///@param w: weight
void rms_norm(
    bf16 * y,
    const bf16 * x,
    const bf16 * w
){
    bf16 *it_y = const_cast<bf16*>(y);
    bf16 *it_x = const_cast<bf16*>(x);
    bf16 *it_w = const_cast<bf16*>(w);
    const float epsilon = 1e-6;

    constexpr int vector_size = 16;
    constexpr float one_over_D = 1.0f / (float)PLI_D;

    aie::accum<accfloat, vector_size> sum_squares = aie::zeros<accfloat>();
    aie::vector<float, vector_size> mx_vec;

    const int F = PLI_D / vector_size;
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

///@brief residual add: y = x + x_buf, and x_buf = y
///@param y: output
///@param x: input
///@param x_buf: copy of x
void residual_add(
    bf16 * y,
    const bf16 * x_buf,
    const bf16 * x
){
    bf16 *it_x = const_cast<bf16*>(x);
    bf16 *it_y = const_cast<bf16*>(y);
    bf16 *it_x_buf = const_cast<bf16*>(x_buf);

    for (int i = 0; i < PLI_D / 16; i++) {
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


///@brief
template<int M>
void _scale(
    bf16 * y,
    const float scale
){
    for (int i = 0; i < M / 16; i++) {
        aie::vector<bf16, 16> y_vec = aie::load_v<16>(y);
        aie::vector<bf16, 16> out_vec = aie::mul(y_vec, aie::broadcast<bf16, 16>((bf16)scale));
        aie::store_v(y, out_vec);
        y += 16;
    }
}

extern "C" {

void proj_layer_embedding(
    bf16 * norm_w,
    bf16 * x0_per_layer,
    bf16 * x0,
    bf16 * x_proj,
    bf16 * y,
    bf16 * proj_w_ping,
    bf16 * proj_w_pong
){

    // self.norm_w_prod_lock = aie.lock(self.ct, lock_id = 0, init = 1)  
    // self.x0_per_layer_prod_lock = aie.lock(self.ct, lock_id = 1, init = 0)  
    // self.x0_prod_lock = aie.lock(self.ct, lock_id = 2, init = 0)
    // self.xw_cons_lock = aie.lock(self.ct, lock_id = 3, init = 0)
    // self.x_prod_lock = aie.lock(self.ct, lock_id = 4, init = self.repeats)
    // self.x_cons_lock = aie.lock(self.ct, lock_id = 5, init = 0)

    // self.x_proj_prod_lock = aie.lock(self.ct, lock_id = 6, init = 1)
    // self.x_proj_cons_lock = aie.lock(self.ct, lock_id = 7, init = 0)

    // self.y_prod_lock = aie.lock(self.ct, lock_id = 8, init = 1)
    // self.y_cons_lock = aie.lock(self.ct, lock_id = 9, init = 0)
    constexpr int norm_w_prod_lock = 0;
    constexpr int x0_per_layer_prod_lock = 1;
    constexpr int x0_prod_lock = 2;
    constexpr int xw_cons_lock = 3;
    constexpr int proj_w_prod_lock = 4;
    constexpr int proj_w_cons_lock = 5;
    constexpr int y_prod_lock = 6;
    constexpr int y_cons_lock = 7;

    AIE_PERSIST_ZERO static bool is_proj_w_ping = false;
    alignas(aie::vector_decl_align) float  y_acc[BF16_PROJ_M_BLOCK];
    _lock_acquire(xw_cons_lock);
    _lock_acquire(y_prod_lock);

    linear_proj<PLI_D, MODEL_DIM>(
        x_proj,
        proj_w_ping,
        proj_w_pong,
        x0,
        y_acc,
        &is_proj_w_ping,
        proj_w_prod_lock,
        proj_w_cons_lock
    );
    _scale<PLI_D>(x_proj, PER_LAYER_INPUT_SCALE);
    
    rms_norm(x_proj, x_proj, norm_w);
    residual_add(x_proj, x0_per_layer, x_proj);

    _scale<PLI_D>(x_proj, PER_LAYER_MODEL_PROJECTION_SCALE);
    #if defined(__chess__)
    memcpy(y + MODEL_DIM + 32, x_proj, PLI_D * sizeof(bf16));
    #else
    copy_vectorized<bf16, PLI_D>(y + MODEL_DIM + 32, x_proj);
    #endif
    #if defined(__chess__)
    memcpy(y, norm_w + PLI_D, (MODEL_DIM + 32) * sizeof(bf16));
    #else
    copy_vectorized<bf16, MODEL_DIM + 32>(y, norm_w + PLI_D);
    #endif
    _lock_release(y_cons_lock);
    _lock_release(x0_per_layer_prod_lock);

}

}