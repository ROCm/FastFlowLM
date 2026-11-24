// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include "aie_array_layout.h"
#include "aie_kernel_utils.h"
#include "typedef.hpp"
#include "bf16_proj.h"
#include "lut_based_ops.h"

///@brief

void _activate(
    bf16 * x
){
    for (int i = 0; i < PLI_D / 16; i++) {
        aie::vector<bf16, 16> x_vec = aie::load_v<16>(x);
        aie::vector<bf16, 16> out_vec = getActivationBf16(x_vec);
        aie::store_v(x, out_vec);
        x += 16;
    }
}


extern "C" {

void gate_layer_embedding(
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
    constexpr int final_x_prod_lock = 7;
    constexpr int final_x_cons_lock = 8;

    AIE_PERSIST_ZERO static bool is_proj_w_ping = false;
    alignas(aie::vector_decl_align) float  y_acc[BF16_PROJ_M_BLOCK];
    _left_lock_acquire(final_x_cons_lock);
    _lock_acquire(y_prod_lock);

    #if defined(__chess__)
    memcpy(y, x, MODEL_DIM * sizeof(bf16));
    #else
    copy_vectorized<bf16, MODEL_DIM>(y, x);
    #endif
    _left_lock_release(final_x_prod_lock);
    linear_proj<PLI_D, MODEL_DIM>(
        y + MODEL_DIM,
        proj_w_ping,
        proj_w_pong,
        y,
        y_acc,
        &is_proj_w_ping,
        proj_w_prod_lock,
        proj_w_cons_lock
    );
    _activate(y + MODEL_DIM);
    
    _lock_release(y_cons_lock);

}

}