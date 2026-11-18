// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include "common.h"

#include "lut_based_ops.h"
#include "gptossParam.hpp"
#include "common_external.hpp"
#include "aie_kernel_utils.h"

template<int L>
void pseduo_gelu(bf16 * y, const bf16 * x) {
    bf16* gate_ptr = const_cast<bf16*>(x);
    bf16* up_ptr = const_cast<bf16*>(x) + (L / 2);
    bf16* y_ptr = y;

    static_assert(L%16 == 0);

    AIE_PREPARE_FOR_PIPELINING
    for (int i = 0; i < L / 2; i += 16) {
        aie::vector<bf16, 16> _gate_vec = aie::load_v<16>(gate_ptr + i);
        aie::vector<bf16, 16> _up_vec = aie::load_v<16>(up_ptr + i);

        aie::vector<bfloat16, 16> gate_min_vec = _gate_vec;
        aie::vector<bfloat16, 16> gate_max_vec = aie::broadcast<bfloat16,16>(static_cast<bfloat16>(MLP_LIMIT));
        aie::vector<bfloat16, 16> up_min_vec = aie::broadcast<bfloat16,16>(static_cast<bfloat16>(-MLP_LIMIT));
        aie::vector<bfloat16, 16> up_max_vec = aie::broadcast<bfloat16,16>(static_cast<bfloat16>(MLP_LIMIT));

        aie::vector<bf16, 16> gate_vec_clamp = aie::clamp(_gate_vec, gate_min_vec, gate_max_vec);
        aie::vector<bf16, 16> up_vec_clamp = aie::clamp(_up_vec, up_min_vec, up_max_vec);

        aie::vector<bfloat16, 16> glu = getCustomSILU(gate_vec_clamp);
        
        aie::accum<accfloat, 16> glu_acc;
        glu_acc.from_vector(glu);
        // glu_acc.from_vector(aie::broadcast<bfloat16, 16>(static_cast<bfloat16>(0.0)));
        glu_acc = aie::mac(glu_acc, up_vec_clamp, glu);
        aie::vector<bfloat16, 16> y_vec = glu_acc.template to_vector<bfloat16>();
        aie::store_v(y_ptr + i, y_vec);
    }
}

constexpr int x_prod_lock = 0;
constexpr int x_cons_lock = 1;
constexpr int y_prod_lock = 2;
constexpr int y_cons_lock = 3;
// phase 1: qkv proj, D -> (D + D // 2) to RoPE
// phase 2: o_proj, D -> D to rms_residual
// phase 3: up/gate proj, D -> D * 8 to MT_GLU
// phase 4: down proj, D * 4 -> D, to rms_residual
// phase 4 is special, do not clear y to zero
extern "C" {

void gelu(
    bf16 * y_ping,
    bf16 * y_pong,
    const bf16 * x_ping,
    const bf16 * x_pong,
    int32_t* y_ping_pong_flag,
    int32_t* x_ping_pong_flag,
    uint32_t request_down_project_lock
)
{

    bfloat16* cur_y;
    const bfloat16* cur_x;
    for(int l = 0; l < X_UP_GATE_REPEAT/2; l++){

        if(*y_ping_pong_flag == 0){
            cur_y = y_ping;
            *y_ping_pong_flag = 1;
        }else{
            cur_y = y_pong;
            *y_ping_pong_flag = 0;
        }
        
        if(*x_ping_pong_flag == 0){
            cur_x = x_ping;
            *x_ping_pong_flag= 1;
        }else{
            cur_x = x_pong;
            *x_ping_pong_flag= 0;
        }
        _lock_acquire(x_cons_lock);
        _lock_acquire(y_prod_lock);

        pseduo_gelu<GELU_SLICE>(cur_y, cur_x);

        _lock_release(y_cons_lock);
        _lock_release(x_prod_lock);

    }

    constexpr uint32_t extra_X_pad = (INTERMEDIATE_PADDED_Q4NX_COL-INTERMEDIATE_PADDED_FOUR_Q4NX_ROW)/(GELU_SLICE/2 / 4); // div 4 because 4 column
    for(int l = 0; l < extra_X_pad; l++){
        // only release a temporary y
        if(*y_ping_pong_flag == 0){
            cur_y = y_ping;
            *y_ping_pong_flag = 1;
        }else{
            cur_y = y_pong;
            *y_ping_pong_flag = 0;
        }
        _lock_acquire(y_prod_lock);   
        _lock_release(y_cons_lock);             
    }


    _lock_release(request_down_project_lock, 1); // Assume the lock belong to this CT
}


}
