// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __BF16_PROJ_H__
#define __BF16_PROJ_H__
#include "aie_kernel_utils.h"
#include "model_spec.h"


template<int M, int K>
void _mvm_bf16_bf16(
    bf16* w,
    bf16* x,
    float* y_acc
){
    aie::vector<float, M> y_acc_vec = aie::load_v<M>(y_acc);
    aie::accum<accfloat, M> acc;

    constexpr int x_vec_size = 32;
    bf16* w_ptr = w;
    acc.from_vector(y_acc_vec);
    for (int i = 0; i < K / x_vec_size; i++) {
        aie::vector<bf16, x_vec_size> b_col = aie::load_v< x_vec_size>(x + i * x_vec_size);
    
        for (int j = 0; j < x_vec_size; j++) {
            aie::vector<bf16, M> w_col = aie::load_v<M>(w_ptr);
            bf16 b_col_j = b_col[j];
            acc = aie::mac(acc, w_col, b_col_j);
            w_ptr += M;
        }
    }
    aie::store_v(y_acc, acc.template to_vector<float>());
}

template<int M, int K>
void linear_proj(
    bf16 * y,
    bf16 * w_ping,
    bf16 * w_pong,
    bf16 * x,
    float* y_acc,
    bool* is_w_ping,
    const int w_prod_lock_id,
    const int w_cons_lock_id
) {

    constexpr int m = BF16_PROJ_M_BLOCK;
    constexpr int k = BF16_PROJ_K_BLOCK;

    for (int i = 0; i < M / m; i++) {
        zero_vectorized<float, m>(y_acc);
        for (int j = 0; j < K / k; j++) {
            (*is_w_ping) = !(*is_w_ping);

            bf16* w_using = (*is_w_ping) ? w_ping : w_pong;
            bf16* x_using = x + j * k;
            _lock_acquire(w_cons_lock_id);
            _mvm_bf16_bf16<m, k>(w_using, x_using, y_acc);
            _lock_release(w_prod_lock_id);
        }

        // store the y
        bf16* y_using = y + i * m;
        aie::vector<float, m> y_acc_vec = aie::load_v<m>(y_acc);
        aie::accum<accfloat, m> c_acc;
        c_acc.from_vector(y_acc_vec);
        aie::vector<bf16, m> y_vec = c_acc.template to_vector<bf16>();
        aie::store_v(y_using, y_vec);
    }
}
#endif // __BF16_PROJ_H__