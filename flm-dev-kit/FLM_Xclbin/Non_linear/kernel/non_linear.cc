// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include <aie_api/aie.hpp>
#include "aie_kernel_utils.h"
#include <math.h>
#include "typedef.hpp"
#include "lut_based_ops.h"
#include "nonlut_based_ops.h"

void gelu_aie(bf16 *__restrict x, bf16 *__restrict o, int D) {

    bf16 *x_it = x;

    constexpr int vector_size = 16;

    aie::vector<bf16, vector_size> x_vec;
    aie::vector<bf16, vector_size> o_vec;

    const int F = D / vector_size;

    for (int i = 0; i < F; i++)
    chess_prepare_for_pipelining chess_loop_range(128, ) {
        x_vec = aie::load_v<vector_size>(x_it);
        o_vec = getGeluBf16(x_vec);
        aie::store_v(o, o_vec);
        x_it += vector_size;
        o += vector_size;
    }
}

void gelu_aie_non_lut_bf16(bf16 *__restrict x, bf16 *__restrict o, int D) {
    bf16 *x_it = x;

    constexpr int vector_size = 16;

    aie::vector<bf16, vector_size> x_vec;
    aie::vector<bf16, vector_size> o_vec;

    const int F = D / vector_size;

    for (int i = 0; i < F; i++)
    chess_prepare_for_pipelining chess_loop_range(128, ) {
        x_vec = aie::load_v<vector_size>(x_it);
        o_vec = getGeluBf16_nonLUT(x_vec);
        aie::store_v(o, o_vec);
        x_it += vector_size;
        o += vector_size;
    }
}

void silu_aie_non_lut_bf16(bf16 *__restrict x, bf16 *__restrict o, int D) {
    bf16 *x_it = x;

    constexpr int vector_size = 16;

    aie::vector<bf16, vector_size> x_vec;
    aie::vector<bf16, vector_size> o_vec;

    const int F = D / vector_size;

    for (int i = 0; i < F; i++)
    chess_prepare_for_pipelining chess_loop_range(128, ) {
        x_vec = aie::load_v<vector_size>(x_it);
        o_vec = getSiluBf16_nonLUT(x_vec);
        aie::store_v(o, o_vec);
        x_it += vector_size;
        o += vector_size;
    }
}

void gelu_aie_non_lut_float(bf16 *__restrict x, bf16 *__restrict o, int D) {

    bf16 *x_it = x;

    constexpr int vector_size = 16;

    aie::vector<bf16, vector_size> x_vec;
    aie::vector<float, vector_size> o_vec;

    const int F = D / vector_size;

    for (int i = 0; i < F; i++)
    chess_prepare_for_pipelining chess_loop_range(128, ) {
        x_vec = aie::load_v<vector_size>(x_it);
        aie::accum<accfloat, vector_size> x_vec_accfloat;
        x_vec_accfloat.from_vector(x_vec);

        o_vec = getGelufloat16_nonLUT(x_vec_accfloat.template to_vector<float>());
        aie::accum<accfloat, vector_size> o_vec_acc;
        o_vec_acc.from_vector(o_vec);
        aie::store_v(o, o_vec_acc.template to_vector<bfloat16>());
        x_it += vector_size;
        o += vector_size;
    }
}

void sigmoid_aie(bf16 *__restrict x, bf16 *__restrict o, int D) {

    bf16 *x_it = x;

    constexpr int vector_size = 16;

    aie::vector<bf16, vector_size> x_vec;
    aie::vector<bf16, vector_size> o_vec;

    const int F = D / vector_size;

    for (int i = 0; i < F; i++)
    chess_prepare_for_pipelining chess_loop_range(128, ) {
        x_vec = aie::load_v<vector_size>(x_it);
        o_vec = getsigmoidBf16(x_vec);
        aie::store_v(o, o_vec);
        x_it += vector_size;
        o += vector_size;
    }
}

void softplus_aie(bf16 *__restrict x, bf16 *__restrict o, int D) {

    bf16 *x_it = x;

    constexpr int vector_size = 16;

    aie::vector<bf16, vector_size> x_vec;
    aie::vector<bf16, vector_size> o_vec;

    const int F = D / vector_size;

    for (int i = 0; i < F; i++)
    chess_prepare_for_pipelining chess_loop_range(128, ) {
        x_vec = aie::load_v<vector_size>(x_it);
        o_vec = getSoftPlusBf16(x_vec);
        aie::store_v(o, o_vec);
        x_it += vector_size;
        o += vector_size;
    }
}

void exp_aie(bf16 *__restrict x, bf16 *__restrict o, int D) {
    bf16 *x_it = x;

    constexpr int vector_size = 16;

    aie::vector<bf16, vector_size> x_vec;
    aie::vector<bf16, vector_size> o_vec;

    const int F = D / vector_size;

    for (int i = 0; i < F; i++)
    chess_prepare_for_pipelining chess_loop_range(128, ) {
        x_vec = aie::load_v<vector_size>(x_it);

        v16accfloat exp_res =getExpBf16(x_vec);
        o_vec =  to_v16bfloat16(exp_res);
        aie::store_v(o, o_vec);
        x_it += vector_size;
        o += vector_size;
    }
}

void exp_aie_clamped(bf16 *__restrict x, bf16 *__restrict o, int D) {
    bf16 *x_it = x;

    constexpr int vector_size = 16;

    aie::vector<bf16, vector_size> x_vec;
    aie::vector<bf16, vector_size> o_vec;

    const int F = D / vector_size;

    for (int i = 0; i < F; i++)
    chess_prepare_for_pipelining chess_loop_range(128, ) {
        x_vec = aie::load_v<vector_size>(x_it);

        v16accfloat exp_res =getExpBf16_clamped(x_vec); 
        // convert to v16accfloat to bfloat16
        o_vec =  to_v16bfloat16(exp_res);
        aie::store_v(o, o_vec);
        x_it += vector_size;
        o += vector_size;
    }
}

void silu_aie(bf16 *__restrict x, bf16 *__restrict o, int D) {

    bf16 *x_it = x;

    constexpr int vector_size = 16;

    aie::vector<bf16, vector_size> x_vec;
    aie::vector<bf16, vector_size> o_vec;

    const int F = D / vector_size;

    for (int i = 0; i < F; i++)
    chess_prepare_for_pipelining chess_loop_range(128, ) {
        x_vec = aie::load_v<vector_size>(x_it);
        o_vec = getSiluBf16(x_vec);
        aie::store_v(o, o_vec);
        x_it += vector_size;
        o += vector_size;
    }
}

void inv_aie(bf16 *__restrict x, bf16 *__restrict o, int D) {

    bf16 *x_it = x;

    constexpr int vector_size = 16;

    aie::vector<bf16, vector_size> x_vec;
    aie::vector<bf16, vector_size> o_vec;

    const int F = D / vector_size;

    for (int i = 0; i < F; i++)
    chess_prepare_for_pipelining chess_loop_range(128, ) {
        x_vec = aie::load_v<vector_size>(x_it);
        o_vec = aie::inv(x_vec);
        aie::store_v(o, o_vec);
        x_it += vector_size;
        o += vector_size;
    }
}

void inv_lut(bf16 *__restrict x, bf16 *__restrict o, int D) {

    bf16 *x_it = x;
    for (int i = 0; i < D; i++)
    chess_prepare_for_pipelining chess_loop_range(128, ) {
        o[i] = getInvBf16(x[i]);
    }
}

void tanh_aie(bf16 *__restrict x, bf16 *__restrict o, int D) {

    bf16 *x_it = x;

    constexpr int vector_size = 16;

    aie::vector<bf16, vector_size> x_vec;
    aie::vector<bf16, vector_size> o_vec;

    const int F = D / vector_size;

    for (int i = 0; i < F; i++)
    chess_prepare_for_pipelining chess_loop_range(128, ) {
        x_vec = aie::load_v<vector_size>(x_it);
        aie::accum<accfloat, vector_size> x_vec_accfloat;
        x_vec_accfloat.from_vector(x_vec);
        o_vec = aie::tanh(x_vec_accfloat.template to_vector<float>());
        aie::store_v(o, o_vec);
        x_it += vector_size;
        o += vector_size;
    }
}
constexpr int x_prod_lock = 0;
constexpr int x_cons_lock = 1;
constexpr int o_prod_lock = 2;
constexpr int o_cons_lock = 3;
constexpr int l_cons_lock = 4;

extern "C" {
    void non_linear(
        bf16 *__restrict o_ping,
        bf16 *__restrict o_pong,
        bf16 *__restrict x_ping,
        bf16 *__restrict x_pong,
        int * rtp_activate_func_impl,
        int D
    ){
        AIE_PERSIST_ZERO static bool is_x_ping = false;
        AIE_PERSIST_ZERO static bool is_o_ping = false;

        _lock_acquire(l_cons_lock);
        int attn_impl = rtp_activate_func_impl[0];

        _lock_acquire(x_cons_lock);
        _lock_acquire(o_prod_lock);
        is_x_ping = !is_x_ping;
        is_o_ping = !is_o_ping;
        bf16 *x = is_x_ping ? x_ping : x_pong;
        bf16 *o = is_o_ping ? o_ping : o_pong;
        if (attn_impl == 0) {
            gelu_aie(x, o, D);
        } else if (attn_impl == 1) {
            silu_aie(x, o, D);
        } else if (attn_impl == 2) {
            sigmoid_aie(x, o, D);
        } else if (attn_impl == 3) {
            exp_aie(x, o, D);
        } else if (attn_impl == 4) {
            softplus_aie(x, o, D);
        } else if (attn_impl == 5) {
            inv_aie(x, o, D);
        } else if (attn_impl == 6) {
            tanh_aie(x, o, D);
        }else {
            inv_lut(x, o, D);
        }
        _lock_release(x_prod_lock);
        _lock_release(o_cons_lock);
    }
}