// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

// PLE stage for 8 token rows (plan S6.8), norm tile: out[t] = (h[t] + rmsnorm(up[t]) * w_post_pli) * layer_scalar for 8 tokens.
//
// up rows arrive token-major from the memtile stage (2 x [2560] ring), the h rows and the constants from the shim: one transfer
// [w_post_pli(2560) | layer_scalar, pad(32)] first, then 8 h rows [2560].  The rms/residual/scale code is the 1-token per_layer_up.cc's.
//
//   locks (own tile): 0 w_prod (init 1)  1 w_cons  2 up_prod (init 2)  3 up_cons  4 h_prod (init 2)  5 h_cons  6 o_prod (init 2)  7 o_cons
#include "aie_array_layout.h"
#include "aie_kernel_utils.h"
#include "typedef.hpp"
#include "lut_based_ops.h"
#include "flm_rsqrt.h"

constexpr int PLE8_T = 8;

// bring-up switch (iron/ple8_aie.py full stage): 0 = the stage, 1 = out := up, 2 = out := h, 3 = out := w_post | layer_scalar

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

void ple8_norm(
    bf16 * wc,                             // [w_post_pli(2560) | layer_scalar | pad(31)]
    bf16 * up_ping, bf16 * up_pong,        // up rows [2560]
    bf16 * h_ping, bf16 * h_pong,          // h rows [2560]
    bf16 * o_ping, bf16 * o_pong           // out rows [2560]
){
    constexpr int w_prod_lock = 0, w_cons_lock = 1, up_prod_lock = 2, up_cons_lock = 3, h_prod_lock = 4, h_cons_lock = 5, o_prod_lock = 6, o_cons_lock = 7;
    bf16* w_post = wc;

    _lock_acquire(w_cons_lock);
    const bf16 layer_scale = wc[MODEL_DIM];   // only valid once the constants transfer has landed
    for (int t = 0; t < PLE8_T; t++) {
        bf16* up = (t % 2 == 0) ? up_ping : up_pong;
        bf16* h = (t % 2 == 0) ? h_ping : h_pong;
        bf16* o = (t % 2 == 0) ? o_ping : o_pong;
        _lock_acquire(up_cons_lock);
        _lock_acquire(h_cons_lock);
        _lock_acquire(o_prod_lock);
        _rms_norm(o, up, w_post);
        for (int i = 0; i < MODEL_DIM / 16; i++) {
            aie::vector<bf16, 16> a = aie::load_v<16>(o + i * 16);
            aie::vector<bf16, 16> r = aie::load_v<16>(h + i * 16);
            aie::vector<bf16, 16> s = aie::add(a, r);
            aie::store_v(o + i * 16, s);
        }
        _scale<MODEL_DIM>(o, layer_scale);
        _lock_release(up_prod_lock);
        _lock_release(h_prod_lock);
        _lock_release(o_cons_lock);
    }
    _lock_release(w_prod_lock);
}

}
