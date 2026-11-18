// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#include "common.h"


void cconv(
    bf16* restrict bx_buffer,
    bf16* restrict y_buffer,
    bf16* restrict past_bx_buffer,
    bf16* restrict bcx_buffer,
    bf16* restrict w_buffer
){
    // past_bx_buffer: (CONV_K-1) rows of D, oldest first; the newest row is bx = b * x.
    // w_buffer: CONV_K rows of D; row 0 multiplies the oldest input, row CONV_K-1 the newest.
    bf16* it_b = bcx_buffer;
    bf16* it_c = bcx_buffer + D;
    bf16* it_x2 = bcx_buffer + D * 2;
    bf16* it_y = y_buffer;
    bf16* it_bx_new = bx_buffer;

    const int vector_size = 32;
    const int F = D / vector_size;
    for (int i = 0; i < F; i++) {
        aie::vector<bf16, vector_size> b_vec = aie::load_v<vector_size>(it_b);
        aie::vector<bf16, vector_size> c_vec = aie::load_v<vector_size>(it_c);
        aie::vector<bf16, vector_size> x2_vec = aie::load_v<vector_size>(it_x2);
        aie::vector<bf16, vector_size> bx_new_vec = aie::mul(b_vec, x2_vec);
        aie::store_v(it_bx_new, bx_new_vec);

        aie::accum<accfloat, vector_size> ACC_vec =
            aie::mul(aie::load_v<vector_size>(w_buffer + D * (CONV_K - 1) + i * vector_size), bx_new_vec);
        for (int t = CONV_K - 2; t >= 0; t--) {
            aie::vector<bf16, vector_size> bx_vec = aie::load_v<vector_size>(past_bx_buffer + D * t + i * vector_size);
            aie::vector<bf16, vector_size> w_vec = aie::load_v<vector_size>(w_buffer + D * t + i * vector_size);
            ACC_vec = aie::mac(ACC_vec, bx_vec, w_vec);
        }

        aie::vector<bf16, vector_size> conv_out_vec = ACC_vec.template to_vector<bf16>();
        conv_out_vec = aie::mul(conv_out_vec, c_vec);

        aie::store_v(it_y, conv_out_vec);
        it_y += vector_size;
        it_b += vector_size;
        it_c += vector_size;
        it_x2 += vector_size;
        it_bx_new += vector_size;
    }
}

extern "C" {

void conv(
    bf16* restrict bx_buffer,
    bf16* restrict y_buffer,
    bf16* restrict past_bx_buffer,
    bf16* restrict bcx_buffer,
    bf16* restrict w_buffer
)
{
    constexpr int conv_start_lock = 0;
    constexpr int conv_end_lock = 1;
    constexpr int bx_prod_lock = 2;
    constexpr int bx_cons_lock = 3;
    constexpr int y_prod_lock = 4;
    constexpr int y_cons_lock = 5;
    constexpr int past_bx_prod_lock = 6;
    constexpr int past_bx_cons_lock = 7;
    _lock_acquire(conv_start_lock);
    _lock_acquire(bx_prod_lock);
    _lock_acquire(y_prod_lock, o_repeats);
    _lock_acquire(past_bx_cons_lock);
    cconv(bx_buffer, y_buffer, past_bx_buffer, bcx_buffer, w_buffer);
    _lock_release(bx_cons_lock);
    _lock_release(y_cons_lock, o_repeats);
    _lock_release(past_bx_prod_lock);
    _lock_release(conv_end_lock);
}


}