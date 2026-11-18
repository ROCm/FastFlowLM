// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#include <aie_api/aie.hpp>
#include "common.h"
#include "aie_kernel_utils.h"
#include "q8_core.h"
#include "flm_rsqrt.h"


#define QWEN3_8_27B       0

#ifndef MODEL_TYPE
#define MODEL_TYPE QWEN3_8_27B
#endif


constexpr int VOCAB_SIZE = 248320;
constexpr int NUM_CTS    = 32;

#if MODEL_TYPE == QWEN3_8_27B
constexpr int MODEL_DIM  = 5120;
#else
#error "Unsupported MODEL_TYPE"
#endif


constexpr int   M_PER_ROUND = 32 * 32;

constexpr int   VOCAB_SIZE_PADDED = (VOCAB_SIZE + M_PER_ROUND - 1) / M_PER_ROUND * M_PER_ROUND;
constexpr int   MODEL_DIM_PADDED  = (MODEL_DIM + QXNX_COL_BLOCK_SIZE - 1) / QXNX_COL_BLOCK_SIZE * QXNX_COL_BLOCK_SIZE;

constexpr int ROUNDS   = VOCAB_SIZE_PADDED / M_PER_ROUND;
constexpr int K_BLOCKS = MODEL_DIM_PADDED  / QXNX_COL_BLOCK_SIZE;
void rms_norm(
    bf16 *restrict y,
    bf16 *restrict x,
    bf16 *restrict w
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
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_RANGE(F, F)
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
    sum = sum + epsilon;
#else
    // Same math as the chess branch, rounded like chess (see AIE_FADD).
    sum = AIE_FMUL(sum, one_over_D);
    sum = AIE_FADD(sum, epsilon);
#endif
    const float divrms = flm_rsqrt(sum);
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_RANGE(F, F)
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

constexpr int x_prod_lock = 0;
constexpr int x_cons_lock = 1;
constexpr int w_prod_lock = 2;
constexpr int w_cons_lock = 3;
constexpr int y_prod_lock = 4;
constexpr int y_cons_lock = 5;

template<int m>
void copy_y(bf16 * y, const float * y_acc) {
    for (int i = 0; i < m / 16; i++){
        aie::vector<float, 16> y_vec = aie::load_v<16>(y_acc + i * 16);
        aie::accum<accfloat, 16> acc;
        acc.from_vector(y_vec);
        aie::store_v(y + i * 16, acc.template to_vector<bf16>());
    }
}


extern "C" {
void qmm(
    bf16* y_ping, bf16* y_pong,
    q80_block_t* w_ping, q80_block_t* w_pong,
    bf16* x
){
    // Peano's CDO loader does not zero .bss; keep this ping/pong selector in .data.
#if !defined(__chess__)
    __attribute__((section(".data")))
#endif
    static bool is_y_ping = false;
#if !defined(__chess__)
    __attribute__((section(".data")))
#endif
    static bool is_w_ping = false;

    alignas(aie::vector_decl_align) float  y_acc[QXNX_ROW_BLOCK_SIZE];

    _lock_acquire(x_cons_lock);

    rms_norm(x, x, x + MODEL_DIM);
    for (int r = 0; r < ROUNDS; r++) {

        zero_vectorized<float, QXNX_ROW_BLOCK_SIZE>(y_acc);

        for (int k = 0; k < K_BLOCKS; k++) {
            bf16* x_using = x + k * QXNX_COL_BLOCK_SIZE;

            _lock_acquire(w_cons_lock);
            is_w_ping = !is_w_ping;
            q80_block_t* w_using = is_w_ping ? w_ping : w_pong;

            _Q80_x_bf16(y_acc, x_using, w_using);

            _lock_release(w_prod_lock);
        }

        _lock_acquire(y_prod_lock);

        is_y_ping = !is_y_ping;
        bf16* y_using = is_y_ping ? y_ping : y_pong;
        copy_y<QXNX_ROW_BLOCK_SIZE>(y_using, y_acc);

        _lock_release(y_cons_lock);
    }
    _lock_release(x_prod_lock);
}
}
