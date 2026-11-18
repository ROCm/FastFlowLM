// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#include "aie_array_layout.h"
#include "aie_kernel_utils.h"
#include "typedef.hpp"
#include "flm_rsqrt.h"

/// @file passthrough.cc
/// @brief Normalises the MTP fusion's two input halves and hands them to the
///        projection engine, one D-wide half at a time.

static_assert(MODEL_DIM % 32 == 0,
              "the loops move 32 bf16 per iteration");

/// @brief y = rms(x) * w over MODEL_DIM, vectorised 2 x 16 per iteration.
///
/// Arithmetically identical to rms_residual.cc's rms_norm, down to the fast
/// inverse square root: the two tiles apply the same norm to the same model and
/// must not disagree. Change one, change both. It is static so the two
/// definitions cannot collide if the objects are ever linked into one core.
///
/// @param y: output
/// @param x: input
/// @param w: gain
static void rms_norm(
    bf16 *restrict y,
    const bf16 *restrict x,
    const bf16 *restrict w
){
    const float epsilon = 1e-6;

    constexpr int vector_size = 16;
    constexpr float one_over_D = 1.0f / (float)MODEL_DIM;

    bf16 *it_y_0 = const_cast<bf16*>(y);
    bf16 *it_y_1 = const_cast<bf16*>(y) + vector_size;
    bf16 *it_x_0 = const_cast<bf16*>(x);
    bf16 *it_x_1 = const_cast<bf16*>(x) + vector_size;
    bf16 *it_w_0 = const_cast<bf16*>(w);
    bf16 *it_w_1 = const_cast<bf16*>(w) + vector_size;

    aie::accum<accfloat, vector_size> sum_squares_0 = aie::zeros<accfloat>();
    aie::accum<accfloat, vector_size> sum_squares_1 = aie::zeros<accfloat>();
    aie::vector<float, vector_size> mx_vec;

    const int F = MODEL_DIM / (2 * vector_size);
    for (int i = 0; i < F; i++){
        auto x_vec_0 = aie::load_v<vector_size>(it_x_0);
        auto x_vec_1 = aie::load_v<vector_size>(it_x_1);
        sum_squares_0 = aie::mac_square(sum_squares_0, x_vec_0);
        sum_squares_1 = aie::mac_square(sum_squares_1, x_vec_1);
        it_x_0 += 2 * vector_size;
        it_x_1 += 2 * vector_size;
    }

    it_x_0 = const_cast<bf16*>(x);
    it_x_1 = const_cast<bf16*>(x) + vector_size;

    mx_vec = aie::add(sum_squares_0.template to_vector<float>(),
                      sum_squares_1.template to_vector<float>());
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

    for (int i = 0; i < F; i++){
        aie::vector<bf16, vector_size> x_vec_0 = aie::load_v<vector_size>(it_x_0);
        aie::vector<bf16, vector_size> x_vec_1 = aie::load_v<vector_size>(it_x_1);
        aie::vector<bf16, vector_size> w_vec_0 = aie::load_v<vector_size>(it_w_0);
        aie::vector<bf16, vector_size> w_vec_1 = aie::load_v<vector_size>(it_w_1);
        aie::vector<float, vector_size> wx_vec_0 = aie::mul(x_vec_0, w_vec_0);
        aie::vector<float, vector_size> wx_vec_1 = aie::mul(x_vec_1, w_vec_1);
        aie::vector<bf16, vector_size> o_vec_0 = aie::mul(wx_vec_0, divrms);
        aie::vector<bf16, vector_size> o_vec_1 = aie::mul(wx_vec_1, divrms);
        aie::store_v(it_y_0, o_vec_0);
        aie::store_v(it_y_1, o_vec_1);
        it_x_0 += 2 * vector_size;
        it_x_1 += 2 * vector_size;
        it_y_0 += 2 * vector_size;
        it_y_1 += 2 * vector_size;
        it_w_0 += 2 * vector_size;
        it_w_1 += 2 * vector_size;
    }
}

extern "C" {

/// @param y         packet-framed output, payload at y + 16
/// @param x_ping    inbound half, ping
/// @param x_pong    inbound half, pong
/// @param w_embed   pre_fc_norm_embedding, for the first half
/// @param w_hidden  pre_fc_norm_hidden,    for the second half
void passthrough(
    bf16 *restrict y,
    bf16 *restrict x_ping,
    bf16 *restrict x_pong,
    bf16 *restrict w_embed,
    bf16 *restrict w_hidden
){
    // Mirrors iron/passthrough.py's lock ids -- the two files are one ABI.
    constexpr int w_prod_lock = 0;
    constexpr int w_cons_lock = 1;
    constexpr int y_prod_lock = 2;
    constexpr int y_cons_lock = 3;
    constexpr int x_prod_lock = 4;
    constexpr int x_cons_lock = 5;

#if defined(__chess__)
    uint32_t* pkt_id_ptr = reinterpret_cast<uint32_t*>(y + 14);
#else
    // y is restrict, so Peano assumes the lock calls cannot read the packet id and
    // drops a store that a later one overwrites. The packet then goes to the
    // wrong consumer and the layer hangs. volatile keeps every store.
    volatile uint32_t* pkt_id_ptr = reinterpret_cast<volatile uint32_t*>(y + 14);
#endif

    // Ping/pong parity is static across invocations: two halves per step, so
    // the toggle lands back where it started and never desyncs from the DMA.
    // Peano's CDO loader does not zero .bss; keep ping/pong selectors in .data.
    AIE_PERSIST_ZERO static bool is_x_ping = false;

    // Both gains stay resident for the whole step, so acquire both up front and
    // release both at the end. Acquiring one per half would let the w chain run
    // a lap ahead and refill w_embed while the hidden half still needs its pair.
    _lock_acquire(w_cons_lock, 2);

    // --- half 0: rms(e, pre_fc_norm_embedding), fc_in cols 0 : D ----------
    _lock_acquire(x_cons_lock);
    _lock_acquire(y_prod_lock);
    is_x_ping = !is_x_ping;
    *pkt_id_ptr = pkt_id_pass_to_proj;
    rms_norm(y + 16, is_x_ping ? x_ping : x_pong, w_embed);
    _lock_release(x_prod_lock);
    _lock_release(y_cons_lock);

    // --- half 1: rms(h, pre_fc_norm_hidden),    fc_in cols D : 2D --------
    _lock_acquire(x_cons_lock);
    _lock_acquire(y_prod_lock);
    is_x_ping = !is_x_ping;
    *pkt_id_ptr = pkt_id_pass_to_proj;
    rms_norm(y + 16, is_x_ping ? x_ping : x_pong, w_hidden);
    _lock_release(x_prod_lock);
    _lock_release(y_cons_lock);

    _lock_release(w_prod_lock, 2);
}

}
