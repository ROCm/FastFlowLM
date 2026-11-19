// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
#include "aie_array_layout.h"
#include "aie_kernel_utils.h"
#include "typedef.hpp"
#include "flm_rsqrt.h"


///@brief  rms norm: y = nrom(x) * w;
///@brief                    x_buf = x
///@param y: output
///@param x_buf: copy of x
///@param x: input
///@param w: weight
void rms_norm(
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


///@brief residual add: y = x + x_buf, and x_buf = y
///@param y: output
///@param x: input
///@param x_buf: copy of x
void residual_add(
    bf16 *restrict y,
    const bf16 *restrict x_buf,
    const bf16 *restrict x
){
    constexpr int vector_size = 16;
    bf16 *it_x_0 = const_cast<bf16*>(x);
    bf16 *it_x_1 = const_cast<bf16*>(x) + vector_size;
    bf16 *it_y_0 = const_cast<bf16*>(y);
    bf16 *it_y_1 = const_cast<bf16*>(y) + vector_size;
    bf16 *it_x_buf_0 = const_cast<bf16*>(x_buf);
    bf16 *it_x_buf_1 = const_cast<bf16*>(x_buf) + vector_size;

    for (int i = 0; i < MODEL_DIM / (2 * vector_size); i++) {
        aie::vector<bf16, vector_size> x_vec_0 = aie::load_v<vector_size>(it_x_0);
        aie::vector<bf16, vector_size> x_vec_1 = aie::load_v<vector_size>(it_x_1);
        aie::vector<bf16, vector_size> x_buf_vec_0 = aie::load_v<vector_size>(it_x_buf_0);
        aie::vector<bf16, vector_size> x_buf_vec_1 = aie::load_v<vector_size>(it_x_buf_1);
        auto out_vec_0 = aie::add(x_vec_0, x_buf_vec_0);
        auto out_vec_1 = aie::add(x_vec_1, x_buf_vec_1);
        aie::store_v(it_y_0, out_vec_0);
        aie::store_v(it_y_1, out_vec_1);
        aie::store_v(it_x_buf_0, out_vec_0);
        aie::store_v(it_x_buf_1, out_vec_1);
        it_x_0 += 2 * vector_size;
        it_x_1 += 2 * vector_size;
        it_y_0 += 2 * vector_size;
        it_y_1 += 2 * vector_size;
        it_x_buf_0 += 2 * vector_size;
        it_x_buf_1 += 2 * vector_size;
    }
}


///@brief


extern "C" {
// #define FORCE_SYNC_X
// STATELESS_X (build variant "<MODEL_TYPE>_SL"): every run takes x from DDR and gives the new x back to DDR,
// so nothing the next run needs stays on the array. Without it the residual stream (x_buf and the x
// ping/pong buffers of this core) is carried from run to run inside a layer group, and a firmware design
// reload between two runs (any other process's command) silently wipes it. FORCE_SYNC_X is the existing
// receive-x-at-the-start / send-x-at-the-end flow; STATELESS_X selects it for every run and makes the
// final residual of the linear runs land in the packet buffer y (which is what gets sent).
#ifdef STATELESS_X
#define FORCE_SYNC_X
#endif
void rms_residual(
    bf16 *restrict y,
    bf16 *restrict x_ping,
    bf16 *restrict x_pong,
    bf16 *restrict w,
    bf16 *restrict y_out,
    bf16 *restrict x_buf,
    int32_t *restrict rtp
){
    FLM_SET_FLOOR_ROUNDING();
    KERNEL_RNE_SCOPE(8)
    constexpr int w_prod_lock = 0;
    constexpr int w_cons_lock = 1;
    constexpr int y_prod_lock = 2;
    constexpr int y_cons_lock = 3;
    constexpr int x_prod_lock = 4;
    constexpr int x_cons_lock = 5;
    constexpr int rtp_available_lock = 6;
    constexpr int ab_out_prod_lock = 7;
    constexpr int ab_out_cons_lock = 8;
    uint32_t* pkt_id_ptr = reinterpret_cast<uint32_t*>(y + 14);
    bf16* x;

    AIE_PERSIST_ZERO static bool is_x_ping = false;
    AIE_PERSIST_ZERO static bool is_ab_ping = false;
    AIE_PERSIST_ZERO static bool is_w_ping = false;

    _lock_acquire(rtp_available_lock);
    const int mode = rtp[0];
    if (mode == RMS_MODE_LM_HEAD) {
        // final norm with model.norm, no residual; the projection engine needs x once per 512-row round
        constexpr int rounds = LM_HEAD_ROWS / M_PER_ROUND;
        constexpr int batch = 32;
        _lock_acquire(x_cons_lock);
        _lock_acquire(w_cons_lock);
        is_x_ping = !is_x_ping;
        x = is_x_ping ? x_ping : x_pong;
        *pkt_id_ptr = pkt_id_rms_to_proj;
        rms_norm(y + 16, x, w);
        _lock_release(x_prod_lock);
        _lock_release(w_prod_lock);
        for (int sent = 0; sent < rounds; sent += batch) {
            const int n = (rounds - sent < batch) ? (rounds - sent) : batch;
            _lock_release(y_cons_lock, n);
            _lock_acquire(y_prod_lock, n);
        }
    }
    else if (mode == RMS_MODE_ATTN) {
        // input later norm + residul
#ifdef FORCE_SYNC_X
        _lock_acquire(x_cons_lock);
        is_x_ping = !is_x_ping;
#endif
        x = is_x_ping ? x_ping : x_pong;
    
        _lock_acquire(w_cons_lock);
        *pkt_id_ptr = pkt_id_rms_to_proj;
#ifdef FORCE_SYNC_X
        _memcpy<MODEL_DIM * sizeof(bf16)>((uint8*)x_buf, (uint8*)x);
#endif
        rms_norm(y + 16, x_buf, w);
        _lock_release(x_prod_lock);
        _lock_release(w_prod_lock);
        _lock_release(y_cons_lock, QKV_REPEATS);


        _lock_acquire(x_cons_lock);

        _lock_acquire(y_prod_lock, QKV_REPEATS);
        *pkt_id_ptr = pkt_id_rms_to_proj;
        is_x_ping = !is_x_ping;
        x = is_x_ping ? x_ping : x_pong;
        residual_add(x_buf, x_buf, x);
        // pre_feed_forward
        _lock_acquire(w_cons_lock);
        rms_norm(y + 16, x_buf, w);
        _lock_release(w_prod_lock);

        _lock_release(x_prod_lock);
        _lock_release(y_cons_lock, UP_GATE_REPEATS);

        _lock_acquire(x_cons_lock);
        _lock_acquire(y_prod_lock, UP_GATE_REPEATS);
        *pkt_id_ptr = pkt_id_rms_to_it;
        is_x_ping = !is_x_ping;
        x = is_x_ping ? x_ping : x_pong;
        residual_add(y + 16, x_buf, x);
        _lock_release(x_prod_lock);
        _lock_release(y_cons_lock, 1);
        _lock_acquire(y_prod_lock, 1);
    }
    else if (mode == RMS_MODE_FIRST_LINEAR){
        // input later norm + residul
        _lock_acquire(x_cons_lock);
        _lock_acquire(w_cons_lock);
        _lock_acquire(ab_out_prod_lock);
        *pkt_id_ptr = pkt_id_rms_to_proj;
        is_x_ping = !is_x_ping;
        x = is_x_ping ? x_ping : x_pong;
        _memcpy<MODEL_DIM * sizeof(bf16)>((uint8*)x_buf, (uint8*)x);
        rms_norm(y_out, x_buf, w);
        _memcpy<MODEL_DIM * sizeof(bf16)>((uint8*)(y + 16), (uint8*)(y_out));
        _lock_release(x_prod_lock);
        _lock_release(w_prod_lock);
        _lock_release(y_cons_lock, L_QKV_REPEATS);
        _lock_release(ab_out_cons_lock);


        _lock_acquire(x_cons_lock);

        _lock_acquire(y_prod_lock, L_QKV_REPEATS);
        *pkt_id_ptr = pkt_id_rms_to_proj;
        is_x_ping = !is_x_ping;
        x = is_x_ping ? x_ping : x_pong;
        residual_add(x_buf, x_buf, x);
        // pre_feed_forward
        _lock_acquire(w_cons_lock);
        rms_norm(y + 16, x_buf, w);
        _lock_release(w_prod_lock);

        _lock_release(x_prod_lock);
        _lock_release(y_cons_lock, UP_GATE_REPEATS);

        _lock_acquire(x_cons_lock);
        _lock_acquire(y_prod_lock, UP_GATE_REPEATS);
        is_x_ping = !is_x_ping;
        x = is_x_ping ? x_ping : x_pong;
#ifdef STATELESS_X
        residual_add(y + 16, x_buf, x);
#else
        residual_add(x, x_buf, x);
#endif
#ifdef FORCE_SYNC_X
        _lock_release(x_prod_lock);
        *pkt_id_ptr = pkt_id_rms_to_it;
        _lock_release(y_cons_lock, 1);
        _lock_acquire(y_prod_lock, 1);
#endif
    }
    else {
#ifdef FORCE_SYNC_X
        _lock_acquire(x_cons_lock);
        is_x_ping = !is_x_ping;
#endif
        _lock_acquire(w_cons_lock);
        _lock_acquire(ab_out_prod_lock);
        *pkt_id_ptr = pkt_id_rms_to_proj;
        x = is_x_ping ? x_ping : x_pong;
#ifdef FORCE_SYNC_X
        _memcpy<MODEL_DIM * sizeof(bf16)>((uint8*)x_buf, (uint8*)x);
#endif
        rms_norm(y_out, x_buf, w);
        _memcpy<MODEL_DIM * sizeof(bf16)>((uint8*)(y + 16), (uint8*)(y_out));
        // memcpy(y_out, y + 16, MODEL_DIM * sizeof(bf16));
        _lock_release(x_prod_lock);
        _lock_release(w_prod_lock);
        _lock_release(y_cons_lock, L_QKV_REPEATS);
        _lock_release(ab_out_cons_lock);


        _lock_acquire(x_cons_lock);

        _lock_acquire(y_prod_lock, L_QKV_REPEATS);
        *pkt_id_ptr = pkt_id_rms_to_proj;
        is_x_ping = !is_x_ping;
        x = is_x_ping ? x_ping : x_pong;
        residual_add(x_buf, x_buf, x);
        // pre_feed_forward
        _lock_acquire(w_cons_lock);
        rms_norm(y + 16, x_buf, w);
        _lock_release(w_prod_lock);

        _lock_release(x_prod_lock);
        _lock_release(y_cons_lock, UP_GATE_REPEATS);

        _lock_acquire(x_cons_lock);
        _lock_acquire(y_prod_lock, UP_GATE_REPEATS);
        is_x_ping = !is_x_ping;
        x = is_x_ping ? x_ping : x_pong;
#ifdef STATELESS_X
        residual_add(y + 16, x_buf, x);
#else
        residual_add(x, x_buf, x);
#endif
        // _lock_release(x_prod_lock);
#ifdef FORCE_SYNC_X
        *pkt_id_ptr = pkt_id_rms_to_it;
        _lock_release(x_prod_lock);
        _lock_release(y_cons_lock, 1);
        _lock_acquire(y_prod_lock, 1);
#endif
    }
}

}