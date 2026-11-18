// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#include "common.h"
#include "flm_rsqrt.h"

///@brief pseduo rms norm: y = x * w
///@param y: output
///@param x_buf: copy of x
///@param x: input
///@param w: weight
///@param overwrite_x_buf: if true, overwrite x_buf with x
#if !defined(__chess__)
__attribute__((noinline))
#endif
void pseduo_rms_norm(
    bf16 *restrict y,
    const bf16 *restrict x_buf,
    const bf16 *restrict x,
    const bf16 *restrict w,
    bool overwrite_x_buf = true
){
    bf16 *it_y = const_cast<bf16*>(y);
    bf16 *it_x_buf = const_cast<bf16*>(x_buf);
    bf16 *it_x = const_cast<bf16*>(x);
    bf16 *it_w = const_cast<bf16*>(w);
    const float epsilon = 1e-5;

    constexpr int vector_size = 16;
    constexpr float one_over_D = 1.0f / (float)D;

    aie::accum<accfloat, vector_size> sum_squares = aie::zeros<accfloat>();
    aie::vector<float, vector_size> mx_vec;

    const int F = D / vector_size;
    for (int i = 0; i < F; i++)
    chess_prepare_for_pipelining chess_loop_range(128, ) {
        auto x_vec = aie::load_v<vector_size>(it_x);
        sum_squares = aie::mac_square(sum_squares, x_vec);
        aie::store_v(it_x_buf, x_vec);
        it_x += vector_size;
        it_x_buf += vector_size;
    }

    it_x = const_cast<bf16*>(x);

    mx_vec = sum_squares.template to_vector<float>();
    float sum = aie::reduce_add(mx_vec);
    sum = AIE_FMUL(sum, one_over_D);
    sum = AIE_FADD(sum, epsilon);
    const float divrms = flm_rsqrt(sum);

    for (int i = 0; i < F; i++)
    chess_prepare_for_pipelining chess_loop_range(128, ) {
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

///@brief pseduo residual add: y = x + x_buf, and x_buf = y
///@param y: output
///@param x: input
///@param x_buf: copy of x
void pseduo_residual_add(
    bf16 *restrict y,
    const bf16 *restrict x_buf,
    const bf16 *restrict x
){
    bf16 *it_x = const_cast<bf16*>(x);
    bf16 *it_y = const_cast<bf16*>(y);
    bf16 *it_x_buf = const_cast<bf16*>(x_buf);

    for (int i = 0; i < D / 16; i++) {
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



extern "C" {

void rms_residual(
    bf16 *restrict y,
    bf16 *restrict x_ping,
    bf16 *restrict x_pong,
    const bf16 *restrict w,
    bf16 *restrict y_out_0,
    bf16 *restrict y_out_1,
    int* IS_ATTN
){
    constexpr int w_prod_lock = 0;
    constexpr int w_cons_lock = 1;
    constexpr int y_prod_lock = 2;
    constexpr int y_cons_lock = 3;
    constexpr int x_prod_lock = 4;
    constexpr int x_cons_lock = 5;
    constexpr int rtp_available_lock = 6;
    constexpr int lm_head_out_prod_lock = 7;
    constexpr int lm_head_out_cons_lock = 8;

#if !defined(__chess__)
    static __attribute__((section(".data"))) bf16 x_buf[D];
    __attribute__((section(".data")))
#else
    static bf16 x_buf[D];
#endif
    static bool is_lm_head_ping = false;
#if !defined(__chess__)
    __attribute__((section(".data")))
#endif
    static bool is_x_ping = false;
#if defined(__chess__)
    bool &lmp = is_lm_head_ping;
    bool &xp = is_x_ping;
    bf16* x;
    uint32_t* pkt_id_ptr = reinterpret_cast<uint32_t*>(y + 14);
#else
    // Peano miscompiles RMW on persisted .data bools; keep ping/pong state in
    // stack locals, sync to .data once at entry/exit.
    bool lmp = is_lm_head_ping;
    bool xp = is_x_ping;
    bf16* x;
    // Peano may reorder the pkt_id store past the triggering lock release, which
    // misroutes the packet-switched output -> consumer waits forever (deadlock).
    // volatile forces the store to keep its program-order position.
    volatile uint32_t* pkt_id_ptr = reinterpret_cast<volatile uint32_t*>(y + 14);
#endif

    _lock_acquire(rtp_available_lock);

    if (IS_ATTN[0] == 0) { // conv layer
        _lock_acquire(x_cons_lock);
        _lock_acquire(w_cons_lock);
        *pkt_id_ptr = 0;
        xp = !xp;
        x = xp ? x_ping : x_pong;
        pseduo_rms_norm(y + 16, x_buf, x, w, true);
        _lock_release(x_prod_lock);
        _lock_release(w_prod_lock);
        _lock_release(y_cons_lock, conv_d3_repeats);


        _lock_acquire(x_cons_lock);
        _lock_acquire(w_cons_lock);
        _lock_acquire(y_prod_lock, conv_d3_repeats);
        *pkt_id_ptr = 0;
        xp = !xp;
        x = xp ? x_ping : x_pong;
        pseduo_residual_add(y + 16, x_buf, x);
        pseduo_rms_norm(y + 16, x_buf, y + 16, w, false);
        _lock_release(x_prod_lock);
        _lock_release(w_prod_lock);
        _lock_release(y_cons_lock, up_gate_repeats);

        _lock_acquire(x_cons_lock);
        _lock_acquire(y_prod_lock, up_gate_repeats);
        *pkt_id_ptr = 1;
        xp = !xp;
        x = xp ? x_ping : x_pong;
        pseduo_residual_add(y + 16, x_buf, x);
        _lock_release(x_prod_lock);
        _lock_release(y_cons_lock, 1);
        _lock_acquire(y_prod_lock, 1);
    } else if (IS_ATTN[0] == 1) { // attn layer
        _lock_acquire(x_cons_lock);
        _lock_acquire(w_cons_lock);
        *pkt_id_ptr = 0;
        xp = !xp;
        x = xp ? x_ping : x_pong;
        pseduo_rms_norm(y + 16, x_buf, x, w, true);
        _lock_release(x_prod_lock);
        _lock_release(w_prod_lock);
        _lock_release(y_cons_lock, qkv_repeats);


        _lock_acquire(x_cons_lock);
        _lock_acquire(w_cons_lock);
        _lock_acquire(y_prod_lock, qkv_repeats);
        *pkt_id_ptr = 0;
        xp = !xp;
        x = xp ? x_ping : x_pong;
        pseduo_residual_add(y + 16, x_buf, x);
        pseduo_rms_norm(y + 16, x_buf, y + 16, w, false);
        _lock_release(x_prod_lock);
        _lock_release(w_prod_lock);
        _lock_release(y_cons_lock, up_gate_repeats);

        _lock_acquire(x_cons_lock);
        _lock_acquire(y_prod_lock, up_gate_repeats);
        *pkt_id_ptr = 1;
        xp = !xp;
        x = xp ? x_ping : x_pong;
        pseduo_residual_add(y + 16, x_buf, x);
        _lock_release(x_prod_lock);
        _lock_release(y_cons_lock, 1);
        _lock_acquire(y_prod_lock, 1);
    } else if (IS_ATTN[0] == 2) { // lm head
        _lock_acquire(x_cons_lock);
        _lock_acquire(w_cons_lock);
        *pkt_id_ptr = 0;
        xp = !xp;
        x = xp ? x_ping : x_pong;
        pseduo_rms_norm(y + 16, x_buf, x, w, true);
        _lock_release(x_prod_lock);
        _lock_release(w_prod_lock);

        constexpr int VOCAB_SIZE = 65536;
        constexpr int M_PER_ROUND = 32 * 16; // 512
        constexpr int VOCAB_SIZE_PER_ROUND = VOCAB_SIZE / D;
        constexpr int y_repeats_per_round = D / M_PER_ROUND;

        // Peano reorders the producer/consumer lock ops in this pipelined loop
        // -> deadlock; disable pipelining so the lock protocol order is preserved.
#if !defined(__chess__)
        _Pragma("clang loop pipeline(disable) unroll(disable)")
#endif
        for (AIE_LOCK_LOOP_CTR i = 0; i < VOCAB_SIZE_PER_ROUND; i++) {
            _lock_release(y_cons_lock, y_repeats_per_round);
            _lock_acquire(lm_head_out_prod_lock, 1);
            _lock_acquire(x_cons_lock);

            lmp = !lmp;
            bf16* lm_head_dst = lmp ? y_out_0 : y_out_1;
            xp = !xp;
            x = xp ? x_ping : x_pong;
            // Peano emits an out-of-line memcpy call (-fno-builtin-memcpy) that
            // wedges the core -> lock deadlock; use an inline vector copy. Chess
            // keeps memcpy (inlined, byte-identical).
#if defined(__chess__)
            memcpy(lm_head_dst, x, D * sizeof(bf16));
#else
            for (int q = 0; q < D; q += 16) {
                aie::store_v(lm_head_dst + q, aie::load_v<16>(x + q));
            }
#endif

            _lock_release(x_prod_lock);
            _lock_release(lm_head_out_cons_lock, 1);
            _lock_acquire(y_prod_lock, y_repeats_per_round);
        }

    }

#if !defined(__chess__)
    is_lm_head_ping = lmp;
    is_x_ping = xp;
#endif
}

}