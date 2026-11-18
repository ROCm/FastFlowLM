// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include "model_spec.h"
#include "aie_kernel_utils.h"
#include "aie_array_layout.h"
#include "flm_rsqrt.h"
// #include "lut_based_ops.h"

void l2_norm(
    bf16 *restrict y
){
    const float epsilon = 1e-6;

    constexpr int vector_size = 16;

    bf16 *it_y_0 = const_cast<bf16*>(y);
    bf16 *it_y_1 = const_cast<bf16*>(y) + vector_size;
    bf16 *it_x_0 = const_cast<bf16*>(y);
    bf16 *it_x_1 = const_cast<bf16*>(y) + vector_size;

    aie::accum<accfloat, vector_size> sum_squares_0 = aie::zeros<accfloat>();
    aie::accum<accfloat, vector_size> sum_squares_1 = aie::zeros<accfloat>();
    aie::vector<float, vector_size> mx_vec;

    const int F = L_V_HEAD_DIM / (2 * vector_size);
    for (int i = 0; i < F; i++){
        auto x_vec_0 = aie::load_v<vector_size>(it_x_0);
        auto x_vec_1 = aie::load_v<vector_size>(it_x_1);
        sum_squares_0 = aie::mac_square(sum_squares_0, x_vec_0);
        sum_squares_1 = aie::mac_square(sum_squares_1, x_vec_1);
        it_x_0 += 2 * vector_size;
        it_x_1 += 2 * vector_size;
    }

    it_x_0 = const_cast<bf16*>(y);
    it_x_1 = const_cast<bf16*>(y) + vector_size;

    mx_vec = aie::add(sum_squares_0.template to_vector<float>(),
                      sum_squares_1.template to_vector<float>());
    float sum = aie::reduce_add(mx_vec);
#if defined(__chess__)
    sum = sum + epsilon;
#else
    // Same math as the chess branch, rounded like chess (see AIE_FADD).
    sum = AIE_FADD(sum, epsilon);
#endif
    const float divrms = flm_rsqrt(sum);

    for (int i = 0; i < F; i++){
        aie::vector<bf16, vector_size> x_vec_0 = aie::load_v<vector_size>(it_x_0);
        aie::vector<bf16, vector_size> x_vec_1 = aie::load_v<vector_size>(it_x_1);
        aie::accum<accfloat, vector_size> x_acc_0;
        aie::accum<accfloat, vector_size> x_acc_1;
        x_acc_0.from_vector(x_vec_0);
        x_acc_1.from_vector(x_vec_1);
        aie::vector<bf16, vector_size> o_vec_0 = aie::mul(x_acc_0.template to_vector<float>(), divrms);
        aie::vector<bf16, vector_size> o_vec_1 = aie::mul(x_acc_1.template to_vector<float>(), divrms);
        aie::store_v(it_y_0, o_vec_0);
        aie::store_v(it_y_1, o_vec_1);
        it_x_0 += 2 * vector_size;
        it_x_1 += 2 * vector_size;
        it_y_0 += 2 * vector_size;
        it_y_1 += 2 * vector_size;
    }
}

void scale_q(
    bf16 *restrict y
){
    const float epsilon = 1e-6;

    constexpr int vector_size = 16;
    const int F = L_V_HEAD_DIM / (2 * vector_size);

    bf16 *it_y_0 = const_cast<bf16*>(y);
    bf16 *it_y_1 = const_cast<bf16*>(y) + vector_size;
    bf16 *it_x_0 = const_cast<bf16*>(y);
    bf16 *it_x_1 = const_cast<bf16*>(y) + vector_size;

    for (int i = 0; i < F; i++){
        aie::vector<bf16, vector_size> x_vec_0 = aie::load_v<vector_size>(it_x_0);
        aie::vector<bf16, vector_size> x_vec_1 = aie::load_v<vector_size>(it_x_1);
        aie::vector<bf16, vector_size> o_vec_0 = aie::mul(x_vec_0, aie::broadcast<bf16, vector_size>(LINEAR_ATTN_SCALE));
        aie::vector<bf16, vector_size> o_vec_1 = aie::mul(x_vec_1, aie::broadcast<bf16, vector_size>(LINEAR_ATTN_SCALE));
        aie::store_v(it_y_0, o_vec_0);
        aie::store_v(it_y_1, o_vec_1);
        it_x_0 += 2 * vector_size;
        it_x_1 += 2 * vector_size;
        it_y_0 += 2 * vector_size;
        it_y_1 += 2 * vector_size;
    }
}

// Lock ids on this tile, mirroring iron/Conv.py.  x and w arrive on separate
// S2MM channels (ch0 / ch1) so each has its own ping/pong pair -- see the Conv
// docstring for why they are not interleaved into one buffer any more.
constexpr int x_prod_lock = 0;
constexpr int x_cons_lock = 1;
constexpr int w_prod_lock = 2;
constexpr int w_cons_lock = 3;
constexpr int y_prod_lock = 4;
constexpr int y_cons_lock = 5;
// alpha / beta never reach this tile any more: the a/b tile scatters them
// straight into the DDR triplet arena, next to the k and v this tile writes.
// Nothing releases these, so acquiring them would deadlock.
// constexpr int ab_prod_lock = 6;
// constexpr int ab_cons_lock = 9;

// Peano's CDO loader does not zero .bss; keep ping/pong selectors in .data.
AIE_PERSIST_ZERO bool is_x_ping = false;
AIE_PERSIST_ZERO bool is_w_ping = false;

// 128 bf16, vectorised -- the packing loop runs 48x per token so this is on the
// critical path and memcpy's byte loop is not worth the risk.
static inline void _copy_head(bf16 *restrict dst, const bf16 *restrict src)
{
    constexpr int vec_size = 32;
    for (int i = 0; i < L_V_HEAD_DIM / vec_size; i++) {
        aie::store_v(dst + i * vec_size, aie::load_v<vec_size>(src + i * vec_size));
    }
}

void _conv(
    bf16* x,
    bf16* w,
    bf16* y
){
    constexpr int vec_size = 16;

    bf16* x_0_a = x;
    bf16* x_0_b = x + vec_size;
    bf16* w_0_a = w;
    bf16* w_0_b = w + vec_size;
    bf16* x_1_a = x + L_V_HEAD_DIM;
    bf16* x_1_b = x + L_V_HEAD_DIM + vec_size;
    bf16* w_1_a = w + L_V_HEAD_DIM;
    bf16* w_1_b = w + L_V_HEAD_DIM + vec_size;
    bf16* x_2_a = x + 2 * L_V_HEAD_DIM;
    bf16* x_2_b = x + 2 * L_V_HEAD_DIM + vec_size;
    bf16* w_2_a = w + 2 * L_V_HEAD_DIM;
    bf16* w_2_b = w + 2 * L_V_HEAD_DIM + vec_size;
    bf16* x_3_a = x + 3 * L_V_HEAD_DIM;
    bf16* x_3_b = x + 3 * L_V_HEAD_DIM + vec_size;
    bf16* w_3_a = w + 3 * L_V_HEAD_DIM;
    bf16* w_3_b = w + 3 * L_V_HEAD_DIM + vec_size;

    bf16* y_a = y;
    bf16* y_b = y + vec_size;

    for (int i = 0; i < L_V_HEAD_DIM / (2 * vec_size); i++){
        aie::vector<bf16, vec_size> x_vec_0_a = aie::load_v<vec_size>(x_0_a);
        aie::vector<bf16, vec_size> x_vec_0_b = aie::load_v<vec_size>(x_0_b);
        aie::vector<bf16, vec_size> w_vec_0_a = aie::load_v<vec_size>(w_0_a);
        aie::vector<bf16, vec_size> w_vec_0_b = aie::load_v<vec_size>(w_0_b);
        aie::vector<bf16, vec_size> x_vec_1_a = aie::load_v<vec_size>(x_1_a);
        aie::vector<bf16, vec_size> x_vec_1_b = aie::load_v<vec_size>(x_1_b);
        aie::vector<bf16, vec_size> w_vec_1_a = aie::load_v<vec_size>(w_1_a);
        aie::vector<bf16, vec_size> w_vec_1_b = aie::load_v<vec_size>(w_1_b);
        aie::vector<bf16, vec_size> x_vec_2_a = aie::load_v<vec_size>(x_2_a);
        aie::vector<bf16, vec_size> x_vec_2_b = aie::load_v<vec_size>(x_2_b);
        aie::vector<bf16, vec_size> w_vec_2_a = aie::load_v<vec_size>(w_2_a);
        aie::vector<bf16, vec_size> w_vec_2_b = aie::load_v<vec_size>(w_2_b);
        aie::vector<bf16, vec_size> x_vec_3_a = aie::load_v<vec_size>(x_3_a);
        aie::vector<bf16, vec_size> x_vec_3_b = aie::load_v<vec_size>(x_3_b);
        aie::vector<bf16, vec_size> w_vec_3_a = aie::load_v<vec_size>(w_3_a);
        aie::vector<bf16, vec_size> w_vec_3_b = aie::load_v<vec_size>(w_3_b);

        aie::accum<accfloat, vec_size> acc_a = aie::mul(x_vec_0_a, w_vec_0_a);
        aie::accum<accfloat, vec_size> acc_b = aie::mul(x_vec_0_b, w_vec_0_b);
        acc_a = aie::mac(acc_a, x_vec_1_a, w_vec_1_a);
        acc_b = aie::mac(acc_b, x_vec_1_b, w_vec_1_b);
        acc_a = aie::mac(acc_a, x_vec_2_a, w_vec_2_a);
        acc_b = aie::mac(acc_b, x_vec_2_b, w_vec_2_b);
        acc_a = aie::mac(acc_a, x_vec_3_a, w_vec_3_a);
        acc_b = aie::mac(acc_b, x_vec_3_b, w_vec_3_b);

        constexpr bfloat16 one = 1.0f;

        aie::vector<float, vec_size> y_vec_f32_a = acc_a.template to_vector<float>();
        aie::vector<float, vec_size> y_vec_f32_b = acc_b.template to_vector<float>();
        aie::vector<bf16, vec_size> y_vec_a      = acc_a.template to_vector<bf16>();
        aie::vector<bf16, vec_size> y_vec_b      = acc_b.template to_vector<bf16>();
        aie::vector<float, vec_size> y_scaled_a = aie::mul(y_vec_a, aie::broadcast<bf16, vec_size>(-exp_scale));
        aie::vector<float, vec_size> y_scaled_b = aie::mul(y_vec_b, aie::broadcast<bf16, vec_size>(-exp_scale));
        aie::vector<bf16, vec_size> y_exp_a = aie::exp2(y_scaled_a);
        aie::vector<bf16, vec_size> y_exp_b = aie::exp2(y_scaled_b);
        aie::vector<bf16, vec_size> vec_one = aie::broadcast<bf16, vec_size>(one);
        aie::accum<accfloat, vec_size> y_exp_acc_a;
        aie::accum<accfloat, vec_size> y_exp_acc_b;
        y_exp_acc_a.from_vector(y_exp_a);
        y_exp_acc_b.from_vector(y_exp_b);
        y_exp_acc_a = aie::add(y_exp_acc_a, vec_one);
        y_exp_acc_b = aie::add(y_exp_acc_b, vec_one);
        aie::vector<float, vec_size> inv_y_a = aie::inv(y_exp_acc_a.template to_vector<float>());
        aie::vector<float, vec_size> inv_y_b = aie::inv(y_exp_acc_b.template to_vector<float>());
        aie::accum<accfloat, vec_size> silu_y_f32_a = aie::mul(inv_y_a, y_vec_f32_a);
        aie::accum<accfloat, vec_size> silu_y_f32_b = aie::mul(inv_y_b, y_vec_f32_b);

        aie::store_v(y_a, silu_y_f32_a.template to_vector<bf16>());
        aie::store_v(y_b, silu_y_f32_b.template to_vector<bf16>());

        x_0_a += 2 * vec_size;
        x_0_b += 2 * vec_size;
        w_0_a += 2 * vec_size;
        w_0_b += 2 * vec_size;
        x_1_a += 2 * vec_size;
        x_1_b += 2 * vec_size;
        w_1_a += 2 * vec_size;
        w_1_b += 2 * vec_size;
        x_2_a += 2 * vec_size;
        x_2_b += 2 * vec_size;
        w_2_a += 2 * vec_size;
        w_2_b += 2 * vec_size;
        x_3_a += 2 * vec_size;
        x_3_b += 2 * vec_size;
        w_3_a += 2 * vec_size;
        w_3_b += 2 * vec_size;
        y_a += 2 * vec_size;
        y_b += 2 * vec_size;
    }
}


extern "C" {
AIE_PERSIST_ZERO static bool is_y_ping = false;

// One invocation = L_TOKENS tokens.  The compute stays at one 128-channel
// slice and loops; the L_TOKENS staging is entirely a memtile concern (the
// memtile replays the CONV_SIZE weight taps once per token and slides the x
// window by one row).  Slice order per token is q, then k, then v, which is
// the order the memtile walks conv_x_buffer's channels in.
void conv(
    bf16 * x_ping,
    bf16 * x_pong,
    bf16 * w_ping,
    bf16 * w_pong,
    bf16 * conv_buffer,
    bf16 * y_buffer_0,
    bf16 * y_buffer_1
)
{
    for (int t = 0; t < L_TOKENS; t++)
    {
        bf16* q_ptr = conv_buffer;
        bf16* k_ptr = conv_buffer + Linear_DQ;
        bf16* v_ptr = conv_buffer + Linear_DQ + Linear_DK;
        for (int i = 0; i < (Linear_DQ) / L_V_HEAD_DIM; i++)
        {
            _lock_acquire(x_cons_lock);
            _lock_acquire(w_cons_lock);
            is_x_ping = !is_x_ping;
            is_w_ping = !is_w_ping;
            bf16* x_using = is_x_ping ? x_ping : x_pong;
            bf16* w_using = is_w_ping ? w_ping : w_pong;
            _conv(x_using, w_using, q_ptr + i * L_V_HEAD_DIM);
            l2_norm(q_ptr + i * L_V_HEAD_DIM);
            scale_q(q_ptr + i * L_V_HEAD_DIM);
            _lock_release(x_prod_lock);
            _lock_release(w_prod_lock);
        }
        for (int i = 0; i < (Linear_DK) / L_V_HEAD_DIM; i++)
        {
            _lock_acquire(x_cons_lock);
            _lock_acquire(w_cons_lock);
            is_x_ping = !is_x_ping;
            is_w_ping = !is_w_ping;
            bf16* x_using = is_x_ping ? x_ping : x_pong;
            bf16* w_using = is_w_ping ? w_ping : w_pong;
            _conv(x_using, w_using, k_ptr + i * L_V_HEAD_DIM);
            l2_norm(k_ptr + i * L_V_HEAD_DIM);
            _lock_release(x_prod_lock);
            _lock_release(w_prod_lock);
        }
        for (int i = 0; i < (Linear_DV) / L_V_HEAD_DIM; i++)
        {
            _lock_acquire(x_cons_lock);
            _lock_acquire(w_cons_lock);
            is_x_ping = !is_x_ping;
            is_w_ping = !is_w_ping;
            bf16* x_using = is_x_ping ? x_ping : x_pong;
            bf16* w_using = is_w_ping ? w_ping : w_pong;
            _conv(x_using, w_using, v_ptr + i * L_V_HEAD_DIM);
            _lock_release(x_prod_lock);
            _lock_release(w_prod_lock);
        }

        // Repack this token into the 48 per-v-head (k | v | q) triplets the
        // delta-rule cores consume.  One 384-word packet per v head, emitted
        // head-ascending; the shim BD behind Y_out scatters each packet into
        // the DDR arena's 130-word sub-blocks, leaving the two-word alpha /
        // beta holes for the a/b tile to fill.  v head hv = G*h + g pairs with
        // qk head h, matching cpu_ref's gq = h / EXPAND.
        constexpr int G = L_V_HEADS / L_QK_HEADS;
        for (int h = 0; h < L_QK_HEADS; h++){
            for (int g = 0; g < G; g++){
                _lock_acquire(y_prod_lock, 1);
                is_y_ping = !is_y_ping;
                bf16* y_using = is_y_ping ? y_buffer_0 : y_buffer_1;
                _copy_head(y_using, k_ptr);
                _copy_head(y_using + L_V_HEAD_DIM, v_ptr);
                _copy_head(y_using + 2 * L_V_HEAD_DIM, q_ptr);
                v_ptr += L_V_HEAD_DIM;
                _lock_release(y_cons_lock, 1);
            }
            q_ptr += L_V_HEAD_DIM;
            k_ptr += L_V_HEAD_DIM;
        }
    }
}


}