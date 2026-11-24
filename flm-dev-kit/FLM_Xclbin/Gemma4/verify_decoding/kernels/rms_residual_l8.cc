// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.
// 8-row rms_residual of the verify layer (plan S6.6, D17): Gemma's sandwich norms on L_TOKENS rows with the residual h[8][D] resident on
// the neighbour tile (3,3) (a core reaches its north window directly, no DMA).  Built only for L_TOKENS > 1; the L_TOKENS = 1 layer keeps
// kernels/rms_residual.cc.
//
//   phase 1  h = x                        y = n_in(h)            -> qkv projection (replay tile, packet id PKT_TO_MAIN)
//   phase 2  h += n_pa(o)                 y = n_pff(h)           -> up|gate projection (PKT_TO_MAIN)
//   phase 3  h += n_pf(down)              y = h                  -> layer output / PLE tail (PKT_OUT; the shim in the stage design)
//
// n_x(v) = bf16(bf16(v * w_x) * 1/sqrt(mean(v^2) + eps)), the arithmetic of the decode kernel (rms_residual.cc rms_norm + residual_add).
// Every phase takes the L_TOKENS rows of its input back to back through the x ping/pong (one row per buffer), folds each into h[] and
// remembers the 1/rms of the next norm; the send pass then builds y from h[] alone, one packet per row.
//
// Norm weights: w = [w_in | w_post_attn | w_pre_ffn | w_post_ffn], 4 x MODEL_DIM resident, one load per dispatch.
#include "aie_array_layout.h"
#include "aie_kernel_utils.h"
#include "typedef.hpp"
#include "flm_rsqrt.h"

#if L_TOKENS > 1
static_assert(L_TOKENS == 8, "the resident rows are passed as h0..h7 (iron/rms_residual_l8.py)");

// packet ids stamped on y (must match iron/design_rmsglu.py)
constexpr uint32_t PKT_TO_MAIN = 16;   // proj_x_in_pkt_id.to_main
constexpr uint32_t PKT_OUT = 24;

static inline float inv_rms(float sum_sq) {
    const float epsilon = 1e-6;
    constexpr float one_over_D = 1.0f / (float)MODEL_DIM;
    float sum = sum_sq;
#if defined(__chess__)
    sum = sum * one_over_D;
#else
    sum = vmulf(sum, one_over_D);
#endif
    sum = sum + epsilon;
    return flm_rsqrt(sum);
}

/// sum of squares of a row
static inline float sum_sq_row(const bf16 *restrict x) {
    constexpr int V = 16;
    const bf16 *it = x;
    aie::accum<accfloat, V> acc = aie::zeros<accfloat>();
    for (int i = 0; i < MODEL_DIM / V; i++) {
        acc = aie::mac_square(acc, aie::load_v<V>(it));
        it += V;
    }
    return aie::reduce_add(acc.template to_vector<float>());
}

/// h = x; returns the sum of squares of x (phase 1: the residual branch is saved, and the norm's reduction rides along)
static inline float save_row(bf16 *restrict h, const bf16 *restrict x) {
    constexpr int V = 16;
    const bf16 *it_x = x;
    bf16 *it_h = h;
    aie::accum<accfloat, V> acc = aie::zeros<accfloat>();
    for (int i = 0; i < MODEL_DIM / V; i++) {
        aie::vector<bf16, V> v = aie::load_v<V>(it_x);
        acc = aie::mac_square(acc, v);
        aie::store_v(it_h, v);
        it_x += V;
        it_h += V;
    }
    return aie::reduce_add(acc.template to_vector<float>());
}

/// h = bf16(h + bf16(bf16-weighted x * divrms)); returns the sum of squares of the new h
static inline float add_normed_row(bf16 *restrict h, const bf16 *restrict x, const bf16 *restrict w, float divrms) {
    constexpr int V = 16;
    const bf16 *it_x = x;
    const bf16 *it_w = w;
    bf16 *it_h = h;
    aie::accum<accfloat, V> acc = aie::zeros<accfloat>();
    for (int i = 0; i < MODEL_DIM / V; i++) {
        aie::vector<bf16, V> x_vec = aie::load_v<V>(it_x);
        aie::vector<bf16, V> w_vec = aie::load_v<V>(it_w);
        aie::vector<bf16, V> h_vec = aie::load_v<V>(it_h);
        aie::vector<float, V> wx = aie::mul(x_vec, w_vec);
        aie::vector<bf16, V> n_vec = aie::mul(wx, divrms);
        aie::vector<bf16, V> out = aie::add(h_vec, n_vec);
        aie::store_v(it_h, out);
        acc = aie::mac_square(acc, out);
        it_x += V;
        it_w += V;
        it_h += V;
    }
    return aie::reduce_add(acc.template to_vector<float>());
}

/// y = bf16(bf16(h * w) * divrms)
static inline void norm_row(bf16 *restrict y, const bf16 *restrict h, const bf16 *restrict w, float divrms) {
    constexpr int V = 16;
    const bf16 *it_h = h;
    const bf16 *it_w = w;
    bf16 *it_y = y;
    for (int i = 0; i < MODEL_DIM / V; i++) {
        aie::vector<bf16, V> h_vec = aie::load_v<V>(it_h);
        aie::vector<bf16, V> w_vec = aie::load_v<V>(it_w);
        aie::vector<float, V> wx = aie::mul(h_vec, w_vec);
        aie::vector<bf16, V> o = aie::mul(wx, divrms);
        aie::store_v(it_y, o);
        it_h += V;
        it_w += V;
        it_y += V;
    }
}

static inline void copy_row(bf16 *restrict y, const bf16 *restrict h) {
    constexpr int V = 16;
    const bf16 *it_h = h;
    bf16 *it_y = y;
    for (int i = 0; i < MODEL_DIM / V; i++) {
        aie::store_v(it_y, aie::load_v<V>(it_h));
        it_h += V;
        it_y += V;
    }
}

extern "C" {
void rms_residual_l8(bf16 *restrict y, bf16 *restrict x_ping, bf16 *restrict x_pong, bf16 *restrict w,
                     bf16 *restrict h0, bf16 *restrict h1, bf16 *restrict h2, bf16 *restrict h3,
                     bf16 *restrict h4, bf16 *restrict h5, bf16 *restrict h6, bf16 *restrict h7) {
    constexpr int w_prod_lock = 0;
    constexpr int w_cons_lock = 1;
    constexpr int y_prod_lock = 2;
    constexpr int y_cons_lock = 3;
    constexpr int x_prod_lock = 4;
    constexpr int x_cons_lock = 5;
    uint32_t *pkt_id_ptr = reinterpret_cast<uint32_t *>(y + 14);
    bf16 *h[L_TOKENS] = {h0, h1, h2, h3, h4, h5, h6, h7};
    const bf16 *w_in = w;
    const bf16 *w_post_attn = w + MODEL_DIM;
    const bf16 *w_pre_ffn = w + 2 * MODEL_DIM;
    const bf16 *w_post_ffn = w + 3 * MODEL_DIM;
    float divrms[L_TOKENS];
    // 3 phases x L_TOKENS rows: even, so the parity survives across invocations (the BD chain starts at x_ping)
    AIE_PERSIST_ZERO static bool is_x_ping = false;

    _lock_acquire(w_cons_lock);

    // ---- phase 1: h = x, y = n_in(h) ----
    for (int t = 0; t < L_TOKENS; t++) {
        _lock_acquire(x_cons_lock);
        is_x_ping = !is_x_ping;
        divrms[t] = inv_rms(save_row(h[t], is_x_ping ? x_ping : x_pong));
        _lock_release(x_prod_lock);
    }
    for (int t = 0; t < L_TOKENS; t++) {
        *pkt_id_ptr = PKT_TO_MAIN;
        norm_row(y + 16, h[t], w_in, divrms[t]);
        _lock_release(y_cons_lock, 1);
        _lock_acquire(y_prod_lock, 1);
    }

    // ---- phase 2: h += n_pa(o), y = n_pff(h) ----
    for (int t = 0; t < L_TOKENS; t++) {
        _lock_acquire(x_cons_lock);
        is_x_ping = !is_x_ping;
        const bf16 *x = is_x_ping ? x_ping : x_pong;
        const float d_o = inv_rms(sum_sq_row(x));
        divrms[t] = inv_rms(add_normed_row(h[t], x, w_post_attn, d_o));
        _lock_release(x_prod_lock);
    }
    for (int t = 0; t < L_TOKENS; t++) {
        *pkt_id_ptr = PKT_TO_MAIN;
        norm_row(y + 16, h[t], w_pre_ffn, divrms[t]);
        _lock_release(y_cons_lock, 1);
        _lock_acquire(y_prod_lock, 1);
    }

    // ---- phase 3: h += n_pf(down); the layer output rows leave (the PLE tail takes them here in the full layer) ----
    for (int t = 0; t < L_TOKENS; t++) {
        _lock_acquire(x_cons_lock);
        is_x_ping = !is_x_ping;
        const bf16 *x = is_x_ping ? x_ping : x_pong;
        const float d_d = inv_rms(sum_sq_row(x));
        (void)add_normed_row(h[t], x, w_post_ffn, d_d);
        _lock_release(x_prod_lock);
    }
    for (int t = 0; t < L_TOKENS; t++) {
        *pkt_id_ptr = PKT_OUT;
        copy_row(y + 16, h[t]);
        _lock_release(y_cons_lock, 1);
        _lock_acquire(y_prod_lock, 1);
    }
    _lock_release(w_prod_lock);
}
}
#endif // L_TOKENS > 1
