// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#include "aie_array_layout.h"
#include "aie_kernel_utils.h"
#include "typedef.hpp"
#include "flm_rsqrt.h"


///@brief  1 / sqrt(sum_sq / MODEL_DIM + eps), the tail every reduction pass shares.
///
/// Split out because all three
/// phases reduce in their *drain* pass, over different data -- x, or h + x --
/// and only the accumulation differs.
static inline float inv_rms_tail(
    float sum_sq
){
    const float epsilon = 1e-6;
    constexpr float one_over_D = 1.0f / (float)MODEL_DIM;

#if defined(__chess__)
    float sum = sum_sq * one_over_D;
    sum = sum + epsilon;
#else
    // Same math as the chess branch, rounded like chess (see AIE_FADD).
    float sum = AIE_FMUL(sum_sq, one_over_D);
    sum = AIE_FADD(sum, epsilon);
#endif
    return flm_rsqrt(sum);
}


///@brief  h = x, and 1 / sqrt(mean(x^2) + eps) on the way. Phase 1's drain pass.
///
/// The copy is the point: it is what lets the core release the input buffer
/// straight away. h lives in a neighbour tile's memory module, so this is an
/// ordinary store through a pointer into the core's west or north window -- no
/// DMA, and the neighbours are idle, so nothing arbitrates against it for the
/// bank. The reduction rides along in a pass that would otherwise be pure loads
/// and stores, so it is free.
static inline float inv_rms_save(
    bf16 *restrict h,
    const bf16 *restrict x
){
    constexpr int vector_size = 16;

    bf16 *it_x_0 = const_cast<bf16*>(x);
    bf16 *it_x_1 = const_cast<bf16*>(x) + vector_size;
    bf16 *it_h_0 = h;
    bf16 *it_h_1 = h + vector_size;

    aie::accum<accfloat, vector_size> sum_squares_0 = aie::zeros<accfloat>();
    aie::accum<accfloat, vector_size> sum_squares_1 = aie::zeros<accfloat>();

    const int F = MODEL_DIM / (2 * vector_size);
    for (int i = 0; i < F; i++){
        auto x_vec_0 = aie::load_v<vector_size>(it_x_0);
        auto x_vec_1 = aie::load_v<vector_size>(it_x_1);
        sum_squares_0 = aie::mac_square(sum_squares_0, x_vec_0);
        sum_squares_1 = aie::mac_square(sum_squares_1, x_vec_1);
        aie::store_v(it_h_0, x_vec_0);
        aie::store_v(it_h_1, x_vec_1);
        it_x_0 += 2 * vector_size;
        it_x_1 += 2 * vector_size;
        it_h_0 += 2 * vector_size;
        it_h_1 += 2 * vector_size;
    }

    aie::vector<float, vector_size> mx_vec =
        aie::add(sum_squares_0.template to_vector<float>(),
                 sum_squares_1.template to_vector<float>());
    return inv_rms_tail(aie::reduce_add(mx_vec));
}


///@brief  h += x, and 1 / sqrt(mean(h^2) + eps) on the way. Phase 2's drain pass.
///
/// h is the layer input saved in phase 1 and x is o_proj's output, so the sum is
/// this layer's h1 -- which is what post_attention_layernorm normalises, what the
/// MLP's residual will want, and after that what the next layer takes as its
/// input. Nothing has to move for any of that; the row stays where it is.
///
/// The sum is in registers when it is stored, so squaring it costs a mac and
/// saves the send pass a whole second read of h across the tile boundary.
static inline float residual_add_inv_rms(
    bf16 *restrict h,
    const bf16 *restrict x
){
    constexpr int vector_size = 16;

    bf16 *it_x_0 = const_cast<bf16*>(x);
    bf16 *it_x_1 = const_cast<bf16*>(x) + vector_size;
    bf16 *it_h_0 = h;
    bf16 *it_h_1 = h + vector_size;

    aie::accum<accfloat, vector_size> sum_squares_0 = aie::zeros<accfloat>();
    aie::accum<accfloat, vector_size> sum_squares_1 = aie::zeros<accfloat>();

    const int F = MODEL_DIM / (2 * vector_size);
    for (int i = 0; i < F; i++){
        aie::vector<bf16, vector_size> x_vec_0 = aie::load_v<vector_size>(it_x_0);
        aie::vector<bf16, vector_size> x_vec_1 = aie::load_v<vector_size>(it_x_1);
        aie::vector<bf16, vector_size> h_vec_0 = aie::load_v<vector_size>(it_h_0);
        aie::vector<bf16, vector_size> h_vec_1 = aie::load_v<vector_size>(it_h_1);
        aie::vector<bf16, vector_size> out_vec_0 = aie::add(x_vec_0, h_vec_0);
        aie::vector<bf16, vector_size> out_vec_1 = aie::add(x_vec_1, h_vec_1);
        aie::store_v(it_h_0, out_vec_0);
        aie::store_v(it_h_1, out_vec_1);
        sum_squares_0 = aie::mac_square(sum_squares_0, out_vec_0);
        sum_squares_1 = aie::mac_square(sum_squares_1, out_vec_1);
        it_x_0 += 2 * vector_size;
        it_x_1 += 2 * vector_size;
        it_h_0 += 2 * vector_size;
        it_h_1 += 2 * vector_size;
    }

    aie::vector<float, vector_size> mx_vec =
        aie::add(sum_squares_0.template to_vector<float>(),
                 sum_squares_1.template to_vector<float>());
    return inv_rms_tail(aie::reduce_add(mx_vec));
}


///@brief  h += x. Phase 3's drain pass -- the MLP residual, which is not normed.
static inline void residual_add_inplace(
    bf16 *restrict h,
    const bf16 *restrict x
){
    constexpr int vector_size = 16;
    bf16 *it_x_0 = const_cast<bf16*>(x);
    bf16 *it_x_1 = const_cast<bf16*>(x) + vector_size;
    bf16 *it_h_0 = h;
    bf16 *it_h_1 = h + vector_size;

    for (int i = 0; i < MODEL_DIM / (2 * vector_size); i++) {
        aie::vector<bf16, vector_size> x_vec_0 = aie::load_v<vector_size>(it_x_0);
        aie::vector<bf16, vector_size> x_vec_1 = aie::load_v<vector_size>(it_x_1);
        aie::vector<bf16, vector_size> h_vec_0 = aie::load_v<vector_size>(it_h_0);
        aie::vector<bf16, vector_size> h_vec_1 = aie::load_v<vector_size>(it_h_1);
        auto out_vec_0 = aie::add(x_vec_0, h_vec_0);
        auto out_vec_1 = aie::add(x_vec_1, h_vec_1);
        aie::store_v(it_h_0, out_vec_0);
        aie::store_v(it_h_1, out_vec_1);
        it_x_0 += 2 * vector_size;
        it_x_1 += 2 * vector_size;
        it_h_0 += 2 * vector_size;
        it_h_1 += 2 * vector_size;
    }
}


///@brief h -> y, the send pass of phase 3. No norm and no weight, so this is a
/// copy across the tile boundary into the packet-framed output buffer.
static inline void copy_row(
    bf16 *restrict y,
    const bf16 *restrict h
){
    constexpr int vector_size = 16;
    bf16 *it_h = const_cast<bf16*>(h);
    bf16 *it_y = y;
    for (int i = 0; i < MODEL_DIM / vector_size; i++) {
        aie::store_v(it_y, aie::load_v<vector_size>(it_h));
        it_h += vector_size;
        it_y += vector_size;
    }
}


///@brief y = h * divrms * w, the send pass of the phases that norm.
///
/// Reads h, not the input buffer: by the time this runs the input has been
/// released and refilled several times over.
static inline void norm_scale(
    bf16 *restrict y,
    const bf16 *restrict h,
    const bf16 *restrict w,
    float divrms
){
    constexpr int vector_size = 16;

    bf16 *it_y_0 = const_cast<bf16*>(y);
    bf16 *it_y_1 = const_cast<bf16*>(y) + vector_size;
    bf16 *it_h_0 = const_cast<bf16*>(h);
    bf16 *it_h_1 = const_cast<bf16*>(h) + vector_size;
    bf16 *it_w_0 = const_cast<bf16*>(w);
    bf16 *it_w_1 = const_cast<bf16*>(w) + vector_size;

    const int F = MODEL_DIM / (2 * vector_size);
    for (int i = 0; i < F; i++){
        aie::vector<bf16, vector_size> h_vec_0 = aie::load_v<vector_size>(it_h_0);
        aie::vector<bf16, vector_size> h_vec_1 = aie::load_v<vector_size>(it_h_1);
        aie::vector<bf16, vector_size> w_vec_0 = aie::load_v<vector_size>(it_w_0);
        aie::vector<bf16, vector_size> w_vec_1 = aie::load_v<vector_size>(it_w_1);
        aie::vector<float, vector_size> wx_vec_0 = aie::mul(h_vec_0, w_vec_0);
        aie::vector<float, vector_size> wx_vec_1 = aie::mul(h_vec_1, w_vec_1);
        aie::vector<bf16, vector_size> o_vec_0 = aie::mul(wx_vec_0, divrms);
        aie::vector<bf16, vector_size> o_vec_1 = aie::mul(wx_vec_1, divrms);
        aie::store_v(it_y_0, o_vec_0);
        aie::store_v(it_y_1, o_vec_1);
        it_h_0 += 2 * vector_size;
        it_h_1 += 2 * vector_size;
        it_y_0 += 2 * vector_size;
        it_y_1 += 2 * vector_size;
        it_w_0 += 2 * vector_size;
        it_w_1 += 2 * vector_size;
    }
}


///@brief Both of the layer's norms, and the two residuals around the MLP.
///
/// The norm tile of the full-attention layer, now running all three phases --
/// the same set as ../../gate_delta_net_verify/kernels/rms_residual.cc, which
/// is the verified copy this was ported from.
///
///   phase 1  h = x                   -- the residual branch, kept on the array
///            y = rmsnorm(x) * w_in   -> the projection engine (qkv)
///   phase 2  h += o                  -- the attention residual, h1
///            y = rmsnorm(h) * w_post -> the projection engine (up|gate)
///   phase 3  h += down               -- the MLP residual, h2, out to DDR
///
/// The same 80 KiB of neighbour memory is h, then h1, then h2: each phase's sum
/// overwrites the row in place, so the layer's whole residual chain costs one
/// allocation and never touches DDR.
///
/// Every phase is two passes. The *drain* pass takes all L_TOKENS input rows
/// back to back and only folds each into h[], releasing the input channel as
/// fast as the DMA can fill it; the *send* pass then builds y out of h[] alone.
/// Draining first is free -- h[] is the bit-exact copy of x the residual branch
/// needs anyway -- and avoids the interleaved version's stall, where the core
/// sits on the y producer lock and rows back up in the fabric behind it.
///
/// The residual branch does not stream: an 8-row bf16 activation is 80 KiB
/// against a 64 KiB tile, so h[] points into the memory modules of two idle
/// neighbours, four rows each, via the core's west and north windows. Keeping it
/// resident is what lets phase 2 take L_TOKENS rows rather than 2 * L_TOKENS --
/// the layer input never has to come back from DDR.
///
/// The weight is acquired once per *norm*, not per row and not per phase: the
/// self-chained w BD reloads the single w_buffer between acquires, so the host
/// sends 2 * MODEL_DIM (input_layernorm, then post_attention_layernorm
/// overwriting it) even though three phases run -- phase 3 is a bare residual
/// add and takes none. That reload is the only reason a four-bank tile can run
/// both norms, and the length has to track the norm count exactly.
///
/// Ping/pong parity: is_x_ping is static across invocations and the BD chain
/// starts at x_ping, so the row count per invocation has to stay even. Three
/// phases are 3 * L_TOKENS = 24.
///
/// Unlike the delta-net copy this tile has no second MM2S consumer: the a/b
/// projection is delta-net-only, so the normed row leaves on the y channel alone
/// and the ab_out lock pair is gone.
///
///@param y: packet-framed output, header at y[14..16), data at y[16..)
///@param x_ping, x_pong: ping/pong input rows, filled by the S2MM ch0 BD chain
///@param w: input_layernorm.weight (then post_attention_layernorm.weight)
///@param h0..h7: the resident hidden state, one D-wide row per token, on the
///       neighbour tiles. Order is the ABI -- iron/rms_residual.py appends them
///       in token order after w.

static_assert(L_TOKENS == 8,
              "the resident hidden state is passed as h0..h7; iron/rms_residual.py "
              "appends one argument per token, so this list has to match L_TOKENS");

extern "C" {
void rms_residual(
    bf16 *restrict y,
    bf16 *restrict x_ping,
    bf16 *restrict x_pong,
    bf16 *restrict w,
    bf16 *restrict h0,
    bf16 *restrict h1,
    bf16 *restrict h2,
    bf16 *restrict h3,
    bf16 *restrict h4,
    bf16 *restrict h5,
    bf16 *restrict h6,
    bf16 *restrict h7
){
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

    bf16* h[L_TOKENS] = {h0, h1, h2, h3, h4, h5, h6, h7};

    // The drain pass hands these to the send pass. L_TOKENS floats, the only
    // state the split costs.
    float divrms[L_TOKENS];

    // Toggled once per row. Every phase is L_TOKENS rows, so the parity survives
    // across invocations and stays in step with the BD chain, which starts at
    // x_ping.
    // Peano's CDO loader does not zero .bss; keep ping/pong selectors in .data.
    AIE_PERSIST_ZERO static bool is_x_ping = false;

    // -----------------------------------------------------------------------
    // Phase 1: input_layernorm.
    // -----------------------------------------------------------------------
    // Drain. The unnormalised row is the residual branch, so it is saved before
    // w touches it -- and saving it is what lets the input buffer go back
    // immediately. This is the layer's only read of x from DDR.
    for (int t = 0; t < L_TOKENS; t++) {
        _lock_acquire(x_cons_lock);
        is_x_ping = !is_x_ping;
        divrms[t] = inv_rms_save(h[t], is_x_ping ? x_ping : x_pong);
        _lock_release(x_prod_lock);
    }

    // Send. One y release per row, not QKV_REPEATS: the x replay for the
    // projection is driven from MT[2], which holds all L_TOKENS * MODEL_DIM and
    // re-emits them, so this tile hands each row over once. At L = 1 the single
    // resident row *was* the replay source, and this released QKV_REPEATS.
    _lock_acquire(w_cons_lock);
    for (int t = 0; t < L_TOKENS; t++) {
        *pkt_id_ptr = pkt_id_rms_to_proj;
        norm_scale(y + 16, h[t], w, divrms[t]);
        _lock_release(y_cons_lock, 1);
        _lock_acquire(y_prod_lock, 1);
    }
    _lock_release(w_prod_lock);

    // -----------------------------------------------------------------------
    // Phase 2: the attention residual, then post_attention_layernorm.
    // -----------------------------------------------------------------------
    // The same two passes and the same channels as phase 1 -- x arrives on the
    // same S2MM chain, now carrying o_proj's rows from MT[3] instead of the
    // layer input from the shim. Nothing here distinguishes the phases; the
    // layer's data dependencies order them, and o cannot exist until the whole
    // attention column has run.
    //
    // h[] is the layer input saved by phase 1, so h += o is this layer's h1 --
    // the residual, never re-read from DDR. The sum overwrites h in place, which
    // is what phase 3 will add down_proj's output to.
    //
    // **The weight acquire is what makes a four-bank tile run two norms.** The
    // self-chained w BD refills the single w_buffer between the two acquires, so
    // the host sends 2 * MODEL_DIM -- input_layernorm then
    // post_attention_layernorm -- and the length has to track the live phase
    // count exactly: too few and this parks on the acquire, too many and the
    // surplus desynchronises the next invocation.
    _lock_acquire(w_cons_lock);
    for (int t = 0; t < L_TOKENS; t++) {
        _lock_acquire(x_cons_lock);
        is_x_ping = !is_x_ping;
        divrms[t] = residual_add_inv_rms(h[t], is_x_ping ? x_ping : x_pong);
        _lock_release(x_prod_lock);
    }
    // pkt_id_rms_to_st: the post-attention norm is the MLP's x, so it goes west
    // to the replay tile -- the same destination phase 1's row goes to, and the
    // two are told apart there by arrival order rather than by the id.
    for (int t = 0; t < L_TOKENS; t++) {
        *pkt_id_ptr = pkt_id_rms_to_st;
        norm_scale(y + 16, h[t], w, divrms[t]);
        _lock_release(y_cons_lock, 1);
        _lock_acquire(y_prod_lock, 1);
    }
    _lock_release(w_prod_lock);

    // -----------------------------------------------------------------------
    // Phase 3: the MLP residual, h2 = h1 + down_proj(hid). No norm, so this is
    // an add and a copy out.
    // -----------------------------------------------------------------------
    // **No weight acquire here**, which is why the host sends exactly
    // 2 * MODEL_DIM: the next norm this row meets is the *next layer's*
    // input_layernorm, which is phase 1 of the next invocation.
    //
    // h2 is the layer's output and also the next layer's input, so it leaves
    // under pkt_id_rms_to_out -- the one phase whose y goes south to the shim
    // rather than west to the replay tile. That is the second and last
    // destination set this master port can carry.
    for (int t = 0; t < L_TOKENS; t++) {
        _lock_acquire(x_cons_lock);
        is_x_ping = !is_x_ping;
        residual_add_inplace(h[t], is_x_ping ? x_ping : x_pong);
        _lock_release(x_prod_lock);
    }
    for (int t = 0; t < L_TOKENS; t++) {
        *pkt_id_ptr = pkt_id_rms_to_out;
        copy_row(y + 16, h[t]);
        _lock_release(y_cons_lock, 1);
        _lock_acquire(y_prod_lock, 1);
    }
}
}
