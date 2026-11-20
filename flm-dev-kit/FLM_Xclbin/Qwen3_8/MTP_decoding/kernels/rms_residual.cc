// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

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
/// @brief The draft layer's two norms and its two residuals, decode shape.
///
///     phase 1   xn = rms(fc_out, input_layernorm)          -> qkv and z
///     phase 2   h  = fc_out + W_o @ o                      -- the attention residual
///               yn = rms(h, post_attention_layernorm)      -> up | gate
///     phase 3   h2 = h + W_down @ silu(gate) * up          -- the MLP residual
///                                                          -> IT[2], the layer output
///
/// All three phases read x on the same S2MM chain, and **nothing in this kernel
/// distinguishes them**: the layer's data dependencies order them, because o
/// cannot exist until the whole attention column has run and down cannot exist
/// until the MLP has. That is the same arrangement ../gated_attention_verify
/// uses across its three phases.
///
/// Phase 1 is replayed. The projection engine's qkv and z are plain mvm()s, so
/// each consumes x a slice at a time and needs the same xn once per row block;
/// this tile's outbound BD self-chains, so one norm is computed and sent
/// QKV_REPEATS times -- 16 for qkv plus 12 for the output gate. Phase 2 is sent
/// once: up|gate is a *resident* mvm that latches its x on the core, so it needs
/// the row once however many row blocks it owns.
///
/// x_buf carries fc_out across the whole attention latency. It is saved in
/// phase 1 rather than re-read at use time because by then the x chain has
/// moved on to carrying o. residual_add writes its sum back into x_buf, so the
/// same slot then carries h across the whole MLP for phase 3 to add down to --
/// one buffer, two residuals, and no second D-wide allocation on a tile that is
/// already within 32 bytes of full.
///
/// **The weight acquire is once per norm, not once per phase**: the self-chained
/// w BD refills the single w_buffer between the two acquires, so the host sends
/// 2 * MODEL_DIM -- input_layernorm then post_attention_layernorm -- and that
/// length has to track the live *norm* count exactly. Too few and this parks on
/// the second acquire; too many and the surplus sits in the channel unconsumed
/// and desynchronises the next invocation. Phase 3 is a bare add and takes no
/// weight, so adding it did not move this number.
///
/// Ping/pong parity: is_x_ping is static across invocations and the inbound BD
/// chain is a bare two-BD alternation, so the toggle and the chain step in
/// lockstep whatever the row count -- three rows per invocation (fc_out, o,
/// down) is as safe as two. What would break it is a phase that acquires x a
/// number of times other than once, or an early return.
///
/// No y_out parameter: it was the gated delta-net's a/b tap, inherited from the
/// Qwen3.5 template, and nothing in this kernel ever read or wrote it. Its
/// buffer still cost 10 KiB of a 64 KiB tile, which put the tile 32 bytes over
/// budget the moment anything else was added.
void rms_residual(
    bf16 *restrict y,
    bf16 *restrict x_ping,
    bf16 *restrict x_pong,
    bf16 *restrict w,
    bf16 *restrict x_buf
){
    // These ids are the ABI with iron/rms_residual.py -- keep them in step.
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

    // One x arrives per invocation, so the toggle alternates across
    // invocations rather than within one. It is static for that reason: the
    // DMA chain's ping/pong position has to survive the return.
    // Peano's CDO loader does not zero .bss; keep ping/pong selectors in .data.
    AIE_PERSIST_ZERO static bool is_x_ping = false;

    // --- input_layernorm: xn = rms(fc_out, w) -----------------------------
    _lock_acquire(x_cons_lock);
    is_x_ping = !is_x_ping;
    bf16* x = is_x_ping ? x_ping : x_pong;

    // Save fc_out for the residual that the o_proj phase will add to.
    _memcpy<MODEL_DIM * sizeof(bf16)>((uint8*)x_buf, (uint8*)x);

    _lock_acquire(w_cons_lock);
    *pkt_id_ptr = pkt_id_rms_to_proj;
    rms_norm(y + 16, x_buf, w);
    _lock_release(x_prod_lock);
    _lock_release(w_prod_lock);

    // Emit once, then wait for every send to land before the core overwrites y
    // in phase 2.
    //
    // QKV_REPEATS, not QKV_ONLY_REPEATS: qkv and z are two separate streaming
    // mvm()s, but they reduce over this same xn, so the row has to stay on the
    // wire for 16 + 12 row blocks. Release only 16 and z starves -- and the
    // stall lands in the projection engine, not here.
    _lock_release(y_cons_lock, QKV_REPEATS);
    _lock_acquire(y_prod_lock, QKV_REPEATS);

    // --- the attention residual, then post_attention_layernorm --------------
    // x now carries o_proj's row, arriving from MT[2] on the same channel the
    // layer input came in on. residual_add writes the sum back into x_buf as
    // well as out, so h stays resident for the MLP residual to add to.
    _lock_acquire(x_cons_lock);
    is_x_ping = !is_x_ping;
    x = is_x_ping ? x_ping : x_pong;
    residual_add(x_buf, x_buf, x);
    _lock_release(x_prod_lock);

    _lock_acquire(w_cons_lock);
    // **pkt_id_rms_to_proj, the same id phase 1 leaves under.** This norm is no
    // longer a DDR tap: it is up|gate's x, so it goes west to the projection
    // engine exactly as xn does, and the engine's x FIFO tells the two apart by
    // arrival order rather than by id -- the same arrangement z and o use on
    // MT[2]. Stamping pkt_id_rms_to_out here instead would leave the packet with
    // no matching route and park this core on the acquire below.
    //
    // The consequence is that the post-attention norm is no longer separately
    // observable; the SwiGLU tap covers it, reducing over it through two
    // projections and an activation.
    *pkt_id_ptr = pkt_id_rms_to_proj;
    rms_norm(y + 16, x_buf, w);
    _lock_release(w_prod_lock);

    // Once, not replayed: up|gate is a *resident* mvm (see proj_main.cc), so it
    // drains this row one slice at a time into x_full and re-reads it locally
    // for all UP_GATE_REPEATS row blocks. Replaying it here would need 68
    // credits and a lock saturates at 63.
    _lock_release(y_cons_lock, 1);
    _lock_acquire(y_prod_lock, 1);

    // --- the MLP residual, h2 = h + down -- the layer's output ---------------
    // x now carries down_proj's row, arriving on the same channel fc_out and o
    // came in on. x_buf still holds h: residual_add wrote its sum back there in
    // phase 2 precisely so this phase would not need its own copy.
    //
    // The sum goes to y, the outbound buffer, at + 16 -- the payload offset the
    // outbound BD reads from, past the packet header slot at + 14.
    // residual_add also leaves its sum in x_buf, which is dead here rather than
    // wrong: nothing downstream reads h again this invocation, and phase 1 of
    // the next one overwrites the slot with that token's fc_out before anything
    // reads it.
    _lock_acquire(x_cons_lock);
    is_x_ping = !is_x_ping;
    x = is_x_ping ? x_ping : x_pong;

    // **No weight acquire here.** This phase is a bare add, not a norm, so it
    // takes nothing from the w chain -- and taking one would park the core, as
    // the host sends exactly two MODEL_DIM-wide norms per invocation.
    *pkt_id_ptr = pkt_id_rms_to_out;
    residual_add(y + 16, x_buf, x);
    _lock_release(x_prod_lock);

    // Once: one D-wide row south to IT[2]. The DDR tap that down_proj used to
    // own moved here when down became this phase's x, so the shim channel is
    // the same one and only the producer changed.
    _lock_release(y_cons_lock, 1);
    _lock_acquire(y_prod_lock, 1);
}

}