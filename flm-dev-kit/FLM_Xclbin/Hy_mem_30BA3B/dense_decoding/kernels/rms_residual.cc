// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include "aie_array_layout.h"
#include "aie_kernel_utils.h"
#include "typedef.hpp"



///@brief  rms norm: y = norm(x) * w
///@param y: output
///@param x: input
///@param w: weight
///@note The body lives in aie_kernel_utils.h so this, rope.cc's per-head Q/K
///      norm and rope_hunyuan.cc's all share one 512-bit-wide implementation.
inline void rms_norm(
    bf16 *restrict y,
    const bf16 *restrict x,
    const bf16 *restrict w
){
    _rms_norm<MODEL_DIM>(y, x, w, RMS_EPS);
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
    constexpr int vector_size = _fit_lanes<MODEL_DIM / 2, AIE_BF16_LANES>();
    static_assert(MODEL_DIM % (2 * vector_size) == 0);
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


// Lock ids, shared with iron/rms_residual.py's ObjectFifo/lock allocation.
constexpr int w_prod_lock           = 0;
constexpr int w_cons_lock           = 1;
constexpr int y_prod_lock           = 2;
constexpr int y_cons_lock           = 3;
constexpr int x_prod_lock           = 4;
constexpr int x_cons_lock           = 5;
constexpr int rtp_available_lock    = 6;
constexpr int lm_head_out_prod_lock = 7;
constexpr int lm_head_out_cons_lock = 8;

// Ping/pong parity is carried across invocations, so it has to outlive the
// call: one core-loop iteration picks up exactly where the previous one left
// off, whether the phases are sequenced by RTP or by the core itself.
#if defined(__chess__)
static bool is_x_ping = false;
static bool is_lm_head_ping = false;
#else
// Peano: kept in a zero-initialized MLIR buffer instead of static data, see
// iron/rms_residual.py.
extern "C" int32_t gdl_rms_state[2];
#define is_x_ping       gdl_rms_state[0]
#define is_lm_head_ping gdl_rms_state[1]
#endif


///@brief One decoder layer: input norm, residual + post-attn norm, residual out.
static void rms_decoder_layer(
    bf16 *restrict y,
    bf16 *restrict x_ping,
    bf16 *restrict x_pong,
    bf16 *restrict w,
    bf16 *restrict x_buf
){
    uint32_t* pkt_id_ptr = reinterpret_cast<uint32_t*>(y + PKT_ID_OFFSET);
    bf16* x;

    // input later norm + residul
    _lock_acquire(x_cons_lock);
    _lock_acquire(w_cons_lock);
    *pkt_id_ptr = pkt_id_rms_to_proj;
    is_x_ping = !is_x_ping;
    x = is_x_ping ? x_ping : x_pong;
    _copy_bf16<MODEL_DIM>(x_buf, x);
    rms_norm(y + PKT_PAYLOAD_OFFSET, x_buf, w);
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
    rms_norm(y + PKT_PAYLOAD_OFFSET, x_buf, w);
    _lock_release(w_prod_lock);

    _lock_release(x_prod_lock);
    _lock_release(y_cons_lock, UP_GATE_REPEATS);

    _lock_acquire(x_cons_lock);
    _lock_acquire(y_prod_lock, UP_GATE_REPEATS);
    *pkt_id_ptr = pkt_id_rms_to_it;
    is_x_ping = !is_x_ping;
    x = is_x_ping ? x_ping : x_pong;
    residual_add(y + PKT_PAYLOAD_OFFSET, x_buf, x);
    _lock_release(x_prod_lock);
    _lock_release(y_cons_lock, 1);
    _lock_acquire(y_prod_lock, 1);
}


///@brief The LM head: final norm, then stream the logits out in vocab rounds.
static void rms_lm_head(
    bf16 *restrict y,
    bf16 *restrict x_ping,
    bf16 *restrict x_pong,
    bf16 *restrict w,
    bf16 *restrict y_out_0,
    bf16 *restrict y_out_1
){
    uint32_t* pkt_id_ptr = reinterpret_cast<uint32_t*>(y + PKT_ID_OFFSET);
    bf16* x;

    _lock_acquire(x_cons_lock);
    _lock_acquire(w_cons_lock);
    *pkt_id_ptr = pkt_id_rms_to_proj;
    is_x_ping = !is_x_ping;
    x = is_x_ping ? x_ping : x_pong;
    rms_norm(y + PKT_PAYLOAD_OFFSET, x, w);
    _lock_release(x_prod_lock);
    _lock_release(w_prod_lock);

    constexpr int vocab_size_per_round = VOCAB_SIZE_PADDED / MODEL_DIM;
    constexpr int y_repeats_per_round = MODEL_DIM / M_PER_ROUND;
    for (int i = 0; i < vocab_size_per_round; i++) {
        _lock_release(y_cons_lock, y_repeats_per_round);
        _lock_acquire(lm_head_out_prod_lock, 1);
        _lock_acquire(x_cons_lock);

        is_lm_head_ping = !is_lm_head_ping;
        bf16* lm_head_dst = is_lm_head_ping ? y_out_0 : y_out_1;
        is_x_ping = !is_x_ping;
        x = is_x_ping ? x_ping : x_pong;
        _copy_bf16<MODEL_DIM>(lm_head_dst, x);

        _lock_release(x_prod_lock);
        _lock_release(lm_head_out_cons_lock, 1);
        _lock_acquire(y_prod_lock, y_repeats_per_round);
    }
}


extern "C" {

///@brief One unit of work for the RMS/residual tile.
///
/// With SELF_SEQUENCED_LM_HEAD (models/<model>.h) one call is one whole token:
/// NUM_HIDDEN_LAYERS decoder layers followed by the LM head, so the core knows
/// which phase it is in from its own position in the loop and IS_ATTN is unused
/// -- the host writes no RTP_RMS_IS_ATTN_BUFFER and never releases lock 6.
/// Without it, one call is one phase and the host names it through IS_ATTN,
/// gated on lock 6.  The parameter stays in the signature either way so the RTP
/// buffer keeps its address; see iron/rms_residual.py.
void rms_residual(
    bf16 *restrict y,
    bf16 *restrict x_ping,
    bf16 *restrict x_pong,
    bf16 *restrict w,
    bf16 *restrict y_out_0,
    bf16 *restrict y_out_1,
    bf16 *restrict x_buf,
    int* IS_ATTN
){
    // _lock_acquire(rtp_available_lock);
    _lock_acquire(x_cons_lock);
    _lock_release(x_cons_lock);
    if (IS_ATTN[0] == 1) {
        rms_decoder_layer(y, x_ping, x_pong, w, x_buf);
    }
    else {
        rms_lm_head(y, x_ping, x_pong, w, y_out_0, y_out_1);
    }
}

}
