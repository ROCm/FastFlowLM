// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#include <aie_api/aie.hpp>
#include "common.h"
#include <math.h>
#include "attn_kernel.h"    

constexpr int q_prod_lock = 0;
constexpr int q_cons_lock = 1;
constexpr int in_prod_lock = 2;
constexpr int in_cons_lock = 3;
constexpr int o_prod_lock = 4;
constexpr int o_cons_lock = 5;
constexpr int w_prod_lock = 6;
constexpr int w_cons_lock = 7;
constexpr int in_fwd_lock = 8;    // the owner has the tile: the partner may read it
constexpr int in_done_lock = 9;   // the owner is done with the tile: the partner frees it

// The core rows of a pair share K / V: the lower row receives the K tiles, the upper row the V tiles, and
// each reads the other's from the neighbouring tile memory. The owner takes a tile from its DMA and
// forwards it; the partner frees it once both are done.
inline void kv_get(bool own, bool owner_up) {
    if (own) { _lock_acquire(in_cons_lock); _lock_release(in_fwd_lock); }
    else if (owner_up) _up_lock_acquire(in_fwd_lock);
    else _down_lock_acquire(in_fwd_lock);
}
inline void kv_put(bool own, bool owner_up) {
    if (own) _lock_release(in_done_lock);
    else if (owner_up) { _up_lock_acquire(in_done_lock); _up_lock_release(in_prod_lock); }
    else { _down_lock_acquire(in_done_lock); _down_lock_release(in_prod_lock); }
}

// Attention with a per-query sink logit (softmax starts at m = sink, l = 1) and a relative-position
// bias: score(q, k) += R_q . W[q - k] for q - k < 1024, W per layer [1024][nr], R per query and head.
// Per round the core gets its query pack, then nwin windows of W (distance blocks nwin-1 .. 0).

extern "C" {
    void attn(
        bf16 *__restrict o_ping,
        bf16 *__restrict o_pong,
        bf16 *__restrict q,
        bf16 *__restrict k_ping,
        bf16 *__restrict k_pong,
        bf16 *__restrict v_ping,
        bf16 *__restrict v_pong,
        bf16 * s,
        bf16 * m,
        float * y,
        bf16 *__restrict w_ping,
        bf16 *__restrict w_pong,
        bf16 *__restrict p,
        int * L_begin_buffer,
        int * L_end_buffer,
        int * window_size_buffer,
        const int row, 
        const int col
    ){
        // fp32 -> bf16 / bfp16 conversions otherwise truncate toward -inf, which biases the output low.
        aie::set_rounding(aie::rounding_mode::conv_even);
        // Force into .data (file-backed) rather than .bss: these persistent
        // ping-pong toggles must start false, but Peano's CDO loader does not
        // reliably zero a .bss tail (FileSiz<MemSiz PT_LOAD), which would leave
        // them garbage on load. Placing them in .data keeps the segment fully
        // file-backed so the initial values load correctly.
        AIE_PERSIST_ZERO static bool is_o_ping = false;

        alignas(aie::vector_decl_align) bf16 prev_m_local[lq];
        alignas(aie::vector_decl_align) bf16 new_m_local[lq];
        alignas(aie::vector_decl_align) float c[lq];
        alignas(aie::vector_decl_align) float l[lq];
        alignas(aie::vector_decl_align) bf16 l_bf16[lq];

        // NOT static: a function-local `static` here is lazily initialized via a
        // guard byte in .bss. Under Peano (llvm-aie) the kernel's .bss lands in a
        // PT_LOAD with FileSiz<MemSiz that the aie-rt CDO loader does not zero
        // reliably; a non-zero guard byte skips initialization and leaves these
        // vectors holding garbage, corrupting the softmax max/normalization
        // (intermittent, depends on residual tile memory). Constructing them each
        // call is a cheap broadcast with no static-init dependency.
        AIE_TRANSIENT const aie::vector<bf16, 16> neg_inf = aie::broadcast<bf16, 16>(-0x1.FEp127f);
        AIE_TRANSIENT const aie::vector<float, lq> one = aie::broadcast<float, lq>(1.0f);

        int i = 0;
        int offset;
        int L_div_l;
        int pointer_block_q = 0;
        int pointer_block_inner_q = 0;
        int pointer_block_k = 0;
        int pointer_block_inner_k = 0;
        int block_location = 0;
        int pointer_block_inner_k_current = 0;
        const bool k_own = (row & 1) == 0;   // lower row of the pair: K here, V in the tile above
        // q - k offsets of a row pair in the 32 lanes of mask_bias_max
        alignas(aie::vector_decl_align) int didx[32];
        for (int z = 0; z < 32; z++) didx[z] = z / 16 - z % 16;
        do{
            aie::store_v(new_m_local, neg_inf);
            aie::store_v(c, one);
            aie::store_v(l, one);
            zero_vectorized<float, lq * dh>(y);

            _lock_acquire(q_cons_lock);
            aie::store_v(prev_m_local, aie::load_v<lq>(q + q_sink_off + col * lq));
            // m: the row references broadcast to 16 lanes, then the lane maxima of each block
            for (int j = 0; j < lq; j++) aie::store_v(m + j * lk, aie::broadcast<bf16, lk>(prev_m_local[j]));
            const bf16* rq = q + q_r_off + col * lq * nr;
            // the other column's 16 queries are unused here: they make room for this core's as bfp16 tiles
            uint8* qb = (uint8*)(q + (1 - col) * lq * dh);
            convert_q(qb, q + col * lq * dh);
            int wnext = nwin - 1;
            // apply softmax
            offset = (L_begin_buffer[0] >> 7);
            // determine lq
            pointer_block_q = L_begin_buffer[0] + i * 128;
            pointer_block_inner_q = pointer_block_q + col * 16 + row * 32;
            // determine lk
            int window_size = window_size_buffer[0];
            pointer_block_k = pointer_block_q - window_size;
            pointer_block_k = pointer_block_k > 0 ? pointer_block_k : 0;
            // determine llk
            pointer_block_inner_k = pointer_block_inner_q - window_size;
            pointer_block_inner_k = pointer_block_inner_k > -16 ? pointer_block_inner_k : -16;
            // key k is visible to query q when 0 <= q - k < wd
            const int wd = pointer_block_inner_q - pointer_block_inner_k;

            const int nblk = (pointer_block_q - pointer_block_k) / 128 + 1;
            for (int block_idx = 0; block_idx < nblk; block_idx++){
                const int n = nblk - 1 - block_idx;
                const bf16* pb = nullptr;
                if (n < nwin) {
                    for (; wnext > n; wnext--) {
                        _lock_acquire(w_cons_lock);
                        _lock_release(w_prod_lock);
                    }
                    _lock_acquire(w_cons_lock);
#if !defined(EXP_NO_ALL) && !defined(EXP_NO_P)
                    compute_p(p, rq, ((nwin - 1 - n) & 1) ? w_pong : w_ping);
#endif
                    _lock_release(w_prod_lock);
                    wnext--;
                    pb = p + 15 + 16 * (1 - col);
                }
                for (int j = 0; j < 8; j++){
                    block_location = block_idx * 8 + j;
                    pointer_block_inner_k_current = pointer_block_k + block_location * 16;
                    kv_get(k_own, false);
                    bf16 *k = (j & 1) ? k_pong : k_ping;
#if !defined(EXP_NO_QK) && !defined(EXP_NO_ALL)
                    attn_qk_bfp16(s + j * lq * lk, qb, (const uint8*)k);
#endif
                    kv_put(k_own, false);
#if !defined(EXP_NO_ALL) && !defined(EXP_NO_SOFT)
                    const int dq = pointer_block_inner_q - pointer_block_inner_k_current;
                    if (pb) mask_bias_max<true>(s + j * lq * lk, m, dq, wd, didx, pb + 16 * j);
                    else mask_bias_max<false>(s + j * lq * lk, m, dq, wd, didx, nullptr);
#endif
                }
#if !defined(EXP_NO_ALL) && !defined(EXP_NO_SOFT)
                // A row moves its reference to the block maximum only when that is more than rescale_slack
                // above it: exp(s - reference) stays below e^rescale_slack, and most blocks need no rescale.
                row_max(new_m_local, m);
                const aie::vector<bf16, lq> prev_m = aie::load_v<lq>(prev_m_local);
                const aie::vector<bf16, lq> blk_m = aie::load_v<lq>(new_m_local);
                const aie::mask<lq> grow = aie::gt(blk_m, aie::add(prev_m, (bf16)rescale_slack));
                const bool moved = !grow.empty();
                aie::store_v(new_m_local, aie::select(prev_m, blk_m, grow));
                for (int j = 0; j < lq; j++) aie::store_v(m + j * lk, aie::broadcast<bf16, lk>(new_m_local[j]));
                apply_softmax(s, m);
                if (moved) calculate_c(c, prev_m_local, new_m_local);
                else aie::store_v(c, one);
                reorder_s(s);
                calculate_l(l, c, s);
                // y is still zero in the first block
                if (moved && block_idx > 0) calculate_y(y, c);
#endif
                for (int j = 0; j < 8; j++){
                    kv_get(!k_own, true);
                    bf16 *v = (j & 1) ? v_pong : v_ping;
#if !defined(EXP_NO_FV) && !defined(EXP_NO_ALL)
                    attn_fv_bfp16(y, s + j * lq * lk, (const uint8*)v);
#endif
                    kv_put(!k_own, true);
                }
                aie::vector<bf16, lq> prev_m_local_vec = aie::load_v<lq>(new_m_local);
                aie::store_v(prev_m_local, prev_m_local_vec);
            }
            for (; wnext >= 0; wnext--) {
                _lock_acquire(w_cons_lock);
                _lock_release(w_prod_lock);
            }
            _lock_release(q_prod_lock);
            aie::vector<float, lq> l_vec = aie::load_v<lq>(l);
            l_vec = aie::inv(l_vec);
            aie::accum<accfloat, lq> l_acc;
            l_acc.from_vector(l_vec);
            aie::store_v(l_bf16, l_acc.template to_vector<bf16>());
            for (int o_row = 0; o_row < lq / 8; o_row++){
                for (int o_col = 0; o_col < dh / 8; o_col++){
                    _lock_acquire(o_prod_lock);
                    is_o_ping = !is_o_ping;
                    bf16 *o = is_o_ping ? o_ping : o_pong;
                    scale_div_aie(o, l_bf16 + o_row * 8, y + o_row * 8 * dh + o_col * 64);
                    // passThrough_aie(y + o_row * 8 * dh + o_col * 64, o);
                    _lock_release(o_cons_lock);
                }
            }
            i++;
            L_div_l = L_end_buffer[0] >> 7;
            L_div_l -= offset;
        } while (i < L_div_l);
    }
}