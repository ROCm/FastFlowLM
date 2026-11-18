// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#include <aie_api/aie.hpp>
#include "common.h"
#include <math.h>
#include "attn_kernel.h"
#include "bfp_ops.h"

constexpr int q_prod_lock = 0;
constexpr int q_cons_lock = 1;
constexpr int in_prod_lock = 2;
constexpr int in_cons_lock = 3;
constexpr int o_prod_lock = 4;
constexpr int o_cons_lock = 5;
constexpr float lazy_tau = 8.0f;
// The current K or V block as bfp16 (pack_k / pack_v), shared by the four q tiles.
alignas(aie::vector_decl_align) static uint8_t kv_bfp[d_blocks * (lk / 8) * bfp_bytes];
extern "C" {
    void attn(
        bf16 *__restrict o_ping,
        bf16 *__restrict o_pong,
        bf16 *__restrict q_ping,
        bf16 *__restrict q_pong,
        bf16 *__restrict q,
        bf16 *__restrict in_ping,
        bf16 *__restrict in_pong,
        bf16 *__restrict s,
        float * y,
        int * L_buffer,
        int col
    ){
        // Every fp32->bf16 conversion in here otherwise truncates toward -inf.
        // Those floors do not cancel: they biased the whole output low by a
        // constant (measured -0.0079, ~43% of |O| on a mean-of-V test case).
        // Round-to-nearest-even makes the error zero-mean instead.
        aie::set_rounding(aie::rounding_mode::conv_even);

#if defined(__chess__)
        static bool is_in_ping = false;
        static bool is_q_ping = false;
        static bool is_o_ping = false;
#else
        // Each call flips every selector an even number of times, so all three are false on
        // entry; Peano keeps them local because its loader does not zero .bss.
        static_assert((lq / 8) * (dh / 8) % 2 == 0, "o selector must return to false at the end of a call");
        bool is_in_ping = false;
        bool is_q_ping = false;
        bool is_o_ping = false;
#endif

        alignas(aie::vector_decl_align) static bf16 prev_m_local[lq];
        alignas(aie::vector_decl_align) static bf16 m_local[lq];
        alignas(aie::vector_decl_align) static bf16 new_m_local[lq];
        alignas(aie::vector_decl_align) static float c_local[lq] = {};
        alignas(aie::vector_decl_align) static float l_local[lq] = {};
#if defined(__chess__)
        static const aie::vector<bf16, lq> neg_inf = aie::broadcast<bf16, lq>(-0x1.FEp127f);
        static const aie::vector<float, lq> one = aie::broadcast<float, lq>(1.0f);
#else
        // Not static under Peano: the guard for a function-local static initializer lives
        // in .bss, which is not zeroed at load, so the initializer could be skipped.
        const aie::vector<bf16, lq> neg_inf = aie::broadcast<bf16, lq>(-0x1.FEp127f);
        const aie::vector<float, lq> one = aie::broadcast<float, lq>(1.0f);
#endif

        int i = 0;
        int L_div_l = 0;
        do{
            zero_vectorized<float, lq>(l_local);
            zero_vectorized<float, lq * dh>(y);

            aie::store_v(prev_m_local, neg_inf);
            aie::store_v(new_m_local, neg_inf);
            aie::store_v(c_local, one);

            int q_blocks = lq * dh / 64;
            // q kept as bfp16 blocks (72 B each, same block order) for qk_bfp.
            aie::block_vector_output_buffer_stream<bfp16ebs8, 64> qs((bfp16ebs8 *)q);
            if (col == 0){
                for (int q_block = 0; q_block < q_blocks; q_block++){
                    _lock_acquire(q_cons_lock);
                    is_q_ping = !is_q_ping;
                    bf16 *q_in = is_q_ping ? q_ping : q_pong;
                    qs << to_bfp(aie::load_v<64>(q_in));
                    _lock_release(q_prod_lock);
                }
                for (int q_block = 0; q_block < q_blocks; q_block++){
                    _lock_acquire(q_cons_lock);
                    is_q_ping = !is_q_ping;
                    bf16 *q_in = is_q_ping ? q_ping : q_pong;
                    _lock_release(q_prod_lock);
                }
            }
            else{
                for (int q_block = 0; q_block < q_blocks; q_block++){
                    _lock_acquire(q_cons_lock);
                    is_q_ping = !is_q_ping;
                    bf16 *q_in = is_q_ping ? q_ping : q_pong;
                    _lock_release(q_prod_lock);
                }
                for (int q_block = 0; q_block < q_blocks; q_block++){
                    _lock_acquire(q_cons_lock);
                    is_q_ping = !is_q_ping;
                    bf16 *q_in = is_q_ping ? q_ping : q_pong;
                    qs << to_bfp(aie::load_v<64>(q_in));
                    _lock_release(q_prod_lock);
                }
            }

            // K/V once per round. Each 128-key chunk arrives exactly once and is
            // used by all four 16-row q tiles before the next one is taken, so
            // the memtile no longer replays every chunk once per q tile. Only
            // ceil(L_valid / lk_mt) chunks are sent; blocks past L_valid inside
            // the last chunk are still delivered (their locks are taken) but
            // not computed.
            const int n_chunks = (L_buffer[1] + lk_mt - 1) / lk_mt;
            constexpr int n_tiles = lq / 16;
            constexpr int bpc = lk_mt / lk;            // 16-key blocks per chunk
            constexpr int s_tile = bpc * 16 * lk;      // s elements per q tile
            for (int j = 0; j < n_chunks; j++){
                const int valid_in_chunk = L_buffer[1] - j * lk_mt;
                const int nb = valid_in_chunk >= lk_mt ? bpc : (valid_in_chunk + lk - 1) / lk;
                for (int t = 0; t < lq; t += 16){
                    aie::store_v(m_local + t, aie::load_v<16>(prev_m_local + t));
                }
                for (int k_idx = 0; k_idx < bpc; k_idx++){
                    _lock_acquire(in_cons_lock);
                    is_in_ping = !is_in_ping;
                    bf16 *k_in = is_in_ping ? in_ping : in_pong;
                    if (k_idx < nb){
                        const int k_base = j * lk_mt + k_idx * lk;
                        pack_k(kv_bfp, k_in);
                        for (int t = 0; t < n_tiles; t++){
                            bf16 *sb = s + t * s_tile + k_idx * 16 * lk;
                            qk_bfp(sb, (const uint8_t *)q + t * 2 * d_blocks * bfp_bytes, kv_bfp);
                            if (k_base + lk > L_buffer[1])
                                mask_tail(sb, k_base, L_buffer[1]);
                        }
                    }
                    _lock_release(in_prod_lock);
                }
                for (int t = 0; t < n_tiles; t++){
                    const int q_index = t * 16;
                    bf16 *st = s + t * s_tile;
#ifndef EXP_NO_SOFT
#ifndef EXP_NO_RMAX
                    rowmax_blocks(st, m_local + q_index, nb);
#endif
                    {
                        // Lazy rescale: a row's max only moves when the chunk max beats it by more than
                        // lazy_tau, so exp(s - m) <= e^lazy_tau and y / l need no rescale otherwise.
                        // l and y always share the same m, so the ratio is unchanged up to rounding.
                        const aie::vector<bf16, 16> mp = aie::load_v<16>(prev_m_local + q_index);
                        const aie::vector<bf16, 16> mc = aie::load_v<16>(m_local + q_index);
                        const aie::vector<bf16, 16> thr = aie::add(mp, aie::broadcast<bf16, 16>(lazy_tau));
                        aie::store_v(new_m_local + q_index, aie::select(mp, mc, aie::gt(mc, thr)));
                    }
#ifndef EXP_NO_EXP
                    softmax(st, new_m_local + q_index, nb);
#endif
#ifndef EXP_NO_C
                    calculate_c(c_local + q_index, prev_m_local + q_index, new_m_local + q_index);
#endif
#ifndef EXP_NO_REORDER
                    reorder_s(st, nb);
#endif
#ifndef EXP_NO_LY
                    calculate_l_and_y(y + q_index * dh, l_local + q_index, c_local + q_index, st, nb, j == 0);
#endif
#endif
                    aie::store_v(prev_m_local + q_index, aie::load_v<16>(new_m_local + q_index));
                }
                for (int v_idx = 0; v_idx < bpc; v_idx++){
                    _lock_acquire(in_cons_lock);
                    is_in_ping = !is_in_ping;
                    bf16 *v_in = is_in_ping ? in_ping : in_pong;
#ifndef EXP_NO_FV
                    if (v_idx < nb){
                        pack_v(kv_bfp, v_in);
                        for (int t = 0; t < n_tiles; t++)
                            fv_bfp(y + t * 16 * dh, s + t * s_tile + v_idx * 16 * lk, kv_bfp);
                    }
#endif
                    _lock_release(in_prod_lock);
                }
            } // chunk
            calculate_l(l_local);
            for (int ll = 0; ll < lq / 8; ll++){
                for (int j = 0; j < dh / 8; j++){
                    _lock_acquire(o_prod_lock);
                    is_o_ping = !is_o_ping;
                    bf16 *o = is_o_ping ? o_ping : o_pong;
                    scale_div_aie(o, l_local + ll * 8, y + ll * 8 * dh + j * 64);
                    _lock_release(o_cons_lock);
                }
            }
            i++;
            L_div_l = L_buffer[0] >> 9;
        }while(i < L_div_l);
    }
}