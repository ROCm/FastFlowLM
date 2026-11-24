// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include <aie_api/aie.hpp>
#include "common.h"
#include <math.h>
#include "attn_kernel.h"    
#include "aie_kernel_utils.h"

constexpr int q_prod_lock = 0;
constexpr int q_cons_lock = 1;
constexpr int in_prod_lock = 2;
constexpr int in_cons_lock = 3;
constexpr int o_prod_lock = 4;
constexpr int o_cons_lock = 5;
constexpr int t_prod_lock = 6;
constexpr int t_cons_lock = 7;

extern "C" {
    void attn0(
        bf16 *__restrict o_ping,
        bf16 *__restrict o_pong,
        bf16 *__restrict q,
        bf16 *__restrict in_ping,
        bf16 *__restrict in_pong,
        bf16 * s,
        float * y,
        int * rtp_buffer
    ){
        __attribute__((section(".data"))) static bool is_in_ping = false;
        __attribute__((section(".data"))) static bool is_o_ping = false;

        alignas(aie::vector_decl_align) bf16 prev_m_local[lq];
        alignas(aie::vector_decl_align) bf16 new_m_local[lq];
        alignas(aie::vector_decl_align) float c_local[lq];
        alignas(aie::vector_decl_align) float l_local[lq];
        alignas(aie::vector_decl_align) bf16 l_bf16[lq];

        const aie::vector<bf16, lq> neg_inf = aie::broadcast<bf16, lq>(-0x1.FEp127f);
        const aie::vector<float, lq> one = aie::broadcast<float, lq>(1.0f);

        aie::store_v(prev_m_local, neg_inf);
        aie::store_v(new_m_local, neg_inf);
        aie::store_v(c_local, one);
        zero_vectorized<float, lq>(l_local);
        zero_vectorized<float, lq * dh>(y);

        _lock_acquire(q_cons_lock);
        // rope table (cos blocks 0-3, sin blocks 4-7, tile layout) arrives in s behind q
        _lock_acquire(t_cons_lock);
        rope_tile(q, s);
        bool is_self = rtp_buffer[3] == 1;

        int blocks = rtp_buffer[0];
        int text_chunk_id = rtp_buffer[1];
        int effective_number = rtp_buffer[2];
        // self layers: the expert's own K/V rows come first (online softmax is order independent);
        // their K is roped here with the table still in s
        if (is_self){
            _lock_acquire(in_cons_lock);
            is_in_ping = !is_in_ping;
            bf16 *k = is_in_ping ? in_ping : in_pong;
            rope_tile(k, s);
            _attn_qk<bf16, bf16, 8, 8, 8>(s, q, k, true, lk);
            _lock_release(in_prod_lock);
            get_max(new_m_local, s, prev_m_local);
            apply_softmax(s, new_m_local, true);
            calculate_c(c_local, prev_m_local, new_m_local);
            calculate_l(l_local, c_local, s);
            calculate_y(y, c_local);
            _lock_acquire(in_cons_lock);
            is_in_ping = !is_in_ping;
            bf16 *v = is_in_ping ? in_ping : in_pong;
            _attn_fv<bf16, float, 8, 8, 8>(y, s, v);
            _lock_release(in_prod_lock);
            for (int j = 0; j < lq; j++) *(prev_m_local + j) = *(new_m_local + j);
        }
        for (int block_idx = 0; block_idx < blocks; block_idx++){
            _lock_acquire(in_cons_lock);
            is_in_ping = !is_in_ping;
            bf16 *k = is_in_ping ? in_ping : in_pong;
            int n_valid = (block_idx == text_chunk_id) ? effective_number : (int)lk;
            _attn_qk<bf16, bf16, 8, 8, 8>(s, q, k, false, n_valid);
            _lock_release(in_prod_lock);
            get_max(new_m_local, s, prev_m_local);
            apply_softmax(s, new_m_local, false);
            calculate_c(c_local, prev_m_local, new_m_local);
            calculate_l(l_local, c_local, s);
            calculate_y(y, c_local);
            _lock_acquire(in_cons_lock);
            is_in_ping = !is_in_ping;
            bf16 *v = is_in_ping ? in_ping : in_pong;
            _attn_fv<bf16, float, 8, 8, 8>(y, s, v);
            _lock_release(in_prod_lock);
            for (int j = 0; j < lq; j++) *(prev_m_local + j) = *(new_m_local + j);
        }
        _lock_release(q_prod_lock);
        _lock_release(t_prod_lock);  // s (table + scores) is dead from here
        aie::vector<float, lq> l_vec = aie::load_v<lq>(l_local);
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
                _lock_release(o_cons_lock);
            }
        }
    }
}