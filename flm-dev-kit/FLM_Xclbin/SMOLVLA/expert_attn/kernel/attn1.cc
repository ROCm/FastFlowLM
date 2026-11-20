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

extern "C" {
    void attn1(
        bf16 *__restrict o_ping,
        bf16 *__restrict o_pong,
        bf16 *__restrict q,
        bf16 *__restrict in_ping,
        bf16 *__restrict in_pong,
        bf16 * s,
        float * y,
        int * rtp_buffer,
        int row
    ){
#if defined(__chess__)
        __attribute__((section(".data"))) static bool is_in_ping = false;
        __attribute__((section(".data"))) static bool is_o_ping = false;
#else
        // Locals under Peano: these cores have no room for .data. Each selector
        // flips an even number of times per call, so every call starts at false.
        static_assert((lq / 8) * (dh / 8) % 2 == 0, "is_o_ping must flip an even number of times");
        bool is_in_ping = false;
        bool is_o_ping = false;
#endif

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

        int blocks = rtp_buffer[0];
        int text_chunk_id = rtp_buffer[1];
        int effective_number = rtp_buffer[2];
        for (int block_idx = 0; block_idx < blocks; block_idx++){
            _lock_acquire(in_cons_lock);
            is_in_ping = !is_in_ping;
            bf16 *k = is_in_ping ? in_ping : in_pong;
            int n_valid = (block_idx == text_chunk_id) ? effective_number : (int)lk;
            _attn_qk<bf16, bf16, 8, 8, 8>(s, q + row * lq * dh, k, false, n_valid);
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
        if (rtp_buffer[3] == 1){
            _lock_acquire(in_cons_lock);
            is_in_ping = !is_in_ping;
            bf16 *k = is_in_ping ? in_ping : in_pong;
            _attn_qk<bf16, bf16, 8, 8, 8>(s, q + row * lq * dh, k, true, lk);
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
        _lock_release(q_prod_lock);
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