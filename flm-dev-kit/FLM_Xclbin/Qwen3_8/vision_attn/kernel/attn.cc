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
            if (col == 0){
                for (int q_block = 0; q_block < q_blocks; q_block++){
                    _lock_acquire(q_cons_lock);
                    is_q_ping = !is_q_ping;
                    bf16 *q_in = is_q_ping ? q_ping : q_pong;
                    passThrough_aie<bf16, bf16, 64>(q_in, q + q_block * 64);
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
                    passThrough_aie<bf16, bf16, 64>(q_in, q + q_block * 64);
                    _lock_release(q_prod_lock);
                }
            }

            for (int j = 0; j < (L_buffer[0] >> 9); j++){
                for (int q_index = 0; q_index < lq; q_index += 16){
                    aie::vector<bf16, 16> m_local_vec = aie::load_v<16>(prev_m_local + q_index);
                    aie::store_v(m_local + q_index, m_local_vec);
                    for (int k_idx = 0; k_idx < lk_mt / lk; k_idx++){
                        _lock_acquire(in_cons_lock);
                        is_in_ping = !is_in_ping;
                        bf16 *k_in = is_in_ping ? in_ping : in_pong;
                        _attn_qk<bf16, bf16, 16 / 8, d_blocks, lk/8, 8, 8, 8>(s + k_idx * 16 * lk, q + q_index * dh, k_in);
                        _lock_release(in_prod_lock);
                        // global row of this 16-key block within the padded K
                        const int k_base = j * lk_mt + k_idx * lk;
                        if (k_base + lk <= L_buffer[1]){
                            get_max(s + k_idx * 16 * lk, m_local + q_index);
                        } else {
                            mask_tail_and_get_max(s + k_idx * 16 * lk, m_local + q_index, k_base, L_buffer[1]);
                        }
                    }
                    m_local_vec = aie::load_v<16>(m_local + q_index);
                    aie::store_v(new_m_local + q_index, m_local_vec);
                    softmax(s, new_m_local + q_index);
                    calculate_c(c_local + q_index, prev_m_local + q_index, new_m_local + q_index);
                    reorder_s(s);
                    calculate_l_and_y(y + q_index * dh, l_local + q_index, c_local + q_index, s);
                    for (int v_idx = 0; v_idx < lk_mt / lk; v_idx++){
                        _lock_acquire(in_cons_lock);
                        is_in_ping = !is_in_ping;
                        bf16 *v_in = is_in_ping ? in_ping : in_pong;
                        _attn_fv<bf16, float, 8, 8, 8>(y + q_index * dh, s + v_idx * 16 * lk, v_in);
                        _lock_release(in_prod_lock);
                    }
                    aie::vector<bf16, 16> prev_m_local_vec = aie::load_v<16>(new_m_local + q_index);
                    aie::store_v(prev_m_local + q_index, prev_m_local_vec);
                }
            } // end of q_index loop
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