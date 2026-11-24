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
        AIE_PERSIST_ZERO static bool is_in_ping = false;
        AIE_PERSIST_ZERO static bool is_q_ping = false;
        AIE_PERSIST_ZERO static bool is_o_ping = false;

        alignas(aie::vector_decl_align) static bf16 prev_m_local[lq];
        alignas(aie::vector_decl_align) static bf16 new_m_local[lq];
        alignas(aie::vector_decl_align) static float c_local[lq] = {};
        alignas(aie::vector_decl_align) static float l_local[lq] = {};
        AIE_TRANSIENT const aie::vector<bf16, lq> neg_inf = aie::broadcast<bf16, lq>(-0x1.FEp127f);
        AIE_TRANSIENT const aie::vector<float, lq> one = aie::broadcast<float, lq>(1.0f);

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
                    scale_copy_aie<bf16, bf16, 64>(q_in, q + q_block * 64);
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
                    scale_copy_aie<bf16, bf16, 64>(q_in, q + q_block * 64);
                    _lock_release(q_prod_lock);
                }
            }

            for (int j = 0; j < chunks; j++){
                for (int q_index = 0; q_index < lq; q_index += 16){
                    // Seed the running maxima from the previous chunk; _attn_qk
                    // folds each block's row maxima straight into new_m_local,
                    // so the separate get_max pass over s is gone.
                    aie::vector<bf16, 16> m_local_vec = aie::load_v<16>(prev_m_local + q_index);
                    aie::store_v(new_m_local + q_index, m_local_vec);
                    for (int k_idx = 0; k_idx < lk_mt / lk; k_idx++){
                        _lock_acquire(in_cons_lock);
                        is_in_ping = !is_in_ping;
                        bf16 *k_in = is_in_ping ? in_ping : in_pong;
                        _attn_qk<bf16, bf16, 16 / 8, dh/8, lk/8, 8, 8, 8>(s + k_idx * 16 * lk, q + q_index * dh, k_in, new_m_local + q_index);
                        _lock_release(in_prod_lock);
                    }
                    aie::vector<float, 16> rowsum = softmax(s, new_m_local + q_index);
                    calculate_c(c_local + q_index, prev_m_local + q_index, new_m_local + q_index);
                    calculate_l_and_y(y + q_index * dh, l_local + q_index, c_local + q_index, rowsum, j == 0);
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