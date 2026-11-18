// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

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
        bf16 *__restrict q,
        bf16 *__restrict in_ping,
        bf16 *__restrict in_pong,
        bf16 * s,
        bf16 * m,
        float * y,
        int * KV_CHUNKS_buffer,
        int * Q_CHUNKS_buffer,
        int * L_buffer,
        const int col
    ){

        AIE_PERSIST_ZERO static bool is_in_ping = false;
        AIE_PERSIST_ZERO static bool is_o_ping = false;

        alignas(aie::vector_decl_align) bf16 prev_m_local[lq];
        alignas(aie::vector_decl_align) bf16 new_m_local[lq];
        alignas(aie::vector_decl_align) float c[lq];
        alignas(aie::vector_decl_align) float l[lq];
        alignas(aie::vector_decl_align) float l_inv[lq];

        AIE_TRANSIENT const aie::vector<bf16, 8> neg_inf = aie::broadcast<bf16, 8>(-0x1.FEp127f);
        AIE_TRANSIENT const aie::vector<float, lq> one = aie::broadcast<float, lq>(1.0f);

        int i = 0;
        int L_div_l;
        int block_location = 0;
        int pointer_block_inner_k_current = 0;
        do{
            aie::store_v(prev_m_local, neg_inf);
            aie::store_v(new_m_local, neg_inf);
            aie::store_v(c, one);
            zero_vectorized<float, lq>(l);
            zero_vectorized<float, lq * dh>(y);

            _lock_acquire(q_cons_lock);
            const int hi = L_buffer[0] - 1;
            const int kv_blocks = KV_CHUNKS_buffer[0];

            for (int block_idx = 0; block_idx < kv_blocks; block_idx++){
                for (int j = 0; j < lq; j++){
                    aie::vector<bf16, lk> m_vec = aie::broadcast<bf16, lk>(*(prev_m_local + j));
                    aie::store_v(m + j * lk, m_vec);
                }
                for (int j = 0; j < 16; j++){
                    block_location = block_idx * 16 + j;
                    pointer_block_inner_k_current = block_location * 8;   //!< kv starts at 0
                    _lock_acquire(in_cons_lock);
                    is_in_ping = !is_in_ping;
                    bf16 *k = is_in_ping ? in_ping : in_pong;
                    _attn_qk<bf16, bf16, lq/8, dh/8, lk/8, 8, 8, 8>(s + j * lq * lk, q + (col & 0x1) * lq * dh, k);
                    _lock_release(in_prod_lock);
                    if (pointer_block_inner_k_current + 7 <= hi){
                        get_max(s + j * lq * lk, m);
                    } else {
                        apply_mask_and_get_max(s + j * lq * lk, m, hi, pointer_block_inner_k_current);
                    }
                }
                for (int j = 0; j < lq; j++){
                    aie::vector<bf16, lk> m_vec = aie::load_v<lk>(m + j * lk);
                    bf16 vm = aie::reduce_max(m_vec);
                    *(new_m_local + j) = (bf16)vm;
                }
                apply_softmax(s, new_m_local);
                calculate_c(c, prev_m_local, new_m_local);
                calculate_l(l, c, s);
                calculate_y(y, c);
                for (int j = 0; j < 16; j++){
                    _lock_acquire(in_cons_lock);
                    is_in_ping = !is_in_ping;
                    bf16 *v = is_in_ping ? in_ping : in_pong;
                    _attn_fv<bf16, float, lq/8, lk/8, dh/8, 8, 8, 8>(y, s + j * lq * lk, v);
                    _lock_release(in_prod_lock);
                }
                aie::vector<bf16, lq> prev_m_local_vec = aie::load_v<lq>(new_m_local);
                aie::store_v(prev_m_local, prev_m_local_vec);
            }
            _lock_release(q_prod_lock);
            
            for (int j = 0; j < lq; j++){
                l_inv[j] = 1.0f / l[j];
            }
            for (int o_row = 0; o_row < lq / 8; o_row++){
                for (int o_col = 0; o_col < dh / 8; o_col++){
                    _lock_acquire(o_prod_lock);
                    is_o_ping = !is_o_ping;
                    bf16 *o = is_o_ping ? o_ping : o_pong;
                    scale_div_aie(o, l_inv + o_row * 8, y + o_row * 8 * dh + o_col * 64);
                    _lock_release(o_cons_lock);
                }
            }
            i++;
            L_div_l = Q_CHUNKS_buffer[0];
        } while (i < L_div_l);
    }
}