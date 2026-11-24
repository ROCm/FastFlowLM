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
        int * L_begin_buffer,
        int * L_chunks_buffer,
        int * window_size_buffer,
        const int row, 
        const int col
    ){
        AIE_PERSIST_ZERO static bool is_in_ping = false;
        AIE_PERSIST_ZERO static bool is_o_ping = false;

        alignas(aie::vector_decl_align) bf16 prev_m_local[lq];
        alignas(aie::vector_decl_align) bf16 new_m_local[lq];
        alignas(aie::vector_decl_align) float c[lq];
        alignas(aie::vector_decl_align) float l[lq];
        alignas(aie::vector_decl_align) float l_inv[lq];

        AIE_TRANSIENT const aie::vector<bf16, 16> neg_inf = aie::broadcast<bf16, 16>(-0x1.FEp127f);
        AIE_TRANSIENT const aie::vector<float, lq> one = aie::broadcast<float, lq>(1.0f);

        int i = 0;
        int L_div_l;
        int pointer_block_q = 0;
        int pointer_block_inner_q = 0;
        int pointer_block_k = 0;
        int pointer_block_inner_k = 0;
        int block_location = 0;
        int pointer_block_inner_k_current = 0;
        do{
            aie::store_v(prev_m_local, neg_inf);
            aie::store_v(new_m_local, neg_inf);
            aie::store_v(c, one);
            zero_vectorized<float, lq>(l);
            zero_vectorized<float, lq * dh>(y);

            _lock_acquire(q_cons_lock);
            // apply softmax
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
            
            int kv_blocks = ((pointer_block_q + 128 + 128 - 1) - pointer_block_k) / 128;
            for (int block_idx = 0; block_idx < kv_blocks; block_idx++){
                for (int j = 0; j < lq; j++){
                    aie::vector<bf16, lk> m_vec = aie::broadcast<bf16, lk>(*(prev_m_local + j));
                    aie::store_v(m + j * lk, m_vec);
                }
                for (int j = 0; j < 8; j++){
                    block_location = block_idx * 8 + j;
                    pointer_block_inner_k_current = pointer_block_k + block_location * 16;
                    _lock_acquire(in_cons_lock);
                    is_in_ping = !is_in_ping;
                    bf16 *k = is_in_ping ? in_ping : in_pong;
                    _attn_qk<bf16, bf16, lq/8, dh/8, lk/8, 8, 8, 8>(s + j * lq * lk, q + col * lq * dh, k);
                    _lock_release(in_prod_lock);
                    apply_mask_and_get_max(s + j * lq * lk, m, pointer_block_inner_k, pointer_block_inner_q, pointer_block_inner_k_current);
                }
                for (int j = 0; j < lq; j++){
                    aie::vector<bf16, lk> m_vec = aie::load_v<lk>(m + j * lk);
                    bf16 vm = aie::reduce_max(m_vec);
                    *(new_m_local + j) = (bf16)vm;
                }
                apply_softmax(s, new_m_local);
                bool c_is_identity = calculate_c(c, prev_m_local, new_m_local);
                reorder_s(s);
                calculate_l(l, c, s);
                // Rescaling y is a no-op in two cases, both exact: on the first
                // KV block y is still the zeroed buffer, and once every row's
                // correction is 1 the running max has stopped moving. Skipping
                // avoids a full 16 KB read-modify-write pass over y.
                if (block_idx != 0 && !c_is_identity){
                    calculate_y(y, c);
                }
                for (int j = 0; j < 8; j++){
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
            // True fp32 divide rather than aie::inv: the approximate reciprocal
            // costs more than a bf16 ULP once l is a generic value.
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
            L_div_l = L_chunks_buffer[0];
        } while (i < L_div_l);
    }
}