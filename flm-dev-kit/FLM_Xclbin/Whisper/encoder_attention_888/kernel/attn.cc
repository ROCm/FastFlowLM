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
        volatile int * L_buffer,
        int col
    ){
        // The core comes out of reset rounding toward -inf. That biases every
        // conversion to bf16 or bfp16 (mmul operands, accumulator stores) low,
        // and on mostly-positive inputs (GELU, softmax) the bias adds up.
        aie::set_rounding(aie::rounding_mode::conv_even);

        AIE_PERSIST_ZERO static AIE_PING_FLAG is_in_ping = false;
        AIE_PERSIST_ZERO static AIE_PING_FLAG is_q_ping = false;
        AIE_PERSIST_ZERO static AIE_PING_FLAG is_o_ping = false;

        alignas(aie::vector_decl_align) static bf16 prev_m_local[lq];
        alignas(aie::vector_decl_align) static bf16 m_local[lq];
        alignas(aie::vector_decl_align) static bf16 new_m_local[lq];
        alignas(aie::vector_decl_align) static float c_local[lq] = {};
        alignas(aie::vector_decl_align) static float l_local[lq] = {};
        alignas(aie::vector_decl_align) static bf16 row_max[16 * 8];    // running per-row max of one chunk (block layout)
        static const aie::vector<bf16, lq> neg_inf = aie::broadcast<bf16, lq>(-0x1.FEp127f);
        static const aie::vector<float, lq> one = aie::broadcast<float, lq>(1.0f);

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
                    scaled_copy_64(q_in, q + q_block * 64, (bf16)scale);   // q / sqrt(dh), see attn_qk_blk
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
                    scaled_copy_64(q_in, q + q_block * 64, (bf16)scale);   // q / sqrt(dh), see attn_qk_blk
                    _lock_release(q_prod_lock);
                }
            }

            // Key chunks of 512 rows and the number of real key rows are runtime
            // parameters, written by the host next to L (RTP slots 1 and 2), so a
            // shorter audio window can use fewer keys. 0 keeps the 30 s defaults.
            int n_chunks = L_buffer[1];
            if (n_chunks <= 0) n_chunks = chunks;
            int l_valid = L_buffer[2];
            if (l_valid <= 0) l_valid = L_valid;

            for (int j = 0; j < n_chunks; j++){
                for (int q_index = 0; q_index < lq; q_index += 16){
                    aie::vector<bf16, 16> m_local_vec = aie::load_v<16>(prev_m_local + q_index);
                    aie::store_v(m_local + q_index, m_local_vec);
                    aie::store_v(row_max, aie::broadcast<bf16, 64>(-0x1.FEp127f));
                    aie::store_v(row_max + 64, aie::broadcast<bf16, 64>(-0x1.FEp127f));
                    for (int k_idx = 0; k_idx < lk_mt / lk; k_idx++){
                        _lock_acquire(in_cons_lock);
                        is_in_ping = !is_in_ping;
                        bf16 *k_in = is_in_ping ? in_ping : in_pong;
                        attn_qk_blk(s + k_idx * 16 * lk, q + q_index * dh, k_in);
                        _lock_release(in_prod_lock);
                        // Padding keys get the lowest score, so exp() gives
                        // them no weight. The tile is in block layout (blk_index).
                        const int key_base = j * chunk_size + k_idx * lk;
                        if (key_base + lk > l_valid){
                            bf16* s_blk = s + k_idx * 16 * lk;
                            const int first_pad = l_valid > key_base ? l_valid - key_base : 0;
                            for (int row = 0; row < 16; row++){
                                for (int col = first_pad; col < lk; col++){
                                    s_blk[blk_index(row, col)] = (bf16)-0x1.FEp127f;
                                }
                            }
                        }
                        accumulate_row_max_blk(s + k_idx * 16 * lk, row_max);
                    }
                    fold_row_max_blk(row_max, m_local + q_index);
                    m_local_vec = aie::load_v<16>(m_local + q_index);
                    aie::store_v(new_m_local + q_index, m_local_vec);
                    softmax_blk(s, new_m_local + q_index, lk_mt / lk);
                    calculate_c(c_local + q_index, prev_m_local + q_index, new_m_local + q_index);
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