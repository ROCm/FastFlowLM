// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#include <aie_api/aie.hpp>
#include "common.h"
#include <math.h>
#include "attn_kernel.h"

// Core rounding mode: its power-on default is floor (toward minus infinity), which biases every narrowing of an accumulator
// or fp32 value to bf16 / bfp16 downwards. Select round-to-nearest-even for the duration of the call, restore on return.
struct KernelRneScope {
    aie::rounding_mode saved;
    inline __attribute__((always_inline)) KernelRneScope() { saved = aie::swap_rounding(aie::rounding_mode::conv_even); }
    inline __attribute__((always_inline)) ~KernelRneScope() { aie::set_rounding(saved); }
};
#define KERNEL_RNE_SCOPE KernelRneScope _kernel_rne_scope;
    

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
        KERNEL_RNE_SCOPE
        static bool is_in_ping AIE_PERSIST_ZERO = false;
        static bool is_o_ping AIE_PERSIST_ZERO = false;

        // each core takes 32 query rows as two 16-row halves h (own s, m, y, running max and sums)
        alignas(aie::vector_decl_align) bf16 prev_m_local[halves][lq];
        alignas(aie::vector_decl_align) bf16 new_m_local[halves][lq];
        alignas(aie::vector_decl_align) float c[halves][lq];
        alignas(aie::vector_decl_align) float l[halves][lq];
        alignas(aie::vector_decl_align) bf16 l_bf16[halves][lq];

#if defined(__chess__)
        static const aie::vector<bf16, 16> neg_inf = aie::broadcast<bf16, 16>(-0x1.FEp127f);
        static const aie::vector<float, lq> one = aie::broadcast<float, lq>(1.0f);
#else
        // Peano: a function-local static const with a constructor needs a guard variable in .bss, which Peano's
        // CDO loader does not zero, so the constructor can be skipped (the old NaN). Build them on every call.
        const aie::vector<bf16, 16> neg_inf = aie::broadcast<bf16, 16>(-0x1.FEp127f);
        const aie::vector<float, lq> one = aie::broadcast<float, lq>(1.0f);
#endif

        int i = 0;
        int L_div_l;
        int pointer_block_q = 0;
        int pointer_block_inner_q[halves];
        int pointer_block_k = 0;
        int pointer_block_inner_k[halves];
        int block_location = 0;
        int pointer_block_inner_k_current = 0;
        do{
            for (int h = 0; h < halves; h++){
                aie::store_v(prev_m_local[h], neg_inf);
                aie::store_v(new_m_local[h], neg_inf);
                aie::store_v(c[h], one);
                zero_vectorized<float, lq>(l[h]);
            }
            zero_vectorized<float, halves * lq * dh>(y);

            _lock_acquire(q_cons_lock);
            pointer_block_q = L_begin_buffer[0] + i * rows_per_round;
            int window_size = window_size_buffer[0];
            pointer_block_k = pointer_block_q - window_size;
            pointer_block_k = pointer_block_k > 0 ? pointer_block_k : 0;
            for (int h = 0; h < halves; h++){
                pointer_block_inner_q[h] = pointer_block_q + row * 2 * rows_per_core + col * rows_per_core + h * lq;
                pointer_block_inner_k[h] = pointer_block_inner_q[h] - window_size;
                pointer_block_inner_k[h] = pointer_block_inner_k[h] > -16 ? pointer_block_inner_k[h] : -16;
            }
            int kv_blocks = ((pointer_block_q + rows_per_round + 128 - 1) - pointer_block_k) / 128;
            for (int block_idx = 0; block_idx < kv_blocks; block_idx++){
                for (int h = 0; h < halves; h++){
                    for (int j = 0; j < lq; j++){
                        aie::vector<bf16, lk> m_vec = aie::broadcast<bf16, lk>(prev_m_local[h][j]);
                        aie::store_v(m + h * lq * lk + j * lk, m_vec);
                    }
                }
                for (int j = 0; j < 8; j++){
                    block_location = block_idx * 8 + j;
                    pointer_block_inner_k_current = pointer_block_k + block_location * 16;
                    _lock_acquire(in_cons_lock);
                    is_in_ping = !is_in_ping;
                    bf16 *k = is_in_ping ? in_ping : in_pong;
                    for (int h = 0; h < halves; h++){
                        _attn_qk_bfp16(s + h * lq * lk_block + j * lq * lk, (const bfp16ebs8*)((const uint8_t*)q + h * q_bfp_bytes), (const bfp16ebs8*)k);
                    }
                    _lock_release(in_prod_lock);
                    for (int h = 0; h < halves; h++){
                        apply_mask_and_get_max(s + h * lq * lk_block + j * lq * lk, m + h * lq * lk, pointer_block_inner_k[h], pointer_block_inner_q[h], pointer_block_inner_k_current);
                    }
                }
                for (int h = 0; h < halves; h++){
                    bf16 *s_h = s + h * lq * lk_block;
                    for (int j = 0; j < lq; j++){
                        aie::vector<bf16, lk> m_vec = aie::load_v<lk>(m + h * lq * lk + j * lk);
                        new_m_local[h][j] = (bf16)aie::reduce_max(m_vec);
                    }
                    {
                        aie::vector<bf16, lq> prev_m = aie::load_v<lq>(prev_m_local[h]);
                        aie::vector<bf16, lq> new_m = aie::load_v<lq>(new_m_local[h]);
                        aie::vector<bf16, lq> limit = aie::add(prev_m, lazy_max_step);
                        aie::store_v(new_m_local[h], aie::select(prev_m, new_m, aie::gt(new_m, limit)));
                    }
                    apply_softmax(s_h, new_m_local[h]);
                    // y only needs rescaling when a row's running max moved (never in the first block)
                    const bool rescale = calculate_c(c[h], prev_m_local[h], new_m_local[h]) && block_idx > 0;
                    reorder_s(s_h);
                    calculate_l(l[h], c[h], s_h);
                    if (rescale){
                        calculate_y(y + h * lq * dh, c[h]);
                    }
                }
                convert_p_bfp16(s);
                for (int j = 0; j < 8; j++){
                    _lock_acquire(in_cons_lock);
                    is_in_ping = !is_in_ping;
                    bf16 *v = is_in_ping ? in_ping : in_pong;
                    _attn_pv_chunk(y, (const bfp16ebs8*)s, (const bfp16ebs8*)v, j);
                    _lock_release(in_prod_lock);
                }
                for (int h = 0; h < halves; h++){
                    aie::store_v(prev_m_local[h], aie::load_v<lq>(new_m_local[h]));
                }
            }
            _lock_release(q_prod_lock);
            for (int h = 0; h < halves; h++){
                aie::vector<float, lq> l_vec = aie::inv(aie::load_v<lq>(l[h]));
                aie::accum<accfloat, lq> l_acc;
                l_acc.from_vector(l_vec);
                aie::store_v(l_bf16[h], l_acc.template to_vector<bf16>());
            }
            for (int h = 0; h < halves; h++){
                for (int o_row = 0; o_row < lq / 8; o_row++){
                    for (int o_col = 0; o_col < dh / 8; o_col++){
                        _lock_acquire(o_prod_lock);
                        is_o_ping = !is_o_ping;
                        bf16 *o = is_o_ping ? o_ping : o_pong;
                        scale_div_aie(o, l_bf16[h] + o_row * 8, y + h * lq * dh + o_row * 8 * dh + o_col * 64);
                        _lock_release(o_cons_lock);
                    }
                }
            }
            i++;
            L_div_l = L_chunks_buffer[0];
        } while (i < L_div_l);
    }
}