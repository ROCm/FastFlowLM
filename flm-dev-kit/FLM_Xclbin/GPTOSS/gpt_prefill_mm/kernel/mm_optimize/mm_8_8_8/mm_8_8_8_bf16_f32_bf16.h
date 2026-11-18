// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __MM_888_H__
#define __MM_888_H__

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <aie_api/aie.hpp>
#include "../mm_optimized_common.h"
#include "../../passThroughCommon.hpp"
template <unsigned m, unsigned k, unsigned n, bool is_b_row_maj, bool is_b_s_t_row_maj,
 bool transpose_b_st_row_at_load, int CT_K_max>
void matmul_vectorized_8x8x8_bf16_f32_ping_pong_a(
    const bfloat16 *__restrict pA_ping, const bfloat16 *__restrict PA_pong,
    const bfloat16 *__restrict pB_ping, const bfloat16 *__restrict pB_pong,
    int *a_ping_pong_flag, int *b_ping_pong_flag,
    int a_prod_lock, int a_con_lock,
    float *__restrict pC,
    int b_prod_lock, int b_con_lock)
{
    constexpr int r = 8;
    constexpr int s = 8;
    constexpr int t = 8;

    static_assert(m % (2 * r) == 0);
    static_assert(k % s == 0);
    static_assert(n % (2 * t) == 0);

    static_assert(k%CT_K_max == 0);
    static_assert( k%(2*CT_K_max) == 0);// for ping pong
    constexpr int num_iter = k/CT_K_max;
    static_assert((k / num_iter) % s == 0);

    for(uint32_t l =0; l < num_iter; l++){
        const bfloat16 * __restrict pB_buf = nullptr;
        if (*(b_ping_pong_flag) == 0){
            pB_buf = pB_ping;
            *(b_ping_pong_flag) = 1;
        }else{
            pB_buf = pB_pong;
            *(b_ping_pong_flag) = 0;
        }

        acquire_greater_equal(b_con_lock, 1);

        if(is_b_row_maj == false &&  is_b_s_t_row_maj==false  && transpose_b_st_row_at_load==false){  // transpose all s_t block from col_major to row_major directly
            // change B's s_t sub blocks from col major to row major
            convert_vectorized_B_s_t_bloc_from_col_major_to_row_major<bfloat16, 
           ((k / num_iter) / s),  (n / t), r, s, t>( (bfloat16*) pB_buf);
        }


        matmul_vectorized_2x2_mmul_modify_buffer<bfloat16, float, (m / r), ((k / num_iter) / s), (n / t),
                                                r, s, t, is_b_row_maj, is_b_s_t_row_maj || (false == transpose_b_st_row_at_load)> // Treat as s_t is row major, expect when explicit runtime transpose is requested
                                                (pA_ping, PA_pong,pB_buf, a_ping_pong_flag, a_prod_lock, a_con_lock,
                                                                        pC);
        release(b_prod_lock, 1);
    }

}






template <unsigned m, unsigned k, unsigned n,  int CT_K_max>
void matmul_vectorized_8x8x8_bf16_f32_load_bias(
    const bfloat16 *__restrict pB_ping, const bfloat16 *__restrict pB_pong,
    bfloat16* bias_buf, // rxn buffer
     int *b_ping_pong_flag,
    int b_prod_lock, int b_con_lock)
{
    constexpr int r = 8;
    constexpr int s = 8;
    constexpr int t = 8;

    static_assert(m % (2 * r) == 0);
    static_assert(k % s == 0);
    static_assert(n % (2 * t) == 0);

    static_assert(k%CT_K_max == 0);
    static_assert( k%(2*CT_K_max) == 0);// for ping pong
    constexpr int num_iter = k/CT_K_max;
    static_assert((k / num_iter) % s == 0);

    for(uint32_t l =0; l < num_iter; l++){
        const bfloat16 * __restrict pB_buf = nullptr;
        if (*(b_ping_pong_flag) == 0){
            pB_buf = pB_ping;
            *(b_ping_pong_flag) = 1;
        }else{
            pB_buf = pB_pong;
            *(b_ping_pong_flag) = 0;
        }

        
        acquire_greater_equal(b_con_lock, 1);
        if(l == 0){
            // only do it the first time, since bias is size of sxn

            static_assert(  k*n >= r*n);
            bfloat16 * r_0 = bias_buf;
            bfloat16* r_1 = r_0 + t;
            bfloat16* r_2 = r_1 + t;
            bfloat16* r_3 = r_2 + t;
            bfloat16* r_4 = r_3 + t;
            bfloat16* r_5 = r_4 + t;
            bfloat16* r_6 = r_5 + t;
            bfloat16* r_7 = r_6 + t;
            // because r == 8
            int n_div_t = n/t;
            static_assert(n%t  == 0);
            for(int idx = 0; idx < n_div_t; idx++){
                aie::vector<bfloat16, t> bias_vec = aie::load_v<t>(pB_buf + idx*(s*t));

                aie::store_v(r_0, bias_vec);
                r_0 += r*t;
                aie::store_v(r_1, bias_vec);
                r_1 += r*t;
                aie::store_v(r_2, bias_vec);
                r_2 += r*t;
                aie::store_v(r_3, bias_vec);
                r_3 += r*t;
                aie::store_v(r_4, bias_vec);
                r_4 += r*t;
                aie::store_v(r_5, bias_vec);
                r_5 += r*t;
                aie::store_v(r_6, bias_vec);
                r_6 += r*t;
                aie::store_v(r_7, bias_vec);
                r_7 += r*t;

            }



        }

        release(b_prod_lock, 1);
    }

}





template <unsigned m, unsigned k, unsigned n,  int CT_K_max>
void matmul_vectorized_8x8x8_bf16_f32_load_bias_col_major(
    const bfloat16 *__restrict pB_ping, const bfloat16 *__restrict pB_pong,
    bfloat16* bias_buf, // rxn buffer
     int *b_ping_pong_flag,
    int b_prod_lock, int b_con_lock)
{
    constexpr int r = 8;
    constexpr int s = 8;
    constexpr int t = 8;

    static_assert(m % (2 * r) == 0);
    static_assert(k % s == 0);
    static_assert(n % (2 * t) == 0);

    static_assert(k%CT_K_max == 0);
    static_assert( k%(2*CT_K_max) == 0);// for ping pong
    constexpr int num_iter = k/CT_K_max;
    static_assert((k / num_iter) % s == 0);

    bfloat16 * r_0 = bias_buf;
    bfloat16* r_1 = r_0 + t;
    bfloat16* r_2 = r_1 + t;
    bfloat16* r_3 = r_2 + t;
    bfloat16* r_4 = r_3 + t;
    bfloat16* r_5 = r_4 + t;
    bfloat16* r_6 = r_5 + t;
    bfloat16* r_7 = r_6 + t;

    static_assert(n%CT_K_max == 0);
    int n_div_ct_k = n/CT_K_max;
    static_assert(k>=n);


    int CT_K_max_div_s = CT_K_max/s;

    static_assert(CT_K_max%s == 0);    

    for(uint32_t l =0; l < num_iter; l++){
        const bfloat16 * __restrict pB_buf = nullptr;
        if (*(b_ping_pong_flag) == 0){
            pB_buf = pB_ping;
            *(b_ping_pong_flag) = 1;
        }else{
            pB_buf = pB_pong;
            *(b_ping_pong_flag) = 0;
        }

        
        acquire_greater_equal(b_con_lock, 1);

        if(l < n_div_ct_k ){

            for(int ct_k_d_s_idx = 0;  ct_k_d_s_idx < CT_K_max_div_s; ct_k_d_s_idx++){

                //This implementation assume s==t

                static_assert(s == t);
                aie::vector<bfloat16, s> bias_vec =  aie::load_v<s>( pB_buf + s*t *ct_k_d_s_idx); 
                aie::store_v(r_0, bias_vec);
                r_0 += r*t;
                aie::store_v(r_1, bias_vec);
                r_1 += r*t;
                aie::store_v(r_2, bias_vec);
                r_2 += r*t;
                aie::store_v(r_3, bias_vec);
                r_3 += r*t;
                aie::store_v(r_4, bias_vec);
                r_4 += r*t;
                aie::store_v(r_5, bias_vec);
                r_5 += r*t;
                aie::store_v(r_6, bias_vec);
                r_6 += r*t;
                aie::store_v(r_7, bias_vec);
                r_7 += r*t;
            }

        }
      
        release(b_prod_lock, 1);
    }

}

// template <unsigned m, unsigned k, unsigned n,  int CT_K_max>
// void matmul_vectorized_8x8x8_bf16_f32_load_bias_col_major(
//     const bfloat16 *__restrict pB_ping, const bfloat16 *__restrict pB_pong,
//     bfloat16* bias_buf, // rxn buffer
//      int *b_ping_pong_flag,
//     int b_prod_lock, int b_con_lock)
// {
//     constexpr int r = 8;
//     constexpr int s = 8;
//     constexpr int t = 8;

//     static_assert(m % (2 * r) == 0);
//     static_assert(k % s == 0);
//     static_assert(n % (2 * t) == 0);

//     static_assert(k%CT_K_max == 0);
//     static_assert( k%(2*CT_K_max) == 0);// for ping pong
//     constexpr int num_iter = k/CT_K_max;
//     static_assert((k / num_iter) % s == 0);







//         // only do it the first time, since bias is size of sxn
//     bfloat16 * r_0 = bias_buf;
//     bfloat16* r_1 = r_0 + t;
//     bfloat16* r_2 = r_1 + t;
//     bfloat16* r_3 = r_2 + t;
//     bfloat16* r_4 = r_3 + t;
//     bfloat16* r_5 = r_4 + t;
//     bfloat16* r_6 = r_5 + t;
//     bfloat16* r_7 = r_6 + t;

//     static_assert(n%s == 0);
//     int n_div_s = n/s;

//     int CT_K_max_div_s = CT_K_max/s;

//     static_assert(CT_K_max%s == 0);


//     for(uint32_t l =0; l < num_iter; l++){
//         const bfloat16 * __restrict pB_buf = nullptr;
//         if (*(b_ping_pong_flag) == 0){
//             pB_buf = pB_ping;
//             *(b_ping_pong_flag) = 1;
//         }else{
//             pB_buf = pB_pong;
//             *(b_ping_pong_flag) = 0;
//         }

//         acquire_greater_equal(b_con_lock, 1);
//         for(int ct_k_d_s_idx = 0;  ct_k_d_s_idx < CT_K_max_div_s; ct_k_d_s_idx++){

//             //This implementation assume s==t

//             static_assert(s == t);
//             aie::vector<bfloat16, s> bias_vec = aie::load_v<s>( pB_buf + s*n *ct_k_d_s_idx); 
//             aie::store_v(r_0, bias_vec);
//             r_0 += r*t;
//             aie::store_v(r_1, bias_vec);
//             r_1 += r*t;
//             aie::store_v(r_2, bias_vec);
//             r_2 += r*t;
//             aie::store_v(r_3, bias_vec);
//             r_3 += r*t;
//             aie::store_v(r_4, bias_vec);
//             r_4 += r*t;
//             aie::store_v(r_5, bias_vec);
//             r_5 += r*t;
//             aie::store_v(r_6, bias_vec);
//             r_6 += r*t;
//             aie::store_v(r_7, bias_vec);
//             r_7 += r*t;
//         }
        

//         release(b_prod_lock, 1);
//     }

// }






template <unsigned m, unsigned k, int CT_K_max>
void matmul_vectorized_8x8x8_bf16_f32_consume_a(
    const bfloat16 *__restrict pA_ping, const bfloat16 *__restrict PA_pong,
    int *a_ping_pong_flag, int *b_ping_pong_flag,
    int a_prod_lock, int a_con_lock)
{
    constexpr int r = 8;
    constexpr int s = 8;
    constexpr int t = 8;

    static_assert(m % (2 * r) == 0);


    static_assert(k%CT_K_max == 0);
    static_assert( k%(2*CT_K_max) == 0);// for ping pong
    constexpr int num_iter = k/CT_K_max;
    static_assert((k / num_iter) % s == 0);

    for(uint32_t l =0; l < num_iter; l++){



        matmul_vectorized_2x2_mmul_consume_a_lock<bfloat16, (m / r)> // Treat as s_t is row major, expect when explicit runtime transpose is requested
                                                (pA_ping, PA_pong, a_ping_pong_flag, a_prod_lock, a_con_lock);

    }

}


#endif