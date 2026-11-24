// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __ATTENTION_OPTIMIZE_COMMON_H__
#define __ATTENTION_OPTIMIZE_COMMON_H__
#define NOCPP
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>
#include "../zero.h"
#include <aie_api/aie.hpp>
#include "../aie_kernel_utils.h" //TODO: a better method?
#include "../lock_util.hpp"


constexpr float NEG_INF_FP32 = -0x1.FEp127f;



// template <int m>
// aie::vector<bfloat16, m> exp2_bf16_noLUT(aie::vector<bfloat16, m> &x){



//   constexpr bfloat16 min_clamp = -87.0f;
//   constexpr bfloat16 max_clamp = 88.0f;
//   aie::vector<bfloat16, 16> min_value = aie::broadcast<bfloat16, 16>(min_clamp);
//   aie::vector<bfloat16, 16> max_value = aie::broadcast<bfloat16, 16>(max_clamp); // max of float

//   aie::vector<bfloat16, m> result_vec;
//   static_assert(m%16 == 0);
//   AIE_PREPARE_FOR_PIPELINING
//   for(int i = 0; i < m/16; i++){
//     aie::vector<bfloat16,16> cur_x = x.template extract<16>(i);

//     cur_x = aie::clamp(cur_x, min_value, max_value);
//     aie::accum<accfloat, 16> Vec_acc = aie::mul(cur_x, exp2_scale);
//     aie::vector<bfloat16, 16> res_temp = aie::exp2<bfloat16>(Vec_acc.template to_vector<float>());
//     result_vec.template insert<16>(i, res_temp);
//   }
//   return result_vec;

// }
template< typename T, unsigned r, unsigned t>
void broadcast_vector_to_mm(
    aie::vector<T, r * t>& out_mm,
    T* __restrict in_vector

){
    
    // This function broadcast a vector of size r to a small mm of rxt size, aka broadcast each element t times
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_RANGE(r, r)
    AIE_PEANO_UNROLL_FULL
    for(int r_idx = 0; r_idx < r; r_idx++){
        aie::vector<T, t> broadcast_vec = aie::broadcast<T, t>(in_vector[r_idx]);
        out_mm.template insert<t>(r_idx, broadcast_vec);
    }

}


constexpr bfloat16 exp2_scale =  (bfloat16)1.4426950408889634f;
template <int m>
inline aie::vector<bfloat16, m> exp2_bf16_noLUT(aie::vector<bfloat16, m> &x)
{

    constexpr bfloat16 min_clamp = -87.0f;
    constexpr bfloat16 max_clamp = 88.0f;
    aie::vector<bfloat16, m> min_value = aie::broadcast<bfloat16, m>(min_clamp);
    aie::vector<bfloat16, m> max_value = aie::broadcast<bfloat16, m>(max_clamp); // max of float

    aie::vector<bfloat16, m> cur_x = aie::clamp(x, min_value, max_value);
    aie::accum<accfloat, m> Vec_acc = aie::mul(cur_x, exp2_scale);
    aie::vector<bfloat16, m> res_temp = aie::exp2<bfloat16>(Vec_acc.template to_vector<float>());

    return res_temp;
}

template<typename T>
inline const T* retrive_ping_pong_buffer(
    const T* ping_buffer,
    const T* pong_buffer,
    int* ping_pong_flag)
{
    const T* out_ptr =  (*ping_pong_flag) == 0? ping_buffer : pong_buffer;
    *ping_pong_flag = 1 - *ping_pong_flag;
    return out_ptr;

}


template<typename T_in, unsigned row, unsigned col, unsigned r,  unsigned s, unsigned t>
void transpose_col_block_mm_t_s_to_s_tmatrix(
    T_in *__restrict matrix_in_out // in rowx col reorder of sxt blocks col block major
){
    //goal: transpose the small sxt blocks from sxt to txs or vice versa, but still keep big matrix in col block order

    static_assert( col % (2*t) ==0);
    static_assert( row % (s) ==0);


    using MMUL = aie::mmul<r, s, t, T_in, T_in, accauto>;

    constexpr unsigned rowB = row/s;
    constexpr unsigned colB = col/t;

    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_RANGE(colB / 2, colB / 2)    
    for(unsigned j = 0; j < colB; j+=2)
    {
        T_in *pB1;    
        T_in *pB2;
        pB1 = matrix_in_out + (j * rowB) * MMUL::size_B;
        pB2 = matrix_in_out + ((j + 1) * rowB) * MMUL::size_B;
        aie::vector<T_in, MMUL::size_B> B0;
        aie::vector<T_in, MMUL::size_B> B1;

        AIE_PREPARE_FOR_PIPELINING
        AIE_LOOP_RANGE(rowB, rowB)
        for (unsigned i = 0; i < rowB; i++)
        {
            B0 = aie::load_v<MMUL::size_B>(pB1);
            B1 = aie::load_v<MMUL::size_B>(pB2);
            aie::store_v(pB1, aie::transpose(B0, t, s));
            aie::store_v(pB2, aie::transpose(B1, t, s));    
            pB1 += MMUL::size_B;
            pB2 += MMUL::size_B;
        }


    }

}



template<typename T_in, typename T_out, unsigned LQ, unsigned DH, unsigned LK_V, 
    unsigned r, unsigned s, unsigned t,
    unsigned lk_internal_size
    >
void matmul_q_k_transpose_mmul_modify_buffer(
    T_in* __restrict Q_buffer, // size of LQ x DH reorder in r x s blocks row block major
    T_in* __restrict K_buffer_ping, // size of (lk_internal_size) x DH reorder in sxt blocks row block major, note: Need transpose when load
    T_in* __restrict K_buffer_pong, // size of (lk_internal_size) x DH reorder in sxt blocks row block major, note: Need transpose when load
    
    T_out* __restrict S_buffer, // size of LQ x LK_V
    int  *k_ping_pong_flag, // ping if == 0, else pong
    int k_prod_lock, int k_con_lock
){
    // The result is a matrix of LQ x LK_V in sxt block row block major


    static_assert( LQ%(2*r) ==0 );
    static_assert(DH%s ==0);

    static_assert(LK_V% (2*t) ==0);

    static_assert(LK_V % lk_internal_size ==0);

    constexpr int lk_internal_div_t = lk_internal_size / (t);

    static_assert(lk_internal_div_t >0);
    static_assert(lk_internal_div_t %2 ==0);  // atleast 2t

    using MMUL = aie::mmul<r, s, t, T_in, T_in, accauto>;

    constexpr unsigned rowA = LQ/r;
    constexpr unsigned colA = DH/s;
    constexpr unsigned colB = LK_V/t;
    
    T_out* __restrict pC = S_buffer;
    
    T_in* k_buffer_cur = nullptr;
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_RANGE(colB / 2, colB / 2)
    for(unsigned j = 0; j < colB; j+=2){

        if(j % (lk_internal_div_t) == 0){
            _lock_acquire(k_con_lock, 1);
            k_buffer_cur = (T_in*)retrive_ping_pong_buffer(
                K_buffer_ping,
                K_buffer_pong,
                k_ping_pong_flag
            );

        }else{
            k_buffer_cur += 2* colA * MMUL::size_B;
        }



        // The k_buffer is in sxt block of (2*t)xDH in row block major, each block is store in txs order
        // Thus, do a block transpose on the small txs-> sxt block
        //BUT still keep the big matrix in (2xt) x DH in row block major, which is same format if view as col block major of DH x (2xt)

        transpose_col_block_mm_t_s_to_s_tmatrix<T_in, DH, (2*t),  r, s, t>(k_buffer_cur);
        
        AIE_PREPARE_FOR_PIPELINING
        AIE_LOOP_RANGE(rowA / 2, rowA / 2)
        for(unsigned z = 0; z < rowA; z+=2){


            T_out *__restrict pC1 = pC + ((z * colB)+j) * MMUL::size_C;
            T_out *__restrict pC2 = pC1 + colB * MMUL::size_C;

            const T_in *__restrict pA1 = Q_buffer + (z * colA) * MMUL::size_A;
            const T_in *__restrict pA2 =pA1 + colA * MMUL::size_A;

            const T_in *__restrict pB1 = k_buffer_cur;
            const T_in *__restrict pB2 = k_buffer_cur + colA * MMUL::size_B;




            aie::vector<T_in, MMUL::size_A> A0;
            aie::vector<T_in, MMUL::size_A> A1;
            aie::vector<T_in, MMUL::size_B> B0;
            aie::vector<T_in, MMUL::size_B> B1;

            //NOTE: zero out the s_buffer
            aie::vector<T_out, MMUL::size_C> zero_vec = aie::zeros<T_out, MMUL::size_C>();

            MMUL C00(zero_vec);
            MMUL C01(zero_vec);
            MMUL C10(zero_vec);
            MMUL C11(zero_vec);

            AIE_PREPARE_FOR_PIPELINING
            AIE_LOOP_RANGE(colA, colA)
            for(unsigned i = 0; i < colA; i++)
            {
                A0 = aie::load_v<MMUL::size_A>(pA1);
                pA1 += MMUL::size_A;
                A1 = aie::load_v<MMUL::size_A>(pA2);
                pA2 += MMUL::size_A;

                //NOTE: block of b is already transposed to sxt order
                B0 = aie::load_v<MMUL::size_B>(pB1);
                pB1 += MMUL::size_B;
                B1 = aie::load_v<MMUL::size_B>(pB2);
                pB2 += MMUL::size_B;

                C00.mac(A0, B0);
                C01.mac(A0, B1);
                C10.mac(A1, B0);
                C11.mac(A1, B1);

            }

            aie::store_v(pC1, C00.template to_vector<T_out>());
            pC1 += MMUL::size_C;
            aie::store_v(pC1, C01.template to_vector<T_out>());
            pC1 += MMUL::size_C;
            aie::store_v(pC2, C10.template to_vector<T_out>());
            pC2 += MMUL::size_C;
            aie::store_v(pC2, C11.template to_vector<T_out>());
            pC2 += MMUL::size_C;


        }
        if((j + 2) % (lk_internal_div_t) == 0){
            _lock_release(k_prod_lock, 1);
        }else{
            //do nothing
        }

    }

}





template<typename T_in, typename T_out, unsigned LQ, unsigned LK_V, unsigned DH, 
    unsigned r, unsigned s, unsigned t, int K_Unroll>
void matmul_pij_vmmul_generic(
    T_in* __restrict Pij_buffer, // size of LQ x LK_V reorder in r x s blocks row block major
    T_in* __restrict V_buffer_ping, // size of (K_Unroll xs)x DH reorder in sxt blocks row block major
    T_in* __restrict V_buffer_pong, // size of (K_Unroll xs)x DH reorder in sxt blocks row block major
    
    T_out* __restrict O_buffer, // size of LQ x DH, in block of rxt blocks with row block major
    int  *v_ping_pong_flag, // ping if == 0, else pong
    int v_prod_lock, int v_con_lock
){
    // The result is a matrix of LQ x LK_V in sxt block row block major


    static_assert( LQ%(2*r) ==0 );
    static_assert(LK_V%(K_Unroll*s) ==0);
    static_assert(DH% (t) ==0);

    using MMUL = aie::mmul<r, s, t, T_in, T_in, accauto>;

    constexpr unsigned rowA = LQ/r;
    constexpr unsigned colA = LK_V/s; 
    constexpr unsigned colB = DH/t;
    
    // The k index
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_RANGE(colA / K_Unroll, colA / K_Unroll)
    for(int i = 0; i < colA; i+= K_Unroll){
        _lock_acquire(v_con_lock,1);

        T_in* v_buffer_cur = (T_in*)retrive_ping_pong_buffer(
            V_buffer_ping,
            V_buffer_pong,
            v_ping_pong_flag
        );

        aie::vector<T_in, MMUL::size_A> A_vecs[2][K_Unroll];
        // 0: row z, 1: row z+1

        AIE_PREPARE_FOR_PIPELINING
        AIE_LOOP_RANGE(rowA / 2, rowA / 2)
        for(int z = 0; z < rowA; z+=2){
            
            T_in* pA1= Pij_buffer + MMUL::size_A* (z*colA +i);
            T_in* pA2 = pA1+ colA * MMUL::size_A;
            
            // Load A blocks
            AIE_LOOP_UNROLL_FULL
            for(int k=0; k<K_Unroll; ++k) {
                A_vecs[0][k] = aie::load_v<MMUL::size_A>(pA1 + k*MMUL::size_A);
                A_vecs[1][k] = aie::load_v<MMUL::size_A>(pA2 + k*MMUL::size_A);
            }

            AIE_PREPARE_FOR_PIPELINING
            AIE_LOOP_RANGE(colB, colB)
            for(int j = 0; j < colB; j++)
            {   
                // Load B blocks
                aie::vector<T_in, MMUL::size_B> B_vecs[K_Unroll];
                AIE_LOOP_UNROLL_FULL
                for(int k=0; k<K_Unroll; ++k){
                     B_vecs[k] = aie::load_v<MMUL::size_B>(v_buffer_cur + (k*colB + j) * MMUL::size_B);
                }

                T_out* pC1 = O_buffer + ((z*colB) + j) * MMUL::size_C;
                T_out* pC2 = pC1  + colB * MMUL::size_C; 

                MMUL C00(aie::load_v<MMUL::size_C>(pC1));
                MMUL C10(aie::load_v<MMUL::size_C>(pC2));

                AIE_LOOP_UNROLL_FULL
                for(int k=0; k<K_Unroll; ++k){
                    C00.mac(A_vecs[0][k], B_vecs[k]);
                    C10.mac(A_vecs[1][k], B_vecs[k]);
                }
                
                // store it back
                aie::store_v(pC1, C00.template to_vector<T_out>());
                aie::store_v(pC2, C10.template to_vector<T_out>());
            }
        }
        _lock_release(v_prod_lock, 1);
    }
}

// template<typename T_in, typename T_out, unsigned LQ, unsigned LK_V, unsigned DH, 
//     unsigned r, unsigned s, unsigned t>
// void matmul_pij_vmmul_modify_buffer(
//     T_in* __restrict Pij_buffer, // size of LQ x LK_V reorder in r x s blocks row block major
//     T_in* __restrict V_buffer_ping, // size of (2xs)x DH reorder in sxt blocks row block major
//     T_in* __restrict V_buffer_pong, // size of (2xs)x DH reorder in sxt blocks row block major
    
//     T_out* __restrict O_buffer, // size of LQ x DH, in block of rxt blocks with row block major
//     int  *v_ping_pong_flag, // ping if == 0, else pong
//     int v_prod_lock, int v_con_lock
// ){
//     matmul_pij_vmmul_generic<T_in, T_out, LQ, LK_V, DH, r, s, t, 2>(
//         Pij_buffer, V_buffer_ping, V_buffer_pong, O_buffer, v_ping_pong_flag, v_prod_lock, v_con_lock
//     );
// }




// template<typename T_in, typename T_out, unsigned LQ, unsigned LK_V, unsigned DH, 
//     unsigned r, unsigned s, unsigned t>
// void matmul_pij_vmmul_modify_buffer_4s_DH_V_buffer(
//     T_in* __restrict Pij_buffer, // size of LQ x LK_V reorder in r x s blocks row block major
//     T_in* __restrict V_buffer_ping, // size of (4xs)x DH reorder in sxt blocks row block major
//     T_in* __restrict V_buffer_pong, // size of (4xs)x DH reorder in sxt blocks row block major
    
//     T_out* __restrict O_buffer, // size of LQ x DH, in block of rxt blocks with row block major
//     int  *v_ping_pong_flag, // ping if == 0, else pong
//     int v_prod_lock, int v_con_lock
// ){
//     matmul_pij_vmmul_generic<T_in, T_out, LQ, LK_V, DH, r, s, t, 4>(
//         Pij_buffer, V_buffer_ping, V_buffer_pong, O_buffer, v_ping_pong_flag, v_prod_lock, v_con_lock
//     );
// }




// template<int m, int k, int r, int t>
// void find_m_new_using_reduce_max(
//     bfloat16* __restrict S_buffer, // size of m x k with rxt block in row block major
//     bfloat16* __restrict m_i, //old m_i buffer of size m, represents old max per row

//     bfloat16* __restrict m_i_new // new m_i buffer of size m, represents new max per row
// ){

//     constexpr int vec_size = 16; // bfloat16 vector size
//     static_assert( k % t ==0 );
//     static_assert( m % r ==0 );
//     static_assert( vec_size >=r && vec_size >= t );


//     for(int i = 0; i < m; i+= vec_size){

//         aie::vector<bfloat16, vec_size> old_m_max = aie::load_v<vec_size>(m_i + i);
//         aie::vector<bfloat16, vec_size> cur_max = aie::broadcast<bfloat16, vec_size>(bfloat16(NEG_INF_FP32));
        
//         for(int j = 0; j < k; j+= t){

//             AIE_LOOP_UNROLL_FULL
//             for(int l = 0; l <vec_size; l+= r){
                
//                 bfloat16* ps = S_buffer + i*(k) + j*r + l*k;

//                 AIE_LOOP_UNROLL_FULL
//                 for(int rr = 0; rr< r; rr++){
//                     aie::vector<bfloat16, t> data_vec = aie::load_v<t>(ps + rr*t);

//                     bfloat16 reduce_val = aie::reduce_max(data_vec);
//                     cur_max[l + rr] =   cur_max[l + rr] > reduce_val ? cur_max[l + rr] : reduce_val;
//                 }
//             }
//         }

//         aie::vector<bfloat16, vec_size> new_m_max = aie::max(old_m_max, cur_max);
//         aie::store_v(m_i_new + i, new_m_max);
//     }

// }




// template<int m, int k, int r, int t>
// void find_m_new_using_gt_and_reduce_max(

//     bfloat16* __restrict S_buffer, // size of m x k with rxt block in row block major
//     bfloat16* __restrict m_i, //old m_i buffer of size m, represents old max per row

//     bfloat16* __restrict m_i_new // new m_i buffer of size m, represents new max per row   
// ){


//     static_assert( k % t ==0 );
//     static_assert( m % r ==0 );
    
//     AIE_PREPARE_FOR_PIPELINING
//     AIE_LOOP_RANGE(m / r, m / r)
//     for(int i = 0; i < m; i += r){
//         aie::vector<bfloat16, r*t> cur_max_s = aie::broadcast<bfloat16, r*t>(bfloat16(NEG_INF_FP32));
//         AIE_PREPARE_FOR_PIPELINING
//         AIE_LOOP_RANGE(k / t, k / t)
//         for(int j = 0; j < k; j+= t){
//             aie::vector<bfloat16, r*t> data_block = aie::load_v<r*t>(S_buffer + i*k + j*r);

//             aie::mask<r*t> gt_mask = aie::gt(data_block, cur_max_s); // 1 if data_block[i] > cur_max_s[i]
//             // select  m[i] == 0? v1[i] : v2[i];
//             cur_max_s = aie::select( cur_max_s, data_block, gt_mask);

//         }

//         // reduce max from cur_max_s to cur_max_r
//         AIE_LOOP_UNROLL_FULL
//         for(unsigned l = 0; l <r; l++){
//             aie::vector<bfloat16, t> cur_row_vec =cur_max_s.template extract<t>(l);
//             bfloat16 reduce_val = aie::reduce_max(cur_row_vec);

//             m_i_new[i + l] = m_i[i + l] > reduce_val ? m_i[i + l] : reduce_val;
//         }

//     }


// }

template<int m, int k, int r, int t, bool casual=false>
void find_m_new_using_gt_and_reduce_max(

    bfloat16* __restrict S_buffer, // size of m x k with rxt block in row block major
    bfloat16* __restrict m_i, //old m_i buffer of size m, represents old max per row

    bfloat16* __restrict m_i_new // new m_i buffer of size m, represents new max per row
){


    static_assert( k % t ==0 );
    static_assert( m % r ==0 );

    aie::vector<bfloat16, r*t> neg_inf_block = aie::broadcast<bfloat16, r*t>(bfloat16(NEG_INF_FP32));

    // Build lower-triangle mask using index vectors to avoid uint64 bit-ordering ambiguity.
    // Element idx = row*t + col; keep (bit=1) where col <= row.
    aie::mask<r*t> causal_triangle_mask;
    if constexpr(casual && r==t){
        static_assert(r == 8 && t == 8);
        alignas(aie::vector_decl_align) int32_t col_idx_arr[r*t];
        alignas(aie::vector_decl_align) int32_t row_idx_arr[r*t];
        AIE_LOOP_UNROLL_FULL
        for(int idx = 0; idx < r*t; idx++){
            col_idx_arr[idx] = idx % t;
            row_idx_arr[idx] = idx / t;
        }
        aie::vector<int32_t, r*t> col_vec = aie::load_v<r*t>(col_idx_arr);
        aie::vector<int32_t, r*t> row_vec = aie::load_v<r*t>(row_idx_arr);
        causal_triangle_mask = aie::le(col_vec, row_vec);
    }

    aie::vector<bfloat16, r*t> cur_max_s;
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_RANGE(m / r, m / r)
    for(int i = 0; i < m; i += r){

        broadcast_vector_to_mm<bfloat16, r, t>(cur_max_s, m_i + i);


        AIE_PREPARE_FOR_PIPELINING
        AIE_LOOP_RANGE(k / t, k / t)
        for(int j = 0; j < k; j+= t){
            aie::vector<bfloat16, r*t> data_block = aie::load_v<r*t>(S_buffer + i*k + j*r);

            if constexpr(casual && r==t){
                static_assert(r == 8 && t == 8);

                if(i == j){
                    // Diagonal tile: upper-triangle → -inf so exp gives 0; lower-triangle → keep score.
                    data_block = aie::select(neg_inf_block, data_block, causal_triangle_mask);

                    aie::mask<r*t> gt_mask = aie::gt(data_block, cur_max_s);
                    cur_max_s = aie::select(cur_max_s, data_block, gt_mask);
                    aie::store_v(S_buffer + i*k + j*r, data_block);

                }else if(i < j){
                    // Upper-triangle tile: all future tokens → -inf so exp gives 0.
                    aie::store_v(S_buffer + i*k + j*r, neg_inf_block);
                }else{
                    // Lower-triangle tile: all valid, normal gt update.
                    aie::mask<r*t> gt_mask = aie::gt(data_block, cur_max_s);
                    cur_max_s = aie::select(cur_max_s, data_block, gt_mask);
                }
            }else{
                aie::mask<r*t> gt_mask = aie::gt(data_block, cur_max_s);
                // select  m[i] == 0? v1[i] : v2[i];
                cur_max_s = aie::select(cur_max_s, data_block, gt_mask);
            }



        }

        // reduce max from cur_max_s to cur_max_r
        AIE_PREPARE_FOR_PIPELINING
        AIE_LOOP_RANGE(r, r)
        AIE_PEANO_UNROLL_FULL
        for(unsigned l = 0; l <r; l++){
            aie::vector<bfloat16, t> cur_row_vec =cur_max_s.template extract<t>(l);
            bfloat16 reduce_val = aie::reduce_max(cur_row_vec);

            m_i_new[i + l] =reduce_val;
        }

    }


}













template<int size>
void calculate_exp_difference(
    bfloat16* __restrict m_i, // old m_i buffer of size size
    bfloat16* __restrict m_i_new, // new m_i buffer of size size
    bfloat16* __restrict exp_diff_vec // output exp difference vector of size size
){
    constexpr int vec_size = size; // bfloat16 vector size
    static_assert( size % vec_size ==0 );


    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_RANGE(size / vec_size, size / vec_size)
    for(int i = 0; i < size; i+= vec_size){
        aie::vector<bfloat16, vec_size> old_m_vec = aie::load_v<vec_size>(m_i + i);
        aie::vector<bfloat16, vec_size> new_m_vec = aie::load_v<vec_size>(m_i_new + i);

        aie::vector<bfloat16, vec_size> diff_vec = aie::sub(old_m_vec, new_m_vec); // m_ii_block - m_new

        aie::store_v(exp_diff_vec + i, exp2_bf16_noLUT<vec_size>(diff_vec));
    }

}






// template<int m, int k, int r, int t>
// void calculate_Pij_unnormalize_and_sum_pij_bf16_Acc(
//     bfloat16* S_buffer,        // mxk buffer order in rxt block row major
//     bfloat16* m_new,           // size of m 
//     float* pij_row_sum_buffer  // output of reduce sum
// ){
//     // Compile-time constants
//     static_assert(m % r == 0);
//     static_assert(k % t == 0);

//     constexpr int accum_vec_size = r * t;
    
//     // NOTE: num_accum_vec is guaranteed to be 1 here because accum_vec_size == r*t
//     // We removed the 'v' loops entirely.

//     // --- MAIN OUTER LOOP (Iterate through Row Blocks) ---
//     // Pipelining here keeps the AIE busy fetching new tiles
//     AIE_PREPARE_FOR_PIPELINING
//     AIE_LOOP_RANGE(m / r, m / r)
//     for(int i = 0; i < m; i += r) {

//         // ---------------------------------------------------------
//         // 1. Prepare Broadcast Vector & Initialize Accumulator
//         // ---------------------------------------------------------
//         // We act on a single vector register representing the r*t tile
//         aie::vector<bfloat16, accum_vec_size> pij_accum = aie::zeros<bfloat16, accum_vec_size>();
//         aie::vector<bfloat16, accum_vec_size> m_broadcast;

//         // Load and broadcast rows. Unroll fully as 'r' is small.
//         AIE_PREPARE_FOR_PIPELINING
//         AIE_LOOP_RANGE(r, r)
//         for(int l = 0; l < r; l++){
//             // Load scalar m_new, broadcast to vector, insert into large vector
//             aie::vector<bfloat16, t> row_vec = aie::broadcast<bfloat16, t>(m_new[i + l]);
//             m_broadcast.template insert<t>(l, row_vec);
//         }

//         // ---------------------------------------------------------
//         // 2. Process Columns (The Heavy Compute Loop)
//         // ---------------------------------------------------------
//         // Pipeline this loop for maximum MAC performance
//         AIE_PREPARE_FOR_PIPELINING
//         AIE_LOOP_RANGE(k / t, k / t)
//         for(int j = 0; j < k; j += t){
//             // Calculate pointer to current block (Standard Block-Layout Arithmetic)
//             bfloat16 * cur_s = S_buffer + i*k + j*r;
            
//             // Load Input Block
//             aie::vector<bfloat16, accum_vec_size> s = aie::load_v<accum_vec_size>(cur_s);
            
//             // Subtract Max
//             s = aie::sub(s, m_broadcast);

//             // Exponentiate
//             s = exp2_bf16_noLUT<accum_vec_size>(s);
            
//             // Accumulate (Add to running sum for these rows)
//             pij_accum = aie::add(pij_accum, s);

//             // Store result back to buffer
//             aie::store_v(cur_s, s);
//         }

//         // ---------------------------------------------------------
//         // 3. Reduce and Write Output
//         // ---------------------------------------------------------
//         // Unroll fully to extract and reduce rows quickly
//         AIE_PREPARE_FOR_PIPELINING
//         AIE_LOOP_RANGE(r, r)
//         for(int l = 0; l < r; l++){
//             // Extract the l-th row (size t)
//             aie::vector<bfloat16, t> row_vec = pij_accum.template extract<t>(l);
            
//             // Reduce sum the row to a single float
//             float reduce_res = (float)aie::reduce_add(row_vec);
            
//             // Accumulate into the global buffer
//             pij_row_sum_buffer[i + l] += reduce_res;
//         }
//     }
// }


// template<int m, int k, int r, int t>
// void calculate_Pij_unnormalize_and_sum_pij_bf16_Acc(
//     bfloat16* S_buffer, // mxk buffer order in rxt block row major
//     bfloat16* m_new, // size of m 
//     float* pij_row_sum_buffer //output of reduce sum
//     //NOTE: will not erase pij_row_sum_buffer, so 
// ){
//     static_assert(m % r == 0);
//     static_assert(k % t == 0);

//     constexpr int accum_vec_size = r*t;
//     static_assert((r * t) % accum_vec_size == 0);
//     static_assert(accum_vec_size >= t);

//     constexpr int r_per_accum_vec = accum_vec_size / t;
//     constexpr int num_accum_vec = (r * t) / accum_vec_size;


//     aie::vector<bfloat16, accum_vec_size> pij_row_accum_vec[num_accum_vec];
//     aie::vector<bfloat16, accum_vec_size> m_new_broadcast_vec[num_accum_vec];
//     // --- MAIN OUTER LOOP (Iterate through Row Blocks) ---

//     AIE_PREPARE_FOR_PIPELINING
//     AIE_LOOP_RANGE(m / r, m / r)
//     for(int i = 0; i < m; i += r) {

//         AIE_PREPARE_FOR_PIPELINING
//         AIE_LOOP_RANGE(num_accum_vec, num_accum_vec)
//         for(int v = 0; v < num_accum_vec; v++){
                    
//             // 1. Initialize accumulators for THIS block of rows
//             pij_row_accum_vec[v] = aie::zeros<bfloat16, accum_vec_size>();

//             // 2. Prepare m_new broadcast for THIS block of rows
//             // Note: We use (m_new + i) to get the correct offset
//             AIE_PREPARE_FOR_PIPELINING
//             AIE_LOOP_RANGE(r_per_accum_vec, r_per_accum_vec)
//             for(int l = 0; l < r_per_accum_vec; l++){
//                 // Calculate global row index relative to current block
//                 int row_in_block = v * r_per_accum_vec + l;
                
//                 // Load m_new for the specific row (i + row_in_block)
//                 aie::vector<bfloat16, t> row_vec = aie::broadcast<bfloat16, t>(m_new[i + row_in_block]);
//                 m_new_broadcast_vec[v].template insert<t>(l, row_vec);
//             }
//         }

//         // 3. Process Columns (Iterate through Col Blocks)
//         AIE_PREPARE_FOR_PIPELINING
//         AIE_LOOP_RANGE(k / t, k / t)
//         for(int j = 0; j < k; j += t){
//             bfloat16 * cur_s = S_buffer + i*k + j*r;
            
//             AIE_LOOP_UNROLL_FULL
//             for(int l = 0; l < num_accum_vec; l++){
//                 aie::vector<bfloat16, accum_vec_size> s = aie::load_v<accum_vec_size>(cur_s);
                

//                 s = aie::sub(s,m_new_broadcast_vec[l]);

//                 // // now, do the exp
//                 s = exp2_bf16_noLUT<accum_vec_size>(s);
                
//                 //pij_row_accum_vec[l] = aie::mac(pij_row_accum_vec[l], (bfloat16)1.0f, s);
//                 pij_row_accum_vec[l] = aie::add(pij_row_accum_vec[l], s);

//                 aie::store_v(cur_s, s);
//                 cur_s += accum_vec_size;
//             }
//         }

//         // 4. Reduce and Write Output for THIS block of rows
//         AIE_PREPARE_FOR_PIPELINING
//         AIE_LOOP_RANGE(num_accum_vec, num_accum_vec)
//         for(int v = 0; v < num_accum_vec; v++){
//             aie::vector<bfloat16, accum_vec_size> vec_float = pij_row_accum_vec[v];
            
//             AIE_PREPARE_FOR_PIPELINING
//             AIE_LOOP_RANGE(r_per_accum_vec, r_per_accum_vec)
//             for(int l = 0; l < r_per_accum_vec; l++){
//                 aie::vector<bfloat16, t> row_vec = vec_float.template extract<t>(l);
//                 float reduce_res =(float) aie::reduce_add(row_vec);
                
//                 // Write to global output buffer at offset i
//                 int row_in_block = v * r_per_accum_vec + l;
//                 pij_row_sum_buffer[i + row_in_block] += reduce_res;
//             }
//         }
//     }
// }









template<int m, int k, int r, int t, typename OUT_T>
void calculate_Pij_unnormalize_and_sum_pij(
    bfloat16* __restrict S_buffer, // mxk buffer order in rxt block row major
    bfloat16* __restrict m_new, // size of m 
    OUT_T* __restrict pij_row_sum_buffer //output of reduce sum
    //NOTE: will not erase pij_row_sum_buffer, so 
){
    static_assert(m % r == 0);
    static_assert(k % t == 0);

    constexpr int accum_vec_size = r*t;
    static_assert((r * t) % accum_vec_size == 0);
    static_assert(accum_vec_size >= t);

    constexpr int r_per_accum_vec = accum_vec_size / t;
    // constexpr int num_accum_vec = (r * t) / accum_vec_size;  // only 1 num_acc_vec is sufficient


    aie::accum<accfloat, accum_vec_size> pij_row_accum_vec;
    aie::vector<bfloat16, accum_vec_size> m_new_broadcast_vec;
    // --- MAIN OUTER LOOP (Iterate through Row Blocks) ---

    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_RANGE(m / r, m / r)
    for(int i = 0; i < m; i += r) {

                    
        // 1. Initialize accumulators for THIS block of rows
        pij_row_accum_vec = aie::zeros<accfloat, accum_vec_size>();

        // // 2. Prepare m_new broadcast for THIS block of rows
        // // Note: We use (m_new + i) to get the correct offset
        // AIE_PREPARE_FOR_PIPELINING
        // AIE_LOOP_RANGE(r_per_accum_vec, r_per_accum_vec)
        // for(int l = 0; l < r_per_accum_vec; l++){
        //     // Calculate global row index relative to current block
        //     int row_in_block = v * r_per_accum_vec + l;
            
        //     // Load m_new for the specific row (i + row_in_block)
        //     aie::vector<bfloat16, t> row_vec = aie::broadcast<bfloat16, t>(m_new[i + row_in_block]);
        //     m_new_broadcast_vec[v].template insert<t>(l, row_vec);
        // }

        broadcast_vector_to_mm<bfloat16, r, t>(
            m_new_broadcast_vec,
            m_new+ i 
        );
    

        // // 3. Process Columns (Iterate through Col Blocks)
        // AIE_PREPARE_FOR_PIPELINING
        // AIE_LOOP_RANGE(k / t, k / t)
        // for(int j = 0; j < k; j += t){
        //     bfloat16 * cur_s = S_buffer + i*k + j*r;
            
        //     AIE_LOOP_UNROLL_FULL
        //     for(int l = 0; l < num_accum_vec; l++){
        //         aie::vector<bfloat16, accum_vec_size> s = aie::load_v<accum_vec_size>(cur_s);
                

        //         s = aie::sub(s,m_new_broadcast_vec[l]);

        //         // now, do the exp
        //         s = exp2_bf16_noLUT<accum_vec_size>(s);
                
        //         pij_row_accum_vec[l] = aie::mac(pij_row_accum_vec[l], (bfloat16)1.0f, s);
        //         // pij_row_accum_vec[l] = aie::add(pij_row_accum_vec[l].template to_vector<bfloat16>(), s);

        //         aie::store_v(cur_s, s);
        //         cur_s += accum_vec_size;
        //     }
        // }

        // 3. Process Columns (Iterate through Col Blocks)
        constexpr int total_iter = (k/t);
        bfloat16 * cur_s = S_buffer + i*k;

        AIE_PREPARE_FOR_PIPELINING
        AIE_LOOP_RANGE(total_iter, total_iter)
        for(int idx = 0; idx < total_iter; idx++){
          
            aie::vector<bfloat16, accum_vec_size> s = aie::load_v<accum_vec_size>(cur_s);

            s = aie::sub(s, m_new_broadcast_vec);

            // now, do the exp
            s = exp2_bf16_noLUT<accum_vec_size>(s);

            pij_row_accum_vec= aie::mac(pij_row_accum_vec, (bfloat16)1.0f, s);
            
            aie::store_v(cur_s, s);
            cur_s += accum_vec_size;
        }
        // // 4. Reduce and Write Output for THIS block of rows
        // AIE_PREPARE_FOR_PIPELINING
        // AIE_LOOP_RANGE(num_accum_vec, num_accum_vec)
        // for(int v = 0; v < num_accum_vec; v++){
        //     aie::vector<float, accum_vec_size> vec_float = pij_row_accum_vec[v].template to_vector<float>();
            
        //     AIE_LOOP_UNROLL_FULL
        //     for(int l = 0; l < r_per_accum_vec; l++){
        //         aie::vector<float, t> row_vec = vec_float.template extract<t>(l);
        //         float reduce_res = aie::reduce_add(row_vec);
                
        //         // Write to global output buffer at offset i
        //         int row_in_block = v * r_per_accum_vec + l;
        //         pij_row_sum_buffer[i + row_in_block] += reduce_res;
        //     }
        // }


        // 4. Reduce and Write Output for THIS block of rows
        aie::vector<float, accum_vec_size> vec_float = pij_row_accum_vec.template to_vector<float>();        
        AIE_PREPARE_FOR_PIPELINING
        AIE_LOOP_RANGE(r, r)
        AIE_PEANO_UNROLL_FULL
        for(int r_idx = 0; r_idx < r; r_idx++){
             aie::vector<float, t> row_vec = vec_float.template extract<t>(r_idx);
             float reduce_res = aie::reduce_add(row_vec);
             
             // Write to global output buffer at offset i
             pij_row_sum_buffer[i + r_idx] = AIE_FADD_NONNEG(pij_row_sum_buffer[i + r_idx], reduce_res);
        }


    }
}




template<int m, int k, int r, int t, typename OUT_T>
void calculate_Pij_unnormalize_and_sum_pij_low_precision(
    bfloat16* __restrict S_buffer, // mxk buffer order in rxt block row major
    bfloat16* __restrict m_new, // size of m 
    OUT_T* __restrict pij_row_sum_buffer //output of reduce sum
    //NOTE: will not erase pij_row_sum_buffer, so 
){
    static_assert(m % r == 0);
    static_assert(k % t == 0);

    constexpr int accum_vec_size = r*t;
    static_assert((r * t) % accum_vec_size == 0);
    static_assert(accum_vec_size >= t);

    constexpr int r_per_accum_vec = accum_vec_size / t;
    // constexpr int num_accum_vec = (r * t) / accum_vec_size;  // only 1 num_acc_vec is sufficient
    using MMUL = aie::mmul<r, t, t, bfloat16, bfloat16, accauto>;

    //aie::accum<accfloat, accum_vec_size> pij_row_accum_vec;
    aie::vector<bfloat16, accum_vec_size> m_new_broadcast_vec;

    aie::vector<bfloat16, MMUL::size_C> zero_vec = aie::zeros<bfloat16, MMUL::size_C>(); 
    aie::vector<bfloat16, MMUL::size_C> one_vec = aie::broadcast<bfloat16, MMUL::size_C>(1.0f);

    alignas(aie::vector_decl_align) float mmu_res[r*t];
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_RANGE(m / r, m / r)
    for(int i = 0; i < m; i += r) {

                    
        // 1. Initialize accumulators for THIS block of rows
        //pij_row_accum_vec = aie::zeros<accfloat, accum_vec_size>();
        MMUL C00(zero_vec);    


        // // 2. Prepare m_new broadcast for THIS block of rows
        // // Note: We use (m_new + i) to get the correct offset

        broadcast_vector_to_mm<bfloat16, r, t>(
            m_new_broadcast_vec,
            m_new+ i 
        );
    

        // 3. Process Columns (Iterate through Col Blocks)
        constexpr int total_iter = (k/t);
        bfloat16 * cur_s = S_buffer + i*k;

        AIE_PREPARE_FOR_PIPELINING
        AIE_LOOP_RANGE(total_iter, total_iter)
        for(int idx = 0; idx < total_iter; idx++) {
          
            aie::vector<bfloat16, accum_vec_size> s = aie::load_v<accum_vec_size>(cur_s);

            s = aie::sub(s, m_new_broadcast_vec);

            // now, do the exp
            s = exp2_bf16_noLUT<accum_vec_size>(s);

            //pij_row_accum_vec= aie::mac(pij_row_accum_vec, (bfloat16)1.0f, s);
            C00.mac( s, one_vec);
            
            aie::store_v(cur_s, s);
            cur_s += accum_vec_size;
        }

        aie::store_v((float*)mmu_res, C00.template to_vector<float>());

        for(int idx = 0; idx < r*t; idx+=t){
            pij_row_sum_buffer[i + idx/t] = AIE_FADD_NONNEG(pij_row_sum_buffer[i + idx/t], mmu_res[idx]);
        }

        // aie::vector<float, accum_vec_size> vec_float  = C00.template to_vector<float>();
        // // 4. Reduce and Write Output for THIS block of rows    
        // AIE_PREPARE_FOR_PIPELINING
        // AIE_LOOP_RANGE(r, r)
        // for(int r_idx = 0; r_idx < r; r_idx++){
        //      aie::vector<float, t> row_vec = vec_float.template extract<t>(r_idx);

        //      // Write to global output buffer at offset i
        //      pij_row_sum_buffer[i + r_idx] += row_vec.get(0);
        // }


    }
}

template<int m, int k, int r, int t, int accum_vec_size=16>
void calculate_Oij_mul_exp_old(
    float * __restrict O_buffer, // mxk buffer order in rxt block row major
    bfloat16 * __restrict exp_old // mx 
    
){
    // calculate O_buffer = exp_old * O_buffer


    static_assert(m % r == 0);
    static_assert(k % t == 0);


    static_assert((r * t) % accum_vec_size == 0);
    static_assert(accum_vec_size >= t);
    static_assert(accum_vec_size % t == 0, "t must divide accum_vec_size for correct broadcasting");

    constexpr int r_per_accum_vec = accum_vec_size / t;
    constexpr int num_accum_vec = (r * t) / accum_vec_size;

   
    aie::vector<float, accum_vec_size> exp_old_broadcast_vec[num_accum_vec];


    for(int i = 0; i < m; i+=r){

        // AIE_PREPARE_FOR_PIPELINING
        // AIE_LOOP_RANGE(num_accum_vec, num_accum_vec)
        // for(int v = 0; v < num_accum_vec; v++){

        //     AIE_PREPARE_FOR_PIPELINING
        //     AIE_LOOP_RANGE(r_per_accum_vec, r_per_accum_vec)
        //     for(int l = 0; l < r_per_accum_vec; l++){
        //         int row_in_block = v * r_per_accum_vec + l;
        //         aie::vector<float, t> row_vec = aie::broadcast<float, t>((float) exp_old[i + row_in_block]);
        //         exp_old_broadcast_vec[v].template insert<t>(l, row_vec);
        //     }
        // }


        AIE_PEANO_UNROLL_FULL
        for(int r_idx = 0; r_idx < r; r_idx++){
            int v = r_idx / r_per_accum_vec;
            int l = r_idx % r_per_accum_vec;
            aie::vector<float, t> row_vec = aie::broadcast<float, t>((float) exp_old[i + r_idx]);
            exp_old_broadcast_vec[v].template insert<t>(l, row_vec);
        }


        // AIE_PREPARE_FOR_PIPELINING
        // AIE_LOOP_RANGE(k/t, k/t)
        // for(int j = 0; j < k; j+= t){

        //     float * cur_o = O_buffer + i*k + j*r;

        //     AIE_PREPARE_FOR_PIPELINING
        //     AIE_LOOP_RANGE(num_accum_vec, num_accum_vec)
        //     for(int l = 0; l < num_accum_vec; l++){
        //         aie::vector<float, accum_vec_size> o = aie::load_v<accum_vec_size>(cur_o);

        //         aie::accum<accfloat,accum_vec_size> res = mul_elem_16_accuracy_safe(o, exp_old_broadcast_vec[l]);

        //         aie::store_v(cur_o, res.template to_vector<float>());
        //         cur_o += accum_vec_size;
        //     }

        // }
        constexpr int total_iter = (k/t) * num_accum_vec;
        float * cur_o = O_buffer + i*k;

        AIE_PREPARE_FOR_PIPELINING
        AIE_LOOP_RANGE(total_iter, total_iter)
        for(int idx = 0; idx < total_iter; idx++){
            int l = idx % num_accum_vec;

            aie::vector<float, accum_vec_size> o = aie::load_v<accum_vec_size>(cur_o);
            aie::accum<accfloat,accum_vec_size> res;
            if constexpr(accum_vec_size == 16){
                res = mul_elem_16_accuracy_safe(o, exp_old_broadcast_vec[l]);
            }else{
                res = aie::mul(o, exp_old_broadcast_vec[l]);
            }

            aie::store_v(cur_o, res.template to_vector<float>());
            cur_o += accum_vec_size;
        }

    }


}

template<int m, int vec_size=16>
void vec_dot_fp32_intrinsic( float* __restrict vec_a, bfloat16* __restrict vec_b, float* __restrict result){

    static_assert( m % vec_size ==0 ); //TODO: for now

    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_RANGE(m / vec_size, m / vec_size)
    for(int i = 0; i< m; i+=vec_size){  

        aie::vector<float, vec_size> a_vec = aie::load_v<vec_size>(vec_a + i);
        aie::vector<bfloat16, vec_size> b_vec = aie::load_v<vec_size>(vec_b + i);



        aie::accum<accfloat, vec_size> b_vec_acc;
        b_vec_acc.from_vector(b_vec);
        aie::accum<accfloat,vec_size> res;
        if constexpr(vec_size == 16){
            res = mul_elem_16_accuracy_safe(a_vec, b_vec_acc.template to_vector<float>());            
        }else{
            res = aie::mul(a_vec, b_vec_acc.template to_vector<float>());
        }


        aie::store_v(result +i, res.template to_vector<float>());
    }

}







template<int m, int k>
void mul_scalar(
    bfloat16* __restrict A_buffer,
    bfloat16 scale_value
){
    constexpr int vec_size = 64;
    static_assert(  (m*k) % vec_size ==0 );

    aie::vector<bfloat16, vec_size> scale_vector = aie::broadcast<bfloat16, vec_size>(scale_value);
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_RANGE((m*k) / vec_size, (m*k) / vec_size)
    AIE_PEANO_NO_PIPELINE
    for(int i = 0; i < m*k; i+= vec_size) {
        aie::vector<bfloat16, vec_size> A_vec = aie::load_v<vec_size>(A_buffer + i);
        aie::accum<accfloat, vec_size> res = aie::mul(A_vec, scale_vector);
        aie::store_v(A_buffer + i, res.template to_vector<bfloat16>());
    }

}

// template<int m, int k, int r, int s>
// void mm_buffer_mul_scale(
//     bfloat16* __restrict A_buffer,
//     bfloat16* __restrict scale_vector
// ){


//     // 1. Pre-load scale vector
//     // This setup remains the same: we use the same 'r' scale factors for every block.
//     aie::vector<bfloat16, s> scale_broadcast_vector[r];
//     AIE_LOOP_UNROLL_FULL
//     for(int i = 0; i < r; i++){
//         scale_broadcast_vector[i] = aie::broadcast<bfloat16, s>(scale_vector[i]);
//     }

//     // 2. Calculate total blocks
//     // M/r = number of block rows
//     // K/s = number of block columns
//     constexpr int total_blocks = (m / r) * (k / s);
    
//     bfloat16* ptr = A_buffer;

//     // 3. Single Flattened Loop
//     // We iterate once per tile/block.
//     AIE_PREPARE_FOR_PIPELINING
//     AIE_LOOP_RANGE(total_blocks, total_blocks)
//     for(int b = 0; b < total_blocks; b++) 
//     {
//         // Inside this loop, we process exactly one r*s tile
//         // The compiler will effectively pipeline this across blocks
//         for(int i = 0; i < r; i++){
//             // Load contiguous vector of size 's'
//             aie::vector<bfloat16, s> A_vec = aie::load_v<s>(ptr);
            
//             // Multiply
//             aie::vector<bfloat16, s> scale_acc = aie::mul(A_vec, scale_broadcast_vector[i]);
            
//             // Store
//             aie::store_v(ptr, scale_acc);
            
//             // Advance pointer to the next row in this block
//             ptr += s;
//         }
        
//         // At the end of the unrolled 'i' loop, 'ptr' has advanced by r*s.
//         // It is now perfectly aligned at the start of the next block.
//     }
// }





template<int m, int k, int r, int t>
void O_normalize(

    float* __restrict O_buffer, // size of m x k with rxt block in row block major
    float* __restrict L_i_buffer // size of mx1

){

    //NOTE: the result store back to O_buffer directly, but in bfloat16 format


    bfloat16* O_bf16_buffer = (bfloat16*) O_buffer;
    //TODO: medrge with calculate_Oij_mul_exp_old ?
    static_assert(m % r == 0);
    static_assert(k % t == 0);

    constexpr int accum_vec_size = r*t;
    static_assert((r * t) % accum_vec_size == 0);
    static_assert(accum_vec_size >= t);
    static_assert(accum_vec_size % t == 0, "t must divide accum_vec_size for correct broadcasting");


    // constexpr int num_accum_vec = (r * t) / accum_vec_size; // 1

   
    aie::vector<bfloat16, accum_vec_size> L_i_broadcast_vec;

    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_RANGE(m/r, m/r)
    for(int i = 0; i < m; i+=r){

        AIE_PREPARE_FOR_PIPELINING
        AIE_LOOP_RANGE(r, r)
        AIE_PEANO_UNROLL_FULL
        for(int idx = 0; idx < r; idx++) {
            
            // 2. Load and Compute
            // Note: 'idx' is mathematically equivalent to (v * r_per_accum_vec + l)
            // So we can use 'idx' directly for the buffer offset.
            bfloat16 val = L_i_buffer[i + idx];
            
            // Compute reciprocal and broadcast
            aie::vector<bfloat16, t> row_vec = aie::broadcast<bfloat16, t>((bfloat16)1.0f / val);

            // 3. Insert into the target vector
            L_i_broadcast_vec.template insert<t>(idx, row_vec);
        }        


        constexpr int total_iter = (k/t) ;
        float * cur_o = O_buffer + i*k;
        bfloat16 * cur_o_bf16 = O_bf16_buffer + i*k;

        aie::accum<accfloat, accum_vec_size> o;
        AIE_PREPARE_FOR_PIPELINING
        AIE_LOOP_RANGE(total_iter, total_iter)
        for(int idx = 0; idx < total_iter; idx++){
            o.from_vector(aie::load_v<accum_vec_size>(cur_o));
            aie::accum<accfloat,accum_vec_size> res = aie::mul(o.template to_vector<bfloat16>(), L_i_broadcast_vec);
            aie::store_v(cur_o_bf16, res.template to_vector<bfloat16>());
            cur_o += accum_vec_size;
            cur_o_bf16 += accum_vec_size;
        }


    }


}


template<int m, int n, int r, int t>
void apply_padding_mask(

    bfloat16 * S_buffer, // size of m x n with rxt block in row block major
    int valid_n // valid length of n dimension
){

    // NOTE: for now
    static_assert(t == 8);


    static_assert(n % t == 0);
    static_assert(m % r == 0);


    int n_mod_t = valid_n % t;
    int n_div_t = valid_n / t;


    aie::vector<bfloat16, r*t> neg_inf_vec = aie::broadcast<bfloat16, r*t>(bfloat16(NEG_INF_FP32));



    // Construct partial mask only if needed
    aie::mask<r*t> partial_padding_mask = aie::mask<r*t>(true);
    if (n_mod_t != 0) {
        for(int i = 0; i < r; i++){
            for(int j = 0; j < n_mod_t; j++){
                partial_padding_mask.clear(i*t + j); 
            }
        }
    }


    // now, apply the mask by using select

    for(int i = 0; i < m; i+= r){
        int j = n_div_t * t;
        
        // 1. Handle the Boundary Block (Partial Pad)
        // Only if valid_n is not perfectly aligned to block boundary
        if(j < n && n_mod_t != 0) {
            aie::vector<bfloat16, r*t> S_block = aie::load_v<r*t>(S_buffer + i*n + j*r);
            aie::vector<bfloat16, r*t> masked_S_block = aie::select(S_block, neg_inf_vec, partial_padding_mask);
            aie::store_v(S_buffer + i*n + j*r, masked_S_block);
            j += t; // Move to next block
        } 
        
        // 2. Handle Remaining Blocks (Full Pad) - No branching
        for(; j < n; j+= t){
            aie::store_v(S_buffer + i*n + j*r, neg_inf_vec);
        }
    }


}


// template<int m, int k, int r, int s>
// void mm_buffer_mul_scale(
//     bfloat16* __restrict A_buffer,   // Added __restrict for optimization
//     bfloat16* __restrict scale_vector
// ){

//     // Pre-load and broadcast scale vector
//     aie::vector<bfloat16, s> scale_broadcast_vector[r];
//     for(int i = 0; i < r; i++){
//         scale_broadcast_vector[i] = aie::broadcast<bfloat16, s>(scale_vector[i]);
//     }

//     static_assert((m % r) == 0);
//     static_assert((k % s) == 0);

//     // Calculate total number of tiles (blocks)
//     // Assuming the buffer is packed as (M/r) * (K/s) tiles of size (r*s)
    
//     // We can iterate linearly through the blocks because the operation 
//     // is independent per tile.
//     bfloat16* ptr = A_buffer;






//     AIE_PREPARE_FOR_PIPELINING
//     AIE_LOOP_RANGE(m/r, m/r)
//     for(unsigned row = 0; row < m; row += r){
//         AIE_PREPARE_FOR_PIPELINING        
//         AIE_LOOP_RANGE(k/s, k/s)
//         for(unsigned col = 0; col < k; col += s){
            
//             // Inside one r*s block
//             // The block is stored as: [Row0_vec, Row1_vec, ... Row(r-1)_vec]
            
//             AIE_LOOP_UNROLL_FULL
//             for(int i = 0; i < r; i++){
//                 // 1. Load vector (size s)
//                 aie::vector<bfloat16, s> A_vec = aie::load_v<s>(ptr);
                
//                 // 2. Multiply with corresponding scale factor for this 'logical row'
//                 aie::vector<bfloat16, s> scale_acc = aie::mul(A_vec, scale_broadcast_vector[i]);
                
//                 // 3. Store result back
//                 aie::store_v(ptr, scale_acc);
                
//                 // 4. Advance pointer to the next row within this block
//                 ptr += s;
//             }
//             // At the end of this inner loop, 'ptr' has advanced by r*s 
//             // and is naturally pointing to the start of the NEXT block.
//         }
//     }







//     // AIE_PREPARE_FOR_PIPELINING
//     // AIE_LOOP_RANGE(m/r, m/r)
//     // for(unsigned row = 0; row < m; row += r){
//     //     AIE_PREPARE_FOR_PIPELINING        
//     //     AIE_LOOP_RANGE(k/s, k/s)
//     //     for(unsigned col = 0; col < k; col += s){
            
//     //         // Inside one r*s block
//     //         // The block is stored as: [Row0_vec, Row1_vec, ... Row(r-1)_vec]
            
//     //         AIE_LOOP_UNROLL_FULL
//     //         for(int i = 0; i < r; i++){
//     //             // 1. Load vector (size s)
//     //             aie::vector<bfloat16, s> A_vec = aie::load_v<s>(ptr);
                
//     //             // 2. Multiply with corresponding scale factor for this 'logical row'
//     //             aie::vector<bfloat16, s> scale_acc = aie::mul(A_vec, scale_broadcast_vector[i]);
                
//     //             // 3. Store result back
//     //             aie::store_v(ptr, scale_acc);
                
//     //             // 4. Advance pointer to the next row within this block
//     //             ptr += s;
//     //         }
//     //         // At the end of this inner loop, 'ptr' has advanced by r*s 
//     //         // and is naturally pointing to the start of the NEXT block.
//     //     }
//     // }
// }
// template<typename T_in_out, int m, int k, int r, int s, int t>
// void mm_buffer_with_fixed_matrix(T_in_out* A_buffer,
//     T_in_out* b_matrix){
//     //NOTE: b_matrix need be aligned load 


//     // The scenario:
//     // A_buffer is m x k reorder in rxs block row major
//     // b_matrix is sxt fixed matrix in col major
//     // out_buffer is m x t matrix
    
//     //NOTE: important assumption: r == s == t
//     static_assert( (r == s) && (s == t) );

//     // use similar to matmul_vectorized_2x2_mmul approach
//     constexpr int rowA = m/r;
//     constexpr int colA = k/s;
//     static_assert( (m % r) ==0);
//     static_assert( (k % s) ==0);
//     static_assert( rowA %2 ==0);

    
//     using MMUL = aie::mmul<r, s, t, T_in_out, T_in_out, accauto>;
//     static_assert(r * s == MMUL::size_A);
    
    
//     AIE_PREPARE_FOR_PIPELINING
//     AIE_LOOP_MAX_ITERATION_COUNT(rowA / 2)
//     for(unsigned z = 0; z< rowA; z+= 2){

//         T_in_out * pC1 = A_buffer + (z * colA) * MMUL::size_C;
//         T_in_out * pC2 = A_buffer + ((z + 1) * colA) * MMUL::size_C;
//         T_in_out* pA1 = pC1;
//         T_in_out* pA2 = pC2;

//         aie::vector<T_in_out, MMUL::size_A> A00;
//         aie::vector<T_in_out, MMUL::size_A> A10;
//         aie::vector<T_in_out, MMUL::size_A> A01;
//         aie::vector<T_in_out, MMUL::size_A> A11;
        
//         aie::vector<T_in_out, MMUL::size_B> B0 = aie::load_v<MMUL::size_B>(b_matrix); // fixed load
//         aie::vector<T_in_out, MMUL::size_B> B1 = aie::load_v<MMUL::size_B>(b_matrix); // fixed load

//         AIE_PREPARE_FOR_PIPELINING
//         AIE_LOOP_MAX_ITERATION_COUNT(colA)
//         for(unsigned i = 0; i < colA; i++)
//         AIE_LOOP_FLATTEN
//         {
//             aie::vector<bfloat16, MMUL::size_C> acc_C0 = aie::zeros<bfloat16, MMUL::size_C>(); 

//             MMUL C00(acc_C0);
//             MMUL C01(acc_C0);
//             MMUL C10(acc_C0);
//             MMUL C11(acc_C0);

//             A00 = aie::load_v<MMUL::size_A>(pA1);
//             pA1 += MMUL::size_A;
//             A01 = aie::load_v<MMUL::size_A>(pA1);
//             pA1 += MMUL::size_A;

//             A10 = aie::load_v<MMUL::size_A>(pA2);
//             pA2 += MMUL::size_A;
//             A11 = aie::load_v<MMUL::size_A>(pA2);
//             pA2 += MMUL::size_A;

//             C00.mac(A00, B0);
//             C01.mac(A01, B1);
//             C10.mac(A10, B0);
//             C11.mac(A11, B1);

//             aie::store_v(pC1, C00.template to_vector<T_in_out>());
//             pC1 += MMUL::size_C;
//             aie::store_v(pC1, C01.template to_vector<T_in_out>());
//             pC1 += MMUL::size_C;
//             aie::store_v(pC2, C10.template to_vector<T_in_out>());
//             pC2 += MMUL::size_C;
//             aie::store_v(pC2, C11.template to_vector<T_in_out>());
//             pC2 += MMUL::size_C;
//         }

//     }




// }



#endif