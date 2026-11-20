// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __WINDOW_ATTENTION_OPTIMIZE_COMMON_H__
#define __WINDOW_ATTENTION_OPTIMIZE_COMMON_H__
#define NOCPP
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>
#include "../zero.h"
#include <aie_api/aie.hpp>
#include "../aie_kernel_utils.h" //TODO: a better method?
#include "../lock_util.hpp"



template<typename T_in, typename T_out, unsigned LQ, unsigned DH, unsigned LK, 
    unsigned r, unsigned s, unsigned t>
void window_attention_matmul_q_k_transpose_mmul_manual_unroll_two(
    T_in* __restrict Q_buffer, // Q is size of LQ x DH in row block major
    T_in* __restrict K_buffer, // K is size of LK x DH in row block major, Thus, need to transpose inside
    T_out* __restrict S_buffer 
){

    constexpr unsigned rowA = LQ / r;
    constexpr unsigned colA = DH / s;
    constexpr unsigned colB = LK / t;

    using MMUL = aie::mmul<r, s, t, T_in, T_in, accauto>;
    static_assert(rowA % 2 == 0, "rowA must be multiple of 2");
    static_assert(colB % 2 == 0, "colB must be multiple of 2");
    static_assert(LQ%r == 0, "LQ must be multiple of r");
    static_assert(DH%s == 0, "DH must be multiple of s");
    static_assert(LK%t == 0, "LK must be multiple of t");

    // Outer loop over columns of K (horizontal movement in the output S)
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_RANGE(colB / 2, colB / 2)
    for(unsigned j = 0; j < colB; j += 2) {
        
        // Calculate the base pointer for the current two-column block of K
        // Each block of B (K) is MMUL::size_B. There are colA such blocks per column.
        T_in* k_block_base = K_buffer + (j * colA * MMUL::size_B);


        // Inner loop over rows of Q (vertical movement in the output S)
        AIE_PREPARE_FOR_PIPELINING
        AIE_LOOP_RANGE(rowA / 2, rowA / 2)
        for(unsigned z = 0; z < rowA; z += 2) {

            // S_buffer indexing: [row_idx * total_cols + col_idx] * block_size
            T_out *__restrict pC1 = S_buffer + ((z * colB) + j) * MMUL::size_C;
            T_out *__restrict pC2 = pC1 + (colB * MMUL::size_C); // Next row of blocks

            // Q_buffer indexing
            const T_in *__restrict pA1 = Q_buffer + (z * colA) * MMUL::size_A;
            const T_in *__restrict pA2 = pA1 + (colA * MMUL::size_A);

            // K_buffer (pB) indexing relative to our j-loop base
            const T_in *__restrict pB1 = k_block_base;
            const T_in *__restrict pB2 = k_block_base + (colA * MMUL::size_B);

            MMUL C00(aie::zeros<T_out, MMUL::size_C>());
            MMUL C01(aie::zeros<T_out, MMUL::size_C>());
            MMUL C10(aie::zeros<T_out, MMUL::size_C>());
            MMUL C11(aie::zeros<T_out, MMUL::size_C>());

            // Reduction loop over DH (the shared dimension)
            AIE_PREPARE_FOR_PIPELINING
            AIE_LOOP_RANGE(colA, colA)
            for(unsigned i = 0; i < colA; i++) AIE_LOOP_FLATTEN {
                aie::vector<T_in, MMUL::size_A> A0 = aie::load_v<MMUL::size_A>(pA1);
                aie::vector<T_in, MMUL::size_A> A1 = aie::load_v<MMUL::size_A>(pA2);
                pA1 += MMUL::size_A;
                pA2 += MMUL::size_A;

                aie::vector<T_in, MMUL::size_B> B0 = aie::load_v<MMUL::size_B>(pB1);
                aie::vector<T_in, MMUL::size_B> B1 = aie::load_v<MMUL::size_B>(pB2);
                static_assert(s==t);
                B0 = aie::transpose(B0, s, t);
                B1 = aie::transpose(B1, s, t);

                pB1 += MMUL::size_B;
                pB2 += MMUL::size_B;

                C00.mac(A0, B0);
                C01.mac(A0, B1);
                C10.mac(A1, B0);
                C11.mac(A1, B1);
            }

            // Store results - note the pC1/pC2 offsets
            aie::store_v(pC1, C00.template to_vector<T_out>());
            aie::store_v(pC1 + MMUL::size_C, C01.template to_vector<T_out>());
            
            aie::store_v(pC2, C10.template to_vector<T_out>());
            aie::store_v(pC2 + MMUL::size_C, C11.template to_vector<T_out>());
        }
    }
}


template<typename T_in, typename T_out, unsigned LQ, unsigned DH, unsigned LK,
    unsigned r, unsigned s, unsigned t>
void window_attention_matmul_q_k_transpose_mmul_normal_loop(
    T_in* __restrict Q_buffer, // Q is size of LQ x DH in row block major
    T_in* __restrict K_buffer, // K is size of LK x DH in row block major, Thus, need to transpose inside
    T_out* __restrict S_buffer
){

    constexpr unsigned rowA = LQ / r;
    constexpr unsigned colA = DH / s;
    constexpr unsigned colB = LK / t;

    using MMUL = aie::mmul<r, s, t, T_in, T_in, accauto>;
    static_assert(LQ%r == 0, "LQ must be multiple of r");
    static_assert(DH%s == 0, "DH must be multiple of s");
    static_assert(LK%t == 0, "LK must be multiple of t");
    static_assert(s==t);

    // Outer loop over columns of K (horizontal movement in the output S)
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_RANGE(colB, colB)
    for(unsigned j = 0; j < colB; j++) {

        // Calculate the base pointer for the current column block of K
        // Each block of B (K) is MMUL::size_B. There are colA such blocks per column.
        T_in* k_block_base = K_buffer + (j * colA * MMUL::size_B);

        // Inner loop over rows of Q (vertical movement in the output S)
        AIE_PREPARE_FOR_PIPELINING
        AIE_LOOP_RANGE(rowA, rowA)
        for(unsigned z = 0; z < rowA; z++) {

            // S_buffer indexing: [row_idx * total_cols + col_idx] * block_size
            T_out *__restrict pC = S_buffer + ((z * colB) + j) * MMUL::size_C;

            // Q_buffer indexing
            const T_in *__restrict pA = Q_buffer + (z * colA) * MMUL::size_A;

            // K_buffer (pB) indexing relative to our j-loop base
            const T_in *__restrict pB = k_block_base;

            MMUL C(aie::zeros<T_out, MMUL::size_C>());

            // Reduction loop over DH (the shared dimension)
            AIE_PREPARE_FOR_PIPELINING
            AIE_LOOP_RANGE(colA, colA)
            for(unsigned i = 0; i < colA; i++) AIE_LOOP_FLATTEN {
                aie::vector<T_in, MMUL::size_A> A = aie::load_v<MMUL::size_A>(pA);
                pA += MMUL::size_A;

                aie::vector<T_in, MMUL::size_B> B = aie::load_v<MMUL::size_B>(pB);
                B = aie::transpose(B, s, t);
                pB += MMUL::size_B;

                C.mac(A, B);
            }

            // Store result
            aie::store_v(pC, C.template to_vector<T_out>());
        }
    }
}













template<typename T_in, typename T_out, unsigned LQ, unsigned DH, unsigned LK,
    unsigned r, unsigned s, unsigned t>
void window_attention_matmul_q_k_transpose_mmul(
    T_in* __restrict Q_buffer, // Q is size of LQ x DH in row block major
    T_in* __restrict K_buffer, // K is size of LK x DH in row block major, Thus, need to transpose inside
    T_out* __restrict S_buffer
){
    // Use the 2x2-unrolled fast path when both output tile dimensions are even;
    // fall back to the plain loop otherwise.
    constexpr unsigned rowA = LQ / r;
    constexpr unsigned colB = LK / t;
    if constexpr (rowA % 2 == 0 && colB % 2 == 0) {
        window_attention_matmul_q_k_transpose_mmul_manual_unroll_two<T_in, T_out, LQ, DH, LK, r, s, t>(
            Q_buffer, K_buffer, S_buffer);
    } else {
        window_attention_matmul_q_k_transpose_mmul_normal_loop<T_in, T_out, LQ, DH, LK, r, s, t>(
            Q_buffer, K_buffer, S_buffer);
    }
}



template<typename T_in, typename T_out, unsigned LQ, unsigned DH, unsigned LK,
    unsigned r, unsigned s, unsigned t>
void window_attention_matmul_pij_v_manual_unroll_two(
    T_in* pij_buffer, // LQxLK buffer in rxs block row major
    T_in*  V_buffer,  // LKxDH buffer in sxt block row major
    T_out* output_buffer
){
    constexpr unsigned rowA = LQ / r;
    constexpr unsigned colB = DH / t;
    constexpr unsigned common = LK / s;

    using MMUL = aie::mmul<r, s, t, T_in, T_in, accauto>;

    // Static assertions for safety
    static_assert(rowA % 2 == 0, "rowA must be multiple of 2 for 2x2 tiling");
    static_assert(LQ % r == 0 && LK % s == 0 && DH % t == 0, "Dimensions must align with tiling factors");

    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_RANGE(rowA / 2, rowA / 2)
    for(unsigned z = 0; z < rowA; z += 2) {

        unsigned j = 0;
        // Main loop: Process 2x2 blocks of output (2 rows of A, 2 columns of B)
        AIE_PREPARE_FOR_PIPELINING
        AIE_LOOP_RANGE(colB / 2, colB / 2)
        for(; j + 1 < colB; j += 2) {

            // Output pointers (C): z and z+1
            T_out * pC1 = output_buffer + (z * colB + j) * MMUL::size_C;
            T_out * pC2 = pC1 + (colB * MMUL::size_C);

            // Matrix A pointers: Row z and Row z+1
            const T_in * pA1 = pij_buffer + (z * common) * MMUL::size_A;
            const T_in * pA2 = pA1 + (common * MMUL::size_A);

            // Matrix B pointers: Column j and Column j+1
            // We start at the top of the matrix for these specific columns
            const T_in * pB1 = V_buffer + (j * MMUL::size_B);
            const T_in * pB2 = pB1 + MMUL::size_B;

            MMUL C00(aie::zeros<T_out, MMUL::size_C>());
            MMUL C01(aie::zeros<T_out, MMUL::size_C>());
            MMUL C10(aie::zeros<T_out, MMUL::size_C>());
            MMUL C11(aie::zeros<T_out, MMUL::size_C>());
            AIE_PREPARE_FOR_PIPELINING
            AIE_LOOP_RANGE(common, common)
            for(unsigned i = 0; i < common; i++) AIE_LOOP_FLATTEN {
                auto vA0 = aie::load_v<MMUL::size_A>(pA1); pA1 += MMUL::size_A;
                auto vA1 = aie::load_v<MMUL::size_A>(pA2); pA2 += MMUL::size_A;

                auto vB0 = aie::load_v<MMUL::size_B>(pB1);
                auto vB1 = aie::load_v<MMUL::size_B>(pB2);

                // Move pB to the next row of blocks in Matrix B
                pB1 += (colB * MMUL::size_B);
                pB2 += (colB * MMUL::size_B);

                C00.mac(vA0, vB0); C01.mac(vA0, vB1);
                C10.mac(vA1, vB0); C11.mac(vA1, vB1);
            }

            aie::store_v(pC1, C00.template to_vector<T_out>());
            aie::store_v(pC1 + MMUL::size_C, C01.template to_vector<T_out>());
            aie::store_v(pC2, C10.template to_vector<T_out>());
            aie::store_v(pC2 + MMUL::size_C, C11.template to_vector<T_out>());
        }

        // Remainder Handling: If colB is odd, process the last column of blocks for rows z and z+1
        if (j < colB) {
            T_out * pC1 = output_buffer + (z * colB + j) * MMUL::size_C;
            T_out * pC2 = pC1 + (colB * MMUL::size_C);

            const T_in * pA1 = pij_buffer + (z * common) * MMUL::size_A;
            const T_in * pA2 = pA1 + (common * MMUL::size_A);
            const T_in * pB1 = V_buffer + (j * MMUL::size_B);

            MMUL C00(aie::zeros<T_out, MMUL::size_C>());
            MMUL C10(aie::zeros<T_out, MMUL::size_C>());

            AIE_PREPARE_FOR_PIPELINING
            AIE_LOOP_RANGE(common, common)
            for(unsigned i = 0; i < common; i++) {
                auto vA0 = aie::load_v<MMUL::size_A>(pA1); pA1 += MMUL::size_A;
                auto vA1 = aie::load_v<MMUL::size_A>(pA2); pA2 += MMUL::size_A;
                auto vB0 = aie::load_v<MMUL::size_B>(pB1);
                pB1 += (colB * MMUL::size_B);

                C00.mac(vA0, vB0);
                C10.mac(vA1, vB0);
            }
            aie::store_v(pC1, C00.template to_vector<T_out>());
            aie::store_v(pC2, C10.template to_vector<T_out>());
        }
    }
}


template<typename T_in, typename T_out, unsigned LQ, unsigned DH, unsigned LK,
    unsigned r, unsigned s, unsigned t>
void window_attention_matmul_pij_v_normal_loop(
    T_in* pij_buffer, // LQxLK buffer in rxs block row major
    T_in*  V_buffer,  // LKxDH buffer in sxt block row major
    T_out* output_buffer
){
    constexpr unsigned rowA = LQ / r;
    constexpr unsigned colB = DH / t;
    constexpr unsigned common = LK / s;

    using MMUL = aie::mmul<r, s, t, T_in, T_in, accauto>;
    static_assert(LQ % r == 0 && LK % s == 0 && DH % t == 0, "Dimensions must align with tiling factors");

    // Outer loop over rows of A
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_RANGE(rowA, rowA)
    for(unsigned z = 0; z < rowA; z++) {

        // Inner loop over columns of B
        AIE_PREPARE_FOR_PIPELINING
        AIE_LOOP_RANGE(colB, colB)
        for(unsigned j = 0; j < colB; j++) {

            // Output pointer (C): row z, column j
            T_out * pC = output_buffer + (z * colB + j) * MMUL::size_C;

            // Matrix A pointer: Row z
            const T_in * pA = pij_buffer + (z * common) * MMUL::size_A;

            // Matrix B pointer: Column j, starting at the top
            const T_in * pB = V_buffer + (j * MMUL::size_B);

            MMUL C(aie::zeros<T_out, MMUL::size_C>());

            // Reduction loop over LK (the shared dimension)
            AIE_PREPARE_FOR_PIPELINING
            AIE_LOOP_RANGE(common, common)
            for(unsigned i = 0; i < common; i++) AIE_LOOP_FLATTEN {
                auto vA = aie::load_v<MMUL::size_A>(pA); pA += MMUL::size_A;
                auto vB = aie::load_v<MMUL::size_B>(pB);

                // Move pB to the next row of blocks in Matrix B
                pB += (colB * MMUL::size_B);

                C.mac(vA, vB);
            }

            aie::store_v(pC, C.template to_vector<T_out>());
        }
    }
}


template<typename T_in, typename T_out, unsigned LQ, unsigned DH, unsigned LK,
    unsigned r, unsigned s, unsigned t>
void window_attention_matmul_pij_v(
    T_in* pij_buffer, // LQxLK buffer in rxs block row major
    T_in*  V_buffer,  // LKxDH buffer in sxt block row major
    T_out* output_buffer
){
    // Use the 2x2-unrolled fast path when rowA is even (colB remainder already handled inside);
    // fall back to the plain loop otherwise.
    constexpr unsigned rowA = LQ / r;
    if constexpr (rowA % 2 == 0) {
        window_attention_matmul_pij_v_manual_unroll_two<T_in, T_out, LQ, DH, LK, r, s, t>(
            pij_buffer, V_buffer, output_buffer);
    } else {
        window_attention_matmul_pij_v_normal_loop<T_in, T_out, LQ, DH, LK, r, s, t>(
            pij_buffer, V_buffer, output_buffer);
    }
}



// does the pij div l_i, or pij/sum(np.exp( x- max(x)))
template<int m, int k, int r, int t>
void pij_normalize(

    bfloat16* __restrict O_buffer, // size of m x k with rxt block in row block major
    bfloat16* __restrict L_i_buffer // size of mx1

){

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
        bfloat16 * cur_o = O_buffer + i*k;
 
        aie::accum<accfloat, accum_vec_size> o;
        AIE_PREPARE_FOR_PIPELINING
        AIE_LOOP_RANGE(total_iter, total_iter)
        for(int idx = 0; idx < total_iter; idx++){
            o.from_vector(aie::load_v<accum_vec_size>(cur_o));
            aie::accum<accfloat,accum_vec_size> res = aie::mul(o.template to_vector<bfloat16>(), L_i_broadcast_vec);
            aie::store_v(cur_o, res.template to_vector<bfloat16>());
            cur_o += accum_vec_size;
        }


    }


}



#endif