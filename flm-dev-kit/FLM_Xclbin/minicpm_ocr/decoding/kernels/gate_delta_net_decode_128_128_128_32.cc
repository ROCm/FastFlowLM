// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>
#include "model_spec.h"
#include "aie_kernel_utils.h"
#include "aie_array_layout.h"
#include "bf16_limb_math.hpp"

constexpr int CT_S_IN_prod_lock = 0;
constexpr int CT_S_IN_con_lock = 1;

constexpr int CT_Q_prod_lock = 2;
constexpr int CT_Q_con_lock = 3;

constexpr int CT_K_prod_lock = 4;
constexpr int CT_K_con_lock = 5;

constexpr int CT_V_prod_lock = 6;
constexpr int CT_V_con_lock = 7;

constexpr int CT_S_O_out_prod_lock = 8;

constexpr int CT_S_O_out_con_lock = 11;

template<typename T>
T* determine_ping_pong_buffer(
    T* ping_buffer, T* pong_buffer,
    int* ping_pong_flag
){
    if(*ping_pong_flag == 0){
        *ping_pong_flag=1;
        return ping_buffer;
    }
    else{
        *ping_pong_flag=0;
        return pong_buffer;
    }

}

extern "C" {

    //TODO: for now
    constexpr int dK = L_Q_HEAD_DIM;
    constexpr int dV = L_V_HEAD_DIM;
    constexpr int dQ = L_Q_HEAD_DIM;
    constexpr int V_chunk = dV / 4;

    ///TODO: have the lock id sync with python version
    // dq, dk, dv, chunk_size
    void gate_delta_net_decode_128_128_128_32(
        float * S_in_chunk_ping, float * S_in_chunk_pong,
        float * S_out_chunk,// size of V_chunk x dK  in col-major
        bfloat16* Q_ping, bfloat16* Q_pong, // size of dQ + 2, 2 is padding in this case
        bfloat16* K_ping, bfloat16* K_pong, // size of dK +2, 2 for a scalar of float of alpha
        bfloat16* V_ping, bfloat16* V_pong, // size of dV +2, 2 for a scalar of float of beta
        bfloat16* out_vec, // size of V_chunk,

        int chunk_idx, // which chunk of V to process, assume chunk_idx <= dV/V_chunk
        int* ping_pong_flag// size of 16 ints, 0 for ping, 1 for pong
    ){

        _lock_acquire(CT_S_IN_con_lock, 1);
        _lock_acquire(CT_K_con_lock, 1);
        _lock_acquire(CT_V_con_lock, 1);
        _lock_acquire(CT_Q_con_lock, 1);

        bfloat16* cur_Q_buffer = determine_ping_pong_buffer(Q_ping, Q_pong, &ping_pong_flag[1]);
        bfloat16* cur_K_buffer = determine_ping_pong_buffer(K_ping, K_pong, &ping_pong_flag[2]);
        bfloat16* cur_V_buffer = determine_ping_pong_buffer(V_ping, V_pong, &ping_pong_flag[3]);
        float* cur_S_in_buffer =determine_ping_pong_buffer(S_in_chunk_ping, S_in_chunk_pong, &ping_pong_flag[4]);
        bfloat16* cur_out_buffer = out_vec;

        float* alpha_ptr = (float*)(cur_K_buffer + dK); // the last 2 bfloat16 is alpha
        float* beta_ptr = (float*)(cur_V_buffer + dV); // the last 2 bfloat16 is beta

        constexpr int vec_len = 32;
        aie::vector<float, vec_len> alpha_vec = aie::broadcast<float, vec_len>(*alpha_ptr);
        aie::vector<float, vec_len> beta_vec = aie::broadcast<float, vec_len>(*beta_ptr);

        static_assert(V_chunk%vec_len == 0, "V_chunk must be divisible by vec_len for this simple implementation");

        aie::accum<accfloat, vec_len> intermediate_acc[V_chunk/vec_len];
        aie::accum<accfloat, vec_len> o_acc[V_chunk/vec_len];
        AIE_PREPARE_FOR_PIPELINING
        AIE_LOOP_UNROLL_FULL
        for(int row_idx = 0; row_idx < V_chunk; row_idx+= vec_len)AIE_LOOP_FLATTEN{
            intermediate_acc[ row_idx/vec_len] = aie::zeros<accfloat, vec_len>();
            o_acc[row_idx/vec_len] = aie::zeros<accfloat, vec_len>();
        }

        // S@K. K is read from a bf16 buffer, so it is exactly representable in
        // bf16; only S is fp32. Splitting S into 2 bf16 limbs lets the MACs run
        // on the native bf16 MMUL datapath instead of the emulated-fp32 path.
        // 2 limbs is enough here: the result feeds u, which is scaled by beta
        // and rounded into a bf16-precision path, so the third limb is discarded
        // downstream anyway.
        aie::accum<accfloat, vec_len> ia_1[V_chunk/vec_len];
        AIE_LOOP_UNROLL_FULL
        for(int r = 0; r < V_chunk; r += vec_len) AIE_LOOP_FLATTEN{
            ia_1[r/vec_len] = aie::zeros<accfloat, vec_len>();
        }
        AIE_PREPARE_FOR_PIPELINING
        AIE_LOOP_RANGE(dK, dK)
        for(int col = 0; col < dK; col++){
            aie::vector<bfloat16, vec_len> k_bf = aie::broadcast<bfloat16, vec_len>(cur_K_buffer[col]);
            AIE_LOOP_UNROLL_FULL
            for(int row = 0; row < V_chunk; row+= vec_len) AIE_LOOP_FLATTEN{
                aie::vector<float, vec_len> res = aie::load_v<vec_len>(cur_S_in_buffer + col*V_chunk + row);
                // split-accumulator form: one MAC of loop-carried depth per limb
                bf16_limb::mac_f32_bf16_low(intermediate_acc[row/vec_len], ia_1[row/vec_len], res, k_bf);
            }
        }
        AIE_LOOP_UNROLL_FULL
        for(int r = 0; r < V_chunk; r += vec_len) AIE_LOOP_FLATTEN{
            intermediate_acc[r/vec_len] = bf16_limb::reduce(intermediate_acc[r/vec_len], ia_1[r/vec_len]);
        }

        // now, intermediate_acc has the value of S@K,

        // u = beta*(v - alpha*S@K) is algebraically beta*v - (beta*alpha)*S@K.
        // Written that way, beta*alpha collapses to a single scalar multiply
        // outside the vector unit, and beta*v becomes f32 x bf16 (v is loaded
        // from a bf16 buffer, so it is exact) at 3 MACs instead of a full
        // emulated fp32 multiply. Both scalars are loop-invariant, so their
        // limbs are built once here.
        aie::vector<float, vec_len> beta_alpha_vec =
#if defined(__chess__)
            aie::broadcast<float, vec_len>((*beta_ptr) * (*alpha_ptr));
#else
            aie::broadcast<float, vec_len>(AIE_FMUL(*beta_ptr, *alpha_ptr));  // round like chess
#endif
        const bf16_limb::limbs3<vec_len> beta_l       = bf16_limb::split_f32_3(beta_vec);
        const bf16_limb::limbs3<vec_len> beta_alpha_l = bf16_limb::split_f32_3(beta_alpha_vec);
        AIE_PREPARE_FOR_PIPELINING
        AIE_LOOP_UNROLL_FULL
        for(int row = 0; row < V_chunk; row+= vec_len) AIE_LOOP_FLATTEN{

            aie::vector<bfloat16, vec_len> v_vec = aie::load_v<vec_len>(cur_V_buffer + chunk_idx * V_chunk + row);

            // beta * v : f32 x bf16, 3 MACs, fp32-exact
            aie::accum<accfloat, vec_len> u = aie::zeros<accfloat, vec_len>();
            u = bf16_limb::mac_limbs_bf16(u, beta_l, v_vec);
            // -= (beta*alpha) * S@K : both fp32, 6 MACs via the triangular form
            u = bf16_limb::msc_limbs3_f32_tri(
                    u, beta_alpha_l, intermediate_acc[row/vec_len].template to_vector<float>());
            intermediate_acc[row/vec_len] = u;

        }

        // now: intermediate_acc holds beta * (v - alpha* S@K)

        // K comes from a bf16 buffer, so K*intermediate is f32 x bf16: only the
        // fp32 side (intermediate) is split, into 2 bf16 limbs -> 2 MACs. The
        // limbs are built here, outside the col loop, because that is where the
        // row reduction ends -- the scheduler would hoist the split either way.
        aie::vector<bfloat16, vec_len> inter_l0[V_chunk/vec_len], inter_l1[V_chunk/vec_len];
        AIE_LOOP_UNROLL_FULL
        for(int r = 0; r < V_chunk; r += vec_len) AIE_LOOP_FLATTEN{
            auto l = bf16_limb::split_f32(intermediate_acc[r/vec_len].template to_vector<float>());
            inter_l0[r/vec_len] = l.first;
            inter_l1[r/vec_len] = l.second;
        }
        // alpha is a scalar broadcast, so its limbs are the same for all dK*V_chunk
        // products below. The reason to split explicitly is the product set, not
        // the split itself: aie::mul(S, alpha) issues all 9 cross products into a
        // serial VADD tree, where the triangular form keeps only the 6 that can
        // land above the fp32 mantissa floor, through a MAC chain.
        const bf16_limb::limbs3<vec_len> alpha_l = bf16_limb::split_f32_3(alpha_vec);
        AIE_PREPARE_FOR_PIPELINING
        AIE_LOOP_RANGE(dK, dK)
        for(int col = 0; col < dK; col++ ){
            bfloat16 q_scalar = cur_Q_buffer[col];
            aie::vector<bfloat16, vec_len> k_bf = aie::broadcast<bfloat16, vec_len>(cur_K_buffer[col]);
            AIE_LOOP_UNROLL_FULL
            for(int row = 0; row < V_chunk; row += vec_len) AIE_LOOP_FLATTEN{
                aie::vector<float, vec_len> S_res = aie::load_v<vec_len>(cur_S_in_buffer + col*V_chunk + row);
                // K is exact in bf16, so u*K is f32 x bf16: 2 MACs, no emulation.
                aie::accum<accfloat, vec_len> nS = aie::zeros<accfloat, vec_len>();
                nS = bf16_limb::mac_limbs_bf16(nS, inter_l0[row/vec_len], inter_l1[row/vec_len], k_bf);
                // alpha * S: 6 MACs, fp32-exact. The decay compounds over the
                // sequence, so the 3-MAC form is not safe here -- but the 3 terms
                // this drops are all below 2^-24 of the result, so it loses nothing
                // the 9-MAC vendor path would have kept.
                nS = bf16_limb::mac_limbs3_f32_tri(nS, alpha_l, S_res);
                aie::store_v(S_out_chunk + col*V_chunk + row, nS.template to_vector<float>());
                o_acc[row/vec_len] = aie::mac(o_acc[row/vec_len], q_scalar, nS.template to_vector<bfloat16>());
            }
        }

        _lock_release(CT_K_prod_lock, 1);
        _lock_release(CT_V_prod_lock, 1);
        _lock_release(CT_Q_prod_lock, 1);
        _lock_release(CT_S_IN_prod_lock, 1);

        _lock_acquire(CT_S_O_out_prod_lock,1);

        // store o buffer; o = S_t @ Q was accumulated in the loop above
        AIE_PREPARE_FOR_PIPELINING
        AIE_LOOP_UNROLL_FULL
        for(int row_idx = 0; row_idx < V_chunk/vec_len; row_idx ++) AIE_LOOP_FLATTEN{
            aie::store_v(cur_out_buffer + row_idx*vec_len, o_acc[row_idx].template to_vector<bfloat16>());
        }

        _lock_release(CT_S_O_out_con_lock, 1);
    }
} // extern "C"