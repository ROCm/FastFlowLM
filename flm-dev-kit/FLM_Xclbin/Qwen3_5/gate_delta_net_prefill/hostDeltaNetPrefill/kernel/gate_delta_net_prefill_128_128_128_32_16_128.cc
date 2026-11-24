// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

#include "common_external.hpp"
#include "lock_util.hpp"
#include "zero.h"
#include "passThroughCommon.hpp"
//TODO: align with python or pass in
#include "aie_kernel_utils.h" //TODO: a better method?
#include "bf16_limb_math.hpp"

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

constexpr float csqrt(float x)
{
    float r = x;
    for (int i = 0; i < 10; ++i)
        r = 0.5f * (r + x / r);
    return r;
}

template <int vec_len, int DQ, int DK, int DV, int V_chunk, bfloat16 inv_sqrt_DH>
void  single_step_gate_delta_net(
    float* alpha_ptr, float* beta_ptr,
    bfloat16* __restrict__ cur_Q_buffer, bfloat16* __restrict__ cur_K_buffer, bfloat16* __restrict__ cur_V_buffer,
    float* S_in_out_chunk, // Note: S_in_chunk and S_out_chunk are in col-major with size of DK x V_chunk
    // The two ptr could point to the same buffer, so don't use __restrict__ here
    bfloat16* __restrict__ cur_out_buffer, // size of V_chunk
    int chunk_idx
){

    aie::vector<float, vec_len> alpha_vec = aie::broadcast<float, vec_len>(*alpha_ptr);
    aie::vector<float, vec_len> beta_vec = aie::broadcast<float, vec_len>(*beta_ptr);

    static_assert(V_chunk%vec_len == 0, "V_chunk must be divisible by vec_len for this simple implementation");

    aie::accum<accfloat, vec_len> intermediate_acc[V_chunk/vec_len];
    aie::accum<accfloat, vec_len> o_acc[V_chunk/vec_len];
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_UNROLL_FULL
    for(int row_idx = 0; row_idx < V_chunk/vec_len; row_idx++)AIE_LOOP_FLATTEN{
        intermediate_acc[ row_idx] = aie::zeros<accfloat, vec_len>();
        o_acc[row_idx] = aie::zeros<accfloat, vec_len>();
    }

    // S@K. K is read from a bf16 buffer, so it is exactly representable in
    // bf16; only S is fp32. Splitting S into 2 bf16 limbs lets the MACs run on
    // the native bf16 MMUL datapath instead of the emulated-fp32 path. 2 limbs
    // is enough here: the result feeds u, which is scaled by beta and rounded
    // into a bf16-precision path, so the third limb is discarded downstream.
    aie::accum<accfloat, vec_len> ia_1[V_chunk/vec_len];
    AIE_LOOP_UNROLL_FULL
    for(int r = 0; r < V_chunk/vec_len; r++) AIE_LOOP_FLATTEN{
        ia_1[r] = aie::zeros<accfloat, vec_len>();
    }
#if defined(__chess__)
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_RANGE(DK, DK)
    for(int col = 0; col < DK; col++){
        aie::vector<bfloat16, vec_len> k_bf = aie::broadcast<bfloat16, vec_len>(cur_K_buffer[col]);

        AIE_LOOP_UNROLL_FULL
        for(int row_idx = 0; row_idx < V_chunk/vec_len; row_idx++) AIE_LOOP_FLATTEN{
            aie::vector<float, vec_len> res = aie::load_v<vec_len>(S_in_out_chunk + col*V_chunk + row_idx*vec_len);
            // split-accumulator form: one MAC of loop-carried depth per limb
            bf16_limb::mac_f32_bf16_low(intermediate_acc[row_idx], ia_1[row_idx], res, k_bf);
        }

    }
#else
    // Walking pointers, and K[col] loaded one column ahead: under Peano both the
    // col * V_chunk address math and the scalar load latency otherwise sit on
    // this loop's critical path. The last column is peeled so the prefetch never
    // reads past K.
    {
        const float *s_it = S_in_out_chunk;
        const bfloat16 *k_it = cur_K_buffer;
        bfloat16 k_s = *k_it++;
        AIE_LOOP_RANGE(DK - 1, DK - 1)
        for(int col = 0; col < DK - 1; col++){
            aie::vector<bfloat16, vec_len> k_bf = aie::broadcast<bfloat16, vec_len>(k_s);
            k_s = *k_it++;
            AIE_LOOP_UNROLL_FULL
            for(int row_idx = 0; row_idx < V_chunk/vec_len; row_idx++){
                aie::vector<float, vec_len> res = aie::load_v<vec_len>(s_it + row_idx*vec_len);
                bf16_limb::mac_f32_bf16_low(intermediate_acc[row_idx], ia_1[row_idx], res, k_bf);
            }
            s_it += V_chunk;
        }
        aie::vector<bfloat16, vec_len> k_bf = aie::broadcast<bfloat16, vec_len>(k_s);
        AIE_LOOP_UNROLL_FULL
        for(int row_idx = 0; row_idx < V_chunk/vec_len; row_idx++){
            aie::vector<float, vec_len> res = aie::load_v<vec_len>(s_it + row_idx*vec_len);
            bf16_limb::mac_f32_bf16_low(intermediate_acc[row_idx], ia_1[row_idx], res, k_bf);
        }
    }
#endif
    AIE_LOOP_UNROLL_FULL
    for(int r = 0; r < V_chunk/vec_len; r++) AIE_LOOP_FLATTEN{
        intermediate_acc[r] = bf16_limb::reduce(intermediate_acc[r], ia_1[r]);
    }

    // u = beta*(v - alpha*S@K) is algebraically beta*v - (beta*alpha)*S@K.
    // Written that way, beta*alpha collapses to a single scalar multiply outside
    // the vector unit, and beta*v becomes f32 x bf16 (v is loaded from a bf16
    // buffer, so it is exact) at 3 MACs instead of a full emulated fp32 multiply.
    // Both scalars are loop-invariant, so their limbs are built once here.
    aie::vector<float, vec_len> beta_alpha_vec =
        aie::broadcast<float, vec_len>((*beta_ptr) * (*alpha_ptr));
    const bf16_limb::limbs3<vec_len> beta_l       = bf16_limb::split_f32_3(beta_vec);
    const bf16_limb::limbs3<vec_len> beta_alpha_l = bf16_limb::split_f32_3(beta_alpha_vec);
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_UNROLL_FULL
    for(int row_idx = 0; row_idx < V_chunk/vec_len; row_idx++) AIE_LOOP_FLATTEN{

        aie::vector<bfloat16, vec_len> v_vec = aie::load_v<vec_len>(cur_V_buffer + chunk_idx * V_chunk + row_idx*vec_len);

        // beta * v : f32 x bf16, 3 MACs, fp32-exact
        aie::accum<accfloat, vec_len> u = aie::zeros<accfloat, vec_len>();
        u = bf16_limb::mac_limbs_bf16(u, beta_l, v_vec);
        // -= (beta*alpha) * S@K : both fp32, 6 MACs via the triangular form
        u = bf16_limb::msc_limbs3_f32_tri(
                u, beta_alpha_l, intermediate_acc[row_idx].template to_vector<float>());
        intermediate_acc[row_idx] = u;

    }

    // K comes from a bf16 buffer, so K*intermediate is f32 x bf16: only the
    // fp32 side (intermediate) is split, into 2 bf16 limbs -> 2 MACs. The limbs
    // are built here, outside the col loop, because that is where the row
    // reduction ends -- the scheduler would hoist the split either way.
    aie::vector<bfloat16, vec_len> inter_l0[V_chunk/vec_len], inter_l1[V_chunk/vec_len];
    AIE_LOOP_UNROLL_FULL
    for(int r = 0; r < V_chunk/vec_len; r++) AIE_LOOP_FLATTEN{
        auto l = bf16_limb::split_f32(intermediate_acc[r].template to_vector<float>());
        inter_l0[r] = l.first;
        inter_l1[r] = l.second;
    }
    // alpha is a scalar broadcast, so its limbs are the same for all DK*V_chunk
    // products below. The reason to split explicitly is the product set, not the
    // split itself: aie::mul(S, alpha) issues all 9 cross products into a serial
    // VADD tree, where the triangular form keeps only the 6 that can land above
    // the fp32 mantissa floor, through a MAC chain.
    const bf16_limb::limbs3<vec_len> alpha_l = bf16_limb::split_f32_3(alpha_vec);
#if defined(__chess__)
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_RANGE(DK, DK) // although it's a loop, we only have 128 iteration with static bound, so it's basically unrolled
    for(int col = 0; col < DK; col++ ){
        bfloat16 q_scalar = cur_Q_buffer[col] * inv_sqrt_DH; // scale the q by 1/sqrt(DH
        aie::vector<bfloat16, vec_len> k_bf = aie::broadcast<bfloat16, vec_len>(cur_K_buffer[col]);

        AIE_LOOP_UNROLL_FULL
        for(int row_idx = 0; row_idx < V_chunk/vec_len; row_idx++) AIE_LOOP_FLATTEN{
            int S_offset =  col*V_chunk + row_idx*vec_len;

            aie::vector<float, vec_len> S_res = aie::load_v<vec_len>(S_in_out_chunk +S_offset);

            // K is exact in bf16, so u*K is f32 x bf16: 2 MACs, no emulation.
            aie::accum<accfloat, vec_len> new_S_acc = aie::zeros<accfloat, vec_len>();
            new_S_acc = bf16_limb::mac_limbs_bf16(new_S_acc, inter_l0[row_idx], inter_l1[row_idx], k_bf);
            // alpha * S: 6 MACs, fp32-exact. The decay compounds over the sequence,
            // so the 3-MAC form is not safe here -- but the 3 terms this drops are
            // all below 2^-24 of the result, so it loses nothing the 9-MAC vendor
            // path would have kept.
            new_S_acc = bf16_limb::mac_limbs3_f32_tri(new_S_acc, alpha_l, S_res);

            // store new S_value for next iteration
            aie::store_v( S_in_out_chunk + S_offset, new_S_acc.template to_vector<float>());

            // at the same time, also calculate the output for this chunk
            o_acc[row_idx] = aie::mac(o_acc[row_idx],   q_scalar,new_S_acc.template to_vector<bfloat16>());
        }
    }

#else
    // Same update with walking pointers and Q[col] / K[col] loaded one column
    // ahead, as in the S@K loop above; the last column is peeled.
    {
        float *s_it = S_in_out_chunk;
        const bfloat16 *q_it = cur_Q_buffer;
        const bfloat16 *k_it = cur_K_buffer;
        bfloat16 q_s = *q_it++ * inv_sqrt_DH;
        bfloat16 k_s = *k_it++;
        AIE_LOOP_RANGE(DK - 1, DK - 1)
        for(int col = 0; col < DK - 1; col++){
            const bfloat16 q_cur = q_s;
            aie::vector<bfloat16, vec_len> k_bf = aie::broadcast<bfloat16, vec_len>(k_s);
            q_s = *q_it++ * inv_sqrt_DH;
            k_s = *k_it++;
            AIE_LOOP_UNROLL_FULL
            for(int row_idx = 0; row_idx < V_chunk/vec_len; row_idx++){
                aie::vector<float, vec_len> S_res = aie::load_v<vec_len>(s_it + row_idx*vec_len);
                aie::accum<accfloat, vec_len> new_S_acc = aie::zeros<accfloat, vec_len>();
                new_S_acc = bf16_limb::mac_limbs_bf16(new_S_acc, inter_l0[row_idx], inter_l1[row_idx], k_bf);
                new_S_acc = bf16_limb::mac_limbs3_f32_tri(new_S_acc, alpha_l, S_res);
                aie::store_v(s_it + row_idx*vec_len, new_S_acc.template to_vector<float>());
                o_acc[row_idx] = aie::mac(o_acc[row_idx], q_cur, new_S_acc.template to_vector<bfloat16>());
            }
            s_it += V_chunk;
        }
        {
            aie::vector<bfloat16, vec_len> k_bf = aie::broadcast<bfloat16, vec_len>(k_s);
            AIE_LOOP_UNROLL_FULL
            for(int row_idx = 0; row_idx < V_chunk/vec_len; row_idx++){
                aie::vector<float, vec_len> S_res = aie::load_v<vec_len>(s_it + row_idx*vec_len);
                aie::accum<accfloat, vec_len> new_S_acc = aie::zeros<accfloat, vec_len>();
                new_S_acc = bf16_limb::mac_limbs_bf16(new_S_acc, inter_l0[row_idx], inter_l1[row_idx], k_bf);
                new_S_acc = bf16_limb::mac_limbs3_f32_tri(new_S_acc, alpha_l, S_res);
                aie::store_v(s_it + row_idx*vec_len, new_S_acc.template to_vector<float>());
                o_acc[row_idx] = aie::mac(o_acc[row_idx], q_s, new_S_acc.template to_vector<bfloat16>());
            }
            s_it += V_chunk;
        }
    }
#endif

    // store o buffer; o = S_t @ Q was accumulated in the loop above
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_UNROLL_FULL
    for(int row_idx = 0; row_idx < V_chunk/vec_len; row_idx ++) AIE_LOOP_FLATTEN{
        aie::store_v(cur_out_buffer + row_idx*vec_len, o_acc[row_idx].template to_vector<bfloat16>());
    }

}

extern "C" {

    //TODO: for now
    constexpr int DK = 128;
    constexpr int DV = 128;
    constexpr int DQ = 128;
    constexpr int V_chunk = 32;
    constexpr int CT_L_size = 16;
    constexpr int MT_L_size = 128;

    static constexpr bfloat16 inv_sqrt_DH = 1.0f / csqrt(static_cast<float>(DQ));

    ///TODO: have the lock id sync with python version
    // dq, dk, dv, chunk_size
    void gate_delta_net_prefill_128_128_128_32_16_128(
        float * S_in_chunk, // size of V_chunk x DK  in col-major
        bfloat16* Q_ping, bfloat16* Q_pong, // size of CT_L_sizex padded(DQ +2), 2 is padding in this case        Due to ptr alignment issuye, the padded(DQ+2) is being padded to padded_QKV_size
        bfloat16* K_ping, bfloat16* K_pong, // size of CT_L_sizex padded(DK +2), 2 for a scalar of float of alpha Due to ptr alignment issuye, the padded(DK+2) is being padded to padded_QKV_size
        bfloat16* V_ping, bfloat16* V_pong, // size of CT_L_sizex padded(DV +2), 2 for a scalar of float of beta  Due to ptr alignment issuye, the padded(DV+2) is being padded to padded_QKV_size
        bfloat16* out_ping, bfloat16* out_pong,// size of CT_L_size x V_chunk
        int padded_QKV_size,

        int chunk_idx, // which chunk of V to process, assume chunk_idx <= DV/V_chunk
        int* ping_pong_flag,// size of 32 ints, 0 for ping, 1 for pong
        int* rtp_buffer,
        int CT_S_IN_prod_lock_id, int CT_S_IN_con_lock_id,
        int CT_Q_prod_lock_id, int CT_Q_con_lock_id,
        int CT_K_prod_lock_id, int CT_K_con_lock_id,
        int CT_V_prod_lock_id, int CT_V_con_lock_id,
        int CT_output_prod_lock_id, int CT_output_con_lock_id
    ){

        // the out_ping/pong buffer is used to send both output and final S_out
        static_assert(  V_chunk* DK*sizeof(float) %   CT_L_size*V_chunk*sizeof(bfloat16) == 0);
        constexpr int iter_for_send_S =   (V_chunk* DK*sizeof(float) )/ ( CT_L_size*V_chunk*sizeof(bfloat16));

        // rtp_buffer[0] tell the number_of_head(S) this CT will process
        // rtp_buffer[1] tell the L size per number_of_head(S), L is not padded

        for(int head_idx = 0; head_idx < rtp_buffer[0]; head_idx++){

            // acquire the S for this head
            acquire_greater_equal(CT_S_IN_con_lock_id, 1 );
            // note: rtp_buffer[1] is not necessarily divisible by CT_L_size

            for(int MT_L_idx = 0; MT_L_idx< rtp_buffer[1]; MT_L_idx+= MT_L_size){

                // because data in MT is transaction in size of MT_L_size, thus we need to loop over the MT_L_size
                //

                for(int ct_l_chunk_idx = 0; ct_l_chunk_idx < MT_L_size; ct_l_chunk_idx+= CT_L_size){

                    int L_idx = MT_L_idx + ct_l_chunk_idx;

                    bfloat16* cur_Q_buffer = determine_ping_pong_buffer(Q_ping, Q_pong, &ping_pong_flag[1]);
                    bfloat16* cur_K_buffer = determine_ping_pong_buffer(K_ping, K_pong, &ping_pong_flag[2]);
                    bfloat16* cur_V_buffer = determine_ping_pong_buffer(V_ping, V_pong, &ping_pong_flag[3]);
                    bfloat16* cur_out_buffer = determine_ping_pong_buffer(out_ping, out_pong, &ping_pong_flag[4]);

                    int cur_iteration_size = (L_idx + CT_L_size <= rtp_buffer[1])? CT_L_size : (rtp_buffer[1] - L_idx);

                    // acquire q, k, v lock
                    // acquire o lock
                    acquire_greater_equal(CT_Q_con_lock_id, 1);
                    acquire_greater_equal(CT_K_con_lock_id, 1);
                    acquire_greater_equal(CT_V_con_lock_id, 1);
                    acquire_greater_equal(CT_output_prod_lock_id, 1);
                    for(int iteration_idx = 0; iteration_idx < cur_iteration_size; iteration_idx++ ){

                        // int Q_iteration_offset = iteration_idx * (DQ + 2); // plus 2 for alpha and beta
                        // int K_iteration_offset = iteration_idx * (DK + 2);
                        // int V_iteration_offset = iteration_idx * (DV + 2);

                        float* alpha_ptr = reinterpret_cast<float*> (cur_K_buffer  + DK); // the last 2 bfloat16 is alpha
                        float* beta_ptr =  reinterpret_cast<float*>(cur_V_buffer   + DV); // the last 2 bfloat16 is beta

                        single_step_gate_delta_net<32, DQ, DK, DV, V_chunk, inv_sqrt_DH>(
                            alpha_ptr, beta_ptr,
                            cur_Q_buffer, cur_K_buffer, cur_V_buffer,
                            S_in_chunk,
                            cur_out_buffer,
                            chunk_idx
                        );
                        // debug, copy Q out
                        // memcpy(cur_out_buffer, cur_Q_buffer, V_chunk*sizeof(bfloat16));
                        // for(int i = 0; i < V_chunk; i++){
                        //     cur_out_buffer[i] = iteration_idx*2.0f + chunk_idx*32.0f + i*1.0f;
                        // }
                        // for(int i = 0; i < V_chunk; i++){
                        //     cur_out_buffer[i] = cur_K_buffer[i];
                        // }
                        cur_Q_buffer += padded_QKV_size;//(DQ + 2);
                        cur_K_buffer += padded_QKV_size;//(DK + 2);
                        cur_V_buffer += padded_QKV_size;//(DV + 2);
                        cur_out_buffer += V_chunk;

                    }
                    // release q, k, v lock
                    //release o lock
                    release(CT_output_con_lock_id, 1);
                    release(CT_V_prod_lock_id,1);
                    release(CT_K_prod_lock_id,1);
                    release(CT_Q_prod_lock_id, 1);

                }

            }

            // for loop of acquire-release o lock for sending out the S for this head
            for(int i = 0; i < iter_for_send_S; i++){
                bfloat16* cur_out_buffer = determine_ping_pong_buffer(out_ping, out_pong, &ping_pong_flag[4]);
                constexpr int S_elements_per_iteration = (CT_L_size*V_chunk*sizeof(bfloat16)) / sizeof(float);
                acquire_greater_equal(CT_output_prod_lock_id, 1);
                copy_float_to_float< S_elements_per_iteration, 32>(
                    (float*)cur_out_buffer, S_in_chunk+ i*S_elements_per_iteration
                );
                release(CT_output_con_lock_id, 1);
            }

            // release the S lock for this head,
            release(CT_S_IN_prod_lock_id, 1);
        }

    }
} // ex