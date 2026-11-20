// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "common.h"

#include <aie_api/aie.hpp>

#include "sort.hpp"
#include "mvm_bf16.hpp"
#include "common_external.hpp"

#include "lut_based_ops.h"
#include "gptossParam.hpp"



#include "packet_control_gen.hpp"

void get_SHM_bd_virtual_address(
    // TODO: improve this, right now assume the lock is being added +48 already
    uint32_t * SHIM_BD_VIRTUAL_ptr,
    uint32_t * control_packet_out_buf, uint32_t* control_packet_in_buf,
    const uint32_t control_packet_out_prod_lock_id, const uint32_t control_packet_out_con_lock_id,
    const uint32_t control_packet_in_prod_lock_id, const uint32_t control_packet_in_con_lock_id,
    
    // configure of this CT, aka bd_id and port of this CT
    const uint32_t CT_to_shimtile_control_bd_id, const uint32_t CT_from_shimtile_control_bd_id,
    const uint32_t CT_to_shimtile_control_bd_port, const uint32_t CT_from_shimtile_control_bd_port,

    // related to packet_flow routing
    const uint32_t SHIM_control_packet_id, const uint32_t SHIM_0_control_response_id,

    const uint32_t SHM_MM2S_channel_idx, // 0 or 1
    const uint32_t SHM_bd_id

){

    // first, get the virtual base address in BD_id0
    
    //configure dmab_bd_id to send only the 
    // control_packet_header read 

    configure_BD_x_MM2S_y_dma_bd_len(
        CT_to_shimtile_control_bd_id,
        CT_to_shimtile_control_bd_port, 2, 0); // packet-flow header + control packet

    // Shim control reads return one register value per response packet. Keep
    // the established one-word response DMA and read X1 and X2 sequentially.
    for (uint32_t word = 0; word < 2; ++word) {
        acquire_greater_equal(control_packet_out_prod_lock_id, 1);
        *control_packet_out_buf = packet_flow_gen(
            0, // 0 since is from CT?
            SHIM_control_packet_id
        );
        *(control_packet_out_buf + 1) = control_packet_gen(
            SHIM_0_control_response_id, 1, 0,
            get_SHM_bd_x_0_address(SHM_bd_id) + 0x4 + word * 0x4
        );
        release(control_packet_out_con_lock_id, 1);

        acquire_greater_equal(control_packet_in_con_lock_id, 1);
        SHIM_BD_VIRTUAL_ptr[word] = *control_packet_in_buf;
        release(control_packet_in_prod_lock_id, 1);
    }
}
template<bool issue_token>
void retrieveData(

    // TODO: improve this, right now assume the lock is being added +48 already
    uint32_t * control_packet_out_buf, uint32_t* control_packet_in_buf,
    const uint32_t control_packet_out_prod_lock_id, const uint32_t control_packet_out_con_lock_id,
    const uint32_t control_packet_in_prod_lock_id, const uint32_t control_packet_in_con_lock_id,
    
    // configure of this CT, aka bd_id and port of this CT
    const uint32_t CT_to_shimtile_control_bd_id, const uint32_t CT_from_shimtile_control_bd_id,
    const uint32_t CT_to_shimtile_control_bd_port, const uint32_t CT_from_shimtile_control_bd_port,

    // related to packet_flow routing
    const uint32_t SHIM_control_packet_id,

    const uint32_t SHM_MM2S_channel_idx, // 0 or 1
    const uint32_t SHM_bd_id,

    const uint32_t shim_bd_id_1_val,
    const uint32_t shim_bd_id_2_val,
    // specific to the stuff we send
    // const uint32_t matrix_size_per_expert_in_bytes, const uint32_t expert_to_fetch, const uint32_t constant_offset_in_byte
    // //constant_offset_in_byte could due to different matrix type, like gate, up, down
    const uint32_t buffer_fetch_size_in_byte, const uint32_t address_offset_in_byte
){


    // const uint32_t address_offset =expert_to_fetch * matrix_size_per_expert_in_bytes+ constant_offset_in_byte; 
    control_Shimtile_transfer_with_packetflow_gen<issue_token>(
        shim_bd_id_1_val, shim_bd_id_2_val,
        buffer_fetch_size_in_byte/4, address_offset_in_byte,  // div 4 due to 4 byte granduality

        control_packet_out_prod_lock_id, control_packet_out_con_lock_id,
        control_packet_out_buf, 

        CT_to_shimtile_control_bd_id, CT_to_shimtile_control_bd_port,
        SHIM_control_packet_id,
        
        SHM_MM2S_channel_idx, SHM_bd_id
    );


}


extern "C" {




    void select_top_k_with_index(
        bfloat16* top_k_data, //top_k_data buffer allocate multiple of 16(even if K_select< 16)
        
        bfloat16* vector_in,  //vector_in  allocate multliple of 16 (even if size <16)
        int32_t* index,   //Index buffer allocate multliple of 16 (even if K_select <16)
 
        uint32_t *SHIM_BD_1_VIRTUAL_ptr,
        // control packet related
        uint32_t * control_packet_out_buf, uint32_t* control_packet_in_buf,
        const uint32_t control_packet_out_prod_lock_id, const uint32_t control_packet_out_con_lock_id,
        const uint32_t control_packet_in_prod_lock_id, const uint32_t control_packet_in_con_lock_id,
        
        // configure of this CT, aka bd_id and port of this CT
        const uint32_t CT_to_shimtile_control_bd_id, const uint32_t CT_from_shimtile_control_bd_id,
        const uint32_t CT_to_shimtile_control_bd_port, const uint32_t CT_from_shimtile_control_bd_port,
        uint32_t request_down_project_lock,

        // buffer out to MOE

        bfloat16* down_projection_res_ptr, bfloat16* _mlp_out_buffer

    ) {
        // NOTE: parameterize this?
        constexpr uint32_t down_proj_res_prod_lock = 0;
        constexpr uint32_t down_proj_res_con_lock = 1;
        constexpr uint32_t mlp_out_prod_lock = 2;
        constexpr uint32_t mlp_out_con_lock = 3;

        _up_lock_acquire(M_SCORE_CON_LOCK, 1);

        // do not change data_ping_pong_flag value, only read
        bfloat16* cur_vector_ptr =vector_in;

        bubble_sort_largest_to_smallest_bf16_with_index<NUM_OF_EXPERT>(cur_vector_ptr, index);
        static uint32_t  SHIM_x_control_packet_id_list[4] = {SHIM_2_CONTROL_PKT, SHIM_3_CONTROL_PKT, SHIM_4_CONTROL_PKT, SHIM_5_CONTROL_PKT};
        static uint32_t  SHIM_x_control_respose_id_list[1] = { SHIM_5_CONTROL_RES_PKT};

        // At this point, request the experts we need 
        {
                    //NOTE: need to be same with runtime sequence
                // RUNTIME sequence will set this lock to enable core to access its registers
                _lock_acquire(15, 1 );

                // at this point, start sending out the control packet to shimtiles
                // send out for MVM weight


                //assume set by

                get_SHM_bd_virtual_address(
                    SHIM_BD_1_VIRTUAL_ptr,
                    control_packet_out_buf, control_packet_in_buf,
                    control_packet_out_prod_lock_id, control_packet_out_con_lock_id,
                    control_packet_in_prod_lock_id, control_packet_in_con_lock_id,
                    CT_to_shimtile_control_bd_id, CT_from_shimtile_control_bd_id,
                    CT_to_shimtile_control_bd_port, CT_from_shimtile_control_bd_port,
                    SHIM_5_CONTROL_PKT, SHIM_5_CONTROL_RES_PKT,  // BIG assume, all MVM shm share same bo buffer
                    0, SHIM_X_gateup_BD_ID
                );
                
   
        
                for(int i = 0; i < TOP_K; i++){
                    int expert_to_fetch = index[i];
                    // now request both gate, up, down
                    retrieveData<false>(
                        control_packet_out_buf, control_packet_in_buf,
                        control_packet_out_prod_lock_id, control_packet_out_con_lock_id,
                        control_packet_in_prod_lock_id, control_packet_in_con_lock_id,
                        CT_to_shimtile_control_bd_id, CT_from_shimtile_control_bd_id,
                        CT_to_shimtile_control_bd_port, CT_from_shimtile_control_bd_port,
                        SHIM_x_control_packet_id_list[i], 
                        0, SHIM_X_gateup_BD_ID,
                        SHIM_BD_1_VIRTUAL_ptr[0], SHIM_BD_1_VIRTUAL_ptr[1],
                        GATE_UP_WEIGHT_BYTE_PER_EXPERT/2, expert_to_fetch * (GATE_UP_WEIGHT_BYTE_PER_EXPERT+DOWN_WEIGHT_BYTE_PER_EXPERT)     //div4 for 4 byte granduality
                    );

                    retrieveData<false>(
                        control_packet_out_buf, control_packet_in_buf,
                        control_packet_out_prod_lock_id, control_packet_out_con_lock_id,
                        control_packet_in_prod_lock_id, control_packet_in_con_lock_id,
                        CT_to_shimtile_control_bd_id, CT_from_shimtile_control_bd_id,
                        CT_to_shimtile_control_bd_port, CT_from_shimtile_control_bd_port,
                        SHIM_x_control_packet_id_list[i], 
                        1, SHIM_X2_gateup_BD_ID,
                        SHIM_BD_1_VIRTUAL_ptr[0], SHIM_BD_1_VIRTUAL_ptr[1],
                        GATE_UP_WEIGHT_BYTE_PER_EXPERT/2, expert_to_fetch * (GATE_UP_WEIGHT_BYTE_PER_EXPERT+DOWN_WEIGHT_BYTE_PER_EXPERT) + 2*Q4NX_BLOCK_BYTE_SIZE    //div4 for 4 byte granduality
                    );                    
                }


        }




        float exp_sum = 0.0;
        constexpr uint32_t k_select_ceil_16_multiple = (TOP_K + 16 -1)/16 *16;

        AIE_LOOP_UNROLL_FULL
        for(int i = 0; i <k_select_ceil_16_multiple; i+= 16 ){
            aie::vector<bfloat16, 16> top_16_vec = aie::load_v<16>(cur_vector_ptr+ i);
            bf16 max_value_vec = aie::reduce_max(top_16_vec);
            aie::vector<bf16, 16> sub_value = aie::sub(top_16_vec, max_value_vec);
            
            constexpr int min_clamp = -87.0f;
            constexpr int max_clamp = 88.0f;
            aie::vector<bf16, 16> min_value = aie::broadcast<bf16, 16>(min_clamp);
            aie::vector<bf16, 16> max_value = aie::broadcast<bf16, 16>(max_clamp);
        
            aie::vector<bf16, 16> top_16_vec_clamped = aie::clamp(sub_value, min_value, max_value);

            aie::accum<accfloat, 16> exp_res_acc = getExpBf16(top_16_vec_clamped);

            aie::store_v((top_k_data + i), exp_res_acc.template to_vector<bfloat16>());

            AIE_LOOP_UNROLL_FULL
            for(int l = 0; l < 16 && (i + l) < TOP_K; l++){
#if defined(__chess__)
                exp_sum += top_k_data[i + l];
#else
                exp_sum = AIE_FADD(exp_sum, (float)top_k_data[i + l]);
#endif
            }

        }
        // normalize
        const float inv_exp_sum = aie::inv(exp_sum);
#if defined(__chess__)
        AIE_LOOP_UNROLL_FULL
        for(int i = 0; i <TOP_K; i++ ){
            top_k_data[i]  = top_k_data[i] * inv_exp_sum;

        }
#else
        // The scale is bf16 x bf16 -> bf16 on the vector unit, which is what
        // chess generates for bfloat16 * float.
        aie::accum<accfloat, 16> inv_acc;
        inv_acc.from_vector(aie::broadcast<float, 16>(inv_exp_sum));
        const aie::vector<bf16, 16> inv_bf16 = inv_acc.template to_vector<bf16>();
        AIE_LOOP_UNROLL_FULL
        for(int i = 0; i < k_select_ceil_16_multiple; i += 16){
            aie::vector<bf16, 16> w = aie::mul(aie::load_v<16>(top_k_data + i), inv_bf16).template to_vector<bf16>();
            AIE_LOOP_UNROLL_FULL
            for(int l = 0; l < 16 && (i + l) < TOP_K; l++){
                top_k_data[i + l] = w[l];
            }
        }
#endif


        _down_lock_acquire(request_down_project_lock, 1);
        for(int i = 0; i < TOP_K; i++){
            int expert_to_fetch = index[i];
            // now request both gate, up, down
            retrieveData<false>(
                control_packet_out_buf, control_packet_in_buf,
                control_packet_out_prod_lock_id, control_packet_out_con_lock_id,
                control_packet_in_prod_lock_id, control_packet_in_con_lock_id,
                CT_to_shimtile_control_bd_id, CT_from_shimtile_control_bd_id,
                CT_to_shimtile_control_bd_port, CT_from_shimtile_control_bd_port,
                SHIM_x_control_packet_id_list[i], 
                0, SHIM_X_down_BD_ID,
                SHIM_BD_1_VIRTUAL_ptr[0], SHIM_BD_1_VIRTUAL_ptr[1],
                DOWN_WEIGHT_BYTE_PER_EXPERT/2, expert_to_fetch * (GATE_UP_WEIGHT_BYTE_PER_EXPERT+DOWN_WEIGHT_BYTE_PER_EXPERT) +GATE_UP_WEIGHT_BYTE_PER_EXPERT      //div4 for 4 byte granduality
            );

            retrieveData<false>(
                control_packet_out_buf, control_packet_in_buf,
                control_packet_out_prod_lock_id, control_packet_out_con_lock_id,
                control_packet_in_prod_lock_id, control_packet_in_con_lock_id,
                CT_to_shimtile_control_bd_id, CT_from_shimtile_control_bd_id,
                CT_to_shimtile_control_bd_port, CT_from_shimtile_control_bd_port,
                SHIM_x_control_packet_id_list[i], 
                1, SHIM_X2_down_BD_ID,
                SHIM_BD_1_VIRTUAL_ptr[0], SHIM_BD_1_VIRTUAL_ptr[1],
                DOWN_WEIGHT_BYTE_PER_EXPERT/2, expert_to_fetch * (GATE_UP_WEIGHT_BYTE_PER_EXPERT+DOWN_WEIGHT_BYTE_PER_EXPERT)+ GATE_UP_WEIGHT_BYTE_PER_EXPERT + 2*Q4NX_BLOCK_BYTE_SIZE    //div4 for 4 byte granduality
            );                    
        }



        _lock_acquire(down_proj_res_con_lock, 1); // lock belong to this CT
        _lock_acquire(mlp_out_prod_lock, 1);

        constexpr uint32_t vec_len  = 16;
        static_assert(MVM_INTERLEAVE_SIZE%vec_len == 0);

        bfloat16* mlp_out_ptr = (_mlp_out_buffer);
        
        AIE_PREPARE_FOR_PIPELINING
        for(int i = 0; i<  TOP_K*D_PADDED_FOUR_Q4NX_ROW; i+= TOP_K*MVM_INTERLEAVE_SIZE){

            AIE_PREPARE_FOR_PIPELINING
            AIE_LOOP_RANGE(MVM_INTERLEAVE_SIZE/vec_len, MVM_INTERLEAVE_SIZE/vec_len )
            for(int l = 0; l <MVM_INTERLEAVE_SIZE; l+=vec_len){
                aie::accum<accfloat, vec_len> acc_res = aie::zeros<accfloat, vec_len>();

                AIE_LOOP_UNROLL_FULL
                for(int k = 0; k < TOP_K; k++){
                    // The TOP_K expert down projection value are stacked interleved at MVM_INTERLEAVE_SIZE granduality
                    bfloat16* exp_ptr = down_projection_res_ptr + (k*MVM_INTERLEAVE_SIZE) +l + i;
                    aie::vector<bfloat16, vec_len> expert_down_val = aie::load_v<vec_len>(exp_ptr);
                    acc_res = aie::mac(acc_res,  top_k_data[k], expert_down_val );
                }


                aie::store_v(mlp_out_ptr, acc_res.template to_vector<bfloat16>());
                mlp_out_ptr += vec_len;
            }

        }

        _lock_release(mlp_out_con_lock, 1);
       _lock_release(down_proj_res_prod_lock, 1);
        
        _up_lock_release(M_SCORE_PROD_LOCK, 1);


    }

    void MVM_bf16_block_32_64_16_with_bias_slave(
        bfloat16* mat_in_ping, bfloat16* mat_in_pong,  
        bfloat16* vec_in,  // assume each buffer is store as 1xtotal_mat_col size already
        float* result_acc_ptr, 
        int32_t *input_ping_pong_flag // slave need to know which buffer of matrix_block to use
    ){
        constexpr int BLOCK_ROWS = EXPERT_BLOCK_ROWS;
        constexpr int BLOCK_COLS = EXPERT_BLOCK_COLS;
        constexpr int TILE_ROWS = EXPERT_BLOCK_TILE_ROWS;
        static_assert(BLOCK_ROWS%TILE_ROWS == 0);

        AIE_PREPARE_FOR_PIPELINING
        AIE_LOOP_RANGE(NUM_OF_EXPERT/BLOCK_ROWS, NUM_OF_EXPERT/BLOCK_ROWS )
        for(int row_idx = 0; row_idx< NUM_OF_EXPERT; row_idx += BLOCK_ROWS){

            AIE_PREPARE_FOR_PIPELINING
            AIE_LOOP_RANGE(D/BLOCK_COLS, D/BLOCK_COLS )
            for(int col_idx = 0; col_idx < D; col_idx += BLOCK_COLS){
                bfloat16* cur_mat_ptr;
                
         
                _up_lock_acquire(M_Prod_S_Con_LOCK);
                volatile int32_t current_input_ping_pong = *input_ping_pong_flag;             
                if(current_input_ping_pong == 0){
                    cur_mat_ptr  = mat_in_ping;
                }else{
                    cur_mat_ptr  = mat_in_pong;
                }                

                // here, release a lock to slave to signal the slave can process
                matvec_vectorized_Bf16_col_major<BLOCK_ROWS/2, BLOCK_COLS, TILE_ROWS>(
                    cur_mat_ptr + (BLOCK_ROWS/2)*BLOCK_COLS, // different for slave
                    vec_in+col_idx,
                    result_acc_ptr +BLOCK_ROWS/2     // different for slave
                );
                // here wait for slave to finish
                _up_lock_release(M_Con_S_Prod_LOCK);
            } // end of for col_idx
        }
    }




}