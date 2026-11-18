// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include "aie_array_layout.h"
#include "aie_kernel_utils.h"
#include "typedef.hpp"
#include "mvm_col_major.hpp"
#include "zero.h"
#include "nonLinearFunction.h"
#include "sort.hpp"
#include "common_external.hpp"
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
    //TODO: have this from function parameter
    // this is from glu (CT above it)
    constexpr int x_in_prod_lock = 6;
    constexpr int x_in_con_lock = 7;
    constexpr int glu_request_router_send_lock = 12;


    constexpr int router_weight_prod_lock = 2;
    constexpr int router_weight_con_lock = 3;
    constexpr int shimtile_configure_ready_lock  = 10;

    constexpr int router_helper_prod_lock = 11;
    constexpr int router_helper_con_lock = 13;
    constexpr int router_helper_finished_lock = 14;
    void router(
        bf16* restrict top_x_in,

        bf16* restrict x_in, // a copy of top_x_in 

        bfloat16* restrict router_w_block_ping,
        bfloat16* restrict router_w_block_pong,   
        
        float* _top_k_score_vector,   float* local_top_k_score_vector,

        int32_t* top_k_exp_index_vector,
        float * share_expert_output,

             
        uint32_t *SHIM_BD_1_VIRTUAL_ptr,
        // control packet related
        uint32_t * control_packet_out_buf, uint32_t* control_packet_in_buf,
        const uint32_t control_packet_out_prod_lock_id, const uint32_t control_packet_out_con_lock_id,
        const uint32_t control_packet_in_prod_lock_id, const uint32_t control_packet_in_con_lock_id,
        
        // configure of this CT, aka bd_id and port of this CT
        const uint32_t CT_to_shimtile_control_bd_id, const uint32_t CT_from_shimtile_control_bd_id,
        const uint32_t CT_to_shimtile_control_bd_port, const uint32_t CT_from_shimtile_control_bd_port,
        int *router_weight_ping_pong_flag

    ){
        

        static uint32_t  SHIM_x_control_packet_id_list[4] = {SHIM_0_CONTROL_PKT, SHIM_1_CONTROL_PKT, SHIM_6_CONTROL_PKT, SHIM_7_CONTROL_PKT};
        static uint32_t  SHIM_x_control_respose_id_list[1] = { SHIM_6_CONTROL_RES_PKT};



        zero_vectorized_512<float, NUM_EXPERTS, 1>(
            local_top_k_score_vector
        );        

        _up_lock_acquire(x_in_con_lock);
        copy_bf16_to_bf16<MODEL_DIM>(
            x_in, top_x_in
        );
        _up_lock_release(x_in_prod_lock);


        for(int i = 0; i < NUM_EXPERTS; i+= router_weight_row_block   ){
            float* res_ptr = local_top_k_score_vector + i;

            
            for(int j = 0; j < MODEL_DIM; j+= router_weight_col_block ){

                _lock_acquire(router_weight_con_lock);
   
                _lock_acquire(router_helper_prod_lock);
                _lock_release(router_helper_con_lock);

                if(*router_weight_ping_pong_flag== 0){
                    matvec_vectorized_col_major<
                        router_weight_row_block/2, router_weight_col_block, router_weight_row_block/2,
                        bfloat16, bfloat16, float,
                        router_weight_row_block
                    >(
                        router_w_block_ping, x_in+j, res_ptr 
                    );
                }else{
                    matvec_vectorized_col_major<
                        router_weight_row_block/2, router_weight_col_block, router_weight_row_block/2,
                        bfloat16, bfloat16, float,
                        router_weight_row_block
                    >(
                        router_w_block_pong, x_in+j, res_ptr 
                    );
                }


                // wait for  finish signal from 
                _lock_acquire(router_helper_finished_lock);
                _lock_release(router_weight_prod_lock);
        
                
                if(*router_weight_ping_pong_flag == 0){
                    *router_weight_ping_pong_flag=1;
                }else{
                    *router_weight_ping_pong_flag=0;
                }
            }
        }



        // Select on the raw float32 logits, not on a bfloat16 softmax of them.
        // softmax is monotone, so it never changed the ranking -- all it did was
        // round 256 logits to bfloat16 first, and top-8-of-256 turns that
        // rounding into a discrete expert swap whenever two logits land inside
        // one ulp of each other (0.031 at |z| = 4, 0.063 at |z| = 8).  Ties in
        // partial_select_largest break toward the lower index, so the collapse
        // is a systematic bias toward low-numbered experts, not noise that
        // averages out.  The exponential is then only needed for the 8 survivors
        // and is done in float32 -- see softmax_topk_f32.
        partial_select_largest<float, NUM_EXPERTS, NUM_ACT_EXPERTS>(
            local_top_k_score_vector,
            top_k_exp_index_vector
        );

        softmax_topk_f32<NUM_ACT_EXPERTS>(
            _top_k_score_vector,
            local_top_k_score_vector
        );


        
        // acquire a sync lock from runtimesequence where runtimesequence tell it finished configuration and is ready

        _lock_acquire(shimtile_configure_ready_lock);

        get_SHM_bd_virtual_address(
            SHIM_BD_1_VIRTUAL_ptr,
            control_packet_out_buf, control_packet_in_buf,
            control_packet_out_prod_lock_id, control_packet_out_con_lock_id,
            control_packet_in_prod_lock_id, control_packet_in_con_lock_id,
            CT_to_shimtile_control_bd_id, CT_from_shimtile_control_bd_id,
            CT_to_shimtile_control_bd_port, CT_from_shimtile_control_bd_port,
            SHIM_6_CONTROL_PKT, SHIM_6_CONTROL_RES_PKT,  // BIG assume, all MVM shm share same bo buffer
            0, SHIM_X_gateup_BD_ID
        );
                


        //TODO: firs acquire a address
        for(int round = 0; round <  NUM_ACT_EXPERTS/NUM_MVM_COLUMN; round++){
            _up_lock_acquire(glu_request_router_send_lock);


            for(int i = 0; i <NUM_MVM_COLUMN; i++ ){
                int expert_to_fetch = top_k_exp_index_vector[i + round*NUM_MVM_COLUMN];
                 //int expert_to_fetch = -1;
  
         
                   // now request both gate, up
                    retrieveData<false>(
                        control_packet_out_buf, control_packet_in_buf,
                        control_packet_out_prod_lock_id, control_packet_out_con_lock_id,
                        control_packet_in_prod_lock_id, control_packet_in_con_lock_id,
                        CT_to_shimtile_control_bd_id, CT_from_shimtile_control_bd_id,
                        CT_to_shimtile_control_bd_port, CT_from_shimtile_control_bd_port,
                        SHIM_x_control_packet_id_list[i], 
                        0,round == 0?  SHIM_X_gateup_BD_ID:Round_2_SHIM_X_gateup_BD_ID ,
                        SHIM_BD_1_VIRTUAL_ptr[0], SHIM_BD_1_VIRTUAL_ptr[1],
                       Q4_K_GATE_UP_WEIGHT_BYTE_PER_EXPERT/2,expert_to_fetch * (Q4_K_GATE_UP_WEIGHT_BYTE_PER_EXPERT)     //div4 for 4 byte granduality
                    );

                 //int expert_to_fetch = -1;

                    retrieveData<false>(
                        control_packet_out_buf, control_packet_in_buf,
                        control_packet_out_prod_lock_id, control_packet_out_con_lock_id,
                        control_packet_in_prod_lock_id, control_packet_in_con_lock_id,
                        CT_to_shimtile_control_bd_id, CT_from_shimtile_control_bd_id,
                        CT_to_shimtile_control_bd_port, CT_from_shimtile_control_bd_port,
                        SHIM_x_control_packet_id_list[i], 
                        1, round == 0?  SHIM_X2_gateup_BD_ID: Round_2_SHIM_X2_gateup_BD_ID,
                        SHIM_BD_1_VIRTUAL_ptr[0], SHIM_BD_1_VIRTUAL_ptr[1],
                        Q4_K_GATE_UP_WEIGHT_BYTE_PER_EXPERT/2, expert_to_fetch * (Q4_K_GATE_UP_WEIGHT_BYTE_PER_EXPERT) + 2*Q4_K_block_byte    //div4 for 4 byte granduality
                    );    
            }




            for(int i = 0; i <NUM_MVM_COLUMN; i++ ){
                int expert_to_fetch = top_k_exp_index_vector[i + round*NUM_MVM_COLUMN];
    
             
                retrieveData<false>(
                    control_packet_out_buf, control_packet_in_buf,
                    control_packet_out_prod_lock_id, control_packet_out_con_lock_id,
                    control_packet_in_prod_lock_id, control_packet_in_con_lock_id,
                    CT_to_shimtile_control_bd_id, CT_from_shimtile_control_bd_id,
                    CT_to_shimtile_control_bd_port, CT_from_shimtile_control_bd_port,
                    SHIM_x_control_packet_id_list[i], 
                    0, round==0 ? SHIM_X_down_BD_ID: Round_2_SHIM_X_down_BD_ID,
                    SHIM_BD_1_VIRTUAL_ptr[0], SHIM_BD_1_VIRTUAL_ptr[1],
                    Q4_K_DOWN_WEIGHT_BYTE_PER_EXPERT/2,  expert_to_fetch *Q4_K_DOWN_WEIGHT_BYTE_PER_EXPERT + NUM_EXPERTS*Q4_K_GATE_UP_WEIGHT_BYTE_PER_EXPERT       //div4 for 4 byte granduality
                );


                retrieveData<false>(
                    control_packet_out_buf, control_packet_in_buf,
                    control_packet_out_prod_lock_id, control_packet_out_con_lock_id,
                    control_packet_in_prod_lock_id, control_packet_in_con_lock_id,
                    CT_to_shimtile_control_bd_id, CT_from_shimtile_control_bd_id,
                    CT_to_shimtile_control_bd_port, CT_from_shimtile_control_bd_port,
                    SHIM_x_control_packet_id_list[i], 
                    1, round==0 ? SHIM_X2_down_BD_ID: Round_2_SHIM_X2_down_BD_ID,
                    SHIM_BD_1_VIRTUAL_ptr[0], SHIM_BD_1_VIRTUAL_ptr[1],
                    Q4_K_DOWN_WEIGHT_BYTE_PER_EXPERT/2, expert_to_fetch *Q4_K_DOWN_WEIGHT_BYTE_PER_EXPERT + 2*Q4_K_block_byte + NUM_EXPERTS*Q4_K_GATE_UP_WEIGHT_BYTE_PER_EXPERT //div4 for 4 byte granduality
                );    
            }            
        }
    

  

        _up_lock_acquire(x_in_con_lock);
        _up_lock_release(x_in_prod_lock);
  

    }



}