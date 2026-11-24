// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#include "aie_array_layout.h"
#include "aie_kernel_utils.h"
#include "typedef.hpp"
#include "mvm_col_major.hpp"
#include "zero.h"
#include "nonLinearFunction.h"
#include "sort.hpp"
#include "common_external.hpp"
#include "packet_control_gen.hpp"
#include "router_math.hpp"
#include "router_block.hpp"

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
    for (AIE_LOCK_LOOP_CTR word = 0; word < 2; ++word) {
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
    const uint32_t buffer_fetch_size_in_byte, const uint32_t address_offset_in_byte
){
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
    constexpr int shimtile_configure_ready_lock  = 10;   // expert BDs written, safe to push expert tasks

    // handshake with the router helper (the tile below), both locks in this tile:
    // x copied, the helper may start / the helper's 64 logits are in local_top_k_score_vector
    constexpr int router_helper_x_ready_lock = 13;
    constexpr int router_helper_finished_lock = 14;
    void router(
        bf16* AIE_DMA_RESTRICT top_x_in,

        bf16* AIE_DMA_RESTRICT x_in, // a copy of top_x_in 

        bfloat16* AIE_DMA_RESTRICT router_w_block_ping,
        bfloat16* AIE_DMA_RESTRICT router_w_block_pong,   
        
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
        int *router_weight_ping_pong_flag,
        const float* expert_bias,     // NUM_EXPERTS fp32, an RTP buffer the host writes each call
        const int32_t* layer_rtp      // [0]: which layer this call is (0 .. MAX_LAYERS-1), RTP

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
        // the helper may start on experts 64..127 (its own weight stream)
        _lock_release(router_helper_x_ready_lock);

        // experts 0..63, from this tile's weight stream (IT5 channel 0)
        for(AIE_LOCK_LOOP_CTR i = 0; i < NUM_EXPERTS / 2; i+= router_weight_row_block   ){
            float* res_ptr = local_top_k_score_vector + i;

            for(AIE_LOCK_LOOP_CTR j = 0; j < MODEL_DIM; j+= router_weight_col_block ){
                _lock_acquire(router_weight_con_lock);
                bfloat16* block = *router_weight_ping_pong_flag == 0 ? router_w_block_ping : router_w_block_pong;
                router_block_matvec(block, x_in + j, res_ptr);
                _lock_release(router_weight_prod_lock);
                *router_weight_ping_pong_flag = 1 - *router_weight_ping_pong_flag;
            }
        }

        // the helper's experts 64..127 are in too
        _lock_acquire(router_helper_finished_lock);



        // Hy routing, all fp32 (see router_math.hpp): sigmoid scores, top 8 of
        // score + bias, weights = score / sum * ROUTER_SCALING. The weights go
        // straight into the GLU's buffer (the tile above).
        router_math::route_sigmoid_topk<NUM_EXPERTS, NUM_ACT_EXPERTS>(
            _top_k_score_vector,
            top_k_exp_index_vector,
            local_top_k_score_vector,
            expert_bias,
            ROUTER_SCALING
        );


        
        // The host sets this after writing this call's expert BDs and pushing its
        // last shared-expert weight task on shims 0, 1, 6, 7 (expert tasks pushed
        // earlier would queue in front of them). Reading the base address before
        // this, while the shared expert streamed (MoE step 2, T3), hung every call.
        _lock_acquire(shimtile_configure_ready_lock);

        // The weight buffer's device address, read from the BD the host points at
        // expert 0. Each layer has its own weight buffer, so one saved address per
        // layer: read on the first call of that layer, reused afterwards (this core
        // keeps running between calls, so the table survives; XRT buffers do not move).
        static uint32_t base_table[MAX_LAYERS][2];
        AIE_PERSIST_ZERO static bool base_known[MAX_LAYERS] = {};
        const int layer = layer_rtp[0];
        if (!base_known[layer]) {
            get_SHM_bd_virtual_address(
                base_table[layer],
                control_packet_out_buf, control_packet_in_buf,
                control_packet_out_prod_lock_id, control_packet_out_con_lock_id,
                control_packet_in_prod_lock_id, control_packet_in_con_lock_id,
                CT_to_shimtile_control_bd_id, CT_from_shimtile_control_bd_id,
                CT_to_shimtile_control_bd_port, CT_from_shimtile_control_bd_port,
                SHIM_6_CONTROL_PKT, SHIM_6_CONTROL_RES_PKT,  // BIG assume, all MVM shm share same bo buffer
                0, SHIM_X_expert_BD_ID
            );
            base_known[layer] = true;
        }
        SHIM_BD_1_VIRTUAL_ptr[0] = base_table[layer][0];
        SHIM_BD_1_VIRTUAL_ptr[1] = base_table[layer][1];

        // Each round: point one BD per channel of shim 0, 1, 6, 7 at one chosen
        // expert each (one expert per proj column). The host stores every expert
        // as [channel 0 stream | channel 1 stream], each stream its gate/up blocks
        // then its down blocks in read order, so one linear BD per channel covers
        // the expert (16 BD patches per token instead of 32).
        //
        // All rounds are pushed at once, after the GLU's first request: each
        // channel's queue then holds round 0's task and round 1's behind it, so
        // round 1's weights stream as soon as round 0's are through (no wait for
        // the GLU's round-1 request; at most 3 tasks per queue, with the last
        // shared-down one). The GLU still asks once per round; those requests
        // are taken after the loop to keep the lock balanced.
        constexpr int rounds = NUM_ACT_EXPERTS / NUM_MVM_COLUMN;
        _up_lock_acquire(glu_request_router_send_lock);
        for(int round = 0; round < rounds; round++){
            for(int i = 0; i <NUM_MVM_COLUMN; i++ ){
                const uint32_t expert_offset = top_k_exp_index_vector[i + round*NUM_MVM_COLUMN] * Q4_K_EXPERT_BYTE;
                for(uint32_t ch = 0; ch < 2; ch++){
                    retrieveData<false>(
                        control_packet_out_buf, control_packet_in_buf,
                        control_packet_out_prod_lock_id, control_packet_out_con_lock_id,
                        control_packet_in_prod_lock_id, control_packet_in_con_lock_id,
                        CT_to_shimtile_control_bd_id, CT_from_shimtile_control_bd_id,
                        CT_to_shimtile_control_bd_port, CT_from_shimtile_control_bd_port,
                        SHIM_x_control_packet_id_list[i],
                        ch, (round == 0 ? SHIM_X_expert_BD_ID : Round_2_SHIM_X_expert_BD_ID) + ch,
                        SHIM_BD_1_VIRTUAL_ptr[0], SHIM_BD_1_VIRTUAL_ptr[1],
                        Q4_K_EXPERT_BYTE_PER_CHANNEL, expert_offset + ch * Q4_K_EXPERT_BYTE_PER_CHANNEL
                    );
                }
            }
        }
        for(AIE_LOCK_LOOP_CTR round = 1; round < rounds; round++){
            _up_lock_acquire(glu_request_router_send_lock);
        }

        _up_lock_acquire(x_in_con_lock);
        _up_lock_release(x_in_prod_lock);
  

    }



}