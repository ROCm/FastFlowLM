#ifndef __GATE_DELTA_NET_PREFILL_HPP__
#define  __GATE_DELTA_NET_PREFILL_HPP__
#include <cassert>
#include <algorithm> // Required for std::max
#include <string>
#include "npu_utils/npu_instr_utils.hpp"



inline npu_bd_id get_S_IN_bd_id(int ping_pong){
    if(ping_pong ==0){
        return  static_cast<npu_bd_id> (0);
    }else{
        return  static_cast<npu_bd_id> (8);
    }
}

inline npu_bd_id get_alpha_beta_bd_id(int ping_pong){
    if(ping_pong ==0){
        return  static_cast<npu_bd_id> (1);
    }else{
        return  static_cast<npu_bd_id> (9);
    }
}

inline npu_bd_id get_q_k_bd_id(int ping_pong){
    if(ping_pong ==0){
        return  static_cast<npu_bd_id> (3);
    }else{
        return  static_cast<npu_bd_id> (11);
    }
}

inline npu_bd_id get_v_bd_id(int ping_pong){
    if(ping_pong ==0){
        return  static_cast<npu_bd_id> (5);
    }else{
        return  static_cast<npu_bd_id> (13);
    }
}

inline  npu_bd_id get_output_o_s_bd_id(int ping_pong){
    if(ping_pong ==0){
        return  static_cast<npu_bd_id> (6);
    }else{
        return  static_cast<npu_bd_id> (14);
    }
}




inline void update_ping_pong_flag(int &flag){
    if(flag ==0){
        flag =1;
    }else{
        flag =0;
    }
}




inline void wait_DMA_queue_if_full(
    npu_sequence &seq,
    int cur_shim_idx, std::vector<int> &SHIM_queue_counter,
    npu_tiles cur_shimtile, dma_direction channel_direction, npu_it_channel it_channel,
    int max_queue_size = 2
){

    //NOTE: it will not wait if max_queue_size is not achieve
    if(SHIM_queue_counter.at(cur_shim_idx) > max_queue_size){
        std::cerr <<"SHIM DMA queue overflow"<< std::endl;
        exit(1);
    } 

    if(SHIM_queue_counter.at(cur_shim_idx) == max_queue_size){
        // need to wait
        seq.npu_dma_wait(
            cur_shimtile, channel_direction, it_channel
        );
        SHIM_queue_counter[cur_shim_idx]--;
    }
}



inline void force_wait_DMA_queue(
    npu_sequence &seq,
    int cur_shim_idx, std::vector<int> &SHIM_queue_counter,
    npu_tiles cur_shimtile, dma_direction channel_direction, npu_it_channel it_channel

){

    if (SHIM_queue_counter.at(cur_shim_idx) >0){
        // need to wait
        seq.npu_dma_wait(
            cur_shimtile, channel_direction, it_channel
        );
        SHIM_queue_counter[cur_shim_idx]--;
    }

}


inline void increment_DMA_queue(

    npu_sequence &seq,
    int cur_shim_idx, std::vector<int> &SHIM_queue_counter,
    int max_queue_size = 2
){

    if(SHIM_queue_counter.at(cur_shim_idx) < max_queue_size){
        SHIM_queue_counter[cur_shim_idx]++;
    }else{
        std::cerr <<"SHIM DMA queue overflow"<< std::endl;
        exit(1);    
    } 

}






void generate_gate_delta_net_prefill_sequence(npu_sequence &seq, 
    
    uint32_t NUM_OF_COLUMN, uint32_t NUM_CT_PER_COLUMN,
    
    
    uint32_t QK_HEAD_GROUP,
    uint32_t NUM_HEAD, // o, and v head
    uint32_t MT_L_size,
   
    uint32_t L, uint32_t L_PADDED, 
    uint32_t DQ, uint32_t DK, uint32_t DV,// DV==DO


    uint32_t L_start,
    uint32_t S_in_out_offset_fp32,
    uint32_t beta_start_offset_fp32

  

){


    //NOTE: assumptions
    // Assume S_in_out is shape of  NUM_HEAD x (DVx DK), where each DVxDK is in column major order

    // assume S is shape of  NUM_HEAD x (DVx DK), where each DVxDK is in column major order
    //if QKV_buffer_row_major: 
    // assume QKV_in is shape of L x  (NUM_Q_K_HEAD x DQ+ NUM_Q_K_HEAD x DK + NUM_HEADx DV), and the entire buffer is in row major oder
    //else
    //  QKV_in is shape of   NUM_Q_K_HEAD * (LxDQ) + NUM_Q_K_HEAD * (LxDK) + NUM_HEAD * (LxDV) in row major order, where each LxDQ, LxDK, LxDV is in column major order.
    
    // assume  o is  in shape of Lx (NUM_HEAD x DV), and each NUM_HEAD x DV is in column major order
    
    uint32_t NUM_Q_K_HEAD = NUM_HEAD / QK_HEAD_GROUP;
    uint32_t o_buffer_row_stride = NUM_HEAD * DV;
    uint32_t  alpha_beta_buffer_row_stride = NUM_HEAD;

    constexpr int CT_lock_address_base = 0x000001F000;
    constexpr int CT_rtp_address = 48128;
    const int Arg_S_in_out = 0;
    const int Arg_QKV_in = 1;
    const int Arg_alpha_beta_in = 2;
    const int Arg_O_out = 3;

    constexpr int CT_rtp_sync_lock_id = 12;
    
    seq.clear_cmds();




    //create list of tiles
    size_t shimtile_size = NUM_OF_COLUMN;
    std::vector<npu_tiles> shim_tiles;

    std::vector<int> s_in_BD_ping_pong(shimtile_size, 0);
    std::vector<int> alpha_beta_qkv_BD_ping_pong(shimtile_size, 0);
    std::vector<int> o_out_BD_ping_pong(shimtile_size, 0);


    std::vector<int> s_in_bd_queue_counter(shimtile_size, 0);
    std::vector<int> alpha_beta_qkv_in_bd_queue_counter(shimtile_size, 0);
    std::vector<int> o_out_bd_queue_counter(shimtile_size, 0);


    uint32_t DV_chunk_per_CT = DV/NUM_CT_PER_COLUMN;


    uint32_t QKV_in_offset_bf16 = L_start * (NUM_Q_K_HEAD *(DQ+DK) + NUM_HEAD*DV);
    uint32_t alpha_beta_in_offset_fp32 = L_start * NUM_HEAD; // each head has one alpha and one beta, and we store in fp32 for better precision, so times NUM_HEAD 
    //TOOD: take this assertion
    if(  NUM_HEAD % NUM_OF_COLUMN){
        std::cerr << "Currently only support NUM_HEAD that is divisible by NUM_OF_COLUMN, please change the configuration or implement the non-divisible case in the future. " << std::endl;
        exit(-1);
    }

    int HEAD_PER_COLUMN = NUM_HEAD / NUM_OF_COLUMN;
    
    for(size_t i = 0; i < shimtile_size; i++){
        shim_tiles.push_back((get_tile(0, i)));
    }
    
    


    // first, setup the rtp buffer and the rtp locks

    for(size_t row_idx = 0; row_idx < NUM_CT_PER_COLUMN; row_idx++){
        for(size_t col_idx = 0; col_idx< NUM_OF_COLUMN; col_idx++){
            auto CT_tile = get_tile(row_idx+2, col_idx);
            // set RTP value
            seq.rtp_write( CT_tile, CT_rtp_address, HEAD_PER_COLUMN );
            seq.rtp_write( CT_tile, CT_rtp_address+4, L );

            // set RTP lock
            seq.rtp_write(CT_tile, CT_lock_address_base+16*(CT_rtp_sync_lock_id), 1); // set lock to 1
        }
    }


    uint32_t num_bf16_for_float = sizeof(float)/sizeof(bf16);
    uint32_t MT_Output_size_bf16 = MT_L_size * DV;
    uint32_t NUM_MT_O_CHUNK_FOR_S_output = ((DK*DV)*sizeof(float)) /(MT_Output_size_bf16* sizeof(bf16)); 

    // pair of group_head_idx(Q, K), head_idx(V, O)
    std::vector<std::vector<std::pair<int, int>>> head_idx_per_transaction;


    std::vector<std::pair<int, int>> cur_transaction_head_idx;

    for(int h_idx = 0; h_idx < NUM_HEAD; h_idx += QK_HEAD_GROUP){
        for(int group_idx = 0; group_idx < QK_HEAD_GROUP; group_idx ++){

            int g_head_idx = h_idx/QK_HEAD_GROUP;
            int head_idx = h_idx + group_idx;
            
            if (cur_transaction_head_idx.size() == NUM_OF_COLUMN){
                head_idx_per_transaction.push_back(cur_transaction_head_idx);
                cur_transaction_head_idx.clear();
            }
            cur_transaction_head_idx.push_back({g_head_idx, head_idx});

        }


    }
    if(cur_transaction_head_idx.size() > 0){
        head_idx_per_transaction.push_back(cur_transaction_head_idx);
    }



    for(auto transaction_head_idx_list: head_idx_per_transaction){

        for(int cur_col_idx = 0; cur_col_idx < transaction_head_idx_list.size(); cur_col_idx ++){

            int g_head_idx = transaction_head_idx_list[cur_col_idx].first;
            int head_idx = transaction_head_idx_list[cur_col_idx].second;


            uint32_t S_offset = head_idx*  DK * DV +S_in_out_offset_fp32;

            wait_DMA_queue_if_full(
                seq, cur_col_idx, 
                s_in_bd_queue_counter,
                shim_tiles.at(cur_col_idx),
                MM2S,it_channel_0
            );

            seq.npu_dma_memcpy_nd(
                sizeof(float),
                Arg_S_in_out,
                MM2S,
                shim_tiles.at(cur_col_idx),
                static_cast<npu_bd_id>(get_S_IN_bd_id(s_in_BD_ping_pong.at(cur_col_idx))),
                it_channel_0,
                {0,0,0,S_offset},
                {1, 1, 1, (uint32_t)DK*DV},
                {0,0,0,1},
                -1, 0, true
            );
            update_ping_pong_flag(s_in_BD_ping_pong.at(cur_col_idx));
            increment_DMA_queue(
                seq, cur_col_idx, 
                s_in_bd_queue_counter
            );

            // // now, receive O from the shimtile
            // wait_DMA_queue_if_full(
            //     seq, cur_col_idx, 
            //     o_out_bd_queue_counter,
            //     shim_tiles.at(cur_col_idx),
            //     S2MM,it_channel_0
            // );
            // int O_offset = head_idx* DV + L_start * o_buffer_row_stride;
            // seq.npu_dma_memcpy_nd(
            //     sizeof(bf16),
            //     Arg_O_out,
            //     S2MM,
            //     shim_tiles.at(cur_col_idx),
            //     static_cast<npu_bd_id>(get_output_o_s_bd_id(o_out_BD_ping_pong.at(cur_col_idx))),
            //     it_channel_0,
            //     {0,0,0,(uint32_t)O_offset},
            //     {(uint32_t)L_PADDED/ MT_L_size, NUM_CT_PER_COLUMN, MT_L_size, DV_chunk_per_CT},
            //     {(uint32_t)MT_L_size*o_buffer_row_stride , DV_chunk_per_CT,   o_buffer_row_stride,1},
            //     -1, 0, true
            // );
            // update_ping_pong_flag(o_out_BD_ping_pong.at(cur_col_idx));
            // increment_DMA_queue(
            //     seq, cur_col_idx, 
            //     o_out_bd_queue_counter
            //  );



            // // now, receive S output
            // uint32_t MT_output_size_per_CT_bf16 = MT_Output_size_bf16/NUM_CT_PER_COLUMN;
            // uint32_t DK_per_S_chunk_per_CT = (MT_output_size_per_CT_bf16/2 ) /DV_chunk_per_CT;  // div 2 because S is float32
            
            // wait_DMA_queue_if_full(
            //     seq, cur_col_idx, 
            //     o_out_bd_queue_counter,
            //     shim_tiles.at(cur_col_idx),
            //     S2MM,it_channel_0
            // );
            // seq.npu_dma_memcpy_nd(
            //     sizeof(float),
            //     Arg_S_in_out,
            //     S2MM,
            //     shim_tiles.at(cur_col_idx),
            //     static_cast<npu_bd_id>(get_output_o_s_bd_id(o_out_BD_ping_pong.at(cur_col_idx))),
            //     it_channel_0,
            //     {0,0,0,S_offset},
            //     {NUM_MT_O_CHUNK_FOR_S_output,  NUM_CT_PER_COLUMN,DK_per_S_chunk_per_CT, DV_chunk_per_CT},
            //     {DK_per_S_chunk_per_CT*DV,  DV_chunk_per_CT, DV, 1},
            //     -1, 0, true
            // );
            // update_ping_pong_flag(o_out_BD_ping_pong.at(cur_col_idx));
            // increment_DMA_queue(
            //     seq, cur_col_idx, 
            //     o_out_bd_queue_counter
            //  );
            
        }
        for(int cur_L = 0; cur_L < L_PADDED; cur_L += MT_L_size){
            for(int cur_col_idx = 0; cur_col_idx < transaction_head_idx_list.size(); cur_col_idx ++){

                int g_head_idx = transaction_head_idx_list[cur_col_idx].first;
                int head_idx = transaction_head_idx_list[cur_col_idx].second;                
                    
                uint32_t q_offset = -1;
                uint32_t v_offset = -1;


                uint32_t a_offset_fp32 =cur_L* alpha_beta_buffer_row_stride + head_idx;
                uint32_t beta_buffer_offset_bf16 =beta_start_offset_fp32; // times 2 because two bfloat16 for 1 float
                
                npu_bd_id alpha_bd_id = get_alpha_beta_bd_id(alpha_beta_qkv_BD_ping_pong.at(cur_col_idx));
                npu_bd_id qkv_bd_id = get_q_k_bd_id(alpha_beta_qkv_BD_ping_pong.at(cur_col_idx));
                npu_bd_id v_bd_id = get_v_bd_id(alpha_beta_qkv_BD_ping_pong.at(cur_col_idx));



                wait_DMA_queue_if_full(
                    seq, cur_col_idx, 
                    alpha_beta_qkv_in_bd_queue_counter,
                    shim_tiles.at(cur_col_idx),
                    MM2S,it_channel_1
                );
                // both alpha, beta
                seq.npu_dma_memcpy_nd(
                    sizeof(float),
                    Arg_alpha_beta_in,
                    MM2S,
                    shim_tiles.at(cur_col_idx),
                    alpha_bd_id,
                    it_channel_1,
                    {0,0,0,a_offset_fp32 +alpha_beta_in_offset_fp32},
                    { 1,2,MT_L_size, 1}, // both alpha and beta
                    {0,beta_buffer_offset_bf16,alpha_beta_buffer_row_stride,1},
                    -1, 0, false
                );
               

                    
                    assert(DQ==DK);
                    // both q, k 
                    uint32_t  qkv_buffer_row_stride = NUM_Q_K_HEAD* DQ + NUM_Q_K_HEAD* DK + NUM_HEAD*DV;
                    q_offset = g_head_idx * DQ + cur_L* qkv_buffer_row_stride + QKV_in_offset_bf16;
                    assert(DQ==DK);
                    v_offset = head_idx * DV + (NUM_Q_K_HEAD* DQ) + (NUM_Q_K_HEAD* DK) + cur_L* qkv_buffer_row_stride +QKV_in_offset_bf16;                    
                    seq.npu_dma_memcpy_nd(
                        sizeof(bf16),
                        Arg_QKV_in,
                        MM2S,
                        shim_tiles.at(cur_col_idx),
                        qkv_bd_id,
                        it_channel_1,
                        {0,0,0,q_offset},
                        {1,2,                 MT_L_size, DQ},
                        {0,(NUM_Q_K_HEAD* DQ),qkv_buffer_row_stride,1},
                        -1, 0, false
                    );
                    // send V
                    seq.npu_dma_memcpy_nd(
                        sizeof(bf16),
                        Arg_QKV_in,
                        MM2S,
                        shim_tiles.at(cur_col_idx),
                        v_bd_id,
                        it_channel_1,
                        {0,0,0,v_offset},
                        {1,1, MT_L_size, DV},
                        {0,0,qkv_buffer_row_stride,1},
                        -1, 0, true
                    );

                update_ping_pong_flag(alpha_beta_qkv_BD_ping_pong.at(cur_col_idx));
                increment_DMA_queue(
                    seq, cur_col_idx, 
                    alpha_beta_qkv_in_bd_queue_counter
                );


                // now, receive O from the shimtile
                wait_DMA_queue_if_full(
                    seq, cur_col_idx, 
                    o_out_bd_queue_counter,
                    shim_tiles.at(cur_col_idx),
                    S2MM,it_channel_0
                );
                int O_offset = head_idx* DV + L_start * o_buffer_row_stride;
                seq.npu_dma_memcpy_nd(
                    sizeof(bf16),
                    Arg_O_out,
                    S2MM,
                    shim_tiles.at(cur_col_idx),
                    static_cast<npu_bd_id>(get_output_o_s_bd_id(o_out_BD_ping_pong.at(cur_col_idx))),
                    it_channel_0,
                    {0,0,0,(uint32_t)O_offset +  cur_L*o_buffer_row_stride},
                    {(uint32_t)1, NUM_CT_PER_COLUMN, MT_L_size, DV_chunk_per_CT},
                    {(uint32_t)0 , DV_chunk_per_CT,   o_buffer_row_stride,1},
                    -1, 0, true
                );
                update_ping_pong_flag(o_out_BD_ping_pong.at(cur_col_idx));
                increment_DMA_queue(
                    seq, cur_col_idx, 
                    o_out_bd_queue_counter
                );

            }

        }



        // now receive the S Out buffer
        for(int s_chunk_idx = 0; s_chunk_idx<NUM_MT_O_CHUNK_FOR_S_output; s_chunk_idx++ ){

            for(int cur_col_idx = 0; cur_col_idx < transaction_head_idx_list.size(); cur_col_idx ++){
            
                int g_head_idx = transaction_head_idx_list[cur_col_idx].first;
                int head_idx = transaction_head_idx_list[cur_col_idx].second;


                uint32_t S_offset = head_idx*  DK * DV +S_in_out_offset_fp32;

        
                // now, receive S output
                uint32_t MT_output_size_per_CT_bf16 = MT_Output_size_bf16/NUM_CT_PER_COLUMN;
                uint32_t DK_per_S_chunk_per_CT = (MT_output_size_per_CT_bf16/2 ) /DV_chunk_per_CT;  // div 2 because S is float32
                
                wait_DMA_queue_if_full(
                    seq, cur_col_idx, 
                    o_out_bd_queue_counter,
                    shim_tiles.at(cur_col_idx),
                    S2MM,it_channel_0
                );
                seq.npu_dma_memcpy_nd(
                    sizeof(float),
                    Arg_S_in_out,
                    S2MM,
                    shim_tiles.at(cur_col_idx),
                    static_cast<npu_bd_id>(get_output_o_s_bd_id(o_out_BD_ping_pong.at(cur_col_idx))),
                    it_channel_0,
                    {0,0,0,S_offset + DK_per_S_chunk_per_CT*DV*s_chunk_idx},
                    {1,  NUM_CT_PER_COLUMN,DK_per_S_chunk_per_CT, DV_chunk_per_CT},
                    {0,  DV_chunk_per_CT, DV, 1},
                    -1, 0, true
                );
                update_ping_pong_flag(o_out_BD_ping_pong.at(cur_col_idx));
                increment_DMA_queue(
                    seq, cur_col_idx, 
                    o_out_bd_queue_counter
                );
                


            }

        }


    }




    int max_remaining_queue_number = -1;
    for (int col_idx = 0; col_idx < shimtile_size; col_idx++){
        max_remaining_queue_number = std::max(max_remaining_queue_number, s_in_bd_queue_counter.at(col_idx));
        max_remaining_queue_number = std::max(max_remaining_queue_number, alpha_beta_qkv_in_bd_queue_counter.at(col_idx));
        max_remaining_queue_number = std::max(max_remaining_queue_number, o_out_bd_queue_counter.at(col_idx));
    }


    for(int i = 0; i < max_remaining_queue_number; i++){
        for(int col_idx = 0; col_idx< NUM_OF_COLUMN; col_idx++){
            force_wait_DMA_queue(
                seq, col_idx, s_in_bd_queue_counter,
                shim_tiles.at(col_idx),
                MM2S,it_channel_0
            );
            force_wait_DMA_queue(
                seq, col_idx, alpha_beta_qkv_in_bd_queue_counter,
                shim_tiles.at(col_idx),
                MM2S,it_channel_1
            );
            force_wait_DMA_queue(
                seq, col_idx, o_out_bd_queue_counter,
                shim_tiles.at(col_idx),
                S2MM,it_channel_0
            );

        }
    }




    seq.cmds2seq();


}




#endif