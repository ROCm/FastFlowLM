#ifndef __Attention_RUNTIME_SEQUENCE_HPP__
#define __Attention_RUNTIME_SEQUENCE_HPP__


//direct copy from gemma3

#include <cassert>
#include <algorithm> // Required for std::max
#include <string>
#include "npu_utils/npu_instr_utils.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"





inline void update_ping_pong_flag(int &flag){
    if(flag ==0){
        flag =1;
    }else{
        flag =0;
    }
}


//ALL SHIMTILE agree on common bds for Q, K, V O that it is using
//for Q, used bd 0, 8
//for k, used bd 1, 9
//for V, used bd 2, 10
//for O, used bd 3, 11
inline int get_Q_bd_id(int ping_pong){
    if(ping_pong ==0){
        return 0;
    }else{
        return 8;
    }
}

inline int get_K_bd_id(int ping_pong){
    if(ping_pong ==0){
        return 1;
    }else{
        return 9;
    }
}
inline int get_V_bd_id(int ping_pong){
    if(ping_pong ==0){
        return 2;
    }else{
        return 10;
    }
}
inline int get_O_bd_id(int ping_pong){
    if(ping_pong ==0){
        return 3;
    }else{
        return 11;
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







template<typename T_in, typename T_out>
void process_SHM_Q_chunk_Q_repeat(
    npu_sequence &seq,
    int NUM_OF_CT_per_column,
    int MT_KV_repeat,
    int DH,
    int NUM_OF_DH,
    int head_idx_start, // current DH index that is being processed

    int SHIM_LQ_start_row,
    int SHIM_S_SEQ_PADDED,

    std::vector<std::vector<int>> &CU_column,
    int LQ_CU_chunk_size,
    int LK_CU_chunk_size,

    int LQ_per_CT,
    int LK_per_CT,

    std::vector<int> &SHIM_Q_queue_counter,
    std::vector<int> &SHIM_K_queue_counter,
    std::vector<int> &SHIM_V_queue_counter,
    std::vector<int> &SHIM_O_queue_counter,

    std::vector<int> &SHIM_Q_ping_pong,
    std::vector<int> &SHIM_K_ping_pong,
    std::vector<int> &SHIM_V_ping_pong,
    std::vector<int> &SHIM_O_ping_pong,

    uint32_t Q_PADDDED_D, uint32_t K_PADDED_D, uint32_t V_PADDED_D, uint32_t O_PADDDED_D,
    uint32_t Q_offset, uint32_t K_offset, uint32_t V_offset, uint32_t O_offset,

    int ARG_Q, int ARG_K, int ARG_V, int ARG_O,
    npu_it_channel it_for_send_Q, npu_it_channel it_for_send_KV,
    npu_it_channel it_for_recv_O,
    std::vector<npu_tiles> &shim_tiles
){

 

    
    uint32_t LQ_per_column = LQ_CU_chunk_size/(CU_column[0].size());

    bool sanity_check = true;

    sanity_check &= SHIM_S_SEQ_PADDED%LK_CU_chunk_size ==0;
    sanity_check &= MT_KV_repeat >1;
    sanity_check &= LK_CU_chunk_size == LK_per_CT;
    sanity_check &= LQ_CU_chunk_size % CU_column[0].size() ==0;

    sanity_check &= NUM_OF_CT_per_column*LQ_per_CT == LQ_per_column;
    if(!sanity_check){
        std::cerr << "process_SHM_Q_chunk_Q_repeat sanity check failed"<< std::endl;
        exit(1);
    }


    for(int cu_idx = 0; cu_idx <CU_column.size(); cu_idx++){
        auto cur_CU_shimtile_cols = CU_column.at(cu_idx);
        int DH_index = head_idx_start+ cu_idx;
        if(DH_index >= NUM_OF_DH){
            break;
        }


        uint32_t SHIM_O_offset =   SHIM_LQ_start_row *  (O_PADDDED_D) + DH_index * DH +O_offset;

        for(int idx = 0; idx < cur_CU_shimtile_cols.size(); idx++){

            auto shim_col_idx = cur_CU_shimtile_cols.at(idx);


            // now, receive the O
            wait_DMA_queue_if_full(
                seq, shim_col_idx, SHIM_O_queue_counter,
                shim_tiles.at(shim_col_idx),
                S2MM, it_for_recv_O
            );
            seq.npu_dma_memcpy_nd(
                sizeof(T_out),
                ARG_O,
                S2MM, 
                shim_tiles.at(shim_col_idx),
                static_cast<npu_bd_id>(get_O_bd_id(SHIM_O_ping_pong.at(shim_col_idx))),
                it_for_recv_O,
                {0,0,0,   (uint32_t)SHIM_O_offset + LQ_per_column*idx*(O_PADDDED_D)},
                {1,1, (uint32_t)LQ_per_column, (uint32_t)DH},
                {0,0, (uint32_t)O_PADDDED_D, 1},
                -1, 0, true
            );
            update_ping_pong_flag(SHIM_O_ping_pong.at(shim_col_idx));
            increment_DMA_queue(
                seq, shim_col_idx, SHIM_O_queue_counter
            );


        

        }

    }


    for(int kv_chunk_idx = 0; kv_chunk_idx < SHIM_S_SEQ_PADDED/LK_CU_chunk_size; kv_chunk_idx++){



        for(int cu_idx = 0; cu_idx <CU_column.size(); cu_idx++){
            auto cur_CU_shimtile_cols = CU_column.at(cu_idx);
            int DH_index = head_idx_start+ cu_idx;
            if(DH_index >= NUM_OF_DH){
                break;
            }
        
        




            uint32_t Shim_Q_offset =     SHIM_LQ_start_row *  (Q_PADDDED_D) + DH_index * DH +Q_offset;

            uint32_t SHIM_K_offset = DH_index * DH + K_offset;
            uint32_t SHIM_V_offset = DH_index * DH + V_offset;

            for(int idx = 0; idx < cur_CU_shimtile_cols.size(); idx++){

                auto shim_col_idx = cur_CU_shimtile_cols.at(idx);
                wait_DMA_queue_if_full(
                    seq, shim_col_idx, SHIM_Q_queue_counter,
                    shim_tiles.at(shim_col_idx),
                    MM2S, it_for_send_Q
                );

                seq.npu_dma_memcpy_nd(
                    sizeof(T_in),
                    ARG_Q,
                    MM2S, 
                    shim_tiles.at(shim_col_idx),
                    static_cast<npu_bd_id>(get_Q_bd_id(SHIM_Q_ping_pong.at(shim_col_idx))),
                    it_for_send_Q,
                    {0,0,0,   (uint32_t)Shim_Q_offset + LQ_per_column*idx*(Q_PADDDED_D)},
                    {1,1, (uint32_t)LQ_per_column, (uint32_t)DH},
                    {0,0, (uint32_t)Q_PADDDED_D, 1},
                    -1, 0, true

                );
                update_ping_pong_flag(SHIM_Q_ping_pong.at(shim_col_idx));
                increment_DMA_queue(
                    seq, shim_col_idx, SHIM_Q_queue_counter
                );
            }


            //only the first shimtile column in cur_CU_shimtile_cols will be used to send KV data
            {

                uint32_t shim_col_idx = cur_CU_shimtile_cols.at(0);

                //NOTE: NO need to wait for k,since we only issue the v
                wait_DMA_queue_if_full(
                    seq, shim_col_idx, SHIM_V_queue_counter,
                    shim_tiles.at(shim_col_idx),
                    MM2S, it_for_send_KV
                );

                seq.npu_dma_memcpy_nd(
                    sizeof(T_in),
                    ARG_K,
                    MM2S, 
                    shim_tiles.at(shim_col_idx),
                    static_cast<npu_bd_id>(get_K_bd_id(SHIM_K_ping_pong.at(shim_col_idx))),
                    it_for_send_KV,
                    {0,0,0,   (uint32_t)SHIM_K_offset + kv_chunk_idx*LK_CU_chunk_size*(K_PADDED_D)},
                    {1,1, (uint32_t)LK_CU_chunk_size, (uint32_t)DH},
                    {0,0, (uint32_t)K_PADDED_D, 1},
                    -1, 0, false,
                    aggressive_cache
                    
                );
                update_ping_pong_flag(SHIM_K_ping_pong.at(shim_col_idx));


                seq.npu_dma_memcpy_nd(
                    sizeof(T_in),
                    ARG_V,
                    MM2S, 
                    shim_tiles.at(shim_col_idx),
                    static_cast<npu_bd_id>(get_V_bd_id(SHIM_V_ping_pong.at(shim_col_idx))),
                    it_for_send_KV,
                    {0,0,0,   (uint32_t)SHIM_V_offset + kv_chunk_idx*LK_CU_chunk_size*(V_PADDED_D)},
                    {1,1, (uint32_t)LK_CU_chunk_size, (uint32_t)DH},
                    {0,0, (uint32_t)V_PADDED_D, 1},
                    -1, 0, true,
                    aggressive_cache
                );
                update_ping_pong_flag(SHIM_V_ping_pong.at(shim_col_idx));
                increment_DMA_queue(
                    seq, shim_col_idx, SHIM_V_queue_counter
                );

            }



        }



    
    
    }









}





template<typename T_in, typename T_out>
void process_SHM_Q_chunk_Q_repeat_reorder_NUM_DH_L_DH(
    npu_sequence &seq,
    int NUM_OF_CT_per_column,
    int MT_KV_repeat,
    int DH,
    int NUM_OF_DH,
    int DH_index_start, // current DH index that is being processed

    int SHIM_LQ_start_row,
    int SHIM_S_SEQ_PADDED,

    std::vector<std::vector<int>> &CU_column,
    int LQ_CU_chunk_size,
    int LK_CU_chunk_size,

    int LQ_per_CT,
    int LK_per_CT,

    std::vector<int> &SHIM_Q_queue_counter,
    std::vector<int> &SHIM_K_queue_counter,
    std::vector<int> &SHIM_V_queue_counter,
    std::vector<int> &SHIM_O_queue_counter,

    std::vector<int> &SHIM_Q_ping_pong,
    std::vector<int> &SHIM_K_ping_pong,
    std::vector<int> &SHIM_V_ping_pong,
    std::vector<int> &SHIM_O_ping_pong,

    uint32_t O_PADDDED_D,
    //[QWEN3_VISION_NUM_HEAD, batch, L_Seq_per_batch, QWEN3_VISION_HEAD_DIM]
    uint32_t Q_total_seq_len_per_head, // Total sequence length per head for Q in layout[NUM_HEAD, batch, L_Seq_per_batch, HEAD_DIM]. Can be L_seq_padded*batch_size (uniform padding) or L_seq*(batch_size-1)+L_seq_padded (only last batch padded)
    uint32_t K_total_seq_len_per_head, // Total sequence length per head for K in layout [NUM_HEAD, batch, L_Seq_per_batch, HEAD_DIM]. Can be L_seq_padded*batch_size (uniform padding) or L_seq*(batch_size-1)+L_seq_padded (only last batch padded)
    uint32_t V_total_seq_len_per_head, // Total sequence length per head for V in layout [NUM_HEAD, batch, L_Seq_per_batch, HEAD_DIM]. Can be L_seq_padded*batch_size (uniform padding) or L_seq*(batch_size-1)+L_seq_padded (only last batch padded)
    uint32_t Q_offset, uint32_t K_offset, uint32_t V_offset, uint32_t O_offset,

    int ARG_Q, int ARG_K, int ARG_V, int ARG_O,
    npu_it_channel it_for_send_Q, npu_it_channel it_for_send_KV,
    npu_it_channel it_for_recv_O,
    std::vector<npu_tiles> &shim_tiles
){


    uint32_t LQ_per_column = LQ_CU_chunk_size/(CU_column[0].size());

    bool sanity_check = true;

    sanity_check &= SHIM_S_SEQ_PADDED%LK_CU_chunk_size ==0;
    sanity_check &= MT_KV_repeat >1;
    sanity_check &= LK_CU_chunk_size == LK_per_CT;
    sanity_check &= LQ_CU_chunk_size % CU_column[0].size() ==0;

    sanity_check &= NUM_OF_CT_per_column*LQ_per_CT == LQ_per_column;
    if(!sanity_check){
        std::cerr << "process_SHM_Q_chunk_Q_repeat sanity check failed"<< std::endl;
        exit(1);
    }



    for(int cu_idx = 0; cu_idx <CU_column.size(); cu_idx++){
        auto cur_CU_shimtile_cols = CU_column.at(cu_idx);
        int DH_index = DH_index_start + cu_idx;
        if(DH_index >= NUM_OF_DH){
            break;
        }

        for(int idx = 0; idx < cur_CU_shimtile_cols.size(); idx++){

            auto shim_col_idx = cur_CU_shimtile_cols.at(idx);

            uint32_t SHIM_O_offset =   SHIM_LQ_start_row *  (O_PADDDED_D) + DH_index * DH +O_offset;
            // now, receive the O
            wait_DMA_queue_if_full(
                seq, shim_col_idx, SHIM_O_queue_counter,
                shim_tiles.at(shim_col_idx),
                S2MM, it_for_recv_O
            );
            seq.npu_dma_memcpy_nd(
                sizeof(T_out),
                ARG_O,
                S2MM, 
                shim_tiles.at(shim_col_idx),
                static_cast<npu_bd_id>(get_O_bd_id(SHIM_O_ping_pong.at(shim_col_idx))),
                it_for_recv_O,
                {0,0,0,   (uint32_t)SHIM_O_offset + LQ_per_column*idx*(O_PADDDED_D)},
                {1,1, (uint32_t)LQ_per_column, (uint32_t)DH},
                {0,0, (uint32_t)O_PADDDED_D, 1},
                -1, 0, true
            );
            update_ping_pong_flag(SHIM_O_ping_pong.at(shim_col_idx));
            increment_DMA_queue(
                seq, shim_col_idx, SHIM_O_queue_counter
            );


        }

    }





    for(int kv_chunk_idx = 0; kv_chunk_idx < SHIM_S_SEQ_PADDED/LK_CU_chunk_size; kv_chunk_idx++){



        for(int cu_idx = 0; cu_idx <CU_column.size(); cu_idx++){
            auto cur_CU_shimtile_cols = CU_column.at(cu_idx);
            int DH_index = DH_index_start + cu_idx;
            if(DH_index >= NUM_OF_DH){
                break;
            }

 
            uint32_t Shim_Q_offset =  SHIM_LQ_start_row* DH      +    ((Q_total_seq_len_per_head * DH) * DH_index) + Q_offset;

            uint32_t SHIM_K_offset =  DH_index * (DH*K_total_seq_len_per_head)  + K_offset;
            uint32_t SHIM_V_offset =  DH_index * (DH*V_total_seq_len_per_head) + V_offset;
            

            for(int idx = 0; idx < cur_CU_shimtile_cols.size(); idx++){

                auto shim_col_idx = cur_CU_shimtile_cols.at(idx);
                wait_DMA_queue_if_full(
                    seq, shim_col_idx, SHIM_Q_queue_counter,
                    shim_tiles.at(shim_col_idx),
                    MM2S, it_for_send_Q
                );

                seq.npu_dma_memcpy_nd(
                    sizeof(T_in),
                    ARG_Q,
                    MM2S, 
                    shim_tiles.at(shim_col_idx),
                    static_cast<npu_bd_id>(get_Q_bd_id(SHIM_Q_ping_pong.at(shim_col_idx))),
                    it_for_send_Q,
                    {0,0,0,   (uint32_t)Shim_Q_offset + LQ_per_column*idx*(DH)},
                    // {1,1, (uint32_t)LQ_per_column, (uint32_t)DH},
                    // {0,0, (uint32_t)DH, 1},
                    {1, 1, 1, (uint32_t)LQ_per_column*DH},
                    {0, 0, 0, 1},
                    -1, 0, true

                );
                update_ping_pong_flag(SHIM_Q_ping_pong.at(shim_col_idx));
                increment_DMA_queue(
                    seq, shim_col_idx, SHIM_Q_queue_counter
                );
            }


            //only the first shimtile column in cur_CU_shimtile_cols will be used to send KV data
            {

                uint32_t shim_col_idx = cur_CU_shimtile_cols.at(0);

                //NOTE: NO need to wait for k,since we only issue the v
                wait_DMA_queue_if_full(
                    seq, shim_col_idx, SHIM_V_queue_counter,
                    shim_tiles.at(shim_col_idx),
                    MM2S, it_for_send_KV
                );

                seq.npu_dma_memcpy_nd(
                    sizeof(T_in),
                    ARG_K,
                    MM2S, 
                    shim_tiles.at(shim_col_idx),
                    static_cast<npu_bd_id>(get_K_bd_id(SHIM_K_ping_pong.at(shim_col_idx))),
                    it_for_send_KV,
                    {0,0,0,   (uint32_t)SHIM_K_offset + kv_chunk_idx*LK_CU_chunk_size*(DH)},
                    //{1,1, (uint32_t)LK_CU_chunk_size, (uint32_t)DH},
                    //{0,0, (uint32_t)DH, 1},
                    {1, 1, 1, (uint32_t)LK_CU_chunk_size*DH},
                    {0, 0, 0, 1},
                    -1, 0, false,
                    aggressive_cache
                    
                );
                update_ping_pong_flag(SHIM_K_ping_pong.at(shim_col_idx));


                seq.npu_dma_memcpy_nd(
                    sizeof(T_in),
                    ARG_V,
                    MM2S, 
                    shim_tiles.at(shim_col_idx),
                    static_cast<npu_bd_id>(get_V_bd_id(SHIM_V_ping_pong.at(shim_col_idx))),
                    it_for_send_KV,
                    {0,0,0,   (uint32_t)SHIM_V_offset + kv_chunk_idx*LK_CU_chunk_size*(DH)},
                    // {1,1, (uint32_t)LK_CU_chunk_size, (uint32_t)DH},
                    // {0,0, (uint32_t)DH, 1},
                    {1, 1, 1, (uint32_t)LK_CU_chunk_size*DH},
                    {0, 0, 0, 1},                
                    -1, 0, true,
                    aggressive_cache
                );
                update_ping_pong_flag(SHIM_V_ping_pong.at(shim_col_idx));
                increment_DMA_queue(
                    seq, shim_col_idx, SHIM_V_queue_counter
                );

            }
    

        }



    
    }

}

template<typename T_in, typename T_out>
void process_SHM_Q_chunk_no_Q_repeat(

    npu_sequence &seq,
    int NUM_OF_CT_per_column,
    int MT_KV_repeat,
    int DH,
    int NUM_OF_DH,
    int DH_index_start, // current DH index that is being processed

    int SHIM_LQ_start_row,
    int SHIM_S_SEQ_PADDED,

    std::vector<std::vector<int>> &CU_column,
    int LQ_CU_chunk_size,
    int LK_CU_chunk_size,

    int LQ_per_CT,
    int LK_per_CT,

    std::vector<int> &SHIM_Q_queue_counter,
    std::vector<int> &SHIM_K_queue_counter,
    std::vector<int> &SHIM_V_queue_counter,
    std::vector<int> &SHIM_O_queue_counter,

    std::vector<int> &SHIM_Q_ping_pong,
    std::vector<int> &SHIM_K_ping_pong,
    std::vector<int> &SHIM_V_ping_pong,
    std::vector<int> &SHIM_O_ping_pong,


    uint32_t Q_PADDDED_D, uint32_t K_PADDED_D, uint32_t V_PADDED_D, uint32_t O_PADDDED_D,
    uint32_t Q_offset, uint32_t K_offset, uint32_t V_offset, uint32_t O_offset,
    int ARG_Q, int ARG_K, int ARG_V, int ARG_O,
    npu_it_channel it_for_send_Q, npu_it_channel it_for_send_KV,
    npu_it_channel it_for_recv_O,
    std::vector<npu_tiles> &shim_tiles

){  
    if(LQ_CU_chunk_size % LQ_per_CT !=0){
        std::cerr << "process_SHM_Q_chunk_no_Q_repeat LQ_CU_chunk_size % LQ_per_CT !=0"<< std::endl;
        exit(1);
    }



    uint32_t LQ_per_column = LQ_CU_chunk_size/(CU_column[0].size());
    bool sanity_check = true;
    sanity_check &= SHIM_S_SEQ_PADDED%LK_CU_chunk_size ==0;
    sanity_check &= MT_KV_repeat == 1;
    sanity_check &= LK_CU_chunk_size == LK_per_CT;
    sanity_check &= NUM_OF_CT_per_column*LQ_per_CT == LQ_per_column;
    if(!sanity_check){
        std::cerr << "process_SHM_Q_chunk_no_Q_repeat sanity check failed"<< std::endl;
        exit(1);
    }





    for(int cu_idx = 0; cu_idx < CU_column.size(); cu_idx++){

        auto cur_CU_shimtile_cols = CU_column.at(cu_idx);
        int DH_index = DH_index_start + cu_idx;
        if(DH_index >= NUM_OF_DH){
            break;// done
        }
        
    
        uint32_t Shim_Q_offset = SHIM_LQ_start_row *  (Q_PADDDED_D) + DH_index * DH + Q_offset;
        uint32_t SHIM_O_offset = SHIM_LQ_start_row *  (O_PADDDED_D) + DH_index * DH + O_offset;

    
    
        for(int idx = 0; idx < cur_CU_shimtile_cols.size(); idx++){

            auto shim_col_idx = cur_CU_shimtile_cols.at(idx);


            wait_DMA_queue_if_full(
                seq, shim_col_idx, SHIM_Q_queue_counter,
                shim_tiles.at(shim_col_idx),
                MM2S, it_for_send_Q
            );

            seq.npu_dma_memcpy_nd(
                sizeof(T_in),
                ARG_Q,
                MM2S, 
                shim_tiles.at(shim_col_idx),
                static_cast<npu_bd_id>(get_Q_bd_id(SHIM_Q_ping_pong.at(shim_col_idx))),
                it_for_send_Q,
                {0,0,0,   (uint32_t)Shim_Q_offset + LQ_per_column*idx*(Q_PADDDED_D)},
                {1,1, (uint32_t)LQ_per_column, (uint32_t)DH},
                {0,0, (uint32_t)Q_PADDDED_D, 1},
                -1, 0, true

            );
            update_ping_pong_flag(SHIM_Q_ping_pong.at(shim_col_idx));
            increment_DMA_queue(
                seq, shim_col_idx, SHIM_Q_queue_counter
            );




            // now, receive the O
            wait_DMA_queue_if_full(
                seq, shim_col_idx, SHIM_O_queue_counter,
                shim_tiles.at(shim_col_idx),
                S2MM, it_for_recv_O
            );
            seq.npu_dma_memcpy_nd(
                sizeof(T_out),
                ARG_O,
                S2MM, 
                shim_tiles.at(shim_col_idx),
                static_cast<npu_bd_id>(get_O_bd_id(SHIM_O_ping_pong.at(shim_col_idx))),
                it_for_recv_O,
                {0,0,0,   (uint32_t)SHIM_O_offset + LQ_per_column*idx*(O_PADDDED_D)},
                {1,1, (uint32_t)LQ_per_column, (uint32_t)DH},
                {0,0, (uint32_t)O_PADDDED_D, 1},
                -1, 0, true
            );
            update_ping_pong_flag(SHIM_O_ping_pong.at(shim_col_idx));
            increment_DMA_queue(
                seq, shim_col_idx, SHIM_O_queue_counter
            );


        }


    }


    for(uint32_t kv_chunk_idx = 0; kv_chunk_idx < SHIM_S_SEQ_PADDED/LK_CU_chunk_size; kv_chunk_idx++){


        for(int cu_idx = 0; cu_idx < CU_column.size(); cu_idx++){

            auto cur_CU_shimtile_cols = CU_column.at(cu_idx);
            int DH_index = DH_index_start + cu_idx;
            if(DH_index >= NUM_OF_DH){
                break;// done
            }

            uint32_t SHIM_K_offset = DH_index * DH + K_offset;
            uint32_t SHIM_V_offset = DH_index * DH + V_offset;
            // only the first shimtile column in cur_CU_shimtile_cols will be used to send KV data
            uint32_t shim_col_idx = cur_CU_shimtile_cols.at(0);


            //NOTE: NO need to wait for k,since we only issue the v
            wait_DMA_queue_if_full(
                seq, shim_col_idx, SHIM_V_queue_counter,
                shim_tiles.at(shim_col_idx),
                MM2S, it_for_send_KV
            );

            seq.npu_dma_memcpy_nd(
                sizeof(T_in),
                ARG_K,
                MM2S, 
                shim_tiles.at(shim_col_idx),
                static_cast<npu_bd_id>(get_K_bd_id(SHIM_K_ping_pong.at(shim_col_idx))),
                it_for_send_KV,
                {0,0,0,   (uint32_t)SHIM_K_offset + kv_chunk_idx*LK_CU_chunk_size*(K_PADDED_D)},
                {1,1, (uint32_t)LK_CU_chunk_size, (uint32_t)DH},
                {0,0, (uint32_t)K_PADDED_D, 1},
                -1, 0, false,
                aggressive_cache
            );
            update_ping_pong_flag(SHIM_K_ping_pong.at(shim_col_idx));


            seq.npu_dma_memcpy_nd(
                sizeof(T_in),
                ARG_V,
                MM2S, 
                shim_tiles.at(shim_col_idx),
                static_cast<npu_bd_id>(get_V_bd_id(SHIM_V_ping_pong.at(shim_col_idx))),
                it_for_send_KV,
                {0,0,0,   (uint32_t)SHIM_V_offset + kv_chunk_idx*LK_CU_chunk_size*(V_PADDED_D)},
                {1,1, (uint32_t)LK_CU_chunk_size, (uint32_t)DH},
                {0,0, (uint32_t)V_PADDED_D, 1},
                -1, 0, true,
                aggressive_cache
            );
            update_ping_pong_flag(SHIM_V_ping_pong.at(shim_col_idx));
            increment_DMA_queue(
                seq, shim_col_idx, SHIM_V_queue_counter
            );

        }


    }

}
















template<typename T_in, typename T_out>
void process_SHM_Q_chunk_no_Q_repeat_qkv_reorder_NUM_DH_L_DH(

    npu_sequence &seq,
    int NUM_OF_CT_per_column,
    int MT_KV_repeat,
    int DH,
    int NUM_OF_DH,
    int head_idx_start, // current DH index that is being processed

    int SHIM_LQ_start_row,
    int SHIM_S_SEQ_PADDED,

    std::vector<std::vector<int>> &CU_column,
    int LQ_CU_chunk_size,
    int LK_CU_chunk_size,

    int LQ_per_CT,
    int LK_per_CT,

    std::vector<int> &SHIM_Q_queue_counter,
    std::vector<int> &SHIM_K_queue_counter,
    std::vector<int> &SHIM_V_queue_counter,
    std::vector<int> &SHIM_O_queue_counter,

    std::vector<int> &SHIM_Q_ping_pong,
    std::vector<int> &SHIM_K_ping_pong,
    std::vector<int> &SHIM_V_ping_pong,
    std::vector<int> &SHIM_O_ping_pong,


    uint32_t O_PADDDED_D,
    //[QWEN3_VISION_NUM_HEAD, batch, L_Seq_per_batch, QWEN3_VISION_HEAD_DIM]
    uint32_t Q_total_seq_len_per_head, // Total sequence length per head for Q in layout[NUM_HEAD, batch, L_Seq_per_batch, HEAD_DIM]. Can be L_seq_padded*batch_size (uniform padding) or L_seq*(batch_size-1)+L_seq_padded (only last batch padded)
    uint32_t K_total_seq_len_per_head, // Total sequence length per head for K in layout [NUM_HEAD, batch, L_Seq_per_batch, HEAD_DIM]. Can be L_seq_padded*batch_size (uniform padding) or L_seq*(batch_size-1)+L_seq_padded (only last batch padded)
    uint32_t V_total_seq_len_per_head, // Total sequence length per head for V in layout [NUM_HEAD, batch, L_Seq_per_batch, HEAD_DIM]. Can be L_seq_padded*batch_size (uniform padding) or L_seq*(batch_size-1)+L_seq_padded (only last batch padded)

    uint32_t Q_offset, uint32_t K_offset, uint32_t V_offset, uint32_t O_offset,

    int ARG_Q, int ARG_K, int ARG_V, int ARG_O,
    npu_it_channel it_for_send_Q, npu_it_channel it_for_send_KV,
    npu_it_channel it_for_recv_O,
    std::vector<npu_tiles> &shim_tiles

){  
    if(LQ_CU_chunk_size % LQ_per_CT !=0){
        std::cerr << "process_SHM_Q_chunk_no_Q_repeat LQ_CU_chunk_size % LQ_per_CT !=0"<< std::endl;
        exit(1);
    }
    uint32_t LQ_per_column = LQ_CU_chunk_size/(CU_column[0].size());
    bool sanity_check = true;
    sanity_check &= SHIM_S_SEQ_PADDED%LK_CU_chunk_size ==0;
    sanity_check &= MT_KV_repeat == 1;
    sanity_check &= LK_CU_chunk_size == LK_per_CT;
    sanity_check &= NUM_OF_CT_per_column*LQ_per_CT == LQ_per_column;
    if(!sanity_check){
        std::cerr << "process_SHM_Q_chunk_no_Q_repeat sanity check failed"<< std::endl;
        exit(1);
    }
    




    for(int cu_idx = 0; cu_idx < CU_column.size(); cu_idx++){

        auto cur_CU_shimtile_cols = CU_column.at(cu_idx);
        int DH_index = head_idx_start + cu_idx;
        if(DH_index >= NUM_OF_DH){
            break;// done
        }

        uint32_t Shim_Q_offset = SHIM_LQ_start_row* DH      +    ((Q_total_seq_len_per_head * DH) * DH_index) + Q_offset;
        uint32_t SHIM_O_offset = SHIM_LQ_start_row * (O_PADDDED_D) + DH_index * DH + O_offset;


        for(int idx = 0; idx < cur_CU_shimtile_cols.size(); idx++){

            auto shim_col_idx = cur_CU_shimtile_cols.at(idx);


            wait_DMA_queue_if_full(
                seq, shim_col_idx, SHIM_Q_queue_counter,
                shim_tiles.at(shim_col_idx),
                MM2S, it_for_send_Q
            );

            seq.npu_dma_memcpy_nd(
                sizeof(T_in),
                ARG_Q,
                MM2S, 
                shim_tiles.at(shim_col_idx),
                static_cast<npu_bd_id>(get_Q_bd_id(SHIM_Q_ping_pong.at(shim_col_idx))),
                it_for_send_Q,
                {0,0,0,   (uint32_t) Shim_Q_offset +   idx*(LQ_per_column*DH)},
                // {1,1, (uint32_t)LQ_per_column, (uint32_t)DH},
                // {0,0, (uint32_t)DH, 1},
                {1, 1, 1, (uint32_t)LQ_per_column*DH},
                {0, 0, 0, 1},
                -1, 0, true

            );
            update_ping_pong_flag(SHIM_Q_ping_pong.at(shim_col_idx));
            increment_DMA_queue(
                seq, shim_col_idx, SHIM_Q_queue_counter
            );




            // now, receive the O
            wait_DMA_queue_if_full(
                seq, shim_col_idx, SHIM_O_queue_counter,
                shim_tiles.at(shim_col_idx),
                S2MM, it_for_recv_O
            );
            seq.npu_dma_memcpy_nd(
                sizeof(T_out),
                ARG_O,
                S2MM, 
                shim_tiles.at(shim_col_idx),
                static_cast<npu_bd_id>(get_O_bd_id(SHIM_O_ping_pong.at(shim_col_idx))),
                it_for_recv_O,
                {0,0,0,   (uint32_t)SHIM_O_offset + LQ_per_column*idx*(O_PADDDED_D)},
                {1,1, (uint32_t)LQ_per_column, (uint32_t)DH},
                {0,0, (uint32_t)O_PADDDED_D, 1},
                -1, 0, true
            );
            update_ping_pong_flag(SHIM_O_ping_pong.at(shim_col_idx));
            increment_DMA_queue(
                seq, shim_col_idx, SHIM_O_queue_counter
            );


        }


    }



    for(uint32_t kv_chunk_idx = 0; kv_chunk_idx < SHIM_S_SEQ_PADDED/LK_CU_chunk_size; kv_chunk_idx++){


        for(int cu_idx = 0; cu_idx < CU_column.size(); cu_idx++){

            auto cur_CU_shimtile_cols = CU_column.at(cu_idx);
            int DH_index = head_idx_start + cu_idx;
            if(DH_index >= NUM_OF_DH){
                break;// done
            }

            uint32_t SHIM_K_offset =  DH_index * (DH*K_total_seq_len_per_head)  + K_offset;
            uint32_t SHIM_V_offset =  DH_index * (DH*V_total_seq_len_per_head) + V_offset;


            // only the first shimtile column in cur_CU_shimtile_cols will be used to send KV data
            uint32_t shim_col_idx = cur_CU_shimtile_cols.at(0);

            //NOTE: NO need to wait for k,since we only issue the v
            wait_DMA_queue_if_full(
                seq, shim_col_idx, SHIM_V_queue_counter,
                shim_tiles.at(shim_col_idx),
                MM2S, it_for_send_KV
            );

            seq.npu_dma_memcpy_nd(
                sizeof(T_in),
                ARG_K,
                MM2S, 
                shim_tiles.at(shim_col_idx),
                static_cast<npu_bd_id>(get_K_bd_id(SHIM_K_ping_pong.at(shim_col_idx))),
                it_for_send_KV,
                {0,0,0,   (uint32_t)SHIM_K_offset + kv_chunk_idx*LK_CU_chunk_size*(DH )},
                //{1,1, (uint32_t)LK_CU_chunk_size, (uint32_t)DH},
                //{0,0, (uint32_t)DH, 1},
                {1,1, 1,(uint32_t)LK_CU_chunk_size *DH},
                {0,0, 0, 1},
                -1, 0, false,
                aggressive_cache
            );
            update_ping_pong_flag(SHIM_K_ping_pong.at(shim_col_idx));


            seq.npu_dma_memcpy_nd(
                sizeof(T_in),
                ARG_V,
                MM2S, 
                shim_tiles.at(shim_col_idx),
                static_cast<npu_bd_id>(get_V_bd_id(SHIM_V_ping_pong.at(shim_col_idx))),
                it_for_send_KV,
                {0,0,0,   (uint32_t)SHIM_V_offset + kv_chunk_idx*LK_CU_chunk_size*(DH)},
                // {1,1, (uint32_t)LK_CU_chunk_size, (uint32_t)DH},
                // {0,0, (uint32_t)DH, 1},
                {1,1, 1,(uint32_t)LK_CU_chunk_size *DH},
                {0,0, 0, 1},
                -1, 0, true,
                aggressive_cache
            );
            update_ping_pong_flag(SHIM_V_ping_pong.at(shim_col_idx));
            increment_DMA_queue(
                seq, shim_col_idx, SHIM_V_queue_counter
            );
        }



    }

}










struct AttentionConfig {
    uint32_t DH;
    uint32_t NUM_OF_DH;
    uint32_t LQ_per_CT;
    uint32_t LK_per_CT;
    uint32_t lq_internal;
    uint32_t lk_lv_internal;
    uint32_t NUM_OF_COLUMNS;
    uint32_t NUM_OF_CT_PER_COLUMN;
    uint32_t CU_mode;
    uint32_t num_of_batches;

    int ARG_Q, ARG_K, ARG_V, ARG_O;


    // Q_PADDED_D, K_PADDED_D, V_PADDED_D is singificant only if QKV_reordered == false
    // since QKV_reordered= false mean
    // Q, K, V buffer is view as [batch, L_seq, NUM_DH*DH]
    uint32_t Q_padded_D; 
    uint32_t K_padded_D;
    uint32_t V_padded_D;
    uint32_t O_PADDDED_D;

    bool QKV_reordered;
    // parameter below is only siginificatn when QKV_reordered = true
    // when QKV_reordered == true, it means 
    // Q, K, V buffer is viewed as [NUM_DH, batch, L_seq, DH]
    uint32_t Q_total_seq_len_per_head;
    uint32_t K_total_seq_len_per_head;
    uint32_t V_total_seq_len_per_head;
};

struct BatchMetadata {
    const std::vector<int>& SHIM_L_SEQ_list;
    const std::vector<int>& SHIM_S_SEQ_list;
    const std::vector<int>& SHIM_S_SEQ_PADDED_list;
    
    const std::vector<int>& Q_batch_offset_list;
    const std::vector<int>& K_batch_offset_list;
    const std::vector<int>& V_batch_offset_list;
    const std::vector<int>& O_batch_offset_list;
};

template<typename T_in, typename T_out>
void setup_SHM_configuration(
    npu_sequence &seq,
    const AttentionConfig& config,
    const BatchMetadata& batch_data,
    float attention_scalar
){
    uint32_t DH = config.DH;
    uint32_t NUM_OF_DH = config.NUM_OF_DH;
    uint32_t LQ_per_CT = config.LQ_per_CT;
    uint32_t LK_per_CT = config.LK_per_CT;
    uint32_t lq_internal = config.lq_internal;
    uint32_t lk_lv_internal = config.lk_lv_internal;
    uint32_t NUM_OF_COLUMNS = config.NUM_OF_COLUMNS;
    uint32_t NUM_OF_CT_PER_COLUMN = config.NUM_OF_CT_PER_COLUMN;
    uint32_t CU_mode = config.CU_mode;
    uint32_t num_of_batches = config.num_of_batches;
    int ARG_Q = config.ARG_Q;
    int ARG_K = config.ARG_K;
    int ARG_V = config.ARG_V;
    int ARG_O = config.ARG_O;
    uint32_t Q_padded_D = config.Q_padded_D;
    uint32_t K_padded_D = config.K_padded_D;
    uint32_t V_padded_D = config.V_padded_D;
    uint32_t O_PADDDED_D = config.O_PADDDED_D;
    bool QKV_reordered = config.QKV_reordered;
    uint32_t Q_total_seq_len_per_head = config.Q_total_seq_len_per_head;
    uint32_t K_total_seq_len_per_head = config.K_total_seq_len_per_head;
    uint32_t V_total_seq_len_per_head = config.V_total_seq_len_per_head;

    const std::vector<int>& SHIM_L_SEQ_list = batch_data.SHIM_L_SEQ_list;
    const std::vector<int>& SHIM_S_SEQ_list = batch_data.SHIM_S_SEQ_list;
    const std::vector<int>& SHIM_S_SEQ_PADDED_list = batch_data.SHIM_S_SEQ_PADDED_list;
    const std::vector<int>& Q_batch_offset_list = batch_data.Q_batch_offset_list;
    const std::vector<int>& K_batch_offset_list = batch_data.K_batch_offset_list;
    const std::vector<int>& V_batch_offset_list = batch_data.V_batch_offset_list;
    const std::vector<int>& O_batch_offset_list = batch_data.O_batch_offset_list;

    if(Q_padded_D < DH*NUM_OF_DH || K_padded_D < DH*NUM_OF_DH || V_padded_D < DH*NUM_OF_DH || O_PADDDED_D < DH*NUM_OF_DH) {
        std::cerr << "padded D less than required D"<< std::endl;
        exit(1);
    }

    if(SHIM_L_SEQ_list.size() != num_of_batches ||
       SHIM_S_SEQ_list.size() != num_of_batches ||
       SHIM_S_SEQ_PADDED_list.size() != num_of_batches){
        std::cerr << "SHIM_L/S_SEQ_LIST size not equal to num_of_batches"<< std::endl;
        exit(1);
    }


    assert(LQ_per_CT % lq_internal ==0);
    int MT_KV_repeat = LQ_per_CT/lq_internal;


    std::vector<npu_tiles> shim_tiles;
    // initialize the shimtiles
    for(int i = 0; i < NUM_OF_COLUMNS; i++){
        shim_tiles.push_back(get_tile(0, i));
    }

    constexpr int CT_lock_address_base = 0x000001F000;

    constexpr npu_it_channel it_for_send_Q = it_channel_0;
    constexpr npu_it_channel it_for_send_KV = it_channel_1; 
    constexpr npu_it_channel it_for_recv_O = it_channel_0;

    constexpr int CT_rtp_lock_id = 10;
    constexpr int CT_rtp_address = 4096;

    // parameter sanity checks
    if(NUM_OF_COLUMNS != 8){
        std::cerr << "Currently only support 8 columns" << std::endl;
        exit(1);
    }
    if(NUM_OF_CT_PER_COLUMN !=4){
        std::cerr << "Currently only support 4 CT per column" << std::endl;
        exit(1);
    }
    // different CU_mode result in different shim->kv broadcast pattern

    std::vector<std::vector<int>>  CU_column;
    if (CU_mode == 0){
        // 1-1
        CU_column = {  {0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}  }; // each column
    }else if(CU_mode == 1){
        // 1-8
        CU_column = {  {0,1,2,3,4,5,6,7}  }; // all columns        
    }else if(CU_mode == 2){
        // 1-4
        CU_column = {  {0,1,2,3}, {4,5,6,7}  }; // each column
    }else if(CU_mode == 3){
        // 1-2
        CU_column = {  {0,1}, {2,3}, {4,5}, {6,7}  }; // each column
    }else{
        std::cerr << "CU_mode not supported" << std::endl;
        exit(1);
    }


    seq.clear_cmds();
    seq.npu_preemption(0);

    int LQ_per_CU = CU_column[0].size() *(LQ_per_CT * NUM_OF_CT_PER_COLUMN);
    

    for(int batch_idx =0; batch_idx < num_of_batches; batch_idx++){
        if(SHIM_L_SEQ_list[batch_idx] % LQ_per_CU !=0){
            std::cerr << "SHIM_L_SEQ must be multiple of LQ_per_CU" << std::endl;
            exit(1);
        }
    }

    std::vector<int> SHIM_Q_queue_counter(NUM_OF_COLUMNS, 0); // counter for each column
    std::vector<int> SHIM_K_queue_counter(NUM_OF_COLUMNS, 0); // counter for each column
    std::vector<int> SHIM_V_queue_counter(NUM_OF_COLUMNS, 0); // counter for each column
    std::vector<int> SHIM_O_queue_counter(NUM_OF_COLUMNS, 0); // counter for each column


    std::vector<int> SHIM_Q_ping_pong(NUM_OF_COLUMNS, 0); // ping pong flag for each column
    std::vector<int> SHIM_K_ping_pong(NUM_OF_COLUMNS, 0); // ping pong flag for each column
    std::vector<int> SHIM_V_ping_pong(NUM_OF_COLUMNS, 0); // ping pong flag for each column
    std::vector<int> SHIM_O_ping_pong(NUM_OF_COLUMNS, 0); // ping pong flag for each column






    for(uint32_t batch_idx = 0; batch_idx < num_of_batches; batch_idx++){



        // # calculate how many SHIM_L chunks we have in the column 
        std::vector<int> LQ_chunk_per_CT_in_column(NUM_OF_COLUMNS, 0);// init to 0

        int NUM_OF_CU_UNITS = CU_column.size();

        for (int head_idx_start = 0; head_idx_start < NUM_OF_DH; head_idx_start+= NUM_OF_CU_UNITS) {
            for (int SHIM_L_chunk_idx = 0; SHIM_L_chunk_idx < SHIM_L_SEQ_list[batch_idx] / LQ_per_CU; SHIM_L_chunk_idx++) {
                for(int cu_idx = 0; cu_idx < CU_column.size(); cu_idx++){
                    int DH_index = head_idx_start + cu_idx;
                    if(DH_index >= NUM_OF_DH){
                        break;
                    }
                    for(auto col_idx : CU_column[cu_idx]){
                        LQ_chunk_per_CT_in_column.at(col_idx)++;
                    }
                }
            }
        }



        //NOTEL for now, we only give SHIM_S_SEQ
        for(auto cur_CU_shimtile_cols : CU_column){
            
            for (auto shim_col_idx: cur_CU_shimtile_cols){
                for(int row_idx= 0; row_idx <  NUM_OF_CT_PER_COLUMN; row_idx++){

                    auto CT_tile = get_tile(row_idx+2, shim_col_idx);
                    seq.rtp_write(CT_tile, CT_rtp_address,  SHIM_S_SEQ_list[batch_idx]); // SHIM_S_SEQ
                    seq.rtp_write(CT_tile, CT_rtp_address+4, LQ_chunk_per_CT_in_column[shim_col_idx] * LQ_per_CT); // how many LQ chunks per CT in this column
                    
                    uint32_t attention_scalar_bits;
                    memcpy(&attention_scalar_bits, &attention_scalar, sizeof(uint32_t));
                    seq.rtp_write(CT_tile, CT_rtp_address+8, attention_scalar_bits);

                    seq.rtp_write(CT_tile, CT_lock_address_base+16*(CT_rtp_lock_id), 1); // set lock to 1
                }


            }
        }



        // uint32_t Q_offset = batch_idx * SHIM_L_SEQ_list[batch_idx]* Q_padded_D + Q_external_data_offset;
        // uint32_t K_offset = batch_idx * SHIM_S_SEQ_PADDED_list[batch_idx] * K_padded_D+ K_external_data_offset;
        // uint32_t V_offset = batch_idx * SHIM_S_SEQ_PADDED_list[batch_idx] * V_padded_D + V_external_data_offset;
        // uint32_t O_offset = batch_idx * SHIM_L_SEQ_list[batch_idx] * O_PADDDED_D + O_external_data_offset;

        uint32_t Q_offset = Q_batch_offset_list[batch_idx];;
        uint32_t K_offset = K_batch_offset_list[batch_idx];
        uint32_t V_offset = V_batch_offset_list[batch_idx];
        uint32_t O_offset = O_batch_offset_list[batch_idx];
  
        

        for (int head_idx_start = 0; head_idx_start < NUM_OF_DH; head_idx_start+= NUM_OF_CU_UNITS)
        {

            for (int SHIM_L_chunk_idx = 0; SHIM_L_chunk_idx < SHIM_L_SEQ_list[batch_idx] / LQ_per_CU; SHIM_L_chunk_idx++)
            {

                if (MT_KV_repeat == 1)
                {
                    if (!QKV_reordered)
                    {
                        process_SHM_Q_chunk_no_Q_repeat<T_in, T_out>(
                            seq,
                            NUM_OF_CT_PER_COLUMN,
                            MT_KV_repeat,
                            DH,
                            NUM_OF_DH,
                            head_idx_start,
                            SHIM_L_chunk_idx * LQ_per_CU,
                            SHIM_S_SEQ_PADDED_list[batch_idx],
                            CU_column,
                            LQ_per_CU,
                            LK_per_CT, // same as LK_CU_chunk_size

                            LQ_per_CT,
                            LK_per_CT,
                            SHIM_Q_queue_counter,
                            SHIM_K_queue_counter,
                            SHIM_V_queue_counter,
                            SHIM_O_queue_counter,
                            SHIM_Q_ping_pong,
                            SHIM_K_ping_pong,
                            SHIM_V_ping_pong,
                            SHIM_O_ping_pong,
                            Q_padded_D, K_padded_D, V_padded_D, O_PADDDED_D,
                            Q_offset, K_offset, V_offset, O_offset,
                            ARG_Q, ARG_K, ARG_V, ARG_O,
                            it_for_send_Q, it_for_send_KV,
                            it_for_recv_O,
                            shim_tiles);
                    }
                    else
                    {
                        process_SHM_Q_chunk_no_Q_repeat_qkv_reorder_NUM_DH_L_DH<T_in, T_out>(
                            seq,
                            NUM_OF_CT_PER_COLUMN,
                            MT_KV_repeat,
                            DH,
                            NUM_OF_DH,
                            head_idx_start,
                            SHIM_L_chunk_idx * LQ_per_CU,
                            SHIM_S_SEQ_PADDED_list[batch_idx],
                            CU_column,
                            LQ_per_CU,
                            LK_per_CT, // same as LK_CU_chunk_size

                            LQ_per_CT,
                            LK_per_CT,
                            SHIM_Q_queue_counter,
                            SHIM_K_queue_counter,
                            SHIM_V_queue_counter,
                            SHIM_O_queue_counter,
                            SHIM_Q_ping_pong,
                            SHIM_K_ping_pong,
                            SHIM_V_ping_pong,
                            SHIM_O_ping_pong,
                            O_PADDDED_D,
                            Q_total_seq_len_per_head, K_total_seq_len_per_head, V_total_seq_len_per_head,
                            Q_offset, K_offset, V_offset, O_offset,
                            ARG_Q, ARG_K, ARG_V, ARG_O,
                            it_for_send_Q, it_for_send_KV,
                            it_for_recv_O,
                            shim_tiles);
                    }
                }
                else
                {
                    if (!QKV_reordered)
                    {
                        process_SHM_Q_chunk_Q_repeat<T_in, T_out>(
                            seq,
                            NUM_OF_CT_PER_COLUMN,
                            MT_KV_repeat,
                            DH,
                            NUM_OF_DH,
                            head_idx_start,
                            SHIM_L_chunk_idx * LQ_per_CU,
                            SHIM_S_SEQ_PADDED_list[batch_idx],
                            CU_column,
                            LQ_per_CU,
                            LK_per_CT, // same as LK_CU_chunk_size

                            LQ_per_CT,
                            LK_per_CT,
                            SHIM_Q_queue_counter,
                            SHIM_K_queue_counter,
                            SHIM_V_queue_counter,
                            SHIM_O_queue_counter,
                            SHIM_Q_ping_pong,
                            SHIM_K_ping_pong,
                            SHIM_V_ping_pong,
                            SHIM_O_ping_pong,
                            Q_padded_D, K_padded_D, V_padded_D, O_PADDDED_D,
                            Q_offset, K_offset, V_offset, O_offset,
                            ARG_Q, ARG_K, ARG_V, ARG_O,
                            it_for_send_Q, it_for_send_KV,
                            it_for_recv_O,
                            shim_tiles);
                    }
                    else
                    {
                        process_SHM_Q_chunk_Q_repeat_reorder_NUM_DH_L_DH<T_in, T_out>(
                            seq,
                            NUM_OF_CT_PER_COLUMN,
                            MT_KV_repeat,
                            DH,
                            NUM_OF_DH,
                            head_idx_start,
                            SHIM_L_chunk_idx * LQ_per_CU,
                            SHIM_S_SEQ_PADDED_list[batch_idx],
                            CU_column,
                            LQ_per_CU,
                            LK_per_CT, // same as LK_CU_chunk_size

                            LQ_per_CT,
                            LK_per_CT,
                            SHIM_Q_queue_counter,
                            SHIM_K_queue_counter,
                            SHIM_V_queue_counter,
                            SHIM_O_queue_counter,
                            SHIM_Q_ping_pong,
                            SHIM_K_ping_pong,
                            SHIM_V_ping_pong,
                            SHIM_O_ping_pong,
                            O_PADDDED_D,
                            Q_total_seq_len_per_head, K_total_seq_len_per_head, V_total_seq_len_per_head,
                            Q_offset, K_offset, V_offset, O_offset,
                            ARG_Q, ARG_K, ARG_V, ARG_O,
                            it_for_send_Q, it_for_send_KV,
                            it_for_recv_O,
                            shim_tiles

                        );
                    }
                }


            }
        }


        // clear and wait for all remaining 
        bool is_all_flushed = false;
        
        while(!is_all_flushed){
            is_all_flushed = true;  // init to True

            // start with q
            for(int idx = 0; idx < NUM_OF_COLUMNS; idx++){
                if(SHIM_Q_queue_counter.at(idx) > 0){
                    is_all_flushed = false;
                    force_wait_DMA_queue(
                        seq, idx, SHIM_Q_queue_counter,
                        shim_tiles.at(idx),
                        MM2S, it_for_send_Q
                    );
                }
            }

            // THEN K, V 
            for(int idx = 0; idx < NUM_OF_COLUMNS; idx++){
                // assert SHIM_K_queue_counter[idx] == 0
                assert(SHIM_K_queue_counter.at(idx) == 0);
                
                if(SHIM_V_queue_counter.at(idx) > 0){
                    is_all_flushed = false;
                    force_wait_DMA_queue(
                        seq, idx, SHIM_V_queue_counter,
                        shim_tiles.at(idx),
                        MM2S, it_for_send_KV
                    );
                }
            }
        }

        is_all_flushed = false;
        while(!is_all_flushed){
            is_all_flushed = true;  // init to True
            
            for(int idx = 0; idx < NUM_OF_COLUMNS; idx++){
                assert(SHIM_O_queue_counter.at(idx) <= 2);
                if(SHIM_O_queue_counter.at(idx) > 0){
                    is_all_flushed = false;
                    force_wait_DMA_queue(
                        seq, idx, SHIM_O_queue_counter,
                        shim_tiles.at(idx),
                        S2MM, it_for_recv_O
                    );
                }
            }
        }


    }











    seq.cmds2seq();

}




#endif