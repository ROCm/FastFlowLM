#ifndef __WindowAttention_RUNTIME_SEQUENCE_HPP__
#define __WindowAttention_RUNTIME_SEQUENCE_HPP__




#include <cassert>
#include <algorithm> // Required for std::max
#include <string>
#include "npu_utils/npu_instr_utils.hpp"



template<typename T_in, typename T_out>
void window_attention_per_batch(
    npu_sequence &seq,
    int LQ,
    int LK,
    int DH,
    int NUM_HEAD,
    int NUM_OF_COLUMNS,
    int NUM_OF_CT_PER_COLUMN,

    int L_SEQ,
    int S_SEQ,

    int batch_Q_offset, int batch_K_offset, int batch_V_offset, int batch_O_offset,
    std::vector<npu_tiles>& shim_tiles,
    int Q_ARG_IDX,
    int K_ARG_IDX,
    int V_ARG_IDX,
    int O_ARG_IDX
 
){

    constexpr npu_it_channel it_for_send_Q = it_channel_0;
    constexpr npu_it_channel it_for_send_KV = it_channel_1; 
    constexpr npu_it_channel it_for_recv_O = it_channel_0;

    if(L_SEQ != S_SEQ){
        std::cerr << "For window attention, L_SEQ should be equal to S_SEQ"<< std::endl;
        exit(1);
    }



    std::vector<int> O_queue_list(NUM_OF_COLUMNS, 0);
    std::vector<int> bd_list(NUM_OF_COLUMNS, 0);


    int num_of_transaction = NUM_HEAD/NUM_OF_COLUMNS;
    if(NUM_HEAD % NUM_OF_COLUMNS != 0){
        num_of_transaction += 1;
    }




    for (int transaction_idx = 0; transaction_idx < num_of_transaction; transaction_idx++){


        for(int blocks_idx = 0; blocks_idx < L_SEQ/ (LQ*NUM_OF_CT_PER_COLUMN); blocks_idx++){


            for(int cur_col_idx = 0; cur_col_idx < NUM_OF_COLUMNS; cur_col_idx++){
                int head_idx = transaction_idx*NUM_OF_COLUMNS + cur_col_idx;
                if(head_idx >= NUM_HEAD){
                    // all heads are processed
                    break;
                }


                int q_offset = batch_Q_offset + head_idx * DH;
                int k_offset = batch_K_offset + head_idx * DH;
                int v_offset = batch_V_offset + head_idx * DH;
                int o_offset = batch_O_offset + head_idx * DH;

                auto cur_shimtile = shim_tiles[cur_col_idx];
                if(O_queue_list[cur_col_idx] == 2){
                    seq.npu_dma_wait(
                        cur_shimtile, S2MM, it_for_recv_O
                    );
                    O_queue_list[cur_col_idx] -= 1;
                }


                uint32_t q_bd_id, k_bd_id, v_bd_id, o_bd_id;
                if(bd_list[cur_col_idx] == 0){
                    q_bd_id = 0;
                    k_bd_id = 1;
                    v_bd_id = 2;
                    o_bd_id = 3;
                    bd_list[cur_col_idx] = 1;
                }else{
                    q_bd_id = 0+8;
                    k_bd_id = 1+8;
                    v_bd_id = 2+8;
                    o_bd_id = 3+8;
                    bd_list[cur_col_idx] = 0;
                }


                seq.npu_dma_memcpy_nd(
                    sizeof(T_in),
                    Q_ARG_IDX,
                    MM2S, 
                    cur_shimtile,
                    static_cast<npu_bd_id>(q_bd_id),
                    it_for_send_Q,
                    {0,0,0, q_offset+  blocks_idx* (NUM_OF_CT_PER_COLUMN*LQ)*(DH * NUM_HEAD) },
                    {1, 1, (NUM_OF_CT_PER_COLUMN*LQ), DH},
                    {0,0, (DH*NUM_HEAD), 1},
                    -1, 0, false
                );
                seq.npu_dma_memcpy_nd(
                    sizeof(T_in),
                    K_ARG_IDX,
                    MM2S, 
                    cur_shimtile,
                    static_cast<npu_bd_id>(k_bd_id),
                    it_for_send_KV,
                    {0,0,0, k_offset + blocks_idx*(NUM_OF_CT_PER_COLUMN*LK) *(DH* NUM_HEAD)},
                    {1, 1, (NUM_OF_CT_PER_COLUMN*LK), DH},
                    {0,0, (DH*NUM_HEAD), 1},
                    -1, 0, false, aggressive_cache
                );
                seq.npu_dma_memcpy_nd(
                    sizeof(T_in),
                    V_ARG_IDX,
                    MM2S, 
                    cur_shimtile,
                    static_cast<npu_bd_id>(v_bd_id),
                    it_for_send_KV,
                    {0,0,0, v_offset + blocks_idx*(NUM_OF_CT_PER_COLUMN*LK) *(DH* NUM_HEAD)},
                    {1, 1, (NUM_OF_CT_PER_COLUMN*LK), DH},
                    {0,0, (DH*NUM_HEAD), 1},
                    -1, 0, false, aggressive_cache
                );


                seq.npu_dma_memcpy_nd(
                    sizeof(T_out),
                    O_ARG_IDX,
                    S2MM, 
                    cur_shimtile,
                    static_cast<npu_bd_id>(o_bd_id),
                    it_for_recv_O,
                    {0,0,0, o_offset + blocks_idx*(NUM_OF_CT_PER_COLUMN*LQ) *(DH* NUM_HEAD)},
                    {1, 1, (NUM_OF_CT_PER_COLUMN*LQ), DH},
                    {0,0, (DH*NUM_HEAD), 1},
                    -1, 0, true

                );
                O_queue_list[cur_col_idx] += 1;


            }





        }



    }



    int max_remaining = -1;
    for(int col_idx = 0; col_idx < NUM_OF_COLUMNS; col_idx++){
        if(O_queue_list[col_idx] > max_remaining){
            max_remaining = O_queue_list[col_idx];  
        }
    }


    for(int i = 0; i < max_remaining; i++){
        for(int col_idx = 0; col_idx < NUM_OF_COLUMNS; col_idx++){
            if(O_queue_list[col_idx] > 0){
                auto cur_shimtile = shim_tiles[col_idx];
                seq.npu_dma_wait(
                    cur_shimtile, S2MM, it_for_recv_O
                );
                O_queue_list[col_idx] -= 1;
            }
        }
    }



}


template<typename T_in, typename T_out>
void window_attention_per_batch_inner_head(
    npu_sequence &seq,
    int LQ,
    int LK,
    int DH,
    int NUM_HEAD,
    int NUM_OF_COLUMNS,
    int NUM_OF_CT_PER_COLUMN,

    int L_SEQ,
    int S_SEQ,

    int batch_Q_offset, int batch_K_offset, int batch_V_offset, int batch_O_offset,
    std::vector<npu_tiles>& shim_tiles,
    int Q_ARG_IDX,
    int K_ARG_IDX,
    int V_ARG_IDX,
    int O_ARG_IDX
 
){

    constexpr npu_it_channel it_for_send_Q = it_channel_0;
    constexpr npu_it_channel it_for_send_KV = it_channel_1; 
    constexpr npu_it_channel it_for_recv_O = it_channel_0;

    if(L_SEQ != S_SEQ){
        std::cerr << "For window attention, L_SEQ should be equal to S_SEQ"<< std::endl;
        exit(1);
    }



    std::vector<int> O_queue_list(NUM_OF_COLUMNS, 0);
    std::vector<int> bd_list(NUM_OF_COLUMNS, 0);



    // NOTE: This implementation is where # each CT process a single head, not each column process a single head
    if(NUM_HEAD % (NUM_OF_CT_PER_COLUMN) != 0){
        std::cerr << "Currently, window attention inner head only support NUM_HEAD % NUM_OF_CT_PER_COLUMN ==0 "<< std::endl;
        exit(1);
    }


    int num_of_transactions = NUM_HEAD / (NUM_OF_CT_PER_COLUMN * NUM_OF_COLUMNS);   //NOTE: because each CT process a single head
    if(NUM_HEAD %( (NUM_OF_CT_PER_COLUMN * NUM_OF_COLUMNS)) != 0){
        // do a round up
        num_of_transactions += 1;
    }


    for(int transaction_idx = 0; transaction_idx < num_of_transactions; transaction_idx++){


        for(int blocks_idx = 0; blocks_idx < L_SEQ/ (LQ); blocks_idx++){ 

            for(int cur_col_idx = 0; cur_col_idx < NUM_OF_COLUMNS; cur_col_idx++){
                
                // start head_idx of the current transaction
                int head_idx = transaction_idx*(NUM_OF_CT_PER_COLUMN * NUM_OF_COLUMNS) + cur_col_idx* NUM_OF_CT_PER_COLUMN;
                if(head_idx >= NUM_HEAD){
                    // all heads are processed
                    break;;
                }
                int q_offset = batch_Q_offset + head_idx * DH;
                int k_offset = batch_K_offset + head_idx * DH;
                int v_offset = batch_V_offset + head_idx * DH;
                int o_offset = batch_O_offset + head_idx * DH;

                auto cur_shimtile = shim_tiles[cur_col_idx];
                if(O_queue_list[cur_col_idx] == 2){
                    seq.npu_dma_wait(
                        cur_shimtile, S2MM, it_for_recv_O
                    );
                    O_queue_list[cur_col_idx] -= 1;
                }


                uint32_t q_bd_id, k_bd_id, v_bd_id, o_bd_id;
                if(bd_list[cur_col_idx] == 0){
                    q_bd_id = 0;
                    k_bd_id = 1;
                    v_bd_id = 2;
                    o_bd_id = 3;
                    bd_list[cur_col_idx] = 1;
                }else{
                    q_bd_id = 0+8;
                    k_bd_id = 1+8;
                    v_bd_id = 2+8;
                    o_bd_id = 3+8;
                    bd_list[cur_col_idx] = 0;
                }


                seq.npu_dma_memcpy_nd(
                    sizeof(T_in),
                    Q_ARG_IDX,
                    MM2S, 
                    cur_shimtile,
                    static_cast<npu_bd_id>(q_bd_id),
                    it_for_send_Q,
                    {0,0,0, uint32_t(q_offset+  blocks_idx* (LQ)*(DH * NUM_HEAD)) },
                    // {1, 1, (LQ), DH*NUM_OF_CT_PER_COLUMN},
                    // {0,0, (DH*NUM_HEAD), 1},
                    {1, uint32_t(NUM_OF_CT_PER_COLUMN), uint32_t(LQ), uint32_t(DH)}, // transaction of NUM_OF_CT_PER_COLUMN head
                    {0, uint32_t(DH), uint32_t(DH*NUM_HEAD), 1},                
                    -1, 0, false, aggressive_cache
                );
                seq.npu_dma_memcpy_nd(
                    sizeof(T_in),
                    K_ARG_IDX,
                    MM2S, 
                    cur_shimtile,
                    static_cast<npu_bd_id>(k_bd_id),
                    it_for_send_KV,
                    {0,0,0, uint32_t(k_offset + blocks_idx*(LK) *(DH* NUM_HEAD))},
                    // {1, 1, (LK), DH*NUM_OF_CT_PER_COLUMN},
                    // {0,0, (DH*NUM_HEAD), 1},
                    {1, (u32)NUM_OF_CT_PER_COLUMN, u32(LK),        u32(DH)},
                    {0, u32(DH), u32(DH*NUM_HEAD), (u32)1},
                    -1, 0, false, aggressive_cache
                );
                seq.npu_dma_memcpy_nd(
                    sizeof(T_in),
                    V_ARG_IDX,
                    MM2S, 
                    cur_shimtile,
                    static_cast<npu_bd_id>(v_bd_id),
                    it_for_send_KV,
                    {0,0,0, uint32_t(v_offset + blocks_idx*(LK) *(DH* NUM_HEAD))},
                    // {1, 1, (LK), DH*NUM_OF_CT_PER_COLUMN},
                    // {0,0, (DH*NUM_HEAD), 1},
                    {1, uint32_t(NUM_OF_CT_PER_COLUMN), uint32_t(LK), uint32_t(DH)},
                    {0, uint32_t(DH), uint32_t(DH*NUM_HEAD), uint32_t(1)},
                    -1, 0, false, aggressive_cache
                );


                seq.npu_dma_memcpy_nd(
                    sizeof(T_out),
                    O_ARG_IDX,
                    S2MM, 
                    cur_shimtile,
                    static_cast<npu_bd_id>(o_bd_id),
                    it_for_recv_O,
                    {0,0,0, uint32_t(o_offset + blocks_idx*(LQ) *(DH* NUM_HEAD))},
                    // {1, 1, (LQ), DH*NUM_OF_CT_PER_COLUMN},
                    // {0,0, (DH*NUM_HEAD), 1},
                    {1, uint32_t(NUM_OF_CT_PER_COLUMN), uint32_t(LQ), uint32_t(DH)},
                    {0, uint32_t(DH), uint32_t(DH*NUM_HEAD), uint32_t(1)},
                    -1, 0, true

                );
                O_queue_list[cur_col_idx] += 1;

            }


        }


    }


    int max_remaining = -1;
    for(int col_idx = 0; col_idx < NUM_OF_COLUMNS; col_idx++){
        if(O_queue_list[col_idx] > max_remaining){
            max_remaining = O_queue_list[col_idx];  
        }
    }


    for(int i = 0; i < max_remaining; i++){
        for(int col_idx = 0; col_idx < NUM_OF_COLUMNS; col_idx++){
            if(O_queue_list[col_idx] > 0){
                auto cur_shimtile = shim_tiles[col_idx];
                seq.npu_dma_wait(
                    cur_shimtile, S2MM, it_for_recv_O
                );
                O_queue_list[col_idx] -= 1;
            }
        }
    }



}





template<typename T_in, typename T_out>
void window_attention_runtimeSequence(
    npu_sequence &seq,
    int batch_size,
    int LQ,
    int LK,
    int DH,
    int NUM_HEAD,
    int NUM_OF_COLUMNS,
    int NUM_OF_CT_PER_COLUMN,
    std::vector<int> L_SEQ_per_batch,
    std::vector<int> S_SEQ_per_batch,

    int external_Q_offset,
    int external_K_offset,
    int external_V_offset,
    int external_O_offset,


    int Q_ARG_IDX,
    int K_ARG_IDX,
    int V_ARG_IDX,
    int O_ARG_IDX

){


    // Note 
    // sanity checks
    
    if(L_SEQ_per_batch.size() != batch_size || S_SEQ_per_batch.size() != batch_size){
        std::cerr << "batch size not match with L_SEQ_per_batch or S_SEQ_per_batch size"<< std::endl;
        exit(1);
    }


    for(int i = 0; i < batch_size; i++){
        if(L_SEQ_per_batch[i] % (LQ * NUM_OF_CT_PER_COLUMN) !=0){
            std::cerr << "L_SEQ_per_batch["<<i<<"] not divisible by L_Q * NUM_OF_CT_PER_COLUMN"<< std::endl;
            exit(1);
        }
        if(S_SEQ_per_batch[i] % (LK * NUM_OF_CT_PER_COLUMN) !=0){
            std::cerr << "S_SEQ_per_batch["<<i<<"] not divisible by L_K * NUM_OF_CT_PER_COLUMN"<< std::endl;
            exit(1);
        }
    }


    seq.clear_cmds();
    seq.npu_preemption(0);
    std::vector<npu_tiles> shim_tiles;
    // initialize the shimtiles
    for(int i = 0; i < NUM_OF_COLUMNS; i++){
        shim_tiles.push_back(get_tile(0, i));
    }



    int batch_Q_offset = external_Q_offset;
    int batch_K_offset = external_K_offset;
    int batch_V_offset = external_V_offset;
    int batch_O_offset = external_O_offset;
    for(int b = 0; b< batch_size; b++){
        

        // window_attention_per_batch<T_in, T_out>(
        //     seq,
        //     LQ, LK, DH, NUM_HEAD, NUM_OF_COLUMNS, NUM_OF_CT_PER_COLUMN,
        //     L_SEQ_per_batch[b], S_SEQ_per_batch[b],
        //     batch_Q_offset, batch_K_offset, batch_V_offset, batch_O_offset,
        //     shim_tiles,
        //     Q_ARG_IDX, K_ARG_IDX, V_ARG_IDX, O_ARG_IDX
        // );
        window_attention_per_batch_inner_head<T_in, T_out>(
            seq,
            LQ, LK, DH, NUM_HEAD, NUM_OF_COLUMNS, NUM_OF_CT_PER_COLUMN,
            L_SEQ_per_batch[b], S_SEQ_per_batch[b],
            batch_Q_offset, batch_K_offset, batch_V_offset, batch_O_offset,
            shim_tiles,
            Q_ARG_IDX, K_ARG_IDX, V_ARG_IDX, O_ARG_IDX
        );        
        batch_Q_offset += L_SEQ_per_batch[b] * DH * NUM_HEAD;
        batch_K_offset += S_SEQ_per_batch[b] * DH * NUM_HEAD;
        batch_V_offset += S_SEQ_per_batch[b] * DH * NUM_HEAD;
        batch_O_offset += L_SEQ_per_batch[b] * DH * NUM_HEAD;

    }


    seq.cmds2seq();
}




#endif