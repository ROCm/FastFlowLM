// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __MM_SEQUENCE_HPP__
#define  __MM_SEQUENCE_HPP__
#include <cassert>
#include <algorithm> // Required for std::max
#include <string>
#include "npu_utils/npu_utils.hpp"






template <typename T_in, typename T_out>
void generate_shimtile_sequence_per_k_block(
    uint32_t shim_index, uint32_t total_npu_row, uint32_t total_npu_col,
    uint32_t mega_block_row_idx, uint32_t mega_block_col_idx,
    uint32_t M_size, uint32_t K_size, uint32_t N_size,
    uint32_t m, uint32_t k, uint32_t n,
    uint32_t Arg_A, uint32_t  Arg_B, uint32_t Arg_C,
    uint32_t A_const_offset, uint32_t B_const_offset,  uint32_t C_const_offset,
    uint32_t bias_offset,

    std::vector<int> &list_A_shim_queue, std::vector<int> &list_B_shim_queue, std::vector<int> &list_C_shim_queue,
    std::vector<int> &list_A_bd_pingpong_flag, std::vector<int> &list_B_bd_pingpong_flag, std::vector<int> &list_C_bd_pingpong_flag,

    bool IS_B_ROW_MAJOR, bool ENABLE_AXI4,
    bool B_in_K_N_block_col_major_order,
    bool VALID_COLUMN,
    bool ADD_BIAS, bool SEND_BIAS,
    std::map<int, int>& valid_A_MT_shimtile_index,
    npu_sequence& seq,
    std::vector<npu_tiles> & shimtile_list
){

    if(B_in_K_N_block_col_major_order){
        assert(IS_B_ROW_MAJOR== false); // on valid for B in col major order
    }
    // When B_in_K_N_block_col_major_order is set to true, it mean
    // B is col-major order &&
    // B is rearrange into kxn blocks, where blocks are in col-major. Moreover, the data in each blocks is 
    // also in col-major order.

    // Basically,B as a col-major matrix goes through
    // stride: [N_size/n,K_size/k  ,n,     k]
    // offset: [K_size*n,k         ,K_SIZE, 1]

    
    // uint32_t AXI_FLAG = 0x2;// 0b1110<<24; 
    // uint32_t DMA_BD_x_5 = 0x000001D014;
    uint32_t K_div_k = K_size/k;

    npu_tiles cur_shimtile = shimtile_list.at(shim_index);

    if (valid_A_MT_shimtile_index.contains(shim_index) && valid_A_MT_shimtile_index[shim_index] < total_npu_row){
        

        if (list_A_shim_queue.at(shim_index) == 2) {   
            seq.npu_dma_wait(
                cur_shimtile, MM2S, it_channel_0
            );
            list_A_shim_queue.at(shim_index)--;
        }


        uint32_t A_offset = mega_block_row_idx *(total_npu_row*m) *K_size;
        A_offset += valid_A_MT_shimtile_index[shim_index]*(m*K_size);
        npu_bd_id A_bd_id;
        if (list_A_bd_pingpong_flag.at(shim_index) ==0){
            A_bd_id = bd_0;
            list_A_bd_pingpong_flag.at(shim_index) =1;
        }else{
            A_bd_id = bd_1;
            list_A_bd_pingpong_flag.at(shim_index) =0;
        }


        seq.npu_dma_memcpy_nd(
            sizeof(T_in),   // bfloat16
            Arg_A, 
            MM2S,
            cur_shimtile,
            A_bd_id,
            it_channel_0,
            {0,0,0,A_offset+ A_const_offset},
            {1, K_div_k, m,k},
            {0, k, K_size, 1},
            -1, 0, true,
            ENABLE_AXI4 ? aggressive_cache: no_cache
        );
        list_A_shim_queue.at(shim_index)++;

        // if( ENABLE_AXI4){
        //     seq.npu_maskwrite(
        //         cur_shimtile,  DMA_BD_x_5 + 32*(uint32_t)(A_bd_id), AXI_FLAG, 0xF000000
        //     );
        // }
    }

    if(shim_index < total_npu_col && VALID_COLUMN){ 
        
        
        if(SEND_BIAS){
            if (list_B_shim_queue[shim_index] == 2){

                seq.npu_dma_wait(
                    cur_shimtile, MM2S, it_channel_1
                );
                list_B_shim_queue[shim_index] -= 1;
            }
            uint32_t _BIAS_DATA_OFFSET = bias_offset+  mega_block_col_idx * (total_npu_col*n) +  shim_index *n;
            seq.npu_dma_memcpy_nd(
                sizeof(T_in),
                Arg_B,
                MM2S,
                cur_shimtile,
                npu_bd_id(bd_6), //reserved for sending bias
                it_channel_1,
                {0,0,0,_BIAS_DATA_OFFSET},
                {1, 1,1, k*n},
                {0, 0, 0, 1},
                -1, 0, true,
                ENABLE_AXI4 ? aggressive_cache: no_cache
            );
            list_B_shim_queue[shim_index]++;
        }

        npu_bd_id b_bd_id;
        if (list_B_shim_queue[shim_index] == 2){

            seq.npu_dma_wait(
                cur_shimtile, MM2S, it_channel_1
            );
            list_B_shim_queue[shim_index] -= 1;
        }


        if (list_B_bd_pingpong_flag[shim_index] == 0) {
            b_bd_id = bd_2;
            list_B_bd_pingpong_flag[shim_index] = 1;
        } else {
            b_bd_id = bd_3;
            list_B_bd_pingpong_flag[shim_index] = 0;
        }
        
        if (IS_B_ROW_MAJOR){
            uint32_t B_offset = mega_block_col_idx* (total_npu_col) * n;
            B_offset += shim_index * n;
            seq.npu_dma_memcpy_nd(
                sizeof(T_in),
                Arg_B,
                MM2S,
                cur_shimtile,
                b_bd_id,
                it_channel_1,
                {0,0,0,B_offset+ B_const_offset },
                {1, K_div_k, k, n},
                {0, k*N_size, N_size, 1},
                -1, 0, true,
                ENABLE_AXI4 ? aggressive_cache: no_cache
            );
        }else{
            uint32_t B_offset = mega_block_col_idx*(total_npu_col*n)*K_size;
            B_offset += shim_index * n*K_size;
            if(B_in_K_N_block_col_major_order){
                seq.npu_dma_memcpy_nd(
                    sizeof(T_in),
                    Arg_B,
                    MM2S,
                    cur_shimtile,
                    b_bd_id,
                    it_channel_1,
                    {0,0,0,B_offset+ B_const_offset },
                    {1, 1,1, K_div_k* n*k},
                    {0, 0, 0, 1},  
                    -1, 0, true,
                    ENABLE_AXI4 ?  aggressive_cache: no_cache
                );
            }else{
                seq.npu_dma_memcpy_nd(
                    sizeof(T_in),
                    Arg_B,
                    MM2S,
                    cur_shimtile,
                    b_bd_id,
                    it_channel_1,
                    {0,0,0,B_offset+ B_const_offset },
                    {1, K_div_k, n, k},
                    {0, k, K_size, 1},
                    -1, 0, true,
                    ENABLE_AXI4 ?  aggressive_cache: no_cache
                );
            }

        }
        list_B_shim_queue[shim_index]++;
        // if (ENABLE_AXI4){
        //     seq.npu_maskwrite(
        //         cur_shimtile, DMA_BD_x_5 + 32*(uint32_t)(b_bd_id), AXI_FLAG, 0xF000000
        //     );
        // }

    }

    uint32_t C_offset = mega_block_col_idx*n*total_npu_col;
    C_offset += mega_block_row_idx* m*total_npu_row*N_size;
    C_offset += shim_index *n;

    if (shim_index < total_npu_col && VALID_COLUMN){
        
        if (list_C_shim_queue.at(shim_index) == 2) {   
            seq.npu_dma_wait(
                cur_shimtile, S2MM, it_channel_0
            );
            list_C_shim_queue.at(shim_index)--;
        }

        npu_bd_id c_bd_id;
        if(list_C_bd_pingpong_flag.at(shim_index) ==0){
            c_bd_id = bd_14;
            list_C_bd_pingpong_flag.at(shim_index) =1;
        }else{
            c_bd_id = bd_15;
            list_C_bd_pingpong_flag.at(shim_index) =0;
        }

        
        seq.npu_dma_memcpy_nd(
            sizeof(T_out),
            Arg_C,
            S2MM, 
            cur_shimtile,
            c_bd_id,
            it_channel_0,
            {0,0,0, C_offset+ C_const_offset},
            {1,1,4*m, n},
            {0,0,N_size, 1},
            -1, 0, true,
            ENABLE_AXI4 ? aggressive_cache: no_cache
        );
        // if( ENABLE_AXI4){
        //     seq.npu_maskwrite(
        //         cur_shimtile, DMA_BD_x_5 + 32*(uint32_t)(c_bd_id), AXI_FLAG, 0xF000000
        //     );
        // }
        list_C_shim_queue.at(shim_index)++;
    }

}




template <typename T_in, typename T_out>
void generate_runtime_sequence(
    uint32_t Arg_A, uint32_t Arg_B, uint32_t Arg_C,
    uint32_t A_const_offset, uint32_t B_const_offset,  uint32_t C_const_offset,    
    uint32_t bias_offset,  
    uint32_t M_size, uint32_t N_size, uint32_t K_size,
    uint32_t m, uint32_t n, uint32_t k,
    uint32_t total_npu_row, uint32_t total_npu_col,
    std::vector<int> &list_A_shim_queue,
    std::vector<int> &list_B_shim_queue,
    std::vector<int> &list_C_shim_queue,
    
    bool IS_B_ROW_MAJOR,  bool ENABLE_AXI4, bool B_in_K_N_block_col_major_order,
    bool ADD_BIAS,
    std::map<int, int>& valid_A_MT_shimtile_index,
    npu_sequence& seq,
    std::vector<npu_tiles> &shim_tiles
){



    uint32_t M_div_num_row_m = M_size/(m*total_npu_row);
    uint32_t N_div_num_col_n = N_size/(n*total_npu_col);

    uint32_t N_div_num_col_n_remainder_blocks =  (N_size % (n*total_npu_col))/ n;


    std::vector<int> list_A_BD_pingpong_flag(std::max(total_npu_row, total_npu_col), 0);
    std::vector<int> list_B_BD_pingpong_flag(std::max(total_npu_row, total_npu_col), 0);
    std::vector<int> list_C_BD_pingpong_flag(std::max(total_npu_row, total_npu_col), 0);

    uint32_t col_block_range = N_div_num_col_n;
    if (N_div_num_col_n_remainder_blocks!= 0){
        col_block_range += 1;
    }

    for(uint32_t mega_block_col_idx=0; mega_block_col_idx<col_block_range; mega_block_col_idx++){
        
        for (uint32_t mega_block_row_idx = 0; mega_block_row_idx < M_div_num_row_m; mega_block_row_idx++) {
            
            for (uint32_t shim_index = 0; shim_index < std::max(total_npu_col, total_npu_row); shim_index++) {
                bool SEND_ADD_BIAS = false;
                if (mega_block_row_idx == 0 &&ADD_BIAS){
                    SEND_ADD_BIAS = true;
                }                  
                if (N_div_num_col_n_remainder_blocks!= 0 && mega_block_col_idx == N_div_num_col_n){
                        
                    if (shim_index < N_div_num_col_n_remainder_blocks){
 

                        generate_shimtile_sequence_per_k_block<T_in, T_out>(
                            shim_index,
                            total_npu_row, total_npu_col,
                            mega_block_row_idx, mega_block_col_idx,
                            M_size, K_size, N_size,
                            m, k, n,
                            Arg_A, Arg_B, Arg_C,
                            A_const_offset, B_const_offset, C_const_offset, bias_offset,
                            list_A_shim_queue, list_B_shim_queue, list_C_shim_queue,
                            list_A_BD_pingpong_flag, list_B_BD_pingpong_flag, list_C_BD_pingpong_flag,
                            IS_B_ROW_MAJOR, ENABLE_AXI4,
                            B_in_K_N_block_col_major_order, 
                            true,
                            ADD_BIAS, SEND_ADD_BIAS,
                            valid_A_MT_shimtile_index,
                            seq, shim_tiles
                        
                        );
                    }
                       
                    else{
                        generate_shimtile_sequence_per_k_block<T_in, T_out>(
                            shim_index,
                            total_npu_row, total_npu_col,
                            mega_block_row_idx, mega_block_col_idx,
                            M_size, K_size, N_size,
                            m, k, n,
                            Arg_A, Arg_B, Arg_C,
                            A_const_offset, B_const_offset, C_const_offset, bias_offset,
                            list_A_shim_queue, list_B_shim_queue, list_C_shim_queue,
                            list_A_BD_pingpong_flag, list_B_BD_pingpong_flag, list_C_BD_pingpong_flag,
                            IS_B_ROW_MAJOR, ENABLE_AXI4,
                            B_in_K_N_block_col_major_order, 
                            false,
                            ADD_BIAS, SEND_ADD_BIAS,
                            valid_A_MT_shimtile_index,
                            seq, shim_tiles
                        );
                    }

                }
                else{
                    generate_shimtile_sequence_per_k_block<T_in, T_out>(
                        shim_index,
                        total_npu_row, total_npu_col,
                        mega_block_row_idx, mega_block_col_idx,
                        M_size, K_size, N_size,
                        m, k, n,
                        Arg_A, Arg_B, Arg_C,
                        A_const_offset, B_const_offset, C_const_offset, bias_offset,
                        list_A_shim_queue, list_B_shim_queue, list_C_shim_queue,
                        list_A_BD_pingpong_flag, list_B_BD_pingpong_flag, list_C_BD_pingpong_flag,
                        IS_B_ROW_MAJOR, ENABLE_AXI4,
                        B_in_K_N_block_col_major_order, 
                        true,
                        ADD_BIAS, SEND_ADD_BIAS,
                        valid_A_MT_shimtile_index,
                        seq, shim_tiles
                    );

                }
            }

        }
    } 
        



}




template <typename T_in, typename T_out>
void generate_mm_sequence(npu_sequence &seq, uint32_t M, uint32_t K, uint32_t N, 
    uint32_t m, uint32_t k, uint32_t n,
    uint32_t r, uint32_t s, uint32_t t,
    uint32_t CT_rtp_address, uint32_t CT_rtp_sync_lock_id,
    uint32_t total_row, uint32_t total_col,
    uint32_t A_const_offset, uint32_t B_const_offset,  uint32_t C_const_offset, uint32_t bias_offset,
    bool IS_B_ROW_MAJOR, bool ENABLE_AXI4,  bool B_in_K_N_block_col_major_order, 
    bool ADD_BIAS, int OUTPUT_MODE, int OUTPUT_CLAMP,
    float output_clamp_min, float output_clamp_max,
    std::string fileoutputName
){

    constexpr int CT_lock_address_base = 0x000001F000;
    const int Arg_A = 0;
    const int Arg_B = 1;
    const int Arg_C = 2;

    const int K_div_k  = K/k;


    // some sanity checks
    assert( M %(m*total_row) == 0);
    assert(K %k == 0);
    assert(N%(n) == 0);
    
    seq.clear_cmds();

    //debug
    assert(total_row == 4);
    assert(total_col == 8);
    // mapping of valid shimtile index for A -> index offset
    std::map<int, int> valid_A_MT_shimtile_index;
    valid_A_MT_shimtile_index[0] = 0;
    valid_A_MT_shimtile_index[2] = 1;
    valid_A_MT_shimtile_index[4] = 2;
    valid_A_MT_shimtile_index[6] = 3;


    //create list of tiles
    uint32_t shimtile_size = std::max(total_col, total_row);
    std::vector<npu_tiles> shim_tiles;

    std::vector<int> list_C_shim_queue;  // int counter of how many DMA_Wait for C
    std::vector<int> list_A_shim_queue;  // int counter of how many DMA_Wait for A
    std::vector<int> list_B_shim_queue;  // int counter of how many DMA_Wait for B
    for(size_t i = 0; i < shimtile_size; i++){
        shim_tiles.push_back((get_tile(0, i)));
        list_C_shim_queue.push_back(0);
        list_A_shim_queue.push_back(0);
        list_B_shim_queue.push_back(0);
    }
    
    // first, setup the rtp buffer and the rtp locks

    for(size_t row_idx = 0; row_idx < total_row; row_idx++){
        for(size_t col_idx = 0; col_idx< total_col; col_idx++){
            auto CT_tile = get_tile(row_idx+2, col_idx);
            // set RTP value
            seq.rtp_write( CT_tile, CT_rtp_address, K_div_k );
            seq.rtp_write( CT_tile, CT_rtp_address+4, M );
            seq.rtp_write( CT_tile, CT_rtp_address+8, N );
            if(ADD_BIAS){
                seq.rtp_write( CT_tile, CT_rtp_address+12, 1 );
            }else{
                seq.rtp_write( CT_tile, CT_rtp_address+12, 0 );
            }                     
            seq.rtp_write( CT_tile, CT_rtp_address+16, OUTPUT_MODE );  // OUTPUT MODE
            seq.rtp_write( CT_tile, CT_rtp_address+20, OUTPUT_CLAMP); // OUTPUT CLAMP 0 means no clamp, 1 means clamp

            int32_t output_min_int, output_max_int;
            std::memcpy(&output_min_int, &output_clamp_min, sizeof(int32_t));
            std::memcpy(&output_max_int, &output_clamp_max, sizeof(int32_t));

            
            seq.rtp_write( CT_tile, CT_rtp_address+24, output_min_int ); // output clamp min value in float32
            seq.rtp_write( CT_tile, CT_rtp_address+28, output_max_int ); // output clamp max value in float32
            // set RTP lock
            seq.rtp_write(CT_tile, CT_lock_address_base+16*(CT_rtp_sync_lock_id), 1); // set lock to 1
        }
    }

    generate_runtime_sequence<T_in, T_out>(
        Arg_A, Arg_B, Arg_C,
        A_const_offset, B_const_offset, C_const_offset, bias_offset,
        M, N, K, m,n,k,
        total_row, total_col, 
        list_A_shim_queue, list_B_shim_queue,
        list_C_shim_queue, 
        IS_B_ROW_MAJOR, ENABLE_AXI4, B_in_K_N_block_col_major_order,
        ADD_BIAS,
        valid_A_MT_shimtile_index, seq, shim_tiles
    );

    int max_C_remain =  0;
    for( auto li: list_C_shim_queue){

        max_C_remain = std::max(max_C_remain, li);
    }

    for(size_t k = 0; k< max_C_remain; k++){
        for(size_t shim_index = 0; shim_index < total_col; shim_index++){
            
            if(list_A_shim_queue.at(shim_index) > 0){
                seq.npu_dma_wait(
                    shim_tiles.at(shim_index),
                    MM2S,
                    it_channel_0
                );
                list_A_shim_queue.at(shim_index)--;
            }
            if(list_B_shim_queue.at(shim_index) > 0){
                seq.npu_dma_wait(
                    shim_tiles.at(shim_index),
                    MM2S,
                    it_channel_1
                );
                list_B_shim_queue.at(shim_index)--;
            }
            if(list_C_shim_queue.at(shim_index) > 0){
                seq.npu_dma_wait(
                    shim_tiles.at(shim_index),
                    S2MM,
                    it_channel_0

                );
                list_C_shim_queue.at(shim_index)--;
            }
        }

    }

    seq.cmds2seq();
    seq.write_out_sequence(fileoutputName);

}


#endif