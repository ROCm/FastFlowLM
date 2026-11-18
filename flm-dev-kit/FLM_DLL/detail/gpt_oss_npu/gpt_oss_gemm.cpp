// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include "gpt_oss_gemm.hpp"
#include <cassert>
#include <algorithm> // Required for std::max
#include <string>

// // constructors
GPT_OSS_Gemm::GPT_OSS_Gemm() {
}

GPT_OSS_Gemm::~GPT_OSS_Gemm() = default;

// Assume both in and out is bfl16
void GPT_OSS_Gemm::generate_seq(npu_sequence *seq, uint32_t M, uint32_t K, uint32_t N,
    const uint32_t weight_offset,
    bool ADD_BIAS, int OUTPUT_MODE, const uint32_t bias_offset
){
    uint32_t A_const_offset = 0;
    uint32_t B_const_offset = weight_offset;
    uint32_t C_const_offset =0;
    constexpr int CT_lock_address_base = 0x000001F000;
    const int Arg_C = 0;
    const int Arg_A = 1;
    const int Arg_B = 2;
    const int Arg_Bias = 3;

    const int K_div_k  = K/k;

    // some sanity checks
    assert( M %(m*total_rows) == 0);
    assert(K %k == 0);
    assert(N%(n) == 0);

    seq->clear_cmds();
    seq->npu_preemption(0);

    // mapping of valid shimtile index for A -> index offset
    std::map<int, int> valid_A_MT_shimtile_index;
    valid_A_MT_shimtile_index[0] = 0;
    valid_A_MT_shimtile_index[2] = 1;
    valid_A_MT_shimtile_index[4] = 2;
    valid_A_MT_shimtile_index[6] = 3;

    //create list of tiles
    uint32_t shimtile_size = std::max(total_cols, total_rows);
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

    for(size_t row_idx = 0; row_idx < total_rows; row_idx++){
        for(size_t col_idx = 0; col_idx< total_cols; col_idx++){
            auto CT_tile = get_tile(row_idx+2, col_idx);
            // set RTP value
            seq->rtp_write( CT_tile, CT_rtp_address, K_div_k );
            seq->rtp_write( CT_tile, CT_rtp_address+4, M );
            seq->rtp_write( CT_tile, CT_rtp_address+8, N );
            if(ADD_BIAS){
                seq->rtp_write( CT_tile, CT_rtp_address+12, 1 );
            }else{
                seq->rtp_write( CT_tile, CT_rtp_address+12, 0 );
            }
            seq->rtp_write( CT_tile, CT_rtp_address+16, OUTPUT_MODE );  // OUTPUT MODE
            // set RTP lock
            seq->rtp_write(CT_tile, CT_lock_address_base+16*(CT_rtp_sync_lock_id), 1); // set lock to 1
        }
    }

    generate_runtime_sequence<bf16, bf16>(
        Arg_A, Arg_B, Arg_C, Arg_Bias,
        A_const_offset, B_const_offset, C_const_offset, bias_offset,
        M, N, K, m,n,k,
        total_rows, total_cols,
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
        for(size_t shim_index = 0; shim_index < total_cols; shim_index++){

            if(list_A_shim_queue.at(shim_index) > 0){
                seq->npu_dma_wait(
                    shim_tiles.at(shim_index),
                    MM2S,
                    it_channel_0
                );
                list_A_shim_queue.at(shim_index)--;
            }
            if(list_B_shim_queue.at(shim_index) > 0){
                seq->npu_dma_wait(
                    shim_tiles.at(shim_index),
                    MM2S,
                    it_channel_1
                );
                list_B_shim_queue.at(shim_index)--;
            }
            if(list_C_shim_queue.at(shim_index) > 0){
                seq->npu_dma_wait(
                    shim_tiles.at(shim_index),
                    S2MM,
                    it_channel_0

                );
                list_C_shim_queue.at(shim_index)--;
            }
        }
    }

    seq->cmds2seq();
}
