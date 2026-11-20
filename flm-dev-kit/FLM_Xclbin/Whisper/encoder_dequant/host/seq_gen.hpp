// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#pragma once
#include "npu_utils/npu_utils.hpp"
#define ENABLE_AXI4

void gen_layernorm_engine_seq(npu_sequence* seq, const uint32_t L_end){
    constexpr int D = 1280;
    constexpr int total_cols = 1;
    constexpr int total_rows = 1;
    constexpr npu_tiles IT[2] = {IT0};
    seq->clear_cmds();

    uint32_t AXI_FLAG = 0b1110<<24; 
    uint32_t DMA_BD_x_5 = 0x000001D014;

    size_t wb_offset = 0;
    seq->npu_dma_memcpy_nd(
        2, 2,
        MM2S, IT[0],
        (npu_bd_id)(2), it_channel_1,
        {0, 0, 0, (uint32_t)wb_offset},
        {1, 1, (uint32_t)1, (uint32_t)2 * D},
        {0, 0, (uint32_t)0, 1},
        -1, 0, false
    );
    for (int round = 0; round < L_end; round++){
        int bd_offset = (round % 2) * 8; 
        for (int col = 0; col < 1; col++){
            // receive y
            size_t y_offset = round * D + col * D; // y does not have history
            seq->npu_dma_memcpy_nd(
                2, 0,
                S2MM, IT[col],
                (npu_bd_id)(bd_offset + 0), it_channel_0,
                {0, 0, 0, (uint32_t)y_offset},
                {1, 1, (uint32_t)1, (uint32_t)D},
                {0, 0, (uint32_t)D, 1},
                -1, 0, true
            );
            // send in
            size_t in_offset = round * D + col * D;
            seq->npu_dma_memcpy_nd(
                2, 1,
                MM2S, IT[col],
                (npu_bd_id)(bd_offset + 1), it_channel_0,
                {0, 0, 0, (uint32_t)in_offset},
                {1, 1, (uint32_t)1, (uint32_t)D},
                {0, 0, (uint32_t)D, 1},
                -1, 0, false
            );
        } // end for col

        if (round > 0){
            for (int col = 0; col < 1; col++){
                seq->npu_dma_wait(
                    IT[col],
                    S2MM,
                    it_channel_0
                );
            }
        }
    }
    for (int col = 0; col < 1; col++){
        seq->npu_dma_wait(
            IT[col],
            S2MM,
            it_channel_0
        );
    }
    seq->cmds2seq();
}

void gen_dequant_q4_1_seq(npu_sequence* seq_ptr, const u32 D_in, const u32 D_out, const u32 weight_offset){
    npu_sequence& seq = *seq_ptr;
    const int w_out_arg_idx = 0;
    const int qw_in_arg_idx = 1;
    // N is din, M is dout
    constexpr int m_tile = 32;
    constexpr int k_tile = 128;
    constexpr int total_cols = 8;
    constexpr npu_tiles IT[8] = {IT0, IT1, IT2, IT3, IT4, IT5, IT6, IT7};
    const int block_size_in_byte = m_tile * k_tile * 5 / 8;
    const int blocks_per_row = D_in / k_tile;
    assert(D_in % (k_tile * 2) == 0);
    const int quant_in_per_column = blocks_per_row * block_size_in_byte;
    int bd_wait_counter[8] = {0,0,0,0,0,0,0,0};
    
    const int row_per_round = 2 * m_tile * total_cols;
    const int down_rounds = (D_out + row_per_round - 1) / row_per_round;
    const int total_column_rounds = D_out / 2 / m_tile;
    std::cout << "row_per_round: " << row_per_round << std::endl;
    std::cout << "down_rounds: " << down_rounds << std::endl;
    std::cout << "total_column_rounds: " << total_column_rounds << std::endl;
    assert(D_out % (2 * m_tile) == 0);
    seq.clear_cmds();
    // first, the dequant of down 
    for(int i = 0; i < down_rounds; i++){

        for(int col = 0; col < 8; col++){
            uint32_t bd_offset = bd_wait_counter[col] * 8;
            uint32_t round_offset = i * 8 + col;

            if(round_offset < total_column_rounds){
                // std::cout << "push dma " << col << " " << round_offset << std::endl;

                uint32_t output_offset_0 = round_offset * 64 * D_in;
                seq.npu_dma_memcpy_nd(
                    2,
                    w_out_arg_idx,
                    S2MM,
                    IT[col],
                    (npu_bd_id)(2+bd_offset),
                    it_channel_0,
                    {0, 0, 0, output_offset_0},
                    {1, 1, 1, 64 * D_in},
                    {0, 0, 0, 1},
                    -1, 0, true
                );
               
                seq.npu_dma_memcpy_nd(
                    sizeof(char),
                    qw_in_arg_idx,
                    MM2S, 
                    IT[col],
                    (npu_bd_id)(0+bd_offset),
                    it_channel_0,
                    {0, 0, 0, weight_offset + round_offset * quant_in_per_column * 2}, 
                    {1, 1, 1, (uint32_t)quant_in_per_column}, //NOTE: being rearrange beforehadn already
                    {0, 0, 0, 1},
                    -1,0, false

                );

                seq.npu_dma_memcpy_nd(
                    sizeof(char),
                    qw_in_arg_idx,
                    MM2S, 
                    IT[col],
                    (npu_bd_id)(1+bd_offset),
                    it_channel_1,
                    {0, 0, 0, weight_offset + round_offset * quant_in_per_column * 2 + quant_in_per_column}, 
                    {1, 1, 1, (uint32_t)quant_in_per_column}, //NOTE: being rearrange beforehadn already
                    {0, 0, 0, 1},
                    -1,0, false

                );
                bd_wait_counter[col]++;
            }
        }
        // std::cout << "dma queue: " << std::flush;
        // for (int j = 0; j < 8; j++){
        //     std::cout << bd_wait_counter[j] << " ";
        // }
        // std::cout << std::endl;
        // note: for now
        for(int col = 0; col < 8; col++){
            if(bd_wait_counter[col] == 2){
                seq.npu_dma_wait(IT[col], S2MM, it_channel_0);
                // std::cout << "wait dma " << col << std::endl;
                bd_wait_counter[col]--;
            }
        }
    }
    int max_loop_iterations = 0;
    // find the global max loop iterations
    for(int col = 0; col < 8; col++){
        max_loop_iterations = std::max(max_loop_iterations, bd_wait_counter[col]);
    }
    for(int i = 0; i < max_loop_iterations; i++){
        for(int col = 0; col < 8; col++){
            if (bd_wait_counter[col] > 0){
                seq.npu_dma_wait(IT[col], S2MM, it_channel_0);
                // std::cout << "wait dma " << col << std::endl;
                bd_wait_counter[col]--;
            }
        }
    }
    seq.cmds2seq();
}