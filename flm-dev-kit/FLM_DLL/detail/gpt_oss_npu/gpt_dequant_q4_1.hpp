// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#pragma once
#include "gpt_oss_params.hpp"

#include "npu_utils/npu_instr_utils.hpp"

void generate_dequant_q4_1_seq(npu_sequence* seq_ptr, const u32 D_in, const u32 D_out, const u32 weight_offset){
    npu_sequence& seq = *seq_ptr;
    const int w_out_arg_idx = 0;
    const int qw_in_arg_idx = 1;
    // N is din, M is dout
    constexpr int m_tile = 32;
    constexpr int k_tile = 128;
    constexpr int total_cols = 8;
    constexpr npu_tiles IT[8] = {IT0, IT1, IT2, IT3, IT4, IT5, IT6, IT7};
    const uint32_t block_size_in_byte = m_tile * k_tile * 5 / 8;
    const uint32_t blocks_per_row = D_in / k_tile;
    assert(D_in % k_tile == 0);
    const int quant_in_per_column = 4 * blocks_per_row * block_size_in_byte;
    int bd_wait_counter[8] = {0,0,0,0,0,0,0,0};

    const int row_per_round = 4 * m_tile * total_cols;
    const int down_rounds = (D_out + row_per_round - 1) / row_per_round;
    const int total_column_rounds = D_out / 4 / m_tile;

    if((2*block_size_in_byte)%128 !=0){
        std::cout << "dequant q4_1: block size not aligned" << std::endl;
        exit(1);
    }
    seq.clear_cmds();
    seq.npu_preemption(0);

    constexpr uint32_t dequant_rtp_address  = 35328;
    for(size_t row_idx = 0; row_idx < 4; row_idx++){
        for(size_t col_idx = 0; col_idx< 8; col_idx++){
            auto CT_tile = get_tile(row_idx+2, col_idx);
            // set RTP value
            seq.rtp_write( CT_tile, dequant_rtp_address, 0 );
        }
    }
    // first, the dequant of down
    for(int i = 0; i < down_rounds; i++){

        for(int col = 0; col < 8; col++){
            uint32_t bd_offset = (i % 2) * 8;
            uint32_t round_offset = i * 8 + col;

            if(round_offset < total_column_rounds){
                seq.npu_dma_memcpy_nd(
                    sizeof(char),
                    qw_in_arg_idx,
                    MM2S,
                    IT[col],
                    (npu_bd_id)(0+bd_offset),
                    it_channel_0,
                    {0, 0, 0, weight_offset + round_offset * quant_in_per_column},
                    { 1,    blocks_per_row ,       (2*block_size_in_byte)/128 , 128},
                    { 0,    4*block_size_in_byte ,                      128,    1},
                    -1,0, false

                );

                seq.npu_dma_memcpy_nd(
                    sizeof(char),
                    qw_in_arg_idx,
                    MM2S,
                    IT[col],
                    (npu_bd_id)(2+bd_offset),
                    it_channel_1,
                    {0, 0, 0, weight_offset + round_offset * quant_in_per_column + 2*block_size_in_byte },   // second the second half of quant data
                    { 1,    blocks_per_row ,       (2*block_size_in_byte)/128 , 128},
                    { 0,    4*block_size_in_byte ,                      128,  1},
                    -1,0, false

                );

                // Each port receive 2*Q4NX_ROWx D_Q4NX_BLOCK_PER_ROW*Q4NX_COL
                uint32_t output_offset_0 = round_offset * 128 * D_in;
                seq.npu_dma_memcpy_nd(
                    2,
                    w_out_arg_idx,
                    S2MM,
                    IT[col],
                    (npu_bd_id)(1+bd_offset),
                    it_channel_0,
                    {0,0,0,  output_offset_0},
                    {1,1,1, 128 * D_in},
                    {0,0,0,1},
                    -1,0,true
                );
                bd_wait_counter[col]++;
            }
        }
        // note: for now
        for(int col = 0; col < 8; col++){
            if(bd_wait_counter[col] == 2){
                seq.npu_dma_wait(IT[col], S2MM, it_channel_0);
                bd_wait_counter[col]--;
            }
        }
    }

    for(int col = 0; col < 8; col++){
        while(bd_wait_counter[col] != 0){
            seq.npu_dma_wait(IT[col], S2MM, it_channel_0);
            bd_wait_counter[col]--;
        }
    }
    seq.cmds2seq();
}

// // Int this case, D_in_mem_padded
//     // N is din, M is dout

//     // first, the dequant of down

//             if(round_offset < total_column_rounds){
//                 seq.npu_dma_memcpy_nd(
//                     sizeof(char),
//                     qw_in_arg_idx,
//                     MM2S,
//                     IT[col],
//                     (npu_bd_id)(0+bd_offset),
//                     it_channel_0,
//                     {0, 0, 0, weight_offset + round_offset * quant_in_per_column},
//                     // {1, 1, 1, (uint32_t)quant_in_per_column}, //NOTE: being rearrange beforehadn already
//                     // {0, 0, 0, 1},
//                     { 1,    blocks_per_row ,       (2*block_size_in_byte)/128 , 128},
//                     { 0,    4*block_size_in_byte ,                      128,    1},
//                     -1,0, false

//                 );

//                 seq.npu_dma_memcpy_nd(
//                     sizeof(char),
//                     qw_in_arg_idx,
//                     MM2S,
//                     IT[col],
//                     (npu_bd_id)(2+bd_offset),
//                     it_channel_1,
//                     {0, 0, 0, weight_offset + round_offset * quant_in_per_column + 2*block_size_in_byte },   // second the second half of quant data
//                     { 1,    blocks_per_row ,       (2*block_size_in_byte)/128 , 128},
//                     { 0,    4*block_size_in_byte ,                      128,  1},
//                     -1,0, false

//                 );

//                 // Each port receive 2*Q4NX_ROWx D_Q4NX_BLOCK_PER_ROW*Q4NX_COL
//                 uint32_t output_offset_0 = round_offset * 128 * D_in_mem_padded;
//                 seq.npu_dma_memcpy_nd(
//                     2,
//                     w_out_arg_idx,
//                     S2MM,
//                     IT[col],
//                     (npu_bd_id)(1+bd_offset),
//                     it_channel_0,
//                     {0,0,0,  output_offset_0},
//                     // {1,1,1, 128 * D_in},
//                     // {0,0,0,1},
//                     {D_in/MM_K_size,                MM_K_size/k_tile        ,m_tile*4           ,k_tile},//n==128
//                     {MM_K_size*(m_tile * 4),         k_tile                 ,MM_K_size     ,1},
//                     -1,0,true
//                 );
//                 seq.npu_dma_memcpy_nd(
//                     2,
//                     w_out_arg_idx,
//                     S2MM,
//                     IT[col],
//                     (npu_bd_id)(3+bd_offset),
//                     it_channel_0,
//                     {0,0,0,  output_offset_0 +  ( D_in/MM_K_size ) * MM_K_size * m_tile * 4},
//                     {1,         (D_in%MM_K_size)/k_tile ,m_tile*4           ,k_tile},//n==128
//                     {0,         k_tile                 ,MM_K_size           ,1},
//                     -1,0,true
//                 );
//                 bd_wait_counter[col]++;

//             }

//         // note: for now
