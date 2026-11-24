// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#pragma once
#include "npu_utils/npu_utils.hpp"

void gen_seq(
    npu_sequence* seq,
    const uint32_t L_PADDED, // assume padded
    const uint32_t D,  
    const uint32_t L_OUT,
    const uint32_t weight_offset
){

    auto round_up_to_multiple = [](int x, int multiple) -> int {
        if (multiple == 0) {
            return x; // Cannot divide by zero
        }
        // This uses integer division to achieve the rounding
        return ((x + multiple - 1) / multiple) * multiple;
    };

    npu_tiles IT[8] = {IT0, IT1, IT2, IT3, IT4, IT5, IT6, IT7};
    seq->clear_cmds();

    const int l_address = 33024;
    const uint32_t d = 512;

    const int num_col = 8;
    for (int row = 2; row < 6; row++){
        for (int col = 0; col < num_col; col++){
            npu_tiles tile = get_tile(row, col);
            seq->rtp_write(tile, l_address, L_OUT);
        }
    }

    for(int col = 0; col < num_col; col++){
        //send w
        size_t w_offset = col * d + weight_offset;
        seq->npu_dma_memcpy_nd(
            2, 2,
            MM2S, IT[col],
            (npu_bd_id)(2), it_channel_1,
            {0, 0, 0, (uint32_t)w_offset},
            {1, 1, (uint32_t)4, (uint32_t)(d)},
            {0, 0, (uint32_t)(D), (uint32_t)1},
            -1, 0, false
        );
    }
    uint32_t L_DIV_PADD_REQUIREMENT = L_PADDED;
    int num_blocks = L_DIV_PADD_REQUIREMENT / 256;
    int remainder  = L_DIV_PADD_REQUIREMENT % 256;
    std::cout << "num_blocks: " << num_blocks << ", remainder: " << remainder << std::endl;
    // send x
    if(num_blocks > 0){
        for (int col = 0; col < num_col; col++){
            uint32_t x_offset = col * d;
            seq->npu_dma_memcpy_nd(
                2, 1,
                MM2S, IT[col],
                (npu_bd_id)(2), it_channel_0,
                {0, 0, 0, x_offset},
                {1, (uint32_t)num_blocks, 256, (uint32_t)d},
                {0, (uint32_t)(256 * D), (uint32_t)D, (uint32_t)1},
                -1, 0, false
            );

            uint32_t x_reminder_offset = col * d + (256 * D) * (num_blocks);
            seq->npu_dma_memcpy_nd(
                2, 1,
                MM2S, IT[col],
                (npu_bd_id)(10), it_channel_0,
                {0, 0, 0, x_reminder_offset},
                {1, 1, (uint32_t)remainder, (uint32_t)d},
                {0, 0, (uint32_t)D, (uint32_t)1},
                -1, 0, false
            );
        }
    }else{
        for (int col = 0; col < num_col; col++){
            // send x
            uint32_t x_offset = col * d;
            seq->npu_dma_memcpy_nd(
                2, 1,
                MM2S, IT[col],
                (npu_bd_id)(2), it_channel_0,
                {0, 0, 0, x_offset},
                {1, 1, (uint32_t)L_DIV_PADD_REQUIREMENT, (uint32_t)d},
                {0, 0, (uint32_t)D, (uint32_t)1},
                -1, 0, false
            );
        } // end for col
    }

    uint32_t y_tranasction_L_size = 1024;
    uint32_t receive_round = round_up_to_multiple(L_OUT, y_tranasction_L_size) / y_tranasction_L_size;
    for (int round = 0; round < receive_round; round++){
        int bd_offset = (round % 2) * 8;
        for (int col = 0; col < num_col; col++){
            // receive o
            size_t o_offset = col * d + round * y_tranasction_L_size * D;
            if(round +1 == receive_round && L_OUT % y_tranasction_L_size != 0){
                uint32_t L_remain = L_OUT  % y_tranasction_L_size;
                seq->npu_dma_memcpy_nd(
                    2, 0,
                    S2MM, IT[col],
                    (npu_bd_id)(bd_offset + 0), it_channel_0,
                    {0, 0, 0, (uint32_t)o_offset},
                    {1, 1,  (uint32_t)L_remain, (uint32_t)d},
                    {0, 0,  (uint32_t)D,        (uint32_t)1},
                    -1, 0, true
                );
            }else{
                seq->npu_dma_memcpy_nd(
                    2, 0,
                    S2MM, IT[col],
                    (npu_bd_id)(bd_offset + 0), it_channel_0,
                    {0, 0, 0, (uint32_t)o_offset},
                    {1, 1,  (uint32_t)y_tranasction_L_size, (uint32_t)d},
                    {0, 0,  (uint32_t)D, (uint32_t)1},
                    -1, 0, true
                );
            }
        }
        if (round > 0){
            for (int col = 0; col < num_col; col++){
                seq->npu_dma_wait(
                    IT[col],
                    S2MM,
                    it_channel_0
                );
            }
        }// round > 0
    }// round
    for (int col = 0; col < num_col; col++){
        seq->npu_dma_wait(
            IT[col],
            S2MM,
            it_channel_0
        );
    }
    seq->cmds2seq();
}