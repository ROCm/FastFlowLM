// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#pragma once
#include "npu_utils/npu_utils.hpp"
#define ENABLE_AXI4

// void gen_conv1d_engine_seq(npu_sequence* seq, const uint32_t L, const uint32_t D, const uint32_t d, const uint32_t weight_offset){
//     constexpr int total_cols = 8;
//     constexpr int total_rows = 4;
//     constexpr npu_tiles IT[total_cols] = {IT0, IT1, IT2, IT3, IT4, IT5, IT6, IT7};

//     seq->clear_cmds();
    
//     for (int row = 2; row < total_rows + 2; row++){
//         for (int col = 0; col < total_cols; col++){
//             npu_tiles tile = get_tile(row, col);
//             seq->rtp_write(tile, 16512, L);
//         }
//     }

//     for (int col = 0; col < total_cols; col++){
//         // receive y
//         int y_offset = col * d;
//         seq->npu_dma_memcpy_nd(
//             2, 0,
//             S2MM, IT[col],
//             (npu_bd_id)(0), it_channel_0,
//             {0, 0, 0, y_offset},
//             {1, 1, 1, d},
//             {0, 0, 0, 1},
//             -1, 0, true
//         );
//         // send w
//         int w_offset = col * d + weight_offset;
//         seq->npu_dma_memcpy_nd(
//             2, 1,
//             MM2S, IT[col],
//             (npu_bd_id)(1), it_channel_0,
//             {0, 0, 0, w_offset},
//             {1, 1, 3, d},
//             {0, 0, D, 1},
//             -1, 0, false
//         );
//         // send x
//         int x_offset = col * d;
//         seq->npu_dma_memcpy_nd(
//             2, 2,
//             MM2S, IT[col],
//             (npu_bd_id)(2), it_channel_1,
//             {0, 0, 0, x_offset},
//             {1, 1, 3, d},
//             {0, 0, D, 1},
//             -1, 0, false
//         );
//         seq->npu_dma_wait(IT[col], S2MM, it_channel_0);
//     } // end for col

//     for(int round = 1; round < L; round++){
//         int bd_offset = (round % 2) * 8;
//         for (int col = 0; col < total_cols; col++){
//             int y_offset = round * D + col * d;
//             seq->npu_dma_memcpy_nd(
//                 2, 0,
//                 S2MM, IT[col],
//                 (npu_bd_id)(bd_offset + 0), it_channel_0,
//                 {0, 0, 0, y_offset},
//                 {1, 1, 1, d},
//                 {0, 0, 0, 1},
//                 -1, 0, true
//             );
//             // send x
//             int x_offset = 2 * D + round * D + col * d;
//             seq->npu_dma_memcpy_nd(
//                 2, 2,
//                 MM2S, IT[col],
//                 (npu_bd_id)(bd_offset + 2), it_channel_1,
//                 {0, 0, 0, x_offset},
//                 {1, 1, 1, d},
//                 {0, 0, 0, 1},
//                 -1, 0, false
//             );
//         } // end for col
//         if (round > 1){
//             for (int col = 0; col < total_cols; col++){
//                 seq->npu_dma_wait(IT[col], S2MM, it_channel_0);
//             }
//         }
//     } // end for round
//     for (int col = 0; col < total_cols; col++){
//         seq->npu_dma_wait(IT[col], S2MM, it_channel_0);
//     }
//     seq->cmds2seq();    
// }



void gen_conv1d_engine_seq(npu_sequence* seq,
    const uint32_t L_PADDED, // assume padded
    const uint32_t D,  
    const uint32_t L_OUT,  // caller handle the calculation of L_OUT through kernel size and stride...
    const uint32_t L_PADDED_REQUIREMENT,
    const uint32_t weight_offset){
    constexpr int total_cols = 8;
    constexpr int total_rows = 4;
    constexpr npu_tiles IT[total_cols] = {IT0, IT1, IT2, IT3, IT4, IT5, IT6, IT7};

    seq->clear_cmds();
    //const uint32_t d = D/(4*8);
    const uint32_t d = D/8;

    auto round_up_to_multiple = [](int x, int multiple) -> int {
        if (multiple == 0) {
            return x; // Cannot divide by zero
        }
        // This uses integer division to achieve the rounding
        return ((x + multiple - 1) / multiple) * multiple;
    };
    

    constexpr int CT_lock_address_base = 0x000001F000;
    constexpr uint32_t CT_rtp_sync_lock_id = 6;
    for (int row = 2; row < total_rows + 2; row++){
        for (int col = 0; col < total_cols; col++){
            npu_tiles tile = get_tile(row, col);
            seq->rtp_write(tile, 16512, L_PADDED);
            seq->rtp_write(tile, CT_lock_address_base+16*(CT_rtp_sync_lock_id), 1); // set lock to 1            
        }
    }



    for(int col = 0; col < total_cols; col++ ){
        // purely send weights
        uint32_t w_offset = col * d + weight_offset;
        seq->npu_dma_memcpy_nd(
            2, 1,
            MM2S, IT[col],
            (npu_bd_id)(1), it_channel_0,
            {0, 0, 0, w_offset},
            {1, 1, 3, d},
            {0, 0, D, 1},
            -1, 0, false
        );
    }


    assert( L_PADDED %L_PADDED_REQUIREMENT == 0);
    uint32_t L_DIV_PADD_REQUIREMENT = L_PADDED / L_PADDED_REQUIREMENT;
    int num_blocks = L_DIV_PADD_REQUIREMENT / 1024;
    int remainder   = L_DIV_PADD_REQUIREMENT % 1024;
    // send x
    if(num_blocks > 0){
        for (int col = 0; col < total_cols; col++){

            uint32_t x_offset = col * d;
            seq->npu_dma_memcpy_nd(
                2, 2,
                MM2S, IT[col],
                (npu_bd_id)( 2), it_channel_1,
                {0, 0, 0, x_offset},
                 {num_blocks,                       1024,                        L_PADDED_REQUIREMENT,         d},
                {1024*L_PADDED_REQUIREMENT*D, L_PADDED_REQUIREMENT*D,                      D,             1},
                -1, 0, false
            );

        } 


        for (int col = 0; col < total_cols; col++){

            uint32_t x_offset = col * d +   (1024*L_PADDED_REQUIREMENT*D) * (num_blocks);
            seq->npu_dma_memcpy_nd(
                2, 2,
                MM2S, IT[col],
                (npu_bd_id)(10 ), it_channel_1,
                {0, 0, 0, x_offset},
                {1,  remainder,                                 L_PADDED_REQUIREMENT,         d},
                {0, L_PADDED_REQUIREMENT*D,                      D,                         1},
                -1, 0, false
            );

        } 


        
    }else{
        //done in single transaction

        for (int col = 0; col < total_cols; col++){

            // send x
            uint32_t x_offset = col * d;
            seq->npu_dma_memcpy_nd(
                2, 2,
                MM2S, IT[col],
                (npu_bd_id)( 2), it_channel_1,
                {0, 0, 0, x_offset},
                {1,  L_DIV_PADD_REQUIREMENT,                L_PADDED_REQUIREMENT, d},
                {0, L_PADDED_REQUIREMENT*D,                      D,             1},
                -1, 0, false
            );

        } // end for col

    }

    std::cout << "L_OUT " << L_OUT << std::endl;
    uint32_t y_tranasction_L_size = 1024;
    uint32_t receive_round = round_up_to_multiple(L_OUT, y_tranasction_L_size) / y_tranasction_L_size;

    int C_wait_counter = 0;  // only keep column 0 should be sufficient for our case
    for(int round = 0; round < receive_round; round++){
        int bd_offset = (round % 2) * 8;
        for (int col = 0; col < total_cols; col++){
            int y_offset = round * (y_tranasction_L_size*D) + col * d;

            if(round +1 == receive_round && L_OUT % y_tranasction_L_size != 0){
                uint32_t L_remain = L_OUT  % y_tranasction_L_size;
                seq->npu_dma_memcpy_nd(
                    2, 0,
                    S2MM, IT[col],
                    (npu_bd_id)(bd_offset + 0), it_channel_0,
                    {0, 0, 0, y_offset},
                    {1, 1,    L_remain, d},
                    {0, 0,  D,        1},
                    -1, 0, true
                );

            }else{
                seq->npu_dma_memcpy_nd(
                    2, 0,
                    S2MM, IT[col],
                    (npu_bd_id)(bd_offset + 0), it_channel_0,
                    {0, 0, 0, y_offset},
                    {1, 1,    y_tranasction_L_size, d},
                    {0, 0,  D, 1},
                    -1, 0, true
                );
            }

        } // end for col

        C_wait_counter ++;
        if (C_wait_counter == 2 ){
            for (int col = 0; col < total_cols; col++){
                seq->npu_dma_wait(IT[col], S2MM, it_channel_0);
            }
            C_wait_counter--;
        }
    } // end for round
    for(int wait_r = 0; wait_r < C_wait_counter; wait_r ++ ){
        for (int col = 0; col < total_cols; col++){
            seq->npu_dma_wait(IT[col], S2MM, it_channel_0);
        }
    }

    seq->cmds2seq();    
}