#pragma once
#include <cassert>
#include <algorithm> // Required for std::max
#include <string>
#include "npu_utils/npu_instr_utils.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"
void conv1d_prefill(
    npu_sequence* seq,
    const uint32_t L_OUT,
    bf16 min_value,
    bf16 max_value,
    const uint32_t external_x_offset, // in bf16
    const uint32_t external_o_offset,
    const uint32_t d
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

    const int l_address = 49664;
    const int round_address = 8704;
    const int min_address = 27136;
    const int max_address = 35840;
    constexpr int CT_lock_address_base = 0x000001F000;
    constexpr int CT_rtp_sync_lock_id = 6;

    assert(d == 1024);

    const int num_col = 8;
    float max_float = (float)max_value;
    float min_float = (float)min_value;

    const int l_column = 256;
    int ROUND = L_OUT / (l_column * num_col);
    int remaining = L_OUT % (l_column * num_col);
    int needed_col = 0;
    int l_left_last_col = 0;

    if (remaining > 0){
        // calculate how many columns are needed for the remaining part
        needed_col = remaining / l_column;
        if (remaining % l_column != 0) {
            // calculate how many rows are needed for the remaining part in the last column
            l_left_last_col = remaining - needed_col * l_column;
        }
    }

    if(ROUND > 0){
        int num_col = 8;
        for (int row = 2; row < 6; row++){
            for (int col = 0; col < num_col; col++){
                npu_tiles tile = get_tile(row, col);
                seq->rtp_write(tile, l_address, l_column);
                seq->rtp_write(tile, round_address, ROUND);
                seq->rtp_write(tile, max_address, *(uint32_t*)(&max_float));
                seq->rtp_write(tile, min_address, *(uint32_t*)(&min_float));
                seq->rtp_write(tile, CT_lock_address_base + 16 * CT_rtp_sync_lock_id, 1); // set lock to 1
            }
        }
        for (int col = 0; col < num_col; col++){
            // send w
            seq->npu_dma_memcpy_nd(
                2, 2,
                MM2S, IT[col],
                (npu_bd_id)(1), it_channel_1,
                {0, 0, 0, (uint32_t)0},
                {1, 1, (uint32_t)1, (uint32_t)(5 * d)},
                {0, 0, (uint32_t)0, (uint32_t)1},
                -1, 0, false
            );
        }
        for (int round = 0; round < ROUND; round++){
            int bd_offset = (round % 2) * 8;
            for (int col = 0; col < num_col; col++){
                // send x
                uint32_t x_offset = external_x_offset + round * l_column * num_col * d + col * l_column * d;
                seq->npu_dma_memcpy_nd(
                    2, 1,
                    MM2S, IT[col],
                    (npu_bd_id)(bd_offset + 2), it_channel_0,
                    {0, 0, 0, x_offset},
                    {1, 1, 1, (uint32_t)((l_column + 4) * d)},
                    {0, 0, 0, (uint32_t)1},
                    -1, 0, false
                );
                // receive o
                uint32_t o_offset = external_o_offset + round * l_column * num_col * d + col * l_column * d;
                seq->npu_dma_memcpy_nd(
                    2, 0,
                    S2MM, IT[col],
                    (npu_bd_id)(bd_offset + 0), it_channel_0,
                    {0, 0, 0, o_offset},
                    {1, 1, 1, (uint32_t)(l_column * d)},
                    {0, 0, 0, (uint32_t)1},
                    -1, 0, true
                );
            }
            if (round > 0){
                for (int col = 0; col < num_col; col++){
                    seq->npu_dma_wait(
                        IT[col],
                        S2MM,
                        it_channel_0
                    );
                }
            }
        }
        for (int col = 0; col < num_col; col++){
            seq->npu_dma_wait(
                IT[col],
                S2MM,
                it_channel_0
            );
        }
    }

    if(remaining > 0){
        int num_col = needed_col;

        for (int row = 2; row < 6; row++){
            for (int col = 0; col < num_col; col++){
                npu_tiles tile = get_tile(row, col);
                seq->rtp_write(tile, l_address, l_column);
                seq->rtp_write(tile, round_address, 1);
                seq->rtp_write(tile, max_address, *(uint32_t*)(&max_float));
                seq->rtp_write(tile, min_address, *(uint32_t*)(&min_float));
                seq->rtp_write(tile, CT_lock_address_base + 16 * CT_rtp_sync_lock_id, 1); // set lock to 1
            }
        }
        for (int col = 0; col < num_col; col++){
            // send w
            seq->npu_dma_memcpy_nd(
                2, 2,
                MM2S, IT[col],
                (npu_bd_id)(1), it_channel_1,
                {0, 0, 0, (uint32_t)0},
                {1, 1, (uint32_t)1, (uint32_t)(5 * d)},
                {0, 0, (uint32_t)0, (uint32_t)1},
                -1, 0, false
            );
        }
        for (int col = 0; col < num_col; col++){
            // send x
            uint32_t x_offset = external_x_offset + ROUND * l_column * 8 * d + col * l_column * d;
            seq->npu_dma_memcpy_nd(
                2, 1,
                MM2S, IT[col],
                (npu_bd_id)(2), it_channel_0,
                {0, 0, 0, x_offset},
                {1, 1, 1, (uint32_t)((l_column + 4) * d)},
                {0, 0, 0, (uint32_t)1},
                -1, 0, false
            );
            // receive o
            uint32_t o_offset = external_o_offset + ROUND * l_column * 8 * d + col * l_column * d;
            seq->npu_dma_memcpy_nd(
                2, 0,
                S2MM, IT[col],
                (npu_bd_id)(0), it_channel_0,
                {0, 0, 0, o_offset},
                {1, 1, 1, (uint32_t)(l_column * d)},
                {0, 0, 0, (uint32_t)1},
                -1, 0, true
            );
            seq->npu_dma_wait(
                IT[col],
                S2MM,
                it_channel_0
            );
        }

        if (remaining % l_column != 0){
            for (int row = 2; row < 6; row++){
                npu_tiles tile = get_tile(row, num_col);
                seq->rtp_write(tile, l_address, l_left_last_col);
                seq->rtp_write(tile, round_address, 1);
                seq->rtp_write(tile, max_address, *(uint32_t*)(&max_float));
                seq->rtp_write(tile, min_address, *(uint32_t*)(&min_float));
                seq->rtp_write(tile, CT_lock_address_base + 16 * CT_rtp_sync_lock_id, 1); // set lock to 1
            }
            // send w
            seq->npu_dma_memcpy_nd(
                2, 2,
                MM2S, IT[num_col],
                (npu_bd_id)(1), it_channel_1,
                {0, 0, 0, (uint32_t)0},
                {1, 1, (uint32_t)1, (uint32_t)(5 * d)},
                {0, 0, (uint32_t)0, (uint32_t)1},
                -1, 0, false
            );
            // send x
            uint32_t x_offset = external_x_offset +  ROUND * l_column * 8 * d + num_col * l_column * d;
            seq->npu_dma_memcpy_nd(
                2, 1,
                MM2S, IT[num_col],
                (npu_bd_id)(2), it_channel_0,
                {0, 0, 0, x_offset},
                {1, 1, 1, (uint32_t)((l_left_last_col + 4) * d)},
                {0, 0, 0, (uint32_t)1},
                -1, 0, false
            );
            // receive o
            uint32_t o_offset = external_o_offset +ROUND * l_column * 8 * d + num_col * l_column * d;
            seq->npu_dma_memcpy_nd(
                2, 0,
                S2MM, IT[num_col],
                (npu_bd_id)(0), it_channel_0,
                {0, 0, 0, o_offset},
                {1, 1, 1, (uint32_t)(l_left_last_col * d)},
                {0, 0, 0, (uint32_t)1},
                -1, 0, true
            );
            seq->npu_dma_wait(
                IT[num_col],
                S2MM,
                it_channel_0
            );
        }
    }

    seq->cmds2seq();
}
