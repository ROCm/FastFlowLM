// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#pragma once
#include "npu_utils/npu_utils.hpp"

void gen_seq(
    npu_sequence* seq,
    const uint32_t D,
    const int rtp_activate_func_impl_id = 0
){
    npu_tiles IT[1] = {IT0};
    seq->clear_cmds();

    const int rtp_address = 10240;

    constexpr int CT_lock_address_base = 0x000001F000;
    constexpr int CT_rtp_sync_lock_id = 4;
    for (int row = 2; row < 3; row++){
        for (int col = 0; col < 1; col++){
            npu_tiles tile = get_tile(row, col);
            seq->rtp_write(tile, rtp_address, rtp_activate_func_impl_id);
            seq->rtp_write(tile, CT_lock_address_base+16*(CT_rtp_sync_lock_id), 1); // set lock to 1
        }
    }

    // send x
    uint32_t x_offset = 0;
    seq->npu_dma_memcpy_nd(
        2, 1,
        MM2S, IT[0],
        (npu_bd_id)(1), it_channel_0,
        {0, 0, 0, x_offset},
        {1, 1, 1, (uint32_t)D},
        {0, 0, 0, (uint32_t)1},
        -1, 0, false
    );
    size_t o_offset = 0;
    seq->npu_dma_memcpy_nd(
        2, 0,
        S2MM, IT[0],
        (npu_bd_id)(0), it_channel_0,
        {0, 0, 0, (uint32_t)o_offset},
        {1, 1, 1, (uint32_t)D},
        {0, 0, 0, (uint32_t)1},
        -1, 0, true
    );
    seq->npu_dma_wait(
        IT[0],
        S2MM,
        it_channel_0
    );
    seq->cmds2seq();
}