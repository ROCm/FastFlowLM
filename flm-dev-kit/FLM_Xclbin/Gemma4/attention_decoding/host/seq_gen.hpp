// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#pragma once
#include "npu_utils/npu_utils.hpp"

void gen_seq(
    npu_sequence* seq,
    const uint32_t L_padded, // assume padded
    const uint32_t L_effective, // the effective sequence length, used for masking
    const uint32_t lc,
    const uint32_t DH, 
    const uint32_t HEADS,
    const uint32_t KV_HEADS
){

    auto round_up_to_multiple = [](int x, int multiple) -> int {
        if (multiple == 0) {
            return x; // Cannot divide by zero
        }
        // This uses integer division to achieve the rounding
        return ((x + multiple - 1) / multiple) * multiple;
    };

    npu_tiles IT[2] = {IT0, IT1};
    seq->clear_cmds();

    const int l_qk_address = 58368;
    const int l_kv_address = 14336;
    constexpr int CT_lock_address_base = 0x000001F000;
    constexpr int CT_rtp_sync_lock_id = 8;

    npu_tiles tile = get_tile(2, 0);
    seq->rtp_write(tile, l_qk_address, L_effective);
    seq->rtp_write(tile, CT_lock_address_base + 16 * CT_rtp_sync_lock_id, 1); // set lock to 1
    tile = get_tile(3, 0);
    seq->rtp_write(tile, l_kv_address, L_effective);
    seq->rtp_write(tile, CT_lock_address_base + 16 * CT_rtp_sync_lock_id, 1); // set lock to 1

    // send k
    size_t k_offset = 0;
    seq->npu_dma_memcpy_nd(
        2, 2,
        MM2S, IT[0],
        (npu_bd_id)(2), it_channel_0,
        {0, 0, 0, (uint32_t)k_offset},
        {1, 1, 1, (uint32_t)(L_padded * DH * KV_HEADS)},
        {0, 0, 0, 1},
        -1, 0, false
    );
    // send v
    size_t v_offset = (size_t)L_padded * DH * KV_HEADS;
    seq->npu_dma_memcpy_nd(
        2, 2,
        MM2S, IT[0],
        (npu_bd_id)(3), it_channel_1,
        {0, 0, 0, (uint32_t)v_offset},
        {1, 1, 1, (uint32_t)(L_padded * DH * KV_HEADS)},
        {0, 0, 0, 1},
        -1, 0, false
    );
    // send q
    size_t q_offset = 0;
    seq->npu_dma_memcpy_nd(
        2, 1,
        MM2S, IT[1],
        (npu_bd_id)(1), it_channel_0,
        {0, 0, 0, (uint32_t)q_offset},
        {1, 1, 1, (uint32_t)(DH * HEADS)},
        {0, 0, 0, 1},
        -1, 0, false
    );
    // receive y
    size_t y_offset = 0; 
    seq->npu_dma_memcpy_nd(
        2, 0,
        S2MM, IT[1],
        (npu_bd_id)(0), it_channel_0,
        {0, 0, 0, (uint32_t)y_offset},
        {1, 1, 1, (uint32_t)(DH * HEADS)},
        {0, 0, 0, 1},
        -1, 0, true
    );
    seq->npu_dma_wait(
        IT[1],
        S2MM,
        it_channel_0
    );
    seq->cmds2seq();
}