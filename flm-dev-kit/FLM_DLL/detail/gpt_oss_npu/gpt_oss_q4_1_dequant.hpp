// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#pragma once

#include "gpt_oss_params.hpp"
#include "npu_utils/npu_instr_utils.hpp"

void gen_dequant_seq(npu_sequence* seq, const u32 D_in, const u32 D_out, const u32 weight_offset){
    constexpr npu_tiles IT[] = {IT0, IT1, IT2, IT3, IT4, IT5, IT6, IT7};

    constexpr u32 total_cols = 8;
    constexpr u32 total_rows = 4;

    constexpr block_row = 32;
    constexpr block_col = 128;

    // N is din, M is dout
    const int rounds = D_out / block_row / total_cols / 2;
    seq->clear_cmds();
    seq->npu_preemption(0);
    uint32_t qw_size = block_row * block_col * 5 / 8 / 4;
    for (int round = 0; round < rounds; round++){
        int bd_offset = (round % 2) * 8;
        npu_bd_id w_out_bd = npu_bd_id(0 + bd_offset);
        npu_bd_id wq_in_up_bd = npu_bd_id(1 + bd_offset);
        npu_bd_id wq_in_down_bd = npu_bd_id(2 + bd_offset);

        for (int c = 0; c < total_cols; c++){
            uint32_t qw_offset = (round * 8 + c) * qw_size * D_in / block_col * 2 + weight_offset / 4;
            uint32_t q_offset = (round * 8 + c) * 64 * D_in;

            seq->npu_dma_memcpy_nd(
                2, 0,
                S2MM, IT[c],
                w_out_bd, it_channel_0,
                {0, 0, 0, q_offset},
                {1, 1, 1, 64 * D_in},
                {0, 0, 0, 1},
                -1, 0, true
            );

            seq->npu_dma_memcpy_nd(
                4, 1,
                MM2S, IT[c],
                wq_in_up_bd, it_channel_0,
                {0, 0, 0, qw_offset},
                {1, 1, 1, qw_size * D_in / block_col},
                {0, 0, 0, 1},
                -1, 0, false
            );

            seq->npu_dma_memcpy_nd(
                4, 1,
                MM2S, IT[c],
                wq_in_down_bd, it_channel_1,
                {0, 0, 0, qw_offset + qw_size * D_in / block_col},
                {1, 1, 1, qw_size * D_in / block_col},
                {0, 0, 0, 1},
                -1, 0, false
            );

            if (round > 0){
                seq->npu_dma_wait(IT[c], S2MM, it_channel_0);
            }
        }
    }// end round
    for (int c = 0; c < total_cols; c++){
        seq->npu_dma_wait(IT[c], S2MM, it_channel_0);
    }

    seq->cmds2seq();
}
