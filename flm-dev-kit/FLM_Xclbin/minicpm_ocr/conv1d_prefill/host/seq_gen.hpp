// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#pragma once
#include <algorithm>
#include <cassert>
#include <vector>
#include "npu_utils/npu_utils.hpp"

// Runtime sequence for conv8.xclbin. The same code is in
// FLM_DLL/detail/minicpm_ocr_npu/minicpm_ocr_npu_sequence.cpp::gen_seq_conv1d();
// keep the two in step.
//
// x is [L_PADDED, D] bf16 (the 3 conv-state rows first), w is [4, D] at
// weight_offset in its own argument, o is [L_OUT, D]. D = q (2048) + k (2048) +
// v (2048) = 6144.
//
// Layout: all 8 columns, every core owns one 128-wide q/k head and 64 v channels.
// Column col therefore takes q/k channels [512 col, 512 col + 512) and v channels
// [4096 + 256 col, ... + 256), sent as two streams:
//     shim MM2S 0: the 4 q/k weight rows, then the q/k part of every x row
//     shim MM2S 1: the 4 v weight rows,   then the v part of every x row
//     shim S2MM 0 / 1: the q/k / v part of every output row
// The weight rows lead the stream so they reach the cores like any other row.
inline void gen_seq(
    npu_sequence* seq,
    const uint32_t L_PADDED, // L_OUT + (window_size - 1)
    const uint32_t D,
    const uint32_t L_OUT,
    const uint32_t weight_offset,
    const int l_address = 33152  // address of the L rtp buffer in conv8.xclbin (all 32 cores)
){
    npu_tiles IT[8] = {IT0, IT1, IT2, IT3, IT4, IT5, IT6, IT7};
    seq->clear_cmds();

    constexpr uint32_t num_col = 8;
    constexpr uint32_t num_row = 4;
    constexpr uint32_t window_size = 4;
    constexpr uint32_t d_qk_col = 512;            // 4 cores x one 128-wide head
    constexpr uint32_t d_v_col = 256;             // 4 cores x 64
    constexpr uint32_t D_qk = num_col * d_qk_col; // 4096 = q + k
    assert(D == D_qk + num_col * d_v_col && "conv8.xclbin is laid out for D = 6144 (q 2048, k 2048, v 2048)");

    constexpr uint32_t x_arg = 1;
    constexpr uint32_t w_arg = 2;
    constexpr uint32_t o_arg = 0;

    // input rows go in blocks of 256 (the BD's row count field), plus a remainder
    constexpr uint32_t X_block = 256;
    const uint32_t num_x_blocks = L_PADDED / X_block;
    const uint32_t x_remainder = L_PADDED % X_block;
    constexpr uint32_t Y_block = 1024;

    for (uint32_t row = 2; row < 2 + num_row; row++) {
        for (uint32_t col = 0; col < num_col; col++) {
            seq->rtp_write(get_tile(row, col), l_address, L_OUT);
        }
    }

    // one input stream: its 4 weight rows, then the x rows
    auto send_stream = [&](uint32_t col, npu_it_channel ch, uint32_t c_off, uint32_t width,
                           uint32_t bd_w, uint32_t bd_x, uint32_t bd_rem) {
        seq->npu_dma_memcpy_nd(
            sizeof(bf16), w_arg, MM2S, IT[col], (npu_bd_id)bd_w, ch,
            {0, 0, 0, weight_offset + c_off},
            {1, 1, window_size, width},
            {0, 0, D, 1},
            -1, 0, false);
        if (num_x_blocks > 0) {
            seq->npu_dma_memcpy_nd(
                sizeof(bf16), x_arg, MM2S, IT[col], (npu_bd_id)bd_x, ch,
                {0, 0, 0, c_off},
                {1, num_x_blocks, X_block, width},
                {0, X_block * D, D, 1},
                -1, 0, false);
        }
        if (x_remainder > 0) {
            seq->npu_dma_memcpy_nd(
                sizeof(bf16), x_arg, MM2S, IT[col], (npu_bd_id)bd_rem, ch,
                {0, 0, 0, c_off + X_block * D * num_x_blocks},
                {1, 1, x_remainder, width},
                {0, 0, D, 1},
                -1, 0, false);
        }
    };
    for (uint32_t col = 0; col < num_col; col++) {
        send_stream(col, it_channel_0, col * d_qk_col, d_qk_col, 1, 2, 3);
        send_stream(col, it_channel_1, D_qk + col * d_v_col, d_v_col, 4, 5, 6);
    }

    // outputs, Y_block rows at a time; each round waits for the one before it
    const uint32_t receive_round = (L_OUT + Y_block - 1) / Y_block;
    for (uint32_t r = 0; r < receive_round; r++) {
        const uint32_t rows = (r + 1 == receive_round && L_OUT % Y_block != 0) ? L_OUT % Y_block : Y_block;
        for (uint32_t col = 0; col < num_col; col++) {
            seq->npu_dma_memcpy_nd(
                sizeof(bf16), o_arg, S2MM, IT[col], (npu_bd_id)((r % 2) ? 8 : 0), it_channel_0,
                {0, 0, 0, col * d_qk_col + r * Y_block * D},
                {1, 1, rows, d_qk_col},
                {0, 0, D, 1},
                -1, 0, true);
            seq->npu_dma_memcpy_nd(
                sizeof(bf16), o_arg, S2MM, IT[col], (npu_bd_id)((r % 2) ? 9 : 7), it_channel_1,
                {0, 0, 0, D_qk + col * d_v_col + r * Y_block * D},
                {1, 1, rows, d_v_col},
                {0, 0, D, 1},
                -1, 0, true);
        }
        if (r > 0) {
            for (uint32_t col = 0; col < num_col; col++) {
                seq->npu_dma_wait(IT[col], S2MM, it_channel_0);
                seq->npu_dma_wait(IT[col], S2MM, it_channel_1);
            }
        }
    }
    for (uint32_t col = 0; col < num_col; col++) {
        seq->npu_dma_wait(IT[col], S2MM, it_channel_0);
        seq->npu_dma_wait(IT[col], S2MM, it_channel_1);
    }

    seq->cmds2seq();
}
