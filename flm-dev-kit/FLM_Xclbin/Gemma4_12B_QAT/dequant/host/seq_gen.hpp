// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef SEQ_GEN_HPP
#define SEQ_GEN_HPP

#pragma once
#include "npu_utils/npu_utils.hpp"
#include "typedef.hpp"
#include "weights_desc.hpp"

inline void generate_dequant_q4_0_seq(npu_sequence* seq, const uint32_t Dout, const uint32_t Din, const u32 weight_offset){
    static constexpr npu_tiles IT[8] = {IT0, IT1, IT2, IT3, IT4, IT5, IT6, IT7}; 

    static constexpr int m = QXNX_ROW_BLOCK_SIZE;
    static constexpr int k = QXNX_COL_BLOCK_SIZE;

    uint32_t a_block_size = get_quantization_byte_size(m * k, flm_q40);

    const uint32_t blocks_per_row = Din / k;

    const uint32_t columns = 8;
    const uint32_t rows_per_column = 4; // one AIE core row per row block

    // A work unit is the `rows_per_column` row blocks handed to one AIE column.
    assert(Dout % (m * rows_per_column) == 0);
    const uint32_t work_units = Dout / (m * rows_per_column);
    // Dout is not always a multiple of m * columns * rows_per_column: o_proj and
    // down_proj have Dout = 3840, i.e. 30 work units = 3 full rounds of 8 plus a
    // tail of 6. Round the round count up and skip the idle columns in the tail --
    // truncating here would silently leave the last rows of the matrix unwritten.
    const uint32_t rounds = (work_units + columns - 1) / columns;
    assert(rounds > 0);

    int bd_wait_counter[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    seq->clear_cmds();

    for (uint32_t round = 0; round < rounds; round++){
        uint32_t bd_offset = (round % 2) * 8;
        for (uint32_t col = 0; col < columns; col++){
            const uint32_t unit = round * columns + col;
            if (unit >= work_units) break; // tail round: this column has no work
            const uint32_t row_block = unit * rows_per_column;
            // quantized weights: byte-addressed, one contiguous group of
            // `rows_per_column` row blocks (already interleaved by the host)
            seq->npu_dma_memcpy_nd(
                sizeof(u8),
                1,
                MM2S,
                IT[col],
                npu_bd_id(bd_1 + bd_offset),
                it_channel_0,
                {0, 0, 0, (uint32_t)(row_block * a_block_size * blocks_per_row + weight_offset)},
                {1, 1, 1, rows_per_column * blocks_per_row * a_block_size},
                {0, 0, 0, 1},
                -1, 0, false, aggressive_cache
            );
            // dequantized weights: bf16-addressed, `blocks_per_row` column blocks
            // of (rows_per_column * m) x k, scattered along the row stride Din
            seq->npu_dma_memcpy_nd(
                sizeof(bf16),
                0,
                S2MM,
                IT[col],
                npu_bd_id(bd_0 + bd_offset),
                it_channel_0,
                {0, 0, 0, (uint32_t)((size_t)row_block * m * Din)},
                {1, blocks_per_row, rows_per_column * m, k},
                {0, k, Din, 1},
                -1, 0, true, aggressive_cache
            );
            bd_wait_counter[col]++;
        }
        // Wait per column rather than for all 8: in the tail round the idle columns
        // have no outstanding BD to wait on.
        for (uint32_t col = 0; col < columns; col++){
            if (bd_wait_counter[col] == 2){
                seq->npu_dma_wait(
                    IT[col],
                    S2MM,
                    it_channel_0
                );
                bd_wait_counter[col]--;
            }
        }
    }
    for (uint32_t col = 0; col < columns; col++){
        while (bd_wait_counter[col] != 0){
            seq->npu_dma_wait(
                IT[col],
                S2MM,
                it_channel_0
            );
            bd_wait_counter[col]--;
        }
    }
    seq->cmds2seq();
}

#endif