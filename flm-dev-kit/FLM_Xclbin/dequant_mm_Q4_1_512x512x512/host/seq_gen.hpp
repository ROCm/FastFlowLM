// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __SEQ_GEN_HPP__
#define __SEQ_GEN_HPP__
#pragma once
#include "npu_utils/npu_utils.hpp"
#include "weights_desc.hpp"

// Mirror of qwen3_5vl_npu_sequence::gen_dequant_mm_512 in
// FLM_DLL/detail/qwen3_5vl_npu/qwen3_5vl_npu_sequence.cpp. Keep the two in sync:
// argument order is (L, Din, Dout, ...), and the quantization type used for every
// byte offset must be the one this xclbin's kernel decodes (q4_1 here).
// These are core-local buffer addresses assigned by aie-assign-buffer-addresses,
// so they move whenever the kernel's data layout changes -- the q4_1 dequant core
// is not the same size as the q4_k one, and its even row lands 768 B higher than
// the 54784/54848 the qwen3_5vl runtime uses. After any bitstream rebuild, re-read
// them from
//   build/bitstream/from_iron/dequant_mm.mlir.prj/input_with_addresses.mlir
// (grep for K_div_k_2_7 / rtp_act_2_7 and K_div_k_3_0 / rtp_act_3_0).
static constexpr uint32_t CT_rtp_address_even_row  = 55552;
static constexpr uint32_t act_rtp_address_even_row = 55616;
static constexpr uint32_t CT_rtp_address_odd_row   = 61696;
static constexpr uint32_t act_rtp_address_odd_row  = 61760;

void gen_dequant_mm_512(npu_sequence* seq, const uint32_t L, const uint32_t Din, const uint32_t Dout, const size_t weight_offset, const size_t out_offset, const int rtp_act){
    static constexpr npu_tiles IT[8] = {IT0, IT1, IT2, IT3, IT4, IT5, IT6, IT7};

    static constexpr int m = QXNX_ROW_BLOCK_SIZE;
    static constexpr int k = QXNX_COL_BLOCK_SIZE;
    static constexpr uint32_t elems = m * k;
    static constexpr uint32_t desired_m = 128;
    static constexpr uint32_t desired_k = 512;
    static constexpr uint32_t desired_n = 64;
    //  %K_div_k_2_7 = aie.buffer(%tile_7_2) {address = 54784 : i32, sym_name = "K_div_k_2_7"} : memref<16xi32>
    // %rtp_act_2_7 = aie.buffer(%tile_7_2) {address = 54848 : i32, sym_name = "rtp_act_2_7"} : memref<16xi32>
    // %K_div_k_3_0 = aie.buffer(%tile_0_3) {address = 61696 : i32, sym_name = "K_div_k_3_0"} : memref<16xi32>
    // %rtp_act_3_0 = aie.buffer(%tile_0_3) {address = 61760 : i32, sym_name = "rtp_act_3_0"} : memref<16xi32>

    uint32_t a_block_size = get_quantization_byte_size(m * k, flm_q41);

    const uint32_t blocks_per_row = Din / k;
    int K_div_k = Din / desired_k;

    const uint32_t columns = 8;
    assert(Dout / m / columns > 0);
    assert(Dout % (m * columns) == 0);

    seq->clear_cmds();

    int M_div_m = L / (desired_m * 4); // 4 row-blocks processed per i iteration (IT[col], col=0..3)
    int N_div_n = Dout / desired_n / columns;
    assert(L % (desired_m * 4) == 0);

    for (int row = 2; row < 6; row+=2){
        for (int col = 0; col < columns; col++){
            npu_tiles tile = get_tile(row, col);
            seq->rtp_write(tile, CT_rtp_address_even_row, K_div_k);
            seq->rtp_write(tile, act_rtp_address_even_row, rtp_act);
        }
    }
    for (int row = 3; row < 6; row+=2){
        for (int col = 0; col < columns; col++){
            npu_tiles tile = get_tile(row, col);
            seq->rtp_write(tile, CT_rtp_address_odd_row, K_div_k);
            seq->rtp_write(tile, act_rtp_address_odd_row, rtp_act);
        }
    }

    for (int i = 0; i < M_div_m; i++){
        for (int j = 0; j < N_div_n; j++){
            int round = i * N_div_n + j;
            int is_pong = round % 2;
            int bd_offset = is_pong * 8;
            uint32_t y_offset = i * desired_m * 4 * Dout + j * desired_n * columns;
            uint32_t x_offset = i * desired_m * 4 * Din;
            int w_offset = get_quantization_byte_size((size_t)j * desired_n * Din * columns, flm_q41) + weight_offset;

            npu_bd_id y_bd_0 = npu_bd_id(0 + bd_offset);
            npu_bd_id y_bd_1 = npu_bd_id(3 + bd_offset);
            npu_bd_id x_bd = npu_bd_id(1 + bd_offset);
            npu_bd_id w_bd_0 = npu_bd_id(2 + bd_offset);
            npu_bd_id w_bd_1 = npu_bd_id(4 + bd_offset);
            for (uint32_t col = 0; col < 4; col++){
                seq->npu_dma_memcpy_nd(
                    sizeof(char),
                    2,
                    MM2S,
                    IT[col],
                    w_bd_0,
                    it_channel_1,
                    {0, 0, 0, (uint32_t)(w_offset + get_quantization_byte_size((size_t)col * desired_n * Din, flm_q41))},
                    {1, 1, 1, (uint32_t)get_quantization_byte_size(desired_n * Din, flm_q41)},
                    {0, 0, 0, 1},
                    -1, 0, false, aggressive_cache
                );
                seq->npu_dma_memcpy_nd(
                    sizeof(char),
                    2,
                    MM2S,
                    IT[col + 4],
                    w_bd_1,
                    it_channel_1,
                    {0, 0, 0, (uint32_t)(w_offset + get_quantization_byte_size((size_t)col * desired_n * Din, flm_q41) + get_quantization_byte_size(4 * desired_n * Din, flm_q41))},
                    {1, 1, 1, (uint32_t)get_quantization_byte_size(desired_n * Din, flm_q41)},
                    {0, 0, 0, 1},
                    -1, 0, false, aggressive_cache
                );
                // receive y
                seq->npu_dma_memcpy_nd(
                    2, 0,
                    S2MM, IT[col],
                    y_bd_0, it_channel_0,
                    {0, 0, 0, (uint32_t)(y_offset + col * desired_m * Dout + out_offset)},
                    {1, 1, desired_m, desired_n * 4},
                    {0, 0, Dout, 1},
                    -1, 0, true
                );
                seq->npu_dma_memcpy_nd(
                    2, 0,
                    S2MM, IT[col + 4],
                    y_bd_1, it_channel_0,
                    {0, 0, 0, (uint32_t)(y_offset + col * desired_m * Dout + 4 * desired_n + out_offset)},
                    {1, 1, desired_m, desired_n * 4},
                    {0, 0, Dout, 1},
                    -1, 0, true
                );
                // send x
                seq->npu_dma_memcpy_nd(
                    2, 1,
                    MM2S, IT[2 * col],
                    x_bd, it_channel_0,
                    {0, 0, 0, x_offset + col * desired_m * Din},
                    {1, (uint32_t)K_div_k, desired_m, desired_k},
                    {0, desired_k, Din, 1},
                    -1, 0, false
                );
            }
            if (round > 0){
                for (int r = 0; r < 4; r++){
                    seq->npu_dma_wait(IT[r], S2MM, it_channel_0);
                    seq->npu_dma_wait(IT[r + 4], S2MM, it_channel_0);
                }
            }
        }// end j
    } // end i
    for (int r = 0; r < 4; r++){
        seq->npu_dma_wait(IT[r], S2MM, it_channel_0);
        seq->npu_dma_wait(IT[r + 4], S2MM, it_channel_0);
    }
    seq->cmds2seq();
}

#endif
