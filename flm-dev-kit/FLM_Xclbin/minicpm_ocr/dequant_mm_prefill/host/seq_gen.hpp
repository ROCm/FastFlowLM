// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#ifndef __SEQ_GEN_HPP__
#define __SEQ_GEN_HPP__
#pragma once
#include "npu_utils/npu_utils.hpp"
#include "weights_desc.hpp"

// Mirror of qwen3_5vl_npu_sequence::gen_dequant_mm_512 (kernels_2b path) in
// FLM_DLL/detail/qwen3_5vl_npu/qwen3_5vl_npu_sequence.cpp. Keep the two in sync.
// RTP addresses: after a rebuild, re-read them from
//   build/<MODEL_TYPE>/bitstream/from_iron/dequant_mm.mlir.prj/input_with_addresses.mlir
// (K_div_k_2_7 / rtp_act_2_7 and K_div_k_3_0 / rtp_act_3_0).
static constexpr uint32_t CT_rtp_address_even_row  = 55552;
static constexpr uint32_t act_rtp_address_even_row = 55616;
static constexpr uint32_t CT_rtp_address_odd_row   = 64512;
static constexpr uint32_t act_rtp_address_odd_row  = 64576;

// GLU slots (README.md "GLU epilogue"): one 16 KB slot per (128-row block, 32 hid features), [8 groups][4 k blocks]
// [2 row blocks][72 B] at its start; slot (rb, fs) at (rb * (IS / 32) + fs) * 16 KB.
static constexpr uint32_t glu_slot_bytes = 16384;
static constexpr uint32_t glu_slot_used = 4608;
static constexpr uint32_t glu_slot_features = 32;
inline size_t glu_slots_bytes(size_t L, size_t IS) { return L / 128 * (IS / glu_slot_features) * glu_slot_bytes; }

void gen_dequant_mm_512(npu_sequence* seq, const uint32_t L, const uint32_t Din, const uint32_t Dout, const size_t weight_offset, const size_t out_offset, const int rtp_act, const bool x_from_glu_slots = false){
    static constexpr npu_tiles IT[8] = {IT0, IT1, IT2, IT3, IT4, IT5, IT6, IT7};

    static constexpr int m = QXNX_ROW_BLOCK_SIZE;
    static constexpr int k = QXNX_COL_BLOCK_SIZE;
    static constexpr uint32_t elems = m * k;
    static constexpr uint32_t desired_m = 128;
    static constexpr uint32_t desired_k = 512;
    static constexpr uint32_t desired_n = 64;

    uint32_t a_block_size = get_quantization_byte_size(m * k, flm_q4k);

    const uint32_t blocks_per_row = Din / k;
    int K_div_k = Din / desired_k;

    const uint32_t columns = 8;
    assert(Dout / m / columns > 0);
    assert(Dout % (m * columns) == 0);

    seq->clear_cmds();

    int M_div_m = L / (desired_m * 4); // 4 row-blocks processed per i iteration (IT[col], col=0..3)
    int N_div_n = Dout / desired_n / columns;
    assert(L % (desired_m * 4) == 0);

    // y from a mem tile: its 4 cores' 128 x 64 tiles one after another
    const std::vector<uint32_t> y_sizes = {1, 4, desired_m, desired_n};
    const std::vector<uint32_t> y_strides = {0, desired_n, Dout, 1};

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
            int w_offset = get_quantization_byte_size((size_t)j * desired_n * Din * columns, flm_q4k) + weight_offset;

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
                    {0, 0, 0, (uint32_t)(w_offset + get_quantization_byte_size((size_t)col * desired_n * Din, flm_q4k))},
                    {1, 1, 1, (uint32_t)get_quantization_byte_size(desired_n * Din, flm_q4k)},
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
                    {0, 0, 0, (uint32_t)(w_offset + get_quantization_byte_size((size_t)col * desired_n * Din, flm_q4k) + get_quantization_byte_size(4 * desired_n * Din, flm_q4k))},
                    {1, 1, 1, (uint32_t)get_quantization_byte_size(desired_n * Din, flm_q4k)},
                    {0, 0, 0, 1},
                    -1, 0, false, aggressive_cache
                );
                // receive y
                seq->npu_dma_memcpy_nd(
                    2, 0,
                    S2MM, IT[col],
                    y_bd_0, it_channel_0,
                    {0, 0, 0, (uint32_t)(y_offset + col * desired_m * Dout + out_offset)},
                    y_sizes,
                    y_strides,
                    -1, 0, true
                );
                seq->npu_dma_memcpy_nd(
                    2, 0,
                    S2MM, IT[col + 4],
                    y_bd_1, it_channel_0,
                    {0, 0, 0, (uint32_t)(y_offset + col * desired_m * Dout + 4 * desired_n + out_offset)},
                    y_sizes,
                    y_strides,
                    -1, 0, true
                );
                // send x: bfp16 tiles [16-row group][Din / 8][2][72 B], per 256-k half 8 groups x 4608 B
                const uint32_t group_bytes = Din * 18;
                const uint32_t first_group = (i * desired_m * 4 + col * desired_m) / 16;
                if (x_from_glu_slots) {
                    // the same stream from the GLU slots: per 256 k, 8 groups x the 8 slots' 576 B
                    const uint32_t rb = i * 4 + col;
                    const uint32_t slots_per_rb = Din / glu_slot_features;
                    seq->npu_dma_memcpy_nd(
                        1, 1,
                        MM2S, IT[2 * col],
                        x_bd, it_channel_0,
                        {0, 0, 0, rb * slots_per_rb * glu_slot_bytes},
                        {Din / 256, desired_m / 16, 8, glu_slot_used / 8},
                        {8 * glu_slot_bytes, glu_slot_used / 8, glu_slot_bytes, 1},
                        -1, 0, false
                    );
                }
                else {
                seq->npu_dma_memcpy_nd(
                    1, 1,
                    MM2S, IT[2 * col],
                    x_bd, it_channel_0,
                    {0, 0, 0, first_group * group_bytes},
                    {Din / 256, desired_m / 16, 9, 512},
                    {4608, group_bytes, 512, 1},
                    -1, 0, false
                );
                }
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

/// up/gate projection with the GLU epilogue (rtp_act 2): weights as the engine stores them, [up 512 rows][gate 512 rows]
/// per 512 hid features, each 64 rows as [256 k][2 x 32-row blocks]. Round j, core column c makes hid features
/// j * 256 + c * 32 .. + 32: its shim sends, per 256 k, the 32 up rows' block then the matching gate rows' block.
/// y: each core's 16 KB goes to its slot; a shim's 4 cores are 4 consecutive slots.
void gen_dequant_mm_glu(npu_sequence* seq, const uint32_t L, const uint32_t Din, const uint32_t IS, const size_t weight_offset, const size_t out_offset){
    static constexpr npu_tiles IT[8] = {IT0, IT1, IT2, IT3, IT4, IT5, IT6, IT7};
    static constexpr uint32_t desired_m = 128;
    static constexpr uint32_t desired_k = 512;
    const uint32_t columns = 8;
    const uint32_t feats_per_round = columns * glu_slot_features;
    const int K_div_k = Din / desired_k;
    assert(IS % 512 == 0);
    assert(Din % desired_k == 0);
    assert(L % (desired_m * 4) == 0);

    seq->clear_cmds();
    const int M_div_m = L / (desired_m * 4);
    const int N_div_n = IS / feats_per_round;
    const uint32_t block = get_quantization_byte_size(QXNX_ROW_BLOCK_SIZE * QXNX_COL_BLOCK_SIZE, flm_q4k);
    const uint32_t up_to_gate = get_quantization_byte_size(512 * Din, flm_q4k);
    const uint32_t slots_per_rb = IS / glu_slot_features;

    for (int row = 2; row < 6; row += 2){
        for (uint32_t col = 0; col < columns; col++){
            npu_tiles tile = get_tile(row, col);
            seq->rtp_write(tile, CT_rtp_address_even_row, K_div_k);
            seq->rtp_write(tile, act_rtp_address_even_row, 2);
        }
    }
    for (int row = 3; row < 6; row += 2){
        for (uint32_t col = 0; col < columns; col++){
            npu_tiles tile = get_tile(row, col);
            seq->rtp_write(tile, CT_rtp_address_odd_row, K_div_k);
            seq->rtp_write(tile, act_rtp_address_odd_row, 2);
        }
    }

    for (int i = 0; i < M_div_m; i++){
        for (int j = 0; j < N_div_n; j++){
            const int round = i * N_div_n + j;
            const int bd_offset = (round % 2) * 8;
            npu_bd_id y_bd_0 = npu_bd_id(0 + bd_offset);
            npu_bd_id y_bd_1 = npu_bd_id(3 + bd_offset);
            npu_bd_id x_bd = npu_bd_id(1 + bd_offset);
            npu_bd_id w_bd_0 = npu_bd_id(2 + bd_offset);
            npu_bd_id w_bd_1 = npu_bd_id(4 + bd_offset);
            for (uint32_t col = 0; col < 4; col++){
                for (uint32_t side = 0; side < 2; side++){
                    const uint32_t c = col + 4 * side;
                    const uint32_t f = j * feats_per_round + c * glu_slot_features;
                    const uint32_t u = (f / 512) * 1024 + f % 512;
                    const uint32_t up = weight_offset + get_quantization_byte_size((size_t)(u / 64) * 64 * Din, flm_q4k) + (size_t)((u % 64) / 32) * block;
                    seq->npu_dma_memcpy_nd(
                        1, 2,
                        MM2S, IT[c],
                        side ? w_bd_1 : w_bd_0, it_channel_1,
                        {0, 0, 0, up},
                        {Din / 256, 2, 2, block / 2},
                        {2 * block, up_to_gate, block / 2, 1},
                        -1, 0, false, aggressive_cache
                    );
                }
                const uint32_t rb = i * 4 + col;
                for (uint32_t side = 0; side < 2; side++){
                    seq->npu_dma_memcpy_nd(
                        1, 0,
                        S2MM, IT[col + 4 * side],
                        side ? y_bd_1 : y_bd_0, it_channel_0,
                        {0, 0, 0, (uint32_t)(out_offset + (rb * slots_per_rb + j * columns + 4 * side) * glu_slot_bytes)},
                        {1, 1, 1, 4 * glu_slot_bytes},
                        {0, 0, 0, 1},
                        -1, 0, true
                    );
                }
                const uint32_t group_bytes = Din * 18;
                const uint32_t first_group = (i * desired_m * 4 + col * desired_m) / 16;
                seq->npu_dma_memcpy_nd(
                    1, 1,
                    MM2S, IT[2 * col],
                    x_bd, it_channel_0,
                    {0, 0, 0, first_group * group_bytes},
                    {Din / 256, desired_m / 16, 9, 512},
                    {4608, group_bytes, 512, 1},
                    -1, 0, false
                );
            }
            if (round > 0){
                for (int r = 0; r < 4; r++){
                    seq->npu_dma_wait(IT[r], S2MM, it_channel_0);
                    seq->npu_dma_wait(IT[r + 4], S2MM, it_channel_0);
                }
            }
        }
    }
    for (int r = 0; r < 4; r++){
        seq->npu_dma_wait(IT[r], S2MM, it_channel_0);
        seq->npu_dma_wait(IT[r + 4], S2MM, it_channel_0);
    }
    seq->cmds2seq();
}

#endif
