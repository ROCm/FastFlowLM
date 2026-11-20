// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file d3_npu_sequence.cpp
/// \brief Instruction sequences of the d3 decision engine: the prefill GEMM only.
/// \note  gen_dequant_mm_512 is minicpm_v_4_7_npu_sequence's (itself qwen3_5vl's) unchanged; it
///        drives FLM_Xclbin/Qwen3_5's mm.xclbin, the one bitstream this engine registers.
#include "d3_npu_sequence.hpp"
#include <algorithm>
#include <stdexcept>
#include <string>

void d3_npu_sequence::gen_dequant_mm_512(npu_sequence* seq, const uint32_t L, const uint32_t Din, const uint32_t Dout, const size_t weight_offset, const size_t out_offset, const int rtp_act){
    static constexpr npu_tiles IT[8] = {IT0, IT1, IT2, IT3, IT4, IT5, IT6, IT7};

    static constexpr int m = QXNX_ROW_BLOCK_SIZE;
    static constexpr uint32_t desired_m = 128;
    static constexpr uint32_t desired_k = 512;
    static constexpr uint32_t desired_n = 64;
    //  %K_div_k_2_7 = aie.buffer(%tile_7_2) {address = 54784 : i32, sym_name = "K_div_k_2_7"} : memref<16xi32>
    // %rtp_act_2_7 = aie.buffer(%tile_7_2) {address = 54848 : i32, sym_name = "rtp_act_2_7"} : memref<16xi32>
    // %K_div_k_3_0 = aie.buffer(%tile_0_3) {address = 61696 : i32, sym_name = "K_div_k_3_0"} : memref<16xi32>
    // %rtp_act_3_0 = aie.buffer(%tile_0_3) {address = 61760 : i32, sym_name = "rtp_act_3_0"} : memref<16xi32>

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

void d3_npu_sequence::gen_bidirectional_attention(npu_sequence* seq, const uint32_t L_valid, const size_t q_base,
                                                  const size_t o_base, const size_t kv_half, const uint32_t q_rows){
    // FLM_Xclbin/Qwen3_5/bidirectional_attention_prefill/host/seq_gen.hpp with base offsets: 4 CUs
    // run 4 q heads at a time, each CU's 8 tiles 16 q rows each; K and V go to every tile.
    static constexpr int lc = 16;
    npu_tiles IT[4][2] = {{IT0, IT1}, {IT2, IT3}, {IT4, IT5}, {IT6, IT7}};
    const int Heads = desc->num_heads;
    const int NKV = desc->num_kv_heads;
    const int DH = desc->DH;
    const int num_cu = 4;
    // Hard checks, not asserts: the release build defines NDEBUG, and a group this schedule did
    // not handle used to map heads to the wrong kv head without any message.
    if (NKV <= 0 || Heads <= 0 || Heads % NKV != 0 || DH != 256)
        throw std::runtime_error("d3_npu: attention array path needs head_dim 256 and num_heads a multiple of num_kv_heads (got "
                                 + std::to_string(Heads) + " heads, " + std::to_string(NKV) + " kv heads, head_dim " + std::to_string(DH) + ")");
    if (L_valid == 0)
        throw std::invalid_argument("d3_npu: attention sequence for zero keys");
    const int group = Heads / NKV;
    // a kv head's q heads run in passes of up to 4 (one per compute unit). A last pass with fewer
    // real heads (GQA 6: 4 + 2) feeds its idle units duplicates of real heads of the same pass:
    // they compute and store the identical rows to the identical addresses.
    const int passes_per_kv = (group + num_cu - 1) / num_cu;
    const uint32_t q_rows_ = q_rows ? q_rows : L_valid;
    const int Q_chunks = (q_rows_ + ATTN_ROWS - 1) / ATTN_ROWS;
    const int kv_length = (int)((L_valid + ATTN_ROWS - 1) / ATTN_ROWS) * ATTN_ROWS;
    const uint32_t q_stride = (uint32_t)DH * Heads;
    const uint32_t kv_stride = (uint32_t)DH * NKV;

    seq->clear_cmds();
    for (int row = 2; row < 6; row++){
        for (int col = 0; col < 8; col++){
            npu_tiles tile = get_tile(row, col);
            seq->rtp_write(tile, attn_l_begin_address, 0);
            seq->rtp_write(tile, attn_l_chunks_address, Q_chunks);
            seq->rtp_write(tile, attn_window_address, L_valid);
        }
    }
    for (int pass = 0; pass < NKV * passes_per_kv; pass++){
        const int kv_head = pass / passes_per_kv;
        const int first = (pass % passes_per_kv) * num_cu;           // first q head of the pass within its group
        const int real = std::min(num_cu, group - first);            // real heads in this pass
        for (int round = 0; round < Q_chunks; round++){
            const int bd_offset = (round % 2) * 8;
            for (int cu = 0; cu < num_cu; cu++){
                const int head = kv_head * group + first + cu % real;
                for (int col = 0; col < 2; col++){
                    const size_t y_offset = o_base + (size_t)head * DH + (size_t)(round * ATTN_ROWS + col * lc * 4) * q_stride;
                    seq->npu_dma_memcpy_nd(2, 0, S2MM, IT[cu][col], (npu_bd_id)(bd_offset + 0), it_channel_0,
                        {0, 0, 0, (uint32_t)y_offset}, {1, 1, (uint32_t)4 * lc, (uint32_t)DH}, {0, 0, q_stride, 1},
                        -1, 0, true);
                }
                const size_t q_offset = q_base + (size_t)head * DH + (size_t)round * ATTN_ROWS * q_stride;
                seq->npu_dma_memcpy_nd(2, 1, MM2S, IT[cu][0], (npu_bd_id)(bd_offset + 1), it_channel_0,
                    {0, 0, 0, (uint32_t)q_offset}, {1, 1, (uint32_t)lc * 4, (uint32_t)DH}, {0, 0, q_stride, 1},
                    -1, 0, false);
                seq->npu_dma_memcpy_nd(2, 1, MM2S, IT[cu][0], (npu_bd_id)(bd_offset + 2), it_channel_1,
                    {0, 0, 0, (uint32_t)(q_offset + (size_t)lc * 4 * q_stride)}, {1, 1, (uint32_t)lc * 4, (uint32_t)DH},
                    {0, 0, q_stride, 1}, -1, 0, false);
            }
            const size_t k_offset = (size_t)kv_head * DH;
            seq->npu_dma_memcpy_nd(2, 2, MM2S, IT[1][1], (npu_bd_id)(bd_offset + 3), it_channel_0,
                {0, 0, 0, (uint32_t)k_offset}, {1, (uint32_t)kv_length / ATTN_ROWS, (uint32_t)ATTN_ROWS, (uint32_t)DH},
                {0, (uint32_t)ATTN_ROWS * kv_stride, kv_stride, 1}, -1, 0, false);
            seq->npu_dma_memcpy_nd(2, 2, MM2S, IT[1][1], (npu_bd_id)(bd_offset + 4), it_channel_1,
                {0, 0, 0, (uint32_t)(k_offset + kv_half)}, {1, (uint32_t)kv_length / ATTN_ROWS, (uint32_t)ATTN_ROWS, (uint32_t)DH},
                {0, (uint32_t)ATTN_ROWS * kv_stride, kv_stride, 1}, -1, 0, false);
            if (round > 0){
                for (int cu = 0; cu < num_cu; cu++)
                    for (int col = 0; col < 2; col++) seq->npu_dma_wait(IT[cu][col], S2MM, it_channel_0);
            }
        }
        for (int cu = 0; cu < num_cu; cu++)
            for (int col = 0; col < 2; col++) seq->npu_dma_wait(IT[cu][col], S2MM, it_channel_0);
    }
    seq->cmds2seq();
}
