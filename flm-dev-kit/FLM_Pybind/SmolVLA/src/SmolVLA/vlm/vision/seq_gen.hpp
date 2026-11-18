// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __SMOLVLA__VISION__SEQ__GEN__HPP__
#define __SMOLVLA__VISION__SEQ__GEN__HPP__
#include "npu_utils/npu_utils.hpp"
#include "weight_desc.hpp"

namespace smolvla_vlm_vision_seq{
/// @brief build the NPU DMA sequence for bidirectional vision self-attention.
///        Attention is computed independently per image: each image's 1024
///        tokens form one block, and blocks are laid out contiguously in the
///        q/k/v buffers, so the sequence loops over images and offsets each by
///        one image's worth of elements (1024*768).
/// @param seq        sequence to populate (cleared and rebuilt)
/// @param L          tokens per image (1024)
/// @param DH         head dimension
/// @param Heads      number of attention heads
/// @param num_images number of images in the batch
/// @param need_set_rtp force re-writing the runtime params into the tiles
inline void gen_vision_attn_seq(
    npu_sequence* seq,
    const uint32_t L,
    const uint32_t DH,
    const int Heads,
    const int num_images,
    bool need_set_rtp = false
){

    static uint32_t last_L = 0;
    static int last_DH = DH;
    static int last_Heads = Heads;


    if (last_L != L) {
        need_set_rtp = true;
        last_L = L;
    }
    if (last_DH != DH) {
        need_set_rtp = true;
        last_DH = DH;
    }
    if (last_Heads != Heads) {
        need_set_rtp = true;
        last_Heads = Heads;
    }

    constexpr int lq = 64;
    npu_tiles IT[2][2] = {{IT0, IT1}, {IT6, IT7}};
    assert(L % (lq * 8) == 0);

    int l_begin_mha_address = 10496;

    seq->clear_cmds();
    if (need_set_rtp) {
        for (int row = 2; row < 6; row++){
            npu_tiles tile0 = get_tile(row, 0);
            npu_tiles tile1 = get_tile(row, 1);
            npu_tiles tile2 = get_tile(row, 6);
            npu_tiles tile3 = get_tile(row, 7);
            seq->rtp_write(tile0, l_begin_mha_address, L);
            seq->rtp_write(tile1, l_begin_mha_address, L);
            seq->rtp_write(tile2, l_begin_mha_address, L);
            seq->rtp_write(tile3, l_begin_mha_address, L);
        }
    }

    // each round, each CU works on 1 head and lc * 8 of l, total_cols / 2 is corresponding to the number of CUs
    const int data_per_round = lq * 8; 
    // zero padding the last round if necessary
    const int down_rounds = (L + data_per_round - 1) / data_per_round;
    for (int image_id = 0; image_id < num_images; image_id++) {
        uint32_t image_offset = 1024 * 768 * image_id;
        for (int head = 0; head < Heads / 2; head++){
            for (int round = 0; round < down_rounds; round++){
                int _round = image_id * Heads / 2 * down_rounds + head * down_rounds + round;
                int bd_offset = (_round % 2) * 8;
                // cu 0,
                for (int cu = 0; cu < 2; cu++){
                    int head_offset = head * 2 + cu;
                    // send q
                    size_t q_offset = head_offset * DH + round * data_per_round * DH * Heads + image_offset;
                    seq->npu_dma_memcpy_nd(
                        2, 1,
                        MM2S, IT[cu][0],
                        (npu_bd_id)(bd_offset + 1), it_channel_0,
                        {0, 0, 0, (uint32_t)q_offset},
                        {1, 1, (uint32_t)lq * 4, (uint32_t)(DH)},
                        {0, 0, (uint32_t)DH * Heads, 1},
                        -1, 0, false
                    );
                    seq->npu_dma_memcpy_nd(
                        2, 1,
                        MM2S, IT[cu][0],
                        (npu_bd_id)(bd_offset + 2), it_channel_1,
                        {0, 0, 0, (uint32_t)(q_offset + (size_t)lq * 4 * DH * Heads)},
                        {1, 1, (uint32_t)lq * 4, (uint32_t)(DH)},
                        {0, 0, (uint32_t)DH * Heads, 1},
                        -1, 0, false
                    );
                    // send k
                    size_t k_offset = head_offset * DH + image_offset;
                    seq->npu_dma_memcpy_nd(
                        2, 2,
                        MM2S, IT[cu][1],
                        (npu_bd_id)(bd_offset + 3), it_channel_0,
                        {0, 0, 0, (uint32_t)k_offset},
                        {1, (L / 512), 512, DH},
                        {0, (uint32_t)512 * DH * Heads, (uint32_t)DH * Heads, 1},
                        -1, 0, false
                    );
                    // send v
                    size_t v_offset = head_offset * DH + image_offset;
                    seq->npu_dma_memcpy_nd(
                        2, 3,
                        MM2S, IT[cu][1],
                        (npu_bd_id)(bd_offset + 4), it_channel_1,
                        {0, 0, 0, (uint32_t)v_offset},
                        {1, (L / 512), 512, DH},
                        {0, (uint32_t)512 * DH * Heads, (uint32_t)DH * Heads, 1},
                        -1, 0, false
                    );
                    for (int col = 0; col < 2; col++){    
                        // receive y
                        size_t y_offset = head_offset * DH + round * data_per_round * DH * Heads + col * lq * 4 * DH * Heads + image_offset; 
                        seq->npu_dma_memcpy_nd(
                            2, 0,
                            S2MM, IT[cu][col],
                            (npu_bd_id)(bd_offset + 0), it_channel_0,
                            {0, 0, 0, (uint32_t)y_offset},
                            {1, 1, (uint32_t)4 * lq, (uint32_t)(DH)},
                            {0, 0, (uint32_t)DH * Heads, 1},
                            -1, 0, true
                        );
                    }
                } // cu
                if (_round > 0){
                    for (int cu = 0; cu < 2; cu++){
                        for (int col = 0; col < 2; col++){
                            seq->npu_dma_wait(
                                IT[cu][col],
                                S2MM,
                                it_channel_0
                            );
                        } // col
                    } // cu
                }// round > 0
            }// round
        } // head
    }
    for (int cu = 0; cu < 2; cu++){
        for (int col = 0; col < 2; col++){
            seq->npu_dma_wait(
                IT[cu][col],
                S2MM,
                it_channel_0
            );
        } // col
    } // cu
    seq->cmds2seq();
}

inline void gen_vision_mm_seq(npu_sequence* seq, const uint32_t M, const weight_desc_t& weight, int32_t rtp_bias, int32_t rtp_act, bool force_rtp = false){
    static uint32_t last_M = 0;
    static uint32_t last_K = 0;
    static uint32_t last_N = 0;
    static int32_t  last_rtp_bias = -1;
    static int32_t  last_rtp_act = -1;
    const uint32_t w_base = (uint32_t)weight.offset / sizeof(bf16);
    constexpr npu_tiles IT[4] = {IT2, IT3, IT4, IT5};
    constexpr int total_cols = 4;
    constexpr int total_rows = 4;
    const uint32_t K = weight.shape[0];
    const uint32_t N = weight.shape[1];
    constexpr int m = 128;
    constexpr int k = 256;
    constexpr int n = 64;
    assert(N % k == 0);
    seq->clear_cmds();

    bool need_reset_rtp = force_rtp;

    constexpr int rtp_address = 56832;
    uint32_t K_div_k = K / k;
    const int M_div_m = M / m / total_rows;
    const int N_div_n  = N / n / total_cols; 

    if (force_rtp || (last_M != M)){
        need_reset_rtp = true;
        last_M = M;
        for (int row = 2; row < 6; row++){
            for (int col = 2; col < total_cols + 2; col++){
                npu_tiles tile = get_tile(row, col);
                seq->rtp_write(tile, rtp_address, M_div_m);
            }
        }
    }
    if (force_rtp || (last_K != K)){
        need_reset_rtp = true;
        for (int row = 2; row < 6; row++){
            for (int col = 2; col < total_cols + 2; col++){
                npu_tiles tile = get_tile(row, col);
                seq->rtp_write(tile, rtp_address + 4, K_div_k);
            }
        }
        last_K = K;
    }
    if (force_rtp || (last_rtp_act != rtp_act)){
        need_reset_rtp = true;
        for (int row = 2; row < 6; row++){
            for (int col = 2; col < total_cols + 2; col++){
                npu_tiles tile = get_tile(row, col);
                seq->rtp_write(tile, rtp_address + 8, rtp_act);
            }
        }
        last_rtp_act = rtp_act;
    }

    for (int j = 0; j < N_div_n; j++){
        if(rtp_bias == 1){
            // send b
            uint32_t b_offset = j * n * total_cols + w_base - N;
            for (int c = 0; c < total_cols; c++){
                seq->npu_dma_memcpy_nd(
                    2, 2,
                    MM2S, IT[c],
                    npu_bd_id(3), it_channel_1,
                    {0, 0, 0, b_offset + c * n},
                    {1, 1, 1, (uint32_t)n * k},
                    {0, 0, 0, 1},
                    -1, 0, false
                );
            }
        }
        for (int i = 0; i < M_div_m; i++){
            int round = j * M_div_m + i;
            int bd_offset = (round % 2) * 8;
            int y_offset = i * m * N * total_rows + j * n * total_cols;
            int x_offset = i * m * K * total_rows;
            uint32_t w_offset = j * n * K * total_cols + w_base;
            npu_bd_id y_bd = npu_bd_id(0 + bd_offset);
            npu_bd_id x_bd = npu_bd_id(1 + bd_offset);
            npu_bd_id w_bd = npu_bd_id(2 + bd_offset);
            // send w
            for (int c = 0; c < total_cols; c++){
                seq->npu_dma_memcpy_nd(
                    2, 2,
                    MM2S, IT[c],
                    w_bd, it_channel_1,
                    {0, 0, 0, w_offset + c * n * K},
                    {1, K_div_k, n, (uint32_t)k},
                    {0, k, K, 1},
                    -1, 0, false
                );
                // receive y
                seq->npu_dma_memcpy_nd(
                    2, 0,
                    S2MM, IT[c],
                    y_bd, it_channel_0,
                    {0, 0, 0, y_offset + c * m * N},
                    {1, 1, m, n * 4},
                    {0, 0, N, 1},
                    -1, 0, true
                );
                // send x
                seq->npu_dma_memcpy_nd(
                    2, 1,
                    MM2S, IT[c],
                    x_bd, it_channel_0,
                    {0, 0, 0, x_offset + c * m * K},
                    {1, (uint32_t)K_div_k, m, k},
                    {0, k, K, 1},
                    -1, 0, false
                );
            }
            if (round > 0){
                for (int r = 0; r < total_rows; r++){
                    seq->npu_dma_wait(IT[r], S2MM, it_channel_0);
                }
            }
        }
    }
    for (int r = 0; r < total_rows; r++){
        seq->npu_dma_wait(IT[r], S2MM, it_channel_0);
    }
    seq->cmds2seq();
}

}
#endif