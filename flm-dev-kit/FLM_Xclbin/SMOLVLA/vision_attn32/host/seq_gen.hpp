// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#pragma once
#include "npu_utils/npu_utils.hpp"

/// Vision attention on all 32 cores: 4 compute units (CU) of 8 cores, one head per CU per round,
/// 4 heads per head group; L (per image) must be a multiple of 512, tensors are [image][L][Heads*DH].
inline void gen_attn32_seq(
    npu_sequence* seq,
    const uint32_t L,
    const int DH,
    const int Heads,
    const int num_images,
    const int ncu = 4   // 4: all 32 cores; 2: the engine's 16-core layout (cols 0,1,6,7) for comparison
){
    constexpr int lq = 64;
    const int NCU = ncu;
    npu_tiles ITa[4][2] = {{IT0, IT1}, {IT2, IT3}, {IT4, IT5}, {IT6, IT7}};
    npu_tiles ITb[2][2] = {{IT0, IT1}, {IT6, IT7}};
    auto IT = [&](int cu, int c) { return NCU == 4 ? ITa[cu][c] : ITb[cu][c]; };
    const int l_begin_mha_address = 10496;

    seq->clear_cmds();
    for (int row = 2; row < 6; row++) {
        for (int col = 0; col < 8; col++) {
            if (NCU == 2 && col != 0 && col != 1 && col != 6 && col != 7) continue;
            seq->rtp_write(get_tile(row, col), l_begin_mha_address, L);
        }
    }

    const int data_per_round = lq * 8;
    const int down_rounds = (L + data_per_round - 1) / data_per_round;
    const int groups = Heads / NCU;
    int _round = 0;
    for (int image_id = 0; image_id < num_images; image_id++) {
        const size_t image_offset = (size_t)L * DH * Heads * image_id;
        for (int g = 0; g < groups; g++) {
            for (int round = 0; round < down_rounds; round++, _round++) {
                const int bd_offset = (_round % 2) * 8;
                for (int cu = 0; cu < NCU; cu++) {
                    const int head_offset = g * NCU + cu;
                    const size_t q_offset = (size_t)head_offset * DH + (size_t)round * data_per_round * DH * Heads + image_offset;
                    seq->npu_dma_memcpy_nd(2, 1, MM2S, IT(cu, 0), (npu_bd_id)(bd_offset + 1), it_channel_0,
                        {0, 0, 0, (uint32_t)q_offset}, {1, 1, (uint32_t)lq * 4, (uint32_t)DH},
                        {0, 0, (uint32_t)DH * Heads, 1}, -1, 0, false);
                    seq->npu_dma_memcpy_nd(2, 1, MM2S, IT(cu, 0), (npu_bd_id)(bd_offset + 2), it_channel_1,
                        {0, 0, 0, (uint32_t)(q_offset + (size_t)lq * 4 * DH * Heads)}, {1, 1, (uint32_t)lq * 4, (uint32_t)DH},
                        {0, 0, (uint32_t)DH * Heads, 1}, -1, 0, false);
                    const size_t kv_offset = (size_t)head_offset * DH + image_offset;
                    seq->npu_dma_memcpy_nd(2, 2, MM2S, IT(cu, 1), (npu_bd_id)(bd_offset + 3), it_channel_0,
                        {0, 0, 0, (uint32_t)kv_offset}, {1, (L / 512), 512, (uint32_t)DH},
                        {0, (uint32_t)512 * DH * Heads, (uint32_t)DH * Heads, 1}, -1, 0, false);
                    seq->npu_dma_memcpy_nd(2, 3, MM2S, IT(cu, 1), (npu_bd_id)(bd_offset + 4), it_channel_1,
                        {0, 0, 0, (uint32_t)kv_offset}, {1, (L / 512), 512, (uint32_t)DH},
                        {0, (uint32_t)512 * DH * Heads, (uint32_t)DH * Heads, 1}, -1, 0, false);
                    for (int col = 0; col < 2; col++) {
                        const size_t y_offset = (size_t)head_offset * DH + (size_t)round * data_per_round * DH * Heads
                                                + (size_t)col * lq * 4 * DH * Heads + image_offset;
                        seq->npu_dma_memcpy_nd(2, 0, S2MM, IT(cu, col), (npu_bd_id)(bd_offset + 0), it_channel_0,
                            {0, 0, 0, (uint32_t)y_offset}, {1, 1, (uint32_t)4 * lq, (uint32_t)DH},
                            {0, 0, (uint32_t)DH * Heads, 1}, -1, 0, true);
                    }
                }
                if (_round > 0) {
                    for (int cu = 0; cu < NCU; cu++)
                        for (int col = 0; col < 2; col++)
                            seq->npu_dma_wait(IT(cu, col), S2MM, it_channel_0);
                }
            }
        }
    }
    for (int cu = 0; cu < NCU; cu++)
        for (int col = 0; col < 2; col++)
            seq->npu_dma_wait(IT(cu, col), S2MM, it_channel_0);
    seq->cmds2seq();
}
