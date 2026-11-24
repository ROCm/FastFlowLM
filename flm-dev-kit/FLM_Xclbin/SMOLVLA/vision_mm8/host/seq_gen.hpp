// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __SMOLVLA__VISION_MM8__SEQ__GEN__HPP__
#define __SMOLVLA__VISION_MM8__SEQ__GEN__HPP__
#include "npu_utils/npu_utils.hpp"

/// 8-column matmul y[M,N] = x[M,K] * w[N,K]^T (+ bias[N], stored right before w).
/// 32 cores: core row r works on rows [r*128, +128) of a 512-row block, core column c on 48 of the 384 columns of
/// an N strip. Memtile / shim column c feeds w for core column c, x for row c%4 (to cores 4*(c/4)..+3) and gathers y
/// of those 4 cores (rows block c%4, columns (c/4)*192..+192 of the strip).
inline void gen_mm8_seq(npu_sequence* seq, uint32_t M, uint32_t K, uint32_t N, uint32_t w_base, int rtp_bias, int rtp_act,
                        int rtp_address, bool set_rtp = true) {
    constexpr npu_tiles IT[8] = {IT0, IT1, IT2, IT3, IT4, IT5, IT6, IT7};
    constexpr int total_cols = 8;
    constexpr int total_rows = 4;
    constexpr uint32_t m = 128, k = 256, n = 48;
    assert(M % (m * total_rows) == 0 && K % k == 0 && N % (n * total_cols) == 0);
    seq->clear_cmds();
    const uint32_t K_div_k = K / k;
    const int M_div_m = M / m / total_rows;
    const int N_div_n = N / n / total_cols;
    if (set_rtp) {
        for (int row = 2; row < 6; row++) {
            for (int col = 0; col < total_cols; col++) {
                npu_tiles tile = get_tile(row, col);
                seq->rtp_write(tile, rtp_address, M_div_m);
                seq->rtp_write(tile, rtp_address + 4, K_div_k);
                seq->rtp_write(tile, rtp_address + 8, rtp_act);
            }
        }
    }
    for (int j = 0; j < N_div_n; j++) {
        if (rtp_bias == 1) {
            const uint32_t b_offset = j * n * total_cols + w_base - N;
            for (int c = 0; c < total_cols; c++) {
                seq->npu_dma_memcpy_nd(2, 2, MM2S, IT[c], npu_bd_id(3), it_channel_1,
                                       {0, 0, 0, b_offset + c * n}, {1, 1, 1, n * k}, {0, 0, 0, 1}, -1, 0, false);
            }
        }
        for (int i = 0; i < M_div_m; i++) {
            const int round = j * M_div_m + i;
            const int bd_offset = (round % 2) * 8;
            const uint32_t row_base = i * m * total_rows;
            const uint32_t w_offset = j * n * K * total_cols + w_base;
            for (int c = 0; c < total_cols; c++) {
                const uint32_t r = c % 4, g = c / 4;
                seq->npu_dma_memcpy_nd(2, 2, MM2S, IT[c], npu_bd_id(2 + bd_offset), it_channel_1,
                                       {0, 0, 0, w_offset + c * n * K}, {1, K_div_k, n, k}, {0, k, K, 1}, -1, 0, false);
                seq->npu_dma_memcpy_nd(2, 0, S2MM, IT[c], npu_bd_id(0 + bd_offset), it_channel_0,
                                       {0, 0, 0, (row_base + r * m) * N + j * n * total_cols + g * n * 4},
                                       {1, 1, m, n * 4}, {0, 0, N, 1}, -1, 0, true);
                if (c < 4) {  // x of row block r: one read, multicast to memtiles r and r + 4
                    seq->npu_dma_memcpy_nd(2, 1, MM2S, IT[c], npu_bd_id(1 + bd_offset), it_channel_0,
                                           {0, 0, 0, (row_base + r * m) * K}, {1, K_div_k, m, k}, {0, k, K, 1}, -1, 0, false);
                }
            }
            if (round > 0) {
                for (int c = 0; c < total_cols; c++) seq->npu_dma_wait(IT[c], S2MM, it_channel_0);
            }
        }
    }
    for (int c = 0; c < total_cols; c++) seq->npu_dma_wait(IT[c], S2MM, it_channel_0);
    seq->cmds2seq();
}
#endif
