// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include "gemma_text_gemm_detail.hpp"
#include "npu_utils/npu_utils.hpp"
#include <cassert>
#include <algorithm> // Required for std::max
#include <string>

// // constructors
GemmaTextGemm::Impl::Impl(LM_Config& config) : config(config) {
}

GemmaTextGemm::Impl::~Impl() = default;

// methods
void GemmaTextGemm::Impl::generate_seq(npu_sequence* seq, const uint32_t M, const uint32_t K, const uint32_t N, const uint32_t weight_offset){
    npu_sequence& seq_ptr = *seq;
    int K_div_k = K / k;
    seq_ptr.clear_cmds();
    seq_ptr.npu_preemption(0);
    // set K_DIV_K
    for (int row = 2; row < 6; row++){
        for (int col = 0; col < 8; col++){
            npu_tiles tile = get_tile(row, col);
            seq_ptr.rtp_write(tile, CT_rtp_address, K_div_k);
        }
    }
    int M_div_m = M / m / total_rows;
    int N_div_n = N / n / total_cols;
    assert(M % (m * total_rows) == 0);
    assert(N % (n * total_cols) == 0);
    for (int j = 0; j < N_div_n; j++){
        for (int i = 0; i < M_div_m; i++){
            int round = j * M_div_m + i;
            int is_pong = round % 2;
            int bd_offset = is_pong * 8;
            int y_offset = i * m * N * total_rows + j * n * total_cols;
            int x_offset = i * m * K * total_rows;
            npu_bd_id y_bd = npu_bd_id(0 + bd_offset);
            npu_bd_id x_bd = npu_bd_id(1 + bd_offset);
            npu_bd_id w_bd = npu_bd_id(2 + bd_offset);
            for (int r = 0; r < total_rows; r++){
                // receive y
                seq_ptr.npu_dma_memcpy_nd(
                    2, 0,
                    S2MM, IT[r],
                    y_bd, it_channel_0,
                    {0, 0, 0, y_offset + r * m * N},
                    {1, 1, m, n * total_rows},
                    {0, 0, N, 1},
                    -1, 0, true
                );

                seq_ptr.npu_dma_memcpy_nd(
                    2, 0,
                    S2MM, IT[r + 4],
                    y_bd, it_channel_0,
                    {0, 0, 0, y_offset + r * m * N + 4 * n},
                    {1, 1, m, n * total_rows},
                    {0, 0, N, 1},
                    -1, 0, true
                );

                // send x
                seq_ptr.npu_dma_memcpy_nd(
                    2, 1,
                    MM2S, IT[r + 2],
                    x_bd, it_channel_0,
                    {0, 0, 0, x_offset + r * m * K},
                    {1, (uint32_t)K_div_k, m, k},
                    {0, k, K, 1},
                    -1, 0, false, no_cache
                );
            } // end r
            for (int c = 0; c < total_cols; c++){
                uint32_t w_offset = j * n * K * total_cols + c * n * K + weight_offset;
                seq_ptr.npu_dma_memcpy_nd(
                    2, 2,
                    MM2S, IT[c],
                    w_bd, it_channel_1,
                    {0, 0, 0, w_offset},
                    {1, 1, 1, (uint32_t)K_div_k * n * k},
                    {0, 0, 0, 1},
                    -1, 0, false, no_cache
                );
            }
            if (round > 0){
                for (int c = 0; c < total_cols; c++){
                    seq_ptr.npu_dma_wait(IT[c], S2MM, it_channel_0);
                }
            }
        }
    }
    for (int c = 0; c < total_cols; c++){
        seq_ptr.npu_dma_wait(IT[c], S2MM, it_channel_0);
    }
    seq_ptr.cmds2seq();
}
// wrappers
GemmaTextGemm::GemmaTextGemm(LM_Config& config) : _impl(new Impl(config)){}
GemmaTextGemm::~GemmaTextGemm(){
    delete _impl;
}

void GemmaTextGemm::generate_seq(npu_sequence* seq, const uint32_t M, const uint32_t K, const uint32_t N, const uint32_t weight_offset){
    _impl->generate_seq(seq, M, K, N, weight_offset);
}
