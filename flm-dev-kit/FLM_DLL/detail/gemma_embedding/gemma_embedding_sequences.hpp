// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#pragma once
#include "npu_utils/npu_utils.hpp"

static constexpr int32_t total_cols = 8;
static constexpr int32_t total_rows = 4;

void gen_mm_seq(npu_sequence* seq, const uint32_t M, const uint32_t K, const uint32_t N, const uint32_t weight_offset){
    static constexpr int m = 16;
    static constexpr int k = 128;
    static constexpr int n = 64;

    npu_sequence& seq_ptr = *seq;
    constexpr npu_tiles IT[8] = {IT0, IT1, IT2, IT3, IT4, IT5, IT6, IT7};
    int K_div_k = K / k;
    // K_div_k RTP of FLM_Xclbin/Embedding_Gemma/MM, pinned there (K_DIV_K_RTP_ADDRESS in iron/mm_aie.py) so a
    // rebuild cannot move it. A wrong address leaves the cores on a stale K count: wrong projections, and DMA /
    // core K-loop mismatches that hang a command until the ~60 s driver timeout. The bitstream must be the one
    // built from that design (not the pre-2026-11 FastFlowLM mm.xclbin, whose RTP was at 46208).
    constexpr uint32_t CT_rtp_address = 61440;
    seq_ptr.clear_cmds();
    seq_ptr.npu_preemption(0);
    // set K_DIV_K
    for (int row = 2; row < 6; row++){
        for (int col = 0; col < 8; col++){
            npu_tiles tile = get_tile(row, col);
            seq_ptr.rtp_write(tile, CT_rtp_address, K_div_k);
        }
    }
    int M_div_m = M / m / total_cols;
    int N_div_n = N / n / total_rows;
    for (int j = 0; j < N_div_n; j++){
        for (int i = 0; i < M_div_m; i++){
            int round = j * M_div_m + i;
            int is_pong = round % 2;
            int bd_offset = is_pong * 8;
            int y_offset = i * m * N * total_cols + j * n * total_rows;
            int x_offset = i * m * K * total_cols;
            npu_bd_id y_bd = npu_bd_id(0 + bd_offset);
            npu_bd_id x_bd = npu_bd_id(1 + bd_offset);
            npu_bd_id w_bd = npu_bd_id(2 + bd_offset);
            for (int c = 0; c < total_cols; c++){
                // receive y
                seq_ptr.npu_dma_memcpy_nd(
                    2, 0,
                    S2MM, IT[c],
                    y_bd, it_channel_0,
                    {0, 0, 0, y_offset + c * m * N},
                    {1, 4, m, n},
                    {0, n, N, 1},
                    -1, 0, true
                );

                // send x
                seq_ptr.npu_dma_memcpy_nd(
                    2, 1,
                    MM2S, IT[c],
                    x_bd, it_channel_0,
                    {0, 0, 0, x_offset + c * m * K},
                    {1, (uint32_t)K_div_k, m, k},
                    {0, k, K, 1},
                    -1, 0, false
                );
            } // end c
            for (int r = 0; r < total_rows; r++){
                uint32_t w_offset = j * n * K * total_rows + r * n * K + weight_offset;
                seq_ptr.npu_dma_memcpy_nd(
                    2, 2,
                    MM2S, IT[r + 2],
                    w_bd, it_channel_1,
                    {0, 0, 0, w_offset},
                    {1, (uint32_t)K_div_k, n, k},
                    {0, k, K, 1},
                    -1, 0, false
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

void gen_mha_engine_seq(npu_sequence* seq, const uint32_t LK_end, const uint32_t LQ_end, const uint32_t L){
    constexpr int DQ = 768;
    constexpr int DK = 256;
    constexpr int DV = DK;
    constexpr int DH = 256;
    constexpr int MAX_L = 2048;
    constexpr int KV_CACHE_SIZE = MAX_L * (DK + DV);
    constexpr int K_OFFSET = 0;
    constexpr int V_OFFSET = KV_CACHE_SIZE / 2;
    constexpr int total_cols = 8;
    constexpr int total_rows = 3;
    constexpr npu_tiles IT[8] = {IT0, IT1, IT2, IT3, IT4, IT5, IT6, IT7};
    constexpr int lc = 16;
    const int G = DQ / DK;
    const int lk_end_mha_address = 11776;
    assert(LQ_end % (16 * 8) == 0);
    seq->clear_cmds();
    seq->npu_preemption(0);
    for (int row = 2; row < total_rows + 2; row++){
        for (int col = 0; col < total_cols; col++){
            npu_tiles tile = get_tile(row, col);
            seq->rtp_write(tile, lk_end_mha_address, L); // 32 is lk
        }
    }
    int new_L = LQ_end;

    for (int round = 0; round < new_L / 16 / 8; round++){
        int bd_offset = (round % 2) * 8;
        for (int col = 0; col < total_cols; col++){ // col is bind to l
            // receive y
            size_t y_offset = (round * lc * total_cols + col * lc) * DQ; // y does not have history
            seq->npu_dma_memcpy_nd(
                2, 0,
                S2MM, IT[col],
                (npu_bd_id)(bd_offset + 0), it_channel_0,
                {0, 0, 0, (uint32_t)y_offset},
                {1, 1, (uint32_t)1, (uint32_t)(lc * DQ)},
                {0, 0, (uint32_t)0, 1},
                -1, 0, true
            );
            // send q
            size_t q_offset = (round * lc * total_cols + col * lc) * DQ; // q does not have history
            seq->npu_dma_memcpy_nd(
                2, 1,
                MM2S, IT[col],
                (npu_bd_id)(bd_offset + 1), it_channel_0,
                {0, 0, 0, (uint32_t)q_offset},
                {1, 1, (uint32_t)1, (uint32_t)(lc * DQ)},
                {0, 0, (uint32_t)0, 1},
                -1, 0, false
            );
        }
        // send k
        seq->npu_dma_memcpy_nd(
            2, 2,
            MM2S, IT[0],
            (npu_bd_id)(bd_offset + 2), it_channel_1,
            {0, 0, 0, K_OFFSET},
            {1, 1, 1, LK_end * DK},
            {0, 0, 0, 1},
            -1, 0, false
        );

        // send v
        seq->npu_dma_memcpy_nd(
            2, 2,
            MM2S, IT[4],
            (npu_bd_id)(bd_offset + 3), it_channel_1,
            {0, 0, 0, (uint32_t)V_OFFSET},
            {1, 1, 1, LK_end * DK},
            {0, 0, 0, 1},
            -1, 0, false
        );

        if (round > 0){
            for (int col = 0; col < total_cols; col++){
                seq->npu_dma_wait(
                    IT[col],
                    S2MM,
                    it_channel_0
                );
            }
        }
    }
    for (int col = 0; col < total_cols; col++){
        seq->npu_dma_wait(
            IT[col],
            S2MM,
            it_channel_0
        );
    }
    seq->cmds2seq();
}

void gen_sliding_mha_engine_seq(npu_sequence* seq, const uint32_t L){
    constexpr int DQ = 768;
    constexpr int DK = 256;
    constexpr int DV = DK;
    constexpr int MAX_L = 2048;
    constexpr int W = 257;
    constexpr int KV_CACHE_SIZE = MAX_L * (DK + DV);
    constexpr uint32_t K_OFFSET = 0;
    constexpr uint32_t V_OFFSET = KV_CACHE_SIZE / 2;
    constexpr int CHUNK_SIZE = 128;
    constexpr int total_cols = 8;
    constexpr int total_rows = 4;
    constexpr npu_tiles IT[8] = {IT0, IT1, IT2, IT3, IT4, IT5, IT6, IT7};
    constexpr int lc = 16;
    const int G = DQ / DK;
    const int L_buffer_address = 14848;
    const int window_size_address = 14880;
    seq->clear_cmds();
    seq->npu_preemption(0);
    for (int row = 2; row < 5; row++){
        for (int col = 0; col < total_cols; col++){
            npu_tiles tile = get_tile(row, col);
            seq->rtp_write(tile, L_buffer_address, L);
            seq->rtp_write(tile, window_size_address, W);
        }
    }
    int L_PADDED = (L + CHUNK_SIZE - 1) / CHUNK_SIZE * CHUNK_SIZE;
    int total_chunks = L_PADDED / CHUNK_SIZE;

    for (int chunk_idx = 0; chunk_idx < total_chunks; chunk_idx++){
        int chunk_q_begin = chunk_idx * CHUNK_SIZE;
        int chunk_q_end = chunk_q_begin + CHUNK_SIZE - 1;
        int chunk_k_begin = chunk_q_begin - W + 1;
        int chunk_k_end = chunk_q_end + W;
        chunk_k_begin = chunk_k_begin < 0 ? 0 : chunk_k_begin;
        chunk_k_end = chunk_k_end > L ? L : chunk_k_end;
        int k_begin = chunk_k_begin / CHUNK_SIZE * CHUNK_SIZE;
        int k_end = (chunk_k_end + CHUNK_SIZE - 1) / CHUNK_SIZE * CHUNK_SIZE;
        int k_length = k_end - k_begin;
        int k_chunks = k_length / CHUNK_SIZE;
        int bd_offset = (chunk_idx % 2) * 8;

        // std::cout << "Chunk " << chunk_idx << " chunk_q_begin " << chunk_q_begin << " chunk_q_end " << chunk_q_end << " chunk_k_begin " << chunk_k_begin << " chunk_k_end " << chunk_k_end << " k_begin " << k_begin << " k_end " << k_end << " k_length " << k_length << std::endl;
        for (int col = 0; col < total_cols; col++){
            // receive y
            size_t y_offset = (chunk_idx * lc * total_cols + col * lc) * DQ; // y does not have history
            seq->npu_dma_memcpy_nd(
                2, 0,
                S2MM, IT[col],
                (npu_bd_id)(bd_offset + 0), it_channel_0,
                {0, 0, 0, (uint32_t)y_offset},
                {1, 1, (uint32_t)1, (uint32_t)(lc * DQ)},
                {0, 0, (uint32_t)0, 1},
                -1, 0, true
            );
            // send q
            size_t q_offset = (chunk_idx * lc * total_cols + col * lc) * DQ; // q does not have history
            seq->npu_dma_memcpy_nd(
                2, 1,
                MM2S, IT[col],
                (npu_bd_id)(bd_offset + 1), it_channel_0,
                {0, 0, 0, (uint32_t)q_offset},
                {1, 1, (uint32_t)1, (uint32_t)(lc * DQ)},
                {0, 0, (uint32_t)0, 1},
                -1, 0, false
            );
        }
        // send k
        seq->npu_dma_memcpy_nd(
            2, 2,
            MM2S, IT[0],
            (npu_bd_id)(bd_offset + 2), it_channel_1,
            {0, 0, 0, K_OFFSET + k_begin * DK},
            {1, 1, 1, (uint32_t)k_length * DK},
            {0, 0, 0, 1},
            -1, 0, false
        );

        // send v
        seq->npu_dma_memcpy_nd(
            2, 2,
            MM2S, IT[4],
            (npu_bd_id)(bd_offset + 3), it_channel_1,
            {0, 0, 0, (uint32_t)V_OFFSET + k_begin * DV},
            {1, 1, 1, (uint32_t)k_length * DV},
            {0, 0, 0, 1},
            -1, 0, false
        );

        if (chunk_idx > 0){
            for (int col = 0; col < total_cols; col++){
                seq->npu_dma_wait(
                    IT[col],
                    S2MM,
                    it_channel_0
                );
            }
        }
    }
    for (int col = 0; col < total_cols; col++){
        seq->npu_dma_wait(
            IT[col],
            S2MM,
            it_channel_0
        );
    }
    seq->cmds2seq();
}

inline int _ceil_to_multiple(int value, int multiple) {
    assert(multiple > 0 && "multiple must be positive");
    return ((value + multiple - 1) / multiple) * multiple;
}

//NOTE: vector is batch x D
//NOTE: weight is M x D
//NOTE: output is batch x M
//NOTE: Operation is batch x (M x D x 1) -> batch x M
void gen_mvm_batch_engine_seq(
    npu_sequence* seq, const uint32_t M_total, const uint32_t D,
    const uint32_t m, const uint32_t d,  // size process for each CT block
    const uint32_t batch_size, const uint32_t weight_offset
){

    constexpr uint32_t Y_ARG_IDX = 0;
    constexpr uint32_t W_ARG_IDX = 1;
    constexpr uint32_t X_ARG_IDX = 2;

    // still assume W(M x K) are in pure row-major format
    const int M_total_per_batch = M_total / batch_size;

    assert(M_total % (4 * m) == 0);
    assert(M_total_per_batch % (4 * m) == 0);

    constexpr int total_cols = 8;
    constexpr int total_rows = 4;
    constexpr npu_tiles IT[total_cols] = {IT0, IT1, IT2, IT3, IT4, IT5, IT6, IT7};
    int bd_wait_counter[total_cols] = {0, 0, 0, 0, 0, 0, 0, 0};

    seq->clear_cmds();
    seq->npu_preemption(0);
    uint32_t D_div_d = D / d;
    assert(D % d == 0);

    uint32_t col_per_batch = M_total_per_batch / (4 * m);
    uint32_t compute_col_needed = batch_size * col_per_batch;

    constexpr uint32_t M_div_m_rtp_address = 1600;
    constexpr uint32_t D_div_d_rtp_address = 49856;
    constexpr uint32_t CT_lock_address_base = 0x000001F000;
    constexpr uint32_t CT_rtp_sync_lock_id = 6;   // 0x1F000 + 16*6 == 0x1F060
    for(int col = 0; col < total_cols; col++){
        uint32_t total_col_div_CT_cols = compute_col_needed / total_cols;
        uint32_t total_col_mode_CT_cols = compute_col_needed % total_cols;
        uint32_t cur_col_M_div_m = total_col_div_CT_cols;

        if( col < total_col_mode_CT_cols){
            cur_col_M_div_m += 1;
        }
        for(int row = 2; row < 2 + total_rows; row++){
            npu_tiles tile = get_tile(row, col);
            seq->rtp_write(tile, M_div_m_rtp_address, cur_col_M_div_m);
            seq->rtp_write(tile, D_div_d_rtp_address, D_div_d);
            seq->rtp_write(tile, CT_lock_address_base + 16 * CT_rtp_sync_lock_id, 1); //set rtp finished lock
        }
    }

    uint32_t round = _ceil_to_multiple(compute_col_needed,total_cols );
    for(int i = 0; i < round / total_cols; i++  ){

        for(int col = 0; col < total_cols; col++){
            uint32_t bd_offset = (i % 2) * 8;
            uint32_t CT_col_idx = i * total_cols + col; //NOTE: column offset

            uint32_t cur_batch  =  CT_col_idx/col_per_batch;
            uint32_t cur_batch_col_index = CT_col_idx % col_per_batch;
            if(CT_col_idx < compute_col_needed ){

                uint32_t w_off = cur_batch * D * (M_total_per_batch) + cur_batch_col_index * D * (4 * m) + weight_offset;// This assume W is in pure row-major order
                uint32_t x_off = cur_batch * D;
                uint32_t y_off = cur_batch * M_total_per_batch + cur_batch_col_index * (4 * m);

                // Y
                seq->npu_dma_memcpy_nd(
                    sizeof(uint16_t), Y_ARG_IDX,
                    S2MM, IT[col],
                    (npu_bd_id)(bd_offset + 0), it_channel_0,
                    {0,0,0, y_off},
                    {1,1,1, 4 * m},
                    {0,0,0,1},
                    -1, 0, true
                );
                // W
                seq->npu_dma_memcpy_nd(
                    sizeof(uint16_t), W_ARG_IDX,
                    MM2S, IT[col],
                    (npu_bd_id)(bd_offset + 1), it_channel_0,
                    {0,0,0, w_off},
                    { 1, D_div_d, (4 * m), d},
                    { 0, d, D, 1},
                    -1, 0, false
                );
                seq->npu_dma_memcpy_nd(
                    sizeof(uint16_t), X_ARG_IDX,
                    MM2S, IT[col],
                    (npu_bd_id)(bd_offset + 2), it_channel_1,
                    {0,0,0, x_off},
                    {1,1,1,  D},
                    {0,0,0,1},
                    -1, 0, false
                );
                bd_wait_counter[col]++;
            }
        }
        for (int col = 0; col < total_cols; col++){
            if(bd_wait_counter[col] == 2){
                seq->npu_dma_wait(IT[col], S2MM, it_channel_0);
                bd_wait_counter[col]--;
            }
        }
    }
    int max_loop_iterations = 0;
    // find the global max loop iterations
    for(int col = 0; col < total_cols; col++){
        max_loop_iterations = std::max(max_loop_iterations, bd_wait_counter[col]);
    }
    for(int i = 0; i < max_loop_iterations; i++){
        for(int col = 0; col < total_cols; col++){
            if (bd_wait_counter[col] > 0){
                seq->npu_dma_wait(IT[col], S2MM, it_channel_0);
                bd_wait_counter[col]--;
            }
        }
    }

    seq->cmds2seq();
}
