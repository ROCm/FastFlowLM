// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#pragma once
#include "npu_utils/npu_utils.hpp"
#define ENABLE_AXI4

namespace {
    inline int _ceil_to_multiple(int value, int multiple) {
        assert(multiple > 0 && "multiple must be positive");
        return ((value + multiple - 1) / multiple) * multiple;
    }
}

//NOTE: vector is batch x D
//NOTE: weight is M x D
//NOTE: output is batch x M
//NOTE: Operation is batch x (M x D x 1) -> batch x M
template<bool DEBUG=false>
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
    if(DEBUG){
        std::cout << M_total << "M_total " << batch_size << "batch_size " << M_total_per_batch << "M_total_per_batch " << std::endl;
    }
    assert(M_total % (4 * m) == 0);
    assert(M_total_per_batch % (4 * m) == 0);


    constexpr int total_cols = 8;
    constexpr int total_rows = 4;
    constexpr npu_tiles IT[total_cols] = {IT0, IT1, IT2, IT3, IT4, IT5, IT6, IT7};
    int bd_wait_counter[total_cols] = {0, 0, 0, 0, 0, 0, 0, 0};

    seq->clear_cmds();
    int D_div_d = D / d;
    assert(D % d == 0);

    uint32_t col_per_batch = M_total_per_batch / (4 * m);
    uint32_t compute_col_needed = batch_size * col_per_batch;

    for(int col = 0; col < total_cols; col++){
        uint32_t total_col_div_CT_cols = compute_col_needed / total_cols;
        uint32_t total_col_mode_CT_cols = compute_col_needed % total_cols;
        uint32_t cur_col_M_div_m = total_col_div_CT_cols;

        if( col < total_col_mode_CT_cols){
            cur_col_M_div_m += 1;
        }
        if(DEBUG){
            std::cout << "col: " << col << ", total_col_div_CT_cols: " << total_col_div_CT_cols << ", total_col_mode_CT_cols: " << total_col_mode_CT_cols << std::endl;
        }
        for(int row = 2; row < 2 + total_rows; row++){
            npu_tiles tile = get_tile(row, col);
            seq->rtp_write(tile, 1600, cur_col_M_div_m);
            seq->rtp_write(tile, 49856, D_div_d);
            seq->rtp_write(tile, 0x000001F060, 1); //set rtp finished lock
        }
    }

    uint32_t round = _ceil_to_multiple(compute_col_needed,total_cols );
    std::cout << "rounds: " << round << std::endl;
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
                
                if(DEBUG){
                    std::cout << "i: " << i << ", CT_col_idx: " << CT_col_idx << std::endl;
                    std::cout << "w_off: " << w_off << ", x_off: " << x_off << ", y_off: " << y_off << std::endl;
                    std::cout << "cur_batch: " << cur_batch << ", cur_batch_col_index: " << cur_batch_col_index << std::endl;
                    std::cout << std::endl;
                }

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
        std::cout << "dma queue: " << std::flush;
        for (int j = 0; j < total_cols; j++){
            std::cout << bd_wait_counter[j] << " ";
        }
        std::cout << std::endl;
        for (int col = 0; col < total_cols; col++){
            if(bd_wait_counter[col] == 2){
                seq->npu_dma_wait(IT[col], S2MM, it_channel_0);
                std::cout << "wait dma " << col << std::endl;
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
                std::cout << "wait dma " << col << std::endl;
                bd_wait_counter[col]--;
            }
        }
    }

    seq->cmds2seq();    
}


void gen_mv_engine_seq(npu_sequence* seq, const uint32_t M, const uint32_t K, const uint32_t N, const uint32_t m, const uint32_t k, const uint32_t n, const uint32_t weight_offset){
    constexpr int total_cols = 4;
    constexpr int total_rows = 4;
    constexpr npu_tiles IT[4] = {IT0, IT1, IT2, IT3};
    seq->clear_cmds();

    npu_sequence& seq_ptr = *seq;

    const int row_per_round = 4 * m * total_cols;
    const int down_rounds = (M + row_per_round - 1) / row_per_round;
    std::cout << "down_rounds: " << down_rounds << std::endl;
    const int total_column_rounds = M / 4 / m;

    int bd_wait_counter[total_cols] = {0, 0, 0, 0};

    assert(M % (4 * m) == 0);

    int K_div_k = K / k;

    for(int col = 0; col < total_cols; col++){
        uint32_t compute_column_needed = M/(4*m);
        uint32_t total_col_div_CT_cols = compute_column_needed/total_cols;
        uint32_t total_col_mode_CT_cols = compute_column_needed %total_cols;
        uint32_t cur_col_M_div_m = total_col_div_CT_cols;

        if( col < total_col_mode_CT_cols){
            cur_col_M_div_m += 1;
        }
        
        for(int row = 2; row < 2+total_rows; row++){
            npu_tiles tile = get_tile(row, col);
            seq->rtp_write(tile, 18432, cur_col_M_div_m);
            seq->rtp_write(tile, 32896, K_div_k);
            seq->rtp_write(tile, 0x000001F060, 1); //set rtp finished lock
        }
    }

    for (int i = 0; i < down_rounds; i++){
        for (int col = 0; col < total_cols; col++){
            uint32_t bd_offset = (i % 2) * 8; // NOTE: 8 is because of the IT ping-pong, bd_id = 16
            uint32_t round_offset = i * total_cols + col; //NOTE: column offset

            if (round_offset < total_column_rounds){
                // receive y
                uint32_t y_offset = round_offset * 4 * m;
                seq_ptr.npu_dma_memcpy_nd(
                    2, 0,
                    S2MM, IT[col],
                    (npu_bd_id)(bd_offset + 0), it_channel_0,
                    {0, 0, 0, (uint32_t)y_offset},
                    {1, 1, 1, (uint32_t)4 * m},
                    {0, 0, 0, 1},
                    -1, 0, true
                );

                // send w (matrix w)
                uint32_t w_offset = round_offset * 4 * m * K + weight_offset;
                seq_ptr.npu_dma_memcpy_nd(
                    2, 1,
                    MM2S, IT[col],
                    (npu_bd_id)(bd_offset + 1), it_channel_0,
                    {0, 0, 0, (uint32_t)w_offset},
                    {1, (uint32_t)K_div_k, (uint32_t)4 * m, (uint32_t)k},
                    {0, k, K, 1},
                    -1, 0, false
                );

                // send x (vector x)
                uint32_t x_offset = 0;
                seq_ptr.npu_dma_memcpy_nd(
                    2, 2,
                    MM2S, IT[col],
                    (npu_bd_id)(bd_offset + 2), it_channel_1,
                    {0, 0, 0, (uint32_t)x_offset},
                    {1, 1, 1, (uint32_t)K},
                    {0, 0, 0, 1},
                    -1, 0, false
                );
                bd_wait_counter[col]++;
            }
        }
        std::cout << "dma queue: " << std::flush;
        for (int j = 0; j < total_cols; j++){
            std::cout << bd_wait_counter[j] << " ";
        }
        std::cout << std::endl;
        for (int col = 0; col < total_cols; col++){
            if(bd_wait_counter[col] == 2){
                seq_ptr.npu_dma_wait(IT[col], S2MM, it_channel_0);
                std::cout << "wait dma " << col << std::endl;
                bd_wait_counter[col]--;
            }
        }
    } // end for i
    int max_loop_iterations = 0;
    // find the global max loop iterations
    for(int col = 0; col < total_cols; col++){
        max_loop_iterations = std::max(max_loop_iterations, bd_wait_counter[col]);
    }
    std::cout << "max loop iterations: " << max_loop_iterations << std::endl;
    for(int i = 0; i < max_loop_iterations; i++){
        for(int col = 0; col < total_cols; col++){
            if (bd_wait_counter[col] > 0){
                seq_ptr.npu_dma_wait(IT[col], S2MM, it_channel_0);
                std::cout << "wait dma " << col << std::endl;
                bd_wait_counter[col]--;
            }
        }
    }
    seq->cmds2seq();
}