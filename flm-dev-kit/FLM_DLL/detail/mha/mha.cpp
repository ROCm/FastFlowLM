// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include "mha_detail.hpp"

// constructors
MHA::Impl::Impl(mha_type_t& mha_ty, int num_kv_heads) : mha_ty(mha_ty), num_kv_heads(num_kv_heads){
    this->l_begin_mha_address = _get_l_begin_addr(mha_ty);
    this->l_end_mha_address = _get_l_end_addr(mha_ty);
    this->window_size_address = _get_window_size_addr(mha_ty);
    this->ct_chunk_size = _get_ct_chunk_size(mha_ty);
    this->head_dim = _get_head_dim(mha_ty);
    this->chunk_size = this->ct_chunk_size * 8; // each CT process 4 tokens at a time
    this->num_q_heads = _get_num_q_heads(mha_ty);
}

MHA::Impl::~Impl() = default;


void MHA::Impl::generate_mha_sequence(npu_sequence* seq, const uint32_t L_begin, const uint32_t L_end, const uint32_t MAX_L, const bool is_sliding_window, const int WINDOW_SIZE){
    switch(this->mha_ty){
        case mha_d64_q4:
            _gen_mha_seq_d64_q4(seq, L_begin, L_end, MAX_L, is_sliding_window, WINDOW_SIZE);
            break;
        case mha_d128_q2:
            _gen_mha_seq_d128_q2(seq, L_begin, L_end, MAX_L, is_sliding_window, WINDOW_SIZE);
            break;
        case mha_d128_q3:
            _gen_mha_seq_d128_q3(seq, L_begin, L_end, MAX_L, is_sliding_window, WINDOW_SIZE);
            break;
        case mha_d128_q4:
            _gen_mha_seq_d128_q4(seq, L_begin, L_end, MAX_L, is_sliding_window, WINDOW_SIZE);
            break;
        case mha_d256_q2:
            _gen_mha_seq_d256_q2(seq, L_begin, L_end, MAX_L, is_sliding_window, WINDOW_SIZE);
            break;
        case mha_d256_q4:
            _gen_mha_seq_d256_q4(seq, L_begin, L_end, MAX_L, is_sliding_window, WINDOW_SIZE);
            break;
        case mha_d128_q4_1cu:
            _gen_mha_seq_d128_q4_1cu(seq, L_begin, L_end, MAX_L, is_sliding_window, WINDOW_SIZE);
            break;
        default:
            throw std::runtime_error("MHA type not supported!");
    }
}

bool MHA::Impl::_parameter_check(const uint32_t L_begin, const uint32_t L_end, const bool is_sliding_window, const int WINDOW_SIZE){
    try {
        if (L_begin >= L_end){
            throw std::runtime_error("L_begin should be less than L_end!");
        }
        if (is_sliding_window && WINDOW_SIZE <= 0){
            throw std::runtime_error("WINDOW_SIZE should be positive in sliding window mode!");
        }
        if (L_begin % this->chunk_size != 0){
            throw std::runtime_error("L_begin should be multiple of chunk size!");
        }
        if (L_end % this->chunk_size != 0){
            throw std::runtime_error("L_end should be multiple of chunk size!");
        }
    } catch (const std::runtime_error& e) {
        std::cerr << "Parameter check error: " << e.what() << std::endl
                    << "L_begin: " << L_begin << ", L_end: " << L_end
                    << ", is_sliding_window: " << is_sliding_window
                    << ", WINDOW_SIZE: " << WINDOW_SIZE << std::endl;
        return false;
    }
    return true;
}

void MHA::Impl::_gen_mha_seq_d64_q4(npu_sequence* seq, const uint32_t L_begin, const uint32_t L_end, const uint32_t MAX_L, const bool is_sliding_window, const int WINDOW_SIZE){
    static constexpr int num_cu = 4;
    // some renames for brief
    const int lc = ct_chunk_size; // length per CT, renamed for clarity
    const int Heads = this->num_q_heads;
    const int DH = this->head_dim;
    const int Num_of_d_in_KV_cache = this->num_kv_heads / 2;
    //
    const uint32_t KV_CACHE_SIZE = MAX_L * DH * this->num_kv_heads * 2;
    if (!_parameter_check(L_begin, L_end, is_sliding_window, WINDOW_SIZE)){
        throw std::runtime_error("MHA parameter check failed!");
        exit(1);
    }
    seq->clear_cmds();

    int using_window_size = is_sliding_window ? WINDOW_SIZE : (L_end);

    for (int row = 2; row < 6; row++){
        for (int col = 0; col < 8; col++){
            npu_tiles tile = get_tile(row, col);
            seq->rtp_write(tile, l_begin_mha_address, L_begin);
            seq->rtp_write(tile, l_end_mha_address, L_end);
            seq->rtp_write(tile, window_size_address, using_window_size);
        }
    }
    // each CU has 8 CTs and works on 1 head.
    int all_data_size = L_end - L_begin;
    const int data_per_round = chunk_size;
    const int down_rounds = (all_data_size + data_per_round - 1) / data_per_round;

    for (int head = 0; head < Heads / num_cu; head++){
        for (int round = 0; round < down_rounds; round++){
            int Lq_current = L_begin + round * data_per_round;
            int kv_begin = ((Lq_current - using_window_size) > 0) ? (Lq_current - using_window_size) : 0;
            int kv_length = Lq_current - kv_begin + data_per_round;
            LOG_VERBOSE(1, "Round " << round << " Lq_current " << Lq_current << " kv_begin " << kv_begin << " kv_length " << kv_length);

            int bd_offset = (round % 2) * 8;
            LOG_VERBOSE(1, " bd_offset: " << bd_offset);

            for (int cu = 0; cu < num_cu; cu++){
                int head_offset = head * num_cu + cu;
                LOG_VERBOSE(1, "head_offset: " << head_offset);
                for (int col = 0; col < 2; col++){
                    // receive y
                    size_t y_offset = head_offset * DH + (round * data_per_round + col * lc * 4) * DH * Heads;
                    seq->npu_dma_memcpy_nd(
                        2, 0,
                        S2MM, IT[cu][col],
                        (npu_bd_id)(bd_offset + 0), it_channel_0,
                        {0, 0, 0, (uint32_t)y_offset},
                        {1, 1, (uint32_t)4 * lc, (uint32_t)(DH)},
                        {0, 0, (uint32_t)DH * Heads, 1},
                        -1, 0, true
                    );
                }
                // send q
                size_t q_offset = head_offset * DH + round * data_per_round * DH * Heads;
                seq->npu_dma_memcpy_nd(
                    2, 1,
                    MM2S, IT[cu][0],
                    (npu_bd_id)(bd_offset + 1), it_channel_0,
                    {0, 0, 0, (uint32_t)q_offset},
                    {1, 1, (uint32_t)lc * 4, (uint32_t)(DH)},
                    {0, 0, (uint32_t)DH * Heads, 1},
                    -1, 0, false
                );
                seq->npu_dma_memcpy_nd(
                    2, 1,
                    MM2S, IT[cu][0],
                    (npu_bd_id)(bd_offset + 2), it_channel_1,
                    {0, 0, 0, (uint32_t)(q_offset + (size_t)lc * 4 * DH * Heads)},
                    {1, 1, (uint32_t)lc * 4, (uint32_t)(DH)},
                    {0, 0, (uint32_t)DH * Heads, 1},
                    -1, 0, false
                );
            } // cu

            int kv_head_offset = head;
            int kv_chunk_offset = kv_head_offset / Num_of_d_in_KV_cache;
            int kv_head_offset_in_chunk = kv_head_offset % Num_of_d_in_KV_cache;
            LOG_VERBOSE(1, " kv_head_offset: " << kv_head_offset << " kv_chunk_offset: " << kv_chunk_offset << " kv_head_offset_in_chunk: " << kv_head_offset_in_chunk);

            size_t k_offset = kv_chunk_offset * MAX_L * DH * Num_of_d_in_KV_cache + kv_head_offset_in_chunk * DH + kv_begin * DH * Num_of_d_in_KV_cache;
            seq->npu_dma_memcpy_nd(
                2, 2,
                MM2S, IT[1][1],
                (npu_bd_id)(bd_offset + 3), it_channel_0,
                {0, 0, 0, (uint32_t)k_offset},
                {1, (uint32_t)kv_length / 256, (uint32_t)256, (uint32_t)(DH)},
                {0, (uint32_t)256 * DH * Num_of_d_in_KV_cache, (uint32_t)DH * Num_of_d_in_KV_cache, 1},
                -1, 0, false
            );
            size_t v_offset = kv_chunk_offset * MAX_L * DH * Num_of_d_in_KV_cache + kv_head_offset_in_chunk * DH + kv_begin * DH * Num_of_d_in_KV_cache + KV_CACHE_SIZE / 2;
            seq->npu_dma_memcpy_nd(
                2, 2,
                MM2S, IT[1][1],
                (npu_bd_id)(bd_offset + 4), it_channel_1,
                {0, 0, 0, (uint32_t)v_offset},
                {1, (uint32_t)kv_length / 256, (uint32_t)256, (uint32_t)(DH)},
                {0, (uint32_t)256 * DH * Num_of_d_in_KV_cache, (uint32_t)DH * Num_of_d_in_KV_cache, 1},
                -1, 0, false
            );
            if (round > 0){
                for (int cu = 0; cu < num_cu; cu++){
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
        for (int cu = 0; cu < num_cu; cu++){
            for (int col = 0; col < 2; col++){
                seq->npu_dma_wait(
                    IT[cu][col],
                    S2MM,
                    it_channel_0
                );
            } // col
        } // cu
    } // head
    seq->cmds2seq();
}

void MHA::Impl::_gen_mha_seq_d128_q2(npu_sequence* seq, const uint32_t L_begin, const uint32_t L_end, const uint32_t MAX_L, const bool is_sliding_window, const int WINDOW_SIZE){
    static constexpr int num_cu = 4;
    // some renames for brief
    const int lc = ct_chunk_size; // length per CT, renamed for clarity
    const int Heads = this->num_q_heads;
    const int DH = this->head_dim;
    const int Num_of_d_in_KV_cache = this->num_kv_heads / 2;
    //
    const uint32_t KV_CACHE_SIZE = MAX_L * DH * this->num_kv_heads * 2;
    if (!_parameter_check(L_begin, L_end, is_sliding_window, WINDOW_SIZE)){
        throw std::runtime_error("MHA parameter check failed!");
        exit(1);
    }
    seq->clear_cmds();

    int using_window_size = is_sliding_window ? WINDOW_SIZE : (L_end);

    for (int row = 2; row < 6; row++){
        for (int col = 0; col < 8; col++){
            npu_tiles tile = get_tile(row, col);
            seq->rtp_write(tile, l_begin_mha_address, L_begin);
            seq->rtp_write(tile, l_end_mha_address, L_end);
            seq->rtp_write(tile, window_size_address, using_window_size);
        }
    }
    // each CU has 8 CTs and works on 1 head.
    int all_data_size = L_end - L_begin;
    const int data_per_round = lc * 8;
    const int down_rounds = (all_data_size + data_per_round - 1) / data_per_round;

    for (int head = 0; head < Heads / num_cu; head++){
        for (int round = 0; round < down_rounds; round++){
            int Lq_current = L_begin + round * data_per_round;
            int kv_begin = ((Lq_current - using_window_size) > 0) ? (Lq_current - using_window_size) : 0;
            int kv_length = Lq_current - kv_begin + data_per_round;
            LOG_VERBOSE(1, "Round " << round << " Lq_current " << Lq_current << " kv_begin " << kv_begin << " kv_length " << kv_length << std::endl);

            int bd_offset = (round % 2) * 8;
            LOG_VERBOSE(1, "bd_offset: " << bd_offset << std::endl);

            for (int cu = 0; cu < num_cu; cu++){
                int head_offset = head * num_cu + cu;
                LOG_VERBOSE(1, "head_offset: " << head_offset << std::endl);
                for (int col = 0; col < 2; col++){
                    // receive y
                    size_t y_offset = head_offset * DH + (round * data_per_round + col * lc * 4) * DH * Heads;
                    seq->npu_dma_memcpy_nd(
                        2, 0,
                        S2MM, IT[cu][col],
                        (npu_bd_id)(bd_offset + 0), it_channel_0,
                        {0, 0, 0, (uint32_t)y_offset},
                        {1, 1, (uint32_t)4 * lc, (uint32_t)(DH)},
                        {0, 0, (uint32_t)DH * Heads, 1},
                        -1, 0, true
                    );
                }
                // send q
                size_t q_offset = head_offset * DH + round * data_per_round * DH * Heads;
                seq->npu_dma_memcpy_nd(
                    2, 1,
                    MM2S, IT[cu][0],
                    (npu_bd_id)(bd_offset + 1), it_channel_0,
                    {0, 0, 0, (uint32_t)q_offset},
                    {1, 1, (uint32_t)lc * 4, (uint32_t)(DH)},
                    {0, 0, (uint32_t)DH * Heads, 1},
                    -1, 0, false
                );
                seq->npu_dma_memcpy_nd(
                    2, 1,
                    MM2S, IT[cu][0],
                    (npu_bd_id)(bd_offset + 2), it_channel_1,
                    {0, 0, 0, (uint32_t)(q_offset + (size_t)lc * 4 * DH * Heads)},
                    {1, 1, (uint32_t)lc * 4, (uint32_t)(DH)},
                    {0, 0, (uint32_t)DH * Heads, 1},
                    -1, 0, false
                );
            } // cu

            // 2 is beacuse 1: 2, 4 CUs has 2 shared CUs
            for (int i = 0; i < 2; i++){
                int kv_head_offset = head * 2 + i;  // 2 is beacuse 1: 2, 2 q shared one k and v
                int kv_chunk_offset = kv_head_offset / Num_of_d_in_KV_cache;
                int kv_head_offset_in_chunk = kv_head_offset % Num_of_d_in_KV_cache;
                LOG_VERBOSE(1, " kv_head_offset: " << kv_head_offset << " kv_chunk_offset: " << kv_chunk_offset << " kv_head_offset_in_chunk: " << kv_head_offset_in_chunk << std::endl);

                size_t k_offset = kv_chunk_offset * MAX_L * DH * Num_of_d_in_KV_cache + kv_head_offset_in_chunk * DH + kv_begin * DH * Num_of_d_in_KV_cache;
                seq->npu_dma_memcpy_nd(
                    2, 2,
                    MM2S, IT[i * 2 + 1][1],
                    (npu_bd_id)(bd_offset + 3), it_channel_0,
                    {0, 0, 0, (uint32_t)k_offset},
                    {1, (uint32_t)kv_length / 128, (uint32_t)128, (uint32_t)(DH)},
                    {0, (uint32_t)128 * DH * Num_of_d_in_KV_cache, (uint32_t)DH * Num_of_d_in_KV_cache, 1},
                    -1, 0, false
                );
                size_t v_offset = kv_chunk_offset * MAX_L * DH * Num_of_d_in_KV_cache + kv_head_offset_in_chunk * DH + kv_begin * DH * Num_of_d_in_KV_cache + KV_CACHE_SIZE / 2;
                seq->npu_dma_memcpy_nd(
                    2, 2,
                    MM2S, IT[i * 2 + 1][1],
                    (npu_bd_id)(bd_offset + 4), it_channel_1,
                    {0, 0, 0, (uint32_t)v_offset},
                    {1, (uint32_t)kv_length / 128, (uint32_t)128, (uint32_t)(DH)},
                    {0, (uint32_t)128 * DH * Num_of_d_in_KV_cache, (uint32_t)DH * Num_of_d_in_KV_cache, 1},
                    -1, 0, false
                );
            }
            if (round > 0){
                for (int cu = 0; cu < num_cu; cu++){
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
        for (int cu = 0; cu < num_cu; cu++){
            for (int col = 0; col < 2; col++){
                seq->npu_dma_wait(
                    IT[cu][col],
                    S2MM,
                    it_channel_0
                );
            } // col
        } // cu
    } // head
    seq->cmds2seq();
}

void MHA::Impl::_gen_mha_seq_d128_q3(npu_sequence* seq, const uint32_t L_begin, const uint32_t L_end, const uint32_t MAX_L, const bool is_sliding_window, const int WINDOW_SIZE){
    static constexpr int num_cu = 4;
    // some renames for brief
    const int lc = ct_chunk_size; // length per CT, renamed for clarity
    const int Heads = this->num_q_heads;
    const int DH = this->head_dim;
    const int Num_of_d_in_KV_cache = this->num_kv_heads / 2;
    //
    const uint32_t KV_CACHE_SIZE = MAX_L * DH * this->num_kv_heads * 2;
    if (!_parameter_check(L_begin, L_end, is_sliding_window, WINDOW_SIZE)){
        throw std::runtime_error("MHA parameter check failed!");
        exit(1);
    }
    seq->clear_cmds();

    int using_window_size = is_sliding_window ? WINDOW_SIZE : (L_end);
    for (int row = 2; row < 6; row++){
        for (int col = 0; col < 8; col++){
            npu_tiles tile = get_tile(row, col);
            seq->rtp_write(tile, l_begin_mha_address, L_begin);
            seq->rtp_write(tile, l_end_mha_address, L_end);
            seq->rtp_write(tile, window_size_address, using_window_size);
        }
    }
    // each CU has 8 CTs and works on 1 head.
    int all_data_size = L_end - L_begin;
    // each round, each CU works on 1 head and lc * 8 of l, total_cols / 2 is corresponding to the number of CUs
    const int data_per_round = lc * 8;
    // zero padding the last round if necessary
    const int down_rounds = (all_data_size + data_per_round - 1) / data_per_round;
    for (int head = 0; head < Heads / 4; head++){
        for (int round = 0; round < down_rounds; round++){
            int Lq_current = L_begin + round * data_per_round;
            int kv_begin = ((Lq_current - using_window_size) > 0) ? (Lq_current - using_window_size) : 0;
            int kv_length = Lq_current - kv_begin + data_per_round;
            LOG_VERBOSE(1, "Round " << round << " Lq_current " << Lq_current << " kv_begin " << kv_begin << " kv_length " << kv_length << std::endl);

            int bd_offset = (round % 2) * 8;
            LOG_VERBOSE(1, "bd_offset: " << bd_offset << std::endl);

            // cu 0, 1, 2
            for (int cu = 0; cu < 3; cu++){
                int head_offset = head * 3 + cu;
                LOG_VERBOSE(1, "head_offset: " << head_offset << std::endl);
                for (int col = 0; col < 2; col++){
                    // receive y
                    size_t y_offset = head_offset * DH + round * data_per_round * DH * Heads + col * lc * 4 * DH * Heads;
                    seq->npu_dma_memcpy_nd(
                        2, 0,
                        S2MM, IT[cu][col],
                        (npu_bd_id)(bd_offset + 0), it_channel_0,
                        {0, 0, 0, (uint32_t)y_offset},
                        {1, 1, (uint32_t)4 * lc, (uint32_t)(DH)},
                        {0, 0, (uint32_t)DH * Heads, 1},
                        -1, 0, true
                    );
                }
                // send q
                size_t q_offset = head_offset * DH + round * data_per_round * DH * Heads;
                seq->npu_dma_memcpy_nd(
                    2, 1,
                    MM2S, IT[cu][0],
                    (npu_bd_id)(bd_offset + 1), it_channel_0,
                    {0, 0, 0, (uint32_t)q_offset},
                    {1, 1, (uint32_t)lc * 4, (uint32_t)(DH)},
                    {0, 0, (uint32_t)DH * Heads, 1},
                    -1, 0, false
                );
                seq->npu_dma_memcpy_nd(
                    2, 1,
                    MM2S, IT[cu][0],
                    (npu_bd_id)(bd_offset + 2), it_channel_1,
                    {0, 0, 0, (uint32_t)(q_offset + (size_t)lc * 4 * DH * Heads)},
                    {1, 1, (uint32_t)lc * 4, (uint32_t)(DH)},
                    {0, 0, (uint32_t)DH * Heads, 1},
                    -1, 0, false
                );
            } // cu
            int kv_chunk_offset = head / Num_of_d_in_KV_cache;
            int kv_head_offset = head % Num_of_d_in_KV_cache;
            LOG_VERBOSE(1, "kv_chunk_offset: " << kv_chunk_offset << " kv_head_offset: " << kv_head_offset << std::endl);
            // send k
            size_t k_offset = kv_chunk_offset * MAX_L * Num_of_d_in_KV_cache * DH + kv_head_offset * DH + kv_begin * DH * Num_of_d_in_KV_cache;
            seq->npu_dma_memcpy_nd(
                2, 2,
                MM2S, IT[0][1],
                (npu_bd_id)(bd_offset + 3), it_channel_0,
                {0, 0, 0, (uint32_t)k_offset},
                {1, (uint32_t)kv_length / 128, (uint32_t)128, (uint32_t)(DH)},
                {0, (uint32_t)128 * DH * Num_of_d_in_KV_cache, (uint32_t)DH * Num_of_d_in_KV_cache, 1},
                -1, 0, false
            );
            // send v
            size_t v_offset = kv_chunk_offset * MAX_L * Num_of_d_in_KV_cache * DH + kv_head_offset * DH + kv_begin * DH * Num_of_d_in_KV_cache + KV_CACHE_SIZE / 2;
            seq->npu_dma_memcpy_nd(
                2, 2,
                MM2S, IT[0][1],
                (npu_bd_id)(bd_offset + 4), it_channel_1,
                {0, 0, 0, (uint32_t)v_offset},
                {1, (uint32_t)kv_length / 128, (uint32_t)128, (uint32_t)(DH)},
                {0, (uint32_t)128 * DH * Num_of_d_in_KV_cache, (uint32_t)DH * Num_of_d_in_KV_cache, 1},
                -1, 0, false
            );
            // cu 3
            int head_offset = head + 18;
            LOG_VERBOSE(1, "head_offset: " << head_offset << std::endl);
            for (int col = 0; col < 2; col++){
                // receive y
                size_t y_offset = head_offset * DH + round * data_per_round * DH * Heads + col * lc * 4 * DH * Heads;
                seq->npu_dma_memcpy_nd(
                    2, 0,
                    S2MM, IT[3][col],
                    (npu_bd_id)(bd_offset + 0), it_channel_0,
                    {0, 0, 0, (uint32_t)y_offset},
                    {1, 1, (uint32_t)4 * lc, (uint32_t)(DH)},
                    {0, 0, (uint32_t)DH * Heads, 1},
                    -1, 0, true
                );
            }
            // send q
            size_t q_offset = head_offset * DH + round * data_per_round * DH * Heads;
            seq->npu_dma_memcpy_nd(
                2, 1,
                MM2S, IT[3][0],
                (npu_bd_id)(bd_offset + 1), it_channel_0,
                {0, 0, 0, (uint32_t)q_offset},
                {1, 1, (uint32_t)lc * 4, (uint32_t)(DH)},
                {0, 0, (uint32_t)DH * Heads, 1},
                -1, 0, false
            );
            seq->npu_dma_memcpy_nd(
                2, 1,
                MM2S, IT[3][0],
                (npu_bd_id)(bd_offset + 2), it_channel_1,
                {0, 0, 0, (uint32_t)(q_offset + (size_t)lc * 4 * DH * Heads)},
                {1, 1, (uint32_t)lc * 4, (uint32_t)(DH)},
                {0, 0, (uint32_t)DH * Heads, 1},
                -1, 0, false
            );
            kv_head_offset = (head / 3 + 6) % Num_of_d_in_KV_cache;
            kv_chunk_offset = 1;
            LOG_VERBOSE(1, "kv_chunk_offset: " << 1 << " kv_head_offset: " << kv_head_offset << std::endl);
            // send k
            k_offset = kv_chunk_offset * MAX_L * Num_of_d_in_KV_cache * DH + kv_head_offset * DH + kv_begin * DH * Num_of_d_in_KV_cache;
            seq->npu_dma_memcpy_nd(
                2, 2,
                MM2S, IT[3][1],
                (npu_bd_id)(bd_offset + 3), it_channel_0,
                {0, 0, 0, (uint32_t)k_offset},
                {1, (uint32_t)kv_length / 128, (uint32_t)128, (uint32_t)(DH)},
                {0, (uint32_t)128 * DH * Num_of_d_in_KV_cache, (uint32_t)DH * Num_of_d_in_KV_cache, 1},
                -1, 0, false
            );
            // send v
            v_offset = kv_chunk_offset * MAX_L * Num_of_d_in_KV_cache * DH + kv_head_offset * DH + kv_begin * DH * Num_of_d_in_KV_cache + KV_CACHE_SIZE / 2;
            seq->npu_dma_memcpy_nd(
                2, 2,
                MM2S, IT[3][1],
                (npu_bd_id)(bd_offset + 4), it_channel_1,
                {0, 0, 0, (uint32_t)v_offset},
                {1, (uint32_t)kv_length / 128, (uint32_t)128, (uint32_t)(DH)},
                {0, (uint32_t)128 * DH * Num_of_d_in_KV_cache, (uint32_t)DH * Num_of_d_in_KV_cache, 1},
                -1, 0, false
            );
            if (round > 0){
                for (int cu = 0; cu < 4; cu++){
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
        for (int cu = 0; cu < 4; cu++){
            for (int col = 0; col < 2; col++){
                seq->npu_dma_wait(
                    IT[cu][col],
                    S2MM,
                    it_channel_0
                );
            } // col
        } // cu
    } // head
    seq->cmds2seq();
}

void MHA::Impl::_gen_mha_seq_d128_q4(npu_sequence* seq, const uint32_t L_begin, const uint32_t L_end, const uint32_t MAX_L, const bool is_sliding_window, const int WINDOW_SIZE){
    static constexpr int num_cu = 4;
    // some renames for brief
    const int lc = ct_chunk_size; // length per CT, renamed for clarity
    const int Heads = this->num_q_heads;
    const int DH = this->head_dim;
    const int Num_of_d_in_KV_cache = this->num_kv_heads / 2;
    //
    const uint32_t KV_CACHE_SIZE = MAX_L * DH * this->num_kv_heads * 2;
    if (!_parameter_check(L_begin, L_end, is_sliding_window, WINDOW_SIZE)){
        throw std::runtime_error("MHA parameter check failed!");
        exit(1);
    }
    seq->clear_cmds();

    int using_window_size = is_sliding_window ? WINDOW_SIZE : (L_end);
    for (int row = 2; row < 6; row++){
        for (int col = 0; col < 8; col++){
            npu_tiles tile = get_tile(row, col);
            seq->rtp_write(tile, l_begin_mha_address, L_begin);
            seq->rtp_write(tile, l_end_mha_address, L_end);
            seq->rtp_write(tile, window_size_address, using_window_size);
        }
    }
    // each CU has 8 CTs and works on 1 head.
    int all_data_size = L_end - L_begin;
    const int data_per_round = lc * 8;
    const int down_rounds = (all_data_size + data_per_round - 1) / data_per_round;

    for (int head = 0; head < Heads / num_cu; head++){
        for (int round = 0; round < down_rounds; round++){
            int Lq_current = L_begin + round * data_per_round;
            int kv_begin = ((Lq_current - using_window_size) > 0) ? (Lq_current - using_window_size) : 0;
            int kv_length = Lq_current - kv_begin + data_per_round;
            LOG_VERBOSE(1, "Round " << round << " Lq_current " << Lq_current << " kv_begin " << kv_begin << " kv_length " << kv_length << std::endl);

            int bd_offset = (round % 2) * 8;
            LOG_VERBOSE(1, "bd_offset: " << bd_offset << std::endl);

            for (int cu = 0; cu < num_cu; cu++){
                int head_offset = head * num_cu + cu;
                LOG_VERBOSE(1, "head_offset: " << head_offset << std::endl);
                for (int col = 0; col < 2; col++){
                    // receive y
                    size_t y_offset = head_offset * DH + (round * data_per_round + col * lc * 4) * DH * Heads;
                    seq->npu_dma_memcpy_nd(
                        2, 0,
                        S2MM, IT[cu][col],
                        (npu_bd_id)(bd_offset + 0), it_channel_0,
                        {0, 0, 0, (uint32_t)y_offset},
                        {1, 1, (uint32_t)4 * lc, (uint32_t)(DH)},
                        {0, 0, (uint32_t)DH * Heads, 1},
                        -1, 0, true
                    );
                }
                // send q
                size_t q_offset = head_offset * DH + round * data_per_round * DH * Heads;
                seq->npu_dma_memcpy_nd(
                    2, 1,
                    MM2S, IT[cu][0],
                    (npu_bd_id)(bd_offset + 1), it_channel_0,
                    {0, 0, 0, (uint32_t)q_offset},
                    {1, 1, (uint32_t)lc * 4, (uint32_t)(DH)},
                    {0, 0, (uint32_t)DH * Heads, 1},
                    -1, 0, false
                );
                seq->npu_dma_memcpy_nd(
                    2, 1,
                    MM2S, IT[cu][0],
                    (npu_bd_id)(bd_offset + 2), it_channel_1,
                    {0, 0, 0, (uint32_t)(q_offset + (size_t)lc * 4 * DH * Heads)},
                    {1, 1, (uint32_t)lc * 4, (uint32_t)(DH)},
                    {0, 0, (uint32_t)DH * Heads, 1},
                    -1, 0, false
                );
            } // cu

            int kv_head_offset = head;
            int kv_chunk_offset = kv_head_offset / Num_of_d_in_KV_cache;
            int kv_head_offset_in_chunk = kv_head_offset % Num_of_d_in_KV_cache;
            LOG_VERBOSE(1, " kv_head_offset: " << kv_head_offset << " kv_chunk_offset: " << kv_chunk_offset << " kv_head_offset_in_chunk: " << kv_head_offset_in_chunk << std::endl);

            size_t k_offset = kv_chunk_offset * MAX_L * DH * Num_of_d_in_KV_cache + kv_head_offset_in_chunk * DH + kv_begin * DH * Num_of_d_in_KV_cache;
            seq->npu_dma_memcpy_nd(
                2, 2,
                MM2S, IT[1][1],
                (npu_bd_id)(bd_offset + 3), it_channel_0,
                {0, 0, 0, (uint32_t)k_offset},
                {1, (uint32_t)kv_length / 128, (uint32_t)128, (uint32_t)(DH)},
                {0, (uint32_t)128 * DH * Num_of_d_in_KV_cache, (uint32_t)DH * Num_of_d_in_KV_cache, 1},
                -1, 0, false
            );
            size_t v_offset = kv_chunk_offset * MAX_L * DH * Num_of_d_in_KV_cache + kv_head_offset_in_chunk * DH + kv_begin * DH * Num_of_d_in_KV_cache + KV_CACHE_SIZE / 2;
            seq->npu_dma_memcpy_nd(
                2, 2,
                MM2S, IT[1][1],
                (npu_bd_id)(bd_offset + 4), it_channel_1,
                {0, 0, 0, (uint32_t)v_offset},
                {1, (uint32_t)kv_length / 128, (uint32_t)128, (uint32_t)(DH)},
                {0, (uint32_t)128 * DH * Num_of_d_in_KV_cache, (uint32_t)DH * Num_of_d_in_KV_cache, 1},
                -1, 0, false
            );
            if (round > 0){
                LOG_VERBOSE(1, "Waiting for previous round DMA transfers to complete..." << std::endl);
                for (int cu = 0; cu < num_cu; cu++){
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
        for (int cu = 0; cu < num_cu; cu++){
            for (int col = 0; col < 2; col++){
                seq->npu_dma_wait(
                    IT[cu][col],
                    S2MM,
                    it_channel_0
                );
            } // col
        } // cu
    } // head
    seq->cmds2seq();
}

void MHA::Impl::_gen_mha_seq_d256_q2(npu_sequence* seq, const uint32_t L_begin, const uint32_t L_end, const uint32_t MAX_L, const bool is_sliding_window, const int WINDOW_SIZE){
    static constexpr int num_cu = 4;
    // some renames for brief
    const int lc = ct_chunk_size; // length per CT, renamed for clarity
    const int Heads = this->num_q_heads;
    const int DH = this->head_dim;
    const int Num_of_d_in_KV_cache = this->num_kv_heads / 2;
    //
    const uint32_t KV_CACHE_SIZE = MAX_L * DH * this->num_kv_heads * 2;
    if (!_parameter_check(L_begin, L_end, is_sliding_window, WINDOW_SIZE)){
        throw std::runtime_error("MHA parameter check failed!");
        exit(1);
    }
    seq->clear_cmds();

    int using_window_size = is_sliding_window ? WINDOW_SIZE : (L_end);
    for (int row = 2; row < 6; row++){
        for (int col = 0; col < 8; col++){
            npu_tiles tile = get_tile(row, col);
            seq->rtp_write(tile, l_begin_mha_address, L_begin);
            seq->rtp_write(tile, l_end_mha_address, L_end);
            seq->rtp_write(tile, window_size_address, using_window_size);
        }
    }
    // each CU has 8 CTs and works on 1 head.
    int all_data_size = L_end - L_begin;
    // each round, each CU works on 1 head and lc * 8 of l, total_cols / 2 is corresponding to the number of CUs
    const int data_per_round = lc * 8;
    // zero padding the last round if necessary
    const int down_rounds = (all_data_size + data_per_round - 1) / data_per_round;

    for (int head = 0; head < Heads / num_cu; head++){
        for (int round = 0; round < down_rounds; round++){
            int Lq_current = L_begin + round * data_per_round;
            int kv_begin = ((Lq_current - using_window_size) > 0) ? (Lq_current - using_window_size) : 0;
            int kv_length = Lq_current - kv_begin + data_per_round;
            LOG_VERBOSE(1, "Round " << round << " Lq_current " << Lq_current << " kv_begin " << kv_begin << " kv_length " << kv_length << std::endl);

            int bd_offset = (round % 2) * 8;
            LOG_VERBOSE(1, "bd_offset: " << bd_offset << std::endl);
            for (int cu = 0; cu < num_cu; cu++){
                int head_offset = head * num_cu + cu;
                LOG_VERBOSE(1, "head_offset: " << head_offset << std::endl);
                for (int col = 0; col < 2; col++){
                    // receive y
                    size_t y_offset = head_offset * DH + (round * data_per_round + col * lc * 4) * DH * Heads;
                    seq->npu_dma_memcpy_nd(
                        2, 0,
                        S2MM, IT[cu][col],
                        (npu_bd_id)(bd_offset + 0), it_channel_0,
                        {0, 0, 0, (uint32_t)y_offset},
                        {1, 1, (uint32_t)4 * lc, (uint32_t)(DH)},
                        {0, 0, (uint32_t)DH * Heads, 1},
                        -1, 0, true
                    );
                }
                // send q
                size_t q_offset = head_offset * DH + round * data_per_round * DH * Heads;
                seq->npu_dma_memcpy_nd(
                    2, 1,
                    MM2S, IT[cu][0],
                    (npu_bd_id)(bd_offset + 1), it_channel_0,
                    {0, 0, 0, (uint32_t)q_offset},
                    {1, 1, (uint32_t)lc * 4, (uint32_t)(DH)},
                    {0, 0, (uint32_t)DH * Heads, 1},
                    -1, 0, false, aggressive_cache
                );
                seq->npu_dma_memcpy_nd(
                    2, 1,
                    MM2S, IT[cu][0],
                    (npu_bd_id)(bd_offset + 2), it_channel_1,
                    {0, 0, 0, (uint32_t)(q_offset + (size_t)lc * 4 * DH * Heads)},
                    {1, 1, (uint32_t)lc * 4, (uint32_t)(DH)},
                    {0, 0, (uint32_t)DH * Heads, 1},
                    -1, 0, false, aggressive_cache
                );
            } // cu

            // 2 is beacuse 1: 2, 4 CUs has 2 shared CUs
            for (int i = 0; i < 2; i++){
                int kv_head_offset = head * 2 + i;  // 2 is beacuse 1: 2, 2 q shared one k and v
                int kv_chunk_offset = kv_head_offset / Num_of_d_in_KV_cache;
                int kv_head_offset_in_chunk = kv_head_offset % Num_of_d_in_KV_cache;
                LOG_VERBOSE(1, " kv_head_offset: " << kv_head_offset << " kv_chunk_offset: " << kv_chunk_offset << " kv_head_offset_in_chunk: " << kv_head_offset_in_chunk << std::endl);

                size_t k_offset = kv_chunk_offset * MAX_L * DH * Num_of_d_in_KV_cache + kv_head_offset_in_chunk * DH + kv_begin * DH * Num_of_d_in_KV_cache;
                seq->npu_dma_memcpy_nd(
                    2, 2,
                    MM2S, IT[i * 2 + 1][1],
                    (npu_bd_id)(bd_offset + 3), it_channel_0,
                    {0, 0, 0, (uint32_t)k_offset},
                    {1, (uint32_t)kv_length / 128, (uint32_t)128, (uint32_t)(DH)},
                    {0, (uint32_t)128 * DH * Num_of_d_in_KV_cache, (uint32_t)DH * Num_of_d_in_KV_cache, 1},
                    -1, 0, false, aggressive_cache
                );
                size_t v_offset = kv_chunk_offset * MAX_L * DH * Num_of_d_in_KV_cache + kv_head_offset_in_chunk * DH + kv_begin * DH * Num_of_d_in_KV_cache + KV_CACHE_SIZE / 2;
                seq->npu_dma_memcpy_nd(
                    2, 2,
                    MM2S, IT[i * 2 + 1][1],
                    (npu_bd_id)(bd_offset + 4), it_channel_1,
                    {0, 0, 0, (uint32_t)v_offset},
                    {1, (uint32_t)kv_length / 128, (uint32_t)128, (uint32_t)(DH)},
                    {0, (uint32_t)128 * DH * Num_of_d_in_KV_cache, (uint32_t)DH * Num_of_d_in_KV_cache, 1},
                    -1, 0, false, aggressive_cache
                );
            }
            if (round > 0){
                for (int cu = 0; cu < num_cu; cu++){
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
        for (int cu = 0; cu < num_cu; cu++){
            for (int col = 0; col < 2; col++){
                seq->npu_dma_wait(
                    IT[cu][col],
                    S2MM,
                    it_channel_0
                );
            } // col
        } // cu
    } // head
    seq->cmds2seq();
}

void MHA::Impl::_gen_mha_seq_d256_q4(npu_sequence* seq, const uint32_t L_begin, const uint32_t L_end, const uint32_t MAX_L, const bool is_sliding_window, const int WINDOW_SIZE){
static constexpr int num_cu = 4;
    // some renames for brief
    const int lc = ct_chunk_size; // length per CT, renamed for clarity
    const int Heads = this->num_q_heads;
    const int DH = this->head_dim;
    const int Num_of_d_in_KV_cache = this->num_kv_heads; // 1 4 is special
    //
    const uint32_t KV_CACHE_SIZE = MAX_L * DH * this->num_kv_heads * 2;
    if (!_parameter_check(L_begin, L_end, is_sliding_window, WINDOW_SIZE)){
        throw std::runtime_error("MHA parameter check failed!");
        exit(1);
    }
    seq->clear_cmds();

    int using_window_size = is_sliding_window ? WINDOW_SIZE : (L_end);
    for (int row = 2; row < 6; row++){
        for (int col = 0; col < 8; col++){
            npu_tiles tile = get_tile(row, col);
            seq->rtp_write(tile, l_begin_mha_address, L_begin);
            seq->rtp_write(tile, l_end_mha_address, L_end);
            seq->rtp_write(tile, window_size_address, using_window_size);
        }
    }
    // each CU has 8 CTs and works on 1 head.
    int all_data_size = L_end - L_begin;
    // each round, each CU works on 1 head and lc * 8 of l, total_cols / 2 is corresponding to the number of CUs
    const int data_per_round = lc * 8;
    // zero padding the last round if necessary
    const int down_rounds = (all_data_size + data_per_round - 1) / data_per_round;
    for (int head = 0; head < Heads / num_cu; head++){
        for (int round = 0; round < down_rounds; round++){
            int Lq_current = L_begin + round * data_per_round;
            int kv_begin = ((Lq_current - using_window_size) > 0) ? (Lq_current - using_window_size) : 0;
            int kv_length = Lq_current - kv_begin + data_per_round;
            LOG_VERBOSE(1, "Round " << round << " Lq_current " << Lq_current << " kv_begin " << kv_begin << " kv_length " << kv_length << std::endl);

            int bd_offset = (round % 2) * 8;
            LOG_VERBOSE(1, "bd_offset: " << bd_offset << std::endl);
            for (int cu = 0; cu < num_cu; cu++){
                int head_offset = head * num_cu + cu;
                LOG_VERBOSE(1, "head_offset: " << head_offset << std::endl);
                for (int col = 0; col < 2; col++){
                    // receive y
                    size_t y_offset = head_offset * DH + (round * data_per_round + col * lc * 4) * DH * Heads;
                    seq->npu_dma_memcpy_nd(
                        2, 0,
                        S2MM, IT[cu][col],
                        (npu_bd_id)(bd_offset + 0), it_channel_0,
                        {0, 0, 0, (uint32_t)y_offset},
                        {1, 1, (uint32_t)4 * lc, (uint32_t)(DH)},
                        {0, 0, (uint32_t)DH * Heads, 1},
                        -1, 0, true
                    );
                }
                // send q
                size_t q_offset = head_offset * DH + round * data_per_round * DH * Heads;
                seq->npu_dma_memcpy_nd(
                    2, 1,
                    MM2S, IT[cu][0],
                    (npu_bd_id)(bd_offset + 1), it_channel_0,
                    {0, 0, 0, (uint32_t)q_offset},
                    {1, 1, (uint32_t)lc * 4, (uint32_t)(DH)},
                    {0, 0, (uint32_t)DH * Heads, 1},
                    -1, 0, false
                );
                seq->npu_dma_memcpy_nd(
                    2, 1,
                    MM2S, IT[cu][0],
                    (npu_bd_id)(bd_offset + 2), it_channel_1,
                    {0, 0, 0, (uint32_t)(q_offset + (size_t)lc * 4 * DH * Heads)},
                    {1, 1, (uint32_t)lc * 4, (uint32_t)(DH)},
                    {0, 0, (uint32_t)DH * Heads, 1},
                    -1, 0, false
                );
            } // cu

            int kv_head_offset = head;
            int kv_chunk_offset = kv_head_offset / Num_of_d_in_KV_cache;
            int kv_head_offset_in_chunk = kv_head_offset % Num_of_d_in_KV_cache;
            LOG_VERBOSE(1, " kv_head_offset: " << kv_head_offset << " kv_chunk_offset: " << kv_chunk_offset << " kv_head_offset_in_chunk: " << kv_head_offset_in_chunk << std::endl);

            size_t k_offset = kv_chunk_offset * MAX_L * DH * Num_of_d_in_KV_cache + kv_head_offset_in_chunk * DH + kv_begin * DH * Num_of_d_in_KV_cache;
            seq->npu_dma_memcpy_nd(
                2, 2,
                MM2S, IT[1][1],
                (npu_bd_id)(bd_offset + 3), it_channel_0,
                {0, 0, 0, (uint32_t)k_offset},
                {1, (uint32_t)kv_length / 128, (uint32_t)128, (uint32_t)(DH)},
                {0, (uint32_t)128 * DH * Num_of_d_in_KV_cache, (uint32_t)DH * Num_of_d_in_KV_cache, 1},
                -1, 0, false
            );
            size_t v_offset = kv_chunk_offset * MAX_L * DH * Num_of_d_in_KV_cache + kv_head_offset_in_chunk * DH + kv_begin * DH * Num_of_d_in_KV_cache + KV_CACHE_SIZE / 2;
            seq->npu_dma_memcpy_nd(
                2, 2,
                MM2S, IT[1][1],
                (npu_bd_id)(bd_offset + 4), it_channel_1,
                {0, 0, 0, (uint32_t)v_offset},
                {1, (uint32_t)kv_length / 128, (uint32_t)128, (uint32_t)(DH)},
                {0, (uint32_t)128 * DH * Num_of_d_in_KV_cache, (uint32_t)DH * Num_of_d_in_KV_cache, 1},
                -1, 0, false
            );
            if (round > 0){
                for (int cu = 0; cu < num_cu; cu++){
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
        for (int cu = 0; cu < num_cu; cu++){
            for (int col = 0; col < 2; col++){
                seq->npu_dma_wait(
                    IT[cu][col],
                    S2MM,
                    it_channel_0
                );
            } // col
        } // cu
    } // head
    seq->cmds2seq();
}

// wrappers
MHA::MHA(mha_type_t& config, int num_kv_heads) : _impl(new Impl(config, num_kv_heads)){}
MHA::~MHA(){
    delete _impl;
}

void MHA::generate_mha_sequence(npu_sequence* seq, const uint32_t L_begin, const uint32_t L_end, const uint32_t MAX_L, const bool is_sliding_window, const int WINDOW_SIZE){
    _impl->generate_mha_sequence(seq, L_begin, L_end, MAX_L, is_sliding_window, WINDOW_SIZE);
}

int MHA::get_chunk_size(){
    return _impl->get_chunk_size();
}

/// Single-compute-unit variant of _gen_mha_seq_d128_q4, for the fused prefill
/// overlay where columns 0-5 are a dequant+mm array and only columns 6-7 are
/// attention.  The array topology inside the CU is unchanged -- it is literally
/// the 4-CU build's CU 3 -- so the only differences are that IT6/IT7 replace the
/// IT[cu] table, k/v broadcast out of IT7 instead of IT3, and the head loop runs
/// every q head itself instead of four at a time.
void MHA::Impl::_gen_mha_seq_d128_q4_1cu(npu_sequence* seq, const uint32_t L_begin, const uint32_t L_end, const uint32_t MAX_L, const bool is_sliding_window, const int WINDOW_SIZE){
    // The CU's two physical columns.  They are the outermost ones so that the
    // six mm columns straddle the DDR-channel boundary at 3/4 evenly; attn.cc
    // derives its q-token offset from `ct_col & 0x1`, and 0/7 is still an
    // (even, odd) pair even though they are not neighbours.
    static constexpr npu_tiles CU[2] = {IT0, IT7};
    // k and v are broadcast from the CU's odd column, matching MT[7] in the
    // overlay's k_in/v_in wiring.
    static constexpr npu_tiles KV = IT7;

    const int lc = ct_chunk_size;
    const int Heads = this->num_q_heads;
    const int DH = this->head_dim;
    const int Num_of_d_in_KV_cache = this->num_kv_heads / 2;
    const int gqa = Heads / this->num_kv_heads;

    const uint32_t KV_CACHE_SIZE = MAX_L * DH * this->num_kv_heads * 2;
    if (!_parameter_check(L_begin, L_end, is_sliding_window, WINDOW_SIZE)){
        throw std::runtime_error("MHA parameter check failed!");
        exit(1);
    }
    seq->clear_cmds();

    int using_window_size = is_sliding_window ? WINDOW_SIZE : (L_end);
    static constexpr int ATTN_COL[2] = {0, 7};
    for (int row = 2; row < 6; row++){
        for (int c = 0; c < 2; c++){
            npu_tiles tile = get_tile(row, ATTN_COL[c]);
            seq->rtp_write(tile, l_begin_mha_address, L_begin);
            seq->rtp_write(tile, l_end_mha_address, L_end);
            seq->rtp_write(tile, window_size_address, using_window_size);
        }
    }

    int all_data_size = L_end - L_begin;
    const int data_per_round = lc * 8;
    const int down_rounds = (all_data_size + data_per_round - 1) / data_per_round;

    // One head per pass now, so unlike the 4-CU build the kv head has to be
    // derived from the q head rather than being the loop variable.
    for (int head = 0; head < Heads; head++){
        for (int round = 0; round < down_rounds; round++){
            int Lq_current = L_begin + round * data_per_round;
            int kv_begin = ((Lq_current - using_window_size) > 0) ? (Lq_current - using_window_size) : 0;
            int kv_length = Lq_current - kv_begin + data_per_round;

            int bd_offset = (round % 2) * 8;
            const int head_offset = head;

            for (int col = 0; col < 2; col++){
                // receive y
                size_t y_offset = head_offset * DH + (round * data_per_round + col * lc * 4) * DH * Heads;
                seq->npu_dma_memcpy_nd(
                    2, 0,
                    S2MM, CU[col],
                    (npu_bd_id)(bd_offset + 0), it_channel_0,
                    {0, 0, 0, (uint32_t)y_offset},
                    {1, 1, (uint32_t)4 * lc, (uint32_t)(DH)},
                    {0, 0, (uint32_t)DH * Heads, 1},
                    -1, 0, true
                );
            }
            // send q
            size_t q_offset = head_offset * DH + round * data_per_round * DH * Heads;
            seq->npu_dma_memcpy_nd(
                2, 1,
                MM2S, CU[0],
                (npu_bd_id)(bd_offset + 1), it_channel_0,
                {0, 0, 0, (uint32_t)q_offset},
                {1, 1, (uint32_t)lc * 4, (uint32_t)(DH)},
                {0, 0, (uint32_t)DH * Heads, 1},
                -1, 0, false
            );
            seq->npu_dma_memcpy_nd(
                2, 1,
                MM2S, CU[0],
                (npu_bd_id)(bd_offset + 2), it_channel_1,
                {0, 0, 0, (uint32_t)(q_offset + (size_t)lc * 4 * DH * Heads)},
                {1, 1, (uint32_t)lc * 4, (uint32_t)(DH)},
                {0, 0, (uint32_t)DH * Heads, 1},
                -1, 0, false
            );

            int kv_head_offset = head / gqa;
            int kv_chunk_offset = kv_head_offset / Num_of_d_in_KV_cache;
            int kv_head_offset_in_chunk = kv_head_offset % Num_of_d_in_KV_cache;

            size_t k_offset = kv_chunk_offset * MAX_L * DH * Num_of_d_in_KV_cache + kv_head_offset_in_chunk * DH + kv_begin * DH * Num_of_d_in_KV_cache;
            seq->npu_dma_memcpy_nd(
                2, 2,
                MM2S, KV,
                (npu_bd_id)(bd_offset + 3), it_channel_0,
                {0, 0, 0, (uint32_t)k_offset},
                {1, (uint32_t)kv_length / 128, (uint32_t)128, (uint32_t)(DH)},
                {0, (uint32_t)128 * DH * Num_of_d_in_KV_cache, (uint32_t)DH * Num_of_d_in_KV_cache, 1},
                -1, 0, false
            );
            size_t v_offset = k_offset + KV_CACHE_SIZE / 2;
            seq->npu_dma_memcpy_nd(
                2, 2,
                MM2S, KV,
                (npu_bd_id)(bd_offset + 4), it_channel_1,
                {0, 0, 0, (uint32_t)v_offset},
                {1, (uint32_t)kv_length / 128, (uint32_t)128, (uint32_t)(DH)},
                {0, (uint32_t)128 * DH * Num_of_d_in_KV_cache, (uint32_t)DH * Num_of_d_in_KV_cache, 1},
                -1, 0, false
            );
            if (round > 0){
                for (int col = 0; col < 2; col++){
                    seq->npu_dma_wait(CU[col], S2MM, it_channel_0);
                }
            }
        }// round
        for (int col = 0; col < 2; col++){
            seq->npu_dma_wait(CU[col], S2MM, it_channel_0);
        }
    } // head
    seq->cmds2seq();
}
