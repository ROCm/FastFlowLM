// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include "gemma_text_npu_sequence_detail.hpp"

void gemma_text_npu_sequence::Impl::_send_x(npu_sequence* seq){
    // send x
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        xy_arg_idx,
        MM2S,
        xyr_tile,
        bd_0,
        it_channel_0,
        {0, 0, 0, 0},
        {1, 1, 1, (uint32_t)D},
        {0, 0, 0, 1},
        0, 0, false, aggressive_cache
    );
}

void gemma_text_npu_sequence::Impl::_send_rope_rms_weights(npu_sequence* seq){
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        w_rope_rms_arg_idx,
        MM2S,
        xyr_tile,
        bd_2,
        it_channel_0,
        {0, 0, 0, 0},
        {1, 1, 1, (uint32_t)DH * 3},
        {0, 0, 0, 1},
        1, 0, false, aggressive_cache
    );
}

void gemma_text_npu_sequence::Impl::_receive_kv_cache(npu_sequence* seq, const int L, bool is_sliding_window){
    int L_local = L - 1;
    if (is_sliding_window){
        L_local = L_local % sliding_window;
    }
    const uint32_t kv_cache_offset = L_local * DH;
    uint32_t k_offset = is_sliding_window ? sliding_K_offset : K_offset;
    uint32_t v_offset = is_sliding_window ? sliding_V_offset : V_offset;
    // receive new k, v
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        kv_cache_arg_idx,
        S2MM,
        kv_tiles,
        bd_0,
        it_channel_0,
        {0, 0, 0, kv_cache_offset + (uint32_t)k_offset},
        {1, 1, 1, (uint32_t)DH},
        {0, 0, 0, 1},
        -1, 0, true
    );
    _move_weights(seq, DQ + DK + DV, D, w_qkv_offset);
    seq->npu_dma_wait(
        kv_tiles,
        S2MM,
        it_channel_0
    );
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        kv_cache_arg_idx,
        S2MM,
        kv_tiles,
        bd_1,
        it_channel_1,
        {0, 0, 0, kv_cache_offset + (uint32_t)v_offset},
        {1, 1, 1, (uint32_t)DH},
        {0, 0, 0, 1},
        -1, 0, true
    );
    seq->npu_dma_wait(
        kv_tiles,
        S2MM,
        it_channel_1
    );
}

void gemma_text_npu_sequence::Impl::_move_weights(npu_sequence* seq, const size_t Dout, const size_t Din, const size_t w_offset){
    const uint32_t columns = 4;
    const uint32_t elems = m * k;
    const uint32_t a_block_size = elems * 5 / 8 / 2;
    const uint32_t blocks_per_row = Din / k;
    const uint32_t cores = columns * 4;

    assert(Din % k == 0);
    assert(blocks_per_row > 0);
    assert(Dout / m / cores > 0);
    assert(Dout % (m * cores) == 0);
    assert(blocks_per_row <= 63);
    for (size_t round = 0; round < Dout / m / cores; round++){
        uint32_t bd_offset = (round % 2) * 8;
        for (uint32_t col = 0; col < columns; col++){
            seq->npu_dma_memcpy_nd(
                sizeof(bf16),
                w_proj_arg_idx,
                MM2S,
                mvm_tiles[col],
                npu_bd_id(bd_1 + bd_offset),
                it_channel_0,
                {0, 0, 0, (uint32_t)((round * cores + col * 4) * a_block_size * blocks_per_row + (uint32_t)w_offset)},
                {blocks_per_row, 2, 8, a_block_size / 8},
                {a_block_size, a_block_size * blocks_per_row, a_block_size / 8, 1},
                -1, 0, true, aggressive_cache
            );
            seq->npu_dma_memcpy_nd(
                sizeof(bf16),
                w_proj_arg_idx,
                MM2S,
                mvm_tiles[col],
                npu_bd_id(bd_2 + bd_offset),
                it_channel_1,
                {0, 0, 0, (uint32_t)((round * cores + col * 4 + 2) * a_block_size * blocks_per_row + (uint32_t)w_offset)},
                {blocks_per_row, 2, 8, a_block_size / 8},
                {a_block_size, a_block_size * blocks_per_row, a_block_size / 8, 1},
                -1, 0, true, aggressive_cache
            );
        }
        if (round > 0){
            for (uint32_t col = 0; col < columns; col++){
                seq->npu_dma_wait(
                    mvm_tiles[col],
                    MM2S,
                    it_channel_0
                );
                seq->npu_dma_wait(
                    mvm_tiles[col],
                    MM2S,
                    it_channel_1
                );
            }
        }
    }
    for (uint32_t col = 0; col < columns; col++){
        seq->npu_dma_wait(
            mvm_tiles[col],
            MM2S,
            it_channel_0
        );
        seq->npu_dma_wait(
            mvm_tiles[col],
            MM2S,
            it_channel_1
        );
    }
}

void gemma_text_npu_sequence::Impl::_move_kv_cache(npu_sequence* seq, const size_t L, bool is_sliding_window){

    uint32_t k_offset = is_sliding_window ? sliding_K_offset : K_offset;
    uint32_t v_offset = is_sliding_window ? sliding_V_offset : V_offset;
    if (is_sliding_window){
        if (L % sliding_window == 0){ // corner case, just from zero to end
            const uint32_t data2move = sliding_window * DH;
            seq->npu_dma_memcpy_nd(
                sizeof(bf16),
                kv_cache_arg_idx,
                MM2S,
                kv_tiles,
                bd_0,
                it_channel_0,
                {0, 0, 0, (uint32_t)k_offset},
                {1, 1, 1, data2move},
                {0, 0, 0, 1},
                -1, 0, false, aggressive_cache
            );
            seq->npu_dma_memcpy_nd(
                sizeof(bf16),
                kv_cache_arg_idx,
                MM2S,
                kv_tiles,
                bd_1,
                it_channel_1,
                {0, 0, 0, (uint32_t)v_offset},
                {1, 1, 1, data2move},
                {0, 0, 0, 1},
                -1, 0, false, aggressive_cache
            );
            return;
        }
        else if (L > sliding_window){ // still single phase, from zero to L
            uint32_t L_begin = L % sliding_window;
            uint32_t L_phase_1 = sliding_window - L_begin;
            uint32_t L_phase_2 = L_begin;
            uint32_t offset = L_begin * DH;
            uint32_t data2move_1 = L_phase_1 * DH;
            uint32_t data2move_2 = L_phase_2 * DH;
            // phase 1
            seq->npu_dma_memcpy_nd(
                sizeof(bf16),
                kv_cache_arg_idx,
                MM2S,
                kv_tiles,
                bd_0,
                it_channel_0,
                {0, 0, 0, (uint32_t)k_offset + offset},
                {1, 1, 1, data2move_1},
                {0, 0, 0, 1},
                -1, 0, true, aggressive_cache
            );
            seq->npu_dma_memcpy_nd(
                sizeof(bf16),
                kv_cache_arg_idx,
                MM2S,
                kv_tiles,
                bd_1,
                it_channel_1,
                {0, 0, 0, (uint32_t)v_offset + offset},
                {1, 1, 1, data2move_1},
                {0, 0, 0, 1},
                -1, 0, true, aggressive_cache
            );
            seq->npu_dma_wait(
                kv_tiles,
                MM2S,
                it_channel_0
            );
            seq->npu_dma_wait(
                kv_tiles,
                MM2S,
                it_channel_1
            );
            // phase 2
            seq->npu_dma_memcpy_nd(
                sizeof(bf16),
                kv_cache_arg_idx,
                MM2S,
                kv_tiles,
                bd_0,
                it_channel_0,
                {0, 0, 0, (uint32_t)k_offset},
                {1, 1, 1, data2move_2},
                {0, 0, 0, 1},
                -1, 0, false, aggressive_cache
            );
            seq->npu_dma_memcpy_nd(
                sizeof(bf16),
                kv_cache_arg_idx,
                MM2S,
                kv_tiles,
                bd_1,
                it_channel_1,
                {0, 0, 0, (uint32_t)v_offset},
                {1, 1, 1, data2move_2},
                {0, 0, 0, 1},
                -1, 0, false, aggressive_cache
            );
            return;
        }
    }
    // fallback path, single phase, from zero to L
    const int L_padded = (L + L_CHUNK - 1) / L_CHUNK * L_CHUNK;
    const uint32_t data2move = L_padded * DH;
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        kv_cache_arg_idx,
        MM2S,
        kv_tiles,
        bd_0,
        it_channel_0,
        {0, 0, 0, (uint32_t)k_offset},
        {1, 1, 1, data2move},
        {0, 0, 0, 1},
        -1, 0, false, aggressive_cache
    );
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        kv_cache_arg_idx,
        MM2S,
        kv_tiles,
        bd_1,
        it_channel_1,
        {0, 0, 0, (uint32_t)v_offset},
        {1, 1, 1, data2move},
        {0, 0, 0, 1},
        -1, 0, false, aggressive_cache
    );
}

gemma_text_npu_sequence::Impl::Impl(LM_Config config, uint32_t MAX_L){
    D = config.get("hidden_size");
    DH = config.get("head_dim");
    DQ = config.get("head_dim") * config.get("num_attention_heads");
    DK = config.get("head_dim") * config.get("num_key_value_heads");
    DV = config.get("head_dim") * config.get("num_key_value_heads");
    this->sliding_window = config.get("sliding_window");
    this->sliding_window_pattern = config.get("sliding_window_pattern");
    this->MAX_L = MAX_L;
    uint32_t PADDED_D = (D + 255) / 256 * 256;

    HIDDEN_SIZE = config.get("intermediate_size");
    kv_cache_size = MAX_L * (DK + DV);
    w_qkv_offset = 0;
    w_o_offset = (DQ + DK + DV) * D * 5 / 8 / sizeof(bf16);
    w_up_offset = w_o_offset + PADDED_D * DQ * 5 / 8 / sizeof(bf16);
    w_down_offset = w_up_offset + (2 * HIDDEN_SIZE) * D * 5 / 8 / sizeof(bf16);
    uint32_t kv_cache_size_sliding = sliding_window * (DK + DV);
    K_offset = 0;
    V_offset = kv_cache_size / 2;
    sliding_K_offset = 0;
    sliding_V_offset = kv_cache_size_sliding / 2;
}

void gemma_text_npu_sequence::Impl::gen_layer_seq(npu_sequence* seq, const uint32_t L, bool is_sliding_window){
    assert(L <= MAX_L + 1);
    assert(L > 0);
    uint32_t PADDED_D = (D + 255) / 256 * 256;
    seq->clear_cmds();
    seq->npu_preemption(0);
    if (is_sliding_window){
        uint32_t L_local;
        if (L <= sliding_window) {
            L_local = L;
        }
        else{
            L_local = sliding_window;
        }
        seq->rtp_write(attn_qk_tiles, l_qk_address, L_local);
        seq->rtp_write(attn_kv_tiles, l_kv_address, L_local);
    }
    else{

        seq->rtp_write(attn_qk_tiles, l_qk_address, L);
        seq->rtp_write(attn_kv_tiles, l_kv_address, L);
    }
    _send_x(seq);
    // _send_rms_weights(seq, 0);
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        w_rms_arg_idx,
        MM2S,
        xyr_tile,
        bd_1,
        it_channel_1,
        {0, 0, 0, 0},
        {1, 1, 1, (uint32_t)D},
        {0, 0, 0, 1},
        0, 0, false
    );
    _send_rope_rms_weights(seq);

    // receive y
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        xy_arg_idx,
        S2MM,
        xyr_tile,
        bd_10,
        it_channel_1,
        {0, 0, 0, 0},
        {1, 1, 1, (uint32_t)D},
        {0, 0, 0, 1},
        -1, 0, true
    );
    _receive_kv_cache(seq, L, is_sliding_window); // also move qkv weights
    _move_kv_cache(seq, L, is_sliding_window);
    _move_weights(seq, PADDED_D, DQ, w_o_offset);

    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        w_rms_arg_idx,
        MM2S,
        xyr_tile,
        bd_1,
        it_channel_1,
        {0, 0, 0, D},
        {1, 1, 1, (uint32_t)D * 2},
        {0, 0, 0, 1},
        0, 0, false
    );

    _move_weights(seq, 2 * HIDDEN_SIZE, D, w_up_offset);
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        w_rms_arg_idx,
        MM2S,
        xyr_tile,
        bd_1,
        it_channel_1,
        {0, 0, 0, D * 3},
        {1, 1, 1, (uint32_t)D},
        {0, 0, 0, 1},
        0, 0, false
    );
    _move_weights(seq, PADDED_D, HIDDEN_SIZE, w_down_offset);

    // wait for receiving y
    seq->npu_dma_wait(
        xyr_tile,
        S2MM,
        it_channel_1
    );

    seq->cmds2seq();
}

void gemma_text_npu_sequence::Impl::gen_mha_engine_seq(npu_sequence* seq, const uint32_t L_begin, const uint32_t L_end, bool is_sliding_window, int buffer_length){
    constexpr int total_cols = 8;
    constexpr int total_rows = 4;
    constexpr npu_tiles IT[8] = {IT0, IT1, IT2, IT3, IT4, IT5, IT6, IT7};
    constexpr int lc = 16;
    const int G = DQ / DK;
    assert(L_begin % (16 * 8) == 0);
    assert(L_end % (16 * 8) == 0);
    int using_window_size = is_sliding_window ? sliding_window : (L_end);
    seq->clear_cmds();
    seq->npu_preemption(0);
    for (int row = 2; row < 6; row++){
        for (int col = 0; col < total_cols; col++){
            npu_tiles tile = get_tile(row, col);
            seq->rtp_write(tile, l_begin_mha_address, L_begin);
            seq->rtp_write(tile, l_end_mha_address, L_end);
            seq->rtp_write(tile, window_size_address, using_window_size);
        }
    }
    int new_L = L_end - L_begin;

    for (int round = 0; round < new_L / 128; round++){
        int bd_offset = (round % 2) * 8;
        int L_current = L_begin + round * 128;
        int kv_begin = ((L_current - using_window_size) > 0) ? (L_current - using_window_size) : 0;
        int kv_length = (L_current - kv_begin + 128) * DK;
        for (int col = 0; col < total_cols; col++){ // col is bind to l
            // receive y
            size_t y_offset = (size_t)(round * lc * total_cols + col * lc) * DQ; // y does not have history
            seq->npu_dma_memcpy_nd(
                2, 0,
                S2MM, IT[col],
                (npu_bd_id)(bd_offset + 0), it_channel_0,
                {0, 0, 0, (uint32_t)y_offset},
                {1, 1, (uint32_t)lc, (uint32_t)(DQ)},
                {0, 0, (uint32_t)DQ, 1},
                -1, 0, true
            );
            // send q
            size_t q_offset = (size_t)(round * lc * total_cols + col * lc) * DQ; // q does not have history
            seq->npu_dma_memcpy_nd(
                2, 1,
                MM2S, IT[col],
                (npu_bd_id)(bd_offset + 1), it_channel_0,
                {0, 0, 0, (uint32_t)q_offset},
                {1, 1, (uint32_t)lc, (uint32_t)(DQ)},
                {0, 0, (uint32_t)DQ, 1},
                -1, 0, false
            );
        } // end of for col
        // send k
        seq->npu_dma_memcpy_nd(
            2, 2,
            MM2S, IT[0],
            (npu_bd_id)(bd_offset + 2), it_channel_1,
            {0, 0, 0, (uint32_t)(this->K_offset + (size_t)kv_begin * DK)},
            {1, 1, 1, (uint32_t)kv_length},
            {0, 0, 0, 1},
            -1, 0, false
        );

        // send v
        seq->npu_dma_memcpy_nd(
            2, 2,
            MM2S, IT[4],
            (npu_bd_id)(bd_offset + 3), it_channel_1,
            {0, 0, 0, (uint32_t)(this->V_offset + (size_t)kv_begin * DV)},
            {1, 1, 1, (uint32_t)kv_length},
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

void gemma_text_npu_sequence::Impl::set_max_length(const uint32_t MAX_L){
    this->MAX_L = MAX_L;
    this->kv_cache_size = MAX_L * (DK + DV);
    this->K_offset = 0;
    this->V_offset = kv_cache_size / 2;
}

// ===============================================
// Externals for gemma_npu_sequence
// ===============================================
/// @brief
/// @param rtp_seq : npu_sequence ptr for layer rtp write (set L)
/// @param layer_seq : npu_sequence ptr for layer sequence
/// @param dequant_seq : npu_sequence ptr for dequant sequence
/// @param mm_engine_seq : npu_sequence ptr for mm engine sequence
/// @param mha_engine_seq : npu_sequence ptr for mha engine sequence
/// @param config : llama config
/// @param MAX_L : max length
gemma_text_npu_sequence::gemma_text_npu_sequence(LM_Config config, uint32_t MAX_L){
    this->_impl = new Impl(config, MAX_L);
}

void gemma_text_npu_sequence::gen_layer_seq(npu_sequence* seq, const uint32_t L, bool is_sliding_window){
    this->_impl->gen_layer_seq(seq, L, is_sliding_window);
}

void gemma_text_npu_sequence::gen_mha_engine_seq(npu_sequence* seq, const uint32_t L_begin, const uint32_t L_end, bool is_sliding_window, int buffer_length){
    this->_impl->gen_mha_engine_seq(seq, L_begin, L_end, is_sliding_window, buffer_length);
}

void gemma_text_npu_sequence::set_max_length(const uint32_t MAX_L){
    this->_impl->set_max_length(MAX_L);
}

size_t gemma_text_npu_sequence::get_k_offset() const{
    return this->_impl->K_offset;
}

size_t gemma_text_npu_sequence::get_v_offset() const{
    return this->_impl->V_offset;
}

gemma_text_npu_sequence::~gemma_text_npu_sequence() = default;
