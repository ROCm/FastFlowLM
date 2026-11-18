// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include "lfm2_npu_sequence.hpp"

void lfm2_npu_sequence::_send_x(npu_sequence* seq){
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

void lfm2_npu_sequence::_send_rms_weights(npu_sequence* seq){
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        w_rms_arg_idx,
        MM2S,
        xyr_tile,
        bd_1,
        it_channel_1,
        {0, 0, 0, 0},
        {1, 1, 1, (uint32_t)D * 2},
        {0, 0, 0, 1},
        0, 0, false, aggressive_cache
    );
}

void lfm2_npu_sequence::_send_rope_weights(npu_sequence* seq){
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        w_rope_arg_idx,
        MM2S,
        xyr_tile,
        bd_2,
        it_channel_0,
        {0, 0, 0, 0},
        {1, 1, 1, (uint32_t)3 * D},
        {0, 0, 0, 1},
        1, 0, false, aggressive_cache
    );
}

void lfm2_npu_sequence::_receive_kv_cache(npu_sequence* seq, const int L){
    const uint32_t kv_cache_offset = (L - 1) * 4 * DH;
    // receive new k, v
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        kv_cache_arg_idx,
        S2MM,
        IT0,
        bd_0,
        it_channel_0,
        {0, 0, 0, kv_cache_offset},
        {1, 1, 1, 4 * (uint32_t)DH},
        {0, 0, 0, 1},
        -1, 0, true
    );
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        kv_cache_arg_idx,
        S2MM,
        IT7,
        bd_0,
        it_channel_0,
        {0, 0, 0, kv_cache_offset + (uint32_t)V03_offset},
        {1, 1, 1, 4 * (uint32_t)DH},
        {0, 0, 0, 1},
        -1, 0, true
    );
    // the band is packed Q|K|V at the front of slot0, so attn_q carries its base
    _move_weights(seq, DQ + DK + DV, D, desc->weight_desc[0]->attn_q);
    seq->npu_dma_wait(
        IT0,
        S2MM,
        it_channel_0
    );
    seq->npu_dma_wait(
        IT7,
        S2MM,
        it_channel_0
    );
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        kv_cache_arg_idx,
        S2MM,
        IT0,
        bd_0,
        it_channel_0,
        {0, 0, 0, kv_cache_offset + (uint32_t)K47_offset},
        {1, 1, 1, 4 * (uint32_t)DH},
        {0, 0, 0, 1},
        -1, 0, true
    );

    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        kv_cache_arg_idx,
        S2MM,
        IT7,
        bd_0,
        it_channel_0,
        {0, 0, 0, kv_cache_offset + (uint32_t)V47_offset},
        {1, 1, 1, 4 * (uint32_t)DH},
        {0, 0, 0, 1},
        -1, 0, true
    );

    seq->npu_dma_wait(
        IT0,
        S2MM,
        it_channel_0
    );
    seq->npu_dma_wait(
        IT7,
        S2MM,
        it_channel_0
    );
}

void lfm2_npu_sequence::_move_weights(npu_sequence* seq, const size_t Dout, const size_t Din, const weight_desc_t& weight){
    // the descriptor's offset is in bytes; every DMA below counts in bf16 elements
    const uint32_t w_offset = (uint32_t)(weight.offset / sizeof(bf16));
    const uint32_t columns = 4;
    const uint32_t elems = m * k;
    const uint32_t a_block_size = (uint32_t)get_quantization_byte_size(elems, weight.dtype) / sizeof(bf16);
    const uint32_t blocks_per_row = Din / k;
    const uint32_t cores = columns * 4;
    assert(Dout / m / cores > 0);
    assert(Dout / m / cores % 2 == 0);
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
                {1, 1, 1, blocks_per_row * 2 * a_block_size},
		        {0, 0, 0, 1},
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
                {1, 1, 1, blocks_per_row * 2 * a_block_size},
		        {0, 0, 0, 1},
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

void lfm2_npu_sequence::_move_kv_cache(npu_sequence* seq, const size_t L){
    const int L_padded = (L + L_CHUNK - 1) / L_CHUNK * L_CHUNK;
    const uint32_t data2move = L_padded * DH * 4;
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        kv_cache_arg_idx,
        MM2S,
        kv_tiles[0],
        bd_0,
        it_channel_0,
        {0, 0, 0, (uint32_t)K03_offset},
        {1, 1, 1, data2move},
        {0, 0, 0, 1},
        -1, 0, false, aggressive_cache
    );

    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        kv_cache_arg_idx,
        MM2S,
        kv_tiles[1],
        bd_0,
        it_channel_0,
        {0, 0, 0, (uint32_t)K47_offset},
        {1, 1, 1, data2move},
        {0, 0, 0, 1},
        -1, 0, false, aggressive_cache
    );

    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        kv_cache_arg_idx,
        MM2S,
        kv_tiles[0],
        bd_1,
        it_channel_1,
        {0, 0, 0, (uint32_t)V03_offset},
        {1, 1, 1, data2move},
        {0, 0, 0, 1},
        -1, 0, false, aggressive_cache
    );

    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        kv_cache_arg_idx,
        MM2S,
        kv_tiles[1],
        bd_1,
        it_channel_1,
        {0, 0, 0, (uint32_t)V47_offset},
        {1, 1, 1, data2move},
        {0, 0, 0, 1},
        -1, 0, false, aggressive_cache
    );
}

void lfm2_npu_sequence::_send_conv_cache(npu_sequence* seq, const int L){
    static constexpr int conv_cache_idx = 4;
    static constexpr npu_tiles conv_it = IT1;
    // two possibilities:
    // 1. L is odd (start from 1), send the conv cache in 0-1 order
    // 2. L is even (start from 2), send the conv cache in 1-0 order
    if (L % 2 == 0){ // even
        seq->npu_dma_memcpy_nd(
            sizeof(bf16),
            conv_cache_idx,
            MM2S,
            conv_it,
            bd_14,
            it_channel_1,
            {0, 0, 0, (uint32_t)D},
            {1, 1, 1, (uint32_t)D},
            {0, 0, 0, 1},
            8, 0, false, aggressive_cache
        );
        seq->npu_dma_memcpy_nd(
            sizeof(bf16),
            conv_cache_idx,
            MM2S,
            conv_it,
            bd_15,
            it_channel_1,
            {0, 0, 0, 0},
            {1, 1, 1, (uint32_t)D},
            {0, 0, 0, 1},
            8, 0, true, aggressive_cache
        );
    }
    else { // odd
        seq->npu_dma_memcpy_nd(
            sizeof(bf16),
            conv_cache_idx,
            MM2S,
            conv_it,
            bd_14,
            it_channel_1,
            {0, 0, 0, (uint32_t)0},
            {1, 1, 2, (uint32_t)D},
            {0, 0, (uint32_t)D, 1},
            8, 0, true, aggressive_cache
        );
    }
    seq->npu_dma_wait(
        conv_it,
        MM2S,
        it_channel_1
    );
}

void lfm2_npu_sequence::_receive_conv_cache(npu_sequence* seq, const int L){
    static constexpr int conv_cache_idx = 4;
    static constexpr npu_tiles conv_it = IT1;
    // two possibilities:
    // 1. L is odd (start from 1), receive the conv cache in 0
    // 2. L is even (start from 2), receive the conv cache in 1
    if (L % 2 == 0){ // even
        seq->npu_dma_memcpy_nd(
            sizeof(bf16),
            conv_cache_idx,
            S2MM,
            conv_it,
            bd_13,
            it_channel_0,
            {0, 0, 0, (uint32_t)D},
            {1, 1, 1, (uint32_t)D},
            {0, 0, 0, 1},
            -1, 0, false
        );
    }
    else { // odd
        seq->npu_dma_memcpy_nd(
            sizeof(bf16),
            conv_cache_idx,
            S2MM,
            conv_it,
            bd_13,
            it_channel_0,
            {0, 0, 0, (uint32_t)0},
            {1, 1, 1, (uint32_t)D},
            {0, 0, 0, 1},
            -1, 0, false
        );
    }
}

lfm2_npu_sequence::lfm2_npu_sequence(lfm2_desc& desc, LM_Config config, uint32_t MAX_L) : desc(&desc){
    D = config.get("hidden_size");
    DH = config.get("head_dim");
    DQ = config.get("head_dim") * config.get("num_attention_heads");
    DK = config.get("head_dim") * config.get("num_key_value_heads");
    DV = config.get("head_dim") * config.get("num_key_value_heads");
    this->MAX_L = MAX_L;
    HIDDEN_SIZE = config.get("intermediate_size");
    kv_cache_size = MAX_L * (DK + DV);
    vocab_size = config.get("vocab_size");

    K03_offset = 0;
    K47_offset = kv_cache_size / 4;
    V03_offset = kv_cache_size / 2;
    V47_offset = 3 * kv_cache_size / 4;
}

void lfm2_npu_sequence::gen_attn_layer_seq(npu_sequence* seq, const uint32_t L){
    assert(L <= MAX_L + 1);
    assert(L > 0);
    seq->clear_cmds();
    seq->npu_preemption(0);
    int proj_rtp_lock_id = 6;
    int rms_rtp_lock_id = 6;

    constexpr size_t CT_lock_address_base = 0x000001F000;

    for (int i = 0; i < 4; i++){
        seq->rtp_write(attn_qk_tiles[i], l_qk_address, L);
        seq->rtp_write(attn_kv_tiles[i], l_kv_address, L);
    }
    for (int i = 0; i < 16; i++){
        seq->rtp_write(proj_tiles[i], proj_addr, 1);
        seq->rtp_write(proj_tiles[i], CT_lock_address_base + 16 * proj_rtp_lock_id, 1); // set lock to 1
    }
    seq->rtp_write(rms_tile, rms_addr, 1);
    seq->rtp_write(rms_tile, CT_lock_address_base + 16 * rms_rtp_lock_id, 1); // set lock to 1
    seq->rtp_write(rope_tile, rope_addr, 1);

    _send_x(seq);
    _send_rms_weights(seq);
    _send_rope_weights(seq);

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

    _receive_kv_cache(seq, L); // also move qkv weights
    _move_kv_cache(seq, L);
    _move_weights(seq, D, DQ, desc->weight_desc[0]->attn_output);
    // up and gate are interleaved into one 2 * HIDDEN_SIZE band based at ffn_up
    _move_weights(seq, 2 * HIDDEN_SIZE, D, desc->weight_desc[0]->ffn_up);
    _move_weights(seq, D, HIDDEN_SIZE, desc->weight_desc[0]->ffn_down);

    // wait for receiving y
    seq->npu_dma_wait(
        xyr_tile,
        S2MM,
        it_channel_1
    );

    seq->npu_preemption(0);
    seq->cmds2seq();
}

void lfm2_npu_sequence::gen_conv_layer_seq(npu_sequence* seq, const uint32_t L){

    constexpr size_t CT_lock_address_base = 0x000001F000;
    int proj_rtp_lock_id = 6;
    int rms_rtp_lock_id = 6;
    assert(L <= MAX_L + 1);
    assert(L > 0);
    seq->clear_cmds();
    seq->npu_preemption(0);
    for (int i = 0; i < 16; i++){
        seq->rtp_write(proj_tiles[i], proj_addr, 0);
        seq->rtp_write(proj_tiles[i], CT_lock_address_base + 16 * proj_rtp_lock_id, 1); // set lock to 1
    }
    seq->rtp_write(rms_tile, rms_addr, 0);
    seq->rtp_write(rms_tile, CT_lock_address_base + 16 * rms_rtp_lock_id, 1); // set lock to 1
    seq->rtp_write(rope_tile, rope_addr, 0);

    _send_x(seq);
    _send_rms_weights(seq);
    _send_conv_cache(seq, L);
    _send_rope_weights(seq);

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

    _move_weights(seq, D * 3, D, desc->weight_desc[0]->conv_in); // send conv proj_weight

    // _send_conv_weights(seq);

    _receive_conv_cache(seq, L);

    _move_weights(seq, D, D, desc->weight_desc[0]->conv_out); // send conv proj_weight
    _move_weights(seq, 2 * HIDDEN_SIZE, D, desc->weight_desc[0]->ffn_up);
    _move_weights(seq, D, HIDDEN_SIZE, desc->weight_desc[0]->ffn_down);

    // wait for receiving y
    seq->npu_dma_wait(
        xyr_tile,
        S2MM,
        it_channel_1
    );

    seq->npu_preemption(0);
    seq->cmds2seq();
}

void lfm2_npu_sequence::gen_mha_engine_seq(npu_sequence* seq, const uint32_t L_begin, const uint32_t L_end){
    // replaced by new implementation

    std::cerr << "gen_mha_engine_seq is replaced by new implementation" << std::endl;
    exit(1);

}

void lfm2_npu_sequence::gen_conv1d_engine_seq(npu_sequence* seq,
    const uint32_t L_PADDED, // assume padded
    const uint32_t D,
    const uint32_t L_OUT,  // caller handle the calculation of L_OUT through kernel size and stride...
    const uint32_t L_PADDED_REQUIREMENT,
    const uint32_t weight_offset){
    constexpr int total_cols = 8;
    constexpr int total_rows = 4;
    constexpr npu_tiles IT[total_cols] = {IT0, IT1, IT2, IT3, IT4, IT5, IT6, IT7};

    seq->clear_cmds();
    seq->npu_preemption(0);
    //const uint32_t d = D/(4*8);
    const uint32_t d = D/8;

    auto round_up_to_multiple = [](int x, int multiple) -> int {
        if (multiple == 0) {
            return x; // Cannot divide by zero
        }
        // This uses integer division to achieve the rounding
        return ((x + multiple - 1) / multiple) * multiple;
    };

    constexpr int CT_lock_address_base = 0x000001F000;
    constexpr uint32_t CT_rtp_sync_lock_id = 6;
    for (int row = 2; row < total_rows + 2; row++){
        for (int col = 0; col < total_cols; col++){
            npu_tiles tile = get_tile(row, col);
            seq->rtp_write(tile, l_conv_address, L_PADDED);
            seq->rtp_write(tile, CT_lock_address_base+16*(CT_rtp_sync_lock_id), 1); // set lock to 1
        }
    }

    for(int col = 0; col < total_cols; col++ ){
        // purely send weights
        uint32_t w_offset = col * d + weight_offset;
        seq->npu_dma_memcpy_nd(
            2, 1,
            MM2S, IT[col],
            (npu_bd_id)(1), it_channel_0,
            {0, 0, 0, w_offset},
            {1, 1, 3, d},
            {0, 0, D, 1},
            -1, 0, false
        );
    }

    assert( L_PADDED %L_PADDED_REQUIREMENT == 0);
    uint32_t L_DIV_PADD_REQUIREMENT = L_PADDED / L_PADDED_REQUIREMENT;
    uint32_t num_blocks = L_DIV_PADD_REQUIREMENT / 1024;
    uint32_t remainder   = L_DIV_PADD_REQUIREMENT % 1024;
    // send x
    if(num_blocks > 0){
        for (int col = 0; col < total_cols; col++){

            uint32_t x_offset = col * d;
            seq->npu_dma_memcpy_nd(
                2, 2,
                MM2S, IT[col],
                (npu_bd_id)( 2), it_channel_1,
                {0, 0, 0, x_offset},
                 {num_blocks,                       1024,                        L_PADDED_REQUIREMENT,         d},
                {1024*L_PADDED_REQUIREMENT*D, L_PADDED_REQUIREMENT*D,                      D,             1},
                -1, 0, false
            );
        }

        for (int col = 0; col < total_cols; col++){

            uint32_t x_offset = col * d +   (1024*L_PADDED_REQUIREMENT*D) * (num_blocks);
            seq->npu_dma_memcpy_nd(
                2, 2,
                MM2S, IT[col],
                (npu_bd_id)(10 ), it_channel_1,
                {0, 0, 0, x_offset},
                {1,  remainder,                                 L_PADDED_REQUIREMENT,         d},
                {0, L_PADDED_REQUIREMENT*D,                      D,                         1},
                -1, 0, false
            );
        }
    }else{
        //done in single transaction

        for (int col = 0; col < total_cols; col++){

            // send x
            uint32_t x_offset = col * d;
            seq->npu_dma_memcpy_nd(
                2, 2,
                MM2S, IT[col],
                (npu_bd_id)( 2), it_channel_1,
                {0, 0, 0, x_offset},
                {1,  L_DIV_PADD_REQUIREMENT,                L_PADDED_REQUIREMENT, d},
                {0, L_PADDED_REQUIREMENT*D,                      D,             1},
                -1, 0, false
            );
        } // end for col
    }

    uint32_t y_tranasction_L_size = 1024;
    uint32_t receive_round = round_up_to_multiple(L_OUT, y_tranasction_L_size) / y_tranasction_L_size;

    int C_wait_counter = 0;  // only keep column 0 should be sufficient for our case
    for(int round = 0; round < receive_round; round++){
        int bd_offset = (round % 2) * 8;
        for (int col = 0; col < total_cols; col++){
            uint32_t y_offset = round * (y_tranasction_L_size*D) + col * d;

            if(round +1 == receive_round && L_OUT % y_tranasction_L_size != 0){
                uint32_t L_remain = L_OUT  % y_tranasction_L_size;
                seq->npu_dma_memcpy_nd(
                    2, 0,
                    S2MM, IT[col],
                    (npu_bd_id)(bd_offset + 0), it_channel_0,
                    {0, 0, 0, y_offset},
                    {1, 1,    L_remain, d},
                    {0, 0,  D,        1},
                    -1, 0, true
                );
            }else{
                seq->npu_dma_memcpy_nd(
                    2, 0,
                    S2MM, IT[col],
                    (npu_bd_id)(bd_offset + 0), it_channel_0,
                    {0, 0, 0, y_offset},
                    {1, 1,    y_tranasction_L_size, d},
                    {0, 0,  D, 1},
                    -1, 0, true
                );
            }
        } // end for col

        C_wait_counter ++;
        if (C_wait_counter == 2 ){
            for (int col = 0; col < total_cols; col++){
                seq->npu_dma_wait(IT[col], S2MM, it_channel_0);
            }
            C_wait_counter--;
        }
    } // end for round
    for(int wait_r = 0; wait_r < C_wait_counter; wait_r ++ ){
        for (int col = 0; col < total_cols; col++){
            seq->npu_dma_wait(IT[col], S2MM, it_channel_0);
        }
    }

    seq->cmds2seq();
}

void lfm2_npu_sequence::set_max_length(const uint32_t MAX_L){
    this->MAX_L = MAX_L;
    this->kv_cache_size = MAX_L * (DK + DV);
    this->K03_offset = 0;
    this->K47_offset = kv_cache_size / 4;
    this->V03_offset = kv_cache_size / 2;
    this->V47_offset = 3 * kv_cache_size / 4;
}

void lfm2_npu_sequence::gen_lm_head_seq(npu_sequence* seq){
    static constexpr npu_tiles lm_head_tiles[] = {IT2, IT3, IT4, IT5};
    static constexpr uint32_t columns = 4;
    static constexpr int lm_head_y_arg_idx = 0;
    static constexpr int lm_head_w_arg_idx = 1;
    static constexpr int lm_head_x_arg_idx = 2;
    static constexpr int lm_head_rms_arg_idx = 3;

    constexpr size_t CT_lock_address_base = 0x000001F000;
    int proj_rtp_lock_id = 6;
    int rms_rtp_lock_id = 6;
    seq->clear_cmds();
    seq->npu_preemption(0);
    for (int i = 0; i < 16; i++){
        seq->rtp_write(proj_tiles[i], proj_addr, 2);
        seq->rtp_write(proj_tiles[i], CT_lock_address_base + 16 * proj_rtp_lock_id, 1); // set lock to 1
    }
    seq->rtp_write(rms_tile, rms_addr, 2);
    seq->rtp_write(rms_tile, CT_lock_address_base + 16 * rms_rtp_lock_id, 1); // set lock to 1

    // send x
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        lm_head_x_arg_idx,
        MM2S,
        xyr_tile,
        bd_0,
        it_channel_0,
        {0, 0, 0, 0},
        {1, 1, 1, (uint32_t)D},
        {0, 0, 0, 1},
        0, 0, false, aggressive_cache
    );
    // send final rms weight
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        lm_head_rms_arg_idx,
        MM2S,
        xyr_tile,
        bd_1,
        it_channel_1,
        {0, 0, 0, 0},
        {1, 1, 1, (uint32_t)D},
        {0, 0, 0, 1},
        0, 0, false, aggressive_cache
    );
    // y_out
    seq->npu_dma_memcpy_nd(
        sizeof(bf16), lm_head_y_arg_idx,
        S2MM, IT0,  bd_15, it_channel_1,
        {0, 0, 0, 0},
        {1, 1, 1, vocab_size},
        {0, 0, 0, 1},
        -1, 0, true
    );
    // send weights

    _move_weights(seq, vocab_size, D, desc->lm_head);

    // wait for y
    seq->npu_dma_wait(
        IT0,
        S2MM,
        it_channel_1
    );
}
lfm2_npu_sequence::~lfm2_npu_sequence() = default;
