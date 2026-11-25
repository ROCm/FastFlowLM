// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#include "gemma_npu_sequence_detail.hpp"

void gemma_npu_sequence::Impl::_send_x(npu_sequence* seq){
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
        0, 0, false
    );
}

void gemma_npu_sequence::Impl::_send_rms_weights(npu_sequence* seq, uint32_t w_offset){
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        w_rms_arg_idx,
        MM2S,
        xyr_tile,
        bd_1,
        it_channel_1,
        {0, 0, 0, w_offset},
        {1, 1, 1, (uint32_t)D * 2},
        {0, 0, 0, 1},
        0, 0, false
    );
}

void gemma_npu_sequence::Impl::_send_rope_rms_weights(npu_sequence* seq){
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
        1, 0, false
    );
}

void gemma_npu_sequence::Impl::_receive_kv_cache(npu_sequence* seq, const int L, bool is_sliding_window){
    int L_local = L - 1;
    if (is_sliding_window){
        L_local = L_local % sliding_window;
    }
    const uint32_t kv_cache_offset = L_local * 2 * DH;
    uint32_t k01_offset = is_sliding_window ? K01_sliding_offset : K01_offset;
    uint32_t k23_offset = is_sliding_window ? K23_sliding_offset : K23_offset;
    uint32_t v01_offset = is_sliding_window ? V01_sliding_offset : V01_offset;
    uint32_t v23_offset = is_sliding_window ? V23_sliding_offset : V23_offset;
    // receive new k, v
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        kv_cache_arg_idx,
        S2MM,
        IT0,
        bd_0,
        it_channel_0,
        {0, 0, 0, kv_cache_offset},
        {1, 1, 1, 2 * (uint32_t)DH},
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
        {0, 0, 0, kv_cache_offset + (uint32_t)v01_offset},
        {1, 1, 1, 2 * (uint32_t)DH},
        {0, 0, 0, 1},
        -1, 0, true
    );
    _move_weights(seq, DQ + DK + DV, D, w_qkv_offset);
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
        {0, 0, 0, kv_cache_offset + (uint32_t)k23_offset},
        {1, 1, 1, 2 * (uint32_t)DH},
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
        {0, 0, 0, kv_cache_offset + (uint32_t)v23_offset},
        {1, 1, 1, 2 * (uint32_t)DH},
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

void gemma_npu_sequence::Impl::_move_weights(npu_sequence* seq, const size_t Dout, const size_t Din, const size_t w_offset){
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
                {1, 1, 1, blocks_per_row * a_block_size * 2},
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
                {1, 1, 1, blocks_per_row * a_block_size * 2},
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

// Finished 16-row chunks the decode layer reads from the bfp16 copy: all but the chunk holding the newest row
// (README "bfp16 KV cache").
uint32_t gemma_npu_sequence::Impl::kv_bfp16_packed_chunks(const uint32_t L, bool is_sliding_window) const{
    if (is_sliding_window && L > sliding_window){
        return sliding_window / L_CHUNK - 1;
    }
    return (L - 1) / L_CHUNK;
}

// bfp16 copy of one layer's cache, after its 4 bf16 regions in the same buffer: per stream (K01, K23, V01, V23) one
// 9,216 B unit per 16-row chunk. Offsets in bf16 elements.
size_t gemma_npu_sequence::Impl::kv_bfp16_offset(bool is_sliding_window, int stream) const{
    const size_t cap = is_sliding_window ? sliding_window : MAX_L;
    const size_t bf16_region = cap * (DK + DV);
    return bf16_region + (size_t)stream * (cap / L_CHUNK) * kv_bfp16_unit_elems;
}

// Finished chunks from the bfp16 copy, in ring order past the window (after the newest chunk to the end, then from
// 0), then the newest chunk as bf16 rows (README "bfp16 KV cache").
void gemma_npu_sequence::Impl::_move_kv_cache(npu_sequence* seq, const size_t L, bool is_sliding_window){
    const bool ring = is_sliding_window && L > sliding_window;
    const uint32_t chunks = (is_sliding_window ? sliding_window : MAX_L) / L_CHUNK;
    const uint32_t newest_row = ring ? (L - 1) % sliding_window : L - 1;
    const uint32_t tail_chunk = newest_row / L_CHUNK;
    const uint32_t row_elems = DH * 2;
    const uint32_t chunk_elems = L_CHUNK * row_elems;

    struct stream_t {
        npu_tiles tile;
        npu_it_channel channel;
        uint32_t bf16_offset;
        int index;
        npu_bd_id bd_packed_1;
        npu_bd_id bd_packed_2;
        npu_bd_id bd_tail;
    };
    const stream_t streams[4] = {
        {kv_tiles[0], it_channel_0, (uint32_t)(is_sliding_window ? K01_sliding_offset : K01_offset), 0, bd_0, bd_2, bd_4},
        {kv_tiles[1], it_channel_0, (uint32_t)(is_sliding_window ? K23_sliding_offset : K23_offset), 1, bd_0, bd_2, bd_4},
        {kv_tiles[0], it_channel_1, (uint32_t)(is_sliding_window ? V01_sliding_offset : V01_offset), 2, bd_1, bd_3, bd_5},
        {kv_tiles[1], it_channel_1, (uint32_t)(is_sliding_window ? V23_sliding_offset : V23_offset), 3, bd_1, bd_3, bd_5},
    };

    for (const stream_t& s : streams){
        const uint32_t packed = (uint32_t)kv_bfp16_offset(is_sliding_window, s.index);
        // packed chunks in read order: after the newest chunk to the end (ring only), then from 0 to the newest chunk
        const uint32_t first_count = ring ? chunks - 1 - tail_chunk : 0;
        const uint32_t second_count = tail_chunk;
        if (first_count > 0){
            seq->npu_dma_memcpy_nd(
                sizeof(bf16), kv_cache_arg_idx, MM2S, s.tile, s.bd_packed_1, s.channel,
                {0, 0, 0, packed + (tail_chunk + 1) * kv_bfp16_unit_elems},
                {1, 1, 1, first_count * kv_bfp16_unit_elems},
                {0, 0, 0, 1},
                -1, 0, false, aggressive_cache
            );
        }
        if (second_count > 0){
            seq->npu_dma_memcpy_nd(
                sizeof(bf16), kv_cache_arg_idx, MM2S, s.tile, s.bd_packed_2, s.channel,
                {0, 0, 0, packed},
                {1, 1, 1, second_count * kv_bfp16_unit_elems},
                {0, 0, 0, 1},
                -1, 0, false, aggressive_cache
            );
        }
        // the newest chunk as bf16: rows 0-8 then rows 8-16, two 9-row units (rows 8 and 16 are not used)
        seq->npu_dma_memcpy_nd(
            sizeof(bf16), kv_cache_arg_idx, MM2S, s.tile, s.bd_tail, s.channel,
            {0, 0, 0, s.bf16_offset + tail_chunk * chunk_elems},
            {1, 2, 9, row_elems},
            {0, 8 * row_elems, row_elems, 1},
            -1, 0, false, aggressive_cache
        );
    }
}

gemma_npu_sequence::Impl::Impl(LM_Config config, uint32_t MAX_L){
    D = config.get("hidden_size");
    DH = config.get("head_dim");
    DQ = config.get("head_dim") * config.get("num_attention_heads");
    DK = config.get("head_dim") * config.get("num_key_value_heads");
    DV = config.get("head_dim") * config.get("num_key_value_heads");
    this->vocab_size = config.get("vocab_size");
    this->lm_head_in_layer_rows = (lm_head_in_layer_vocab + D - 1) / D * D;
    this->sliding_window = config.get("sliding_window");
    this->sliding_window_pattern = config.get("sliding_window_pattern");
    this->MAX_L = MAX_L;
    HIDDEN_SIZE = config.get("intermediate_size");
    kv_cache_size = MAX_L * (DK + DV);
    w_qkv_offset = 0;
    w_o_offset = (DQ + DK + DV) * D * 5 / 8 / sizeof(bf16);
    w_up_offset = w_o_offset + D * DQ * 5 / 8 / sizeof(bf16);
    w_down_offset = w_up_offset + (2 * HIDDEN_SIZE) * D * 5 / 8 / sizeof(bf16);
    uint32_t kv_cache_size_sliding = sliding_window * (DK + DV);
    K01_offset = 0;
    K23_offset = kv_cache_size / 4;
    V01_offset = kv_cache_size / 2;
    V23_offset = 3 * kv_cache_size / 4;
    K01_sliding_offset = 0;
    K23_sliding_offset = kv_cache_size_sliding / 4;
    V01_sliding_offset = kv_cache_size_sliding / 2;
    V23_sliding_offset = 3 * kv_cache_size_sliding / 4;
}

void gemma_npu_sequence::Impl::gen_layer_seq(npu_sequence* seq, const uint32_t L, bool is_sliding_window){
    assert(L <= MAX_L + 1);
    assert(L > 0);
    constexpr size_t CT_lock_address_base = 0x000001F000;
    int proj_rtp_lock_id = 6;
    int rms_rtp_lock_id = 6;
    seq->clear_cmds();

    seq->npu_preemption(0);

    const uint32_t qk_address = l_qk_address;
    const uint32_t kv_address = l_kv_address;
    if (is_sliding_window){
        uint32_t L_local;
        if (L <= 1024) {
            L_local = L;
        }
        else{
            L_local = 1024;
        }
        for (int i = 0; i < 4; i++){
            seq->rtp_write(attn_qk_tiles[i], qk_address, L_local);
            seq->rtp_write(attn_kv_tiles[i], kv_address, L_local);
        }
    }
    else{

        for (int i = 0; i < 4; i++){
            seq->rtp_write(attn_qk_tiles[i], qk_address, L);
            seq->rtp_write(attn_kv_tiles[i], kv_address, L);
        }
    }
    const uint32_t packed_chunks = kv_bfp16_packed_chunks(L, is_sliding_window);
    for (int i = 0; i < 4; i++){
        seq->rtp_write(attn_qk_tiles[i], qk_address + sizeof(int32_t), packed_chunks);
        seq->rtp_write(attn_kv_tiles[i], kv_address + sizeof(int32_t), packed_chunks);
    }
    for (int i = 0; i < 16; i++){
        seq->rtp_write(proj_tiles[i], proj_addr, 1);
        seq->rtp_write(proj_tiles[i], CT_lock_address_base + 16 * proj_rtp_lock_id, 1); // set lock to 1
    }
    seq->rtp_write(rms_tile, rms_addr, 1);
    seq->rtp_write(rms_tile, CT_lock_address_base + 16 * rms_rtp_lock_id, 1); // set lock to 1

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
    _move_weights(seq, D, DQ, w_o_offset);

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
    _move_weights(seq, D, HIDDEN_SIZE, w_down_offset);

    // wait for receiving y
    seq->npu_dma_wait(
        xyr_tile,
        S2MM,
        it_channel_1
    );

    seq->cmds2seq();
}

void gemma_npu_sequence::Impl::gen_mha_engine_seq(npu_sequence* seq, const uint32_t L_begin, const uint32_t L_end, bool is_sliding_window, int buffer_length){
    constexpr int total_cols = 8;
    constexpr int total_rows = 4;
    constexpr npu_tiles IT[8] = {IT0, IT1, IT2, IT3, IT4, IT5, IT6, IT7};
    constexpr int lc = 16;
    const int G = DQ / DK;
    assert(L_begin % (16 * 4) == 0);
    assert(L_end % (16 * 4) == 0);
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

    for (int round = 0; round < new_L / 64; round++){
        int bd_offset = (round % 2) * 8;
        int L_current = L_begin + round * 64;
        int kv_begin = ((L_current - using_window_size) > 0) ? (L_current - using_window_size) : 0;
        int kv_length = (L_current - kv_begin + 64) * DK / 2;
        for (int col = 0; col < 4; col++){ // col is bind to l
            for (int i = 0; i < 2; i++){ // i is bind to 0~3, 4~7
                // receive y
                size_t y_offset = (round * lc * 4 + col * lc) * DQ + i * 4 * DH; // y does not have history
                seq->npu_dma_memcpy_nd(
                    2, 0,
                    S2MM, IT[i * 4 + col],
                    (npu_bd_id)(bd_offset + 0), it_channel_0,
                    {0, 0, 0, (uint32_t)y_offset},
                    {1, 1, (uint32_t)lc, (uint32_t)(DH * 4)},
                    {0, 0, (uint32_t)DQ, 1},
                    -1, 0, true
                );
                // send q
                size_t q_offset = (round * lc * 4 + col * lc) * DQ + i * 4 * DH; // q does not have history
                seq->npu_dma_memcpy_nd(
                    2, 1,
                    MM2S, IT[i * 4 + col],
                    (npu_bd_id)(bd_offset + 1), it_channel_0,
                    {0, 0, 0, (uint32_t)q_offset},
                    {1, 1, (uint32_t)lc, (uint32_t)(DH * 4)},
                    {0, 0, (uint32_t)DQ, 1},
                    -1, 0, false
                );
            }
        } // end of for col
        // send k
        seq->npu_dma_memcpy_nd(
            2, 2,
            MM2S, IT[0],
            (npu_bd_id)(bd_offset + 2), it_channel_1,
            {0, 0, 0, (uint32_t)(this->K01_offset + kv_begin * DK / 2)},
            {1, 1, 1, (uint32_t)kv_length},
            {0, 0, 0, 1},
            -1, 0, false
        );

        // send k
        seq->npu_dma_memcpy_nd(
            2, 2,
            MM2S, IT[4],
            (npu_bd_id)(bd_offset + 2), it_channel_1,
            {0, 0, 0, (uint32_t)(this->K23_offset + kv_begin * DK / 2)},
            {1, 1, 1, (uint32_t)kv_length},
            {0, 0, 0, 1},
            -1, 0, false
        );

        // send v
        seq->npu_dma_memcpy_nd(
            2, 2,
            MM2S, IT[1],
            (npu_bd_id)(bd_offset + 3), it_channel_1,
            {0, 0, 0, (uint32_t)(this->V01_offset + kv_begin * DV / 2)},
            {1, 1, 1, (uint32_t)kv_length},
            {0, 0, 0, 1},
            -1, 0, false
        );

        // send v
        seq->npu_dma_memcpy_nd(
            2, 2,
            MM2S, IT[5],
            (npu_bd_id)(bd_offset + 3), it_channel_1,
            {0, 0, 0, (uint32_t)(this->V23_offset + kv_begin * DV / 2)},
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

void gemma_npu_sequence::Impl::gen_lm_head_in_layer_seq(npu_sequence* seq){
    assert(vocab_size == lm_head_in_layer_vocab);
    assert(lm_head_in_layer_rows % (m * 16) == 0);
    constexpr size_t CT_lock_address_base = 0x000001F000;
    constexpr int rtp_lock_id = 6;
    seq->clear_cmds();

    seq->npu_preemption(0);

    for (int i = 0; i < 16; i++){
        seq->rtp_write(proj_tiles[i], proj_addr, lm_head_proj_mode);
        seq->rtp_write(proj_tiles[i], CT_lock_address_base + 16 * rtp_lock_id, 1);
    }
    seq->rtp_write(rms_tile, rms_addr, lm_head_rms_passthrough_mode);
    seq->rtp_write(rms_tile, CT_lock_address_base + 16 * rtp_lock_id, 1);

    _send_x(seq);
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

    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        lm_head_in_layer_logits_arg_idx,
        S2MM,
        xyr_tile,
        bd_10,
        it_channel_0,
        {0, 0, 0, 0},
        {1, 1, 1, lm_head_in_layer_rows},
        {0, 0, 0, 1},
        -1, 0, true
    );

    _move_weights(seq, lm_head_in_layer_rows, D, 0);

    seq->npu_dma_wait(
        xyr_tile,
        S2MM,
        it_channel_0
    );

    seq->cmds2seq();
}

void gemma_npu_sequence::Impl::set_max_length(const uint32_t MAX_L){
    this->MAX_L = MAX_L;
    this->kv_cache_size = MAX_L * (DK + DV);
    this->K01_offset = 0;

    this->K23_offset = kv_cache_size / 4;
    this->V01_offset = kv_cache_size / 2;
    this->V23_offset = 3 * kv_cache_size / 4;
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
gemma_npu_sequence::gemma_npu_sequence(LM_Config config, uint32_t MAX_L){
    this->_impl = new Impl(config, MAX_L);
}

void gemma_npu_sequence::gen_layer_seq(npu_sequence* seq, const uint32_t L, bool is_sliding_window){
    this->_impl->gen_layer_seq(seq, L, is_sliding_window);
}

size_t gemma_npu_sequence::get_kv_bfp16_offset(bool is_sliding_window, int stream) const{
    return this->_impl->kv_bfp16_offset(is_sliding_window, stream);
}

void gemma_npu_sequence::gen_mha_engine_seq(npu_sequence* seq, const uint32_t L_begin, const uint32_t L_end, bool is_sliding_window, int buffer_length){
    this->_impl->gen_mha_engine_seq(seq, L_begin, L_end, is_sliding_window, buffer_length);
}

void gemma_npu_sequence::gen_lm_head_in_layer_seq(npu_sequence* seq){
    this->_impl->gen_lm_head_in_layer_seq(seq);
}

uint32_t gemma_npu_sequence::get_lm_head_in_layer_rows() const{
    return this->_impl->lm_head_in_layer_rows;
}

void gemma_npu_sequence::set_max_length(const uint32_t MAX_L){
    this->_impl->set_max_length(MAX_L);
}

size_t gemma_npu_sequence::get_k01_offset() const{
    return this->_impl->K01_offset;
}

size_t gemma_npu_sequence::get_k23_offset() const{
    return this->_impl->K23_offset;
}

size_t gemma_npu_sequence::get_v01_offset() const{
    return this->_impl->V01_offset;
}

size_t gemma_npu_sequence::get_v23_offset() const{
    return this->_impl->V23_offset;
}

gemma_npu_sequence::~gemma_npu_sequence() = default;
