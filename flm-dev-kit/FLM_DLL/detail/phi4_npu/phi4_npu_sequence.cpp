// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include "phi4_npu_sequence_detail.hpp"

void phi4_npu_sequence::Impl::_send_x(npu_sequence* seq){
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

void phi4_npu_sequence::Impl::_send_rms_weights(npu_sequence* seq){
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

void phi4_npu_sequence::Impl::_send_rope_weights(npu_sequence* seq){
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        w_rope_arg_idx,
        MM2S,
        xyr_tile,
        bd_2,
        it_channel_0,
        {0, 0, 0, 0},
        {1, 1, 1, (uint32_t)DH},
        {0, 0, 0, 1},
        1, 0, false, aggressive_cache
    );
}

void phi4_npu_sequence::Impl::_receive_kv_cache(npu_sequence* seq, const int L){
    const uint32_t kv_cache_offset = (L - 1) * 4 * DH;
    // receive new k, v
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        kv_cache_arg_idx,
        S2MM,
        attn_tiles[0],
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
        attn_tiles[1],
        bd_0,
        it_channel_0,
        {0, 0, 0, kv_cache_offset + (uint32_t)V03_offset},
        {1, 1, 1, 4 * (uint32_t)DH},
        {0, 0, 0, 1},
        -1, 0, true
    );
    // K and V heads 4..7. Both halves are armed up front, on a second bd, so
    // the whole kv-cache write is already posted when the projection starts:
    // the weights are packed K|V|Q, so K and V come off the array first and
    // must not wait on the host to enqueue the second pair.
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        kv_cache_arg_idx,
        S2MM,
        attn_tiles[0],
        bd_1,
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
        attn_tiles[1],
        bd_1,
        it_channel_0,
        {0, 0, 0, kv_cache_offset + (uint32_t)V47_offset},
        {1, 1, 1, 4 * (uint32_t)DH},
        {0, 0, 0, 1},
        -1, 0, true
    );

    // the band is packed K|V|Q, so attn_k carries the base offset of all three
    _move_weights(seq, DK + DV + DQ, D, desc->weight_desc[0]->attn_k);

    seq->npu_dma_wait(
        attn_tiles[0],
        S2MM,
        it_channel_0
    );
    seq->npu_dma_wait(
        attn_tiles[1],
        S2MM,
        it_channel_0
    );

    seq->npu_dma_wait(
        attn_tiles[0],
        S2MM,
        it_channel_0
    );
    seq->npu_dma_wait(
        attn_tiles[1],
        S2MM,
        it_channel_0
    );
}

void phi4_npu_sequence::Impl::_move_weights(npu_sequence* seq, const size_t Dout, const size_t Din, const weight_desc_t& weight){
    // the descriptor's offset is in bytes; every DMA below counts in bf16 elements
    const uint32_t w_offset = (uint32_t)(weight.offset / sizeof(bf16));
    const uint32_t columns = 4;
    const uint32_t elems = m * k;
    const uint32_t a_block_size = get_quantization_byte_size(elems, weight.dtype) / sizeof(bf16);
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

void phi4_npu_sequence::Impl::_move_kv_cache(npu_sequence* seq, const size_t L){
    const int L_padded = (L + L_CHUNK - 1) / L_CHUNK * L_CHUNK;
    const uint32_t data2move = L_padded * DH * 4;
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        kv_cache_arg_idx,
        MM2S,
        attn_tiles[0],
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
        attn_tiles[1],
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
        attn_tiles[0],
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
        attn_tiles[1],
        bd_1,
        it_channel_1,
        {0, 0, 0, (uint32_t)V47_offset},
        {1, 1, 1, data2move},
        {0, 0, 0, 1},
        -1, 0, false, aggressive_cache
    );
}

phi4_npu_sequence::Impl::Impl(phi4_desc& desc, LM_Config config, uint32_t MAX_L) : desc(&desc){
    D = config.get("hidden_size");
    DH = config.get("head_dim");
    DQ = config.get("head_dim") * config.get("num_attention_heads");
    DK = config.get("head_dim") * config.get("num_key_value_heads");
    DV = config.get("head_dim") * config.get("num_key_value_heads");
    this->MAX_L = MAX_L;
    HIDDEN_SIZE = config.get("intermediate_size");
    kv_cache_size = MAX_L * (DK + DV);

    K03_offset = 0;
    K47_offset = kv_cache_size / 4;
    V03_offset = kv_cache_size / 2;
    V47_offset = 3 * kv_cache_size / 4;

    vocab_size = config.get("vocab_size");
    vocab_size_padded = (vocab_size + D - 1) / D * D;
}
void phi4_npu_sequence::Impl::gen_dequant_seq(npu_sequence* seq, const size_t D_in, const size_t D_out, const size_t weight_offset){
    // N is din, M is dout
    constexpr int total_cols = 8;
    constexpr int total_rows = 4;
    const int rounds = D_out / 32 / total_cols / 4;
    if (rounds * 4 * total_cols * 32 != D_out) {
        header_print("ERROR", "D_out is not aligned with 32 * 4 * total_cols");
        exit(1);
    }
    constexpr npu_tiles IT[total_cols] = {IT0, IT1, IT2, IT3, IT4, IT5, IT6, IT7};
    seq->clear_cmds();
    seq->npu_preemption(0);
    uint32_t qw_size = 32 * 256 * 5 / 8 / 4;
    for (int round = 0; round < rounds; round++){
        int bd_offset = (round % 2) * 8;
        npu_bd_id w_out_up_bd = npu_bd_id(0 + bd_offset);
        npu_bd_id w_out_down_bd = npu_bd_id(1 + bd_offset);
        npu_bd_id wq_in_up_bd = npu_bd_id(2 + bd_offset);
        npu_bd_id wq_in_down_bd = npu_bd_id(3 + bd_offset);

        for (int c = 0; c < total_cols; c++){
            uint32_t qw_offset_up = (round * 16 + c * 2) * qw_size * D_in / 256 * 2 + weight_offset / 4;
            uint32_t qw_offset_down = qw_offset_up + qw_size * D_in / 256 * 2;
            uint32_t q_offset_up = (round * 16 + c * 2) * 64 * D_in;
            uint32_t q_offset_down = q_offset_up + 64 * D_in;

            seq->npu_dma_memcpy_nd(
                2, 0,
                S2MM, IT[c],
                w_out_up_bd, it_channel_0,
                {0, 0, 0, q_offset_up},
                {1, 1, 1, (uint32_t)(64 * D_in)},
                {0, 0, 0, 1},
                -1, 0, true
            );

            seq->npu_dma_memcpy_nd(
                2, 0,
                S2MM, IT[c],
                w_out_down_bd, it_channel_1,
                {0, 0, 0, q_offset_down},
                {1, 1, 1, (uint32_t)(64 * D_in)},
                {0, 0, 0, 1},
                -1, 0, true
            );

            seq->npu_dma_memcpy_nd(
                4, 1,
                MM2S, IT[c],
                wq_in_up_bd, it_channel_0,
                {0, 0, 0, qw_offset_up},
                {1, 1, 1, (uint32_t)(2 * qw_size * D_in / 256)},
                {0, 0, 0, 1},
                -1, 0, false
            );

            seq->npu_dma_memcpy_nd(
                4, 1,
                MM2S, IT[c],
                wq_in_down_bd, it_channel_1,
                {0, 0, 0, qw_offset_down},
                {1, 1, 1, (uint32_t)(2 * qw_size * D_in / 256)},
                {0, 0, 0, 1},
                -1, 0, false
            );

            if (round > 0){
                seq->npu_dma_wait(IT[c], S2MM, it_channel_0);
                seq->npu_dma_wait(IT[c], S2MM, it_channel_1);
            }
        }
    }// end round
    for (int c = 0; c < total_cols; c++){
        seq->npu_dma_wait(IT[c], S2MM, it_channel_0);
        seq->npu_dma_wait(IT[c], S2MM, it_channel_1);
    }

    seq->cmds2seq();
}

/// @brief Writes the per-token RTPs (context length, is_attn) once.
/// @note  These are identical for every decoder layer, so they used to be
///        rewritten num_hidden_layers times per token. This sequence runs once
///        at the head of the layer runlist instead.
void phi4_npu_sequence::Impl::gen_layer_rtp_seq(npu_sequence* seq, const uint32_t L){
    assert(L <= MAX_L + 1);
    assert(L > 0);
    constexpr size_t CT_lock_address_base = 0x000001F000;
    int proj_rtp_lock_id = 6;
    int rms_rtp_lock_id = 6;
    seq->clear_cmds();
    seq->npu_preemption(0);

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

    seq->cmds2seq();
}

void phi4_npu_sequence::Impl::gen_layer_seq(npu_sequence* seq, const uint32_t L){
    assert(L <= MAX_L + 1);
    assert(L > 0);
    seq->clear_cmds();
    seq->npu_preemption(0);

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

void phi4_npu_sequence::Impl::gen_mha_engine_seq(npu_sequence* seq, const uint32_t L_begin, const uint32_t L_end){
    // replaced by new implementation

    std::cerr << "Replacing gen_mha_engine_seq with new implementation" << std::endl;
    exit(1);

}

void phi4_npu_sequence::Impl::gen_lm_head_seq(npu_sequence* seq){
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
        seq->rtp_write(proj_tiles[i], proj_addr, 0);
        seq->rtp_write(proj_tiles[i], CT_lock_address_base + 16 * proj_rtp_lock_id, 1); // set lock to 1
    }
    seq->rtp_write(rms_tile, rms_addr, 0);
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
        S2MM, IT3,  bd_15, it_channel_1,
        {0, 0, 0, 0},
        {1, 1, 1, vocab_size_padded},
        {0, 0, 0, 1},
        -1, 0, true
    );
    // send weights

    _move_weights(seq, vocab_size_padded, D, desc->lm_head);

    // wait for y
    seq->npu_dma_wait(
        IT3,
        S2MM,
        it_channel_1
    );
}

void phi4_npu_sequence::Impl::set_max_length(const uint32_t MAX_L){
    this->MAX_L = MAX_L;
    this->kv_cache_size = MAX_L * (DK + DV);
    this->K03_offset = 0;
    this->K47_offset = kv_cache_size / 4;
    this->V03_offset = kv_cache_size / 2;
    this->V47_offset = 3 * kv_cache_size / 4;
}

// ===============================================
// Externals for phi4_npu_sequence
// ===============================================
/// @brief
/// @param rtp_seq : npu_sequence ptr for layer rtp write (set L)
/// @param layer_seq : npu_sequence ptr for layer sequence
/// @param dequant_seq : npu_sequence ptr for dequant sequence
/// @param mm_engine_seq : npu_sequence ptr for mm engine sequence
/// @param mha_engine_seq : npu_sequence ptr for mha engine sequence
/// @param config : phi4 config
/// @param MAX_L : max length
phi4_npu_sequence::phi4_npu_sequence(phi4_desc& desc, LM_Config config, uint32_t MAX_L){
    this->_impl = new Impl(desc, config, MAX_L);
}

void phi4_npu_sequence::gen_layer_seq(npu_sequence* seq, const uint32_t L){
    this->_impl->gen_layer_seq(seq, L);
}

void phi4_npu_sequence::gen_layer_rtp_seq(npu_sequence* seq, const uint32_t L){
    this->_impl->gen_layer_rtp_seq(seq, L);
}

void phi4_npu_sequence::gen_dequant_seq(npu_sequence* seq, const size_t D_in, const size_t D_out, const size_t weight_offset){
    this->_impl->gen_dequant_seq(seq, D_in, D_out, weight_offset);
}

void phi4_npu_sequence::gen_mha_engine_seq(npu_sequence* seq, const uint32_t L_begin, const uint32_t L_end){
    this->_impl->gen_mha_engine_seq(seq, L_begin, L_end);
}

void phi4_npu_sequence::gen_lm_head_seq(npu_sequence* seq){
    this->_impl->gen_lm_head_seq(seq);
}

void phi4_npu_sequence::set_max_length(const uint32_t MAX_L){
    this->_impl->set_max_length(MAX_L);
}

size_t phi4_npu_sequence::get_k03_offset() const{
    return this->_impl->K03_offset;
}

size_t phi4_npu_sequence::get_k47_offset() const{
    return this->_impl->K47_offset;
}

size_t phi4_npu_sequence::get_v03_offset() const{
    return this->_impl->V03_offset;
}

size_t phi4_npu_sequence::get_v47_offset() const{
    return this->_impl->V47_offset;
}

phi4_npu_sequence::~phi4_npu_sequence() = default;
