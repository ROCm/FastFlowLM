// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include "gemma4_12b_npu_sequence.hpp"
#include "weight_desc.hpp" // flm_q40 / get_quantization_byte_size, for gen_dequant_mm_512

#include <algorithm>

gemma4_12b_npu_sequence::gemma4_12b_npu_sequence(gemma4_12b_seq_gen_parameters_t params, uint32_t MAX_L){
    D = params.D;
    DH = params.DH;
    DQ = params.DQ;
    DK = params.DK;
    DV = params.DV;
    SWA_DH = params.SWA_DH;
    SWA_DQ = params.SWA_DQ;
    SWA_DK = params.SWA_DK;
    SWA_DV = params.SWA_DV;
    num_attn_heads = params.NUM_ATTENTION_HEADS;
    num_kv_heads = params.NUM_KEY_VALUE_HEADS;
    swa_num_kv_heads = params.SWA_NUM_KEY_VALUE_HEADS;
    num_kv_per_round = num_kv_heads;
    swa_num_kv_per_round = swa_num_kv_heads;
    INTERMEDIATE_SIZE = params.INTERMEDIATE_SIZE;
    VOCAB_SIZE = params.VOCAB_SIZE_PADDED;
    SLIDING_LENGTH = params.SLIDING_WINDOW_SIZE;

    this->buffer_offsets = buffer_offsets;

    this->MAX_L = MAX_L;

    if (INTERMEDIATE_SIZE == 15360){
        e12b_rtp_addresses = rtp_addresses;
    }
    else{
        std::cerr << "Unsupported intermediate size: " << INTERMEDIATE_SIZE << std::endl;
        throw std::runtime_error("SEQ Unsupported intermediate size");
    }
    buffer_offsets.clear(); // global, swa, global_skip, swa_skip

    DEBUG_BLOCK(1,
    header_print_g("info", "Sequence Gen Params: ");
    std::cout << "\tD: " << D << std::endl;
    std::cout << "\tDH: " << DH << std::endl;
    std::cout << "\tDQ: " << DQ << std::endl;
    std::cout << "\tDK: " << DK << std::endl;
    std::cout << "\tDV: " << DV << std::endl;
    std::cout << "\tSWA_DH: " << SWA_DH << std::endl;
    std::cout << "\tSWA_DQ: " << SWA_DQ << std::endl;
    std::cout << "\tSWA_DK: " << SWA_DK << std::endl;
    std::cout << "\tSWA_DV: " << SWA_DV << std::endl;
    std::cout << "\tMAX_L: " << MAX_L << std::endl;
    std::cout << "\tINTERMEDIATE_SIZE: " << INTERMEDIATE_SIZE << std::endl;
    std::cout << "\tNUM_ATTENTION_HEADS: " << num_attn_heads << std::endl;
    std::cout << "\tNUM_KEY_VALUE_HEADS: " << num_kv_heads << std::endl;
    std::cout << "\tSLIDING_WINDOW_SIZE: " << SLIDING_LENGTH << std::endl;
    std::cout << "\tVOCAB_SIZE_PADDED: " << VOCAB_SIZE << std::endl;
    )
    DEBUG_BLOCK(1,
    header_print_g("info", "RTP ADDRESS BOOK: ");
    std::cout << "\tl_qk_address: " << rtp_addresses.l_qk_address << std::endl;
    std::cout << "\tl_kv_address: " << rtp_addresses.l_kv_address << std::endl;
    std::cout << "\tswa_l_qk_address: " << rtp_addresses.swa_l_qk_address << std::endl;
    std::cout << "\tswa_l_kv_address: " << rtp_addresses.swa_l_kv_address << std::endl;
    std::cout << "\tproj_swa_address: " << rtp_addresses.proj_swa_address << std::endl;
    std::cout << "\trms_swa_address: " << rtp_addresses.rms_swa_address << std::endl;
    )
}

void gemma4_12b_npu_sequence::set_max_length(const uint32_t MAX_L){
    this->MAX_L = MAX_L;
}

void gemma4_12b_npu_sequence::gen_layer_scale_seq(npu_sequence* seq, const bf16 layer_scale, const int L){
    // The rms core scales the whole layer output (residual included) by this value, which
    // it reads as the low half of the RTP word at rms_swa_address + 4. Every layer has its
    // own value while gen_layer_seq() is shared by all layers of a type, so the write lives
    // in this tiny sequence, run right before the layer it belongs to.
    uint32_t scale_bits = (uint32_t)(*((const uint16_t*)(&layer_scale)));
    seq->clear_cmds();
    seq->rtp_write(rms_tile, rtp_addresses.rms_swa_address + 4, scale_bits);

    // The attention cores keep their L RTP until someone overwrites it, and every layer of
    // a token runs with the same context length, so L only needs writing once per token.
    // set_context_length() passes it here for layer 0 and leaves L < 0 for the rest.
    if (L >= 0){
        // The global kv cache is MAX_L rows and not a ring, and the caller reaches
        // L = MAX_L + 1 on the token that fills the last slot. Saturate so the whole
        // sequence agrees on one in-bounds length: the newest row keeps being overwritten,
        // which is wrong but bounded, rather than an out of bounds DMA.
        // set_context_length() already warns when it saturates, so stay quiet here.
        const uint32_t l_global = std::min((uint32_t)L, MAX_L);
        // Not the true context length: _move_kv_cache() streams at most SLIDING_LENGTH rows
        // into the swa engine (the ring is only that big), so telling the cores more makes
        // them read past what was sent. gemma4e does the same clamp.
        const uint32_t l_swa    = std::min((uint32_t)L, SLIDING_LENGTH);
        for (int i = 0; i < 2; i++){
            seq->rtp_write(attn_qk_tile[i],     rtp_addresses.l_qk_address,      l_global);
            seq->rtp_write(attn_kv_tile[i],     rtp_addresses.l_kv_address,      l_global);
            seq->rtp_write(swa_attn_qk_tile[i], rtp_addresses.swa_l_qk_address,  l_swa);
            seq->rtp_write(swa_attn_kv_tile[i], rtp_addresses.swa_l_kv_address,  l_swa);
        }
    }
    seq->cmds2seq();
}

void gemma4_12b_npu_sequence::gen_layer_seq(npu_sequence* seq, const uint32_t L, gemma4_12b_layer_type_t layer_type){
    constexpr size_t CT_lock_address_base = 0x000001F000;
    LAYER_SPECIFIC_DIM(layer_type)
    DEBUG_BLOCK(2,
    header_print_g("info", "Generating sequence for layer type " + std::to_string(layer_type) + " with L = " + std::to_string(L));
    header_print_g("info", "Layer specific dimensions: ");
    std::cout << "\t_ D: "  << D << std::endl;
    std::cout << "\t_ DH: " << _DH << std::endl;
    std::cout << "\t_ DQ: " << _DQ << std::endl;
    std::cout << "\t_ DK: " << _DK << std::endl;
    std::cout << "\t_ DV: " << _DV << std::endl;
    std::cout << "\t_ INTERMEDIATE_SIZE: " << _INTERMEDIATE_SIZE << std::endl;
    std::cout << "\t_ QKV_OFFSET: " << buffer_offsets[int(layer_type)].qkv_offset << std::endl;
    std::cout << "\t_ O_OFFSET: " << buffer_offsets[int(layer_type)].o_offset << std::endl;
    std::cout << "\t_ UP_OFFSET: " << buffer_offsets[int(layer_type)].upgate_offset << std::endl;
    std::cout << "\t_ DOWN_OFFSET: " << buffer_offsets[int(layer_type)].down_offset << std::endl;
    std::cout << "\t_ IS_SWA: " << is_swa_layer(layer_type) << std::endl;
    )
    int proj_rtp_lock_id = 6;
    seq->clear_cmds();

    // Same MAX_L saturation as gen_layer_scale_seq(): clamp instead of letting the DMAs
    // address past the end of the (non ring) global kv cache.
    uint32_t L_eff = L;
    if (!is_swa_layer(layer_type) && L_eff > MAX_L){
        DEBUG_BLOCK(1,
        header_print_r("WARN", "gen_layer_seq: L " << L << " > MAX_L " << MAX_L << ", clamping");
        )
        L_eff = MAX_L;
    }

    for (int i = 0; i < 16; i++){
        seq->rtp_write(proj_tiles[i], rtp_addresses.proj_swa_address, is_swa_layer(layer_type) ? 1 : 0);
        seq->rtp_write(proj_tiles[i], CT_lock_address_base + 16 * proj_rtp_lock_id, 1); // set lock to 1
    }
    seq->rtp_write(rms_tile, rtp_addresses.rms_swa_address, is_swa_layer(layer_type) ? 1 : 0);
    // rms_swa_address + 4 holds the per layer output scale, and the attention cores' L RTPs
    // are the same for every layer of a token: both are written by gen_layer_scale_seq(),
    // because this sequence is shared by every layer of this type.

    // rope before rms, the order the reference host uses. Both ride IT3 MM2S on the
    // same channel for a sliding layer, and the rope send is the one that waits.
    _send_hidden_states(seq);
    _send_rope_rms_weights(seq, layer_type);
    _send_rms_weights(seq);

    // receive y

    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        x_arg_id,
        S2MM,
        xr_tile,
        bd_10,
        it_channel_0,
        {0, 0, 0, 0},
        {1, 1, 1, (uint32_t)D},
        {0, 0, 0, 1},
        -1, 0, true
    );
    // Without this the 16 proj cores block on w_cons_lock before their first MVM round.
    // The global layers share k and v (attention_k_eq_v), so they carry no v_proj.
    if (is_swa_layer(layer_type)){
        _move_weights(seq, _DQ + _DK + _DV, D, buffer_offsets[int(layer_type)].qkv_offset);
    }
    else {
        _move_weights(seq, _DQ + _DK, D, buffer_offsets[int(layer_type)].qkv_offset);
    }

    _receive_kv_cache(seq, L_eff, layer_type); // also move qkv weights
    _move_kv_cache(seq, L_eff, layer_type);
    _move_weights(seq, D, _DQ, buffer_offsets[int(layer_type)].o_offset);
    _move_weights(seq, 2 * _INTERMEDIATE_SIZE, D, buffer_offsets[int(layer_type)].upgate_offset);
    _move_weights(seq, D, _INTERMEDIATE_SIZE, buffer_offsets[int(layer_type)].down_offset);

    // wait for receiving y
    seq->npu_dma_wait(
        xr_tile,
        S2MM,
        it_channel_0
    );
    seq->cmds2seq();
}

void gemma4_12b_npu_sequence::gen_lm_head_seq(npu_sequence* seq, float final_scale){
    static constexpr int y_arg_id = 0;
    static constexpr int w_arg_id = 1;
    static constexpr int x_arg_id = 2;
    static constexpr int M_PER_ROUND = 16 * 32;
    static constexpr int M    = 16;
    static constexpr int M_PER_COL    = M * 4;
    static constexpr int COLS = 8;
    static npu_tiles ITs[] = {IT0, IT1, IT2, IT3, IT4, IT5, IT6, IT7};
    // Packet ids for the two flows that share IT0's MM2S 0 -- must match X_PKT_ID /
    // W_PKT_ID in FLM_Xclbin/Gemma4_12B_QAT/lm_head_npu_bin/iron/q4k_mm.py. Every
    // other shim MM2S port carries a single circuit-switched flow.
    static constexpr int X_PKT_ID = 0;
    static constexpr int W_PKT_ID = 1;
    static constexpr int NO_PKT   = -1;
    // Rounds of shim BD ping-pong; also how far the y waits lag the y issues. Three
    // sets in flight ahead of the sequencer was the knee, four bought nothing.
    static constexpr int LOOKAHEAD = 3;
    const uint32_t final_scale_address = this->rtp_addresses.lm_head_final_tune_address;
    assert(VOCAB_SIZE % M_PER_ROUND == 0);
    int rounds = VOCAB_SIZE / M_PER_ROUND;
    // lm_head.weight is q8_0, not q4_0 like the rest of the checkpoint: 17/16 byte per
    // weight (one int8 quant + one bf16 scale per 32-wide group), so a 16x256 block is
    // 4352 B. The trailing / 2 converts bytes to bf16 elements, which is the element
    // size the BDs below are programmed with. Getting this wrong (9/16) walks the weight
    // buffer at roughly half stride and every core multiplies against the wrong rows.
    size_t TOTAL_W_SIZE = size_t(D) * size_t(VOCAB_SIZE) * 17 / 16 / 2;
    size_t W_PER_COL = (size_t(M_PER_COL) * D * 17 / 16 / 2);
    size_t W_PER_ROUND = W_PER_COL * COLS;
    // A column's four cores are fed by two independent shim MM2S channels, one per
    // pair of cores: a single channel saturates at roughly a quarter of what the
    // column can consume, which is what made this kernel purely data-movement bound.
    // load_lm_head_weights has already laid the two halves out back to back in DDR.
    size_t W_PER_STREAM = W_PER_COL / 2;

    DEBUG_BLOCK(2,
    header_print_g("info", "Generating lm_head sequence: ");
    std::cout << "\trounds: " << rounds << std::endl;
    std::cout << "\tw per shim tile: " << TOTAL_W_SIZE / std::size(ITs) << " bf16" << std::endl;
    std::cout << "\ty per shim tile: " << VOCAB_SIZE / (int)std::size(ITs) << " bf16" << std::endl;
    std::cout << "\tfinal softcap: " << final_scale << " -> rtp " << final_scale_address << std::endl;
    )

    // the rtp slot takes the raw float bits; bit_cast instead of a type-punned pointer,
    // which -O3 -ffast-math is free to reorder around
    uint32_t scale_int_view = std::bit_cast<uint32_t>(final_scale);
    seq->clear_cmds();
    for (int row = 0; row < 4; row++){
        for (int col = 0; col < 8; col++){
            npu_tiles tile = get_tile(row + 2, col);
            seq->rtp_write(tile, final_scale_address, scale_int_view);
        }
    }
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        x_arg_id,
        MM2S,
        ITs[0],
        npu_bd_id(bd_0),
        it_channel_0,
        {0, 0, 0, 0},
        {1, 1, 1, (uint32_t)D * 2},
        {0, 0, 0, 1},
        X_PKT_ID, 0, false, aggressive_cache
    );

    for (int r = 0; r < rounds; r++) {
        int bd_offset = (r % (LOOKAHEAD + 1)) * 3;
        npu_bd_id bd_y  = npu_bd_id(bd_1 + bd_offset);
        npu_bd_id bd_wa = npu_bd_id(bd_2 + bd_offset);
        npu_bd_id bd_wb = npu_bd_id(bd_3 + bd_offset);
        for (size_t col = 0; col < std::size(ITs); col++){
            size_t w_col_offset = r * W_PER_ROUND + col * W_PER_COL;
            uint32_t y_offset = r * M_PER_ROUND + col * M_PER_COL;
            seq->npu_dma_memcpy_nd(
                sizeof(bf16),
                y_arg_id,
                S2MM,
                ITs[col],
                bd_y,
                it_channel_0,
                {0, 0, 0, (uint32_t)(y_offset)},
                {1, 1, 1, (uint32_t)M_PER_COL},
                {0, 0, 0, 1},
                -1, 0, true, aggressive_cache
            );
            // Group a (cores 0/1) on channel 1, group b (cores 2/3) on channel 0.
            // Channel 0 of IT0 is the one shared with the x broadcast, so only that
            // transfer needs a packet id.
            seq->npu_dma_memcpy_nd(
                sizeof(bf16),
                w_arg_id,
                MM2S,
                ITs[col],
                bd_wa,
                it_channel_1,
                {0, 0, 0, (uint32_t)(w_col_offset)},
                {1, 1, 1, (uint32_t)W_PER_STREAM},
                {0, 0, 0, 1},
                NO_PKT, 0, false, aggressive_cache
            );
            seq->npu_dma_memcpy_nd(
                sizeof(bf16),
                w_arg_id,
                MM2S,
                ITs[col],
                bd_wb,
                it_channel_0,
                {0, 0, 0, (uint32_t)(w_col_offset + W_PER_STREAM)},
                {1, 1, 1, (uint32_t)W_PER_STREAM},
                {0, 0, 0, 1},
                col == 0 ? W_PKT_ID : NO_PKT, 0, false, aggressive_cache
            );
            // Four sets of shim BDs are in flight, so a round only has to wait for the
            // y of the round three back rather than the one immediately before. That
            // keeps three rounds of weight transfers queued ahead of the sequencer.
            if (r > LOOKAHEAD - 1){
                seq->npu_dma_wait(
                    ITs[col],
                    S2MM,
                    it_channel_0
                );
            }
        }
    }
    for (size_t col = 0; col < std::size(ITs); col++){
        for (int i = 0; i < LOOKAHEAD; i++){
            seq->npu_dma_wait(
                ITs[col],
                S2MM,
                it_channel_0
            );
        }
    }
    seq->cmds2seq();
}


void gemma4_12b_npu_sequence::_send_hidden_states(npu_sequence* seq){
    // send x
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        x_arg_id,
        MM2S,
        xr_tile,
        bd_0,
        it_channel_0,
        {0, 0, 0, 0},
        {1, 1, 1, (uint32_t)D},
        {0, 0, 0, 1},
        9, 0, true, aggressive_cache
    );
    seq->npu_dma_wait(
        xr_tile,
        MM2S,
        it_channel_0
    );
}

void gemma4_12b_npu_sequence::_send_rms_weights(npu_sequence* seq){
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        rms_arg_id,
        MM2S,
        xr_tile,
        bd_1,
        it_channel_1,
        {0, 0, 0, 0},
        {1, 1, 1, (uint32_t)D * 4},
        {0, 0, 0, 1},
        // fire and forget: the rms core consumes this once the proj cores have run, so
        // issuing a token and waiting on it here deadlocks the control sequence.
        9, 0, false, aggressive_cache
    );
}

void gemma4_12b_npu_sequence::_send_rope_rms_weights(npu_sequence* seq, gemma4_12b_layer_type_t layer_type){
    LAYER_SPECIFIC_DIM(layer_type)
    if (is_swa_layer(layer_type)){
        seq->npu_dma_memcpy_nd(
            sizeof(bf16),
            rope_rms_arg_id,
            MM2S,
            xr_tile,
            bd_4,
            it_channel_1,
            {0, 0, 0, 0},
            {1, 1, 1, (uint32_t)(_DH * 3)},
            {0, 0, 0, 1},
            1, 0, true, no_cache
        );
        seq->npu_dma_wait(
            xr_tile,
            MM2S,
            it_channel_1
        );
    }
    else {
        seq->npu_dma_memcpy_nd(
            sizeof(bf16),
            rope_rms_arg_id,
            MM2S,
            xr_tile,
            bd_4,
            it_channel_0,
            {0, 0, 0, 0},
            {1, 1, 1, (uint32_t)(_DH * 3)},
            {0, 0, 0, 1},
            1, 0, true, no_cache
        );
        seq->npu_dma_wait(
            xr_tile,
            MM2S,
            it_channel_0
        );
    }

}

void gemma4_12b_npu_sequence::_receive_kv_cache(npu_sequence* seq, const int L, gemma4_12b_layer_type_t layer_type){
    LAYER_SPECIFIC_DIM(layer_type)
    DEBUG_BLOCK(2,
    header_print_g("info", "Moving KV cache for layer type " + std::to_string(layer_type) + " with L = " + std::to_string(L));
    std::cout << "\t_ L: " << L << std::endl;
    std::cout << "\t_ _DK: " << _DK << std::endl;
    std::cout << "\t_ _DV: " << _DV << std::endl;
    std::cout << "\t_ MAX_L: " << MAX_L << std::endl;
    )
    int L_local = L - 1;
    if (is_swa_layer(layer_type)) {
        L_local = L_local % SLIDING_LENGTH;
    }
    uint32_t kv_cache_size;
    if (is_swa_layer(layer_type)){
        kv_cache_size = SLIDING_LENGTH * (_DK + _DV);
    }
    else {
        kv_cache_size = (_DK + _DV) * MAX_L;
    }
    uint32_t v_offset = kv_cache_size / 2;
    uint32_t L_offset = L_local * _DK;

    npu_it_channel receiving_channel = is_swa_layer(layer_type)? it_channel_1 : it_channel_0;

    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        kv_cache_arg_id,
        S2MM,
        attn_tile,
        bd_0,
        receiving_channel,
        {0, 0, 0, (uint32_t)L_offset},
        {1, 1, 1, (uint32_t)_DK},
        {0, 0, 0, 1},
        -1, 0, true
    );

    seq->npu_dma_wait(
        attn_tile,
        S2MM,
        receiving_channel
    );
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        kv_cache_arg_id,
        S2MM,
        attn_tile,
        bd_1,
        receiving_channel,
        {0, 0, 0, (uint32_t)(L_offset + v_offset)},
        {1, 1, 1, (uint32_t)_DV},
        {0, 0, 0, 1},
        -1, 0, true
    );
    seq->npu_dma_wait(
        attn_tile,
        S2MM,
        receiving_channel
    );
}


void gemma4_12b_npu_sequence::_move_kv_cache(npu_sequence* seq, const size_t L, gemma4_12b_layer_type_t layer_type){
    LAYER_SPECIFIC_DIM(layer_type)
    DEBUG_BLOCK(2,
    header_print_g("info", "Moving KV cache for layer type " + std::to_string(layer_type) + " with L = " + std::to_string(L));
    std::cout << "\t_ L: " << L << std::endl;
    std::cout << "\t_ _DK: " << _DK << std::endl;
    std::cout << "\t_ _DV: " << _DV << std::endl;
    std::cout << "\t_ MAX_L: " << MAX_L << std::endl;
    )
    uint32_t kv_cache_size;
    if (is_swa_layer(layer_type)){
        kv_cache_size = SLIDING_LENGTH * (_DK + _DV);
    }
    else {
        kv_cache_size = (_DK + _DV) * MAX_L;
    }
    uint32_t v_offset = kv_cache_size / 2;
    int pkt_id = is_swa_layer(layer_type) ? 13 : 12;
    // the cache is replayed into the engine that owns this layer
    const npu_tiles src_tile = is_swa_layer(layer_type) ? swa_attn_tile : attn_tile;

    if (is_swa_layer(layer_type)) {
        if (L % SLIDING_LENGTH == 0){ // corner, from 0 to end
            // move the oldest block to the new block
            uint32_t data2move = SLIDING_LENGTH * _DK;
            seq->npu_dma_memcpy_nd(
                sizeof(bf16),
                kv_cache_arg_id,
                MM2S,
                src_tile,
                bd_8,
                it_channel_0,
                {0, 0, 0, 0},
                {1, 1, 1, data2move},
                {0, 0, 0, 1},
                pkt_id, 0, false, aggressive_cache
            );

            seq->npu_dma_memcpy_nd(
                sizeof(bf16),
                kv_cache_arg_id,
                MM2S,
                src_tile,
                bd_9,
                it_channel_1,
                {0, 0, 0, (uint32_t)(v_offset)},
                {1, 1, 1, data2move},
                {0, 0, 0, 1},
                pkt_id, 0, false, aggressive_cache
            );
            return ;
        }
        else if (L > SLIDING_LENGTH) { // dual phase
            uint32_t L_begin = L % SLIDING_LENGTH;
            uint32_t L_phase_1 = SLIDING_LENGTH - L_begin;
            uint32_t L_phase_2 = L_begin;
            uint32_t offset = L_begin * _DK;
            uint32_t data2move_1 = L_phase_1 * _DK;
            uint32_t data2move_2 = L_phase_2 * _DK;

            seq->npu_dma_memcpy_nd(
                sizeof(bf16),
                kv_cache_arg_id,
                MM2S,
                src_tile,
                bd_8,
                it_channel_0,
                {0, 0, 0, offset},
                {1, 1, 1, data2move_1},
                {0, 0, 0, 1},
                pkt_id, 0, false, aggressive_cache
            );

            seq->npu_dma_memcpy_nd(
                sizeof(bf16),
                kv_cache_arg_id,
                MM2S,
                src_tile,
                bd_9,
                it_channel_1,
                {0, 0, 0, (uint32_t)(v_offset + offset)},
                {1, 1, 1, data2move_1},
                {0, 0, 0, 1},
                pkt_id, 0, false, aggressive_cache
            );
            // phase 2
            seq->npu_dma_memcpy_nd(
                sizeof(bf16),
                kv_cache_arg_id,
                MM2S,
                src_tile,
                bd_10,
                it_channel_0,
                {0, 0, 0, 0},
                {1, 1, 1, data2move_2},
                {0, 0, 0, 1},
                pkt_id, 0, false, aggressive_cache
            );

            seq->npu_dma_memcpy_nd(
                sizeof(bf16),
                kv_cache_arg_id,
                MM2S,
                src_tile,
                bd_11,
                it_channel_1,
                {0, 0, 0, (uint32_t)(v_offset)},
                {1, 1, 1, data2move_2},
                {0, 0, 0, 1},
                pkt_id, 0, false, aggressive_cache
            );
            return;
        }
    }
    DEBUG_BLOCK(2,
    std::cout << "Single phase move for layer type (Fall back path) " << layer_type << std::endl;
    )
    // fallback path, single phase, from zero to L.
    // L is rounded up to a whole L_CHUNK, which can push it past the number of rows the
    // cache physically holds (MAX_L for global layers, SLIDING_LENGTH for sliding ones).
    // Since V starts at kv_cache_size/2, an unclamped L_padded makes the second
    // descriptor below read off the end of the buffer, so bound it here: this is the
    // function that emits the address, so this is where the bound belongs.
    const uint32_t cache_rows = is_swa_layer(layer_type) ? (uint32_t)SLIDING_LENGTH : MAX_L;
    uint32_t L_padded = (uint32_t)((L + L_CHUNK - 1) / L_CHUNK * L_CHUNK);
    if (L_padded > cache_rows) L_padded = cache_rows;
    const uint32_t data2move = L_padded * _DK;
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        kv_cache_arg_id,
        MM2S,
        src_tile,
        bd_8,
        it_channel_0,
        {0, 0, 0, 0},
        {1, 1, 1, data2move},
        {0, 0, 0, 1},
        pkt_id, 0, true, aggressive_cache
    );

    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        kv_cache_arg_id,
        MM2S,
        src_tile,
        bd_9,
        it_channel_1,
        {0, 0, 0, (uint32_t)(v_offset)},
        {1, 1, 1, data2move},
        {0, 0, 0, 1},
        pkt_id, 0, true, aggressive_cache
    );

    seq->npu_dma_wait(
        src_tile,
        MM2S,
        it_channel_0
    );
     seq->npu_dma_wait(
        src_tile,
        MM2S,
        it_channel_1
    );
}

void gemma4_12b_npu_sequence::_move_weights(npu_sequence* seq, const size_t Dout, const size_t Din, const size_t w_offset){
    // 16x256 q4_0 blocks, the tiling every gemma4-12b kernel expects (Q4NX_ROW_BLOCK_SIZE
    // in decoding/models/gemma4-12b.h). 12 rows here made the assert below fire on qkv.
    const int m = 16;
    const int k = 256;
    const uint32_t columns = 4;
    const uint32_t elems = m * k;
    // Q4_0 block in bf16 elements: 2304 B per 16x256 block (256 B bf16 scales + 2048 B
    // signed nibbles), halved because the DMA element size is sizeof(bf16).
    const uint32_t a_block_size = elems * 9 / 16 / 2;
    const uint32_t blocks_per_row = Din / k;
    const uint32_t cores = columns * 4;
    assert(Dout / m / cores > 0);
    assert(Dout % (m * cores) == 0);
    for (size_t round = 0; round < Dout / m / cores; round++){
        uint32_t bd_offset = (round % 2) * 8;
        for (uint32_t col = 0; col < columns; col++){
            seq->npu_dma_memcpy_nd(
                sizeof(bf16),
                proj_arg_id,
                MM2S,
                mvm_tiles[col],
                npu_bd_id(bd_1 + bd_offset),
                it_channel_0,
                {0, 0, 0, (uint32_t)((round * cores + col * 4) * a_block_size * blocks_per_row + (uint32_t)w_offset)},
                {1, 1, 1, 2 * blocks_per_row * a_block_size},
                {0, 0, 0, 1},
                -1, 0, true, aggressive_cache
            );
            seq->npu_dma_memcpy_nd(
                sizeof(bf16),
                proj_arg_id,
                MM2S,
                mvm_tiles[col],
                npu_bd_id(bd_2 + bd_offset),
                it_channel_1,
                {0, 0, 0, (uint32_t)((round * cores + col * 4 + 2) * a_block_size * blocks_per_row + (uint32_t)w_offset)},
                {1, 1, 1, 2 * blocks_per_row * a_block_size},
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

/// \brief Prefill projection: y[L, D_out] = dequant(w) * x[L, D_in].

void gemma4_12b_npu_sequence::gen_dequant_mm_512(npu_sequence* seq, const uint32_t L,
                                                 const uint32_t D_in, const uint32_t D_out,
                                                 const size_t weight_offset, const size_t out_offset,
                                                 const int rtp_act, const uint32_t glu_slab,
                                                 const dequant_output_mode_t glu_part){
    static constexpr npu_tiles IT[8] = {IT0, IT1, IT2, IT3, IT4, IT5, IT6, IT7};

    static constexpr uint32_t desired_m = 128;  // L rows per core row
    static constexpr uint32_t desired_k = 256;  // D_in per k-tile, one q4_0 block wide
    static constexpr uint32_t desired_n = 64;   // D_out columns per core column

    static constexpr uint32_t half_lanes = 4;   // shims per half: 4 core columns == 4 core rows
    static constexpr uint32_t unit_m = desired_m * half_lanes;  // 512 L rows per unit
    static constexpr uint32_t unit_n = desired_n * half_lanes;  // 256 D_out columns per unit

    // RTP slots of the two core rows of a dequant_mm pair, read off the overlay.
    static constexpr uint32_t even_row_k_addr   = 54528;
    static constexpr uint32_t even_row_act_addr = 54592;
    static constexpr uint32_t odd_row_k_addr    = 61696;
    static constexpr uint32_t odd_row_act_addr  = 61760;

    static constexpr uint32_t y_arg_idx = 0;
    static constexpr uint32_t x_arg_idx = 1;
    static constexpr uint32_t w_arg_idx = 2;

    const uint32_t Dout = D_out;
    const uint32_t Din  = D_in;

    assert(Din  % desired_k == 0);
    assert(Dout % unit_n    == 0);
    assert(L    % unit_m    == 0);
    assert(L > 0 && Dout > 0);
    if (glu_slab != 0){
        // a unit is the smallest thing the y scatter can redirect, so a seam must
        // never fall inside one, and the slabs must pair up
        assert(glu_slab % unit_n == 0);
        assert(Dout % (2 * glu_slab) == 0);
    }
    else {
        assert(glu_part == NORMAL_DEQUANT && "half selection needs a slab period");
    }

    const uint32_t K_div_k = Din / desired_k;  // also the q4_0 blocks per weight row
    // K_div_k rides the BD iteration dimension, which the hardware caps at 6 bits
    assert(K_div_k < 64 && "D_in too wide for the weight gather's iteration dimension");

    // one 16x256 q4_0 block: 2048 B of nibbles + 256 B of bf16 scales. Half a
    // QXNX_ROW_BLOCK_SIZE chunk, so get_quantization_byte_size() cannot supply it.
    static constexpr uint32_t blk = 16 * 256 * 9 / 16;  // 2304 B

    const uint32_t T_all = Dout / unit_n;   // Dout tiles present in the weight buffer
    // ... of which this app computes all, or every other slab: the activation RTP is
    // broadcast to every core, so GeLU-ing gate alone means running it alone.
    const uint32_t slab_units = (glu_slab != 0) ? (glu_slab / unit_n) : T_all;
    const uint32_t T      = (glu_part == NORMAL_DEQUANT) ? T_all : T_all / 2;
    const uint32_t C      = L / unit_m;      // L chunks
    const uint32_t units  = T * C;
    const uint32_t rounds = (units + 1) / 2;

    // compact tile index of this run -> tile index in the weight buffer
    auto phys_tile = [&](uint32_t t) -> uint32_t {
        if (glu_part == NORMAL_DEQUANT){
            return t;
        }
        const uint32_t parity = (glu_part == GATE_MATRIX) ? 1 : 0;
        return (t / slab_units) * 2 * slab_units + parity * slab_units + t % slab_units;
    };

    seq->clear_cmds();

    for (uint32_t row = 2; row < 6; row += 2){
        for (uint32_t col = 0; col < 8; col++){
            npu_tiles tile = get_tile(row, col);
            seq->rtp_write(tile, even_row_k_addr, K_div_k);
            seq->rtp_write(tile, even_row_act_addr, (uint32_t)rtp_act);
        }
    }
    for (uint32_t row = 3; row < 6; row += 2){
        for (uint32_t col = 0; col < 8; col++){
            npu_tiles tile = get_tile(row, col);
            seq->rtp_write(tile, odd_row_k_addr, K_div_k);
            seq->rtp_write(tile, odd_row_act_addr, (uint32_t)rtp_act);
        }
    }

    // byte offset of the 64-row slab this lane owns; the slab base is the same in
    // either block ordering, only the order inside it differs (see send_w)
    auto w_byte_off = [&](uint32_t t, uint32_t lane) -> uint32_t {
        return (uint32_t)(weight_offset +
            get_quantization_byte_size((size_t)(t * unit_n + lane * desired_n) * Din, flm_q40));
    };
    auto x_elem_off = [&](uint32_t c, uint32_t lane) -> uint32_t {
        return (c * unit_m + lane * desired_m) * Din;
    };
    // physical Dout base of tile `t` -> its column base in y. Identity for a plain
    // matrix; for fused up/gate it undoes the slab interleave into [ up | gate ].
    auto y_dout_base = [&](uint32_t t) -> uint32_t {
        const uint32_t phys = t * unit_n;
        if (glu_slab == 0) {
            return phys;
        }
        const uint32_t slab = phys / glu_slab;   // even -> up, odd -> gate
        return (slab % 2) * (Dout / 2)           // which half of y
             + (slab / 2) * glu_slab             // which slab within that half
             + phys % glu_slab;                  // offset inside the slab
    };
    auto y_elem_off = [&](uint32_t c, uint32_t t, uint32_t lane) -> uint32_t {
        return (uint32_t)((c * unit_m + lane * desired_m) * Dout + y_dout_base(t) + out_offset);
    };

    auto send_w = [&](uint32_t shim, npu_bd_id bd, uint32_t t, uint32_t lane){
        seq->npu_dma_memcpy_nd(
            sizeof(char), w_arg_idx, MM2S, IT[shim], bd, it_channel_1,
            {0, 0, 0, w_byte_off(t, lane)},
            {K_div_k, 2, 2, blk},
            {2 * blk, 2 * K_div_k * blk, blk, 1},
            -1, 0, false, aggressive_cache
        );
    };
    auto recv_y = [&](uint32_t shim, npu_bd_id bd, uint32_t c, uint32_t t, uint32_t lane){
        seq->npu_dma_memcpy_nd(
            sizeof(bf16), y_arg_idx, S2MM, IT[shim], bd, it_channel_0,
            {0, 0, 0, y_elem_off(c, t, lane)},
            {1, 1, desired_m, unit_n},
            {0, 0, Dout, 1},
            -1, 0, true
        );
    };
    auto send_x = [&](uint32_t shim, npu_bd_id bd, uint32_t c, uint32_t lane){
        seq->npu_dma_memcpy_nd(
            sizeof(bf16), x_arg_idx, MM2S, IT[shim], bd, it_channel_0,
            {0, 0, 0, x_elem_off(c, lane)},
            {1, K_div_k, desired_m, desired_k},
            {0, desired_k, Din, 1},
            -1, 0, false
        );
    };

    bool prev_valid = false;   // is there a round in flight to drain?
    bool prev_right = false;   // did it program the right half?

    for (uint32_t r = 0; r < rounds; r++){
        const uint32_t u_l = 2 * r;
        const uint32_t u_r = 2 * r + 1;

        // Only the very last round can come up short, and only when T * C is odd.
        const bool right_active = (u_r < units);

        const uint32_t cl = u_l / T, tl = phys_tile(u_l % T);
        const uint32_t cr = right_active ? u_r / T : 0;
        const uint32_t tr = right_active ? phys_tile(u_r % T) : 0;

        const int bd_offset = (int)(r % 2) * 8;
        npu_bd_id y_bd_0 = npu_bd_id(0 + bd_offset);
        npu_bd_id y_bd_1 = npu_bd_id(1 + bd_offset);
        npu_bd_id x_bd_0 = npu_bd_id(2 + bd_offset);
        npu_bd_id x_bd_1 = npu_bd_id(3 + bd_offset);
        npu_bd_id w_bd_0 = npu_bd_id(4 + bd_offset);
        npu_bd_id w_bd_1 = npu_bd_id(5 + bd_offset);

        for (uint32_t lane = 0; lane < half_lanes; lane++){
            send_w(lane, w_bd_0, tl, lane);
            if (right_active) send_w(lane + 4, w_bd_1, tr, lane);

            recv_y(lane, y_bd_0, cl, tl, lane);
            if (right_active) recv_y(lane + 4, y_bd_1, cr, tr, lane);

            send_x(lane, x_bd_0, cl, lane);
            if (right_active) send_x(lane + 4, x_bd_1, cr, lane);
        }

        if (prev_valid){
            for (uint32_t s = 0; s < half_lanes; s++){
                seq->npu_dma_wait(IT[s], S2MM, it_channel_0);
                if (prev_right) seq->npu_dma_wait(IT[s + 4], S2MM, it_channel_0);
            }
        }
        prev_valid = true;
        prev_right = right_active;
    }

    if (prev_valid){
        for (uint32_t s = 0; s < half_lanes; s++){
            seq->npu_dma_wait(IT[s], S2MM, it_channel_0);
            if (prev_right) seq->npu_dma_wait(IT[s + 4], S2MM, it_channel_0);
        }
    }

    seq->cmds2seq();
}

/// \brief Prefill flash attention for a global layer: o[q_rows, DQ] = softmax(QK^T + causal) V.
///
/// Ported from FLM_Xclbin/Gemma4_12B_QAT/attention_DH_512_prefill/host/seq_gen.hpp, which is
/// the standalone testbench for this overlay. Two differences from that testbench, both
/// deliberate:
///   - it passes its own L_padded as MAX_L because its kv buffer is exactly that tall. Here
///     the kv cache is the real per layer cache, so kv_rows is its true row capacity.
///   - the > 128*1024 kv branch is dropped. It reuses the same two BD ids inside one round,
///     which is a hazard, and 128 * 1024 rows is far past any context this model runs.

void gemma4_12b_npu_sequence::gen_attn_prefill_global(npu_sequence* seq, const uint32_t L_begin,
                                                      const uint32_t q_rows, const uint32_t kv_rows,
                                                      const size_t kv_elems){
    // The DH_512 floorplan is two compute units of four shim columns each; a column drains
    // lc * 4 of the 128 query rows a round covers.
    static constexpr npu_tiles IT[2][4] = {{IT0, IT1, IT2, IT3},
                                           {IT4, IT5, IT6, IT7}};
    static constexpr uint32_t lc     = 8;
    static constexpr uint32_t num_cu = 2;
    static constexpr uint32_t cols   = 4;
    static constexpr uint32_t rows_per_round = lc * 16;  // 128 query rows per round
    static constexpr uint32_t kv_block       = 128;      // kv rows per BD iteration step
    static constexpr uint32_t max_kv_iters   = 1024;     // 6 bit iteration count, x kv_block

    static constexpr int o_arg  = 0;
    static constexpr int q_arg  = 1;
    static constexpr int kv_arg = 2;

    const uint32_t dh    = this->DH;
    const uint32_t heads = (uint32_t)this->num_attn_heads;
    const uint32_t q_row = dh * heads;  // == DQ, the row stride of q and o

    assert(this->num_kv_heads == 1 && "the DH_512 overlay streams a single kv head");
    assert(this->DK == dh && this->DV == dh);
    assert(heads % num_cu == 0);
    assert(q_rows > 0 && q_rows % rows_per_round == 0 && "prefill attention needs q_rows % 128 == 0");

    const uint32_t Q_chunks = q_rows / rows_per_round;
    // Causal, not windowed. A window as wide as the cache can never clip a real position,
    // so the kernel's max(0, q - window) collapses to 0 for every row.
    const uint32_t window = kv_rows;
    // Where the V half of this cache starts, in elements.
    const uint32_t v_base = (uint32_t)(kv_elems / 2);

    seq->clear_cmds();

    for (uint32_t row = 2; row < 6; row++){
        for (uint32_t col = 0; col < 8; col++){
            npu_tiles tile = get_tile(row, col);
            seq->rtp_write(tile, global_attn_prefill_rtp.l_begin_address,     L_begin);
            seq->rtp_write(tile, global_attn_prefill_rtp.l_chunks_address,    Q_chunks);
            seq->rtp_write(tile, global_attn_prefill_rtp.window_size_address, window);
        }
    }

    for (uint32_t head = 0; head < heads / num_cu; head++){
        for (uint32_t round = 0; round < Q_chunks; round++){
            const uint32_t Lq = L_begin + round * rows_per_round;
            // kv_begin is 0 for every round of a causal pass; keep the windowed form so the
            // sliding design can reuse this shape unchanged.
            const uint32_t kv_begin = (Lq > window) ? (Lq - window) : 0;
            // the round covers rows [Lq, Lq + 128), so keys run up to the block that holds
            // its last row: round_up(Lq + 128, 128)
            const uint32_t kv_end = (Lq + 2 * rows_per_round - 1) / rows_per_round * rows_per_round;
            const uint32_t kv_len = kv_end - kv_begin;

            assert(kv_end <= kv_rows && "the kv cache is too short for this round's tail read");
            assert(kv_len <= kv_block * max_kv_iters && "kv stream longer than one BD sweep");

            const uint32_t bd = (round % 2) * 8;  // double buffer the descriptors

            for (uint32_t cu = 0; cu < num_cu; cu++){
                const uint32_t h = head * num_cu + cu;

                // receive o
                for (uint32_t col = 0; col < cols; col++){
                    const uint32_t o_off = h * dh
                                         + (round * rows_per_round + col * lc * 4) * q_row;
                    seq->npu_dma_memcpy_nd(
                        sizeof(bf16), o_arg,
                        S2MM, IT[cu][col],
                        (npu_bd_id)(bd + 0), it_channel_0,
                        {0, 0, 0, o_off},
                        {1, 1, 4 * lc, dh},
                        {0, 0, q_row, 1},
                        -1, 0, true
                    );
                }

                // send q, as four quarters over the two channels of the unit's outer shims
                const uint32_t q_off = h * dh + round * rows_per_round * q_row;
                const npu_tiles      q_tile[4] = {IT[cu][0], IT[cu][0], IT[cu][3], IT[cu][3]};
                const npu_it_channel q_chan[4] = {it_channel_0, it_channel_1,
                                                  it_channel_0, it_channel_1};
                for (uint32_t part = 0; part < 4; part++){
                    seq->npu_dma_memcpy_nd(
                        sizeof(bf16), q_arg,
                        MM2S, q_tile[part], (npu_bd_id)(bd + 1 + part), q_chan[part],
                        {0, 0, 0, q_off + part * lc * 4 * q_row},
                        {1, 1, 4 * lc, dh},
                        {0, 0, q_row, 1},
                        -1, 0, false
                    );
                }
            }

            // send k and v: one kv head, broadcast from a single shim to every core
            const uint32_t k_off = kv_begin * dh;
            seq->npu_dma_memcpy_nd(
                sizeof(bf16), kv_arg,
                MM2S, IT[0][2], (npu_bd_id)(bd + 5), it_channel_0,
                {0, 0, 0, k_off},
                {1, kv_len / kv_block, kv_block, dh},
                {0, kv_block * dh, dh, 1},
                -1, 0, false
            );
            seq->npu_dma_memcpy_nd(
                sizeof(bf16), kv_arg,
                MM2S, IT[0][2], (npu_bd_id)(bd + 6), it_channel_1,
                {0, 0, 0, k_off + v_base},
                {1, kv_len / kv_block, kv_block, dh},
                {0, kv_block * dh, dh, 1},
                -1, 0, false
            );

            // drain the previous round while this one is in flight
            if (round > 0){
                for (uint32_t cu = 0; cu < num_cu; cu++){
                    for (uint32_t col = 0; col < cols; col++){
                        seq->npu_dma_wait(IT[cu][col], S2MM, it_channel_0);
                    }
                }
            }
        }
        for (uint32_t cu = 0; cu < num_cu; cu++){
            for (uint32_t col = 0; col < cols; col++){
                seq->npu_dma_wait(IT[cu][col], S2MM, it_channel_0);
            }
        }
    }

    seq->cmds2seq();
}

/// \brief Prefill attention for the query rows of one vision block of a global layer.
///
/// Ported from FLM_Xclbin/Gemma4_12B_QAT/attention_global_image/host/seq_gen.hpp. The
/// floorplan and every descriptor shape are gen_attn_prefill_global()'s -- same two compute
/// units of four shim columns, same q quartering, same single broadcast kv shim -- so what
/// is below is that function with the position arithmetic removed. Three deliberate
/// differences from the testbench, the first two being the same ones the causal port made:
///   - the V half is found at kv_elems / 2 from the *real* cache rather than at
///     L_padded * DK. The testbench's kv buffer is exactly as tall as its padded chunk; the
///     DLL's is the layer's MAX_L cache.
///   - the > 128*1024 kv branch is dropped. It reuses the same two BD ids twice inside one
///     round, and 128k keys is far past any context this model runs; assert instead.
///   - the testbench derives kv_length from its own L_padded, which conflates "rows the
///     buffer has" with "rows the block may see". Those are two different numbers here, so
///     the extent is rounded up from kv_valid_rows and checked against the capacity.
void gemma4_12b_npu_sequence::gen_attn_prefill_global_image(npu_sequence* seq,
                                                            const uint32_t q_rows,
                                                            const uint32_t kv_valid_rows,
                                                            const uint32_t kv_rows,
                                                            const size_t kv_elems){
    static constexpr npu_tiles IT[2][4] = {{IT0, IT1, IT2, IT3},
                                           {IT4, IT5, IT6, IT7}};
    static constexpr uint32_t lc     = 8;
    static constexpr uint32_t num_cu = 2;
    static constexpr uint32_t cols   = 4;
    static constexpr uint32_t rows_per_round = lc * 16;  // 128 query rows per round
    static constexpr uint32_t kv_block       = 128;      // kv rows per BD iteration step
    static constexpr uint32_t max_kv_iters   = 1024;     // 6 bit iteration count, x kv_block

    static constexpr int o_arg  = 0;
    static constexpr int q_arg  = 1;
    static constexpr int kv_arg = 2;

    const uint32_t dh    = this->DH;
    const uint32_t heads = (uint32_t)this->num_attn_heads;
    const uint32_t q_row = dh * heads;  // == DQ, the row stride of q and o

    assert(this->num_kv_heads == 1 && "the DH_512 overlay streams a single kv head");
    assert(this->DK == dh && this->DV == dh);
    assert(heads % num_cu == 0);
    assert(q_rows > 0 && q_rows % rows_per_round == 0 && "prefill attention needs q_rows % 128 == 0");
    assert(kv_valid_rows > 0 && "a block with no visible keys cannot be attended");

    const uint32_t Q_chunks = q_rows / rows_per_round;
    // One flat extent from row 0, rounded up to whole blocks. The kernel walks exactly
    // this many blocks and masks every key from kv_valid_rows on, so the two counts have
    // to be derived from the same rounding -- hence kv_blocks rather than a second
    // expression that merely agrees today.
    const uint32_t kv_len    = (kv_valid_rows + kv_block - 1) / kv_block * kv_block;
    const uint32_t kv_blocks = kv_len / kv_block;
    // Where the V half of this cache starts, in elements.
    const uint32_t v_base = (uint32_t)(kv_elems / 2);

    assert(kv_len <= kv_rows && "the kv cache is too short for this block's tail read");
    assert(kv_blocks <= max_kv_iters && "kv stream longer than one BD sweep");

    seq->clear_cmds();

    for (uint32_t row = 2; row < 6; row++){
        for (uint32_t col = 0; col < 8; col++){
            npu_tiles tile = get_tile(row, col);
            seq->rtp_write(tile, global_image_attn_prefill_rtp.kv_chunks_address, kv_blocks);
            seq->rtp_write(tile, global_image_attn_prefill_rtp.q_chunks_address,  Q_chunks);
            seq->rtp_write(tile, global_image_attn_prefill_rtp.l_address,         kv_valid_rows);
        }
    }

    for (uint32_t head = 0; head < heads / num_cu; head++){
        for (uint32_t round = 0; round < Q_chunks; round++){
            const uint32_t bd = (round % 2) * 8;  // double buffer the descriptors

            for (uint32_t cu = 0; cu < num_cu; cu++){
                const uint32_t h = head * num_cu + cu;

                // receive o
                for (uint32_t col = 0; col < cols; col++){
                    const uint32_t o_off = h * dh
                                         + (round * rows_per_round + col * lc * 4) * q_row;
                    seq->npu_dma_memcpy_nd(
                        sizeof(bf16), o_arg,
                        S2MM, IT[cu][col],
                        (npu_bd_id)(bd + 0), it_channel_0,
                        {0, 0, 0, o_off},
                        {1, 1, 4 * lc, dh},
                        {0, 0, q_row, 1},
                        -1, 0, true
                    );
                }

                // send q, as four quarters over the two channels of the unit's outer shims
                const uint32_t q_off = h * dh + round * rows_per_round * q_row;
                const npu_tiles      q_tile[4] = {IT[cu][0], IT[cu][0], IT[cu][3], IT[cu][3]};
                const npu_it_channel q_chan[4] = {it_channel_0, it_channel_1,
                                                  it_channel_0, it_channel_1};
                for (uint32_t part = 0; part < 4; part++){
                    seq->npu_dma_memcpy_nd(
                        sizeof(bf16), q_arg,
                        MM2S, q_tile[part], (npu_bd_id)(bd + 1 + part), q_chan[part],
                        {0, 0, 0, q_off + part * lc * 4 * q_row},
                        {1, 1, 4 * lc, dh},
                        {0, 0, q_row, 1},
                        -1, 0, false
                    );
                }
            }

            // send k and v: one kv head, broadcast from a single shim to every core. Unlike
            // the causal pass the extent does not move with the round -- the mask is the
            // same for every query row, so every round streams the same keys.
            seq->npu_dma_memcpy_nd(
                sizeof(bf16), kv_arg,
                MM2S, IT[0][2], (npu_bd_id)(bd + 5), it_channel_0,
                {0, 0, 0, 0},
                {1, kv_blocks, kv_block, dh},
                {0, kv_block * dh, dh, 1},
                -1, 0, false
            );
            seq->npu_dma_memcpy_nd(
                sizeof(bf16), kv_arg,
                MM2S, IT[0][2], (npu_bd_id)(bd + 6), it_channel_1,
                {0, 0, 0, v_base},
                {1, kv_blocks, kv_block, dh},
                {0, kv_block * dh, dh, 1},
                -1, 0, false
            );

            // drain the previous round while this one is in flight
            if (round > 0){
                for (uint32_t cu = 0; cu < num_cu; cu++){
                    for (uint32_t col = 0; col < cols; col++){
                        seq->npu_dma_wait(IT[cu][col], S2MM, it_channel_0);
                    }
                }
            }
        }
        for (uint32_t cu = 0; cu < num_cu; cu++){
            for (uint32_t col = 0; col < cols; col++){
                seq->npu_dma_wait(IT[cu][col], S2MM, it_channel_0);
            }
        }
    }

    seq->cmds2seq();
}

/// \brief Prefill flash attention for a sliding window layer.
///
/// Ported from FLM_Xclbin/Gemma4_12B_QAT/attention_DH_256_prefill/host/seq_gen.hpp. Three
/// differences from that testbench, all deliberate:
///   - its `Num_of_d_in_KV_cache` generality is dropped. It exists so a kv cache split into
///     several [MAX_L][n_heads][DH] slabs can be addressed; the buffer handed over here is
///     one slab holding all SWA_NUM_KEY_VALUE_HEADS heads, so kv_chunk_offset is always 0
///     and the whole expression collapses to `kv_head * DH + kv_begin * DK`.
///   - the > 128*1024 kv branch is dropped, for the same BD reuse hazard as in
///     gen_attn_prefill_global(). A sliding round never streams more than window + 256 rows.
///   - the kv block count is taken from the kernel's own expression rather than from
///     round_up(Lq + 256, 128) - kv_begin. The two agree only while the window is a multiple
///     of 128; copying the kernel makes that a non-issue instead of an invariant to remember.
void gemma4_12b_npu_sequence::gen_attn_prefill_swa(npu_sequence* seq, const uint32_t L_begin,
                                                   const uint32_t q_rows, const uint32_t kv_rows,
                                                   const uint32_t window, const size_t kv_elems){
    // The DH_256 floorplan is four compute units of two shim columns each, and the two units
    // of a half share one kv head, so a q head pair reads the same keys.
    static constexpr npu_tiles IT[4][2] = {{IT0, IT1}, {IT2, IT3}, {IT4, IT5}, {IT6, IT7}};
    static constexpr uint32_t lc     = 16;
    static constexpr uint32_t num_cu = 4;
    static constexpr uint32_t cols   = 2;
    static constexpr uint32_t gqa    = 2;            // q heads sharing one kv head
    static constexpr uint32_t rows_per_round = lc * 8;   // 128 query rows per round
    static constexpr uint32_t kv_block       = 128;      // kv rows per BD iteration step
    static constexpr uint32_t max_kv_iters   = 1024;     // 6 bit iteration count, x kv_block

    static constexpr int o_arg  = 0;
    static constexpr int q_arg  = 1;
    static constexpr int kv_arg = 2;

    const uint32_t dh    = this->SWA_DH;
    const uint32_t heads = (uint32_t)this->num_attn_heads;
    const uint32_t q_row = dh * heads;              // == SWA_DQ, the row stride of q and o
    const uint32_t kv_row = (uint32_t)this->SWA_DK; // == dh * swa_num_kv_heads

    assert(heads % num_cu == 0);
    assert((uint32_t)this->swa_num_kv_heads * gqa == heads
           && "the DH_256 overlay hardwires two q heads per kv head");
    assert(this->SWA_DK == this->SWA_DV);
    assert(window % kv_block == 0 && "a window off a 128 boundary desynchronises the kv stream");
    assert(q_rows > 0 && q_rows % rows_per_round == 0 && "prefill attention needs q_rows % 128 == 0");

    const uint32_t Q_chunks = q_rows / rows_per_round;
    // Where the V half of the unrolled window starts, in elements.
    const uint32_t v_base = (uint32_t)(kv_elems / 2);

    seq->clear_cmds();

    for (uint32_t row = 2; row < 6; row++){
        for (uint32_t col = 0; col < 8; col++){
            npu_tiles tile = get_tile(row, col);
            seq->rtp_write(tile, swa_attn_prefill_rtp.l_begin_address,     L_begin);
            seq->rtp_write(tile, swa_attn_prefill_rtp.l_chunks_address,    Q_chunks);
            seq->rtp_write(tile, swa_attn_prefill_rtp.window_size_address, window);
        }
    }

    for (uint32_t head = 0; head < heads / num_cu; head++){
        for (uint32_t round = 0; round < Q_chunks; round++){
            const uint32_t Lq = L_begin + round * rows_per_round;
            const uint32_t kv_begin = (Lq > window) ? (Lq - window) : 0;
            // exactly the kernel's kv_blocks: it walks blocks from kv_begin and stops once
            // it has passed the round's last query row, so this count *is* the mask geometry
            const uint32_t kv_iters = (Lq + 2 * rows_per_round - 1 - kv_begin) / kv_block;

            assert(kv_begin + kv_iters * kv_block <= kv_rows
                   && "the unrolled window is too short for this round's tail read");
            assert(kv_iters <= max_kv_iters && "kv stream longer than one BD sweep");

            const uint32_t bd = (round % 2) * 8;  // double buffer the descriptors

            for (uint32_t cu = 0; cu < num_cu; cu++){
                const uint32_t h = head * num_cu + cu;

                // receive o, half the round's rows down each of the unit's two columns
                for (uint32_t col = 0; col < cols; col++){
                    const uint32_t o_off = h * dh
                                         + (round * rows_per_round + col * lc * 4) * q_row;
                    seq->npu_dma_memcpy_nd(
                        sizeof(bf16), o_arg,
                        S2MM, IT[cu][col],
                        (npu_bd_id)(bd + 0), it_channel_0,
                        {0, 0, 0, o_off},
                        {1, 1, 4 * lc, dh},
                        {0, 0, q_row, 1},
                        -1, 0, true
                    );
                }

                // send q, as two halves over the two channels of the unit's first shim
                const uint32_t q_off = h * dh + round * rows_per_round * q_row;
                const npu_it_channel q_chan[2] = {it_channel_0, it_channel_1};
                for (uint32_t part = 0; part < 2; part++){
                    seq->npu_dma_memcpy_nd(
                        sizeof(bf16), q_arg,
                        MM2S, IT[cu][0], (npu_bd_id)(bd + 1 + part), q_chan[part],
                        {0, 0, 0, q_off + part * lc * 4 * q_row},
                        {1, 1, 4 * lc, dh},
                        {0, 0, q_row, 1},
                        -1, 0, false
                    );
                }
            }

            // send k and v: one kv head per pair of units, from that half's second shim.
            // The kv head of q head h is h / gqa, and cu 2i / 2i+1 carry q heads
            // head * 4 + 2i / + 2i + 1, so the pair shares kv head head * 2 + i.
            for (uint32_t i = 0; i < num_cu / gqa; i++){
                const uint32_t kv_head = head * (num_cu / gqa) + i;
                const uint32_t k_off = kv_head * dh + kv_begin * kv_row;
                seq->npu_dma_memcpy_nd(
                    sizeof(bf16), kv_arg,
                    MM2S, IT[i * 2 + 1][1], (npu_bd_id)(bd + 3), it_channel_0,
                    {0, 0, 0, k_off},
                    {1, kv_iters, kv_block, dh},
                    {0, kv_block * kv_row, kv_row, 1},
                    -1, 0, false
                );
                seq->npu_dma_memcpy_nd(
                    sizeof(bf16), kv_arg,
                    MM2S, IT[i * 2 + 1][1], (npu_bd_id)(bd + 4), it_channel_1,
                    {0, 0, 0, k_off + v_base},
                    {1, kv_iters, kv_block, dh},
                    {0, kv_block * kv_row, kv_row, 1},
                    -1, 0, false
                );
            }

            // drain the previous round while this one is in flight
            if (round > 0){
                for (uint32_t cu = 0; cu < num_cu; cu++){
                    for (uint32_t col = 0; col < cols; col++){
                        seq->npu_dma_wait(IT[cu][col], S2MM, it_channel_0);
                    }
                }
            }
        }
        for (uint32_t cu = 0; cu < num_cu; cu++){
            for (uint32_t col = 0; col < cols; col++){
                seq->npu_dma_wait(IT[cu][col], S2MM, it_channel_0);
            }
        }
    }

    seq->cmds2seq();
}

gemma4_12b_npu_sequence::~gemma4_12b_npu_sequence() = default;

// ===========================================================================
//  The multi modal projector matmuls -- audio_image_mm
//
//  Ported from FLM_Xclbin/Gemma4_12B_QAT/audio_image_mm/host/seq_gen.hpp, the standalone
//  testbench for this overlay. Three differences from that testbench, all deliberate:
//
//    - gen_mm_seq()'s `weight_offset` parameter is dropped: it was never read, and every
//      offset this sequence needs is derived from M, K and N.
//    - the bias descriptor is double buffered by output block (BD 3 / BD 11) the same way
//      the x/y/w descriptors are double buffered by round. The testbench re-arms BD 3 on
//      every block with no intervening wait on it, which is the same reuse-without-a-wait
//      hazard that kept the > 128k branch of gen_attn_prefill_global() from being ported.
//      Two ids cost nothing -- 3 and 11 are the two slots the round parity scheme leaves
//      free -- and the alternation is the idiom already in use three lines above.
//    - the testbench's asserts on M and N become asserts on the *padded* row count the
//      projector computes, since here the row count follows the prompt rather than a
//      constant in a testbench.
// ===========================================================================

namespace {

using mmg = mm_projector_geometry;

/// \brief The body both halves share: the round loop over output column blocks x row
///        blocks, issuing w / y / x per shim column and draining the previous round.
///
/// \param IT         the four shim columns of this half
/// \param bias_elems elements of `w` that sit in front of the weight region, i.e. N when a
///                   bias is packed there and 0 otherwise
void gen_mm_projector_rounds(npu_sequence* seq, const npu_tiles IT[4],
                             const uint32_t M, const uint32_t K, const uint32_t N,
                             const uint32_t k_tile, const uint32_t bias_elems,
                             const bool send_bias){
    static constexpr int y_arg = 0;
    static constexpr int x_arg = 1;
    static constexpr int w_arg = 2;

    const uint32_t M_blocks = M / mmg::m / mmg::rows;  //!< rounds of cols * m = 256 rows
    const uint32_t N_blocks = N / mmg::n / mmg::cols;  //!< rounds of cols * n = 256 columns
    const uint32_t K_tiles  = K / k_tile;

    for (uint32_t j = 0; j < N_blocks; j++){
        if (send_bias){
            // One full w tile's worth of elements starting at this block's bias, of which
            // the core keeps the leading cols * n; the rest is over-read that lands inside
            // the weight region behind the bias, which is exactly why the two have to live
            // in one buffer. Parity by output block, see the note at the top.
            const uint32_t b_off  = j * mmg::n * mmg::cols;
            const npu_bd_id b_bd  = (npu_bd_id)(3 + (j % 2) * 8);
            for (uint32_t c = 0; c < mmg::cols; c++){
                seq->npu_dma_memcpy_nd(
                    sizeof(bf16), w_arg,
                    MM2S, IT[c], b_bd, it_channel_1,
                    {0, 0, 0, b_off + c * mmg::n},
                    {1, 1, 1, mmg::n * k_tile},
                    {0, 0, 0, 1},
                    -1, 0, false
                );
            }
        }

        for (uint32_t i = 0; i < M_blocks; i++){
            const uint32_t round = j * M_blocks + i;
            const uint32_t bd    = (round % 2) * 8;

            const uint32_t y_off = i * mmg::m * N * mmg::rows + j * mmg::n * mmg::cols;
            const uint32_t x_off = i * mmg::m * K * mmg::rows;
            const uint32_t w_off = j * mmg::n * K * mmg::cols + bias_elems;

            const npu_bd_id y_bd = (npu_bd_id)(0 + bd);
            const npu_bd_id x_bd = (npu_bd_id)(1 + bd);
            const npu_bd_id w_bd = (npu_bd_id)(2 + bd);

            for (uint32_t c = 0; c < mmg::cols; c++){
                // w: output column band c of this block, [n][K] read one k tile at a time
                seq->npu_dma_memcpy_nd(
                    sizeof(bf16), w_arg,
                    MM2S, IT[c], w_bd, it_channel_1,
                    {0, 0, 0, w_off + c * mmg::n * K},
                    {1, K_tiles, mmg::n, k_tile},
                    {0, k_tile, K, 1},
                    -1, 0, false
                );

                // y: row band c of this block, all cols * n output columns of it
                seq->npu_dma_memcpy_nd(
                    sizeof(bf16), y_arg,
                    S2MM, IT[c], y_bd, it_channel_0,
                    {0, 0, 0, y_off + c * mmg::m * N},
                    {1, 1, mmg::m, mmg::n * mmg::cols},
                    {0, 0, N, 1},
                    -1, 0, true
                );

                // x: the same row band, k tile by k tile
                seq->npu_dma_memcpy_nd(
                    sizeof(bf16), x_arg,
                    MM2S, IT[c], x_bd, it_channel_0,
                    {0, 0, 0, x_off + c * mmg::m * K},
                    {1, K_tiles, mmg::m, k_tile},
                    {0, k_tile, K, 1},
                    -1, 0, false
                );
            }

            // drain the previous round while this one is in flight
            if (round > 0){
                for (uint32_t c = 0; c < mmg::cols; c++){
                    seq->npu_dma_wait(IT[c], S2MM, it_channel_0);
                }
            }
        }
    }

    for (uint32_t c = 0; c < mmg::cols; c++){
        seq->npu_dma_wait(IT[c], S2MM, it_channel_0);
    }
}

} // namespace

void gen_mm_projector_plain(npu_sequence* seq, const uint32_t M, const uint32_t K,
                            const uint32_t N){
    static constexpr npu_tiles IT[4] = {IT0, IT1, IT2, IT3};

    // mm.cc walks K with `NPU_K_TILE / k_slice` as a compile time bound and no RTP, so this
    // half is not a K-parametric engine: it does 640 or it does nothing. Audio is the only
    // caller and audio_embed_dim is 640.
    assert(K == mmg::k_plain && "the plain half's K loop is static, see kernel/typedef.hpp");
    assert(M > 0 && M % mmg::M_ALIGN == 0 && "projector rows must be padded to M_ALIGN");
    assert(N > 0 && N % mmg::N_ALIGN == 0);

    seq->clear_cmds();
    gen_mm_projector_rounds(seq, IT, M, K, N, mmg::k_plain, 0, false);
    seq->cmds2seq();
}

void gen_mm_projector_bias(npu_sequence* seq, const uint32_t M, const uint32_t K,
                           const uint32_t N, const bool add_bias){
    static constexpr npu_tiles IT[4] = {IT4, IT5, IT6, IT7};

    assert(M > 0 && M % mmg::M_ALIGN == 0 && "projector rows must be padded to M_ALIGN");
    assert(N > 0 && N % mmg::N_ALIGN == 0);
    assert(K > 0 && K % mmg::k_bias == 0 && "K must be a whole number of bias half k tiles");

    const uint32_t M_blocks = M / mmg::m / mmg::rows;
    const uint32_t N_blocks = N / mmg::n / mmg::cols;
    const uint32_t K_tiles  = K / mmg::k_bias;

    seq->clear_cmds();

    // Unlike the plain half, one mm_bias() invocation covers the whole block grid: the core
    // reads its loop bounds from the RTPs and only then releases into the DMA stream. They
    // have to be republished on every run, because the row count moves with the prompt.
    for (uint32_t row = 2; row < 2 + mmg::rows; row++){
        for (uint32_t col = 4; col < 4 + mmg::cols; col++){
            const npu_tiles tile = get_tile(row, col);
            seq->rtp_write(tile, mm_projector_rtp.rtp_base_address +  0, N_blocks);
            seq->rtp_write(tile, mm_projector_rtp.rtp_base_address +  4, add_bias ? 1u : 0u);
            seq->rtp_write(tile, mm_projector_rtp.rtp_base_address +  8, M_blocks);
            seq->rtp_write(tile, mm_projector_rtp.rtp_base_address + 12, K_tiles);
            seq->rtp_write(tile, mm_projector_rtp.sync_lock_address, 1);
        }
    }

    gen_mm_projector_rounds(seq, IT, M, K, N, mmg::k_bias, add_bias ? N : 0, add_bias);
    seq->cmds2seq();
}