#include "gemma4e_npu_sequence.hpp"

gemma4e_npu_sequence::gemma4e_npu_sequence(gemma4e_seq_gen_parameters_t params, uint32_t MAX_L){
    D = params.D;
    DH = params.DH;
    DQ = params.DQ;
    DK = params.DK;
    DV = params.DV;
    SWA_DH = params.SWA_DH;
    SWA_DQ = params.SWA_DQ;
    SWA_DK = params.SWA_DK;
    SWA_DV = params.SWA_DV;
    PLI_D = params.PLI_D;
    num_attn_heads = params.NUM_ATTENTION_HEADS;
    num_kv_heads = params.NUM_KEY_VALUE_HEADS;
    num_kv_per_round = num_kv_heads;
    INTERMEDIATE_SIZE = params.INTERMEDIATE_SIZE;
    VOCAB_SIZE = params.VOCAB_SIZE_PADDED;
    SLIDING_LENGTH = params.SLIDING_WINDOW_SIZE;
    enable_double_wide_mlp = params.enable_double_wide_mlp;

    this->MAX_L = MAX_L;

    if (INTERMEDIATE_SIZE == 6144){ // e2b
        rtp_addresses = e2b_rtp_addresses;
    }
    else if (INTERMEDIATE_SIZE == 10240){
        rtp_addresses = e4b_rtp_addresses;
    }
    else{
        std::cerr << "Unsupported intermediate size: " << INTERMEDIATE_SIZE << std::endl;
        throw std::runtime_error("SEQ Unsupported intermediate size");
    }

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
    std::cout << "\tPLI_D: " << PLI_D << std::endl;
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
    std::cout << "\tproj_skip_address: " << rtp_addresses.proj_skip_address << std::endl;
    std::cout << "\trms_swa_address: " << rtp_addresses.rms_swa_address << std::endl;
    std::cout << "\trms_skip_address: " << rtp_addresses.rms_skip_address << std::endl;
    std::cout << "\trope_skip_kv_address: " << rtp_addresses.rope_skip_kv_address << std::endl;
    std::cout << "\tswa_rope_skip_kv_address: " << rtp_addresses.swa_rope_skip_kv_address << std::endl;
    )
}

void gemma4e_npu_sequence::set_max_length(const uint32_t MAX_L){
    this->MAX_L = MAX_L;
}

void gemma4e_npu_sequence::gen_layer_seq(npu_sequence* seq, const uint32_t L, gemma4e_layer_type_t layer_type){
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
    std::cout << "\t_ QKV_OFFSET: " << weight_elem_offset(layer_weights(layer_type).attn_qkv) << std::endl;
    std::cout << "\t_ O_OFFSET: " << weight_elem_offset(layer_weights(layer_type).attn_output) << std::endl;
    std::cout << "\t_ UP_OFFSET: " << weight_elem_offset(layer_weights(layer_type).ffn_up_gate) << std::endl;
    std::cout << "\t_ DOWN_OFFSET: " << weight_elem_offset(layer_weights(layer_type).ffn_down) << std::endl;
    std::cout << "\t_ PLI_DOWN_OFFSET: " << weight_elem_offset(layer_weights(layer_type).pli_down_proj) << std::endl;
    std::cout << "\t_ PLI_GATE_OFFSET: " << weight_elem_offset(layer_weights(layer_type).pli_gate_proj) << std::endl;
    std::cout << "\t_ PLI_UP_OFFSET: " << weight_elem_offset(layer_weights(layer_type).pli_up_proj) << std::endl;
    std::cout << "\t_ IS_SWA: " << is_swa_layer(layer_type) << std::endl;
    std::cout << "\t_ IS_SKIP: " << is_skip_layer(layer_type) << std::endl;
    )
    int proj_rtp_lock_id = 6;
    int rms_rtp_lock_id = 6;
    int glu_rtp_lock_id = 6;
    seq->clear_cmds();
    int L_local = L;
    if (is_swa_layer(layer_type)) {
        if (L > SLIDING_LENGTH){
            L_local = SLIDING_LENGTH;
        }
    }

    for (int i = 0; i < 16; i++){
        seq->rtp_write(proj_tiles[i], rtp_addresses.proj_swa_address, is_swa_layer(layer_type) ? 1 : 0);
        seq->rtp_write(proj_tiles[i], rtp_addresses.proj_skip_address, is_skip_layer(layer_type) ? 1 : 0);
        seq->rtp_write(proj_tiles[i], CT_lock_address_base + 16 * proj_rtp_lock_id, 1); // set lock to 1
    }
    seq->rtp_write(attn_qk_tile, rtp_addresses.l_qk_address, L_local);
    seq->rtp_write(attn_kv_tile, rtp_addresses.l_kv_address, L_local);
    seq->rtp_write(swa_attn_qk_tile, rtp_addresses.swa_l_qk_address, L_local);
    seq->rtp_write(swa_attn_kv_tile, rtp_addresses.swa_l_kv_address, L_local);
    seq->rtp_write(rms_tile, rtp_addresses.rms_swa_address, is_swa_layer(layer_type) ? 1 : 0);
    seq->rtp_write(rms_tile, rtp_addresses.rms_skip_address, is_skip_layer(layer_type) ? 1 : 0);
    seq->rtp_write(rope_ct, rtp_addresses.rope_skip_kv_address, is_skip_layer(layer_type) ? 1 : 0);
    seq->rtp_write(swa_rope_ct, rtp_addresses.swa_rope_skip_kv_address, is_skip_layer(layer_type) ? 1 : 0);
    seq->rtp_write(glu_tile, rtp_addresses.glu_skip_address, (is_skip_layer(layer_type) && enable_double_wide_mlp) ? 1 : 0);
    seq->rtp_write(rms_tile, CT_lock_address_base + 16 * rms_rtp_lock_id, 1); // set lock to 1
    seq->rtp_write(glu_tile, CT_lock_address_base + 16 * glu_rtp_lock_id, 1); // set lock to 1

    _send_hidden_states(seq);
    _send_rms_weights(seq);
    _send_rope_rms_weights(seq, layer_type);

    // receive y

    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        x_arg_id,
        S2MM,
        xr_tile,
        bd_15,
        it_channel_0,
        {0, 0, 0, 0},
        {1, 1, 1, (uint32_t)D},
        {0, 0, 0, 1},
        -1, 0, true
    );
    gemma4e_layer_weight_def& W = layer_weights(layer_type);
    // attn_qkv already spans q alone on skip layers, which carry no k/v projection.
    _move_weights(seq, W.attn_qkv);
    if (!is_skip_layer(layer_type)){
        _receive_kv_cache(seq, L, layer_type);
    }

    _gen_pli_path_seq(seq, layer_type);
    _move_kv_cache(seq, L, layer_type);
    _move_weights(seq, W.attn_output);
    _move_weights(seq, W.ffn_up_gate);
    _move_weights(seq, W.ffn_down);

    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        proj_arg_id,
        MM2S,
        IT5,
        bd_14,
        it_channel_0,
        {0, 0, 0, weight_elem_offset(layer_weights(layer_type).pli_gate_proj)},
        {1, 1, 1, (uint32_t)(PLI_D * D)},
        {0, 0, 0, 1},
        -1, 0, true, aggressive_cache
    );

    seq->npu_dma_wait(
        IT5,
        MM2S,
        it_channel_0
    );

    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        proj_arg_id,
        MM2S,
        IT5,
        bd_15,
        it_channel_1,
        {0, 0, 0, weight_elem_offset(layer_weights(layer_type).pli_up_proj)},
        {1, 1, 1, (uint32_t)(PLI_D * D)},
        {0, 0, 0, 1},
        -1, 0, true, aggressive_cache
    );

    seq->npu_dma_wait(
        IT5,
        MM2S,
        it_channel_1
    );
    // wait for receiving y
    seq->npu_dma_wait(
        xr_tile,
        S2MM,
        it_channel_0
    );
    seq->cmds2seq();
}

void gemma4e_npu_sequence::gen_lm_head_seq(npu_sequence* seq, float final_scale){
    static constexpr int y_arg_id = 0;
    static constexpr int w_arg_id = 1;
    static constexpr int x_arg_id = 2;
    static constexpr int M_PER_ROUND = 32 * 32;
    static constexpr int M    = 32;
    static constexpr int M_PER_COL    = M * 4;
    static constexpr int COLS = 8;
    static npu_tiles ITs[] = {IT0, IT1, IT2, IT3, IT4, IT5, IT6, IT7};
    const uint32_t final_scale_address = this->rtp_addresses.lm_head_final_tune_address;
    assert(VOCAB_SIZE % M_PER_ROUND == 0);
    int rounds = VOCAB_SIZE / M_PER_ROUND;
    size_t TOTAL_W_SIZE = size_t(D) * size_t(VOCAB_SIZE) * 5 / 8 / 2;
    size_t WEIGHTS_PER_IT = TOTAL_W_SIZE / std::size(ITs);
    size_t W_PER_COL = (M_PER_COL * D * 5 / 8 / 2);
    size_t W_PER_ROUND = W_PER_COL * COLS;

    size_t w_offset = WEIGHTS_PER_IT;
    size_t y_offset = VOCAB_SIZE / std::size(ITs);
    uint32_t scale_int_view = *((uint32_t*)(&final_scale));
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
        -1, 0, false, aggressive_cache
    );

    for (int r = 0; r < rounds; r++) {
        int bd_offset = (r % 2) * 8;
        npu_bd_id bd_y = npu_bd_id(bd_1 + bd_offset);
        npu_bd_id bd_w = npu_bd_id(bd_2 + bd_offset);
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
            seq->npu_dma_memcpy_nd(
                sizeof(bf16),
                w_arg_id,
                MM2S,
                ITs[col],
                bd_w,
                it_channel_1,
                {0, 0, 0, (uint32_t)(w_col_offset)},
                {1, 1, 1, (uint32_t)W_PER_COL},
                {0, 0, 0, 1},
                -1, 0, false, aggressive_cache
            );
            if (r > 0){
                seq->npu_dma_wait(
                    ITs[col],
                    S2MM,
                    it_channel_0
                );
            }
        }
    }
    for (size_t col = 0; col < std::size(ITs); col++){
        seq->npu_dma_wait(
            ITs[col],
            S2MM,
            it_channel_0
        );
    }
    seq->cmds2seq();
}

void gemma4e_npu_sequence::_send_hidden_states(npu_sequence* seq){
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
        0, 0, true, aggressive_cache
    );
    seq->npu_dma_wait(
        xr_tile,
        MM2S,
        it_channel_0
    );
}

void gemma4e_npu_sequence::_send_rms_weights(npu_sequence* seq){
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
        0, 0, true, aggressive_cache
    );

    seq->npu_dma_wait(
        xr_tile,
        MM2S,
        it_channel_1
    );
}

void gemma4e_npu_sequence::_send_rope_rms_weights(npu_sequence* seq, gemma4e_layer_type_t layer_type){
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

void gemma4e_npu_sequence::_receive_kv_cache(npu_sequence* seq, const int L, gemma4e_layer_type_t layer_type){
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

void gemma4e_npu_sequence::_move_kv_cache(npu_sequence* seq, const size_t L, gemma4e_layer_type_t layer_type){
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

    if (is_swa_layer(layer_type)) {
        if (L % SLIDING_LENGTH == 0){ // corner, from 0 to end
            // move the oldest block to the new block
            uint32_t data2move = SLIDING_LENGTH * _DK;
            seq->npu_dma_memcpy_nd(
                sizeof(bf16),
                kv_cache_arg_id,
                MM2S,
                attn_tile,
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
                attn_tile,
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
                attn_tile,
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
                attn_tile,
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
                attn_tile,
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
                attn_tile,
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
    // fallback path, single phase, from zero to L
    const int L_padded = (L + L_CHUNK - 1) / L_CHUNK * L_CHUNK;
    const uint32_t data2move = L_padded * _DK;
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        kv_cache_arg_id,
        MM2S,
        attn_tile,
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
        attn_tile,
        bd_9,
        it_channel_1,
        {0, 0, 0, (uint32_t)(v_offset)},
        {1, 1, 1, data2move},
        {0, 0, 0, 1},
        pkt_id, 0, true, aggressive_cache
    );

    seq->npu_dma_wait(
        attn_tile,
        MM2S,
        it_channel_0
    );
     seq->npu_dma_wait(
        attn_tile,
        MM2S,
        it_channel_1
    );
}

/// \brief Stream one quantized weight from DDR into the mvm cores.
/// \param seq the sequence to append the DMAs to
/// \param weight the weight to move; its shape is {input dim, output dim} and its
///        offset is a byte offset into the layer's projection buffer
/// \note The proj port addresses DDR in bf16 elements, so both the descriptor's
///       byte offset and the block size are halved here. Nothing about this
///       function is tied to a particular 4-bit layout any more: switch
///       gemma4e_desc::PROJ_DTYPE and the block size follows.
void gemma4e_npu_sequence::_move_weights(npu_sequence* seq, weight_desc_t& weight){
    assert(weight.added && "weight must be placed in its buffer before it can be moved");
    assert(is_quantize(weight.dtype) && "_move_weights moves quantized projections only");
    const int m = QXNX_ROW_BLOCK_SIZE;
    const int k = QXNX_COL_BLOCK_SIZE;
    const uint32_t columns = 4;
    const size_t Din  = (size_t)weight.shape[0];
    const size_t Dout = (size_t)weight.shape[1];
    const uint32_t a_block_size = (uint32_t)(get_quantization_byte_size((size_t)m * k, weight.dtype) / sizeof(bf16));
    const uint32_t w_offset = weight_elem_offset(weight);
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

void gemma4e_npu_sequence::_gen_pli_path_seq(npu_sequence* seq, gemma4e_layer_type_t layer_type){
    LAYER_SPECIFIC_DIM(layer_type)
    DEBUG_BLOCK(2,
    header_print_g("info", "Generating PLI path sequence for layer type " + std::to_string(layer_type));
    std::cout << "\t_ D: "  << D << std::endl;
    std::cout << "\t_ DH: " << _DH << std::endl;
    std::cout << "\t_ PLI_D: " << PLI_D << std::endl;
    std::cout << "\t_ pli_down_proj_offset: " << weight_elem_offset(layer_weights(layer_type).pli_down_proj) << std::endl;
    std::cout << "\t_ pli_gate_proj_offset: " << weight_elem_offset(layer_weights(layer_type).pli_gate_proj) << std::endl;
    std::cout << "\t_ pli_up_proj_offset: " << weight_elem_offset(layer_weights(layer_type).pli_up_proj) << std::endl;
    )
    // move per layer input weights
    // send x, x, w
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        rope_rms_arg_id,
        MM2S,
        IT4,
        bd_11,
        it_channel_0,
        {0, 0, 0, (uint32_t)(_DH * 3)},
        {1, 1, 1, (uint32_t)(PLI_D * 2 + D + MIN_BF16_PAD)},
        {0, 0, 0, 1},
        -1, 0, true, no_cache
    );
    seq->npu_dma_wait(
        IT4,
        MM2S,
        it_channel_0
    );

    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        x_arg_id,
        MM2S,
        IT4,
        bd_12,
        it_channel_0,
        {0, 0, 0, (uint32_t)D * 2},
        {1, 1, 1, (uint32_t)D},
        {0, 0, 0, 1},
        -1, 0, false, aggressive_cache
    );

    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        proj_arg_id,
        MM2S,
        IT4,
        bd_13,
        it_channel_1,
        {0, 0, 0, weight_elem_offset(layer_weights(layer_type).pli_down_proj)},
        {1, 1, 1, (uint32_t)(PLI_D * D)},
        {0, 0, 0, 1},
        -1, 0, true, aggressive_cache
    );

    seq->npu_dma_wait(
        IT4,
        MM2S,
        it_channel_1
    );
}

/// \brief Build the sequence that dequantizes one weight into a bf16 buffer.
/// \param seq_ptr the sequence to fill
/// \param weight the weight to dequantize; shape is {input dim, output dim} and
///        offset is a byte offset into the layer's projection buffer
/// \param output_mode which half of an interleaved up/gate region to emit, or
///        NORMAL_DEQUANT for a weight that is not interleaved
void gemma4e_npu_sequence::generate_dequant_seq(npu_sequence* seq_ptr, weight_desc_t& weight, dequant_output_mode_t output_mode){
    assert(weight.added && "weight must be placed in its buffer before it can be dequantized");
    assert(is_quantize(weight.dtype) && "only quantized weights need dequantizing");
    const u32 D_in  = (u32)weight.shape[0];
    const u32 D_out = (u32)weight.shape[1];
    const u32 weight_offset = (u32)weight.offset;
    static constexpr npu_tiles IT[] = {IT0, IT1, IT2, IT3, IT4, IT5, IT6, IT7};

    static constexpr u32 total_cols = 8;
    static constexpr u32 total_rows = 4;

    static constexpr int w_out_arg_idx = 0;
    static constexpr int qw_in_arg_idx = 1;

    static constexpr int m_tile_q4 = 32;
    static constexpr int k_tile_q4 = 256;

    const uint32_t block_size_in_byte_q4 = (uint32_t)get_quantization_byte_size((size_t)m_tile_q4 * k_tile_q4, weight.dtype);

    static constexpr int m_tile_q8 = 32;
    static constexpr int k_tile_q8 = 128;
    static constexpr uint32_t block_size_in_byte_q8 = m_tile_q8 * k_tile_q8 * (10)/8;

    static constexpr int quant_block_col_stride = 2;

    static constexpr int desired_k_dequant = 512;
    static constexpr int desired_m_dequant = 128;

    static constexpr int glu_slice = 1024;
    static constexpr int gate_up_m_interleave_size = glu_slice / 2;
    if (D_in % k_tile_q4 != 0) {
        std::cerr << "D_in % k_tile_q4 != 0" << std::endl;
        exit(1);
    }
    int bd_wait_counter[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    // although each data block is in mxk block, but the data block could be reorder in col-stride on block view
    /*
        For example, quant_block_col_stride = 2 means

        //This is the logical view of the data block, each block of m_tile_q4 x k_tile_q4
        [block0, block1, ...... blockD,
        blockD+1, blockD+2, ......
        ]

        But in memory order, the data block is arrange as block0, blockD+1, block1, blockD+2 ....

    */

    if(D_in % desired_k_dequant != 0){
        std::cerr << "D_in % desired_k_dequant != 0" << std::endl;
        exit(1);
    }

    const uint32_t blocks_per_row = D_in / k_tile_q4;

    if(D_out % desired_m_dequant != 0 ){
        std::cerr << "D_out % desired_m_dequant != 0" << std::endl;
        exit(1);
    }

    const int quant_in_per_column =  (desired_m_dequant / m_tile_q4) * blocks_per_row * block_size_in_byte_q4;
    const int total_column_rounds = D_out / (desired_m_dequant);

    const int row_per_round = desired_m_dequant * total_cols;
    // down rounds, go though D_out
    const int down_rounds = (D_out + row_per_round - 1) / row_per_round;

    npu_sequence& seq = *seq_ptr;
    seq.clear_cmds();

    for(int row = 0; row < 4; row++){
        for (int col = 0; col < 8; col++){
            npu_tiles tile = get_tile(row + 2, col);
            seq_ptr->rtp_write(tile, dequant_rtp_address, 0);
        }
    }
    uint32_t input_offset = weight_offset;

    if(output_mode == dequant_output_mode_t::GATE_MATRIX){
        input_offset += (gate_up_m_interleave_size / m_tile_q4) * blocks_per_row * block_size_in_byte_q4;
    }
    uint32_t gate_up_interleave_counter= 0;

    // first, the dequant of down
    for(int i = 0; i < down_rounds; i++){
        for(int col = 0; col < 8; col++){
            uint32_t bd_offset = (i % 2) * 8;
            uint32_t round_offset = i * 8 + col;
            if(round_offset < total_column_rounds){
                seq.npu_dma_memcpy_nd(
                    sizeof(char),
                    qw_in_arg_idx,
                    MM2S,
                    IT[col],
                    (npu_bd_id)(0+bd_offset),
                    it_channel_0,
                    {0, 0, 0, input_offset},
                    //NOTE: this for now only work if desired_m_dequant == quant_block_col_stride*m_tile_q4
                    {
                        blocks_per_row,
                        (desired_m_dequant / m_tile_q4) / quant_block_col_stride,
                        quant_block_col_stride * block_size_in_byte_q4 / 512,
                        512
                    },
                    {
                        quant_block_col_stride * block_size_in_byte_q4,
                        quant_block_col_stride * block_size_in_byte_q4 * blocks_per_row,
                        512,
                        1
                    },
                    -1 ,0, false
                );

                if(output_mode == dequant_output_mode_t::NORMAL_DEQUANT){
                    input_offset += quant_in_per_column;
                }
                else{
                    gate_up_interleave_counter++;
                    input_offset += quant_in_per_column;
                    if(gate_up_interleave_counter == (gate_up_m_interleave_size / desired_m_dequant) ){
                        gate_up_interleave_counter = 0;
                        input_offset += (gate_up_m_interleave_size / m_tile_q4) * blocks_per_row * block_size_in_byte_q4;
                    }
                }

                // Each port receive 2*Q4NX_ROWx D_Q4NX_BLOCK_PER_ROW*Q4NX_COL
                uint32_t output_offset_0 = round_offset * desired_m_dequant  * D_in;

                seq.npu_dma_memcpy_nd(
                    sizeof(uint16_t),//bf16 outpout
                    w_out_arg_idx,
                    S2MM,
                    IT[col],
                    (npu_bd_id)(1+bd_offset),
                    it_channel_0,
                    {0, 0, 0, output_offset_0},
                    {
                        (uint32_t)D_in/desired_k_dequant,
                        desired_k_dequant/k_tile_q4,
                        desired_m_dequant,
                        k_tile_q4
                    },
                    {
                        desired_m_dequant * desired_k_dequant,
                        k_tile_q4,
                        desired_k_dequant,
                        1
                    },
                    -1, 0, true,
                    aggressive_cache
                );
                bd_wait_counter[col]++;
            }
        }
        // note: for now
        for(int col = 0; col < 8; col++){
            if(bd_wait_counter[col] == 2){
                seq.npu_dma_wait(IT[col], S2MM, it_channel_0);
                bd_wait_counter[col]--;
            }
        }
    }

    for(int col = 0; col < 8; col++){
        while(bd_wait_counter[col] != 0){
            seq.npu_dma_wait(IT[col], S2MM, it_channel_0);
            bd_wait_counter[col]--;
        }
    }
    seq.cmds2seq();
}

void gemma4e_npu_sequence::gen_mha_engine_seq(
    npu_sequence* seq,
    const uint32_t L_begin,
    const uint32_t L_end
){
    const int lc = 8; // local chunk size, each CU works on lc*8 of l at a time. This is determined by the hardware design.

    npu_tiles IT[2][4] = {{IT0, IT1, IT2, IT3}, {IT4, IT5, IT6, IT7}};
    assert(L_begin % (lc * 16) == 0);
    assert(L_end % (lc * 16) == 0);
    int using_window_size = L_end;
    const int Heads = num_attn_heads;
    const int GQA   = num_attn_heads / num_kv_heads;
    const int Num_of_d_in_KV_cache = num_kv_heads;
    const int DH = this->DH;
    const uint32_t KV_CACHE_SIZE = (DK + DV) * MAX_L;
    seq->clear_cmds();

    const int l_begin_mha_address = 59520;
    const int l_end_mha_address = 11520;
    const int window_size_address = 59552;
    for (int row = 2; row < 6; row++){
        for (int col = 0; col < 8; col++){
            npu_tiles tile = get_tile(row, col);
            seq->rtp_write(tile, l_begin_mha_address, L_begin);
            seq->rtp_write(tile, l_end_mha_address, L_end);
            seq->rtp_write(tile, window_size_address, using_window_size);
        }
    }
    int all_data_size = L_end - L_begin;
    const int data_per_round = lc * 16;
    const int down_rounds = (all_data_size + data_per_round - 1) / data_per_round;
    int num_cu = 2;
    for (int head = 0; head < Heads / num_cu; head++){
        for (int round = 0; round < down_rounds; round++){
            int Lq_current = L_begin + round * data_per_round;
            int kv_begin = ((Lq_current - using_window_size) > 0) ? (Lq_current - using_window_size) : 0;
            int kv_length = Lq_current - kv_begin + data_per_round;

            int bd_offset = (round % 2) * 8;
            for (int cu = 0; cu < num_cu; cu++){
                int head_offset = head * num_cu + cu;
                for (int col = 0; col < 4; col++){
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
                    {0, 0, 0, (uint32_t)(q_offset + lc * 4 * DH * Heads)},
                    {1, 1, (uint32_t)lc * 4, (uint32_t)(DH)},
                    {0, 0, (uint32_t)DH * Heads, 1},
                    -1, 0, false
                );
                seq->npu_dma_memcpy_nd(
                    2, 1,
                    MM2S, IT[cu][3],
                    (npu_bd_id)(bd_offset + 3), it_channel_0,
                    {0, 0, 0, (uint32_t)(q_offset + lc * 8 * DH * Heads)},
                    {1, 1, (uint32_t)lc * 4, (uint32_t)(DH)},
                    {0, 0, (uint32_t)DH * Heads, 1},
                    -1, 0, false
                );
                seq->npu_dma_memcpy_nd(
                    2, 1,
                    MM2S, IT[cu][3],
                    (npu_bd_id)(bd_offset + 4), it_channel_1,
                    {0, 0, 0, (uint32_t)(q_offset + lc * 12 * DH * Heads)},
                    {1, 1, (uint32_t)lc * 4, (uint32_t)(DH)},
                    {0, 0, (uint32_t)DH * Heads, 1},
                    -1, 0, false
                );
            } // cu
            int kv_head_offset = head / (GQA / num_cu);
            int kv_chunk_offset = kv_head_offset / Num_of_d_in_KV_cache;
            int kv_head_offset_in_chunk = kv_head_offset % Num_of_d_in_KV_cache;
            size_t k_offset;
            size_t v_offset;
            if (kv_length <= 128 * 1024){
                k_offset = kv_chunk_offset * MAX_L * DH * Num_of_d_in_KV_cache + kv_head_offset_in_chunk * DH + kv_begin * DH * Num_of_d_in_KV_cache;
                seq->npu_dma_memcpy_nd(
                    2, 2,
                    MM2S, IT[0][2],
                    (npu_bd_id)(bd_offset + 5), it_channel_0,
                    {0, 0, 0, (uint32_t)k_offset},
                    {1, (uint32_t)kv_length / 128, (uint32_t)128, (uint32_t)(DH)},
                    {0, (uint32_t)128 * DH * Num_of_d_in_KV_cache, (uint32_t)DH * Num_of_d_in_KV_cache, 1},
                    -1, 0, false
                );
                v_offset = k_offset + KV_CACHE_SIZE / 2;
                seq->npu_dma_memcpy_nd(
                    2, 2,
                    MM2S, IT[0][2],
                    (npu_bd_id)(bd_offset + 6), it_channel_1,
                    {0, 0, 0, (uint32_t)v_offset},
                    {1, (uint32_t)kv_length / 128, (uint32_t)128, (uint32_t)(DH)},
                    {0, (uint32_t)128 * DH * Num_of_d_in_KV_cache, (uint32_t)DH * Num_of_d_in_KV_cache, 1},
                    -1, 0, false
                );
            }
            else{
                k_offset = kv_chunk_offset * MAX_L * DH * Num_of_d_in_KV_cache + kv_head_offset_in_chunk * DH + kv_begin * DH * Num_of_d_in_KV_cache;
                seq->npu_dma_memcpy_nd(
                    2, 2,
                    MM2S, IT[0][2],
                    (npu_bd_id)(bd_offset + 5), it_channel_0,
                    {0, 0, 0, (uint32_t)k_offset},
                    {1, (uint32_t)1024, (uint32_t)128, (uint32_t)(DH)},
                    {0, (uint32_t)128 * DH * Num_of_d_in_KV_cache, (uint32_t)DH * Num_of_d_in_KV_cache, 1},
                    -1, 0, false
                );
                v_offset = k_offset + KV_CACHE_SIZE / 2;
                seq->npu_dma_memcpy_nd(
                    2, 2,
                    MM2S, IT[0][2],
                    (npu_bd_id)(bd_offset + 6), it_channel_1,
                    {0, 0, 0, (uint32_t)v_offset},
                    {1, (uint32_t)1024, (uint32_t)128, (uint32_t)(DH)},
                    {0, (uint32_t)128 * DH * Num_of_d_in_KV_cache, (uint32_t)DH * Num_of_d_in_KV_cache, 1},
                    -1, 0, false
                );
                int remaining_kv_length = kv_length - 1024 * 128;
                k_offset = kv_chunk_offset * MAX_L * DH * Num_of_d_in_KV_cache + kv_head_offset_in_chunk * DH + kv_begin * DH * Num_of_d_in_KV_cache + 1024 * 128 * DH * Num_of_d_in_KV_cache;
                seq->npu_dma_memcpy_nd(
                    2, 2,
                    MM2S, IT[0][2],
                    (npu_bd_id)(bd_offset + 5), it_channel_0,
                    {0, 0, 0, (uint32_t)k_offset},
                    {1, (uint32_t)remaining_kv_length / 128, (uint32_t)128, (uint32_t)(DH)},
                    {0, (uint32_t)128 * DH * Num_of_d_in_KV_cache, (uint32_t)DH * Num_of_d_in_KV_cache, 1},
                    -1, 0, false
                );
                v_offset = k_offset + KV_CACHE_SIZE / 2 + 1024 * 128 * DH * Num_of_d_in_KV_cache;
                seq->npu_dma_memcpy_nd(
                    2, 2,
                    MM2S, IT[0][2],
                    (npu_bd_id)(bd_offset + 6), it_channel_1,
                    {0, 0, 0, (uint32_t)v_offset},
                    {1, (uint32_t)remaining_kv_length / 128, (uint32_t)128, (uint32_t)(DH)},
                    {0, (uint32_t)128 * DH * Num_of_d_in_KV_cache, (uint32_t)DH * Num_of_d_in_KV_cache, 1},
                    -1, 0, false
                );
            }
            if (round > 0){
                for (int cu = 0; cu < num_cu; cu++){
                    for (int col = 0; col < 4; col++){
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
            for (int col = 0; col < 4; col++){
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

void gemma4e_npu_sequence::gen_swa_engine_seq(
    npu_sequence* seq,
    const uint32_t L_begin,
    const uint32_t L_end
){
    const int lc = 16; // local chunk size, each CU works on lc*8 of l at a time. This is determined by the hardware design.
    npu_tiles IT[4][2] = {{IT0, IT1}, {IT2, IT3}, {IT4, IT5}, {IT6, IT7}};
    assert(L_begin % (lc * 8) == 0);
    assert(L_end % (lc * 8) == 0);
    int using_window_size = SLIDING_LENGTH;
    const int Heads = num_attn_heads;
    const int GQA   = num_attn_heads / num_kv_heads;
    const int Num_of_d_in_KV_cache = num_kv_heads;
    const int DH = this->SWA_DH;
    const uint32_t KV_CACHE_SIZE = (SWA_DK + SWA_DV) * MAX_L;
    seq->clear_cmds();

    const int l_begin_mha_address = 61568;
    const int l_end_mha_address = 11904;
    const int window_size_address = 61600;
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

    int num_cu = 4;
    for (int head = 0; head < Heads / num_cu; head++){
        for (int round = 0; round < down_rounds; round++){
            int Lq_current = L_begin + round * data_per_round;
            int kv_begin = ((Lq_current - using_window_size) > 0) ? (Lq_current - using_window_size) : 0;
            int kv_length = Lq_current - kv_begin + data_per_round;

            int bd_offset = (round % 2) * 8;

            for (int cu = 0; cu < num_cu; cu++){
                int head_offset = head * num_cu + cu;

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
                    {0, 0, 0, (uint32_t)(q_offset + lc * 4 * DH * Heads)},
                    {1, 1, (uint32_t)lc * 4, (uint32_t)(DH)},
                    {0, 0, (uint32_t)DH * Heads, 1},
                    -1, 0, false
                );
            } // cu

            int kv_head_offset = head / (GQA / num_cu);
            int kv_chunk_offset = kv_head_offset / Num_of_d_in_KV_cache;
            int kv_head_offset_in_chunk = kv_head_offset % Num_of_d_in_KV_cache;

            size_t k_offset;
            size_t v_offset;
            if (kv_length <= 128 * 1024){
                k_offset = kv_chunk_offset * MAX_L * DH * Num_of_d_in_KV_cache + kv_head_offset_in_chunk * DH + kv_begin * DH * Num_of_d_in_KV_cache;
                seq->npu_dma_memcpy_nd(
                    2, 2,
                    MM2S, IT[1][1],
                    (npu_bd_id)(bd_offset + 3), it_channel_0,
                    {0, 0, 0, (uint32_t)k_offset},
                    {1, (uint32_t)kv_length / 128, (uint32_t)128, (uint32_t)(DH)},
                    {0, (uint32_t)128 * DH * Num_of_d_in_KV_cache, (uint32_t)DH * Num_of_d_in_KV_cache, 1},
                    -1, 0, false
                );
                v_offset = kv_chunk_offset * MAX_L * DH * Num_of_d_in_KV_cache + kv_head_offset_in_chunk * DH + kv_begin * DH * Num_of_d_in_KV_cache + KV_CACHE_SIZE / 2;
                seq->npu_dma_memcpy_nd(
                    2, 2,
                    MM2S, IT[1][1],
                    (npu_bd_id)(bd_offset + 4), it_channel_1,
                    {0, 0, 0, (uint32_t)v_offset},
                    {1, (uint32_t)kv_length / 128, (uint32_t)128, (uint32_t)(DH)},
                    {0, (uint32_t)128 * DH * Num_of_d_in_KV_cache, (uint32_t)DH * Num_of_d_in_KV_cache, 1},
                    -1, 0, false
                );
            } else{
                k_offset = kv_chunk_offset * MAX_L * DH * Num_of_d_in_KV_cache + kv_head_offset_in_chunk * DH + kv_begin * DH * Num_of_d_in_KV_cache;
                seq->npu_dma_memcpy_nd(
                    2, 2,
                    MM2S, IT[1][1],
                    (npu_bd_id)(bd_offset + 3), it_channel_0,
                    {0, 0, 0, (uint32_t)k_offset},
                    {1, (uint32_t)1024, (uint32_t)128, (uint32_t)(DH)},
                    {0, (uint32_t)128 * DH * Num_of_d_in_KV_cache, (uint32_t)DH * Num_of_d_in_KV_cache, 1},
                    -1, 0, false
                );
                v_offset = kv_chunk_offset * MAX_L * DH * Num_of_d_in_KV_cache + kv_head_offset_in_chunk * DH + kv_begin * DH * Num_of_d_in_KV_cache + KV_CACHE_SIZE / 2;
                seq->npu_dma_memcpy_nd(
                    2, 2,
                    MM2S, IT[1][1],
                    (npu_bd_id)(bd_offset + 4), it_channel_1,
                    {0, 0, 0, (uint32_t)v_offset},
                    {1, (uint32_t)1024, (uint32_t)128, (uint32_t)(DH)},
                    {0, (uint32_t)128 * DH * Num_of_d_in_KV_cache, (uint32_t)DH * Num_of_d_in_KV_cache, 1},
                    -1, 0, false
                );
                int remaining_kv_length = kv_length - 1024 * 128;
                k_offset = kv_chunk_offset * MAX_L * DH * Num_of_d_in_KV_cache + kv_head_offset_in_chunk * DH + kv_begin * DH * Num_of_d_in_KV_cache + 1024 * 128 * DH * Num_of_d_in_KV_cache;
                seq->npu_dma_memcpy_nd(
                    2, 2,
                    MM2S, IT[1][1],
                    (npu_bd_id)(bd_offset + 5), it_channel_0,
                    {0, 0, 0, (uint32_t)k_offset},
                    {1, (uint32_t)remaining_kv_length / 128, (uint32_t)128, (uint32_t)(DH)},
                    {0, (uint32_t)128 * DH * Num_of_d_in_KV_cache, (uint32_t)DH * Num_of_d_in_KV_cache, 1},
                    -1, 0, false
                );
                v_offset = kv_chunk_offset * MAX_L * DH * Num_of_d_in_KV_cache + kv_head_offset_in_chunk * DH + kv_begin * DH * Num_of_d_in_KV_cache + KV_CACHE_SIZE / 2 + 1024 * 128 * DH * Num_of_d_in_KV_cache;
                seq->npu_dma_memcpy_nd(
                    2, 2,
                    MM2S, IT[1][1],
                    (npu_bd_id)(bd_offset + 6), it_channel_1,
                    {0, 0, 0, (uint32_t)v_offset},
                    {1, (uint32_t)remaining_kv_length / 128, (uint32_t)128, (uint32_t)(DH)},
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

gemma4e_npu_sequence::~gemma4e_npu_sequence() = default;
