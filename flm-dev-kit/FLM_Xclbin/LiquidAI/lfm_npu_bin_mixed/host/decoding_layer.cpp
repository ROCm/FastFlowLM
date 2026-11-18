// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include "decoding_layer.hpp"

decoding_layer::decoding_layer(model_list::model_t model_type, LM_Config& lm_config, npu_manager* xclbin_mgr, int context_length)
    : model_type(model_type), lm_config(lm_config), xclbin_mgr(xclbin_mgr), MAX_L(context_length) {
    header_print("info", "Initializing decoding layer");
    this->rope_weights = model_list::get_model_rotary_frequencies(this->model_type);

    npu_app_desc layer_desc {
        .xclbin_name = lm_config.layer_xclbin_name,
        .app_name    = "deocding_layer"
    };

    this->layer_app = this->xclbin_mgr->create_app(layer_desc);
    this->current_context_length = 0;

    DH = lm_config.head_dim;
    DQ = lm_config.head_dim * lm_config.num_attention_heads;
    DK = lm_config.head_dim * lm_config.num_key_value_heads;
    DV = lm_config.head_dim * lm_config.num_key_value_heads;
    intermediate_size = lm_config.intermediate_size;
    D = lm_config.hidden_size;
    kv_cache_size = (size_t)MAX_L * (DK + DV);
    w_qkv_offset = 0;
    w_o_offset = (DQ + DK + DV) * D * 5 / 8 / sizeof(bf16);
    w_up_offset = w_o_offset + D * DQ * 5 / 8 / sizeof(bf16);
    w_down_offset = w_up_offset + (2 * intermediate_size) * D * 5 / 8 / sizeof(bf16);

    K03_offset = 0;
    K47_offset = kv_cache_size / 4;
    V03_offset = kv_cache_size / 2;
    V47_offset = 3 * kv_cache_size / 4;
    l_qk_address = lm_config.addr_qk;
    l_kv_address = lm_config.addr_kv;
    l_begin_mha_address = lm_config.addr_l_begin_mha;
    l_end_mha_address = lm_config.addr_l_end_mha;
    kk_address = lm_config.addr_kk;
    rms_address = model_list::get_rms_addr(this->model_type);

    header_print("info", "Decoding layer config: ");
    header_print("info", "  D : " << D);
    header_print("info", "  DQ: " << DQ);
    header_print("info", "  DK: " << DK);
    header_print("info", "  DV: " << DV);
    header_print("info", "  Intermediate Size: " << intermediate_size);
    has_qk_norm = model_list::has_qk_norm(this->model_type);
    header_print("info", "  Has QK Norm: " << (has_qk_norm ? "Yes" : "No"));
    this->rope_weights = model_list::get_model_rotary_frequencies(this->model_type);
    this->rope_scaling = model_list::get_rope_scaling(this->model_type);

    size_t total_proj_weights = 
          (DQ + DK + DV) * D
        + DQ * D
        + 3 * D * intermediate_size;

    size_t total_proj_weights_byte_size = total_proj_weights * 5 / 8;
    size_t kv_cache_size = (size_t)MAX_L * (DK + DV);

    this->hidden_states_buffer = this->layer_app.create_bo_buffer<bf16>(D, x_gropu_id);
    this->all_proj_weight_buffer = this->layer_app.create_bo_buffer<u8>(total_proj_weights_byte_size, proj_group_id);
    this->rms_weights_buffer = this->layer_app.create_bo_buffer<bf16>(2 * D, rms_group_id);
    if (this->has_qk_norm) {
        this->rope_rms_buffer = this->layer_app.create_bo_buffer<bf16>(DH * 3, rope_rms_group_id);
    }
    else {
        this->rope_rms_buffer = this->layer_app.create_bo_buffer<bf16>(DH, rope_rms_group_id);
    }
    this->kv_cache = this->layer_app.create_bo_buffer<bf16>(kv_cache_size, kv_cache_group_id);

}

void decoding_layer::load_weights(Q4NX& q4nx) {
    int layer_idx = 0;
    u8* w_ptr = this->all_proj_weight_buffer.data();
    u8* w_ptr_end = this->all_proj_weight_buffer.end();
    {
        LOG_VERBOSE(1, "Loading qkv weights");
        assert(w_ptr < w_ptr_end);
        buffer<u8> w_q;
        buffer<u8> w_k;
        buffer<u8> w_v;
        q4nx.load_weights(w_q, "model.layers." + std::to_string(layer_idx) + ".self_attn.q_proj.weight");
        q4nx.load_weights(w_k, "model.layers." + std::to_string(layer_idx) + ".self_attn.k_proj.weight");
        q4nx.load_weights(w_v, "model.layers." + std::to_string(layer_idx) + ".self_attn.v_proj.weight");
        LOG_VERBOSE(1, "Copying qkv weights to buffer");
        memcpy(w_ptr, w_q.data(), DQ * D * 5 / 8);
        w_ptr += DQ * D * 5 / 8;
        memcpy(w_ptr, w_k.data(), DK * D * 5 / 8);
        w_ptr += DK * D * 5 / 8;
        memcpy(w_ptr, w_v.data(), DV * D * 5 / 8);
        w_ptr += DV * D * 5 / 8;
    }
    // read o weights
    {
        LOG_VERBOSE(1, "Loading o weights");
        assert(w_ptr < w_ptr_end);
        buffer<u8> w_o;
        q4nx.load_weights(w_o, "model.layers." + std::to_string(layer_idx) + ".self_attn.o_proj.weight");
        memcpy(w_ptr, w_o.data(), DQ * D * 5 / 8);
        w_ptr += DQ * D * 5 / 8;
    }
    // read up weights
    {
        LOG_VERBOSE(1, "Loading up weights");
        assert(w_ptr < w_ptr_end);
        buffer<u8> w_up;
        buffer<u8> w_gate;
        q4nx.load_weights(w_up, "model.layers." + std::to_string(layer_idx) + ".mlp.up_proj.weight");
        q4nx.load_weights(w_gate, "model.layers." + std::to_string(layer_idx) + ".mlp.gate_proj.weight");
        // For every D rows, half is up, half is gate
        const int blocks = 1024 / 2 / 32 * (D / 256);
        const int chunk_size = blocks * 32 * 256 * 5 / 8;
        const int total_size = intermediate_size * D * 5 / 8;
        const int phases = total_size / chunk_size;
        for (int i = 0; i < phases; i++){
            memcpy(w_ptr, w_up.data() + i * chunk_size, chunk_size);
            w_ptr += chunk_size;
            memcpy(w_ptr, w_gate.data() + i * chunk_size, chunk_size);
            w_ptr += chunk_size;
        }
    }
    // read down weights
    {
        LOG_VERBOSE(1, "Loading down weights");
        assert(w_ptr < w_ptr_end);
        buffer<u8> w_down;
        q4nx.load_weights(w_down, "model.layers." + std::to_string(layer_idx) + ".mlp.down_proj.weight");
        memcpy(w_ptr, w_down.data(), intermediate_size * D * 5 / 8);
        w_ptr += intermediate_size * D * 5 / 8;
    }
    assert(w_ptr == w_ptr_end);
    // read rms weights
    {
        LOG_VERBOSE(1, "Loading rms weights");
        buffer<bf16> w_rms_pre;
        buffer<bf16> w_rms_post;

        bf16* w_rms_ptr = this->rms_weights_buffer.data();
        q4nx.load_weights(w_rms_pre, "model.layers." + std::to_string(layer_idx) + ".input_layernorm.weight");
        q4nx.load_weights(w_rms_post, "model.layers." + std::to_string(layer_idx) + ".post_attention_layernorm.weight");

        // w_rms_post.memset(1);
        memcpy((unsigned char*)w_rms_ptr, w_rms_pre.data(), D * sizeof(bf16));
        w_rms_ptr += D;      
        memcpy((unsigned char*)w_rms_ptr, w_rms_post.data(), D * sizeof(bf16));
        w_rms_ptr += D;
    }
    this->_set_rope_weights(0);
    this->kv_cache.memset((bf16)0.0f);

    this->all_proj_weight_buffer.sync_to_device();
    this->rms_weights_buffer.sync_to_device();
    this->rope_rms_buffer.sync_to_device();
    this->kv_cache.sync_to_device();
}

buffer<bf16> decoding_layer::operator()(buffer<bf16>& input) {
    memcpy(this->hidden_states_buffer.data(), input.data(), D * sizeof(bf16));
    this->hidden_states_buffer.sync_to_device();
    this->layer_app(hidden_states_buffer, all_proj_weight_buffer, rms_weights_buffer, rope_rms_buffer, kv_cache);
    this->hidden_states_buffer.sync_from_device();
    this->set_context_length(this->current_context_length + 1);
    return this->hidden_states_buffer;
}

void decoding_layer::_set_rope_weights(int idx){
    static bool printed = false;
    bf16* w_rope_ptr = this->rope_rms_buffer.data();
    for (uint32_t i = 0; i < DH / 2; i++){
        w_rope_ptr[i] = bf16(std::cos(this->rope_weights[i] * idx) * this->rope_scaling);
        w_rope_ptr[i + DH / 2] = bf16(std::sin(this->rope_weights[i] * idx) * this->rope_scaling);
    }
    if (!printed){
        utils::print_matrix(rope_rms_buffer, 16, 16);
        printed = true;
    }
    this->rope_rms_buffer.sync_to_device();
}

void decoding_layer::set_context_length(int L){

    this->current_context_length = L;

    this->_gen_sequence(this->layer_app.instr_seq, L + 1);

    this->_set_rope_weights(L);
}

void decoding_layer::_gen_sequence(npu_sequence* seq, int L) {
    constexpr size_t CT_lock_address_base = 0x000001F000;
    int proj_rtp_lock_id = 6;
    int rms_rtp_lock_id = 6;
    seq->clear_cmds();

    for (int i = 0; i < 4; i++){
        seq->rtp_write(attn_qk_tiles[i], l_qk_address, L);
        seq->rtp_write(attn_kv_tiles[i], l_kv_address, L);
    }
    for (int i = 0; i < 16; i++){
        seq->rtp_write(proj_tiles[i], 33280, 1);
        seq->rtp_write(proj_tiles[i], CT_lock_address_base + 16 * proj_rtp_lock_id, 1); // set lock to 1
    }
    seq->rtp_write(rms_tile, rms_address, 1);
    seq->rtp_write(rms_tile, CT_lock_address_base + 16 * rms_rtp_lock_id, 1); // set lock to 1

    _send_hidden_states(seq);
    _send_rms_weights(seq);
    _send_rope_rms_weights(seq);

    // receive y
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        x_arg_id,
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
    _move_weights(seq, D, DQ, w_o_offset);
    _move_weights(seq, 2 * intermediate_size, D, w_up_offset);
    _move_weights(seq, D, intermediate_size, w_down_offset);

    // wait for receiving y
    seq->npu_dma_wait(
        xyr_tile,
        S2MM,
        it_channel_1
    );

    seq->cmds2seq();
}



void decoding_layer::_send_hidden_states(npu_sequence* seq){
    // send x
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        x_arg_id,
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

void decoding_layer::_send_rms_weights(npu_sequence* seq){
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        rms_arg_id,
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

void decoding_layer::_send_rope_rms_weights(npu_sequence* seq){
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        rope_rms_arg_id,
        MM2S,
        xyr_tile,
        bd_2,
        it_channel_0,
        {0, 0, 0, 0},
        {1, 1, 1, (uint32_t)DH},
        {0, 0, 0, 1},
        1, 0, false, no_cache
    );
}

void decoding_layer::_receive_kv_cache(npu_sequence* seq, const int L){
    const uint32_t kv_cache_offset = (L - 1) * 4 * DH;
    // receive new k, v
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        kv_cache_arg_id,
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
        kv_cache_arg_id,
        S2MM,
        IT7,
        bd_0,
        it_channel_0,
        {0, 0, 0, kv_cache_offset + (uint32_t)V03_offset},
        {1, 1, 1, 4 * (uint32_t)DH},
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
        kv_cache_arg_id,
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
        kv_cache_arg_id,
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

void decoding_layer::_move_weights(npu_sequence* seq, const size_t Dout, const size_t Din, const size_t w_offset){
    const int m = 32;
    const int k = 256;
    const uint32_t columns = 4;
    const uint32_t elems = m * k;
    const uint32_t a_block_size = elems * 5 / 8 / 2;
    const uint32_t blocks_per_row = Din / k;
    const uint32_t cores = columns * 4;
    assert(Dout / m / cores > 0);
    assert(Dout / m / cores % 2 == 0);
    uint32_t AXI_FLAG = 0b1110<<24; 
    uint32_t DMA_BD_x_5 = 0x000001D014;
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
                {blocks_per_row, 2, 8, a_block_size / 8},
                {a_block_size, a_block_size * blocks_per_row, a_block_size / 8, 1},
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
    
void decoding_layer::_move_kv_cache(npu_sequence* seq, const size_t L){
    const int L_padded = (L + L_CHUNK - 1) / L_CHUNK * L_CHUNK;
    const uint32_t data2move = L_padded * DH * 4;
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        kv_cache_arg_id,
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
        kv_cache_arg_id,
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
        kv_cache_arg_id,
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
        kv_cache_arg_id,
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

buffer<bf16> decoding_layer::get_k_cache(int idx){
    this->kv_cache.sync_from_device();
    buffer<bf16> k_cache(DK);
    memcpy(k_cache.data(), this->kv_cache.data() + K03_offset + idx * DK / 2, DK * sizeof(bf16) / 2);
    memcpy(k_cache.data() + DK / 2, this->kv_cache.data() + K47_offset + idx * DK / 2, DK * sizeof(bf16) / 2);
    return k_cache;
}

buffer<bf16> decoding_layer::get_v_cache(int idx){
    this->kv_cache.sync_from_device();
    buffer<bf16> v_cache(DV);
    memcpy(v_cache.data(), this->kv_cache.data() + V03_offset + idx * DV / 2, DV * sizeof(bf16) / 2);
    memcpy(v_cache.data() + DV / 2, this->kv_cache.data() + V47_offset + idx * DV / 2, DV * sizeof(bf16) / 2);
    return v_cache;
}