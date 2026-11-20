// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include "decoding_layer.hpp"

decoding_layer::decoding_layer(model_list::model_t model_type, LM_Config& lm_config, npu_app_manager* layer_app_mgr, int context_length)
    : model_type(model_type), lm_config(lm_config), layer_app_mgr(layer_app_mgr), MAX_L(context_length) {
    header_print("info", "Initializing decoding layer");
    this->rope_weights = model_list::get_model_rotary_frequencies(this->model_type);


    this->layer_app = this->layer_app_mgr->create_app();
    this->lm_head = this->layer_app_mgr->create_app();
    this->current_context_length = 0;

    DH = lm_config.head_dim;
    DQ = lm_config.head_dim * lm_config.num_attention_heads;
    DK = lm_config.head_dim * lm_config.num_key_value_heads;
    DV = lm_config.head_dim * lm_config.num_key_value_heads;
    intermediate_size = lm_config.intermediate_size;
    D = lm_config.hidden_size;
    vocab_size = lm_config.vocab_size;
    vocab_size_padded = ((vocab_size + D - 1) / D) * D; // pad to multiple of D
    kv_cache_size = (size_t)MAX_L * (DK + DV);
    w_qkv_offset = 0;
    w_o_offset = (DQ + DK + DV) * D * 5 / 8 / sizeof(bf16);
    w_up_offset = w_o_offset + D * DQ * 5 / 8 / sizeof(bf16);
    w_down_offset = w_up_offset + (2 * intermediate_size) * D * 5 / 8 / sizeof(bf16);

    V_offset = kv_cache_size / 2;
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

    this->hidden_states_buffer = this->layer_app.create_bo_buffer<bf16>(D);
    this->all_proj_weight_buffer = this->layer_app.create_bo_buffer<u8>(total_proj_weights_byte_size);
    this->rms_weights_buffer = this->layer_app.create_bo_buffer<bf16>(2 * D + DQ + DK + DV);
    if (this->has_qk_norm) {
        this->rope_rms_buffer = this->layer_app.create_bo_buffer<bf16>(DH * 3 + DQ + DK + DV);
    }
    else {
        this->rope_rms_buffer = this->layer_app.create_bo_buffer<bf16>(DH);
    }
    logits = this->lm_head.create_bo_buffer<bf16>(vocab_size_padded);
    lm_head_weights = this->lm_head.create_bo_buffer<u8>(vocab_size_padded * D * 5 / 8);
    lm_head_buffer_x = this->lm_head.create_bo_buffer<bf16>(D);
    final_rms_weight = this->lm_head.create_bo_buffer<bf16>(D);
    logits_valid = buffer<bf16>(logits.data(), vocab_size);
    this->kv_cache = this->layer_app.create_bo_buffer<bf16>(kv_cache_size);
    this->kv_cache.memset((bf16)0);
    this->kv_cache.sync_to_device();

    _gen_lm_head(lm_head.seq());
}

void reorder_cpy(u8 *dst, buffer<u8> &src, const int col, const int vertical_blocks = 2)
{
    // memcpy(dst, src.data(), src.size());
    // return;
    const int a_block_size = 32 * 256 * 5 / 8;
    const int blocks_per_row = col / 256;

    const int rows = src.size() / a_block_size / blocks_per_row;

    u8 *dst_ptr = dst;
    std::vector<u8 *> src_ptr(vertical_blocks);
    for (int i = 0; i < vertical_blocks; i++)
    {
        src_ptr[i] = src.data() + i * a_block_size * blocks_per_row;
    }
    for (int r = 0; r < rows; r += vertical_blocks)
    {
        for (int c = 0; c < blocks_per_row; c++)
        {
            for (int i = 0; i < vertical_blocks; i++)
            {
                memcpy(dst_ptr, src_ptr[i], a_block_size);
                dst_ptr += a_block_size;
                src_ptr[i] += a_block_size;
            }
        }
        for (int i = 0; i < vertical_blocks; i++)
        {
            src_ptr[i] += (vertical_blocks - 1) * a_block_size * blocks_per_row;
            if (src_ptr[i] + a_block_size * blocks_per_row > src.end())
            {
                src_ptr[i] = src.data(); // useless padding
            }
        }
    }
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
        reorder_cpy(w_ptr, w_q, D);
        w_ptr += DQ * D * 5 / 8;
        reorder_cpy(w_ptr, w_k, D);
        w_ptr += DK * D * 5 / 8;
        reorder_cpy(w_ptr, w_v, D);
        w_ptr += DV * D * 5 / 8;
    }
    // read o weights
    {
        LOG_VERBOSE(1, "Loading o weights");
        assert(w_ptr < w_ptr_end);
        buffer<u8> w_o;
        q4nx.load_weights(w_o, "model.layers." + std::to_string(layer_idx) + ".self_attn.o_proj.weight");
        reorder_cpy(w_ptr, w_o, DQ);
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
        tensor_2d<u8> w_up_tensor(w_up, chunk_size, 0);
        tensor_2d<u8> w_gate_tensor(w_gate, chunk_size, 0);
        for (int i = 0; i < phases; i++)
        {
            reorder_cpy(w_ptr, w_up_tensor[i], D);
            w_ptr += chunk_size;
            reorder_cpy(w_ptr, w_gate_tensor[i], D);
            w_ptr += chunk_size;
        }
    }
    // read down weights
    {
        LOG_VERBOSE(1, "Loading down weights");
        assert(w_ptr < w_ptr_end);
        buffer<u8> w_down;
        q4nx.load_weights(w_down, "model.layers." + std::to_string(layer_idx) + ".mlp.down_proj.weight");
        reorder_cpy(w_ptr, w_down, intermediate_size);
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

        // append qkv bias
        buffer<bf16> b_q;
        buffer<bf16> b_k;
        buffer<bf16> b_v;
        q4nx.load_weights(b_q, "model.layers." + std::to_string(layer_idx) + ".self_attn.q_proj.bias");
        q4nx.load_weights(b_k, "model.layers." + std::to_string(layer_idx) + ".self_attn.k_proj.bias");
        q4nx.load_weights(b_v, "model.layers." + std::to_string(layer_idx) + ".self_attn.v_proj.bias");
        memcpy((unsigned char*)w_rms_ptr, b_q.data(), DQ * sizeof(bf16));
        w_rms_ptr += DQ;
        memcpy((unsigned char*)w_rms_ptr, b_k.data(), DK * sizeof(bf16));
        w_rms_ptr += DK;
        memcpy((unsigned char*)w_rms_ptr, b_v.data(), DV * sizeof(bf16));
        w_rms_ptr += DV;
    }
    
    {
        buffer<u8> lm_head_w;
        q4nx.load_weights(lm_head_w, "lm_head.weight");
        reorder_cpy(lm_head_weights.data(), lm_head_w, D);
        lm_head_weights.sync_to_device();
    }
    q4nx.load_weights(this->final_rms_weight, "model.norm.weight");
    this->final_rms_weight.sync_to_device();
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

buffer<bf16> decoding_layer::get_logits(buffer<bf16>& x){
    memcpy(this->lm_head_buffer_x.data(), x.data(), D * sizeof(bf16));
    this->lm_head_buffer_x.sync_to_device();
    this->lm_head_weights.sync_to_device();
    this->final_rms_weight.sync_to_device();
    this->lm_head(logits, lm_head_weights, lm_head_buffer_x, final_rms_weight);
    this->logits.sync_from_device();
    return this->logits_valid;
}

float decoding_layer::bench(buffer<bf16>& input) {
    this->set_context_length(8);
    memcpy(this->hidden_states_buffer.data(), input.data(), D * sizeof(bf16));
    int layers = 36;
    xrt::runlist rl = this->layer_app_mgr->create_runlist();
    for (int i = 0; i < layers; i++){
        rl.add(this->layer_app.create_run(hidden_states_buffer, all_proj_weight_buffer, rms_weights_buffer, rope_rms_buffer, kv_cache));
    }
    time_utils::time_point start = time_utils::now();
    rl.execute();
    rl.wait();
    this->hidden_states_buffer.sync_from_device();
    this->get_logits(this->hidden_states_buffer);
    time_utils::time_point end = time_utils::now();
    time_utils::time_with_unit total_ms = time_utils::duration_ms(start, end);
    return 1000.0 / total_ms.first;
}


void decoding_layer::_set_rope_weights(int idx){
    bf16* w_rope_ptr = this->rope_rms_buffer.data();
    for (uint32_t i = 0; i < DH / 2; i++){
        w_rope_ptr[i] = bf16(std::cos(this->rope_weights[i] * idx) * this->rope_scaling);
        w_rope_ptr[i + DH / 2] = bf16(std::sin(this->rope_weights[i] * idx) * this->rope_scaling);
    }
    this->rope_rms_buffer.sync_to_device();
}

void decoding_layer::set_context_length(int L){

    this->current_context_length = L;

    this->_gen_sequence(this->layer_app.seq(), L + 1);

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
        1, 0, true, no_cache
    );
    seq->npu_dma_wait(
        xyr_tile,
        MM2S,
        it_channel_0
    );
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        rms_arg_id,
        MM2S,
        xyr_tile,
        bd_3,
        it_channel_0,
        {0, 0, 0, (uint32_t)D * 2},
        {1, 1, 1, (uint32_t)DQ + DK + DV},
        {0, 0, 0, 1},
        1, 0, false, no_cache
    );
}

void decoding_layer::_receive_kv_cache(npu_sequence* seq, const int L){
    const uint32_t kv_cache_offset = (L - 1) * 2 * DH;
    // receive new k, v
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        kv_cache_arg_id,
        S2MM,
        IT7,
        bd_0,
        it_channel_0,
        {0, 0, 0, kv_cache_offset},
        {1, 1, 1, 2 * (uint32_t)DH},
        {0, 0, 0, 1},
        -1, 0, true
    );
    _move_weights(seq, DQ + DK + DV, D, w_qkv_offset);
    seq->npu_dma_wait(
        IT7,
        S2MM,
        it_channel_0
    );
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        kv_cache_arg_id,
        S2MM,
        IT7,
        bd_0,
        it_channel_0,
        {0, 0, 0, kv_cache_offset + (uint32_t)V_offset},
        {1, 1, 1, 2 * (uint32_t)DH},
        {0, 0, 0, 1},
        -1, 0, true
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
    
void decoding_layer::_move_kv_cache(npu_sequence* seq, const size_t L){
    const int L_padded = (L + L_CHUNK - 1) / L_CHUNK * L_CHUNK;
    const uint32_t data2move = L_padded * DH * 2;
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        kv_cache_arg_id,
        MM2S,
        attn_tiles[0],
        bd_0,
        it_channel_0,
        {0, 0, 0, (uint32_t)0},
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
        {0, 0, 0, (uint32_t)V_offset},
        {1, 1, 1, data2move},
        {0, 0, 0, 1},
        -1, 0, false, aggressive_cache
    );

}

void decoding_layer::_gen_lm_head(npu_sequence* seq) {
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
        seq->rtp_write(proj_tiles[i], 33280, 0);
        seq->rtp_write(proj_tiles[i], CT_lock_address_base + 16 * proj_rtp_lock_id, 1); // set lock to 1
    }
    seq->rtp_write(rms_tile, rms_address, 0);
    seq->rtp_write(rms_tile, CT_lock_address_base + 16 * rms_rtp_lock_id, 1); // set lock to 1
    
    // y_out
    seq->npu_dma_memcpy_nd(
        sizeof(bf16), lm_head_y_arg_idx, 
        S2MM, IT0,  bd_15, it_channel_1,
        {0, 0, 0, 0},
        {1, 1, 1, (uint32_t)vocab_size_padded},
        {0, 0, 0, 1},
        -1, 0, true
    );
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
    // send weights

    _move_weights(seq, vocab_size_padded, D, 0);

    // wait for y
    seq->npu_dma_wait(
        IT0,
        S2MM,
        it_channel_1
    );
    seq->cmds2seq();
}

buffer<bf16> decoding_layer::get_k_cache(int idx){
    this->kv_cache.sync_from_device();
    buffer<bf16> k_cache(DK);
    memcpy(k_cache.data(), this->kv_cache.data() + idx * DK, DK * sizeof(bf16));
    return k_cache;
}

buffer<bf16> decoding_layer::get_v_cache(int idx){
    this->kv_cache.sync_from_device();
    buffer<bf16> v_cache(DV);
    memcpy(v_cache.data(), this->kv_cache.data() + V_offset + idx * DV, DV * sizeof(bf16));
    return v_cache;
}
