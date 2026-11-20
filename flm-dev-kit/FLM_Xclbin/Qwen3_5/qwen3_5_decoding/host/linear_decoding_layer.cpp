// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include "linear_decoding_layer.hpp"

linear_decoding_layer::linear_decoding_layer(model_list::model_t model_type, LM_Config& lm_config, npu_app_manager* layer_app_mgr, int tb_size)
    : model_type(model_type), lm_config(lm_config), layer_app_mgr(layer_app_mgr), tb_size(tb_size) {
    header_print("info", "Initializing decoding layer");

    this->layer_app = this->layer_app_mgr->create_app();
    this->lm_head = this->layer_app_mgr->create_app();
    this->current_context_length = 0;

    num_hidden_layers = lm_config.num_hidden_layers;
    linear_qk_head_dim = lm_config.linear_key_head_dim;
    linear_v_head_dim = lm_config.linear_value_head_dim;
    linear_num_qk_heads = lm_config.linear_num_key_heads;
    linear_num_v_heads = lm_config.linear_num_value_heads;

    linear_num_v_heads_padded = (linear_num_v_heads + 31) / 32 * 32;

    conv_kernel_size = lm_config.linear_conv_kernel_dim;
    Linear_DQ = linear_qk_head_dim * linear_num_qk_heads;
    Linear_DK = linear_qk_head_dim * linear_num_qk_heads;
    Linear_DV = linear_v_head_dim * linear_num_v_heads;
    intermediate_size = lm_config.intermediate_size;
    D = lm_config.hidden_size;
  
    linear_w_qkv_offset = 0;
    linear_w_gate_offset = (Linear_DQ + Linear_DK + Linear_DV) * D * 5 / 8 / sizeof(bf16);
    linear_w_o_offset = linear_w_gate_offset + Linear_DV * D * 5 / 8 / sizeof(bf16);
    linear_w_upgate_offset = linear_w_o_offset + (D * Linear_DV + D * Linear_DV / 32 * 2)/ sizeof(bf16);
    linear_w_down_offset = linear_w_upgate_offset + (2 * intermediate_size) * D * 5 / 8 / sizeof(bf16);
    w_ab_offset = linear_w_down_offset + (intermediate_size * D) * 5 / 8 / sizeof(bf16);

    l_qk_address = lm_config.addr_qk;
    l_kv_address = lm_config.addr_kv;
    l_begin_mha_address = lm_config.addr_l_begin_mha;
    l_end_mha_address = lm_config.addr_l_end_mha;
    kk_address = lm_config.addr_kk;
    rms_address = model_list::get_rms_addr(this->model_type);

    header_print("info", "Decoding layer config: ");
    header_print("info", "  D : " << D);
    header_print("info", "  Linear_DQ: " << Linear_DQ);
    header_print("info", "  Linear_DK: " << Linear_DK);
    header_print("info", "  Linear_DV: " << Linear_DV);
    header_print("info", "  Conv Kernel Size: " << conv_kernel_size);
    header_print("info", "  Intermediate Size: " << intermediate_size);
    has_qk_norm = model_list::has_qk_norm(this->model_type);
    
    size_t total_q4_1_proj_weights = 
          (Linear_DQ + Linear_DK + Linear_DV) * D
        + Linear_DV * D
        + 3 * D * intermediate_size;

    size_t total_q8_0_proj_weights = 
        + Linear_DV * D
        + 2 * (linear_num_v_heads_padded) * D; // alpha and beta q8

    size_t total_proj_weights_byte_size = total_q4_1_proj_weights * 5 / 8 + total_q8_0_proj_weights + total_q8_0_proj_weights / 32 * 2;
    linear_kv_cache_size = (conv_kernel_size - 1) * (Linear_DQ + Linear_DK + Linear_DV) + linear_num_v_heads * linear_v_head_dim * linear_qk_head_dim * 2; // conv cache + alpha beta cache
    s_offset = (size_t)(conv_kernel_size - 1) * (Linear_DQ + Linear_DK + Linear_DV);
    conv_w_size = (size_t)conv_kernel_size * (Linear_DQ + Linear_DK + Linear_DV);

    header_print("info", "  CONV_W_SIZE: " << conv_w_size);

    this->hidden_states_buffer = this->layer_app.create_bo_buffer<bf16>(D + tb_size);
    this->all_proj_weight_buffer = this->layer_app.create_bo_buffer<u8>(total_proj_weights_byte_size);
    this->rms_weights_buffer = this->layer_app.create_bo_buffer<bf16>(2 * D);

    this->rope_rms_buffer = this->layer_app.create_bo_buffer<bf16>(conv_w_size + linear_v_head_dim + linear_num_v_heads_padded * 2 * 2);
    this->kv_cache = this->layer_app.create_bo_buffer<bf16>(linear_kv_cache_size);
    this->kv_cache.memset((bf16)0);
    this->kv_cache.sync_to_device();
    this->_gen_linear_sequence(this->layer_app.seq());

}

void linear_decoding_layer::reorder_cpy(u8 *dst, buffer<u8> &src, const int col, const int vertical_blocks)
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
            for (int j = 0; j < a_block_size / 512; j++){
                for (int i = 0; i < vertical_blocks; i++){
                    memcpy(dst_ptr, src_ptr[i], 512);
                    dst_ptr += 512;
                    src_ptr[i] += 512;
                }
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

void linear_decoding_layer::reorder_cpy_q80(u8 *dst, buffer<u8> &src, const int col, const int vertical_blocks)
{
    // memcpy(dst, src.data(), src.size());
    // return;
    const int a_block_size = 32 * 256 + 256 * 2;
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
            for (int j = 0; j < a_block_size / 512; j++){
                for (int i = 0; i < vertical_blocks; i++){
                    memcpy(dst_ptr, src_ptr[i], 512);
                    dst_ptr += 512;
                    src_ptr[i] += 512;
                }
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


void linear_decoding_layer::load_weights(Q4NX& q4nx) {
    int layer_idx = 0;
    u8* w_ptr = this->all_proj_weight_buffer.data();
    u8* w_ptr_end = this->all_proj_weight_buffer.end();
    {
        LOG_VERBOSE(1, "Loading qkv weights");
        assert(w_ptr < w_ptr_end);
        buffer<u8> w_qkv;
        q4nx.load_weights(w_qkv, "model.layers." + std::to_string(layer_idx) + ".linear_attn.qkv_proj.weight");
        LOG_VERBOSE(1, "Copying qkv weights to buffer");
        reorder_cpy(w_ptr, w_qkv, D);
        w_ptr += (Linear_DQ + Linear_DK + Linear_DV) * D * 5 / 8;
    }
    // read gate weights
    {
        LOG_VERBOSE(1, "Loading gate weights");
        buffer<u8> w_gate;
        q4nx.load_weights(w_gate, "model.layers." + std::to_string(layer_idx) + ".self_attn.gate_proj.weight");
        reorder_cpy(w_ptr, w_gate, D);
        w_ptr += Linear_DV * D * 5 / 8;
    }
    // read o weights
    {
        LOG_VERBOSE(1, "Loading o weights");
        assert(w_ptr < w_ptr_end);
        buffer<u8> w_o;
        q4nx.load_weights(w_o, "model.layers." + std::to_string(layer_idx) + ".linear_attn.ssm_out_proj.weight");
        reorder_cpy_q80(w_ptr, w_o, Linear_DV);
        w_ptr += (Linear_DV * D + Linear_DV * D / 32 * 2);
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
            LOG_VERBOSE(1, "Copying up and gate weights to buffer, phase " << i + 1 << "/" << phases);
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
    // read alpha_beta_weights
    {
        LOG_VERBOSE(1, "Loading alpha and beta weights");
        assert(w_ptr < w_ptr_end);
        buffer<u8> w_alpha;
        buffer<u8> w_beta;
        q4nx.load_weights(w_alpha, "model.layers." + std::to_string(layer_idx) + ".linear_attn.ssm_alpha_proj.weight");
        q4nx.load_weights(w_beta, "model.layers." + std::to_string(layer_idx) + ".linear_attn.ssm_beta_proj.weight");
        memcpy(w_ptr, w_alpha.data(), w_alpha.size());
        w_ptr += linear_num_v_heads_padded * D  + linear_num_v_heads_padded * D / 32 * 2;
        memcpy(w_ptr, w_beta.data(), w_beta.size());
        w_ptr += linear_num_v_heads_padded * D  + linear_num_v_heads_padded * D / 32 * 2;
    }
    assert(w_ptr == w_ptr_end);
    this->all_proj_weight_buffer.sync_to_device();
    // read rms weights
    {
        LOG_VERBOSE(1, "Loading rms weights");
        buffer<bf16> w_rms_pre;
        buffer<bf16> w_rms_post;

        bf16* w_rms_ptr = this->rms_weights_buffer.data();
        q4nx.load_weights(w_rms_pre, "model.layers." + std::to_string(layer_idx) + ".input_layernorm.weight");
        q4nx.load_weights(w_rms_post, "model.layers." + std::to_string(layer_idx) + ".post_attention_layernorm.weight");
        header_print("info", "Input layer_norm:");
        utils::print_matrix(w_rms_pre, D);

        // w_rms_post.memset(1);
        memcpy((unsigned char*)w_rms_ptr, w_rms_pre.data(), D * sizeof(bf16));
        w_rms_ptr += D;      
        memcpy((unsigned char*)w_rms_ptr, w_rms_post.data(), D * sizeof(bf16));
        w_rms_ptr += D;
    }
    bf16* other_weights_ptr = rope_rms_buffer.data();
    {
        // load conv_weights
        buffer<bf16> conv_weights;
        q4nx.load_weights(conv_weights, "model.layers." + std::to_string(layer_idx) + ".linear_attn.ssm_conv1d.weight");
        memcpy(other_weights_ptr, conv_weights.data(), conv_weights.size() * sizeof(bf16));
        other_weights_ptr += conv_weights.size();
    }

    {
        // load ssm_norm_weights
        buffer<bf16> norm_weights;
        q4nx.load_weights(norm_weights, "model.layers." + std::to_string(layer_idx) + ".linear_attn.ssm_norm.weight");
        memcpy(other_weights_ptr, norm_weights.data(), linear_v_head_dim * sizeof(bf16));
        other_weights_ptr += linear_v_head_dim;
    }
    {
        // load ssm a, dt
        buffer<float> a;
        buffer<float> dt;

        q4nx.load_weights(a, "model.layers." + std::to_string(layer_idx) + ".linear_attn.ssm_a");
        q4nx.load_weights(dt, "model.layers." + std::to_string(layer_idx) + ".linear_attn.ssm_dt.bias");
        memcpy(other_weights_ptr, a.data(), linear_num_v_heads * sizeof(float));
        other_weights_ptr += linear_num_v_heads_padded * sizeof(float) / sizeof(bf16);
        memcpy(other_weights_ptr, dt.data(), linear_num_v_heads * sizeof(float));
        other_weights_ptr += linear_num_v_heads_padded * sizeof(float) / sizeof(bf16);
    }

    assert (other_weights_ptr == rope_rms_buffer.end());

    this->kv_cache.memset((bf16)0.0f);
    // for (int i = 0; i < kv_cache.size(); i++){
    //     this->kv_cache[i] = (bf16)((i / 4096) + 1);
    // }

    this->rms_weights_buffer.sync_to_device();
    this->rope_rms_buffer.sync_to_device();
    this->kv_cache.sync_to_device();
    header_print("info", "Weights loaded!");
}

buffer<bf16> linear_decoding_layer::operator()(buffer<bf16>& input) {
    memcpy(this->hidden_states_buffer.data(), input.data(), D * sizeof(bf16));
    this->hidden_states_buffer.sync_to_device();
    this->layer_app(hidden_states_buffer, all_proj_weight_buffer, rms_weights_buffer, rope_rms_buffer, kv_cache);
    this->hidden_states_buffer.sync_from_device();
    return this->hidden_states_buffer;
}

float linear_decoding_layer::bench(buffer<bf16>& input) {
    memcpy(this->hidden_states_buffer.data(), input.data(), D * sizeof(bf16));
    xrt::runlist rl = this->layer_app_mgr->create_runlist();
    for (int i = 0; i < num_hidden_layers; i++){
        rl.add(this->layer_app.create_run(hidden_states_buffer, all_proj_weight_buffer, rms_weights_buffer, rope_rms_buffer, kv_cache));
    }
    time_utils::time_point start = time_utils::now();
    rl.execute();
    rl.wait();
    this->hidden_states_buffer.sync_from_device();
    time_utils::time_point end = time_utils::now();
    time_utils::time_with_unit total_ms = time_utils::duration_ms(start, end);
    return 1000.0 / total_ms.first;
}

void linear_decoding_layer::_gen_linear_sequence(npu_sequence* seq) {
    constexpr size_t CT_lock_address_base = 0x000001F000;
    int proj_rtp_lock_id = 6;
    int rms_rtp_lock_id = 6;
    int gate_rtp_lock_id = 6;
    seq->clear_cmds();

    for (int i = 0; i < 16; i++){
        seq->rtp_write(proj_tiles[i], 17408, 0);
        seq->rtp_write(proj_tiles[i], CT_lock_address_base + 16 * proj_rtp_lock_id, 1); // set lock to 1
    }
    seq->rtp_write(rms_tile, rms_address, 0);
    seq->rtp_write(gate_tile, 49152, 0);
    seq->rtp_write(rms_tile, CT_lock_address_base + 16 * rms_rtp_lock_id, 1); // set lock to 1
    seq->rtp_write(gate_tile, CT_lock_address_base + 16 * gate_rtp_lock_id, 1); // set lock to 1
    _send_hidden_states(seq);
    _send_rms_weights(seq);
    _send_linear_conv_weights(seq);

    // receive y
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        x_arg_id,
        S2MM,
        IT2,
        bd_10,
        it_channel_1,
        {0, 0, 0, 0},
        {1, 1, 1, (uint32_t)D},
        {0, 0, 0, 1},
        -1, 0, true
    );  
    //receive tb
    if (tb_size > 0)
        seq->npu_dma_memcpy_nd(
            sizeof(bf16),
            x_arg_id,
            S2MM,
            IT5,
            bd_10,
            it_channel_0,
            {0, 0, 0, (uint32_t)D},
            {1, 1, 1, (uint32_t)tb_size},
            {0, 0, 0, 1},
            -1, 0, true
        );

    _move_linear_kv_cache(seq);
    _move_alpha_beta_weights(seq, linear_num_v_heads_padded * 2, D, w_ab_offset);
    _move_weights(seq, Linear_DQ + Linear_DK + Linear_DV, D, linear_w_qkv_offset);
    _receive_linear_kv_cache(seq); // also move qkv weights
    _move_s(seq);
    _move_weights(seq, Linear_DV, D, linear_w_gate_offset);
    _move_weights_q80(seq, D, Linear_DV, linear_w_o_offset);
    _move_weights(seq, 2 * intermediate_size, D, linear_w_upgate_offset);
    _move_weights(seq, D, intermediate_size, linear_w_down_offset);

    // wait for receiving y
    seq->npu_dma_wait(
        IT2,
        S2MM,
        it_channel_1
    );
    if (tb_size > 0)
        seq->npu_dma_wait(
            IT5,
            S2MM,
            it_channel_0
        );
    seq->cmds2seq();
    seq->write_out_sequence("debug_seq.bin");
}



void linear_decoding_layer::_send_hidden_states(npu_sequence* seq){
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

void linear_decoding_layer::_send_rms_weights(npu_sequence* seq){
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

void linear_decoding_layer::_send_linear_conv_weights(npu_sequence* seq){
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        rope_rms_arg_id,
        MM2S,
        IT5,
        bd_2,
        it_channel_1,
        {0, 0, 0, 0},
        {1, 1, 1, (uint32_t)conv_w_size + linear_v_head_dim + linear_num_v_heads_padded * 2 * 2},
        {0, 0, 0, 1},
        -1, 0, true, no_cache
    );
    seq->npu_dma_wait(
        IT5,
        MM2S,
        it_channel_1
    );
}

void linear_decoding_layer::_receive_linear_kv_cache(npu_sequence* seq){
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        kv_cache_arg_id,
        S2MM,
        IT5,
        bd_2,
        it_channel_1,
        {0, 0, 0, 0},
        {1, 1, 1, (uint32_t)(conv_kernel_size - 1) * (Linear_DQ + Linear_DK + Linear_DV)},
        {0, 0, 0, 1},
        -1, 0, true, no_cache
    );
    seq->npu_dma_wait(
        IT5,
        S2MM,
        it_channel_1
    );
}

void linear_decoding_layer::_move_weights(npu_sequence* seq, const size_t Dout, const size_t Din, const size_t w_offset){
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
    

void linear_decoding_layer::_move_weights_q80(npu_sequence* seq, const size_t Dout, const size_t Din, const size_t w_offset){
    const int m = 32;
    const int k = 256;
    const uint32_t columns = 4;
    const uint32_t elems = m * k;
    const uint32_t a_block_size = (elems + elems / 32 * 2) / 2;
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

void linear_decoding_layer::_move_alpha_beta_weights(npu_sequence* seq, const size_t Dout, const size_t Din, const size_t w_offset){
    // currently, only on DMA channel 0 of IT5
    const int m = 32;
    const int k = 256;
    constexpr uint32_t block_size = (m * k + m * k / 32 * 2)/ sizeof(bf16);
    const uint32_t total_blocks = (Dout / m) * (Din / k);
    const uint32_t data2move = block_size * total_blocks;
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        proj_arg_id,
        MM2S,
        IT5,
        npu_bd_id(bd_13),
        it_channel_0,
        {0, 0, 0, (uint32_t)w_offset},
        {1, 1, 1, (uint32_t)data2move},
        {0, 0, 0, 1},
        -1, 0, true, aggressive_cache
    );
    seq->npu_dma_wait(
        IT5,
        MM2S,
        it_channel_0
    );
}

void linear_decoding_layer::_move_linear_kv_cache(npu_sequence* seq){
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        kv_cache_arg_id,
        MM2S,
        IT5,
        bd_2,
        it_channel_1,
        {0, 0, 0, 0},
        {1, 1, 1, (uint32_t)(conv_kernel_size - 1) * (Linear_DQ + Linear_DK + Linear_DV)},
        {0, 0, 0, 1},
        -1, 0, true, no_cache
    );
    seq->npu_dma_wait(
        IT5,
        MM2S,
        it_channel_1
    );
}

void linear_decoding_layer::_move_s(npu_sequence* seq){
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        kv_cache_arg_id,
        MM2S,
        IT4,
        bd_2,
        it_channel_0,
        {0, 0, 0, (uint32_t)s_offset},
        {1, 1, 1, (uint32_t)(linear_num_v_heads * linear_v_head_dim * linear_qk_head_dim * 2)},
        {0, 0, 0, 1},
        -1, 0, false, no_cache
    );
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        kv_cache_arg_id,
        S2MM,
        IT4,
        bd_3,
        it_channel_0,
        {0, 0, 0, (uint32_t)s_offset},
        {1, 1, 1, (uint32_t)(linear_num_v_heads * linear_v_head_dim * linear_qk_head_dim * 2)},
        {0, 0, 0, 1},
        -1, 0, false, no_cache
    );
}

buffer<bf16> linear_decoding_layer::get_conv_cache(){
    buffer<bf16> conv_cache_buffer = buffer<bf16>((size_t)(conv_kernel_size - 1) * (Linear_DQ + Linear_DK + Linear_DV));
    this->kv_cache.sync_from_device();
    memcpy(conv_cache_buffer.data(), this->kv_cache.data(), (size_t)(conv_kernel_size - 1) * (Linear_DQ + Linear_DK + Linear_DV) * sizeof(bf16));
    return conv_cache_buffer;
}


buffer<bf16> linear_decoding_layer::get_S_cache(int head){
    buffer<bf16> s_cache_buffer = buffer<bf16>((size_t)linear_v_head_dim * linear_qk_head_dim);
    this->kv_cache.sync_from_device();
    float * s_begin = (float*)(this->kv_cache.data() + s_offset + head * linear_v_head_dim * linear_qk_head_dim * 2);
    for (int i = 0; i < linear_v_head_dim * linear_qk_head_dim; i++) {
        s_cache_buffer[i] = (bf16)s_begin[i];
    }
    return s_cache_buffer;
}

void linear_decoding_layer::clear_cache(){
    this->kv_cache.sync_from_device();
    this->kv_cache.memset((bf16)0.0f);
    this->kv_cache.sync_to_device();
}