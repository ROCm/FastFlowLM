// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include "whisper_decoder.hpp"

WhisperDecoder::WhisperDecoder(Whisper_Config config, npu_xclbin_manager *npu)
: config(config), npu(npu){
    this->embedding = std::make_unique<Embedding>(config.get("vocab_size"), config.get("hidden_size"));

    // register all xclbins
    this->layer_app_manager = npu->register_xclbin(utils::path_join(config.exec_path, "xclbins", config.model_name, "decoder_mv.xclbin"));

    this->proj_self_attn_qkv = this->layer_app_manager->create_app();
    this->proj_self_attn_o = this->layer_app_manager->create_app();
    this->proj_cross_attn_q = this->layer_app_manager->create_app();
    this->proj_cross_attn_o = this->layer_app_manager->create_app();
    this->self_attn_qkT = this->layer_app_manager->create_app();
    this->self_attn_sv = this->layer_app_manager->create_app();
    this->cross_attn_qkT = this->layer_app_manager->create_app();
    this->cross_attn_sv = this->layer_app_manager->create_app();
    this->proj_up = this->layer_app_manager->create_app();
    this->proj_down = this->layer_app_manager->create_app();

    // allocate weight buffers (CPU side)
    this->proj_weights.resize(num_hidden_layers);
    this->self_attn_norm_weights.resize(num_hidden_layers);
    this->self_attn_norm_biases.resize(num_hidden_layers);
    this->cross_attn_norm_weights.resize(num_hidden_layers);
    this->cross_attn_norm_biases.resize(num_hidden_layers);
    this->mlp_norm_weights.resize(num_hidden_layers);
    this->mlp_norm_biases.resize(num_hidden_layers);
    this->self_attn_q_biases.resize(num_hidden_layers);
    this->self_attn_v_biases.resize(num_hidden_layers);
    this->self_attn_o_biases.resize(num_hidden_layers);
    this->cross_attn_q_biases.resize(num_hidden_layers);
    this->cross_attn_o_biases.resize(num_hidden_layers);
    this->fc1_biases.resize(num_hidden_layers);
    this->fc2_biases.resize(num_hidden_layers);

    this->self_attn_k_cache.resize(num_hidden_layers);
    this->self_attn_v_cache.resize(num_hidden_layers);
    this->cross_attn_k_cache.resize(num_hidden_layers);
    this->cross_attn_v_cache.resize(num_hidden_layers);

    this->buf_q_o_d = this->proj_self_attn_qkv.create_bo_buffer<bf16>(D);
    this->buf_qkv = this->proj_self_attn_qkv.create_bo_buffer<bf16>(DQ + DK + DV);
    this->buf_self_attn_score = this->self_attn_qkT.create_bo_buffer<bf16>(num_key_value_heads * SELF_ATTN_L);
    this->buf_cross_attn_score = this->cross_attn_qkT.create_bo_buffer<bf16>(num_key_value_heads * CROSS_ATTN_L);
    this->buf_up = this->proj_up.create_bo_buffer<bf16>(intermediate_size);
    this->buf_x = this->proj_down.create_bo_buffer<bf16>(D);

    for (size_t i = 0; i < num_hidden_layers; i++){
        this->self_attn_k_cache[i] = this->self_attn_qkT.create_bo_buffer<bf16>(SELF_ATTN_L * DK);
        this->self_attn_v_cache[i] = this->self_attn_sv.create_bo_buffer<bf16>(SELF_ATTN_L * DV);
        this->cross_attn_k_cache[i] = this->cross_attn_qkT.create_bo_buffer<bf16>(CROSS_ATTN_L * DK);
        this->cross_attn_v_cache[i] = this->cross_attn_sv.create_bo_buffer<bf16>(CROSS_ATTN_L * DV);
        this->proj_weights[i] = this->proj_self_attn_qkv.create_bo_buffer<u8>(total_proj_weights_size);
    }

    for (size_t i = 0; i < num_hidden_layers; i++){
        this->self_attn_norm_weights[i] = buffer<bf16>(D);
        this->self_attn_norm_biases[i] = buffer<bf16>(D);
        this->cross_attn_norm_weights[i] = buffer<bf16>(D);
        this->cross_attn_norm_biases[i] = buffer<bf16>(D);
        this->mlp_norm_weights[i] = buffer<bf16>(D);
        this->mlp_norm_biases[i] = buffer<bf16>(D);
        this->self_attn_q_biases[i] = buffer<bf16>(D);
        this->self_attn_v_biases[i] = buffer<bf16>(D);
        this->self_attn_o_biases[i] = buffer<bf16>(D);
        this->cross_attn_q_biases[i] = buffer<bf16>(D);
        this->cross_attn_o_biases[i] = buffer<bf16>(D);
        this->fc1_biases[i] = buffer<bf16>(intermediate_size);
        this->fc2_biases[i] = buffer<bf16>(D);
    }
    this->positional_embedding = buffer<bf16>(SELF_ATTN_L * D);
    this->positional_embedding_tensor = tensor_2d<bf16>(this->positional_embedding, D);
    this->decoder_layer_norm_weights = buffer<bf16>(D);
    this->decoder_layer_norm_biases = buffer<bf16>(D);

    this->buf_q = buffer<bf16>(buf_qkv.data(), DQ);
    this->buf_k = buffer<bf16>(buf_qkv.data() + DQ, DK);
    this->buf_v = buffer<bf16>(buf_qkv.data() + DQ + DK, DV);

    this->residual_buffer = buffer<bf16>(D);
    this->residual_f32.assign(D, 0.0f);
    this->hidden_states_buffer = buffer<bf16>(D);

    prepare_proj(proj_self_attn_qkv.seq(), D, DQ + DK + DV, 0);
    prepare_proj(proj_self_attn_o.seq(), DQ, D, D * (DQ + DK + DV) * 5 / 8);
    prepare_proj(proj_cross_attn_q.seq(), D, DQ, D * (DQ + DK + DV + D) * 5 / 8);
    prepare_proj(proj_cross_attn_o.seq(), DQ, D, D * (DQ + DK + DV + D + DQ) * 5 / 8);
    prepare_proj(proj_up.seq(), D, intermediate_size, D * (DQ + DK + DV + D + DQ + D) * 5 / 8);
    prepare_proj(proj_down.seq(), intermediate_size, D, D * (DQ + DK + DV + D + DQ + D + intermediate_size) * 5 / 8);

    set_context_length(0);
}

WhisperDecoder::~WhisperDecoder() = default;

void WhisperDecoder::load_weights(Q4NX& q4nx){
    this->embedding->init_weights(&q4nx, "model.decoder.embed_tokens");
    q4nx.load_weights(this->positional_embedding, "model.decoder.embed_positions.weight");

    for (int layer_idx = 0; layer_idx < num_hidden_layers; layer_idx++){
        // load norm w & b
        q4nx.load_weights(this->self_attn_norm_weights[layer_idx], "model.decoder.layers." + std::to_string(layer_idx) + ".self_attn_layer_norm.weight");
        q4nx.load_weights(this->self_attn_norm_biases[layer_idx], "model.decoder.layers." + std::to_string(layer_idx) + ".self_attn_layer_norm.bias");
        q4nx.load_weights(this->cross_attn_norm_weights[layer_idx], "model.decoder.layers." + std::to_string(layer_idx) + ".encoder_attn_layer_norm.weight");
        q4nx.load_weights(this->cross_attn_norm_biases[layer_idx], "model.decoder.layers." + std::to_string(layer_idx) + ".encoder_attn_layer_norm.bias");
        q4nx.load_weights(this->mlp_norm_weights[layer_idx], "model.decoder.layers." + std::to_string(layer_idx) + ".final_layer_norm.weight");
        q4nx.load_weights(this->mlp_norm_biases[layer_idx], "model.decoder.layers." + std::to_string(layer_idx) + ".final_layer_norm.bias");

        // load all biases
        q4nx.load_weights(this->self_attn_q_biases[layer_idx], "model.decoder.layers." + std::to_string(layer_idx) + ".self_attn.q_proj.bias");
        q4nx.load_weights(this->self_attn_v_biases[layer_idx], "model.decoder.layers." + std::to_string(layer_idx) + ".self_attn.v_proj.bias");
        q4nx.load_weights(this->self_attn_o_biases[layer_idx], "model.decoder.layers." + std::to_string(layer_idx) + ".self_attn.out_proj.bias");
        q4nx.load_weights(this->cross_attn_q_biases[layer_idx], "model.decoder.layers." + std::to_string(layer_idx) + ".encoder_attn.q_proj.bias");
        q4nx.load_weights(this->cross_attn_o_biases[layer_idx], "model.decoder.layers." + std::to_string(layer_idx) + ".encoder_attn.out_proj.bias");
        q4nx.load_weights(this->fc1_biases[layer_idx], "model.decoder.layers." + std::to_string(layer_idx) + ".fc1.bias");
        q4nx.load_weights(this->fc2_biases[layer_idx], "model.decoder.layers." + std::to_string(layer_idx) + ".fc2.bias");

        // load proj_weights
        this->proj_weights[layer_idx].sync_from_device();
        load_all_decoder_proj_weights(this->proj_weights[layer_idx], q4nx, layer_idx);
        this->proj_weights[layer_idx].sync_to_device();
    }
    q4nx.load_weights(this->decoder_layer_norm_weights, "model.decoder.layer_norm.weight");
    q4nx.load_weights(this->decoder_layer_norm_biases, "model.decoder.layer_norm.bias");
}

void WhisperDecoder::load_cross_attn_kv_cache(buffer<bf16>& kv_cache){
    tensor_2d<bf16> kv_cache_tensor(kv_cache, 1536 * 1280);
    // 0: k_0
    // 1: v_0
    // 2: k_1
    // 3: v_1
    // ...
    // The copies and transposes of all layers are independent: split them into row slices over all
    // threads (the old single-threaded scalar transpose took ~30 ms per window).
    constexpr int slices = 16;
    // one flat loop over (layer, slice): OpenMP 2.0 compilers have no collapse()
    #pragma omp parallel for schedule(dynamic)
    for (int t = 0; t < num_hidden_layers * slices; t++){
        {
            const int i = t / slices, s = t % slices;
            const int r0 = CROSS_ATTN_L * s / slices, r1 = CROSS_ATTN_L * (s + 1) / slices;
            memcpy(this->cross_attn_k_cache[i].data() + (size_t)r0 * DK, kv_cache_tensor[2 * i].data() + (size_t)r0 * DK,
                   (size_t)(r1 - r0) * DK * sizeof(bf16));
            transpose_v_cache(this->cross_attn_v_cache[i].data(), kv_cache_tensor[2 * i + 1].data(), CROSS_ATTN_L, DV, CROSS_ATTN_L, r0, r1);
        }
    }
    for (int i = 0; i < num_hidden_layers; i++){
        this->cross_attn_k_cache[i].sync_to_device();
        this->cross_attn_v_cache[i].sync_to_device();
    }
}

buffer<bf16> WhisperDecoder::forward(int ids){
    int L_padded = (context_length + 1 + 63) / 64 * 64;
    prepare_qkt(self_attn_qkT.seq(), L_padded, DK, SELF_ATTN_L, 16, 64, 20, 0);
    prepare_sv(self_attn_sv.seq(), DV, SELF_ATTN_L, L_padded, 16, 64, 20, 0);
    prepare_qkt(cross_attn_qkT.seq(), CROSS_ATTN_L, DK, CROSS_ATTN_L, 16, 64, 20, 0);
    prepare_sv(cross_attn_sv.seq(), DV, CROSS_ATTN_L, CROSS_ATTN_L, 16, 64, 20, 0);
    // embed tokens
    this->embedding->forward(ids, this->residual_buffer);
    bf16_to_f32(residual_f32.data(), residual_buffer.data(), D);
    add_bf16_to_f32(residual_f32.data(), this->positional_embedding_tensor[context_length].data(), D);
    buffer<bf16> temporary_buffer(intermediate_size);
    buffer<bf16> temporary_score_buffer(buf_cross_attn_score.size());

    for (int layer_idx = 0; layer_idx < num_hidden_layers; layer_idx++){
        // step 1: layer norm
        apply_layernorm_f32(this->buf_x, residual_f32.data(), 1, this->self_attn_norm_weights[layer_idx], this->self_attn_norm_biases[layer_idx], 1e-5f);
        // qkv projection
        this->buf_x.sync_to_device();

        // this->proj_weights[layer_idx].sync_to_device();
        proj_self_attn_qkv(this->buf_qkv, this->proj_weights[layer_idx], this->buf_x);
        this->buf_qkv.sync_from_device();

        this->self_attn_k_cache[layer_idx].sync_from_device();
        memcpy(this->self_attn_k_cache[layer_idx].data() + context_length * DK, this->buf_k.data(), DK * sizeof(bf16));
        buf_qkv.sync_to_device();
        self_attn_k_cache[layer_idx].sync_to_device();
        // add biases
        residual_add(this->buf_q, this->buf_q, this->self_attn_q_biases[layer_idx]);
        residual_add(this->buf_v, this->buf_v, this->self_attn_v_biases[layer_idx]);

        // self attention
        // load new k into self_attn_k_cache

        buf_qkv.sync_to_device();

        self_attn_qkT(this->buf_self_attn_score, this->self_attn_k_cache[layer_idx], this->buf_qkv);
        this->buf_self_attn_score.sync_from_device();

        scale_and_mask_bf16_avx512(temporary_score_buffer.data(), this->buf_self_attn_score.data(), SCALING, num_key_value_heads, SELF_ATTN_L, context_length, L_padded);
        softmax_bf16_avx512(this->buf_self_attn_score.data(), temporary_score_buffer.data(), num_key_value_heads, L_padded, SELF_ATTN_L);
        this->buf_self_attn_score.sync_to_device();

        // // load new v into self_attn_v_cache
        this->self_attn_v_cache[layer_idx].sync_from_device();
        move_v(this->self_attn_v_cache[layer_idx].data(), this->buf_v.data(), context_length, DV, SELF_ATTN_L);
        this->self_attn_v_cache[layer_idx].sync_to_device();
        self_attn_sv(this->buf_x, this->self_attn_v_cache[layer_idx], this->buf_self_attn_score);
        this->buf_x.sync_from_device();

        // // o projection
        this->buf_x.sync_to_device();
        // this->proj_weights[layer_idx].sync_to_device();
        proj_self_attn_o(this->buf_q_o_d, this->proj_weights[layer_idx], this->buf_x);
        this->buf_q_o_d.sync_from_device();

        // // add o bias
        residual_add(this->buf_q_o_d, this->buf_q_o_d, this->self_attn_o_biases[layer_idx]);

        buf_q_o_d.sync_to_device();

        // // residual add
        add_bf16_to_f32(residual_f32.data(), this->buf_q_o_d.data(), D);

        // // step 2: layer norm
        apply_layernorm_f32(this->buf_x, residual_f32.data(), 1, this->cross_attn_norm_weights[layer_idx], this->cross_attn_norm_biases[layer_idx], 1e-5f);

        // // q projection
        this->buf_x.sync_to_device();
        // this->proj_weights[layer_idx].sync_to_device();
        proj_cross_attn_q(this->buf_q_o_d, this->proj_weights[layer_idx], this->buf_x);
        this->buf_q_o_d.sync_from_device();

        // add q bias
        residual_add(this->buf_q_o_d, this->buf_q_o_d, this->cross_attn_q_biases[layer_idx]);

        // // cross attention
        this->buf_q_o_d.sync_to_device();
        // this->cross_attn_k_cache[layer_idx].sync_to_device();
        cross_attn_qkT(this->buf_cross_attn_score, this->cross_attn_k_cache[layer_idx], this->buf_q_o_d);
        this->buf_cross_attn_score.sync_from_device();

        // Only the first WHISPER_AUDIO_FRAMES rows are audio; mask the padding
        // (the mask keeps indices <= its context_length argument).
        scale_and_mask_bf16_avx512(temporary_score_buffer.data(), this->buf_cross_attn_score.data(), SCALING, num_key_value_heads, CROSS_ATTN_L, WHISPER_AUDIO_FRAMES - 1, CROSS_ATTN_L);
        softmax_bf16_avx512(this->buf_cross_attn_score.data(), temporary_score_buffer.data(), num_key_value_heads, CROSS_ATTN_L, CROSS_ATTN_L);

        this->buf_cross_attn_score.sync_to_device();

        cross_attn_sv(this->buf_x, this->cross_attn_v_cache[layer_idx], this->buf_cross_attn_score);
        this->buf_x.sync_from_device();

        // // o projection
        this->buf_x.sync_to_device();
        // this->proj_weights[layer_idx].sync_to_device();
        proj_cross_attn_o(this->buf_q_o_d, this->proj_weights[layer_idx], this->buf_x);
        this->buf_q_o_d.sync_from_device();

        // // add o bias
        residual_add(this->buf_q_o_d, this->buf_q_o_d, this->cross_attn_o_biases[layer_idx]);

        buf_q_o_d.sync_to_device();

        // // residual add
        add_bf16_to_f32(residual_f32.data(), this->buf_q_o_d.data(), D);

        // // step 3: layer norm
        apply_layernorm_f32(this->buf_x, residual_f32.data(), 1, this->mlp_norm_weights[layer_idx], this->mlp_norm_biases[layer_idx], 1e-5f);

        // // up gate projection
        this->buf_x.sync_to_device();
        this->buf_up.sync_from_device();
        // this->proj_weights[layer_idx].sync_to_device();
        proj_up(this->buf_up, this->proj_weights[layer_idx], this->buf_x);
        this->buf_up.sync_from_device();

        // add up bias
        residual_add(temporary_buffer, this->buf_up, this->fc1_biases[layer_idx]);

        // gelu
        apply_gelu_avx512(this->buf_up, temporary_buffer);

        // down projection
        this->buf_up.sync_to_device();
        proj_down(this->buf_x, this->proj_weights[layer_idx], this->buf_up);
        this->buf_x.sync_from_device();

        // add down bias
        add_bf16_to_f32(residual_f32.data(), this->buf_x.data(), D);
        // residual add
        add_bf16_to_f32(residual_f32.data(), this->fc2_biases[layer_idx].data(), D);
    }
    this->context_length++;
    // bf16 copy for callers; final_layernorm() reads the fp32 stream itself
    for (int i = 0; i < D; i++) this->residual_buffer[i] = bf16(residual_f32[i]);
    return this->residual_buffer;
}

// for general MV, qkv projection etc.
void WhisperDecoder::prepare_proj(npu_sequence* seq, const uint32_t D_in, const uint32_t D_out, const uint32_t weight_offset){
    constexpr int total_cols = 4;
    constexpr int total_rows = 4;
    constexpr npu_tiles IT[4] = {IT2, IT3, IT4, IT5};
    seq->clear_cmds();
    seq->npu_preemption(0);

    npu_sequence& seq_ptr = *seq;

    constexpr int m_tile = 32;
    constexpr int k_tile = 128;

    const uint32_t block_size_in_byte = m_tile * k_tile * 5 / 8;
    const uint32_t blocks_per_row = D_in / k_tile;
    assert(D_in % k_tile == 0);
    const uint32_t quant_in_per_column = 4 * blocks_per_row * block_size_in_byte;
    const uint32_t row_per_round = 4 * m_tile * total_cols;
    const uint32_t down_rounds = (D_out + row_per_round - 1) / row_per_round;
    // std::cout << "down_rounds: " << down_rounds << std::endl;
    const uint32_t total_column_rounds = D_out / 4 / m_tile;

    int bd_wait_counter[total_cols] = {0, 0, 0, 0};

    int K_div_k = D_in / k_tile;
    constexpr uint32_t CT_lock_address_base = 0x000001F000;
    constexpr uint32_t CT_rtp_sync_lock_id = 6;   // 0x1F000 + 16*6 == 0x1F060
    for(int col = 2; col < total_cols + 2; col++){
        uint32_t compute_column_needed = D_out/(4 * m_tile);
        uint32_t total_col_div_CT_cols = compute_column_needed/total_cols;
        uint32_t total_col_mode_CT_cols = compute_column_needed %total_cols;
        uint32_t cur_col_M_div_m = total_col_div_CT_cols;

        if( col < total_col_mode_CT_cols){
            cur_col_M_div_m += 1;
        }

        for(int row = 2; row < 2+total_rows; row++){
            npu_tiles tile = get_tile(row, col);
            seq->rtp_write(tile, proj_M_div_m_rtp_address, cur_col_M_div_m);
            seq->rtp_write(tile, proj_CT_rtp_address, K_div_k);
            seq->rtp_write(tile, CT_lock_address_base + 16 * CT_rtp_sync_lock_id, 1); //set rtp finished lock
        }
    }

    for (int i = 0; i < down_rounds; i++){
        for (int col = 0; col < total_cols; col++){
            uint32_t bd_offset = (i % 2) * 8; // NOTE: 8 is because of the IT ping-pong, bd_id = 16
            uint32_t round_offset = i * total_cols + col; //NOTE: column offset

            if (round_offset < total_column_rounds){
                // receive y
                uint32_t y_offset = round_offset * 4 * m_tile;
                seq_ptr.npu_dma_memcpy_nd(
                    2, 0,
                    S2MM, IT[col],
                    (npu_bd_id)(bd_offset + 0), it_channel_0,
                    {0, 0, 0, (uint32_t)y_offset},
                    {1, 1, 1, (uint32_t)4 * m_tile},
                    {0, 0, 0, 1},
                    -1, 0, true
                );

                // send w (matrix w)
                uint32_t w_offset = round_offset * quant_in_per_column + weight_offset;
                seq_ptr.npu_dma_memcpy_nd(
                    sizeof(char), 1,
                    MM2S, IT[col],
                    (npu_bd_id)(bd_offset + 1), it_channel_0,
                    {0, 0, 0, (uint32_t)w_offset},
                    {blocks_per_row, 4, 4, block_size_in_byte / 4},
                    {block_size_in_byte, block_size_in_byte * blocks_per_row, block_size_in_byte / 4, 1},
                    -1, 0, false
                );

                // send x (vector x)
                uint32_t x_offset = 0;
                seq_ptr.npu_dma_memcpy_nd(
                    2, 2,
                    MM2S, IT[col],
                    (npu_bd_id)(bd_offset + 2), it_channel_1,
                    {0, 0, 0, (uint32_t)x_offset},
                    {1, 1, 1, (uint32_t)D_in},
                    {0, 0, 0, 1},
                    -1, 0, false
                );
                bd_wait_counter[col]++;
            }
        }
        for (int col = 0; col < total_cols; col++){
            if(bd_wait_counter[col] == 2){
                seq_ptr.npu_dma_wait(IT[col], S2MM, it_channel_0);
                // std::cout << "wait dma " << col << std::endl;
                bd_wait_counter[col]--;
            }
        }
    } // end for i
    int max_loop_iterations = 0;
    // find the global max loop iterations
    for(int col = 0; col < total_cols; col++){
        max_loop_iterations = std::max(max_loop_iterations, bd_wait_counter[col]);
    }
    // std::cout << "max loop iterations: " << max_loop_iterations << std::endl;
    for(int i = 0; i < max_loop_iterations; i++){
        for(int col = 0; col < total_cols; col++){
            if (bd_wait_counter[col] > 0){
                seq_ptr.npu_dma_wait(IT[col], S2MM, it_channel_0);
                // std::cout << "wait dma " << col << std::endl;
                bd_wait_counter[col]--;
            }
        }
    }
    seq->cmds2seq();
}

///@brief Do q * k'
///@note q: (1 * DQ) -> (20, 64)
///@note k: (L_MAX * DQ) -> (L_MAX, 20, 64)
///@note y: (20, 1, L_padded), 20 is the number of attention heads or batch
///@note W -> k
///@note X -> q
///@note output s: (20, 1, L_MAX)
///@param seq: npu_sequence pointer
///@param L_padded: L_padded, real useful L in k cache (was named as M)
///@param DK: DK (was named as K)
///@param L_MAX: L_MAX, the dimension of k cache
///@param m: m, tiling size for L
///@param k: k, tiling size for DQ
///@param batch_size: batch_size, or number of attention heads
///@param weight_offset: weight offset
void WhisperDecoder::prepare_qkt(
    npu_sequence* seq, const uint32_t L_padded, const uint32_t DK, const uint32_t L_MAX,
    const uint32_t m, const uint32_t k,  // size process for each CT block
    const uint32_t batch_size, const uint32_t weight_offset
){

    constexpr uint32_t Y_ARG_IDX = 0;
    constexpr uint32_t W_ARG_IDX = 1;
    constexpr uint32_t X_ARG_IDX = 2;

    //NOTE: although is in batch MVM, we assume M is in row-major order, not affect by the batch
    assert( DK % batch_size == 0);
    const uint32_t K_total_per_batch = DK / batch_size; // shall be 64

    assert(K_total_per_batch % k  == 0 );

    constexpr int total_cols = 4;
    constexpr int total_rows = 4;
    constexpr npu_tiles IT[total_cols] = {IT0, IT1, IT6, IT7};
    int bd_wait_counter[total_cols] = {0, 0, 0, 0};

    seq->clear_cmds();
    seq->npu_preemption(0);

    npu_sequence& seq_ptr = *seq;
    uint32_t K_batch_div_k = K_total_per_batch / k;

    assert(L_padded % (4 * m) == 0);
    uint32_t col_per_batch = L_padded / (4 * m);
    uint32_t compute_col_needed = batch_size * col_per_batch;

    constexpr uint32_t CT_lock_address_base = 0x000001F000;
    constexpr uint32_t CT_rtp_sync_lock_id = 6;   // 0x1F000 + 16*6 == 0x1F060
    for(int col = 0; col < 2; col++){

        uint32_t total_col_div_CT_cols = compute_col_needed / total_cols;
        uint32_t total_col_mode_CT_cols = compute_col_needed % total_cols;
        uint32_t cur_col_M_div_m = total_col_div_CT_cols;

        if( col < total_col_mode_CT_cols){
            cur_col_M_div_m += 2;
        }

        for(int row = 2; row < 2 + total_rows; row++){
            npu_tiles tile = get_tile(row, col);
            npu_tiles tile_6 = get_tile(row, col + 6);
            seq->rtp_write(tile, attn_M_div_m_rtp_address, cur_col_M_div_m);
            seq->rtp_write(tile_6, attn_M_div_m_rtp_address, cur_col_M_div_m);
            seq->rtp_write(tile, attn_CT_rtp_address, K_batch_div_k);
            seq->rtp_write(tile_6, attn_CT_rtp_address, K_batch_div_k);
            seq->rtp_write(tile, CT_lock_address_base + 16 * CT_rtp_sync_lock_id, 1); //set rtp finished lock
            seq->rtp_write(tile_6, CT_lock_address_base + 16 * CT_rtp_sync_lock_id, 1); //set rtp finished lock
        }
    }

    uint32_t round = _ceil_to_multiple(compute_col_needed, total_cols);
    for(int i = 0; i <round / total_cols; i++){

        for(int col = 0; col < total_cols; col++){
            uint32_t bd_offset = (i % 2) * 8;
            uint32_t CT_col_idx = i * total_cols + col; //NOTE: column offset

            uint32_t cur_batch  =  CT_col_idx / col_per_batch;
            uint32_t cur_batch_col_index = CT_col_idx % col_per_batch;
            if(CT_col_idx < compute_col_needed ){
                uint32_t w_off = cur_batch * K_total_per_batch + cur_batch_col_index * (4 * m) * DK + weight_offset;// This assume W is in pure row-major order
                uint32_t x_off = cur_batch * K_total_per_batch;
                uint32_t y_off = cur_batch * L_MAX             + cur_batch_col_index * (4 * m);

                // Y
                seq->npu_dma_memcpy_nd(
                    sizeof(uint16_t), Y_ARG_IDX,
                    S2MM, IT[col],
                    (npu_bd_id)(bd_offset + 0), it_channel_0,
                    {0,0,0, y_off},
                    {1, 1, 1, 4 * m},
                    {0, 0, 0, 1},
                    -1, 0, true
                );
                // W
                seq->npu_dma_memcpy_nd(
                    sizeof(uint16_t), W_ARG_IDX,
                    MM2S, IT[col],
                    (npu_bd_id)(bd_offset + 1), it_channel_0,
                    {0, 0, 0, w_off},
                    {1, 1, 4u * m, K_total_per_batch},
                    {0, 0, DK, 1u},
                    -1, 0, false
                );
                seq->npu_dma_memcpy_nd(
                    sizeof(uint16_t), X_ARG_IDX,
                    MM2S, IT[col],
                    (npu_bd_id)(bd_offset + 2), it_channel_1,
                    {0, 0, 0, x_off},
                    {1, 1, 1, K_total_per_batch},
                    {0, 0, 0, 1},
                    -1, 0, false
                );
                bd_wait_counter[col]++;
            }
        }
        for (int col = 0; col < total_cols; col++){
            if(bd_wait_counter[col] == 1){
                seq_ptr.npu_dma_wait(IT[col], S2MM, it_channel_0);
                bd_wait_counter[col]--;
            }
        }
    }
    int max_loop_iterations = 0;
    // find the global max loop iterations
    for(int col = 0; col < total_cols; col++){
        max_loop_iterations = std::max(max_loop_iterations, bd_wait_counter[col]);
    }
    for(int i = 0; i < max_loop_iterations; i++){
        for(int col = 0; col < total_cols; col++){
            if (bd_wait_counter[col] > 0){
                seq_ptr.npu_dma_wait(IT[col], S2MM, it_channel_0);
                bd_wait_counter[col]--;
            }
        }
    }
    seq->cmds2seq();
}

///@brief Do s * v
///@note s: (20, L_MAX)
///@note v: (L_MAX * DV) -> (L_MAX, 20, 64)
///@note y: (20, 1, 64), 20 is the number of attention heads or batch
///@note W -> v
///@note X -> s
///@param seq: npu_sequence pointer
///@param L_padded: L_padded, real useful L in k cache (was named as K)
///@param DV: DV (was named as M)
///@param L_MAX: L_MAX, the dimension of k cache
///@param m: m, tiling size for DV
///@param k: k, tiling size for L_MAX
///@param batch_size: batch_size, or number of attention heads
///@param weight_offset: weight offset
void WhisperDecoder::prepare_sv(
    npu_sequence* seq, const uint32_t DV, const uint32_t L_MAX, const uint32_t L_padded,
    const uint32_t m, const uint32_t k,  // size process for each CT block
    const uint32_t batch_size, const uint32_t weight_offset
){
    // Y -> s v
    // W -> v
    // X -> s
    constexpr uint32_t Y_ARG_IDX = 0;
    constexpr uint32_t W_ARG_IDX = 1;
    constexpr uint32_t X_ARG_IDX = 2;

    // still assume W(M x K) are in pure row-major format
    const int M_total_per_batch = DV / batch_size;

    assert(DV % (4 * m) == 0);
    assert(M_total_per_batch % (4 * m) == 0);

    constexpr int total_cols = 4;
    constexpr int total_rows = 4;
    constexpr npu_tiles IT[total_cols] = {IT0, IT1, IT6, IT7};
    int bd_wait_counter[total_cols] = {0, 0, 0, 0};

    seq->clear_cmds();
    seq->npu_preemption(0);
    uint32_t K_div_k = L_padded / k;
    assert( L_padded % k == 0);

    uint32_t col_per_batch = M_total_per_batch / (4 * m);
    uint32_t compute_col_needed = batch_size * col_per_batch;

    constexpr uint32_t CT_lock_address_base = 0x000001F000;
    constexpr uint32_t CT_rtp_sync_lock_id = 6;   // 0x1F000 + 16*6 == 0x1F060
    for(int col = 0; col < 2; col++){

        uint32_t total_col_div_CT_cols = compute_col_needed / total_cols;
        uint32_t total_col_mode_CT_cols = compute_col_needed % total_cols;
        uint32_t cur_col_M_div_m = total_col_div_CT_cols;

        if( col < total_col_mode_CT_cols){
            cur_col_M_div_m += 2;
        }

        for(int row = 2; row < 2+total_rows; row++){
            npu_tiles tile = get_tile(row, col);
            npu_tiles tile_6 = get_tile(row, col+6);
            seq->rtp_write(tile, attn_M_div_m_rtp_address, cur_col_M_div_m);
            seq->rtp_write(tile_6, attn_M_div_m_rtp_address, cur_col_M_div_m);
            seq->rtp_write(tile, attn_CT_rtp_address, K_div_k);
            seq->rtp_write(tile_6, attn_CT_rtp_address, K_div_k);
            seq->rtp_write(tile, CT_lock_address_base + 16 * CT_rtp_sync_lock_id, 1); //set rtp finished lock
            seq->rtp_write(tile_6, CT_lock_address_base + 16 * CT_rtp_sync_lock_id, 1); //set rtp finished lock
        }
    }

    uint32_t round = _ceil_to_multiple(compute_col_needed,total_cols );
    for(int i = 0; i <round/total_cols; i++){

        for(int col = 0; col < total_cols; col++){
            uint32_t bd_offset = (i % 2) * 8;
            uint32_t CT_col_idx = i * total_cols + col; //NOTE: column offset

            uint32_t cur_batch  =  CT_col_idx / col_per_batch;
            uint32_t cur_batch_col_index = CT_col_idx % col_per_batch;
            if(CT_col_idx < compute_col_needed ){

                uint32_t w_off = cur_batch * L_MAX * (M_total_per_batch) + cur_batch_col_index * L_MAX * (4 * m) + weight_offset;// This assume W is in pure row-major order
                uint32_t x_off = cur_batch *  L_MAX;
                uint32_t y_off = cur_batch * M_total_per_batch + cur_batch_col_index * (4 * m);

                // Y
                seq->npu_dma_memcpy_nd(
                    sizeof(uint16_t), Y_ARG_IDX,
                    S2MM, IT[col],
                    (npu_bd_id)(bd_offset + 0), it_channel_0,
                    {0, 0, 0, y_off},
                    {1, 1, 1, 4 * m},
                    {0, 0, 0, 1},
                    -1, 0, true
                );
                // W
                seq->npu_dma_memcpy_nd(
                    sizeof(uint16_t), W_ARG_IDX,
                    MM2S, IT[col],
                    (npu_bd_id)(bd_offset + 1), it_channel_0,
                    {0,0,0, w_off},
                    {1, K_div_k, (4 * m),  k},
                    {0, k,        L_MAX,   1},
                    -1,0,false
                );
                seq->npu_dma_memcpy_nd(
                    sizeof(uint16_t), X_ARG_IDX,
                    MM2S, IT[col],
                    (npu_bd_id)(bd_offset + 2), it_channel_1,
                    {0,0,0, x_off},
                    {1, 1, 1, L_padded},
                    {0, 0, 0, 1},
                    -1, 0, false
                );
                bd_wait_counter[col]++;
            }
        }
        for (int col = 0; col < total_cols; col++){
            if(bd_wait_counter[col] == 1){
                seq->npu_dma_wait(IT[col], S2MM, it_channel_0);
                bd_wait_counter[col]--;
            }
        }
    }
    int max_loop_iterations = 0;
    // find the global max loop iterations
    for(int col = 0; col < total_cols; col++){
        max_loop_iterations = std::max(max_loop_iterations, bd_wait_counter[col]);
    }
    for(int i = 0; i < max_loop_iterations; i++){
        for(int col = 0; col < total_cols; col++){
            if (bd_wait_counter[col] > 0){
                seq->npu_dma_wait(IT[col], S2MM, it_channel_0);
                bd_wait_counter[col]--;
            }
        }
    }

    seq->cmds2seq();
}
