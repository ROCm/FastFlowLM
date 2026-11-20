// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#include "whisper_encoder.hpp"
#include "whisper_prof.hpp"

WhisperEncoder::WhisperEncoder(Whisper_Config config, npu_xclbin_manager *npu)
: config(config), npu(npu){
    
    // register all xclbins
    this->mm_xclbin        = npu->register_xclbin(utils::path_join(config.exec_path, "xclbins", config.model_name, "encoder_mm.xclbin"));
    this->attention_xclbin = npu->register_xclbin(utils::path_join(config.exec_path, "xclbins", config.model_name, "encoder_attn.xclbin"));
    this->dequant_xclbin   = npu->register_xclbin(utils::path_join(config.exec_path, "xclbins", config.model_name, "encoder_dequant.xclbin"));

    this->q_proj = this->mm_xclbin->create_app();
    this->k_proj = this->mm_xclbin->create_app();
    this->v_proj = this->mm_xclbin->create_app();
    this->o_proj = this->mm_xclbin->create_app();
    this->up_proj = this->mm_xclbin->create_app();
    this->down_proj = this->mm_xclbin->create_app();
    this->conv1_mm = this->mm_xclbin->create_app();
    this->conv2_mm = this->mm_xclbin->create_app();
    this->conv1_x = this->conv1_mm.create_bo_buffer<bf16>((size_t)CONV1_ROWS * 128);
    this->conv1_w = this->conv1_mm.create_bo_buffer<bf16>((size_t)1280 * CONV1_K);
    this->conv2_x = this->conv2_mm.create_bo_buffer<bf16>((size_t)CONV1_ROWS * 1280);
    this->conv2_w = this->conv2_mm.create_bo_buffer<bf16>((size_t)1280 * CONV2_K);
    for (int h = 0; h < 2; h++){
        this->o_half[h] = this->mm_xclbin->create_app();
        this->up_half[h] = this->mm_xclbin->create_app();
        this->down_half[h] = this->mm_xclbin->create_app();
    }

    this->k_cache_proj = this->mm_xclbin->create_app();
    this->v_cache_proj = this->mm_xclbin->create_app();

    this->mha_engine = this->attention_xclbin->create_app();

    this->dequant_qkv = this->dequant_xclbin->create_app();
    this->dequant_o = this->dequant_xclbin->create_app();
    this->dequant_up = this->dequant_xclbin->create_app();
    this->dequant_down = this->dequant_xclbin->create_app();
    this->dequant_kv_cache = this->dequant_xclbin->create_app();

    // allocate computing buffers
    this->buf_q_o_d = this->q_proj.create_bo_buffer<bf16>(L * D);
    this->buf_up = this->up_proj.create_bo_buffer<bf16>(L * intermediate_size);
    this->kv_cache = this->mha_engine.create_bo_buffer<bf16>(kv_cache_size);
    this->buf_x = this->mha_engine.create_bo_buffer<bf16>(L * D);

    // allocate weight buffers (CPU side)
    this->proj_weights.resize(num_hidden_layers);
    this->attn_norm_weights.resize(num_hidden_layers);
    this->attn_norm_biases.resize(num_hidden_layers);
    this->mlp_norm_weights.resize(num_hidden_layers);
    this->mlp_norm_biases.resize(num_hidden_layers);
    this->q_biases.resize(num_hidden_layers);
    this->v_biases.resize(num_hidden_layers);
    this->o_biases.resize(num_hidden_layers);
    this->fc1_biases.resize(num_hidden_layers);
    this->fc2_biases.resize(num_hidden_layers);

    this->cross_attn_kv_proj_weights.resize(WHISPER_DECODER_LAYERS);
    this->cross_attn_v_proj_biases.resize(WHISPER_DECODER_LAYERS);

    for (size_t i = 0; i < num_hidden_layers; i++){
        this->proj_weights[i] = this->dequant_qkv.create_bo_buffer<u8>(total_proj_weights_size);
    }

    for (size_t i = 0; i < WHISPER_DECODER_LAYERS; i++){
        this->cross_attn_kv_proj_weights[i] = dequant_kv_cache.create_bo_buffer<u8>(DK * DK * 5 / 8 + DV * DV * 5 / 8);
    }
    
    this->weights_qkv = this->dequant_qkv.create_bo_buffer<bf16>(weight_q_k_v_size);
    this->weights_o = this->dequant_o.create_bo_buffer<bf16>(weight_o_size);
    this->weights_up = this->dequant_up.create_bo_buffer<bf16>(weight_up_size);
    this->weights_down = this->dequant_down.create_bo_buffer<bf16>(weight_down_size);
    
    this->buf_attn_output = this->mha_engine.create_bo_buffer<bf16>(1536 * 1280);

    this->KV_CACHE = buffer<bf16>(WHISPER_DECODER_LAYERS * (L * DK + L * DV));
    memset(this->KV_CACHE.data(), 0, this->KV_CACHE.size() * sizeof(bf16));  // rows past a short window are masked but must be finite
    this->buf_k_cache = this->k_cache_proj.create_bo_buffer<bf16>(L * DK);
    this->buf_v_cache = this->v_cache_proj.create_bo_buffer<bf16>(L * DV);
    this->buf_kv_cache_proj_weight = this->dequant_kv_cache.create_bo_buffer<bf16>(DK * DK + DV * DV);

    for (size_t i = 0; i < WHISPER_DECODER_LAYERS; i++){
        this->cross_attn_v_proj_biases[i] = buffer<bf16>(DV);
    }

    for (size_t i = 0; i < num_hidden_layers; i++){
        this->attn_norm_weights[i] = buffer<bf16>(hidden_size);
        this->attn_norm_biases[i] = buffer<bf16>(hidden_size);
        this->mlp_norm_weights[i] = buffer<bf16>(hidden_size);
        this->mlp_norm_biases[i] = buffer<bf16>(hidden_size);
        this->q_biases[i] = buffer<bf16>(hidden_size);
        this->v_biases[i] = buffer<bf16>(hidden_size);
        this->o_biases[i] = buffer<bf16>(hidden_size);
        this->fc1_biases[i] = buffer<bf16>(intermediate_size);
        this->fc2_biases[i] = buffer<bf16>(hidden_size);
    }
    this->conv1d_1_weight = buffer<bf16>(1280 * 128 * 3);
    this->conv1d_1_bias   = buffer<bf16>(1280);
    this->conv1d_2_weight = buffer<bf16>(1280 * 1280 * 3);
    this->conv1d_2_bias   = buffer<bf16>(1280);

    this->positional_embedding = buffer<bf16>(1536 * 1280);

    this->encoder_out = buffer<bf16>(1536 * 1280);

    this->residual_buffer = buffer<bf16>(1536 * 1280);
    this->residual_f32.assign((size_t)1536 * 1280, 0.0f);

    
    prepare_sequences();
    for (int h = 0; h < 2; h++){
        const int H = L / 2;
        prepare_mm(o_half[h].seq(), H, DQ, D, 96, 256, 64, 0, h * H * D, h * H * DQ);
        prepare_mm(up_half[h].seq(), H, D, intermediate_size, 96, 256, 64, 0, h * H * intermediate_size, h * H * D);
        prepare_mm(down_half[h].seq(), H, intermediate_size, D, 96, 256, 64, 0, h * H * D, h * H * intermediate_size);
    }
    prepare_dequant(dequant_qkv.seq(), D, DQ + DK + DV, 0);
    prepare_dequant(dequant_o.seq(), D, D, D * (DQ + DK + DV) * 5 / 8);
    prepare_dequant(dequant_up.seq(), D, intermediate_size, D * (DQ + DK + DV + D) * 5 / 8);
    prepare_dequant(dequant_down.seq(), intermediate_size, D, D * (DQ + DK + DV + DQ + intermediate_size) * 5 / 8);
    prepare_dequant(dequant_kv_cache.seq(), DK, DK + DV, 0);
}

WhisperEncoder::~WhisperEncoder(){
}

void WhisperEncoder::load_weights(Q4NX& q4nx){
    q4nx.load_weights(this->conv1d_1_weight, "model.encoder.conv1.weight");
    q4nx.load_weights(this->conv1d_1_bias,   "model.encoder.conv1.bias");
    q4nx.load_weights(this->conv1d_2_weight, "model.encoder.conv2.weight");
    q4nx.load_weights(this->conv1d_2_bias,   "model.encoder.conv2.bias");

    q4nx.load_weights(this->positional_embedding, "model.encoder.embed_positions.weight");
    pack_conv_weights();

    for (int layer_idx = 0; layer_idx < num_hidden_layers; layer_idx++){
        // load norm w & b
        q4nx.load_weights(this->attn_norm_weights[layer_idx], "model.encoder.layers." + std::to_string(layer_idx) + ".self_attn_layer_norm.weight");
        q4nx.load_weights(this->attn_norm_biases[layer_idx], "model.encoder.layers." + std::to_string(layer_idx) + ".self_attn_layer_norm.bias");
        q4nx.load_weights(this->mlp_norm_weights[layer_idx], "model.encoder.layers." + std::to_string(layer_idx) + ".final_layer_norm.weight");
        q4nx.load_weights(this->mlp_norm_biases[layer_idx], "model.encoder.layers." + std::to_string(layer_idx) + ".final_layer_norm.bias");

        // load all biases
        q4nx.load_weights(this->q_biases[layer_idx], "model.encoder.layers." + std::to_string(layer_idx) + ".self_attn.q_proj.bias");
        q4nx.load_weights(this->v_biases[layer_idx], "model.encoder.layers." + std::to_string(layer_idx) + ".self_attn.v_proj.bias");
        q4nx.load_weights(this->o_biases[layer_idx], "model.encoder.layers." + std::to_string(layer_idx) + ".self_attn.out_proj.bias");
        q4nx.load_weights(this->fc1_biases[layer_idx], "model.encoder.layers." + std::to_string(layer_idx) + ".fc1.bias");
        q4nx.load_weights(this->fc2_biases[layer_idx], "model.encoder.layers." + std::to_string(layer_idx) + ".fc2.bias");

        // load proj_weights
        this->proj_weights[layer_idx].sync_from_device();
        load_all_proj_weights(this->proj_weights[layer_idx], q4nx, layer_idx);
        this->proj_weights[layer_idx].sync_to_device();
    }

    for (int layer_idx = 0; layer_idx < WHISPER_DECODER_LAYERS; layer_idx++){
        buffer<u8> k_weight;
        buffer<u8> v_weight;
        u8* w_ptr = this->cross_attn_kv_proj_weights[layer_idx].data();
        q4nx.load_weights(k_weight, "model.decoder.layers." + std::to_string(layer_idx) + ".encoder_attn.k_proj.weight");
        q4nx.load_weights(v_weight, "model.decoder.layers." + std::to_string(layer_idx) + ".encoder_attn.v_proj.weight");
        memcpy(w_ptr, k_weight.data(), DK * DK * 5 / 8);
        memcpy(w_ptr + DK * DK * 5 / 8, v_weight.data(), DV * DV * 5 / 8);
        this->cross_attn_kv_proj_weights[layer_idx].sync_to_device();
        q4nx.load_weights(this->cross_attn_v_proj_biases[layer_idx], "model.decoder.layers." + std::to_string(layer_idx) + ".encoder_attn.v_proj.bias");
    }

    // fp32 copies of the fc1/fc2 biases for the fused passes; o's is made on first use (fold_v_bias)
    fc1_bias_f32.assign(num_hidden_layers, {});
    fc2_bias_f32.assign(num_hidden_layers, {});
    o_bias_f32.assign(num_hidden_layers, {});
    for (int layer_idx = 0; layer_idx < num_hidden_layers; layer_idx++){
        fc1_bias_f32[layer_idx].assign(fc1_biases[layer_idx].begin(), fc1_biases[layer_idx].end());
        fc2_bias_f32[layer_idx].assign(fc2_biases[layer_idx].begin(), fc2_biases[layer_idx].end());
    }

    q4nx.load_weights(this->encoder_layer_norm_weights, "model.encoder.layer_norm.weight");
    q4nx.load_weights(this->encoder_layer_norm_biases, "model.encoder.layer_norm.bias");
}

buffer<bf16> WhisperEncoder::forward(buffer<bf16>& mel_feature){
    PROF_START();
    // Always the 30 s window: 1500 encoder rows of audio, padded to L = 1536.
    constexpr int rows = L;
    constexpr int valid_rows = WHISPER_AUDIO_FRAMES;

    conv_front_end(mel_feature);
    PROF_LAP("embed");

    // residual_f32 keeps the residual stream; each layer norm writes straight into buf_x
    for (int layer_idx = 0; layer_idx < num_hidden_layers; layer_idx++){
        // dequant all projection weights, into this layer's cache slot if the
        // cache is on (allocated on first use), else into the shared buffers
        if (cache_dequant && !dequant_cached[layer_idx]) {
            cache_qkv[layer_idx] = this->dequant_qkv.create_bo_buffer<bf16>(weight_q_k_v_size);
            cache_o[layer_idx] = this->dequant_o.create_bo_buffer<bf16>(weight_o_size);
            cache_up[layer_idx] = this->dequant_up.create_bo_buffer<bf16>(weight_up_size);
            cache_down[layer_idx] = this->dequant_down.create_bo_buffer<bf16>(weight_down_size);
        }
        buffer<bf16>& w_qkv = cache_dequant ? cache_qkv[layer_idx] : weights_qkv;
        buffer<bf16>& w_o = cache_dequant ? cache_o[layer_idx] : weights_o;
        buffer<bf16>& w_up = cache_dequant ? cache_up[layer_idx] : weights_up;
        buffer<bf16>& w_down = cache_dequant ? cache_down[layer_idx] : weights_down;
        if (!cache_dequant || !dequant_cached[layer_idx]) {
            dequant_qkv(w_qkv, this->proj_weights[layer_idx]);
            dequant_o(w_o, this->proj_weights[layer_idx]);
            dequant_up(w_up, this->proj_weights[layer_idx]);
            dequant_down(w_down, this->proj_weights[layer_idx]);
            dequant_cached[layer_idx] = true;
        }
        PROF_LAP("dequant");

        // The V bias is folded into the o-proj bias once per layer (see fold_v_bias).
        if (o_bias_f32[layer_idx].empty()) fold_v_bias(layer_idx, w_o);

        // Step 1: layer norm straight into the NPU input. The padding rows are zeroed: k has no
        // bias and v's bias is folded out, so their keys and values come out zero as well.
        if (ln1_first_half_done) {   // rows 0..L/2-1 were done during the previous layer's last matmul
            const int H = L / 2;
            apply_layernorm_f32(buf_x.data() + (size_t)H * D, residual_f32.data() + (size_t)H * D, rows - H,
                                this->attn_norm_weights[layer_idx], this->attn_norm_biases[layer_idx], 1e-5f);
            ln1_first_half_done = false;
        } else {
            apply_layernorm_f32(buf_x, residual_f32.data(), rows, this->attn_norm_weights[layer_idx], this->attn_norm_biases[layer_idx], 1e-5f);
        }
        if (rows > valid_rows) memset(buf_x.data() + (size_t)valid_rows * D, 0, (size_t)(rows - valid_rows) * D * sizeof(bf16));
        buf_x.sync_to_device();
        PROF_LAP("ln1");

        // Step 2: q, then k and v straight into the two halves of the attention's kv_cache.
        // The q bias is added on the CPU while the NPU computes k.
        q_proj(buf_q_o_d, buf_x, w_qkv);
        PROF_LAP("mm_q");
        auto k_run = k_proj.create_run(kv_cache, buf_x, w_qkv);
        k_run.start();
        buf_q_o_d.sync_from_device();
        add_bias_optimized(buf_q_o_d, buf_q_o_d, q_biases[layer_idx], rows);
        buf_q_o_d.sync_to_device();
        PROF_LAP("q_bias");
        flm::npu::wait_for_completion(k_run, "k_run");
        PROF_LAP("mm_k");
        v_proj(kv_cache, buf_x, w_qkv);
        PROF_LAP("mm_v");

        // Step 3: attention
        mha_engine(buf_attn_output, buf_q_o_d, kv_cache);
        PROF_LAP("mha");

        // Steps 4-5 (o-proj, fc1, fc2 and their CPU parts) on two row halves
        post_attention_split(layer_idx, w_o, w_up, w_down);
    }

    // prepare kv cache for cross attention
    // memcpy(buf_x.data(), residual_buffer.data(), rows * D * sizeof(bf16));
    apply_layernorm_f32(buf_x, residual_f32.data(), rows, this->encoder_layer_norm_weights, this->encoder_layer_norm_biases, 1e-5f);
    buf_x.sync_to_device();
    
    bf16* out_ptr = KV_CACHE.data();

    for (int layer_idx = 0; layer_idx < WHISPER_DECODER_LAYERS; layer_idx++){
        // dequant the K/V projection weights (cached per decoder layer if the cache is on)
        if (cache_dequant && !kv_dequant_cached[layer_idx]) {
            cache_kv[layer_idx] = this->dequant_kv_cache.create_bo_buffer<bf16>(DK * DK + DV * DV);
        }
        buffer<bf16>& w_kv = cache_dequant ? cache_kv[layer_idx] : buf_kv_cache_proj_weight;
        if (!cache_dequant || !kv_dequant_cached[layer_idx]) {
            dequant_kv_cache(w_kv, this->cross_attn_kv_proj_weights[layer_idx]);
            kv_dequant_cached[layer_idx] = true;
        }

        k_cache_proj(buf_k_cache, buf_x, w_kv);
        buf_k_cache.sync_from_device();
        memcpy(out_ptr, buf_k_cache.data(), (size_t)rows * DK * sizeof(bf16));
        buf_k_cache.sync_to_device();
        out_ptr += L * DK;
        
        v_cache_proj(buf_v_cache, buf_x, w_kv);
        buf_v_cache.sync_from_device();
        add_bias_optimized(buf_v_cache, buf_v_cache, this->cross_attn_v_proj_biases[layer_idx], rows);
        buf_v_cache.sync_to_device();
        memcpy(out_ptr, buf_v_cache.data(), (size_t)rows * DV * sizeof(bf16));
        buf_v_cache.sync_to_device();
        out_ptr += L * DV;
    }
    
    PROF_LAP("xattn_kv");
    // (reported in encode_audio, after the decoder took the cross-attention K/V)
    return KV_CACHE;
}

void WhisperEncoder::prepare_mm(npu_sequence* seq, const uint32_t M, const uint32_t K, const uint32_t N, const uint32_t m, const uint32_t k, const uint32_t n, const uint32_t weight_offset, const uint32_t y_base, const uint32_t x_base, const uint32_t x_row_stride){
    // distance between two x rows in the buffer: K normally, less for the conv front end's overlapping rows
    const uint32_t xs = x_row_stride ? x_row_stride : K;
    constexpr int total_rows = 4;
    constexpr int total_cols = 8;
    npu_sequence& seq_ptr = *seq;
    constexpr npu_tiles IT[8] = {IT0, IT1, IT2, IT3, IT4, IT5, IT6, IT7};
    int K_div_k = K / k;
    seq_ptr.clear_cmds();
    seq->npu_preemption(0);
    // set K_DIV_K
    for (int row = 2; row < 6; row++){
        for (int col = 0; col < 8; col++){
            npu_tiles tile = get_tile(row, col);
            // seq_ptr.rtp_write(tile, 46208, K_div_k);
            seq_ptr.rtp_write(tile, CT_rtp_address, K_div_k);
        }
    }
    int M_div_m = M / m / total_cols;
    int N_div_n = N / n / total_rows;
    for (int j = 0; j < N_div_n; j++){
        for (int i = 0; i < M_div_m; i++){
            int round = j * M_div_m + i;
            int is_pong = round % 2;
            int bd_offset = is_pong * 8;
            int y_offset = y_base + i * m * N * total_cols + j * n * total_rows;
            int x_offset = x_base + i * m * xs * total_cols;
            npu_bd_id y_bd = npu_bd_id(0 + bd_offset);
            npu_bd_id x_bd = npu_bd_id(1 + bd_offset);
            npu_bd_id w_bd = npu_bd_id(2 + bd_offset);
            for (int c = 0; c < total_cols; c++){
                // receive y
                seq_ptr.npu_dma_memcpy_nd(
                    2, 0,
                    S2MM, IT[c],
                    y_bd, it_channel_0,
                    {0, 0, 0, y_offset + c * m * N},
                    {1, 4, m, n},
                    {0, n, N, 1},
                    -1, 0, true
                );

                // send x
                seq_ptr.npu_dma_memcpy_nd(
                    2, 1,
                    MM2S, IT[c],
                    x_bd, it_channel_0,
                    {0, 0, 0, x_offset + c * m * xs},
                    {1, (uint32_t)K_div_k, m, k},
                    {0, k, xs, 1},
                    -1, 0, false
                );
                // seq_ptr.npu_maskwrite(
                //     IT[c],  DMA_BD_x_5 + 32*(uint32_t)(x_bd), AXI_FLAG, 0xF000000
                // );
            } // end c
            for (int r = 0; r < total_rows; r++){
                uint32_t w_offset = j * n * K * total_rows + r * n * K + weight_offset;
                seq_ptr.npu_dma_memcpy_nd(
                    2, 2,
                    MM2S, IT[r + 2],
                    w_bd, it_channel_1,
                    {0, 0, 0, w_offset},
                    {1, 1, 1, (uint32_t)K_div_k * n * k},
                    {0, 0, 0, 1},
                    -1, 0, false
                );
                // seq_ptr.npu_maskwrite(
                //     IT[r + 2],  DMA_BD_x_5 + 32*(uint32_t)(w_bd), AXI_FLAG, 0xF000000
                // );
            }
            if (round > 0){
                for (int c = 0; c < total_cols; c++){
                    seq_ptr.npu_dma_wait(IT[c], S2MM, it_channel_0);
                }
            }
        }
    }
    for (int c = 0; c < total_cols; c++){
        seq_ptr.npu_dma_wait(IT[c], S2MM, it_channel_0);
    }
    seq_ptr.cmds2seq();
}

void WhisperEncoder::prepare_dequant(npu_sequence* seq_ptr, const u32 D_in, const u32 D_out, const u32 weight_offset){
    constexpr int total_rows = 4;
    npu_sequence& seq = *seq_ptr;
    const int w_out_arg_idx = 0;
    const int qw_in_arg_idx = 1;
    // N is din, M is dout
    constexpr int m_tile = 32;
    constexpr int k_tile = 128;
    constexpr int total_cols = 8;
    constexpr npu_tiles IT[8] = {IT0, IT1, IT2, IT3, IT4, IT5, IT6, IT7};
    const int block_size_in_byte = m_tile * k_tile * 5 / 8;
    const int blocks_per_row = D_in / k_tile;
    assert(D_in % (k_tile * 2) == 0);
    const int quant_in_per_column = blocks_per_row * block_size_in_byte;
    int bd_wait_counter[8] = {0,0,0,0,0,0,0,0};
    int bd_ping_pong[8] = {0,0,0,0,0,0,0,0};
    
    const int row_per_round = 2 * m_tile * total_cols;
    const int down_rounds = (D_out + row_per_round - 1) / row_per_round;
    const int total_column_rounds = D_out / 2 / m_tile;
    assert(D_out % (2 * m_tile) == 0);
    seq.clear_cmds();
    seq.npu_preemption(0);
    // first, the dequant of down 
    for(int i = 0; i < down_rounds; i++){

        for(int col = 0; col < 8; col++){
            uint32_t bd_offset = (bd_ping_pong[col] % 2) * 8;
            uint32_t round_offset = i * 8 + col;

            if(round_offset < total_column_rounds){
                // std::cout << "push dma " << col << " " << round_offset << std::endl;

                uint32_t output_offset_0 = round_offset * 64 * D_in;
                seq.npu_dma_memcpy_nd(
                    2,
                    w_out_arg_idx,
                    S2MM,
                    IT[col],
                    (npu_bd_id)(2+bd_offset),
                    it_channel_0,
                    {0, 0, 0, output_offset_0},
                    {1, 1, 1, 64 * D_in},
                    {0, 0, 0, 1},
                    -1, 0, true
                );
               
                seq.npu_dma_memcpy_nd(
                    sizeof(char),
                    qw_in_arg_idx,
                    MM2S, 
                    IT[col],
                    (npu_bd_id)(0+bd_offset),
                    it_channel_0,
                    {0, 0, 0, weight_offset + round_offset * quant_in_per_column * 2}, 
                    {1, 1, 1, (uint32_t)quant_in_per_column}, //NOTE: being rearrange beforehadn already
                    {0, 0, 0, 1},
                    -1,0, false

                );

                seq.npu_dma_memcpy_nd(
                    sizeof(char),
                    qw_in_arg_idx,
                    MM2S, 
                    IT[col],
                    (npu_bd_id)(1+bd_offset),
                    it_channel_1,
                    {0, 0, 0, weight_offset + round_offset * quant_in_per_column * 2 + quant_in_per_column}, 
                    {1, 1, 1, (uint32_t)quant_in_per_column}, //NOTE: being rearrange beforehadn already
                    {0, 0, 0, 1},
                    -1,0, false

                );
                bd_wait_counter[col]++;
                bd_ping_pong[col]++;
            }
        }
        // note: for now
        for(int col = 0; col < 8; col++){
            if(bd_wait_counter[col] == 2){
                seq.npu_dma_wait(IT[col], S2MM, it_channel_0);
                // std::cout << "wait dma " << col << std::endl;
                bd_wait_counter[col]--;
            }
        }
    }
    int max_loop_iterations = 0;
    // find the global max loop iterations
    for(int col = 0; col < 8; col++){
        max_loop_iterations = std::max(max_loop_iterations, bd_wait_counter[col]);
    }
    for(int i = 0; i < max_loop_iterations; i++){
        for(int col = 0; col < 8; col++){
            if (bd_wait_counter[col] > 0){
                seq.npu_dma_wait(IT[col], S2MM, it_channel_0);
                // std::cout << "wait dma " << col << std::endl;
                bd_wait_counter[col]--;
            }
        }
    }
    seq.cmds2seq();
}

void WhisperEncoder::prepare_attention(npu_sequence* seq, const uint32_t Lq, const uint32_t Lk, const uint32_t valid_keys){
    static constexpr int Heads = 20;
    static constexpr int DH = 64;
    static constexpr int lq = 64;
    static constexpr int MAX_L = 1536;

    npu_tiles IT[4][2] = {{IT0, IT1}, {IT2, IT3}, {IT4, IT5}, {IT6, IT7}};

    int l_begin_mha_address = 61440;  // pinned in encoder_attention_88x/iron/attn_aie.py

    seq->clear_cmds();
    for (int row = 2; row < 6; row++){
        for (int col = 0; col < 8; col++){
            npu_tiles tile = get_tile(row, col);
            seq->rtp_write(tile, l_begin_mha_address, Lq);                 // query rows
            seq->rtp_write(tile, l_begin_mha_address + 4, Lk / 512);       // key chunks of 512 rows
            seq->rtp_write(tile, l_begin_mha_address + 8, valid_keys);     // key rows that are not padding
        }
    }

    // each round, each CU works on 1 head and lc * 8 of l, total_cols / 2 is corresponding to the number of CUs
    const int data_per_round = lq * 8; 
    // zero padding the last round if necessary
    const int down_rounds = (Lq + data_per_round - 1) / data_per_round;

    for (int head = 0; head < Heads / 4; head++){
        for (int round = 0; round < down_rounds; round++){

            int bd_offset = (round % 2) * 8;

            // cu 0,
            for (int cu = 0; cu < 4; cu++){
                int head_offset = head * 4 + cu;
                for (int col = 0; col < 2; col++){    
                    // receive y
                    size_t y_offset = head_offset * DH + round * data_per_round * DH * Heads + col * lq * 4 * DH * Heads; 
                    seq->npu_dma_memcpy_nd(
                        2, 0,
                        S2MM, IT[cu][col],
                        (npu_bd_id)(bd_offset + 0), it_channel_0,
                        {0, 0, 0, (uint32_t)y_offset},
                        {1, 1, (uint32_t)4 * lq, (uint32_t)(DH)},
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
                    {1, 1, (uint32_t)lq * 4, (uint32_t)(DH)},
                    {0, 0, (uint32_t)DH * Heads, 1},
                    -1, 0, false
                );
                seq->npu_dma_memcpy_nd(
                    2, 1,
                    MM2S, IT[cu][0],
                    (npu_bd_id)(bd_offset + 2), it_channel_1,
                    {0, 0, 0, (uint32_t)(q_offset + lq * 4 * DH * Heads)},
                    {1, 1, (uint32_t)lq * 4, (uint32_t)(DH)},
                    {0, 0, (uint32_t)DH * Heads, 1},
                    -1, 0, false
                );
                // send k
                size_t k_offset = head_offset * DH;
                seq->npu_dma_memcpy_nd(
                    2, 2,
                    MM2S, IT[cu][1],
                    (npu_bd_id)(bd_offset + 3), it_channel_0,
                    {0, 0, 0, (uint32_t)k_offset},
                    {1, (Lk / 512), 512, DH},
                    {0, (uint32_t)512 * DH * Heads, (uint32_t)DH * Heads, 1},
                    -1, 0, false
                );
                // send v
                size_t v_offset = head_offset * DH + MAX_L * DH * Heads;
                seq->npu_dma_memcpy_nd(
                    2, 2,
                    MM2S, IT[cu][1],
                    (npu_bd_id)(bd_offset + 4), it_channel_1,
                    {0, 0, 0, (uint32_t)v_offset},
                    {1, (Lk / 512), 512, DH},
                    {0, (uint32_t)512 * DH * Heads, (uint32_t)DH * Heads, 1},
                    -1, 0, false
                );
            } // cu
            if (round > 0){
                for (int cu = 0; cu < 4; cu++){
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
        for (int cu = 0; cu < 4; cu++){
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

// Matmul and attention sequences for the 30 s window (L rows).
void WhisperEncoder::prepare_sequences(){
    constexpr int rows = L;
    // conv1: one matmul row per mel frame (3000 frames padded to 3072 for a full window)
    constexpr int conv1_m = 3072;   // 3000 mel frames, padded to the matmul's 768-row step
    prepare_mm(conv1_mm.seq(), conv1_m, CONV1_K, 1280, 96, 256, 64, 0, 0, 0, 128);
    // conv2 (stride 2): one matmul row per encoder row, reading three conv1 rows two rows apart
    prepare_mm(conv2_mm.seq(), rows, CONV2_K, 1280, 96, 256, 64, 0, 0, 0, 2 * 1280);
    prepare_mm(q_proj.seq(), rows, D, DQ, 96, 256, 64, 0);
    prepare_mm(k_proj.seq(), rows, D, DK, 96, 256, 64, D * DQ, 0);                   // K half of kv_cache
    prepare_mm(v_proj.seq(), rows, D, DV, 96, 256, 64, D * (DQ + DK), L * DK);      // V half of kv_cache
    prepare_mm(o_proj.seq(), rows, DQ, D, 96, 256, 64, 0);
    prepare_mm(up_proj.seq(), rows, D, intermediate_size, 96, 256, 64, 0);
    prepare_mm(down_proj.seq(), rows, intermediate_size, D, 96, 256, 64, 0);

    prepare_mm(k_cache_proj.seq(), rows, DK, DK, 96, 256, 64, 0);
    prepare_mm(v_cache_proj.seq(), rows, DV, DV, 96, 256, 64, DK * DK);

    // the attention kernel works in blocks of 512 query rows and 512 key rows
    const int blocks = (rows + 511) / 512 * 512;
    prepare_attention(mha_engine.seq(), blocks, blocks, WHISPER_AUDIO_FRAMES);
}

// P (V + b_v) = P V + b_v because every softmax row sums to 1, so the V bias can move behind the
// attention: o = (P V) W_o^T + (b_v W_o^T + b_o). b_v W_o^T is computed once with the o-proj
// kernel itself (every input row = b_v), so V needs no CPU pass at all.
void WhisperEncoder::fold_v_bias(int layer_idx, buffer<bf16>& w_o){
    for (int r = 0; r < L; r++) {
        memcpy(buf_x.data() + (size_t)r * D, v_biases[layer_idx].data(), D * sizeof(bf16));
    }
    buf_x.sync_to_device();
    o_proj(buf_q_o_d, buf_x, w_o);
    buf_q_o_d.sync_from_device();
    o_bias_f32[layer_idx].resize(D);
    for (int c = 0; c < D; c++) {
        o_bias_f32[layer_idx][c] = (float)o_biases[layer_idx][c] + (float)buf_q_o_d[c];
    }
}

// Steps 4-5 of a layer for a full window, on two halves of the rows: while the NPU runs a matmul on one
// half, the CPU does the bias / residual / layer norm / GELU of the other. Each row gets exactly the
// arithmetic of the plain path. CPU and NPU never touch the same rows of a buffer at the same time; a
// whole-buffer invalidate while the NPU writes the other half only drops clean cache lines.
void WhisperEncoder::post_attention_split(int layer_idx, buffer<bf16>& w_o, buffer<bf16>& w_up, buffer<bf16>& w_down){
    const int H = L / 2;
    float* resid = residual_f32.data();
    const float* o_bias = o_bias_f32[layer_idx].data();
    const float* fc1_bias = fc1_bias_f32[layer_idx].data();
    const float* fc2_bias = fc2_bias_f32[layer_idx].data();
    auto residual_and_ln2 = [&](int h){   // o rows of half h -> residual -> layer norm -> buf_x rows of half h
        add_biased_bf16_to_f32(resid + (size_t)h * H * D, buf_q_o_d.data() + (size_t)h * H * D, o_bias, H, D);
        apply_layernorm_f32(buf_x.data() + (size_t)h * H * D, resid + (size_t)h * H * D, H,
                            this->mlp_norm_weights[layer_idx], this->mlp_norm_biases[layer_idx], 1e-5f);
    };
    auto gelu_half = [&](int h){
        add_bias_gelu(buf_up.data() + (size_t)h * H * intermediate_size, fc1_bias, H, intermediate_size);
    };
    auto down_residual = [&](int h){
        add_biased_bf16_to_f32(resid + (size_t)h * H * D, buf_q_o_d.data() + (size_t)h * H * D, fc2_bias, H, D);
    };

    o_half[0](buf_q_o_d, buf_attn_output, w_o);
    PROF_LAP("mm_o");
    auto run = o_half[1].create_run(buf_q_o_d, buf_attn_output, w_o);
    run.start();
    buf_q_o_d.sync_from_device();
    residual_and_ln2(0);
    flm::npu::wait_for_completion(run, "whisper layer " + std::to_string(layer_idx) + " o 1");
    buf_x.sync_to_device();
    PROF_LAP("o|ln2");

    run = up_half[0].create_run(buf_up, buf_x, w_up);
    run.start();
    buf_q_o_d.sync_from_device();
    residual_and_ln2(1);
    flm::npu::wait_for_completion(run, "whisper layer " + std::to_string(layer_idx) + " up 0");
    buf_x.sync_to_device();
    PROF_LAP("up|ln2");

    run = up_half[1].create_run(buf_up, buf_x, w_up);
    run.start();
    buf_up.sync_from_device();
    gelu_half(0);
    flm::npu::wait_for_completion(run, "whisper layer " + std::to_string(layer_idx) + " up 1");
    buf_up.sync_to_device();
    PROF_LAP("up|gelu");

    run = down_half[0].create_run(buf_q_o_d, buf_up, w_down);
    run.start();
    buf_up.sync_from_device();
    gelu_half(1);
    flm::npu::wait_for_completion(run, "whisper layer " + std::to_string(layer_idx) + " down 0");
    buf_up.sync_to_device();
    PROF_LAP("down|gelu");

    run = down_half[1].create_run(buf_q_o_d, buf_up, w_down);
    run.start();
    buf_q_o_d.sync_from_device();
    down_residual(0);
    // the next layer's first layer norm for this half, unless that layer still has to fold its V bias
    // (fold_v_bias uses buf_x)
    if (layer_idx + 1 < num_hidden_layers && !o_bias_f32[layer_idx + 1].empty()) {
        apply_layernorm_f32(buf_x.data(), resid, H, this->attn_norm_weights[layer_idx + 1], this->attn_norm_biases[layer_idx + 1], 1e-5f);
        ln1_first_half_done = true;
    }
    flm::npu::wait_for_completion(run, "whisper layer " + std::to_string(layer_idx) + " down 1");
    buf_q_o_d.sync_from_device();
    down_residual(1);
    PROF_LAP("down|resid");
}

// ---------------------------------------------------------------------------------------------
// Conv front end on the NPU. Both convs are matmuls over overlapping rows, so no im2col copy is
// needed: with the input stored one frame per row (row 0 = the zero padding frame), the K inputs
// of output row t are consecutive rows of that buffer, and the matmul's x row stride is set to
// one frame (conv1) or two frames (conv2, stride 2) instead of K.
//   conv1: K = 4 frames x 128 mel bins (the 4th frame has zero weights; K must be a multiple of 256)
//   conv2: K = 3 frames x 1280 channels
// The weights are reordered once to match: W[oc][k * C + c] = w[oc][c][k], stored the way the matmul
// kernel streams W: blocks of 64 output rows, and inside a block one 64 x 256 row-major tile per 256
// inputs (mm_w_index).
// ---------------------------------------------------------------------------------------------
// Element (n, kk) of an N x K matmul weight in the layout prepare_mm streams it from (the layout the
// dequant kernel writes; read back from the hardware with one-hot inputs): 64-row blocks, 256-wide K
// chunks inside a block, and inside a chunk [n / 8][kk % 8][kk / 8][n % 8].
static inline size_t mm_w_index(int n, int kk, int K){
    const int blk = n / 64, nn = n % 64;
    const int chunk = kk / 256, kc = kk % 256;
    return (size_t)blk * 64 * K + (size_t)chunk * 64 * 256
         + (size_t)(nn / 8) * 2048 + (kc % 8) * 256 + (kc / 8) * 8 + (nn % 8);
}

void WhisperEncoder::pack_conv_weights(){
    memset(conv1_w.data(), 0, conv1_w.size() * sizeof(bf16));
    for (int oc = 0; oc < 1280; oc++)
        for (int c = 0; c < 128; c++)
            for (int k = 0; k < 3; k++)
                conv1_w[mm_w_index(oc, k * 128 + c, CONV1_K)] = conv1d_1_weight[((size_t)oc * 128 + c) * 3 + k];
    for (int oc = 0; oc < 1280; oc++)
        for (int ic = 0; ic < 1280; ic++)
            for (int k = 0; k < 3; k++)
                conv2_w[mm_w_index(oc, k * 1280 + ic, CONV2_K)] = conv1d_2_weight[((size_t)oc * 1280 + ic) * 3 + k];
    conv1_w.sync_to_device();
    conv2_w.sync_to_device();
    conv1_bias_f32.assign(conv1d_1_bias.begin(), conv1d_1_bias.end());
    conv2_bias_f32.assign(conv1d_2_bias.begin(), conv1d_2_bias.end());
}

void WhisperEncoder::conv_front_end(const buffer<bf16>& mel){
    constexpr int MEL = 128, MEL_FRAMES = 3000, C = 1280;
    constexpr int rows = L, frames = MEL_FRAMES;
    // conv1 input: frame f of the mel at row f + 1; row 0 and rows past `frames` are zero padding
    bf16* x1 = conv1_x.data();
    memset(x1, 0, conv1_x.size() * sizeof(bf16));
    #pragma omp parallel for schedule(static)
    for (int f = 0; f < frames; f++)
        for (int c = 0; c < MEL; c++)
            x1[(size_t)(f + 1) * MEL + c] = mel[(size_t)c * MEL_FRAMES + f];
    conv1_x.sync_to_device();
    conv1_mm(buf_up, conv1_x, conv1_w);             // [conv1_m][1280]
    buf_up.sync_from_device();
    PROF_LAP("conv1");

    // conv2 input: gelu(conv1 + b1) for frame f at row f + 1, zero padding around it
    bf16* x2 = conv2_x.data();
    memset(x2, 0, C * sizeof(bf16));
    memset(x2 + (size_t)(frames + 1) * C, 0, (conv2_x.size() - (size_t)(frames + 1) * C) * sizeof(bf16));
    add_bias_gelu_to(x2 + C, buf_up.data(), conv1_bias_f32.data(), frames, C);
    conv2_x.sync_to_device();
    conv2_mm(buf_q_o_d, conv2_x, conv2_w);          // [rows][1280]; row t reads conv1 rows 2t-1 .. 2t+1
    buf_q_o_d.sync_from_device();
    PROF_LAP("conv2");

    // residual = gelu(conv2 + b2) + positional embedding, rounded to bf16 as before; padding rows zero
    const int n_out = frames / 2;
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < rows; t++){
        float* out = residual_f32.data() + (size_t)t * D;
        if (t >= n_out) { memset(out, 0, D * sizeof(float)); continue; }
        const bf16* y = buf_q_o_d.data() + (size_t)t * C;
        const bf16* pos = positional_embedding.data() + (size_t)t * D;
        for (int c = 0; c < C; c += 16){
            __m512 g = round_bf16_ps(gelu_avx512(round_bf16_ps(_mm512_add_ps(loadu_bf16x16_to_ps(y + c), _mm512_loadu_ps(conv2_bias_f32.data() + c)))));
            _mm512_storeu_ps(out + c, round_bf16_ps(_mm512_add_ps(g, loadu_bf16x16_to_ps(pos + c))));
        }
    }
}
