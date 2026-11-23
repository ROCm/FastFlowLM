// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#include "gemma_npu_detail.hpp"

gemma_npu::Impl::Impl(LM_Config config, npu_xclbin_manager *npu_instance, int MAX_L)
    : config(config), npu(npu_instance){
    is_vlm = config.get<bool>("is_vlm", false);
    num_hidden_layers = config.get("num_hidden_layers");
    current_context_length = 0;
    D = config.get("hidden_size");
    DH = config.get("head_dim");
    DQ = config.get("head_dim") * config.get("num_attention_heads");
    DK = config.get("head_dim") * config.get("num_key_value_heads");
    DV = config.get("head_dim") * config.get("num_key_value_heads");
    INTERMEDIATE_SIZE = config.get("intermediate_size");
    HIDDEN_SIZE = config.get("intermediate_size");
    SLIDING_WINDOW = config.get("sliding_window");
    SLIDING_WINDOW_PATTERN = config.get("sliding_window_pattern");
    vocab_size = config.get("vocab_size");

    this->MAX_L = std::max(MAX_L, 8192);
    // make MAX_L a power of 2
    float log2_MAX_L = std::log2(this->MAX_L);
    if (log2_MAX_L != std::floor(log2_MAX_L)){
        this->MAX_L = 1 << ((int)std::floor(log2_MAX_L) + 1);
    }
    MAX_L = this->MAX_L; // synchronize MAX_L

    is_checkpoint_valid = false;

    mha_type_t mha_type = mha_d256_q2;

    this->sequence = std::make_unique<gemma_npu_sequence>(config, MAX_L);
    this->dequant_seq_generator =  std::make_unique<Dequant>(config);
    this->gemm_seq_generator = std::make_unique<Gemm>(config);
    this->mha_seq_generator = std::make_unique<MHA>(mha_type, 4);

    if (is_vlm){
        this->gemma_image_encoder = std::make_unique<GemmaImageEncoder>(config, npu);
    }

    layer_app_manager = npu->register_xclbin(utils::path_join(config.exec_path, "xclbins", config.model_name, "layer_lm.xclbin"));
    this->layer = layer_app_manager->create_app();
    this->sliding_layer = layer_app_manager->create_app();
    if (!this->npu->is_preemption_enabled()){
        this->layers_run = layer_app_manager->create_runlist();
    }

    gemm_app_manager = npu->register_xclbin(utils::path_join(config.exec_path, "xclbins", config.model_name, "mm.xclbin"));
    this->q_proj = gemm_app_manager->create_app();
    this->k_proj = gemm_app_manager->create_app();
    this->v_proj = gemm_app_manager->create_app();
    this->o_proj = gemm_app_manager->create_app();
    this->up_proj = gemm_app_manager->create_app();
    this->gate_proj= gemm_app_manager->create_app();
    this->down_proj = gemm_app_manager->create_app();

    dequant_app_manager = npu->register_xclbin(utils::path_join(config.exec_path, "xclbins", config.model_name, "dequant.xclbin"));
    this->qkv_dequant = dequant_app_manager->create_app();
    this->o_dequant = dequant_app_manager->create_app();
    this->gate_dequant = dequant_app_manager->create_app();
    this->up_dequant = dequant_app_manager->create_app();
    this->down_dequant = dequant_app_manager->create_app();
    if (!this->npu->is_preemption_enabled()){
        this->dequant_all = dequant_app_manager->create_runlist();
    }
    mha_app_manager = npu->register_xclbin(utils::path_join(config.exec_path, "xclbins", config.model_name, "attn.xclbin"));
    this->mha_engine = mha_app_manager->create_app();
    this->sliding_mha_engine = mha_app_manager->create_app();

    this->lm_head = layer_app_manager->create_app();

    // allocate all buffers
    rms_weights.resize(num_hidden_layers);
    proj_weights.resize(num_hidden_layers);
    kv_caches.resize(num_hidden_layers);
    attn_mask.resize(num_hidden_layers);
    for (uint32_t i = 0; i < num_hidden_layers; i++) attn_mask[i] = !is_sliding_window(i);
    rope_rms_weights.resize(num_hidden_layers);

    size_t total_proj_weights =
          (config.get("num_attention_heads") * config.get("head_dim")) * config.get("hidden_size")
        + (config.get("num_key_value_heads") * config.get("head_dim")) * config.get("hidden_size") * 2
        + (config.get("num_attention_heads") * config.get("head_dim")) * config.get("hidden_size")
        + config.get("intermediate_size") * config.get("hidden_size") * 3;
    size_t total_proj_weights_size = total_proj_weights * 5 / 8;
    size_t kv_cache_size = (size_t)MAX_L * config.get("num_key_value_heads") * config.get("head_dim") * 2;
    size_t kv_cache_size_sliding = (size_t)SLIDING_WINDOW * config.get("num_key_value_heads") * config.get("head_dim") * 2;

    for (size_t i = 0; i < num_hidden_layers; i++){
        proj_weights[i] = layer.create_bo_buffer<u8>(total_proj_weights_size);
        rms_weights[i] = layer.create_bo_buffer<bf16>(config.get("hidden_size") * 4);
        kv_caches[i] = layer.create_bo_buffer<bf16>(is_sliding_window(i) ? kv_cache_size_sliding : kv_cache_size);
        rope_rms_weights[i] = layer.create_bo_buffer<bf16>(config.get("head_dim") * 3);
        this->kv_caches[i].sync_to_device();
    }
    this->sliding_temp_cache = this->mha_engine.create_bo_buffer<bf16>(kv_cache_size);
    x = layer.create_bo_buffer<bf16>(D);
    const size_t lm_head_rows = this->sequence->get_lm_head_in_layer_rows();
    this->logits = this->lm_head.create_bo_buffer<bf16>(lm_head_rows);
    this->logits_valid = buffer<bf16>(this->logits.data(), vocab_size);
    this->lm_head_weights = this->lm_head.create_bo_buffer<u8>(lm_head_rows * D * 5 / 8);
    this->lm_head_norm = this->lm_head.create_bo_buffer<bf16>(D);
    this->lm_head_unused = this->lm_head.create_bo_buffer<bf16>(D);

    this->K01_offset = sequence->get_k01_offset();
    this->K23_offset = sequence->get_k23_offset();
    this->V01_offset = sequence->get_v01_offset();
    this->V23_offset = sequence->get_v23_offset();
    this->sliding_K01_offset = SLIDING_WINDOW * config.get("num_key_value_heads") * config.get("head_dim") / 2 * 0;
    this->sliding_K23_offset = SLIDING_WINDOW * config.get("num_key_value_heads") * config.get("head_dim") / 2 * 1;
    this->sliding_V01_offset = SLIDING_WINDOW * config.get("num_key_value_heads") * config.get("head_dim") / 2 * 2;
    this->sliding_V23_offset = SLIDING_WINDOW * config.get("num_key_value_heads") * config.get("head_dim") / 2 * 3;
    this->set_context_length(0);
    this->embedding = std::make_unique<Embedding>(config.get("vocab_size"), config.get("hidden_size"));

    this->dequanted_qkv_weights = this->qkv_dequant.create_bo_buffer<bf16>((size_t)D * (DQ + DK + DV));
    this->dequanted_o_weights = this->o_dequant.create_bo_buffer<bf16>((size_t)DQ * D);

    this->dequanted_up_weights = this->up_dequant.create_bo_buffer<bf16>((size_t)D * INTERMEDIATE_SIZE );
    this->dequanted_gate_weights = this->gate_dequant.create_bo_buffer<bf16>((size_t)D * INTERMEDIATE_SIZE );
    this->dequanted_down_weights = this->down_dequant.create_bo_buffer<bf16>((size_t)INTERMEDIATE_SIZE * D);

    this->dequant_seq_generator->generate_dequant_q4_1_seq(this->qkv_dequant.seq(),  D,  DQ + DK + DV, 0,              0);
    this->dequant_seq_generator->generate_dequant_q4_1_seq(this->o_dequant.seq(),    DQ, D, D * (DQ + DK + DV) * 5 / 8, 0);
    this->dequant_seq_generator->generate_dequant_q4_1_seq(this->up_dequant.seq(),   D,  INTERMEDIATE_SIZE , D * (DQ + DK + DV + DQ) * 5 / 8, 1);
    this->dequant_seq_generator->generate_dequant_q4_1_seq(this->gate_dequant.seq(), D,  INTERMEDIATE_SIZE , D * (DQ + DK + DV + DQ) * 5 / 8, 2);

    this->dequant_seq_generator->generate_dequant_q4_1_seq(this->down_dequant.seq(), INTERMEDIATE_SIZE, D, D * (DQ + DK + DV + DQ + 2 * INTERMEDIATE_SIZE) * 5 / 8,    0);

    this->sequence->gen_lm_head_in_layer_seq(this->lm_head.seq());
    this->lm_head_run = this->lm_head.create_run(this->x, this->lm_head_weights, this->lm_head_norm, this->lm_head_unused, this->logits);
}

void reorder_cpy(u8* dst, buffer<u8>& src, const int col, const int vertical_blocks = 2){
    const int a_block_size = 32 * 256 * 5 / 8;
    const int blocks_per_row = col / 256;

    const int rows = src.size() / a_block_size / blocks_per_row;

    u8* dst_ptr = dst;
    std::vector<u8*> src_ptr(vertical_blocks);
    for (int i = 0; i < vertical_blocks; i++){
        src_ptr[i] = src.data() + i * a_block_size * blocks_per_row;
    }
    for (int r = 0; r < rows; r += vertical_blocks){
        for (int c = 0; c < blocks_per_row; c++){
            for (int i = 0; i < vertical_blocks; i++){
                memcpy(dst_ptr, src_ptr[i], a_block_size);
                dst_ptr += a_block_size;
                src_ptr[i] += a_block_size;
            }
        }
        for (int i = 0; i < vertical_blocks; i++){
            src_ptr[i] += (vertical_blocks - 1) * a_block_size * blocks_per_row;
            if (src_ptr[i] + a_block_size * blocks_per_row > src.end()){
                src_ptr[i] = src.data(); // useless padding
            }
        }
    }
}

void gemma_npu::Impl::load_weights(Q4NX& q4nx){
    // read qkv weights
    if (is_vlm){
        SafeTensors vision_weights(config.get<std::string>("vision_model_weight", ""));
        this->gemma_image_encoder->init_weights(vision_weights);
    }
    this->embedding->init_weights(&q4nx, "model.embed_tokens");
    q4nx.load_weights(this->final_rms_weight, "model.norm.weight");

    for (uint32_t layer_idx = 0; layer_idx < num_hidden_layers; layer_idx++){
        u8* w_ptr = this->proj_weights[layer_idx].data();
        u8* w_ptr_end = this->proj_weights[layer_idx].end();
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
            // memcpy(w_ptr, w_q.data(), DQ * D * 5 / 8);
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
            const int total_size = HIDDEN_SIZE * D * 5 / 8;
            const int phases = total_size / chunk_size;
            tensor_2d<u8> w_up_tensor(w_up, chunk_size, 0);
            tensor_2d<u8> w_gate_tensor(w_gate, chunk_size, 0);
            for (int i = 0; i < phases; i++){
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
            reorder_cpy(w_ptr, w_down, HIDDEN_SIZE);
            w_ptr += HIDDEN_SIZE * D * 5 / 8;
        }
        assert(w_ptr == w_ptr_end);
        // read rms weights
        {
            LOG_VERBOSE(1, "Loading rms weights");
            buffer<bf16> w_rms_pre;
            buffer<bf16> w_rms_post;
            buffer<bf16> w_rms_pre_mlp;
            buffer<bf16> w_rms_post_mlp;

            bf16* w_rms_ptr = this->rms_weights[layer_idx].data();
            q4nx.load_weights(w_rms_pre, "model.layers." + std::to_string(layer_idx) + ".input_layernorm.weight");
            q4nx.load_weights(w_rms_post, "model.layers." + std::to_string(layer_idx) + ".post_attention_layernorm.weight");
            q4nx.load_weights(w_rms_pre_mlp, "model.layers." + std::to_string(layer_idx) + ".pre_feedforward_layernorm.weight");
            q4nx.load_weights(w_rms_post_mlp, "model.layers." + std::to_string(layer_idx) + ".post_feedforward_layernorm.weight");

            // w_rms_post.memset(1);
            memcpy((unsigned char*)w_rms_ptr, w_rms_pre.data(), D * sizeof(bf16));
            w_rms_ptr += D;
            memcpy((unsigned char*)w_rms_ptr, w_rms_post.data(), D * sizeof(bf16));
            w_rms_ptr += D;
            memcpy((unsigned char*)w_rms_ptr, w_rms_pre_mlp.data(), D * sizeof(bf16));
            w_rms_ptr += D;
            memcpy((unsigned char*)w_rms_ptr, w_rms_post_mlp.data(), D * sizeof(bf16));
            w_rms_ptr += D;
        }
        // read attention norm weights
        {
            LOG_VERBOSE(1, "Loading attention norm weights");
            buffer<bf16> q_attn_norm;
            buffer<bf16> k_attn_norm;
            q4nx.load_weights(q_attn_norm, "model.layers." + std::to_string(layer_idx) + ".self_attn.q_norm.weight");
            q4nx.load_weights(k_attn_norm, "model.layers." + std::to_string(layer_idx) + ".self_attn.k_norm.weight");
            memcpy(this->rope_rms_weights[layer_idx].data() + DH, q_attn_norm.data(), DH * sizeof(bf16));
            memcpy(this->rope_rms_weights[layer_idx].data() + DH * 2, k_attn_norm.data(), DH * sizeof(bf16));
        }

        this->proj_weights[layer_idx].sync_to_device();
        this->rms_weights[layer_idx].sync_to_device();
        this->rope_rms_weights[layer_idx].sync_to_device();
    }
    {
        LOG_VERBOSE(1, "Loading lm head weights");
        buffer<u8> w_lm_head;
        q4nx.load_weights(w_lm_head, "lm_head.weight");
        reorder_cpy(this->lm_head_weights.data(), w_lm_head, D, 2);
        this->lm_head_weights.sync_to_device();
        memcpy(this->lm_head_norm.data(), this->final_rms_weight.data(), (size_t)D * sizeof(bf16));
        this->lm_head_norm.sync_to_device();
    }
    this->clear_context();
}

void gemma_npu::Impl::_set_rope_rms_weights(int idx){

    buffer<bf16> cos_buf_global(DH / 2);
    buffer<bf16> sin_buf_global(DH / 2);
    buffer<bf16> cos_buf_local(DH / 2);
    buffer<bf16> sin_buf_local(DH / 2);
    for (int j = 0; j < DH / 2; j++){
        cos_buf_global[j] = (bf16)cos(inv_freq[j] * idx);
        sin_buf_global[j] = (bf16)sin(inv_freq[j] * idx);
        cos_buf_local[j] = (bf16)cos(inv_freq_local[j] * idx);
        sin_buf_local[j] = (bf16)sin(inv_freq_local[j] * idx);
    }

    for (int i = 0; i < num_hidden_layers; i++){
        bool is_sliding = ((i + 1) % 6 != 0);
        bf16* w_rope_ptr = this->rope_rms_weights[i].data();
        if (is_sliding){
            memcpy(w_rope_ptr, cos_buf_local.data(), DH / 2 * sizeof(bf16));
            w_rope_ptr += DH / 2;
            memcpy(w_rope_ptr, sin_buf_local.data(), DH / 2 * sizeof(bf16));
        } else {
            memcpy(w_rope_ptr, cos_buf_global.data(), DH / 2 * sizeof(bf16));
            w_rope_ptr += DH / 2;
            memcpy(w_rope_ptr, sin_buf_global.data(), DH / 2 * sizeof(bf16));
        }
        this->rope_rms_weights[i].sync_to_device();
    }
}

void gemma_npu::Impl::set_context_length(int L){
    this->current_context_length = L;
    this->dirty_rows = std::max(this->dirty_rows, static_cast<size_t>(L));

    this->sequence->gen_layer_seq(this->layer.seq(), L + 1, false);
    this->sequence->gen_layer_seq(this->sliding_layer.seq(), L + 1, true);

    this->_set_rope_rms_weights(L);

    if (!this->npu->is_preemption_enabled()){
        this->layers_run.reset();

        for (uint32_t i = 0; i < num_hidden_layers; i++){
            if (is_sliding_window(i)){
                this->layers_run.add(this->sliding_layer.create_run(this->x, this->proj_weights[i], this->rms_weights[i], this->rope_rms_weights[i], this->kv_caches[i]));
            }
            else{
                this->layers_run.add(this->layer.create_run(this->x, this->proj_weights[i], this->rms_weights[i], this->rope_rms_weights[i], this->kv_caches[i]));
            }
        }
    }
}

buffer<bf16> gemma_npu::Impl::forward(int ids){
    this->embedding->forward(ids, this->x);
    this->x.sync_to_device();
    if (!this->npu->is_preemption_enabled()){
        this->layers_run.execute();
        flm::npu::wait_for_completion(this->layers_run, "layers_run");
    }
    else{
        for (uint32_t i = 0; i < num_hidden_layers; i++){
            if (is_sliding_window(i)){
                this->sliding_layer(this->x, this->proj_weights[i], this->rms_weights[i], this->rope_rms_weights[i], this->kv_caches[i]);
            }
            else{
                this->layer(this->x, this->proj_weights[i], this->rms_weights[i], this->rope_rms_weights[i], this->kv_caches[i]);
            }
        }
    }

    this->x.sync_from_device();
    // buffer<bf16> logits = get_logits(this->x);
    this->_rms_norm(this->x, this->x, this->final_rms_weight);
    this->x.sync_to_device();
    this->lm_head_run.start();
    this->set_context_length(this->current_context_length + 1);
    flm::npu::wait_for_completion(this->lm_head_run, "lm_head_run");
    this->logits.sync_from_device();
    return this->logits_valid;
}

buffer<bf16> gemma_npu::Impl::prefill(std::vector<int>& ids, void* payload){
#ifdef MVPREFILL
    header_print("warning", "Prefilling with Decoder, Slow!");
    return this->_prefill_with_mv(ids, payload);
#else
    return this->_prefill_with_mm(ids, payload);
#endif
}

buffer<bf16> gemma_npu::Impl::_prefill_with_mv(std::vector<int>& ids, void* payload){
    for (int i = 0; i < ids.size(); i++){
        this->embedding->forward(ids[i], this->x);
        this->x.sync_to_device();
        if (!this->npu->is_preemption_enabled()){
            this->layers_run.execute();
            flm::npu::wait_for_completion(this->layers_run, "layers_run");
        }
        else{
            for (uint32_t i = 0; i < num_hidden_layers; i++){
                if (is_sliding_window(i)){
                    this->sliding_layer(this->x, this->proj_weights[i], this->rms_weights[i], this->rope_rms_weights[i], this->kv_caches[i]);
                }
                else{
                    this->layer(this->x, this->proj_weights[i], this->rms_weights[i], this->rope_rms_weights[i], this->kv_caches[i]);
                }
            }
        }
        this->set_context_length(this->current_context_length + 1);
    }
    this->x.sync_from_device();
    buffer<bf16> logits = get_logits(this->x);

    return logits;
}

buffer<bf16> gemma_npu::Impl::_prefill_with_mm(std::vector<int>& ids, void* payload){
    const int L_MIN = 256;
    const int L_CHUNK_SIZE = this->mha_seq_generator->get_chunk_size();
    int L_in = ids.size();
    int L_begin = this->current_context_length;
    int L_end = L_begin + L_in;
    // get the L in product of 16
    int L_begin_chunked = (L_begin / L_CHUNK_SIZE) * L_CHUNK_SIZE;
    int L_offset = L_begin - L_begin_chunked;
    int L_end_chunked = L_begin_chunked + (L_in + L_offset + L_MIN - 1) / L_MIN * L_MIN;
    int L_padded = L_end_chunked - L_begin_chunked;

    int L_effective = L_end - L_begin;
    int sliding_l_begin = (L_begin_chunked - SLIDING_WINDOW) > 0 ? (L_begin_chunked - SLIDING_WINDOW) : 0;
    //allocate temporary buffers

    // The scratch window the sliding layers attend over is rebuilt from the rings on every call, but what an earlier call left in the
    // rows outside what it writes is not harmless to the kernel: start every prefill from zeros, so a prefill continued live and one
    // that follows a restored state see the same scratch.
    this->_zero_temp_cache();
    this->gemm_seq_generator->generate_seq(this->q_proj.seq(),
        L_padded, D, DQ, 0,
        false, Gemm::NO_Activation, 0 //  for NOW
    );
    this->gemm_seq_generator->generate_seq(this->k_proj.seq(),
        L_padded, D, DK, D * DQ,
        false, Gemm::NO_Activation, 0 //  for NOW
    );
    this->gemm_seq_generator->generate_seq(this->v_proj.seq(),
        L_padded, D, DV, D * (DQ + DK),
        false, Gemm::NO_Activation, 0 //  for NOW
    );
    this->gemm_seq_generator->generate_seq(this->o_proj.seq(),
        L_padded, DQ, D, 0,
        false, Gemm::NO_Activation, 0 //  for NOW
    );
    this->gemm_seq_generator->generate_seq(this->up_proj.seq(),
        L_padded, D,  INTERMEDIATE_SIZE, 0,
        false, Gemm::NO_Activation, 0 //  for NOW
    );

    this->gemm_seq_generator->generate_seq(this->gate_proj.seq(),
        L_padded, D,  INTERMEDIATE_SIZE, 0,
        false, Gemm::GeLU, 0 //  gelu
    );

    this->gemm_seq_generator->generate_seq(this->down_proj.seq(),
        L_padded, INTERMEDIATE_SIZE, D, 0,
        false, Gemm::NO_Activation, 0 //  for NOW
    );

    buffer<bf16> buffer_q = this->q_proj.create_bo_buffer<bf16>((size_t)L_padded * DQ);
    buffer<bf16> buffer_k = this->k_proj.create_bo_buffer<bf16>((size_t)L_padded * DK);
    buffer<bf16> buffer_v = this->v_proj.create_bo_buffer<bf16>((size_t)L_padded * DV);
    buffer<bf16> buffer_o = this->o_proj.create_bo_buffer<bf16>((size_t)L_padded * D);
    buffer<bf16> buffer_up = this->up_proj.create_bo_buffer<bf16>((size_t)L_padded *  INTERMEDIATE_SIZE); //shared for o, down
    buffer<bf16> buffer_gate = this->gate_proj.create_bo_buffer<bf16>((size_t)L_padded *  INTERMEDIATE_SIZE); //shared for o, down

    buffer<bf16> buffer_hid = this->down_proj.create_bo_buffer<bf16>((size_t)L_padded * INTERMEDIATE_SIZE);
    buffer<bf16> buffer_x = this->q_proj.create_bo_buffer<bf16>((size_t)L_padded * D);
    buffer<bf16> buffer_attn_out = this->mha_engine.create_bo_buffer<bf16>((size_t)L_padded * DQ);
    buffer<bf16> residual = buffer<bf16>((size_t)L_padded * D); //shared for o, down

    // tensor_2d<bf16> tensor_x(buffer_x, D, L_offset);
    tensor_2d<bf16> tensor_residual(residual, D, L_offset);
    tensor_2d<bf16> tensor_x(buffer_x, D, L_offset);
    tensor_2d<bf16> tensor_attn_out(buffer_o, D, L_offset);

    tensor_2d<bf16> tensor_q(buffer_q, DQ, L_offset);
    tensor_2d<bf16> tensor_k(buffer_k, DK, L_offset);
    // embedding
    buffer<bf16> image_embeddings;
    int image_counter = 0;
    buffer<bf16>* image_buffer = static_cast<buffer<bf16>*>(payload);
    if (is_vlm && image_buffer != nullptr){

        image_embeddings = this->gemma_image_encoder->encode(*image_buffer);

        // // load the reference tensor

        // // swap the image_buffer with the "image_hidden_states_ref" tensor in the reference tensor

        // std::cout <<"ref_image_hidden_states size: " << ref_image_hidden_states.size() << std::endl;

        // image_embeddings = this->gemma_image_encoder->encode<true,true>(ref_image_hidden_states, &reference_tensor);
    }
    for (int i = 0; i < L_effective; i++){
        if (is_vlm && ids[i] == img_soft_token_id){
            assert(payload != nullptr);
            memcpy(tensor_residual[i].data(), image_embeddings.data() + image_counter * 256 * D, 256 * D * sizeof(bf16));
            i += 255;
            image_counter++;
        }
        else{
            this->embedding->forward(ids[i], tensor_residual[i]);
        }
    }

    this->mha_seq_generator->generate_mha_sequence(this->sliding_mha_engine.seq(), L_begin_chunked, L_end_chunked, MAX_L, true, SLIDING_WINDOW);
    this->mha_seq_generator->generate_mha_sequence(this->mha_engine.seq(), L_begin_chunked, L_end_chunked, MAX_L, false, SLIDING_WINDOW);
    for (int layer_idx = 0; layer_idx < num_hidden_layers; layer_idx++){
        tensor_2d<bf16> rms_w(this->rms_weights[layer_idx], D);
        tensor_2d<bf16> rope_rms_w(this->rope_rms_weights[layer_idx], DH);
        if (!this->npu->is_preemption_enabled()){
            this->dequant_all.reset();
            // step 1: dequant all weights
            auto dequant_0_run = this->qkv_dequant.create_run(this->dequanted_qkv_weights, this->proj_weights[layer_idx]);
            auto dequant_1_run = this->o_dequant.create_run(this->dequanted_o_weights, this->proj_weights[layer_idx]);
            auto dequant_2_run = this->up_dequant.create_run(this->dequanted_up_weights, this->proj_weights[layer_idx]);
            auto dequant_3_run = this->gate_dequant.create_run(this->dequanted_gate_weights, this->proj_weights[layer_idx]);
            auto dequant_4_run = this->down_dequant.create_run(this->dequanted_down_weights, this->proj_weights[layer_idx]);
            this->dequant_all.add(dequant_0_run);
            this->dequant_all.add(dequant_1_run);
            this->dequant_all.add(dequant_2_run);
            this->dequant_all.add(dequant_3_run);
            this->dequant_all.add(dequant_4_run);
            this->dequant_all.execute();
            // step 2: apply rms norm
            for (int i = 0; i < L_effective; i++){
                _rms_norm(tensor_x[i], tensor_residual[i], rms_w[0]);
            }
            buffer_x.sync_to_device();
            flm::npu::wait_for_completion(this->dequant_all, "dequant_all");
        }
        else{
            auto dequant_0_run = this->qkv_dequant.create_run(this->dequanted_qkv_weights, this->proj_weights[layer_idx]);
            auto dequant_1_run = this->o_dequant.create_run(this->dequanted_o_weights, this->proj_weights[layer_idx]);
            auto dequant_2_run = this->up_dequant.create_run(this->dequanted_up_weights, this->proj_weights[layer_idx]);
            auto dequant_3_run = this->gate_dequant.create_run(this->dequanted_gate_weights, this->proj_weights[layer_idx]);
            auto dequant_4_run = this->down_dequant.create_run(this->dequanted_down_weights, this->proj_weights[layer_idx]);
            dequant_0_run.start();
            flm::npu::wait_for_completion(dequant_0_run, "dequant_0_run");
            dequant_1_run.start();
            flm::npu::wait_for_completion(dequant_1_run, "dequant_1_run");
            dequant_2_run.start();
            flm::npu::wait_for_completion(dequant_2_run, "dequant_2_run");
            dequant_3_run.start();
            flm::npu::wait_for_completion(dequant_3_run, "dequant_3_run");
            dequant_4_run.start();
            // step 2: apply rms norm
            for (int i = 0; i < L_effective; i++){
                _rms_norm(tensor_x[i], tensor_residual[i], rms_w[0]);
            }
            buffer_x.sync_to_device();
            flm::npu::wait_for_completion(dequant_4_run, "dequant_4_run");
        }

        // step 3: qkv projection
        this->q_proj(buffer_q, buffer_x, this->dequanted_qkv_weights, dequanted_qkv_weights);
        auto run_k = this->k_proj.create_run(buffer_k, buffer_x, this->dequanted_qkv_weights);
        run_k.start();
        // Step 2: apply rope
        buffer_q.sync_from_device();
        for (int ll = 0; ll < L_effective; ll++){
            _rope_rms(tensor_q[ll], ll + L_begin, rope_rms_w[1], layer_idx);
        }
        buffer_q.sync_to_device();
        flm::npu::wait_for_completion(run_k, "run_k");
        buffer_k.sync_from_device();
        auto run_v = this->v_proj.create_run(buffer_v, buffer_x, this->dequanted_qkv_weights);
        run_v.start();
        for (int ll = 0; ll < L_effective; ll++){
            _rope_rms(tensor_k[ll], ll + L_begin, rope_rms_w[2], layer_idx);
        }
        buffer_k.sync_to_device();
        flm::npu::wait_for_completion(run_v, "run_v");
        buffer_q.sync_to_device();
        buffer_v.sync_from_device();

        if (is_sliding_window(layer_idx)){
            fill_kv_cache_sliding(this->sliding_temp_cache, buffer_k, buffer_v, L_offset, L_begin, L_end, L_effective, sliding_l_begin, L_end_chunked, layer_idx);
            this->sliding_temp_cache.sync_to_device();
            this->temp_dirty = true;
        }
        else{
            fill_kv_cache(buffer_k, buffer_v, L_offset, L_begin, L_effective, layer_idx);
        }
        this->kv_caches[layer_idx].sync_to_device();
        // start = time_utils::now();
        // Step 5: apply mha
        if (is_sliding_window(layer_idx)){
            this->sliding_mha_engine(buffer_attn_out, buffer_q, this->sliding_temp_cache);
        }
        else{
            this->mha_engine(buffer_attn_out, buffer_q, this->kv_caches[layer_idx]);
        }
        buffer_attn_out.sync_from_device();
        // end = time_utils::now();
        // mha_time.first += time_utils::duration_us(start, end).first;
        // Step 6: o projection
        this->o_proj(buffer_o, buffer_attn_out, this->dequanted_o_weights);
        buffer_o.sync_from_device();
        for (int i = 0; i < L_effective; i++){
            // Step 7: past_attention_norm
            _rms_norm(tensor_attn_out[i], tensor_attn_out[i], rms_w[1]);
            // Step 8: residual add
            _residual_add(tensor_residual[i], tensor_residual[i], tensor_attn_out[i]);
        }
        // Step 9: rms norm
        for (int i = 0; i < L_effective; i++){
            _rms_norm(tensor_x[i], tensor_residual[i], rms_w[2]);
        }
        buffer_x.sync_to_device();
        // Step 10: up gate projection
        this->up_proj(buffer_up, buffer_x, this->dequanted_up_weights);
        buffer_up.sync_from_device();

        this->gate_proj(buffer_gate, buffer_x, this->dequanted_gate_weights);
        buffer_gate.sync_from_device();
        this->dequanted_gate_weights.sync_from_device();
        // Step 11: glu
        _elementwise_mul(buffer_hid,buffer_gate, buffer_up  );
        buffer_hid.sync_to_device();

        buffer_o.sync_from_device();
        buffer_o.sync_to_device();
        // Step 12: down projection
        this->down_proj(buffer_o, buffer_hid, this->dequanted_down_weights);
        buffer_o.sync_from_device();
        for (int i = 0; i < L_effective; i++){
            // Step 13: post mlp norm
            _rms_norm(tensor_attn_out[i], tensor_attn_out[i], rms_w[3]);
            // Step 14: residual add
            _residual_add(tensor_residual[i], tensor_residual[i], tensor_attn_out[i]);
        }
        buffer_o.sync_to_device();
    }

    buffer<bf16> predict = tensor_residual[L_effective - 1];
    LOG_VERBOSE(1, "predict: " << predict.size() << " " << predict.data());
    buffer<bf16> logits = get_logits(predict);
    this->set_context_length(this->current_context_length + L_effective);

    return logits;
}

buffer<bf16> gemma_npu::Impl::get_k_cache(int layer_idx, int idx){
    buffer<bf16> k_cache(DK);
    if (is_sliding_window(layer_idx)){
        if (idx < this->current_context_length - SLIDING_WINDOW){
            header_print("warning", "Out of sliding window range!");
            k_cache.memset((bf16)0.0f);
        }
        this->kv_caches[layer_idx].sync_from_device();
        bf16* k_cache_ptr = this->kv_caches[layer_idx].data();
        uint32_t read_ptr = idx % SLIDING_WINDOW;
        uint32_t offset = read_ptr * DK / 2;
        uint32_t k23_offset = SLIDING_WINDOW * DK / 2;
        memcpy(static_cast<void*>(k_cache.data()), static_cast<void*>(k_cache_ptr + offset), DK / 2 * sizeof(bf16));
        memcpy(static_cast<void*>(k_cache.data() + DK / 2), static_cast<void*>(k_cache_ptr + offset + k23_offset), DK / 2 * sizeof(bf16));
    }
    else{
        this->kv_caches[layer_idx].sync_from_device();
        bf16* k_cache_ptr = this->kv_caches[layer_idx].data();
        uint32_t offset = idx * DK / 2;
        memcpy(static_cast<void*>(k_cache.data()), static_cast<void*>(k_cache_ptr + offset), DK / 2 * sizeof(bf16));
        memcpy(static_cast<void*>(k_cache.data() + DK / 2), static_cast<void*>(k_cache_ptr + offset + K23_offset), DK / 2 * sizeof(bf16));
    }
    return k_cache;
}

buffer<bf16> gemma_npu::Impl::get_v_cache(int layer_idx, int idx){
    buffer<bf16> v_cache(DV);
    if (is_sliding_window(layer_idx)){
        if (idx < this->current_context_length - SLIDING_WINDOW){
            header_print("warning", "Out of sliding window range!");
            v_cache.memset((bf16)0.0f);
        }
        this->kv_caches[layer_idx].sync_from_device();
        bf16* v_cache_ptr = this->kv_caches[layer_idx].data();
        uint32_t read_ptr = idx % SLIDING_WINDOW;
        uint32_t offset = read_ptr * DV / 2;
        uint32_t v23_offset = SLIDING_WINDOW * DV / 2 * 3;
        uint32_t v01_offset = SLIDING_WINDOW * DV;
        memcpy(static_cast<void*>(v_cache.data()), static_cast<void*>(v_cache_ptr + offset + v01_offset), DV / 2 * sizeof(bf16));
        memcpy(static_cast<void*>(v_cache.data() + DV / 2), static_cast<void*>(v_cache_ptr + offset + v23_offset), DV / 2 * sizeof(bf16));
    }
    else{
        this->kv_caches[layer_idx].sync_from_device();
        bf16* v_cache_ptr = this->kv_caches[layer_idx].data();
        uint32_t offset = idx * DH * 2;
        memcpy(static_cast<void*>(v_cache.data()), static_cast<void*>(v_cache_ptr + offset + V01_offset), DV / 2 * sizeof(bf16));
        memcpy(static_cast<void*>(v_cache.data() + DV / 2), static_cast<void*>(v_cache_ptr + offset + V23_offset), DV / 2 * sizeof(bf16));
    }
    return v_cache;
}

buffer<bf16> gemma_npu::Impl::get_logits(buffer<bf16>& predict){
    this->_rms_norm(this->x, predict, this->final_rms_weight);
    this->x.sync_to_device();
    this->logits.sync_to_device();
    this->lm_head_run.start();
    flm::npu::wait_for_completion(this->lm_head_run, "lm_head_run");
    this->logits.sync_from_device();
    return this->logits_valid;
}

void gemma_npu::Impl::fill_kv_cache(buffer<bf16>& buffer_k, buffer<bf16>& buffer_v, int L_offset, int L_begin, int L_effective, int layer_idx){
    tensor_2d<bf16> k_cache(buffer_k, DK / 2, L_offset * 2);
    tensor_2d<bf16> v_cache(buffer_v, DV / 2, L_offset * 2);

    buffer<bf16> kv_cache = this->kv_caches[layer_idx];
    kv_cache.sync_from_device();

    bf16* k_03_ptr = kv_cache.data() + this->K01_offset + L_begin * DK / 2;
    bf16* k_47_ptr = kv_cache.data() + this->K23_offset + L_begin * DK / 2;
    bf16* v_03_ptr = kv_cache.data() + this->V01_offset + L_begin * DV / 2;
    bf16* v_47_ptr = kv_cache.data() + this->V23_offset + L_begin * DV / 2;

    for (int l = 0; l < L_effective; l++){
        memcpy(k_03_ptr, k_cache[2 * l].data(), DK / 2 * sizeof(bf16));
        memcpy(k_47_ptr, k_cache[2 * l + 1].data(), DK / 2 * sizeof(bf16));
        memcpy(v_03_ptr, v_cache[2 * l].data(), DV / 2 * sizeof(bf16));
        memcpy(v_47_ptr, v_cache[2 * l + 1].data(), DV / 2 * sizeof(bf16));

        k_03_ptr += DK / 2;
        k_47_ptr += DK / 2;
        v_03_ptr += DV / 2;
        v_47_ptr += DV / 2;
    }
}

///@brief move new kv to sliding window kv cache
void gemma_npu::Impl::fill_kv_cache_sliding(buffer<bf16>& local_kv_cache, buffer<bf16>& buffer_k, buffer<bf16>& buffer_v, int L_offset, int L_begin, int L_end, int L_effective, int sliding_l_begin, int sliding_l_end, int layer_idx){
    int in_memory = (L_begin - SLIDING_WINDOW) > 0 ? L_begin - SLIDING_WINDOW : 0;
    int l_idx = in_memory % SLIDING_WINDOW;
    int local_offset = in_memory - sliding_l_begin;

    tensor_2d<bf16> k_cache(buffer_k, DK / 2, L_offset * 2);
    tensor_2d<bf16> v_cache(buffer_v, DV / 2, L_offset * 2);
    int local_length = sliding_l_end - sliding_l_begin;

    this->kv_caches[layer_idx].sync_from_device();
    local_kv_cache.sync_from_device();
    bf16* k_01_ptr = local_kv_cache.data() + this->K01_offset + in_memory * DK / 2;
    bf16* k_23_ptr = local_kv_cache.data() + this->K23_offset + in_memory * DK / 2;
    bf16* v_01_ptr = local_kv_cache.data() + this->V01_offset + in_memory * DV / 2;
    bf16* v_23_ptr = local_kv_cache.data() + this->V23_offset + in_memory * DV / 2;

    buffer<bf16> system_k01 = buffer<bf16>(this->kv_caches[layer_idx].data() + this->sliding_K01_offset, SLIDING_WINDOW * DK / 2);
    buffer<bf16> system_k23 = buffer<bf16>(this->kv_caches[layer_idx].data() + this->sliding_K23_offset, SLIDING_WINDOW * DK / 2);
    buffer<bf16> system_v01 = buffer<bf16>(this->kv_caches[layer_idx].data() + this->sliding_V01_offset, SLIDING_WINDOW * DV / 2);
    buffer<bf16> system_v23 = buffer<bf16>(this->kv_caches[layer_idx].data() + this->sliding_V23_offset, SLIDING_WINDOW * DV / 2);

    tensor_2d<bf16> tensor_system_k01(system_k01, DK / 2, 0);
    tensor_2d<bf16> tensor_system_k23(system_k23, DK / 2, 0);
    tensor_2d<bf16> tensor_system_v01(system_v01, DV / 2, 0);
    tensor_2d<bf16> tensor_system_v23(system_v23, DV / 2, 0);
    // | sliding_l_begin -> in_memory | in_memory -> L_begin | L_begin -> L_end | L_end -> sliding_l_end |
    // | Part 1                       | Part 2               | Part 3           | Part 4                 |

    // part 1: padded, no actual data exists

    // part 2: copy from system kv to local kv
    for (int i = 0; i < (L_begin - in_memory); i++){
        int absolute_idx = in_memory + i;
        int ring_idx = linear2ring(absolute_idx);

        memcpy(k_01_ptr, tensor_system_k01[ring_idx].data(), DK / 2 * sizeof(bf16));
        memcpy(k_23_ptr, tensor_system_k23[ring_idx].data(), DK / 2 * sizeof(bf16));
        memcpy(v_01_ptr, tensor_system_v01[ring_idx].data(), DV / 2 * sizeof(bf16));
        memcpy(v_23_ptr, tensor_system_v23[ring_idx].data(), DV / 2 * sizeof(bf16));

        k_01_ptr += DK / 2;
        k_23_ptr += DK / 2;
        v_01_ptr += DV / 2;
        v_23_ptr += DV / 2;
    }
    // part 3: copy new kv to local kv and system kv
    for (int i = 0; i < (L_end - L_begin); i++){
        int absolute_idx = L_begin + i;
        int ring_idx = linear2ring(absolute_idx);

        memcpy(k_01_ptr, k_cache[2 * i].data(), DK / 2 * sizeof(bf16));
        memcpy(k_23_ptr, k_cache[2 * i + 1].data(), DK / 2 * sizeof(bf16));
        memcpy(v_01_ptr, v_cache[2 * i].data(), DV / 2 * sizeof(bf16));
        memcpy(v_23_ptr, v_cache[2 * i + 1].data(), DV / 2 * sizeof(bf16));

        memcpy(tensor_system_k01[ring_idx].data(), k_cache[2 * i].data(), DK / 2 * sizeof(bf16));
        memcpy(tensor_system_k23[ring_idx].data(), k_cache[2 * i + 1].data(), DK / 2 * sizeof(bf16));
        memcpy(tensor_system_v01[ring_idx].data(), v_cache[2 * i].data(), DV / 2 * sizeof(bf16));
        memcpy(tensor_system_v23[ring_idx].data(), v_cache[2 * i + 1].data(), DV / 2 * sizeof(bf16));

        k_01_ptr += DK / 2;
        k_23_ptr += DK / 2;
        v_01_ptr += DV / 2;
        v_23_ptr += DV / 2;
    }
}
void gemma_npu::Impl::_rms_norm(buffer<bf16>& y, buffer<bf16>& x, buffer<bf16>& rms_weight){
    int n = x.size();
    const int simd_width = 16;
    // NOTE(bit-exactness): index through raw pointers, NOT buffer<T>::operator[].
    // Indexing via buffer<T>::operator[] inside these SIMD loops causes the
    // compiler to emit numerically-divergent code for the identical intrinsic
    // sequence, because the buffer<T> type differs between device-buffer
    // backends. Hoisting to raw pointers pins codegen so results are
    // bit-identical across backends.
    const bf16* xp = x.data();
    const bf16* wp = rms_weight.data();
    bf16* yp = y.data();

    // Step 1: Compute sum of squares using AVX-512 with 4-way unrolling
    __m512 sum_xx_vec0 = _mm512_setzero_ps();
    __m512 sum_xx_vec1 = _mm512_setzero_ps();
    __m512 sum_xx_vec2 = _mm512_setzero_ps();
    __m512 sum_xx_vec3 = _mm512_setzero_ps();

    int i = 0;
    const int unroll = simd_width * 4;
    for (; i <= n - unroll; i += unroll) {
        __m256i bf16_vals0 = _mm256_loadu_si256((__m256i*)&xp[i]);
        __m256i bf16_vals1 = _mm256_loadu_si256((__m256i*)&xp[i + simd_width]);
        __m256i bf16_vals2 = _mm256_loadu_si256((__m256i*)&xp[i + simd_width * 2]);
        __m256i bf16_vals3 = _mm256_loadu_si256((__m256i*)&xp[i + simd_width * 3]);

        __m512 fp32_vals0 = bf16o_fp32_512(bf16_vals0);
        __m512 fp32_vals1 = bf16o_fp32_512(bf16_vals1);
        __m512 fp32_vals2 = bf16o_fp32_512(bf16_vals2);
        __m512 fp32_vals3 = bf16o_fp32_512(bf16_vals3);

        sum_xx_vec0 = _mm512_fmadd_ps(fp32_vals0, fp32_vals0, sum_xx_vec0);
        sum_xx_vec1 = _mm512_fmadd_ps(fp32_vals1, fp32_vals1, sum_xx_vec1);
        sum_xx_vec2 = _mm512_fmadd_ps(fp32_vals2, fp32_vals2, sum_xx_vec2);
        sum_xx_vec3 = _mm512_fmadd_ps(fp32_vals3, fp32_vals3, sum_xx_vec3);
    }

    // Handle remaining elements
    for (; i <= n - simd_width; i += simd_width) {
        __m256i bf16_vals = _mm256_loadu_si256((__m256i*)&xp[i]);
        __m512 fp32_vals = bf16o_fp32_512(bf16_vals);
        sum_xx_vec0 = _mm512_fmadd_ps(fp32_vals, fp32_vals, sum_xx_vec0);
    }

    // Combine accumulators
    __m512 sum_combined = _mm512_add_ps(_mm512_add_ps(sum_xx_vec0, sum_xx_vec1),
                                         _mm512_add_ps(sum_xx_vec2, sum_xx_vec3));

    // Efficient horizontal sum using AVX-512 reduction
    f32 sum_xx = _mm512_reduce_add_ps(sum_combined);

    sum_xx /= (f32)n;
    f32 rms_x = sqrtf(sum_xx + 1e-6f);

    // Step 2: Compute element-wise operation y[i] = w[i] * x[i] / rms_x using AVX-512
    __m512 rms_x_vec = _mm512_set1_ps(rms_x);
    __m512 inv_rms_x_vec = _mm512_rcp14_ps(rms_x_vec);  // Fast reciprocal approximation

    // 4-way unrolled normalization loop
    for (i = 0; i <= n - unroll; i += unroll) {
        __m256i bf16_vals0 = _mm256_loadu_si256((__m256i*)&xp[i]);
        __m256i bf16_vals1 = _mm256_loadu_si256((__m256i*)&xp[i + simd_width]);
        __m256i bf16_vals2 = _mm256_loadu_si256((__m256i*)&xp[i + simd_width * 2]);
        __m256i bf16_vals3 = _mm256_loadu_si256((__m256i*)&xp[i + simd_width * 3]);

        __m512 fp32_vals0 = bf16o_fp32_512(bf16_vals0);
        __m512 fp32_vals1 = bf16o_fp32_512(bf16_vals1);
        __m512 fp32_vals2 = bf16o_fp32_512(bf16_vals2);
        __m512 fp32_vals3 = bf16o_fp32_512(bf16_vals3);

        __m256i bf16_vals_w0 = _mm256_loadu_si256((__m256i*)&wp[i]);
        __m256i bf16_vals_w1 = _mm256_loadu_si256((__m256i*)&wp[i + simd_width]);
        __m256i bf16_vals_w2 = _mm256_loadu_si256((__m256i*)&wp[i + simd_width * 2]);
        __m256i bf16_vals_w3 = _mm256_loadu_si256((__m256i*)&wp[i + simd_width * 3]);

        __m512 fp32_vals_w0 = bf16o_fp32_512(bf16_vals_w0);
        __m512 fp32_vals_w1 = bf16o_fp32_512(bf16_vals_w1);
        __m512 fp32_vals_w2 = bf16o_fp32_512(bf16_vals_w2);
        __m512 fp32_vals_w3 = bf16o_fp32_512(bf16_vals_w3);

        __m512 y_vec0 = _mm512_mul_ps(_mm512_mul_ps(fp32_vals_w0, fp32_vals0), inv_rms_x_vec);
        __m512 y_vec1 = _mm512_mul_ps(_mm512_mul_ps(fp32_vals_w1, fp32_vals1), inv_rms_x_vec);
        __m512 y_vec2 = _mm512_mul_ps(_mm512_mul_ps(fp32_vals_w2, fp32_vals2), inv_rms_x_vec);
        __m512 y_vec3 = _mm512_mul_ps(_mm512_mul_ps(fp32_vals_w3, fp32_vals3), inv_rms_x_vec);

        __m256i result_bf16_0 = f32o_bf16_512(y_vec0);
        __m256i result_bf16_1 = f32o_bf16_512(y_vec1);
        __m256i result_bf16_2 = f32o_bf16_512(y_vec2);
        __m256i result_bf16_3 = f32o_bf16_512(y_vec3);

        _mm256_storeu_si256((__m256i*)&yp[i], result_bf16_0);
        _mm256_storeu_si256((__m256i*)&yp[i + simd_width], result_bf16_1);
        _mm256_storeu_si256((__m256i*)&yp[i + simd_width * 2], result_bf16_2);
        _mm256_storeu_si256((__m256i*)&yp[i + simd_width * 3], result_bf16_3);
    }

    // Handle remaining elements
    for (; i <= n - simd_width; i += simd_width) {
        __m256i bf16_vals = _mm256_loadu_si256((__m256i*)&xp[i]);
        __m512 fp32_vals = bf16o_fp32_512(bf16_vals);
        __m256i bf16_vals_w = _mm256_loadu_si256((__m256i*)&wp[i]);
        __m512 fp32_vals_w = bf16o_fp32_512(bf16_vals_w);
        __m512 y_vec = _mm512_mul_ps(_mm512_mul_ps(fp32_vals_w, fp32_vals), inv_rms_x_vec);
        __m256i result_bf16 = f32o_bf16_512(y_vec);
        _mm256_storeu_si256((__m256i*)&yp[i], result_bf16);
    }
}

void gemma_npu::Impl::_rope_rms(buffer<bf16>& x, int idx, buffer<bf16>& rms_weight, int layer_idx){
    int heads = x.size() / DH;
    buffer<f32> cos_buf(DH / 2);
    buffer<f32> sin_buf(DH / 2);

    // Pre-compute cos/sin buffers
    // Note: There's no efficient SIMD implementation for trigonometric functions in standard intrinsics.
    // Intel SVML provides vectorized sin/cos (_mm512_sin_ps, _mm512_cos_ps) but requires linking with libsvml.
    // For portability, we use scalar computation here, which is done once per token and is relatively fast.
    for (uint32_t i = 0; i < DH / 2; i++){
        if (is_sliding_window(layer_idx)){
            cos_buf[i] = std::cos(this->inv_freq_local[i] * idx);
            sin_buf[i] = std::sin(this->inv_freq_local[i] * idx);
        }
        else{
            cos_buf[i] = std::cos(this->inv_freq[i] * idx);
            sin_buf[i] = std::sin(this->inv_freq[i] * idx);
        }
    }

    const int simd_width = 16;  // AVX-512 processes 16 float32 at once

    for (int h = 0; h < heads; h++){
        buffer<bf16> x_head(x.data() + h * DH, DH);
        _rms_norm(x_head, x_head, rms_weight);
        buffer<bf16> x_left(x_head.data(), DH / 2);
        buffer<bf16> x_right(x_head.data() + DH / 2, DH / 2);

        int i = 0;
        // AVX-512 vectorized main loop - process 16 elements at a time
        for (; i + simd_width <= DH / 2; i += simd_width) {
            // Load 16 BF16 → float32 using AVX-512
            __m256i left_bf16  = _mm256_loadu_si256((__m256i*)(x_left.data()  + i));
            __m256i right_bf16 = _mm256_loadu_si256((__m256i*)(x_right.data() + i));
            __m512  L = bf16o_fp32_512(left_bf16);
            __m512  R = bf16o_fp32_512(right_bf16);

            // Load sin/cos (16 float32 values)
            __m512 C = _mm512_loadu_ps(cos_buf.data() + i);
            __m512 S = _mm512_loadu_ps(sin_buf.data() + i);

            // new_left  = L*C - R*S using FMA for better performance
            __m512 newL = _mm512_fmsub_ps(L, C, _mm512_mul_ps(R, S));
            // new_right = L*S + R*C using FMA
            __m512 newR = _mm512_fmadd_ps(L, S, _mm512_mul_ps(R, C));

            // Convert back to BF16 and store
            __m256i newL_bf16 = f32o_bf16_512(newL);
            __m256i newR_bf16 = f32o_bf16_512(newR);
            _mm256_storeu_si256((__m256i*)(x_left.data()  + i), newL_bf16);
            _mm256_storeu_si256((__m256i*)(x_right.data() + i), newR_bf16);
        }

        // Handle remaining elements with AVX2 (8 elements at a time)
        const int simd_avx2 = 8;
        for (; i + simd_avx2 <= DH / 2; i += simd_avx2) {
            __m128i left_bf16  = _mm_loadu_si128((__m128i*)(x_left.data()  + i));
            __m128i right_bf16 = _mm_loadu_si128((__m128i*)(x_right.data() + i));
            __m256  L = bf16o_fp32(left_bf16);
            __m256  R = bf16o_fp32(right_bf16);

            __m256 C = _mm256_loadu_ps(cos_buf.data() + i);
            __m256 S = _mm256_loadu_ps(sin_buf.data() + i);

            __m256 newL = _mm256_sub_ps(_mm256_mul_ps(L, C), _mm256_mul_ps(R, S));
            __m256 newR = _mm256_add_ps(_mm256_mul_ps(L, S), _mm256_mul_ps(R, C));

            __m128i newL_bf16 = f32o_bf16(newL);
            __m128i newR_bf16 = f32o_bf16(newR);
            _mm_storeu_si128((__m128i*)(x_left.data()  + i), newL_bf16);
            _mm_storeu_si128((__m128i*)(x_right.data() + i), newR_bf16);
        }
    }
}

void gemma_npu::Impl::_residual_add(buffer<bf16>& dest, buffer<bf16>& src, buffer<bf16>& residual){
    int n = src.size();
    const int simd_width = 16;
    const int unroll = simd_width * 4;

    int i = 0;
    // 4-way unrolled loop for maximum throughput
    for (; i <= n - unroll; i += unroll){
        __m256i bf16_vals0 = _mm256_loadu_si256((__m256i*)&src[i]);
        __m256i bf16_vals1 = _mm256_loadu_si256((__m256i*)&src[i + simd_width]);
        __m256i bf16_vals2 = _mm256_loadu_si256((__m256i*)&src[i + simd_width * 2]);
        __m256i bf16_vals3 = _mm256_loadu_si256((__m256i*)&src[i + simd_width * 3]);

        __m256i bf16_vals_r0 = _mm256_loadu_si256((__m256i*)&residual[i]);
        __m256i bf16_vals_r1 = _mm256_loadu_si256((__m256i*)&residual[i + simd_width]);
        __m256i bf16_vals_r2 = _mm256_loadu_si256((__m256i*)&residual[i + simd_width * 2]);
        __m256i bf16_vals_r3 = _mm256_loadu_si256((__m256i*)&residual[i + simd_width * 3]);

        __m512 fp32_vals0 = bf16o_fp32_512(bf16_vals0);
        __m512 fp32_vals1 = bf16o_fp32_512(bf16_vals1);
        __m512 fp32_vals2 = bf16o_fp32_512(bf16_vals2);
        __m512 fp32_vals3 = bf16o_fp32_512(bf16_vals3);

        __m512 fp32_vals_r0 = bf16o_fp32_512(bf16_vals_r0);
        __m512 fp32_vals_r1 = bf16o_fp32_512(bf16_vals_r1);
        __m512 fp32_vals_r2 = bf16o_fp32_512(bf16_vals_r2);
        __m512 fp32_vals_r3 = bf16o_fp32_512(bf16_vals_r3);

        __m512 result_vec0 = _mm512_add_ps(fp32_vals0, fp32_vals_r0);
        __m512 result_vec1 = _mm512_add_ps(fp32_vals1, fp32_vals_r1);
        __m512 result_vec2 = _mm512_add_ps(fp32_vals2, fp32_vals_r2);
        __m512 result_vec3 = _mm512_add_ps(fp32_vals3, fp32_vals_r3);

        __m256i result_bf16_0 = f32o_bf16_512(result_vec0);
        __m256i result_bf16_1 = f32o_bf16_512(result_vec1);
        __m256i result_bf16_2 = f32o_bf16_512(result_vec2);
        __m256i result_bf16_3 = f32o_bf16_512(result_vec3);

        _mm256_storeu_si256((__m256i*)&dest[i], result_bf16_0);
        _mm256_storeu_si256((__m256i*)&dest[i + simd_width], result_bf16_1);
        _mm256_storeu_si256((__m256i*)&dest[i + simd_width * 2], result_bf16_2);
        _mm256_storeu_si256((__m256i*)&dest[i + simd_width * 3], result_bf16_3);
    }

    // Handle remaining elements
    for (; i <= n - simd_width; i += simd_width){
        __m256i bf16_vals = _mm256_loadu_si256((__m256i*)&src[i]);
        __m256i bf16_vals_r = _mm256_loadu_si256((__m256i*)&residual[i]);
        __m512 fp32_vals = bf16o_fp32_512(bf16_vals);
        __m512 fp32_vals_r = bf16o_fp32_512(bf16_vals_r);
        __m512 result_vec = _mm512_add_ps(fp32_vals, fp32_vals_r);
        __m256i result_bf16 = f32o_bf16_512(result_vec);
        _mm256_storeu_si256((__m256i*)&dest[i], result_bf16);
    }
}

void gemma_npu::Impl::_elementwise_mul(buffer<bf16>& dest, buffer<bf16>& a, buffer<bf16>& b){
    int n = a.size();
    const int simd_width = 16;
    const int unroll = simd_width * 4;

    int i = 0;
    // 4-way unrolled loop for maximum throughput
    for (; i <= n - unroll; i += unroll){
        __m256i bf16_vals_a0 = _mm256_loadu_si256((__m256i*)&a[i]);
        __m256i bf16_vals_a1 = _mm256_loadu_si256((__m256i*)&a[i + simd_width]);
        __m256i bf16_vals_a2 = _mm256_loadu_si256((__m256i*)&a[i + simd_width * 2]);
        __m256i bf16_vals_a3 = _mm256_loadu_si256((__m256i*)&a[i + simd_width * 3]);

        __m256i bf16_vals_b0 = _mm256_loadu_si256((__m256i*)&b[i]);
        __m256i bf16_vals_b1 = _mm256_loadu_si256((__m256i*)&b[i + simd_width]);
        __m256i bf16_vals_b2 = _mm256_loadu_si256((__m256i*)&b[i + simd_width * 2]);
        __m256i bf16_vals_b3 = _mm256_loadu_si256((__m256i*)&b[i + simd_width * 3]);

        __m512 fp32_vals_a0 = bf16o_fp32_512(bf16_vals_a0);
        __m512 fp32_vals_a1 = bf16o_fp32_512(bf16_vals_a1);
        __m512 fp32_vals_a2 = bf16o_fp32_512(bf16_vals_a2);
        __m512 fp32_vals_a3 = bf16o_fp32_512(bf16_vals_a3);

        __m512 fp32_vals_b0 = bf16o_fp32_512(bf16_vals_b0);
        __m512 fp32_vals_b1 = bf16o_fp32_512(bf16_vals_b1);
        __m512 fp32_vals_b2 = bf16o_fp32_512(bf16_vals_b2);
        __m512 fp32_vals_b3 = bf16o_fp32_512(bf16_vals_b3);

        __m512 result_vec0 = _mm512_mul_ps(fp32_vals_a0, fp32_vals_b0);
        __m512 result_vec1 = _mm512_mul_ps(fp32_vals_a1, fp32_vals_b1);
        __m512 result_vec2 = _mm512_mul_ps(fp32_vals_a2, fp32_vals_b2);
        __m512 result_vec3 = _mm512_mul_ps(fp32_vals_a3, fp32_vals_b3);

        __m256i result_bf16_0 = f32o_bf16_512(result_vec0);
        __m256i result_bf16_1 = f32o_bf16_512(result_vec1);
        __m256i result_bf16_2 = f32o_bf16_512(result_vec2);
        __m256i result_bf16_3 = f32o_bf16_512(result_vec3);

        _mm256_storeu_si256((__m256i*)&dest[i], result_bf16_0);
        _mm256_storeu_si256((__m256i*)&dest[i + simd_width], result_bf16_1);
        _mm256_storeu_si256((__m256i*)&dest[i + simd_width * 2], result_bf16_2);
        _mm256_storeu_si256((__m256i*)&dest[i + simd_width * 3], result_bf16_3);
    }

    // Handle remaining elements
    for (; i <= n - simd_width; i += simd_width){
        __m256i bf16_vals_a = _mm256_loadu_si256((__m256i*)&a[i]);
        __m256i bf16_vals_b = _mm256_loadu_si256((__m256i*)&b[i]);
        __m512 fp32_vals_a = bf16o_fp32_512(bf16_vals_a);
        __m512 fp32_vals_b = bf16o_fp32_512(bf16_vals_b);
        __m512 result_vec = _mm512_mul_ps(fp32_vals_a, fp32_vals_b);
        __m256i result_bf16 = f32o_bf16_512(result_vec);
        _mm256_storeu_si256((__m256i*)&dest[i], result_bf16);
    }
}

void gemma_npu::Impl::clear_context(){
    // prefill attends over whole chunks, so rows past the prompt must not keep
    // an earlier request's k/v: they make identical requests give different logits
    for (auto& kv_cache : this->kv_caches){
        kv_cache.sync_from_device();
        memset(static_cast<void*>(kv_cache.data()), 0, kv_cache.size() * sizeof(bf16));
        kv_cache.sync_to_device();
    }
    this->dirty_rows = 0;
    this->_zero_temp_cache();
    this->set_context_length(0);
}

void gemma_npu::Impl::_zero_temp_cache(){
    if (!this->temp_dirty) return;
    this->sliding_temp_cache.memset((bf16)0);
    this->sliding_temp_cache.sync_to_device();
    this->temp_dirty = false;
}

kv_row_layout gemma_npu::Impl::kv_layout() const{
    kv_row_layout l;
    l.DK = DK; l.DV = DV; l.striped = true;
    l.k47 = K23_offset; l.v03 = V01_offset; l.v47 = V23_offset;
    return l;
}

std::unique_ptr<hybrid_engine_state> gemma_npu::Impl::new_state(){
    size_t ring_bytes = 0, global_layers = 0;
    for (size_t i = 0; i < kv_caches.size(); i++){
        if (attn_mask[i]) global_layers++;
        else ring_bytes += kv_caches[i].size() * sizeof(bf16);
    }
    return kv_rows_new_state(global_layers, kv_layout(), ring_bytes);
}

int gemma_npu::Impl::read_state(hybrid_engine_state& state, int kv_from){
    header_print_r("FLM", "Reading state at context length " + std::to_string(this->current_context_length) + ", K/V rows from " + std::to_string(kv_from));
    const int length = kv_rows_read(state, this->kv_caches, kv_layout(), this->current_context_length, kv_from, -1, &attn_mask);
    if (length < 0) return -1;
    for (size_t i = 0; i < kv_caches.size(); i++){
        if (attn_mask[i]) continue;
        this->kv_caches[i].sync_from_device();
        state.state[i] = buffer<bf16>(this->kv_caches[i].size());
        memcpy(static_cast<void*>(state.state[i].data()), static_cast<void*>(this->kv_caches[i].data()), this->kv_caches[i].size() * sizeof(bf16));
    }
    return length;
}

int gemma_npu::Impl::write_state_kv(const hybrid_engine_state& state){
    return kv_rows_write(state, this->kv_caches, kv_layout(), this->MAX_L, &attn_mask);
}

int gemma_npu::Impl::write_state(const hybrid_engine_state& state){
    // check the rings before anything is written
    bool ok = state.state.size() == this->kv_caches.size();
    for (size_t i = 0; ok && i < kv_caches.size(); i++){
        if (!attn_mask[i] && state.state[i].size() != this->kv_caches[i].size()) ok = false;
    }
    const int length = ok ? kv_rows_write(state, this->kv_caches, kv_layout(), this->MAX_L, &attn_mask) : -1;
    if (length < 0){
        header_print_r("FLM", "No valid state to write");
        return -1;
    }
    header_print_r("FLM", "Writing state at context length " + std::to_string(length));
    for (size_t i = 0; i < kv_caches.size(); i++){
        if (attn_mask[i]) continue;
        this->kv_caches[i].sync_from_device();
        memcpy(static_cast<void*>(this->kv_caches[i].data()), static_cast<const void*>(state.state[i].data()), this->kv_caches[i].size() * sizeof(bf16));
        this->kv_caches[i].sync_to_device();
    }
    // rows past the length hold what the requests since the state was read left there: put back the zeros of a fresh engine
    kv_rows_zero(this->kv_caches, kv_layout(), static_cast<size_t>(length), this->dirty_rows, &attn_mask);
    this->dirty_rows = static_cast<size_t>(length);
    this->set_context_length(length);
    return length;
}

int gemma_npu::Impl::get_current_context_length(){
    return this->current_context_length;
}

void gemma_npu::Impl::update_max_length(uint32_t MAX_L){
    if (MAX_L <= this->MAX_L){
        header_print("FLM", "New length is shorter than the current length, no need to update!");
        return; // no need to update
	}
    this->MAX_L = MAX_L;
    size_t kv_cache_size = (size_t)MAX_L * DK * 2;   // DK == num_key_value_heads * head_dim
    buffer<bf16> kv_cache_temp(this->kv_caches[this->SLIDING_WINDOW_PATTERN - 1].size());
    size_t original_k01_offset = this->K01_offset;
    size_t original_k23_offset = this->K23_offset;
    size_t original_v01_offset = this->V01_offset;
    size_t original_v23_offset = this->V23_offset;
    this->sequence->set_max_length(MAX_L);
    this->K01_offset = this->sequence->get_k01_offset();
    this->K23_offset = this->sequence->get_k23_offset();
    this->V01_offset = this->sequence->get_v01_offset();
    this->V23_offset = this->sequence->get_v23_offset();
    for (uint32_t i = 0; i < num_hidden_layers; i++){
        if (is_sliding_window(i)){
            continue;
        }
        this->kv_caches[i].sync_from_device();
        memcpy(static_cast<void*>(kv_cache_temp.data()), static_cast<void*>(this->kv_caches[i].data()), this->kv_caches[i].size() * sizeof(bf16));
        this->kv_caches[i] = layer.create_bo_buffer<bf16>(kv_cache_size);
        this->kv_caches[i].memset((bf16)0); // the rows past the old size must start zero

        memcpy(static_cast<void*>(this->kv_caches[i].data() + this->K01_offset), static_cast<void*>(kv_cache_temp.data() + original_k01_offset), (kv_cache_temp.size() / 4) * sizeof(bf16));
        memcpy(static_cast<void*>(this->kv_caches[i].data() + this->V01_offset), static_cast<void*>(kv_cache_temp.data() + original_v01_offset), (kv_cache_temp.size() / 4) * sizeof(bf16));
        memcpy(static_cast<void*>(this->kv_caches[i].data() + this->K23_offset), static_cast<void*>(kv_cache_temp.data() + original_k23_offset), (kv_cache_temp.size() / 4) * sizeof(bf16));
        memcpy(static_cast<void*>(this->kv_caches[i].data() + this->V23_offset), static_cast<void*>(kv_cache_temp.data() + original_v23_offset), (kv_cache_temp.size() / 4) * sizeof(bf16));
        this->kv_caches[i].sync_to_device();
    }
    this->sliding_temp_cache = this->mha_engine.create_bo_buffer<bf16>(kv_cache_size);
    this->temp_dirty = true; // the new scratch was never zeroed
    this->set_context_length(this->current_context_length); // regenerate sequence
    this->is_checkpoint_valid = false; // invalidate checkpoint since the underlying buffer has changed
}

void gemma_npu::Impl::_allocate_checkpoint_buffers(){
    this->kv_checkpoint.clear();
    this->kv_checkpoint.resize(num_hidden_layers);
    for (int i = 0; i < num_hidden_layers; i++){
        if (is_sliding_window(i)){
            this->kv_checkpoint[i] = layer.create_bo_buffer<bf16>(this->kv_caches[i].size());
        }
    }
}

int gemma_npu::Impl::checkpoint() {
    header_print_r("FLM", "Creating checkpoint at context length " + std::to_string(this->current_context_length));
    if (!is_checkpoint_valid) {
        _allocate_checkpoint_buffers();
    } // lazy allocation of checkpoint buffers, only allocate when checkpoint is called for the first time
    for (int i = 0; i < num_hidden_layers; i++){
        if (is_sliding_window(i)){
            this->kv_caches[i].sync_from_device();
            memcpy(static_cast<void*>(this->kv_checkpoint[i].data()), static_cast<void*>(this->kv_caches[i].data()), this->kv_caches[i].size() * sizeof(bf16));
            this->kv_caches[i].sync_to_device();
        }
    }
    is_checkpoint_valid = true;
    checkpoint_context_length = this->current_context_length;
    return checkpoint_context_length;
}

int gemma_npu::Impl::restore() {
    if (!is_checkpoint_valid) {
        header_print("FLM", "No valid checkpoint found!");
        return -1;
    }
    header_print_r("FLM", "Restoring checkpoint at context length " + std::to_string(checkpoint_context_length));
    for (int i = 0; i < num_hidden_layers; i++){
        if (is_sliding_window(i)){
            this->kv_caches[i].sync_from_device();
            memcpy(static_cast<void*>(this->kv_caches[i].data()), static_cast<void*>(this->kv_checkpoint[i].data()), this->kv_caches[i].size() * sizeof(bf16));
            this->kv_caches[i].sync_to_device();
        }
    }
    this->set_context_length(checkpoint_context_length); // regenerate sequence
    return checkpoint_context_length;
}

///@brief destructor of gemma_npu
gemma_npu::Impl::~Impl() = default;
/*
gemma_npu::Impl::~Impl(){
    for (int i = 0; i < num_hidden_layers; i++){
        this->kv_caches[i].free();
        this->proj_weights[i].free();
        this->rms_weights[i].free();
    }
    this->kv_caches.clear();
    this->rms_weights.clear();
    this->proj_weights.clear();
    this->rope_weights.release();
    this->weights_p0.release();
    this->weights_p1.release();
    this->x.release();
    this->lm_head.reset();
}
*/
// ===============================================
// Externals for gemma_npu
// ===============================================
gemma_npu::gemma_npu(LM_Config config, npu_xclbin_manager *npu_instance, int MAX_L){
    this->_impl = new Impl(config, npu_instance, MAX_L);
}

buffer<bf16> gemma_npu::forward(int ids){
    return this->_impl->forward(ids);
}

buffer<bf16> gemma_npu::prefill(std::vector<int>& ids, void* payload){
    try {
        buffer<bf16> result = this->_impl->prefill(ids, payload);
        return result;
    }
    catch (const std::runtime_error& e) {
        header_print("FLM", e.what());
        throw;
    }
}

void gemma_npu::set_context_length(int L){
    this->_impl->set_context_length(L);
}

void gemma_npu::load_weights(Q4NX& q4nx){
    this->_impl->load_weights(q4nx);
}

void gemma_npu::update_max_length(uint32_t MAX_L){
    this->_impl->update_max_length(MAX_L);
}

void gemma_npu::clear_context(){
    this->_impl->clear_context();
}

buffer<bf16> gemma_npu::get_k_cache(int layer_idx, int idx){
    return this->_impl->get_k_cache(layer_idx, idx);
}

buffer<bf16> gemma_npu::get_v_cache(int layer_idx, int idx){
    return this->_impl->get_v_cache(layer_idx, idx);
}

int gemma_npu::get_current_context_length(){
    return this->_impl->get_current_context_length();
}

int gemma_npu::checkpoint(){
    return this->_impl->checkpoint();
}

int gemma_npu::restore(){
    return this->_impl->restore();
}

std::unique_ptr<hybrid_engine_state> gemma_npu::new_state(){
    return this->_impl->new_state();
}

int gemma_npu::read_state(hybrid_engine_state& state, int kv_from){
    return this->_impl->read_state(state, kv_from);
}

int gemma_npu::write_state(const hybrid_engine_state& state){
    return this->_impl->write_state(state);
}

int gemma_npu::write_state_kv(const hybrid_engine_state& state){
    return this->_impl->write_state_kv(state);
}

extern "C" const flm_state_api* gemma_npu_state_api(void* engine){
    return hybrid_state_api<gemma_npu>(engine);
}

gemma_npu::~gemma_npu(){
    delete this->_impl;
}
