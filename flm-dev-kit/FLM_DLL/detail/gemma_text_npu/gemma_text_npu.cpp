// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include "gemma_text_npu_detail.hpp"
#include <chrono>
#include <thread>
#include "metrices.hpp"

gemma_text_npu::Impl::Impl(LM_Config config, npu_xclbin_manager *npu_instance, int MAX_L)
    : config(config), npu(npu_instance){
    num_hidden_layers = config.get("num_hidden_layers");
    current_context_length = 0;
    D = config.get("hidden_size");
    uint32_t PADDED_D = (D + 255) / 256 * 256;
    total_error = 0.0f;
    is_checkpoint_valid = false;
    DH = config.get("head_dim");
    DQ = config.get("head_dim") * config.get("num_attention_heads");
    DK = config.get("head_dim") * config.get("num_key_value_heads");
    DV = config.get("head_dim") * config.get("num_key_value_heads");
    INTERMEDIATE_SIZE = config.get("intermediate_size");
    HIDDEN_SIZE = config.get("intermediate_size");
    SLIDING_WINDOW = config.get("sliding_window");
    SLIDING_WINDOW_PATTERN = config.get("sliding_window_pattern");

    if (config.get("hidden_size") == 640){
        this->MAX_L = std::max(MAX_L, 16384);
    } else if (config.get("hidden_size") == 1152){
        this->MAX_L = std::max(MAX_L, 8192);
    } else {
        throw std::runtime_error("Unsupported hidden size: " + std::to_string(config.get("hidden_size")));
    }
    // make MAX_L a power of 2
    float log2_MAX_L = std::log2(this->MAX_L);
    if (log2_MAX_L != std::floor(log2_MAX_L)){
        this->MAX_L = 1 << ((int)std::floor(log2_MAX_L) + 1);
    }
    MAX_L = this->MAX_L; // synchronize MAX_L

    mha_type_t mha_type = mha_type_t::mha_d256_q4;

    this->gemm = std::make_unique<GemmaTextGemm>(config);
    this->dequant = std::make_unique<GemmaTextDequant>(config);

    this->mha_sequence_gen = std::make_unique<MHA>(mha_type, 1);

    this->sequence = std::make_unique<gemma_text_npu_sequence>(config, MAX_L);

    layer_app_manager = npu->register_xclbin(utils::path_join(config.exec_path, "xclbins", config.model_name, "layer.xclbin"));
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
    this->up_gate_proj = gemm_app_manager->create_app();
    this->down_proj = gemm_app_manager->create_app();

    dequant_app_manager = npu->register_xclbin(utils::path_join(config.exec_path, "xclbins", config.model_name, "dequant.xclbin"));
    this->dequant_0 = dequant_app_manager->create_app();
    this->dequant_1 = dequant_app_manager->create_app();
    this->dequant_2 = dequant_app_manager->create_app();
    this->dequant_3 = dequant_app_manager->create_app();
    if (!this->npu->is_preemption_enabled()){
        this->dequant_all = dequant_app_manager->create_runlist();
    }
    mha_app_manager = npu->register_xclbin(utils::path_join(config.exec_path, "xclbins", config.model_name, "attn.xclbin"));
    this->mha_engine = mha_app_manager->create_app();
    this->sliding_mha_engine = mha_app_manager->create_app();

    this->lm_head = std::make_unique<GemmaTextLMHead>(config, npu);

    // allocate all buffers
    rms_weights.resize(num_hidden_layers);
    proj_weights.resize(num_hidden_layers);
    kv_caches.resize(num_hidden_layers);
    rope_rms_weights.resize(num_hidden_layers);
    attn_mask.resize(num_hidden_layers);
    for (uint32_t i = 0; i < num_hidden_layers; i++) attn_mask[i] = !is_sliding_window(i);

    size_t total_proj_weights =
          (config.get("num_attention_heads") * config.get("head_dim")) * config.get("hidden_size")
        + (config.get("num_key_value_heads") * config.get("head_dim")) * config.get("hidden_size") * 2
        + (config.get("num_attention_heads") * config.get("head_dim")) * PADDED_D
        + config.get("intermediate_size") * config.get("hidden_size") * 2
        + config.get("intermediate_size") * PADDED_D;
    size_t total_proj_weights_size = total_proj_weights * 5 / 8;
    size_t kv_cache_size = (size_t)MAX_L * config.get("num_key_value_heads") * config.get("head_dim") * 2;
    size_t kv_cache_size_sliding = (size_t)SLIDING_WINDOW * config.get("num_key_value_heads") * config.get("head_dim") * 2;

    this->sliding_temp_cache = this->mha_engine.create_bo_buffer<bf16>(kv_cache_size);
    this->sliding_temp_cache.memset((bf16)0);
    this->sliding_temp_cache.sync_to_device();

    for (int i = num_hidden_layers - 1; i >= 0; i--){
        proj_weights[i] = layer.create_bo_buffer<u8>(total_proj_weights_size);
        kv_caches[i] = layer.create_bo_buffer<bf16>(is_sliding_window(i) ? kv_cache_size_sliding : kv_cache_size);
        rope_rms_weights[i] = layer.create_bo_buffer<bf16>(config.get("head_dim") * 3);
        rms_weights[i] = layer.create_bo_buffer<bf16>(config.get("hidden_size") * 4);
        this->kv_caches[i].memset((bf16)0);
        this->kv_caches[i].sync_to_device();
    }
    x = layer.create_bo_buffer<bf16>(D);

    this->K_offset = sequence->get_k_offset();
    this->V_offset = sequence->get_v_offset();
    this->sliding_K_offset = SLIDING_WINDOW * config.get("num_key_value_heads") * config.get("head_dim") * 0;
    this->sliding_V_offset = (size_t)SLIDING_WINDOW * config.get("num_key_value_heads") * config.get("head_dim") * 1;

    this->set_context_length(0);
    this->embedding = std::make_unique<Embedding>(config.get("vocab_size"), config.get("hidden_size"));

    this->x_lm_head_exposed = this->lm_head->x_exposed();
    this->logits_temp = buffer<bf16>(this->config.get("vocab_size"));
    this->weights_p0 = this->dequant_0.create_bo_buffer<bf16>((size_t)D * (DQ + DK + DV));
    this->weights_p1 = this->dequant_1.create_bo_buffer<bf16>((size_t)DQ * PADDED_D);
    this->weights_p2 = this->dequant_2.create_bo_buffer<bf16>((size_t)D * INTERMEDIATE_SIZE * 2);
    this->weights_p3 = this->dequant_3.create_bo_buffer<bf16>((size_t)INTERMEDIATE_SIZE * PADDED_D);

    this->dequant->generate_seq(this->dequant_0.seq(), D, DQ + DK + DV, 0);
    this->dequant->generate_seq(this->dequant_1.seq(), DQ, PADDED_D, D * (DQ + DK + DV) * 5 / 8);
    this->dequant->generate_seq(this->dequant_2.seq(), D, INTERMEDIATE_SIZE * 2, (D * (DQ + DK + DV) + DQ * PADDED_D) * 5 / 8);
    this->dequant->generate_seq(this->dequant_3.seq(), INTERMEDIATE_SIZE, PADDED_D, ((D * (DQ + DK + DV) + DQ * PADDED_D) + D * INTERMEDIATE_SIZE * 2) * 5 / 8);
}

void gemma_text_npu::Impl::load_weights(Q4NX& q4nx){
    // read qkv weights
    uint32_t PADDED_D = (D + 255) / 256 * 256;
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
            buffer<u8> w_o_padded = buffer<u8>(w_ptr, (PADDED_D - D) * DQ * 5 / 8);
            w_o_padded.memset(0);
            w_ptr += (PADDED_D - D) * DQ * 5 / 8;
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
            const int blocks = 256 / 2 / 16 * (D / 128);
            const int chunk_size = blocks * 16 * 128 * 5 / 8;
            const int total_size = HIDDEN_SIZE * D * 5 / 8;
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
            memcpy(w_ptr, w_down.data(), HIDDEN_SIZE * D * 5 / 8);
            w_ptr += HIDDEN_SIZE * D * 5 / 8;
            buffer<u8> w_down_padded = buffer<u8>(w_ptr, (PADDED_D - D) * HIDDEN_SIZE * 5 / 8);
            w_down_padded.memset(0);
            w_ptr += (PADDED_D - D) * HIDDEN_SIZE * 5 / 8;
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
    this->lm_head->load_weights(q4nx);
    this->clear_context();
}

void gemma_text_npu::Impl::_set_rope_rms_weights(int idx){

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
        bool is_sliding = ((i + 1) % SLIDING_WINDOW_PATTERN != 0);
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

void gemma_text_npu::Impl::set_context_length(int L){
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

buffer<bf16> gemma_text_npu::Impl::forward(int ids){
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
    this->_rms_norm(this->x_lm_head_exposed, this->x, this->final_rms_weight);
    // utils::print_matrix(this->x, D);
    this->lm_head->execute();
    this->set_context_length(this->current_context_length + 1);
    buffer<bf16> logits = this->lm_head->wait();
    return logits;
}
buffer<bf16> gemma_text_npu::Impl::prefill(std::vector<int>& ids, void* payload){
#ifdef MVPREFILL
    header_print("warning", "Prefilling with Decoder, Slow!");
    return this->_prefill_with_mv(ids);
#else
    return this->_prefill_with_mm(ids);
#endif
}

buffer<bf16> gemma_text_npu::Impl::_prefill_with_mv(std::vector<int>& ids){
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
        this->x.sync_from_device();
        this->set_context_length(this->current_context_length + 1);
    }
    buffer<bf16> logits = get_logits(this->x);

    return logits;
}

buffer<bf16> gemma_text_npu::Impl::_prefill_with_mm(std::vector<int>& ids){
#ifndef MVPREFILL
    const int L_MIN = 256;
    const int L_CHUNK_SIZE = this->mha_sequence_gen->get_chunk_size();
    uint32_t PADDED_D = (D + 255) / 256 * 256;
    int L_in = ids.size();
    int L_begin = this->current_context_length;
    int L_end = L_begin + L_in;

    int L_begin_chunked = (L_begin / L_CHUNK_SIZE) * L_CHUNK_SIZE;
    int L_offset = L_begin - L_begin_chunked;
    int L_end_chunked = L_begin_chunked + (L_in + L_offset + L_MIN - 1) / L_MIN * L_MIN;
    int L_padded = L_end_chunked - L_begin_chunked;

    int L_effective = L_end - L_begin;
    int sliding_l_begin = (L_begin_chunked - SLIDING_WINDOW) > 0 ? (L_begin_chunked - SLIDING_WINDOW) : 0;
    int sliding_l_length = (L_end_chunked - sliding_l_begin);
    //allocate temporary buffers
    // The scratch window the sliding layers attend over is rebuilt from the rings on every call, but what an earlier call left in the
    // rows outside what it writes is not harmless to the kernel: start every prefill from zeros, so a prefill continued live and one
    // that follows a restored state see the same scratch.
    this->_zero_temp_cache();
    this->gemm->generate_seq(this->q_proj.seq(), L_padded, D, DQ, 0);
    this->gemm->generate_seq(this->k_proj.seq(), L_padded, D, DK, D * DQ);
    this->gemm->generate_seq(this->v_proj.seq(), L_padded, D, DV, D * (DQ + DK));
    this->gemm->generate_seq(this->o_proj.seq(), L_padded, DQ, PADDED_D, 0);
    this->gemm->generate_seq(this->up_gate_proj.seq(), L_padded, D, 2 * INTERMEDIATE_SIZE, 0);
    this->gemm->generate_seq(this->down_proj.seq(), L_padded, INTERMEDIATE_SIZE, PADDED_D, 0);
    buffer<bf16> buffer_q = this->q_proj.create_bo_buffer<bf16>((size_t)L_padded * DQ);
    buffer<bf16> buffer_k = this->k_proj.create_bo_buffer<bf16>((size_t)L_padded * DK);
    buffer<bf16> buffer_v = this->v_proj.create_bo_buffer<bf16>((size_t)L_padded * DV);
    buffer<bf16> buffer_upGate = this->up_gate_proj.create_bo_buffer<bf16>((size_t)L_padded * 2 * INTERMEDIATE_SIZE); //shared for o, down
    buffer<bf16> buffer_hid = this->down_proj.create_bo_buffer<bf16>((size_t)L_padded * INTERMEDIATE_SIZE);
    buffer<bf16> buffer_attn_out = this->mha_engine.create_bo_buffer<bf16>((size_t)L_padded * DQ);
    buffer<bf16> buffer_x = this->q_proj.create_bo_buffer<bf16>((size_t)L_padded * D);
    buffer<bf16> buffer_o = this->o_proj.create_bo_buffer<bf16>((size_t)L_padded * PADDED_D);
    buffer<bf16> residual = buffer<bf16>((size_t)L_padded * D); //shared for o, down
    buffer<bf16> buffer_upGate_cpu = buffer<bf16>((size_t)L_padded * 2 * INTERMEDIATE_SIZE); //shared for o, down
    buffer<bf16> buffer_hid_cpu = buffer<bf16>((size_t)L_padded * INTERMEDIATE_SIZE);
    tensor_2d<bf16> tensor_residual(residual, D, L_offset);
    tensor_2d<bf16> tensor_x(buffer_x, D, L_offset);
    tensor_2d<bf16> tensor_attn_out(buffer_o, PADDED_D, L_offset);

    tensor_2d<bf16> tensor_q(buffer_q, DQ, L_offset);
    tensor_2d<bf16> tensor_k(buffer_k, DK, L_offset);
    buffer_q.memset((bf16)0);
    buffer_k.memset((bf16)0);
    buffer_v.memset((bf16)0);
    buffer_upGate.memset((bf16)0);
    buffer_hid.memset((bf16)0);
    buffer_attn_out.memset((bf16)0);
    buffer_x.memset((bf16)0);
    buffer_o.memset((bf16)0);
    residual.memset((bf16)0);

    // embedding
    for (int i = 0; i < L_effective; i++){
        this->embedding->forward(ids[i], tensor_residual[i]);
    }

    this->mha_sequence_gen->generate_mha_sequence(this->mha_engine.seq(), L_begin_chunked, L_end_chunked, MAX_L, false, SLIDING_WINDOW);
    this->mha_sequence_gen->generate_mha_sequence(this->sliding_mha_engine.seq(), L_begin_chunked, L_end_chunked, MAX_L, true, SLIDING_WINDOW);
    for (int layer_idx = 0; layer_idx < num_hidden_layers; layer_idx++){
        buffer<bf16> rms_w_real(this->rms_weights[layer_idx].data(), D * 4);
        tensor_2d<bf16> rms_w(rms_w_real, D);
        tensor_2d<bf16> rope_rms_w(this->rope_rms_weights[layer_idx], DH);
        if (!this->npu->is_preemption_enabled()){
            this->dequant_all.reset();
            this->weights_p0.sync_to_device();
            this->weights_p1.sync_to_device();
            this->weights_p2.sync_to_device();
            this->weights_p3.sync_to_device();
            auto dequant_0_run = this->dequant_0.create_run(this->weights_p0, this->proj_weights[layer_idx]);
            auto dequant_1_run = this->dequant_1.create_run(this->weights_p1, this->proj_weights[layer_idx]);
            auto dequant_2_run = this->dequant_2.create_run(this->weights_p2, this->proj_weights[layer_idx]);
            auto dequant_3_run = this->dequant_3.create_run(this->weights_p3, this->proj_weights[layer_idx]);
            this->dequant_all.add(dequant_0_run);
            this->dequant_all.add(dequant_1_run);
            this->dequant_all.add(dequant_2_run);
            this->dequant_all.add(dequant_3_run);
            this->dequant_all.execute();
            for (int i = 0; i < L_effective; i++){
                _rms_norm(tensor_x[i], tensor_residual[i], rms_w[0]);
            }
            buffer_x.sync_to_device();
            flm::npu::wait_for_completion(this->dequant_all, "dequant_all");
        }
        else{
            this->weights_p0.sync_to_device();
            this->weights_p1.sync_to_device();
            this->weights_p2.sync_to_device();
            this->weights_p3.sync_to_device();
            auto dequant_0_run = this->dequant_0.create_run(this->weights_p0, this->proj_weights[layer_idx]);
            auto dequant_1_run = this->dequant_1.create_run(this->weights_p1, this->proj_weights[layer_idx]);
            auto dequant_2_run = this->dequant_2.create_run(this->weights_p2, this->proj_weights[layer_idx]);
            auto dequant_3_run = this->dequant_3.create_run(this->weights_p3, this->proj_weights[layer_idx]);
            dequant_0_run.start();
            flm::npu::wait_for_completion(dequant_0_run, "dequant_0_run");
            dequant_1_run.start();
            flm::npu::wait_for_completion(dequant_1_run, "dequant_1_run");
            dequant_2_run.start();
            flm::npu::wait_for_completion(dequant_2_run, "dequant_2_run");
            dequant_3_run.start();
            for (int i = 0; i < L_effective; i++){
                _rms_norm(tensor_x[i], tensor_residual[i], rms_w[0]);
            }
            buffer_x.sync_to_device();
            flm::npu::wait_for_completion(dequant_3_run, "dequant_3_run");
        }

        // step 3: qkv projection
        buffer_q.sync_to_device();
        buffer_k.sync_to_device();
        // this->weights_p0.sync_to_device();
        this->q_proj(buffer_q, buffer_x, this->weights_p0);
        auto run_k = this->k_proj.create_run(buffer_k, buffer_x, this->weights_p0);
        run_k.start();
        // Step 2: apply rope
        buffer_q.sync_from_device();

        for (int ll = 0; ll < L_effective; ll++){
            _rope_rms(tensor_q[ll], ll + L_begin, rope_rms_w[1], layer_idx);
        }
        buffer_q.sync_to_device();
        flm::npu::wait_for_completion(run_k, "run_k");
        buffer_k.sync_from_device();

        auto run_v = this->v_proj.create_run(buffer_v, buffer_x, this->weights_p0);
        buffer_v.sync_to_device();
        run_v.start();
        for (int ll = 0; ll < L_effective; ll++){
            _rope_rms(tensor_k[ll], ll + L_begin, rope_rms_w[2], layer_idx);
        }
        buffer_k.sync_to_device();
        flm::npu::wait_for_completion(run_v, "run_v");

        buffer_v.sync_from_device();

        buffer_q.sync_to_device();
        if (is_sliding_window(layer_idx)){
            this->sliding_temp_cache.sync_from_device();
            fill_kv_cache_sliding(this->sliding_temp_cache, buffer_k, buffer_v, L_offset, L_begin, L_end, L_effective, sliding_l_begin, L_end_chunked, layer_idx);
            this->sliding_temp_cache.sync_to_device();
            this->temp_dirty = true;
        }
        else{
            fill_kv_cache(buffer_k, buffer_v, L_offset, L_begin, L_effective, layer_idx);
        }
        this->kv_caches[layer_idx].sync_to_device();
        // Step 5: apply mha
        buffer_attn_out.sync_to_device();
        if (is_sliding_window(layer_idx)){
            this->sliding_mha_engine(buffer_attn_out, buffer_q, this->sliding_temp_cache);
        }
        else{
            this->mha_engine(buffer_attn_out, buffer_q, this->kv_caches[layer_idx]);
        }
        buffer_attn_out.sync_from_device();
        buffer_attn_out.sync_to_device();
        buffer_o.sync_to_device();
        // this->weights_p1.sync_to_device();
        this->o_proj(buffer_o, buffer_attn_out, this->weights_p1);
        buffer_o.sync_from_device();

        for (int i = 0; i < L_effective; i++){
            // Step 7: past_attention_norm
            _rms_norm(tensor_attn_out[i], tensor_attn_out[i], rms_w[1]);
            // Step 8: residual add
            _residual_add(tensor_residual[i], tensor_residual[i], tensor_attn_out[i]);
        }
        buffer_o.sync_to_device();
        // Step 9: rms norm
        for (int i = 0; i < L_effective; i++){
            _rms_norm(tensor_x[i], tensor_residual[i], rms_w[2]);
        }
        buffer_x.sync_to_device();

        // Step 10: up gate projection
        buffer_upGate.sync_to_device();
        this->up_gate_proj(buffer_upGate, buffer_x, this->weights_p2);
        buffer_upGate.sync_from_device();
        memcpy(buffer_upGate_cpu.data() + L_offset * 2 * INTERMEDIATE_SIZE, buffer_upGate.data() + L_offset * 2 * INTERMEDIATE_SIZE, (size_t)L_effective * 2 * INTERMEDIATE_SIZE * sizeof(bf16));

        // Step 11: glu
        _gelu(buffer_hid_cpu, buffer_upGate_cpu, L_offset, L_effective);
        buffer_upGate.sync_to_device();
        buffer_hid.sync_to_device();
        memcpy(buffer_hid.data() + L_offset * INTERMEDIATE_SIZE, buffer_hid_cpu.data() + L_offset * INTERMEDIATE_SIZE, (size_t)L_effective * INTERMEDIATE_SIZE * sizeof(bf16));

        buffer_hid.sync_to_device();
        this->weights_p3.sync_from_device(); // This is critical??
        this->weights_p3.sync_to_device();
        buffer_o.sync_to_device();

        // Step 12: down projection
        this->down_proj(buffer_o, buffer_hid, this->weights_p3);
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
    buffer<bf16> logits = get_logits(predict);
    this->set_context_length(this->current_context_length + L_effective);
    return logits;
#else
    return this->_prefill_with_mv(ids);
#endif
}

buffer<bf16> gemma_text_npu::Impl::get_k_cache(int layer_idx, int idx){
    buffer<bf16> k_cache(DK);
    if (is_sliding_window(layer_idx)){
        if (idx < this->current_context_length - SLIDING_WINDOW){
            header_print("warning", "Out of sliding window range!");
            k_cache.memset((bf16)0.0f);
        }
        this->kv_caches[layer_idx].sync_from_device();
        bf16* k_cache_ptr = this->kv_caches[layer_idx].data();
        uint32_t read_ptr = idx % SLIDING_WINDOW;
        uint32_t offset = read_ptr * DK;
        memcpy(static_cast<void*>(k_cache.data()), static_cast<void*>(k_cache_ptr + offset), DK * sizeof(bf16));
    }
    else{
        this->kv_caches[layer_idx].sync_from_device();
        bf16* k_cache_ptr = this->kv_caches[layer_idx].data();
        uint32_t offset = idx * DK;
        memcpy(static_cast<void*>(k_cache.data()), static_cast<void*>(k_cache_ptr + offset), DK * sizeof(bf16));
    }
    return k_cache;
}

buffer<bf16> gemma_text_npu::Impl::get_v_cache(int layer_idx, int idx){
    buffer<bf16> v_cache(DV);
    if (is_sliding_window(layer_idx)){
        if (idx < this->current_context_length - SLIDING_WINDOW){
            header_print("warning", "Out of sliding window range!");
            v_cache.memset((bf16)0.0f);
        }
        this->kv_caches[layer_idx].sync_from_device();
        bf16* v_cache_ptr = this->kv_caches[layer_idx].data();
        uint32_t read_ptr = idx % SLIDING_WINDOW;
        uint32_t offset = read_ptr * DV;
        uint32_t v_offset = SLIDING_WINDOW * DV;
        memcpy(static_cast<void*>(v_cache.data()), static_cast<void*>(v_cache_ptr + offset + v_offset), DV * sizeof(bf16));
    }
    else{
        this->kv_caches[layer_idx].sync_from_device();
        bf16* v_cache_ptr = this->kv_caches[layer_idx].data();
        uint32_t offset = idx * DH;
        memcpy(static_cast<void*>(v_cache.data()), static_cast<void*>(v_cache_ptr + offset + V_offset), DV * sizeof(bf16));
    }
    return v_cache;
}

buffer<bf16> gemma_text_npu::Impl::get_logits(buffer<bf16>& x){
    this->x_lm_head_exposed.sync_from_device();
    this->_rms_norm(this->x_lm_head_exposed, x, this->final_rms_weight);
    this->lm_head->execute();
    return this->lm_head->wait();
}

void gemma_text_npu::Impl::fill_kv_cache(buffer<bf16>& buffer_k, buffer<bf16>& buffer_v, int L_offset, int L_begin, int L_effective, int layer_idx){
    tensor_2d<bf16> k_cache(buffer_k, DK, L_offset);
    tensor_2d<bf16> v_cache(buffer_v, DV, L_offset);

    buffer<bf16> kv_cache = this->kv_caches[layer_idx];
    kv_cache.sync_from_device();

    bf16* k_ptr = kv_cache.data() + this->K_offset + L_begin * DK;
    bf16* v_ptr = kv_cache.data() + this->V_offset + L_begin * DV;

    for (int l = 0; l < L_effective; l++){
        memcpy(k_ptr, k_cache[l].data(), DK * sizeof(bf16));
        memcpy(v_ptr, v_cache[l].data(), DV * sizeof(bf16));

        k_ptr += DK;
        v_ptr += DV;
    }
}

///@brief move new kv to sliding window kv cache
void gemma_text_npu::Impl::fill_kv_cache_sliding(buffer<bf16>& local_kv_cache, buffer<bf16>& buffer_k, buffer<bf16>& buffer_v, int L_offset, int L_begin, int L_end, int L_effective, int sliding_l_begin, int sliding_l_end, int layer_idx){
    int in_memory = (L_begin - SLIDING_WINDOW) > 0 ? L_begin - SLIDING_WINDOW : 0;
    int l_idx = in_memory % SLIDING_WINDOW;
    int local_offset = in_memory - sliding_l_begin;

    tensor_2d<bf16> k_cache(buffer_k, DK, L_offset);
    tensor_2d<bf16> v_cache(buffer_v, DV, L_offset);
    int local_length = sliding_l_end - sliding_l_begin;

    this->kv_caches[layer_idx].sync_from_device();
    local_kv_cache.sync_from_device();
    bf16* k_ptr = local_kv_cache.data() + this->K_offset + in_memory * DK;
    bf16* v_ptr = local_kv_cache.data() + this->V_offset + in_memory * DV;

    buffer<bf16> system_k = buffer<bf16>(this->kv_caches[layer_idx].data() + this->sliding_K_offset, (size_t)SLIDING_WINDOW * DK);
    buffer<bf16> system_v = buffer<bf16>(this->kv_caches[layer_idx].data() + this->sliding_V_offset, (size_t)SLIDING_WINDOW * DV);

    tensor_2d<bf16> tensor_system_k(system_k, DK, 0);
    tensor_2d<bf16> tensor_system_v(system_v, DV, 0);
    // | sliding_l_begin -> in_memory | in_memory -> L_begin | L_begin -> L_end | L_end -> sliding_l_end |
    // | Part 1                       | Part 2               | Part 3           | Part 4                 |

    // part 1: padded, no actual data exists

    // part 2: copy from system kv to local kv
    for (int i = 0; i < (L_begin - in_memory); i++){
        int absolute_idx = in_memory + i;
        int ring_idx = linear2ring(absolute_idx);

        memcpy(k_ptr, tensor_system_k[ring_idx].data(), DK * sizeof(bf16));
        memcpy(v_ptr, tensor_system_v[ring_idx].data(), DV * sizeof(bf16));

        k_ptr += DK;
        v_ptr += DV;
    }
    // part 3: copy new kv to local kv and system kv
    for (int i = 0; i < (L_end - L_begin); i++){
        int absolute_idx = L_begin + i;
        int ring_idx = linear2ring(absolute_idx);

        memcpy(k_ptr, k_cache[i].data(), DK * sizeof(bf16));
        memcpy(v_ptr, v_cache[i].data(), DV * sizeof(bf16));

        memcpy(tensor_system_k[ring_idx].data(), k_cache[i].data(), DK * sizeof(bf16));
        memcpy(tensor_system_v[ring_idx].data(), v_cache[i].data(), DV * sizeof(bf16));

        k_ptr += DK;
        v_ptr += DV;
    }
}
void gemma_text_npu::Impl::_rms_norm(buffer<bf16>& y, buffer<bf16>& x, buffer<bf16>& rms_weight){
    int n = rms_weight.size();
    const int simd_width = 8;

    __m256 sum_xx_vec = _mm256_setzero_ps();

    // Step 1: Compute sum of squares using AVX
    int i = 0;
    for (; i <= n - simd_width; i += simd_width) {
        __m128i bf16_vals = _mm_loadu_si128((__m128i*)&x[i]);

        // Convert BF16 to FP32
        __m256 fp32_vals = bf16o_fp32(bf16_vals);
        sum_xx_vec = _mm256_fmadd_ps(fp32_vals, fp32_vals, sum_xx_vec);  // sum_xx += x[i] * x[i];
    }

    // Horizontal sum of the AVX register
    f32 sum_xx_arr[simd_width];
    _mm256_storeu_ps(sum_xx_arr, sum_xx_vec);
    f32 sum_xx = sum_xx_arr[0] + sum_xx_arr[1] + sum_xx_arr[2] + sum_xx_arr[3] +
                      sum_xx_arr[4] + sum_xx_arr[5] + sum_xx_arr[6] + sum_xx_arr[7];

    sum_xx /= (f32)n;
    f32 rms_x = sqrtf(sum_xx + 1e-6f);

    // Step 2: Compute element-wise operation y[i] = w[i] * x[i] / rms_x
    __m256 rms_x_vec = _mm256_set1_ps(rms_x);
    for (i = 0; i <= n - simd_width; i += simd_width) {
        __m128i bf16_vals = _mm_loadu_si128((__m128i*)&x[i]);
        __m256 fp32_vals = bf16o_fp32(bf16_vals);
        __m128i bf16_vals_w = _mm_loadu_si128((__m128i*)&rms_weight[i]);
        __m256 fp32_vals_w = bf16o_fp32(bf16_vals_w);
        __m256 y_vec = _mm256_div_ps(_mm256_mul_ps(fp32_vals_w, fp32_vals), rms_x_vec);
        __m128i result_bf16 = f32o_bf16(y_vec);
        _mm_storeu_si128((__m128i*)&y[i], result_bf16);
    }
}

void gemma_text_npu::Impl::_rope_rms(buffer<bf16>& x, int idx, buffer<bf16>& rms_weight, int layer_idx){
    int heads = x.size() / DH;
    buffer<f32> cos_buf(DH / 2);
    buffer<f32> sin_buf(DH / 2);
    for (uint32_t i = 0; i < DH / 2; i++){
        // use avx2 to calculate cos and sin
        if (is_sliding_window(layer_idx)){
            cos_buf[i] = std::cos(this->inv_freq_local[i] * idx);
            sin_buf[i] = std::sin(this->inv_freq_local[i] * idx);
        }
        else{
            cos_buf[i] = std::cos(this->inv_freq[i] * idx);
            sin_buf[i] = std::sin(this->inv_freq[i] * idx);
        }
    }
    for (int h = 0; h < heads; h++){
        buffer<bf16> x_head(x.data() + h * DH, DH);
        _rms_norm(x_head, x_head, rms_weight);
        buffer<bf16> x_left(x_head.data(), DH / 2);
        buffer<bf16> x_right(x_head.data() + DH / 2, DH / 2);
        int i = 0, simd = 8;
        // vectorized main loop
        for (; i + simd <= DH / 2; i += simd) {
            // load 8 BF16 → float32
            __m128i left_bf16  = _mm_loadu_si128((__m128i*)(x_left.data()  + i));
            __m128i right_bf16 = _mm_loadu_si128((__m128i*)(x_right.data() + i));
            __m256  L = bf16o_fp32(left_bf16);
            __m256  R = bf16o_fp32(right_bf16);

            // load sin/cos
            __m256 C = _mm256_loadu_ps(cos_buf.data() + i);
            __m256 S = _mm256_loadu_ps(sin_buf.data() + i);

            // new_left  = L*C - R*S
            __m256 newL = _mm256_sub_ps(_mm256_mul_ps(L, C),
                                        _mm256_mul_ps(R, S));
            // new_right = L*S + R*C
            __m256 newR = _mm256_add_ps(_mm256_mul_ps(L, S),
                                        _mm256_mul_ps(R, C));
            // store back
            __m128i newL_bf16 = f32o_bf16(newL);
            __m128i newR_bf16 = f32o_bf16(newR);
            _mm_storeu_si128((__m128i*)(x_left.data()  + i), newL_bf16);
            _mm_storeu_si128((__m128i*)(x_right.data() + i), newR_bf16);
        }
    }
}

void gemma_text_npu::Impl::_residual_add(buffer<bf16>& dest, buffer<bf16>& src, buffer<bf16>& residual){
    int n = src.size();
    const int simd_width = 8;

    for (int i = 0; i < n; i += simd_width){
        __m128i bf16_vals = _mm_loadu_si128((__m128i*)&src[i]);
        __m128i bf16_vals_r = _mm_loadu_si128((__m128i*)&residual[i]);
        __m256 fp32_vals = bf16o_fp32(bf16_vals);
        __m256 fp32_vals_r = bf16o_fp32(bf16_vals_r);
        __m256 result_vec = _mm256_add_ps(fp32_vals, fp32_vals_r);
        __m128i result_bf16 = f32o_bf16(result_vec);
        _mm_storeu_si128((__m128i*)&dest[i], result_bf16);
    }
}

__m256 gemma_text_npu::Impl::_mm256_exp_ps(__m256 x){
    // clamp x to avoid overflow
    const __m256 max_val = _mm256_set1_ps(88.3762626647949f);
    const __m256 min_val = _mm256_set1_ps(-88.3762626647949f);
    x = _mm256_min_ps(x, max_val);
    x = _mm256_max_ps(x, min_val);

    // express exp(x) = 2^(x / ln2)
    const __m256 ln2_inv = _mm256_set1_ps(1.44269504088896341f);  // 1/ln(2)
    __m256 fx = _mm256_mul_ps(x, ln2_inv);
    // round to nearest int
    fx = _mm256_add_ps(fx, _mm256_set1_ps(0.5f));
    __m256i emm0 = _mm256_cvttps_epi32(fx);
    fx = _mm256_cvtepi32_ps(emm0);

    // rebuild x' = x - fx*ln2
    const __m256 ln2_hi = _mm256_set1_ps(0.693359375f);
    const __m256 ln2_lo = _mm256_set1_ps(-2.12194440e-4f);
    __m256 tmp = _mm256_mul_ps(fx, ln2_hi);
    __m256 z   = _mm256_mul_ps(fx, ln2_lo);
    x = _mm256_sub_ps(x, tmp);
    x = _mm256_sub_ps(x, z);

    // polynomial approximation on reduced x
    const __m256 c0 = _mm256_set1_ps(1.9875691500E-4f);
    const __m256 c1 = _mm256_set1_ps(1.3981999507E-3f);
    const __m256 c2 = _mm256_set1_ps(8.3334519073E-3f);
    const __m256 c3 = _mm256_set1_ps(4.1665795894E-2f);
    const __m256 c4 = _mm256_set1_ps(1.6666665459E-1f);
    const __m256 c5 = _mm256_set1_ps(5.0000001201E-1f);

    __m256 y = c0;
    y = _mm256_fmadd_ps(y, x, c1);
    y = _mm256_fmadd_ps(y, x, c2);
    y = _mm256_fmadd_ps(y, x, c3);
    y = _mm256_fmadd_ps(y, x, c4);
    y = _mm256_fmadd_ps(y, x, c5);
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.0f));

    // build 2^n from the integer part
    emm0 = _mm256_add_epi32(emm0, _mm256_set1_epi32(0x7F));
    emm0 = _mm256_slli_epi32(emm0, 23);
    __m256 pow2n = _mm256_castsi256_ps(emm0);

    // combine
    return _mm256_mul_ps(y, pow2n);
}

void gemma_text_npu::Impl::_gelu(buffer<bf16>& dest, buffer<bf16>& up_gate, int L_offset, int L_effective){
    const int simd_width = 8;
    const int GLU_SLICE = 256;
    const __m256 max_arg = _mm256_set1_ps(10.0f);
    const __m256 min_arg = _mm256_set1_ps(-10.0f);
    const int n = L_effective * 2 * INTERMEDIATE_SIZE;
    const int chunks = n / GLU_SLICE;
    bf16 * in_ptr = up_gate.data() + L_offset * 2 * INTERMEDIATE_SIZE;
    bf16 * out_ptr = dest.data() + L_offset * INTERMEDIATE_SIZE;
    for (int i = 0; i < chunks; i++){
        bf16* up_ptr = in_ptr;
        bf16* gate_ptr = up_ptr + GLU_SLICE / 2;

        for (int j = 0; j < GLU_SLICE / 2; j += simd_width){
            __m128i up_vec = _mm_loadu_si128((__m128i*)&up_ptr[j]);
            __m128i gate_vec = _mm_loadu_si128((__m128i*)&gate_ptr[j]);
            __m256 up_vec_fp32 = bf16o_fp32(up_vec);
            __m256 gate_vec_fp32 = bf16o_fp32(gate_vec);
            // ---- GELU(gate) with tanh approximation: 0.5 * gate * (1 + tanh(sqrt(2/pi) * (gate + 0.044715 * gate^3)))  ----
            __m256 half = _mm256_set1_ps(0.5f);
            __m256 sqrt_2_pi = _mm256_set1_ps(0.7978845608f); // sqrt(2/pi)
            __m256 coeff = _mm256_set1_ps(0.044715f);

            // Compute gate^3
            __m256 gate_squared = _mm256_mul_ps(gate_vec_fp32, gate_vec_fp32);
            __m256 gate_cubed = _mm256_mul_ps(gate_squared, gate_vec_fp32);

            // Compute gate + 0.044715 * gate^3
            __m256 inner_term = _mm256_add_ps(gate_vec_fp32, _mm256_mul_ps(coeff, gate_cubed));

            // Compute sqrt(2/pi) * (gate + 0.044715 * gate^3)
            __m256 scaled_term = _mm256_mul_ps(sqrt_2_pi, inner_term);

            // Compute tanh using proper approximation: tanh(x) ≈ (exp(x) - exp(-x)) / (exp(x) + exp(-x))
            __m256 scaled_term_clamped = _mm256_max_ps(min_arg, _mm256_min_ps(max_arg, scaled_term));
            __m256 exp_pos = _mm256_exp_ps(scaled_term_clamped);
            __m256 exp_neg = _mm256_exp_ps(_mm256_sub_ps(_mm256_setzero_ps(), scaled_term));
            __m256 numerator = _mm256_sub_ps(exp_pos, exp_neg);
            __m256 denominator = _mm256_add_ps(exp_pos, exp_neg);
            __m256 tanh_approx = _mm256_div_ps(numerator, denominator);

            // Compute 1 + tanh(...)
            __m256 one_plus_tanh = _mm256_add_ps(_mm256_set1_ps(1.0f), tanh_approx);

            // Compute 0.5 * gate * (1 + tanh(...))
            __m256 gelu = _mm256_mul_ps(half, _mm256_mul_ps(gate_vec_fp32, one_plus_tanh));

            __m256 result_vec = _mm256_mul_ps(gelu, up_vec_fp32);
            __m128i result_bf16 = f32o_bf16(result_vec);
            _mm_storeu_si128((__m128i*)&out_ptr[j], result_bf16);
        }
        in_ptr += GLU_SLICE;
        out_ptr += GLU_SLICE / 2;
    }
}

void gemma_text_npu::Impl::clear_context(){
    this->set_context_length(0);
    for (int i = 0; i < this->num_hidden_layers; i++){
        this->kv_caches[i].memset((bf16)0);
        this->kv_caches[i].sync_to_device();
    }
    this->sliding_temp_cache.memset((bf16)0);
    this->sliding_temp_cache.sync_to_device();
    this->temp_dirty = false;
    this->dirty_rows = 0;
}

void gemma_text_npu::Impl::_zero_temp_cache(){
    if (!this->temp_dirty) return;
    this->sliding_temp_cache.memset((bf16)0);
    this->sliding_temp_cache.sync_to_device();
    this->temp_dirty = false;
}

kv_row_layout gemma_text_npu::Impl::kv_layout() const{
    kv_row_layout l;
    l.DK = DK; l.DV = DV; l.striped = false;
    l.v_offset = V_offset;
    return l;
}

std::unique_ptr<hybrid_engine_state> gemma_text_npu::Impl::new_state(){
    size_t ring_bytes = 0, global_layers = 0;
    for (size_t i = 0; i < kv_caches.size(); i++){
        if (attn_mask[i]) global_layers++;
        else ring_bytes += kv_caches[i].size() * sizeof(bf16);
    }
    return kv_rows_new_state(global_layers, kv_layout(), ring_bytes);
}

int gemma_text_npu::Impl::read_state(hybrid_engine_state& state, int kv_from){
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

int gemma_text_npu::Impl::write_state_kv(const hybrid_engine_state& state){
    return kv_rows_write(state, this->kv_caches, kv_layout(), this->MAX_L, &attn_mask);
}

int gemma_text_npu::Impl::write_state(const hybrid_engine_state& state){
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

int gemma_text_npu::Impl::get_current_context_length(){
    return this->current_context_length;
}

void gemma_text_npu::Impl::_allocate_checkpoint_buffers(){
    this->kv_checkpoint.clear();
    this->kv_checkpoint.resize(num_hidden_layers);
    for (int i = 0; i < num_hidden_layers; i++){
        if (is_sliding_window(i)){
            this->kv_checkpoint[i] = layer.create_bo_buffer<bf16>(this->kv_caches[i].size());
        }
    }
}

int gemma_text_npu::Impl::checkpoint(){
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

int gemma_text_npu::Impl::restore(){
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

void gemma_text_npu::Impl::update_max_length(uint32_t MAX_L){
    if (MAX_L <= this->MAX_L){
        header_print("FLM", "New length is shorter than the current length, no need to update!");
        return; // no need to update
	}
    this->MAX_L = MAX_L;
    size_t kv_cache_size = (size_t)MAX_L * DK * 2;   // DK == num_key_value_heads * head_dim
    buffer<bf16> kv_cache_temp(this->kv_caches[0].size());
    size_t original_k_offset = this->K_offset;
    size_t original_v_offset = this->V_offset;
    this->sequence->set_max_length(MAX_L);
    this->K_offset = this->sequence->get_k_offset();
    this->V_offset = this->sequence->get_v_offset();
    for (uint32_t i = 0; i < num_hidden_layers; i++){
        if (is_sliding_window(i)){
            continue;
        }
        this->kv_caches[i] = layer.create_bo_buffer<bf16>(kv_cache_size);
        this->kv_caches[i].memset((bf16)0);
        this->kv_caches[i].sync_to_device();
    }
    this->sliding_temp_cache = this->mha_engine.create_bo_buffer<bf16>(kv_cache_size);
    this->sliding_temp_cache.memset((bf16)0);
    this->sliding_temp_cache.sync_to_device();
    this->temp_dirty = false;
    this->dirty_rows = 0; // everything global was zeroed
    this->set_context_length(0); // regenerate sequence
    is_checkpoint_valid = false; // invalidate checkpoint since the underlying buffer has changed
}

///@brief destructor of gemma_npu
gemma_text_npu::Impl::~Impl() = default;
/*
gemma_text_npu::Impl::~Impl(){
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
gemma_text_npu::gemma_text_npu(LM_Config config, npu_xclbin_manager *npu_instance, int MAX_L){
    this->_impl = new Impl(config, npu_instance, MAX_L);
}

buffer<bf16> gemma_text_npu::forward(int ids){
    return this->_impl->forward(ids);
}

buffer<bf16> gemma_text_npu::prefill(std::vector<int>& ids, void* payload){
    try {
        buffer<bf16> result = this->_impl->prefill(ids, payload);
        return result;
    }
    catch (const std::runtime_error& e) {
        header_print("FLM", e.what());
        throw;
    }
}

void gemma_text_npu::set_context_length(int L){
    this->_impl->set_context_length(L);
}

void gemma_text_npu::load_weights(Q4NX& q4nx){
    this->_impl->load_weights(q4nx);
}

void gemma_text_npu::update_max_length(uint32_t MAX_L){
    this->_impl->update_max_length(MAX_L);
}

void gemma_text_npu::clear_context(){
    this->_impl->clear_context();
}

buffer<bf16> gemma_text_npu::get_k_cache(int layer_idx, int idx){
    return this->_impl->get_k_cache(layer_idx, idx);
}

buffer<bf16> gemma_text_npu::get_v_cache(int layer_idx, int idx){
    return this->_impl->get_v_cache(layer_idx, idx);
}

int gemma_text_npu::get_current_context_length(){
    return this->_impl->get_current_context_length();
}

int gemma_text_npu::checkpoint(){
    return this->_impl->checkpoint();
}

int gemma_text_npu::restore(){
    return this->_impl->restore();
}

std::unique_ptr<hybrid_engine_state> gemma_text_npu::new_state(){
    return this->_impl->new_state();
}

int gemma_text_npu::read_state(hybrid_engine_state& state, int kv_from){
    return this->_impl->read_state(state, kv_from);
}

int gemma_text_npu::write_state(const hybrid_engine_state& state){
    return this->_impl->write_state(state);
}

int gemma_text_npu::write_state_kv(const hybrid_engine_state& state){
    return this->_impl->write_state_kv(state);
}

extern "C" const flm_state_api* gemma_text_npu_state_api(void* engine){
    return hybrid_state_api<gemma_text_npu>(engine);
}

gemma_text_npu::~gemma_text_npu(){
    delete this->_impl;
}
