// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file qwen3_5_omni_thinker.cpp
/// \brief qwen3_5_omni thinker (text LM backbone) implementation.
/// \author FastFlowLM Team
/// \note Migrated from qwen3_5vl_npu (language part only).
#include <omp.h>
#include "qwen3_5_omni_thinker_detail.hpp"
#include "metrices.hpp"
#include "utils/error_measure.hpp"
#include "npu_sequences/GateDeltaNetPrefillRuntimeSequence.hpp"

qwen3_5_omni_thinker::Impl::Impl(LM_Config config, npu_xclbin_manager *npu_instance, int MAX_L)
    : config(config), npu(npu_instance){
    is_vlm = config.get<bool>("is_vlm", false);
    is_audio = config.get<bool>("is_audio", false);

    this->image_encoder = std::make_unique<qwen3_5_omni_image_embed>(config, npu_instance);
    this->audio_encoder = std::make_unique<qwen3_5_omni_audio_embed>(config, npu_instance);
    current_context_length = 0;
    // build the weight-shape definition (dimensions + per-layer weight descriptors)
    desc.build(config);

    vocab_size = desc.vocab_size;
    vocab_size_padded = desc.vocab_size_padded;

    this->MAX_L = std::max(MAX_L, 4096);
    // make MAX_L a power of 2
    float log2_MAX_L = std::log2(this->MAX_L);
    if (log2_MAX_L != std::floor(log2_MAX_L)){
        this->MAX_L = 1 << ((int)std::floor(log2_MAX_L) + 1);
    }
    MAX_L = this->MAX_L; // synchronize MAX_L

    this->sequence = std::make_unique<qwen3_5_omni_thinker_sequence>(this->desc, config, MAX_L);
    this->V_offset = this->sequence->get_V_offset();

    layer_app_manager = npu->register_xclbin(utils::path_join(config.exec_path, "xclbins", config.model_name, "layer.xclbin"));

    this->layer = layer_app_manager->create_app();
    this->linear_layer = layer_app_manager->create_app();
    this->first_linear_layer = layer_app_manager->create_app();
    this->set_layer_rtp = layer_app_manager->create_app();
    if (!this->npu->is_preemption_enabled()){
        this->layers_run = layer_app_manager->create_runlist();
    }

    lm_head_app_manager = npu->register_xclbin(utils::path_join(config.exec_path, "xclbins", config.model_name, "lm_head.xclbin"));
    this->lm_head = lm_head_app_manager->create_app();

    // allocate all buffers
    rms_weights.resize(desc.num_hidden_layers);
    proj_weights.resize(desc.num_hidden_layers);
    kv_caches.resize(desc.num_hidden_layers);
    rope_rms_weights.resize(desc.num_hidden_layers);
    alpha_proj_cpu.resize(desc.num_hidden_layers);
    beta_proj_cpu.resize(desc.num_hidden_layers);

    size_t attn_total_proj_weights_byte_size = desc.attn_total_proj_weights_byte_size;
    size_t linear_total_proj_weights_byte_size = desc.linear_total_proj_weights_byte_size;
    size_t attn_rms_weights_byte_size = desc.attn_rms_weights_byte_size;

    size_t linear_rms_weights_byte_size = desc.linear_rms_weights_byte_size;
    size_t linear_ssm_weights_byte_size = desc.linear_ssm_weights_byte_size;
    size_t kv_cache_size = desc.get_kv_cache_size(this->MAX_L);

    size_t linear_kv_cache_byte_size = desc.get_linear_cache_size();
    size_t conv_w_byte_size = desc.get_conv_w_size();

    for (size_t i = 0; i < desc.num_hidden_layers; i++){
        if (is_attn_layer(i)){
            this->proj_weights[i] = this->layer.create_bo_buffer<u8>(attn_total_proj_weights_byte_size);
            this->rms_weights[i] = this->layer.create_bo_buffer<bf16>(attn_rms_weights_byte_size / sizeof(bf16));
            this->rope_rms_weights[i] = this->layer.create_bo_buffer<bf16>(desc.RoPE_DIM + desc.DH * 2);
            this->kv_caches[i] = this->layer.create_bo_buffer<bf16>(kv_cache_size);
            this->kv_caches[i].memset((bf16)0);
            this->kv_caches[i].sync_to_device();
        }
        else{
            this->proj_weights[i] = this->linear_layer.create_bo_buffer<u8>(linear_total_proj_weights_byte_size);
            this->rms_weights[i] = this->linear_layer.create_bo_buffer<bf16>(linear_rms_weights_byte_size / sizeof(bf16));
            this->rope_rms_weights[i] = this->linear_layer.create_bo_buffer<bf16>(linear_ssm_weights_byte_size / sizeof(bf16));
            this->kv_caches[i] = this->linear_layer.create_bo_buffer<bf16>(linear_kv_cache_byte_size / sizeof(bf16));
            this->kv_caches[i].memset((bf16)0);
            this->kv_caches[i].sync_to_device();
            this->alpha_proj_cpu[i] = buffer<bf16>((size_t)desc.D * desc.linear_num_v_heads);
            this->beta_proj_cpu[i] = buffer<bf16>((size_t)desc.D * desc.linear_num_v_heads);
        }
    }

    is_checkpoint_valid = false; // checkpoint is not valid until we load kv cache to device after resizing buffers
    x = layer.create_bo_buffer<bf16>(desc.D * 2);
    lm_head_weights = layer.create_bo_buffer<u8>(desc.get_lm_head_w_size());
    logits = layer.create_bo_buffer<bf16>(desc.vocab_size_padded);
    logits_valid = buffer<bf16>(logits.data(), desc.vocab_size);

    this->clear_context();
    // the embedding table is created in load_weights, once we know whether the
    // checkpoint stores model.embed_tokens or only the tied q8_0 lm_head.

    pctx = std::make_unique<qwen3_5_omni_thinker_prefill_context>(npu, &desc, sequence.get(), config);

    lm_head = this->lm_head_app_manager->create_app();
    this->sequence->gen_lm_head_seq(lm_head.seq(), desc.vocab_size_padded, desc.D);
    this->lm_head_run = lm_head.create_run(this->logits, this->lm_head_weights, this->x);

    this->sequence->gen_layer_seq(this->linear_layer.seq(), 0, false, false);
    this->sequence->gen_layer_seq(this->first_linear_layer.seq(), 0, false, true);

    set_context_length(0, 0);
}

void qwen3_5_omni_thinker::Impl::load_weights(Q4NX& q4nx){
    // Newer converters tie the word embeddings and emit only the quantized
    // lm_head, so fall back to dequantizing the embedding out of it.
    if (q4nx.has_tensor("model.embed_tokens.weight")){
        this->embedding = std::make_unique<Embedding>(desc.vocab_size, desc.D);
        this->embedding->init_weights(&q4nx, "model.embed_tokens");
    }
    else{
        this->embedding_q8 = std::make_unique<embedding_q8_0>(desc.vocab_size, desc.D);
        this->embedding_q8->init_weights(q4nx, "lm_head.weight");
    }
    if(is_vlm){
        SafeTensors vision_weights(config.get<std::string>("vision_model_weight", ""));

        this->image_encoder->load_weights(vision_weights);
    }
    if(is_audio){
        SafeTensors audio_weights(config.get<std::string>("audio_model_weight", ""));
        this->audio_encoder->load_weights(audio_weights);
    }
    for (int i = 0; i < desc.num_hidden_layers; i++){
        if (desc.is_attn_layer(i)){
            desc.load_attn_weights(i, q4nx, this->proj_weights[i], this->rms_weights[i], this->rope_rms_weights[i]);
        }
        else{
            desc.load_linear_weights(i, q4nx, this->proj_weights[i], this->rms_weights[i], this->rope_rms_weights[i], this->alpha_proj_cpu[i], this->beta_proj_cpu[i]);
        }
    }

    {
        buffer<bf16> final_rms_norm_weight;
        q4nx.load_weights(final_rms_norm_weight, "model.norm.weight");
        memcpy(this->x.data() + desc.D, final_rms_norm_weight.data(), desc.D * sizeof(bf16));
        this->x.sync_to_device();
    }
    {
        buffer<u8> lm_head_w;
        q4nx.load_weights(lm_head_w, "lm_head.weight");

        desc.reorder_cpy(this->lm_head_weights.data(), lm_head_w, desc.D, flm_q80, 4);
        this->lm_head_weights.sync_to_device();
    }

    this->clear_context();
}

void qwen3_5_omni_thinker::Impl::_set_rope_rms_weights(int idx){
    int RoPE_DIM = desc.RoPE_DIM;
    buffer<bf16> cos_buf(RoPE_DIM / 2);
    buffer<bf16> sin_buf(RoPE_DIM / 2);

    for (int j = 0; j < RoPE_DIM / 2; j++){
        cos_buf[j] = (bf16)cos(desc.inv_freq[j] * idx);
        sin_buf[j] = (bf16)sin(desc.inv_freq[j] * idx);
    }

    for (int i = 0; i < desc.num_hidden_layers; i++){
        if (is_attn_layer(i)){
            bf16* w_rope_ptr = this->rope_rms_weights[i].data();
            memcpy(w_rope_ptr, cos_buf.data(), RoPE_DIM * sizeof(bf16) / 2);
            memcpy(w_rope_ptr + RoPE_DIM / 2, sin_buf.data(), RoPE_DIM * sizeof(bf16) / 2);
            this->rope_rms_weights[i].sync_to_device();
        }
    }
}

void qwen3_5_omni_thinker::Impl::set_context_length(int L, int new_general_token_idx){
    this->current_context_length = L;
    this->general_token_idx = new_general_token_idx;

    this->sequence->gen_set_context(this->set_layer_rtp.seq(), L + 1);
    this->sequence->gen_layer_seq(this->layer.seq(), L + 1, true);
    if (!this->npu->is_preemption_enabled()){
        this->layers_run.reset();

        this->layers_run.add(this->set_layer_rtp.create_run());
        for (uint32_t i = 0; i < desc.num_hidden_layers; i++){
            if (is_attn_layer(i)){
                this->layers_run.add(this->layer.create_run(this->x, this->proj_weights[i], this->rms_weights[i], this->rope_rms_weights[i], this->kv_caches[i]));
            }
            else {
                if (is_first_linear_layer(i)) {
                    this->layers_run.add(this->first_linear_layer.create_run(this->x, this->proj_weights[i], this->rms_weights[i], this->rope_rms_weights[i], this->kv_caches[i]));
                }
                else  {
                    this->layers_run.add(this->linear_layer.create_run(this->x, this->proj_weights[i], this->rms_weights[i], this->rope_rms_weights[i], this->kv_caches[i]));
                }
            }
        }
    }

    this->_set_rope_rms_weights(new_general_token_idx);
}

buffer<bf16> qwen3_5_omni_thinker::Impl::forward(int ids){
    this->embed_forward(ids, this->x);
    this->x.sync_to_device();

    if (!this->npu->is_preemption_enabled()){
        this->layers_run.execute();
        flm::npu::wait_for_completion(this->layers_run, "layers_run");
    }
    else{
        set_layer_rtp();
        for (uint32_t i = 0; i < desc.num_hidden_layers; i++){
            if (is_attn_layer(i)){
                this->layer(this->x, this->proj_weights[i], this->rms_weights[i], this->rope_rms_weights[i], this->kv_caches[i]);
            }
            else if (is_first_linear_layer(i)){
                    this->first_linear_layer(this->x, this->proj_weights[i], this->rms_weights[i], this->rope_rms_weights[i], this->kv_caches[i]);
            }
            else{
                this->linear_layer(this->x, this->proj_weights[i], this->rms_weights[i], this->rope_rms_weights[i], this->kv_caches[i]);
            }
        }
    }

    this->x.sync_to_device();
    this->lm_head_run.start();
    this->set_context_length(this->current_context_length + 1, this->general_token_idx + 1);
    flm::npu::wait_for_completion(this->lm_head_run, "lm_head_run");
    this->logits.sync_from_device();
    return this->logits_valid;
}

buffer<bf16> qwen3_5_omni_thinker::Impl::prefill(std::vector<int>& ids, void* payload){
#ifdef MVPREFILL
    header_print("warning", "Prefilling with Decoder, Slow!");
    return this->_prefill_with_mv(ids);
#else
    return this->_prefill_with_mm(ids, payload);
#endif
}

buffer<bf16> qwen3_5_omni_thinker::Impl::_prefill_with_mv(std::vector<int>& ids, void* payload){
#ifdef ENABLE_DEBUG
    header_print("info", "USING DEBUG!");
    SafeTensors aref(utils::path_join(config.model_path, "qwen3_5_omni_thinker.safetensors"));

    buffer<int> debug_ids;
    aref.load_weights(debug_ids, "input_ids");
    ids.resize(debug_ids.size());
    memcpy(ids.data(), debug_ids.data(), debug_ids.size() * sizeof(int));

#endif
    for (int i = 0; i < (int)ids.size(); i++){
        this->embed_forward(ids[i], this->x);
        this->x.sync_to_device();
        set_layer_rtp();
        for (uint32_t l = 0; l < desc.num_hidden_layers; l++){
#ifdef ENABLE_DEBUG
            buffer<bf16> ref;
            char nn[256];
            sprintf(nn, "layer%02d_layer_output", l);
            aref.load_weights(ref, std::string(nn));
            std::cout << nn << std::endl;
#endif
            if (is_attn_layer(l)){
                this->layer(this->x, this->proj_weights[l], this->rms_weights[l], this->rope_rms_weights[l], this->kv_caches[l]);
            }
            else if (is_first_linear_layer(l)) {
                this->first_linear_layer(this->x, this->proj_weights[l], this->rms_weights[l], this->rope_rms_weights[l], this->kv_caches[l]);
            }
            else {
                this->linear_layer(this->x, this->proj_weights[l], this->rms_weights[l], this->rope_rms_weights[l], this->kv_caches[l]);
            }

            this->x.sync_from_device();
#ifdef ENABLE_DEBUG
            print_error_metrics(
                x.data(), ref.data() + i * desc.D,
                1,
                1, desc.D,
                1, desc.D
            );
#endif
        }
        this->set_context_length(this->current_context_length + 1, this->general_token_idx + 1);
#ifdef ENABLE_DEBUG
        exit(0);
#endif
    }
    this->lm_head_run.start();
    flm::npu::wait_for_completion(this->lm_head_run, "lm_head_run");
    this->logits.sync_from_device();

    return logits_valid;
}

buffer<bf16> qwen3_5_omni_thinker::Impl::_prefill_with_mm(std::vector<int>& ids, void* payload){

#ifdef ENABLE_DEBUG
    header_print("info", "USING DEBUG!");
    SafeTensors aref(utils::path_join(config.model_path, "qwen3_5_omni_thinker.safetensors"));

    buffer<int> debug_ids;
    aref.load_weights(debug_ids, "input_ids");
    ids.resize(debug_ids.size());
    memcpy(ids.data(), debug_ids.data(), debug_ids.size() * sizeof(int));

#endif
    int L_in = ids.size();

    std::vector<bool> vision_mask(L_in);
    std::vector<int> input_idx = this->get_input_idx(vision_mask, ids, (qwen3_5_omni_image_payload_t*)payload);
    buffer<bf16> hidden_states((size_t)desc.D * L_in);
    tensor_2d<bf16> hidden_states_2d(hidden_states, desc.D);

    buffer<bf16> image_embedding;
    buffer<bf16> audio_embedding;
    if(payload != nullptr){
        // for some info, let us compare to the input
        qwen3_5_omni_payload_t* mulit_modal_payload_ptr = (qwen3_5_omni_payload_t*)(payload);
        if(is_vlm && mulit_modal_payload_ptr->image_payload.num_images > 0){
            //SafeTensors vision_ref(utils::path_join(config.model_path, "vision_data_debug.safetensors"));
            image_embedding = this->image_encoder->forward(mulit_modal_payload_ptr->image_payload);
        }
        if(is_audio && mulit_modal_payload_ptr->audio_payload.num_audios>0){
            //std::cout << "TODO: do audio payload prefill here" << std::endl;
            audio_embedding = this->audio_encoder->forward(mulit_modal_payload_ptr->audio_payload);
        }
    }
    bf16 * vision_token_ptr = (payload != nullptr)? image_embedding.data() : nullptr;
    bf16 * audio_token_ptr = (payload != nullptr)? audio_embedding.data() : nullptr;

    for (int l = 0; l < L_in; l++) {
        if (vision_mask[l] && ((payload != nullptr))){
            memcpy(hidden_states_2d[l].data(), vision_token_ptr, desc.D * sizeof(bf16));
            vision_token_ptr += desc.D;
        }
        else if (ids[l] == desc.audio_token_id && (payload != nullptr)){
            memcpy(hidden_states_2d[l].data(), audio_token_ptr, desc.D * sizeof(bf16));
            audio_token_ptr += desc.D;
        }
        else {
            this->embed_forward(ids[l], hidden_states_2d[l]);
        }
    }

#ifdef ENABLE_DEBUG
    buffer<bf16> embed_ref;
    aref.load_weights(embed_ref, "embeddings");
    print_error_metrics(
        hidden_states.data(), embed_ref.data(),
        1,
        L_in, desc.D,
        L_in, desc.D
    );
#endif

    time_utils::time_point start = time_utils::now();
    for (int layer_idx = 0; layer_idx < desc.num_hidden_layers; layer_idx++) {
        auto ret = pctx->forward(layer_idx, hidden_states, L_in, proj_weights[layer_idx], rms_weights[layer_idx], rope_rms_weights[layer_idx], kv_caches[layer_idx], input_idx, alpha_proj_cpu[layer_idx], beta_proj_cpu[layer_idx], this->current_context_length);
#ifdef ENABLE_DEBUG
        buffer<bf16> ref;
        char nn[256];
        sprintf(nn, "layer%02d_input_layernorm", layer_idx);
        aref.load_weights(ref, std::string(nn));
        std::cout << nn << std::endl;

        print_error_metrics(
            ret.data(), ref.data(),
            1,
            L_in, desc.D,
            L_in, desc.D
        );
    exit(0);
#endif
    }

    buffer<bf16> predict = hidden_states_2d[L_in - 1];

    this->lm_head_weights.sync_to_device();
    get_logits(predict);
    this->set_context_length(this->current_context_length + L_in, input_idx[L_in * 3 - 1] + 1);

    return logits_valid;
}

std::vector<int> qwen3_5_omni_thinker::Impl::get_input_idx(std::vector<bool>& vision_mask, std::vector<int>& tokens, qwen3_5_omni_image_payload_t* image_payload){
    int l = tokens.size();
    bool in_image = false;
    int image_counter = 0;

    // Optimization: Cached width and counters to avoid expensive division/modulo operations
    int current_w = 0;
    int cur_row = 0;
    int cur_col = 0;

    std::vector<int> input_idx(l * 3);
    for (int i = 0; i < tokens.size(); i++){
        vision_mask[i] = (tokens[i] == desc.image_token_id);
        if (vision_mask[i]){
            if (!in_image){
                in_image = true;
                // Update cached width for the new image block
                current_w = image_payload->image_grid_h_w[image_counter][0] / 2;
                cur_row = 0;
                cur_col = 0;
            }

            input_idx[i * 3 + 0] = general_token_idx;
            input_idx[i * 3 + 1] = general_token_idx + cur_row;
            input_idx[i * 3 + 2] = general_token_idx + cur_col;

            // Advance counters
            cur_col++;
            if (cur_col == current_w) {
                cur_col = 0;
                cur_row++;
            }
        }
        else {
            if (in_image){
                in_image = false;
                general_token_idx += (image_payload->image_grid_h_w[image_counter][0]  / 2);
                image_counter++;
            }
            input_idx[i * 3 + 0] = general_token_idx;
            input_idx[i * 3 + 1] = general_token_idx;
            input_idx[i * 3 + 2] = general_token_idx;
            general_token_idx += 1;
        }
    }

    return input_idx;
}

buffer<bf16> qwen3_5_omni_thinker::Impl::get_k_cache(int layer_idx, int idx){
    this->kv_caches[layer_idx].sync_from_device();
    buffer<bf16> k_cache(desc.DK);
    bf16* k_cache_ptr = this->kv_caches[layer_idx].data();
    uint32_t offset = idx * desc.DK;
    memcpy(static_cast<void*>(k_cache.data()), static_cast<void*>(k_cache_ptr + offset), desc.DK * sizeof(bf16));

    return k_cache;
}

buffer<bf16> qwen3_5_omni_thinker::Impl::get_v_cache(int layer_idx, int idx){
    this->kv_caches[layer_idx].sync_from_device();
    buffer<bf16> v_cache(desc.DV);
    bf16* v_cache_ptr = this->kv_caches[layer_idx].data();
    uint32_t offset = idx * desc.DV;
    memcpy(static_cast<void*>(v_cache.data()), static_cast<void*>(v_cache_ptr + offset), desc.DV * sizeof(bf16));
    return v_cache;
}

buffer<bf16> qwen3_5_omni_thinker::Impl::get_logits(buffer<bf16>& predict){
    memcpy(this->x.data(), predict.data(), desc.D * sizeof(bf16));
    this->x.sync_to_device();
    this->lm_head_run.start();
    flm::npu::wait_for_completion(this->lm_head_run, "lm_head_run");
    this->logits.sync_from_device();
    return this->logits_valid;
}

void qwen3_5_omni_thinker::Impl::_allocate_checkpoint_buffers() {
    this->cache_checkpoint.clear();
    this->cache_checkpoint.resize(desc.num_hidden_layers);
    is_checkpoint_valid = false;
    for (int i = 0; i < desc.num_hidden_layers; i++) {
        if (!is_attn_layer(i)) {
            this->cache_checkpoint[i] = buffer<bf16>(this->kv_caches[i].size());
        }
    }
}

int qwen3_5_omni_thinker::Impl::checkpoint() {
    if (!is_checkpoint_valid) {
        _allocate_checkpoint_buffers();
    } // lazy allocation of checkpoint buffers on the first checkpoint call, to avoid unnecessary memory usage if checkpoint/restore is never used
    header_print_r("FLM", "Creating checkpoint at context length " + std::to_string(this->current_context_length));
    for (int i = 0; i < desc.num_hidden_layers; i++) {
        if (!is_attn_layer(i)) {
            this->kv_caches[i].sync_from_device();
            memcpy(this->cache_checkpoint[i].data(), this->kv_caches[i].data(), this->kv_caches[i].size() * sizeof(bf16));
            this->kv_caches[i].sync_to_device();
        }
    }
    is_checkpoint_valid = true;
    checkpoint_context_length = this->current_context_length;
    checkpoint_general_token_idx = this->general_token_idx;
    return this->current_context_length;
}

int qwen3_5_omni_thinker::Impl::restore() {
    if (!is_checkpoint_valid) {
        header_print_r("FLM", "No valid checkpoint to restore");
        return -1;
    }
    header_print_r("FLM", "Restoring checkpoint at context length " + std::to_string(checkpoint_context_length));
    for (int i = 0; i < desc.num_hidden_layers; i++) {
        if (!is_attn_layer(i)) {
            this->kv_caches[i].sync_from_device();
            memcpy(this->kv_caches[i].data(), this->cache_checkpoint[i].data(), this->cache_checkpoint[i].size() * sizeof(bf16));
            this->kv_caches[i].sync_to_device();
        }
        else {
            this->kv_caches[i].sync_from_device();

            int offset = (size_t)current_context_length * (desc.DK);
            // first half is K, second half is V, and they are stored contiguously in kv_cache
            memset(this->kv_caches[i].data() + offset, 0, (this->kv_caches[i].size() / 2 - offset) * sizeof(bf16)); // zero out the part that is not restored for global layers, since we only restore the sliding part for global layers
            memset(this->kv_caches[i].data() + this->kv_caches[i].size() / 2 + offset, 0, (this->kv_caches[i].size() / 2 - offset) * sizeof(bf16)); // zero out the second half for swa
            this->kv_caches[i].sync_to_device();
        }
    }
    this->current_context_length = this->checkpoint_context_length;
    this->general_token_idx = this->checkpoint_general_token_idx;
    this->set_context_length(this->current_context_length, this->general_token_idx);
    return this->current_context_length;
}

void qwen3_5_omni_thinker::Impl::clear_context(){
    this->set_context_length(0, 0);
    for (int i = 0; i < desc.num_hidden_layers; i++){
        this->kv_caches[i].sync_from_device();
        memset(this->kv_caches[i].data(), 0, this->kv_caches[i].size() * sizeof(bf16));
        this->kv_caches[i].sync_to_device();
    }
}

int qwen3_5_omni_thinker::Impl::get_current_context_length(){
    return this->current_context_length;
}

void qwen3_5_omni_thinker::Impl::update_max_length(uint32_t MAX_L){
    if (MAX_L <= this->MAX_L){
        header_print("FLM", "New length is shorter than the current length, no need to update!");
        return; // no need to update
	}
    this->MAX_L = MAX_L;
    size_t kv_cache_size = (size_t)MAX_L * (desc.DK + desc.DV);
    buffer<bf16> kv_cache_temp(this->kv_caches[0].size());
    size_t original_V_offset = this->V_offset;
    this->sequence->set_max_length(MAX_L);
    this->V_offset = this->sequence->get_V_offset();
    for (uint32_t i = 0; i < desc.num_hidden_layers; i++){
        if (!is_attn_layer(i)){
            continue; // nothing to do with linear layer
        }
        this->kv_caches[i].sync_from_device();
        memcpy(static_cast<void*>(kv_cache_temp.data()), static_cast<void*>(this->kv_caches[i].data()), this->kv_caches[i].size() * sizeof(bf16));
        this->kv_caches[i] = layer.create_bo_buffer<bf16>(kv_cache_size);
        this->kv_caches[i].sync_to_device();
        memcpy(static_cast<void*>(this->kv_caches[i].data()), static_cast<void*>(kv_cache_temp.data()), (kv_cache_temp.size() / 2) * sizeof(bf16));
        memcpy(static_cast<void*>(this->kv_caches[i].data() + this->V_offset), static_cast<void*>(kv_cache_temp.data() + original_V_offset), (kv_cache_temp.size() / 2) * sizeof(bf16));
    }
    this->set_context_length(this->current_context_length, this->general_token_idx); // regenerate sequence
    is_checkpoint_valid = false; // invalidate checkpoint since the underlying kv cache has changed
}

///@brief destructor
qwen3_5_omni_thinker::Impl::~Impl() = default;

// ===============================================
// Externals for qwen3_5_omni_thinker
// ===============================================
qwen3_5_omni_thinker::qwen3_5_omni_thinker(LM_Config config, npu_xclbin_manager *npu_instance, int MAX_L){
    this->_impl = new Impl(config, npu_instance, MAX_L);
}

buffer<bf16> qwen3_5_omni_thinker::forward(int ids){
    return this->_impl->forward(ids);
}

buffer<bf16> qwen3_5_omni_thinker::prefill(std::vector<int>& ids, void* payload){
    try {
        buffer<bf16> result = this->_impl->prefill(ids, payload);
        return result;
    }
    catch (const std::runtime_error& e) {
        header_print("FLM", e.what());
        throw;
    }
}

void qwen3_5_omni_thinker::set_context_length(int L){
    std::cout << "Setting context length is not supported for qwen3_5_omni_thinker!" << std::endl;
}

void qwen3_5_omni_thinker::load_weights(Q4NX& q4nx){
    this->_impl->load_weights(q4nx);
}

void qwen3_5_omni_thinker::update_max_length(uint32_t MAX_L){
    this->_impl->update_max_length(MAX_L);
}

void qwen3_5_omni_thinker::clear_context(){
    this->_impl->clear_context();
}

buffer<bf16> qwen3_5_omni_thinker::get_k_cache(int layer_idx, int idx){
    return this->_impl->get_k_cache(layer_idx, idx);
}

buffer<bf16> qwen3_5_omni_thinker::get_v_cache(int layer_idx, int idx){
    return this->_impl->get_v_cache(layer_idx, idx);
}

int qwen3_5_omni_thinker::get_current_context_length(){
    return this->_impl->get_current_context_length();
}

int qwen3_5_omni_thinker::checkpoint(){
    return this->_impl->checkpoint();
}

int qwen3_5_omni_thinker::restore(){
    return this->_impl->restore();
}

qwen3_5_omni_thinker::~qwen3_5_omni_thinker(){
    delete this->_impl;
}
