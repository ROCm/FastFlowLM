// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include "whisper_npu_detail.hpp"
#include "whisper_prof.hpp"

whisper_npu::Impl::Impl(Whisper_Config config, npu_xclbin_manager *npu_instance, int MAX_L)
    : config(config), npu(npu_instance){
    current_context_length = 0;
    this->MAX_L = 448;
    this->encoder = std::make_unique<WhisperEncoder>(config, npu_instance);
    this->decoder = std::make_unique<WhisperDecoder>(config, npu_instance);
    this->lm_head = std::make_unique<WhisperLMHead>(config, npu_instance);

    this->final_layer_norm_weights = buffer<bf16>(WHISPER_D);
    this->final_layer_norm_biases = buffer<bf16>(WHISPER_D);
}

void whisper_npu::Impl::load_weights(Q4NX& q4nx){
    this->encoder->load_weights(q4nx);
    this->decoder->load_weights(q4nx);
    this->lm_head->load_weights(q4nx);

    q4nx.load_weights(this->final_layer_norm_weights, "model.decoder.layer_norm.weight");
    q4nx.load_weights(this->final_layer_norm_biases, "model.decoder.layer_norm.bias");

    this->clear_context();
}

void whisper_npu::Impl::set_context_length(int L){
    this->current_context_length = L;
}

buffer<bf16> whisper_npu::Impl::encode_audio(buffer<bf16>& mel_feature){
    // static int count = 0;
    buffer<bf16> kv_cache = this->encoder->forward(mel_feature);
    // write kv_cache to file
    this->decoder->load_cross_attn_kv_cache(kv_cache);
    PROF_LAP("kv_load");
    PROF_REPORT("encoder");
    return kv_cache;
}
buffer<bf16> whisper_npu::Impl::decode_audio(int ids){
    buffer<bf16> output = this->decoder->forward(ids);

    this->decoder->final_layernorm(this->lm_head->lm_head_x, this->final_layer_norm_weights, this->final_layer_norm_biases);
    this->lm_head->lm_head_x.sync_to_device();

    this->lm_head->execute();

    buffer<bf16> result = this->lm_head->wait();

    current_context_length += 1;
    return result;
}

buffer<bf16> whisper_npu::Impl::get_logits(buffer<bf16>& x){
    this->lm_head->execute();
    return this->lm_head->wait();
}

void whisper_npu::Impl::clear_context(){
    this->set_context_length(0);
    this->decoder->set_context_length(0);
}

int whisper_npu::Impl::get_current_context_length(){
    return this->current_context_length;
}

///@brief destructor of whisper_npu
whisper_npu::Impl::~Impl() = default;
/*
whisper_npu::Impl::~Impl(){
    for (int i = 0; i < config.get("num_hidden_layers"); i++){
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
// Externals for whisper_npu
// ===============================================
whisper_npu::whisper_npu(Whisper_Config config, npu_xclbin_manager *npu_instance, int MAX_L){
    this->_impl = new Impl(config, npu_instance, MAX_L);
}

buffer<bf16> whisper_npu::encode_audio(buffer<bf16>& mel_feature){
    return this->_impl->encode_audio(mel_feature);
}

buffer<bf16> whisper_npu::decode_audio(int ids){
    return this->_impl->decode_audio(ids);
}

void whisper_npu::load_weights(Q4NX& q4nx){
    this->_impl->load_weights(q4nx);
}

void whisper_npu::clear_context(){
    this->_impl->clear_context();
}

int whisper_npu::get_current_context_length(){
    return this->_impl->get_current_context_length();
}

whisper_npu::~whisper_npu(){
    delete this->_impl;
}
