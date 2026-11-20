// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#include "lfm2_npu_detail.hpp"

lfm2_npu::Impl::Impl(LM_Config config, npu_xclbin_manager *npu_instance, int MAX_L)
    : config(config), npu(npu_instance){
    current_context_length = 0;
    is_checkpoint_valid = false;

    this->desc.build(config);

    this->MAX_L = std::max((uint32_t)MAX_L, desc.min_max_length);
    // make MAX_L a power of 2
    float log2_MAX_L = std::log2(this->MAX_L);
    if (log2_MAX_L != std::floor(log2_MAX_L)){
        this->MAX_L = 1 << ((int)std::floor(log2_MAX_L) + 1);
    }
    MAX_L = this->MAX_L; // synchronize MAX_L

    this->sequence = std::make_unique<lfm2_npu_sequence>(this->desc, config, MAX_L);

    layer_app_manager = npu->register_xclbin(utils::path_join(config.exec_path, "xclbins", config.model_name, "layer.xclbin"));
    this->attn_layer = layer_app_manager->create_app();
    this->conv_layer = layer_app_manager->create_app();
    this->lm_head = layer_app_manager->create_app();

    if (!this->npu->is_preemption_enabled()){
        this->layers_run = layer_app_manager->create_runlist();
    }

    this->pctx = std::make_unique<lfm2_prefill_context>(npu, &this->desc, this->sequence.get(), config, this->MAX_L);

    // allocate all buffers
    const uint32_t D = desc.D;
    const uint32_t num_hidden_layers = desc.num_hidden_layers;

    rms_weights.resize(num_hidden_layers);
    proj_weights.resize(num_hidden_layers);
    kv_caches.resize(num_hidden_layers);
    rope_rms_weights.resize(num_hidden_layers);
    attn_mask.resize(num_hidden_layers);
    for (uint32_t i = 0; i < num_hidden_layers; i++) attn_mask[i] = is_attn_layer(i);

    size_t kv_cache_size = desc.get_kv_cache_size(MAX_L);

    lm_head_weights = lm_head.create_bo_buffer<u8>(desc.get_lm_head_w_size());
    logits = lm_head.create_bo_buffer<bf16>(desc.vocab_size);
    final_rms_weight = lm_head.create_bo_buffer<bf16>(D);

    for (uint32_t i = 0; i < num_hidden_layers; i++){
        proj_weights[i] = conv_layer.create_bo_buffer<u8>(desc.total_proj_weights_byte_size);
        rms_weights[i] = attn_layer.create_bo_buffer<bf16>(desc.rms_weights_byte_size / sizeof(bf16));
        if (is_attn_layer(i)){
            kv_caches[i] = attn_layer.create_bo_buffer<bf16>(kv_cache_size);
        }
        else {
            kv_caches[i] = conv_layer.create_bo_buffer<bf16>(desc.get_conv_cache_size());
        }
        rope_rms_weights[i] = conv_layer.create_bo_buffer<bf16>(desc.rope_rms_weights_byte_size / sizeof(bf16));
        kv_caches[i].memset((bf16)0.0f);
        this->kv_caches[i].sync_to_device();
    }

    x = attn_layer.create_bo_buffer<bf16>(D);

    this->K03_offset = sequence->get_k03_offset();
    this->K47_offset = sequence->get_k47_offset();
    this->V03_offset = sequence->get_v03_offset();
    this->V47_offset = sequence->get_v47_offset();
    this->pctx->set_kv_layout(this->K03_offset, this->K47_offset, this->V03_offset, this->V47_offset);

    this->set_context_length(0);
    this->embedding = std::make_unique<Embedding>(desc.vocab_size, D);

    this->sequence->gen_lm_head_seq(this->lm_head.seq());

    this->lm_head_run = this->lm_head.create_run(this->logits, this->lm_head_weights, this->x, this->final_rms_weight);
}

void lfm2_npu::Impl::load_weights(Q4NX& q4nx){
    this->embedding->init_weights(&q4nx, "model.token_embd");
    this->desc.load_head_weights(q4nx, this->lm_head_weights, this->final_rms_weight);

    for (uint32_t layer_idx = 0; layer_idx < desc.num_hidden_layers; layer_idx++){
        this->desc.load_layer_weights(layer_idx, q4nx,
            this->proj_weights[layer_idx],
            this->rms_weights[layer_idx],
            this->rope_rms_weights[layer_idx]);
    }

    this->clear_context();
}

void lfm2_npu::Impl::set_context_length(int L){
    this->current_context_length = L;
    this->dirty_rows = std::max(this->dirty_rows, static_cast<size_t>(L));

    this->sequence->gen_attn_layer_seq(this->attn_layer.seq(), L + 1);
    this->sequence->gen_conv_layer_seq(this->conv_layer.seq(), L + 1);

    this->attn_layer.update_ctrl_seq();
    this->conv_layer.update_ctrl_seq();

    if (!this->npu->is_preemption_enabled()){
        this->layers_run.reset();
        for (uint32_t i = 0; i < desc.num_hidden_layers; i++){
            if (is_attn_layer(i)){
                this->layers_run.add(this->attn_layer.create_run(this->x, this->proj_weights[i], this->rms_weights[i], this->rope_rms_weights[i], this->kv_caches[i]));
            }
            else{
                this->layers_run.add(this->conv_layer.create_run(this->x, this->proj_weights[i], this->rms_weights[i], this->rope_rms_weights[i], this->kv_caches[i]));
            }
        }
    }

    this->desc.set_rope_weights(L, this->rope_rms_weights);
}

buffer<bf16> lfm2_npu::Impl::forward(int ids){
    this->embedding->forward(ids, this->x);
    this->x.sync_to_device();

    if (!this->npu->is_preemption_enabled()){
        this->layers_run.execute();
        flm::npu::wait_for_completion(this->layers_run, "layers_run");
    }
    else{
        for (uint32_t i = 0; i < desc.num_hidden_layers; i++){
            if (is_attn_layer(i)){
                this->attn_layer(this->x, this->proj_weights[i], this->rms_weights[i], this->rope_rms_weights[i], this->kv_caches[i]);
            }
            else{
                this->conv_layer(this->x, this->proj_weights[i], this->rms_weights[i], this->rope_rms_weights[i], this->kv_caches[i]);
            }
        }
    }

    lm_head_run.start();
    this->set_context_length(this->current_context_length + 1);

    flm::npu::wait_for_completion(lm_head_run, "lm_head_run");
    this->logits.sync_from_device();
    return this->logits;
}

buffer<bf16> lfm2_npu::Impl::prefill(std::vector<int>& ids, void* payload){
#ifdef MVPREFILL
    header_print("warning", "Prefilling with Decoder, Slow!");
    return this->_prefill_with_mv(ids);
#else
    return this->_prefill_with_mm(ids);
#endif
}

buffer<bf16> lfm2_npu::Impl::_prefill_with_mv(std::vector<int>& ids){
    for (int i = 0; i < ids.size(); i++){
        this->embedding->forward(ids[i], this->x);
        this->x.sync_to_device();

        if (!this->npu->is_preemption_enabled()){
            this->layers_run.execute();
            flm::npu::wait_for_completion(this->layers_run, "layers_run");
        }
        else{
            for (uint32_t j = 0; j < desc.num_hidden_layers; j++){
                if (is_attn_layer(j)){
                    this->attn_layer(this->x, this->proj_weights[j], this->rms_weights[j], this->rope_rms_weights[j], this->kv_caches[j]);
                }
                else{
                    this->conv_layer(this->x, this->proj_weights[j], this->rms_weights[j], this->rope_rms_weights[j], this->kv_caches[j]);
                }
            }
        }
        this->set_context_length(this->current_context_length + 1);
    }
    buffer<bf16> logits = get_logits(this->x);

    return logits;
}

buffer<bf16> lfm2_npu::Impl::_prefill_with_mm(std::vector<int>& ids){
    int L_effective = ids.size();
    lfm2_prefill_shape shape = this->pctx->setup(L_effective, this->current_context_length);

    // embedding
    for (int i = 0; i < L_effective; i++){
        this->embedding->forward(ids[i], this->pctx->residual_row(i));
    }

    for (uint32_t layer_idx = 0; layer_idx < desc.num_hidden_layers; layer_idx++){
        this->pctx->forward(layer_idx, shape,
            this->proj_weights[layer_idx],
            this->rms_weights[layer_idx],
            this->rope_rms_weights[layer_idx],
            this->kv_caches[layer_idx]);
    }

    buffer<bf16> predict = this->pctx->residual_row(L_effective - 1);
    memcpy(this->x.data(), predict.data(), desc.D * sizeof(bf16));
    this->x.sync_to_device();
    this->final_rms_weight.sync_to_device();
    this->lm_head_weights.sync_to_device();
    buffer<bf16> logits = get_logits(this->x);
    this->set_context_length(this->current_context_length + L_effective);
    return logits;
}

buffer<bf16> lfm2_npu::Impl::get_k_cache(int layer_idx, int idx){
    if (!is_attn_layer(layer_idx)){
        header_print("warning", "Not an attn layer!");
        return buffer<bf16>(0);
    }
    const uint32_t DK = desc.DK;
    this->kv_caches[layer_idx].sync_from_device();
    buffer<bf16> k_cache(DK);
    bf16* k_cache_ptr = this->kv_caches[layer_idx].data();
    uint32_t offset = idx * desc.DH * 4;
    memcpy(static_cast<void*>(k_cache.data()), static_cast<void*>(k_cache_ptr + offset), DK / 2 * sizeof(bf16));
    memcpy(static_cast<void*>(k_cache.data() + DK / 2), static_cast<void*>(k_cache_ptr + offset + K47_offset), DK / 2 * sizeof(bf16));

    return k_cache;
}

buffer<bf16> lfm2_npu::Impl::get_v_cache(int layer_idx, int idx){
    if (!is_attn_layer(layer_idx)){
        header_print("warning", "Not an attn layer!");
        return buffer<bf16>(0);
    }
    const uint32_t DV = desc.DV;
    this->kv_caches[layer_idx].sync_from_device();
    buffer<bf16> v_cache(DV);
    bf16* v_cache_ptr = this->kv_caches[layer_idx].data();
    uint32_t offset = idx * desc.DH * 4;
    memcpy(static_cast<void*>(v_cache.data()), static_cast<void*>(v_cache_ptr + offset + V03_offset), DV / 2 * sizeof(bf16));
    memcpy(static_cast<void*>(v_cache.data() + DV / 2), static_cast<void*>(v_cache_ptr + offset + V47_offset), DV / 2 * sizeof(bf16));
    return v_cache;
}

buffer<bf16> lfm2_npu::Impl::get_logits(buffer<bf16>& x){
    this->lm_head(this->logits, this->lm_head_weights, this->x, this->final_rms_weight);
    logits.sync_from_device();
    return this->logits;
}

void lfm2_npu::Impl::clear_context(){
    for (uint32_t i = 0; i < desc.num_hidden_layers; i++){
        this->kv_caches[i].sync_from_device();
        memset(static_cast<void*>(this->kv_caches[i].data()), 0, this->kv_caches[i].size() * sizeof(bf16));
        this->kv_caches[i].sync_to_device();
    }
    this->dirty_rows = 0;
    this->set_context_length(0);
}

int lfm2_npu::Impl::get_current_context_length(){
    return this->current_context_length;
}

void lfm2_npu::Impl::_allocate_checkpoint_buffers(){
    this->kv_checkpoint.clear();
    this->kv_checkpoint.resize(desc.num_hidden_layers);
    for (uint32_t i = 0; i < desc.num_hidden_layers; i++){
        if (!is_attn_layer(i)){
            this->kv_checkpoint[i] = conv_layer.create_bo_buffer<bf16>(this->kv_caches[i].size());
        }
    }
}

int lfm2_npu::Impl::checkpoint(){
    header_print_r("FLM", "Creating checkpoint at context length " + std::to_string(this->current_context_length));
    if (!is_checkpoint_valid) {
        _allocate_checkpoint_buffers();
    } // lazy allocation of checkpoint buffers, only allocate when checkpoint is called for the first time
    for (uint32_t i = 0; i < desc.num_hidden_layers; i++){
        if (!is_attn_layer(i)){
            this->kv_caches[i].sync_from_device();
            memcpy(static_cast<void*>(this->kv_checkpoint[i].data()), static_cast<void*>(this->kv_caches[i].data()), this->kv_caches[i].size() * sizeof(bf16));
            this->kv_caches[i].sync_to_device();
        }
    }
    is_checkpoint_valid = true;
    checkpoint_context_length = this->current_context_length;
    return checkpoint_context_length;
}

int lfm2_npu::Impl::restore(){
    if (!is_checkpoint_valid) {
        header_print("FLM", "No valid checkpoint found!");
        return -1;
    }
    header_print_r("FLM", "Restoring checkpoint at context length " + std::to_string(checkpoint_context_length));
    for (uint32_t i = 0; i < desc.num_hidden_layers; i++){
        if (!is_attn_layer(i)){
            this->kv_caches[i].sync_from_device();
            memcpy(static_cast<void*>(this->kv_caches[i].data()), static_cast<void*>(this->kv_checkpoint[i].data()), this->kv_caches[i].size() * sizeof(bf16));
            this->kv_caches[i].sync_to_device();
        }
    }
    this->set_context_length(checkpoint_context_length); // regenerate sequence
    return checkpoint_context_length;
}

void lfm2_npu::Impl::update_max_length(uint32_t MAX_L){
    if (MAX_L <= this->MAX_L){
        header_print("FLM", "New length is shorter than the current length, no need to update!");
        return; // no need to update
	}
    this->MAX_L = MAX_L;
    const size_t kv_cache_size = desc.get_kv_cache_size(MAX_L);
    const size_t original_offsets[4] = {this->K03_offset, this->K47_offset, this->V03_offset, this->V47_offset};
    this->sequence->set_max_length(MAX_L);
    this->K03_offset = this->sequence->get_k03_offset();
    this->K47_offset = this->sequence->get_k47_offset();
    this->V03_offset = this->sequence->get_v03_offset();
    this->V47_offset = this->sequence->get_v47_offset();
    const size_t new_offsets[4] = {this->K03_offset, this->K47_offset, this->V03_offset, this->V47_offset};
    for (uint32_t i = 0; i < desc.num_hidden_layers; i++){
        if (!is_attn_layer(i)){
            continue; // a conv layer's history does not depend on MAX_L
        }
        // sizes come from this layer's own buffer (layer 0 is a conv layer with a much smaller one)
        this->kv_caches[i].sync_from_device();
        const size_t old_quarter = this->kv_caches[i].size() / 4;
        buffer<bf16> kv_cache_temp(this->kv_caches[i].size());
        memcpy(static_cast<void*>(kv_cache_temp.data()), static_cast<void*>(this->kv_caches[i].data()), this->kv_caches[i].size() * sizeof(bf16));
        this->kv_caches[i] = attn_layer.create_bo_buffer<bf16>(kv_cache_size);
        this->kv_caches[i].memset((bf16)0);
        for (int k = 0; k < 4; k++){
            memcpy(static_cast<void*>(this->kv_caches[i].data() + new_offsets[k]), static_cast<void*>(kv_cache_temp.data() + original_offsets[k]), old_quarter * sizeof(bf16));
        }
        this->kv_caches[i].sync_to_device(); // after the copy: the device must see the migrated rows
    }
    // both the cache layout and MAX_L are baked into the cached mha sequence
    this->pctx->set_max_length(MAX_L);
    this->pctx->set_kv_layout(this->K03_offset, this->K47_offset, this->V03_offset, this->V47_offset);
    this->set_context_length(this->current_context_length); // regenerate sequence
    this->is_checkpoint_valid = false; // invalidate checkpoint since the underlying buffer has changed
}

kv_row_layout lfm2_npu::Impl::kv_layout() const{
    kv_row_layout l;
    l.DK = desc.DK; l.DV = desc.DV; l.striped = true;
    l.k47 = K47_offset; l.v03 = V03_offset; l.v47 = V47_offset;
    return l;
}

std::unique_ptr<hybrid_engine_state> lfm2_npu::Impl::new_state(){
    size_t conv_layers = 0;
    for (uint32_t i = 0; i < desc.num_hidden_layers; i++) conv_layers += attn_mask[i] ? 0 : 1;
    return kv_rows_new_state(desc.num_hidden_layers - conv_layers, kv_layout(), conv_layers * desc.get_conv_cache_size() * sizeof(bf16));
}

int lfm2_npu::Impl::read_state(hybrid_engine_state& state, int kv_from){
    header_print_r("FLM", "Reading state at context length " + std::to_string(this->current_context_length) + ", K/V rows from " + std::to_string(kv_from));
    const int length = kv_rows_read(state, this->kv_caches, kv_layout(), this->current_context_length, kv_from, -1, &attn_mask);
    if (length < 0) return -1;
    for (uint32_t i = 0; i < desc.num_hidden_layers; i++){
        if (attn_mask[i]) continue;
        this->kv_caches[i].sync_from_device();
        state.state[i] = buffer<bf16>(this->kv_caches[i].size());
        memcpy(static_cast<void*>(state.state[i].data()), static_cast<void*>(this->kv_caches[i].data()), this->kv_caches[i].size() * sizeof(bf16));
    }
    return length;
}

int lfm2_npu::Impl::write_state_kv(const hybrid_engine_state& state){
    return kv_rows_write(state, this->kv_caches, kv_layout(), this->MAX_L, &attn_mask);
}

int lfm2_npu::Impl::write_state(const hybrid_engine_state& state){
    // check the conv blobs before anything is written
    bool ok = state.state.size() == this->kv_caches.size();
    for (uint32_t i = 0; ok && i < desc.num_hidden_layers; i++){
        if (!attn_mask[i] && state.state[i].size() != this->kv_caches[i].size()) ok = false;
    }
    const int length = ok ? kv_rows_write(state, this->kv_caches, kv_layout(), this->MAX_L, &attn_mask) : -1;
    if (length < 0){
        header_print_r("FLM", "No valid state to write");
        return -1;
    }
    header_print_r("FLM", "Writing state at context length " + std::to_string(length));
    for (uint32_t i = 0; i < desc.num_hidden_layers; i++){
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

///@brief destructor of lfm2_npu
lfm2_npu::Impl::~Impl() = default;

// ===============================================
// Externals for lfm2_npu
// ===============================================
lfm2_npu::lfm2_npu(LM_Config config, npu_xclbin_manager *npu_instance, int MAX_L){
    this->_impl = new Impl(config, npu_instance, MAX_L);
}

buffer<bf16> lfm2_npu::forward(int ids){
    return this->_impl->forward(ids);
}

buffer<bf16> lfm2_npu::prefill(std::vector<int>& ids, void* payload){
    try {
        buffer<bf16> result = this->_impl->prefill(ids, payload);
        return result;
    }
    catch (const std::runtime_error& e) {
        header_print("FLM", e.what());
        throw;
    }
}

void lfm2_npu::set_context_length(int L){
    this->_impl->set_context_length(L);
}

void lfm2_npu::load_weights(Q4NX& q4nx){
    this->_impl->load_weights(q4nx);
}

void lfm2_npu::update_max_length(uint32_t MAX_L){
    this->_impl->update_max_length(MAX_L);
}

void lfm2_npu::clear_context(){
    this->_impl->clear_context();
}

buffer<bf16> lfm2_npu::get_k_cache(int layer_idx, int idx){
    return this->_impl->get_k_cache(layer_idx, idx);
}

buffer<bf16> lfm2_npu::get_v_cache(int layer_idx, int idx){
    return this->_impl->get_v_cache(layer_idx, idx);
}

int lfm2_npu::get_current_context_length(){
    return this->_impl->get_current_context_length();
}

int lfm2_npu::checkpoint(){
    return this->_impl->checkpoint();
}

int lfm2_npu::restore(){
    return this->_impl->restore();
}

std::unique_ptr<hybrid_engine_state> lfm2_npu::new_state(){
    return this->_impl->new_state();
}

int lfm2_npu::read_state(hybrid_engine_state& state, int kv_from){
    return this->_impl->read_state(state, kv_from);
}

int lfm2_npu::write_state(const hybrid_engine_state& state){
    return this->_impl->write_state(state);
}

int lfm2_npu::write_state_kv(const hybrid_engine_state& state){
    return this->_impl->write_state_kv(state);
}

extern "C" const flm_state_api* lfm2_npu_state_api(void* engine){
    return hybrid_state_api<lfm2_npu>(engine);
}

lfm2_npu::~lfm2_npu(){
    delete this->_impl;
}
