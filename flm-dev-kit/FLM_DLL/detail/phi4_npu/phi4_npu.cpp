// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#include "phi4_npu_detail.hpp"

phi4_npu::Impl::Impl(LM_Config config, npu_xclbin_manager *npu_instance, int MAX_L)
    : config(config), npu(npu_instance){
    current_context_length = 0;
    is_checkpoint_valid = false;

    this->desc.build(config);

    this->MAX_L = std::max((uint32_t)MAX_L, this->desc.min_max_length);
    // make MAX_L a power of 2
    float log2_MAX_L = std::log2(this->MAX_L);
    if (log2_MAX_L != std::floor(log2_MAX_L)){
        this->MAX_L = 1 << ((int)std::floor(log2_MAX_L) + 1);
    }
    MAX_L = this->MAX_L; // synchronize MAX_L

    // the rope factors are picked once, from the context this model is built for
    this->desc.select_rope(this->MAX_L);

    this->sequence = std::make_unique<phi4_npu_sequence>(this->desc, config, MAX_L);

    layer_app_manager = npu->register_xclbin(utils::path_join(config.exec_path, "xclbins", config.model_name, "layer.xclbin"));
    this->layer = layer_app_manager->create_app();
    this->set_layer_rtp = layer_app_manager->create_app();
    this->lm_head = layer_app_manager->create_app();
    if (!this->npu->is_preemption_enabled()){
        this->layers_run = layer_app_manager->create_runlist();
    }

    this->pctx = std::make_unique<phi4_prefill_context>(npu, &this->desc, config, this->MAX_L);

    // allocate all buffers
    rms_weights.resize(desc.num_hidden_layers);
    proj_weights.resize(desc.num_hidden_layers);
    kv_caches.resize(desc.num_hidden_layers);

    size_t kv_cache_size = desc.get_kv_cache_size(MAX_L);

    for (size_t i = 0; i < desc.num_hidden_layers; i++){
        proj_weights[i] = layer.create_bo_buffer<u8>(desc.total_proj_weights_byte_size);
        rms_weights[i] = layer.create_bo_buffer<bf16>(desc.rms_weights_byte_size / sizeof(bf16));
        kv_caches[i] = layer.create_bo_buffer<bf16>(kv_cache_size);
        kv_caches[i].memset((bf16)0.0f);
        this->kv_caches[i].sync_to_device();
    }

    rope_weights = layer.create_bo_buffer<bf16>(desc.DH);
    x = layer.create_bo_buffer<bf16>(desc.D);
    this->logits = layer.create_bo_buffer<bf16>(desc.vocab_size_padded);
    this->logits_valid = buffer<bf16>(this->logits.data(), desc.vocab_size);
    this->lm_head_weights = layer.create_bo_buffer<u8>(desc.get_lm_head_w_size());
    this->final_rms_weight = layer.create_bo_buffer<bf16>(desc.D);

    this->K03_offset = sequence->get_k03_offset();
    this->K47_offset = sequence->get_k47_offset();
    this->V03_offset = sequence->get_v03_offset();
    this->V47_offset = sequence->get_v47_offset();
    this->pctx->set_kv_layout(this->K03_offset, this->K47_offset, this->V03_offset, this->V47_offset);

    this->set_context_length(0);
    this->embedding = std::make_unique<Embedding>(desc.vocab_size, desc.D);

    this->sequence->gen_lm_head_seq(this->lm_head.seq());
    this->lm_head_run = this->lm_head.create_run(this->logits, this->lm_head_weights, this->x, this->final_rms_weight);
}

void phi4_npu::Impl::load_weights(Q4NX& q4nx){
    this->embedding->init_weights(&q4nx, "model.embed_tokens");
    this->desc.load_head_weights(q4nx, this->lm_head_weights, this->final_rms_weight);
    for (uint32_t layer_idx = 0; layer_idx < desc.num_hidden_layers; layer_idx++){
        this->desc.load_layer_weights(layer_idx, q4nx, this->proj_weights[layer_idx], this->rms_weights[layer_idx]);
    }
    for (uint32_t i = 0; i < desc.DH; i++){
        this->rope_weights[i] = (bf16)1.0f;
    }
    this->rope_weights.sync_to_device();
    this->clear_context();
}

void phi4_npu::Impl::_set_rope_weights(int idx){
    this->desc.set_rope_weights(this->rope_weights, idx);
}

void phi4_npu::Impl::set_context_length(int L){
    this->current_context_length = L;
    this->dirty_rows = std::max(this->dirty_rows, static_cast<size_t>(L));

    this->sequence->gen_layer_rtp_seq(this->set_layer_rtp.seq(), L + 1);
    this->sequence->gen_layer_seq(this->layer.seq(), L + 1);

    DO_VERBOSE(2, auto now = time_utils::now();)
    this->set_layer_rtp.update_ctrl_seq();
    this->layer.update_ctrl_seq();
    DO_VERBOSE(2, auto end = time_utils::now();)
    LOG_VERBOSE(2, "Time taken to update ctrl seq: " << time_utils::duration_us(now, end).first << " us");

    if (!this->npu->is_preemption_enabled()){
        this->layers_run.reset();
        // the per-token RTPs are the same for every layer, so they are written
        // once at the head of the runlist instead of once per layer.
        this->layers_run.add(this->set_layer_rtp.create_run());

        for (uint32_t i = 0; i < desc.num_hidden_layers; i++){
            this->layers_run.add(this->layer.create_run(this->x, this->proj_weights[i], this->rms_weights[i], this->rope_weights, this->kv_caches[i]));
        }
    }

    this->_set_rope_weights(L);
}

buffer<bf16> phi4_npu::Impl::forward(int ids){
    this->embedding->forward(ids, this->x);
    this->x.sync_to_device();

    if (!this->npu->is_preemption_enabled()){
        this->layers_run.execute();
        flm::npu::wait_for_completion(this->layers_run, "layers_run");
    }
    else{
        this->set_layer_rtp();
        for (uint32_t i = 0; i < desc.num_hidden_layers; i++){
            this->layer(this->x, this->proj_weights[i], this->rms_weights[i], this->rope_weights, this->kv_caches[i]);
        }
    }

    this->lm_head_run.start();
    this->set_context_length(this->current_context_length + 1);
    flm::npu::wait_for_completion(this->lm_head_run, "lm_head_run");
    this->logits.sync_from_device();
    return this->logits_valid;
}

buffer<bf16> phi4_npu::Impl::prefill(std::vector<int>& ids, void* payload){
#ifdef MVPREFILL
    header_print("warning", "Prefilling with Decoder, Slow!");
    return this->_prefill_with_mv(ids);
#else
    return this->_prefill_with_mm(ids);
#endif
}

buffer<bf16> phi4_npu::Impl::_prefill_with_mv(std::vector<int>& ids){
    for (int i = 0; i < ids.size(); i++){
        this->embedding->forward(ids[i], this->x);
        this->x.sync_to_device();

        this->set_layer_rtp();
        for (uint32_t i = 0; i < desc.num_hidden_layers; i++){
            this->layer(this->x, this->proj_weights[i], this->rms_weights[i], this->rope_weights, this->kv_caches[i]);
        }
        this->set_context_length(this->current_context_length + 1);
    }
    buffer<bf16> logits = get_logits(this->x);
    return logits;
}

buffer<bf16> phi4_npu::Impl::_prefill_with_mm(std::vector<int>& ids){
    int L_effective = ids.size();
    phi4_prefill_shape shape = this->pctx->setup(L_effective, this->current_context_length);

    for (int i = 0; i < L_effective; i++){
        this->embedding->forward(ids[i], this->pctx->residual_row(i));
    }

    for (uint32_t layer_idx = 0; layer_idx < desc.num_hidden_layers; layer_idx++){
        this->pctx->forward(layer_idx, shape, this->proj_weights[layer_idx],
            this->rms_weights[layer_idx], this->kv_caches[layer_idx]);
    }

    buffer<bf16> predict = this->pctx->residual_row(L_effective - 1);
    buffer<bf16> logits_result = get_logits(predict);

    this->set_context_length(this->current_context_length + L_effective);
    return logits_result;
}

buffer<bf16> phi4_npu::Impl::get_k_cache(int layer_idx, int idx){
    this->kv_caches[layer_idx].sync_from_device();
    buffer<bf16> k_cache(desc.DK);
    bf16* k_cache_ptr = this->kv_caches[layer_idx].data();
    uint32_t offset = idx * desc.DH * 4;
    memcpy(static_cast<void*>(k_cache.data()), static_cast<void*>(k_cache_ptr + offset), desc.DK / 2 * sizeof(bf16));
    memcpy(static_cast<void*>(k_cache.data() + desc.DK / 2), static_cast<void*>(k_cache_ptr + offset + K47_offset), desc.DK / 2 * sizeof(bf16));

    return k_cache;
}

buffer<bf16> phi4_npu::Impl::get_v_cache(int layer_idx, int idx){
    this->kv_caches[layer_idx].sync_from_device();
    buffer<bf16> v_cache(desc.DV);
    bf16* v_cache_ptr = this->kv_caches[layer_idx].data();
    uint32_t offset = idx * desc.DH * 4;
    memcpy(static_cast<void*>(v_cache.data()), static_cast<void*>(v_cache_ptr + offset + V03_offset), desc.DV / 2 * sizeof(bf16));
    memcpy(static_cast<void*>(v_cache.data() + desc.DV / 2), static_cast<void*>(v_cache_ptr + offset + V47_offset), desc.DV / 2 * sizeof(bf16));
    return v_cache;
}

buffer<bf16> phi4_npu::Impl::get_logits(buffer<bf16>& predict){
    memcpy(this->x.data(), predict.data(), desc.D * sizeof(bf16));
    this->x.sync_to_device();
    this->lm_head_run.start();
    flm::npu::wait_for_completion(this->lm_head_run, "lm_head_run");
    this->logits.sync_from_device();
    return this->logits_valid;
}

void phi4_npu::Impl::clear_context(){
    // saved states hold their own copies of the rows; the live ones must not leak into the next conversation
    kv_rows_zero(this->kv_caches, kv_layout(), 0, this->dirty_rows);
    this->dirty_rows = 0;
    this->set_context_length(0);
}

int phi4_npu::Impl::get_current_context_length(){
    return this->current_context_length;
}

void phi4_npu::Impl::_allocate_checkpoint_buffers(){}

int phi4_npu::Impl::checkpoint(){
    header_print_r("FLM", "Creating checkpoint at context length " + std::to_string(this->current_context_length));
    is_checkpoint_valid = true;
    checkpoint_context_length = this->current_context_length;
    return checkpoint_context_length;
}

int phi4_npu::Impl::restore(){
    if (!is_checkpoint_valid) {
        header_print("FLM", "No valid checkpoint found!");
        return -1;
    }
    header_print_r("FLM", "Restoring checkpoint at context length " + std::to_string(checkpoint_context_length));
    this->set_context_length(checkpoint_context_length);
    return checkpoint_context_length;
}

kv_row_layout phi4_npu::Impl::kv_layout() const {
    kv_row_layout l;
    l.DK = desc.DK; l.DV = desc.DV; l.striped = true;
    l.k47 = K47_offset; l.v03 = V03_offset; l.v47 = V47_offset;
    return l;
}

std::unique_ptr<hybrid_engine_state> phi4_npu::Impl::new_state(){
    return kv_rows_new_state(desc.num_hidden_layers, kv_layout());
}

int phi4_npu::Impl::read_state(hybrid_engine_state& state, int kv_from){
    header_print_r("FLM", "Reading state at context length " + std::to_string(this->current_context_length) + ", K/V rows from " + std::to_string(kv_from));
    return kv_rows_read(state, this->kv_caches, kv_layout(), this->current_context_length, kv_from);
}

int phi4_npu::Impl::write_state_kv(const hybrid_engine_state& state){
    return kv_rows_write(state, this->kv_caches, kv_layout(), this->MAX_L);
}

int phi4_npu::Impl::write_state(const hybrid_engine_state& state){
    const int length = kv_rows_write(state, this->kv_caches, kv_layout(), this->MAX_L);
    if (length < 0){
        header_print_r("FLM", "No valid state to write");
        return -1;
    }
    header_print_r("FLM", "Writing state at context length " + std::to_string(length));
    // rows past the length hold what the requests since the state was read left there: put back the zeros of a fresh engine
    kv_rows_zero(this->kv_caches, kv_layout(), static_cast<size_t>(length), this->dirty_rows);
    this->dirty_rows = static_cast<size_t>(length);
    this->set_context_length(length);
    return length;
}

void phi4_npu::Impl::update_max_length(uint32_t MAX_L){
    if (MAX_L <= this->MAX_L){
        header_print("FLM", "New length is shorter than the current length, no need to update!");
        return; // no need to update
	}
    this->MAX_L = MAX_L;
    size_t kv_cache_size = desc.get_kv_cache_size(MAX_L);
    buffer<bf16> kv_cache_temp(this->kv_caches[0].size());
    size_t original_k03_offset = this->K03_offset;
    size_t original_k47_offset = this->K47_offset;
    size_t original_v03_offset = this->V03_offset;
    size_t original_v47_offset = this->V47_offset;
    this->sequence->set_max_length(MAX_L);
    this->K03_offset = this->sequence->get_k03_offset();
    this->K47_offset = this->sequence->get_k47_offset();
    this->V03_offset = this->sequence->get_v03_offset();
    this->V47_offset = this->sequence->get_v47_offset();
    // both the cache layout and MAX_L are baked into the cached mha sequence
    this->pctx->set_max_length(MAX_L);
    this->pctx->set_kv_layout(this->K03_offset, this->K47_offset, this->V03_offset, this->V47_offset);
    for (uint32_t i = 0; i < desc.num_hidden_layers; i++){
        this->kv_caches[i].sync_from_device();
        memcpy(static_cast<void*>(kv_cache_temp.data()), static_cast<void*>(this->kv_caches[i].data()), this->kv_caches[i].size() * sizeof(bf16));
        this->kv_caches[i] = layer.create_bo_buffer<bf16>(kv_cache_size);
        this->kv_caches[i].sync_to_device();
        memcpy(static_cast<void*>(this->kv_caches[i].data() + this->K03_offset), static_cast<void*>(kv_cache_temp.data() + original_k03_offset), (kv_cache_temp.size() / 4) * sizeof(bf16));
        memcpy(static_cast<void*>(this->kv_caches[i].data() + this->V03_offset), static_cast<void*>(kv_cache_temp.data() + original_v03_offset), (kv_cache_temp.size() / 4) * sizeof(bf16));
        memcpy(static_cast<void*>(this->kv_caches[i].data() + this->K47_offset), static_cast<void*>(kv_cache_temp.data() + original_k47_offset), (kv_cache_temp.size() / 4) * sizeof(bf16));
        memcpy(static_cast<void*>(this->kv_caches[i].data() + this->V47_offset), static_cast<void*>(kv_cache_temp.data() + original_v47_offset), (kv_cache_temp.size() / 4) * sizeof(bf16));
    }
    this->set_context_length(this->current_context_length); // regenerate sequence
}
///@brief destructor of phi4_npu
phi4_npu::Impl::~Impl() = default;

// ===============================================
// Externals for phi4_npu
// ===============================================
phi4_npu::phi4_npu(LM_Config config, npu_xclbin_manager *npu_instance, int MAX_L){
    this->_impl = new Impl(config, npu_instance, MAX_L);
}

buffer<bf16> phi4_npu::forward(int ids){
    return this->_impl->forward(ids);
}

buffer<bf16> phi4_npu::prefill(std::vector<int>& ids, void* payload){
    try {
        buffer<bf16> result = this->_impl->prefill(ids, payload);
        return result;
    }
    catch (const std::runtime_error& e) {
        header_print("FLM", e.what());
        throw;
    }
}

void phi4_npu::set_context_length(int L){
    this->_impl->set_context_length(L);
}

void phi4_npu::load_weights(Q4NX& q4nx){
    this->_impl->load_weights(q4nx);
}

void phi4_npu::update_max_length(uint32_t MAX_L){
    this->_impl->update_max_length(MAX_L);
}

void phi4_npu::clear_context(){
    this->_impl->clear_context();
}

buffer<bf16> phi4_npu::get_k_cache(int layer_idx, int idx){
    return this->_impl->get_k_cache(layer_idx, idx);
}

buffer<bf16> phi4_npu::get_v_cache(int layer_idx, int idx){
    return this->_impl->get_v_cache(layer_idx, idx);
}

int phi4_npu::get_current_context_length(){
    return this->_impl->get_current_context_length();
}

int phi4_npu::checkpoint(){
    return this->_impl->checkpoint();
}

int phi4_npu::restore(){
    return this->_impl->restore();
}

std::unique_ptr<hybrid_engine_state> phi4_npu::new_state(){
    return this->_impl->new_state();
}

int phi4_npu::read_state(hybrid_engine_state& state, int kv_from){
    return this->_impl->read_state(state, kv_from);
}

int phi4_npu::write_state(const hybrid_engine_state& state){
    return this->_impl->write_state(state);
}

int phi4_npu::write_state_kv(const hybrid_engine_state& state){
    return this->_impl->write_state_kv(state);
}

phi4_npu::~phi4_npu(){
    delete this->_impl;
}

extern "C" const flm_state_api* phi4_npu_state_api(void* engine){
    return hybrid_state_api<phi4_npu>(engine);
}
