// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#include "qwen2_npu_detail.hpp"

qwen2_npu::Impl::Impl(LM_Config config, npu_xclbin_manager *npu_instance, int MAX_L)
    : config(config), npu(npu_instance)
{
    current_context_length = 0;
    is_checkpoint_valid = false;

    this->desc.build(this->config);

    this->MAX_L = MAX_L;
    // make MAX_L a power of 2
    float log2_MAX_L = std::log2(this->MAX_L);
    if (log2_MAX_L != std::floor(log2_MAX_L)){
        this->MAX_L = 1 << ((int)std::floor(log2_MAX_L) + 1);
    }

    this->sequence = std::make_unique<qwen2_npu_sequence>(this->config, this->MAX_L);

    layer_app_manager = npu->register_xclbin(utils::path_join(config.exec_path, "xclbins", config.model_name, "layer.xclbin"));
    this->layer = layer_app_manager->create_app();
    if (!this->npu->is_preemption_enabled()){
        this->layers_run = layer_app_manager->create_runlist();
    }
    this->lm_head = layer_app_manager->create_app();

    // allocate all buffers
    rms_weights.resize(desc.num_hidden_layers);
    proj_weights.resize(desc.num_hidden_layers);
    kv_caches.resize(desc.num_hidden_layers);
    rope_rms_weights.resize(desc.num_hidden_layers);

    for (size_t i = 0; i < desc.num_hidden_layers; i++){
        proj_weights[i] = layer.create_bo_buffer<u8>(desc.total_proj_weights_byte_size);
        rms_weights[i] = layer.create_bo_buffer<bf16>(desc.rms_weights_byte_size / sizeof(bf16));
        kv_caches[i] = layer.create_bo_buffer<bf16>(desc.get_kv_cache_size(this->MAX_L));
        rope_rms_weights[i] = layer.create_bo_buffer<bf16>(desc.DH * 3);
        this->kv_caches[i].sync_to_device();
    }
    x = layer.create_bo_buffer<bf16>(desc.D);
    lm_head_weights = layer.create_bo_buffer<u8>(desc.get_lm_head_w_size());
    final_rms_weight = layer.create_bo_buffer<bf16>(desc.D);
    logits = layer.create_bo_buffer<bf16>(desc.vocab_size_padded);
    logits_valid = buffer<bf16>(logits.data(), desc.vocab_size);

    this->V_offset = sequence->get_V_Offset();

    this->set_context_length(0);
    this->embedding = std::make_unique<Embedding>(desc.vocab_size, desc.D);

    this->pctx = std::make_unique<qwen2_prefill_context>(npu, &this->desc, this->sequence.get(), this->config);
    this->pctx->set_kv_layout(this->V_offset);

    this->sequence->gen_lm_head_seq(this->lm_head.seq());
    this->lm_head_run = this->lm_head.create_run(this->logits, this->lm_head_weights, this->x, this->final_rms_weight);
}

void qwen2_npu::Impl::load_weights(Q4NX &q4nx)
{
    this->embedding->init_weights(&q4nx, "model.embed_tokens");
    this->desc.load_head_weights(q4nx, this->lm_head_weights, this->final_rms_weight);

    for (uint32_t layer_idx = 0; layer_idx < desc.num_hidden_layers; layer_idx++){
        this->desc.load_layer_weights(layer_idx, q4nx, this->proj_weights[layer_idx], this->rms_weights[layer_idx]);
        this->rope_rms_weights[layer_idx].sync_to_device();
    }
    this->clear_context();
}

void qwen2_npu::Impl::_set_rope_rms_weights(int idx)
{
    buffer<bf16> cos_buf(desc.DH / 2);
    buffer<bf16> sin_buf(desc.DH / 2);
    for (uint32_t j = 0; j < desc.DH / 2; j++){
        cos_buf[j] = (bf16)cos(desc.inv_freq[j] * idx);
        sin_buf[j] = (bf16)sin(desc.inv_freq[j] * idx);
    }

    for (uint32_t i = 0; i < desc.num_hidden_layers; i++){
        bf16 *w_rope_ptr = this->rope_rms_weights[i].data();
        for (uint32_t j = 0; j < desc.DH / 2; j++){
            w_rope_ptr[j] = cos_buf[j];
            w_rope_ptr[j + desc.DH / 2] = sin_buf[j];
        }
        this->rope_rms_weights[i].sync_to_device();
    }
}

void qwen2_npu::Impl::set_context_length(int L)
{
    this->current_context_length = L;
    this->dirty_rows = std::max(this->dirty_rows, static_cast<size_t>(L));

    this->sequence->gen_layer_seq(this->layer.seq(), L + 1);
    if (!this->npu->is_preemption_enabled()){
        this->layers_run.reset();

        for (uint32_t i = 0; i < desc.num_hidden_layers; i++){
            this->layers_run.add(this->layer.create_run(this->x, this->proj_weights[i], this->rms_weights[i], this->rope_rms_weights[i], this->kv_caches[i]));
        }
    }
    this->_set_rope_rms_weights(L);
}

buffer<bf16> qwen2_npu::Impl::forward(int ids)
{
    this->embedding->forward(ids, this->x);
    this->x.sync_to_device();

    if (!this->npu->is_preemption_enabled()){
        this->layers_run.execute();
        flm::npu::wait_for_completion(this->layers_run, "layers_run");
    }
    else{
        for (uint32_t i = 0; i < desc.num_hidden_layers; i++){
            this->layer(this->x, this->proj_weights[i], this->rms_weights[i], this->rope_rms_weights[i], this->kv_caches[i]);
        }
    }

    this->lm_head_run.start();
    this->set_context_length(this->current_context_length + 1);
    flm::npu::wait_for_completion(this->lm_head_run, "lm_head_run");
    this->logits.sync_from_device();
    return this->logits_valid;
}

buffer<bf16> qwen2_npu::Impl::prefill(std::vector<int> &ids, void *payload)
{
#ifdef MVPREFILL
    header_print("warning", "Prefilling with Decoder, Slow!");
    return this->_prefill_with_mv(ids);
#else
    return this->_prefill_with_mm(ids);
#endif
}

buffer<bf16> qwen2_npu::Impl::_prefill_with_mv(std::vector<int> &ids)
{
    for (int i = 0; i < ids.size(); i++){
        this->embedding->forward(ids[i], this->x);
        this->x.sync_to_device();

        for (uint32_t l = 0; l < desc.num_hidden_layers; l++){
            this->layer(this->x, this->proj_weights[l], this->rms_weights[l], this->rope_rms_weights[l], this->kv_caches[l]);
        }
        this->set_context_length(this->current_context_length + 1);
    }
    get_logits(this->x);
    this->logits.sync_from_device();
    return this->logits_valid;
}

buffer<bf16> qwen2_npu::Impl::_prefill_with_mm(std::vector<int> &ids)
{
    int L_in = ids.size();
    qwen2_prefill_shape shape = this->pctx->setup(L_in, this->current_context_length);

    // embedding
    for (int i = 0; i < L_in; i++){
        this->embedding->forward(ids[i], this->pctx->residual_row(i));
    }

    for (uint32_t layer_idx = 0; layer_idx < desc.num_hidden_layers; layer_idx++){
        this->pctx->forward(
            layer_idx,
            shape,
            this->proj_weights[layer_idx],
            this->rms_weights[layer_idx],
            this->kv_caches[layer_idx]
        );
    }

    buffer<bf16> predict = this->pctx->residual_row(L_in - 1);
    buffer<bf16> logits_result = get_logits(predict);

    this->set_context_length(this->current_context_length + L_in);
    return logits_result;
}

buffer<bf16> qwen2_npu::Impl::get_k_cache(int layer_idx, int idx)
{
    this->kv_caches[layer_idx].sync_from_device();
    buffer<bf16> k_cache(desc.DK);
    memcpy(k_cache.data(), this->kv_caches[layer_idx].data() + (size_t)idx * desc.DK, desc.DK * sizeof(bf16));
    return k_cache;
}

buffer<bf16> qwen2_npu::Impl::get_v_cache(int layer_idx, int idx)
{
    this->kv_caches[layer_idx].sync_from_device();
    buffer<bf16> v_cache(desc.DV);
    memcpy(v_cache.data(), this->kv_caches[layer_idx].data() + this->V_offset + (size_t)idx * desc.DV, desc.DV * sizeof(bf16));
    return v_cache;
}

buffer<bf16> qwen2_npu::Impl::get_logits(buffer<bf16> &predict)
{
    memcpy(this->x.data(), predict.data(), desc.D * sizeof(bf16));
    this->x.sync_to_device();
    this->lm_head_run.start();
    flm::npu::wait_for_completion(this->lm_head_run, "lm_head_run");
    this->logits.sync_from_device();
    return this->logits_valid;
}

void qwen2_npu::Impl::clear_context()
{
    // saved states hold their own copies of the rows; the live ones must not leak into the next conversation
    kv_rows_zero(this->kv_caches, kv_layout(), 0, this->dirty_rows);
    this->dirty_rows = 0;
    this->set_context_length(0);
}

int qwen2_npu::Impl::get_current_context_length()
{
    return this->current_context_length;
}

void qwen2_npu::Impl::_allocate_checkpoint_buffers(){}

int qwen2_npu::Impl::checkpoint()
{
    header_print_r("FLM", "Creating checkpoint at context length " + std::to_string(this->current_context_length));
    is_checkpoint_valid = true;
    checkpoint_context_length = this->current_context_length;
    return checkpoint_context_length;
}

int qwen2_npu::Impl::restore()
{
    if (!is_checkpoint_valid) {
        header_print("FLM", "No valid checkpoint found!");
        return -1;
    }
    header_print_r("FLM", "Restoring checkpoint at context length " + std::to_string(checkpoint_context_length));
    this->set_context_length(checkpoint_context_length);
    return checkpoint_context_length;
}

kv_row_layout qwen2_npu::Impl::kv_layout() const
{
    kv_row_layout l;
    l.DK = desc.DK; l.DV = desc.DV; l.striped = false;
    l.v_offset = V_offset; // K rows start at the beginning of the layer buffer, V rows at the middle
    return l;
}

std::unique_ptr<hybrid_engine_state> qwen2_npu::Impl::new_state()
{
    return kv_rows_new_state(desc.num_hidden_layers, kv_layout());
}

int qwen2_npu::Impl::read_state(hybrid_engine_state& state, int kv_from)
{
    header_print_r("FLM", "Reading state at context length " + std::to_string(this->current_context_length) + ", K/V rows from " + std::to_string(kv_from));
    return kv_rows_read(state, this->kv_caches, kv_layout(), this->current_context_length, kv_from);
}

int qwen2_npu::Impl::write_state_kv(const hybrid_engine_state& state)
{
    return kv_rows_write(state, this->kv_caches, kv_layout(), this->MAX_L);
}

int qwen2_npu::Impl::write_state(const hybrid_engine_state& state)
{
    const int length = kv_rows_write(state, this->kv_caches, kv_layout(), this->MAX_L);
    if (length < 0)
    {
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

void qwen2_npu::Impl::update_max_length(uint32_t MAX_L)
{
    if (MAX_L <= this->MAX_L){
        header_print("FLM", "New length is shorter than the current length, no need to update!");
        return; // no need to update
    }
    this->MAX_L = MAX_L;
    size_t kv_cache_size = desc.get_kv_cache_size(MAX_L);
    buffer<bf16> kv_cache_temp(this->kv_caches[0].size());
    size_t original_V_offset = this->V_offset;
    this->sequence->set_max_length(MAX_L);
    this->V_offset = this->sequence->get_V_Offset();
    for (uint32_t i = 0; i < desc.num_hidden_layers; i++){
        this->kv_caches[i].sync_from_device();
        memcpy(static_cast<void *>(kv_cache_temp.data()), static_cast<void *>(this->kv_caches[i].data()), this->kv_caches[i].size() * sizeof(bf16));
        this->kv_caches[i] = layer.create_bo_buffer<bf16>(kv_cache_size);
        memcpy(static_cast<void *>(this->kv_caches[i].data()), static_cast<void *>(kv_cache_temp.data()), (kv_cache_temp.size() / 2) * sizeof(bf16));
        memcpy(static_cast<void *>(this->kv_caches[i].data() + V_offset), static_cast<void *>(kv_cache_temp.data() + original_V_offset), (kv_cache_temp.size() / 2) * sizeof(bf16));
        this->kv_caches[i].sync_to_device();
    }
    // the kv cache layout is baked into the mha prefill sequence
    this->pctx->set_kv_layout(this->V_offset);
    this->set_context_length(this->current_context_length); // regenerate sequence
}

///@brief destructor of qwen2_npu
qwen2_npu::Impl::~Impl() = default;

// ===============================================
// Externals for qwen2_npu
// ===============================================
qwen2_npu::qwen2_npu(LM_Config config, npu_xclbin_manager *npu_instance, int MAX_L)
{
    this->_impl = new Impl(config, npu_instance, MAX_L);
}

buffer<bf16> qwen2_npu::forward(int ids)
{
    return this->_impl->forward(ids);
}

buffer<bf16> qwen2_npu::prefill(std::vector<int> &ids, void *payload){
    try {
        buffer<bf16> result = this->_impl->prefill(ids, payload);
        return result;
    }
    catch (const std::runtime_error& e) {
        header_print("FLM", e.what());
        throw;
    }
}

void qwen2_npu::set_context_length(int L)
{
    this->_impl->set_context_length(L);
}

void qwen2_npu::load_weights(Q4NX &q4nx)
{
    this->_impl->load_weights(q4nx);
}

void qwen2_npu::update_max_length(uint32_t MAX_L)
{
    this->_impl->update_max_length(MAX_L);
}

void qwen2_npu::clear_context()
{
    this->_impl->clear_context();
}

buffer<bf16> qwen2_npu::get_k_cache(int layer_idx, int idx)
{
    return this->_impl->get_k_cache(layer_idx, idx);
}

buffer<bf16> qwen2_npu::get_v_cache(int layer_idx, int idx)
{
    return this->_impl->get_v_cache(layer_idx, idx);
}

int qwen2_npu::get_current_context_length()
{
    return this->_impl->get_current_context_length();
}

int qwen2_npu::checkpoint()
{
    return this->_impl->checkpoint();
}

int qwen2_npu::restore()
{
    return this->_impl->restore();
}

std::unique_ptr<hybrid_engine_state> qwen2_npu::new_state()
{
    return this->_impl->new_state();
}

int qwen2_npu::read_state(hybrid_engine_state& state, int kv_from)
{
    return this->_impl->read_state(state, kv_from);
}

int qwen2_npu::write_state(const hybrid_engine_state& state)
{
    return this->_impl->write_state(state);
}

int qwen2_npu::write_state_kv(const hybrid_engine_state& state)
{
    return this->_impl->write_state_kv(state);
}

extern "C" const flm_state_api* qwen2_npu_state_api(void* engine)
{
    return hybrid_state_api<qwen2_npu>(engine);
}

qwen2_npu::~qwen2_npu()
{
    delete this->_impl;
}
