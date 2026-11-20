// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include "qwen3vl_flash_detail.hpp"
#include <algorithm>

#include <exception>
#include <thread>

qwen3vl_flash::Impl::Impl(LM_Config config, npu_xclbin_manager *npu_instance, int MAX_L)
    : config(config), npu(npu_instance)
{
    is_vlm = config.get<bool>("is_vlm", false);
    current_context_length = 0;
    is_checkpoint_valid = false;

    this->desc.build(config);

    // This engine is pinned to a 1 k context: the requested length is ignored and
    // update_max_length() below refuses to grow it. The prefill scratch is cut for
    // whatever chunk arrives, and on a machine whose npu carve-out has no headroom
    // a chunk longer than 1 k fails to create its bos -- capping the context is
    // what keeps a longer chunk from ever being asked for.
    static constexpr uint32_t FORCED_MAX_L = 1024;
    this->MAX_L = FORCED_MAX_L;
    MAX_L = this->MAX_L; // synchronize MAX_L

    this->sequence = std::make_unique<qwen3vl_flash_sequence>(this->desc, config, MAX_L);

    if (is_vlm)
    {
        this->qwen3vl_image_encoder = std::make_unique<Qwen3VLImageEncoder>(config, npu);
    }

    layer_app_manager = npu->register_xclbin(utils::path_join(config.exec_path, "xclbins", config.model_name, "layer.xclbin"));
    this->layer = layer_app_manager->create_app();
    if (!this->npu->is_preemption_enabled())
    {
        this->layers_run = layer_app_manager->create_runlist();
    }
    this->lm_head = layer_app_manager->create_app();

    this->pctx = std::make_unique<qwen3vl_prefill_context>(npu, &this->desc, this->config, this->sequence.get(), this->MAX_L);

    // allocate all buffers
    rms_weights.resize(desc.num_hidden_layers);
    proj_weights.resize(desc.num_hidden_layers);
    kv_caches.resize(desc.num_hidden_layers);
    rope_rms_weights.resize(desc.num_hidden_layers);

    size_t kv_cache_size = desc.get_kv_cache_size(MAX_L);

    for (size_t i = 0; i < desc.num_hidden_layers; i++)
    {
        proj_weights[i] = layer.create_bo_buffer<u8>(desc.total_proj_weights_byte_size);
        rms_weights[i] = layer.create_bo_buffer<bf16>(desc.rms_weights_byte_size / sizeof(bf16));
        kv_caches[i] = layer.create_bo_buffer<bf16>(kv_cache_size);
        rope_rms_weights[i] = layer.create_bo_buffer<bf16>(desc.rope_rms_weights_byte_size / sizeof(bf16));
        this->kv_caches[i].sync_to_device();
    }
    x = layer.create_bo_buffer<bf16>(desc.D);
    lm_head_weights = layer.create_bo_buffer<u8>(desc.get_lm_head_w_size());
    final_rms_weight = layer.create_bo_buffer<bf16>(desc.D);
    logits = layer.create_bo_buffer<bf16>(desc.vocab_size_padded);
    logits_valid = buffer<bf16>(logits.data(), desc.vocab_size);

    // the prefill mlp pre-arms one up/gate runlist per layer, so it needs the
    // weight buffers before the first setup(); the handles are stable for life
    this->pctx->set_weights(this->proj_weights);

    this->K03_offset = sequence->get_k03_offset();
    this->K47_offset = sequence->get_k47_offset();
    this->V03_offset = sequence->get_v03_offset();
    this->V47_offset = sequence->get_v47_offset();
    this->pctx->set_kv_layout(K03_offset, K47_offset, V03_offset, V47_offset);

    this->clear_context();
    this->embedding = std::make_unique<Embedding>(desc.vocab_size, desc.D);

    this->sequence->gen_lm_head_seq(this->lm_head.seq());
    this->lm_head_run = this->lm_head.create_run(this->logits, this->lm_head_weights, this->x, this->final_rms_weight);
}

void qwen3vl_flash::Impl::load_weights(Q4NX &q4nx)
{
    if (is_vlm)
    {
        SafeTensors vision_weights(config.get<std::string>("vision_model_weight", ""));
        this->qwen3vl_image_encoder->init_weights(vision_weights);
    }
    this->embedding->init_weights(&q4nx, "model.embed_tokens");
    this->desc.load_head_weights(q4nx, this->lm_head_weights, this->final_rms_weight);
    for (uint32_t layer_idx = 0; layer_idx < desc.num_hidden_layers; layer_idx++)
    {
        this->desc.load_layer_weights(layer_idx, q4nx,
            this->proj_weights[layer_idx],
            this->rms_weights[layer_idx],
            this->rope_rms_weights[layer_idx]);
    }
    this->clear_context();
}

void qwen3vl_flash::Impl::_set_rope_rms_weights(int idx)
{
    const uint32_t DH = desc.DH;
    buffer<bf16> cos_buf(DH / 2);
    buffer<bf16> sin_buf(DH / 2);
    for (uint32_t j = 0; j < DH / 2; j++)
    {
        cos_buf[j] = (bf16)cos(desc.inv_freq[j] * idx);
        sin_buf[j] = (bf16)sin(desc.inv_freq[j] * idx);
    }

    for (uint32_t i = 0; i < desc.num_hidden_layers; i++)
    {
        bf16 *w_rope_ptr = this->rope_rms_weights[i].data();
        for (uint32_t j = 0; j < DH / 2; j++)
        {
            w_rope_ptr[j] = cos_buf[j];
            w_rope_ptr[j + DH / 2] = sin_buf[j];
        }
        this->rope_rms_weights[i].sync_to_device();
    }
}

void qwen3vl_flash::Impl::set_context_length(int L, int new_general_token_idx)
{
    this->current_context_length = L;
    this->general_token_idx = new_general_token_idx;

    this->sequence->gen_layer_seq(this->layer.seq(), L + 1);
    if (!this->npu->is_preemption_enabled())
    {
        this->layers_run.reset();

        for (uint32_t i = 0; i < desc.num_hidden_layers; i++)
        {
            this->layers_run.add(this->layer.create_run(this->x, this->proj_weights[i], this->rms_weights[i], this->rope_rms_weights[i], this->kv_caches[i]));
        }
    }

    // decode rotates by the temporal clock only, so the on-device table is 1-D
    this->_set_rope_rms_weights(new_general_token_idx);
}

buffer<bf16> qwen3vl_flash::Impl::forward(int ids)
{
    this->embedding->forward(ids, this->x);
    this->x.sync_to_device();

    if (!this->npu->is_preemption_enabled())
    {
        this->layers_run.execute();
        flm::npu::wait_for_completion(this->layers_run, "layers_run");
    }
    else
    {
        for (uint32_t i = 0; i < desc.num_hidden_layers; i++)
        {
            this->layer(this->x, this->proj_weights[i], this->rms_weights[i], this->rope_rms_weights[i], this->kv_caches[i]);
        }
    }

    this->lm_head_run.start();
    this->set_context_length(this->current_context_length + 1, this->general_token_idx + 1);
    flm::npu::wait_for_completion(this->lm_head_run, "lm_head_run");
    this->logits.sync_from_device();
    return this->logits_valid;
}

buffer<bf16> qwen3vl_flash::Impl::prefill(std::vector<int> &ids, void *payload)
{
    // Nothing below bounds-checks the prompt against the pinned context: a longer
    // one sizes the prefill scratch for rows the kv cache has no room for, and the
    // mha sequence then addresses past the end of the cache bo -- a silent
    // out-of-bounds write on the device, not a failure. So refuse it here.
    if ((size_t)this->current_context_length + ids.size() > (size_t)this->MAX_L) {
        throw std::runtime_error(
            "This model only supports a " + std::to_string(this->MAX_L) + " token context: "
            + std::to_string(this->current_context_length) + " already cached + "
            + std::to_string(ids.size()) + " requested exceeds it");
    }
#ifdef MVPREFILL
    header_print("warning", "Prefilling with Decoder, Slow!");
    return this->_prefill_with_mv(ids);
#else
    return this->_prefill_with_mm(ids, payload);
#endif
}

buffer<bf16> qwen3vl_flash::Impl::_prefill_with_mv(std::vector<int> &ids)
{
    for (int i = 0; i < ids.size(); i++)
    {
        this->embedding->forward(ids[i], this->x);
        this->x.sync_to_device();

        for (uint32_t l = 0; l < desc.num_hidden_layers; l++)
        {
            this->layer(this->x, this->proj_weights[l], this->rms_weights[l], this->rope_rms_weights[l], this->kv_caches[l]);
        }
        this->set_context_length(this->current_context_length + 1, this->general_token_idx + 1);
    }
    buffer<bf16> logits = get_logits(this->x);

    return logits;
}

buffer<bf16> qwen3vl_flash::Impl::_prefill_with_mm(std::vector<int> &ids, void *payload)
{
    int L_effective = ids.size();

    // the mrope indices, and which rows are vision patches, come from the prompt
    std::vector<bool> vision_mask(L_effective);
    std::vector<int> input_idx = this->get_input_idx(vision_mask, ids, (qwen3vl_image_payload_t *)payload);

    // the encoder returns four embeddings per patch: [0..2] are the deepstack
    // residuals for the first three layers, [3] is the input embedding
    qwen3vl_prefill_shape shape;
    std::vector<std::vector<bf16>> image_embedding;
    if (is_vlm && payload != nullptr)
    {
        // the vision encoder spends most of its time waiting on NPU dispatches,
        // while the text prefill setup is pure host work on a disjoint set of
        // apps and buffers -- so the two run side by side and the setup is free.
        qwen3vl_vision_shape vision_shape = this->qwen3vl_image_encoder->setup(payload);

        std::exception_ptr setup_exception;
        std::thread setup_thread([&]() {
            try
            {
                shape = this->pctx->setup(L_effective, this->current_context_length);
            }
            catch (...)
            {
                setup_exception = std::current_exception();
            }
        });

        try
        {
            image_embedding = this->qwen3vl_image_encoder->encode(payload, vision_shape);
        }
        catch (...)
        {
            setup_thread.join();
            throw;
        }
        setup_thread.join();
        if (setup_exception)
        {
            std::rethrow_exception(setup_exception);
        }
    }
    else
    {
        shape = this->pctx->setup(L_effective, this->current_context_length);
    }

    // embedding straight into the residual stream, vision rows spliced in
    bf16 *vision_token_ptr = (payload != nullptr) ? image_embedding[3].data() : nullptr;
    for (int i = 0; i < L_effective; i++)
    {
        if (vision_mask[i] && (payload != nullptr))
        {
            memcpy(this->pctx->residual_row(i).data(), vision_token_ptr, desc.D * sizeof(bf16));
            vision_token_ptr += desc.D;
        }
        else
        {
            this->embedding->forward(ids[i], this->pctx->residual_row(i));
        }
    }

    for (uint32_t layer_idx = 0; layer_idx < desc.num_hidden_layers; layer_idx++)
    {
        this->pctx->forward(layer_idx, shape, input_idx,
            this->proj_weights[layer_idx],
            this->rms_weights[layer_idx],
            this->rope_rms_weights[layer_idx],
            this->kv_caches[layer_idx]);

        // deepstack: the first three layers each get their own vision residual
        if (is_vlm && (payload != nullptr) && layer_idx < 3)
        {
            vision_token_ptr = image_embedding[layer_idx].data();
            for (int ll = 0; ll < L_effective; ll++)
            {
                if (vision_mask[ll])
                {
                    buffer<bf16> temp_buf = buffer<bf16>(vision_token_ptr, desc.D);
                    buffer<bf16> &row = this->pctx->residual_row(ll);
                    qwen3vl_cpu_func::_residual_add(row, temp_buf, row);
                    vision_token_ptr += desc.D;
                }
            }
        }
    }

    buffer<bf16> predict = this->pctx->residual_row(L_effective - 1);
    get_logits(predict);
    this->set_context_length(this->current_context_length + L_effective, input_idx[L_effective * 3 - 1] + 1);
    return this->logits_valid;
}

std::vector<int> qwen3vl_flash::Impl::get_input_idx(std::vector<bool> &vision_mask, std::vector<int> &tokens, qwen3vl_image_payload_t *image_payload)
{
    int l = tokens.size();
    bool in_image = false;
    int image_counter = 0;
    int pixel_counter = 0;
    std::vector<int> input_idx(l * 3);
    for (int i = 0; i < tokens.size(); i++)
    {
        vision_mask[i] = (tokens[i] == image_soft_token_id);
        if (vision_mask[i])
        {
            if (!in_image)
            {
                in_image = true;
                pixel_counter = 0;
            }
            int h = image_payload->images[image_counter].grid_h;
            int w = image_payload->images[image_counter].grid_w;
            h = h / 2;
            w = w / 2;
            // a patch keeps the temporal clock and adds its own (row, column)
            input_idx[i * 3 + 0] = general_token_idx;
            input_idx[i * 3 + 1] = general_token_idx + pixel_counter / w;
            input_idx[i * 3 + 2] = general_token_idx + pixel_counter % w;
            pixel_counter += 1;
        }
        else
        {
            if (in_image)
            {
                in_image = false;
                // the image as a whole costs its longer side (in merged tokens) of temporal clock
                general_token_idx += std::max(image_payload->images[image_counter].grid_h,
                                           image_payload->images[image_counter].grid_w) / 2;
                image_counter++;
                pixel_counter = 0;
            }
            input_idx[i * 3 + 0] = general_token_idx;
            input_idx[i * 3 + 1] = general_token_idx;
            input_idx[i * 3 + 2] = general_token_idx;
            general_token_idx += 1;
        }
    }

    return input_idx;
}

buffer<bf16> qwen3vl_flash::Impl::get_k_cache(int layer_idx, int idx)
{
    this->kv_caches[layer_idx].sync_from_device();
    buffer<bf16> k_cache(desc.DK);
    bf16 *k_cache_ptr = this->kv_caches[layer_idx].data();
    uint32_t offset = idx * desc.DH * 4;
    memcpy(static_cast<void *>(k_cache.data()), static_cast<void *>(k_cache_ptr + offset), desc.DK / 2 * sizeof(bf16));
    memcpy(static_cast<void *>(k_cache.data() + desc.DK / 2), static_cast<void *>(k_cache_ptr + offset + K47_offset), desc.DK / 2 * sizeof(bf16));

    return k_cache;
}

buffer<bf16> qwen3vl_flash::Impl::get_v_cache(int layer_idx, int idx)
{
    this->kv_caches[layer_idx].sync_from_device();
    buffer<bf16> v_cache(desc.DV);
    bf16 *v_cache_ptr = this->kv_caches[layer_idx].data();
    uint32_t offset = idx * desc.DH * 4;
    memcpy(static_cast<void *>(v_cache.data()), static_cast<void *>(v_cache_ptr + offset + V03_offset), desc.DV / 2 * sizeof(bf16));
    memcpy(static_cast<void *>(v_cache.data() + desc.DV / 2), static_cast<void *>(v_cache_ptr + offset + V47_offset), desc.DV / 2 * sizeof(bf16));
    return v_cache;
}

buffer<bf16> qwen3vl_flash::Impl::get_logits(buffer<bf16> &predict)
{
    memcpy(this->x.data(), predict.data(), desc.D * sizeof(bf16));
    this->x.sync_to_device();
    this->lm_head_run.start();
    flm::npu::wait_for_completion(this->lm_head_run, "lm_head_run");
    this->logits.sync_from_device();
    return this->logits_valid;
}

void qwen3vl_flash::Impl::clear_context()
{
    // prefill attends over whole chunks, so rows past the prompt must not keep
    // an earlier request's k/v: they make identical requests give different logits
    for (auto& kv_cache : this->kv_caches){
        kv_cache.sync_from_device();
        memset(static_cast<void*>(kv_cache.data()), 0, kv_cache.size() * sizeof(bf16));
        kv_cache.sync_to_device();
    }
    this->set_context_length(0, 0);
}

int qwen3vl_flash::Impl::get_current_context_length()
{
    return this->current_context_length;
}

void qwen3vl_flash::Impl::_allocate_checkpoint_buffers() {}

int qwen3vl_flash::Impl::checkpoint()
{
    header_print_r("FLM", "Creating checkpoint at context length " + std::to_string(this->current_context_length));
    is_checkpoint_valid = true;
    checkpoint_context_length = this->current_context_length;
    checkpoint_general_token_idx = this->general_token_idx;
    return checkpoint_context_length;
}

int qwen3vl_flash::Impl::restore()
{
    if (!is_checkpoint_valid) {
        header_print("FLM", "No valid checkpoint found!");
        return -1;
    }
    header_print_r("FLM", "Restoring checkpoint at context length " + std::to_string(checkpoint_context_length));
    this->set_context_length(checkpoint_context_length, checkpoint_general_token_idx);
    return checkpoint_context_length;
}

void qwen3vl_flash::Impl::update_max_length(uint32_t MAX_L)
{
    // the context is pinned at construction, so there is nothing to grow into:
    // a longer context would re-cut the prefill scratch for a longer chunk, which
    // is exactly the allocation this engine is pinned to avoid
    if (MAX_L != this->MAX_L) {
        header_print("WARNING", "This model only supports a 1k context; the request for " + std::to_string(MAX_L) + " tokens is ignored and the context stays at " + std::to_string(this->MAX_L));
    }
    return;

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
    this->pctx->set_kv_layout(K03_offset, K47_offset, V03_offset, V47_offset);
    for (uint32_t i = 0; i < desc.num_hidden_layers; i++)
    {
        this->kv_caches[i].sync_from_device();
        memcpy(static_cast<void *>(kv_cache_temp.data()), static_cast<void *>(this->kv_caches[i].data()), this->kv_caches[i].size() * sizeof(bf16));
        this->kv_caches[i] = layer.create_bo_buffer<bf16>(kv_cache_size);
        this->kv_caches[i].sync_to_device();
        memcpy(static_cast<void *>(this->kv_caches[i].data() + this->K03_offset), static_cast<void *>(kv_cache_temp.data() + original_k03_offset), (kv_cache_temp.size() / 4) * sizeof(bf16));
        memcpy(static_cast<void *>(this->kv_caches[i].data() + this->V03_offset), static_cast<void *>(kv_cache_temp.data() + original_v03_offset), (kv_cache_temp.size() / 4) * sizeof(bf16));
        memcpy(static_cast<void *>(this->kv_caches[i].data() + this->K47_offset), static_cast<void *>(kv_cache_temp.data() + original_k47_offset), (kv_cache_temp.size() / 4) * sizeof(bf16));
        memcpy(static_cast<void *>(this->kv_caches[i].data() + this->V47_offset), static_cast<void *>(kv_cache_temp.data() + original_v47_offset), (kv_cache_temp.size() / 4) * sizeof(bf16));
    }
    this->set_context_length(this->current_context_length, this->general_token_idx); // regenerate sequence
}
///@brief destructor of qwen3vl_flash
qwen3vl_flash::Impl::~Impl() = default;

// ===============================================
// Externals for qwen3vl_flash
// ===============================================
qwen3vl_flash::qwen3vl_flash(LM_Config config, npu_xclbin_manager *npu_instance, int MAX_L)
{
    this->_impl = new Impl(config, npu_instance, MAX_L);
}

buffer<bf16> qwen3vl_flash::forward(int ids)
{
    return this->_impl->forward(ids);
}

buffer<bf16> qwen3vl_flash::prefill(std::vector<int> &ids, void *payload)
{
    try {
        buffer<bf16> result = this->_impl->prefill(ids, payload);
        return result;
    }
    catch (const std::runtime_error& e) {
        header_print("FLM", e.what());
        throw;
    }
}

void qwen3vl_flash::set_context_length(int L)
{
    // the mrope clock cannot be recovered from L alone, so this is a no-op
    std::cout << "Setting context length is not supported for qwen3vl_flash!" << std::endl;
}

void qwen3vl_flash::load_weights(Q4NX &q4nx)
{
    this->_impl->load_weights(q4nx);
}

void qwen3vl_flash::update_max_length(uint32_t MAX_L)
{
    this->_impl->update_max_length(MAX_L);
}

void qwen3vl_flash::clear_context()
{
    this->_impl->clear_context();
}

buffer<bf16> qwen3vl_flash::get_k_cache(int layer_idx, int idx)
{
    return this->_impl->get_k_cache(layer_idx, idx);
}

buffer<bf16> qwen3vl_flash::get_v_cache(int layer_idx, int idx)
{
    return this->_impl->get_v_cache(layer_idx, idx);
}

int qwen3vl_flash::get_current_context_length()
{
    return this->_impl->get_current_context_length();
}

int qwen3vl_flash::checkpoint()
{
    return this->_impl->checkpoint();
}

int qwen3vl_flash::restore()
{
    return this->_impl->restore();
}

qwen3vl_flash::~qwen3vl_flash()
{
    delete this->_impl;
}
