// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#include <limits>
#include <stdexcept>
#include <algorithm>
#include <omp.h>
#include "minicpm_v_4_7_moe_npu_detail.hpp"
#include "image/vision_common.hpp"
#include "image/vision_tower.hpp"
#include "image/npu/vision_op_factory.hpp"
#include <filesystem>
#include "metrices.hpp"
#include "utils/error_measure.hpp"
#include "npu_sequences/GateDeltaNetPrefillRuntimeSequence.hpp"
minicpm_v_4_7_moe_npu::Impl::Impl(LM_Config config, npu_xclbin_manager *npu_instance, minicpm_v_4_7_moe_npu* parent_ptr, int MAX_L )
    : config(config), npu(npu_instance), parent_ptr(parent_ptr){
    is_vlm = config.get<bool>("is_vlm", false);
    current_context_length = 0;
    // build the weight-shape definition (dimensions + per-layer weight descriptors)
    desc.build(config);
    // an attention layer's cache is K rows then V rows, each half MAX_L rows long; the slot copies
    // and update_max_length find the V half at the middle of the buffer, which needs DK == DV
    if (desc.DK != desc.DV) {
        throw std::runtime_error("minicpm_v_4_7_moe_npu: DK != DV, the K/V cache layout assumed by checkpoints does not hold");
    }

    this->MAX_L = std::max(MAX_L, 65536);
    // make MAX_L a power of 2
    float log2_MAX_L = std::log2(this->MAX_L);
    if (log2_MAX_L != std::floor(log2_MAX_L)){
        this->MAX_L = 1 << ((int)std::floor(log2_MAX_L) + 1);
    }
    MAX_L = this->MAX_L; // synchronize MAX_L

    this->sequence = std::make_unique<minicpm_v_4_7_moe_npu_sequence>(this->desc, config, MAX_L);
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

    static constexpr size_t half_gib = 512ull * 1024 * 1024;
    size_t attn_rms_weights_byte_size = desc.attn_rms_weights_byte_size;

    size_t linear_rms_weights_byte_size = desc.linear_rms_weights_byte_size;
    size_t linear_ssm_weights_byte_size = desc.linear_ssm_weights_byte_size;
    size_t kv_cache_size = desc.get_kv_cache_size(this->MAX_L);

    size_t linear_kv_cache_byte_size = desc.get_linear_cache_size();
    size_t conv_w_byte_size = desc.get_conv_w_size();

    for (size_t layer_idx = 0; layer_idx < desc.num_hidden_layers; layer_idx++){
        if (is_attn_layer(layer_idx)){
            this->proj_weights[layer_idx] = this->layer.create_bo_buffer<u8>(half_gib);
        }
        else{
            this->proj_weights[layer_idx] = this->linear_layer.create_bo_buffer<u8>(half_gib);
        }
    }

    for (size_t i = 0; i < desc.num_hidden_layers; i++){
        if (is_attn_layer(i)){
            this->rms_weights[i] = this->layer.create_bo_buffer<bf16>(attn_rms_weights_byte_size / sizeof(bf16));
            this->rope_rms_weights[i] = this->layer.create_bo_buffer<bf16>(desc.RoPE_DIM + desc.DH * 2);
            this->kv_caches[i] = this->layer.create_bo_buffer<bf16>(kv_cache_size);
            this->kv_caches[i].memset((bf16)0);
            this->kv_caches[i].sync_to_device();
        }
        else{
            this->rms_weights[i] = this->linear_layer.create_bo_buffer<bf16>(linear_rms_weights_byte_size / sizeof(bf16));
            this->rope_rms_weights[i] = this->linear_layer.create_bo_buffer<bf16>(linear_ssm_weights_byte_size / sizeof(bf16));
            this->kv_caches[i] = this->linear_layer.create_bo_buffer<bf16>(linear_kv_cache_byte_size / sizeof(bf16));
            this->kv_caches[i].memset((bf16)0);
            this->kv_caches[i].sync_to_device();
        }
    }

    x = layer.create_bo_buffer<bf16>(desc.D * 2);
    lm_head_weights = layer.create_bo_buffer<u8>(desc.get_lm_head_w_size());
    logits = layer.create_bo_buffer<bf16>(desc.vocab_size_padded);
    logits_valid = buffer<bf16>(logits.data(), desc.vocab_size);

    this->clear_context();
    this->embedding = std::make_unique<Embedding>(desc.vocab_size, desc.D);

    pctx = std::make_unique<minicpm_v_4_7_moe_prefill_context>(npu, &desc, sequence.get(), config);

    lm_head = this->lm_head_app_manager->create_app();
    this->sequence->gen_lm_head_seq(lm_head.seq(), desc.vocab_size_padded, desc.D);
    this->lm_head_run = lm_head.create_run(this->logits, this->lm_head_weights, this->x);

    this->sequence->gen_layer_seq(this->linear_layer.seq(), 0, false, false);
    this->sequence->gen_layer_seq(this->first_linear_layer.seq(), 0, false, true);

    set_context_length(0, 0);

    // Last, so the language model's xclbins and buffers are all in place: the
    // tower uploads its 1.1 GB of weights into BOs here, at load, not per image.
    if (is_vlm){
        this->_load_vision();
    }
}

void minicpm_v_4_7_moe_npu::Impl::load_weights(Q4NX& q4nx){

    this->embedding->init_weights(&q4nx, "model.embed_tokens");

    for (int i = 0; i < desc.num_hidden_layers; i++){
        if (desc.is_attn_layer(i)){
            desc.load_attn_weights(i, q4nx, this->proj_weights[i], this->rms_weights[i], this->rope_rms_weights[i]);
        }
        else{
            desc.load_linear_weights(i, q4nx, this->proj_weights[i], this->rms_weights[i], this->rope_rms_weights[i]);
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

        desc.mcvmoe_reorder_cpy_lm_head(this->lm_head_weights.data(), lm_head_w, desc.D, flm_q80, 4);
        this->lm_head_weights.sync_to_device();
    }

    this->clear_context();
}

void minicpm_v_4_7_moe_npu::Impl::_set_rope_rms_weights(int idx){
    int RoPE_DIM = desc.RoPE_DIM;
    buffer<bf16> cos_buf(RoPE_DIM / 2);
    buffer<bf16> sin_buf(RoPE_DIM / 2);

    for (int j = 0; j < RoPE_DIM / 2; j++){
        cos_buf[j] = (bf16)cos(inv_freq[j] * idx);
        sin_buf[j] = (bf16)sin(inv_freq[j] * idx);
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

void minicpm_v_4_7_moe_npu::Impl::set_context_length(int L, int new_general_token_idx){
    this->current_context_length = L;
    this->general_token_idx = new_general_token_idx;

    this->sequence->gen_set_context(this->set_layer_rtp.seq(), L + 1);
    this->sequence->gen_layer_seq(this->layer.seq(), L + 1, true);
    // utils::print_matrix(this->x, 64, 10, 10);
    if (!this->npu->is_preemption_enabled()){
        this->layers_run.reset();

        this->layers_run.add(this->set_layer_rtp.create_run());
        for (uint32_t i = 0; i < desc.num_hidden_layers; i++){
            if (is_attn_layer(i)){
                this->layers_run.add(this->layer.create_run(this->proj_weights[i], this->x, this->rms_weights[i], this->rope_rms_weights[i], this->kv_caches[i]));
            }
            else {
                if (is_first_linear_layer(i)) {
                    this->layers_run.add(this->first_linear_layer.create_run(this->proj_weights[i], this->x, this->rms_weights[i], this->rope_rms_weights[i], this->kv_caches[i]));
                }
                else {
                    this->layers_run.add(this->linear_layer.create_run(this->proj_weights[i], this->x, this->rms_weights[i], this->rope_rms_weights[i], this->kv_caches[i]));
                }
            }
        }
    }

    this->_set_rope_rms_weights(new_general_token_idx);
}

buffer<bf16> minicpm_v_4_7_moe_npu::Impl::forward(int ids){
    // exit(0);
    this->embedding->forward(ids, this->x);
    this->x.sync_to_device();

    // if (!this->npu->is_preemption_enabled()){
    if (!this->npu->is_preemption_enabled()){
        this->layers_run.execute();
        flm::npu::wait_for_completion(this->layers_run, "layers_run");
    }
    else{
        this->set_layer_rtp();
        for (uint32_t i = 0; i < desc.num_hidden_layers; i++){
            if (is_attn_layer(i)){
                this->layer(this->proj_weights[i], this->x, this->rms_weights[i], this->rope_rms_weights[i], this->kv_caches[i]);
            }
            else if (is_first_linear_layer(i)) {
                this->first_linear_layer(this->proj_weights[i], this->x, this->rms_weights[i], this->rope_rms_weights[i], this->kv_caches[i]);
                this->x.sync_from_device();
            }
            else {
                this->linear_layer(this->proj_weights[i], this->x, this->rms_weights[i], this->rope_rms_weights[i], this->kv_caches[i]);
                this->x.sync_from_device();
            }
        }
    }

    this->lm_head_run.start();
    this->set_context_length(this->current_context_length + 1, this->general_token_idx + 1);
    flm::npu::wait_for_completion(this->lm_head_run, "lm_head_run");
    this->logits.sync_from_device();
    // utils::print_matrix(this->logits_valid, this->logits_valid.size() / 4);
    return this->logits_valid;
}

buffer<bf16> minicpm_v_4_7_moe_npu::Impl::prefill(std::vector<int>& ids, void* payload){
#ifdef MVPREFILL
    header_print("warning", "Prefilling with Decoder, Slow!");
    return this->_prefill_with_mv(ids);
#else
    return this->_prefill_with_mm(ids, payload);
#endif
}

buffer<bf16> minicpm_v_4_7_moe_npu::Impl::_prefill_with_mv(std::vector<int>& ids, void* payload){
    // For debuging
    // SafeTensors aref(utils::path_join(config.model_path, "activations.safetensors"));

    for (int i = 0; i < (int)ids.size(); i++){
        this->embedding->forward(ids[i], this->x);
        utils::print_matrix(this->x, desc.D * 2);
        this->x.sync_to_device();
        this->set_layer_rtp();
        for (uint32_t i = 0; i < desc.num_hidden_layers; i++){
            if (is_attn_layer(i)){
                this->layer(this->proj_weights[i], this->x, this->rms_weights[i], this->rope_rms_weights[i], this->kv_caches[i]);
                this->x.sync_from_device();
                utils::print_matrix(this->x, desc.D * 2);
            }
            else {
                if (is_first_linear_layer(i)) {
                    this->first_linear_layer(this->proj_weights[i], this->x, this->rms_weights[i], this->rope_rms_weights[i], this->kv_caches[i]);
                }
                else {
                    this->linear_layer(this->proj_weights[i], this->x, this->rms_weights[i], this->rope_rms_weights[i], this->kv_caches[i]);
                }
            }
        }
        std::cout << "Token finished!" << std::endl;
        this->x.sync_from_device();
        utils::print_matrix(this->x, desc.D * 2);
        this->set_context_length(this->current_context_length + 1, this->general_token_idx + 1);
    }

    this->lm_head_run.start();
    flm::npu::wait_for_completion(this->lm_head_run, "lm_head_run");
    this->logits.sync_from_device();

    return logits_valid;
}

buffer<bf16> minicpm_v_4_7_moe_npu::Impl::_prefill_with_mm(std::vector<int>& ids, void* payload){
#ifdef ENABLE_DEBUG
    SafeTensors aref(utils::path_join(config.model_path, "activations.safetensors"));

    buffer<int> input_ids; aref.load_weights(input_ids, "input_ids");
#endif
    int L_in = ids.size();

#ifdef ENABLE_DEBUG
    std::cout << std::endl;
    header_print("DEBUG", "DEBUG ENABLED!");
    L_in = input_ids.size();
    ids.resize(L_in);
    memcpy(ids.data(), input_ids.data(), L_in * sizeof(int));
    header_print("DEBUG", "Debuging input ids:" << L_in);
#endif

    buffer<bf16> hidden_states((size_t)desc.D * L_in);
    tensor_2d<bf16> hidden_states_2d(hidden_states, desc.D);

    const auto* p = static_cast<const minicpm_v_moe_image_payload_t*>(payload);
    const bool has_images = p != nullptr && !p->images.empty();

    int n_image_tokens = 0;
    for (int id : ids){
        if (id == video_token_id){
            throw std::runtime_error("the prompt carries video tokens (" + std::to_string(video_token_id) +
                                     "): this engine has no video path");
        }
        n_image_tokens += (id == canvas_ids.image_pad);
    }
    std::vector<mcvm::view_grid> views;
    std::vector<bf16> image_rows;
    if (has_images){
        image_rows = this->_encode_images(p, n_image_tokens, views);
    }
    else if (n_image_tokens > 0){
        throw std::runtime_error("the prompt carries " + std::to_string(n_image_tokens) +
                                 " image tokens but no image payload");
    }

    // the tower's rows overwrite the image_token_id rows in order, bypassing the embedding table
    const bf16* vision_row = image_rows.data();
    for (int l = 0; l < L_in; l++) {
        if (has_images && ids[l] == canvas_ids.image_pad){
            memcpy(hidden_states_2d[l].data(), vision_row, desc.D * sizeof(bf16));
            vision_row += desc.D;
        }
        else {
            this->embedding->forward(ids[l], hidden_states_2d[l]);
        }
    }

    // positions: MiniCPM's canvas m-rope when images are present, plain text positions otherwise
    std::vector<int> input_idx(L_in * 3);
    int next_position;
    if (has_images){
        next_position = mcvm::canvas_mrope(ids.data(), L_in, canvas_ids, views, general_token_idx, input_idx.data());
    }
    else {
        for (int l = 0; l < L_in; l++){
            input_idx[l * 3 + 0] = input_idx[l * 3 + 1] = input_idx[l * 3 + 2] = general_token_idx + l;
        }
        next_position = general_token_idx + L_in;
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
        pctx->forward(layer_idx, hidden_states, L_in, proj_weights[layer_idx], rms_weights[layer_idx], rope_rms_weights[layer_idx], kv_caches[layer_idx], input_idx, this->current_context_length);
    }

    buffer<bf16> predict = hidden_states_2d[L_in - 1];
    this->lm_head_weights.sync_to_device();
    get_logits(predict);
    this->set_context_length(this->current_context_length + L_in, next_position);

    return logits_valid;
}

/// @brief The vision tower, on the array.
/// @note  The attention runs on vision_attn.xclbin and the 115 matmuls on the two
///        vision_mm designs -- Qwen3.8-27B's bitstreams, reused unchanged. All three
///        are required: there is no host tower to fall back to, because a silent
///        fallback would encode an image ~5x slower and read as a working feature.
///        FLM_MINICPM_V_VISION=0 declines the tower as a FEATURE (text-only), which
///        is a different thing from choosing an implementation of it.
void minicpm_v_4_7_moe_npu::Impl::_load_vision(){
    vision_desc.build(config);
    if (!vision_desc.present){
        is_vlm = false;
        return;
    }
    if (const char* e = std::getenv("FLM_MINICPM_V_VISION"); e && e[0] == '0'){
        header_print("FLM", "FLM_MINICPM_V_VISION=0 -- no vision tower; images will be rejected");
        is_vlm = false;
        return;
    }
    if (!std::filesystem::exists(vision_desc.weight_path)){
        header_print("FLM", "config declares vision_model_weight but there is no " + vision_desc.weight_path +
                            " -- text-only, images will be rejected");
        is_vlm = false;
        return;
    }
    if (vision_desc.out_hidden_size != desc.D){
        throw std::runtime_error("vision out_hidden_size " + std::to_string(vision_desc.out_hidden_size) +
                                 " != language hidden_size " + std::to_string(desc.D));
    }
    canvas_ids.image_pad = config.get<int>("image_token_id", image_soft_token_id);

    const std::string dir = utils::path_join(config.exec_path, "xclbins", config.model_name);
    const std::string attn = utils::path_join(dir, "vision_attn.xclbin");
    const std::string mm_a = utils::path_join(dir, "vision_mm_a.xclbin");
    const std::string mm_b = utils::path_join(dir, "vision_mm_b.xclbin");
    for (const std::string* p : {&attn, &mm_a, &mm_b}){
        if (!std::filesystem::exists(*p)){
            throw std::runtime_error("no " + *p + " -- the vision tower needs vision_attn.xclbin and both "
                                     "vision_mm designs (Qwen3_8/vision_attn, Qwen3_8/vision_mm A and B)");
        }
    }
    time_utils::time_point t0 = time_utils::now();
    // FLM_MINICPM_V_VISION_BACKEND=cpu is a DIAGNOSTIC arm, not a configuration:
    // the host tower is the reference's arithmetic (exact in fp32) and ~7 s an
    // image. It exists so an end-to-end answer can be A/B'd against the array
    // with the language model held fixed -- the only way to attribute a changed
    // answer to the tower's precision rather than to the Q4 language model.
    const char* be = std::getenv("FLM_MINICPM_V_VISION_BACKEND");
    if (be && std::string(be) == "cpu"){
        header_print("WARNING", "FLM_MINICPM_V_VISION_BACKEND=cpu -- the vision tower runs on the HOST "
                                "(diagnostic only, several seconds per image)");
        this->image_mod = std::make_unique<mcvm::vision_tower>(vision_desc,
                                                              std::make_unique<mcvm::vision::cpu_op_factory>());
    }
    else {
        auto factory = std::make_unique<mcvm_npu::vision_op_factory>(npu->register_xclbin(attn));
        factory->enable_mm(npu->register_xclbin(mm_a), npu->register_xclbin(mm_b));
        this->image_mod = std::make_unique<mcvm::vision_tower>(vision_desc, std::move(factory));
    }
    // header_print("FLM", "vision tower loaded (" + std::to_string(vision_desc.depth) + " blocks, " +
    //                     std::string(image_mod->backend()) + ") in " +
    //                     std::to_string((int)time_utils::duration_ms(t0, time_utils::now()).first) + " ms");
}


std::vector<bf16> minicpm_v_4_7_moe_npu::Impl::_encode_images(const minicpm_v_moe_image_payload_t* p, int n_image_tokens,
                                                      std::vector<mcvm::view_grid>& views){
    if (!image_mod){
        throw std::runtime_error("images supplied but no vision tower is loaded -- either the config has no "
                                 "vision_model_weight or FLM_MINICPM_V_VISION=0");
    }
    const size_t patch_dim = vision_desc.patch_dim();
    size_t patches = 0;
    int rows = 0;
    for (const minicpm_v_moe_image_t& im : p->images){
        views.push_back(mcvm::view_grid{im.grid_h, im.grid_w});
        patches += (size_t)im.grid_h * im.grid_w;
        rows += image_mod->tokens_for(views.back());
    }
    if (rows != n_image_tokens){
        throw std::runtime_error("the prompt has " + std::to_string(n_image_tokens) + " image tokens but its " +
                                 std::to_string(views.size()) + " views produce " + std::to_string(rows));
    }
    if (p->_data__processed.size() != patches * patch_dim){
        throw std::runtime_error("image payload holds " + std::to_string(p->_data__processed.size()) +
                                 " values, its views need " + std::to_string(patches * patch_dim));
    }

    time_utils::time_point t0 = time_utils::now();
    std::vector<float> pixels(p->_data__processed.size());
    mcvm::simd::bf16_to_f32(pixels.data(), p->_data__processed.data(), pixels.size());
    std::vector<float> out((size_t)rows * desc.D);
    image_mod->encode(pixels.data(), views, out.data());

    std::vector<bf16> image_rows(out.size());
    mcvm::simd::f32_to_bf16(image_rows.data(), out.data(), out.size());
    if (std::getenv("FLM_MINICPM_V_VISION_PROFILE")){
        header_print_r("FLM", "vision: " + std::to_string(views.size()) + " views, " + std::to_string(patches) +
                              " patches -> " + std::to_string(rows) + " rows in " +
                              std::to_string((int)time_utils::duration_ms(t0, time_utils::now()).first) + " ms (" +
                              image_mod->backend() + ")");
    }
    return image_rows;
}

buffer<bf16> minicpm_v_4_7_moe_npu::Impl::get_k_cache(int layer_idx, int idx){
    this->kv_caches[layer_idx].sync_from_device();
    buffer<bf16> k_cache(desc.DK);
    bf16* k_cache_ptr = this->kv_caches[layer_idx].data();
    uint32_t offset = idx * desc.DK;
    memcpy(static_cast<void*>(k_cache.data()), static_cast<void*>(k_cache_ptr + offset), desc.DK * sizeof(bf16));

    return k_cache;
}

buffer<bf16> minicpm_v_4_7_moe_npu::Impl::get_v_cache(int layer_idx, int idx){
    this->kv_caches[layer_idx].sync_from_device();
    buffer<bf16> v_cache(desc.DV);
    bf16* v_cache_ptr = this->kv_caches[layer_idx].data();
    uint32_t offset = idx * desc.DV;
    memcpy(static_cast<void*>(v_cache.data()), static_cast<void*>(v_cache_ptr + offset), desc.DV * sizeof(bf16));
    return v_cache;
}

buffer<bf16> minicpm_v_4_7_moe_npu::Impl::get_logits(buffer<bf16>& predict){
    memcpy(this->x.data(), predict.data(), desc.D * sizeof(bf16));
    this->x.sync_to_device();
    this->lm_head_run.start();
    flm::npu::wait_for_completion(this->lm_head_run, "lm_head_run");
    this->logits.sync_from_device();
    return this->logits_valid;
}

size_t minicpm_v_4_7_moe_npu::Impl::state_row_bytes() {
    size_t attention_layers = 0;
    for (uint32_t i = 0; i < desc.num_hidden_layers; i++) {
        attention_layers += is_attn_layer(i) ? 1 : 0;
    }
    return attention_layers * (static_cast<size_t>(desc.DK) + desc.DV) * sizeof(bf16);
}

size_t minicpm_v_4_7_moe_npu::Impl::state_fixed_bytes() {
    size_t bytes = 0;
    for (uint32_t i = 0; i < desc.num_hidden_layers; i++) {
        if (!is_attn_layer(i)) {
            bytes += this->kv_caches[i].size() * sizeof(bf16);
        }
    }
    return bytes;
}

std::unique_ptr<hybrid_engine_state> minicpm_v_4_7_moe_npu::Impl::new_state() {
    return std::make_unique<hybrid_engine_state>(state_fixed_bytes(), state_row_bytes());
}

int minicpm_v_4_7_moe_npu::Impl::read_state(hybrid_engine_state& base, int kv_from) {
    auto* st = &base;
    if (st == nullptr) {
        return -1;
    }
    const int length = this->current_context_length;
    kv_from = std::clamp(kv_from, 0, length);
    st->state.resize(desc.num_hidden_layers);
    st->kv_k.resize(desc.num_hidden_layers);
    st->kv_v.resize(desc.num_hidden_layers);
    header_print_r("FLM", "Reading state at context length " + std::to_string(length) + ", K/V rows from " + std::to_string(kv_from));
    const size_t rows = static_cast<size_t>(length - kv_from);
    for (int i = 0; i < desc.num_hidden_layers; i++) {
        this->kv_caches[i].sync_from_device();
        if (!is_attn_layer(i)) {
            if (st->state[i].size() != this->kv_caches[i].size()) st->state[i] = buffer<bf16>(this->kv_caches[i].size());
            memcpy(st->state[i].data(), this->kv_caches[i].data(), this->kv_caches[i].size() * sizeof(bf16));
        }
        else if (rows > 0) {
            // K rows then V rows, one DK / DV row per position; the V half starts at the middle of the buffer
            if (st->kv_k[i].size() != rows * desc.DK) st->kv_k[i] = buffer<bf16>(rows * desc.DK);
            if (st->kv_v[i].size() != rows * desc.DV) st->kv_v[i] = buffer<bf16>(rows * desc.DV);
            memcpy(st->kv_k[i].data(), this->kv_caches[i].data() + static_cast<size_t>(kv_from) * desc.DK, rows * desc.DK * sizeof(bf16));
            memcpy(st->kv_v[i].data(), this->kv_caches[i].data() + (this->kv_caches[i].size() / 2) + static_cast<size_t>(kv_from) * desc.DV, rows * desc.DV * sizeof(bf16));
        }
        else {
            // nothing to keep: a reused state must not hold rows of its previous contents
            st->kv_k[i] = buffer<bf16>();
            st->kv_v[i] = buffer<bf16>();
        }
        if (!is_attn_layer(i)) {
            this->kv_caches[i].sync_to_device();
        }
    }
    st->filled = true;
    st->context_length = length;
    st->general_token_idx = this->general_token_idx;
    st->kv_from = kv_from;
    return length;
}

/// @brief whether `st` holds what this engine's write_state needs (a state of another engine would not)
static bool minicpm_v_4_7_moe_state_fits(const hybrid_engine_state* st, const std::vector<buffer<bf16>>& kv_caches, const minicpm_v_4_7_moe_desc& desc, int full_attention_interval) {
    if (st == nullptr || !st->filled || st->state.size() != desc.num_hidden_layers) {
        return false;
    }
    const size_t rows = static_cast<size_t>(st->context_length - st->kv_from);
    for (int i = 0; i < desc.num_hidden_layers; i++) {
        const bool attn = (i % full_attention_interval) == (full_attention_interval - 1);
        if (attn ? (rows > 0 && (st->kv_k[i].size() != rows * desc.DK || st->kv_v[i].size() != rows * desc.DV))
                 : st->state[i].size() != kv_caches[i].size()) {
            return false;
        }
    }
    return true;
}

int minicpm_v_4_7_moe_npu::Impl::write_state_kv(const hybrid_engine_state& base) {
    const auto* st = &base;
    if (!minicpm_v_4_7_moe_state_fits(st, this->kv_caches, desc, full_attention_interval)) {
        return -1;
    }
    const size_t rows = static_cast<size_t>(st->context_length - st->kv_from);
    if (rows == 0) {
        return st->context_length;
    }
    for (int i = 0; i < desc.num_hidden_layers; i++) {
        if (is_attn_layer(i)) {
            this->kv_caches[i].sync_from_device();
            memcpy(this->kv_caches[i].data() + static_cast<size_t>(st->kv_from) * desc.DK, st->kv_k[i].data(), rows * desc.DK * sizeof(bf16));
            memcpy(this->kv_caches[i].data() + (this->kv_caches[i].size() / 2) + static_cast<size_t>(st->kv_from) * desc.DV, st->kv_v[i].data(), rows * desc.DV * sizeof(bf16));
            this->kv_caches[i].sync_to_device();
        }
    }
    return st->context_length;
}

int minicpm_v_4_7_moe_npu::Impl::write_state(const hybrid_engine_state& base) {
    const auto* st = &base;
    if (!minicpm_v_4_7_moe_state_fits(st, this->kv_caches, desc, full_attention_interval)) {
        header_print_r("FLM", "No valid state to write");
        return -1;
    }
    header_print_r("FLM", "Writing state at context length " + std::to_string(st->context_length));
    const size_t rows = static_cast<size_t>(st->context_length - st->kv_from);
    for (int i = 0; i < desc.num_hidden_layers; i++) {
        this->kv_caches[i].sync_from_device();
        if (!is_attn_layer(i)) {
            memcpy(this->kv_caches[i].data(), st->state[i].data(), st->state[i].size() * sizeof(bf16));
        }
        else {
            if (rows > 0) {
                memcpy(this->kv_caches[i].data() + static_cast<size_t>(st->kv_from) * desc.DK, st->kv_k[i].data(), rows * desc.DK * sizeof(bf16));
                memcpy(this->kv_caches[i].data() + (this->kv_caches[i].size() / 2) + static_cast<size_t>(st->kv_from) * desc.DV, st->kv_v[i].data(), rows * desc.DV * sizeof(bf16));
            }
            // Rows past the restored length hold whatever the requests since the
            // state was read left there. The kernels do not rely on the length alone to
            // keep them out of attention, so put back the zeros a fresh context has.
            const size_t from = static_cast<size_t>(st->context_length);
            const size_t half = this->kv_caches[i].size() / 2;
            memset(this->kv_caches[i].data() + from * desc.DK, 0, (half - from * desc.DK) * sizeof(bf16));
            memset(this->kv_caches[i].data() + half + from * desc.DV, 0, (half - from * desc.DV) * sizeof(bf16));
        }
        this->kv_caches[i].sync_to_device();
    }
    this->current_context_length = st->context_length;
    this->general_token_idx = st->general_token_idx;
    this->set_context_length(this->current_context_length, this->general_token_idx);
    return this->current_context_length;
}

void minicpm_v_4_7_moe_npu::Impl::clear_context(){
    // Checkpoint slots survive a clear: each holds its own copy of the recurrent state and of
    // the K/V rows it needs, so zeroing the live cache does not touch them. The caller must only
    // restore a slot that keeps rows from position 0 (kv_from 0), or put the rows before its
    // kv_from back first (restore_slot_kv of the slot that holds them).
    this->set_context_length(0, 0);
    for (int i = 0; i < desc.num_hidden_layers; i++){
        this->kv_caches[i].sync_from_device();
        memset(this->kv_caches[i].data(), 0, this->kv_caches[i].size() * sizeof(bf16));
        this->kv_caches[i].sync_to_device();
    }
}

int minicpm_v_4_7_moe_npu::Impl::get_current_context_length(){
    return this->current_context_length;
}

void minicpm_v_4_7_moe_npu::Impl::update_max_length(uint32_t MAX_L){
    if (MAX_L <= this->MAX_L){
        header_print("FLM", "New length is shorter than the current length, no need to update!");
        return; // no need to update
	}
    this->MAX_L = MAX_L;
    const size_t kv_cache_size = desc.get_kv_cache_size(MAX_L);
    this->sequence->set_max_length(MAX_L);
    this->V_offset = this->sequence->get_V_offset();
    for (uint32_t i = 0; i < desc.num_hidden_layers; i++){
        if (!is_attn_layer(i)){
            continue; // nothing to do with linear layer
        }
        // sizes come from this layer's own buffer: layer 0 is a linear layer with a differently sized state,
        // and the old V half starts at the middle of the old buffer
        this->kv_caches[i].sync_from_device();
        const size_t old_half = this->kv_caches[i].size() / 2;
        buffer<bf16> kv_cache_temp(this->kv_caches[i].size());
        memcpy(static_cast<void*>(kv_cache_temp.data()), static_cast<void*>(this->kv_caches[i].data()), this->kv_caches[i].size() * sizeof(bf16));
        this->kv_caches[i] = layer.create_bo_buffer<bf16>(kv_cache_size);
        this->kv_caches[i].memset((bf16)0);
        const size_t new_half = this->kv_caches[i].size() / 2;
        memcpy(static_cast<void*>(this->kv_caches[i].data()), static_cast<void*>(kv_cache_temp.data()), old_half * sizeof(bf16));
        memcpy(static_cast<void*>(this->kv_caches[i].data() + new_half), static_cast<void*>(kv_cache_temp.data() + old_half), old_half * sizeof(bf16));
        this->kv_caches[i].sync_to_device();
    }
    this->set_context_length(this->current_context_length, this->general_token_idx); // regenerate sequence
    this->legacy_checkpoint.reset(); // the legacy copy was taken against the old buffers
}
///@brief destructor of qwen3vl_npu
minicpm_v_4_7_moe_npu::Impl::~Impl() = default;

// ===============================================
// Externals for qwen3vl_npu
// ===============================================
minicpm_v_4_7_moe_npu::minicpm_v_4_7_moe_npu(LM_Config config, npu_xclbin_manager *npu_instance, int MAX_L){

    this->load_vision_preprocess_parameters(config);
    this->_impl = new Impl(config, npu_instance, this,   MAX_L);
}

buffer<bf16> minicpm_v_4_7_moe_npu::forward(int ids){
    return this->_impl->forward(ids);
}

buffer<bf16> minicpm_v_4_7_moe_npu::prefill(std::vector<int>& ids, void* payload){
    try {
        buffer<bf16> result = this->_impl->prefill(ids, payload);
        return result;
    }
    catch (const std::runtime_error& e) {
        header_print("FLM", e.what());
        throw;
    }
}

void minicpm_v_4_7_moe_npu::set_context_length(int L){
    std::cout << "Setting context length is not supported for qwen3vl_npu!" << std::endl;
}

void minicpm_v_4_7_moe_npu::load_weights(Q4NX& q4nx){
    this->_impl->load_weights(q4nx);
}

void minicpm_v_4_7_moe_npu::update_max_length(uint32_t MAX_L){
    this->_impl->update_max_length(MAX_L);
}

void minicpm_v_4_7_moe_npu::clear_context(){
    this->_impl->clear_context();
}

buffer<bf16> minicpm_v_4_7_moe_npu::get_k_cache(int layer_idx, int idx){
    return this->_impl->get_k_cache(layer_idx, idx);
}

buffer<bf16> minicpm_v_4_7_moe_npu::get_v_cache(int layer_idx, int idx){
    return this->_impl->get_v_cache(layer_idx, idx);
}

int minicpm_v_4_7_moe_npu::get_current_context_length(){
    return this->_impl->get_current_context_length();
}

int minicpm_v_4_7_moe_npu::checkpoint(){
    // the single legacy copy keeps no K/V rows: its caller continues from the same cache
    if (!this->_impl->legacy_checkpoint) {
        this->_impl->legacy_checkpoint = std::make_unique<hybrid_engine_state>(this->_impl->state_fixed_bytes(), this->_impl->state_row_bytes());
    }
    return this->_impl->read_state(*this->_impl->legacy_checkpoint, std::numeric_limits<int>::max());
}

int minicpm_v_4_7_moe_npu::restore(){
    if (!this->_impl->legacy_checkpoint) {
        return -1;
    }
    return this->_impl->write_state(*this->_impl->legacy_checkpoint);
}

std::unique_ptr<hybrid_engine_state> minicpm_v_4_7_moe_npu::new_state(){
    return this->_impl->new_state();
}

int minicpm_v_4_7_moe_npu::read_state(hybrid_engine_state& state, int kv_from){
    return this->_impl->read_state(state, kv_from);
}

int minicpm_v_4_7_moe_npu::write_state(const hybrid_engine_state& state){
    return this->_impl->write_state(state);
}

int minicpm_v_4_7_moe_npu::write_state_kv(const hybrid_engine_state& state){
    return this->_impl->write_state_kv(state);
}

minicpm_v_4_7_moe_npu::~minicpm_v_4_7_moe_npu(){
    delete this->_impl;
}

extern "C" const flm_state_api* minicpm_v_4_7_moe_state_api(void* engine){
    return hybrid_state_api<minicpm_v_4_7_moe_npu>(engine);
}
