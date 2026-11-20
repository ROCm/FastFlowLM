// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#include <chrono>
#include "minicpm_ocr_npu_detail.hpp"
#include "npu_utils/runtime_wait.hpp"
#include "image/vision_common.hpp"
#include "image/vision_tower.hpp"
#include "image/npu/vision_op_factory.hpp"
#include <filesystem>
#include "lm_head_in_layer.hpp"
minicpm_ocr_npu::Impl::Impl(LM_Config config, npu_xclbin_manager *npu_instance, minicpm_ocr_npu* parent_ptr, int MAX_L )
    : config(config), npu(npu_instance), parent_ptr(parent_ptr){
    is_vlm = config.get<bool>("is_vlm", false);
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

    this->sequence = std::make_unique<minicpm_ocr_npu_sequence>(this->desc, config, MAX_L);
    this->V_offset = this->sequence->get_V_offset();

    // The descriptors that never go through a weight_container: q_norm / k_norm sit
    // behind the rope table, and the bf16 alpha/beta twins live in host buffers. Without
    // indp() their ready() stays false however they are loaded (ADDING_A_NEW_MODEL.md S1.2).
    {
        auto& attn = static_cast<minicpm_ocr_attn_layer&>(*desc.weight_desc[desc.attn_idx]);
        auto& lin  = static_cast<minicpm_ocr_linear_layer&>(*desc.weight_desc[desc.lin_idx]);
        attn.attn_q_norm.indp();
        attn.attn_k_norm.indp();
        lin.ssm_alpha_cpu.indp();
        lin.ssm_beta_cpu.indp();
    }

    // layer.xclbin and lm_head.xclbin belong to decode; with DECODE=cpu neither is
    // registered and a token runs through the prefill graph at L = 1 plus a host lm_head.
    // layer_with_lm_head.xclbin, if present, replaces both (README "lm_head inside layer.xclbin");
    // FLM_MINICPM_OCR_LM_HEAD_IN_LAYER=0 keeps the two-xclbin path.
#if MINICPM_OCR_DECODE_ON_NPU
    {
        const std::string dir = utils::path_join(config.exec_path, "xclbins", config.model_name);
        const std::string with_lm_head = utils::path_join(dir, "layer_with_lm_head.xclbin");
        const std::string with_kvb = utils::path_join(dir, "layer_with_lm_head_kvb.xclbin");
        const char* env = std::getenv("FLM_MINICPM_OCR_LM_HEAD_IN_LAYER");
        const char* kvb_env = std::getenv("FLM_MINICPM_OCR_DECODE_KV_BFP16");
        const bool lm_head_allowed = !(env && env[0] == '0');
        // layer_with_lm_head_kvb.xclbin is layer_with_lm_head.xclbin with bfp16 K / V reads; preferred when present
        decode_kv_bfp16 = lm_head_allowed && std::filesystem::exists(with_kvb) && !(kvb_env && kvb_env[0] == '0');
        lm_head_in_layer = lm_head_allowed && (decode_kv_bfp16 || std::filesystem::exists(with_lm_head));
        sequence->write_modes = lm_head_in_layer;
        sequence->kv_bfp16 = decode_kv_bfp16;
        // README "Rescale skip": on unless FLM_MINICPM_OCR_DECODE_RESCALE_SKIP=0
        const char* skip_env = std::getenv("FLM_MINICPM_OCR_DECODE_RESCALE_SKIP");
        sequence->kv_bfp16_rescale_skip = !(skip_env && skip_env[0] == '0');
        layer_app_manager = npu->register_xclbin(decode_kv_bfp16 ? with_kvb
                                                 : lm_head_in_layer ? with_lm_head : utils::path_join(dir, "layer.xclbin"));
    }

    this->layer = layer_app_manager->create_app();
    this->linear_layer = layer_app_manager->create_app();
    this->first_linear_layer = layer_app_manager->create_app();
    this->set_layer_rtp = layer_app_manager->create_app();
    if (!this->npu->is_preemption_enabled()){
        this->layers_run = layer_app_manager->create_runlist();
    }

    lm_head_app_manager = lm_head_in_layer ? layer_app_manager
                                           : npu->register_xclbin(utils::path_join(config.exec_path, "xclbins", config.model_name, "lm_head.xclbin"));
    this->lm_head = lm_head_app_manager->create_app();
#endif

    // allocate all buffers. The weights, caches, x and logits are device BOs in every
    // configuration: create_bo_buffer needs no registered xclbin, and
    // minicpm_ocr_desc::load_*_weights() sync them unconditionally. So even the all-host
    // build opens the device (but registers no xclbin and opens no hw context).
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

    for (size_t i = 0; i < desc.num_hidden_layers; i++){
        if (is_attn_layer(i)){
            this->proj_weights[i] = npu->create_bo_buffer<u8>(attn_total_proj_weights_byte_size);
            this->rms_weights[i] = npu->create_bo_buffer<bf16>(attn_rms_weights_byte_size / sizeof(bf16));
            this->rope_rms_weights[i] = npu->create_bo_buffer<bf16>(desc.RoPE_DIM + desc.DH * 2);
            this->kv_caches[i] = npu->create_bo_buffer<bf16>(kv_cache_size);
            this->kv_caches[i].memset((bf16)0);
            this->kv_caches[i].sync_to_device();
        }
        else{
            this->proj_weights[i] = npu->create_bo_buffer<u8>(linear_total_proj_weights_byte_size);
            this->rms_weights[i] = npu->create_bo_buffer<bf16>(linear_rms_weights_byte_size / sizeof(bf16));
            this->rope_rms_weights[i] = npu->create_bo_buffer<bf16>(linear_ssm_weights_byte_size / sizeof(bf16));
            this->kv_caches[i] = npu->create_bo_buffer<bf16>(linear_kv_cache_byte_size / sizeof(bf16));
            this->kv_caches[i].memset((bf16)0);
            this->kv_caches[i].sync_to_device();
            this->alpha_proj_cpu[i] = buffer<bf16>((size_t)desc.D * desc.linear_num_v_heads);
            this->beta_proj_cpu[i] = buffer<bf16>((size_t)desc.D * desc.linear_num_v_heads);
        }
    }

    if (decode_kv_bfp16){
        std::vector<int> attn_layers;
        for (uint32_t i = 0; i < desc.num_hidden_layers; i++){
            if (is_attn_layer(i)){
                attn_layers.push_back((int)i);
            }
        }
        decode_kv.init(npu, desc.num_hidden_layers, attn_layers, desc.num_kv_heads, desc.DH);
    }

    is_checkpoint_valid = false; // checkpoint is not valid until we load kv cache to device after resizing buffers
    x = npu->create_bo_buffer<bf16>(desc.D * 2);
    lm_head_weights = npu->create_bo_buffer<u8>(lm_head_in_layer ? minicpm_ocr_lm_head::weight_bytes(desc) : desc.get_lm_head_w_size());
    // reorder_cpy_lm_head fills only the bands the checkpoint has; zero the rest so the
    // host lm_head never reads uninitialised scales
    memset(lm_head_weights.data(), 0, lm_head_weights.size());
    lm_head_norm = npu->create_bo_buffer<bf16>(desc.D);
    assert(minicpm_ocr_lm_head::rows(desc) <= (int)desc.vocab_size_padded);
    logits = npu->create_bo_buffer<bf16>(desc.vocab_size_padded);
    logits_valid = buffer<bf16>(logits.data(), desc.vocab_size);
    decode_hidden = buffer<bf16>(desc.D);

    this->clear_context();
    // the embedding table is created in load_weights, once we know whether the
    // checkpoint stores model.embed_tokens or only the tied q8_0 lm_head.

    pctx = std::make_unique<minicpm_ocr_prefill_context>(npu, &desc, sequence.get(), config);
    pctx->attn_block->kv_cache_elems = desc.get_kv_cache_size(this->MAX_L);

#if MINICPM_OCR_DECODE_ON_NPU
    lm_head = this->lm_head_app_manager->create_app();
    if (lm_head_in_layer){
        this->sequence->gen_lm_head_in_layer_seq(lm_head.seq());
        this->lm_head_run = lm_head.create_run(this->x, this->lm_head_weights, this->lm_head_norm, this->lm_head_norm, this->logits);
    }
    else {
        this->sequence->gen_lm_head_seq(lm_head.seq(), desc.vocab_size_padded, desc.D);
        this->lm_head_run = lm_head.create_run(this->logits, this->lm_head_weights, this->x);
    }

    this->sequence->gen_layer_seq(this->linear_layer.seq(), 0, false, false);
    this->sequence->gen_layer_seq(this->first_linear_layer.seq(), 0, false, true);
#endif

    set_context_length(0, 0);

    // Last, so the language model's xclbins and buffers are all in place: the
    // tower uploads its 1.1 GB of weights into BOs here, at load, not per image.
    if (is_vlm){
        this->_load_vision();
    }
}

/// @brief The vision tower, on the array.
/// @note  The attention runs on vision_attn.xclbin and the 115 matmuls on the two
///        vision_mm designs -- Qwen3.8-27B's bitstreams, reused unchanged. All three
///        are required: there is no host tower to fall back to, because a silent
///        fallback would encode an image ~5x slower and read as a working feature.
///        FLM_MINICPM_OCR_VISION=0 declines the tower as a FEATURE (text-only), which
///        is a different thing from choosing an implementation of it.
void minicpm_ocr_npu::Impl::_load_vision(){
    vision_desc.build(config);
    if (!vision_desc.present){
        is_vlm = false;
        return;
    }
    if (const char* e = std::getenv("FLM_MINICPM_OCR_VISION"); e && e[0] == '0'){
        header_print("FLM", "FLM_MINICPM_OCR_VISION=0 -- no vision tower; images will be rejected");
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
    // vision_overlay.xclbin (FLM_Xclbin/minicpm_ocr/vision_overlay) puts the attention and the
    // design-A matmuls on one xclbin; vision_mm_b.xclbin is still needed. FLM_MINICPM_OCR_VISION_OVERLAY=0
    // keeps the three-xclbin path.
    const std::string overlay = utils::path_join(dir, "vision_overlay.xclbin");
    const char* ov = std::getenv("FLM_MINICPM_OCR_VISION_OVERLAY");
    const bool use_overlay = std::filesystem::exists(overlay) && !(ov && ov[0] == '0') &&
                             std::filesystem::exists(mm_b);
    for (const std::string* p : {&attn, &mm_a, &mm_b}){
        if (use_overlay && p != &mm_b) continue;
        if (!std::filesystem::exists(*p)){
            throw std::runtime_error("no " + *p + " -- the vision tower needs vision_attn.xclbin and both "
                                     "vision_mm designs (Qwen3_8/vision_attn, Qwen3_8/vision_mm A and B)");
        }
    }
    time_utils::time_point t0 = time_utils::now();
    // FLM_MINICPM_OCR_VISION_BACKEND=cpu is a DIAGNOSTIC arm, not a configuration:
    // the host tower is the reference's arithmetic (exact in fp32) and ~7 s an
    // image. It exists so an end-to-end answer can be A/B'd against the array
    // with the language model held fixed -- the only way to attribute a changed
    // answer to the tower's precision rather than to the Q4 language model.
    const char* be = std::getenv("FLM_MINICPM_OCR_VISION_BACKEND");
    if (be && std::string(be) == "cpu"){
        header_print("WARNING", "FLM_MINICPM_OCR_VISION_BACKEND=cpu -- the vision tower runs on the HOST "
                                "(diagnostic only, several seconds per image)");
        this->image_mod = std::make_unique<mco::vision_tower>(vision_desc,
                                                              std::make_unique<mco::vision::cpu_op_factory>());
    }
    else if (use_overlay) {
        auto factory = std::make_unique<mco_npu::vision_op_factory>(npu->register_xclbin(overlay));
        factory->enable_overlay(npu->register_xclbin(mm_b));
        this->image_mod = std::make_unique<mco::vision_tower>(vision_desc, std::move(factory));
    }
    else {
        auto factory = std::make_unique<mco_npu::vision_op_factory>(npu->register_xclbin(attn));
        factory->enable_mm(npu->register_xclbin(mm_a), npu->register_xclbin(mm_b));
        this->image_mod = std::make_unique<mco::vision_tower>(vision_desc, std::move(factory));
    }
    // header_print("FLM", "vision tower loaded (" + std::to_string(vision_desc.depth) + " blocks, " +
    //                     std::string(image_mod->backend()) + ") in " +
    //                     std::to_string((int)time_utils::duration_ms(t0, time_utils::now()).first) + " ms");
}

void minicpm_ocr_npu::Impl::load_weights(Q4NX& q4nx){
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
        memcpy(this->lm_head_norm.data(), final_rms_norm_weight.data(), desc.D * sizeof(bf16));
        this->lm_head_norm.sync_to_device();
    }
    {
        buffer<u8> lm_head_w;
        q4nx.load_weights(lm_head_w, "lm_head.weight");

        if (lm_head_in_layer){
            minicpm_ocr_lm_head::reorder_cpy(desc, this->lm_head_weights.data(), lm_head_w);
        }
        else {
            desc.reorder_cpy_lm_head(this->lm_head_weights.data(), lm_head_w, desc.D, flm_q80, 4);
        }
        this->lm_head_weights.sync_to_device();
    }

    this->_check_descriptors(q4nx);
    this->clear_context();
}

/// @brief the "did I forget a tensor" check (ADDING_A_NEW_MODEL.md S1.4), run after every load
/// @note  minicpm_ocr_desc keeps one representative descriptor per layer kind (lin_idx,
///        attn_idx) and the loaders mark only those, so ready() is checked there, and every
///        layer's own tensors are checked against model.q4nx by name and byte size. Throws
///        on any failure; the table prints at DEBUG_LEVEL >= 1.
void minicpm_ocr_npu::Impl::_check_descriptors(Q4NX& q4nx){
    auto& attn = static_cast<minicpm_ocr_attn_layer&>(*desc.weight_desc[desc.attn_idx]);
    auto& lin  = static_cast<minicpm_ocr_linear_layer&>(*desc.weight_desc[desc.lin_idx]);
    // all() leaves out the host-only bf16 twins; attn_gate aliases q_proj
    std::vector<weight_desc_t*> attn_all = attn.all();
    std::vector<weight_desc_t*> lin_all  = lin.all();
    lin_all.push_back(&lin.ssm_alpha_cpu);
    lin_all.push_back(&lin.ssm_beta_cpu);

    std::vector<std::string> failures;
    DEBUG_BLOCK(1,
        std::cout << "================ minicpm_ocr descriptors ================" << std::endl;
        std::cout << std::left << std::setw(56) << "descriptor (representative layer)" << std::setw(8) << "dtype"
                  << std::setw(12) << "bytes" << std::setw(8) << "added" << std::setw(8) << "loaded" << "ready" << std::endl;
    )
    for (int kind = 0; kind < 2; kind++){
        const int rep = kind ? desc.attn_idx : desc.lin_idx;
        for (weight_desc_t* d : (kind ? attn_all : lin_all)){
            DEBUG_BLOCK(1,
                std::cout << std::left << std::setw(56) << d->format_name(rep) << std::setw(8) << (int)d->dtype
                          << std::setw(12) << d->get_size() << std::setw(8) << d->added << std::setw(8) << d->loaded
                          << (d->ready() ? "yes" : "NO") << std::endl;
            )
            if (!d->ready()){
                failures.push_back(d->format_name(rep) + ": ready() is false");
            }
            for (uint32_t i = 0; i < desc.num_hidden_layers; i++){
                if (desc.is_attn_layer(i) != (kind == 1)) continue;
                const std::string name = d->format_name(i);
                if (!q4nx.has_tensor(name)){
                    failures.push_back(name + ": missing from model.q4nx");
                    continue;
                }
                // q_proj holds q and the gate (both descriptors alias it); ssm_a / ssm_dt
                // are 16 floats under a 32-padded descriptor
                const size_t have = q4nx.get_tensor_metadata(name).byte_size;
                size_t want = d->get_size();
                if (d == &attn.attn_q || d == &attn.attn_gate) want *= 2;
                if (d == &lin.ssm_a || d == &lin.ssm_dt) want = desc.linear_num_v_heads * sizeof(float);
                if (have != want){
                    failures.push_back(name + ": " + std::to_string(have) + " B in model.q4nx, descriptor expects " + std::to_string(want));
                }
            }
        }
    }
    DEBUG_BLOCK(1,
        std::cout << "checked " << lin_all.size() << " linear + " << attn_all.size() << " attention descriptors over "
                  << desc.num_hidden_layers << " layers: " << (failures.empty() ? "all ready" : "FAILURES") << std::endl;
        std::cout << "========================================================" << std::endl;
    )
    if (!failures.empty()){
        for (auto& f : failures) header_print("FLM", "descriptor check: " << f);
        throw std::runtime_error("minicpm_ocr_npu: " + std::to_string(failures.size()) + " weight descriptor check(s) failed");
    }
}

void minicpm_ocr_npu::Impl::_set_rope_rms_weights(int idx){
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
            if (decode_kv_bfp16){
                decode_kv.set_head(i, this->rope_rms_weights[i]);
            }
        }
    }
}

void minicpm_ocr_npu::Impl::set_context_length(int L, int new_general_token_idx){
    this->current_context_length = L;
    this->general_token_idx = new_general_token_idx;

#if MINICPM_OCR_DECODE_ON_NPU
    if (decode_kv_bfp16){
        this->sequence->kv_bfp16_cap = decode_kv.cap;
    }
    this->sequence->gen_set_context(this->set_layer_rtp.seq(), L + 1);
    this->sequence->gen_layer_seq(this->layer.seq(), L + 1, true);
    if (!this->npu->is_preemption_enabled()){
        this->layers_run.reset();

        this->layers_run.add(this->set_layer_rtp.create_run());
        for (uint32_t i = 0; i < desc.num_hidden_layers; i++){
            if (is_attn_layer(i) && decode_kv_bfp16){
                this->layers_run.add(this->layer.create_run(this->x, this->proj_weights[i], this->rms_weights[i], this->decode_kv.bos[i], this->kv_caches[i]));
            }
            else if (is_attn_layer(i)){
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
#endif
}

/// @note  The host fallback (MINICPM_OCR_DECODE_ON_NPU == 0) runs the prefill graph at
///        L = 1, so its conv state, recurrent state and kv cache are the same buffers in
///        the same layout the array decode reads. It is the golden reference for S4.
buffer<bf16> minicpm_ocr_npu::Impl::forward(int ids){
#if MINICPM_OCR_DECODE_ON_NPU
#if MINICPM_OCR_DECODE_PROFILE
    static double t_embed = 0, t_layers = 0, t_lmstart = 0, t_setctx = 0, t_lmwait = 0, t_sync = 0;
    static int n_tok = 0;
    auto _t = std::chrono::steady_clock::now();
    #define MINICPM_OCR_TICK(acc) { acc += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - _t).count(); \
                                  _t = std::chrono::steady_clock::now(); }
#else
    #define MINICPM_OCR_TICK(acc) do {} while (0)
#endif
    this->embed_forward(ids, this->x);
    this->x.sync_to_device();
    // after a prefill, restore or state write: convert the chunks the copies lack (and remake the runs if they grew)
    if (decode_kv_bfp16 && this->_decode_kv_catch_up(this->current_context_length)){
        this->set_context_length(this->current_context_length, this->general_token_idx);
    }
    MINICPM_OCR_TICK(t_embed);

    if (!this->npu->is_preemption_enabled()){
        this->layers_run.execute();
        flm::npu::wait_for_completion(this->layers_run, "layers_run");
    }
    else{
        set_layer_rtp();
        for (uint32_t i = 0; i < desc.num_hidden_layers; i++){
            if (is_attn_layer(i) && decode_kv_bfp16){
                this->layer(this->x, this->proj_weights[i], this->rms_weights[i], this->decode_kv.bos[i], this->kv_caches[i]);
            }
            else if (is_attn_layer(i)){
                this->layer(this->x, this->proj_weights[i], this->rms_weights[i], this->rope_rms_weights[i], this->kv_caches[i]);
            }
            else if (is_first_linear_layer(i)) {
                this->first_linear_layer(this->x, this->proj_weights[i], this->rms_weights[i], this->rope_rms_weights[i], this->kv_caches[i]);
            }
            else  {
                this->linear_layer(this->x, this->proj_weights[i], this->rms_weights[i], this->rope_rms_weights[i], this->kv_caches[i]);
            }
        }
    }

    MINICPM_OCR_TICK(t_layers);
    this->x.sync_to_device();
    this->lm_head_run.start();
    MINICPM_OCR_TICK(t_lmstart);
    // under the lm_head run: the chunk this token's row finished goes into the bfp16 copies before the next runs are made
    if (decode_kv_bfp16){
        this->_decode_kv_catch_up(this->current_context_length + 1);
    }
    this->set_context_length(this->current_context_length + 1, this->general_token_idx + 1);
    MINICPM_OCR_TICK(t_setctx);
    flm::npu::wait_for_completion(this->lm_head_run, "lm_head_run");
    MINICPM_OCR_TICK(t_lmwait);
    this->logits.sync_from_device();
#if MINICPM_OCR_DECODE_PROFILE
    MINICPM_OCR_TICK(t_sync);
    // set_ctx sits between lm_head's start() and wait(), so it is free until it exceeds
    // the lm_head run; lm_start + set_ctx + lm_wait is the lm_head's true cost.
    if (++n_tok % 32 == 0){
        fprintf(stderr, "[minicpm_ocr decode%s] %d tok, avg ms/token: embed %.3f  layers %.3f  lm_head.start %.3f"
                        "  set_context %.3f  lm_head.wait %.3f  logits sync %.3f  total %.3f\n",
                decode_kv_bfp16 ? (sequence->kv_bfp16_rescale_skip ? ", lm_head in layer, kv bfp16, rescale skip" : ", lm_head in layer, kv bfp16")
                                : lm_head_in_layer ? ", lm_head in layer" : "", n_tok, t_embed / n_tok, t_layers / n_tok, t_lmstart / n_tok, t_setctx / n_tok,
                t_lmwait / n_tok, t_sync / n_tok,
                (t_embed + t_layers + t_lmstart + t_setctx + t_lmwait + t_sync) / n_tok);
    }
#endif
#undef MINICPM_OCR_TICK
    return this->logits_valid;
#else
    this->embed_forward(ids, this->decode_hidden);
    std::vector<int> input_idx(3, this->general_token_idx);
    for (uint32_t layer_idx = 0; layer_idx < desc.num_hidden_layers; layer_idx++) {
        pctx->forward(layer_idx, this->decode_hidden, 1, proj_weights[layer_idx], rms_weights[layer_idx], rope_rms_weights[layer_idx], kv_caches[layer_idx], input_idx, alpha_proj_cpu[layer_idx], beta_proj_cpu[layer_idx], this->current_context_length);
    }
    this->get_logits(this->decode_hidden);
    this->set_context_length(this->current_context_length + 1, this->general_token_idx + 1);
    return this->logits_valid;
#endif
}

bool minicpm_ocr_npu::Impl::_decode_kv_catch_up(int L){
    // a run with L cached tokens reads chunks [0, L / 16) from the copies (the newest, unfinished one from the cache)
    return decode_kv.ensure(npu, L / minicpm_ocr_decode_kv::chunk_rows, this->kv_caches, this->V_offset, this->rope_rms_weights);
}

buffer<bf16> minicpm_ocr_npu::Impl::prefill(std::vector<int>& ids, void* payload){
    return this->_prefill_with_mm(ids, payload);
}

/// @note  `payload` is a minicpm_ocr_image_payload_t* or null. Its `images` are VIEWS,
///        not pictures: the app's slicer turns one picture into a source view plus up
///        to max_slice_nums slices, each with its own patch grid, and `_data__processed`
///        holds every view's patches back to back -- raster order within a view, each
///        patch flattened (c, kh, kw), 3*14*14 bf16. That is the HF processor's NaViT
///        `[3, 14, T*14]` tensor transposed per patch.
///
///        The tower's rows overwrite the image_token_id rows in order, bypassing the
///        embedding table (ADDING_A_NEW_MODEL.md S7). Positions are MiniCPM's canvas
///        m-rope (image/mrope.hpp) whenever images are present.
buffer<bf16> minicpm_ocr_npu::Impl::_prefill_with_mm(std::vector<int>& ids, void* payload){
    PREFILL_PROF_INIT();
    int L_in = ids.size();
    // prefill writes cache rows from current_context_length on; the bfp16 decode copies never hold those (no-op guard)
    decode_kv.cache_changed(this->current_context_length);
    const auto* p = static_cast<const minicpm_ocr_image_payload_t*>(payload);
    const bool has_images = p != nullptr && !p->images.empty();

    int n_image_tokens = 0;
    for (int id : ids){
        if (id == video_token_id){
            throw std::runtime_error("the prompt carries video tokens (" + std::to_string(video_token_id) +
                                     "): this engine has no video path");
        }
        n_image_tokens += (id == canvas_ids.image_pad);
    }

    std::vector<mco::view_grid> views;
    std::vector<bf16> image_rows;
    if (has_images){
        image_rows = this->_encode_images(p, n_image_tokens, views);
    }
    else if (n_image_tokens > 0){
        throw std::runtime_error("the prompt carries " + std::to_string(n_image_tokens) +
                                 " image tokens but no image payload");
    }

    buffer<bf16> hidden_states((size_t)desc.D * L_in);
    tensor_2d<bf16> hidden_states_2d(hidden_states, desc.D);
    const bf16* vision_row = image_rows.data();
    for (int l = 0; l < L_in; l++) {
        if (has_images && ids[l] == canvas_ids.image_pad){
            memcpy(hidden_states_2d[l].data(), vision_row, desc.D * sizeof(bf16));
            vision_row += desc.D;
        }
        else {
            this->embed_forward(ids[l], hidden_states_2d[l]);
        }
    }

    // Positions are plain sequential RoPE, images or not -- this is the ONE place where
    // MiniCPM-OCR genuinely differs from MiniCPM-V-4.7 in code rather than in a dimension.
    //
    // 4.7 calls mco::canvas_mrope() here whenever the prompt carries images, laying each image
    // group out on a (t, h, w) canvas. That machinery belongs to 4.7: upstream,
    // MiniCPMOcrForConditionalGeneration is MiniCPMV4_6ForConditionalGeneration with `pass`,
    // and it is MiniCPMV4_7 that adds get_rope_index / compute_3d_position_ids on top of 4.6.
    // MiniCPM-OCR never inherited them, and its rope_parameters carry no mrope_section.
    //
    // The three lanes are still filled identically rather than collapsed to one, because
    // everything downstream (the RoPE tables, the prefill sequence) indexes input_idx by
    // l * 3 + axis. Keeping the shape means this is the only diff.
    //
    // Do NOT "restore" the mRoPE branch for image prompts: the language model was trained on
    // whichever rule its own checkpoint uses, so using 4.7's here would be wrong in a way that
    // reads as a subtly worse model rather than as an error.
    std::vector<int> input_idx(L_in * 3);
    int next_position;
    for (int l = 0; l < L_in; l++){
        input_idx[l * 3 + 0] = input_idx[l * 3 + 1] = input_idx[l * 3 + 2] = general_token_idx + l;
    }
    next_position = general_token_idx + L_in;

    // FLM_DUMP_DIR scoring hook, DEBUG_LEVEL >= 1 only; decode never dumps
    minicpm_ocr_cpu::dumper().active = true;
    MINICPM_OCR_DUMP("embed_tokens", hidden_states.data(), L_in, desc.D);

    PREFILL_PROF_TICK("prefill:embed+positions");
    time_utils::time_point start = time_utils::now();
    // FLM_MINICPM_OCR_PREFILL_PROFILE splits prefill by layer kind. The two kinds have very
    // different backends -- the GatedDeltaNet layers carry conv1d + the delta rule, the
    // attention layers carry the mha overlay -- so which one dominates is the first thing
    // worth knowing. For a finer split inside a layer, build with -DMINICPM_OCR_PREFILL_PROFILE
    // (minicpm_ocr_prefill.hpp).
    const bool profile_layers = std::getenv("FLM_MINICPM_OCR_PREFILL_PROFILE") != nullptr;
    double t_attn = 0, t_linear = 0;
    int n_attn = 0, n_linear = 0;
    for (int layer_idx = 0; layer_idx < desc.num_hidden_layers; layer_idx++) {
        time_utils::time_point t0 = time_utils::now();
        buffer<bf16>* next_rms = layer_idx + 1 < (int)desc.num_hidden_layers ? &rms_weights[layer_idx + 1] : nullptr;
        pctx->forward(layer_idx, hidden_states, L_in, proj_weights[layer_idx], rms_weights[layer_idx], rope_rms_weights[layer_idx], kv_caches[layer_idx], input_idx, alpha_proj_cpu[layer_idx], beta_proj_cpu[layer_idx], this->current_context_length, next_rms);
        if (profile_layers){
            const bool attn = is_attn_layer(layer_idx);
            double ms = time_utils::duration_ms(t0, time_utils::now()).first;
            (attn ? t_attn : t_linear) += ms;
            (attn ? n_attn : n_linear)++;
            fprintf(stderr, "[minicpm_ocr prefill] L=%d layer %2d %-6s %7.2f ms\n", L_in, layer_idx,
                    attn ? "attn" : "linear", ms);
        }
    }
    if (profile_layers){
        fprintf(stderr, "[minicpm_ocr prefill] L=%d  %d attn %.1f ms  %d linear %.1f ms\n",
                L_in, n_attn, t_attn, n_linear, t_linear);
    }

    time_utils::time_point t_layers = time_utils::now();

    buffer<bf16> predict = hidden_states_2d[L_in - 1];
    // no lm_head weight sync here: load_weights() wrote and synced them once, nothing writes them since
    time_utils::time_point t_lmsync = time_utils::now();
    get_logits(predict);
    // only the last row's logits exist; score against logits.npy[-1]
    MINICPM_OCR_DUMP("logits_last", logits_valid.data(), 1, desc.vocab_size);
    minicpm_ocr_cpu::dumper().active = false;
    // VISION_PROFILE reports it next to the tower's own line, for TTFT work; PREFILL_PROFILE
    // reports it for text-only prompts too, which is the only way to time a text prefill from
    // the CLI (`flm run` prints no profile; only the test harness does).
    if ((has_images && std::getenv("FLM_MINICPM_OCR_VISION_PROFILE")) || profile_layers){
        time_utils::time_point done = time_utils::now();
        header_print_r("FLM", "language prefill: " + std::to_string(L_in) + " tokens in " +
                              std::to_string((int)time_utils::duration_ms(start, done).first) + " ms" +
                              " (layers " + std::to_string((int)time_utils::duration_ms(start, t_layers).first) +
                              ", lm_head w sync " + std::to_string((int)time_utils::duration_ms(t_layers, t_lmsync).first) +
                              ", lm_head " + std::to_string((int)time_utils::duration_ms(t_lmsync, done).first) + ")");
    }
    this->set_context_length(this->current_context_length + L_in, next_position);

    return logits_valid;
}

std::vector<bf16> minicpm_ocr_npu::Impl::_encode_images(const minicpm_ocr_image_payload_t* p, int n_image_tokens,
                                                      std::vector<mco::view_grid>& views){
    if (!image_mod){
        throw std::runtime_error("images supplied but no vision tower is loaded -- either the config has no "
                                 "vision_model_weight or FLM_MINICPM_OCR_VISION=0");
    }
    const size_t patch_dim = vision_desc.patch_dim();
    size_t patches = 0;
    int rows = 0;
    for (const minicpm_ocr_image_t& im : p->images){
        views.push_back(mco::view_grid{im.grid_h, im.grid_w});
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
    mco::simd::bf16_to_f32(pixels.data(), p->_data__processed.data(), pixels.size());
    std::vector<float> out((size_t)rows * desc.D);
    image_mod->encode(pixels.data(), views, out.data());

    std::vector<bf16> image_rows(out.size());
    mco::simd::f32_to_bf16(image_rows.data(), out.data(), out.size());
    if (std::getenv("FLM_MINICPM_OCR_VISION_PROFILE")){
        header_print_r("FLM", "vision: " + std::to_string(views.size()) + " views, " + std::to_string(patches) +
                              " patches -> " + std::to_string(rows) + " rows in " +
                              std::to_string((int)time_utils::duration_ms(t0, time_utils::now()).first) + " ms (" +
                              image_mod->backend() + ")");
    }
    return image_rows;
}

buffer<bf16> minicpm_ocr_npu::Impl::get_k_cache(int layer_idx, int idx){
    this->kv_caches[layer_idx].sync_from_device();
    buffer<bf16> k_cache(desc.DK);
    bf16* k_cache_ptr = this->kv_caches[layer_idx].data();
    uint32_t offset = idx * desc.DK;
    memcpy(static_cast<void*>(k_cache.data()), static_cast<void*>(k_cache_ptr + offset), desc.DK * sizeof(bf16));

    return k_cache;
}

/// @note  The cache slot is [K for MAX_L rows | V for MAX_L rows], so V starts at
///        V_offset. Without it this returned K and silently agreed with get_k_cache --
///        which is how it read during S4 bring-up: every attention layer's "k cos" and
///        "v cos" against the host path were equal to six digits.
buffer<bf16> minicpm_ocr_npu::Impl::get_v_cache(int layer_idx, int idx){
    this->kv_caches[layer_idx].sync_from_device();
    buffer<bf16> v_cache(desc.DV);
    bf16* v_cache_ptr = this->kv_caches[layer_idx].data();
    size_t offset = this->V_offset + (size_t)idx * desc.DV;
    memcpy(static_cast<void*>(v_cache.data()), static_cast<void*>(v_cache_ptr + offset), desc.DV * sizeof(bf16));
    return v_cache;
}

buffer<bf16> minicpm_ocr_npu::Impl::get_logits(buffer<bf16>& predict){
    memcpy(this->x.data(), predict.data(), desc.D * sizeof(bf16));
    this->x.sync_to_device();
#if MINICPM_OCR_DECODE_ON_NPU
    this->lm_head_run.start();
    flm::npu::wait_for_completion(this->lm_head_run, "lm_head_run");
    this->logits.sync_from_device();
#else
    this->_lm_head_host();
#endif
    return this->logits_valid;
}

/// @brief the lm_head.xclbin run on the host: final RMSNorm of x[0, D) by the weight the
///        loader stored behind it at x[D, 2D), then the q8_0 matvec over the lm_head bands
///        in reorder_cpy_lm_head's layout. Leaves logits exactly where the array would.
void minicpm_ocr_npu::Impl::_lm_head_host(){
    buffer<bf16> normed(desc.D);
    cpu_func::_rms_norm(normed.data(), this->x.data(), this->x.data() + desc.D, desc.D, 1, 0, 0);
    MINICPM_OCR_DUMP("final_norm_last", normed.data(), 1, desc.D);
    minicpm_ocr_cpu::dequant_mm(this->logits.data(), desc.vocab_size_padded, this->lm_head_weights.data(),
                              normed.data(), desc.D, 1, desc.D, desc.vocab_size, minicpm_ocr_cpu::LM_HEAD_LAYOUT);
    this->logits.sync_to_device();
}

void minicpm_ocr_npu::Impl::_allocate_checkpoint_buffers() {
    this->cache_checkpoint.clear();
    this->cache_checkpoint.resize(desc.num_hidden_layers);
    is_checkpoint_valid = false;
    for (int i = 0; i < desc.num_hidden_layers; i++) {
        if (!is_attn_layer(i)) {
            this->cache_checkpoint[i] = buffer<bf16>(this->kv_caches[i].size());
        }
    }
}

int minicpm_ocr_npu::Impl::checkpoint() {
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

int minicpm_ocr_npu::Impl::restore() {
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

            // rows past the checkpoint's length: what the requests since it was taken left there (not the live length, which
            // is larger and would leave the rows in between stale)
            int offset = (size_t)checkpoint_context_length * (desc.DK);
            // first half is K, second half is V, and they are stored contiguously in kv_cache
            memset(this->kv_caches[i].data() + offset, 0, (this->kv_caches[i].size() / 2 - offset) * sizeof(bf16)); // zero out the part that is not restored for global layers, since we only restore the sliding part for global layers
            memset(this->kv_caches[i].data() + this->kv_caches[i].size() / 2 + offset, 0, (this->kv_caches[i].size() / 2 - offset) * sizeof(bf16)); // zero out the second half for swa
            this->kv_caches[i].sync_to_device();
        }
    }
    pctx->attn_block->kv_cache_changed(checkpoint_context_length);
    decode_kv.cache_changed(checkpoint_context_length);
    this->current_context_length = this->checkpoint_context_length;
    this->general_token_idx = this->checkpoint_general_token_idx;
    this->set_context_length(this->current_context_length, this->general_token_idx);
    return this->current_context_length;
}

void minicpm_ocr_npu::Impl::clear_context(){
    this->set_context_length(0, 0);
    for (int i = 0; i < desc.num_hidden_layers; i++){
        this->kv_caches[i].sync_from_device();
        memset(this->kv_caches[i].data(), 0, this->kv_caches[i].size() * sizeof(bf16));
        this->kv_caches[i].sync_to_device();
    }
    if (pctx) pctx->attn_block->kv_cache_changed(0);
    decode_kv.cache_changed(0);
}

size_t minicpm_ocr_npu::Impl::state_row_bytes() {
    size_t attention_layers = 0;
    for (int i = 0; i < desc.num_hidden_layers; i++) {
        attention_layers += is_attn_layer(i) ? 1 : 0;
    }
    return attention_layers * (static_cast<size_t>(desc.DK) + desc.DV) * sizeof(bf16);
}

size_t minicpm_ocr_npu::Impl::state_fixed_bytes() {
    size_t bytes = 0;
    for (int i = 0; i < desc.num_hidden_layers; i++) {
        if (!is_attn_layer(i)) {
            bytes += this->kv_caches[i].size() * sizeof(bf16);
        }
    }
    return bytes;
}

std::unique_ptr<hybrid_engine_state> minicpm_ocr_npu::Impl::new_state() {
    return std::make_unique<hybrid_engine_state>(state_fixed_bytes(), state_row_bytes());
}

int minicpm_ocr_npu::Impl::read_state(hybrid_engine_state& st, int kv_from) {
    const int length = this->current_context_length;
    kv_from = std::clamp(kv_from, 0, length);
    st.state.resize(desc.num_hidden_layers);
    st.kv_k.resize(desc.num_hidden_layers);
    st.kv_v.resize(desc.num_hidden_layers);
    header_print_r("FLM", "Reading state at context length " + std::to_string(length) + ", K/V rows from " + std::to_string(kv_from));
    const size_t rows = static_cast<size_t>(length - kv_from);
    for (int i = 0; i < desc.num_hidden_layers; i++) {
        this->kv_caches[i].sync_from_device();
        if (!is_attn_layer(i)) {
            if (st.state[i].size() != this->kv_caches[i].size()) st.state[i] = buffer<bf16>(this->kv_caches[i].size());
            memcpy(st.state[i].data(), this->kv_caches[i].data(), this->kv_caches[i].size() * sizeof(bf16));
        }
        else if (rows > 0) {
            // K rows then V rows, one DK / DV row per position; the V half starts at the middle of the buffer
            if (st.kv_k[i].size() != rows * desc.DK) st.kv_k[i] = buffer<bf16>(rows * desc.DK);
            if (st.kv_v[i].size() != rows * desc.DV) st.kv_v[i] = buffer<bf16>(rows * desc.DV);
            memcpy(st.kv_k[i].data(), this->kv_caches[i].data() + static_cast<size_t>(kv_from) * desc.DK, rows * desc.DK * sizeof(bf16));
            memcpy(st.kv_v[i].data(), this->kv_caches[i].data() + (this->kv_caches[i].size() / 2) + static_cast<size_t>(kv_from) * desc.DV, rows * desc.DV * sizeof(bf16));
        }
        else {
            // nothing to keep: a reused state must not hold rows of its previous contents
            st.kv_k[i] = buffer<bf16>();
            st.kv_v[i] = buffer<bf16>();
        }
        if (!is_attn_layer(i)) {
            this->kv_caches[i].sync_to_device();
        }
    }
    st.filled = true;
    st.context_length = length;
    st.general_token_idx = this->general_token_idx;
    st.kv_from = kv_from;
    return length;
}

/// @brief whether `st` holds what this engine's write_state needs (a state of another engine would not)
static bool minicpm_ocr_state_fits(const hybrid_engine_state& st, const std::vector<buffer<bf16>>& kv_caches, int num_layers, int DK, int DV, int full_attention_interval) {
    if (!st.filled || static_cast<int>(st.state.size()) != num_layers || st.kv_k.size() != st.state.size() || st.kv_v.size() != st.state.size() || st.context_length < st.kv_from) {
        return false;
    }
    const size_t rows = static_cast<size_t>(st.context_length - st.kv_from);
    for (int i = 0; i < num_layers; i++) {
        const bool attn = (i % full_attention_interval) == (full_attention_interval - 1);
        if (attn ? (rows > 0 && (st.kv_k[i].size() != rows * DK || st.kv_v[i].size() != rows * DV))
                 : st.state[i].size() != kv_caches[i].size()) {
            return false;
        }
    }
    return true;
}

int minicpm_ocr_npu::Impl::write_state_kv(const hybrid_engine_state& st) {
    if (!minicpm_ocr_state_fits(st, this->kv_caches, desc.num_hidden_layers, desc.DK, desc.DV, full_attention_interval)) {
        return -1;
    }
    const size_t rows = static_cast<size_t>(st.context_length - st.kv_from);
    if (rows == 0) {
        return st.context_length;
    }
    for (int i = 0; i < desc.num_hidden_layers; i++) {
        if (is_attn_layer(i)) {
            this->kv_caches[i].sync_from_device();
            memcpy(this->kv_caches[i].data() + static_cast<size_t>(st.kv_from) * desc.DK, st.kv_k[i].data(), rows * desc.DK * sizeof(bf16));
            memcpy(this->kv_caches[i].data() + (this->kv_caches[i].size() / 2) + static_cast<size_t>(st.kv_from) * desc.DV, st.kv_v[i].data(), rows * desc.DV * sizeof(bf16));
            this->kv_caches[i].sync_to_device();
        }
    }
    pctx->attn_block->kv_cache_changed(st.kv_from);
    decode_kv.cache_changed(st.kv_from);
    return st.context_length;
}

int minicpm_ocr_npu::Impl::write_state(const hybrid_engine_state& st) {
    if (!minicpm_ocr_state_fits(st, this->kv_caches, desc.num_hidden_layers, desc.DK, desc.DV, full_attention_interval)) {
        header_print_r("FLM", "No valid state to write");
        return -1;
    }
    header_print_r("FLM", "Writing state at context length " + std::to_string(st.context_length));
    const size_t rows = static_cast<size_t>(st.context_length - st.kv_from);
    for (int i = 0; i < desc.num_hidden_layers; i++) {
        this->kv_caches[i].sync_from_device();
        if (!is_attn_layer(i)) {
            memcpy(this->kv_caches[i].data(), st.state[i].data(), st.state[i].size() * sizeof(bf16));
        }
        else {
            if (rows > 0) {
                memcpy(this->kv_caches[i].data() + static_cast<size_t>(st.kv_from) * desc.DK, st.kv_k[i].data(), rows * desc.DK * sizeof(bf16));
                memcpy(this->kv_caches[i].data() + (this->kv_caches[i].size() / 2) + static_cast<size_t>(st.kv_from) * desc.DV, st.kv_v[i].data(), rows * desc.DV * sizeof(bf16));
            }
            // Rows past the restored length hold whatever the requests since the state was read left there. The kernels do not
            // rely on the length alone to keep them out of attention, so put back the zeros a fresh context has.
            const size_t from = static_cast<size_t>(st.context_length);
            const size_t half = this->kv_caches[i].size() / 2;
            memset(this->kv_caches[i].data() + from * desc.DK, 0, (half - from * desc.DK) * sizeof(bf16));
            memset(this->kv_caches[i].data() + half + from * desc.DV, 0, (half - from * desc.DV) * sizeof(bf16));
        }
        this->kv_caches[i].sync_to_device();
    }
    pctx->attn_block->kv_cache_changed(st.kv_from);
    decode_kv.cache_changed(st.kv_from);
    this->current_context_length = st.context_length;
    this->general_token_idx = st.general_token_idx;
    this->set_context_length(this->current_context_length, this->general_token_idx);
    return this->current_context_length;
}

int minicpm_ocr_npu::Impl::get_current_context_length(){
    return this->current_context_length;
}

void minicpm_ocr_npu::Impl::update_max_length(uint32_t MAX_L){
    if (MAX_L <= this->MAX_L){
        header_print("FLM", "New length is shorter than the current length, no need to update!");
        return; // no need to update
	}
    this->MAX_L = MAX_L;
    size_t kv_cache_size = (size_t)MAX_L * (desc.DK + desc.DV);
    // sized from an attention layer: layer 0 is a linear layer, whose cache slot is a
    // conv + recurrent state of a different size
    buffer<bf16> kv_cache_temp(this->kv_caches[desc.attn_idx].size());
    size_t original_V_offset = this->V_offset;
    this->sequence->set_max_length(MAX_L);
    this->V_offset = this->sequence->get_V_offset();
    for (uint32_t i = 0; i < desc.num_hidden_layers; i++){
        if (!is_attn_layer(i)){
            continue; // nothing to do with linear layer
        }
        this->kv_caches[i].sync_from_device();
        memcpy(static_cast<void*>(kv_cache_temp.data()), static_cast<void*>(this->kv_caches[i].data()), this->kv_caches[i].size() * sizeof(bf16));
        this->kv_caches[i] = npu->create_bo_buffer<bf16>(kv_cache_size);
        this->kv_caches[i].memset((bf16)0); // the rows past the old size must start zero
        memcpy(static_cast<void*>(this->kv_caches[i].data()), static_cast<void*>(kv_cache_temp.data()), (kv_cache_temp.size() / 2) * sizeof(bf16));
        memcpy(static_cast<void*>(this->kv_caches[i].data() + this->V_offset), static_cast<void*>(kv_cache_temp.data() + original_V_offset), (kv_cache_temp.size() / 2) * sizeof(bf16));
        this->kv_caches[i].sync_to_device(); // after the copy: the device must see the migrated rows
    }
    this->set_context_length(this->current_context_length, this->general_token_idx); // regenerate sequence
    pctx->attn_block->kv_cache_changed(original_V_offset / desc.DK); // the old size: rows past it are new zeros
    decode_kv.cache_changed(original_V_offset / desc.DK);
    is_checkpoint_valid = false; // invalidate checkpoint since the underlying kv cache has changed
}
///@brief destructor of minicpm_ocr_npu
minicpm_ocr_npu::Impl::~Impl() = default;

// ===============================================
// Externals for minicpm_ocr_npu
// ===============================================
minicpm_ocr_npu::minicpm_ocr_npu(LM_Config config, npu_xclbin_manager *npu_instance, int MAX_L){

    this->load_vision_preprocess_parameters(config);
    this->_impl = new Impl(config, npu_instance, this,   MAX_L);
}

buffer<bf16> minicpm_ocr_npu::forward(int ids){
    return this->_impl->forward(ids);
}

buffer<bf16> minicpm_ocr_npu::prefill(std::vector<int>& ids, void* payload){
    try {
        buffer<bf16> result = this->_impl->prefill(ids, payload);
        return result;
    }
    catch (const std::runtime_error& e) {
        header_print("FLM", e.what());
        throw;
    }
}

void minicpm_ocr_npu::set_context_length(int L){
    std::cout << "Setting context length is not supported for minicpm_ocr_npu!" << std::endl;
}

void minicpm_ocr_npu::load_weights(Q4NX& q4nx){
    this->_impl->load_weights(q4nx);
}

void minicpm_ocr_npu::update_max_length(uint32_t MAX_L){
    this->_impl->update_max_length(MAX_L);
}

void minicpm_ocr_npu::clear_context(){
    this->_impl->clear_context();
}

buffer<bf16> minicpm_ocr_npu::get_k_cache(int layer_idx, int idx){
    return this->_impl->get_k_cache(layer_idx, idx);
}

buffer<bf16> minicpm_ocr_npu::get_v_cache(int layer_idx, int idx){
    return this->_impl->get_v_cache(layer_idx, idx);
}

int minicpm_ocr_npu::get_current_context_length(){
    return this->_impl->get_current_context_length();
}

int minicpm_ocr_npu::checkpoint(){
    return this->_impl->checkpoint();
}

int minicpm_ocr_npu::restore(){
    return this->_impl->restore();
}

std::unique_ptr<hybrid_engine_state> minicpm_ocr_npu::new_state(){
    return this->_impl->new_state();
}

int minicpm_ocr_npu::read_state(hybrid_engine_state& state, int kv_from){
    return this->_impl->read_state(state, kv_from);
}

int minicpm_ocr_npu::write_state(const hybrid_engine_state& state){
    return this->_impl->write_state(state);
}

int minicpm_ocr_npu::write_state_kv(const hybrid_engine_state& state){
    return this->_impl->write_state_kv(state);
}

extern "C" const flm_state_api* minicpm_ocr_state_api(void* engine){
    return hybrid_state_api<minicpm_ocr_npu>(engine);
}

minicpm_ocr_npu::~minicpm_ocr_npu(){
    delete this->_impl;
}
