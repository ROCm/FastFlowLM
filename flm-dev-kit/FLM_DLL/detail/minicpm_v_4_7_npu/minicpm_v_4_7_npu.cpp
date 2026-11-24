// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#include <random>
#if !defined(_WIN32) && !defined(__WINDOWS__)
// amdxdna driver access (the fault-injection test hooks and the single-context checks): Linux only
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include "npu_utils/amdxdna_accel.h"
#endif
#include <cstring>
#include <chrono>
#include "minicpm_v_4_7_npu_detail.hpp"
#include "host_executor.hpp"
#include "image/vision_common.hpp"
#include "image/vision_tower.hpp"
#include "image/vision_profile.hpp"
#include "image/npu/vision_op_factory.hpp"
#include "image/npu/vision_bo.hpp"
#include <filesystem>
minicpm_v_4_7_npu::Impl::Impl(LM_Config config, npu_xclbin_manager *npu_instance, minicpm_v_4_7_npu* parent_ptr, int MAX_L )
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

    this->sequence = std::make_unique<minicpm_v_4_7_npu_sequence>(this->desc, config, MAX_L);
    this->V_offset = this->sequence->get_V_offset();

    // The descriptors that never go through a weight_container: q_norm / k_norm sit
    // behind the rope table, and the bf16 alpha/beta twins live in host buffers. Without
    // indp() their ready() stays false however they are loaded (ADDING_A_NEW_MODEL.md S1.2).
    {
        auto& attn = static_cast<minicpm_v_attn_layer&>(*desc.weight_desc[desc.attn_idx]);
        auto& lin  = static_cast<minicpm_v_linear_layer&>(*desc.weight_desc[desc.lin_idx]);
        attn.attn_q_norm.indp();
        attn.attn_k_norm.indp();
        lin.ssm_alpha_cpu.indp();
        lin.ssm_beta_cpu.indp();
    }

    // layer.xclbin and lm_head.xclbin belong to decode; with DECODE=cpu neither is
    // registered and a token runs through the prefill graph at L = 1 plus a host lm_head.
#if MINICPM_V_DECODE_ON_NPU
    {
        const std::string xdir = utils::path_join(config.exec_path, "xclbins", config.model_name);
        const char* env = std::getenv("FLM_MINICPM_LM_IN_LAYER");
        const char* env_sl = std::getenv("FLM_MINICPM_LM_STATELESS");   // =0: do not use layer_lm_sl.xclbin even if the folder has it
        const bool lm_off = env && std::atoi(env) == 0;
        // layer_lm_sl.xclbin (every run takes x from DDR and returns it: correct under any interleaving with other processes' NPU commands) is preferred
        // to layer_lm.xclbin (x stays on the array inside a layer group: needs the foreign-context witness), which is preferred to the two-xclbin path.
        this->lm_stateless = !lm_off && !(env_sl && std::atoi(env_sl) == 0) && std::filesystem::exists(utils::path_join(xdir, "layer_lm_sl.xclbin"));
        this->lm_in_layer = this->lm_stateless || (!lm_off && std::filesystem::exists(utils::path_join(xdir, "layer_lm.xclbin")));
        this->sequence->lm_in_layer = this->lm_in_layer;
        this->sequence->stateless = this->lm_stateless;
        if (this->lm_in_layer){
            header_print_r("INFO", (this->lm_stateless ? "minicpm_v: lm_head runs inside layer_lm_sl.xclbin (one xclbin; every run takes x from DDR, so other processes' NPU use cannot corrupt a token)"
                                                       : "minicpm_v: lm_head runs inside layer_lm.xclbin (no xclbin switch per token)"));
        }
    }
    layer_app_manager = npu->register_xclbin(utils::path_join(config.exec_path, "xclbins", config.model_name,
                                              this->lm_stateless ? "layer_lm_sl.xclbin" : this->lm_in_layer ? "layer_lm.xclbin" : "layer.xclbin"));
#ifdef FLM_MINICPM_FAULT_RELOAD
    if (std::getenv("FLM_MINICPM_FAULT_RELOAD")){
        auto dm = npu->register_xclbin(utils::path_join(config.exec_path, "xclbins", config.model_name, "conv.xclbin"));
        this->fault_dummy = std::make_unique<npu_app>(dm->create_app());
        this->fault_dummy->seq()->clear_cmds();
        this->fault_dummy->seq()->cmds2seq();
    }
#endif

    this->layer = layer_app_manager->create_app();
    this->linear_layer = layer_app_manager->create_app();
    this->first_linear_layer = layer_app_manager->create_app();
    this->set_layer_rtp = layer_app_manager->create_app();
    if (!this->npu->is_preemption_enabled()){
        this->layers_run = layer_app_manager->create_runlist();
    }

    if (!this->lm_in_layer){
        lm_head_app_manager = npu->register_xclbin(utils::path_join(config.exec_path, "xclbins", config.model_name, "lm_head.xclbin"));
        this->lm_head = lm_head_app_manager->create_app();
    }
#endif

    // allocate all buffers. The weights, caches, x and logits are device BOs in every
    // configuration: create_bo_buffer needs no registered xclbin, and
    // minicpm_v_desc::load_*_weights() sync them unconditionally. So even the all-host
    // build opens the device (but registers no xclbin and opens no hw context).
    rms_weights.resize(desc.num_hidden_layers);
    proj_weights.resize(desc.num_hidden_layers);
    kv_caches.resize(desc.num_hidden_layers);
    rope_rms_weights.resize(desc.num_hidden_layers);
    alpha_proj_cpu.resize(desc.num_hidden_layers);
    beta_proj_cpu.resize(desc.num_hidden_layers);

#if MINICPM_V_MM_ON_NPU
    {
        // fused GLU matmul (default on when mm_rtm2_glu.xclbin is in the model's xclbin folder; FLM_MINICPM_GLU_FUSED=0 switches it off).
        // Every layer's proj_weights BO then carries a re-packed copy of the up/gate weights behind the regular ones.
        const char* glu_env = std::getenv("FLM_MINICPM_GLU_FUSED");
        if (!(glu_env && std::atoi(glu_env) == 0) &&
            std::filesystem::exists(utils::path_join(config.exec_path, "xclbins", config.model_name, "mm_rtm2_glu.xclbin"))){
            desc.enable_glu_fused();
            LOG_VERBOSE(1, "Fused GLU matmul enabled: up/gate weights re-packed");
        }
    }
#endif
    size_t attn_total_proj_weights_byte_size = desc.proj_weights_bo_bytes(desc.attn_total_proj_weights_byte_size);
    size_t linear_total_proj_weights_byte_size = desc.proj_weights_bo_bytes(desc.linear_total_proj_weights_byte_size);
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

    is_checkpoint_valid = false; // checkpoint is not valid until we load kv cache to device after resizing buffers
    x = npu->create_bo_buffer<bf16>(desc.D * 2);
    lm_head_weights = npu->create_bo_buffer<u8>(this->lm_in_layer ? desc.get_lm_head_in_layer_w_size() : desc.get_lm_head_w_size());
    // reorder_cpy_lm_head fills only the bands the checkpoint has; zero the rest so the
    // host lm_head never reads uninitialised scales
    memset(lm_head_weights.data(), 0, lm_head_weights.size());
    logits = npu->create_bo_buffer<bf16>(desc.vocab_size_padded);
    logits_valid = buffer<bf16>(logits.data(), desc.vocab_size);
    decode_hidden = buffer<bf16>(desc.D);

    this->clear_context();
    // the embedding table is created in load_weights, once we know whether the
    // checkpoint stores model.embed_tokens or only the tied q8_0 lm_head.

    pctx = std::make_unique<minicpm_v_prefill_context>(npu, &desc, sequence.get(), config);

#if MINICPM_V_DECODE_ON_NPU
    if (this->lm_in_layer){
        // the lm_head is a run of the layer design: x, lm_head weights, final norm weight, (unused), logits
        this->lm_head = this->layer_app_manager->create_app();
        this->lm_head_norm = npu->create_bo_buffer<bf16>(desc.D);
        this->sequence->gen_lm_head_in_layer_seq(lm_head.seq());
    }
    else{
        lm_head = this->lm_head_app_manager->create_app();
        this->sequence->gen_lm_head_seq(lm_head.seq(), desc.vocab_size_padded, desc.D);
    }
    this->make_lm_head_run();
#if FLM_MINICPM_RECONFIG_SUPPORTED
    if (dec_single_ctx_wanted() && !this->lm_in_layer && !this->npu->is_preemption_enabled()){
        try{
            const std::string dir = utils::path_join(config.exec_path, "xclbins", config.model_name);
            auto mk = [&] { return layer_app_manager->create_app(); };
            this->dec_to_lm.build(dir + "/lm_head.xclbin", dir + "/layer.xclbin", mk);
            this->dec_to_layer.build(dir + "/layer.xclbin", dir + "/lm_head.xclbin", mk);
            this->lm_head_sc = layer_app_manager->create_app();
            this->sequence->gen_lm_head_seq(lm_head_sc.seq(), desc.vocab_size_padded, desc.D);
            this->dec_sc_built = true;
            header_print_r("INFO", "minicpm_v: single-context decode built (" << minicpm_reconfig::versions() << "); known-answer test when the weights are loaded");
        } catch (const std::exception& e){
            header_print_r("WARNING", "minicpm_v: single-context decode unavailable: " << e.what());
        }
    }
#endif

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
///        FLM_MINICPM_V_VISION=0 declines the tower as a FEATURE (text-only), which
///        is a different thing from choosing an implementation of it.
void minicpm_v_4_7_npu::Impl::_load_vision(){
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
        this->image_mod = std::make_unique<mcv::vision_tower>(vision_desc,
                                                              std::make_unique<mcv::vision::cpu_op_factory>());
    }
    else {
        // vision_attn_kv.xclbin (MiniCPM_V/vision_attn_kv, k/v once per round, ~0.4 s less at 120 frames): used when the folder
        // has it (FLM_MINICPM_V_VISION_ATTN_KV=0 keeps vision_attn.xclbin). Not bit-identical to vision_attn.xclbin, and no
        // further from the fp32 tower (cosine 0.9920 vs 0.9912 at 120 frames).
        const std::string attn_kv = utils::path_join(dir, "vision_attn_kv.xclbin");
        const char* kv_env = std::getenv("FLM_MINICPM_V_VISION_ATTN_KV");
        mcv_npu::vision_attn_rtp_addresses attn_rtp{};
        npu_app_manager* attn_mgr = nullptr;
        std::string attn_used = attn;
        if (std::filesystem::exists(attn_kv) && !(kv_env && std::atoi(kv_env) == 0)){
            try { attn_mgr = npu->register_xclbin(attn_kv); attn_rtp.kv_once = true; attn_used = attn_kv; }
            catch (const std::exception& e){
                header_print("WARNING", "vision_attn_kv.xclbin is present but did not load (" << e.what() << "), using vision_attn.xclbin");
            }
        }
        if (!attn_mgr) attn_mgr = npu->register_xclbin(attn);
        auto factory = std::make_unique<mcv_npu::vision_op_factory>(attn_mgr, attn_rtp);
        // optional design P (C on packed bfp16 operands, C's bits, 16-28% less device time): when the folder has it,
        // it serves every op C would at every M and C is not loaded (FLM_MINICPM_V_VISION_MM_P=0 falls back to C).
        // Small images then get C's rounding instead of A's: same tower as C/A at 120 frames, at 1 frame cosine vs
        // the fp32 tower 0.997066 (C/A mix 0.997066, A alone 0.995685), and a frame no longer depends on the batch.
        const std::string mm_p = utils::path_join(dir, "vision_mm_p.xclbin");
        npu_app_manager* mm_p_mgr = nullptr;
        if (std::filesystem::exists(mm_p) && !mcv_npu::mm_p_disabled() && mcv_npu::preround_enabled()){
            try { mm_p_mgr = npu->register_xclbin(mm_p); }
            catch (const std::exception& e){
                header_print("WARNING", "vision_mm_p.xclbin is present but did not load (" << e.what() << "), using design C / A / B");
            }
        }
        // optional design C (512-row tiles, 1.2-1.6x faster from 512 rows up; rounds differently from A): used when the folder has it
        // (FLM_MINICPM_V_VISION_MM_C=0 keeps the matmuls on A / B)
        const std::string mm_c = utils::path_join(dir, "vision_mm_c.xclbin");
        npu_app_manager* mm_c_mgr = nullptr;
        if (!mm_p_mgr && std::filesystem::exists(mm_c) && !mcv_npu::mm_c_disabled()){
            try { mm_c_mgr = npu->register_xclbin(mm_c); }
            catch (const std::exception& e){
                header_print("WARNING", "vision_mm_c.xclbin is present but did not load (" << e.what() << "), using designs A / B");
            }
        }
        factory->enable_mm(npu->register_xclbin(mm_a), npu->register_xclbin(mm_b), mcv_npu::vision_mm_rtp_addresses{}, mm_c_mgr, mm_p_mgr);
#if FLM_MINICPM_RECONFIG_SUPPORTED
        // opt-in (FLM_MINICPM_V_VISION_SINGLE_CTX=1): attention dispatches in vision_mm_a's context between in-stream reconfigurations
        factory->enable_single_context(npu->is_preemption_enabled(), dir, attn_used);
#endif
        this->image_mod = std::make_unique<mcv::vision_tower>(vision_desc, std::move(factory));
    }
    // header_print("FLM", "vision tower loaded (" + std::to_string(vision_desc.depth) + " blocks, " +
    //                     std::string(image_mod->backend()) + ") in " +
    //                     std::to_string((int)time_utils::duration_ms(t0, time_utils::now()).first) + " ms");
}

void minicpm_v_4_7_npu::Impl::load_weights(Q4NX& q4nx){
    // Newer converters tie the word embeddings and emit only the quantized
    // lm_head, so fall back to dequantizing the embedding out of it.
    if (q4nx.has_tensor("model.embed_tokens.weight")){
        this->embedding = std::make_unique<EmbeddingBF16Mapped>(desc.vocab_size, desc.D);
        this->embedding->init_weights(q4nx, utils::path_join(config.model_path, "model.q4nx"), "model.embed_tokens");
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
        if (this->lm_in_layer){
            memcpy(this->lm_head_norm.data(), final_rms_norm_weight.data(), desc.D * sizeof(bf16));
            this->lm_head_norm.sync_to_device();
        }
    }
    {
        buffer<u8> lm_head_w;
        q4nx.load_weights(lm_head_w, "lm_head.weight");

        if (this->lm_in_layer){
            desc.reorder_cpy_lm_head_in_layer(this->lm_head_weights.data(), lm_head_w);
        }
        else{
            desc.reorder_cpy_lm_head(this->lm_head_weights.data(), lm_head_w, desc.D, flm_q80, 4);
        }
        this->lm_head_weights.sync_to_device();
    }

    this->_check_descriptors(q4nx);
#if MINICPM_V_DECODE_ON_NPU && FLM_MINICPM_RECONFIG_SUPPORTED
    if (this->dec_sc_built) this->dec_sc_selftest();
#endif
#if MINICPM_V_ATTN_ON_NPU && MINICPM_V_MM_ON_NPU && FLM_MINICPM_RECONFIG_SUPPORTED
    // opt-in single-context attention: known-answer test on the real weights and the real K/V buffer (rows it dirties are zeroed below)
    pctx->validate_single_context(this->proj_weights[desc.attn_idx], this->kv_caches[desc.attn_idx], this->kv_hwm_rows);
#endif
    this->clear_context();
}

/// @brief the "did I forget a tensor" check (ADDING_A_NEW_MODEL.md S1.4), run after every load
/// @note  minicpm_v_desc keeps one representative descriptor per layer kind (lin_idx,
///        attn_idx) and the loaders mark only those, so ready() is checked there, and every
///        layer's own tensors are checked against model.q4nx by name and byte size. Throws
///        on any failure; the table prints at DEBUG_LEVEL >= 1.
void minicpm_v_4_7_npu::Impl::_check_descriptors(Q4NX& q4nx){
    auto& attn = static_cast<minicpm_v_attn_layer&>(*desc.weight_desc[desc.attn_idx]);
    auto& lin  = static_cast<minicpm_v_linear_layer&>(*desc.weight_desc[desc.lin_idx]);
    // all() leaves out the host-only bf16 twins; attn_gate aliases q_proj
    std::vector<weight_desc_t*> attn_all = attn.all();
    std::vector<weight_desc_t*> lin_all  = lin.all();
    lin_all.push_back(&lin.ssm_alpha_cpu);
    lin_all.push_back(&lin.ssm_beta_cpu);

    std::vector<std::string> failures;
    DEBUG_BLOCK(1,
        std::cout << "================ minicpm_v descriptors ================" << std::endl;
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
        throw std::runtime_error("minicpm_v_4_7_npu: " + std::to_string(failures.size()) + " weight descriptor check(s) failed");
    }
}

void minicpm_v_4_7_npu::Impl::_set_rope_rms_weights(int idx){
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

void minicpm_v_4_7_npu::Impl::set_context_length(int L, int new_general_token_idx){
    this->current_context_length = L;
    this->general_token_idx = new_general_token_idx;
    // a decode step writes row L (index) before the length moves past it
    kv_hwm_rows = std::max(kv_hwm_rows, L + 1);

#if MINICPM_V_DECODE_ON_NPU
    this->sequence->gen_set_context(this->set_layer_rtp.seq(), L + 1);
    this->sequence->gen_layer_seq(this->layer.seq(), L + 1, true);
    if (!this->npu->is_preemption_enabled()){
        this->layers_run.reset();
#if MINICPM_V_DECODE_PROFILE
        this->prof_runs.clear();
#endif

        this->layers_run.add(this->set_layer_rtp.create_run());
#if MINICPM_V_DECODE_PROFILE
        this->prof_runs.push_back(this->set_layer_rtp.create_run());
#endif
#if FLM_MINICPM_RECONFIG_SUPPORTED
        const bool build_sc = this->dec_sc_built && !minicpm_reconfig::disabled();
        if (build_sc){
            if (!this->dec_rl_ready){
                this->dec_rl1 = layer_app_manager->create_runlist();
                this->dec_rl2 = layer_app_manager->create_runlist();
                this->dec_rl_ready = true;
            } else {
                this->dec_rl1.reset();
            }
            this->dec_rl1.add(this->set_layer_rtp.create_run());
            for (uint32_t i = 0; i < desc.num_hidden_layers; i++){
                npu_app& a = is_attn_layer(i) ? this->layer : (is_first_linear_layer(i) ? this->first_linear_layer : this->linear_layer);
                this->dec_rl1.add(a.create_run(this->x, this->proj_weights[i], this->rms_weights[i], this->rope_rms_weights[i], this->kv_caches[i]));
            }
            this->dec_to_lm.add_to(this->dec_rl1);
        }
#endif
        for (uint32_t i = 0; i < desc.num_hidden_layers; i++){
            if (is_attn_layer(i)){
                this->layers_run.add(this->layer.create_run(this->x, this->proj_weights[i], this->rms_weights[i], this->rope_rms_weights[i], this->kv_caches[i]));
#if MINICPM_V_DECODE_PROFILE
                this->prof_runs.push_back(this->layer.create_run(this->x, this->proj_weights[i], this->rms_weights[i], this->rope_rms_weights[i], this->kv_caches[i]));
#endif
            }
            else {
                if (is_first_linear_layer(i)) {
                    this->layers_run.add(this->first_linear_layer.create_run(this->x, this->proj_weights[i], this->rms_weights[i], this->rope_rms_weights[i], this->kv_caches[i]));
#if MINICPM_V_DECODE_PROFILE
                    this->prof_runs.push_back(this->first_linear_layer.create_run(this->x, this->proj_weights[i], this->rms_weights[i], this->rope_rms_weights[i], this->kv_caches[i]));
#endif
                }
                else  {
                    this->layers_run.add(this->linear_layer.create_run(this->x, this->proj_weights[i], this->rms_weights[i], this->rope_rms_weights[i], this->kv_caches[i]));
#if MINICPM_V_DECODE_PROFILE
                    this->prof_runs.push_back(this->linear_layer.create_run(this->x, this->proj_weights[i], this->rms_weights[i], this->rope_rms_weights[i], this->kv_caches[i]));
#endif
                }
            }
        }
    }

    this->_set_rope_rms_weights(new_general_token_idx);
#endif
}

#if MINICPM_V_DECODE_ON_NPU && FLM_MINICPM_RECONFIG_SUPPORTED
/// @brief one decode token after the embedding, without hw-context switches (see Impl::dec_chain_t)
buffer<bf16> minicpm_v_4_7_npu::Impl::forward_single_ctx(){
    const long tok = this->dec_sc_tokens;
    try{
        this->dec_rl1.execute();
        this->wait_layers.start();
        this->wait_layers.wait(this->dec_rl1);
    } catch (const std::exception& e){
        this->dec_to_lm.release();
        minicpm_reconfig::disable(std::string("decode layers round failed at single-context token ") + std::to_string(tok) + ": " + e.what());
        this->dec_sc_recover();
        throw;
    }
    this->dec_to_lm.release();
    this->x.sync_to_device();
    this->dec_rl2.reset();
    this->dec_rl2.add(this->lm_head_sc.create_run(this->logits, this->lm_head_weights, this->x));
    this->dec_to_layer.add_to(this->dec_rl2);
    this->dec_rl2.execute();
    this->wait_lm.start();
    // the next token's runlists (and the layer/rtp instruction streams for L + 1) are built while the lm_head round runs: it uses none of them
    this->set_context_length(this->current_context_length + 1, this->general_token_idx + 1);
    try{
        this->wait_lm.wait(this->dec_rl2);
    } catch (const std::exception& e){
        this->dec_to_layer.release();
        minicpm_reconfig::disable(std::string("decode lm_head round failed at single-context token ") + std::to_string(tok) + ": " + e.what());
        this->dec_sc_recover();
        throw;
    }
    this->dec_to_layer.release();
    this->logits.sync_from_device();
    this->dec_sc_tokens++;
    return this->logits_valid;
}

/// @brief after a failed single-context round the firmware may still believe the layer PDI is loaded while the array holds something else; a run in
///        the lm_head context makes it load a different PDI, so the next layer run reloads its own
void minicpm_v_4_7_npu::Impl::dec_sc_recover(){
    try{
        this->x.sync_to_device();
        this->lm_head_run.start();
        flm::npu::wait_for_completion(this->lm_head_run, "lm_head_run");
    } catch (const std::exception& e){
        header_print_r("WARNING", "minicpm_v: recovery dispatch after a failed single-context round also failed: " << e.what());
    }
}

/// @brief known-answer test on the real weights: 160 greedy decode tokens (each step feeds back its own argmax, so the data is the model's own text, not random
///        ids: forced random ids did not trigger the old unwaited-token hang) through the two-context path and through the single-context path from the same
///        empty state; the logits of every token must match bit for bit.
void minicpm_v_4_7_npu::Impl::dec_sc_selftest(){
    constexpr int N = 160, START = 64;
    auto fnv = [&](const buffer<bf16>& b){ uint64_t h = 1469598103934665603ull; const uint16_t* p = reinterpret_cast<const uint16_t*>(b.data()); for (int i = 0; i < vocab_size; i++){ h ^= p[i]; h *= 1099511628211ull; } return h; };
    if (!minicpm_reconfig::npu_exclusive()){
        header_print_r("WARNING", "minicpm_v: single-context decode not validated (another process is on the NPU): two-context decode");
        return;
    }
    std::vector<uint64_t> ref(N, 0), got(N, 1);
    auto run = [&](bool single, std::vector<uint64_t>& out){
        this->clear_context();
        this->set_context_length(START, START);
        this->dec_force_two_ctx = !single;
        this->dec_sc_testing = single;
        this->dec_sc_tokens = 0;
        int id = 1000;
        for (int k = 0; k < N; k++){
            buffer<bf16> lg = this->forward(id);
            out[k] = fnv(lg);
            const uint16_t* p = reinterpret_cast<const uint16_t*>(lg.data());
            float best = -1e30f; int arg = 0;
            for (int i = 0; i < vocab_size; i++){ uint32_t u = (uint32_t)p[i] << 16; float f; std::memcpy(&f, &u, 4); if (f > best){ best = f; arg = i; } }
            id = arg;
        }
    };
    bool ok = false;
    try{
        run(false, ref);
        run(true, got);
        ok = (this->dec_sc_tokens == N) && !minicpm_reconfig::disabled() && ref == got;
        if (!ok && !minicpm_reconfig::disabled()) minicpm_reconfig::disable("decode self-test: logits differ from the two-context path or the single-context path was not taken");
    } catch (const std::exception& e){
        minicpm_reconfig::disable(std::string("decode self-test failed: ") + e.what());
    }
    this->dec_force_two_ctx = false;
    this->dec_sc_testing = false;
    this->dec_sc_validated = ok;
    if (!ok) this->dec_sc_recover();
    header_print_r("INFO", "minicpm_v: single-context decode self-test " << (ok ? "passed" : "FAILED (two-context decode)") << " (" << N << " greedy tokens)");
    this->dec_sc_tokens = 0;
}
#endif

/// @note  The host fallback (MINICPM_V_DECODE_ON_NPU == 0) runs the prefill graph at
///        L = 1, so its conv state, recurrent state and kv cache are the same buffers in
///        the same layout the array decode reads. It is the golden reference for S4.
#ifdef FLM_MINICPM_FAULT_RELOAD
// FLM_MINICPM_FAULT_RELOAD=<kinds> dispatches the empty conv.xclbin sequence before every run of the listed kinds (R = the context-length run,
// F = first linear, L = other linear, A = attention, H = the lm_head run; "all" = all of them), or before each run with probability P percent
// ("rand:P"), or before the runs of the token with the listed indices ("at:2,5"). Test builds only (make FAULT_RELOAD=1).
static bool fault_hit(char kind, long run = -1){
    static const char* m = std::getenv("FLM_MINICPM_FAULT_RELOAD");
    if (!m) return false;
    if (!std::strcmp(m, "all")) return true;
    if (!std::strncmp(m, "at:", 3)){   // "at:2,5": before the runs with these indices (0 = the context-length run, 1..24 = the layers)
        const std::string a = std::string(",") + (m + 3) + ",";
        return run >= 0 && a.find("," + std::to_string(run) + ",") != std::string::npos;
    }
    if (!std::strncmp(m, "rand:", 5)){ static std::mt19937 rng(20261107); return (int)(rng() % 100) < std::atoi(m + 5); }
    return std::strchr(m, kind) != nullptr;
}
// FLM_MINICPM_FAULT_STATUS=<dir> (test builds): dump the driver's AIE status (every tile's core / DMA channel / lock state, 504 bytes a column)
// after every run of the first FLM_MINICPM_FAULT_STATUS_TOKENS decode tokens, as <dir>/s_t<token>_r<run>.bin
static void fault_dump_status(const char* dir, int tok, int run){
    int fd = ::open("/dev/accel/accel0", O_RDWR | O_CLOEXEC);
    if (fd < 0) return;
    std::vector<char> buf(8 * 504, 0);
    amdxdna_drm_query_aie_status stt{};
    stt.buffer = (unsigned long)buf.data(); stt.buffer_size = (uint32_t)buf.size();
    amdxdna_drm_get_info g{};
    g.param = DRM_AMDXDNA_QUERY_AIE_STATUS; g.buffer_size = sizeof stt; g.buffer = (unsigned long)&stt;
    if (::ioctl(fd, DRM_IOCTL_AMDXDNA_GET_INFO, &g) == 0){
        char name[256]; std::snprintf(name, sizeof name, "%s/s_t%d_r%d.bin", dir, tok, run);
        if (FILE* f = std::fopen(name, "wb")){ std::fwrite(buf.data(), 1, buf.size(), f); std::fclose(f); }
    }
    ::close(fd);
}
void minicpm_v_4_7_npu::Impl::fault_layers(){
    std::vector<flm_rt::run> rs;
    std::vector<char> kind;
    rs.push_back(this->set_layer_rtp.create_run()); kind.push_back('R');
    for (uint32_t i = 0; i < desc.num_hidden_layers; i++){
        npu_app& a = is_attn_layer(i) ? this->layer : (is_first_linear_layer(i) ? this->first_linear_layer : this->linear_layer);
        rs.push_back(a.create_run(this->x, this->proj_weights[i], this->rms_weights[i], this->rope_rms_weights[i], this->kv_caches[i]));
        kind.push_back(is_attn_layer(i) ? 'A' : (is_first_linear_layer(i) ? 'F' : 'L'));
    }
    // FLM_MINICPM_FAULT_TRACE=1: the hash of x in DDR after every run of the token (the stateless design hands x over through DDR in every run)
    static const bool trace = std::getenv("FLM_MINICPM_FAULT_TRACE") != nullptr;
    static const char* st_dir = std::getenv("FLM_MINICPM_FAULT_STATUS");
    static const int st_max = std::getenv("FLM_MINICPM_FAULT_STATUS_TOKENS") ? std::atoi(std::getenv("FLM_MINICPM_FAULT_STATUS_TOKENS")) : 3;
    static int st_tok = 0;
    struct tok_counter { int& t; ~tok_counter(){ t++; } } tok_count{st_tok};
    std::string tr;
    if (const char* sd = std::getenv("FLM_MINICPM_FAULT_SEQDIR"); sd && st_tok == 0){   // the instruction streams of the four app kinds, once
        auto dumpseq = [&](const char* nm, npu_sequence* q){ auto d = q->dump(); std::string fn = std::string(sd) + "/seq_" + nm + ".bin";
            if (FILE* f = std::fopen(fn.c_str(), "wb")){ std::fwrite(d.first, 4, d.second, f); std::fclose(f); } };
        dumpseq("rtp", this->set_layer_rtp.seq()); dumpseq("first_linear", this->first_linear_layer.seq());
        dumpseq("linear", this->linear_layer.seq()); dumpseq("attn", this->layer.seq());
    }
    for (size_t r = 0; r < rs.size(); r++){
        if (fault_hit(kind[r], (long)r)) (*this->fault_dummy)();
        rs[r].start();
        flm::npu::wait_for_completion(rs[r], "minicpm layer run");
        if (st_dir && st_tok < st_max) fault_dump_status(st_dir, st_tok, (int)r);
        if (trace){
            this->x.sync_from_device();
            uint64_t h = 1469598103934665603ull;
            for (uint32_t k = 0; k < desc.D; k++){ uint16_t u; std::memcpy(&u, &this->x.data()[k], 2); h = (h ^ u) * 1099511628211ull; }
            char b[48]; std::snprintf(b, sizeof b, "%c%zu:%04llx ", kind[r], r, (unsigned long long)(h & 0xffff)); tr += b;
        }
    }
    if (trace) std::fprintf(stderr, "[FAULTTRACE] %s\n", tr.c_str());
}
void minicpm_v_4_7_npu::Impl::fault_before_lm_head(){
    if (this->fault_dummy && fault_hit('H')) (*this->fault_dummy)();
}
#endif

buffer<bf16> minicpm_v_4_7_npu::Impl::forward(int ids){
#if MINICPM_V_DECODE_ON_NPU
#if MINICPM_V_DECODE_PROFILE
    static double t_embed = 0, t_layers = 0, t_lmstart = 0, t_setctx = 0, t_lmwait = 0, t_sync = 0;
    static int n_tok = 0;
    auto _t = std::chrono::steady_clock::now();
    #define MINICPM_V_TICK(acc) { acc += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - _t).count(); \
                                  _t = std::chrono::steady_clock::now(); }
#else
    #define MINICPM_V_TICK(acc) do {} while (0)
#endif
    this->embed_forward(ids, this->x);
    this->x.sync_to_device();
    MINICPM_V_TICK(t_embed);
#if FLM_MINICPM_RECONFIG_SUPPORTED
    if (!this->npu->is_preemption_enabled() && this->dec_sc_usable()) return this->forward_single_ctx();
#endif

#if MINICPM_V_DECODE_PROFILE
    static const bool per_layer = std::getenv("FLM_MCV_DEC_PERLAYER") != nullptr;
    static double t_kind[4] = {0,0,0,0}; // set_rtp, first linear, linear, attn
    static int n_kind[4] = {0,0,0,0};
    static double t_rl = 0; static int n_rl = 0;
#endif
    if (!this->npu->is_preemption_enabled()){
#if MINICPM_V_DECODE_PROFILE
        if (per_layer){
            for (size_t r = 0; r < this->prof_runs.size(); r++){
                int kind = 0;
                if (r > 0){
                    int i = (int)r - 1;
                    kind = is_attn_layer(i) ? 3 : (is_first_linear_layer(i) ? 1 : 2);
                }
                auto a = std::chrono::steady_clock::now();
                this->prof_runs[r].start();
                flm::npu::wait_for_completion(this->prof_runs[r], "minicpm profile run");
                t_kind[kind] += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count();
                n_kind[kind]++;
            }
        }
        else
#endif
        {
        // Witness for foreign NPU work: the residual stream stays on the array across the runs of a layer group, so another process's command
        // between two of them makes the firmware reload this design and the token comes out as garbage (never an error). Refuse instead.
        // The stateless design (layer_lm_sl.xclbin) does not need the witness: its tokens are correct whatever runs in between; it stays available
        // through an explicit FLM_MINICPM_NPU_GUARD=1 (refuse instead of sharing the NPU).
        const bool guarded = this->lm_stateless ? minicpm_reconfig::guard_forced() : minicpm_reconfig::guard_enabled();
        const minicpm_reconfig::ForeignSig g0 = guarded ? minicpm_reconfig::foreign_sig() : minicpm_reconfig::ForeignSig{};
#ifdef FLM_MINICPM_FAULT_RELOAD
        if (this->fault_dummy) this->fault_layers(); else
#endif
        {
        this->layers_run.execute();
        this->wait_layers.start();
        this->wait_layers.wait(this->layers_run, "layers_run");
        }
        if (guarded && g0.ok){
            const minicpm_reconfig::ForeignSig g1 = minicpm_reconfig::foreign_sig();
            if (g1.ok && g1.h != g0.h){
                header_print_r("WARNING", "minicpm_v: another process ran on the NPU during a decode token (FLM_MINICPM_NPU_GUARD=0 disables this check)");
                throw std::runtime_error("minicpm_v: another process used the NPU while a decode token was running, so the array's state was lost; "
                                         "the request is aborted instead of returning a corrupted answer");
            }
        }
        }
    }
    else{
        set_layer_rtp();
        for (uint32_t i = 0; i < desc.num_hidden_layers; i++){
            if (is_attn_layer(i)){
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

    MINICPM_V_TICK(t_layers);
    this->x.sync_to_device();
#ifdef FLM_MINICPM_FAULT_RELOAD
    this->fault_before_lm_head();
#endif
    this->lm_head_run.start();
    this->wait_lm.start();
    MINICPM_V_TICK(t_lmstart);
    this->set_context_length(this->current_context_length + 1, this->general_token_idx + 1);
    MINICPM_V_TICK(t_setctx);
#ifdef FLM_MINICPM_FAULT_RUNSTATE
    this->fault_runstate("d");
#endif
    this->wait_lm.wait(this->lm_head_run, "lm_head_run");
    MINICPM_V_TICK(t_lmwait);
    this->logits.sync_from_device();
#if MINICPM_V_DECODE_PROFILE
    MINICPM_V_TICK(t_sync);
    // set_ctx sits between lm_head's start() and wait(), so it is free until it exceeds
    // the lm_head run; lm_start + set_ctx + lm_wait is the lm_head's true cost.
    if (++n_tok % 32 == 0){
        if (per_layer){
            fprintf(stderr, "[minicpm_v decode/perlayer] set_rtp %.3f ms/run  first_lin %.3f  lin %.3f  attn %.3f  (ms per run; per token: %.3f)\n",
                    t_kind[0]/std::max(1,n_kind[0]), t_kind[1]/std::max(1,n_kind[1]), t_kind[2]/std::max(1,n_kind[2]), t_kind[3]/std::max(1,n_kind[3]),
                    (t_kind[0]+t_kind[1]+t_kind[2]+t_kind[3])/n_tok);
        }
        fprintf(stderr, "[minicpm_v decode] %d tok, avg ms/token: embed %.3f  layers %.3f  lm_head.start %.3f"
                        "  set_context %.3f  lm_head.wait %.3f  logits sync %.3f  total %.3f\n",
                n_tok, t_embed / n_tok, t_layers / n_tok, t_lmstart / n_tok, t_setctx / n_tok,
                t_lmwait / n_tok, t_sync / n_tok,
                (t_embed + t_layers + t_lmstart + t_setctx + t_lmwait + t_sync) / n_tok);
    }
#endif
#undef MINICPM_V_TICK
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

/// Accuracy-suite taps (inert unless the env var is set; one getenv per prefill / image encode).
///   FLM_TF_HIDDEN_FILE  appends, per prefill call: "TFH1", int32 L, int32 D, int32 ids[L], bf16 rows[L][D]
///                       (the residual stream after the last layer, i.e. the final norm's input, for EVERY prompt row)
///   FLM_TF_IMGEMB_FILE  appends, per image encode: "TFE1", int32 rows, int32 D, bf16 rows[rows][D] (the tower's output rows)
static void tf_append(const char* path, const char tag[4], const std::vector<int>* ids, int rows, int D, const bf16* data){
    FILE* f = std::fopen(path, "ab");
    if (!f) return;
    std::fwrite(tag, 1, 4, f);
    int32_t hdr[2] = {(int32_t)rows, (int32_t)D};
    std::fwrite(hdr, sizeof(int32_t), 2, f);
    if (ids) std::fwrite(ids->data(), sizeof(int), (size_t)rows, f);
    std::fwrite(data, sizeof(bf16), (size_t)rows * (size_t)D, f);
    std::fclose(f);
}

/// FLM_TF_PIXELS_FILE: per image encode, appends "TFP1", int32 n_views, {int32 grid_h, int32 grid_w} x n_views, int32 patch_dim, bf16 patches[total][patch_dim]
static void tf_append_pixels(const char* path, const std::vector<std::pair<int,int>>& grids, int patch_dim, const bf16* data, size_t n_values){
    FILE* f = std::fopen(path, "ab");
    if (!f) return;
    std::fwrite("TFP1", 1, 4, f);
    int32_t nv = (int32_t)grids.size(); std::fwrite(&nv, 4, 1, f);
    for (auto& g : grids){ int32_t gg[2] = {g.first, g.second}; std::fwrite(gg, 4, 2, f); }
    int32_t pd = patch_dim; std::fwrite(&pd, 4, 1, f);
    std::fwrite(data, sizeof(bf16), n_values, f);
    std::fclose(f);
}

buffer<bf16> minicpm_v_4_7_npu::Impl::prefill(std::vector<int>& ids, void* payload){
    return this->_prefill_with_mm(ids, payload);
}

/// @note  `payload` is a minicpm_v_image_payload_t* or null. Its `images` are VIEWS,
///        not pictures: the app's slicer turns one picture into a source view plus up
///        to max_slice_nums slices, each with its own patch grid, and `_data__processed`
///        holds every view's patches back to back -- raster order within a view, each
///        patch flattened (c, kh, kw), 3*14*14 bf16. That is the HF processor's NaViT
///        `[3, 14, T*14]` tensor transposed per patch.
///
///        The tower's rows overwrite the image_token_id rows in order, bypassing the
///        embedding table (ADDING_A_NEW_MODEL.md S7). Positions are MiniCPM's canvas
///        m-rope (image/mrope.hpp) whenever images are present.
buffer<bf16> minicpm_v_4_7_npu::Impl::_prefill_with_mm(std::vector<int>& ids, void* payload){
    int L_in = ids.size();
    const auto* p = static_cast<const minicpm_v_image_payload_t*>(payload);
    const bool has_images = p != nullptr && !p->images.empty();

    // batch the page-ins of the prompt's rows if the mapped table is cold; the kernel reads
    // ahead while the vision tower (if any) runs
    this->embed_prefetch(ids.data(), (size_t)L_in);

    int n_image_tokens = 0;
    for (int id : ids){
        if (id == video_token_id){
            throw std::runtime_error("the prompt carries video tokens (" + std::to_string(video_token_id) +
                                     "): this engine has no video path");
        }
        n_image_tokens += (id == canvas_ids.image_pad);
    }

    std::vector<mcv::view_grid> views;
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

    std::vector<int> input_idx(L_in * 3);
    int next_position;
    if (has_images){
        next_position = mcv::canvas_mrope(ids.data(), L_in, canvas_ids, views, general_token_idx, input_idx.data());
    }
    else {
        for (int l = 0; l < L_in; l++){
            input_idx[l * 3 + 0] = input_idx[l * 3 + 1] = input_idx[l * 3 + 2] = general_token_idx + l;
        }
        next_position = general_token_idx + L_in;
    }

    // FLM_DUMP_DIR scoring hook, DEBUG_LEVEL >= 1 only; decode never dumps
    minicpm_v_cpu::dumper().active = true;
    MINICPM_V_DUMP("embed_tokens", hidden_states.data(), L_in, desc.D);

    time_utils::time_point start = time_utils::now();
    // FLM_MINICPM_V_PREFILL_PROFILE splits prefill by layer kind. The two kinds have very
    // different backends -- the GatedDeltaNet layers carry conv1d + the delta rule, the
    // attention layers carry the mha overlay -- so which one dominates is the first thing
    // worth knowing. For a finer split inside a layer, build with -DMINICPM_V_PREFILL_PROFILE
    // (minicpm_v_prefill.hpp).
    const bool profile_layers = std::getenv("FLM_MINICPM_V_PREFILL_PROFILE") != nullptr;
    // the attention layers write rows [current_context_length, +L_in) of their K/V cache below
    kv_hwm_rows = std::max(kv_hwm_rows, this->current_context_length + L_in);
    double t_attn = 0, t_linear = 0;
    int n_attn = 0, n_linear = 0;
    for (int layer_idx = 0; layer_idx < desc.num_hidden_layers; layer_idx++) {
        time_utils::time_point t0 = time_utils::now();
        pctx->forward(layer_idx, hidden_states, L_in, proj_weights[layer_idx], rms_weights[layer_idx], rope_rms_weights[layer_idx], kv_caches[layer_idx], input_idx, alpha_proj_cpu[layer_idx], beta_proj_cpu[layer_idx], this->current_context_length);
        if (profile_layers){
            const bool attn = is_attn_layer(layer_idx);
            double ms = time_utils::duration_ms(t0, time_utils::now()).first;
            (attn ? t_attn : t_linear) += ms;
            (attn ? n_attn : n_linear)++;
            fprintf(stderr, "[minicpm_v prefill] L=%d layer %2d %-6s %7.2f ms\n", L_in, layer_idx,
                    attn ? "attn" : "linear", ms);
        }
    }
    if (profile_layers){
        fprintf(stderr, "[minicpm_v prefill] L=%d  %d attn %.1f ms  %d linear %.1f ms\n",
                L_in, n_attn, t_attn, n_linear, t_linear);
    }

    time_utils::time_point t_layers = time_utils::now();

    if (const char* tf = std::getenv("FLM_TF_HIDDEN_FILE")){
        tf_append(tf, "TFH1", &ids, L_in, desc.D, hidden_states.data());
    }

    buffer<bf16> predict = hidden_states_2d[L_in - 1];
    // lm_head_weights are synced once, when the loader fills them; the prefill used to
    // flush all 271 MB again before every get_logits (~2.4 ms per request)
    time_utils::time_point t_lmsync = time_utils::now();
    get_logits(predict);
    // only the last row's logits exist; score against logits.npy[-1]
    MINICPM_V_DUMP("logits_last", logits_valid.data(), 1, desc.vocab_size);
    minicpm_v_cpu::dumper().active = false;
    // VISION_PROFILE reports it next to the tower's own line, for TTFT work; PREFILL_PROFILE
    // reports it for text-only prompts too, which is the only way to time a text prefill from
    // the CLI (`flm run` prints no profile; only the test harness does).
    if ((has_images && std::getenv("FLM_MINICPM_V_VISION_PROFILE")) || profile_layers){
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

std::vector<bf16> minicpm_v_4_7_npu::Impl::_encode_images(const minicpm_v_image_payload_t* p, int n_image_tokens,
                                                      std::vector<mcv::view_grid>& views){
    if (!image_mod){
        throw std::runtime_error("images supplied but no vision tower is loaded -- either the config has no "
                                 "vision_model_weight or FLM_MINICPM_V_VISION=0");
    }
    const size_t patch_dim = vision_desc.patch_dim();
    size_t patches = 0;
    int rows = 0;
    for (const minicpm_v_image_t& im : p->images){
        views.push_back(mcv::view_grid{im.grid_h, im.grid_w});
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
    if (const char* tf = std::getenv("FLM_TF_PIXELS_FILE")){
        std::vector<std::pair<int,int>> grids;
        for (const minicpm_v_image_t& im : p->images) grids.push_back({(int)im.grid_h, (int)im.grid_w});
        tf_append_pixels(tf, grids, (int)patch_dim, p->_data__processed.data(), p->_data__processed.size());
    }
    std::vector<float> pixels(p->_data__processed.size());
    mcv::simd::bf16_to_f32(pixels.data(), p->_data__processed.data(), pixels.size());
    std::vector<float> out((size_t)rows * desc.D);
    mcv::vision::prof::reset();
    image_mod->encode(pixels.data(), views, out.data());
    // per-stage table; compiled out unless built with VISION_PROFILE=1
    mcv::vision::prof::report(time_utils::duration_ms(t0, time_utils::now()).first);

    std::vector<bf16> image_rows(out.size());
    mcv::simd::f32_to_bf16(image_rows.data(), out.data(), out.size());
    if (const char* tf = std::getenv("FLM_TF_IMGEMB_FILE")){
        tf_append(tf, "TFE1", nullptr, rows, (int)desc.D, image_rows.data());
    }
    if (std::getenv("FLM_MINICPM_V_VISION_PROFILE")){
        header_print_r("FLM", "vision: " + std::to_string(views.size()) + " views, " + std::to_string(patches) +
                              " patches -> " + std::to_string(rows) + " rows in " +
                              std::to_string((int)time_utils::duration_ms(t0, time_utils::now()).first) + " ms (" +
                              image_mod->backend() + ") hash " + [&]{
                                  uint64_t h = 1469598103934665603ull;   // FNV-1a over the fp32 bits: A/B identity checks
                                  for (float f : out){ uint32_t u; std::memcpy(&u, &f, 4); h = (h ^ u) * 1099511628211ull; }
                                  char b[24]; std::snprintf(b, sizeof b, "%016llx", (unsigned long long)h); return std::string(b); }());
    }
    return image_rows;
}

buffer<bf16> minicpm_v_4_7_npu::Impl::get_k_cache(int layer_idx, int idx){
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
buffer<bf16> minicpm_v_4_7_npu::Impl::get_v_cache(int layer_idx, int idx){
    this->kv_caches[layer_idx].sync_from_device();
    buffer<bf16> v_cache(desc.DV);
    bf16* v_cache_ptr = this->kv_caches[layer_idx].data();
    size_t offset = this->V_offset + (size_t)idx * desc.DV;
    memcpy(static_cast<void*>(v_cache.data()), static_cast<void*>(v_cache_ptr + offset), desc.DV * sizeof(bf16));
    return v_cache;
}

#if MINICPM_V_DECODE_ON_NPU
void minicpm_v_4_7_npu::Impl::make_lm_head_run(){
    if (this->lm_in_layer){
        // the lm_head is a run of the layer design: x, lm_head weights, final norm weight, (unused), logits
        this->lm_head_run = lm_head.create_run(this->x, this->lm_head_weights, this->lm_head_norm, this->lm_head_norm, this->logits);
    }
    else{
        this->lm_head_run = lm_head.create_run(this->logits, this->lm_head_weights, this->x);
    }
}
#endif

void minicpm_v_4_7_npu::Impl::npu_recover(const char* where){
#if MINICPM_V_DECODE_ON_NPU
    static const bool off = [] { const char* e = std::getenv("FLM_MINICPM_RECOVER"); return e && std::atoi(e) == 0; }();
    if (off) return;
    try{
#if FLM_MINICPM_RECONFIG_SUPPORTED
        this->dec_to_lm.release();
        this->dec_to_layer.release();
        if (this->dec_rl_ready){
            this->dec_rl1.reset();
            this->dec_rl2.reset();
        }
#endif
        if (!this->npu->is_preemption_enabled()) this->layers_run.reset();
        // The persistent run may still be in flight (an exception between start() and wait()): XRT aborts the process when a run object is destroyed
        // while its command is in progress. It is waited for first (XRT builds) and then parked instead of destroyed.
        static std::vector<flm_rt::run> parked;
#if !defined(FLM_USE_HRX)
        try{
            const ert_cmd_state st = this->lm_head_run.state();
            if (st == ERT_CMD_STATE_QUEUED || st == ERT_CMD_STATE_RUNNING || st == ERT_CMD_STATE_SUBMITTED) flm::npu::wait_for_completion(this->lm_head_run, "lm_head_run");
        } catch (const std::exception&){}
#endif
        parked.push_back(std::move(this->lm_head_run));
        this->make_lm_head_run();
        this->npu_recoveries++;
        header_print_r("WARNING", "minicpm_v: a request failed in " << where << "; the persistent NPU runs were recreated (recovery " << this->npu_recoveries << "), the next request starts clean");
    } catch (const std::exception& e){
        header_print_r("WARNING", "minicpm_v: recovery after a failed request (" << where << ") failed: " << e.what());
    }
#else
    (void)where;
#endif
}

buffer<bf16> minicpm_v_4_7_npu::Impl::get_logits(buffer<bf16>& predict){
    memcpy(this->x.data(), predict.data(), desc.D * sizeof(bf16));
    this->x.sync_to_device();
#if MINICPM_V_DECODE_ON_NPU
    this->lm_head_run.start();
#ifdef FLM_MINICPM_FAULT_RUNSTATE
    this->fault_runstate("p");
#endif
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
void minicpm_v_4_7_npu::Impl::_lm_head_host(){
    buffer<bf16> normed(desc.D);
    cpu_func::_rms_norm(normed.data(), this->x.data(), this->x.data() + desc.D, desc.D, 1, 0, 0);
    MINICPM_V_DUMP("final_norm_last", normed.data(), 1, desc.D);
    minicpm_v_cpu::dequant_mm(this->logits.data(), desc.vocab_size_padded, this->lm_head_weights.data(),
                              normed.data(), desc.D, 1, desc.D, desc.vocab_size, minicpm_v_cpu::LM_HEAD_LAYOUT);
    this->logits.sync_to_device();
}

void minicpm_v_4_7_npu::Impl::_allocate_checkpoint_buffers() {
    this->cache_checkpoint.clear();
    this->cache_checkpoint.resize(desc.num_hidden_layers);
    is_checkpoint_valid = false;
    for (int i = 0; i < desc.num_hidden_layers; i++) {
        if (!is_attn_layer(i)) {
            this->cache_checkpoint[i] = buffer<bf16>(this->kv_caches[i].size());
        }
    }
}

int minicpm_v_4_7_npu::Impl::checkpoint() {
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

int minicpm_v_4_7_npu::Impl::restore() {
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
    if (pctx) pctx->kv_cache_changed(this->checkpoint_context_length);
    this->current_context_length = this->checkpoint_context_length;
    this->general_token_idx = this->checkpoint_general_token_idx;
    this->set_context_length(this->current_context_length, this->general_token_idx);
    return this->current_context_length;
}

void minicpm_v_4_7_npu::Impl::clear_context(){
    this->set_context_length(0, 0);
    // What a fresh context needs: every attention K/V row zero (the kernels do not rely on
    // the length alone to keep rows out of attention, see write_state), and the recurrent
    // layers' conv + state slots zero. The attention buffers are 64 MB each, and only rows
    // below the high-water mark of what any request wrote can be non-zero, so zero and flush
    // just those; the rest is zero on both sides since the allocation. Nothing is read back:
    // everything flushed here is overwritten on the host first.
    const size_t rows = (size_t)std::min<int>(kv_hwm_rows, (int)(this->MAX_L));
    for (int i = 0; i < desc.num_hidden_layers; i++){
        buffer<bf16>& kv = this->kv_caches[i];
        if (!is_attn_layer(i)){
            memset(kv.data(), 0, kv.size() * sizeof(bf16));
            kv.sync_to_device();
            continue;
        }
        const size_t half = kv.size() / 2;
#if defined(FLM_USE_HRX)
        // HRX flushes whole buffers lazily and would clobber rows the device wrote: keep the full round trip
        kv.sync_from_device();
        memset(kv.data(), 0, kv.size() * sizeof(bf16));
        kv.sync_to_device();
        (void)rows; (void)half;
#else
        if (rows == 0) continue;
        memset(kv.data(), 0, rows * desc.DK * sizeof(bf16));
        memset(kv.data() + half, 0, rows * desc.DV * sizeof(bf16));
        mcv_npu::sync_to_device_range(kv, 0, rows * desc.DK);
        mcv_npu::sync_to_device_range(kv, half, rows * desc.DV);
#endif
    }
    kv_hwm_rows = 0;
    if (pctx) pctx->kv_cache_changed(0);
}

size_t minicpm_v_4_7_npu::Impl::state_row_bytes() {
    size_t attention_layers = 0;
    for (int i = 0; i < desc.num_hidden_layers; i++) {
        attention_layers += is_attn_layer(i) ? 1 : 0;
    }
    return attention_layers * (static_cast<size_t>(desc.DK) + desc.DV) * sizeof(bf16);
}

size_t minicpm_v_4_7_npu::Impl::state_fixed_bytes() {
    size_t bytes = 0;
    for (int i = 0; i < desc.num_hidden_layers; i++) {
        if (!is_attn_layer(i)) {
            bytes += this->kv_caches[i].size() * sizeof(bf16);
        }
    }
    return bytes;
}

std::unique_ptr<hybrid_engine_state> minicpm_v_4_7_npu::Impl::new_state() {
    return std::make_unique<hybrid_engine_state>(state_fixed_bytes(), state_row_bytes());
}

int minicpm_v_4_7_npu::Impl::read_state(hybrid_engine_state& st, int kv_from) {
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
static bool minicpm_v_state_fits(const hybrid_engine_state& st, const std::vector<buffer<bf16>>& kv_caches, int num_layers, int DK, int DV, int full_attention_interval) {
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

int minicpm_v_4_7_npu::Impl::write_state_kv(const hybrid_engine_state& st) {
    if (!minicpm_v_state_fits(st, this->kv_caches, desc.num_hidden_layers, desc.DK, desc.DV, full_attention_interval)) {
        return -1;
    }
    const size_t rows = static_cast<size_t>(st.context_length - st.kv_from);
    if (rows == 0) {
        return st.context_length;
    }
    if (pctx) pctx->kv_cache_changed(static_cast<int>(st.kv_from));
    for (int i = 0; i < desc.num_hidden_layers; i++) {
        if (is_attn_layer(i)) {
            this->kv_caches[i].sync_from_device();
            memcpy(this->kv_caches[i].data() + static_cast<size_t>(st.kv_from) * desc.DK, st.kv_k[i].data(), rows * desc.DK * sizeof(bf16));
            memcpy(this->kv_caches[i].data() + (this->kv_caches[i].size() / 2) + static_cast<size_t>(st.kv_from) * desc.DV, st.kv_v[i].data(), rows * desc.DV * sizeof(bf16));
            this->kv_caches[i].sync_to_device();
        }
    }
    return st.context_length;
}

int minicpm_v_4_7_npu::Impl::write_state(const hybrid_engine_state& st) {
    if (!minicpm_v_state_fits(st, this->kv_caches, desc.num_hidden_layers, desc.DK, desc.DV, full_attention_interval)) {
        header_print_r("FLM", "No valid state to write");
        return -1;
    }
    header_print_r("FLM", "Writing state at context length " + std::to_string(st.context_length));
    const size_t rows = static_cast<size_t>(st.context_length - st.kv_from);
    if (pctx) pctx->kv_cache_changed(static_cast<int>(st.kv_from));
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
    this->current_context_length = st.context_length;
    this->general_token_idx = st.general_token_idx;
    this->set_context_length(this->current_context_length, this->general_token_idx);
    return this->current_context_length;
}

int minicpm_v_4_7_npu::Impl::get_current_context_length(){
    return this->current_context_length;
}

void minicpm_v_4_7_npu::Impl::update_max_length(uint32_t MAX_L){
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
    if (pctx) pctx->kv_cache_changed(0);   // the bfp16 copies of the keys / values are rebuilt from the migrated rows
    this->set_context_length(this->current_context_length, this->general_token_idx); // regenerate sequence
    is_checkpoint_valid = false; // invalidate checkpoint since the underlying kv cache has changed
}
///@brief destructor of minicpm_v_4_7_npu
minicpm_v_4_7_npu::Impl::~Impl() = default;

// ===============================================
// Externals for minicpm_v_4_7_npu
// ===============================================
minicpm_v_4_7_npu::minicpm_v_4_7_npu(LM_Config config, npu_xclbin_manager *npu_instance, int MAX_L){

    this->load_vision_preprocess_parameters(config);
    this->_impl = new Impl(config, npu_instance, this,   MAX_L);
}

buffer<bf16> minicpm_v_4_7_npu::forward(int ids){
    try {
        return this->_impl->forward(ids);
    }
    catch (const std::exception&) {
        this->_impl->npu_recover("decode");
        throw;
    }
}

buffer<bf16> minicpm_v_4_7_npu::prefill(std::vector<int>& ids, void* payload){
    try {
        // On the one persistent host thread (host_executor.hpp): the OpenMP team the host stages and the
        // vision tower use then exists once, not once per FLM I/O thread.
        buffer<bf16> result;
        mcv::HostExecutor::instance().run([&] { result = this->_impl->prefill(ids, payload); });
        return result;
    }
    catch (const std::exception& e) {
        header_print("FLM", e.what());
        this->_impl->npu_recover("prefill");
        throw;
    }
}

void minicpm_v_4_7_npu::set_context_length(int L){
    std::cout << "Setting context length is not supported for minicpm_v_4_7_npu!" << std::endl;
}

void minicpm_v_4_7_npu::load_weights(Q4NX& q4nx){
    this->_impl->load_weights(q4nx);
}

void minicpm_v_4_7_npu::update_max_length(uint32_t MAX_L){
    this->_impl->update_max_length(MAX_L);
}

void minicpm_v_4_7_npu::clear_context(){
    this->_impl->clear_context();
}

buffer<bf16> minicpm_v_4_7_npu::get_k_cache(int layer_idx, int idx){
    return this->_impl->get_k_cache(layer_idx, idx);
}

buffer<bf16> minicpm_v_4_7_npu::get_v_cache(int layer_idx, int idx){
    return this->_impl->get_v_cache(layer_idx, idx);
}

int minicpm_v_4_7_npu::get_current_context_length(){
    return this->_impl->get_current_context_length();
}

int minicpm_v_4_7_npu::checkpoint(){
    return this->_impl->checkpoint();
}

int minicpm_v_4_7_npu::restore(){
    return this->_impl->restore();
}

std::unique_ptr<hybrid_engine_state> minicpm_v_4_7_npu::new_state(){
    return this->_impl->new_state();
}

int minicpm_v_4_7_npu::read_state(hybrid_engine_state& state, int kv_from){
    return this->_impl->read_state(state, kv_from);
}

int minicpm_v_4_7_npu::write_state(const hybrid_engine_state& state){
    return this->_impl->write_state(state);
}

int minicpm_v_4_7_npu::write_state_kv(const hybrid_engine_state& state){
    return this->_impl->write_state_kv(state);
}

extern "C" const flm_state_api* minicpm_v_4_7_state_api(void* engine){
    return hybrid_state_api<minicpm_v_4_7_npu>(engine);
}

minicpm_v_4_7_npu::~minicpm_v_4_7_npu(){
    delete this->_impl;
}
