// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#include <cstdlib>
#include <chrono>
#include "gemma4e_prof.hpp"
#include <omp.h>
#include "flm_override.hpp"
#include "gemma4e_dflash_flash_detail.hpp"
#include "metrices.hpp"
#include <chrono>
#include "utils/error_measure.hpp"
#include "mmRuntimeSequence.hpp"

gemma4e_dflash_flash::Impl::Impl(LM_Config config, npu_xclbin_manager *npu_instance, gemma4e_dflash_flash* parent_ptr, int MAX_L )
    : config(config), npu(npu_instance), parent_ptr(parent_ptr){
    FLM_OVERRIDE(engine_init, (void)0, this->npu, this->config);
    is_vlm = config.get<bool>("is_vlm", false);
    is_audio = config.get<bool>("is_audio", false);

    current_context_length = 0;

    // This engine is pinned to a 1 k context: the requested length is ignored and
    // update_max_length() below refuses to grow it. The prefill scratch is cut for
    // whatever chunk arrives, and on a machine whose npu carve-out has no headroom
    // a chunk longer than 1 k fails to create its bos -- capping the context is
    // what keeps a longer chunk from ever being asked for.
    static constexpr uint32_t FORCED_MAX_L = 1024;
    this->MAX_L = FORCED_MAX_L;
    MAX_L = this->MAX_L; // synchronize MAX_L
    is_preload_launched = false;


    try{
        this->desc.build(this->config);

        D                       = desc.D;
        vocab_size              = desc.vocab_size;
        vocab_size_padded       = desc.vocab_size_padded;
        DH                      = desc.DH;
        DQ                      = desc.DQ;
        DK                      = desc.DK;
        DV                      = desc.DV;
        SWA_DH                  = desc.SWA_DH;
        SWA_DQ                  = desc.SWA_DQ;
        SWA_DK                  = desc.SWA_DK;
        SWA_DV                  = desc.SWA_DV;
        num_hidden_layers       = desc.num_hidden_layers;
        SLIDING_LENGTH          = desc.SLIDING_LENGTH;
        PLI_D                   = desc.PLI_D;
        num_kv_shared_layers    = desc.num_kv_shared_layers;
        final_logit_softcapping = desc.final_logit_softcapping;
        non_skip_layers         = desc.non_skip_layers;
        INTERMEDIATE_SIZE       = desc.INTERMEDIATE_SIZE;
        enable_double_wide_mlp  = desc.enable_double_wide_mlp;
        global_layer_period     = desc.global_layer_period;
        layer_types             = desc.layer_types;

        if (this->is_global_layer_idx(non_skip_layers - 1)){
            last_global_kv_cache_layer_idx = non_skip_layers - 1;
            last_swa_kv_cache_layer_idx = non_skip_layers - 2;
        }
        else {
            last_swa_kv_cache_layer_idx = non_skip_layers - 1;
            last_global_kv_cache_layer_idx = (last_swa_kv_cache_layer_idx / global_layer_period) * global_layer_period - 1;
        }
        DEBUG_BLOCK(1,
        header_print_g("info", "Gemma4e NPU config:");
        std::cout << "\tD: " << D << std::endl;
        std::cout << "\tPLI_D: " << PLI_D << std::endl;
        std::cout << "\tDH: " << DH << std::endl;
        std::cout << "\tDQ: " << DQ << std::endl;
        std::cout << "\tDK: " << DK << std::endl;
        std::cout << "\tDV: " << DV << std::endl;
        std::cout << "\tSWA_DH: " << SWA_DH << std::endl;
        std::cout << "\tSWA_DQ: " << SWA_DQ << std::endl;
        std::cout << "\tSWA_DK: " << SWA_DK << std::endl;
        std::cout << "\tSWA_DV: " << SWA_DV << std::endl;
        std::cout << "\tSLIDING_LENGTH: " << SLIDING_LENGTH << std::endl;
        std::cout << "\tINTERMEDIATE_SIZE: " << INTERMEDIATE_SIZE << std::endl;
        std::cout << "\tFINAL_LOGIT_SOFTCAPPING: " << final_logit_softcapping << std::endl;
        std::cout << "\tvocab_size: " << vocab_size << " (" << vocab_size_padded << " padded)" << std::endl;
        std::cout << "\tnum_hidden_layers: " << num_hidden_layers << std::endl;
        std::cout << "\tnum_kv_shared_layers: " << num_kv_shared_layers << std::endl;
        std::cout << "\tnon_skip_layers: " << non_skip_layers << std::endl;
        std::cout << "\tglobal_layer_period: " << global_layer_period << std::endl;
        std::cout << "\tlast_swa_kv_cache_layer_idx: " << last_swa_kv_cache_layer_idx << std::endl;
        std::cout << "\tlast_global_kv_cache_layer_idx: " << last_global_kv_cache_layer_idx << std::endl;

        )
        DEBUG_BLOCK(2,
        std::cout << "\tlayer_types: \n";
        for (int i = 0; i < num_hidden_layers; i++){
            std::string type_str;
            switch (layer_types[i]){
                case e_gemma4e_swa_layer:
                    type_str = "SWA";
                    break;
                case e_gemma4e_global_layer:
                    type_str = "Global";
                    break;
                case e_gemma4e_swa_layer_skip:
                    type_str = "SWA_Skip";
                    break;
                case e_gemma4e_global_layer_skip:
                    type_str = "Global_Skip";
                    break;
                default:
                    type_str = "Unknown";
            }
            std::cout << "\t\t layer " << i << ": " << type_str << " " << std::endl;
        }
        std::cout << std::endl;
        )
    }
    catch (std::exception& e){
        header_print_r("ERROR", "Failed to parse model config: " << e.what());
        throw e;
    }

    gemma4e_seq_gen_parameters_t seq_gen_params = {
        .D = D,
        .DH = DH,
        .DQ = DQ,
        .DK = DK,
        .DV = DV,
        .SWA_DH = SWA_DH,
        .SWA_DQ = SWA_DQ,
        .SWA_DK = SWA_DK,
        .SWA_DV = SWA_DV,
        .PLI_D = PLI_D,
        .INTERMEDIATE_SIZE = INTERMEDIATE_SIZE,
        .NUM_ATTENTION_HEADS = (int)config.get("num_attention_heads"),
        .NUM_KEY_VALUE_HEADS = (int)config.get("num_key_value_heads"),
        .SLIDING_WINDOW_SIZE = SLIDING_LENGTH,
        .VOCAB_SIZE_PADDED = vocab_size_padded,
        .enable_double_wide_mlp = enable_double_wide_mlp
    };
    this->sequence = std::make_unique<gemma4e_dflash_flash_sequence>(seq_gen_params, MAX_L);
    // Every sequence below addresses weights through the descriptors, so bind the
    // description before generating any of them.
    this->sequence->set_desc(&this->desc);

    DEBUG_BLOCK(1,
        header_print_g("info", "VLM Enabled: " << (is_vlm ? "Yes" : "No"));
        header_print_g("info", "Audio Model Enabled: " << (is_audio ? "Yes" : "No"));
    );

    if (is_vlm){
        this->gemma4e_image_encoder = std::make_unique<Gemma4e_ImageEncoder>(config, npu, this->parent_ptr);
    }
    if(is_audio){
        this->gemma4e_audio_encoder = std::make_unique<Gemma4e_AudioEncoder>(config, npu, this->parent_ptr);
    }
    layer_app_manager = npu->register_xclbin(utils::path_join(config.exec_path, "xclbins", config.model_name, "layer.xclbin"));

    this->global_layer = layer_app_manager->create_app();
    this->swa_layer = layer_app_manager->create_app();
    this->global_skip_layer = layer_app_manager->create_app();
    this->swa_skip_layer = layer_app_manager->create_app();
    this->layer_pre_load = layer_app_manager->create_app();

    if (!this->npu->is_preemption_enabled()){
        this->layers_run = layer_app_manager->create_runlist();
    }

    lm_head_app_manager = npu->register_xclbin(utils::path_join(config.exec_path, "xclbins", config.model_name, "lm_head.xclbin"));
    this->lm_head = lm_head_app_manager->create_app();

    // allocate all buffers
    rms_weights.resize(num_hidden_layers);
    proj_weights.resize(num_hidden_layers);
    pli_gate_up_weights.resize(num_hidden_layers);
    kv_caches.resize(non_skip_layers); // only allocate kv_cache for non-skip layers
    attn_mask.resize(non_skip_layers);
    for (int i = 0; i < non_skip_layers; i++) attn_mask[i] = is_global_layer_idx(i);
    rope_rms_weights.resize(num_hidden_layers);
    layer_scale = buffer<float>(num_hidden_layers);

    this->x = layer_app_manager->create_bo_buffer<bf16>((3 * D + 8191) / 8192 * 8192); // Iterative Hidden States (D) + Final RMS NORM (D) + INITIAL EMBEDDING (D)

    for (int layer_idx = 0; layer_idx < num_hidden_layers; layer_idx++){
        gemma4e_layer_type_t type = layer_types[layer_idx];
        size_t proj_buffer_size = desc.get_proj_weights_byte_size(type);
        proj_weights[layer_idx] = layer_app_manager->create_bo_buffer<u8>(proj_buffer_size); // allocate 32MB for each layer, which is enough for current model scale. For larger model scale, we may need to dynamically load weights
        this->proj_weights[layer_idx].memset((uint8_t)0);
        this->proj_weights[layer_idx].sync_to_device();
        // packed for the fused overlay's bf16 passthrough, gate then up; the up
        // half is zero padded from PLI_D to the kernel's 512-column minimum
        this->pli_gate_up_weights[layer_idx] = layer_app_manager->create_bo_buffer<bf16>(desc.pli_gate_up_packed_elems());
        DEBUG_BLOCK(2,
        header_print("info", "Buffer sizes (in bytes): proj_weights=" + std::to_string(proj_weights[layer_idx].size()));
        )
    }

    for (int layer_idx = 0; layer_idx < num_hidden_layers; layer_idx++){
        rms_weights[layer_idx] = layer_app_manager->create_bo_buffer<bf16>(desc.get_rms_elems(layer_types[layer_idx])); // input_layer_norm (D) + post_attn_layer_norm (D) + pre_ffn_layer_norm (D) + post_ffn_layer_norm (D) for each layer from host.
    }
    for (int layer_idx = 0; layer_idx < num_hidden_layers; layer_idx++){
        gemma4e_layer_type_t type = layer_types[layer_idx];
        rope_rms_weights[layer_idx] = layer_app_manager->create_bo_buffer<bf16>(desc.get_rope_rms_elems(type)); // COS/SIN (_DH) + Q_NORM (_DH) + K_NORM (_DH) + PLI_EMBED (PLI_D) + PLI_NORM (PLI_D) + POST_PLI_NORM (D) + layer_scale
    }

    for (int layer_idx = 0; layer_idx < non_skip_layers; layer_idx++){
        gemma4e_layer_type_t type = layer_types[layer_idx];
        kv_caches[layer_idx] = layer_app_manager->create_bo_buffer<bf16>(is_swa_layer(type) ? swa_buffer_elems() : desc.get_kv_cache_size(type, MAX_L)); // sliding layers: linear window buffer (gemma4e_dflash_flash_swa.hpp)
        kv_caches[layer_idx].memset((bf16)0);
        kv_caches[layer_idx].sync_to_device();
        DEBUG_BLOCK(2,
        header_print("info", "Buffer sizes (in bytes): kv_cache for layer " + std::to_string(layer_idx) + " = " + std::to_string(kv_caches[layer_idx].size()));
        )
    }
    is_checkpoint_valid = false; // checkpoint is not valid until we load kv cache to device after resizing buffers
    pli_down_weights = layer_app_manager->create_bo_buffer<bf16>(desc.pli_down_packed_elems()); // down projection weights for per layer input, which will be added to the input of each layer after the first layer

    lm_head_weights = lm_head_app_manager->create_bo_buffer<u8>(desc.get_lm_head_w_size());
    logits = lm_head_app_manager->create_bo_buffer<bf16>(vocab_size_padded);
    logits_valid = buffer<bf16>(logits.data(), vocab_size);

    this->embedding = std::make_unique<embedding_q8_0>(vocab_size, D);
    this->pli_embedding = std::make_unique<embedding_q8_0>(vocab_size, PLI_D * num_hidden_layers); // per layer input embedding, which will be added to the input of each layer after the first layer

    // Everything below this point belongs to the prefill path: its own xclbins,
    // sequences, dequantized weights and scratch buffers.
    this->prefill_ctx = std::make_unique<gemma4e_prefill_context>(
        npu, &this->desc, this->config, this->sequence.get(), MAX_L
    );

    this->sequence->gen_lm_head_seq(this->lm_head.seq(), final_logit_softcapping);
    this->lm_head_run = FLM_OVERRIDE(lm_head,
        this->lm_head.create_run(this->logits, this->lm_head_weights, this->x));

    // empty seq for preload xclbin
    npu_sequence* pre_load_seq = this->layer_pre_load.seq();
    pre_load_seq->clear_cmds();
    pre_load_seq->cmds2seq();

    // this->sequence->gen_layer_seq(this->linear_layer.seq(), 0, false);
    this->spec_create();
}

void gemma4e_dflash_flash::Impl::_set_rope_rms_weights(int idx){

    buffer<bf16> cos_buf(DH / 2);
    buffer<bf16> sin_buf(DH / 2);
    buffer<bf16> swa_cos_buf(SWA_DH / 2);
    buffer<bf16> swa_sin_buf(SWA_DH / 2);

    for (int j = 0; j < DH / 2; j++){
        cos_buf[j] = (bf16)cos(gemma4e_cpu_func::inv_freq_global[j] * idx);
        sin_buf[j] = (bf16)sin(gemma4e_cpu_func::inv_freq_global[j] * idx);
    }
    for (int j = 0; j < SWA_DH / 2; j++){
        swa_cos_buf[j] = (bf16)cos(gemma4e_cpu_func::inv_freq_swa[j] * idx);
        swa_sin_buf[j] = (bf16)sin(gemma4e_cpu_func::inv_freq_swa[j] * idx);
    }

    for (int i = 0; i < num_hidden_layers; i++){
        bf16* w_rope_ptr = this->rope_rms_weights[i].data();
        if (is_swa_layer(layer_types[i])){
            memcpy(w_rope_ptr, swa_cos_buf.data(), SWA_DH * sizeof(bf16) / 2);
            memcpy(w_rope_ptr + SWA_DH / 2, swa_sin_buf.data(), SWA_DH * sizeof(bf16) / 2);
        }
        else {
            memcpy(w_rope_ptr, cos_buf.data(), DH * sizeof(bf16) / 2);
            memcpy(w_rope_ptr + DH / 2, sin_buf.data(), DH * sizeof(bf16) / 2);
        }
        // the cos / sin table is the buffer's first (DH or SWA_DH) elements; everything after it is static
        this->rope_rms_weights[i].sync_range_to_device(0, (is_swa_layer(layer_types[i]) ? SWA_DH : DH) * sizeof(bf16));
    }
}

void gemma4e_dflash_flash::Impl::set_context_length(int L){
    this->decode_seq_stale = false;
    this->current_context_length = L;
    this->dirty_rows = std::max(this->dirty_rows, static_cast<size_t>(L));
    DEBUG_BLOCK(2,
    header_print_r("info", "Setting context length to " + std::to_string(L) + " for all layers in the sequence");
    )
    // the next step appends row L - base: compact the sliding buffers first when it would not fit (between two steps, never inside a dispatch)
    if (swa_needs_compaction(L, this->swa_base)) this->swa_compact(L);
    this->sequence->swa_base = this->swa_base;
    this->sequence->gen_layer_seq(this->global_layer.seq(),      L + 1, e_gemma4e_global_layer);
    this->sequence->gen_layer_seq(this->swa_layer.seq(),         L + 1, e_gemma4e_swa_layer);
    this->sequence->gen_layer_seq(this->global_skip_layer.seq(), L + 1, e_gemma4e_global_layer_skip);
    this->sequence->gen_layer_seq(this->swa_skip_layer.seq(),    L + 1, e_gemma4e_swa_layer_skip);

    if (!this->npu->is_preemption_enabled()){
        const bool tap_mode = this->spec_tap_mode();
        this->layers_run.reset();
        if (tap_mode) for (auto& r : this->seg_runs) r.reset();
        auto run_list = [&](uint32_t layer) -> flm_rt::runlist& { return tap_mode ? this->seg_runs[this->spec_tap_segment(layer)] : this->layers_run; };

        DEBUG_BLOCK(2,
        header_print_r("info", "Create  runs for all layers in the sequence with the new context length");
        )
        for (uint32_t i = 0; i < num_hidden_layers; i++){

            DEBUG_BLOCK(2,
            header_print_r("info", "Generate run for layer " + std::to_string(i) + " of type " + std::to_string(layer_types[i]));
            )
            switch(layer_types[i]){
                case e_gemma4e_global_layer:
                    run_list(i).add(FLM_OVERRIDE(global_layer_run,
                        this->global_layer.create_run(this->x, this->proj_weights[i], this->rms_weights[i], this->rope_rms_weights[i], this->kv_caches[i]), layer_types[i], i, L, this->MAX_L));
                    break;
                case e_gemma4e_swa_layer:
                    run_list(i).add(FLM_OVERRIDE(swa_layer_run,
                        this->swa_layer.create_run(this->x, this->proj_weights[i], this->rms_weights[i], this->rope_rms_weights[i], this->kv_caches[i]), layer_types[i], i, L, this->MAX_L));
                    break;
                case e_gemma4e_global_layer_skip:
                    run_list(i).add(FLM_OVERRIDE(global_skip_layer_run,
                        this->global_skip_layer.create_run(this->x, this->proj_weights[i], this->rms_weights[i], this->rope_rms_weights[i], this->kv_caches[last_global_kv_cache_layer_idx]), layer_types[i], i, L, this->MAX_L));
                    break;
                case e_gemma4e_swa_layer_skip:
                    run_list(i).add(FLM_OVERRIDE(swa_skip_layer_run,
                        this->swa_skip_layer.create_run(this->x, this->proj_weights[i], this->rms_weights[i], this->rope_rms_weights[i], this->kv_caches[last_swa_kv_cache_layer_idx]), layer_types[i], i, L, this->MAX_L));
                    break;
            }
        }
    }

    this->_set_rope_rms_weights(L);
}

buffer<bf16> gemma4e_dflash_flash::Impl::forward(int ids){
#ifdef GEMMA4E_PROF
    static double dec_t[8] = {0}; static int dec_n = 0; static const char* dec_name[8] = {"preload_wait", "embed", "layers", "lm_head_start+set_ctx", "lm_head_wait", "logits_sync", "preload_create", "total"};
    auto dec_last = gemma4e_prof::now(); const auto dec_t0 = dec_last;
    auto dec_tick = [&](int k){ auto n = gemma4e_prof::now(); dec_t[k] += std::chrono::duration<double>(n - dec_last).count(); dec_last = n; };
#define DEC_TICK(k) dec_tick(k)
#else
#define DEC_TICK(k) do {} while (0)
#endif
    // The embedding work only touches host memory and buffer syncs, so it runs while the empty preload run is still
    // switching the NPU back to the layer design; the layers are the first thing that needs the NPU.
    _process_embedding(ids);
    DEC_TICK(1);
    if (is_preload_launched){
        is_preload_launched = false;
        flm::npu::wait_for_completion(this->pre_load_run, "pre_load_run");
    }
    if (this->decode_seq_stale) this->set_context_length(this->current_context_length);   // a fused verify commit left the decode streams behind
    this->spec_forward_begin(ids);
    DEC_TICK(0);
    if (!this->npu->is_preemption_enabled()){
        if (this->spec_tap_mode()){
            // the 42 layers run as 6 segments so the hidden state after layers 5,11,17,23,35 can be read (DFlash taps)
            for (size_t s = 0; s < this->seg_runs.size(); s++){
                this->seg_runs[s].execute();
                flm::npu::wait_for_completion(this->seg_runs[s], std::string("gemma4e dflash-flash segment ") + std::to_string(s));
                if (s < 5) this->spec_capture_tap((int)s);
            }
        }
        else{
            this->layers_run.execute();
            flm::npu::wait_for_completion(this->layers_run, "layers_run");
        }
    }
    else{
        for (uint32_t i = 0; i < num_hidden_layers; i++){
            switch(layer_types[i]){
                case e_gemma4e_global_layer:
                    FLM_OVERRIDE(global_layer,
                        this->global_layer(this->x, this->proj_weights[i], this->rms_weights[i], this->rope_rms_weights[i], this->kv_caches[i]), layer_types[i], i, this->current_context_length, this->MAX_L);
                    break;
                case e_gemma4e_swa_layer:
                    FLM_OVERRIDE(swa_layer,
                        this->swa_layer(this->x, this->proj_weights[i], this->rms_weights[i], this->rope_rms_weights[i], this->kv_caches[i]), layer_types[i], i, this->current_context_length, this->MAX_L);
                    break;
                case e_gemma4e_global_layer_skip:
                    FLM_OVERRIDE(global_skip_layer,
                        this->global_skip_layer(this->x, this->proj_weights[i], this->rms_weights[i], this->rope_rms_weights[i], this->kv_caches[last_global_kv_cache_layer_idx]), layer_types[i], i, this->current_context_length, this->MAX_L);
                    break;
                case e_gemma4e_swa_layer_skip:
                    FLM_OVERRIDE(swa_skip_layer,
                        this->swa_skip_layer(this->x, this->proj_weights[i], this->rms_weights[i], this->rope_rms_weights[i], this->kv_caches[last_swa_kv_cache_layer_idx]), layer_types[i], i, this->current_context_length, this->MAX_L);
                    break;
            }

            if (this->spec_tap_mode()){
                const int t = this->spec_tap_index((int)i);
                if (t >= 0) this->spec_capture_tap(t);
            }
            DEBUG_BLOCK(2,
            header_print("info", "Finished layer " + std::to_string(i) + " of type " + std::to_string(layer_types[i]) + " for the current token");
            this->x.sync_from_device();
            )
        }
    }
    DEC_TICK(2);
    DEBUG_BLOCK(2,
        std::cout << std::endl;
    header_print("info", "Finished executing all layers for the current token:" + std::to_string(ids));
    this->x.sync_from_device();
    buffer<bf16> valid_x = buffer<bf16>(this->x.data(), D);
    utils::print_matrix(valid_x, D);
    this->x.sync_to_device();
    )

    this->lm_head_run.start();
    this->set_context_length(this->current_context_length + 1);
    DEC_TICK(3);
    flm::npu::wait_for_completion(this->lm_head_run, "lm_head_run");
    DEC_TICK(4);
    this->logits.sync_from_device();
    DEC_TICK(5);
    DEBUG_BLOCK(2,
    header_print("info", "Finished LM head run for the current token " + std::to_string(current_context_length));
    utils::print_matrix(this->logits_valid, vocab_size);
    )

    this->pre_load_run = FLM_OVERRIDE(layer_pre_load, this->layer_pre_load.create_run()); // create run for the preload xclbin, which has an empty sequence and will be used to preload the next layer's weights in the background while the current token is being processed
    this->pre_load_run.start();
    is_preload_launched = true;
    this->spec_forward_end();
    DEC_TICK(6);
#ifdef GEMMA4E_PROF
    dec_t[7] += std::chrono::duration<double>(gemma4e_prof::now() - dec_t0).count();
    if (++dec_n % 32 == 0) { std::fprintf(stderr, "[decode-prof] ms/token over %d tokens:", dec_n); for (int k = 0; k < 8; k++) std::fprintf(stderr, " %s %.2f |", dec_name[k], 1e3 * dec_t[k] / dec_n); std::fprintf(stderr, "\n"); }
#endif
    return this->logits_valid;
}

buffer<bf16> gemma4e_dflash_flash::Impl::prefill(std::vector<int>& ids, void* payload){
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
    if (is_preload_launched){
        is_preload_launched = false;
        flm::npu::wait_for_completion(this->pre_load_run, "pre_load_run");
    }
    DEBUG_BLOCK(1,
    std::cout << "DEBUG: Entering prefill, input length: " << ids.size() << std::endl;
    )
#ifdef MVPREFILL
    header_print("warning", "Prefilling with Decoder, Slow!");
    return this->_prefill_with_mv(ids);
#else
    return this->_prefill_with_mm(ids, payload);
#endif
}

buffer<bf16> gemma4e_dflash_flash::Impl::_prefill_with_mv(std::vector<int>& ids, void* payload){
    DEBUG_BLOCK(1,
    header_print("info", "Entering prefill with MV, input length: " + std::to_string(ids.size()));
    )
    for (int i = 0; i < ids.size(); i++){
        _process_embedding(ids[i]);
        for (uint32_t l = 0; l < num_hidden_layers; l++){

            DEBUG_BLOCK(2,
            header_print_r("info", "Running layer " + std::to_string(l) + " of type " + std::to_string(layer_types[l]) + " for token " + std::to_string(ids[i]));
            )
            // int test_layer = 15;
            switch(layer_types[l]){
                case e_gemma4e_global_layer:
                    FLM_OVERRIDE(global_layer_mv, this->global_layer(this->x, this->proj_weights[l], this->rms_weights[l], this->rope_rms_weights[l], this->kv_caches[l]), layer_types[l], l);
                    break;
                case e_gemma4e_swa_layer:
                    FLM_OVERRIDE(swa_layer_mv, this->swa_layer(this->x, this->proj_weights[l], this->rms_weights[l], this->rope_rms_weights[l], this->kv_caches[l]), layer_types[l], l);
                    break;
                case e_gemma4e_global_layer_skip:
                    FLM_OVERRIDE(global_skip_layer_mv, this->global_skip_layer(this->x, this->proj_weights[l], this->rms_weights[l], this->rope_rms_weights[l], this->kv_caches[last_global_kv_cache_layer_idx]), layer_types[l], l);
                    break;
                case e_gemma4e_swa_layer_skip:
                    FLM_OVERRIDE(swa_skip_layer_mv, this->swa_skip_layer(this->x, this->proj_weights[l], this->rms_weights[l], this->rope_rms_weights[l], this->kv_caches[last_swa_kv_cache_layer_idx]), layer_types[l], l);
                    break;
            }
            DEBUG_BLOCK(2,
            header_print("info", "Finished layer " + std::to_string(l) + " of type " + std::to_string(layer_types[l]) + " for the current token");
            this->x.sync_from_device();
            this->x.sync_to_device();
            buffer<bf16> valid_x = buffer<bf16>(this->x.data(), D);
            utils::print_matrix(valid_x, 256);
            )
        }

        DEBUG_BLOCK(2, exit(0); )
        this->set_context_length(this->current_context_length + 1);
    }

    DEBUG_BLOCK(1,
    header_print("info", "Finished all layers");
    this->x.sync_from_device();
    this->x.sync_to_device();
    buffer<bf16> valid_x = buffer<bf16>(this->x.data(), D);
    utils::print_matrix(valid_x, 256);
    )
    this->lm_head_weights.sync_to_device();
    this->lm_head_run.start();
    flm::npu::wait_for_completion(this->lm_head_run, "lm_head_run");
    this->logits.sync_from_device();

    return logits_valid;
}

buffer<bf16> gemma4e_dflash_flash::Impl::_prefill_with_mm(std::vector<int>& ids, void* payload){
    int L_in = ids.size();

    std::unique_ptr<SafeTensors> reference;
    buffer<int> input_ids;
    DEBUG_BLOCK(2,
    std::cout << "DEBUG: Entering prefill with mm, input length: " << ids.size() << std::endl;
    std::string reference_path = utils::path_join(config.model_path, "gemma4_ref.safetensors");
    reference = std::make_unique<SafeTensors>(reference_path);
    reference->load_weights(input_ids, "input_ids");
    L_in = input_ids.size();
    )

    // Sizes the batch and readies every block; sequences and buffers that already
    // match this geometry are kept as they are.
    PROF_BEGIN();
    const auto prof_wall0__ = std::chrono::steady_clock::now();
    PROF_GROUP("engine");
    // sliding-window decode buffer: where the window rows will sit after this call (see gemma4e_dflash_flash_swa.hpp)
    const int swa_L_end = this->current_context_length + L_in;
    const int swa_new_base = swa_needs_compaction(swa_L_end, this->swa_base)
                                 ? std::max(this->swa_base, swa_base_after_compaction(swa_L_end, (int)SLIDING_LENGTH)) : this->swa_base;
    const int swa_zero_hi = std::min<int>(kSwaBufferRows, (int)std::max<size_t>(this->dirty_rows, (size_t)swa_L_end) - this->swa_base + 1);
    const gemma4e_prefill_shape s = this->prefill_ctx->setup(L_in, this->current_context_length, this->swa_base, swa_new_base, swa_zero_hi);
    this->spec_prefill_begin(ids, std::min<int>(L_in, (int)ids.size()));
    this->prefill_ctx->zero_scratch();
    PROF_TICK("setup");
    gemma4e_common_buffers& bufs = this->prefill_ctx->bufs;

    DEBUG_BLOCK(1,
    std::cout << "DEBUG: L_begin: " << s.L_begin << ", L_end: " << s.L_end << std::endl;
    std::cout << "DEBUG: L_begin_chunked: " << s.L_begin_chunked << ", L_end_chunked: " << s.L_end_chunked << ", L_padded: " << s.L_padded << std::endl;
    std::cout << "DEBUG: L_effective: " << s.L_effective << std::endl;
    std::cout << "DEBUG: sliding_l_begin: " << s.sliding_l_begin << std::endl;
    )

    {
        // batch the page-ins of the mapped tables when they were dropped from the page cache
        const size_t n_ids = std::min<size_t>((size_t)s.L_effective, ids.size());
        this->embedding->prefetch(ids.data(), n_ids);
        this->pli_embedding->prefetch(ids.data(), n_ids);
    }
    for (int i = 0; i < s.L_effective; i++){
        DEBUG_BLOCK(2,
            this->pli_embedding->forward(input_ids[i], this->prefill_ctx->pli_embed_row(i));
            this->embedding->forward(input_ids[i], this->prefill_ctx->residual_row(i));
            continue;
        )
        if ((ids[i] == image_token_id) || (ids[i] == audio_token_id)){
            this->pli_embedding->forward(0, this->prefill_ctx->pli_embed_row(i));
            continue;
        }
        else{
            this->embedding->forward(ids[i], this->prefill_ctx->residual_row(i));
            this->pli_embedding->forward(ids[i], this->prefill_ctx->pli_embed_row(i));
        }
    }

    PROF_TICK("embed_lookup");
    if (payload != nullptr){
        std::vector<bf16> image_embedding;
        std::vector<bf16> audio_embedding;

        gemma4e_multi_modal_payload_t* multi_modal_payload_ptr = (gemma4e_multi_modal_payload_t*)payload;

        if(is_audio && multi_modal_payload_ptr->audio_payload.num_audios > 0){
            audio_embedding = this->gemma4e_audio_encoder->encode(&multi_modal_payload_ptr->audio_payload);
            PROF_TICK("audio.encode");
        }
        if(is_vlm && multi_modal_payload_ptr->image_payload.num_images > 0){
            { auto _t=std::chrono::steady_clock::now();
            image_embedding = this->gemma4e_image_encoder->encode(&multi_modal_payload_ptr->image_payload);}
            PROF_TICK("vision.encode");
            // fprintf(stderr,"[time] image encode %.2f ms\n", std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-_t).count()); 
        }

        bf16* image_token_ptr = image_embedding.data();
        bf16* audio_token_ptr = audio_embedding.data();

        for (int i = 0; i < ids.size(); i++) {
            if (is_vlm && (ids[i] == image_token_id)){
                // copy image embedding to pli_embed_buffer
                memcpy(this->prefill_ctx->residual_row(i).data(), image_token_ptr, D * sizeof(bf16));
                image_token_ptr += D;
            }
            else if (is_audio && (ids[i] == audio_token_id)){
                // copy audio embedding to pli_embed_buffer
                memcpy(this->prefill_ctx->residual_row(i).data(), audio_token_ptr, D * sizeof(bf16));
                audio_token_ptr += D;
            }
        }
    }

    DEBUG_BLOCK(2,
        header_print("info", "Input embedding for the first " + std::to_string(s.L_effective) + " tokens:");
        buffer<bf16> embedding_valid = buffer<bf16>(bufs.pli_embed_buffer.data() + s.L_offset * PLI_D * num_hidden_layers, (size_t)s.L_effective * PLI_D * num_hidden_layers);
        utils::print_matrix(embedding_valid, PLI_D * num_hidden_layers);
    )

    DEBUG_BLOCK(2,
        buffer<bf16> embedding_ref;
        reference->load_weights(embedding_ref, "input_embeds");
        buffer<bf16> embedding_valid = buffer<bf16>(bufs.residual_buffer.data() + s.L_offset * D, (size_t)s.L_effective * D);
        buffer<bf16> embedding_valid_ref = buffer<bf16>(embedding_ref.data() + s.L_offset * D, (size_t)s.L_effective * D);
        print_error_metrics(get_error_metrics(embedding_valid, embedding_valid_ref), "Input Embedding Error: ");
    )

    // fold the token embeddings into the per layer input stream, once for the batch
    buffer<bf16> pli_input_norm(this->rope_rms_weights[0].data() + desc.get_pli_norm_offset(layer_types[0]), PLI_D);
    { auto _t=std::chrono::steady_clock::now();
    this->prefill_ctx->pli->pre_pass(s, this->pli_down_weights, pli_input_norm);
    // fprintf(stderr,"[time] pli pre_pass %.2f ms\n", std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-_t).count()); 
    }
    PROF_TICK("pli_pre_pass");

    for (uint32_t layer_idx = 0; layer_idx < num_hidden_layers; layer_idx++){
        gemma4e_layer_type_t type = layer_types[layer_idx];

        DEBUG_BLOCK(2,
        if (layer_idx > 0){
            buffer<bf16> residual_overide;
            reference->load_weights(residual_overide, "layer_" + std::to_string(layer_idx - 1));
            memcpy(bufs.residual_buffer.data() + s.L_offset * D, residual_overide.data() + s.L_offset * D, (size_t)s.L_effective * D * sizeof(bf16));
        }
        )

        this->prefill_ctx->forward(
            layer_idx, type, s,
            this->proj_weights[layer_idx],
            this->rms_weights[layer_idx],
            this->rope_rms_weights[layer_idx],
            this->pli_gate_up_weights[layer_idx],
            this->kv_caches[is_skip_layer(type)
                          ? (is_swa_layer(type) ? last_swa_kv_cache_layer_idx : last_global_kv_cache_layer_idx)
                          : (int)layer_idx],
            this->layer_scale[layer_idx],
            reference.get()
        );
        this->spec_prefill_tap((int)layer_idx, s.L_effective);
    }

    DEBUG_BLOCK(2,
            reference.reset();
            header_print_r("info", "DEBUG EXIT");
            exit(0);
    )
    buffer<bf16> predict = this->prefill_ctx->residual_row(s.L_effective - 1);
    this->lm_head_weights.sync_to_device();
    get_logits(predict);
    this->spec_prefill_end(s.L_effective);
    this->swa_base = swa_new_base;
    PROF_TICK("lm_head_logits");
    this->set_context_length(this->current_context_length + s.L_effective);
    PROF_TICK("set_context_length");
    PROF_REPORT(std::chrono::duration<double>(std::chrono::steady_clock::now() - prof_wall0__).count());

    this->pre_load_run = FLM_OVERRIDE(layer_pre_load, this->layer_pre_load.create_run());
    this->pre_load_run.start();
    is_preload_launched = true;
    return logits_valid;
}

buffer<bf16> gemma4e_dflash_flash::Impl::get_k_cache(int layer_idx, int idx){
    this->kv_caches[layer_idx].sync_from_device();
    buffer<bf16> k_cache(DK);
    bf16* k_cache_ptr = this->kv_caches[layer_idx].data();
    uint32_t offset = idx * DK;
    memcpy(static_cast<void*>(k_cache.data()), static_cast<void*>(k_cache_ptr + offset), DK * sizeof(bf16));

    return k_cache;
}

buffer<bf16> gemma4e_dflash_flash::Impl::get_v_cache(int layer_idx, int idx){
    this->kv_caches[layer_idx].sync_from_device();
    buffer<bf16> v_cache(DV);
    bf16* v_cache_ptr = this->kv_caches[layer_idx].data();
    uint32_t offset = idx * DV;
    memcpy(static_cast<void*>(v_cache.data()), static_cast<void*>(v_cache_ptr + offset), DV * sizeof(bf16));
    return v_cache;
}

buffer<bf16> gemma4e_dflash_flash::Impl::get_logits(buffer<bf16>& predict){
    memcpy(this->x.data(), predict.data(), D * sizeof(bf16));
    this->x.sync_to_device();
    this->lm_head_run.start();
    flm::npu::wait_for_completion(this->lm_head_run, "lm_head_run");
    this->logits.sync_from_device();
    return this->logits_valid;
}

void gemma4e_dflash_flash::Impl::clear_context(){
    this->swa_base = 0;
    this->set_context_length(0);
    this->spec_cold(0);
    for (int i = 0; i < non_skip_layers; i++){
        this->kv_caches[i].sync_from_device();
        memset(this->kv_caches[i].data(), 0, this->kv_caches[i].size() * sizeof(bf16));
        this->kv_caches[i].sync_to_device();
    }
    this->dirty_rows = 0;
}

kv_row_layout gemma4e_dflash_flash::Impl::kv_layout() const{
    kv_row_layout l;
    l.DK = DK; l.DV = DV; l.striped = false;
    l.v_offset = static_cast<size_t>(MAX_L) * DK; // a global layer holds K rows, then V rows
    return l;
}

std::unique_ptr<hybrid_engine_state> gemma4e_dflash_flash::Impl::new_state(){
    size_t ring_bytes = 0, global_layers = 0;
    for (size_t i = 0; i < kv_caches.size(); i++){
        if (attn_mask[i]) global_layers++;
        else ring_bytes += swa_window_elems() * sizeof(bf16);     // the last SLIDING_LENGTH rows of the window, oldest first
    }
    return kv_rows_new_state(global_layers, kv_layout(), ring_bytes);
}

int gemma4e_dflash_flash::Impl::read_state(hybrid_engine_state& state, int kv_from){
    header_print_r("FLM", "Reading state at context length " + std::to_string(this->current_context_length) + ", K/V rows from " + std::to_string(kv_from));
    const int length = kv_rows_read(state, this->kv_caches, kv_layout(), this->current_context_length, kv_from, -1, &attn_mask);
    if (length < 0) return -1;
    for (size_t i = 0; i < kv_caches.size(); i++){
        if (attn_mask[i]) continue;
        state.state[i] = buffer<bf16>(swa_window_elems());
        this->swa_extract((int)i, this->current_context_length, state.state[i].data());
    }
    return length;
}

int gemma4e_dflash_flash::Impl::write_state_kv(const hybrid_engine_state& state){
    return kv_rows_write(state, this->kv_caches, kv_layout(), this->MAX_L, &attn_mask);
}

int gemma4e_dflash_flash::Impl::write_state(const hybrid_engine_state& state){
    // check the rings before anything is written
    bool ok = state.state.size() == this->kv_caches.size();
    for (size_t i = 0; ok && i < kv_caches.size(); i++){
        if (!attn_mask[i] && state.state[i].size() != swa_window_elems()) ok = false;
    }
    const int length = ok ? kv_rows_write(state, this->kv_caches, kv_layout(), this->MAX_L, &attn_mask) : -1;
    if (length < 0){
        header_print_r("FLM", "No valid state to write");
        return -1;
    }
    header_print_r("FLM", "Writing state at context length " + std::to_string(length));
    {
        std::vector<buffer<bf16>> windows(kv_caches.size());
        for (size_t i = 0; i < kv_caches.size(); i++) if (!attn_mask[i]) windows[i] = state.state[i];
        this->swa_restore_all(length, windows);
    }
    // rows past the length hold what the requests since the state was read left there: put back the zeros of a fresh engine
    kv_rows_zero(this->kv_caches, kv_layout(), static_cast<size_t>(length), this->dirty_rows, &attn_mask);
    this->dirty_rows = static_cast<size_t>(length);
    this->set_context_length(length);
    this->spec_cold(length);
    return length;
}

int gemma4e_dflash_flash::Impl::get_current_context_length(){
    return this->current_context_length;
}

void gemma4e_dflash_flash::Impl::update_max_length(uint32_t MAX_L){
    // the context is pinned at construction, so there is nothing to grow into:
    // a longer context would re-cut the prefill scratch for a longer chunk, which
    // is exactly the allocation this engine is pinned to avoid
    if (MAX_L != this->MAX_L) {
        header_print("WARNING", "This model only supports a 1k context; the request for " + std::to_string(MAX_L) + " tokens is ignored and the context stays at " + std::to_string(this->MAX_L));
    }
    return;

    this->MAX_L = MAX_L;
    this->swa_base = 0;
    size_t kv_cache_size = MAX_L * (DK + DV);
    this->sequence->set_max_length(MAX_L);

    this->prefill_ctx->set_max_length(MAX_L);

    for (uint32_t i = 0; i < non_skip_layers; i++){
        if (!is_swa_layer(layer_types[i])){
            this->kv_caches[i] = buffer<bf16>(kv_cache_size);
            memset(this->kv_caches[i].data(), 0, kv_cache_size * sizeof(bf16));
            this->kv_caches[i].sync_to_device();
        }
        else{
            memset(this->kv_caches[i].data(), 0, this->kv_caches[i].size() * sizeof(bf16));
            this->kv_caches[i].sync_to_device();
        }
    }
    is_checkpoint_valid = false; // force reload checkpoint to clear kv cache on device
    this->dirty_rows = 0; // every global buffer was zeroed
    this->set_context_length(0); // clear
    this->spec_cold(0);
}

void gemma4e_dflash_flash::Impl::load_weights(Q4NX& q4nx){

    const std::string q4nx_path = utils::path_join(config.model_path, "model.q4nx");
    this->embedding->init_weights(q4nx, q4nx_path, "model.embed_tokens");
    this->pli_embedding->init_weights(q4nx, q4nx_path, "model.per_layer_token_embd");

    // Shared by every layer, so it is read once and handed to each of them.
    buffer<bf16> per_layer_norm;
    q4nx.load_weights(per_layer_norm, "model.per_layer_proj_norm.weight");

    for (int layer_idx = 0; layer_idx < num_hidden_layers; layer_idx++){
        this->desc.load_layer_weights(layer_idx, q4nx,
                                      this->proj_weights[layer_idx],
                                      this->rms_weights[layer_idx],
                                      this->rope_rms_weights[layer_idx],
                                      this->pli_gate_up_weights[layer_idx],
                                      per_layer_norm,
                                      this->layer_scale[layer_idx]);
        this->pli_gate_up_weights[layer_idx].sync_to_device();
        this->proj_weights[layer_idx].sync_to_device();
        this->rms_weights[layer_idx].sync_to_device();
        this->rope_rms_weights[layer_idx].sync_to_device();
        FLM_OVERRIDE(layer_weights_loaded, (void)0, &this->desc, layer_idx);
        DEBUG_BLOCK(1,
        header_print("info", "Finished loading weights for layer " + std::to_string(layer_idx));
        )
    }

    // model.norm.weight lives in the second D-sized slot of x, behind the hidden state.
    this->desc.load_head_weights(q4nx, this->lm_head_weights, this->x.data() + D, this->pli_down_weights);

    this->pli_down_weights.sync_to_device();
    this->lm_head_weights.sync_to_device();
    FLM_OVERRIDE(head_weights_loaded, (void)0, &this->desc);

    DEBUG_BLOCK(1,
    header_print("info", "Finished loading all layer weights");
    )

    DEBUG_BLOCK(1,
    header_print("info", "Finished updating sequence buffer offsets based on loaded weights");
    )

    this->clear_context();

    DEBUG_BLOCK(1,
    header_print("info", "Finished clearing context after loading weights");
    )

    // read qkv weights
    if(is_vlm){
        DEBUG_BLOCK(1,
        std::cout << "[DBG] gemma4e_dflash_flash::Impl::load_weights: entering VLM branch, path="
                  << config.get<std::string>("vision_model_weight", "") << std::endl;
        )
        {
            SafeTensors vision_weights(config.get<std::string>("vision_model_weight", ""));
            DEBUG_BLOCK(1,
            std::cout << "[DBG] gemma4e_dflash_flash::Impl::load_weights: SafeTensors constructed" << std::endl;
            )
            this->gemma4e_image_encoder->init_weights(vision_weights);
            DEBUG_BLOCK(1,
            std::cout << "[DBG] gemma4e_dflash_flash::Impl::load_weights: init_weights returned" << std::endl;
            )
        }
        DEBUG_BLOCK(1,
        std::cout << "[DBG] gemma4e_dflash_flash::Impl::load_weights: SafeTensors out of scope" << std::endl;
        )
    }
    if(is_audio){
        SafeTensors audio_weights(config.get<std::string>("audio_model_weight", ""));
        this->gemma4e_audio_encoder->init_weights(audio_weights);
    }
}

int gemma4e_dflash_flash::Impl::checkpoint(){
    if (!is_checkpoint_valid){
        _allocate_checkpoint_buffers();
    } // use lazy allocation
    header_print_r("FLM", "Creating checkpoint at context length " + std::to_string(this->current_context_length));
    checkpoint_context_length = this->current_context_length;
    this->spec_checkpoint();
    for (int i = 0; i < non_skip_layers; i++){
        if (!is_global_layer_idx(i)){
            this->swa_extract(i, this->current_context_length, this->kv_checkpoint[i].data());
        }
    }
    is_checkpoint_valid = true;
    return this->checkpoint_context_length;
}

int gemma4e_dflash_flash::Impl::restore(){
    if (!is_checkpoint_valid){
        header_print("FLM", "No valid checkpoint found, cannot restore context!");
        return -1;
    }
    header_print_r("FLM", "Restoring checkpoint at context length " + std::to_string(checkpoint_context_length));
    this->current_context_length = checkpoint_context_length;
    this->spec_cold(this->current_context_length);
    this->spec_restore_warm(this->current_context_length);
    {
        // sliding layers: lay the saved window back at the base it implies, zero everything stale behind it
        std::vector<buffer<bf16>> windows(kv_caches.size());
        for (int i = 0; i < non_skip_layers; i++) if (!is_global_layer_idx(i)) windows[i] = this->kv_checkpoint[i];
        this->swa_restore_all(this->current_context_length, windows);
    }
    for (int i = 0; i < non_skip_layers; i++){
        if (is_global_layer_idx(i)){
            this->kv_caches[i].sync_from_device();
            int offset = (size_t)current_context_length * (DK);
            // first half is K, second half is V, and they are stored contiguously in kv_cache
            memset(this->kv_caches[i].data() + offset, 0, (this->kv_caches[i].size() / 2 - offset) * sizeof(bf16)); // zero out the part that is not restored for global layers, since we only restore the sliding part for global layers
            memset(this->kv_caches[i].data() + this->kv_caches[i].size() / 2 + offset, 0, (this->kv_caches[i].size() / 2 - offset) * sizeof(bf16)); // zero out the second half for swa
            this->kv_caches[i].sync_to_device();
        }
    }
    this->dirty_rows = (size_t)this->current_context_length;   // every row past the context is zero again
    set_context_length(this->current_context_length);          // after the base is settled: the sequences address rows by it
    return this->current_context_length;
}

void gemma4e_dflash_flash::Impl::_allocate_checkpoint_buffers(){
    // allocate kv cache checkpoint for sliding layers
    this->kv_checkpoint.clear();
    this->kv_checkpoint.resize(non_skip_layers);
    for (int i = 0; i < non_skip_layers; i++){
        if(!is_global_layer_idx(i)){
            this->kv_checkpoint[i] = buffer<bf16>(swa_window_elems());
        }
    }
    is_checkpoint_valid = false;
}


// ---------------------------------------------------------------------------------------------------------------------
// sliding-window K/V buffers: linear window with host compaction (gemma4e_dflash_flash_swa.hpp)
// ---------------------------------------------------------------------------------------------------------------------

void gemma4e_dflash_flash::Impl::swa_zero_rows(int layer, int from_row, int to_row){
    to_row = std::min(to_row, kSwaBufferRows);
    if (to_row <= from_row) return;
    bf16* kv = this->kv_caches[layer].data();
    const size_t n = (size_t)(to_row - from_row);
    memset(kv + (size_t)from_row * SWA_DK, 0, n * SWA_DK * sizeof(bf16));
    memset(kv + (size_t)kSwaBufferRows * SWA_DK + (size_t)from_row * SWA_DV, 0, n * SWA_DV * sizeof(bf16));
}

void gemma4e_dflash_flash::Impl::swa_compact(int L){
    const int new_base = swa_base_after_compaction(L, (int)SLIDING_LENGTH);
    if (new_base <= this->swa_base) return;
    const size_t move_rows = (size_t)std::max(0, L - new_base);
    const size_t from_row = (size_t)(new_base - this->swa_base);
    const int hi = std::min<int>(kSwaBufferRows, (int)std::max<size_t>(this->dirty_rows, (size_t)L) - this->swa_base + 1);
    const auto t_compact = std::chrono::steady_clock::now();
    for (int i = 0; i < non_skip_layers; i++){
        if (is_global_layer_idx(i)) continue;
        this->kv_caches[i].sync_from_device();
        bf16* kv = this->kv_caches[i].data();
        memmove(kv, kv + from_row * SWA_DK, move_rows * SWA_DK * sizeof(bf16));
        memmove(kv + (size_t)kSwaBufferRows * SWA_DK, kv + (size_t)kSwaBufferRows * SWA_DK + from_row * SWA_DV, move_rows * SWA_DV * sizeof(bf16));
        this->swa_zero_rows(i, (int)move_rows, hi);
        this->kv_caches[i].sync_to_device();
    }
    const double compact_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_compact).count();
    header_print_r("FLM", "SWA window buffers compacted at context length " + std::to_string(L) + ": base " + std::to_string(this->swa_base) + " -> " + std::to_string(new_base) + " in " + std::to_string(compact_ms) + " ms");
    this->swa_base = new_base;
}

void gemma4e_dflash_flash::Impl::swa_extract(int layer, int L, bf16* dst){
    const int n = swa_window_rows(L);
    memset(dst, 0, swa_window_elems() * sizeof(bf16));
    if (n <= 0) return;
    this->kv_caches[layer].sync_from_device();
    const bf16* kv = this->kv_caches[layer].data();
    const size_t r0 = (size_t)((L - n) - this->swa_base);
    memcpy(dst, kv + r0 * SWA_DK, (size_t)n * SWA_DK * sizeof(bf16));
    memcpy(dst + (size_t)SLIDING_LENGTH * SWA_DK, kv + (size_t)kSwaBufferRows * SWA_DK + r0 * SWA_DV, (size_t)n * SWA_DV * sizeof(bf16));
}

void gemma4e_dflash_flash::Impl::swa_relay(int layer, int L, const bf16* src, int new_base, int zero_hi){
    const int n = swa_window_rows(L);
    this->kv_caches[layer].sync_from_device();
    bf16* kv = this->kv_caches[layer].data();
    this->swa_zero_rows(layer, 0, zero_hi);
    if (n > 0){
        const size_t r0 = (size_t)((L - n) - new_base);
        memcpy(kv + r0 * SWA_DK, src, (size_t)n * SWA_DK * sizeof(bf16));
        memcpy(kv + (size_t)kSwaBufferRows * SWA_DK + r0 * SWA_DV, src + (size_t)SLIDING_LENGTH * SWA_DK, (size_t)n * SWA_DV * sizeof(bf16));
    }
    this->kv_caches[layer].sync_to_device();
}

void gemma4e_dflash_flash::Impl::swa_restore_all(int L, const std::vector<buffer<bf16>>& windows){
    const int new_base = swa_base_after_compaction(L, (int)SLIDING_LENGTH);
    // everything ever written sits below max(dirty_rows, L) - old base; zero that much (+1), then lay the window
    const int zero_hi = std::min<int>(kSwaBufferRows, std::max<int>(1, (int)std::max<size_t>(this->dirty_rows, (size_t)L) - this->swa_base + 1));
    for (int i = 0; i < non_skip_layers; i++){
        if (is_global_layer_idx(i)) continue;
        this->swa_relay(i, L, windows[i].data(), new_base, zero_hi);
    }
    this->swa_base = new_base;
}


void gemma4e_dflash_flash::Impl::swa_check_zero(const char* where, int ctx){
    static const bool on = [] { const char* e = std::getenv("FLM_DFLASH_CHECK_SWA"); return e != nullptr && e[0] == '1'; }();
    if (!on) return;
    int bad_swa = 0, bad_glob = 0, first_swa_layer = -1, first_swa_row = -1, first_glob_layer = -1, first_glob_row = -1;
    for (int i = 0; i < non_skip_layers; i++){
        this->kv_caches[i].sync_from_device();
        const bf16* kv = this->kv_caches[i].data();
        if (is_global_layer_idx(i)){
            for (int r = ctx; r < std::min<int>((int)this->dirty_rows + 1, this->MAX_L); r++){
                bool nz = false;
                for (int c = 0; c < DK && !nz; c++) if ((float)kv[(size_t)r * DK + c] != 0.0f) nz = true;
                for (int c = 0; c < DV && !nz; c++) if ((float)kv[(size_t)this->MAX_L * DK + (size_t)r * DV + c] != 0.0f) nz = true;
                if (nz){ if (!bad_glob++) { first_glob_layer = i; first_glob_row = r; } }
            }
        } else {
            for (int r = ctx - this->swa_base; r < kSwaBufferRows; r++){
                if (r < 0) continue;
                bool nz = false;
                for (int c = 0; c < SWA_DK && !nz; c++) if ((float)kv[(size_t)r * SWA_DK + c] != 0.0f) nz = true;
                for (int c = 0; c < SWA_DV && !nz; c++) if ((float)kv[(size_t)kSwaBufferRows * SWA_DK + (size_t)r * SWA_DV + c] != 0.0f) nz = true;
                if (nz){ if (!bad_swa++) { first_swa_layer = i; first_swa_row = r; } }
            }
        }
    }
    if (bad_swa || bad_glob)
        header_print_r("FLM", std::string("CHECK_SWA ") + where + " ctx " + std::to_string(ctx) + " base " + std::to_string(this->swa_base) + ": non-zero rows behind the context: swa " + std::to_string(bad_swa) + " (first layer " + std::to_string(first_swa_layer) + " row " + std::to_string(first_swa_row) + "), global " + std::to_string(bad_glob) + " (first layer " + std::to_string(first_glob_layer) + " row " + std::to_string(first_glob_row) + ")");
}
///@brief destructor of qwen3vl_npu
gemma4e_dflash_flash::Impl::~Impl(){
    this->spec_destroy();
    // a preload run may still be in flight; it must complete before the
    // xrt objects it references are torn down
    if (is_preload_launched){
        try{
            flm::npu::wait_for_completion(this->pre_load_run, "pre_load_run");
        }
        catch (const std::exception& e){
            header_print("FLM", std::string("Failed to wait for the pending preload run: ") + e.what());
        }
        is_preload_launched = false;
    }
}
/*
qwen3vl_npu::Impl::~Impl(){
    for (int i = 0; i < config.get("num_hidden_layers"); i++){
        this->kv_caches[i].free();
        this->proj_weights[i].free();
        this->rms_weights[i].free();
    }
    this->kv_caches.clear();
    this->rms_weights.clear();
    this->proj_weights.clear();
    this->rope_weights.release();
    this->dequantized_qkv_weights.release();
    this->dequantized_o_weights.release();
    this->x.release();
    this->lm_head.reset();
}
*/
// ===============================================
// Externals for qwen3vl_npu
// ===============================================
gemma4e_dflash_flash::gemma4e_dflash_flash(LM_Config config, npu_xclbin_manager *npu_instance, int MAX_L){

    this->load_vision_preprocess_parameters(config);
    this->load_audio_preprocess_parameters(config);
    this->_impl = new Impl(config, npu_instance, this,   MAX_L);
}

buffer<bf16> gemma4e_dflash_flash::forward(int ids){
    return this->_impl->forward(ids);
}

buffer<bf16> gemma4e_dflash_flash::prefill(std::vector<int>& ids, void* payload){
    try {
        buffer<bf16> result = this->_impl->prefill(ids, payload);
        return result;
    }
    catch (const std::runtime_error& e) {
        header_print("FLM", e.what());
        throw;
    }
}

void gemma4e_dflash_flash::set_context_length(int L){
    std::cout << "Setting context length is not supported for qwen3vl_npu!" << std::endl;
}

void gemma4e_dflash_flash::load_weights(Q4NX& q4nx){
    this->_impl->load_weights(q4nx);
}

void gemma4e_dflash_flash::update_max_length(uint32_t MAX_L){
    this->_impl->update_max_length(MAX_L);
}

void gemma4e_dflash_flash::clear_context(){
    this->_impl->clear_context();
}

int gemma4e_dflash_flash::checkpoint(){
    return this->_impl->checkpoint();
}

int gemma4e_dflash_flash::restore(){
    return this->_impl->restore();
}

std::unique_ptr<hybrid_engine_state> gemma4e_dflash_flash::new_state(){
    return this->_impl->new_state();
}

int gemma4e_dflash_flash::read_state(hybrid_engine_state& state, int kv_from){
    return this->_impl->read_state(state, kv_from);
}

int gemma4e_dflash_flash::write_state(const hybrid_engine_state& state){
    return this->_impl->write_state(state);
}

int gemma4e_dflash_flash::write_state_kv(const hybrid_engine_state& state){
    return this->_impl->write_state_kv(state);
}

extern "C" const flm_state_api* gemma4e_dflash_flash_state_api(void* engine){
    return hybrid_state_api<gemma4e_dflash_flash>(engine);
}

buffer<bf16> gemma4e_dflash_flash::get_k_cache(int layer_idx, int idx){
    return this->_impl->get_k_cache(layer_idx, idx);
}

buffer<bf16> gemma4e_dflash_flash::get_v_cache(int layer_idx, int idx){
    return this->_impl->get_v_cache(layer_idx, idx);
}

int gemma4e_dflash_flash::get_current_context_length(){
    return this->_impl->get_current_context_length();
}


// ---- DFlash speculative decoding: forwards to Impl (implemented in gemma4e_dflash_flash_spec.cpp) ----
bool gemma4e_dflash_flash::load_draft(const std::string& d){ return this->_impl->load_draft(d); }
bool gemma4e_dflash_flash::draft_ready() const { return this->_impl->draft_ready(); }
void gemma4e_dflash_flash::set_stop_tokens(const std::vector<int>& ids){ this->_impl->set_stop_tokens(ids); }
int gemma4e_dflash_flash::dflash_draft(int last_token, int* drafts){ return this->_impl->dflash_draft(last_token, drafts); }
const bf16* gemma4e_dflash_flash::dflash_verify(int last_token, const int* drafts, int n){ return this->_impl->dflash_verify(last_token, drafts, n); }
void gemma4e_dflash_flash::dflash_commit(int n_keep){ this->_impl->dflash_commit(n_keep); }
int gemma4e_dflash_flash::vocab_size() const { return this->_impl->spec_vocab_size(); }
uint64_t gemma4e_dflash_flash::last_speculation_prime_us() const { return this->_impl->last_speculation_prime_us(); }
const bf16* gemma4e_dflash_flash::last_speculation_logits(int& rows, int& vocab) const { return this->_impl->last_speculation_logits(rows, vocab); }
bool gemma4e_dflash_flash::supports_speculation() const { return this->_impl->supports_speculation(); }
std::vector<int> gemma4e_dflash_flash::speculate(int last_token, int max_draft){ return this->_impl->speculate(last_token, max_draft); }
double gemma4e_dflash_flash::mean_accepted_length() const { return this->_impl->mean_accepted_length(); }
double gemma4e_dflash_flash::draft_hit_rate() const { return this->_impl->draft_hit_rate(); }
int gemma4e_dflash_flash::speculation_cycles() const { return this->_impl->speculation_cycles(); }
std::string gemma4e_dflash_flash::speculation_stats() const { return this->_impl->speculation_stats(); }
std::string gemma4e_dflash_flash::speculation_timing() const { return this->_impl->speculation_timing(); }
void gemma4e_dflash_flash::reset_speculation_stats(){ this->_impl->reset_speculation_stats(); }

gemma4e_dflash_flash::~gemma4e_dflash_flash(){
    delete this->_impl;
}
