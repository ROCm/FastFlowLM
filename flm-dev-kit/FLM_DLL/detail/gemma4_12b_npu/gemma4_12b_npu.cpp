// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include "gemma4_12b_npu_detail.hpp"
#include "metrices.hpp"

gemma4_12b_npu::Impl::Impl(LM_Config config, npu_xclbin_manager *npu_instance, gemma4_12b_npu* parent_ptr, int MAX_L )
    : config(config), npu(npu_instance), parent_ptr(parent_ptr){
    current_context_length = 0;

    this->MAX_L = std::max(MAX_L, 4096);
    // make MAX_L a power of 2
    float log2_MAX_L = std::log2(this->MAX_L);
    if (log2_MAX_L != std::floor(log2_MAX_L)){
        this->MAX_L = 1 << ((int)std::floor(log2_MAX_L) + 1);
    }
    MAX_L = this->MAX_L; // synchronize MAX_L
    is_preload_launched = false;
    // the dequant_mm xclbin is only registered if a prefill actually asks for it
    prefill_mm = nullptr;

    proj_offsets.resize(2);

    try{
        // the model description owns the config parsing and the weight layout
        desc.build(config);

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
        num_kv_shared_layers    = desc.num_kv_shared_layers;
        SLIDING_LENGTH          = desc.sliding_window;
        final_logit_softcapping = desc.final_logit_softcapping;
        INTERMEDIATE_SIZE       = desc.intermediate_size;
        global_layer_period     = desc.sliding_window_pattern;

        if (global_layer_period == 0){
            throw std::runtime_error("sliding_window_pattern must be non zero");
        }
        // construct layer_types
        this->layer_types.resize(num_hidden_layers);
        for (int i = 0; i < num_hidden_layers; i++){
            this->layer_types[i] = desc.layer_type(i);
        }
        DEBUG_BLOCK(2,
        std::cout << "\tlayer_types: \n";
        for (int i = 0; i < num_hidden_layers; i++){
            std::string type_str;
            switch (layer_types[i]){
                case gemma4_12b_swa_layer:
                    type_str = "SWA";
                    break;
                case gemma4_12b_global_layer:
                    type_str = "Global";
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

    gemma4_12b_seq_gen_parameters_t seq_gen_params = {
        .D = D,
        .DH = DH,
        .DQ = DQ,
        .DK = DK,
        .DV = DV,
        .SWA_DH = SWA_DH,
        .SWA_DQ = SWA_DQ,
        .SWA_DK = SWA_DK,
        .SWA_DV = SWA_DV,
        .INTERMEDIATE_SIZE = INTERMEDIATE_SIZE,
        .NUM_ATTENTION_HEADS = (int)desc.num_attention_heads,
        .NUM_KEY_VALUE_HEADS = (int)desc.num_kv_heads,
        .SWA_NUM_KEY_VALUE_HEADS = (int)desc.swa_num_kv_heads,
        .SLIDING_WINDOW_SIZE = SLIDING_LENGTH,
        .VOCAB_SIZE_PADDED = vocab_size_padded,
    };

    this->sequence = std::make_unique<gemma4_12b_npu_sequence>(seq_gen_params, MAX_L);

    layer_app_manager = npu->register_xclbin(utils::path_join(config.exec_path, "xclbins", config.model_name, "layer.xclbin"));
    
    this->global_layer = layer_app_manager->create_app();
    this->swa_layer = layer_app_manager->create_app();
    this->layer_pre_load = layer_app_manager->create_app();

    this->layer_scale_apps.resize(num_hidden_layers);
    for (int i = 0; i < num_hidden_layers; i++){
        this->layer_scale_apps[i] = layer_app_manager->create_app();
        npu_sequence* scale_seq = this->layer_scale_apps[i].seq();
        scale_seq->clear_cmds();
        scale_seq->cmds2seq();
    }

    // layers_run is filled in by set_context_length(), which is the only place that knows
    // the control sequence for the current context length.

    lm_head_app_manager = npu->register_xclbin(utils::path_join(config.exec_path, "xclbins", config.model_name, "lm_head.xclbin"));
    this->lm_head = lm_head_app_manager->create_app();

    // allocate all buffers, every size comes from the model description
    rms_weights.resize(num_hidden_layers);
    proj_weights.resize(num_hidden_layers);
    rope_rms_weights.resize(num_hidden_layers);
    kv_caches.resize(num_hidden_layers);
    attn_mask.resize(num_hidden_layers);
    for (int i = 0; i < num_hidden_layers; i++) attn_mask[i] = is_global_layer_idx(i);
    layer_scale = buffer<bf16>(num_hidden_layers);

    this->x = layer_app_manager->create_bo_buffer<bf16>((3 * D + 8191) / 8192 * 8192); // Iterative Hidden States (D) + Final RMS NORM (D) + INITIAL EMBEDDING (D)

    for (int layer_idx = 0; layer_idx < num_hidden_layers; layer_idx++){
        gemma4_12b_layer_type_t type = layer_types[layer_idx];

        // q + k (+ v) + o + up/gate + down, all q4_0
        proj_weights[layer_idx] = layer_app_manager->create_bo_buffer<u8>(desc.get_proj_weights_byte_size(type));
        this->proj_weights[layer_idx].memset((uint8_t)0);
        this->proj_weights[layer_idx].sync_to_device();

        // input_layer_norm (D) + post_attn_layer_norm (D) + pre_ffn_layer_norm (D) + post_ffn_layer_norm (D)
        rms_weights[layer_idx] = layer_app_manager->create_bo_buffer<bf16>(desc.get_rms_elems());

        // COS/SIN (_DH) + Q_NORM (_DH) + K_NORM (_DH) + layer_scale + padding
        rope_rms_weights[layer_idx] = layer_app_manager->create_bo_buffer<bf16>(desc.get_rope_rms_elems(type));

        kv_caches[layer_idx] = layer_app_manager->create_bo_buffer<bf16>(desc.get_kv_cache_size(type, MAX_L));
        kv_caches[layer_idx].memset((bf16)0);
        kv_caches[layer_idx].sync_to_device();

        DEBUG_BLOCK(2,
        header_print("info", "Buffer sizes for layer " + std::to_string(layer_idx)
                     + ": proj=" + std::to_string(proj_weights[layer_idx].size()) + " B"
                     + ", rms=" + std::to_string(rms_weights[layer_idx].size()) + " bf16"
                     + ", rope_rms=" + std::to_string(rope_rms_weights[layer_idx].size()) + " bf16"
                     + ", kv_cache=" + std::to_string(kv_caches[layer_idx].size()) + " bf16");
        )
    }

    for (int type = 0; type < gemma4_12b_total_layer_types; type++){
        gemma4_12b_layer_type_t layer_type = static_cast<gemma4_12b_layer_type_t>(type);
        proj_offsets[type].qkv_offset    = desc.get_qkv_offset(layer_type);
        proj_offsets[type].o_offset      = desc.get_o_offset(layer_type);
        proj_offsets[type].upgate_offset = desc.get_upgate_offset(layer_type);
        proj_offsets[type].down_offset   = desc.get_down_offset(layer_type);
    }
    this->sequence->update_buffer_offsets(this->proj_offsets);

    is_checkpoint_valid = false; // checkpoint is not valid until we load kv cache to device after resizing buffers

    lm_head_weights = lm_head_app_manager->create_bo_buffer<u8>(desc.get_lm_head_w_size());
    logits = lm_head_app_manager->create_bo_buffer<bf16>(vocab_size_padded);
    logits_valid = buffer<bf16>(logits.data(), vocab_size);

    this->embedding = std::make_unique<EmbeddingBF16Mapped>(vocab_size, D);

    this->sequence->gen_lm_head_seq(this->lm_head.seq(), final_logit_softcapping);
    this->lm_head_run = this->lm_head.create_run(this->logits, this->lm_head_weights, this->x);

    // empty seq for preload xclbin
    npu_sequence* pre_load_seq = this->layer_pre_load.seq();
    pre_load_seq->clear_cmds();
    pre_load_seq->cmds2seq();
    
    this->clear_context();

}

void gemma4_12b_npu::Impl::_set_rope_rms_weights(int idx){

    buffer<bf16> cos_buf(DH / 2);
    buffer<bf16> sin_buf(DH / 2);
    buffer<bf16> swa_cos_buf(SWA_DH / 2);
    buffer<bf16> swa_sin_buf(SWA_DH / 2);
    
    for (int j = 0; j < DH / 2; j++){
        cos_buf[j] = (bf16)cos(desc.inv_freq_global[j] * idx);
        sin_buf[j] = (bf16)sin(desc.inv_freq_global[j] * idx);
    }
    for (int j = 0; j < SWA_DH / 2; j++){
        swa_cos_buf[j] = (bf16)cos(desc.inv_freq_swa[j] * idx);
        swa_sin_buf[j] = (bf16)sin(desc.inv_freq_swa[j] * idx);
    }

    for (int i = 0; i < num_hidden_layers; i++){
        // the cos/sin table sits at the head of the rope buffer, see gemma4_12b_desc
        bf16* w_rope_ptr = this->rope_rms_weights[i].data() + desc.get_rope_cos_offset(layer_types[i]);
        if (is_swa_layer(layer_types[i])){
            memcpy(w_rope_ptr, swa_cos_buf.data(), SWA_DH * sizeof(bf16) / 2);
            memcpy(w_rope_ptr + SWA_DH / 2, swa_sin_buf.data(), SWA_DH * sizeof(bf16) / 2);
        }
        else {
            memcpy(w_rope_ptr, cos_buf.data(), DH * sizeof(bf16) / 2);
            memcpy(w_rope_ptr + DH / 2, sin_buf.data(), DH * sizeof(bf16) / 2);
        }
        this->rope_rms_weights[i].sync_to_device();
    }
}

void gemma4_12b_npu::Impl::set_context_length(int L){
    if (L > (int)this->MAX_L){
        if (this->current_context_length <= (int)this->MAX_L){
            header_print_r("FLM", "Context is full at " + std::to_string(this->MAX_L)
                           + " tokens. Further tokens keep overwriting the newest kv cache"
                           + " row, so the output past this point is not meaningful.");
        }
        L = (int)this->MAX_L;
    }
    this->current_context_length = L;
    // the prefill works on whole 512-row blocks: what it leaves in the rows past the length is zeroed up to here
    this->dirty_rows = std::max(this->dirty_rows, static_cast<size_t>(L) + 512);
    DEBUG_BLOCK(2,
    header_print_r("info", "Setting context length to " + std::to_string(L) + " for all layers in the sequence");
    )
    this->sequence->gen_layer_seq(this->global_layer.seq(),      L + 1, gemma4_12b_global_layer);
    this->sequence->gen_layer_seq(this->swa_layer.seq(),         L + 1, gemma4_12b_swa_layer);
    // layer 0's scale run publishes the context length for the whole token; the other
    // layers reuse the RTP the attention cores are already holding.
    this->sequence->gen_layer_scale_seq(this->layer_scale_apps[0].seq(), this->layer_scale[0], L + 1);

    if (!this->npu->is_preemption_enabled()){
        this->layers_run = layer_app_manager->create_runlist();

        DEBUG_BLOCK(2,
        header_print_r("info", "Create  runs for all layers in the sequence with the new context length");
        )
        for (uint32_t i = 0; i < num_hidden_layers; i++){

            DEBUG_BLOCK(2,
            header_print_r("info", "Generate run for layer " + std::to_string(i) + " of type " + std::to_string(layer_types[i]));
            )
            // publish this layer's output scale before the layer itself runs
            this->layers_run.add(this->layer_scale_apps[i].create_run());
            switch(layer_types[i]){
                case gemma4_12b_global_layer:
                    this->layers_run.add(this->global_layer.create_run(this->x, this->proj_weights[i], this->rms_weights[i], this->rope_rms_weights[i], this->kv_caches[i]));
                    break;
                case gemma4_12b_swa_layer:
                    this->layers_run.add(this->swa_layer.create_run(this->x, this->proj_weights[i], this->rms_weights[i], this->rope_rms_weights[i], this->kv_caches[i]));
                    break;
            }
        }
    }

    this->_set_rope_rms_weights(L);

}

buffer<bf16> gemma4_12b_npu::Impl::forward(int ids){
    this->_drain_preload();
    
    _process_embedding(ids);
    if (!this->npu->is_preemption_enabled()){
        this->layers_run.execute();
        flm::npu::wait_for_completion(this->layers_run, "layers_run");
    }
    else{
        for (uint32_t i = 0; i < num_hidden_layers; i++){
            this->layer_scale_apps[i]();
            switch(layer_types[i]){
                case gemma4_12b_global_layer:
                    this->global_layer(this->x, this->proj_weights[i], this->rms_weights[i], this->rope_rms_weights[i], this->kv_caches[i]);
                    break;
                case gemma4_12b_swa_layer:
                    this->swa_layer(this->x, this->proj_weights[i], this->rms_weights[i], this->rope_rms_weights[i], this->kv_caches[i]);
                    break;
            }

            DEBUG_BLOCK(2,
            header_print("info", "Finished layer " + std::to_string(i) + " of type " + std::to_string(layer_types[i]) + " for the current token");
            this->x.sync_from_device();
            )
        }
    }
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
    flm::npu::wait_for_completion(this->lm_head_run, "lm_head_run");
    this->logits.sync_from_device();
    DEBUG_BLOCK(2,
    header_print("info", "Finished LM head run for the current token " + std::to_string(current_context_length));
    utils::print_matrix(this->logits_valid, vocab_size);
    )
    
    this->pre_load_run = this->layer_pre_load.create_run(); // create run for the preload xclbin, which has an empty sequence and will be used to preload the next layer's weights in the background while the current token is being processed
    this->pre_load_run.start();
    is_preload_launched = true;
    return this->logits_valid;
}

buffer<bf16> gemma4_12b_npu::Impl::prefill(std::vector<int>& ids, void* payload){
    // the preload run of the previous token is still resident; both prefill paths
    // reassign pre_load_run on their way out, which would destroy it mid flight
    this->_drain_preload();

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

/// @brief Prefill by replaying the decoding graph one token at a time.
/// @param ids the prompt tokens to append to the current context
/// @param payload unused, the multimodal payload is only handled by the mm path
/// @return the logits of the last token
buffer<bf16> gemma4_12b_npu::Impl::_prefill_with_mv(std::vector<int>& ids, void* payload){
    (void)payload; // no multimodal payload on the decoding path

    const int  L_in   = (int)ids.size();
    const int* tokens = ids.data();

    DEBUG_BLOCK(1,
    header_print("info", "Entering prefill with MV, input length: " + std::to_string(L_in)
                 + ", context length: " + std::to_string(this->current_context_length));
    )

    if (L_in == 0){
        header_print("warning", "Prefill received an empty token list, keeping the previous logits");
        return this->logits_valid;
    }

    if ((uint32_t)(this->current_context_length + L_in) > this->MAX_L){
        throw std::runtime_error("Prefilling " + std::to_string(L_in) + " tokens at context length "
                                 + std::to_string(this->current_context_length)
                                 + " exceeds the maximum context length of " + std::to_string(this->MAX_L));
    }

    // The runlist is only valid when preemption is disabled, otherwise the layers have to
    // be launched one at a time.
    const bool step_by_layer = this->npu->is_preemption_enabled();

    for (int i = 0; i < L_in; i++){
        this->_process_embedding(tokens[i]);

        if (!step_by_layer){
            // set_context_length() rebuilt this runlist for the current position
            this->layers_run.execute();
            flm::npu::wait_for_completion(this->layers_run, "layers_run");
        }
        else {
            for (uint32_t l = 0; l < num_hidden_layers; l++){
                this->layer_scale_apps[l]();
                switch(layer_types[l]){
                    case gemma4_12b_global_layer:
                        this->global_layer(this->x, this->proj_weights[l], this->rms_weights[l], this->rope_rms_weights[l], this->kv_caches[l]);
                        break;
                    case gemma4_12b_swa_layer:
                        this->swa_layer(this->x, this->proj_weights[l], this->rms_weights[l], this->rope_rms_weights[l], this->kv_caches[l]);
                        break;
                }
            }
        }

        DEBUG_BLOCK(2,
        header_print("info", "Finished token " + std::to_string(i) + "/" + std::to_string(L_in)
                     + " (id " + std::to_string(tokens[i]) + ") at position " + std::to_string(this->current_context_length));
        this->x.sync_from_device();
        buffer<bf16> valid_x = buffer<bf16>(this->x.data(), D);
        utils::print_matrix(valid_x, 256);
        this->x.sync_to_device();
        )

        // Advance one position: regenerates the layer sequences, the runlist and the
        // rope tables for the next token.
        this->set_context_length(this->current_context_length + 1);
    }

    // x still holds the hidden states of the last token, so a single lm_head run
    // yields the logits the caller samples from.
    this->lm_head_run.start();
    flm::npu::wait_for_completion(this->lm_head_run, "lm_head_run");
    this->logits.sync_from_device();

    DEBUG_BLOCK(1,
    header_print("info", "Finished LM head run after prefilling " + std::to_string(L_in) + " tokens");
    utils::print_matrix(this->logits_valid, vocab_size);
    )

    // keep the layer xclbin resident while the caller samples, same as forward()
    this->pre_load_run = this->layer_pre_load.create_run();
    this->pre_load_run.start();
    is_preload_launched = true;
    return this->logits_valid;
}

/// @brief Bring up the image/audio front end.
///
/// @note Called from the first prefill that carries a payload. Reading
///       image_audio_weights.q4nx in the constructor would make every text only session
///       pay ~105 MB of file IO, and would make a text only checkpoint fail to load.
void gemma4_12b_npu::Impl::_init_mm_projector(){
    if (this->mm_projector != nullptr){
        return;
    }
    this->mm_projector = std::make_unique<gemma4_12b_mm_projector>(this->npu, this->config);
    this->mm_projector->load_weights();
}

/// @brief Project the payload's media and write the result over the soft token rows.
///
/// gemma4-12b is encoder free, so a media item is simply a maximal run of one soft token
/// id in the prompt, and the payload carries exactly one feature block per run, in the
/// same left to right order. This walks both in lockstep: one run, one payload entry, one
/// projector call, one memcpy into `residual`.
///
/// @note The rows written here bypass the sqrt(hidden_size) embedding scale on purpose --
///       see the note on the declaration. `residual` already holds scaled *text* rows;
///       the media rows overwrite them and must stay unscaled.
void gemma4_12b_npu::Impl::_embed_media(const int* tokens, int L_in, buffer<bf16>& residual,
                                        void* payload){
    if (payload == nullptr){
        return;
    }
    gemma4_12b_multi_modal_payload_t* mm = (gemma4_12b_multi_modal_payload_t*)payload;
    const gemma4_12b_image_payload_t& ip = mm->image_payload;
    const gemma4_12b_audio_payload_t& ap = mm->audio_payload;
    if (ip.num_images == 0 && ap.num_audios == 0){
        return;
    }

    this->_init_mm_projector();
    const gemma4_12b_mm_dims& dims = this->mm_projector->dims();

    unsigned int next_image = 0;
    unsigned int next_audio = 0;

    //!< reused across items so a prompt with several images allocates once
    std::vector<bf16> feats;   //!< compacted input rows
    std::vector<int>  pos;     //!< their (x, y), vision only
    std::vector<bf16> proj;    //!< the projected rows

    for (int i = 0; i < L_in; ){
        const int id = tokens[i];
        // A video is a sequence of frames, each of which goes through the *image* tower
        // and arrives in the image payload; only its soft token budget differs.
        const bool is_vision = (id == image_token_id || id == video_token_id);
        const bool is_audio  = (id == audio_token_id);
        if (!is_vision && !is_audio){
            i++;
            continue;
        }

        int j = i;
        while (j < L_in && tokens[j] == id) j++;
        const size_t run = (size_t)(j - i);

        size_t rows = 0;
        if (is_vision){
            if (next_image >= ip.num_images
                || next_image >= ip.pixel_values.size()
                || next_image >= ip.image_grid_pairs_per_image.size()){
                header_print_r("warning", "prefill: image soft tokens at " + std::to_string(i)
                               + " have no payload entry, leaving them as text embeddings");
                i = j;
                continue;
            }

            const std::vector<bf16>& px = ip.pixel_values[next_image];
            const std::vector<int>&  gp = ip.image_grid_pairs_per_image[next_image];
            const size_t K = (size_t)dims.patch_dim;
            const size_t avail = px.size() / K;

            // The preprocessor pads its buffer out to the soft token budget with zero
            // patches and (-1, -1) positions. Those rows are not zero once they have been
            // through a LayerNorm and a biased matmul -- they are finite garbage -- so
            // they are dropped here rather than projected and thrown away.
            feats.clear();
            pos.clear();
            feats.reserve(avail * K);
            pos.reserve(avail * 2);
            for (size_t r = 0; r < avail; r++){
                const int x = (2 * r + 1 < gp.size()) ? gp[2 * r + 0] : -1;
                const int y = (2 * r + 1 < gp.size()) ? gp[2 * r + 1] : -1;
                if (x < 0 || y < 0){
                    continue;
                }
                feats.insert(feats.end(), px.begin() + r * K, px.begin() + (r + 1) * K);
                pos.push_back(x);
                pos.push_back(y);
            }
            rows = pos.size() / 2;

            if (rows != run){
                header_print_r("warning", "prefill: image #" + std::to_string(next_image + 1)
                               + " has " + std::to_string(rows) + " valid patches but "
                               + std::to_string(run) + " soft tokens, using "
                               + std::to_string(std::min(rows, run)));
                rows = std::min(rows, run);
            }

            proj.resize(rows * (size_t)D);
            this->mm_projector->project_vision(feats.data(), pos.data(), rows, proj.data());
            next_image++;
        }
        else {
            if (next_audio >= ap.num_audios || next_audio >= ap.mel_spectrograms.size()){
                header_print_r("warning", "prefill: audio soft tokens at " + std::to_string(i)
                               + " have no payload entry, leaving them as text embeddings");
                i = j;
                continue;
            }

            // Despite the field name this is raw PCM, audio_embed_dim samples per row, and
            // every row of it is valid: the extractor sizes the buffer to the frame count.
            const std::vector<bf16>& pcm = ap.mel_spectrograms[next_audio];
            rows = pcm.size() / (size_t)dims.audio_embed_dim;
            if (rows != run){
                header_print_r("warning", "prefill: audio #" + std::to_string(next_audio + 1)
                               + " has " + std::to_string(rows) + " frames but "
                               + std::to_string(run) + " soft tokens, using "
                               + std::to_string(std::min(rows, run)));
                rows = std::min(rows, run);
            }

            proj.resize(rows * (size_t)D);
            this->mm_projector->project_audio(pcm.data(), rows, proj.data());
            next_audio++;
        }

        if (rows > 0){
            memcpy(residual.data() + (size_t)i * D, proj.data(), rows * (size_t)D * sizeof(bf16));
        }

        DEBUG_BLOCK(1,
        header_print("prefill-media", std::string(is_vision ? "image" : "audio")
                     + " rows [" + std::to_string(i) + ", " + std::to_string(i + (int)rows)
                     + ") projected into the prompt embeddings");
        )

        i = j;
    }

    if (next_image != ip.num_images || next_audio != ap.num_audios){
        header_print_r("warning", "prefill: payload has " + std::to_string(ip.num_images)
                       + " image(s) / " + std::to_string(ap.num_audios) + " audio, but the prompt"
                       + " consumed " + std::to_string(next_image) + " / "
                       + std::to_string(next_audio));
    }
}

#ifndef MVPREFILL
/// @brief Bring up the batched (matmul) prefill path.
///
/// @note Called from the first _prefill_with_mm(). Registering the second xclbin in the
///       constructor would make every decode only session pay for it, and would make
///       models shipped without dequant_mm.xclbin fail to load at all.
///
/// @note Everything it builds -- the xclbin registration, the per (layer kind,
///       projection) apps, their control sequences and the activation scratch -- lives
///       in gemma4_12b_prefill_mm_context, see gemma4_12b_prefill.hpp.
void gemma4_12b_npu::Impl::_init_prefill_mm(){
    if (this->prefill_mm != nullptr){
        return;
    }
    this->prefill_mm = std::make_unique<gemma4_12b_prefill_mm_context>(
        this->npu, &this->desc, this->sequence.get(), this->config);
}

buffer<bf16> gemma4_12b_npu::Impl::_prefill_with_mm(std::vector<int>& ids, void* payload){

    int        L_in   = (int)ids.size();
    const int* tokens = ids.data();

    DEBUG_BLOCK(1,
    header_print("info", "Entering prefill with MM, input length: " + std::to_string(L_in)
                 + ", context length: " + std::to_string(this->current_context_length)
                 + ", host threads: " + std::to_string(gemma4_12b_cpu::prefill_threads()));
    )

    DEBUG_BLOCK(1,
    // Where the media sits in this prompt. gemma4-12b is encoder free, so image and audio
    // soft tokens sit inline in `ids` and one left to right pass finds them: a media item
    // is a maximal run of soft tokens of one kind, broken by the boi/eoi/boa/eoa markers
    // the processor always emits around it (those are plain text ids).
    int n_image = 0;
    int n_audio = 0;
    int tok_image = 0;
    int tok_audio = 0;
    int prev = -1;
    for (int i = 0; i < L_in; i++){
        const int id = tokens[i];
        if (id != image_token_id && id != audio_token_id){
            prev = -1;
            continue;
        }
        if (id != prev){
            // start of a new run: find where it ends so it can be reported in one line
            int j = i;
            while (j < L_in && tokens[j] == id) j++;
            const int len = j - i;
            if (id == image_token_id){ n_image++; tok_image += len; }
            else                     { n_audio++; tok_audio += len; }
            header_print("prefill-media",
                         std::string(id == image_token_id ? "image" : "audio") + " #"
                         + std::to_string(id == image_token_id ? n_image : n_audio)
                         + ": tokens [" + std::to_string(i) + ", " + std::to_string(j) + ")"
                         + ", positions [" + std::to_string(this->current_context_length + i) + ", "
                         + std::to_string(this->current_context_length + j) + ")"
                         + ", " + std::to_string(len) + " soft tokens");
        }
        prev = id;
    }
    header_print("prefill-media", "prompt has " + std::to_string(n_image) + " image(s) ("
                 + std::to_string(tok_image) + " soft tokens) and "
                 + std::to_string(n_audio) + " audio segment(s) ("
                 + std::to_string(tok_audio) + " soft tokens)");

    // ...and what the preprocessor actually handed us for those runs. The soft token
    // counts here must line up one for one with the runs above: that is how many rows of
    // projected features get scattered into the embeddings.
    if (payload == nullptr){
        header_print("prefill-media", "no multimodal payload");
    }
    else {
        gemma4_12b_multi_modal_payload_t* mm = (gemma4_12b_multi_modal_payload_t*)payload;
        const gemma4_12b_image_payload_t& ip = mm->image_payload;
        const gemma4_12b_audio_payload_t& ap = mm->audio_payload;

        header_print("prefill-media", "payload: " + std::to_string(ip.num_images)
                     + " image(s), " + std::to_string(ap.num_audios) + " audio segment(s)");

        for (unsigned int i = 0; i < ip.num_images; i++){
            std::string line = "  image payload #" + std::to_string(i + 1) + ": ";
            if (i < ip.pixel_values.size()){
                line += "pixel_values " + std::to_string(ip.pixel_values[i].size()) + " bf16";
            }
            if (i < ip.image_patch__element_per_patch.size()){
                line += ", patch grid " + std::to_string(ip.image_patch__element_per_patch[i].first)
                      + " x " + std::to_string(ip.image_patch__element_per_patch[i].second);
            }
            if (i < ip.valid_patch_size_per_image.size()){
                line += ", valid patches " + std::to_string(ip.valid_patch_size_per_image[i]);
            }
            if (i < ip.image_grid_pairs_per_image.size()){
                line += ", grid pairs " + std::to_string(ip.image_grid_pairs_per_image[i].size());
            }
            if (i < ip.num_soft_tokens_per_image.size()){
                line += ", soft tokens " + std::to_string(ip.num_soft_tokens_per_image[i]);
            }
            header_print("prefill-media", line);
        }

        for (unsigned int i = 0; i < ap.num_audios; i++){
            std::string line = "  audio payload #" + std::to_string(i + 1) + ": ";
            if (i < ap.mel_spectrograms.size()){
                line += "mel " + std::to_string(ap.mel_spectrograms[i].size()) + " bf16";
            }
            if (i < ap.mel_spectrogram_frames_per_audio.size()
                && i < ap.mel_spectrogram_bins_per_audio.size()){
                line += " (" + std::to_string(ap.mel_spectrogram_frames_per_audio[i])
                      + " frames x " + std::to_string(ap.mel_spectrogram_bins_per_audio[i]) + " bins)";
            }
            if (i < ap.num_soft_tokens_per_audio.size()){
                line += ", soft tokens " + std::to_string(ap.num_soft_tokens_per_audio[i]);
            }
            header_print("prefill-media", line);
        }

        //!< a mismatch here means the token runs and the features disagree, which would
        //!< scatter the wrong number of rows into the embeddings
        if ((int)ip.num_images != n_image || (int)ap.num_audios != n_audio){
            header_print_r("prefill-media", "payload/token mismatch: tokens say "
                           + std::to_string(n_image) + " image(s) / " + std::to_string(n_audio)
                           + " audio, payload says " + std::to_string(ip.num_images)
                           + " / " + std::to_string(ap.num_audios));
        }
    }
    )

    if (L_in == 0){
        header_print("warning", "Prefill received an empty token list, keeping the previous logits");
        return this->logits_valid;
    }

    if ((uint32_t)(this->current_context_length + L_in) > this->MAX_L){
        throw std::runtime_error("Prefilling " + std::to_string(L_in) + " tokens at context length "
                                 + std::to_string(this->current_context_length)
                                 + " exceeds the maximum context length of " + std::to_string(this->MAX_L));
    }

    this->_init_prefill_mm();

    const int L_prompt_begin = this->current_context_length;

    //!< a debug build scores the npu matmuls against the host kernels as it goes, see
    //!< gemma4_12b_prefill_xref.hpp. That needs no dedicated prompt and no early exit, so
    //!< the prefill below is the same one a release build runs.
    const int layers = num_hidden_layers;

    // The whole prompt goes through the stack in one pass, so the graph is sized for it
    // here: the scratch, and the row count every projection sequence is generated for.
    gemma4_12b_prefill_mm_shape shape;
    shape.update(L_prompt_begin, L_in);
    this->prefill_mm->reserve(shape);

    // Image and video soft tokens attend to each other in both directions, which is the
    // one place gemma4-12b's mask departs from plain causal / sliding. The runs are found
    // once here and every layer's attention corrects its own rows for them; audio is
    // deliberately not included, it is masked exactly like text.
    this->prefill_mm->set_vision_blocks(tokens, L_in, L_prompt_begin,
                                        image_token_id, video_token_id);

    //!< r0 / r1 / r2. The only stage-to-stage tensor no projection touches, so it is the
    //!< one that stays a plain host buffer. Everything else lives in the device scratch
    //!< of `prefill_mm` and never surfaces here. Padded like the device scratch, and
    //!< zeroed because the host stages only ever write the first L_in rows.
    buffer<bf16> residual((size_t)shape.L_padded * D);
    memset(residual.data(), 0, residual.size() * sizeof(bf16));

    DEBUG_BLOCK(1,
    header_print("info", "Prefill rows [" + std::to_string(shape.L_begin) + ", "
                 + std::to_string(shape.L_end) + "), padded to "
                 + std::to_string(shape.L_padded));
    )

    // ---- embeddings ----
    // Embedding::forward() already multiplies by sqrt(hidden_size); _embed_media() then
    // overwrites the soft token rows with projected features, which are *not* scaled.
    this->embedding->prefetch(tokens, (size_t)L_in); // batch the page-ins if the mapped table was dropped
    for (int i = 0; i < L_in; i++){
        buffer<bf16> e = this->embedding->forward(tokens[i]);
        memcpy(residual.data() + (size_t)i * D, e.data(), D * sizeof(bf16));
    }
    this->_embed_media(tokens, L_in, residual, payload);


    for (int l = 0; l < layers; l++){
        this->prefill_mm->forward(l, residual, shape,
                                  this->proj_weights[l], this->rms_weights[l],
                                  this->rope_rms_weights[l], this->kv_caches[l],
                                  this->layer_scale[l]);

        DEBUG_BLOCK(2,
        header_print("info", "Finished layer " + std::to_string(l) + "/" + std::to_string(layers));
        )
    }

    DEBUG_BLOCK(2,
    header_print("info", "Finished all layers, last row at position "
                 + std::to_string(shape.L_end - 1));
    buffer<bf16> last_row = buffer<bf16>(residual.data() + (size_t)(L_in - 1) * D, D);
    utils::print_matrix(last_row, 256);
    )

    // The hidden state of the last prompt token is what the lm head consumes; the final
    // rms norm weights live at x + D and are applied by the lm head kernel itself.
    memcpy(this->x.data(), residual.data() + (size_t)(L_in - 1) * D, D * sizeof(bf16));
    this->x.sync_to_device();

    // Advance the context: regenerates the decoding sequences, the runlist and the rope
    // tables so that the first sampled token continues from the right position.

    this->lm_head_weights.sync_to_device();
    this->lm_head_run.start();
    this->set_context_length(L_prompt_begin + L_in);
    flm::npu::wait_for_completion(this->lm_head_run, "lm_head_run");
    this->logits.sync_from_device();

    DEBUG_BLOCK(1,
    header_print("info", "Finished LM head run after prefilling " + std::to_string(L_in) + " tokens");
    utils::print_matrix(this->logits_valid, vocab_size);
    )

    // keep the layer xclbin resident while the caller samples, same as forward()
    this->pre_load_run = this->layer_pre_load.create_run();
    this->pre_load_run.start();
    is_preload_launched = true;
    return this->logits_valid;
}
#endif // !MVPREFILL

/// @brief Row index of absolute position `idx` inside the cache of `layer_idx`.
/// @note  Global layers are indexed by the absolute position; sliding ones are a ring of
///        SLIDING_LENGTH rows, exactly like _receive_kv_cache() writes them.
size_t gemma4_12b_npu::Impl::_kv_cache_row(int layer_idx, int idx) const {
    return is_global_layer(layer_types[layer_idx]) ? (size_t)idx
                                                   : (size_t)(idx % SLIDING_LENGTH);
}

buffer<bf16> gemma4_12b_npu::Impl::get_k_cache(int layer_idx, int idx){
    const size_t _DK = is_global_layer(layer_types[layer_idx]) ? (size_t)DK : (size_t)SWA_DK;
    this->kv_caches[layer_idx].sync_from_device();
    buffer<bf16> k_cache(_DK);
    bf16* k_cache_ptr = this->kv_caches[layer_idx].data();
    const size_t offset = this->_kv_cache_row(layer_idx, idx) * _DK;
    memcpy(static_cast<void*>(k_cache.data()), static_cast<void*>(k_cache_ptr + offset), _DK * sizeof(bf16));

    return k_cache;
}

buffer<bf16> gemma4_12b_npu::Impl::get_v_cache(int layer_idx, int idx){
    const size_t _DV = is_global_layer(layer_types[layer_idx]) ? (size_t)DV : (size_t)SWA_DV;
    this->kv_caches[layer_idx].sync_from_device();
    buffer<bf16> v_cache(_DV);
    // the V half starts at the midpoint of the layer's cache, see _attend_global()
    bf16* v_cache_ptr = this->kv_caches[layer_idx].data() + this->kv_caches[layer_idx].size() / 2;
    const size_t offset = this->_kv_cache_row(layer_idx, idx) * _DV;
    memcpy(static_cast<void*>(v_cache.data()), static_cast<void*>(v_cache_ptr + offset), _DV * sizeof(bf16));
    return v_cache;
}

void gemma4_12b_npu::Impl::clear_context(){
    this->set_context_length(0);
    for (int i = 0; i < num_hidden_layers; i++){
        this->kv_caches[i].sync_from_device();
        memset(this->kv_caches[i].data(), 0, this->kv_caches[i].size() * sizeof(bf16));
        this->kv_caches[i].sync_to_device();
    }
    this->dirty_rows = 0;
}

kv_row_layout gemma4_12b_npu::Impl::kv_layout() const{
    kv_row_layout l;
    l.DK = DK; l.DV = DV; l.striped = false;
    l.v_offset = static_cast<size_t>(MAX_L) * DK; // a global layer holds K rows, then V rows
    return l;
}

std::unique_ptr<hybrid_engine_state> gemma4_12b_npu::Impl::new_state(){
    size_t ring_bytes = 0, global_layers = 0;
    for (size_t i = 0; i < kv_caches.size(); i++){
        if (attn_mask[i]) global_layers++;
        else ring_bytes += kv_caches[i].size() * sizeof(bf16);
    }
    return kv_rows_new_state(global_layers, kv_layout(), ring_bytes);
}

int gemma4_12b_npu::Impl::read_state(hybrid_engine_state& state, int kv_from){
    header_print_r("FLM", "Reading state at context length " + std::to_string(this->current_context_length) + ", K/V rows from " + std::to_string(kv_from));
    const int length = kv_rows_read(state, this->kv_caches, kv_layout(), this->current_context_length, kv_from, -1, &attn_mask);
    if (length < 0) return -1;
    for (size_t i = 0; i < kv_caches.size(); i++){
        if (attn_mask[i]) continue;
        this->kv_caches[i].sync_from_device();
        state.state[i] = buffer<bf16>(this->kv_caches[i].size());
        memcpy(static_cast<void*>(state.state[i].data()), static_cast<void*>(this->kv_caches[i].data()), this->kv_caches[i].size() * sizeof(bf16));
    }
    return length;
}

int gemma4_12b_npu::Impl::write_state_kv(const hybrid_engine_state& state){
    return kv_rows_write(state, this->kv_caches, kv_layout(), this->MAX_L, &attn_mask);
}

int gemma4_12b_npu::Impl::write_state(const hybrid_engine_state& state){
    // check the rings before anything is written
    bool ok = state.state.size() == this->kv_caches.size();
    for (size_t i = 0; ok && i < kv_caches.size(); i++){
        if (!attn_mask[i] && state.state[i].size() != this->kv_caches[i].size()) ok = false;
    }
    const int length = ok ? kv_rows_write(state, this->kv_caches, kv_layout(), this->MAX_L, &attn_mask) : -1;
    if (length < 0){
        header_print_r("FLM", "No valid state to write");
        return -1;
    }
    header_print_r("FLM", "Writing state at context length " + std::to_string(length));
    for (size_t i = 0; i < kv_caches.size(); i++){
        if (attn_mask[i]) continue;
        this->kv_caches[i].sync_from_device();
        memcpy(static_cast<void*>(this->kv_caches[i].data()), static_cast<const void*>(state.state[i].data()), this->kv_caches[i].size() * sizeof(bf16));
        this->kv_caches[i].sync_to_device();
    }
    // rows past the length hold what the requests since the state was read left there: put back the zeros of a fresh engine
    kv_rows_zero(this->kv_caches, kv_layout(), static_cast<size_t>(length), std::min(this->dirty_rows, static_cast<size_t>(this->MAX_L)), &attn_mask);
    this->dirty_rows = static_cast<size_t>(length);
    this->set_context_length(length);
    return length;
}

int gemma4_12b_npu::Impl::get_current_context_length(){
    return this->current_context_length;
}

void gemma4_12b_npu::Impl::update_max_length(uint32_t MAX_L){
    // Round to a power of two exactly like the constructor does. Beyond keeping the two
    // paths consistent, this is what guarantees MAX_L is a whole number of L_CHUNKs, so
    // _move_kv_cache()'s padded replay length can never exceed the rows the cache holds.
    if (MAX_L > 0){
        float log2_MAX_L = std::log2(MAX_L);
        if (log2_MAX_L != std::floor(log2_MAX_L)){
            MAX_L = 1u << ((uint32_t)std::floor(log2_MAX_L) + 1);
        }
    }
    if (MAX_L <= this->MAX_L){
        header_print("FLM", "New length is shorter than the current length, no need to update!");
        return; // no need to update
	}
    const size_t old_K_rows = (size_t)this->MAX_L; //!< rows the old global halves held
    this->MAX_L = MAX_L;
    this->sequence->set_max_length(MAX_L);

    // The rows already in the caches stay valid, they just move: the global halves are
    // [MAX_L * DK | MAX_L * DV], so growing MAX_L pushes the V half further out. Migrate
    // both halves into the new layout instead of dropping the conversation, the way
    // qwen3_5vl_npu::Impl::update_max_length() does.
    const size_t rows_in_use = std::min((size_t)this->current_context_length, old_K_rows);
    buffer<bf16> kv_cache_temp;
    for (int i = 0; i < num_hidden_layers; i++){
        gemma4_12b_layer_type_t type = layer_types[i];
        // only the global layers grow with the context, the sliding ones are already
        // sized for their window, so their ring is left exactly as it is
        if (!is_global_layer(type)){
            continue;
        }
        this->kv_caches[i].sync_from_device();
        const size_t old_size = this->kv_caches[i].size();
        if (kv_cache_temp.size() < old_size){
            kv_cache_temp = buffer<bf16>(old_size);
        }
        memcpy(static_cast<void*>(kv_cache_temp.data()),
               static_cast<void*>(this->kv_caches[i].data()), old_size * sizeof(bf16));

        this->kv_caches[i] = layer_app_manager->create_bo_buffer<bf16>(desc.get_kv_cache_size(type, MAX_L));
        this->kv_caches[i].memset((bf16)0); //!< everything past rows_in_use must read as zero
        bf16* new_K = this->kv_caches[i].data();
        bf16* new_V = new_K + this->kv_caches[i].size() / 2;
        memcpy(static_cast<void*>(new_K),
               static_cast<void*>(kv_cache_temp.data()), rows_in_use * DK * sizeof(bf16));
        memcpy(static_cast<void*>(new_V),
               static_cast<void*>(kv_cache_temp.data() + old_size / 2), rows_in_use * DV * sizeof(bf16));
        this->kv_caches[i].sync_to_device();
    }

    // the prefill graph caches nothing that depends on MAX_L any more, but it does hold
    // apps bound to the old cache buffers, so let the next prefill rebuild it
    this->prefill_mm.reset();

    is_checkpoint_valid = false; //!< the checkpoint was taken against the old layout
    this->set_context_length(this->current_context_length); // regenerate the sequences
}


void gemma4_12b_npu::Impl::load_weights(Q4NX& q4nx){
    // model.embed_tokens.weight is bf16 in the checkpoint; Embedding appends ".weight" itself
    this->embedding->init_weights(q4nx, utils::path_join(config.model_path, "model.q4nx"), "model.embed_tokens");
    desc.embed_tokens.load();

    for (int layer_idx = 0; layer_idx < num_hidden_layers; layer_idx++){
        gemma4_12b_layer_type_t type = layer_types[layer_idx];
        DEBUG_BLOCK(2,
        header_print("info", "Loading weights for layer " + std::to_string(layer_idx)
                     + " of type " + std::string(is_global_layer(type) ? "Global" : "SWA"));
        )
        // the description owns the buffer layout, so it performs the copies itself
        this->layer_scale[layer_idx] = desc.load_layer_weights(layer_idx, type, q4nx,
                                                               this->proj_weights[layer_idx],
                                                               this->rms_weights[layer_idx],
                                                               this->rope_rms_weights[layer_idx]);
    }

    // now that every layer_scale is known, bake it into its own tiny sequence
    for (int layer_idx = 0; layer_idx < num_hidden_layers; layer_idx++){
        this->sequence->gen_layer_scale_seq(this->layer_scale_apps[layer_idx].seq(),
                                            this->layer_scale[layer_idx]);
    }

    // the final rms norm lives right behind the hidden states in x
    desc.load_final_norm(q4nx, this->x.data() + D);
    // lm_head is stored explicitly even though the embeddings are tied
    desc.load_lm_head_weights(q4nx, this->lm_head_weights);

    this->lm_head_weights.sync_to_device();

    for (int layer_idx = 0; layer_idx < num_hidden_layers; layer_idx++){
        this->proj_weights[layer_idx].sync_to_device();
        this->rms_weights[layer_idx].sync_to_device();
        this->rope_rms_weights[layer_idx].sync_to_device();
    }

    DEBUG_BLOCK(1,
    header_print("info", "Finished loading all layer weights");
    )
    this->sequence->update_buffer_offsets(this->proj_offsets);


    DEBUG_BLOCK(1,
    header_print("info", "Finished updating sequence buffer offsets based on loaded weights");
    )

    this->clear_context();

    DEBUG_BLOCK(1,
    header_print("info", "Finished clearing context after loading weights");
    )
}
int gemma4_12b_npu::Impl::checkpoint(){
    if (!is_checkpoint_valid){
        _allocate_checkpoint_buffers();
    } // use lazy allocation
    header_print_r("FLM", "Creating checkpoint at context length " + std::to_string(this->current_context_length));
    checkpoint_context_length = this->current_context_length;
    for (int i = 0; i < num_hidden_layers; i++){
        if (!is_global_layer_idx(i)){
            this->kv_caches[i].sync_from_device();
            memcpy(this->kv_checkpoint[i].data(), this->kv_caches[i].data(), this->kv_caches[i].size() * sizeof(bf16));
            this->kv_caches[i].sync_to_device();
        }
    }
    is_checkpoint_valid = true;
    return this->checkpoint_context_length;
}

int gemma4_12b_npu::Impl::restore(){
    if (!is_checkpoint_valid){
        header_print("FLM", "No valid checkpoint found, cannot restore context!");
        return -1;
    }
    header_print_r("FLM", "Restoring checkpoint at context length " + std::to_string(checkpoint_context_length));
    this->current_context_length = checkpoint_context_length;
    set_context_length(this->current_context_length);
    for (int i = 0; i < num_hidden_layers; i++){
        this->kv_caches[i].sync_from_device();
        if (!is_global_layer_idx(i)){
            this->kv_caches[i].sync_from_device();
            memcpy(this->kv_caches[i].data(), this->kv_checkpoint[i].data(), this->kv_caches[i].size() * sizeof(bf16));
        }
        else {
            int offset = (size_t)current_context_length * (DK);
            // first half is K, second half is V, and they are stored contiguously in kv_cache
            memset(this->kv_caches[i].data() + offset, 0, (this->kv_caches[i].size() / 2 - offset) * sizeof(bf16)); // zero out the part that is not restored for global layers, since we only restore the sliding part for global layers
            memset(this->kv_caches[i].data() + this->kv_caches[i].size() / 2 + offset, 0, (this->kv_caches[i].size() / 2 - offset) * sizeof(bf16)); // zero out the second half for swa
        }
        this->kv_caches[i].sync_to_device();
    }
    return this->current_context_length;
}

void gemma4_12b_npu::Impl::_allocate_checkpoint_buffers(){
    // allocate kv cache checkpoint for sliding layers
    this->kv_checkpoint.clear();
    this->kv_checkpoint.resize(num_hidden_layers);
    for (int i = 0; i < num_hidden_layers; i++){
        if(!is_global_layer_idx(i)){
            this->kv_checkpoint[i] = buffer<bf16>(this->kv_caches[i].size());
        }
    }
    is_checkpoint_valid = false;
}

gemma4_12b_npu::Impl::~Impl(){
    // a preload run may still be in flight; it must complete before the
    // xrt objects it references are torn down. Destruction must not throw.
    try{
        this->_drain_preload();
    }
    catch (const std::exception& e){
        header_print("FLM", std::string("Failed to wait for the pending preload run: ") + e.what());
    }
}

void gemma4_12b_npu::Impl::_drain_preload(){
    if (!is_preload_launched){
        return;
    }
    is_preload_launched = false;
    flm::npu::wait_for_completion(this->pre_load_run, "pre_load_run");
}

// ===============================================
// Externals for gemma4_12b_npu
// ===============================================
gemma4_12b_npu::gemma4_12b_npu(LM_Config config, npu_xclbin_manager *npu_instance, int MAX_L){
    
    this->load_vision_preprocess_parameters(config);
    this->load_audio_preprocess_parameters(config);
    this->_impl = new Impl(config, npu_instance, this,   MAX_L);
}

buffer<bf16> gemma4_12b_npu::forward(int ids){
    return this->_impl->forward(ids);
}

buffer<bf16> gemma4_12b_npu::prefill(std::vector<int>& ids, void* payload){
    try {
        buffer<bf16> result = this->_impl->prefill(ids, payload);
        return result;
    }
    catch (const std::runtime_error& e) {
        header_print("FLM", e.what());
        throw;
    }
}

void gemma4_12b_npu::set_context_length(int L){
    std::cout << "Setting context length is not supported for qwen3vl_npu!" << std::endl;
}

void gemma4_12b_npu::load_weights(Q4NX& q4nx){
    this->_impl->load_weights(q4nx);
}

void gemma4_12b_npu::update_max_length(uint32_t MAX_L){
    this->_impl->update_max_length(MAX_L);
}

void gemma4_12b_npu::clear_context(){
    this->_impl->clear_context();
}

int gemma4_12b_npu::checkpoint(){
    return this->_impl->checkpoint();
}

int gemma4_12b_npu::restore(){
    return this->_impl->restore();
}

std::unique_ptr<hybrid_engine_state> gemma4_12b_npu::new_state(){
    return this->_impl->new_state();
}

int gemma4_12b_npu::read_state(hybrid_engine_state& state, int kv_from){
    return this->_impl->read_state(state, kv_from);
}

int gemma4_12b_npu::write_state(const hybrid_engine_state& state){
    return this->_impl->write_state(state);
}

int gemma4_12b_npu::write_state_kv(const hybrid_engine_state& state){
    return this->_impl->write_state_kv(state);
}

extern "C" const flm_state_api* gemma4_12b_npu_state_api(void* engine){
    return hybrid_state_api<gemma4_12b_npu>(engine);
}

buffer<bf16> gemma4_12b_npu::get_k_cache(int layer_idx, int idx){
    return this->_impl->get_k_cache(layer_idx, idx);
}

buffer<bf16> gemma4_12b_npu::get_v_cache(int layer_idx, int idx){
    return this->_impl->get_v_cache(layer_idx, idx);
}

int gemma4_12b_npu::get_current_context_length(){
    return this->_impl->get_current_context_length();
}

gemma4_12b_npu::~gemma4_12b_npu(){
    delete this->_impl;
}
