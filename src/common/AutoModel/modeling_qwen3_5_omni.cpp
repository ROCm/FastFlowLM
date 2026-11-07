/// \file modeling_qwen3_5_omni.cpp
/// \brief Qwen3_5_Omni driver: tokenizer/template/sampler + prefill/decode loop.
/// \author FastFlowLM Team

#include "AutoModel/modeling_qwen3_5_omni.hpp"

Qwen3_5_Omni::Qwen3_5_Omni(flm_rt::device* npu_device_inst)
    : AutoModel(npu_device_inst) {}

/// \brief Pull the thinker-scope vision config (with VL fallbacks).
static uint32_t json_u32(const nlohmann::json& j, const char* key, uint32_t dflt) {
    if (j.contains(key) && !j[key].is_null()) return j[key].get<uint32_t>();
    return dflt;
}


static float json_fp32(const nlohmann::json& j, const char* key, float dflt) {
    if (j.contains(key) && !j[key].is_null()) return j[key].get<float>();
    return dflt;
}

void Qwen3_5_Omni::setup_tokenizer(std::string model_path) {
    std::string tokenizer_config_path = model_path + "/tokenizer_config.json";
    std::ifstream fs_config(tokenizer_config_path, std::ios::in | std::ios::binary);
    if (fs_config.fail()) {
        header_print("ERROR", "Cannot open " << tokenizer_config_path);
        exit(1);
    }
    std::string data_config((std::istreambuf_iterator<char>(fs_config)), std::istreambuf_iterator<char>());
    fs_config.close();
    auto tokenizer_config = nlohmann::json::parse(data_config);

    this->has_bos_token = !tokenizer_config["bos_token"].is_null();

    std::string chat_template_path = model_path + "/chat_template.jinja";
    if (std::filesystem::exists(chat_template_path)) {
        std::ifstream fs_template(chat_template_path, std::ios::in | std::ios::binary);
        std::string chat_template((std::istreambuf_iterator<char>(fs_template)), std::istreambuf_iterator<char>());
        fs_template.close();
        tokenizer_config["chat_template"] = chat_template;
    } else if (tokenizer_config.find("chat_template") == tokenizer_config.end()) {
        header_print("ERROR", "No template file found and no chat_template in tokenizer_config.json");
        exit(1);
    }

    if (tokenizer_config["eos_token"].is_null()) {
        header_print("ERROR", "eos_token is null in tokenizer_config.json");
        exit(1);
    }

    this->chat_tmpl = std::make_unique<minja::chat_template>(
        tokenizer_config["chat_template"],
        this->has_bos_token ? tokenizer_config["bos_token"] : "",
        tokenizer_config["eos_token"]);

    this->bos_token_id = this->has_bos_token ? tokenizer_config["bos_token_id"].get<int>() : -1;
    this->eos_token = tokenizer_config["eos_token"].get<std::string>();
    this->eos_token_ids.clear();
    for (auto& token : tokenizer_config["eos_token_id"]) {
        this->eos_token_ids.push_back(token.get<int>());
    }
    this->user_system_prompt = "";
    this->extra_context["user_system_prompt"] = this->user_system_prompt;
    this->_detect_chat_format(tokenizer_config["chat_template"].get<std::string>());
}

void Qwen3_5_Omni::load_model(std::string model_path, json model_info, int default_context_length, bool enable_preemption, const std::string& backend) {
    if (this->is_model_loaded && this->model_path == model_path) {
        header_print("FLM", "Model already loaded: " << this->model_path);
        return;
    }
    this->model_path = model_path;
    header_print("FLM", "Loading model: " << this->model_path);

    this->lm_config = std::make_unique<LM_Config>();
    this->lm_config->from_pretrained(this->model_path);

    // The omni thinker vocab lives in thinker_config.text_config, so the
    // top-level "vocab_size" is 0. Republish it at the top level so every
    // downstream reader (set_sampler, set_topk, ...) sees the real vocab.
    {
        auto& jc = this->lm_config->_json_config;
        if (jc.contains("thinker_config") && jc["thinker_config"].contains("text_config")) {
            jc["vocab_size"] =
                json_u32(jc["thinker_config"]["text_config"], "vocab_size", this->lm_config->get("vocab_size"));
        }
    }

    if (this->npu_device_inst == nullptr) {
        header_print("ERROR", "NPU device instance is nullptr");
        exit(1);
    }
    this->npu = std::make_unique<npu_xclbin_manager>(npu_device::device_npu2, this->npu_device_inst, enable_preemption);
    this->enable_preemption = enable_preemption;

    if (default_context_length != -1) {
        this->MAX_L = default_context_length;
    } else {
        this->MAX_L = model_info.value("default_context_length", 4096);
    }

    // vision config (thinker_config.vision_config) with VL fallbacks
    const auto& jc = this->lm_config->_json_config;
    if (jc.contains("thinker_config") && jc["thinker_config"].contains("vision_config")) {
        const auto& vc = jc["thinker_config"]["vision_config"];
        const nlohmann::json &vision_config = this->lm_config->_json_config["vision_config"];

        this->patch_size          = json_u32(vc, "patch_size", this->patch_size);
        this->temporal_patch_size = json_u32(vc, "temporal_patch_size", this->temporal_patch_size);


        this->merge_size          = json_u32(vision_config, "merge_size", this->merge_size);
        this->shortest_edge       = json_u32(vision_config, "shortest_edge", this->shortest_edge);
        this->longest_edge        = json_u32(vision_config, "longest_edge", this->longest_edge);
        this->vision_rescale_factor = json_fp32(vision_config, "rescale_factor", this->vision_rescale_factor);
        this->vision_image_mean = json_fp32(vision_config, "rescale_mean", this->vision_image_mean);
        this->vision_image_std = json_fp32(vision_config, "rescale_std", this->vision_image_std);
    }


    this->engine = std::make_unique<qwen3_5_omni>(*this->lm_config, this->npu.get(), this->MAX_L);
    {
        Q4NX q4nx(this->model_path);
        this->engine->load_weights(q4nx);
    }
    this->engine->clear_context();

    this->tokenizer = std::make_unique<Tokenizer>(this->model_path);
    this->setup_tokenizer(this->model_path);

    // default greedy-ish sampler (overridable via set_* helpers)
    sampler_config config;
    config.top_k = 20;
    config.top_p = 0.8;
    config.min_p = 0.0;
    config.temperature = 0.7;
    config.rep_penalty = 1.0;
    config.freq_penalty = 1.0;
    config.pre_penalty = 1.5f;
    this->set_sampler(config);

    this->is_model_loaded = true;
    this->last_token = -1;
    this->total_tokens = 0;
    this->token_history.clear();
    this->token_history.reserve(this->MAX_L);
    for (size_t i = 0; i < PROFILER_TYPE_NUM; i++) {
        this->profiler_list[i].reset();
    }





}

std::string Qwen3_5_Omni::apply_chat_template(nlohmann::ordered_json& messages, nlohmann::ordered_json tools, bool continue_final_message) {
    minja::chat_template_inputs inputs;
    inputs.messages = messages;
    inputs.extra_context = this->extra_context;
    if (!tools.empty()) inputs.tools = tools;
    return this->_render_chat_template(inputs, continue_final_message);
}

bool Qwen3_5_Omni::insert(chat_meta_info_t& meta_info, lm_uniform_input_t& input, std::function<bool()> is_cancelled) {
    this->profiler_list[TKOEN_ENCODE_TIME].start();

    if (input.messages.empty() && input.prompt.empty()) {
        header_print("WARNING", "No messages or prompt provided");
        return false;
    }

    qwen3_5_omni_payload_t payload;
    qwen3_5_omni_image_payload_t& image_payload = payload.image_payload;
    qwen3_5_omni_audio_payload_t& audio_payload = payload.audio_payload;
    image_payload.num_images = 0;
    audio_payload.num_audios = 0;

    std::vector<audio_data_t> audio_data_list;


    // ---- build templated text ----
    std::string templated_text;

    if (!input.messages.empty()) { // Server Processing
        int total_images = 0;
        // Build qwen_messages (used for the template) in the same pass that
        // decodes images/audio, so an invalid image (e.g. bad base64) never
        // gets an image-soft-token placeholder; otherwise image_payload would
        // fall out of sync with the placeholders and the prompt would still
        // try to prefill a nonexistent image.
        nlohmann::ordered_json qwen_messages = nlohmann::ordered_json::array();
        for (const auto& message : input.messages) {
            if (!message.contains("images") && !message.contains("audios")) {
                qwen_messages.push_back(message);
                continue;
            }
            nlohmann::ordered_json new_content = nlohmann::ordered_json::array();
            // ---- image preprocessing ----
            if (message.contains("images")) {
                for (auto& img : message["images"]) {
                    std::string img_str = img.get<std::string>();

                    qwen3_5_omni_image_t image = this->load_image_base64(img_str);
                    if (image.width <= 0 || image.height <= 0) {
                        header_print("ERROR", "Skipping invalid base64 image; prefilling language only for this item");
                        continue;
                    }

                    std::vector<int> grid_pair;
                    uint32_t valid_patch_size = 0;
                    uint32_t num_soft_tokens = 0;

                    this->preprocess_image(image, image_payload._processed_pixel_values, grid_pair, valid_patch_size, num_soft_tokens);
                    if (grid_pair.size() < 2 || grid_pair[0] <= 0 || grid_pair[1] <= 0) {
                        header_print("ERROR", "Skipping image that failed to preprocess");
                        continue;
                    }

                    image_payload.image_grid_h_w.push_back(grid_pair);
                    image_payload.num_soft_tokens_per_image.push_back(num_soft_tokens);
                    image_payload.num_images++;
                    total_images++;

                    new_content.push_back({{"type", "image"}, {"image", img}});
                }
            }
            // ---- audio preprocessing ----
            if (message.contains("audios")) {
                for (auto& aud : message["audios"]) {
                    std::string aud_str = aud.get<std::string>();
                    audio_data_t audio_data = this->load_audio_base64(aud_str, this->audio_sampling_rate, MonoDownmixMode::MEAN);
                    if (audio_data.channels > 1) {
                        header_print("ERROR", "only mono audio is supported");
                        meta_info.error_message = "Only mono audio is supported";
                        return false;
                    }
                    audio_data_list.push_back(audio_data);
                    new_content.push_back({{"type", "audio"}, {"audio", aud}});
                }
            }
            new_content.push_back({{"type", "text"}, {"text", message.value("content", "")}});
            nlohmann::ordered_json new_item = {
                {"role", message.value("role", "user")},
                {"content", new_content}
            };
            qwen_messages.push_back(new_item);
        }
        // once for every clip of the conversation, after the loop: run per message, it re-padded and re-extracted the
        // clips of earlier messages and appended them again, so the payload no longer matched the placeholders
        if(audio_data_list.size()> 0){
            pad_audio(audio_data_list);
            this->extract_spectrogram(audio_data_list, audio_payload);
            for (unsigned int i = 0; i < audio_payload.num_audios; i++) {
                audio_payload.audio_tokens.push_back(
                    get_audio_tokens(
                        _get_feat_extract_output_length(
                            audio_payload.mel_spectrogram_frames_per_audio[i]
                        )
                    )
                );
            }
        }
        header_print("FLM", "Total images: " << total_images);
        templated_text = this->apply_chat_template(qwen_messages, input.tools, meta_info.continue_final_message);
    }
    else { // CLI Processing
        // ---- image preprocessing ----
        for (const auto& img_str : input.images) {
            qwen3_5_omni_image_t image = this->load_image(img_str);
            std::vector<int> grid_pair;
            uint32_t valid_patch_size = 0;
            uint32_t num_soft_tokens = 0;

            this->preprocess_image(image, image_payload._processed_pixel_values, grid_pair, valid_patch_size, num_soft_tokens);

            image_payload.image_grid_h_w.push_back(grid_pair);
            image_payload.num_soft_tokens_per_image.push_back(num_soft_tokens);
            image_payload.num_images++;
        }

        // ---- audio preprocessing ----
        for (const auto& aud_str : input.audios) {
            audio_data_t audio_data = this->load_audio(aud_str, this->audio_sampling_rate, MonoDownmixMode::MEAN);
            if (audio_data.channels > 1) {
                header_print("ERROR", "only mono audio is supported");
                meta_info.error_message = "Only mono audio is supported";
                return false;
            }
          
            audio_data_list.push_back(audio_data);
        }
        if(audio_data_list.size()> 0){
            pad_audio(audio_data_list);
            this->extract_spectrogram(audio_data_list, audio_payload);
            for (unsigned int i = 0; i < audio_payload.num_audios; i++) {
                audio_payload.audio_tokens.push_back(
                    get_audio_tokens(
                        _get_feat_extract_output_length(
                            audio_payload.mel_spectrogram_frames_per_audio[i]
                        )
                    )
                );
            }
        }

        // ---- build templated text (CLI) ----
        nlohmann::ordered_json messages;
        nlohmann::ordered_json content;
        content["role"] = "user";
        content["content"] = nlohmann::ordered_json::array();
        for (size_t i = 0; i < input.images.size(); i++) {
            content["content"].push_back({{"type", "image"}, {"image", input.images[i]}});
        }
        for (size_t i = 0; i < input.audios.size(); i++) {
            content["content"].push_back({{"type", "audio"}, {"audio", input.audios[i]}});
        }
        content["content"].push_back({{"type", "text"}, {"text", input.prompt}});
        messages.push_back(content);
        templated_text = this->apply_chat_template(messages);
    }


    std::vector<int> tokens_init = this->tokenizer->encode(templated_text);

    // ---- expand soft-token placeholders ----
    int total_image_tokens = 0;
    for (unsigned int i = 0; i < image_payload.num_images; i++) {
        total_image_tokens += image_payload.num_soft_tokens_per_image[i];
    }


    std::vector<int> tokens;
    tokens.reserve(tokens_init.size() + total_image_tokens);
   auto total_audio_tokens= 0;



    int image_counter = 0;
    int audio_counter = 0;
    for (size_t i = 0; i < tokens_init.size(); i++) {
        if (tokens_init[i] == image_token_id) {
            for (unsigned int j = 0; j < image_payload.num_soft_tokens_per_image[image_counter]; j++) {
                tokens.push_back(image_token_id);
            }
            image_counter++;
        } else if (tokens_init[i] == audio_token_id) {
            // first, do a tokenizer on the new message
            // insert this into the tokens_init
            std::vector<int> new_audio_tokens = this->tokenizer->encode(audio_payload.audio_tokens[audio_counter]);
            total_audio_tokens+= new_audio_tokens.size();
            tokens.push_back(audio_start_token_id);
            for(size_t j = 0; j < new_audio_tokens.size(); j++){
                tokens.push_back(new_audio_tokens[j]);
            }
            tokens.push_back(audio_end_token_id);
            audio_counter++;
        } else {
            tokens.push_back(tokens_init[i]);
        }
    }
    this->profiler_list[TKOEN_ENCODE_TIME].stop(tokens.size());

    // header_print("FLM", "Prompt tokens: " << tokens.size()
    //     << " (images: " << image_payload.num_images << ", soft image tokens: " << total_image_tokens
    //     << "; audios: " << audio_payload.num_audios << ", soft audio tokens: " << total_audio_tokens << ")");

    // Restore KV cache from checkpoint if caller allows it
    if (meta_info.restore_allowed) {
        int restore_idx = this->engine->restore();
        this->total_tokens = restore_idx;
        this->_restore_history();
    }

    size_t n = tokens.size();
    // a continued message keeps all its tokens: they are the client's text, not the generation prompt
    tokens.resize(n - (meta_info.continue_final_message ? 0 : 4));
    this->_begin_turn(meta_info, tokens);

    // where each image and audio clip sits in the prompt that is prefilled: images are runs of
    // image_token_id as expanded above, each audio clip is bracketed by audio_start/audio_end
    std::vector<prompt_media_t> media;
    const std::vector<size_t> image_token_counts(image_payload.num_soft_tokens_per_image.begin(),
                                                 image_payload.num_soft_tokens_per_image.end());
    prompt_media_from_counts(tokens, image_token_id, image_token_counts, prompt_media_t::kImage, media);
    prompt_media_from_brackets(tokens, audio_start_token_id, audio_end_token_id, prompt_media_t::kAudio, media);

    // the tokens of an item only depend on its size: hash what the engine reads (before the drop below),
    // so a cached item is only kept when its pixels / mel frames are the same
    {
        size_t image_idx = 0, audio_idx = 0, pixel_offset = 0;
        const auto& pixels = image_payload._processed_pixel_values;
        for (prompt_media_t& m : media) {
            if (m.kind == prompt_media_t::kImage && image_idx < image_payload.image_grid_h_w.size()) {
                const auto& grid = image_payload.image_grid_h_w[image_idx++];
                const int grid_h = grid.size() > 0 ? grid[0] : 0;
                const int grid_w = grid.size() > 1 ? grid[1] : 0;
                const size_t count = static_cast<size_t>(std::max(grid_h, 0)) * static_cast<size_t>(std::max(grid_w, 0))
                    * this->patch_size * this->patch_size * this->temporal_patch_size * 3;
                uint64_t h = 0xcbf29ce484222325ull ^ (static_cast<uint64_t>(static_cast<uint32_t>(grid_h)) << 32 | static_cast<uint32_t>(grid_w));
                if (pixel_offset + count <= pixels.size()) {
                    h = prefix_hash_bytes(pixels.data() + pixel_offset, count * sizeof(pixels[0]), h);
                }
                pixel_offset += count;
                m.hash = h != 0 ? h : 1;   // 0 means "not hashed"
            } else if (m.kind == prompt_media_t::kAudio && audio_idx < audio_payload.mel_spectrograms.size()) {
                const auto& mel = audio_payload.mel_spectrograms[audio_idx];
                const int frames = audio_payload.mel_spectrogram_frames_per_audio[audio_idx];
                const int bins = audio_payload.mel_spectrogram_bins_per_audio[audio_idx];
                audio_idx++;
                uint64_t h = 0x84222325cbf29ce4ull ^ (static_cast<uint64_t>(static_cast<uint32_t>(frames)) << 32 | static_cast<uint32_t>(bins));
                h = prefix_hash_bytes(mel.data(), mel.size() * sizeof(mel[0]), h);
                m.hash = h != 0 ? h : 1;
            }
        }
    }

    // the skip and the payload trim come from the same plan, decided against the history the engine
    // holds now (after the restore); this insert does its own prefill, so the plan is not handed on
    const prompt_reuse_t plan = this->_plan_reuse(tokens, media);
    this->planned_skip_.reset();
    this->planned_media_.reset();   // committed into history_media_ below, once the prompt is in token_history

    auto drop_front = [](auto& vec, size_t count) {
        if (count == 0) return;
        if (count >= vec.size()) { vec.clear(); return; }
        vec.erase(vec.begin(), vec.begin() + count);
    };

    // drop the leading images/audios whose tokens fall within the reused prefix
    // so the payload stays aligned with the surviving tokens
    const size_t images_to_drop = std::min<size_t>(plan.skipped_media[prompt_media_t::kImage], image_payload.num_images);
    if (images_to_drop > 0) {
        // Compute how many bf16 elements to erase from the flat pixel buffer.
        // Each image contributed grid_h * grid_w * patch_size^2 * temporal_patch_size * 3 values.
        size_t pixel_elems_to_drop = 0;
        for (size_t i = 0; i < images_to_drop; i++) {
            const int grid_h = image_payload.image_grid_h_w[i][0];
            const int grid_w = image_payload.image_grid_h_w[i][1];
            pixel_elems_to_drop += static_cast<size_t>(grid_h) * grid_w
                * this->patch_size * this->patch_size
                * this->temporal_patch_size * 3;
        }
        drop_front(image_payload._processed_pixel_values, pixel_elems_to_drop);
        drop_front(image_payload.image_grid_h_w,              images_to_drop);
        drop_front(image_payload.num_soft_tokens_per_image,   images_to_drop);
        image_payload.num_images -= static_cast<unsigned>(images_to_drop);
    }

    const size_t audios_to_drop = std::min<size_t>(plan.skipped_media[prompt_media_t::kAudio], audio_payload.num_audios);
    if (audios_to_drop > 0) {
        // Each audio has its own mel_spectrograms[i] buffer (not a flat concat),
        // so dropping is a simple front-erase on each per-audio vector.
        drop_front(audio_payload.mel_spectrograms,                 audios_to_drop);
        drop_front(audio_payload.mel_spectrogram_frames_per_audio, audios_to_drop);
        drop_front(audio_payload.mel_spectrogram_bins_per_audio,   audios_to_drop);
        drop_front(audio_payload.audio_tokens,                     audios_to_drop);
        audio_payload.num_audios -= static_cast<unsigned>(audios_to_drop);
    }
    const bool has_modal = (image_payload.num_images > 0) || (audio_payload.num_audios > 0);

    // _plan_reuse cleared already when the history ends inside an item or an item changed; the other clears
    // (divergence, the whole prompt cached: no last token to sample from) are ours, as in _shared_insert
    if (plan.clear && plan.reason != prompt_reuse_t::kInsideMedia && plan.reason != prompt_reuse_t::kMediaChanged) {
        clear_context();
    }
    const size_t skip_count = plan.skip;
    tokens.erase(tokens.begin(), tokens.begin() + skip_count);

    if (this->total_tokens + tokens.size() >= this->MAX_L){
        header_print("WARNING", "Max length reached, stopping prefilling...");
        meta_info.error_message = "Prompt does not fit in the context window: " + std::to_string(tokens.size()) + " new prompt tokens on top of the " + std::to_string(this->total_tokens) + " already held exceed the context length of " + std::to_string(this->MAX_L) + " tokens";
        return false;
    }
    for (int token : tokens){
        this->token_history.push_back(token);
    }
    // as _shared_insert does with its plan: these are now the items token_history holds
    this->history_media_ = std::move(media);

    auto prefill_start = this->profiler_list[PREFILL_TIME].start();

    int max_prefill_len = meta_info.max_prefill_len;

    // Index (exclusive) of the last multimodal token in the surviving
    // token sequence. The chunker uses this to guarantee the first prefill
    // chunk is large enough to carry the entire multimodal payload.
    int first_len_run = static_cast<int>(plan.media_end);

    max_prefill_len = 1 << static_cast<int>(std::ceil(std::log2(max_prefill_len)));
    if (max_prefill_len < 512) {
        this->last_thinker_result = this->engine->prefill(tokens, has_modal ? &payload : nullptr);
    }
    else{
        if (first_len_run > 0) {
            int new_max_len = 1 << static_cast<int>(std::ceil(std::log2(first_len_run)));
            if (new_max_len > max_prefill_len) {
                max_prefill_len = new_max_len;
            }
        }
        int chunks = (tokens.size() + max_prefill_len - 1) / max_prefill_len;
        for (int i = 0; i < chunks; i++) {
            if (is_cancelled()) {
                meta_info.stop_reason = CANCEL_DETECTED;
                break;
            }
            int start = i * max_prefill_len;
            int end = std::min(static_cast<int>(tokens.size()), (i + 1) * max_prefill_len);
            std::vector<int> chunk_tokens(tokens.begin() + start, tokens.begin() + end);
            header_print("FLM", "Prefill chunk " + std::to_string(i+1) + "/" + std::to_string(chunks) + " with " + std::to_string(chunk_tokens.size()) + " tokens");
            auto chunk_thinker_result = this->engine->prefill(chunk_tokens, (i == 0) ? &payload : nullptr);
            if (i == chunks - 1) {
                this->last_thinker_result = chunk_thinker_result;
            }
        }
    }

    auto prefill_end = this->profiler_list[PREFILL_TIME].stop(tokens.size());
    meta_info.prefill_duration = (uint64_t)time_utils::duration_ns(prefill_start, prefill_end).first;
    // As in AutoModel::_shared_insert, `tokens` holds only the uncached suffix, so
    // the cached prefix is added back to report the whole prompt. The relevant count
    // is `skip_count`, which is what was erased from `tokens` above.
    meta_info.cached_prompt_tokens = static_cast<int>(skip_count);
    meta_info.prompt_tokens = static_cast<int>(skip_count + tokens.size());

    this->total_tokens += tokens.size();

    this->profiler_list[SAMPLING_TIME].start();
    this->last_token = this->_sample(this->last_thinker_result.logits, meta_info);
    meta_info.insert_sample_pending = true;
    this->profiler_list[SAMPLING_TIME].stop(1);

    this->_checkpoint_history();
    this->engine->checkpoint();
    return true;
}

std::string Qwen3_5_Omni::generate(chat_meta_info_t& meta_info, int length_limit, std::ostream& os, std::function<bool()> is_cancelled) {
    std::vector<int> sampled_tokens;
    std::string result;
    if (length_limit > 0){
        sampled_tokens.reserve(length_limit);
    }
    else{
        sampled_tokens.reserve(4096);
    }
    assert(this->last_token != -1);
    stop_reason_t reason = EOT_DETECTED;

    // The preamble below feeds the engine four tokens but only the sampled one
    // is counted, so the engine runs three rows ahead of total_tokens (one more
    // for the forward after an EOS). Keep that much room under MAX_L, or those
    // forwards write past the end of the engine's KV window and corrupt it for
    // every later request. A continued message has no preamble: no lead.
    const uint32_t engine_lead = meta_info.continue_final_message ? 0 : 4;
    if (!meta_info.continue_final_message && this->total_tokens + engine_lead >= this->MAX_L) {
        header_print("WARNING", "Max length reached, stopping generation...");
        meta_info.stop_reason = MAX_LENGTH_REACHED;
        return result;
    }

    this->profiler_list[DECODING_TIME].reset();
    this->profiler_list[TKOEN_DECODE_TIME].reset();
    std::string token_str;
    int sampled_token;
    int last_sampled_token;

    if (meta_info.continue_final_message) {
        // the prompt ends inside the client's message (insert() trimmed nothing):
        // no empty think block, the output starts from insert()'s sample
        last_sampled_token = this->last_token;
        this->token_history.push_back(last_sampled_token);
        // the sample insert() took is the reply's first token, and counts toward the limit
        meta_info.generated_tokens++;
        if (this->is_normal_token(last_sampled_token)) {
            token_str = this->tokenizer->run_time_decoder(last_sampled_token);
            result += token_str;
            os << token_str << std::flush;
        }
        if (this->is_eos(last_sampled_token)) {
            // sampled from the prefill, so the engine never ran it
            this->token_history.pop_back();
            meta_info.stop_reason = reason;
            return result;
        }
    }
    // manually decoding <think> \n <\think> \n
    else {
        this->token_history.push_back(think_start_id);
        // fed for the KV cache only, it never reaches the output
        this->_skip_insert_sample(meta_info);
        this->engine->forward(think_start_id);
        token_str = this->tokenizer->run_time_decoder(think_start_id);

        // \n\n
        this->token_history.push_back(271);
        this->profiler_list[DECODING_TIME].start();
        this->engine->forward(271);
        this->profiler_list[DECODING_TIME].stop(1);
        token_str = this->tokenizer->run_time_decoder(271);
        
        this->token_history.push_back(think_end_id);
        this->profiler_list[DECODING_TIME].start();
        this->engine->forward(think_end_id);
        this->profiler_list[DECODING_TIME].stop(1);
        token_str = this->tokenizer->run_time_decoder(think_end_id);

        this->token_history.push_back(271);
        this->profiler_list[DECODING_TIME].start();
        this->last_thinker_result = this->engine->forward(271);
        this->profiler_list[DECODING_TIME].stop(1);
        token_str = this->tokenizer->run_time_decoder(271);
        sampled_token = this->_sample(this->last_thinker_result.logits, meta_info);

        this->total_tokens++;
        // the first sample after the preamble is the reply's first token, and counts toward the limit
        meta_info.generated_tokens++;
        last_sampled_token = sampled_token;
        this->token_history.push_back(last_sampled_token);
        token_str = this->tokenizer->run_time_decoder(last_sampled_token);
        result += token_str;
        os << token_str << std::flush;
    }

    if (this->is_eos(last_sampled_token)) {
        // only the preamble path gets here with an eos: the engine sampled it, so feed it as the loop does
        this->engine->forward(last_sampled_token);
    }
    else if (this->total_tokens + engine_lead >= this->MAX_L){
        header_print("WARNING", "Max length reached, stopping generation...");
        reason = MAX_LENGTH_REACHED;
    }
    else if ((length_limit > 0) && (meta_info.generated_tokens >= length_limit)) {
        reason = MAX_LENGTH_REACHED;
    }
    else while (true) {
        // Running out of window is a truncation; testing here rather than in
        // the loop condition records it.
        if (this->total_tokens + engine_lead >= this->MAX_L) {
            header_print("WARNING", "Max length reached, stopping generation...");
            reason = MAX_LENGTH_REACHED;
            break;
        }
        if (is_cancelled()) {
            reason = CANCEL_DETECTED;
            break;
        }
        this->profiler_list[DECODING_TIME].start();
        this->last_thinker_result = this->engine->forward(last_sampled_token);
        this->profiler_list[DECODING_TIME].stop(1);

        this->profiler_list[SAMPLING_TIME].start();
        int sampled_token = this->_sample(this->last_thinker_result.logits, meta_info);
        this->profiler_list[SAMPLING_TIME].stop(1);
        this->total_tokens++;
        last_sampled_token = sampled_token;
 
        this->profiler_list[TKOEN_DECODE_TIME].start();
        if (this->is_normal_token(sampled_token)) { // filter out special tokens
            std::string token_str = this->tokenizer->run_time_decoder(sampled_token);
            os << token_str << std::flush;
            result += token_str;
        }
        this->profiler_list[TKOEN_DECODE_TIME].stop(1);
        this->token_history.push_back(sampled_token);
        if (this->is_eos(sampled_token)) {
            meta_info.generated_tokens++;
            this->engine->forward(last_sampled_token);
            break;
        }
        meta_info.generated_tokens++;
        if ((length_limit > 0) && (meta_info.generated_tokens >= length_limit)) {
            reason = MAX_LENGTH_REACHED;
            break;
        }
    }
    meta_info.decoding_duration = (uint64_t)(time_utils::cast_to_us(this->profiler_list[DECODING_TIME].get_total_time()).first) * 1e3;
    meta_info.stop_reason = reason;
    if (reason != EOT_DETECTED) {
        // Cut short after sampling its last token: no engine has run that one,
        // so the next prompt has to carry it.
        this->token_history.pop_back();
    }

    std::cout << std::endl;
    header_print("FLM", "Model RAW Output: \n" + result);
    return result;
}

std::string Qwen3_5_Omni::generate_with_prompt(chat_meta_info_t& meta_info, lm_uniform_input_t& input, int length_limit, std::ostream& os) {
    if (!this->insert(meta_info, input)) {
        return "";
    }
    header_print("FLM", "Prompt inserted, starting generation...");
    return this->generate(meta_info, length_limit, os);
}

buffer<bf16> Qwen3_5_Omni::say(std::string wav_out_path) {
    try {
        buffer<bf16> audio = this->engine->say(this->last_thinker_result);
        // TODO: once the talker/codec path lands, decode `audio` to PCM and write
        //       a WAV file at wav_out_path.
        if (!wav_out_path.empty()) {
            header_print("FLM", "Audio output produced (" << audio.size()
                << " elements); WAV writing not implemented yet.");
        }
        return audio;
    } catch (const std::exception& e) {
        header_print("FLM", "Audio output not available yet: " << e.what());
        return buffer<bf16>();
    }
}

void Qwen3_5_Omni::set_sampler(sampler_config& sampler_config) {
    if (this->sampler != nullptr) this->sampler.reset();
    this->sampler = std::make_unique<Sampler>((int)this->lm_config->get("vocab_size"), sampler_config);
}

void Qwen3_5_Omni::clear_context() {
    this->total_tokens = 0;
    this->last_token = -1;
    this->token_history.clear();
    this->checkpoint_his.clear();
    this->_forget_history_media();   // the items went with the history they sat in
    this->engine->clear_context();
    this->total_tokens = 0;
    this->sampler->reset_penalties();
    for (size_t i = 0; i < PROFILER_TYPE_NUM; i++) {
        this->profiler_list[i].reset();
    }
    this->last_prefill_time = { 0, "us" };
}

void Qwen3_5_Omni::set_max_length(unsigned int new_max) {
    this->MAX_L = std::max(new_max, this->MAX_L);
    if (this->engine != nullptr) this->engine->update_max_length(this->MAX_L);
}

int Qwen3_5_Omni::get_current_context_length() {
    return this->total_tokens;
}

std::string Qwen3_5_Omni::show_model_info() {
    try {
        return this->lm_config->_str();
    } catch (const std::exception& e) {
        return "Error showing model info: " + std::string(e.what());
    }
}

std::string Qwen3_5_Omni::show_profile() {
    std::stringstream ss;
    time_utils::time_with_unit time = this->profiler_list[TOTAL_TIME].get_total_time();
    ss << "  Statistics:" << std::endl;
    ss << "    Total tokens:        " << this->get_current_context_length() << std::endl;
    ss << "    Total time:          " << time.first << " " << time.second << std::endl;
    time = this->profiler_list[DECODING_TIME].get_total_time();
    ss << "    Decoding time:       " << time.first << " " << time.second << std::endl;
    time = this->profiler_list[PREFILL_TIME].get_total_time();
    ss << "    Prefill time:        " << time.first << " " << time.second << std::endl;
    ss << "    Average decoding speed:       " << this->profiler_list[DECODING_TIME].get_average_speed() << " tokens/s" << std::endl;
    ss << "    Average prefill  speed:       " << this->profiler_list[PREFILL_TIME].get_average_speed() << " tokens/s" << std::endl;
    return ss.str();
}

std::pair<std::string, std::vector<int>> Qwen3_5_Omni::get_history() {
    std::vector<int> history = this->token_history;
    std::string all_context = this->tokenizer->decode(history);
    return std::make_pair(all_context, history);
}
