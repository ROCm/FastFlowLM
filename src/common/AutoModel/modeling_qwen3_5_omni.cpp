/// \file modeling_qwen3_5_omni.cpp
/// \brief Qwen3_5_Omni driver: tokenizer/template/sampler + prefill/decode loop.
/// \author FastFlowLM Team

#include "AutoModel/modeling_qwen3_5_omni.hpp"

Qwen3_5_Omni::Qwen3_5_Omni(xrt::device* npu_device_inst)
    : AutoModel(npu_device_inst) {}

/// \brief Pull the thinker-scope vision config (with VL fallbacks).
static uint32_t json_u32(const nlohmann::json& j, const char* key, uint32_t dflt) {
    if (j.contains(key) && !j[key].is_null()) return j[key].get<uint32_t>();
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
}

void Qwen3_5_Omni::load_model(std::string model_path, json model_info, int default_context_length, bool enable_preemption) {
    if (this->is_model_loaded && this->model_path == model_path) {
        header_print("FLM", "Model already loaded: " << this->model_path);
        return;
    }
    this->model_path = model_path;
    header_print("FLM", "Loading model: " << this->model_path);

    this->lm_config = std::make_unique<LM_Config>();
    this->lm_config->from_pretrained(this->model_path);

    // The omni thinker vocab lives in thinker_config.text_config, so the
    // top-level "vocab_size" is 0. Resolve it here so every downstream reader
    // (set_sampler, set_topk, ...) sees the real vocab instead of 0.
    {
        const auto& jc = this->lm_config->_json_config;
        if (jc.contains("thinker_config") && jc["thinker_config"].contains("text_config")) {
            this->lm_config->vocab_size =
                json_u32(jc["thinker_config"]["text_config"], "vocab_size", this->lm_config->vocab_size);
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
        this->patch_size          = json_u32(vc, "patch_size", this->patch_size);
        this->temporal_patch_size = json_u32(vc, "temporal_patch_size", this->temporal_patch_size);
        this->merge_size          = json_u32(vc, "spatial_merge_size", this->merge_size);
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

std::string Qwen3_5_Omni::apply_chat_template(nlohmann::ordered_json& messages, nlohmann::ordered_json tools) {
    minja::chat_template_inputs inputs;
    inputs.add_generation_prompt = true;
    inputs.messages = messages;
    inputs.extra_context = this->extra_context;
    if (!tools.empty()) inputs.tools = tools;
    return this->chat_tmpl->apply(inputs);
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

    // ---- image preprocessing ----
    for (const auto& img_str : input.images) {
        qwen3_5_omni_image_t image = this->load_image(img_str);
        std::vector<int> grid_pair;
        uint32_t valid_patch_size = 0;
        uint32_t num_soft_tokens = 0;
        std::vector<bf16> pixel_values;
        this->preprocess_image(image, pixel_values, grid_pair, valid_patch_size, num_soft_tokens);
        image_payload.pixel_values.push_back(std::move(pixel_values));
        image_payload.image_grid_pairs_per_image.push_back(grid_pair);
        image_payload.valid_patch_size_per_image.push_back(valid_patch_size);
        image_payload.num_soft_tokens_per_image.push_back(num_soft_tokens);
        image_payload.image_patch__element_per_patch.push_back({image.width_resized, image.height_resized});
        image_payload.num_images++;
    }

    // ---- audio preprocessing ----
    constexpr double max_support_audio_length_seconds = 30.0;
    std::vector<audio_data_t> audio_data_list;
    for (const auto& aud_str : input.audios) {
        audio_data_t audio_data = this->load_audio(aud_str, this->audio_resample_rate, MonoDownmixMode::MEAN);
        if (audio_data.channels > 1) {
            header_print("ERROR", "only mono audio is supported");
            return false;
        }
        std::vector<audio_data_t> clipped = this->clip_audio_length(audio_data, max_support_audio_length_seconds);
        audio_data_list.insert(audio_data_list.end(), clipped.begin(), clipped.end());
        if (clipped.size() > 1) {
            header_print_g("FLM", "Audio split into " + std::to_string(clipped.size()) + " chunks for processing.");
        }
    }
    if (!audio_data_list.empty()) {
        this->extract_spectrogram(audio_data_list, audio_payload);
        for (unsigned int i = 0; i < audio_payload.num_audios; i++) {
            audio_payload.num_soft_tokens_per_audio.push_back(
                this->compute_audio_soft_tokens(audio_payload.mel_spectrogram_frames_per_audio[i]));
        }
    }

    // ---- build templated text ----
    std::string templated_text;
    if (!input.messages.empty()) {
        templated_text = this->apply_chat_template(input.messages, input.tools);
    } else {
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
    int total_audio_tokens = 0;
    for (unsigned int i = 0; i < audio_payload.num_audios; i++) {
        total_audio_tokens += audio_payload.num_soft_tokens_per_audio[i] + 2; // +boa/eoa
    }

    std::vector<int> tokens;
    tokens.reserve(tokens_init.size() + total_image_tokens + total_audio_tokens);
    int image_counter = 0;
    int audio_counter = 0;
    for (size_t i = 0; i < tokens_init.size(); i++) {
        if (tokens_init[i] == image_token_id) {
            for (unsigned int j = 0; j < image_payload.num_soft_tokens_per_image[image_counter]; j++) {
                tokens.push_back(image_token_id);
            }
            image_counter++;
        } else if (tokens_init[i] == audio_token_id) {
            tokens.push_back(audio_start_token_id);
            for (unsigned int j = 0; j < audio_payload.num_soft_tokens_per_audio[audio_counter]; j++) {
                tokens.push_back(audio_token_id);
            }
            tokens.push_back(audio_end_token_id);
            audio_counter++;
        } else {
            tokens.push_back(tokens_init[i]);
        }
    }
    this->profiler_list[TKOEN_ENCODE_TIME].stop(tokens.size());

    header_print("FLM", "Prompt tokens: " << tokens.size()
        << " (images: " << image_payload.num_images << ", soft image tokens: " << total_image_tokens
        << "; audios: " << audio_payload.num_audios << ", soft audio tokens: " << total_audio_tokens << ")");

    if (this->total_tokens + tokens.size() >= this->MAX_L) {
        header_print("WARNING", "Max length reached, stopping prefilling...");
        return false;
    }
    for (int t : tokens) this->token_history.push_back(t);

    const bool has_modal = (image_payload.num_images > 0) || (audio_payload.num_audios > 0);

    auto prefill_start = this->profiler_list[PREFILL_TIME].start();
    this->last_thinker_result = this->engine->prefill(tokens, has_modal ? &payload : nullptr);
    auto prefill_end = this->profiler_list[PREFILL_TIME].stop(tokens.size());
    meta_info.prefill_duration = (uint64_t)time_utils::duration_ns(prefill_start, prefill_end).first;
    meta_info.prompt_tokens = tokens.size();

    this->total_tokens += tokens.size();

    this->profiler_list[SAMPLING_TIME].start();
    this->last_token = this->sampler->sample(this->last_thinker_result.logits);
    this->profiler_list[SAMPLING_TIME].stop(1);
    return true;
}

std::string Qwen3_5_Omni::generate(chat_meta_info_t& meta_info, int length_limit, std::ostream& os, std::function<bool()> is_cancelled) {
    std::string result;
    assert(this->last_token != -1);
    stop_reason_t reason = EOT_DETECTED;

    int last_sampled_token = this->last_token;
    this->token_history.push_back(this->last_token);
    if (this->is_normal_token(last_sampled_token)) {
        std::string token_str = this->tokenizer->run_time_decoder(last_sampled_token);
        result += token_str;
        os << token_str << std::flush;
    }
    if (this->is_eos(last_sampled_token)) {
        return result;
    }

    this->profiler_list[DECODING_TIME].reset();
    while (this->total_tokens < this->MAX_L) {
        if (is_cancelled()) {
            reason = CANCEL_DETECTED;
            break;
        }
        this->profiler_list[DECODING_TIME].start();
        this->last_thinker_result = this->engine->forward(last_sampled_token);
        this->profiler_list[DECODING_TIME].stop(1);

        this->profiler_list[SAMPLING_TIME].start();
        int sampled_token = this->sampler->sample(this->last_thinker_result.logits);
        this->profiler_list[SAMPLING_TIME].stop(1);
        this->total_tokens++;
        last_sampled_token = sampled_token;

        if (this->is_normal_token(sampled_token)) {
            std::string token_str = this->tokenizer->run_time_decoder(sampled_token);
            os << token_str << std::flush;
            result += token_str;
        }
        this->token_history.push_back(sampled_token);
        if (this->is_eos(sampled_token)) {
            meta_info.generated_tokens++;
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
    this->sampler = std::make_unique<Sampler>((int)this->lm_config->vocab_size, sampler_config);
}

void Qwen3_5_Omni::clear_context() {
    this->total_tokens = 0;
    this->last_token = -1;
    this->token_history.clear();
    this->engine->clear_context();
    if (this->sampler != nullptr) this->sampler->reset_penalties();
    for (size_t i = 0; i < PROFILER_TYPE_NUM; i++) {
        this->profiler_list[i].reset();
    }
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
