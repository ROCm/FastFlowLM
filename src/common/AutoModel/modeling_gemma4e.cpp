/// \file modeling_gemma4e.cpp
/// \brief Gemma4e class
/// \author FastFlowLM Team
/// \date 2025-09-01
/// \version 0.9.24
/// \note This is a source file for the Gemma4e class


#include "AutoModel/modeling_gemma4e.hpp"
#include "metrices.hpp"

#include <cctype>

namespace {
std::string trim(const std::string& text) {
    const auto first = std::find_if(text.begin(), text.end(), [](unsigned char c) { return !std::isspace(c); });
    const auto last = std::find_if(text.rbegin(), text.rend(), [](unsigned char c) { return !std::isspace(c); }).base();
    return first < last ? std::string(first, last) : std::string();
}
} // namespace

/************              Gemma4e family            **************/
Gemma4e::Gemma4e(flm_rt::device* npu_device_inst) : AutoModel(npu_device_inst, "Gemma4e") {}

gemma4e_engine_config_t Gemma4e::engine_config() const {
    return read_gemma4e_engine_config<gemma4e_npu>(this->lm_engine);
}

gemma4e_engine_config_t Gemma4e_Flash::engine_config() const {
    return read_gemma4e_engine_config<gemma4e_flash>(this->lm_engine);
}

const flm_state_api* Gemma4e::engine_state_api() const {
    return gemma4e_npu_state_api(this->lm_engine);
}

int Gemma4e_Flash::_pin_system_prefix(const std::string& system_text) {
    this->lm_engine->clear_context();
    this->token_history.clear();
    this->checkpoint_his.clear();
    this->_forget_history_media();
    this->total_tokens = 0;
    this->last_token = -1;
    this->system_tokens = 0;
    this->system_his.clear();
    this->pinned_system_text = system_text;

    if (system_text.empty()) {
        return 0;
    }

    // Find the longest token prefix shared by all prompts that carry this
    // system turn. Template two otherwise-identical prompts that differ only
    // in the first character of the user message and take the common prefix.
    auto templated = [&](const std::string& user_text) {
        nlohmann::ordered_json messages = nlohmann::ordered_json::array();
        messages.push_back({ {"role", "system"}, {"content", system_text} });
        messages.push_back({ {"role", "user"}, {"content", user_text} });
        return this->apply_chat_template(messages);
    };
    std::vector<int> probe_a = this->tokenizer->encode(templated("A"));
    std::vector<int> probe_b = this->tokenizer->encode(templated("\xe4\xbd\xa0")); // U+4F60, CJK probe

    size_t shared = 0;
    while (shared < probe_a.size() && shared < probe_b.size() && probe_a[shared] == probe_b[shared]) {
        shared++;
    }
    if (shared == 0) {
        header_print("WARNING", "Gemma4e_Flash: could not isolate a reusable system prefix, every turn will prefill in full");
        return 0;
    }

    std::vector<int> tokens(probe_a.begin(), probe_a.begin() + shared);
    chat_meta_info_t pin_meta;
    pin_meta.restore_allowed = false;
    if (!this->_shared_insert(pin_meta, tokens, [] { return false; }, nullptr)) {
        this->lm_engine->clear_context();
        this->token_history.clear();
        this->checkpoint_his.clear();
        this->_forget_history_media();
        return 0;
    }

    this->system_his = this->token_history;
    // the probes are text only, read in without a payload: the pin holds no image or audio
    this->history_media_ = std::vector<prompt_media_t>{};
    this->_checkpoint_history();
    this->lm_engine->checkpoint();
    this->system_tokens = static_cast<int>(shared);

    for (size_t i = 0; i < PROFILER_TYPE_NUM; i++) {
        this->profiler_list[i].reset();
    }

    header_print("FLM", "Gemma4e_Flash: system prefix pinned (" + std::to_string(this->system_tokens) + " tokens).");
    return this->system_tokens;
}

void Gemma4e_Flash::_forget_pin() {
    this->pinned_system_text.clear();
    this->system_his.clear();
    this->system_tokens = 0;
}

void Gemma4e_Flash::clear_context() {
    Gemma4e::clear_context();
    this->_forget_pin();
}

void Gemma4e_Flash::_reset_turn() {
    if (this->system_tokens > 0) {
        this->total_tokens = static_cast<uint32_t>(this->lm_engine->restore());
        this->token_history = this->system_his;
        this->checkpoint_his = this->system_his;
        // the pin is a text-only system prefix (_pin_system_prefix), so it holds no image or audio
        this->history_media_ = std::vector<prompt_media_t>{};
        this->checkpoint_media_ = std::vector<prompt_media_t>{};
    } else {
        this->lm_engine->clear_context();
        this->token_history.clear();
        this->checkpoint_his.clear();
        this->_forget_history_media();
        this->total_tokens = 0;
    }
    this->sampler->reset_penalties();
    this->last_token = -1;
    this->profiler_list[PREFILL_TIME].reset();
    this->profiler_list[DECODING_TIME].reset();
    this->profiler_list[SAMPLING_TIME].reset();
    this->profiler_list[TKOEN_ENCODE_TIME].reset();
    this->profiler_list[TKOEN_DECODE_TIME].reset();
}

bool Gemma4e_Flash::insert(chat_meta_info_t& meta_info, lm_uniform_input_t& input, std::function<bool()> is_cancelled) {
    // Extract system text from the first message (if present).
    std::string sys_text;
    if (!input.messages.empty() && input.messages[0].value("role", "") == "system") {
        const auto& content = input.messages[0]["content"];
        if (content.is_string()) {
            sys_text = content.get<std::string>();
        } else if (content.is_array()) {
            for (const auto& item : content) {
                if (item.value("type", "") == "text") {
                    sys_text += item.value("text", "");
                }
            }
        }
    }

    if (sys_text != this->pinned_system_text) {
        this->_pin_system_prefix(sys_text);
    }

    this->_reset_turn();

    // Disable the prompt-cache restore path in the parent: _reset_turn() has
    // already positioned the engine at the correct starting point (pin or
    // clear). The parent's token_history now holds the pinned prefix, so
    // _shared_insert's own prefix-match will skip those tokens automatically.
    meta_info.restore_allowed = false;

    return Gemma4e::insert(meta_info, input, is_cancelled);
}

std::string Gemma4e_Flash::generate(chat_meta_info_t& meta_info, int length_limit, std::ostream& os, std::function<bool()> is_cancelled) {
    std::string result;
    assert(this->last_token != -1);

    stop_reason_t reason = EOT_DETECTED;
    int last_sampled_token = this->last_token;

    this->token_history.push_back(last_token);
    // the sample insert() took is the reply's first token, and counts toward the limit
    meta_info.generated_tokens++;

    if (this->is_normal_token(last_sampled_token) && last_sampled_token != -1) {
        std::string token_str = this->tokenizer->run_time_decoder(last_sampled_token);
        result += token_str;
        os << token_str << std::flush;
    }
    if (this->is_eos(last_sampled_token)) {
        // Sampled from the prefill, so no engine has run it.
        this->token_history.pop_back();
        meta_info.stop_reason = reason;
        return result;
    }
    this->profiler_list[DECODING_TIME].reset();
    this->profiler_list[TKOEN_DECODE_TIME].reset();
    if (this->total_tokens >= this->MAX_L) {
        header_print("WARNING", "Max length reached, stopping generation...");
        reason = MAX_LENGTH_REACHED;
    }
    else if ((length_limit > 0) && (meta_info.generated_tokens >= length_limit)) {
        reason = MAX_LENGTH_REACHED;
    }
    else while (true) {
        // Running out of window is a truncation: test it here so it is recorded.
        if (this->total_tokens >= this->MAX_L) {
            reason = MAX_LENGTH_REACHED;
            break;
        }
        if (is_cancelled()) {
            reason = CANCEL_DETECTED;
            break;
        }

        this->profiler_list[DECODING_TIME].start();
        buffer<bf16> y = this->lm_engine->forward(last_sampled_token);
        this->profiler_list[DECODING_TIME].stop(1);

        this->profiler_list[SAMPLING_TIME].start();
        int sampled_token = this->_sample(y, meta_info);
        this->profiler_list[SAMPLING_TIME].stop(1);
        this->total_tokens++;
        last_sampled_token = sampled_token;

        this->profiler_list[TKOEN_DECODE_TIME].start();
        if (this->is_normal_token(sampled_token)) {
            std::string token_str = this->tokenizer->run_time_decoder(sampled_token);
            os << token_str << std::flush;
            result += token_str;
        }
        this->profiler_list[TKOEN_DECODE_TIME].stop(1);

        this->token_history.push_back(sampled_token);
        meta_info.generated_tokens++;

        if (this->is_eos(sampled_token)) {
            // Single-turn: no forward_on_eos needed — kv cache is discarded
            // by _reset_turn() at the start of every new request.
            // The engine never ran it, so the history drops it.
            this->token_history.pop_back();
            break;
        }
        if ((length_limit > 0) && (meta_info.generated_tokens >= length_limit)) {
            reason = MAX_LENGTH_REACHED;
            break;
        }
    }

    meta_info.decoding_duration = (uint64_t)(time_utils::cast_to_us(this->profiler_list[DECODING_TIME].get_total_time()).first) * 1e3;
    meta_info.stop_reason = reason;
    if (reason != EOT_DETECTED) {
        // Cut short after sampling its last token: no engine has run that one.
        this->token_history.pop_back();
    }
    if (this->total_tokens >= this->MAX_L) {
        header_print("WARNING", "Max length reached, stopping generation...");
    }
    std::cout << std::endl;
    header_print("FLM", "Model RAW Output: \n" + result);

    // Single-turn: do not checkpoint here — the only checkpoint that matters
    // is the pinned system prefix, which _pin_system_prefix() already owns.

    return result;
}

std::string Gemma4e_Flash::generate_with_prompt(chat_meta_info_t& meta_info, lm_uniform_input_t& input, int length_limit, std::ostream& os) {
    if (!this->insert(meta_info, input)) {
        return "";
    }
    return this->generate(meta_info, length_limit, os);
}

void Gemma4e::load_model(std::string model_path, json model_info, int default_context_length, bool enable_preemption, const std::string& backend) {
    
    this->_shared_load_backend(model_path, model_info, default_context_length, enable_preemption, backend);
    this->setup_tokenizer(model_path);
    this->sampler.reset();

    this->enable_tool = (model_info["size"] > 800000000)? true : false;

    sampler_config config;
    config.top_k = 64;
    config.top_p = 0.95;
    config.min_p = 0.0;
    config.temperature = 1.0;
    config.rep_penalty = 1.0;
    config.freq_penalty = 1.0;
    config.pre_penalty = 1.0f;

    this->set_sampler(config);
    for (size_t i = 0; i < PROFILER_TYPE_NUM; i++) {
        this->profiler_list[i].reset();
    }
    // Gemma4e_Flash shares this function but runs a different engine: the table is nullptr for it, the cache stays off.
    // Text-only prompts only (see insert), so no image placeholder id is given.
    _prefix_attach(lm_engine, this->engine_state_api());
}

void Gemma4e::setup_tokenizer(std::string model_path) {
    auto tokenizer_config = this->_shared_setup_tokenizer(model_path);
}

std::string Gemma4e::apply_chat_template(nlohmann::ordered_json& messages, nlohmann::ordered_json tools, bool continue_final_message) {
    // Same as Gemma4_12B: the Gemma4 template emits unquoted keys (argument_needle:<|"|>...<|"|>), which
    // minja's capability probe does not recognize, so it would wrongly polyfill tool calls and tool
    // responses into a generic JSON blob. The template handles them natively.
    minja::chat_template_options opt;
    opt.polyfill_tools = false;
    opt.polyfill_tool_calls = false;
    opt.polyfill_tool_responses = false;
    opt.polyfill_object_arguments = false;

    // The template raises if tool_calls[].function.arguments is a string, but OpenAI-style clients send it
    // serialized. Deserialize it here (what polyfill_object_arguments would do).
    nlohmann::ordered_json normalized = messages;
    for (auto& message : normalized) {
        if (!message.is_object() || !message.contains("tool_calls")) continue;
        for (auto& tool_call : message["tool_calls"]) {
            if (!tool_call.is_object() || !tool_call.contains("function")) continue;
            auto& function = tool_call["function"];
            if (!function.is_object() || !function.contains("arguments")) continue;
            auto& arguments = function["arguments"];
            if (!arguments.is_string()) continue;
            auto raw = arguments.get<std::string>();
            if (raw.empty()) {
                arguments = nlohmann::ordered_json::object();
                continue;
            }
            auto parsed = nlohmann::ordered_json::parse(raw, nullptr, /* allow_exceptions= */ false);
            if (parsed.is_object()) arguments = parsed;
        }
    }

    minja::chat_template_inputs inputs;
    inputs.messages = normalized;
    inputs.extra_context = this->extra_context;
    inputs.extra_context["enable_thinking"] = this->enable_think;
    inputs.extra_context["conversational_mode"] = this->conversational_mode;
    if (!tools.empty())
        inputs.tools = tools;
    return this->_render_chat_template(inputs, continue_final_message, opt);
}

bool Gemma4e::_continue_verbatim(std::vector<int>& tokens, std::optional<int> requested_max_new_tokens) {
    const std::vector<int>& ran = this->token_history;
    if (ran.empty() || static_cast<size_t>(this->lm_engine->get_current_context_length()) != ran.size()) {
        return false;
    }
    const std::vector<int> header = this->tokenizer->encode("<|turn>model\n");
    const std::vector<int> turn_end = this->tokenizer->encode("<turn|>");
    if (header.empty() || turn_end.size() != 1) {
        return false;
    }
    // the reply the engine generated follows the last model header it ran
    const auto at = std::find_end(ran.begin(), ran.end(), header.begin(), header.end());
    if (at == ran.end()) {
        return false;
    }
    const size_t start = static_cast<size_t>(at - ran.begin()) + header.size();
    if (tokens.size() <= start || !std::equal(ran.begin(), ran.begin() + start, tokens.begin())) {
        return false;
    }
    // the prompt's copy of that reply ends at the first <turn|> after its header
    const auto reply_end = std::find(tokens.begin() + start, tokens.end(), turn_end[0]);
    if (reply_end == tokens.end()) {
        return false;
    }
    // an eos that ended the reply was run; a reply cut at the length limit lacks its last token
    const bool ended = ran.back() == turn_end[0];
    // A reply's content follows its thought channel; a reply cut inside that channel has none.
    const auto content = [&](std::vector<int>::const_iterator begin, std::vector<int>::const_iterator end) {
        if (begin != end && *begin == think_start_id) {
            const auto close = std::find(begin, end, think_end_id);
            if (close == end) return std::string();
            begin = close + 1;
        }
        return trim(this->tokenizer->decode(std::vector<int>(begin, end)));
    };
    const std::string generated = content(ran.begin() + start, ended ? ran.end() - 1 : ran.end());
    const std::string sent = content(tokens.begin() + start, reply_end);
    std::vector<int> continued(ran);
    if (generated != sent) {
        // the client did get the token the engine did not run
        if (ended || generated.empty() || sent.size() <= generated.size() || sent.compare(0, generated.size(), generated) != 0) {
            return false;
        }
        const std::vector<int> rest = this->tokenizer->encode(sent.substr(generated.size()));
        continued.insert(continued.end(), rest.begin(), rest.end());
    }
    continued.insert(continued.end(), ended ? reply_end + 1 : reply_end, tokens.end());
    // the history keeps the thought channel the rendered prompt leaves out, so it can be the longer one
    try {
        this->_shared_validate_capacity(continued.size(), requested_max_new_tokens);
    } catch (const ModelRequestError&) {
        return false;
    }
    header_print("FLM", "Continuing the " << ran.size() << " tokens the engine ran, last reply included");
    tokens = std::move(continued);
    return true;
}

bool Gemma4e::insert(chat_meta_info_t& meta_info, lm_uniform_input_t& input, std::function<bool()> is_cancelled) {
    this->profiler_list[TKOEN_ENCODE_TIME].start();
    std::string templated_text;
    
    if (input.messages.empty() && input.prompt.empty()) {
        header_print("WARNING", "No messages or prompt provided");
        return false;
    }
    // On OGA these models run text only. A ModelRequestError, not the engine_config()
    // error: the handlers turn only the former into an HTTP 400.
    if (this->backend_id() == flm::backend::kOgaHipepBackendId) {
        bool images = !input.images.empty();
        bool audios = !input.audios.empty();
        for (const auto& message : input.messages) {
            images = images || (message.contains("images") && !message["images"].empty());
            audios = audios || (message.contains("audios") && !message["audios"].empty());
        }
        if (images || audios) {
            throw ModelRequestError(400, false,
                "the loaded backend '" + this->backend_id() + "' does not accept " +
                    (images ? "image" : "audio") + " input",
                "invalid_request_error");
        }
    }

    constexpr bool DEBUG_IMAGE_PREPROCESS = false;
    gemma4e_image_payload_t image_payload;
    gemma4e_audio_payload_t audio_payload;
    audio_payload.num_audios = 0;
    image_payload.num_images = 0;
    
    float max_support_audio_length_seconds = 30.0f;
    int total_audio_clips = 0;
    std::vector<audio_data_t> audio_data_list;

    nlohmann::ordered_json gemma4_message = nlohmann::ordered_json::array();
    if (!input.messages.empty()) { // Server Processing
        // Build gemma4_message (used for the template) in the same pass that
        // decodes images/audio, so an invalid image (e.g. bad base64) never
        // gets an image-token placeholder; otherwise image_payload would
        // fall out of sync with the placeholders and the prompt would still
        // try to prefill a nonexistent image.
        int total_images = 0;
        for (auto& message : input.messages) {
            if (!message.contains("images") && !message.contains("audios")) {
                gemma4_message.push_back(message);
                continue;
            }

            nlohmann::ordered_json newContent = nlohmann::ordered_json::array();
            // Process Images
            if (message.contains("images")) {
                for (auto& img : message["images"]) {
                    std::string img_str = img.get<std::string>();

                    gemma4e_image_t image = this->load_image_base64(img_str);
                    if (image.width <= 0 || image.height <= 0) {
                        header_print("ERROR", "Skipping invalid base64 image; prefilling language only for this item");
                        continue;
                    }

                    std::vector<bf16> pixel_values;
                    std::pair<int, int> patch_element_per_patch;
                    uint32_t valid_patch_size = 0;
                    uint32_t num_soft_tokens = 0;
                    std::vector<int> image_grid_pairs;

                    preprocess_image(image, patch_element_per_patch, valid_patch_size, pixel_values, image_grid_pairs, num_soft_tokens);

                    image_payload.image_patch__element_per_patch.push_back(patch_element_per_patch);
                    image_payload.valid_patch_size_per_image.push_back(valid_patch_size);
                    image_payload.pixel_values.push_back(pixel_values);
                    image_payload.image_grid_pairs_per_image.push_back(image_grid_pairs);
                    image_payload.num_soft_tokens_per_image.push_back(num_soft_tokens);
                    image_payload.num_images++;
                    total_images++;

                    newContent.push_back({{"type", "image"}, {"image", img}});
                }
            }
            // Process Audios
            if (message.contains("audios")) {
                const gemma4e_engine_config_t gemma4e_engine = this->engine_config();
                for (auto& aud : message["audios"]) {
                    std::string audio_str = aud.get<std::string>();
                    audio_data_t audio_data = this->load_audio_base64(audio_str, gemma4e_engine.Gemma4E_Audio_resample_rate, MonoDownmixMode::MEAN);
                    if (audio_data.channels > 1) {
                        std::cerr << "only mono audio is supported." << std::endl;
                        exit(-1);
                    }
                    std::vector<audio_data_t> clipped_audio_data = this->clip_audio_length(audio_data, max_support_audio_length_seconds);
                    if (this->max_audio_chunks > 0 && (int)clipped_audio_data.size() > this->max_audio_chunks) {
                        header_print_g("FLM", "Audio in message is " + std::to_string((int)clipped_audio_data.size())
                            + " chunks long; cutting off after the first " + std::to_string(this->max_audio_chunks) + ".");
                        clipped_audio_data.resize(this->max_audio_chunks);
                    }
                    audio_data_list.insert(audio_data_list.end(), clipped_audio_data.begin(), clipped_audio_data.end());
                    total_audio_clips += clipped_audio_data.size();
                    if (clipped_audio_data.size() > 1) {
                        header_print_g("FLM", "Audio in message is split into " + std::to_string(clipped_audio_data.size()) + " chunks for processing.");
                        std::cout << std::endl;
                    }
                    newContent.push_back({{"type", "audio"}, {"audio", aud}});
                }
            }
            newContent.push_back({{"type", "text"}, {"text", message.value("content", "")}});

            nlohmann::ordered_json newItem = {
                {"role", message.value("role", "user")},
                {"content", newContent}
            };
            gemma4_message.push_back(newItem);
        }
        header_print("FLM", "Total images: " << total_images);
    }
    else { // CLI Processing
        if (input.audios.size() > 0) {
            const gemma4e_engine_config_t gemma4e_engine = this->engine_config();
            for (int i = 0; i < input.audios.size(); i++) {
                std::string audio_str = input.audios[i];
                audio_data_t audio_data = this->load_audio(audio_str, gemma4e_engine.Gemma4E_Audio_resample_rate, MonoDownmixMode::MEAN); 
                
                if (audio_data.channels > 1) {
                    std::cerr << "only mono audio is supported, but got " << audio_data.original_channels << " channels. Please convert it to mono first." << std::endl;
                    exit(-1);
                }

                // apply clipping
                std::vector<audio_data_t> clipped_audio_data = this->clip_audio_length(audio_data, max_support_audio_length_seconds);
                if (this->max_audio_chunks > 0 && (int)clipped_audio_data.size() > this->max_audio_chunks) {
                    header_print_g("FLM", "Audio[" + std::to_string(i) + "] is " + std::to_string((int)clipped_audio_data.size())
                        + " chunks long; cutting off after the first " + std::to_string(this->max_audio_chunks) + ".");
                    clipped_audio_data.resize(this->max_audio_chunks);
                }
                audio_data_list.insert(audio_data_list.end(), clipped_audio_data.begin(), clipped_audio_data.end());
                total_audio_clips += clipped_audio_data.size();
                
                if (clipped_audio_data.size() > 1) {
                    header_print_g("FLM", "Audio[" + std::to_string(i) + "] is split into " + std::to_string(clipped_audio_data.size()) + " chunks for processing.");
                    std::cout << std::endl;
                }
            }
        }

        if (input.images.size() > 0) {
            for (const auto& img_str : input.images) {
                gemma4e_image_t image = this->load_image(img_str);
                std::vector<bf16> pixel_values;
                std::pair<int, int> patch_element_per_patch;
                uint32_t valid_patch_size = 0;
                uint32_t num_soft_tokens = 0;
                std::vector<int> image_grid_pairs; 

                preprocess_image(image, patch_element_per_patch, valid_patch_size, pixel_values, image_grid_pairs, num_soft_tokens);

                image_payload.image_patch__element_per_patch.push_back(patch_element_per_patch);
                image_payload.valid_patch_size_per_image.push_back(valid_patch_size);
                image_payload.pixel_values.push_back(pixel_values);
                image_payload.image_grid_pairs_per_image.push_back(image_grid_pairs);
                image_payload.num_soft_tokens_per_image.push_back(num_soft_tokens);
                image_payload.num_images++;
            } 
        }
    }

    if (!audio_data_list.empty()) {
        this->extract_spectrogram(audio_data_list, audio_payload);

        const gemma4e_engine_config_t gemma4e_engine = this->engine_config();
        const unsigned int conv2d_kernel = gemma4e_engine.Gemma4E_Audio_conv2d_kernel_size;
        const unsigned int conv2d_stride = gemma4e_engine.Gemma4E_Audio_conv2d_Stride;
        const unsigned int conv2d_padding = gemma4e_engine.Gemma4e_Audio_conv2d_Padding;
        const unsigned int max_audio_seq_length = max_support_audio_length_seconds * gemma4e_engine.Gemma4E_Audio_resample_rate;    
        
        constexpr float frame_length_ms = 20.0f;
        constexpr float hop_length_ms   = 10.0f;

        for (int i = 0; i < audio_payload.num_audios; i++) {
            const int num_samples = static_cast<int>(audio_data_list[i].num_samples);
            const int sampling_rate = audio_data_list[i].sample_rate;

            const int frame_length = static_cast<int>(std::round(sampling_rate * frame_length_ms / 1000.0f));
            const int hop_length   = static_cast<int>(std::round(sampling_rate * hop_length_ms / 1000.0f));
            const int frame_size_for_unfold = frame_length + 1;

            const int pad_left = frame_length / 2;
            const int padded_samples = num_samples + pad_left;
            int num_mel_frames = (padded_samples - frame_size_for_unfold) / hop_length + 1;

            unsigned int num_tokens = 0;
            if (num_mel_frames > 0) {
                int t = num_mel_frames;
                for (int layer = 0; layer < 2; layer++) {
                    int t_padded = t + 2 * static_cast<int>(conv2d_padding);
                    t = (t_padded - static_cast<int>(conv2d_kernel)) / static_cast<int>(conv2d_stride) + 1;
                }
                assert(t < max_audio_seq_length);
                num_tokens = t;
            }
            audio_payload.num_soft_tokens_per_audio.push_back(num_tokens);
        }
    }
    if (!input.messages.empty()) { // already a formated messages, usually from REST API
        // gemma4_message was already built above, in the same pass that
        // decoded images/audio.
        templated_text = this->apply_chat_template(gemma4_message, input.tools, meta_info.continue_final_message);
    }
    else if (!input.prompt.empty()) { // a pure text, usually from the cli
        nlohmann::ordered_json messages;
        nlohmann::ordered_json content;
        content["role"] = "user";
        content["content"] = nlohmann::ordered_json::array();
        
        for (int i = 0; i < input.images.size(); i++) {
            content["content"].push_back({{"type", "image"}, {"image", input.images[i]}});
        }
        for (int i = 0; i < total_audio_clips; i++) {
            content["content"].push_back({{"type", "audio"}, {"audio", input.audios[0]}}); // placeholder
        }
        
        content["content"].push_back({{"type", "text"}, {"text", input.prompt}});
        messages.push_back(content);
        templated_text = this->apply_chat_template(messages);
    }

    std::vector<int> tokens_init = this->tokenizer->encode(templated_text);

    // update the tokens to include the image tokens
    std::vector<int> tokens;
    
    int total_image_tokens = 0;
    for (int i = 0; i < image_payload.num_images; i++) {
        total_image_tokens += image_payload.num_soft_tokens_per_image[i];
    }
    
    int total_audio_tokens = 0;
    for (int i = 0; i < audio_payload.num_audios; i++) {
        total_audio_tokens += audio_payload.num_soft_tokens_per_audio[i];
    }

    tokens.reserve(tokens_init.size() + total_image_tokens + total_audio_tokens);
    
    int image_counter = 0;
    int audio_counter = 0;
    // soft tokens of each placeholder as expanded, in prompt order: the prompt-cache plan locates the items by them
    std::vector<size_t> image_token_counts;
    std::vector<size_t> audio_token_counts;
   
    for (int i = 0; i < tokens_init.size(); i++) {
        if (tokens_init[i] == image_token_id) {
            tokens.push_back(boi_token_id); // the first image soft token id, which is reserved for the model to identify the image position, the rest of the soft tokens for this image will be continuous following this id
            for (int j = 0; j < image_payload.num_soft_tokens_per_image[image_counter]; j++) {
                tokens.push_back(image_token_id);
            }
            tokens.push_back(eoi_token_id); // a separator token between images, not necessary but can help the model to better distinguish different images
            image_token_counts.push_back(image_payload.num_soft_tokens_per_image[image_counter]);
            image_counter++;
        } 
        else if (tokens_init[i] == audio_token_id){
            tokens.push_back(boa_token_id); // the first audio soft token id, which is reserved for the model to identify the audio position, the rest of the soft tokens for this audio will be continuous following this id
            for (int j = 0; j < audio_payload.num_soft_tokens_per_audio[audio_counter]; j++) {
                tokens.push_back(audio_token_id);
            }
            tokens.push_back(eoa_token_id); // a separator token between audios, not necessary but can help the model to better distinguish different audios
            audio_token_counts.push_back(audio_payload.num_soft_tokens_per_audio[audio_counter]);
            audio_counter++;
        }
        else {
            tokens.push_back(tokens_init[i]);
        }
    }
    assert(image_counter == image_payload.num_images);
    assert(audio_counter == audio_payload.num_audios);
      
    this->profiler_list[TKOEN_ENCODE_TIME].stop(tokens.size());

    // Prefix slots, text-only prompts. A prompt that carries pictures or audio, or a text turn whose history still holds their
    // soft tokens (the payload is gone by then), takes the in-place path below. The generation prompt ("<|turn>model\n") is how
    // the next turn renders this reply's header: the whole prompt is saved (trim 0).
    const bool modal_tokens = std::any_of(tokens.begin(), tokens.end(), [this](int t) {
        return t == image_token_id || t == audio_token_id || t == boi_token_id || t == boa_token_id;
    });
    if (image_payload.num_images == 0 && audio_payload.num_audios == 0 && !modal_tokens
        && this->_prefix_slot_mode(meta_info, tokens.size(), 0)) {
        checkpoint_his.clear();
        return this->_prefix_text_insert(meta_info, tokens, 0, "<|turn>user", is_cancelled);
    }
    this->_prefix_legacy_path();

    // OGA cannot reuse a prompt that leaves the history it ran, so it runs on from that history instead
    const bool continued = meta_info.restore_allowed && !modal_tokens &&
                           this->backend_id() == flm::backend::kOgaHipepBackendId &&
                           this->_continue_verbatim(tokens, input.requested_max_new_tokens);

    int restore_idx = -1;
    // The legacy checkpoint keeps only the rings: it relies on the global rows still being the ones it was taken with. A slot request
    // overwrites them, so it drops the checkpoint's token history, and the in-place path below then has nothing to restore.
    if (meta_info.restore_allowed && !checkpoint_his.empty() && !continued) {
        restore_idx = this->lm_engine->restore();
        this->total_tokens = restore_idx;
        this->_restore_history(); // restore the token history to be consistent with the restored KV cache, which is crucial for correct functioning of _shared_insert's prefix-matching logic
    }

    // The payload carries every image and audio of the prompt, earlier turns included. _shared_insert skips what the engine
    // already holds, so the items inside that skip leave the payload; _plan_reuse makes the same skip decision against the
    // same token_history. Audio is located per placeholder, as it was expanded.
    std::vector<prompt_media_t> media;
    prompt_media_from_counts(tokens, image_token_id, image_token_counts, prompt_media_t::kImage, media);
    prompt_media_from_counts(tokens, audio_token_id, audio_token_counts, prompt_media_t::kAudio, media);
    // Content hash of each item over exactly what the engine reads for it, taken before the drop: the soft-token count
    // only follows the aspect ratio (images) or the sample count (audio), so a same-sized other picture or clip would
    // otherwise match the history and be reused.
    size_t next_image = 0;
    size_t next_audio = 0;
    for (prompt_media_t& m : media) {
        uint64_t h = 0;
        if (m.kind == prompt_media_t::kImage && next_image < image_payload.num_images) {
            const size_t i = next_image++;
            const auto& dims = image_payload.image_patch__element_per_patch[i];
            const uint64_t shape[4] = { static_cast<uint64_t>(static_cast<uint32_t>(dims.first)), static_cast<uint64_t>(static_cast<uint32_t>(dims.second)),
                                        image_payload.valid_patch_size_per_image[i], image_payload.num_soft_tokens_per_image[i] };
            const auto& pixels = image_payload.pixel_values[i];
            const auto& grid = image_payload.image_grid_pairs_per_image[i];
            h = prefix_hash_bytes(shape, sizeof(shape), 0xcbf29ce484222325ull);
            h = prefix_hash_bytes(pixels.data(), pixels.size() * sizeof(pixels[0]), h);
            h = prefix_hash_bytes(grid.data(), grid.size() * sizeof(grid[0]), h);
        } else if (m.kind == prompt_media_t::kAudio && next_audio < audio_payload.num_audios) {
            const size_t i = next_audio++;
            const uint64_t shape[3] = { static_cast<uint64_t>(static_cast<uint32_t>(audio_payload.mel_spectrogram_frames_per_audio[i])),
                                        static_cast<uint64_t>(static_cast<uint32_t>(audio_payload.mel_spectrogram_bins_per_audio[i])),
                                        audio_payload.num_soft_tokens_per_audio[i] };
            const auto& mel = audio_payload.mel_spectrograms[i];
            h = prefix_hash_bytes(shape, sizeof(shape), 0x84222325cbf29ce4ull);
            h = prefix_hash_bytes(mel.data(), mel.size() * sizeof(mel[0]), h);
        } else {
            continue;
        }
        m.hash = h != 0 ? h : 1; // 0 means "not hashed"
    }
    const prompt_reuse_t plan = this->_plan_reuse(tokens, media);

    auto drop_front = [](auto& vec, size_t n) {
        if (n == 0) return;
        if (n >= vec.size()) { vec.clear(); return; }
        vec.erase(vec.begin(), vec.begin() + n);
    };
    const size_t images_to_drop = std::min<size_t>(plan.skipped_media[prompt_media_t::kImage], image_payload.num_images);
    if (images_to_drop > 0) {
        drop_front(image_payload.image_patch__element_per_patch, images_to_drop);
        drop_front(image_payload.valid_patch_size_per_image,     images_to_drop);
        drop_front(image_payload.pixel_values,                   images_to_drop);
        drop_front(image_payload.image_grid_pairs_per_image,     images_to_drop);
        drop_front(image_payload.num_soft_tokens_per_image,      images_to_drop);
        image_payload.num_images -= static_cast<unsigned>(images_to_drop);
    }
    const size_t audios_to_drop = std::min<size_t>(plan.skipped_media[prompt_media_t::kAudio], audio_payload.num_audios);
    if (audios_to_drop > 0) {
        drop_front(audio_payload.mel_spectrograms,                audios_to_drop);
        drop_front(audio_payload.mel_spectrogram_frames_per_audio, audios_to_drop);
        drop_front(audio_payload.mel_spectrogram_bins_per_audio,   audios_to_drop);
        drop_front(audio_payload.num_soft_tokens_per_audio,        audios_to_drop);
        audio_payload.num_audios -= static_cast<unsigned>(audios_to_drop);
    }

    // one past the last prefilled image row, relative to the skip: the first chunk is widened to hold the images,
    // which travel with it alone. Audio never widened it here, so it is left out.
    const int last_image_token_index = static_cast<int>(plan.media_end_by_kind[prompt_media_t::kImage]);

    // hardware
    gemma4e_multi_modal_payload_t multi_modal_payload;
    multi_modal_payload.image_payload = image_payload;
    multi_modal_payload.audio_payload = audio_payload;

    const bool has_multimodal = image_payload.num_images > 0 || audio_payload.num_audios > 0;

    bool success = has_multimodal
        ? this->_shared_insert(meta_info, tokens, is_cancelled, &multi_modal_payload, last_image_token_index)
        : this->_shared_insert(meta_info, tokens, is_cancelled, nullptr);

    // Single-turn (flash): the only checkpoint is the pinned system prefix, which Gemma4e_Flash owns. Taking this one would
    // overwrite it with the whole first prompt, and every later request would restore into that stale context.
    if (this->enable_think && !this->single_turn) {
        this->_checkpoint_history();
        int checkpoint_idx = this->lm_engine->checkpoint();
    }
    
    return success;
}

std::string Gemma4e::generate(chat_meta_info_t& meta_info, int length_limit, std::ostream& os, std::function<bool()> is_cancelled) {
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
    int last_sampled_token = this->last_token;

    this->token_history.push_back(last_token);
    // the sample insert() took is the reply's first token, and counts toward the limit
    meta_info.generated_tokens++;

    if (this->is_normal_token(last_sampled_token) && last_sampled_token != -1){
        std::string token_str = this->tokenizer->run_time_decoder(last_sampled_token);
        result += token_str;
        os << token_str << std::flush;

    }
    if (this->is_eos(last_sampled_token)){
        // Sampled from the prefill, so no engine has run it.
        this->token_history.pop_back();
        meta_info.stop_reason = reason;
        return result;
    }
    this->profiler_list[DECODING_TIME].reset();
    this->profiler_list[TKOEN_DECODE_TIME].reset();
    if (this->total_tokens >= this->MAX_L){
        header_print("WARNING", "Max length reached, stopping generation...");
        reason = MAX_LENGTH_REACHED;
    }
    else if ((length_limit > 0) && (meta_info.generated_tokens >= length_limit)){
        reason = MAX_LENGTH_REACHED;
    }
    else while (true){
        // Running out of window is a truncation: test it here so it is recorded.
        if (this->total_tokens >= this->MAX_L){
            reason = MAX_LENGTH_REACHED;
            break;
        }
        if (is_cancelled()) {
            reason = CANCEL_DETECTED;
            break;
        }
        this->profiler_list[DECODING_TIME].start();
        buffer<bf16> y = this->lm_engine->forward(last_sampled_token);
        this->profiler_list[DECODING_TIME].stop(1);

        this->profiler_list[SAMPLING_TIME].start();
        int sampled_token = this->_sample(y, meta_info);
        this->profiler_list[SAMPLING_TIME].stop(1);
        this->total_tokens++;
        last_sampled_token = sampled_token;

        this->profiler_list[TKOEN_DECODE_TIME].start();
        if (this->is_normal_token(sampled_token)){ // filter out special tokens
            std::string token_str = this->tokenizer->run_time_decoder(sampled_token);
            os << token_str << std::flush;
            result += token_str;
        }
        this->profiler_list[TKOEN_DECODE_TIME].stop(1);
        this->token_history.push_back(sampled_token);
        if (this->is_eos(sampled_token)){
            meta_info.generated_tokens++;
            // Its row is past the end of the engine's KV window once the context is
            // full: forwarding it there corrupts the engine for every later request,
            // so the next prompt has to carry it instead.
            if (this->total_tokens < this->MAX_L) {
                this->lm_engine->forward(last_sampled_token);
            } else {
                this->token_history.pop_back();
            }
            break;
        }
        meta_info.generated_tokens++;
        if ((length_limit > 0) && (meta_info.generated_tokens >= length_limit)){
            reason = MAX_LENGTH_REACHED;
            break;
        }
    }
    meta_info.decoding_duration = (uint64_t)(time_utils::cast_to_us(this->profiler_list[DECODING_TIME].get_total_time()).first) * 1e3;
    meta_info.stop_reason = reason;
    if (this->total_tokens >= this->MAX_L){
        header_print("WARNING", "Max length reached, stopping generation...");
    }
    std::cout << std::endl;
    header_print("FLM", "Model RAW Output: \n" + result);
    
    if (reason != EOT_DETECTED) {
        // Cut short after sampling its last token: no engine has run that one,
        // so the next prompt has to carry it.
        this->token_history.pop_back();
    }
    if (!this->enable_think) {
        // the history the checkpoint holds: an eos ending the turn was run, a cut-off token was not
        int checkpoint_idx = this->lm_engine->checkpoint();
        this->_checkpoint_history();
    }

    return result;
}

std::string Gemma4e::generate_with_prompt(chat_meta_info_t& meta_info, lm_uniform_input_t& input, int length_limit, std::ostream& os) {
    if (!this->insert(meta_info, input)) {
        return "";
    }
    if (this->enable_think) {
        os << "<think>\n" << std::flush;
    }

    int checkpoint_idx = this->lm_engine->checkpoint();
    int restore_idx = this->lm_engine->restore();
    header_print_r("FLM", "Checkpoint before generation: " << checkpoint_idx << ", restore point: " << restore_idx << ", user context length: " << this->token_history.size());
    return this->_shared_generate(meta_info, length_limit, os);
}
