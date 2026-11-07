/// \file deepseek.cpp
/// \brief deepseek class
/// \author FastFlowLM Team
/// \date 2025-09-01
/// \version 0.9.24
/// \note This is a source file for the deepseek class


#include "AutoModel/modeling_qwen3vl.hpp"
#include "metrices.hpp"


/************              Qwen3VL family            **************/
Qwen3VL::Qwen3VL(flm_rt::device* npu_device_inst) : AutoModel(npu_device_inst, "Qwen3VL") {}

void Qwen3VL::load_model(std::string model_path, json model_info, int default_context_length, bool enable_preemption, const std::string& backend) {
    this->_shared_load_backend(model_path, model_info, default_context_length, enable_preemption, backend);
    this->setup_tokenizer(model_path);
    this->sampler.reset();

    sampler_config config;
    config.top_k = 40;
    config.top_p = 0.9;
    config.min_p = 0.1;
    config.temperature = 0.8;
    config.rep_penalty = 1.05;
    config.freq_penalty = 1.05;

    this->set_sampler(config);
    for (size_t i = 0; i < PROFILER_TYPE_NUM; i++) {
        this->profiler_list[i].reset();
    }
    this->_prefix_attach(this->lm_engine, qwen3vl_npu_state_api(this->lm_engine), IMAGE_SOFT_TOKEN_ID);
}

void Qwen3VL::setup_tokenizer(std::string model_path) {
    auto tokenizer_config = this->_shared_setup_tokenizer(model_path);
}

std::string Qwen3VL::apply_chat_template(nlohmann::ordered_json& messages, nlohmann::ordered_json tools, bool continue_final_message) {
    minja::chat_template_inputs inputs;
    inputs.messages = messages;

    if (!tools.empty())
        inputs.tools = tools;
    inputs.extra_context = this->extra_context;
    return this->_render_chat_template(inputs, continue_final_message);
}

// the <|image_pad|> rows one image is expanded into: one per merge_size x merge_size block of patches.
// The expansion and the prompt-cache media list both use it, so they cannot disagree.
static size_t qwen3vl_image_token_count(const qwen3vl_image_t& image) {
    return static_cast<size_t>(image.grid_h) * static_cast<size_t>(image.grid_w) / (QWEN3_MERGE_SIZE * QWEN3_MERGE_SIZE);
}

bool Qwen3VL::_build_vl_tokens(lm_uniform_input_t& input,
                               qwen3vl_image_payload_t& image_payload,
                               std::vector<int>& tokens,
                               bool continue_final_message) {
    if (input.messages.empty() && input.prompt.empty()) {
        header_print("WARNING", "No messages or prompt provided");
        return false;
    }

    this->profiler_list[TKOEN_ENCODE_TIME].start();
    std::string templated_text;

    image_payload.num_images = 0;
    if (input.images.size() > 0) {
        for(const auto& img_str : input.images){
            qwen3vl_image_t image = this->load_image(img_str);
            if (image.width <= 0 || image.height <= 0) {
                header_print("ERROR", "Skipping image that failed to load: " << img_str);
                continue;
            }

            preprocess_image(image, image_payload._data__processed);
            if (image.grid_h <= 0 || image.grid_w <= 0) {
                header_print("ERROR", "Skipping image that failed to preprocess: " << img_str);
                continue;
            }
            // Push the image AFTER preprocessing so grid_h and grid_w are set
            image_payload.images.push_back(image);
            image_payload.num_images++;
        }
    }
    if (!input.messages.empty()) { // already a formated messages, usually from REST API
        json qwenvl_message = json::array();
        int total_images = 0;
        for (const auto& item : input.messages) {
            if (!item.contains("images")) {
                qwenvl_message.push_back(item);
                continue;
            }

            json newContent = json::array();
            for (const auto& img : item["images"]) {
                std::string img_str = img.get<std::string>();

                // Decode/preprocess up front so an invalid image (e.g. bad
                // base64) never gets an image-soft-token placeholder in the
                // template below; otherwise image_payload.images would fall
                // out of sync with the placeholders and the prompt would
                // still try to prefill a nonexistent image.
                qwen3vl_image_t image = this->load_image_base64(img_str);
                if (image.width <= 0 || image.height <= 0) {
                    header_print("ERROR", "Skipping invalid base64 image; prefilling language only for this item");
                    continue;
                }

                preprocess_image(image, image_payload._data__processed);
                if (image.grid_h <= 0 || image.grid_w <= 0) {
                    header_print("ERROR", "Skipping image that failed to preprocess");
                    continue;
                }

                image_payload.images.push_back(image);
                image_payload.num_images++;
                total_images++;

                newContent.push_back({
                    {"type", "image"},
                    {"image", img}
                });
            }
            newContent.push_back({
                {"type", "text"},
                {"text", item["content"]}
            });

            json newItem = {
                {"role", item["role"]},
                {"content", newContent}
            };

            qwenvl_message.push_back(newItem);
        }
        templated_text = this->apply_chat_template(qwenvl_message, input.tools, continue_final_message);
        header_print("FLM", "Total images: " << total_images);
    }
    else if (!input.prompt.empty()) { // a pure text, usually from the cli
        nlohmann::ordered_json messages;
        nlohmann::ordered_json content;
        content["role"] = "user";
        content["content"] = nlohmann::ordered_json::array();
        
        // Add image objects to content array
        for (int i = 0; i < input.images.size(); i++) {
            nlohmann::ordered_json image_obj;
            image_obj["type"] = "image";
            image_obj["image"] = input.images[i];
            content["content"].push_back(image_obj);
        }
        
        // Add text object to content array
        nlohmann::ordered_json text_obj;
        text_obj["type"] = "text";
        text_obj["text"] = input.prompt;
        content["content"].push_back(text_obj);
        
        messages.push_back(content);
        templated_text = this->apply_chat_template(messages);
    }
    std::vector<int> tokens_init = this->tokenizer->encode(templated_text);

    // update the tokens to include the image tokens
    int total_image_tokens = 0;
    // Use image_payload.images.size() (not input.images.size()), because on
    // the REST API path images come from `messages` and input.images is empty.
    for (size_t i = 0; i < image_payload.images.size(); i++) {
        total_image_tokens += image_payload.images[i].grid_h * image_payload.images[i].grid_w;
    }
    tokens.reserve(tokens_init.size() + total_image_tokens);
    int image_counter = 0;
    for (int i = 0; i < tokens_init.size(); i++) {
        if (tokens_init[i] == IMAGE_SOFT_TOKEN_ID) {
            for (size_t j = 0; j < qwen3vl_image_token_count(image_payload.images[image_counter]); j++) {
                tokens.push_back(IMAGE_SOFT_TOKEN_ID);
            }
            image_counter++;
        } else {
            tokens.push_back(tokens_init[i]);
        }
    }

    this->profiler_list[TKOEN_ENCODE_TIME].stop(tokens.size());
    return true;
}

bool Qwen3VL::insert(chat_meta_info_t& meta_info, lm_uniform_input_t& input, std::function<bool()> is_cancelled) {
    constexpr int image_soft_token_id = IMAGE_SOFT_TOKEN_ID;
    qwen3vl_image_payload_t image_payload;
    std::vector<int> tokens;
    if (!this->_build_vl_tokens(input, image_payload, tokens, meta_info.continue_final_message)) {
        return false;
    }

    // ---- prefix slots: resume from the longest saved prefix of this prompt ----
    // The generation prompt ("<|im_start|>assistant\n") is how the next turn renders this reply's header: the whole prompt
    // is saved (trim 0). After a resume token_history is the slot's tokens, so the skip count below is the match length.
    const bool slot_mode = this->_prefix_slot_mode(meta_info, tokens.size(), 0);
    size_t match_len = 0;
    std::vector<int> full_tokens;          // what this prompt leaves in the engine
    // content of each of its images (pixels + grid), over the full payload: the slot cache and the prompt cache both use it
    std::vector<uint64_t> image_hashes;
    if (!image_payload.images.empty()) {
        image_hashes = prefix_image_hashes(image_payload, QWEN3_PATCH_SIZE, QWEN3_TEMPORAL_PATCH_SIZE);
    }
    if (slot_mode) {
        full_tokens = tokens;
        match_len = this->_prefix_resume(this->prefix_cache_, full_tokens, image_hashes);
    } else {
        this->_prefix_legacy_path(); // the in-place path below moves the engine on without telling the cache
    }

    // ---- prompt cache: drop the images the engine already holds ----
    // The payload carries the pixels of every image of the prompt, but _shared_insert skips the part of the
    // prompt the engine holds; plan_prompt_reuse tells both sides the same skip, so the images left in the
    // payload are exactly the ones on the rows that get prefilled.
    std::vector<prompt_media_t> media;
    {
        std::vector<size_t> image_token_counts;
        image_token_counts.reserve(image_payload.images.size());
        for (const auto& img : image_payload.images) {
            image_token_counts.push_back(qwen3vl_image_token_count(img));
        }
        prompt_media_from_counts(tokens, image_soft_token_id, image_token_counts, prompt_media_t::kImage, media);
        // so a same-size but different image in the cached part of the prompt is not taken for the old one
        for (size_t i = 0; i < media.size() && i < image_hashes.size(); i++) {
            media[i].hash = image_hashes[i] != 0 ? image_hashes[i] : 1; // 0 means "not hashed"
        }
    }
    const prompt_reuse_t plan = this->_plan_reuse(tokens, media);
    const size_t images_to_drop = plan.skipped_media[prompt_media_t::kImage];
    if (images_to_drop > 0) {
        size_t bf16_to_drop = 0;
        for (size_t i = 0; i < images_to_drop; i++) {
            const auto& img = image_payload.images[i];
            bf16_to_drop += static_cast<size_t>(img.grid_h) * QWEN3_PATCH_SIZE *
                            static_cast<size_t>(img.grid_w) * QWEN3_PATCH_SIZE *
                            3u * QWEN3_TEMPORAL_PATCH_SIZE;
        }
        image_payload.images.erase(
            image_payload.images.begin(),
            image_payload.images.begin() + images_to_drop);
        image_payload.num_images -= static_cast<int>(images_to_drop);
        if (bf16_to_drop >= image_payload._data__processed.size()) {
            image_payload._data__processed.clear();
        } else {
            image_payload._data__processed.erase(
                image_payload._data__processed.begin(),
                image_payload._data__processed.begin() + bf16_to_drop);
        }
    }

    // one past the last image row that gets prefilled, relative to the skip (0 when none is)
    const int last_image_token_index = static_cast<int>(plan.media_end);

    // hardware
    if (slot_mode) {
        return this->_prefix_insert(this->prefix_cache_, meta_info, tokens, full_tokens, image_hashes, match_len, "<|im_start|>user",
                                    image_payload.num_images > 0 ? static_cast<void*>(&image_payload) : nullptr, last_image_token_index, is_cancelled);
    }
    if (image_payload.num_images > 0){
        return this->_shared_insert(meta_info, tokens, is_cancelled, &image_payload, last_image_token_index);
    }else{
        return this->_shared_insert(meta_info, tokens, is_cancelled, nullptr);
    }

}

std::string Qwen3VL::generate(chat_meta_info_t& meta_info, int length_limit, std::ostream& os, std::function<bool()> is_cancelled) {
    return this->_shared_generate(meta_info, length_limit, os, is_cancelled);
}

std::string Qwen3VL::generate_with_prompt(chat_meta_info_t& meta_info, lm_uniform_input_t& input, int length_limit, std::ostream& os) {
    if (!this->insert(meta_info, input)) {
        return "";
    }
    return this->_shared_generate(meta_info, length_limit, os);
}

/************              Qwen3VL_Flash            **************/

int Qwen3VL_Flash::_pin_system_prefix(const std::string& system_text) {
    // Full clear — drop any previous pin before building the new one.
    this->lm_engine->clear_context();
    this->token_history.clear();
    this->total_tokens = 0;
    this->last_token = -1;
    this->system_tokens = 0;
    this->system_his.clear();
    this->pinned_system_text = system_text;

    if (system_text.empty()) {
        return 0;
    }

    // Find how many tokens the system turn plus the opening of the user turn
    // share across different user texts (the variable part). Template two
    // otherwise identical prompts that differ only in the first character of
    // the user message and take the longest common prefix.
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
        header_print("WARNING", "Qwen3VL_Flash: could not isolate a reusable system prefix, every turn will prefill in full");
        return 0;
    }

    std::vector<int> tokens(probe_a.begin(), probe_a.begin() + shared);
    chat_meta_info_t pin_meta;
    pin_meta.restore_allowed = false;
    if (!this->_shared_insert(pin_meta, tokens, [] { return false; }, nullptr)) {
        this->lm_engine->clear_context();
        this->token_history.clear();
        return 0;
    }

    this->system_his = this->token_history;
    this->_checkpoint_history();
    this->lm_engine->checkpoint();
    this->system_tokens = static_cast<int>(shared);

    // Reset profilers so the pin prefill does not count against the first turn.
    for (size_t i = 0; i < PROFILER_TYPE_NUM; i++) {
        this->profiler_list[i].reset();
    }

    header_print("FLM", "Qwen3VL_Flash: system prefix pinned (" + std::to_string(this->system_tokens) + " tokens).");
    return this->system_tokens;
}

void Qwen3VL_Flash::_forget_pin() {
    this->pinned_system_text.clear();
    this->system_his.clear();
    this->system_tokens = 0;
}

void Qwen3VL_Flash::clear_context() {
    this->AutoModel::clear_context();
    this->_forget_pin();
}

void Qwen3VL_Flash::_reset_turn() {
    if (this->system_tokens > 0) {
        // Rewind to the pinned system prefix instead of clearing from scratch.
        this->total_tokens = static_cast<uint32_t>(this->lm_engine->restore());
        this->token_history = this->system_his;
        this->checkpoint_his = this->system_his;
        // the pinned prefix is text only (a system string, read in without a payload)
        this->history_media_ = std::vector<prompt_media_t>{};
        this->checkpoint_media_ = std::vector<prompt_media_t>{};
    } else {
        this->lm_engine->clear_context();
        this->token_history.clear();
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

bool Qwen3VL_Flash::insert(chat_meta_info_t& meta_info, lm_uniform_input_t& input, std::function<bool()> is_cancelled) {
    // Extract the system text from the incoming messages (first message only).
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

    // Re-pin only when the system text actually changed (or is being set for
    // the first time). The initial state has pinned_system_text == "" and
    // system_tokens == 0, so the first request with a system prompt triggers
    // a pin; subsequent requests with the same text skip straight to _reset_turn.
    if (sys_text != this->pinned_system_text) {
        this->_pin_system_prefix(sys_text);
    }

    this->_reset_turn();

    qwen3vl_image_payload_t image_payload;
    std::vector<int> tokens;
    if (!this->_build_vl_tokens(input, image_payload, tokens)) {
        return false;
    }

    if (tokens.size() >= this->MAX_L) {
        header_print("WARNING", "Prompt does not fit in the context window, stopping prefilling...");
        meta_info.error_message = "Prompt does not fit in the context window: " + std::to_string(tokens.size()) + " tokens, the context length is " + std::to_string(this->MAX_L);
        return false;
    }

    // When a system prefix is pinned the engine's kv cache already holds those
    // tokens (restored by _reset_turn). Strip them from the front of `tokens`
    // so we only prefill the part the engine has not seen yet.
    int skip = 0;
    if (this->system_tokens > 0
        && static_cast<int>(tokens.size()) > this->system_tokens) {
        bool prefix_matches = std::equal(
            this->system_his.begin(), this->system_his.end(), tokens.begin());
        if (prefix_matches) {
            skip = this->system_tokens;
        }
    }
    if (this->system_tokens > 0 && skip == 0) {
        // The prompt does not start with the pinned tokens (the user text can
        // merge with the end of the probe prefix, e.g. one starting with a
        // newline), yet the engine holds them: prefilling on top would leave it
        // more rows than total_tokens counts, and the system turn twice. Start
        // from an empty cache; the prefill below overwrites the pinned rows, so
        // the pin goes too, and the next request with this system text pins it again.
        this->lm_engine->clear_context();
        this->checkpoint_his.clear();
        this->checkpoint_media_ = std::vector<prompt_media_t>{};
        this->_forget_pin();
    }

    // The vision encoder runs once, on the chunk that carries the payload, so
    // that chunk has to be long enough to hold every image soft token.
    int first_len_run = 0;
    for (int i = static_cast<int>(tokens.size()) - 1; i >= 0; i--) {
        if (tokens[i] == IMAGE_SOFT_TOKEN_ID) {
            first_len_run = std::max(0, i + 1 - skip);
            break;
        }
    }

    this->token_history = tokens;
    this->_begin_turn(meta_info, tokens);
    std::vector<int> tokens_to_prefill(tokens.begin() + skip, tokens.end());

    auto prefill_start_time = this->profiler_list[PREFILL_TIME].start();
    buffer<bf16> y = this->_chunked_insert(
        meta_info, tokens_to_prefill, is_cancelled,
        image_payload.num_images > 0 ? &image_payload : nullptr,
        first_len_run);
    auto prefill_end_time = this->profiler_list[PREFILL_TIME].stop(tokens_to_prefill.size());

    meta_info.prefill_duration = (uint64_t)time_utils::duration_ns(prefill_start_time, prefill_end_time).first;
    meta_info.prompt_tokens = static_cast<int>(tokens.size()); // report full prompt length to caller
    // The pinned system prefix is already in the kv cache, so it is part of the
    // prompt but was not prefilled on this turn.
    meta_info.cached_prompt_tokens = skip;

    if (meta_info.stop_reason == CANCEL_DETECTED || meta_info.stop_reason == ERROR_DETECTED) {
        return false;
    }

    this->total_tokens = static_cast<uint32_t>(tokens.size());

    this->profiler_list[SAMPLING_TIME].start();
    this->last_token = this->_sample(y, meta_info);
    meta_info.insert_sample_pending = true;
    this->profiler_list[SAMPLING_TIME].stop(1);
    return true;
}

std::string Qwen3VL_Flash::generate(chat_meta_info_t& meta_info, int length_limit, std::ostream& os, std::function<bool()> is_cancelled) {
    std::string result;
    if (this->last_token == -1) {
        header_print("ERROR", "generate() called without a successful insert()");
        meta_info.stop_reason = ERROR_DETECTED;
        return result;
    }

    auto emit = [&](int token) {
        if (!this->is_normal_token(token)) {
            return;
        }
        std::string token_str = this->tokenizer->run_time_decoder(token);
        result += token_str;
        os << token_str << std::flush;
    };

    stop_reason_t reason = EOT_DETECTED;
    int last_sampled_token = this->last_token;
    this->token_history.push_back(last_sampled_token);
    // the sample insert() took is the reply's first token, and counts toward the limit
    meta_info.generated_tokens++;
    emit(last_sampled_token);

    if ((length_limit > 0) && (meta_info.generated_tokens >= length_limit)
        && !this->is_eos(last_sampled_token)) {
        reason = MAX_LENGTH_REACHED;
    }
    else if (!this->is_eos(last_sampled_token) && this->total_tokens < this->MAX_L) {
        while (true) {
            if (is_cancelled()) {
                reason = CANCEL_DETECTED;
                break;
            }

            this->profiler_list[DECODING_TIME].start();
            buffer<bf16> y = this->lm_engine->forward(last_sampled_token);
            this->profiler_list[DECODING_TIME].stop(1);

            this->profiler_list[SAMPLING_TIME].start();
            last_sampled_token = this->_sample(y, meta_info);
            this->profiler_list[SAMPLING_TIME].stop(1);
            this->total_tokens++;

            this->profiler_list[TKOEN_DECODE_TIME].start();
            emit(last_sampled_token);
            this->profiler_list[TKOEN_DECODE_TIME].stop(1);

            this->token_history.push_back(last_sampled_token);
            meta_info.generated_tokens++;

            if (this->is_eos(last_sampled_token)) {
                break;
            }
            if ((length_limit > 0) && (meta_info.generated_tokens >= length_limit)) {
                reason = MAX_LENGTH_REACHED;
                break;
            }
            if (this->total_tokens >= this->MAX_L) {
                header_print("WARNING", "Max length reached, stopping generation...");
                reason = MAX_LENGTH_REACHED;
                break;
            }
        }
    }

    meta_info.decoding_duration = (uint64_t)(time_utils::cast_to_us(this->profiler_list[DECODING_TIME].get_total_time()).first) * 1e3;
    meta_info.stop_reason = reason;
    // Whatever the stop, eos included, its last sampled token was never
    // forward()ed (this loop does not feed an eos back), so the history drops it.
    this->token_history.pop_back();
    if (this->log_raw_output) {
        std::cout << std::endl;
        header_print("FLM", "Model RAW Output: \n" + result);
    }
    return result;
}

std::string Qwen3VL_Flash::generate_with_prompt(chat_meta_info_t& meta_info, lm_uniform_input_t& input, int length_limit, std::ostream& os) {
    if (!this->insert(meta_info, input)) {
        return "";
    }
    return this->generate(meta_info, length_limit, os);
}


/************              Qwen3VL_Thinking            **************/

std::string Qwen3VL_Thinking::generate_with_prompt(chat_meta_info_t& meta_info, lm_uniform_input_t& input, int length_limit, std::ostream& os) {
    if (!this->insert(meta_info, input)) {
        return "";
    }
    return this->generate(meta_info, length_limit, os);
}


std::string Qwen3VL_Thinking::generate(chat_meta_info_t& meta_info, int length_limit, std::ostream& os, std::function<bool()> is_cancelled) {
    // the echo stands for the generation prompt's "<think>\n"; a continued message
    // has no generation prompt, its output picks up right after the client's text
    if (meta_info.continue_final_message) {
        return this->_shared_generate(meta_info, length_limit, os, is_cancelled);
    }
    std::string result;
    os << "<think>\n\n";
    result = this->_shared_generate(meta_info, length_limit, os, is_cancelled);
    result = "<think>\n\n" + result;
    return result;
}
