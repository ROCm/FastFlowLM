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

void Qwen3VL::create_engine() {
    this->lm_engine = std::make_unique<qwen3vl_npu>(*this->lm_config, this->npu.get(), this->MAX_L);
}

void Qwen3VL_Flash::create_engine() {
    this->lm_engine = std::make_unique<qwen3vl_flash>(*this->lm_config, this->npu.get(), this->MAX_L);
}

void Qwen3VL::load_model(std::string model_path, json model_info, int default_context_length, bool enable_preemption) {
    this->_shared_load_model(model_path, model_info, default_context_length, enable_preemption);
    
    this->q4nx = std::make_unique<Q4NX>(this->model_path);
    // lm_config->get<std::string>("model_type", "") == qwen3
    this->create_engine();

    this->lm_engine->load_weights(*this->q4nx);
    //free the q4nx
    this->q4nx.reset();
    this->lm_engine->clear_context();
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
}

void Qwen3VL::setup_tokenizer(std::string model_path) {
    auto tokenizer_config = this->_shared_setup_tokenizer(model_path);
}

std::string Qwen3VL::apply_chat_template(nlohmann::ordered_json& messages, nlohmann::ordered_json tools) {
    minja::chat_template_inputs inputs;
    inputs.add_generation_prompt = true;
    inputs.messages = messages;
    
    if (!tools.empty())
        inputs.tools = tools;
    inputs.extra_context = this->extra_context;
    return this->chat_tmpl->apply(inputs);
}

bool Qwen3VL::_build_vl_tokens(lm_uniform_input_t& input,
                               qwen3vl_image_payload_t& image_payload,
                               std::vector<int>& tokens) {
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
        templated_text = this->apply_chat_template(qwenvl_message, input.tools);
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
            for (int j = 0; j < image_payload.images[image_counter].grid_h * image_payload.images[image_counter].grid_w / 4; j++) {
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
    if (!this->_build_vl_tokens(input, image_payload, tokens)) {
        return false;
    }

    // ----------------------------------------------------------------------
    // Prompt-cache aware image alignment.
    //
    // AutoModel::_shared_insert prefix-matches `tokens` against `token_history`
    // over the FULL length of `token_history`. If every token matches, it
    // erases that prefix before prefilling; otherwise it calls clear_context()
    // and skips nothing. We must NOT erase `tokens` here -- _shared_insert
    // needs the untrimmed sequence to run that very check. What we DO need to
    // fix up locally is the image payload (pixels for the WHOLE prompt,
    // including already-cached images from earlier turns): drop the
    // fully-cached leading images so the surviving payload aligns with the
    // surviving image tokens after _shared_insert performs its own erase.
    // ----------------------------------------------------------------------
    size_t prefix_skip_count = 0;
    {
        const size_t idx = this->token_history.size();
        for (size_t i = 0; i < idx; i++) {
            if (i < tokens.size() && tokens[i] == this->token_history[i]) {
                prefix_skip_count++;
            } else {
                break;
            }
        }
        // Must match the entirety of token_history, otherwise _shared_insert
        // will clear the context and not skip anything.
        if (prefix_skip_count != idx) {
            prefix_skip_count = 0;
        }

        if (prefix_skip_count > 0 && !image_payload.images.empty()) {
            // Count image-soft tokens in the cached prefix.
            int skipped_image_tokens = 0;
            for (size_t i = 0; i < prefix_skip_count; i++) {
                if (tokens[i] == image_soft_token_id) skipped_image_tokens++;
            }

            // Walk through images and drop those whose entire token block
            // sits within the cached prefix. (The chat-template prefix
            // boundary always falls between messages, so an image's token
            // block is never partially cached.)
            size_t images_to_drop = 0;
            size_t bf16_to_drop = 0;
            int consumed_image_tokens = 0;
            for (const auto& img : image_payload.images) {
                const int img_tokens =
                    (img.grid_h * img.grid_w) / 4;
                const size_t img_bf16 =
                    static_cast<size_t>(img.grid_h) * QWEN3_PATCH_SIZE *
                    static_cast<size_t>(img.grid_w) * QWEN3_PATCH_SIZE *
                    3u * QWEN3_TEMPORAL_PATCH_SIZE;
                if (consumed_image_tokens + img_tokens <= skipped_image_tokens) {
                    consumed_image_tokens += img_tokens;
                    bf16_to_drop += img_bf16;
                    images_to_drop++;
                } else {
                    break;
                }
            }

            if (images_to_drop > 0) {
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
                header_print("FLM",
                    "Prompt-cache hit: dropped " << images_to_drop
                    << " cached image(s) from payload");
            }
        }
    }

    // find the last image token index, expressed relative to the tokens that
    // will SURVIVE _shared_insert's prefix erase (i.e. shifted by -prefix_skip_count).
    int last_image_token_index = -1;
    for (int i = static_cast<int>(prefix_skip_count); i < (int)tokens.size(); i++) {
        if (tokens[i] == image_soft_token_id) {
            last_image_token_index = i - static_cast<int>(prefix_skip_count);
        }
    }
    last_image_token_index++; // plus the end of image tokens

    // hardware
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

// Non-stream
NonStreamResult Qwen3VL::parse_nstream_content(const std::string response_text) {
    NonStreamResult result;

    const std::string tool_start_tag = "<tool_call>";
    const std::string tool_end_tag = "</tool_call>";

    size_t search_from = 0;
    while (true) {
        size_t start_pos = response_text.find(tool_start_tag, search_from);
        if (start_pos == std::string::npos) break;

        size_t block_content_start = start_pos + tool_start_tag.length();
        size_t end_pos = response_text.find(tool_end_tag, block_content_start);
        size_t block_end = (end_pos != std::string::npos) ? end_pos : response_text.length();
        search_from = (end_pos != std::string::npos) ? end_pos + tool_end_tag.length() : block_end;

        std::string json_str = response_text.substr(block_content_start, block_end - block_content_start);

        std::string name, arguments;
        try {
            auto j = nlohmann::json::parse(json_str);
            if (j.contains("name")) name = j["name"].get<std::string>();
            if (j.contains("arguments")) {
                arguments = j["arguments"].is_string()
                    ? j["arguments"].get<std::string>()
                    : j["arguments"].dump();
            }
        } catch (...) {
            std::string key_name = "\"name\": \"";
            size_t ns = json_str.find(key_name);
            if (ns != std::string::npos) {
                ns += key_name.length();
                size_t ne = json_str.find("\"", ns);
                if (ne != std::string::npos) name = json_str.substr(ns, ne - ns);
            }
            std::string key_args = "\"arguments\":";
            size_t ap = json_str.find(key_args);
            if (ap != std::string::npos) {
                size_t bs = json_str.find("{", ap);
                size_t be = json_str.rfind("}");
                if (bs != std::string::npos && be != std::string::npos && be > bs)
                    arguments = json_str.substr(bs, be - bs + 1);
            }
        }

        result.tool_calls_list.emplace_back(name, arguments);
    }

    if (result.tool_calls_list.empty()) {
        result.content = response_text;
    } else {
        result.tool_name = result.tool_calls_list[0].first;
        result.tool_args = result.tool_calls_list[0].second;
    }

    return result;
}

// Stream
StreamResult Qwen3VL::parse_stream_content(const std::string content) {
    std::string tool_start_tag = "<tool_call>";
    std::string tool_end_tag = "</tool_call>";

    StreamResult result;
    result.type = StreamEventType::CONTENT;

    if (content.find(tool_start_tag) != std::string::npos) {
        is_in_tool_block_ = true;
        tool_name_.clear();
        result.type = StreamEventType::WAITING;
        return result;
    }

    if (content.find("</tool_call>") != std::string::npos) {
        is_in_tool_block_ = false;

        try {
            auto j = nlohmann::json::parse(tool_name_);

            result.type = StreamEventType::TOOL_DONE;
            result.tool_id = "call_" + std::to_string(std::time(nullptr));

            if (j.contains("name")) {
                result.tool_name = j["name"].get<std::string>();
            }

            if (j.contains("arguments")) {
                if (j["arguments"].is_string()) {
                    result.tool_args_str = j["arguments"].get<std::string>();
                }
                else {
                    result.tool_args_str = j["arguments"].dump();
                }
            }
        }
        catch (...) {
            result.type = StreamEventType::CONTENT;
            result.content = "[Error parsing tool call]";
        }
        return result;
    }

    if (is_in_tool_block_) {
        tool_name_ += content;
        result.type = StreamEventType::WAITING;
        return result;
    }

    result.content = content;
    return result;

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
    this->checkpoint_his = this->token_history;
    this->lm_engine->checkpoint();
    this->system_tokens = static_cast<int>(shared);

    // Reset profilers so the pin prefill does not count against the first turn.
    for (size_t i = 0; i < PROFILER_TYPE_NUM; i++) {
        this->profiler_list[i].reset();
    }

    header_print("FLM", "Qwen3VL_Flash: system prefix pinned (" + std::to_string(this->system_tokens) + " tokens).");
    return this->system_tokens;
}

void Qwen3VL_Flash::_reset_turn() {
    if (this->system_tokens > 0) {
        // Rewind to the pinned system prefix instead of clearing from scratch.
        this->total_tokens = static_cast<uint32_t>(this->lm_engine->restore());
        this->token_history = this->system_his;
        this->checkpoint_his = this->system_his;
    } else {
        this->lm_engine->clear_context();
        this->token_history.clear();
        this->total_tokens = 0;
    }
    this->sampler->reset_penalties();
    this->last_token = -1;
    this->reset_parser();
    this->tool_name_.clear();
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

    if (meta_info.stop_reason == CANCEL_DETECTED) {
        return false;
    }

    this->total_tokens = static_cast<uint32_t>(tokens.size());

    this->profiler_list[SAMPLING_TIME].start();
    this->last_token = this->sampler->sample(y);
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
    emit(last_sampled_token);

    if (!this->is_eos(last_sampled_token) && this->total_tokens < this->MAX_L) {
        while (true) {
            if (is_cancelled()) {
                reason = CANCEL_DETECTED;
                this->reset_parser();
                this->tool_name_.clear();
                break;
            }

            this->profiler_list[DECODING_TIME].start();
            buffer<bf16> y = this->lm_engine->forward(last_sampled_token);
            this->profiler_list[DECODING_TIME].stop(1);

            this->profiler_list[SAMPLING_TIME].start();
            last_sampled_token = this->sampler->sample(y);
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
    std::string result;
    os << "<think>\n\n";
    result = this->_shared_generate(meta_info, length_limit, os, is_cancelled);
    result = "<think>\n\n" + result;
    return result;
}
