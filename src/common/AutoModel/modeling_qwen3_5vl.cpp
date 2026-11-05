/// \file deepseek.cpp
/// \brief deepseek class
/// \author FastFlowLM Team
/// \date 2025-09-01
/// \version 0.9.24
/// \note This is a source file for the deepseek class


#include "AutoModel/modeling_qwen3_5vl.hpp"
#include "metrices.hpp"


/************              Qwen3_5VL family            **************/
Qwen3_5VL::Qwen3_5VL(flm_rt::device* npu_device_inst) : AutoModel(npu_device_inst, "Qwen3_5VL") {}

void Qwen3_5VL::load_model(std::string model_path, json model_info, int default_context_length, bool enable_preemption, const std::string& backend) {
    this->_shared_load_backend(model_path, model_info, default_context_length, enable_preemption, backend);
    this->setup_tokenizer(model_path);
    this->sampler.reset();

    this->enable_tool = (model_info["size"] > 800000000)? true : false;

    sampler_config config;
    config.top_k = 20;
    config.top_p = 0.8;
    config.min_p = 0.0;
    config.temperature = 0.7;
    config.rep_penalty = 1.0;
    config.freq_penalty = 1.0;
    config.pre_penalty = 1.5f;

    this->set_sampler(config);
    for (size_t i = 0; i < PROFILER_TYPE_NUM; i++) {
        this->profiler_list[i].reset();
    }
    this->_attach_prefix_cache();
}

void Qwen3_5VL::setup_tokenizer(std::string model_path) {
    auto tokenizer_config = this->_shared_setup_tokenizer(model_path);
}

std::string Qwen3_5VL::apply_chat_template(nlohmann::ordered_json& messages, nlohmann::ordered_json tools, bool continue_final_message) {
    minja::chat_template_inputs inputs;
    inputs.messages = messages;
    inputs.extra_context = this->extra_context;
    inputs.extra_context["enable_thinking"] = this->enable_think;
    if (!tools.empty() && this->enable_tool)
        inputs.tools = tools;
    return this->_render_chat_template(inputs, continue_final_message);
}


Qwen3_5VL::vision_params_t Qwen3_5VL::_vision_params() const {
    auto read = [](const auto& e) {
        return vision_params_t{e.QWEN3_5_PATCH_SIZE, e.QWEN3_5_IMAGE_MERGE_SIZE, e.QWEN3_5_SHORTEST_EDGE, e.QWEN3_5_LONGEST_EDGE,
                               e.QWEN3_5_TEMPORAL_PATCH_SIZE, e.QWEN3_5_MERGE_SIZE, e.QWEN3_5_VISION_RESCALE_FACTOR,
                               e.QWEN3_5_VISION_RESCALE_IMAGE_MEAN, e.QWEN3_5_VISION_RESCALE_IMAGE_STD};
    };
    if (const auto* e = dynamic_cast<const qwen3_5vl_npu*>(this->lm_engine)) {
        return read(*e);
    }
    if (const auto* e = dynamic_cast<const qwen3_5vl_flash*>(this->lm_engine)) {
        return read(*e);
    }
    throw std::runtime_error("Qwen3_5VL: the loaded engine is neither qwen3_5vl_npu nor qwen3_5vl_flash");
}

/************              prefix slots            **************/
namespace {
constexpr int kImageSoftTokenId = 248056;

/// \brief the soft tokens one image expands into: one per merge x merge block of patches
/// \note _build_tokens expands with it and insert() describes the images with it, so the two cannot disagree
size_t image_soft_tokens(const qwen3_5vl_image_t& image, unsigned int merge_size) {
    return static_cast<size_t>(image.grid_h) * static_cast<size_t>(image.grid_w) / (static_cast<size_t>(merge_size) * merge_size);
}
} // namespace

void Qwen3_5VL::_attach_prefix_cache() {
    this->_prefix_attach(this->lm_engine, qwen3_5vl_state_api(this->lm_engine), kImageSoftTokenId);
}

bool Qwen3_5VL::_build_tokens(chat_meta_info_t& meta_info, lm_uniform_input_t& input, qwen3_5vl_image_payload_t& image_payload, std::vector<int>& tokens) {
    // preprocess
    constexpr int image_soft_token_id = 248056;
    this->profiler_list[TKOEN_ENCODE_TIME].start();
    std::string templated_text;
    if (input.messages.empty() && input.prompt.empty()) {
        header_print("WARNING", "No messages or prompt provided");
        return false;
    }

    constexpr bool DEBUG_IMAGE_PREPROCESS = false;
    image_payload.num_images = 0;
    if (input.images.size() > 0) {


        // header_print("info", "Processing images...");
        
        // time_utils::time_point preprocess_start = time_utils::now();
        for(const auto& img_str : input.images){
            qwen3_5vl_image_t image = this->load_image(img_str);
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
                qwen3_5vl_image_t image = this->load_image_base64(img_str);
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
        templated_text = this->apply_chat_template(qwenvl_message, input.tools, meta_info.continue_final_message);
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
    tokens.clear();
    int total_image_tokens = 0;
    // Use image_payload.images.size() (not input.images.size()), because on
    // the REST API path images come from `messages` and input.images is empty.
    for (size_t i = 0; i < image_payload.images.size(); i++) {
        total_image_tokens += image_payload.images[i].grid_h * image_payload.images[i].grid_w;
    }
    tokens.reserve(tokens_init.size() + total_image_tokens);
    // only asked with images: the core backend has no vision config and throws
    const unsigned int merge_size = image_payload.images.empty() ? 0 : this->_vision_params().QWEN3_5_MERGE_SIZE;
    int image_counter = 0;
    for (int i = 0; i < tokens_init.size(); i++) {
        if (tokens_init[i] == image_soft_token_id) {
            const size_t image_tokens = image_soft_tokens(image_payload.images[image_counter], merge_size);
            for (size_t j = 0; j < image_tokens; j++) {
                tokens.push_back(image_soft_token_id);
            }
            image_counter++;
        } else {
            tokens.push_back(tokens_init[i]);
        }
    }

    this->profiler_list[TKOEN_ENCODE_TIME].stop(tokens.size());
    return true;
}

bool Qwen3_5VL::insert(chat_meta_info_t& meta_info, lm_uniform_input_t& input, std::function<bool()> is_cancelled) {
    constexpr int image_soft_token_id = 248056;
    qwen3_5vl_image_payload_t image_payload;
    std::vector<int> tokens;
    if (!this->_build_tokens(meta_info, input, image_payload, tokens)) {
        return false;
    }

    // ---- prefix slots: resume from the longest saved prefix of this prompt ----
    // generate() feeds the generation prompt's think tokens itself, so they are left out of the prefill. A continued
    // message keeps all its tokens: they are the client's text, not the generation prompt.
    const size_t trim_tokens = meta_info.continue_final_message ? 0 : (this->enable_think ? 2 : 4);
    qwen3_5vl_npu* slot_eng = dynamic_cast<qwen3_5vl_npu*>(this->lm_engine);
    const bool slot_mode = this->prefix_cache_.enabled() && slot_eng != nullptr && meta_info.restore_allowed && tokens.size() > trim_tokens;
    size_t match_len = 0;
    std::vector<int> full_tokens;          // what this prompt leaves in the engine
    std::vector<uint64_t> image_hashes;    // content of each of its images
    if (this->prefix_cache_.enabled() && !slot_mode) {
        // the single-checkpoint path below moves the engine on without telling the cache
        this->prefix_cache_.forget_all();
    }
    if (slot_mode) {
        full_tokens.assign(tokens.begin(), tokens.end() - trim_tokens);
        image_hashes = prefix_image_hashes(image_payload, slot_eng->QWEN3_5_PATCH_SIZE, slot_eng->QWEN3_5_TEMPORAL_PATCH_SIZE);
        match_len = this->_prefix_resume(this->prefix_cache_, full_tokens, image_hashes);
    }

    // hardware
    int restore_idx = -1;
    if (!slot_mode && meta_info.restore_allowed) {
        restore_idx = this->lm_engine->restore();
        this->total_tokens = restore_idx;
        this->token_history = checkpoint_his; // restore the token history to be consistent with the restored KV cache, which is crucial for correct functioning of _shared_insert's prefix-matching logic
    }

    size_t n = tokens.size();
    tokens.resize(n - trim_tokens);

    // ---- prompt cache: drop the images the engine already holds ----
    // The payload holds the pixels of every image of the prompt, but _shared_insert prefills only what follows the
    // part of the prompt the engine keeps; ask for that skip on the exact tokens it will get, against the history
    // it will compare with (restored, resumed from a slot, or live), and drop the images inside it.
    std::vector<prompt_media_t> media;
    if (!image_payload.images.empty()) {
        const unsigned int merge_size = this->_vision_params().QWEN3_5_MERGE_SIZE;
        std::vector<size_t> image_tokens;
        image_tokens.reserve(image_payload.images.size());
        for (const auto& img : image_payload.images) {
            image_tokens.push_back(image_soft_tokens(img, merge_size));
        }
        prompt_media_from_counts(tokens, image_soft_token_id, image_tokens, prompt_media_t::kImage, media);
    }
    const prompt_reuse_t plan = this->_plan_reuse(tokens, media);

    const size_t images_to_drop = plan.skipped_media[prompt_media_t::kImage];
    if (images_to_drop > 0) {
        // Per-image bf16 footprint depends on runtime patch/temporal
        // config carried by the engine.
        const auto vision = this->_vision_params();
        const unsigned patch_size = vision.QWEN3_5_PATCH_SIZE;
        const unsigned temporal_patch = vision.QWEN3_5_TEMPORAL_PATCH_SIZE;

        size_t bf16_to_drop = 0;
        for (size_t i = 0; i < images_to_drop; i++) {
            const auto& img = image_payload.images[i];
            bf16_to_drop +=
                static_cast<size_t>(img.grid_h) * patch_size *
                static_cast<size_t>(img.grid_w) * patch_size *
                3u * temporal_patch;
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

    // one past the last image token, relative to the tokens that survive _shared_insert's skip (0: no image left)
    const int last_image_token_index = static_cast<int>(plan.media_end);
    const bool has_images = image_payload.num_images > 0;

    if (slot_mode) {
        return this->_prefix_insert(this->prefix_cache_, meta_info, tokens, full_tokens, image_hashes, match_len, "<|im_start|>user",
                                    has_images ? static_cast<void*>(&image_payload) : nullptr, last_image_token_index, is_cancelled);
    }

    bool success = has_images
        ? this->_shared_insert(meta_info, tokens, is_cancelled, &image_payload, last_image_token_index)
        : this->_shared_insert(meta_info, tokens, is_cancelled, nullptr);

    checkpoint_his = token_history;
    int checkpoint_idx = this->lm_engine->checkpoint();
    return success;
}

std::string Qwen3_5VL::generate(chat_meta_info_t& meta_info, int length_limit, std::ostream& os, std::function<bool()> is_cancelled) {
    // the think preamble below completes the generation prompt insert() left out; a continued
    // message was prefilled whole, so its output starts from the sample insert() made
    if (meta_info.continue_final_message) {
        return this->_shared_generate(meta_info, length_limit, os, std::move(is_cancelled));
    }
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

    // The engine runs ahead of total_tokens: the four preamble tokens below are
    // fed to it but only the sampled one is counted, so the next forward lands
    // on row total_tokens + 3, and the extra forward after an EOS one further.
    // Keep that much room under MAX_L, or those forwards write past the end of
    // the engine's KV window and corrupt it for every later request.
    constexpr uint32_t kEngineLead = 4;
    if (this->total_tokens + kEngineLead >= this->MAX_L){
        header_print("WARNING", "Max length reached, stopping generation...");
        meta_info.stop_reason = MAX_LENGTH_REACHED;
        return result;
    }

    this->profiler_list[DECODING_TIME].reset();
    this->profiler_list[TKOEN_DECODE_TIME].reset();
    std::string token_str;
    int sampled_token;
    if(this->enable_think) {
        this->token_history.push_back(think_start_id);
        this->profiler_list[DECODING_TIME].start();
        this->_force_token(meta_info, think_start_id);
        this->lm_engine->forward(think_start_id);
        this->profiler_list[DECODING_TIME].stop(1);
        token_str = this->tokenizer->run_time_decoder(think_start_id);
        result += token_str;
        os << token_str << std::flush;

        // \n
        this->token_history.push_back(198);
        this->profiler_list[DECODING_TIME].start();
        this->_force_token(meta_info, 198);
        buffer<bf16> y = this->lm_engine->forward(198);
        this->profiler_list[DECODING_TIME].stop(1);
        token_str = this->tokenizer->run_time_decoder(198);
        result += token_str;
        // First token of the reasoning block.
        sampled_token = this->_sample(y, meta_info);
        os << token_str << std::flush;
    }
    else{
        this->token_history.push_back(think_start_id);
        // fed for the KV cache only, it never reaches the output
        this->_skip_insert_sample(meta_info);
        this->lm_engine->forward(think_start_id);
        token_str = this->tokenizer->run_time_decoder(think_start_id);

        // \n\n
        this->token_history.push_back(271);
        this->profiler_list[DECODING_TIME].start();
        this->lm_engine->forward(271);
        this->profiler_list[DECODING_TIME].stop(1);
        token_str = this->tokenizer->run_time_decoder(271);
        
        this->token_history.push_back(think_end_id);
        this->profiler_list[DECODING_TIME].start();
        this->lm_engine->forward(think_end_id);
        this->profiler_list[DECODING_TIME].stop(1);
        token_str = this->tokenizer->run_time_decoder(think_end_id);

        this->token_history.push_back(271);
        this->profiler_list[DECODING_TIME].start();
        buffer<bf16> y = this->lm_engine->forward(271);
        this->profiler_list[DECODING_TIME].stop(1);
        token_str = this->tokenizer->run_time_decoder(271);
        sampled_token = this->_sample(y, meta_info);
    }
    this->total_tokens++;
    // the first sample after the preamble is the reply's first token, and counts toward the limit
    meta_info.generated_tokens++;
    int last_sampled_token = sampled_token;
    this->token_history.push_back(last_sampled_token);
    token_str = this->tokenizer->run_time_decoder(last_sampled_token);
    result += token_str;
    os << token_str << std::flush;

    if (this->is_eos(last_sampled_token)){
        this->lm_engine->forward(last_sampled_token);
    }
    else if (this->total_tokens + kEngineLead >= this->MAX_L){
        header_print("WARNING", "Max length reached, stopping generation...");
        reason = MAX_LENGTH_REACHED;
    }
    else if ((length_limit > 0) && (meta_info.generated_tokens >= length_limit)){
        reason = MAX_LENGTH_REACHED;
    }
    else while (true){
        // Running out of window is a truncation; testing here rather than in
        // the loop condition records it.
        if (this->total_tokens + kEngineLead >= this->MAX_L){
            header_print("WARNING", "Max length reached, stopping generation...");
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
            this->lm_engine->forward(last_sampled_token);
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
    if (reason != EOT_DETECTED) {
        // Cut short after sampling its last token: no engine has run that one,
        // so the next prompt has to carry it.
        this->token_history.pop_back();
    }

    std::cout << std::endl;
    header_print("FLM", "Model RAW Output: \n" + result);
    
    return result;
}

std::string Qwen3_5VL::generate_with_prompt(chat_meta_info_t& meta_info, lm_uniform_input_t& input, int length_limit, std::ostream& os) {
    if (!this->insert(meta_info, input)) {
        return "";
    }
    header_print("FLM", "Prompt inserted, starting generation...");
    int checkpoint_idx = this->lm_engine->checkpoint();
    int restore_idx = this->lm_engine->restore();
    header_print_r("FLM", "Checkpoint before generation: " << checkpoint_idx << ", restore point: " << restore_idx << ", user context length: " << this->token_history.size());
    if (this->enable_think) {
        os << "<think>\n" << std::flush;
    }
    return this->_shared_generate(meta_info, length_limit, os);
}


/************              Qwen3_5VL_Flash            **************/

void Qwen3_5VL_Flash::load_model(std::string model_path, json model_info, int default_context_length, bool enable_preemption, const std::string& backend) {
    Qwen3_5VL::load_model(model_path, model_info, default_context_length, enable_preemption, backend);
    // no tool calling in a flash model, whatever the catalog size says
    this->enable_tool = false;
    // single-turn: nothing to reuse between requests, and the state table the base wrapper attached belongs to the
    // other engine class (the flash engine's table is qwen3_5vl_flash_state_api)
    this->_prefix_attach(nullptr, nullptr, kImageSoftTokenId);
    this->_forget_pin();
}

int Qwen3_5VL_Flash::_pin_system_prefix(const std::string& system_text) {
    // Full clear: drop any previous pin before building the new one.
    this->lm_engine->clear_context();
    this->token_history.clear();
    this->checkpoint_his.clear();
    this->total_tokens = 0;
    this->last_token = -1;
    this->system_tokens = 0;
    this->system_his.clear();
    this->pinned_system_text = system_text;

    // debugging switch, like FLM_DISABLE_PREFIX_SLOTS: no pin, so a request prefills its whole prompt (A/B for the pin)
    static const bool pin_disabled = std::getenv("FLM_DISABLE_FLASH_PIN") != nullptr;
    if (system_text.empty() || pin_disabled) {
        return 0;
    }

    // Find how many tokens the system turn plus the opening of the user turn share across different user texts (the
    // variable part). Template two otherwise identical prompts that differ only in the first character of the user
    // message and take the longest common prefix.
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
        header_print("WARNING", "Qwen3_5VL_Flash: could not isolate a reusable system prefix, every turn will prefill in full");
        return 0;
    }

    std::vector<int> tokens(probe_a.begin(), probe_a.begin() + shared);
    chat_meta_info_t pin_meta;
    pin_meta.restore_allowed = false;
    pin_meta.max_prefill_len = this->MAX_L;
    bool pinned = false;
    try {
        // prefill only: nothing is sampled for a prefix
        pinned = this->_shared_insert(pin_meta, tokens, [] { return false; }, nullptr, 0, std::nullopt, false);
    } catch (const ModelRequestError&) {
        // a system prompt that does not fit the window is the request's problem, reported by its own prefill
        pinned = false;
    }
    if (!pinned) {
        this->lm_engine->clear_context();
        this->token_history.clear();
        this->total_tokens = 0;
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

    header_print("FLM", "Qwen3_5VL_Flash: system prefix pinned (" + std::to_string(this->system_tokens) + " tokens).");
    return this->system_tokens;
}

void Qwen3_5VL_Flash::_forget_pin() {
    this->pinned_system_text.clear();
    this->system_his.clear();
    this->system_tokens = 0;
}

void Qwen3_5VL_Flash::clear_context() {
    this->AutoModel::clear_context();
    this->_forget_pin();
}

void Qwen3_5VL_Flash::_reset_turn() {
    if (this->system_tokens > 0) {
        // Rewind to the pinned system prefix instead of clearing from scratch.
        this->total_tokens = static_cast<uint32_t>(this->lm_engine->restore());
        this->token_history = this->system_his;
        this->checkpoint_his = this->system_his;
    } else {
        this->lm_engine->clear_context();
        this->token_history.clear();
        this->checkpoint_his.clear();
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

bool Qwen3_5VL_Flash::insert(chat_meta_info_t& meta_info, lm_uniform_input_t& input, std::function<bool()> is_cancelled) {
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

    // Re-pin only when the system text actually changed (or is set for the first time).
    if (sys_text != this->pinned_system_text) {
        this->_pin_system_prefix(sys_text);
    }

    this->_reset_turn();

    constexpr int image_soft_token_id = 248056;
    qwen3_5vl_image_payload_t image_payload;
    std::vector<int> tokens;
    if (!this->_build_tokens(meta_info, input, image_payload, tokens)) {
        return false;
    }

    // generate() feeds the generation prompt's think tokens itself, so they are left out of the prefill
    const size_t trim_tokens = this->enable_think ? 2 : 4;
    if (tokens.size() <= trim_tokens) {
        header_print("WARNING", "Qwen3_5VL_Flash: the rendered prompt is shorter than its generation prompt");
        return false;
    }
    tokens.resize(tokens.size() - trim_tokens);

    // _shared_insert skips whatever the engine already holds (the pinned tokens) and clears the context, through
    // clear_context() and so the pin as well, when the prompt does not start with them. The image payload and the
    // image index have to be cut the same way: the pin never holds an image.
    const bool on_pin = this->system_tokens > 0 && tokens.size() > static_cast<size_t>(this->system_tokens) &&
        std::equal(this->system_his.begin(), this->system_his.end(), tokens.begin());
    const int skip = on_pin ? this->system_tokens : 0;
    int last_image_token_index = -1;
    for (int i = skip; i < static_cast<int>(tokens.size()); i++) {
        if (tokens[i] == image_soft_token_id) {
            last_image_token_index = i - skip;
        }
    }
    last_image_token_index++; // plus the end of image tokens

    const bool has_images = image_payload.num_images > 0;
    // no checkpoint() afterwards: the one checkpoint is the pinned system prefix
    return has_images
        ? this->_shared_insert(meta_info, tokens, is_cancelled, &image_payload, last_image_token_index)
        : this->_shared_insert(meta_info, tokens, is_cancelled, nullptr);
}

std::string Qwen3_5VL_Flash::generate_with_prompt(chat_meta_info_t& meta_info, lm_uniform_input_t& input, int length_limit, std::ostream& os) {
    if (!this->insert(meta_info, input)) {
        return "";
    }
    return this->generate(meta_info, length_limit, os);
}
