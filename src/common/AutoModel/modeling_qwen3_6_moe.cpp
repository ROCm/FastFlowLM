/// \file modeling_qwen3_6_moe.cpp
/// \brief Qwen3_6_MOE class
/// \author FastFlowLM Team
/// \date 2026-01-23
/// \version 0.9.28
/// \note This is a source file for the Qwen3_6_MOE class


#include "AutoModel/modeling_qwen3_6_moe.hpp"
#include "metrices.hpp"


/************              Qwen3_6_MOE family            **************/
Qwen3_6_MOE::Qwen3_6_MOE(flm_rt::device* npu_device_inst) : AutoModel(npu_device_inst, "Qwen3_6_MOE") {}

/************              prefix slots            **************/
namespace {
constexpr int kImageSoftTokenId = 248056;
} // namespace

void Qwen3_6_MOE::_attach_prefix_cache() {
    this->_prefix_attach(this->lm_engine, qwen3_6_moe_state_api(this->lm_engine), kImageSoftTokenId);
}

void Qwen3_6_MOE::load_model(std::string model_path, json model_info, int default_context_length, bool enable_preemption, const std::string& backend) {
    this->_shared_load_backend(model_path, model_info, default_context_length, enable_preemption, backend);
    this->setup_tokenizer(model_path);
    this->sampler.reset();

    this->enable_tool =true;

    sampler_config config;
    config.top_k = 20;
    config.top_p = 0.8;
    config.min_p = 0.0;
    config.temperature = 0.7;
    config.rep_penalty = 1.0;
    config.freq_penalty = 0.0;
    config.pre_penalty = 1.5f;

    this->set_sampler(config);
    for (size_t i = 0; i < PROFILER_TYPE_NUM; i++) {
        this->profiler_list[i].reset();
    }
    this->_attach_prefix_cache();
}

void Qwen3_6_MOE::setup_tokenizer(std::string model_path) {
    auto tokenizer_config = this->_shared_setup_tokenizer(model_path);
}

std::string Qwen3_6_MOE::apply_chat_template(nlohmann::ordered_json& messages, nlohmann::ordered_json tools, bool continue_final_message) {
    minja::chat_template_inputs inputs;
    inputs.messages = messages;
    inputs.extra_context = this->extra_context;
    inputs.extra_context["enable_thinking"] = this->enable_think;
    if (!tools.empty() && this->enable_tool)
        inputs.tools = tools;
    return this->_render_chat_template(inputs, continue_final_message);
}

bool Qwen3_6_MOE::insert(chat_meta_info_t& meta_info, lm_uniform_input_t& input, std::function<bool()> is_cancelled) {
    // preprocess
    constexpr int image_soft_token_id = 248056;
    this->profiler_list[TKOEN_ENCODE_TIME].start();
    std::string templated_text;
    if (input.messages.empty() && input.prompt.empty()) {
        header_print("WARNING", "No messages or prompt provided");
        return false;
    }
    if (this->backend_id() == flm::backend::kCoreBackendId) {
        bool has_image = !input.images.empty();
        for (const auto& item : input.messages) {
            if (item.contains("images") && !item["images"].empty()) has_image = true;
        }
        if (has_image)
            throw ModelRequestError(400, false,
                "Qwen3.6 on backend 'core' runs text only; this request has images");
    }

    constexpr bool DEBUG_IMAGE_PREPROCESS = false;
    qwen3_6_moe_image_payload_t image_payload;
    image_payload.num_images = 0;
    if (input.images.size() > 0) {


        // header_print("info", "Processing images...");

        // time_utils::time_point preprocess_start = time_utils::now();
        for(const auto& img_str : input.images){
            qwen3_6_moe_image_t image = this->load_image(img_str);
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
                qwen3_6_moe_image_t image = this->load_image_base64(img_str);
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
    std::vector<int> tokens;
    int total_image_tokens = 0;
    // Use image_payload.images.size() (not input.images.size()), because on
    // the REST API path images come from `messages` and input.images is empty.
    for (size_t i = 0; i < image_payload.images.size(); i++) {
        total_image_tokens += image_payload.images[i].grid_h * image_payload.images[i].grid_w;
    }
    tokens.reserve(tokens_init.size() + total_image_tokens);
    // placeholders of each image, recorded as expanded: the prompt-cache plan below locates the images by them
    std::vector<size_t> image_token_counts;
    int image_counter = 0;
    for (int i = 0; i < tokens_init.size(); i++) {
        if (tokens_init[i] == image_soft_token_id) {
            // one token per merge x merge patches; the same config value smart_resize rounds the grid to
            const int merge = static_cast<int>(reinterpret_cast<qwen3_6_moe_npu*>(this->lm_engine)->QWEN3_6_MOE_IMAGE_MERGE_SIZE);
            const int n_image_tokens = image_payload.images[image_counter].grid_h * image_payload.images[image_counter].grid_w / (merge * merge);
            for (int j = 0; j < n_image_tokens; j++) {
                tokens.push_back(image_soft_token_id);
            }
            image_token_counts.push_back(static_cast<size_t>(n_image_tokens));
            image_counter++;
        } else {
            tokens.push_back(tokens_init[i]);
        }
    }

    this->profiler_list[TKOEN_ENCODE_TIME].stop(tokens.size());

    // ---- prefix slots: resume from the longest saved prefix of this prompt ----
    // a continued message keeps all its tokens: they are the client's text, not the generation prompt
    const size_t trim_tokens = meta_info.continue_final_message ? 0 : (this->enable_think ? 2 : 4);
    qwen3_6_moe_npu* slot_eng = dynamic_cast<qwen3_6_moe_npu*>(this->lm_engine);
    const bool slot_mode = this->prefix_cache_.enabled() && slot_eng != nullptr && meta_info.restore_allowed && tokens.size() > trim_tokens;
    size_t match_len = 0;
    std::vector<int> full_tokens;          // what this prompt leaves in the engine
    std::vector<uint64_t> image_hashes;    // content of each of its images
    if (!image_payload.images.empty() && slot_eng != nullptr) {
        // over the full payload, before the prompt cache drops any image; shared by the slot cache and _plan_reuse
        image_hashes = prefix_image_hashes(image_payload, slot_eng->QWEN3_6_MOE_PATCH_SIZE, slot_eng->QWEN3_6_MOE_TEMPORAL_PATCH_SIZE);
    }
    if (this->prefix_cache_.enabled() && !slot_mode) {
        // the single-checkpoint path below moves the engine on without telling the cache
        this->prefix_cache_.forget_all();
    }
    if (slot_mode) {
        full_tokens.assign(tokens.begin(), tokens.end() - trim_tokens);
        match_len = this->_prefix_resume(this->prefix_cache_, full_tokens, image_hashes);
    }

    // hardware
    int restore_idx = -1;
    if (!slot_mode && meta_info.restore_allowed) {
        restore_idx = this->lm_engine->restore();
        this->total_tokens = restore_idx;
        this->_restore_history(); // restore the token history to be consistent with the restored KV cache, which is crucial for correct functioning of _shared_insert's prefix-matching logic
    }

    size_t n = tokens.size();
    tokens.resize(n - trim_tokens);

    // ---- prompt cache: drop the images the engine already holds ----
    // The payload carries the pixels of every image of the prompt, but _shared_insert only prefills the
    // tokens after the part the engine keeps. Ask the shared rule, against the history and the exact
    // tokens _shared_insert will see, how much that is, and drop the images inside it.
    std::vector<prompt_media_t> media;
    prompt_media_from_counts(tokens, image_soft_token_id, image_token_counts, prompt_media_t::kImage, media);
    // media items are the images in payload order; 0 means "not hashed", so never use it as a hash
    for (size_t i = 0; i < media.size() && i < image_hashes.size(); i++) {
        media[i].hash = image_hashes[i] != 0 ? image_hashes[i] : 1;
    }
    const prompt_reuse_t plan = this->_plan_reuse(tokens, media);

    const size_t images_to_drop = std::min(plan.skipped_media[prompt_media_t::kImage], image_payload.images.size());
    if (images_to_drop > 0) {
        // Per-image bf16 footprint depends on runtime patch/temporal
        // config carried by the engine.
        auto* eng = reinterpret_cast<qwen3_6_moe_npu*>(this->lm_engine);
        const unsigned patch_size = eng->QWEN3_6_MOE_PATCH_SIZE;
        const unsigned temporal_patch = eng->QWEN3_6_MOE_TEMPORAL_PATCH_SIZE;

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

    // one past the last image token, relative to the tokens that survive _shared_insert's skip
    const int last_image_token_index = static_cast<int>(plan.media_end);
    const bool has_images = image_payload.num_images > 0;

    if (slot_mode) {
        return this->_prefix_insert(this->prefix_cache_, meta_info, tokens, full_tokens, image_hashes, match_len, "<|im_start|>user",
                                    has_images ? static_cast<void*>(&image_payload) : nullptr, last_image_token_index, is_cancelled);
    }

    bool success = has_images
        ? this->_shared_insert(meta_info, tokens, is_cancelled, &image_payload, last_image_token_index)
        : this->_shared_insert(meta_info, tokens, is_cancelled, nullptr);

    this->_checkpoint_history();
    int checkpoint_idx = this->lm_engine->checkpoint();
    return success;
}

std::string Qwen3_6_MOE::generate(chat_meta_info_t& meta_info, int length_limit, std::ostream& os, std::function<bool()> is_cancelled) {
    if (meta_info.continue_final_message) {
        // insert() trimmed nothing: no think opening to feed, the output starts from insert()'s sample
        return this->_shared_generate(meta_info, length_limit, os, is_cancelled);
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

    this->profiler_list[DECODING_TIME].reset();
    this->profiler_list[TKOEN_DECODE_TIME].reset();
    std::string token_str;
    int sampled_token;
    // Some backends refuse to decode past a limit of their own, below MAX_L.
    const uint32_t decode_cap = this->decode_cap();
    // insert() trimmed the opening of the think block; it is decoded here, and
    // every one of those steps occupies the context.
    const uint32_t forced = this->enable_think ? 2 : 4;
    if (this->total_tokens + forced >= decode_cap) {
        header_print("WARNING", "Max length reached, stopping generation...");
        meta_info.stop_reason = MAX_LENGTH_REACHED;
        return result;
    }
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
    this->total_tokens += forced;
    // the first sample after the preamble is the reply's first token, and counts toward the limit
    meta_info.generated_tokens++;
    int last_sampled_token = sampled_token;
    this->token_history.push_back(last_sampled_token);
    token_str = this->tokenizer->run_time_decoder(last_sampled_token);
    result += token_str;
    os << token_str << std::flush;

    if (this->is_eos(last_sampled_token)){
        if (this->forward_on_eos &&
            (!this->backend_ || this->backend_->forwards_past_eos())) {
            this->lm_engine->forward(last_sampled_token);
        } else {
            // The engine never ran it, so the next prompt has to carry it.
            this->token_history.pop_back();
        }
    }
    else if ((length_limit > 0) && (meta_info.generated_tokens >= length_limit)){
        reason = MAX_LENGTH_REACHED;
    }
    else while (true){
        // Running out of window is a truncation, not the model choosing to
        // stop; testing here rather than in the loop condition records it.
        if (this->total_tokens >= decode_cap){
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
            // The forward writes row total_tokens, which is past the end of the
            // engine's KV window once the context is full.
            if (this->forward_on_eos && this->total_tokens < decode_cap &&
                (!this->backend_ || this->backend_->forwards_past_eos())) {
                this->lm_engine->forward(last_sampled_token);
            } else {
                // The engine never ran it, so the next prompt has to carry it.
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
    if (reason != EOT_DETECTED) {
        // Cut short after sampling its last token: no engine has run that one,
        // so the next prompt has to carry it.
        this->token_history.pop_back();
    }
    if (this->total_tokens >= decode_cap){
        header_print("WARNING", "Max length reached, stopping generation...");
    }

    std::cout << std::endl;
    header_print("FLM", "Model RAW Output: \n" + result);

    return result;
}

std::string Qwen3_6_MOE::generate_with_prompt(chat_meta_info_t& meta_info, lm_uniform_input_t& input, int length_limit, std::ostream& os) {
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
