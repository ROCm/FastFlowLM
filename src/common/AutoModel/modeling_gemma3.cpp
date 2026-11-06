/// \file gemma3.cpp
/// \brief gemma3 class
/// \author FastFlowLM Team
/// \date 2025-09-03
/// \version 0.9.24
/// \note This is a source file for the gemma3 class

#include "AutoModel/modeling_gemma3.hpp"

/************              Gemma3 family            **************/
Gemma3::Gemma3(flm_rt::device* npu_device_inst) : AutoModel(npu_device_inst, "Gemma3") {}

void Gemma3::load_model(std::string model_path, json model_info, int default_context_length, bool enable_preemption, const std::string& backend) {
    this->_shared_load_backend(model_path, model_info, default_context_length, enable_preemption, backend);
    this->setup_tokenizer(model_path);
    this->sampler.reset();

    sampler_config config;
    config.top_k = 40;
    config.top_p = 0.95;
    config.min_p = 0.1;
    config.temperature = 0.8;
    config.rep_penalty = 1.05;
    config.freq_penalty = 1.05;

    this->set_sampler(config);
    for (size_t i = 0; i < PROFILER_TYPE_NUM; i++) {
        this->profiler_list[i].reset();
    }
    _prefix_attach(lm_engine, gemma_npu_state_api(lm_engine), 262144); // 262144 = the image placeholder id (IMAGE_TOKEN_ID in insert)
}

void Gemma3::setup_tokenizer(std::string model_path) {
    auto tokenizer_config = this->_shared_setup_tokenizer(model_path);
}

std::string Gemma3::apply_chat_template(nlohmann::ordered_json& messages, nlohmann::ordered_json tools, bool continue_final_message) {
    minja::chat_template_inputs inputs;
    inputs.messages = messages;
    inputs.extra_context = this->extra_context;
    return this->_render_chat_template(inputs, continue_final_message);
}

bool Gemma3::insert(chat_meta_info_t& meta_info, lm_uniform_input_t& input, std::function<bool()> is_cancelled) {
    static constexpr int IMAGE_TOKEN_ID = 262144; // replace with actual image token ID
    this->profiler_list[TKOEN_ENCODE_TIME].start();
    std::string templated_text;
    if (input.messages.empty() && input.prompt.empty()) {
        header_print("WARNING", "No messages or prompt provided");
        return false;
    }
    // On OGA the model runs text only. A ModelRequestError, not a payload the
    // engine rejects: the handlers turn only the former into an HTTP 400.
    if (this->backend_id() == flm::backend::kOgaHipepBackendId) {
        bool images = !input.images.empty();
        for (const auto& message : input.messages) {
            images = images || (message.contains("images") && !message["images"].empty());
        }
        if (images) {
            throw ModelRequestError(400, false,
                "the loaded backend '" + this->backend_id() + "' does not accept image input",
                "invalid_request_error");
        }
    }
    if (!input.messages.empty()) { // already a formated messages, usually from REST API
        // Templating happens below, together with image decoding, so that a
        // failed image never gets an image-token placeholder in the prompt.
    }
    else if (!input.prompt.empty()) { // a pure text, usually from the cli
        nlohmann::ordered_json messages;
        if (input.images.size() > 0) {
            nlohmann::ordered_json content;
            content["role"] = "user";
            content["content"] = input.prompt;
            content["images"] = nlohmann::ordered_json::array();
            for (int i = 0; i < input.images.size(); i++) {
                content["images"].push_back(input.images[i]);
            }
            messages.push_back(content);
        }
        else {
            messages.push_back({ {"role", "user"}, {"content", input.prompt} });
        }
        templated_text = this->apply_chat_template(messages);
    }
    // process all images

    bytes pixel_values;
    if (!input.messages.empty()) {
        // Decode/preprocess images up front and drop any that fail (e.g. bad
        // base64) from a filtered copy of the messages before templating, so
        // a failed image never gets an image-token placeholder; otherwise
        // pixel blocks (each a fixed 896x896 slot) would fall out of sync
        // with the placeholders and the prompt would still try to prefill a
        // nonexistent image.
        nlohmann::ordered_json filtered_messages = nlohmann::ordered_json::array();
        std::vector<bytes> valid_pixel_blocks;
        int total_images = 0;
        for (auto& message : input.messages) {
            nlohmann::ordered_json::array_t images = message.value("images", nlohmann::ordered_json::array());
            if (images.empty()) {
                filtered_messages.push_back(message);
                continue;
            }

            nlohmann::ordered_json filtered_message = message;
            filtered_message["images"] = nlohmann::ordered_json::array();
            for (auto& image : images) {
                std::string image_str = image.get<std::string>();
                bytes image_rgb = load_image_base64(image_str);
                if (image_rgb.size() == 0) {
                    header_print("ERROR", "Skipping invalid base64 image; prefilling language only for this item");
                    continue;
                }
                buffer<bf16> pv = preprocess_image(image_rgb);
                bytes pv_bytes(pv.size() * sizeof(bf16));
                memcpy(pv_bytes.data(), pv.data(), pv_bytes.size());
                valid_pixel_blocks.push_back(std::move(pv_bytes));
                filtered_message["images"].push_back(image);
                total_images++;
            }
            filtered_messages.push_back(filtered_message);
        }
        header_print("FLM", "Total images: " << total_images);
        templated_text = this->apply_chat_template(filtered_messages, nlohmann::ordered_json::object(), meta_info.continue_final_message);

        if (total_images > 0) {
            pixel_values.resize(3 * 896 * 896 * sizeof(bf16) * total_images);
            uint8_t* pixel_values_ptr = pixel_values.data();
            for (auto& block : valid_pixel_blocks) {
                memcpy(pixel_values_ptr, block.data(), block.size());
                pixel_values_ptr += block.size();
            }
        }
    }
    else { // from cli, typically only one image, typically a file path
        if (input.images.size() > 0){
            pixel_values.resize(3 * 896 * 896 * sizeof(bf16) * input.images.size());
            uint8_t* pixel_values_ptr = pixel_values.data();
            uint8_t* pixel_values_base = pixel_values.data();
            auto start_time = std::chrono::high_resolution_clock::now();
            for (auto& image : input.images){
                bytes image_rgb = load_image(image);
                if (image_rgb.size() == 0){
                    header_print("FLM", "Error: Could not load image: " << image);
                    header_print("FLM", "Please check if the file exists and is readable.");
                    continue;
                }
                
                buffer<bf16> pv = preprocess_image(image_rgb);
                if (pv.size() == 0){
                    header_print("FLM", "Error: Could not preprocess image: " << image);
                    header_print("FLM", "Please check if the image is valid.");
                    continue;
                }
                memcpy(pixel_values_ptr, pv.data(), pv.size() * sizeof(bf16));
                pixel_values_ptr += pv.size() * sizeof(bf16);
            }
            // Shrink pixel_values to what was actually written so that any
            // failed image loads do not leave uninitialized trailing data
            // (which would misalign pixels vs. image tokens during prefill).
            // `bytes::resize` reallocates without preserving content, so copy
            // into a fresh buffer instead.
            const size_t written = static_cast<size_t>(pixel_values_ptr - pixel_values_base);
            if (written < pixel_values.size()) {
                if (written == 0) {
                    pixel_values = bytes();
                } else {
                    bytes trimmed(written);
                    memcpy(trimmed.data(), pixel_values.data(), written);
                    pixel_values = std::move(trimmed);
                }
            }
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            header_print("FLM", "Image loaded in " << duration.count() << "ms");
        }
    }
    std::vector<int> tokens = this->tokenizer->encode(templated_text);

    // To avoid redownloading the model, a temporary fix, shall be removed in the next big update on gemma
    // if (tokens[tokens.size() - 1] != 107){
    //     tokens.push_back(107);
    // }
    
    this->profiler_list[TKOEN_ENCODE_TIME].stop(tokens.size());

    // ---- prefix slots: resume from the longest saved prefix of this prompt ----
    // The generation prompt ("<start_of_turn>model\n") is how the next turn renders this reply's header: the whole prompt is saved
    // (trim 0). The template folds the system prompt into the first user turn, so the shared head before "<start_of_turn>user" is a
    // BOS token only: no boundary slot, the end slots carry the reuse. After a resume token_history is the slot's tokens, so the
    // plan's skip below is the match length.
    const bool slot_mode = this->_prefix_slot_mode(meta_info, tokens.size(), 0);
    size_t match_len = 0;
    std::vector<int> full_tokens;          // what this prompt leaves in the engine
    std::vector<uint64_t> image_hashes;    // content of each of its images (a fixed 3*896*896 bf16 block each)
    // computed over the untrimmed pixels, for the slot cache and for the prompt-reuse check below
    const size_t block_bytes = static_cast<size_t>(3) * 896 * 896 * sizeof(bf16);
    for (size_t off = 0; off + block_bytes <= pixel_values.size(); off += block_bytes) {
        image_hashes.push_back(prefix_hash_bytes(pixel_values.data() + off, block_bytes, 0xcbf29ce484222325ull));
    }
    if (slot_mode) {
        full_tokens = tokens;
        match_len = this->_prefix_resume(this->prefix_cache_, full_tokens, image_hashes);
    } else {
        this->_prefix_legacy_path(); // the in-place path below moves the engine on without telling the cache
    }

    // ---- prompt cache: drop the images the engine already holds ----
    // pixel_values carries every image of the prompt, earlier turns' included. Each image is a fixed 896x896 block and
    // expands to the same number of placeholders (the chat template does the expansion), so the per-image count is the
    // placeholder total over the blocks; the last image takes any remainder so the items cover every placeholder.
    const size_t per_img_bytes = static_cast<size_t>(3) * 896 * 896 * sizeof(bf16);
    const size_t total_images_in_payload = pixel_values.size() / per_img_bytes;
    std::vector<prompt_media_t> media;
    if (total_images_in_payload > 0) {
        const size_t total_image_tokens = static_cast<size_t>(std::count(tokens.begin(), tokens.end(), IMAGE_TOKEN_ID));
        std::vector<size_t> counts(total_images_in_payload, total_image_tokens / total_images_in_payload);
        counts.back() += total_image_tokens % total_images_in_payload;
        prompt_media_from_counts(tokens, IMAGE_TOKEN_ID, counts, prompt_media_t::kImage, media);
        // so a different picture under the same placeholders is not taken for the one the engine holds (0 = not hashed)
        for (size_t i = 0; i < media.size() && i < image_hashes.size(); i++) {
            media[i].hash = image_hashes[i] != 0 ? image_hashes[i] : 1;
        }
    }
    // after the slot resume: it sees the token_history _shared_insert will see
    const prompt_reuse_t plan = this->_plan_reuse(tokens, media);
    const size_t drop_bytes = std::min(plan.skipped_media[prompt_media_t::kImage] * per_img_bytes, pixel_values.size());
    if (drop_bytes > 0) {
        const size_t remaining = pixel_values.size() - drop_bytes;
        if (remaining == 0) {
            pixel_values = bytes();
        } else {
            bytes trimmed(remaining);
            memcpy(trimmed.data(), pixel_values.data() + drop_bytes, remaining);
            pixel_values = std::move(trimmed);
        }
    }

    // hardware
    void* payload = pixel_values.size() > 0 ? static_cast<void*>(&pixel_values) : nullptr;

    // the image rows have to sit in the first prefill chunk: one past the last placeholder that is prefilled,
    // relative to the tokens that survive _shared_insert's skip
    int last_image_token_index = -1;
    for (int i = static_cast<int>(plan.skip); i < (int)tokens.size(); i++) {
        if (tokens[i] == IMAGE_TOKEN_ID) {
            last_image_token_index = i - static_cast<int>(plan.skip);
        }
    }
    last_image_token_index++;

    if (slot_mode) {
        return this->_prefix_insert(this->prefix_cache_, meta_info, tokens, full_tokens, image_hashes, match_len, "<start_of_turn>user",
                                    payload, last_image_token_index, is_cancelled);
    }
    return this->_shared_insert(meta_info, tokens, is_cancelled, payload, last_image_token_index);
}

// 106 eos
// 107 \n
std::string Gemma3::generate(chat_meta_info_t& meta_info, int length_limit, std::ostream& os, std::function<bool()> is_cancelled) {
    return this->_shared_generate(meta_info, length_limit, os, is_cancelled);
}

std::string Gemma3::generate_with_prompt(chat_meta_info_t& meta_info, lm_uniform_input_t& input, int length_limit, std::ostream& os) {
    if (!this->insert(meta_info, input)) {
        return "";
    }
    return this->_shared_generate(meta_info, length_limit, os);
}


