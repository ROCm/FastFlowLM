/// \file deepseek.cpp
/// \brief deepseek class
/// \author FastFlowLM Team
/// \date 2025-09-01
/// \version 0.9.24
/// \note This is a source file for the deepseek class


#include "AutoModel/modeling_qwen2vl.hpp"
#include "metrices.hpp"
#include <string_view>


/************              Qwen2VL family            **************/
Qwen2VL::Qwen2VL(flm_rt::device* npu_device_inst) : AutoModel(npu_device_inst, "Qwen2VL") {}

void Qwen2VL::load_model(std::string model_path, json model_info, int default_context_length, bool enable_preemption, const std::string& backend) {
    this->_shared_load_backend(model_path, model_info, default_context_length, enable_preemption, backend);
    this->setup_tokenizer(model_path);
    this->sampler.reset();

    sampler_config config;
    config.rep_penalty = 1.05;
    config.temperature = 0.6;
    config.top_p = 0.8;
    config.top_k = 10;
    config.rep_penalty_window = 1024;
    config.freq_penalty = 1.05;
    config.freq_penalty_window = 1024;
    this->set_sampler(config);
    for (size_t i = 0; i < PROFILER_TYPE_NUM; i++) {
        this->profiler_list[i].reset();
    }
    this->_prefix_attach(this->lm_engine, qwen2vl_npu_state_api(this->lm_engine), 151655);
}

void Qwen2VL::setup_tokenizer(std::string model_path) {
    auto tokenizer_config = this->_shared_setup_tokenizer(model_path);
}

std::string Qwen2VL::apply_chat_template(nlohmann::ordered_json& messages, nlohmann::ordered_json tools, bool continue_final_message) {
    minja::chat_template_inputs inputs;
    inputs.messages = messages;
    inputs.extra_context = this->extra_context;
    return this->_render_chat_template(inputs, continue_final_message);
}

bool Qwen2VL::insert(chat_meta_info_t& meta_info, lm_uniform_input_t& input, std::function<bool()> is_cancelled) {
    // preprocess
    constexpr int image_soft_token_id = 151655;
    this->profiler_list[TKOEN_ENCODE_TIME].start();
    std::string templated_text;
    if (input.messages.empty() && input.prompt.empty()) {
        header_print_r("WARNING", "No messages or prompt provided");
        return false;
    }

    constexpr bool DEBUG_IMAGE_PREPROCESS = false;
    qwen2vl_image_payload_t image_payload;
    image_payload.num_images = 0;
    if (input.images.size() > 0) {


        // header_print("info", "Processing images...");
        
        // time_utils::time_point preprocess_start = time_utils::now();
        for(const auto& img_str : input.images){
            qwen2vl_image_t image = this->load_image(img_str);
            if (image.width <= 0 || image.height <= 0) {
                header_print_r("ERROR", "Skipping image that failed to load: " + img_str);
                continue;
            }

            preprocess_image(image, image_payload._data__processed);
            if (image.grid_h <= 0 || image.grid_w <= 0) {
                header_print_r("ERROR", "Skipping image that failed to preprocess: " + img_str);
                continue;
            }

            // Push the image AFTER preprocessing so grid_h and grid_w are set
            image_payload.images.push_back(image);
            image_payload.num_images++;
        }
    }
    try {
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
                    if (!img.is_string() || img.get_ref<const std::string&>().empty()) {
                        continue;
                    }
                    const std::string& img_str = img.get_ref<const std::string&>();

                    // Decode/preprocess up front so an invalid image (e.g.
                    // bad base64) never gets an image-soft-token placeholder
                    // in the template below; otherwise image_payload.images
                    // would fall out of sync with the placeholders and the
                    // prompt would still try to prefill a nonexistent image.
                    qwen2vl_image_t image = this->load_image_base64(img_str);
                    if (image.width <= 0 || image.height <= 0) {
                        header_print_r("ERROR", "Skipping invalid base64 image; prefilling language only for this item");
                        continue;
                    }

                    preprocess_image(image, image_payload._data__processed);
                    if (image.grid_h <= 0 || image.grid_w <= 0) {
                        header_print_r("ERROR", "Skipping image that failed to preprocess");
                        continue;
                    }

                    image_payload.images.push_back(image);
                    image_payload.num_images++;
                    total_images++;

                    newContent.push_back({
                        {"type", "image"}
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
            templated_text = this->apply_chat_template(qwenvl_message, nlohmann::ordered_json::object(), meta_info.continue_final_message);
            header_print_g("FLM", "Total images: " + std::to_string(total_images));
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
    } catch (const ModelRequestError&) {
        throw; // a final message that cannot be continued is the client's error (400), not a failed prefill
    } catch (const std::exception& e) {
        header_print_r("ERROR", std::string("Exception during chat template application: ") + e.what());
        meta_info.error_message = std::string("Applying the chat template failed: ") + e.what();
        return false;
    }
    std::vector<int> tokens_init = this->tokenizer->encode(templated_text);

    // update the tokens to include the image tokens
    std::vector<int> tokens;
    size_t total_image_tokens = 0;
    // Use image_payload.images.size() (not input.images.size()), because on
    // the REST API path images come from `messages` and input.images is empty.
    // Each image is one token per merged patch; the prompt-cache plan below reads the same counts.
    std::vector<size_t> image_token_counts;
    for (size_t i = 0; i < image_payload.images.size(); i++) {
        image_token_counts.push_back(static_cast<size_t>(image_payload.images[i].grid_h * image_payload.images[i].grid_w) /
                                     (QWEN2_MERGE_SIZE * QWEN2_MERGE_SIZE));
        total_image_tokens += image_token_counts.back();
    }
    tokens.reserve(tokens_init.size() + total_image_tokens);
    int image_counter = 0;
    for (int i = 0; i < tokens_init.size(); i++) {
        if (tokens_init[i] == image_soft_token_id) {
            for (size_t j = 0; j < image_token_counts[image_counter]; j++) {
                tokens.push_back(image_soft_token_id);
            }
            image_counter++;
        } else {
            tokens.push_back(tokens_init[i]);
        }
    }

    this->profiler_list[TKOEN_ENCODE_TIME].stop(tokens.size());

    // ---- prefix slots: resume from the longest saved prefix of this prompt ----
    // The generation prompt ("<|im_start|>assistant\n") is how the next turn renders this reply's header: the whole prompt
    // is saved (trim 0). After a resume token_history is the slot's tokens, so the skip count below is the match length.
    const bool slot_mode = this->_prefix_slot_mode(meta_info, tokens.size(), 0);
    size_t match_len = 0;
    std::vector<int> full_tokens;          // what this prompt leaves in the engine
    std::vector<uint64_t> image_hashes;    // content of each of its images
    if (slot_mode) {
        full_tokens = tokens;
        image_hashes = prefix_image_hashes(image_payload, QWEN2_PATCH_SIZE, QWEN2_TEMPORAL_PATCH_SIZE);
        match_len = this->_prefix_resume(this->prefix_cache_, full_tokens, image_hashes);
    } else {
        this->_prefix_legacy_path(); // the in-place path below moves the engine on without telling the cache
    }

    // Pixels cover every image of the prompt, but the engine keeps the part of it it already holds (after the slot
    // resume, or the live history): drop the images inside that part. _shared_insert asks the same plan, so it skips
    // exactly the tokens these images sit in.
    std::vector<prompt_media_t> media;
    prompt_media_from_counts(tokens, image_soft_token_id, image_token_counts, prompt_media_t::kImage, media);
    // Same-size pictures render the same tokens: the plan compares these content hashes (taken before the drop) so
    // another picture is never matched to the rows of the old one. The slot path's hashes are the same values.
    if (!media.empty()) {
        if (image_hashes.empty()) {
            image_hashes = prefix_image_hashes(image_payload, QWEN2_PATCH_SIZE, QWEN2_TEMPORAL_PATCH_SIZE);
        }
        for (size_t i = 0; i < media.size() && i < image_hashes.size(); i++) {
            media[i].hash = image_hashes[i] != 0 ? image_hashes[i] : 1; // 0 means "not hashed"
        }
    }
    const prompt_reuse_t plan = this->_plan_reuse(tokens, media);
    const size_t images_to_drop = std::min(plan.skipped_media[prompt_media_t::kImage], image_payload.images.size());
    if (images_to_drop > 0) {
        size_t bf16_to_drop = 0;
        for (size_t i = 0; i < images_to_drop; i++) {
            const auto& img = image_payload.images[i];
            bf16_to_drop += static_cast<size_t>(img.grid_h) * QWEN2_PATCH_SIZE *
                            static_cast<size_t>(img.grid_w) * QWEN2_PATCH_SIZE *
                            3u * QWEN2_TEMPORAL_PATCH_SIZE;
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

    // one past the last image token, relative to the tokens that survive _shared_insert's skip:
    // every remaining image goes into the first prefill chunk
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

std::string Qwen2VL::generate(chat_meta_info_t& meta_info, int length_limit, std::ostream& os, std::function<bool()> is_cancelled) {
    return this->_shared_generate(meta_info, length_limit, os, is_cancelled);
}

std::string Qwen2VL::generate_with_prompt(chat_meta_info_t& meta_info, lm_uniform_input_t& input, int length_limit, std::ostream& os) {
    if (!this->insert(meta_info, input)) {
        return "";
    }
    return this->_shared_generate(meta_info, length_limit, os);
}
