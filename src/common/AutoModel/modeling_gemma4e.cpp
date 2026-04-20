/// \file modeling_gemma4e.cpp
/// \brief Gemma4e class
/// \author FastFlowLM Team
/// \date 2025-09-01
/// \version 0.9.24
/// \note This is a source file for the Gemma4e class


#include "AutoModel/modeling_gemma4e.hpp"
#include "metrices.hpp"
#include "error_measure.hpp" //TODO: FIXME: remove later


/************              Gemma4e family            **************/
Gemma4e::Gemma4e(xrt::device* npu_device_inst) : AutoModel(npu_device_inst, "Gemma4e") {}

void Gemma4e::load_model(std::string model_path, json model_info, int default_context_length, bool enable_preemption) {
    
    this->_shared_load_model(model_path, model_info, default_context_length, enable_preemption);
    
    this->q4nx = std::make_unique<Q4NX>(this->model_path);
    this->lm_engine = std::make_unique<gemma4e_npu>(*this->lm_config, this->npu.get(), this->MAX_L);

    this->lm_engine->load_weights(*this->q4nx);
    //free the q4nx
    this->q4nx.reset();
    this->lm_engine->clear_context();
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
}

void Gemma4e::setup_tokenizer(std::string model_path) {
    auto tokenizer_config = this->_shared_setup_tokenizer(model_path);
}

std::string Gemma4e::apply_chat_template(nlohmann::ordered_json& messages, nlohmann::ordered_json tools) {
    minja::chat_template_inputs inputs;
    inputs.add_generation_prompt = true;
    inputs.messages = messages;
    inputs.extra_context = this->extra_context;
    inputs.extra_context["enable_thinking"] = this->enable_think;
    if (!tools.empty())
        inputs.tools = tools;
    return this->chat_tmpl->apply(inputs);
}

bool Gemma4e::insert(chat_meta_info_t& meta_info, lm_uniform_input_t& input) {
    this->profiler_list[TKOEN_ENCODE_TIME].start();
    std::string templated_text;
    
    if (input.messages.empty() && input.prompt.empty()) {
        header_print("WARNING", "No messages or prompt provided");
        return false;
    }

    constexpr bool DEBUG_IMAGE_PREPROCESS = false;
    gemma4e_image_payload_t image_payload;
    gemma4e_audio_payload_t audio_payload;
    audio_payload.num_audios = 0;
    image_payload.num_images = 0;
    
    float max_support_audio_length_seconds = 30.0f;
    int total_audio_clips = 0;
    std::vector<audio_data_t> audio_data_list;

    if (!input.messages.empty()) { // Server Processing
        int total_images = 0;
        for (auto& message : input.messages) {
            // Process Images
            if (message.contains("images")) {
                for (auto& img : message["images"]) {
                    std::string img_str = img.get<std::string>();
                    if (!img_str.empty()) total_images++;
                    
                    gemma4e_image_t image = this->load_image_base64(img_str);
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
            // Process Audios
            if (message.contains("audios")) {
                gemma4e_npu *gemma4e_engine = dynamic_cast<gemma4e_npu*>(this->lm_engine.get());
                for (auto& aud : message["audios"]) {
                    std::string audio_str = aud.get<std::string>();
                    audio_data_t audio_data = this->load_audio_base64(audio_str, gemma4e_engine->Gemma4E_Audio_resample_rate, MonoDownmixMode::MEAN);
                    if (audio_data.channels > 1) {
                        std::cerr << "only mono audio is supported." << std::endl;
                        exit(-1);
                    }
                    std::vector<audio_data_t> clipped_audio_data = this->clip_audio_length(audio_data, max_support_audio_length_seconds);
                    audio_data_list.insert(audio_data_list.end(), clipped_audio_data.begin(), clipped_audio_data.end());
                    total_audio_clips += clipped_audio_data.size();
                    if (clipped_audio_data.size() > 1) {
                        header_print_g("FLM", "Audio in message is clipped to " + std::to_string(clipped_audio_data.size()) + " chunks.");
                        std::cout << std::endl;
                    }
                }
            }
        }
        header_print("FLM", "Total images: " << total_images);
    }
    else { // CLI Processing
        if (input.audios.size() > 0) {
            gemma4e_npu *gemma4e_engine = dynamic_cast<gemma4e_npu*>(this->lm_engine.get());
            for (int i = 0; i < input.audios.size(); i++) {
                std::string audio_str = input.audios[i];
                audio_data_t audio_data = this->load_audio(audio_str, gemma4e_engine->Gemma4E_Audio_resample_rate, MonoDownmixMode::MEAN); 
                
                if (audio_data.channels > 1) {
                    std::cerr << "only mono audio is supported, but got " << audio_data.original_channels << " channels. Please convert it to mono first." << std::endl;
                    exit(-1);
                }

                // apply clipping
                std::vector<audio_data_t> clipped_audio_data = this->clip_audio_length(audio_data, max_support_audio_length_seconds);
                audio_data_list.insert(audio_data_list.end(), clipped_audio_data.begin(), clipped_audio_data.end());
                total_audio_clips += clipped_audio_data.size();
                
                if (clipped_audio_data.size() > 1) {
                    header_print_g("FLM", "Audio[" + std::to_string(i) + "] is clipped to " + std::to_string(clipped_audio_data.size()) + " chunks.");
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

        gemma4e_npu *gemma4e_engine = dynamic_cast<gemma4e_npu*>(this->lm_engine.get());
        const unsigned int conv2d_kernel = gemma4e_engine->Gemma4E_Audio_conv2d_kernel_size;
        const unsigned int conv2d_stride = gemma4e_engine->Gemma4E_Audio_conv2d_Stride;
        const unsigned int conv2d_padding = gemma4e_engine->Gemma4e_Audio_conv2d_Padding;
        const unsigned int max_audio_seq_length = max_support_audio_length_seconds * gemma4e_engine->Gemma4E_Audio_resample_rate;    
        
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
        nlohmann::ordered_json gemma4_message = nlohmann::ordered_json::array();
        for (const auto& item : input.messages) {
            if (!item.contains("images") && !item.contains("audios")) {
                gemma4_message.push_back(item);
                continue;
            }

            nlohmann::ordered_json newContent = nlohmann::ordered_json::array();
            if (item.contains("images")) {
                for (const auto& img : item["images"]) {
                    newContent.push_back({{"type", "image"}, {"image", img}});
                }
            }
            if (item.contains("audios")) {
                for (const auto& aud : item["audios"]) {
                    newContent.push_back({{"type", "audio"}, {"audio", aud}});
                }
            }
            newContent.push_back({{"type", "text"}, {"text", item.value("content", "")}});

            nlohmann::ordered_json newItem = {
                {"role", item.value("role", "user")},
                {"content", newContent}
            };
            gemma4_message.push_back(newItem);
        }
        templated_text = this->apply_chat_template(gemma4_message, input.tools);
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
   
    for (int i = 0; i < tokens_init.size(); i++) {
        if (tokens_init[i] == image_token_id) {
            tokens.push_back(boi_token_id); // the first image soft token id, which is reserved for the model to identify the image position, the rest of the soft tokens for this image will be continuous following this id
            for (int j = 0; j < image_payload.num_soft_tokens_per_image[image_counter]; j++) {
                tokens.push_back(image_token_id);
            }
            tokens.push_back(eoi_token_id); // a separator token between images, not necessary but can help the model to better distinguish different images
            image_counter++;
        } 
        else if (tokens_init[i] == audio_token_id){
            tokens.push_back(boa_token_id); // the first audio soft token id, which is reserved for the model to identify the audio position, the rest of the soft tokens for this audio will be continuous following this id
            for (int j = 0; j < audio_payload.num_soft_tokens_per_audio[audio_counter]; j++) {
                tokens.push_back(audio_token_id);
            }
            tokens.push_back(eoa_token_id); // a separator token between audios, not necessary but can help the model to better distinguish different audios
            audio_counter++;
        }
        else {
            tokens.push_back(tokens_init[i]);
        }
    }
    assert(image_counter == image_payload.num_images);
    assert(audio_counter == audio_payload.num_audios);
      
    this->profiler_list[TKOEN_ENCODE_TIME].stop(tokens.size());

    // hardware
    gemma4e_multi_modal_payload_t multi_modal_payload;
    multi_modal_payload.image_payload = image_payload;
    multi_modal_payload.audio_payload = audio_payload;

    if (image_payload.num_images > 0 || audio_payload.num_audios > 0) {
        return this->_shared_insert(meta_info, tokens, &multi_modal_payload);
    } 
    else {
        return this->_shared_insert(meta_info, tokens, nullptr);
    }
}

std::string Gemma4e::generate(chat_meta_info_t& meta_info, int length_limit, std::ostream& os, std::function<bool()> is_cancelled) {
    return this->_shared_generate(meta_info, length_limit, os, is_cancelled);
}

std::string Gemma4e::generate_with_prompt(chat_meta_info_t& meta_info, lm_uniform_input_t& input, int length_limit, std::ostream& os) {
    if (!this->insert(meta_info, input)) {
        return "";
    }
    if (this->enable_think) {
        os << "<think>\n" << std::flush;
    }
    return this->_shared_generate(meta_info, length_limit, os);
}

// Non-stream
NonStreamResult Gemma4e::parse_nstream_content(const std::string response_text) {
    NonStreamResult result;

    std::string think_start_tag = "<|channel>thought";
    std::string think_end_tag = "<channel|>";
    std::string tool_start_tag = "<|tool_call>";
    std::string tool_end_tag = "<tool_call|>";
    std::string tool_resp_tag = "<|tool_response>";
    std::string custom_quote_tag = "<|\"|>";

    size_t think_start_pos = response_text.find(think_start_tag);
    size_t think_end_pos = response_text.find(think_end_tag);
    size_t tool_start_pos = response_text.find(tool_start_tag);
    size_t tool_end_pos = response_text.find(tool_end_tag);

    bool is_reasoning = (think_start_pos != std::string::npos && think_end_pos != std::string::npos && think_end_pos > think_start_pos);
    bool is_tool = (tool_start_pos != std::string::npos && tool_end_pos != std::string::npos && tool_end_pos > tool_start_pos);

    // 1. Parse Reasoning Content
    if (is_reasoning) {
        size_t start = think_start_pos + think_start_tag.length();
        result.reasoning_content = response_text.substr(start, think_end_pos - start);
    }

    // 2. Parse Tool Calling
    if (is_tool) {
        size_t start = tool_start_pos + tool_start_tag.length();
        std::string tool_content = response_text.substr(start, tool_end_pos - start);

        // Remove the "call:" prefix if it exists
        std::string prefix = "call:";
        if (tool_content.find(prefix) == 0) {
            tool_content = tool_content.substr(prefix.length());
        }

        // Split into Name and Arguments
        size_t brace_pos = tool_content.find('{');
        if (brace_pos != std::string::npos) {
            result.tool_name = tool_content.substr(0, brace_pos);
            
            // Keep the {} brackets for the arguments
            std::string args_str = tool_content.substr(brace_pos);

            // Convert custom quote tags <|"|> to standard double quotes "
            size_t quote_pos = 0;
            while ((quote_pos = args_str.find(custom_quote_tag, quote_pos)) != std::string::npos) {
                args_str.replace(quote_pos, custom_quote_tag.length(), "\"");
                quote_pos += 1;
            }

            // Wrap unquoted keys in double quotes to create valid JSON
            // e.g., {query:"ai"} -> {"query":"ai"}
            std::regex key_regex("([{,])\\s*([a-zA-Z0-9_]+)\\s*:");
            args_str = std::regex_replace(args_str, key_regex, "$1\"$2\":");

            result.tool_args = args_str;
        } else {
            // Fallback if no arguments were provided
            result.tool_name = tool_content;
            result.tool_args = "{}";
        }
    }
    // 3. Parse Normal Content
    else {
        if (is_reasoning) {
            // Content is whatever comes AFTER the reasoning block
            result.content = response_text.substr(think_end_pos + think_end_tag.length());
        } else {
            // No reasoning, no tools -> the whole text is content
            result.content = response_text;
        }

        // Cleanup: Strip out <|tool_response> if the model accidentally hallucinated it into plain text
        size_t resp_pos = 0;
        while ((resp_pos = result.content.find(tool_resp_tag, resp_pos)) != std::string::npos) {
            result.content.erase(resp_pos, tool_resp_tag.length());
        }
    }

    return result;
}


// Stream
StreamResult Gemma4e::parse_stream_content(const std::string content) {
    const std::string MARKER_THINK_START = "<|channel>thought";
    const std::string MARKER_THINK_END = "<channel|>";
    const std::string MARKER_TOOL_START = "<|tool_call>";
    const std::string MARKER_TOOL_END = "<tool_call|>";
    const std::string MARKER_TOOL_RESP = "<|tool_response>";
    const std::string MARKER_CUSTOM_QUOTE = "<|\"|>"; 

    StreamResult result;
    buffer_ += content;

    while (true) {
        if (is_in_tool_block_) {
            size_t tool_end_pos = buffer_.find(MARKER_TOOL_END);
            
            if (tool_end_pos != std::string::npos) {
                std::string tool_content = buffer_.substr(0, tool_end_pos);
                buffer_ = buffer_.substr(tool_end_pos + MARKER_TOOL_END.length());
                is_in_tool_block_ = false;

                result.type = StreamEventType::TOOL_DONE;
                
                static int tool_counter = 0;
                result.tool_id = "call_" + std::to_string(std::time(nullptr)) + "_" + std::to_string(tool_counter++);

                std::string prefix = "call:";
                if (tool_content.find(prefix) == 0) {
                    tool_content = tool_content.substr(prefix.length());
                }

                size_t brace_pos = tool_content.find('{');
                if (brace_pos != std::string::npos) {
                    result.tool_name = tool_content.substr(0, brace_pos);
                    
                    std::string args_str = tool_content.substr(brace_pos);
                    
                    // Convert custom quote tags to standard double quotes
                    size_t quote_pos = 0;
                    while ((quote_pos = args_str.find(MARKER_CUSTOM_QUOTE, quote_pos)) != std::string::npos) {
                        args_str.replace(quote_pos, MARKER_CUSTOM_QUOTE.length(), "\"");
                        quote_pos += 1; 
                    }

                    // Wrap unquoted keys in double quotes
                    // Because we kept the '{', this regex will now correctly trigger for the FIRST key too
                    std::regex key_regex("([{,])\\s*([a-zA-Z0-9_]+)\\s*:");
                    args_str = std::regex_replace(args_str, key_regex, "$1\"$2\":");
                    
                    result.tool_args_str = args_str;
                } 
                else {
                    result.tool_name = tool_content;
                    result.tool_args_str = "{}"; // Fallback if absolutely no braces exist
                }
                return result;
            } 
            else {
                result.type = StreamEventType::WAITING;
                return result;
            }
        }

        // Find the earliest occurring marker in the buffer to avoid skipping tags
        size_t pos_tool_start = buffer_.find(MARKER_TOOL_START);
        size_t pos_tool_resp  = buffer_.find(MARKER_TOOL_RESP);
        size_t pos_think      = std::string::npos;

        if (current_mode_ == StreamEventType::CONTENT) {
            pos_think = buffer_.find(MARKER_THINK_START);
        } 
        else if (current_mode_ == StreamEventType::REASONING) {
            pos_think = buffer_.find(MARKER_THINK_END);
        }

        size_t min_pos = std::string::npos;
        if (pos_tool_start != std::string::npos) min_pos = std::min(min_pos, pos_tool_start);
        if (pos_tool_resp != std::string::npos)  min_pos = std::min(min_pos, pos_tool_resp);
        if (pos_think != std::string::npos)      min_pos = std::min(min_pos, pos_think);

        // Flush the text content before the earliest marker
        if (min_pos != std::string::npos && min_pos > 0) {
            result.content = buffer_.substr(0, min_pos);
            result.type = current_mode_;
            buffer_ = buffer_.substr(min_pos);
            return result;
        }

        // Process the exact marker located at index 0
        if (min_pos == 0) {
            if (pos_tool_resp == 0) {
                buffer_ = buffer_.substr(MARKER_TOOL_RESP.length());
                continue;
            }
            if (pos_tool_start == 0) {
                is_in_tool_block_ = true;
                buffer_ = buffer_.substr(MARKER_TOOL_START.length());
                continue;
            }
            if (pos_think == 0) {
                if (current_mode_ == StreamEventType::CONTENT) {
                    buffer_ = buffer_.substr(MARKER_THINK_START.length());
                    current_mode_ = StreamEventType::REASONING;
                } else {
                    buffer_ = buffer_.substr(MARKER_THINK_END.length());
                    current_mode_ = StreamEventType::CONTENT;
                }
                continue;
            }
        }

        // Safe Flush Mechanism
        std::vector<std::string> active_markers;
        active_markers.push_back(MARKER_TOOL_START);
        active_markers.push_back(MARKER_TOOL_RESP);

        if (current_mode_ == StreamEventType::CONTENT) {
            active_markers.push_back(MARKER_THINK_START);
        } 
        else if (current_mode_ == StreamEventType::REASONING) {
            active_markers.push_back(MARKER_THINK_END);
        }

        size_t safe_flush_len = buffer_.length();
        for (const auto& marker : active_markers) {
            for (size_t i = 1; i <= marker.length() && i <= buffer_.length(); ++i) {
                if (buffer_.compare(buffer_.length() - i, i, marker, 0, i) == 0) {
                    safe_flush_len = std::min(safe_flush_len, buffer_.length() - i);
                }
            }
        }

        if (safe_flush_len > 0) {
            result.content = buffer_.substr(0, safe_flush_len);
            result.type = current_mode_;
            buffer_ = buffer_.substr(safe_flush_len);
            return result;
        } 
        else if (buffer_.length() > 0) {
            result.type = StreamEventType::WAITING;
            return result;
        }

        break;
    }

    result.type = current_mode_;
    return result;
}