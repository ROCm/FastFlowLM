/// \file deepseek.cpp
/// \brief deepseek class
/// \author FastFlowLM Team
/// \date 2025-09-01
/// \version 0.9.24
/// \note This is a source file for the deepseek class


#include "AutoModel/modeling_qwen3_5vl.hpp"
#include "metrices.hpp"


/************              Qwen3_5VL family            **************/
Qwen3_5VL::Qwen3_5VL(xrt::device* npu_device_inst) : AutoModel(npu_device_inst, "Qwen3_5VL") {}

void Qwen3_5VL::load_model(std::string model_path, json model_info, int default_context_length, bool enable_preemption) {
    this->_shared_load_model(model_path, model_info, default_context_length, enable_preemption);
    
    this->q4nx = std::make_unique<Q4NX>(this->model_path);
    // lm_config->model_type == qwen3
    this->lm_engine = std::make_unique<qwen3_5vl_npu>(*this->lm_config, this->npu.get(), this->MAX_L);

    this->lm_engine->load_weights(*this->q4nx);
    //free the q4nx
    this->q4nx.reset();
    this->lm_engine->clear_context();
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
}

void Qwen3_5VL::setup_tokenizer(std::string model_path) {
    auto tokenizer_config = this->_shared_setup_tokenizer(model_path);
}

std::string Qwen3_5VL::apply_chat_template(nlohmann::ordered_json& messages, nlohmann::ordered_json tools) {
    minja::chat_template_inputs inputs;
    inputs.add_generation_prompt = true;
    inputs.messages = messages;
    inputs.extra_context = this->extra_context;
    inputs.extra_context["enable_thinking"] = this->enable_think;
    if (!tools.empty() && this->enable_tool)
        inputs.tools = tools;
    return this->chat_tmpl->apply(inputs);
}

bool Qwen3_5VL::insert(chat_meta_info_t& meta_info, lm_uniform_input_t& input, std::function<bool()> is_cancelled) {
    // preprocess
    constexpr int image_soft_token_id = 248056;
    this->profiler_list[TKOEN_ENCODE_TIME].start();
    std::string templated_text;
    if (input.messages.empty() && input.prompt.empty()) {
        header_print("WARNING", "No messages or prompt provided");
        return false;
    }

    constexpr bool DEBUG_IMAGE_PREPROCESS = false;
    qwen3_5vl_image_payload_t image_payload;
    image_payload.num_images = 0;
    if (input.images.size() > 0) {


        // header_print("info", "Processing images...");
        
        // time_utils::time_point preprocess_start = time_utils::now();
        for(const auto& img_str : input.images){
            qwen3_5vl_image_t image = this->load_image(img_str);
            
            preprocess_image(image, image_payload._data__processed);
            // Push the image AFTER preprocessing so grid_h and grid_w are set
            image_payload.images.push_back(image);
            image_payload.num_images++;
        } 
    }
    if (!input.messages.empty()) { // already a formated messages, usually from REST API
        json qwenvl_message = json::array();
        for (const auto& item : input.messages) {
            if (!item.contains("images")) {
                qwenvl_message.push_back(item);
                continue;
            }

            json newContent = json::array();
            for (const auto& img : item["images"]) {
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
        int total_images = 0;
        for (auto& message : qwenvl_message) {
            auto content = message.value("content", nlohmann::ordered_json::array());
            for (auto& item : content) {
                if (item.contains("type") && item["type"] == "image") {
                    std::string img_str = item.value("image", "");
                    if (!img_str.empty()) {
                        total_images++;
                    }
                    qwen3_5vl_image_t image = this->load_image_base64(img_str);
                    preprocess_image(image, image_payload._data__processed);
                    image_payload.images.push_back(image);
                    image_payload.num_images++;
                }
            }
        }
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
    for (int i = 0; i < input.images.size(); i++) {
        total_image_tokens += image_payload.images[i].grid_h * image_payload.images[i].grid_w;
    }
    tokens.reserve(tokens_init.size() + total_image_tokens);
    int image_counter = 0;
    for (int i = 0; i < tokens_init.size(); i++) {
        if (tokens_init[i] == image_soft_token_id) {
            for (int j = 0; j < image_payload.images[image_counter].grid_h * image_payload.images[image_counter].grid_w / 4; j++) {
                tokens.push_back(image_soft_token_id);
            }
            image_counter++;
        } else {
            tokens.push_back(tokens_init[i]);
        }
    }

    this->profiler_list[TKOEN_ENCODE_TIME].stop(tokens.size());

    // find the last image token index
    int last_image_token_index = -1;
    for (int i = 0; i < tokens.size(); i++) {
        if (tokens[i] == image_soft_token_id) {
            last_image_token_index = i;
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

std::string Qwen3_5VL::generate(chat_meta_info_t& meta_info, int length_limit, std::ostream& os, std::function<bool()> is_cancelled) {
    if (this->enable_think) {
        os << "<think>\n" << std::flush;
    }
    return this->_shared_generate(meta_info, length_limit, os, is_cancelled);
}

std::string Qwen3_5VL::generate_with_prompt(chat_meta_info_t& meta_info, lm_uniform_input_t& input, int length_limit, std::ostream& os) {
    if (!this->insert(meta_info, input)) {
        return "";
    }
    if (this->enable_think) {
        os << "<think>\n" << std::flush;
    }
    return this->_shared_generate(meta_info, length_limit, os);
}

// Non-stream
NonStreamResult Qwen3_5VL::parse_nstream_content(const std::string response_text) {
    NonStreamResult result;

    std::string start_tag = "<tool_call>";
    std::string end_tag = "</tool_call>";
    std::string func_end_tag = "</function>";

    size_t start_pos = response_text.find(start_tag);
    size_t end_pos = response_text.find(end_tag);

    if (start_pos == std::string::npos) {
        // pure content
        result.content = response_text;
        return result;
    }

    start_pos += start_tag.length();

    if (end_pos == std::string::npos) {
        end_pos = response_text.find(func_end_tag, start_pos);
        if (end_pos != std::string::npos) {
            end_pos += func_end_tag.length();
        }
        else {
            end_pos = response_text.length();
        }
    }

    std::string block = response_text.substr(start_pos, end_pos - start_pos);

    auto trim_tool_value = [](std::string value) {
        while (!value.empty() && (value.front() == '\n' || value.front() == '\r' || value.front() == ' ' || value.front() == '\t')) {
            value.erase(0, 1);
        }
        while (!value.empty() && (value.back() == '\n' || value.back() == '\r' || value.back() == ' ' || value.back() == '\t')) {
            value.pop_back();
        }
        return value;
    };

    std::string func_open = "<function=";
    size_t func_start = block.find(func_open);
    if (func_start != std::string::npos) {
        func_start += func_open.length();
        size_t func_name_end = block.find(">", func_start);
        if (func_name_end != std::string::npos) {
            result.tool_name = sanitize_tool_argument_json_strings(block.substr(func_start, func_name_end - func_start));
        }
    }

    nlohmann::json args = nlohmann::json::object();
    std::string param_open = "<parameter=";
    std::string param_close = "</parameter>";
    size_t search_pos = 0;

    while (true) {
        size_t param_start = block.find(param_open, search_pos);
        if (param_start == std::string::npos) break;

        param_start += param_open.length();
        size_t param_name_end = block.find(">", param_start);
        if (param_name_end == std::string::npos) break;

        std::string param_name = block.substr(param_start, param_name_end - param_start);
        size_t value_start = param_name_end + 1;
        size_t value_end = block.find(param_close, value_start);

        size_t next_param_pos = block.find(param_open, value_start);
        size_t func_boundary_pos = block.find(func_end_tag, value_start);

        auto use_earlier_boundary = [&value_end](size_t boundary_pos) {
            if (boundary_pos != std::string::npos && (value_end == std::string::npos || boundary_pos < value_end)) {
                value_end = boundary_pos;
            }
        };

        use_earlier_boundary(next_param_pos);
        use_earlier_boundary(func_boundary_pos);

        if (value_end == std::string::npos) {
            value_end = block.length();
        }

        std::string param_value = trim_tool_value(block.substr(value_start, value_end - value_start));

        try {
            args[param_name] = nlohmann::json::parse(param_value);
        }
        catch (...) {
            args[param_name] = param_value;
        }

        search_pos = value_end;
        if (block.compare(value_end, param_close.length(), param_close) == 0) {
            search_pos += param_close.length();
        }
    }

    result.tool_args = args.dump();

    return result;
}

// Stream
StreamResult Qwen3_5VL::parse_stream_content(const std::string content) {
    return parse_stream_content_impl(content, false);
}

StreamResult Qwen3_5VL::parse_stream_content_final(const std::string content) {
    return parse_stream_content_impl(content, true);
}

StreamResult Qwen3_5VL::parse_stream_content_impl(const std::string content, bool is_final) {
    const std::string MARKER_THINK_START = "<think>";
    const std::string MARKER_THINK_END = "</think>";
    const std::string MARKER_TOOL_START = "<tool_call>";
    const std::string MARKER_TOOL_END = "</tool_call>";
    const std::string MARKER_FUNC_END = "</function>"; 


    StreamResult result;
    buffer_ += content;

    while (true) {
        if (!is_in_tool_block_) {
            size_t stray_end_pos = buffer_.find(MARKER_TOOL_END);
            if (stray_end_pos != std::string::npos) {
                buffer_.erase(stray_end_pos, MARKER_TOOL_END.length());
            }
        }

        if (!is_in_tool_block_) {
            size_t tool_start_pos = buffer_.find(MARKER_TOOL_START);
            if (tool_start_pos != std::string::npos) {
                if (tool_start_pos > 0) {
                    result.content = buffer_.substr(0, tool_start_pos);
                    result.type = current_mode_;
                    buffer_ = buffer_.substr(tool_start_pos);
                    return result;
                }

                is_in_tool_block_ = true;
                buffer_ = buffer_.substr(MARKER_TOOL_START.length());
                result.type = StreamEventType::WAITING;
                return result;
            }
        }

        // tool calling process
        if (is_in_tool_block_) {
            size_t tool_end_pos = buffer_.find(MARKER_TOOL_END);
            size_t func_end_pos = buffer_.find(MARKER_FUNC_END);
            
            if (tool_end_pos != std::string::npos || func_end_pos != std::string::npos || (is_final && !buffer_.empty())) {
                size_t actual_end_pos = buffer_.size();
                size_t skip_length = 0;

                if (tool_end_pos != std::string::npos) {
                    actual_end_pos = tool_end_pos;
                    skip_length = MARKER_TOOL_END.length();
                } 
                else if (func_end_pos != std::string::npos) {
                    actual_end_pos = func_end_pos;
                    skip_length = MARKER_FUNC_END.length();
                }

                std::string block = buffer_.substr(0, actual_end_pos + skip_length);
                buffer_ = buffer_.substr(actual_end_pos + skip_length);
                is_in_tool_block_ = false;

                try {
                    result.type = StreamEventType::TOOL_DONE;
                    result.tool_id = "call_" + std::to_string(std::time(nullptr));

                    // parse function name
                    std::string func_open = "<function=";
                    size_t func_start = block.find(func_open);
                    if (func_start != std::string::npos) {
                        func_start += func_open.length();
                        size_t func_end = block.find(">", func_start);
                        if (func_end != std::string::npos) {
                            result.tool_name = sanitize_tool_argument_json_strings(block.substr(func_start, func_end - func_start));
                        }
                    }

                    // parse parameters
                    nlohmann::json args = nlohmann::json::object();
                    std::string param_open = "<parameter=";
                    std::string param_close = "</parameter>";
                    size_t search_pos = 0;
                    
                    while (true) {
                        size_t p_start = block.find(param_open, search_pos);
                        if (p_start == std::string::npos) break;
                        p_start += param_open.length();
                        size_t p_name_end = block.find(">", p_start);
                        if (p_name_end == std::string::npos) break;
                        std::string param_name = block.substr(p_start, p_name_end - p_start);

                        size_t val_start = p_name_end + 1;
                        if (val_start < block.size() && block[val_start] == '\n') val_start++;

                        size_t param_close_pos = block.find(param_close, val_start);
                        size_t val_end = param_close_pos;

                        size_t next_param_pos = block.find(param_open, val_start);
                        size_t func_boundary_pos = block.find(MARKER_FUNC_END, val_start);
                        size_t tool_boundary_pos = block.find(MARKER_TOOL_END, val_start);

                        auto use_earlier_boundary = [&val_end](size_t boundary_pos) {
                            if (boundary_pos != std::string::npos && (val_end == std::string::npos || boundary_pos < val_end)) {
                                val_end = boundary_pos;
                            }
                        };

                        use_earlier_boundary(next_param_pos);
                        use_earlier_boundary(func_boundary_pos);
                        use_earlier_boundary(tool_boundary_pos);

                        if (val_end == std::string::npos && is_final) {
                            val_end = block.size();
                        }
                        if (val_end == std::string::npos) break;

                        std::string param_value = block.substr(val_start, val_end - val_start);
                        
                        // Enhanced trim: handle multiple newlines or spaces that the model may generate after a parameter
                        while(!param_value.empty() && (param_value.back() == '\n' || param_value.back() == '\r' || param_value.back() == ' ')) {
                            param_value.pop_back();
                        }

                        try {
                            // Try to parse as native JSON type (Integer, Float, Boolean, Array, Object)
                            args[param_name] = nlohmann::json::parse(param_value);
                        } 
                        catch (...) {
                            args[param_name] = param_value;
                        }

                        search_pos = param_close_pos != std::string::npos && val_end == param_close_pos
                            ? val_end + param_close.length()
                            : val_end;
                    }
                    result.tool_args_str = args.dump();
                    return result;
                }
                catch (...) {
                    result.type = StreamEventType::CONTENT;
                    result.content = "[Error parsing tool call]";
                    return result;
                }
            }
            else {
                result.type = StreamEventType::WAITING;
                return result;
            }
        }

        if (current_mode_ == StreamEventType::CONTENT) {
            size_t think_start_pos = buffer_.find(MARKER_THINK_START);
            if (think_start_pos != std::string::npos) {
                if (think_start_pos > 0) {
                    result.content = buffer_.substr(0, think_start_pos);
                    result.type = StreamEventType::CONTENT;
                    buffer_ = buffer_.substr(think_start_pos);
                    return result;
                }
                buffer_ = buffer_.substr(MARKER_THINK_START.length());
                current_mode_ = StreamEventType::REASONING;
                continue;
            }
        }
        else if (current_mode_ == StreamEventType::REASONING) {
            size_t think_end_pos = buffer_.find(MARKER_THINK_END);
            if (think_end_pos != std::string::npos) {
                if (think_end_pos > 0) {
                    result.content = buffer_.substr(0, think_end_pos);
                    result.type = StreamEventType::REASONING;
                    buffer_ = buffer_.substr(think_end_pos);
                    return result;
                }
                buffer_ = buffer_.substr(MARKER_THINK_END.length());
                current_mode_ = StreamEventType::CONTENT;
                continue;
            }
        }

        if (!buffer_.empty()) {
            size_t last_lt = buffer_.rfind('<');
            // If '<' appears at the end (possibly an incomplete <tool_call> or <think> tag)
            if (last_lt != std::string::npos && (buffer_.length() - last_lt) <= 15) {
                if (last_lt > 0) {
                    // Only output the content before '<'
                    result.content = buffer_.substr(0, last_lt);
                    result.type = current_mode_;
                    buffer_ = buffer_.substr(last_lt); 
                    return result;
                } else {
                    // If '<' is the first character in the buffer, directly wait for the next chunk
                    result.type = StreamEventType::WAITING;
                    return result;
                }
            }

            result.content = buffer_;
            result.type = current_mode_;
            buffer_.clear();
            return result;
        }

        break;
    }

    result.type = current_mode_;
    return result;
}