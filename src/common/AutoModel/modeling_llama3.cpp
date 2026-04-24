/// \file llama3.cpp
/// \brief llama3 class
/// \author FastFlowLM Team
/// \date 2025-09-04
/// \version 0.9.24
/// \note This is a source file for the llama3 class

#include "AutoModel/modeling_llama3.hpp"

/************              Llama3 family            **************/
Llama3::Llama3(xrt::device* npu_device_inst) : AutoModel(npu_device_inst, "Llama3") {}

void Llama3::load_model(std::string model_path, json model_info, int default_context_length, bool enable_preemption) {
    this->_shared_load_model(model_path, model_info, default_context_length, enable_preemption);
    
    this->q4nx = std::make_unique<Q4NX>(this->model_path);
    // model_type == llama
    this->lm_engine = std::make_unique<llama_npu>(*this->lm_config, this->npu.get(), this->MAX_L);

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

    this->set_sampler(config);
    for (size_t i = 0; i < PROFILER_TYPE_NUM; i++) {
        this->profiler_list[i].reset();
    }
}

void Llama3::setup_tokenizer(std::string model_path) {
    auto tokenizer_config = this->_shared_setup_tokenizer(model_path);
}

std::string Llama3::apply_chat_template(nlohmann::ordered_json& messages, nlohmann::ordered_json tools) {
    minja::chat_template_inputs inputs;
    inputs.add_generation_prompt = true;
    inputs.messages = messages;
    inputs.extra_context = this->extra_context;
    return this->chat_tmpl->apply(inputs);
}

bool Llama3::insert(chat_meta_info_t& meta_info, lm_uniform_input_t& input, std::function<bool()> is_cancelled) {
    // preprocess
    this->profiler_list[TKOEN_ENCODE_TIME].start();
    std::string templated_text;
    if (input.messages.empty() && input.prompt.empty()) {
        header_print("WARNING", "No messages or prompt provided");
        return false;
    }
    if (!input.messages.empty()) { // already a formated messages, usually from REST API
        templated_text = this->apply_chat_template(input.messages);
    }
    else if (!input.prompt.empty()) { // a pure text, usually from the cli
        nlohmann::ordered_json messages;

        messages.push_back({ {"role", "user"}, {"content", input.prompt} });
        templated_text = this->apply_chat_template(messages);
    }

    std::vector<int> tokens = this->tokenizer->encode(templated_text);

    // some models are very sensitive to this bos token, such as lfm2
    if (this->is_first_prompt == false) {
        tokens.erase(tokens.begin()); // remove bos token in multi round conversation
    }
    this->is_first_prompt = false; // always set to false if the insert is ever called

    this->profiler_list[TKOEN_ENCODE_TIME].stop(tokens.size());
    // hardware

    return this->_shared_insert(meta_info, tokens, is_cancelled);
}


std::string Llama3::generate(chat_meta_info_t& meta_info, int length_limit, std::ostream& os, std::function<bool()> is_cancelled) {
    //header_print("is_cancelled", is_cancelled);
    return this->_shared_generate(meta_info, length_limit, os, is_cancelled);
}

std::string Llama3::generate_with_prompt(chat_meta_info_t& meta_info, lm_uniform_input_t& input, int length_limit, std::ostream& os) {
    if (!this->insert(meta_info, input)) {
        return "";
    }
    return this->_shared_generate(meta_info, length_limit, os);
}

/************              DeepSeek_r1_8b family            **************/
DeepSeek_r1_8b::DeepSeek_r1_8b(xrt::device* npu_device_inst) : AutoModel(npu_device_inst) {}

void DeepSeek_r1_8b::load_model(std::string model_path, json model_info, int default_context_length, bool enable_preemption) {
    this->_shared_load_model(model_path, model_info, default_context_length, enable_preemption);
    
    this->q4nx = std::make_unique<Q4NX>(this->model_path);
    // model_type == llama
    this->lm_engine = std::make_unique<llama_npu>(*this->lm_config, this->npu.get(), this->MAX_L);

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

void DeepSeek_r1_8b::setup_tokenizer(std::string model_path) {
    auto tokenizer_config = this->_shared_setup_tokenizer(model_path);
    this->think_marker_id = this->tokenizer->encode("<think>")[0];
}

std::string DeepSeek_r1_8b::apply_chat_template(nlohmann::ordered_json& messages, nlohmann::ordered_json tools) {
    minja::chat_template_inputs inputs;
    inputs.add_generation_prompt = true;
    inputs.messages = messages;
    inputs.extra_context = this->extra_context;
    return this->chat_tmpl->apply(inputs);
}

bool DeepSeek_r1_8b::insert(chat_meta_info_t& meta_info, lm_uniform_input_t& input, std::function<bool()> is_cancelled) {
    // preprocess
    this->profiler_list[TKOEN_ENCODE_TIME].start();
    std::string templated_text;
    if (input.messages.empty() && input.prompt.empty()) {
        header_print("WARNING", "No messages or prompt provided");
        return false;
    }
    if (!input.messages.empty()) { // already a formated messages, usually from REST API
        templated_text = this->apply_chat_template(input.messages);
    }
    else if (!input.prompt.empty()) { // a pure text, usually from the cli
        nlohmann::ordered_json messages;

        messages.push_back({ {"role", "user"}, {"content", input.prompt} });
        templated_text = this->apply_chat_template(messages);
    }

    std::vector<int> tokens = this->tokenizer->encode(templated_text);

    // some models are very sensitive to this bos token, such as lfm2
    if (this->is_first_prompt == false) {
        tokens.erase(tokens.begin()); // remove bos token in multi round conversation
    }
    this->is_first_prompt = false; // always set to false if the insert is ever called

    this->profiler_list[TKOEN_ENCODE_TIME].stop(tokens.size());
    // hardware

    return this->_shared_insert(meta_info, tokens, is_cancelled);
}

std::string DeepSeek_r1_8b::generate(chat_meta_info_t& meta_info, int length_limit, std::ostream& os, std::function<bool()> is_cancelled) {
    os << "<think>\n\n";
    std::string result = this->_shared_generate(meta_info, length_limit, os, is_cancelled);
    result = "<think>\n\n" + result;
    return result;
}

std::string DeepSeek_r1_8b::generate_with_prompt(chat_meta_info_t& meta_info, lm_uniform_input_t& input, int length_limit, std::ostream& os) {
    if (!this->insert(meta_info, input)) {
        return "";
    }
    return this->generate(meta_info, length_limit, os);
}

NonStreamResult DeepSeek_r1_8b::parse_nstream_content(const std::string response_text) {
    NonStreamResult result;

    std::string content, reasoning_content;

    std::string think_start_tag = "<think>";
    std::string think_end_tag = "</think>";

    size_t think_start_pos = response_text.find(think_start_tag);
    size_t think_end_pos = response_text.find(think_end_tag);


    think_start_pos += think_start_tag.length();
    std::string reasoning_str = response_text.substr(think_start_pos, think_end_pos - think_start_pos);
    result.reasoning_content = reasoning_str;

    std::string content_str = response_text.substr(think_end_pos + think_end_tag.length());
    result.content = content_str;

    return result;
}

StreamResult DeepSeek_r1_8b::parse_stream_content(const std::string content) {
    const std::string MARKER_THINK_START = "<think>";
    const std::string MARKER_THINK_END = "</think>";

    StreamResult result;
    buffer_ += content;

    while (true) {
        if (current_mode_ == StreamEventType::CONTENT) {
            // Check for the start of a thought block
            size_t pos = buffer_.find(MARKER_THINK_START);

            if (pos != std::string::npos) {
                // Emit content before the tag
                result.content += buffer_.substr(0, pos);
                result.type = StreamEventType::CONTENT;

                // Remove "<think>\n" and switch mode
                buffer_ = buffer_.substr(pos + MARKER_THINK_START.length());
                current_mode_ = StreamEventType::REASONING;
                continue;
            }
        }
        else if (current_mode_ == StreamEventType::REASONING) {
            // Check for the end of a thought block
            size_t pos = buffer_.find(MARKER_THINK_END);

            if (pos != std::string::npos) {
                // Emit content before the tag
                result.content += buffer_.substr(0, pos);
                result.type = StreamEventType::REASONING;

                // Remove "</think>\n" and switch mode
                buffer_ = buffer_.substr(pos + MARKER_THINK_END.length());
                current_mode_ = StreamEventType::CONTENT;
                continue;
            }
        }

        // Flush remaining buffer
        result.content += buffer_;
        result.type = current_mode_;
        buffer_.clear();
        break;
    }

    return result;
}