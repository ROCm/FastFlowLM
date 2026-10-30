/// \file llama3.cpp
/// \brief llama3 class
/// \author FastFlowLM Team
/// \date 2025-09-04
/// \version 0.9.24
/// \note This is a source file for the llama3 class

#include "AutoModel/modeling_llama3.hpp"

/************              Llama3 family            **************/
Llama3::Llama3(flm_rt::device* npu_device_inst) : AutoModel(npu_device_inst, "Llama3") {}

void Llama3::load_model(std::string model_path, json model_info, int default_context_length, bool enable_preemption, const std::string& backend) {
    this->_shared_load_backend(model_path, model_info, default_context_length, enable_preemption, backend);
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
    this->_prefix_attach(this->lm_engine, llama_npu_state_api(this->lm_engine));
}

void Llama3::setup_tokenizer(std::string model_path) {
    auto tokenizer_config = this->_shared_setup_tokenizer(model_path);
}

std::string Llama3::apply_chat_template(nlohmann::ordered_json& messages, nlohmann::ordered_json tools, bool continue_final_message) {
    minja::chat_template_inputs inputs;
    inputs.messages = messages;
    inputs.extra_context = this->extra_context;
    return this->_render_chat_template(inputs, continue_final_message);
}

void Llama3::fail_inference(const std::string& detail) {
    const bool poisoned = this->backend_ && this->backend_->poisoned();
    this->_shared_after_inference_failure(poisoned);
    std::string message = poisoned
        ? "Inference failed; unload/reload is required because the model is poisoned"
        : "Inference failed; the current conversation was cleared";
    if (!detail.empty()) message += " (" + detail + ")";
    throw ModelRequestError(500, true, message);
}

bool Llama3::insert(chat_meta_info_t& meta_info, lm_uniform_input_t& input, std::function<bool()> is_cancelled) {
    this->_shared_guard_poisoned();
    // preprocess
    this->profiler_list[TKOEN_ENCODE_TIME].start();
    std::string templated_text;
    if (input.messages.empty() && input.prompt.empty()) {
        header_print("WARNING", "No messages or prompt provided");
        return false;
    }
    if (!input.messages.empty()) { // already a formated messages, usually from REST API
        templated_text = this->apply_chat_template(input.messages, nlohmann::ordered_json::object(), meta_info.continue_final_message);
    }
    else if (!input.prompt.empty()) { // a pure text, usually from the cli
        nlohmann::ordered_json messages;

        messages.push_back({ {"role", "user"}, {"content", input.prompt} });
        templated_text = this->apply_chat_template(messages);
    }

    std::vector<int> tokens = this->tokenizer->encode(templated_text);

    this->profiler_list[TKOEN_ENCODE_TIME].stop(tokens.size());
    // hardware

    // The template's generation prompt ("<|start_header_id|>assistant<|end_header_id|>\n\n") is how the next turn
    // renders this reply's header, so the whole prompt can be saved: nothing is left out of the prefill (trim 0).
    if (this->_prefix_slot_mode(meta_info, tokens.size(), 0)) {
        try {
            return this->_prefix_text_insert(meta_info, tokens, 0, "<|start_header_id|>user", is_cancelled, input.requested_max_new_tokens);
        } catch (const ModelRequestError&) {
            throw;
        } catch (const std::exception& error) {
            this->fail_inference(error.what());
        } catch (...) {
            this->fail_inference();
        }
    }
    this->_prefix_legacy_path();
    try {
        return this->_shared_insert(meta_info, tokens, is_cancelled, nullptr, 0, input.requested_max_new_tokens);
    } catch (const ModelRequestError&) {
        throw;
    } catch (const std::exception& error) {
        this->fail_inference(error.what());
    } catch (...) {
        this->fail_inference();
    }
}


std::string Llama3::generate(chat_meta_info_t& meta_info, int length_limit, std::ostream& os, std::function<bool()> is_cancelled) {
    this->_shared_guard_poisoned();
    try {
        return this->_shared_generate(meta_info, length_limit, os, is_cancelled);
    } catch (const ModelRequestError&) {
        throw;
    } catch (const std::exception& error) {
        this->fail_inference(error.what());
    } catch (...) {
        this->fail_inference();
    }
}

std::string Llama3::generate_with_prompt(chat_meta_info_t& meta_info, lm_uniform_input_t& input, int length_limit, std::ostream& os) {
    if (!this->insert(meta_info, input)) {
        return "";
    }
    return this->generate(meta_info, length_limit, os);
}

/************              DeepSeek_r1_8b family            **************/
DeepSeek_r1_8b::DeepSeek_r1_8b(flm_rt::device* npu_device_inst) : AutoModel(npu_device_inst) {}

void DeepSeek_r1_8b::load_model(std::string model_path, json model_info, int default_context_length, bool enable_preemption, const std::string& backend) {
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
    this->_prefix_attach(this->lm_engine, llama_npu_state_api(this->lm_engine));
}

void DeepSeek_r1_8b::setup_tokenizer(std::string model_path) {
    auto tokenizer_config = this->_shared_setup_tokenizer(model_path);
    this->think_marker_id = this->tokenizer->encode("<think>")[0];
}

std::string DeepSeek_r1_8b::apply_chat_template(nlohmann::ordered_json& messages, nlohmann::ordered_json tools, bool continue_final_message) {
    minja::chat_template_inputs inputs;
    inputs.messages = messages;
    inputs.extra_context = this->extra_context;
    return this->_render_chat_template(inputs, continue_final_message);
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
        templated_text = this->apply_chat_template(input.messages, nlohmann::ordered_json::object(), meta_info.continue_final_message);
    }
    else if (!input.prompt.empty()) { // a pure text, usually from the cli
        nlohmann::ordered_json messages;

        messages.push_back({ {"role", "user"}, {"content", input.prompt} });
        templated_text = this->apply_chat_template(messages);
    }

    std::vector<int> tokens = this->tokenizer->encode(templated_text);
    tokens.erase(tokens.begin());
    this->profiler_list[TKOEN_ENCODE_TIME].stop(tokens.size());
    // hardware
    int restore_idx = -1;
    llama_npu *llama_engine = dynamic_cast<llama_npu*>(this->lm_engine);

    // The generation prompt ("<｜Assistant｜>\n") is how the next turn renders this reply's header: save the whole prompt (trim 0).
    if (this->_prefix_slot_mode(meta_info, tokens.size(), 0)) {
        return this->_prefix_text_insert(meta_info, tokens, 0, "<｜User｜>", is_cancelled);
    }
    this->_prefix_legacy_path();
    if (meta_info.restore_allowed) {
        restore_idx = llama_engine->restore();
        this->total_tokens = restore_idx;
        this->token_history = checkpoint_his; // restore the token history to be consistent with the restored KV cache, which is crucial for correct functioning of _shared_insert's prefix-matching logic
    }
    bool success = this->_shared_insert(meta_info, tokens, is_cancelled, nullptr);

    checkpoint_his = token_history;
    int checkpoint_idx = llama_engine->checkpoint();

    return success;
}

std::string DeepSeek_r1_8b::generate(chat_meta_info_t& meta_info, int length_limit, std::ostream& os, std::function<bool()> is_cancelled) {
    std::vector<int> sampled_tokens;
    std::string result;
    // A continued message already carries its own text: no think preamble of ours.
    if (!meta_info.continue_final_message) {
        os << "<think>\n\n";
    }
    if (length_limit > 0){
        sampled_tokens.reserve(length_limit);
    }
    else{
        sampled_tokens.reserve(4096);
    }
    assert(this->last_token != -1);

    stop_reason_t reason = EOT_DETECTED;
    int last_sampled_token = this->last_token;

    token_history.push_back(this->last_token);
    // the sample insert() took is the reply's first token, and counts toward the limit
    meta_info.generated_tokens++;
    if (this->is_normal_token(last_sampled_token) && last_sampled_token != -1){
        std::string token_str = this->tokenizer->run_time_decoder(last_sampled_token);
        result += token_str;
        os << token_str << std::flush;
    }
    if (this->is_eos(last_sampled_token)){
        // Sampled from the prefill, so no engine has run it.
        token_history.pop_back();
        meta_info.stop_reason = reason;
        return result;
    }
    this->profiler_list[DECODING_TIME].reset();
    this->profiler_list[TKOEN_DECODE_TIME].reset();
    if (this->total_tokens >= this->MAX_L){
        reason = MAX_LENGTH_REACHED;
    }
    else if ((length_limit > 0) && (meta_info.generated_tokens >= length_limit)){
        reason = MAX_LENGTH_REACHED;
    }
    else while (true){
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
        if (this->is_normal_token(sampled_token)){
            std::string token_str = this->tokenizer->run_time_decoder(sampled_token);
            os << token_str << std::flush;
            result += token_str;
        }
        this->profiler_list[TKOEN_DECODE_TIME].stop(1);
        token_history.push_back(sampled_token);
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
        token_history.pop_back();
    }
    if (this->total_tokens >= this->MAX_L){
        header_print("WARNING", "Max length reached, stopping generation...");
    }
    std::cout << std::endl;
    header_print("FLM", "Model RAW Output: \n" + result);
    if (!meta_info.continue_final_message) {
        result = "<think>\n\n" + result;
    }
    return result;
}

std::string DeepSeek_r1_8b::generate_with_prompt(chat_meta_info_t& meta_info, lm_uniform_input_t& input, int length_limit, std::ostream& os) {
    if (!this->insert(meta_info, input)) {
        return "";
    }
    return this->generate(meta_info, length_limit, os);
}
