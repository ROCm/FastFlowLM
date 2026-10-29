/// \file lfm2.cpp
/// \brief lfm2 class
/// \author FastFlowLM Team
/// \date 2025-09-04
/// \version 0.9.15
/// \note This is a source file for the lfm2 class

#include "AutoModel/modeling_lfm2.hpp"
#include "utils/utils.hpp"
#include "metrices.hpp"

/************              LFM2 family            **************/
LFM2::LFM2(flm_rt::device* npu_device_inst) : AutoModel(npu_device_inst, "LFM2") {}

void LFM2::load_model(std::string model_path, json model_info, int default_context_length, bool enable_preemption, const std::string& backend) {
    this->_shared_load_backend(model_path, model_info, default_context_length, enable_preemption, backend);
    this->setup_tokenizer(model_path);
    this->sampler.reset();

    sampler_config config;
    config.top_p = 0.95;
    config.top_k = 10;
    config.min_p = 0.1;
    config.temperature = 0.3;
    
    this->set_sampler(config);
    for (size_t i = 0; i < PROFILER_TYPE_NUM; i++) {
        this->profiler_list[i].reset();
    }
    _prefix_attach(lm_engine, lfm2_npu_state_api(lm_engine));
}

void LFM2::setup_tokenizer(std::string model_path) {
    auto tokenizer_config = this->_shared_setup_tokenizer(model_path);
    this->tokenizer->is_doubled_encoded = true;
}

std::string LFM2::apply_chat_template(nlohmann::ordered_json& messages, nlohmann::ordered_json tools, bool continue_final_message) {
    minja::chat_template_inputs inputs;
    minja::chat_template_options opt;
    opt.polyfill_tool_responses = false;
    inputs.messages = messages;
    inputs.extra_context = this->extra_context;
    // inputs.tools = tools;
    return this->_render_chat_template(inputs, continue_final_message, opt);
}

bool LFM2::insert(chat_meta_info_t& meta_info, lm_uniform_input_t& input, std::function<bool()> is_cancelled) {
    // preprocess
    this->profiler_list[TKOEN_ENCODE_TIME].start();
    std::string templated_text;
    if (input.messages.empty() && input.prompt.empty()) {
        header_print("WARNING", "No messages or prompt provided");
        return false;
    }
    if (!input.messages.empty()) { // already a formated messages, usually from REST API
        //templated_text = this->apply_chat_template(input.messages);
        templated_text = this->apply_chat_template(input.messages, input.tools, meta_info.continue_final_message);
    }
    else if (!input.prompt.empty()) { // a pure text, usually from the cli
        nlohmann::ordered_json messages;

        messages.push_back({ {"role", "user"}, {"content", input.prompt} });
        templated_text = this->apply_chat_template(messages);
    }

    std::vector<int> tokens = this->tokenizer->encode(templated_text);

    this->profiler_list[TKOEN_ENCODE_TIME].stop(tokens.size());
    // hardware

    // Prefix slots. The generation prompt ("<|im_start|>assistant\n") is how the next turn renders this reply's header: save the whole prompt.
    if (this->_prefix_slot_mode(meta_info, tokens.size(), 0)) {
        return this->_prefix_text_insert(meta_info, tokens, 0, "<|im_start|>user", is_cancelled);
    }
    this->_prefix_legacy_path();

    return this->_shared_insert(meta_info, tokens, is_cancelled);
}


std::string LFM2::generate(chat_meta_info_t& meta_info, int length_limit, std::ostream& os, std::function<bool()> is_cancelled) {
    return this->_shared_generate(meta_info, length_limit, os, is_cancelled);
}

std::string LFM2::generate_with_prompt(chat_meta_info_t& meta_info, lm_uniform_input_t& input, int length_limit, std::ostream& os) {
    if (!this->insert(meta_info, input)) {
        return "";
    }
    return this->generate(meta_info, length_limit, os);
}

/***********              LFM2_5_TK family            ***********/
LFM2_5_TK::LFM2_5_TK(flm_rt::device* npu_device_inst) : AutoModel(npu_device_inst, "LFM2_5_TK") {}

void LFM2_5_TK::load_model(std::string model_path, json model_info, int default_context_length, bool enable_preemption, const std::string& backend) {
    this->_shared_load_backend(model_path, model_info, default_context_length, enable_preemption, backend);
    this->setup_tokenizer(model_path);
    this->sampler.reset();

    sampler_config config;
    config.rep_penalty = 1.05;
    config.temperature = 0.1;
    config.top_p = 0.1;
    config.top_k = 10;
    config.rep_penalty_window = 1024;
    config.freq_penalty = 1.0;
    config.freq_penalty_window = 1024;
    this->set_sampler(config);
    for (size_t i = 0; i < PROFILER_TYPE_NUM; i++) {
        this->profiler_list[i].reset();
    }
    _prefix_attach(lm_engine, lfm2_npu_state_api(lm_engine));
}

void LFM2_5_TK::setup_tokenizer(std::string model_path) {
    auto tokenizer_config = this->_shared_setup_tokenizer(model_path);
    this->tokenizer->is_doubled_encoded = true;
}

std::string LFM2_5_TK::apply_chat_template(nlohmann::ordered_json& messages, nlohmann::ordered_json tools, bool continue_final_message) {
    minja::chat_template_inputs inputs;
    minja::chat_template_options opt;
    opt.polyfill_tool_responses = false;
    inputs.messages = messages;
    inputs.extra_context = this->extra_context;
    // inputs.tools = tools;
    return this->_render_chat_template(inputs, continue_final_message, opt);
}

bool LFM2_5_TK::insert(chat_meta_info_t& meta_info, lm_uniform_input_t& input, std::function<bool()> is_cancelled) {
    // preprocess
    this->profiler_list[TKOEN_ENCODE_TIME].start();
    std::string templated_text;
    if (input.messages.empty() && input.prompt.empty()) {
        header_print("WARNING", "No messages or prompt provided");
        return false;
    }
    if (!input.messages.empty()) { // already a formated messages, usually from REST API
        templated_text = this->apply_chat_template(input.messages, nlohmann::ordered_json::object(), meta_info.continue_final_message);
        // templated_text = this->apply_chat_template(input.messages, input.tools);
    }
    else if (!input.prompt.empty()) { // a pure text, usually from the cli
        nlohmann::ordered_json messages;

        messages.push_back({ {"role", "user"}, {"content", input.prompt} });
        templated_text = this->apply_chat_template(messages);
    }

    std::vector<int> tokens = this->tokenizer->encode(templated_text);

    this->profiler_list[TKOEN_ENCODE_TIME].stop(tokens.size());

    // hardware
    int restore_idx = -1;
    lfm2_npu *lfm2_engine = dynamic_cast<lfm2_npu*>(this->lm_engine);

    // Prefix slots. The two tokens left out ("<think>\n") are fed by generate(); the next turn does not render them.
    // A continued message keeps all its tokens: they are the client's text, not the generation prompt.
    const size_t think_trim = meta_info.continue_final_message ? 0 : 2;
    if (this->_prefix_slot_mode(meta_info, tokens.size(), think_trim)) {
        return this->_prefix_text_insert(meta_info, tokens, think_trim, "<|im_start|>user", is_cancelled);
    }
    this->_prefix_legacy_path();

    if (meta_info.restore_allowed) {
        restore_idx = lfm2_engine->restore();
        this->total_tokens = restore_idx;
        this->token_history = checkpoint_his; // restore the token history to be consistent with the restored KV cache, which is crucial for correct functioning of _shared_insert's prefix-matching logic
    }

    // Remove the starting reasoning placeholder appended by the chat template: "<think>\n" -> token ids 151667 708.
    size_t n = tokens.size();
    tokens.resize(n - think_trim);

    bool success = this->_shared_insert(meta_info, tokens, is_cancelled, nullptr);

    checkpoint_his = token_history;
    int checkpoint_idx = lfm2_engine->checkpoint();

    return success;
}


std::string LFM2_5_TK::generate(chat_meta_info_t& meta_info, int length_limit, std::ostream& os, std::function<bool()> is_cancelled) {
    if (meta_info.continue_final_message) {
        // insert() held back no "<think>\n": the message goes on from the sample insert() made
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

    this->token_history.push_back(think_start_id);
    this->profiler_list[DECODING_TIME].start();
    this->_force_token(meta_info, think_start_id);
    this->lm_engine->forward(think_start_id);
    this->profiler_list[DECODING_TIME].stop(1);
    token_str = this->tokenizer->run_time_decoder(think_start_id);
    result += token_str;
    os << token_str << std::flush;

    // \n
    this->token_history.push_back(708);
    this->profiler_list[DECODING_TIME].start();
    this->_force_token(meta_info, 708);
    buffer<bf16> y = this->lm_engine->forward(708);
    this->profiler_list[DECODING_TIME].stop(1);
    token_str = this->tokenizer->run_time_decoder(708);
    result += token_str;
    sampled_token = this->_sample(y, meta_info);
    os << token_str << std::flush;

    this->total_tokens++;
    // the first sample after the preamble is the reply's first token, and counts toward the limit
    meta_info.generated_tokens++;
    int last_sampled_token = sampled_token;
    token_history.push_back(last_sampled_token);
    token_str = this->tokenizer->run_time_decoder(last_sampled_token);
    result += token_str;
    os << token_str << std::flush;

    if (this->is_eos(last_sampled_token)){
        this->lm_engine->forward(last_sampled_token);
    }
    else if (this->total_tokens >= this->MAX_L){
        header_print("WARNING", "Max length reached, stopping generation...");
        reason = MAX_LENGTH_REACHED;
    }
    else if ((length_limit > 0) && (meta_info.generated_tokens >= length_limit)){
        reason = MAX_LENGTH_REACHED;
    }
    else while (true){
        // Running out of window is a truncation; testing here rather than in
        // the loop condition records it.
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
        if (this->is_normal_token(sampled_token)){ // filter out special tokens
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
    return result;
}

std::string LFM2_5_TK::generate_with_prompt(chat_meta_info_t& meta_info, lm_uniform_input_t& input, int length_limit, std::ostream& os) {
    if (!this->insert(meta_info, input)) {
        return "";
    }
    return this->generate(meta_info, length_limit, os);
}
