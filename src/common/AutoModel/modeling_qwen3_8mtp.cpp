/// \file modeling_qwen3_8mtp.cpp
/// \brief Qwen3_8MTP class
/// \author FastFlowLM Team
/// \date 2026-09-16
/// \version 0.9.28
/// \note AutoModel wrapper for Qwen3.8-27B. Text-only.
/// \note Speculative decode IS wired up: causal_lm carries the
///       supports_speculation()/speculate() hooks and AutoModel::
///       _shared_generate() drives them under a greedy sampler.
/// \note generate() therefore delegates its decode loop to _shared_generate()
///       and keeps only the think-preamble replay. It previously carried a
///       private copy of the loop that called forward() + sample() directly,
///       which left speculation dead on the main chat path -- correct output,
///       no speedup, and a hit rate that never printed because no cycle ran.

#include "AutoModel/modeling_qwen3_8mtp.hpp"


/************              Qwen3_8MTP family            **************/
Qwen3_8MTP::Qwen3_8MTP(flm_rt::device* npu_device_inst) : AutoModel(npu_device_inst, "Qwen3_8MTP") {}

void Qwen3_8MTP::load_model(std::string model_path, json model_info, int default_context_length, bool enable_preemption) {
    this->_shared_load_model(model_path, model_info, default_context_length, enable_preemption);

    this->q4nx = std::make_unique<Q4NX>(this->model_path);
    // lm_config->get<std::string>("model_type", "") == qwen3_5
    this->lm_engine = std::make_unique<qwen3_8mtp_npu>(*this->lm_config, this->npu.get(), this->MAX_L);

    this->lm_engine->load_weights(*this->q4nx);
    //free the q4nx
    this->q4nx.reset();
    this->lm_engine->clear_context();
    this->setup_tokenizer(model_path);
    this->sampler.reset();

    this->enable_tool = true;

    // Qwen3.5-family recommended decoding defaults; the test harness overrides
    // these with set_topk(1) when it wants a reproducible stream.
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

void Qwen3_8MTP::setup_tokenizer(std::string model_path) {
    // tokenizer_config.json already lists eos_token_id as
    // [248044 <|endoftext|>, 248046 <|im_end|>, 248048], and
    // _shared_setup_tokenizer registers every entry of that array. config.json
    // alone would only give 248044, which the chat template never emits --
    // that mismatch is the runaway-generation failure mode.
    auto tokenizer_config = this->_shared_setup_tokenizer(model_path);
}

std::string Qwen3_8MTP::apply_chat_template(nlohmann::ordered_json& messages, nlohmann::ordered_json tools) {
    minja::chat_template_inputs inputs;
    inputs.add_generation_prompt = true;
    inputs.messages = messages;
    inputs.extra_context = this->extra_context;
    inputs.extra_context["enable_thinking"] = this->enable_think;
    if (!tools.empty() && this->enable_tool)
        inputs.tools = tools;
    return this->chat_tmpl->apply(inputs);
}

bool Qwen3_8MTP::insert(chat_meta_info_t& meta_info, lm_uniform_input_t& input, std::function<bool()> is_cancelled) {
    // preprocess
    this->profiler_list[TKOEN_ENCODE_TIME].start();
    std::string templated_text;
    if (input.messages.empty() && input.prompt.empty()) {
        header_print("WARNING", "No messages or prompt provided");
        return false;
    }

    // Text-only wrapper. model.q4nx carries no vision tower, so an image would
    // be templated into <|vision_start|><|image_pad|>... placeholders that
    // nothing ever fills -- the prompt would prefill garbage rather than fail.
    // Drop them loudly instead.
    if (!input.images.empty()) {
        header_print("WARNING", "Qwen3.8-27B is loaded text-only; ignoring " << input.images.size() << " image(s)");
        input.images.clear();
    }
    if (!input.audios.empty()) {
        header_print("WARNING", "Qwen3.8-27B is loaded text-only; ignoring " << input.audios.size() << " audio clip(s)");
        input.audios.clear();
    }

    if (!input.messages.empty()) { // already a formated messages, usually from REST API
        // Strip per-message media keys for the same reason as above.
        json text_only_messages = json::array();
        for (const auto& item : input.messages) {
            json entry = item;
            entry.erase("images");
            entry.erase("audios");
            text_only_messages.push_back(entry);
        }
        nlohmann::ordered_json ordered_messages = text_only_messages;
        templated_text = this->apply_chat_template(ordered_messages, input.tools);
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
    qwen3_8mtp_npu *qwen3_8mtp_engine = dynamic_cast<qwen3_8mtp_npu*>(this->lm_engine.get());

    if (meta_info.restore_allowed) {
        restore_idx = qwen3_8mtp_engine->restore();
        this->total_tokens = restore_idx;
        this->token_history = checkpoint_his; // restore the token history to be consistent with the restored KV cache, which is crucial for correct functioning of _shared_insert's prefix-matching logic
    }

    // The chat template's generation prompt ends with the think preamble, and
    // generate() re-feeds those exact tokens so it can stream them. Trim them
    // here or they get prefilled twice.
    //   thinking on : "<think>" "\n"                      -> 2 tokens
    //   thinking off: "<think>" "\n\n" "</think>" "\n\n"   -> 4 tokens
    size_t n = tokens.size();
    tokens.resize(n - (this->enable_think ? 2 : 4));

    bool success = this->_shared_insert(meta_info, tokens, is_cancelled, nullptr);

    checkpoint_his = token_history;
    int checkpoint_idx = qwen3_8mtp_engine->checkpoint();
    return success;
}

std::string Qwen3_8MTP::generate(chat_meta_info_t& meta_info, int length_limit, std::ostream& os, std::function<bool()> is_cancelled) {
    std::string result;
    assert(this->last_token != -1);

    // Only the preamble's own forward() calls are charged here; the loop's
    // profilers are reset inside _shared_generate().
    this->profiler_list[DECODING_TIME].reset();
    this->profiler_list[TKOEN_DECODE_TIME].reset();
    std::string token_str;
    int sampled_token;
    // Replay the think preamble insert() trimmed off, so the tokens land in
    // both the KV cache and the visible stream.
    if(this->enable_think) {
        this->token_history.push_back(think_start_id);
        this->profiler_list[DECODING_TIME].start();
        this->lm_engine->forward(think_start_id);
        this->profiler_list[DECODING_TIME].stop(1);
        token_str = this->tokenizer->run_time_decoder(think_start_id);
        result += token_str;
        os << token_str << std::flush;

        // \n
        this->token_history.push_back(198);
        this->profiler_list[DECODING_TIME].start();
        buffer<bf16> y = this->lm_engine->forward(198);
        this->profiler_list[DECODING_TIME].stop(1);
        token_str = this->tokenizer->run_time_decoder(198);
        result += token_str;
        sampled_token = this->sampler->sample(y);
        os << token_str << std::flush;
    }
    else{
        this->token_history.push_back(think_start_id);
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
        sampled_token = this->sampler->sample(y);
    }
    if (this->total_tokens >= this->MAX_L){
        header_print("WARNING", "Max length reached, stopping generation...");
        meta_info.stop_reason = MAX_LENGTH_REACHED;
        return result;
    }

    // Hand the decode loop to the base class rather than running a private
    // copy of it. This function used to carry its own while() calling
    // forward() + sample() directly, which meant the whole speculative path --
    // and with it the MTP hit rate -- was dead on the main chat path: only
    // generate_with_prompt() and the milestone driver ever reached
    // speculate(). The preamble above is the only part that is genuinely
    // specific to this model, so it stays; the loop is not.
    //
    // _shared_generate() seeds from this->last_token, emits it, then feeds it
    // to forward(). The preamble's final sample is exactly that seed.
    this->last_token = sampled_token;
    result += this->_shared_generate(meta_info, length_limit, os, is_cancelled);

    // The engine counts every draft/verify cycle, so this prints whenever
    // speculation actually ran, and prints nothing otherwise -- a checkpoint
    // without an MTP head, or a non-greedy sampler, stays silent.
    //
    // Worth printing at all because a broken draft path is invisible in the
    // text: verify overrides every rejected draft with the base model's own
    // argmax, so bad drafting costs only speed.
    if (auto* mtp = dynamic_cast<qwen3_8mtp_npu*>(this->lm_engine.get())) {
        mtp->report_speculation_stats();
    }

    return result;
}

std::string Qwen3_8MTP::generate_with_prompt(chat_meta_info_t& meta_info, lm_uniform_input_t& input, int length_limit, std::ostream& os) {
    if (!this->insert(meta_info, input)) {
        return "";
    }
    header_print("FLM", "Prompt inserted, starting generation...");
    if (this->enable_think) {
        os << "<think>\n" << std::flush;
    }
    return this->_shared_generate(meta_info, length_limit, os);
}

// Non-stream
NonStreamResult Qwen3_8MTP::parse_nstream_content(const std::string response_text) {
    NonStreamResult result;

    // Qwen3.5-style tool syntax:
    //   <tool_call><function=NAME><parameter=KEY>VALUE</parameter></function></tool_call>
    std::string start_tag = "<tool_call>";
    std::string end_tag = "</tool_call>";
    std::string func_end_tag = "</function>";
    std::string func_open = "<function=";
    std::string param_open = "<parameter=";
    std::string param_close = "</parameter>";

    auto trim_tool_value = [](std::string value) {
        while (!value.empty() && (value.front() == '\n' || value.front() == '\r' || value.front() == ' ' || value.front() == '\t')) {
            value.erase(0, 1);
        }
        while (!value.empty() && (value.back() == '\n' || value.back() == '\r' || value.back() == ' ' || value.back() == '\t')) {
            value.pop_back();
        }
        return value;
    };

    const std::string think_start_tag = "<think>";
    const std::string think_end_tag = "</think>";
    size_t think_start_pos = response_text.find(think_start_tag);
    size_t think_end_pos = response_text.find(think_end_tag);
    bool is_reasoning = (think_start_pos != std::string::npos && think_end_pos != std::string::npos);
    if (is_reasoning) {
        size_t start = think_start_pos + think_start_tag.length();
        result.reasoning_content = response_text.substr(start, think_end_pos - start);
    }

    size_t search_from = 0;

    while (true) {
        size_t start_pos = response_text.find(start_tag, search_from);
        if (start_pos == std::string::npos) break;

        size_t block_content_start = start_pos + start_tag.length();
        size_t end_pos = response_text.find(end_tag, block_content_start);

        size_t block_end;
        if (end_pos != std::string::npos) {
            block_end = end_pos;
            search_from = end_pos + end_tag.length();
        } else {
            // Unclosed tag — search for </function> fallback
            size_t func_end_pos = response_text.find(func_end_tag, block_content_start);
            if (func_end_pos != std::string::npos) {
                block_end = func_end_pos + func_end_tag.length();
            } else {
                block_end = response_text.length();
            }
            search_from = block_end;
        }

        std::string block = response_text.substr(block_content_start, block_end - block_content_start);

        std::string tool_name;
        size_t func_start = block.find(func_open);
        if (func_start != std::string::npos) {
            func_start += func_open.length();
            size_t func_name_end = block.find(">", func_start);
            if (func_name_end != std::string::npos) {
                tool_name = block.substr(func_start, func_name_end - func_start);
            }
        }

        nlohmann::json args = nlohmann::json::object();
        size_t pos = 0;

        while (true) {
            size_t param_start = block.find(param_open, pos);
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

            pos = value_end;
            if (block.compare(value_end, param_close.length(), param_close) == 0) {
                pos += param_close.length();
            }
        }

        result.tool_calls_list.emplace_back(tool_name, args.dump());
    }

    if (result.tool_calls_list.empty()) {
        size_t content_start = is_reasoning ? think_end_pos + think_end_tag.length() : 0;
        result.content = response_text.substr(content_start);
    } else {
        // Populate legacy single-tool fields from the first call for backward compatibility
        result.tool_name = result.tool_calls_list[0].first;
        result.tool_args = result.tool_calls_list[0].second;
        // Extract content before the first <tool_call>
        size_t first_tool = response_text.find(start_tag);
        size_t content_start = is_reasoning ? think_end_pos + think_end_tag.length() : 0;
        if (first_tool != std::string::npos && first_tool > content_start) {
            result.content = trim_tool_value(response_text.substr(content_start, first_tool - content_start));
        }
    }

    return result;
}

// Stream
StreamResult Qwen3_8MTP::parse_stream_content(const std::string content) {
    return parse_stream_content_impl(content, false);
}

StreamResult Qwen3_8MTP::parse_stream_content_final(const std::string content) {
    return parse_stream_content_impl(content, true);
}

StreamResult Qwen3_8MTP::parse_stream_content_impl(const std::string content, bool is_final) {
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
                            result.tool_name = block.substr(func_start, func_end - func_start);
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
