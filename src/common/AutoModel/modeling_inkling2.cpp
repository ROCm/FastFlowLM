/// \file modeling_inkling2.cpp
/// \brief Inkling2 class
/// \author FastFlowLM Team
/// \date 2026-11-21
/// \note Portions of this file consist of AI-generated content.
/// \note AutoModel wrapper for the Inkling-2-Mini engine (inkling2_npu).

#include "AutoModel/modeling_inkling2.hpp"
#include "models/inkling2/flm/aie2p/inkling2_npu.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <stdexcept>

namespace {

/// The tool declaration as the reference template writes it: keys sorted, no spaces.
/// minja's tojson has neither sort_keys nor separators, so the package's template
/// prints this string instead of building the list itself.
std::string tool_specs_json(const nlohmann::ordered_json& tools) {
    auto field_or = [](const nlohmann::ordered_json& o, const char* key, nlohmann::json fallback) {
        if (!o.contains(key)) return fallback;
        const auto& v = o[key];
        const bool empty = v.is_null() || (v.is_string() && v.get<std::string>().empty()) || (v.is_object() && v.empty());
        return empty ? fallback : nlohmann::json(v);
    };
    nlohmann::json specs = nlohmann::json::array();
    for (const auto& tool : tools) {
        const nlohmann::ordered_json& fn = tool.contains("function") ? tool["function"] : tool;
        specs.push_back({{"description", field_or(fn, "description", "")},
                         {"name", fn.value("name", nlohmann::ordered_json())},
                         {"parameters", field_or(fn, "parameters", nlohmann::json::object())},
                         {"type", field_or(tool, "type", "function")}});
    }
    return specs.dump();   // nlohmann::json keeps its object keys sorted
}

/// One past call as the model writes it: {"name":..,"args":{..}}, the arguments' top-level
/// keys sorted and their values compact, as given. Arguments may come as a JSON string.
/// \throws std::invalid_argument when the arguments are not an object
std::string call_json(const std::string& name, const nlohmann::ordered_json& arguments) {
    nlohmann::ordered_json args = arguments;
    if (args.is_string()) {
        const std::string text = args.get<std::string>();
        try {
            args = text.empty() ? nlohmann::ordered_json::object() : nlohmann::ordered_json::parse(text);
        } catch (const nlohmann::json::parse_error&) {
            throw std::invalid_argument("tool call arguments of " + name + " are not valid JSON");
        }
    }
    if (args.is_null()) args = nlohmann::ordered_json::object();
    if (!args.is_object()) throw std::invalid_argument("tool call arguments of " + name + " must be an object");
    std::vector<std::string> keys;
    for (const auto& item : args.items()) keys.push_back(item.key());
    std::sort(keys.begin(), keys.end());
    std::string out = "{\"name\":" + nlohmann::ordered_json(name).dump() + ",\"args\":{";
    for (size_t i = 0; i < keys.size(); i++) {
        out += (i ? "," : "") + nlohmann::ordered_json(keys[i]).dump() + ":" + args[keys[i]].dump();
    }
    return out + "}}";
}

}  // namespace

/************              Inkling 2 family            **************/
Inkling2::Inkling2(flm_rt::device* npu_device_inst) : AutoModel(npu_device_inst, "Inkling2") {}

void Inkling2::load_model(std::string model_path, json model_info, int default_context_length, bool enable_preemption, const std::string& backend) {
    this->_shared_load_backend(model_path, model_info, default_context_length, enable_preemption, backend);
    this->setup_tokenizer(model_path);
    this->sampler.reset();

    // generation_config.json: plain sampling at temperature 1.0, no top-k / top-p
    sampler_config config;
    config.top_k = 0;
    config.top_p = 1.0f;
    config.min_p = 0.0f;
    config.temperature = 1.0f;
    config.rep_penalty = 1.0f;
    this->set_sampler(config);
    for (size_t i = 0; i < PROFILER_TYPE_NUM; i++) {
        this->profiler_list[i].reset();
    }
    this->_prefix_attach(this->lm_engine, inkling2_npu_state_api(this->lm_engine));
}

void Inkling2::setup_tokenizer(std::string model_path) {
    auto tokenizer_config = this->_shared_setup_tokenizer(model_path);
}

bool Inkling2::valid_effort(const std::string& effort) {
    static const char* names[] = {"none", "minimal", "low", "medium", "high", "xhigh", "max"};
    for (const char* name : names) {
        if (effort == name) return true;
    }
    char* end = nullptr;
    const double level = std::strtod(effort.c_str(), &end);
    return !effort.empty() && end && *end == '\0' && level >= 0.0 && level <= 0.99;
}

/// \note The package's chat_template.jinja is the reference template made to run in minja:
///       it prints `inkling_tool_specs` and each call's `inkling_call` (built here) where
///       the reference builds them with tojson(sort_keys=true, separators=...).
std::string Inkling2::apply_chat_template(nlohmann::ordered_json& messages, nlohmann::ordered_json tools, bool continue_final_message) {
    minja::chat_template_inputs inputs;
    minja::chat_template_options opt;
    opt.polyfill_tool_responses = false;
    inputs.extra_context = this->extra_context;
    // a number goes in as a number: the template prints it ("Thinking effort level: 0.5")
    if (std::isdigit(static_cast<unsigned char>(this->reasoning_effort[0]))) {
        inputs.extra_context["reasoning_effort"] = std::strtod(this->reasoning_effort.c_str(), nullptr);
    } else {
        inputs.extra_context["reasoning_effort"] = this->reasoning_effort;
    }
    nlohmann::ordered_json prepared = messages;
    try {
        for (auto& message : prepared) {
            if (!message.contains("tool_calls") || !message["tool_calls"].is_array()) continue;
            for (auto& call : message["tool_calls"]) {
                const nlohmann::ordered_json& fn = call.contains("function") ? call["function"] : call;
                call["inkling_call"] = call_json(fn.value("name", ""), fn.contains("arguments") ? fn["arguments"] : nlohmann::ordered_json());
            }
        }
    } catch (const std::invalid_argument& e) {
        throw ModelRequestError(400, false, e.what(), "invalid_request_error");
    }
    if (tools.is_array() && !tools.empty()) {
        inputs.tools = tools;
        inputs.extra_context["inkling_tool_specs"] = tool_specs_json(tools);
    }
    inputs.messages = prepared;
    return this->_render_chat_template(inputs, continue_final_message, opt);
}

bool Inkling2::configure_parameter(std::string parameter_name, const std::any& value) {
    if (parameter_name == "reasoning_effort") {
        try {
            const std::string effort = std::any_cast<std::string>(value);
            if (!valid_effort(effort)) {
                header_print("WARNING", "Reasoning effort must be none, minimal, low, medium, high, xhigh, max or a number in [0, 0.99]!");
                return true;
            }
            this->reasoning_effort = effort;
            return true;
        } catch (const std::bad_any_cast&) {
            return false;
        }
    }
    if (parameter_name == "enable_think") {
        try {
            const bool on = std::any_cast<bool>(value);
            if (!on) {
                this->reasoning_effort = "none";
            } else if (this->reasoning_effort == "none") {
                this->reasoning_effort = "high";
            }
            return true;
        } catch (const std::bad_any_cast&) {
            return false;
        }
    }
    if (parameter_name == "toggle_think") {
        this->reasoning_effort = this->reasoning_effort == "none" ? "high" : "none";
        header_print("FLM", "Thinking effort: " << this->reasoning_effort);
        return true;
    }
    return this->AutoModel::configure_parameter(parameter_name, value);
}

bool Inkling2::insert(chat_meta_info_t& meta_info, lm_uniform_input_t& input, std::function<bool()> is_cancelled) {
    if (input.messages.empty() && input.prompt.empty()) {
        header_print("WARNING", "No messages or prompt provided");
        return false;
    }

    // Language only: drop any multimodal payload rather than failing the request.
    if (!input.images.empty()) {
        header_print("WARNING", "Inkling-2-Mini runs text only here, ignoring " << input.images.size() << " image(s)");
        input.images.clear();
        input.image_payload_types.clear();
    }
    if (!input.audios.empty()) {
        header_print("WARNING", "Inkling-2-Mini is a text-only model, ignoring " << input.audios.size() << " audio input(s)");
        input.audios.clear();
        input.audio_payload_types.clear();
    }

    this->profiler_list[TKOEN_ENCODE_TIME].start();
    std::string templated_text;
    if (!input.messages.empty()) {   // already formatted messages, usually from the REST API
        for (auto& message : input.messages) {
            message.erase("images");
            message.erase("audios");
        }
        templated_text = this->apply_chat_template(input.messages, input.tools, meta_info.continue_final_message);
    }
    else {                           // plain text, usually from the cli
        nlohmann::ordered_json messages = nlohmann::ordered_json::array();
        messages.push_back({ {"role", "user"}, {"content", input.prompt} });
        templated_text = this->apply_chat_template(messages);
    }
    std::vector<int> tokens = this->tokenizer->encode(templated_text);
    this->profiler_list[TKOEN_ENCODE_TIME].stop(tokens.size());

    // Prefix slots. The prompt ends with the user's <|end_message|>, which is also how the next
    // turn renders it, so the whole prompt is saved (trim 0). The tool declaration and the effort
    // line come before the first <|message_user|>: that head gets its own snapshot.
    if (this->_prefix_slot_mode(meta_info, tokens.size(), 0)) {
        return this->_prefix_text_insert(meta_info, tokens, 0, "<|message_user|>", is_cancelled);
    }
    this->_prefix_legacy_path();

    return this->_shared_insert(meta_info, tokens, is_cancelled, nullptr);
}

std::string Inkling2::generate(chat_meta_info_t& meta_info, int length_limit, std::ostream& os, std::function<bool()> is_cancelled) {
    return this->_shared_generate(meta_info, length_limit, os, is_cancelled);
}

std::string Inkling2::generate_with_prompt(chat_meta_info_t& meta_info, lm_uniform_input_t& input, int length_limit, std::ostream& os) {
    if (!this->insert(meta_info, input)) {
        return "";
    }
    return this->generate(meta_info, length_limit, os);
}
