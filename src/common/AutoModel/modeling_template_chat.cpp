/// \file modeling_template_chat.cpp
/// \brief Text chat models that only run on OGA hip-ep: Laguna 2.1, Nemotron 3.5 Lightning
/// \author FastFlowLM Team

#include "AutoModel/modeling_template_chat.hpp"

namespace {

sampler_config recommended_sampling(float temperature, float top_p, int top_k) {
    sampler_config config;
    config.temperature = temperature;
    config.top_p = top_p;
    config.top_k = top_k;
    config.min_p = 0.0f;
    return config;
}

} // namespace

TemplateChat::TemplateChat(flm_rt::device* npu_device_inst, const std::string& current_model,
                           const sampler_config& sampling)
    : AutoModel(npu_device_inst, current_model), sampling_(sampling) {}

void TemplateChat::load_model(std::string model_path, json model_info, int default_context_length, bool enable_preemption, const std::string& backend) {
    this->_shared_load_backend(model_path, model_info, default_context_length, enable_preemption, backend);
    this->_shared_setup_tokenizer(model_path);
    this->sampler.reset();
    sampler_config config = this->sampling_;
    this->set_sampler(config);
    for (size_t i = 0; i < PROFILER_TYPE_NUM; i++) {
        this->profiler_list[i].reset();
    }
    this->_prefix_attach(this->lm_engine, nullptr);
}

std::string TemplateChat::apply_chat_template(nlohmann::ordered_json& messages, nlohmann::ordered_json tools, bool continue_final_message) {
    minja::chat_template_inputs inputs;
    inputs.messages = messages;
    inputs.extra_context = this->extra_context;
    inputs.extra_context["enable_thinking"] = this->enable_think;
    if (!tools.empty()) {
        inputs.tools = tools;
    }
    return this->_render_chat_template(inputs, continue_final_message);
}

bool TemplateChat::insert(chat_meta_info_t& meta_info, lm_uniform_input_t& input, std::function<bool()> is_cancelled) {
    this->profiler_list[TKOEN_ENCODE_TIME].start();
    if (input.messages.empty() && input.prompt.empty()) {
        header_print("WARNING", "No messages or prompt provided");
        return false;
    }
    // A ModelRequestError: the handlers turn only that into an HTTP 400.
    bool images = !input.images.empty();
    for (const auto& message : input.messages) {
        images = images || (message.contains("images") && !message["images"].empty());
    }
    if (images) {
        throw ModelRequestError(400, false, this->current_model + " does not accept image input", "invalid_request_error");
    }
    std::string templated_text;
    if (!input.messages.empty()) {
        templated_text = this->apply_chat_template(input.messages, input.tools, meta_info.continue_final_message);
    }
    else {
        nlohmann::ordered_json messages;
        messages.push_back({ {"role", "user"}, {"content", input.prompt} });
        templated_text = this->apply_chat_template(messages);
    }
    std::vector<int> tokens = this->tokenizer->encode(templated_text);
    this->profiler_list[TKOEN_ENCODE_TIME].stop(tokens.size());

    this->_prefix_legacy_path();
    return this->_shared_insert(meta_info, tokens, is_cancelled);
}

std::string TemplateChat::generate(chat_meta_info_t& meta_info, int length_limit, std::ostream& os, std::function<bool()> is_cancelled) {
    return this->_shared_generate(meta_info, length_limit, os, is_cancelled);
}

std::string TemplateChat::generate_with_prompt(chat_meta_info_t& meta_info, lm_uniform_input_t& input, int length_limit, std::ostream& os) {
    if (!this->insert(meta_info, input)) {
        return "";
    }
    return this->_shared_generate(meta_info, length_limit, os);
}

bool TemplateChat::configure_parameter(std::string parameter_name, const std::any& value) {
    if (parameter_name == "enable_think") {
        try {
            this->enable_think = std::any_cast<bool>(value);
            return true;
        } catch (const std::bad_any_cast&) {
            return false;
        }
    }
    if (parameter_name == "reasoning_effort") {
        try {
            const std::string effort = std::any_cast<std::string>(value);
            if (effort == "high" || effort == "medium" || effort == "low")
                this->enable_think = true;
            else if (effort == "none")
                this->enable_think = false;
            else
                header_print("WARNING", "Reasoning effort must be 'none', 'low', 'medium' or 'high'!");
            return true;
        } catch (const std::bad_any_cast&) {
            return false;
        }
    }
    if (parameter_name == "toggle_think") {
        this->enable_think = !this->enable_think;
        return true;
    }
    return AutoModel::configure_parameter(parameter_name, value);
}

// generation_config.json: temperature 1.0, top_p 1.0, top_k 20
Laguna::Laguna(flm_rt::device* npu_device_inst)
    : TemplateChat(npu_device_inst, "Laguna", recommended_sampling(1.0f, 1.0f, 20)) {}

// generation_config.json: temperature 1.0, top_p 0.95; no top_k, so transformers' 50
NemotronH::NemotronH(flm_rt::device* npu_device_inst)
    : TemplateChat(npu_device_inst, "NemotronH", recommended_sampling(1.0f, 0.95f, 50)) {}
