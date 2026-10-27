/// \file modeling_hy_memory.cpp
/// \brief HyMemory class
/// \author FastFlowLM Team
/// \date 2026-10-26
/// \note AutoModel wrapper for the Hy-Memory-30B-A3B engine (hy_mem_npu).

#include "AutoModel/modeling_hy_memory.hpp"
#include "models/hy_mem/flm/aie2p/hy_mem_npu.hpp"

/************              Hy-Memory family            **************/
HyMemory::HyMemory(flm_rt::device* npu_device_inst) : AutoModel(npu_device_inst, "HyMemory") {}

void HyMemory::load_model(std::string model_path, json model_info, int default_context_length, bool enable_preemption, const std::string& backend) {
    this->_shared_load_backend(model_path, model_info, default_context_length, enable_preemption, backend);
    this->setup_tokenizer(model_path);
    this->sampler.reset();

    // Greedy, as the hy_memory engine calls it (temperature 0): its replies are
    // JSON that a parser reads, not prose. A REST request can still override this.
    sampler_config config;
    config.top_k = 1;
    config.top_p = 1.0f;
    config.min_p = 0.0f;
    config.temperature = 1.0f;
    config.rep_penalty = 1.0f;
    config.freq_penalty = 0.0f;
    config.pre_penalty = 0.0f;

    this->set_sampler(config);
    for (size_t i = 0; i < PROFILER_TYPE_NUM; i++) {
        this->profiler_list[i].reset();
    }
}

void HyMemory::setup_tokenizer(std::string model_path) {
    auto tokenizer_config = this->_shared_setup_tokenizer(model_path);
}

/// \note The template reads `reasoning_effort` ("high" | "low" | "no_think", default
///       "no_think") and, when there are no tools, appends
///       <｜reasoning_mode｜>reasoning_effort:<value> to the system prompt.
std::string HyMemory::apply_chat_template(nlohmann::ordered_json& messages, nlohmann::ordered_json tools) {
    minja::chat_template_inputs inputs;
    inputs.add_generation_prompt = true;
    inputs.messages = messages;
    if (tools.is_array() && !tools.empty()) {
        inputs.tools = tools;
    }
    inputs.extra_context = this->extra_context;
    inputs.extra_context["reasoning_effort"] = this->reasoning_effort;
    return this->chat_tmpl->apply(inputs);
}

bool HyMemory::configure_parameter(std::string parameter_name, const std::any& value) {
    if (parameter_name == "reasoning_effort") {
        try {
            const std::string effort = std::any_cast<std::string>(value);
            if (effort != "no_think" && effort != "low" && effort != "high") {
                return false;
            }
            this->reasoning_effort = effort;
            return true;
        } catch (const std::bad_any_cast&) {
            return false;
        }
    }
    return this->AutoModel::configure_parameter(parameter_name, value);
}

bool HyMemory::insert(chat_meta_info_t& meta_info, lm_uniform_input_t& input, std::function<bool()> is_cancelled) {
    if (input.messages.empty() && input.prompt.empty()) {
        header_print("WARNING", "No messages or prompt provided");
        return false;
    }

    // Language only: drop any multimodal payload rather than failing the request.
    if (!input.images.empty()) {
        header_print("WARNING", "Hy-Memory-30B-A3B is a text-only model, ignoring " << input.images.size() << " image(s)");
        input.images.clear();
        input.image_payload_types.clear();
    }
    if (!input.audios.empty()) {
        header_print("WARNING", "Hy-Memory-30B-A3B is a text-only model, ignoring " << input.audios.size() << " audio input(s)");
        input.audios.clear();
        input.audio_payload_types.clear();
    }

    nlohmann::ordered_json messages = nlohmann::ordered_json::array();
    if (!input.messages.empty()) {   // already formatted messages, usually from the REST API
        for (auto& message : input.messages) {
            message.erase("images");
            message.erase("audios");
        }
        messages = input.messages;
    }
    else {                           // plain text, usually from the cli
        messages.push_back({ {"role", "user"}, {"content", input.prompt} });
    }

    this->profiler_list[TKOEN_ENCODE_TIME].start();
    std::string templated_text = this->apply_chat_template(messages, input.tools);
    // the template already starts with <bos>; encode() adds no second one
    std::vector<int> tokens = this->tokenizer->encode(templated_text);
    this->profiler_list[TKOEN_ENCODE_TIME].stop(tokens.size());

    return this->_shared_insert(meta_info, tokens, is_cancelled, nullptr);
}

std::string HyMemory::generate(chat_meta_info_t& meta_info, int length_limit, std::ostream& os, std::function<bool()> is_cancelled) {
    return this->_shared_generate(meta_info, length_limit, os, is_cancelled);
}

std::string HyMemory::generate_with_prompt(chat_meta_info_t& meta_info, lm_uniform_input_t& input, int length_limit, std::ostream& os) {
    if (!this->insert(meta_info, input)) {
        return "";
    }
    return this->generate(meta_info, length_limit, os);
}
