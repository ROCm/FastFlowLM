/// \file modeling_inkling2.hpp
/// \brief Inkling2 class
/// \author FastFlowLM Team
/// \date 2026-11-21
/// \note Portions of this file consist of AI-generated content.
/// \note AutoModel wrapper for the Inkling-2-Mini engine (inkling2_npu).
///       A turn is a run of messages, <|message_model|>[name]<|content_KIND|>body<|end_message|>,
///       ended by <|content_model_end_sampling|>; the model always opens with a thinking
///       message, whose length the system line "Thinking effort level: x" steers.
///       reasoning_effort: none / minimal / low / medium / high (default) / xhigh / max,
///       or a number in [0, 0.99]. Thinking off means effort "none" (level 0), which
///       still thinks, briefly. Language only: images and audios are dropped.

#pragma once
#include "AutoModel/automodel.hpp"

/************              Inkling 2 family            **************/
class Inkling2 : public AutoModel {
private:
    void setup_tokenizer(std::string model_path);

    /// \brief the template's reasoning_effort: a level name, or a number as text
    std::string reasoning_effort = "high";

    /// \brief whether `effort` is a level name or a number in [0, 0.99]
    static bool valid_effort(const std::string& effort);

public:
    Inkling2(flm_rt::device* npu_device_inst);

    void load_model(std::string model_path, json model_info, int default_context_length = -1, bool enable_preemption = false, const std::string& backend = "") override;
    bool insert(chat_meta_info_t& meta_info, lm_uniform_input_t& input, std::function<bool()> is_cancelled = [] { return false; }) override;
    std::string generate(chat_meta_info_t& meta_info, int length_limit, std::ostream& os, std::function<bool()> is_cancelled = [] { return false; }) override;
    std::string generate_with_prompt(chat_meta_info_t& meta_info, lm_uniform_input_t& input, int length_limit, std::ostream& os = std::cout) override;
    std::string apply_chat_template(nlohmann::ordered_json& messages, nlohmann::ordered_json tools = nlohmann::ordered_json::object(), bool continue_final_message = false) override;
    bool supports_continue_final_message() const override {
        return true;
    }

    /// \brief request settings the model loaded with (see AutoModel::save_request_settings)
    std::string loaded_reasoning_effort = "high";
    void save_model_request_settings() override {
        loaded_reasoning_effort = this->reasoning_effort;
    }
    void restore_model_request_settings() override {
        this->configure_parameter("reasoning_effort", loaded_reasoning_effort);
    }
    /// \brief enable_think (false = effort "none"), toggle_think, reasoning_effort, plus the base-class ones
    bool configure_parameter(std::string parameter_name, const std::any& value) override;
    using AutoModel::configure_parameter;   // keep the typed template overload visible
};
