/// \file modeling_template_chat.hpp
/// \brief Text chat models that only run on OGA hip-ep: Laguna 2.1, Nemotron 3.5 Lightning, Step 3.5 Flash
/// \author FastFlowLM Team
#pragma once

#include "AutoModel/automodel.hpp"

/// \brief A text chat model whose package says everything about the prompt: its chat
///        template renders the turns, the tools and the reasoning switch, and the OGA
///        backend runs it
/// \note No NPU engine exists for these families, so there is no state API for prefix
///       slots; OGA keeps its own checkpoint.
class TemplateChat : public AutoModel {
private:
    sampler_config sampling_;
    bool think_toggleable_;
    bool enable_think;

    bool _insert_checkpointing_reply_header(chat_meta_info_t& meta_info, std::vector<int>& tokens,
                                            std::function<bool()> is_cancelled, std::optional<int> requested_max_new_tokens);

protected:
    /// \brief Checkpoint each prompt at its reply header, before the reasoning block the
    ///        template opens there, so the next turn can rewind to it
    bool checkpoint_reply_header_ = false;

public:
    /// \param sampling the model card's recommended sampling
    /// \param think_toggleable false for a template that always opens a reasoning block
    TemplateChat(flm_rt::device* npu_device_inst, const std::string& current_model, const sampler_config& sampling,
                 bool think_toggleable);

    void load_model(std::string model_path, json model_inf, int default_context_length = -1, bool enable_preemption = false, const std::string& backend = "") override;
    bool insert(chat_meta_info_t& meta_info, lm_uniform_input_t& input, std::function<bool()> is_cancelled = [] { return false; }) override;
    std::string generate(chat_meta_info_t& meta_info, int length_limit, std::ostream& os, std::function<bool()> is_cancelled = [] { return false; }) override;
    std::string generate_with_prompt(chat_meta_info_t& meta_info, lm_uniform_input_t& input, int length_limit, std::ostream& os = std::cout) override;
    std::string apply_chat_template(nlohmann::ordered_json& messages, nlohmann::ordered_json tools = nlohmann::ordered_json::object(), bool continue_final_message = false) override;
    bool supports_continue_final_message() const override { return true; }

    bool thinking_enabled() const override {
        return this->enable_think;
    }
    bool configure_parameter(std::string parameter_name, const std::any& value) override;
};

/// \brief poolside Laguna-XS-2.1 and Laguna-S-2.1; GLM-style tool calls
class Laguna : public TemplateChat {
public:
    explicit Laguna(flm_rt::device* npu_device_inst);
};

/// \brief NVIDIA Nemotron 3.5 Lightning 30B-A3B, a Mamba/attention hybrid; Qwen XML tool calls
class NemotronH : public TemplateChat {
public:
    explicit NemotronH(flm_rt::device* npu_device_inst);
};

/// \brief StepFun Step-3.5-Flash; its template always opens a reasoning block; Qwen XML tool calls
class Step3_5 : public TemplateChat {
public:
    explicit Step3_5(flm_rt::device* npu_device_inst);
};
