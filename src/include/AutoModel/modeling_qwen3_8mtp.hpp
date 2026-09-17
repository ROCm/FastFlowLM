/// \file modeling_qwen3_8mtp.hpp
/// \brief Qwen3_8MTP class
/// \author FastFlowLM Team
/// \date 2026-09-16
/// \version 0.9.28
/// \note AutoModel wrapper for Qwen3.8-27B (model_type qwen3_5): 64 layers,
///       48 GatedDeltaNet + 16 full attention, plus a 1-layer MTP draft head.
/// \note Text-only. The checkpoint ships a vision tower but model.q4nx carries
///       no vision weights, so images are rejected rather than silently
///       mis-prefilled. Modelled on Qwen3_5VL minus the image pipeline.
/// \note The think ids are 248068/248069, NOT the 151667/151668 that
///       modeling_qwen3.hpp hardcodes. Copying those gives a model whose
///       reasoning block never closes.

#pragma once
#include "AutoModel/automodel.hpp"


/************              Qwen3_8MTP            **************/
class Qwen3_8MTP : public AutoModel {
private:

    bool enable_think = false;
    bool enable_tool = true;

    /// \note Qwen3.8 vocabulary, verified against tokenizer.json:
    ///       <think> 248068, </think> 248069. The chat template emits
    ///       "<think>\n" when thinking is on and "<think>\n\n</think>\n\n"
    ///       when it is off, exactly as Qwen3.5 does.
    int think_start_id = 248068;
    int think_end_id = 248069;

    void setup_tokenizer(std::string model_path);

public:
    Qwen3_8MTP(flm_rt::device* npu_device_inst);

    void load_model(std::string model_path, json model_inf, int default_context_length = -1, bool enable_preemption = false) override;
    bool insert(chat_meta_info_t& meta_info, lm_uniform_input_t& input, std::function<bool()> is_cancelled = [] { return false; }) override;
    std::string generate(chat_meta_info_t& meta_info, int length_limit, std::ostream& os, std::function<bool()> is_cancelled = [] { return false; }) override;
    std::string generate_with_prompt(chat_meta_info_t& meta_info, lm_uniform_input_t& input, int length_limit, std::ostream& os = std::cout) override;
    std::string apply_chat_template(nlohmann::ordered_json& messages, nlohmann::ordered_json tools = nlohmann::ordered_json::object()) override;
    NonStreamResult parse_nstream_content(const std::string response_text);
    StreamResult parse_stream_content(const std::string content);
    StreamResult parse_stream_content_final(const std::string content) override;

private:
    StreamResult parse_stream_content_impl(const std::string content, bool is_final);

public:

    /// \brief Configure a parameter with type-erased value
    /// \param parameter_name the name of the parameter
    /// \param value the value to set (can be any type)
    /// \return true if the parameter was configured successfully, false otherwise
    bool configure_parameter(std::string parameter_name, const std::any& value) override {
        if (parameter_name == "enable_think") {
            try {
                this->enable_think = std::any_cast<bool>(value);
                return true;
            } catch (const std::bad_any_cast&) {
                return false;
            }
        }
        else if (parameter_name == "reasoning_effort") {
            std::string reasoning_effort;
            try {
                reasoning_effort = std::any_cast<std::string>(value);
                if (reasoning_effort == "high" || reasoning_effort == "medium" || reasoning_effort == "low")
                    this->enable_think = true;
                else if (reasoning_effort == "none")
                    this->enable_think = false;
                else
                    header_print("WARNING", "Reasoning effort must be 'none', 'low', 'medium' or 'high'!");
                return true;
            } catch (const std::bad_any_cast&) {
                return false;
            }
        }
        else if (parameter_name == "toggle_think") {
            this->enable_think = !this->enable_think;
            return true;
        }
        else if (parameter_name == "system_prompt") {
            try {
                this->user_system_prompt = std::any_cast<std::string>(value);
                this->extra_context["user_system_prompt"] = this->user_system_prompt;
                return true;
            } catch (const std::bad_any_cast&) {
                return false;
            }
        }
        return AutoModel::configure_parameter(parameter_name, value);
    }
};
