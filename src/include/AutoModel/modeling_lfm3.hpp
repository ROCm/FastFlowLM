/// \file modeling_lfm3.hpp
/// \brief LFM3 class
/// \note LiquidAI LFM3 (conv + sliding-window + full attention), ChatML template with a thinking switch.

#pragma once
#include "AutoModel/automodel.hpp"

/************              LFM3 family            **************/
class LFM3 : public AutoModel {
private:
    /// the template opens a reasoning block (`<think>`) by default; turning it off renders `<think></think>`
    bool enable_think = true;

    void setup_tokenizer(std::string model_path);

public:
    LFM3(flm_rt::device* npu_device_inst);

    void load_model(std::string model_path, json model_inf, int default_context_length = -1, bool enable_preemption = false, const std::string& backend = "") override;
    bool insert(chat_meta_info_t& meta_info, lm_uniform_input_t& input, std::function<bool()> is_cancelled = [] { return false; }) override;
    std::string generate(chat_meta_info_t& meta_info, int length_limit, std::ostream& os, std::function<bool()> is_cancelled = [] { return false; }) override;
    std::string generate_with_prompt(chat_meta_info_t& meta_info, lm_uniform_input_t& input, int length_limit, std::ostream& os = std::cout) override;
    std::string apply_chat_template(nlohmann::ordered_json& messages, nlohmann::ordered_json tools = nlohmann::ordered_json::object(), bool continue_final_message = false) override;
    bool thinking_enabled() const override {
        return this->enable_think;
    }
    bool supports_continue_final_message() const override {
        return true;
    }

    /// \brief enable_think / toggle_think / reasoning_effort ("none" turns thinking off) and everything the base class handles
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
        else if (parameter_name == "toggle_think") {
            this->enable_think = !this->enable_think;
            return true;
        }
        return AutoModel::configure_parameter(parameter_name, value);
    }
};
