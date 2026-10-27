/// \file modeling_hy_memory.hpp
/// \brief HyMemory class
/// \author FastFlowLM Team
/// \date 2026-10-26
/// \note AutoModel wrapper for the Hy-Memory-30B-A3B engine (hy_mem_npu).
///       Hy-Memory is not a chat model in the usual sense: it is the LLM behind the
///       hy_memory engine, which sends it one-shot extract / reconcile prompts and
///       parses a JSON object or array out of the reply. Each request is a whole
///       conversation (the base class keeps only the token prefix it shares with
///       the previous one), decoding defaults to greedy (the engine's temperature
///       0), and the chat template's reasoning_effort defaults to "no_think",
///       which makes the prompt end in <｜hy_Assistant｜><think></think>.
///       Language only: images and audios on the uniform input are dropped.

#pragma once
#include "AutoModel/automodel.hpp"

/************              Hy-Memory family            **************/
class HyMemory : public AutoModel {
private:
    void setup_tokenizer(std::string model_path);

    /// \brief chat template variable: "no_think" (default), "low" or "high"
    std::string reasoning_effort = "no_think";

public:
    HyMemory(flm_rt::device* npu_device_inst);

    void load_model(std::string model_path, json model_info, int default_context_length = -1, bool enable_preemption = false, const std::string& backend = "") override;
    bool insert(chat_meta_info_t& meta_info, lm_uniform_input_t& input, std::function<bool()> is_cancelled = [] { return false; }) override;
    std::string generate(chat_meta_info_t& meta_info, int length_limit, std::ostream& os, std::function<bool()> is_cancelled = [] { return false; }) override;
    std::string generate_with_prompt(chat_meta_info_t& meta_info, lm_uniform_input_t& input, int length_limit, std::ostream& os = std::cout) override;
    std::string apply_chat_template(nlohmann::ordered_json& messages, nlohmann::ordered_json tools = nlohmann::ordered_json::object()) override;

    /// \brief "reasoning_effort" (std::string: no_think / low / high), plus the base-class ones
    bool configure_parameter(std::string parameter_name, const std::any& value) override;
    using AutoModel::configure_parameter;   // keep the typed template overload visible
};
