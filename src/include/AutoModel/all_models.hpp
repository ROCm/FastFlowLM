/// \file all_models.hpp
/// \brief all_models class
/// \author FastFlowLM Team
/// \date 2025-09-10
/// \version 0.9.24
/// \note This is a header file for the all_models class
#pragma once

#include "AutoModel/automodel.hpp"
#include "AutoModel/model_frontend.hpp"
#include "model_list.hpp"

inline std::pair<std::string, flm::frontend::FrontendPtr> get_auto_model(
    const std::string& model_tag, model_list& available_models,
    flm_rt::device* npu_device_inst) {
    if (available_models.is_model_supported(model_tag) == false) {
        header_print_r("ERROR", "Model tag '" << model_tag << "' is not supported. Please check the model list.");
        // Fall through: get_model_info resolves the platform's fallback entry,
        // and the frontend must come from that entry's family, not from a
        // literal tag that may be pruned on this platform.
    }

    auto [new_model_tag, model_info] = available_models.get_model_info(model_tag);
    std::string family = model_info["details"]["family"];

    auto& registry = flm::frontend::FrontendRegistry::instance();
    if (!registry.has(family)) {
        // These built-in non-generation families use the CLI fallback. Any
        // other unregistered family remains an error so a missing extension
        // cannot silently select another model.
        if (family == "whisper-v3" || family == "embed-gemma" ||
            family == "d3") {
            header_print_r("ERROR", "Unsupported model family or non-llm: "
                                        << family);
            family = "llama3";
            new_model_tag = "llama3.2:1b";
        }
    }

    return {new_model_tag, registry.create(family, npu_device_inst)};
}
