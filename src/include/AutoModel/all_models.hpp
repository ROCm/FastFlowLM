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
#include "utils/utils.hpp"
#include "AutoEmbeddingModel/all_embedding_model.hpp"

struct ModelLoadPlan {
    std::string tag;
    std::string family;
    nlohmann::json info;
    flm::backend::BackendSelection selection;

    flm_rt::device* prepare_device(flm_rt::device* shared, const std::string& power_mode) const {
        const auto traits = flm::backend::BackendRegistry::instance().traits(family, selection.backend, selection.provider);
        return prepare_backend_device(traits, [&] { return utils::require_npu_device(shared, power_mode); });
    }
};

inline ModelLoadPlan resolve_model_load(const std::string& model_tag, model_list& available_models,
                                        const std::string& requested_backend) {
    if (available_models.is_model_supported(model_tag) == false) {
        header_print_r("ERROR", "Model tag '" << model_tag << "' is not supported. Please check the model list.");
        // Fall through: get_model_info resolves the platform's fallback entry,
        // and the frontend must come from that entry's family, not from a
        // literal tag that may be pruned on this platform.
    }

    auto [new_model_tag, model_info] = available_models.get_model_info(model_tag);
    if (is_embedding_model(model_info))
        throw std::invalid_argument("Model '" + new_model_tag + "' is an embedding model; use 'flm serve' for embeddings");
    std::string family = model_info["details"]["family"];

    auto& registry = flm::frontend::FrontendRegistry::instance();
    if (!registry.has(family)) {
        // These built-in non-generation families use the CLI fallback. Any
        // other unregistered family remains an error so a missing extension
        // cannot silently select another model.
        if (family == "whisper-v3" || family == "d3") {
            header_print_r("ERROR", "Unsupported model family or non-llm: "
                                        << family);
            auto fallback = available_models.get_model_info("llama3.2:1b");
            new_model_tag = std::move(fallback.first);
            model_info = std::move(fallback.second);
            family = model_info.at("details").at("family").get<std::string>();
        }
        if (!registry.has(family)) throw std::runtime_error("No frontend for model family '" + family + "'");
    }
    auto selection = flm::backend::resolve_model_backend(model_info, requested_backend);
    return {std::move(new_model_tag), std::move(family), std::move(model_info), std::move(selection)};
}

inline std::pair<std::string, flm::frontend::FrontendPtr> get_auto_model(
    const std::string& model_tag, model_list& available_models,
    flm_rt::device* npu_device_inst, const std::string& requested_backend = "",
    const std::string& power_mode = "performance") {
    auto plan = resolve_model_load(model_tag, available_models, requested_backend);
    auto* device = plan.prepare_device(npu_device_inst, power_mode);
    return {plan.tag, flm::frontend::FrontendRegistry::instance().create(plan.family, device, plan.info.value("frontend_provider", ""))};
}
