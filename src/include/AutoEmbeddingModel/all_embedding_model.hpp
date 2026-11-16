/// \file all_embedding_models.hpp
/// \brief get_auto_embedding_model func
/// \author FastFlowLM Team
/// \date 2025-10-23
/// \version 0.9.24
/// \note This is a header file for get_auto_embedding_model func
#pragma once
#include "AutoModel/model_frontend.hpp"
#include "auto_embedding_model.hpp"

inline bool is_embedding_model(const nlohmann::json& info) {
    // Legacy built-in catalogs omit model_kind. New embedding entries must
    // declare model_kind="embedding" rather than rely on the family name.
    return flm::catalog::model_kind(info) == flm::catalog::kEmbeddingModelKind ||
           info.at("details").at("family") == "embed-gemma";
}

inline std::unique_ptr<AutoEmbeddingModel> get_auto_embedding_model(
    const nlohmann::json& info, flm_rt::device* device) {
    if (!is_embedding_model(info))
        throw std::invalid_argument("Model is not an embedding model");
    return flm::frontend::FrontendRegistry::instance().create_embedding(
        info.at("details").at("family"), device, info.value("frontend_provider", ""));
}
