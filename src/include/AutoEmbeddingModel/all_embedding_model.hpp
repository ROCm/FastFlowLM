/// \file all_embedding_models.hpp
/// \brief get_auto_embedding_model func
/// \author FastFlowLM Team
/// \date 2025-10-23
/// \version 0.9.24
/// \note This is a header file for get_auto_embedding_model func
#pragma once
#include "AutoModel/model_frontend.hpp"
#include "auto_embedding_model.hpp"

#ifndef FASTFLOWLM_LINUX_LIMITED_MODELS
#include "modeling_gemma_embedding.hpp"
#endif

inline bool is_embedding_model(const nlohmann::json& info) {
    return flm::catalog::model_kind(info) == flm::catalog::kEmbeddingModelKind ||
           info.at("details").at("family") == "embed-gemma";
}

inline std::unique_ptr<AutoEmbeddingModel> get_auto_embedding_model(
    const nlohmann::json& info, flm_rt::device* device) {
    if (!is_embedding_model(info))
        throw std::invalid_argument("Model is not an embedding model");
    if (flm::catalog::model_kind(info) == flm::catalog::kEmbeddingModelKind)
        return flm::frontend::FrontendRegistry::instance().create_embedding(
            info.at("details").at("family"), device);
#ifndef FASTFLOWLM_LINUX_LIMITED_MODELS
    return std::make_unique<Gemma_Embedding>(device);
#else
    throw std::runtime_error("The built-in embedding model is not supported in this build");
#endif
}
