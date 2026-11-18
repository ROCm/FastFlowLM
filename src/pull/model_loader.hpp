/*!
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * \file model_loader.hpp
 * \brief Resolve model downloads before selecting a generation frontend.
 */
#pragma once

#include "AutoModel/all_models.hpp"
#include "model_downloader.hpp"

inline std::pair<std::string, flm::frontend::FrontendPtr> get_auto_model(
    const std::string& model_tag, model_list& models, flm_rt::device* device,
    ModelDownloader& downloader, bool modelscope,
    const std::string& requested_backend = "", const std::string& power_mode = "performance") {
    auto plan = resolve_model_load(model_tag, models, requested_backend);
    plan.prepare_device(device, power_mode);
    const auto& tag = plan.tag;
    switch (downloader.is_model_downloaded(tag, false, /*fast_check=*/true)) {
        case ModelDownloader::ModelStatus::Ready:
            break;
        case ModelDownloader::ModelStatus::Outdated:
        case ModelDownloader::ModelStatus::Missing:
            if (!downloader.pull_model(tag, modelscope))
                throw std::runtime_error("Failed to download " + tag);
            break;
        case ModelDownloader::ModelStatus::Incompatible:
            throw std::runtime_error(tag + " is incompatible with this version of FastFlowLM");
    }
    // preferred_source can change frontend_provider as well as the model files.
    return get_auto_model(tag, models, device, requested_backend, power_mode);
}
