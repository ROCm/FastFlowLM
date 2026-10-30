/// \file oga_hipep_backend.hpp
/// \brief The oga-hipep backend: an OGA package on the GPU, through a separately installed hip-ep
/// \note Compiled only when FLM_ENABLE_OGA_HIPEP is on, which is Windows only.
#pragma once

#include "AutoModel/model_backend.hpp"
#include "oga/oga_api.hpp"

#include <filesystem>

namespace flm::oga {

/// \brief create the OGA model in `root` with every session on the AMDGPU EP's HIP profile
/// \throws std::runtime_error if OGA places the model on any other device
OgaApi::Owned<OgaModel> OpenModel(const OgaApi& api, const std::filesystem::path& root);

inline flm::backend::BackendTraits hipep_traits() {
    flm::backend::BackendTraits traits;
    traits.needs_npu_xclbin = false;
    traits.supports_preemption = false;
    return traits;
}

/// \brief a factory building the backend from the OGA package in the model directory
/// \note Everything the package states is read from genai_config.json, so one
///       factory serves every family it is registered for. Preemption, the
///       context and the package's token ids are checked before hip-ep is loaded.
flm::backend::BackendFactory hipep_factory();

}  // namespace flm::oga
