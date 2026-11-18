// Copyright (c) 2026 Advanced Micro Devices, Inc.

/// \file oga_hipep_backend.hpp
/// \brief The oga-hipep backend: an OGA package on the GPU, through a separately installed hip-ep
/// \note Compiled only when FLM_ENABLE_OGA_HIPEP is on, which is Windows only.
#pragma once

#include "AutoModel/model_backend.hpp"
#include "oga/oga_api.hpp"

#include <filesystem>

namespace flm::oga {

/// \brief create the OGA model in `root` with its decoder on the EP registered under
///        kAmdGpuProviderName alone, and no CPU fallback
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

/// \brief `flm uninstall hipep`: remove the hip-ep runtime flm installed for this user, and the
///        installer's files; every other hip-ep package, and every model, stays
/// \return whether nothing flm installed is left
/// \throws std::runtime_error when it cannot tell what flm installed
bool UninstallHipEp();

/// \brief end the process with `code` if it loaded hip-ep, skipping every DLL's teardown;
///        return otherwise
/// \note Returning from main and the default Ctrl+C handler both end in ExitProcess, which
///       hangs in the HIP runtime's teardown: the process keeps its GPU memory and cannot
///       even be killed. TerminateProcess skips that teardown, and Windows frees the memory.
void EndProcessIfHipEpLoaded(unsigned int code);

}  // namespace flm::oga
