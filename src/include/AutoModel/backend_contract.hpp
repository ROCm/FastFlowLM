/// \file backend_contract.hpp
/// \brief Lightweight backend and catalog identifiers shared by model lookup
///        and execution code
#pragma once

#include <map>
#include <string>
#include <vector>

namespace flm::backend {

/// \brief FastFlowLM's own NPU kernel flow
/// \note Every model family registers this backend, so it is the default for a
///       catalog entry that names none.
inline constexpr const char* kFlmBackendId = "flm";

/// \brief Kernels reached through ryzenai-corelib
/// \note This backend is added only for supported families when corelib is
///       enabled; it does not replace FastFlowLM's backend.
inline constexpr const char* kCoreBackendId = "core";

/// \brief OGA packages run on the GPU through a separately installed hip-ep
inline constexpr const char* kOgaHipepBackendId = "oga-hipep";

/// \brief Provider id for implementations linked into FastFlowLM itself.
/// \note A backend identifies an artifact/execution contract; a provider
///       identifies one implementation of that contract.
inline constexpr const char* kBuiltinProviderId = "builtin";

/// \brief Catalog entry key naming the required backend contract
inline constexpr const char* kBackendKey = "backend";

/// \brief Catalog entry key naming an implementation of the backend contract
inline constexpr const char* kBackendProviderKey = "backend_provider";

/// \brief Model-family suffix selecting the ryzenai-corelib contract
/// \note Corelib models have distinct artifacts and therefore distinct tags.
inline constexpr const char* kCoreFamilySuffix = "-core";

/// \brief Model-family suffix selecting an OGA package run through hip-ep
inline constexpr const char* kOgaHipepFamilySuffix = "-oga-hipep";

using BackendProviderInventory =
    std::map<std::string,
             std::map<std::string, std::vector<std::string>>>;

}  // namespace flm::backend
