/// \file model_extension_backend.hpp
/// \brief Backend-registration bridge kept out of the hardware-free loader
#pragma once

#include "AutoModel/model_extension.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace flm::frontend {

struct ExtensionBackendRegistration {
    std::string family;
    std::string backend;
    std::string provider;
    flm_backend_traits_v1 traits{};
    std::vector<std::int32_t> forced_eos_ids;
    flm_backend_create_v1 create = nullptr;
    void* factory_context = nullptr;
};

struct ExtensionBackendKey {
    std::string family;
    std::string backend;
    std::string provider;
};

/// \brief Validate and publish backend providers staged by one extension.
std::vector<ExtensionBackendKey> register_extension_backends(
    std::vector<ExtensionBackendRegistration> registrations,
    std::string extension_root, std::string asset_root,
    std::shared_ptr<void> provider_lifetime);

}  // namespace flm::frontend
