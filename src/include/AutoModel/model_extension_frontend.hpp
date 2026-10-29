/// \file model_extension_frontend.hpp
/// \brief Host adapter for stable-C model frontend providers
#pragma once

#include "AutoModel/model_extension.h"
#include "AutoModel/model_frontend.hpp"

#include <memory>
#include <string>
#include <vector>

namespace flm::frontend {

struct ExtensionFrontendRegistration {
    std::string family;
    std::string provider;
    flm_frontend_create_v1 create = nullptr;
    void* factory_context = nullptr;
};

/// \brief Build a registry factory for one provider-owned frontend.
FrontendFactory make_extension_frontend_factory(
    ExtensionFrontendRegistration registration,
    std::string extension_root, std::string asset_root,
    std::shared_ptr<void> provider_lifetime);

}  // namespace flm::frontend
