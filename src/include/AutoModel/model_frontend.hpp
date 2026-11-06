/// \file model_frontend.hpp
/// \brief Frontend factories keyed by model family
#pragma once

#include "AutoModel/model_catalog_overlay.hpp"
#include "AutoModel/model_extension.h"

#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_set>
#include <vector>

class AutoModel;

namespace flm::frontend {

/// \brief Type-erased ownership used by built-in frontend factories.
struct FrontendDeleter {
    void* context = nullptr;
    void (*destroy)(void*, void*) = nullptr;
    std::shared_ptr<void> provider_lifetime;

    void operator()(AutoModel* frontend) noexcept;
};

using FrontendPtr = std::unique_ptr<AutoModel, FrontendDeleter>;
using FrontendFactory = std::function<FrontendPtr(void* device)>;

/// \brief Make a host-owned frontend use the registry's ownership shape.
FrontendPtr own_frontend(std::unique_ptr<AutoModel> frontend);

/// \brief The family -> frontend factory table.
class FrontendRegistry {
public:
    FrontendRegistry();
    ~FrontendRegistry();

    FrontendRegistry(const FrontendRegistry&) = delete;
    FrontendRegistry& operator=(const FrontendRegistry&) = delete;

    /// \brief Process-wide registry populated with built-in frontends.
    static FrontendRegistry& instance();

    /// \brief Register one frontend family.
    /// \throws std::runtime_error for an empty value or duplicate family
    void register_frontend(
        std::string family, FrontendFactory factory,
        std::string provider = "builtin");

    bool has(const std::string& family) const;
    std::vector<std::string> available() const;

    /// \brief Construct a frontend.
    /// \throws std::runtime_error when the family is not registered
    FrontendPtr create(const std::string& family, void* device) const;

    /// \brief Load and atomically register one stable-ABI backend extension.
    /// \param manifest_path extension.json naming the library and optional
    ///        catalog/assets relative to its bundle root
    /// \note Provider libraries remain loaded for this registry's lifetime.
    /// \note Load extensions during process setup, before model creation or
    ///       concurrent backend-registry access. Failure is transactionally
    ///       rolled back; dynamic mutation while serving is not supported.
    void load_extension(
        const std::string& manifest_path,
        const std::unordered_set<std::string>& reserved_model_tags);

    /// \brief Unregister extension providers and release their libraries.
    /// \note Call only during process shutdown, after all model instances have
    ///       been destroyed. Built-in frontend and backend registrations remain.
    void unload_extensions() noexcept;

    /// \brief Validated catalog overlays contributed by loaded extensions.
    std::vector<catalog::ModelCatalogOverlay> catalog_overlays() const;

private:
    struct LoadedExtension;

    mutable std::mutex mutex_;
    std::map<std::string, FrontendFactory> factories_;
    std::map<std::string, std::string> providers_;
    std::vector<std::shared_ptr<LoadedExtension>> extensions_;
};

/// \brief Register every causal-generation frontend shipped with FastFlowLM.
void register_builtin_frontends(FrontendRegistry& registry);

}  // namespace flm::frontend
