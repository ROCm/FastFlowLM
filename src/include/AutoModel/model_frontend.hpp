/// \file model_frontend.hpp
/// \brief Frontend factories keyed by model family
#pragma once

#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

class AutoModel;

namespace flm::frontend {

/// \brief Deletes a frontend in the module that allocated it.
struct FrontendDeleter {
    void* context = nullptr;
    void (*destroy)(void*, void*) = nullptr;
    std::shared_ptr<void> provider_lifetime;

    void operator()(AutoModel* frontend) const noexcept;
};

using FrontendPtr = std::unique_ptr<AutoModel, FrontendDeleter>;
using FrontendFactory = std::function<FrontendPtr(void* device)>;

/// \brief Make a host-owned frontend use the same ownership shape as plugins.
FrontendPtr own_frontend(std::unique_ptr<AutoModel> frontend);

/// \brief The family -> frontend factory table.
class FrontendRegistry {
public:
    FrontendRegistry() = default;
    ~FrontendRegistry();

    FrontendRegistry(const FrontendRegistry&) = delete;
    FrontendRegistry& operator=(const FrontendRegistry&) = delete;

    /// \brief Process-wide registry populated with built-in frontends.
    static FrontendRegistry& instance();

    /// \brief Register one frontend family.
    /// \throws std::runtime_error for an empty value or duplicate family
    void register_frontend(std::string family, FrontendFactory factory);

    bool has(const std::string& family) const;
    std::vector<std::string> available() const;

    /// \brief Construct a frontend.
    /// \throws std::runtime_error when the family is not registered
    FrontendPtr create(const std::string& family, void* device) const;

private:
    mutable std::mutex mutex_;
    std::map<std::string, FrontendFactory> factories_;
};

/// \brief Register every causal-generation frontend shipped with FastFlowLM.
void register_builtin_frontends(FrontendRegistry& registry);

}  // namespace flm::frontend
