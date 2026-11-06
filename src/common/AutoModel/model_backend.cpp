/// \file model_backend.cpp
/// \brief The backend registry and the rules for picking a backend
#include "AutoModel/model_backend.hpp"

#include "utils/utils.hpp"

#include <cstdlib>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace flm::backend {
namespace {

/// \brief render a list of ids for an error message
/// \param ids the ids
/// \return "a, b, c", or "(none)" when the list is empty
std::string Join(const std::vector<std::string>& ids) {
    if (ids.empty()) return "(none)";
    std::ostringstream out;
    for (std::size_t i = 0; i < ids.size(); ++i) {
        if (i) out << ", ";
        out << ids[i];
    }
    return out.str();
}

/// \brief read FLM_BACKEND
/// \return the override, or empty when it is unset
std::string EnvBackend() {
    const char* configured = std::getenv("FLM_BACKEND");
    if (!configured || !*configured) return {};
    return configured;
}

}  // namespace

BackendRegistry& BackendRegistry::instance() {
    static BackendRegistry registry;
    static std::once_flag once;
    // register_builtin_backends must not call instance(), or this deadlocks.
    std::call_once(once, [] { register_builtin_backends(registry); });
    return registry;
}

void BackendRegistry::register_backend(const std::string& family,
                                       const std::string& id,
                                       BackendFactory factory,
                                       BackendTraits traits) {
    register_provider(family, id, kBuiltinProviderId, std::move(factory),
                      traits);
}

void BackendRegistry::register_provider(
    const std::string& family, const std::string& id,
    std::string provider, BackendFactory factory, BackendTraits traits,
    std::string asset_root) {
    if (family.empty()) throw std::runtime_error("backend family is empty");
    if (id.empty()) throw std::runtime_error("backend id is empty");
    if (provider.empty()) throw std::runtime_error("backend provider is empty");
    if (!factory) throw std::runtime_error("backend factory is null");

    std::lock_guard lock(mutex_);
    auto& providers = factories_[family][id];
    if (providers.count(provider)) {
        throw std::runtime_error(
            "backend provider '" + provider + "' is already registered for '" +
            family + "/" + id + "'");
    }
    providers.emplace(std::move(provider),
                      Entry{std::move(factory), traits,
                            std::move(asset_root)});
}

void BackendRegistry::unregister_provider(
    const std::string& family, const std::string& id,
    const std::string& provider) noexcept {
    if (provider == kBuiltinProviderId) return;
    std::lock_guard lock(mutex_);
    const auto per_family = factories_.find(family);
    if (per_family == factories_.end()) return;
    const auto per_backend = per_family->second.find(id);
    if (per_backend == per_family->second.end()) return;
    per_backend->second.erase(provider);
    if (per_backend->second.empty()) per_family->second.erase(per_backend);
    if (per_family->second.empty()) factories_.erase(per_family);
}

void BackendRegistry::replace_backend(std::string family, std::string id,
                                      BackendFactory factory,
                                      BackendTraits traits) {
    if (family.empty()) throw std::runtime_error("backend family is empty");
    if (id.empty()) throw std::runtime_error("backend id is empty");
    if (!factory) throw std::runtime_error("backend factory is null");

    std::lock_guard lock(mutex_);
    factories_[std::move(family)][std::move(id)][kBuiltinProviderId] =
        Entry{std::move(factory), traits, {}};
}

std::vector<std::string> BackendRegistry::available(
    const std::string& family) const {
    std::lock_guard lock(mutex_);
    std::vector<std::string> ids;
    const auto per_family = factories_.find(family);
    if (per_family == factories_.end()) return ids;
    ids.reserve(per_family->second.size());
    for (const auto& [id, providers] : per_family->second) {
        (void)providers;
        ids.push_back(id);
    }
    return ids;  // std::map keeps them sorted
}

std::vector<std::string> BackendRegistry::available_providers(
    const std::string& family, const std::string& id) const {
    std::lock_guard lock(mutex_);
    std::vector<std::string> providers;
    const auto per_family = factories_.find(family);
    if (per_family == factories_.end()) return providers;
    const auto per_backend = per_family->second.find(id);
    if (per_backend == per_family->second.end()) return providers;
    providers.reserve(per_backend->second.size());
    for (const auto& [provider, entry] : per_backend->second) {
        (void)entry;
        providers.push_back(provider);
    }
    return providers;
}

BackendRegistry::Entry BackendRegistry::lookup(const std::string& family,
                                               const std::string& id,
                                               const std::string& provider) const {
    bool backend_exists = false;
    {
        std::lock_guard lock(mutex_);
        const auto per_family = factories_.find(family);
        if (per_family != factories_.end()) {
            const auto per_backend = per_family->second.find(id);
            if (per_backend != per_family->second.end()) {
                backend_exists = true;
                const auto found = per_backend->second.find(provider);
                if (found != per_backend->second.end()) return found->second;
            }
        }
    }
    if (!backend_exists) {
        throw std::runtime_error(
            "Model family '" + family + "' has no backend '" + id +
            "'. This build provides: " + Join(available(family)));
    }
    throw std::runtime_error(
        "Model family '" + family + "' has no provider '" + provider +
        "' for backend '" + id + "'. This backend provides: " +
        Join(available_providers(family, id)));
}

BackendTraits BackendRegistry::traits(const std::string& family,
                                      const std::string& id,
                                      const std::string& provider) const {
    return lookup(family, id, provider).traits;
}

std::string BackendRegistry::asset_root(
    const std::string& family, const std::string& id,
    const std::string& provider) const {
    return lookup(family, id, provider).asset_root;
}

bool BackendRegistry::has(const std::string& family,
                          const std::string& id) const {
    std::lock_guard lock(mutex_);
    const auto per_family = factories_.find(family);
    return per_family != factories_.end() && per_family->second.count(id) != 0;
}

bool BackendRegistry::has_provider(
    const std::string& family, const std::string& id,
    const std::string& provider) const {
    std::lock_guard lock(mutex_);
    const auto per_family = factories_.find(family);
    if (per_family == factories_.end()) return false;
    const auto per_backend = per_family->second.find(id);
    return per_backend != per_family->second.end() &&
           per_backend->second.count(provider) != 0;
}

std::vector<std::string> BackendRegistry::backend_ids() const {
    std::lock_guard lock(mutex_);
    std::set<std::string> ids;
    for (const auto& [family, backends] : factories_) {
        (void)family;
        for (const auto& [id, providers] : backends) {
            (void)providers;
            ids.insert(id);
        }
    }
    return {ids.begin(), ids.end()};
}

BackendRegistry::ProviderInventory BackendRegistry::provider_inventory() const {
    std::lock_guard lock(mutex_);
    ProviderInventory inventory;
    for (const auto& [family, backends] : factories_) {
        for (const auto& [id, providers] : backends) {
            auto& ids = inventory[family][id];
            ids.reserve(providers.size());
            for (const auto& [provider, entry] : providers) {
                (void)entry;
                ids.push_back(provider);
            }
        }
    }
    return inventory;
}

std::unique_ptr<ModelBackend> BackendRegistry::create(
    const std::string& family, const std::string& id,
    const std::string& provider,
    const BackendContext& context) const {
    return lookup(family, id, provider).factory(context);
}

BackendSelection resolve_backend_selection(
    const std::string& family, const std::string& catalog_backend,
    const std::string& catalog_provider,
    const std::string& requested_backend, std::string* source) {
    std::string chosen;
    std::string chosen_source;
    if (!requested_backend.empty()) {
        chosen = requested_backend;
        chosen_source = "--backend";
    } else if (std::string env = EnvBackend(); !env.empty()) {
        chosen = std::move(env);
        chosen_source = "FLM_BACKEND";
    } else {
        chosen = catalog_backend;
        chosen_source = "model catalog";
    }

    if (chosen != catalog_backend) {
        throw std::runtime_error(
            "Backend '" + chosen + "' (from " + chosen_source +
            ") is not compatible with this model package; its catalog entry "
            "requires '" + catalog_backend + "/" + catalog_provider + "'");
    }

    auto& registry = BackendRegistry::instance();
    if (!registry.has(family, chosen)) {
        throw std::runtime_error(
            "Backend '" + chosen + "' (from " + chosen_source +
            ") is not available for model family '" + family +
            "'. It provides: " + Join(registry.available(family)));
    }
    if (!registry.has_provider(family, chosen, catalog_provider)) {
        throw std::runtime_error(
            "Backend provider '" + catalog_provider +
            "' from the model catalog is not available for '" + family +
            "/" + chosen + "'. This backend provides: " +
            Join(registry.available_providers(family, chosen)));
    }

    if (source) *source = chosen_source;
    return BackendSelection{std::move(chosen), catalog_provider};
}

}  // namespace flm::backend
