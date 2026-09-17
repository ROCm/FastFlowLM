/// \file model_backend.cpp
/// \brief The backend registry and the rules for picking a backend
#include "AutoModel/model_backend.hpp"

#include "utils/utils.hpp"

#include <cstdlib>
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

void BackendRegistry::register_backend(std::string family, std::string id,
                                       BackendFactory factory,
                                       BackendTraits traits) {
    if (family.empty()) throw std::runtime_error("backend family is empty");
    if (id.empty()) throw std::runtime_error("backend id is empty");
    if (!factory) throw std::runtime_error("backend factory is null");

    std::lock_guard lock(mutex_);
    auto& per_family = factories_[std::move(family)];
    if (per_family.count(id)) {
        throw std::runtime_error("backend '" + id + "' is already registered");
    }
    per_family.emplace(std::move(id), Entry{std::move(factory), traits});
}

void BackendRegistry::replace_backend(std::string family, std::string id,
                                      BackendFactory factory,
                                      BackendTraits traits) {
    if (family.empty()) throw std::runtime_error("backend family is empty");
    if (id.empty()) throw std::runtime_error("backend id is empty");
    if (!factory) throw std::runtime_error("backend factory is null");

    std::lock_guard lock(mutex_);
    factories_[std::move(family)][std::move(id)] =
        Entry{std::move(factory), traits};
}

std::vector<std::string> BackendRegistry::available(
    const std::string& family) const {
    std::lock_guard lock(mutex_);
    std::vector<std::string> ids;
    const auto per_family = factories_.find(family);
    if (per_family == factories_.end()) return ids;
    ids.reserve(per_family->second.size());
    for (const auto& [id, entry] : per_family->second) ids.push_back(id);
    return ids;  // std::map keeps them sorted
}

BackendRegistry::Entry BackendRegistry::lookup(const std::string& family,
                                               const std::string& id) const {
    {
        std::lock_guard lock(mutex_);
        const auto per_family = factories_.find(family);
        if (per_family != factories_.end()) {
            const auto found = per_family->second.find(id);
            if (found != per_family->second.end()) return found->second;
        }
    }
    throw std::runtime_error("Model family '" + family +
                             "' has no backend '" + id +
                             "'. This build provides: " +
                             Join(available(family)));
}

BackendTraits BackendRegistry::traits(const std::string& family,
                                      const std::string& id) const {
    return lookup(family, id).traits;
}

bool BackendRegistry::has(const std::string& family,
                          const std::string& id) const {
    std::lock_guard lock(mutex_);
    const auto per_family = factories_.find(family);
    return per_family != factories_.end() && per_family->second.count(id) != 0;
}

std::unique_ptr<ModelBackend> BackendRegistry::create(
    const std::string& family, const std::string& id,
    const BackendContext& context) const {
    return lookup(family, id).factory(context);
}

std::string resolve_backend_id(const std::string& family,
                               const std::string& fallback,
                               const std::string& requested,
                               std::string* source) {
    std::string chosen;
    std::string chosen_source;
    if (!requested.empty()) {
        chosen = requested;
        chosen_source = "--backend";
    } else if (std::string env = EnvBackend(); !env.empty()) {
        chosen = std::move(env);
        chosen_source = "FLM_BACKEND";
    } else {
        chosen = fallback;
        chosen_source = "build default";
    }

    if (!BackendRegistry::instance().has(family, chosen)) {
        throw std::runtime_error(
            "Backend '" + chosen + "' (from " + chosen_source +
            ") is not compiled into this build of flm. Model family '" +
            family + "' provides: " +
            Join(BackendRegistry::instance().available(family)));
    }

    if (source) *source = chosen_source;
    return chosen;
}

}  // namespace flm::backend
