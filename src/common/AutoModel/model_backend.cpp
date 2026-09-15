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

/// \brief the catalog's own backend choice for an entry
/// \param model_info the resolved model_list.json entry
/// \return details.execution_backend, or empty when the entry does not set it
std::string CatalogBackend(const nlohmann::ordered_json& model_info) {
    const auto details = model_info.find("details");
    if (details == model_info.end() || !details->is_object()) return {};
    const auto backend = details->find("execution_backend");
    if (backend == details->end()) return {};
    if (!backend->is_string()) {
        throw std::runtime_error(
            "details.execution_backend must be a string");
    }
    return backend->get<std::string>();
}

/// \brief read FLM_BACKEND
/// \return the override, or empty when it is unset
std::string EnvBackend() {
    const char* configured = std::getenv("FLM_BACKEND");
    if (!configured || !*configured) return {};
    return configured;
}

bool Contains(const std::vector<std::string>& ids, const std::string& id) {
    for (const auto& candidate : ids) {
        if (candidate == id) return true;
    }
    return false;
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

std::vector<std::string> supported_backends(const nlohmann::ordered_json& model_info) {
    const auto listed = model_info.find("supported_backends");
    if (listed != model_info.end()) {
        if (!listed->is_array() || listed->empty()) {
            throw std::runtime_error(
                "supported_backends must be a non-empty array");
        }
        std::vector<std::string> ids;
        ids.reserve(listed->size());
        for (const auto& entry : *listed) {
            if (!entry.is_string()) {
                throw std::runtime_error(
                    "supported_backends must hold strings");
            }
            ids.push_back(entry.get<std::string>());
        }
        return ids;
    }

    // No list: the entry allows exactly what it has always run on.
    const std::string catalog = CatalogBackend(model_info);
    return {catalog.empty() ? std::string(kDefaultBackendId) : catalog};
}

std::string resolve_backend_id(const std::string& family,
                               const nlohmann::ordered_json& model_info,
                               const std::string& requested,
                               std::string* source) {
    const std::vector<std::string> allowed = supported_backends(model_info);

    std::string chosen;
    std::string chosen_source;
    if (!requested.empty()) {
        chosen = requested;
        chosen_source = "--backend";
    } else if (std::string env = EnvBackend(); !env.empty()) {
        chosen = std::move(env);
        chosen_source = "FLM_BACKEND";
    } else if (std::string catalog = CatalogBackend(model_info);
               !catalog.empty()) {
        chosen = std::move(catalog);
        chosen_source = "model catalog";
    } else {
        chosen = kDefaultBackendId;
        chosen_source = "default";
    }

    if (!Contains(allowed, chosen)) {
        throw std::runtime_error(
            "Backend '" + chosen + "' (from " + chosen_source +
            ") is not available for this model. It supports: " + Join(allowed));
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
