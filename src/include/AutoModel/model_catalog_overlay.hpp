/// \file model_catalog_overlay.hpp
/// \brief Parsed, versioned model-catalog contributions from extensions
#pragma once

#include "AutoModel/backend_contract.hpp"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace flm::catalog {

struct ModelCatalogOverlay {
    std::string path;
    nlohmann::json document;
};

namespace detail {
inline std::filesystem::path path_from_utf8(const std::string& encoded) {
    return std::filesystem::path(std::u8string(
        reinterpret_cast<const char8_t*>(encoded.data()),
        reinterpret_cast<const char8_t*>(
            encoded.data() + encoded.size())));
}
}  // namespace detail

inline std::string backend_for_model_type(const std::string& model_type) {
    const std::string suffix = backend::kCoreFamilySuffix;
    if (model_type.size() > suffix.size() &&
        model_type.compare(model_type.size() - suffix.size(),
                           suffix.size(), suffix) == 0) {
        return backend::kCoreBackendId;
    }
    return backend::kFlmBackendId;
}

inline ModelCatalogOverlay load_model_catalog_overlay(
    const std::string& overlay_path) {
    std::ifstream input(detail::path_from_utf8(overlay_path));
    if (!input) {
        throw std::runtime_error(
            "Cannot open model extension catalog: " + overlay_path);
    }

    nlohmann::json overlay;
    try {
        overlay = nlohmann::json::parse(input);
    } catch (const std::exception& error) {
        throw std::runtime_error(
            "Cannot parse model extension catalog '" + overlay_path +
            "': " + error.what());
    }
    if (!overlay.is_object() ||
        overlay.value("schema_version", 0) != 1) {
        throw std::runtime_error(
            "Model extension catalog '" + overlay_path +
            "' must use schema_version 1");
    }
    for (const auto& [field, value] : overlay.items()) {
        (void)value;
        if (field != "schema_version" && field != "models") {
            throw std::runtime_error(
                "Model extension catalog '" + overlay_path +
                "' has unsupported field '" + field + "'");
        }
    }
    if (!overlay.contains("models") ||
        !overlay["models"].is_object()) {
        throw std::runtime_error(
            "Model extension catalog '" + overlay_path +
            "' must contain a models object");
    }

    for (const auto& [model_type, sizes] :
         overlay["models"].items()) {
        if (!sizes.is_object() || sizes.empty()) {
            throw std::runtime_error(
                "Model extension catalog family '" + model_type +
                "' must contain model sizes");
        }
        for (const auto& [size, entry] : sizes.items()) {
            const std::string tag = model_type + ":" + size;
            if (!entry.is_object() || !entry.contains("name") ||
                !entry["name"].is_string() ||
                entry["name"].get_ref<const std::string&>().empty() ||
                !entry.contains(backend::kBackendKey) ||
                !entry.contains(backend::kBackendProviderKey) ||
                !entry.contains("default_context_length") ||
                !entry["default_context_length"].is_number_integer() ||
                entry["default_context_length"].get<int>() <= 0 ||
                !entry.contains("max_prefill_len") ||
                !entry["max_prefill_len"].is_number_integer() ||
                entry["max_prefill_len"].get<int>() <= 0 ||
                !entry.contains("details") ||
                !entry["details"].is_object() ||
                !entry["details"].contains("family") ||
                !entry["details"]["family"].is_string() ||
                entry["details"]["family"]
                    .get_ref<const std::string&>().empty()) {
                throw std::runtime_error(
                    "Model extension catalog entry '" + tag +
                    "' must have non-empty name, backend, "
                    "backend_provider, positive default_context_length and "
                    "max_prefill_len, and details.family");
            }
            for (const char* field :
                 {backend::kBackendKey,
                  backend::kBackendProviderKey}) {
                if (entry.contains(field) &&
                    (!entry[field].is_string() ||
                     entry[field]
                         .get_ref<const std::string&>().empty())) {
                    throw std::runtime_error(
                        "Model extension catalog entry '" + tag +
                        "' field '" + field +
                        "' must be a non-empty string");
                }
            }
            if (entry.contains("supported_platforms")) {
                const auto& platforms = entry["supported_platforms"];
                if (!platforms.is_array() || platforms.empty()) {
                    throw std::runtime_error(
                        "Model extension catalog entry '" + tag +
                        "' supported_platforms must be a non-empty array");
                }
                std::unordered_set<std::string> seen;
                for (const auto& platform : platforms) {
                    if (!platform.is_string()) {
                        throw std::runtime_error(
                            "Model extension catalog entry '" + tag +
                            "' supported_platforms must contain strings");
                    }
                    const auto value = platform.get<std::string>();
                    if ((value != "aie2p" && value != "dev04") ||
                        !seen.insert(value).second) {
                        throw std::runtime_error(
                            "Model extension catalog entry '" + tag +
                            "' has an unknown or duplicate platform '" +
                            value + "'");
                    }
                }
            }
            if ((entry.value("vlm", false) ||
                 entry.value("asr", false)) &&
                (!entry.contains("frontend_provider") ||
                 !entry["frontend_provider"].is_string() ||
                 entry["frontend_provider"]
                     .get_ref<const std::string&>().empty())) {
                throw std::runtime_error(
                    "Multimodal model extension catalog entry '" + tag +
                    "' must name a frontend_provider");
            }
        }
    }
    return ModelCatalogOverlay{overlay_path, std::move(overlay)};
}

inline void validate_model_catalog_overlay(
    const ModelCatalogOverlay& overlay,
    const std::vector<std::string>& frontends,
    const backend::BackendProviderInventory& providers,
    const std::unordered_set<std::string>& reserved_tags) {
    std::unordered_map<std::string, std::string> reserved_aliases;
    // Bare tags resolve to the lexicographically first size. An extension may
    // add later sizes, but must not silently change an existing resolution.
    for (const auto& tag : reserved_tags) {
        const auto separator = tag.rfind(':');
        if (separator == std::string::npos) continue;
        const std::string model_type = tag.substr(0, separator);
        auto [alias, inserted] =
            reserved_aliases.emplace(model_type, tag);
        if (!inserted && tag < alias->second) {
            alias->second = tag;
        }
    }
    for (const auto& [model_type, sizes] :
         overlay.document.at("models").items()) {
        const auto reserved_alias = reserved_aliases.find(model_type);
        if (reserved_alias != reserved_aliases.end()) {
            const std::string overlay_alias =
                model_type + ":" + sizes.begin().key();
            if (overlay_alias < reserved_alias->second) {
                throw std::runtime_error(
                    "Model extension catalog changes existing bare model "
                    "tag '" + model_type + "' from '" +
                    reserved_alias->second + "' to '" + overlay_alias + "'");
            }
        }
        for (const auto& [size, entry] : sizes.items()) {
            const std::string tag = model_type + ":" + size;
            if (reserved_tags.count(tag)) {
                throw std::runtime_error(
                    "Model extension catalog duplicates model tag '" +
                    tag + "'");
            }
            const std::string family =
                entry["details"]["family"].get<std::string>();
            if (std::find(frontends.begin(), frontends.end(), family) ==
                frontends.end()) {
                throw std::runtime_error(
                    "Model extension catalog entry '" + tag +
                    "' names unregistered frontend family '" +
                    family + "'");
            }
            const std::string selected_backend =
                entry.at(backend::kBackendKey).get<std::string>();
            const std::string provider =
                entry.at(backend::kBackendProviderKey).get<std::string>();
            const auto family_entry = providers.find(family);
            if (family_entry == providers.end()) {
                throw std::runtime_error(
                    "Model extension catalog entry '" + tag +
                    "' names unavailable backend provider '" + family +
                    "/" + selected_backend + "/" + provider + "'");
            }
            const auto backend_entry =
                family_entry->second.find(selected_backend);
            if (backend_entry == family_entry->second.end() ||
                std::find(backend_entry->second.begin(),
                          backend_entry->second.end(), provider) ==
                    backend_entry->second.end()) {
                throw std::runtime_error(
                    "Model extension catalog entry '" + tag +
                    "' names unavailable backend provider '" + family +
                    "/" + selected_backend + "/" + provider + "'");
            }
        }
    }
}

inline std::unordered_set<std::string> model_catalog_tags(
    const nlohmann::json& catalog, const std::string& catalog_path) {
    if (!catalog.is_object() || !catalog.contains("models") ||
        !catalog["models"].is_object()) {
        throw std::runtime_error(
            "Model catalog '" + catalog_path +
            "' must contain a models object");
    }
    std::unordered_set<std::string> tags;
    for (const auto& [model_type, sizes] :
         catalog["models"].items()) {
        if (!sizes.is_object()) {
            throw std::runtime_error(
                "Model catalog family '" + model_type +
                "' must contain model sizes");
        }
        for (const auto& [size, entry] : sizes.items()) {
            (void)entry;
            tags.insert(model_type + ":" + size);
        }
    }
    return tags;
}

inline std::unordered_set<std::string> load_model_catalog_tags(
    const std::string& catalog_path) {
    std::ifstream input(detail::path_from_utf8(catalog_path));
    if (!input) {
        throw std::runtime_error(
            "Cannot open model catalog: " + catalog_path);
    }
    try {
        return model_catalog_tags(
            nlohmann::json::parse(input), catalog_path);
    } catch (const nlohmann::json::exception& error) {
        throw std::runtime_error(
            "Cannot parse model catalog '" + catalog_path +
            "': " + error.what());
    }
}

}  // namespace flm::catalog
