/// \file model_catalog_overlay.hpp
/// \brief Parsed, versioned model-catalog contributions from extensions
#pragma once

#include "AutoModel/backend_contract.hpp"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <cctype>
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

/// \brief whether a catalog path stays inside the folder it is joined to
/// \param value a '/'-separated relative path
/// \param single_component true when `value` must be one folder or file name
/// \note Backslashes and colons are rejected on every platform, so a catalog
///       cannot name a rooted, drive-relative or UNC path on Windows and the
///       same catalog is accepted everywhere. So are the names Windows
///       rewrites: it drops a trailing dot or space ("Model." opens "Model",
///       another entry's folder) and maps CON, NUL, COM1 and the like,
///       including COM and LPT with a superscript 1 to 3, with any
///       extension, to devices.
inline bool is_contained_path(const std::string& value,
                              bool single_component) {
    if (value.empty() || value.find_first_of("\\:") != std::string::npos) {
        return false;
    }
    const auto is_device = [](const std::string& component) {
        std::string stem = component.substr(0, component.find('.'));
        stem.erase(stem.find_last_not_of(' ') + 1);
        for (auto& c : stem) {
            c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
        }
        if (stem == "CON" || stem == "PRN" || stem == "AUX" || stem == "NUL") {
            return true;
        }
        if (stem.compare(0, 3, "COM") != 0 && stem.compare(0, 3, "LPT") != 0) {
            return false;
        }
        // COM1-COM9 and LPT1-LPT9, and the same with a superscript 1, 2 or 3
        // (U+00B9, U+00B2, U+00B3) in UTF-8.
        const std::string port = stem.substr(3);
        return (port.size() == 1 && port[0] >= '1' && port[0] <= '9') ||
               port == "\xC2\xB9" || port == "\xC2\xB2" || port == "\xC2\xB3";
    };
    std::size_t begin = 0;
    while (true) {
        const std::size_t end = value.find('/', begin);
        const std::string component = value.substr(
            begin, end == std::string::npos ? std::string::npos : end - begin);
        if (component.empty() || component.back() == '.' ||
            component.back() == ' ' || is_device(component)) {
            return false;
        }
        if (end == std::string::npos) return true;
        if (single_component) return false;
        begin = end + 1;
    }
}

inline std::string ascii_lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) {
                       return static_cast<char>(std::tolower(c));
                   });
    return value;
}

inline void require_contained_name(const nlohmann::json& entry,
                                   const char* field,
                                   const std::string& tag) {
    if (!entry.contains(field)) return;
    if (!entry[field].is_string() ||
        !is_contained_path(entry[field].get_ref<const std::string&>(),
                           true)) {
        throw std::runtime_error(
            "Model extension catalog entry '" + tag + "' field '" + field +
            "' must be a single relative folder name");
    }
}

inline void require_contained_files(const nlohmann::json& files,
                                    const std::string& tag) {
    for (const auto& file : files) {
        if (!file.is_string() ||
            !is_contained_path(file.get_ref<const std::string&>(), false)) {
            throw std::runtime_error(
                "Model extension catalog entry '" + tag +
                "' files must be relative paths inside the model folder");
        }
    }
}

/// \brief the folders, lower-cased, that a catalog's entries download into
inline std::unordered_set<std::string> model_catalog_folders(
    const nlohmann::json& catalog) {
    std::unordered_set<std::string> folders;
    if (!catalog.is_object() || !catalog.contains("models") ||
        !catalog["models"].is_object()) {
        return folders;
    }
    const auto add = [&](const nlohmann::json& entry) {
        if (entry.is_object() && entry.contains("name") &&
            entry["name"].is_string()) {
            folders.insert(ascii_lower(entry["name"].get<std::string>()));
        }
    };
    for (const auto& [model_type, sizes] : catalog["models"].items()) {
        (void)model_type;
        if (!sizes.is_object()) continue;
        for (const auto& [size, entry] : sizes.items()) {
            (void)size;
            add(entry);
            if (entry.is_object() && entry.contains("preferred_source")) {
                add(entry["preferred_source"]);
            }
        }
    }
    return folders;
}

/// \brief the entry flm pull and flm run use once `entry`'s preferred_source
///        applies
/// \note model_list applies preferred_source as a JSON merge patch when a pull
///       chose it or its files are on disk, so the merged entry must pass the
///       checks the entry itself does.
inline nlohmann::json with_preferred_source(nlohmann::json entry) {
    nlohmann::json preferred = std::move(entry["preferred_source"]);
    entry.erase("preferred_source");
    entry.merge_patch(preferred);
    return entry;
}

/// \brief refuse a catalog entry flm pull, flm remove or flm run cannot use
/// \param entry a size entry, or the one its preferred_source makes
/// \param tag names the entry in error messages
inline void validate_catalog_entry(const nlohmann::json& entry,
                                   const std::string& tag) {
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
    // flm pull reads these unconditionally for every catalog entry.
    for (const char* field : {"url", "flm_min_version"}) {
        if (!entry.contains(field) || !entry[field].is_string() ||
            entry[field].get_ref<const std::string&>().empty()) {
            throw std::runtime_error(
                "Model extension catalog entry '" + tag +
                "' field '" + field +
                "' must be a non-empty string");
        }
    }
    if (!entry.contains("files") || !entry["files"].is_array() ||
        entry["files"].empty() ||
        std::any_of(entry["files"].begin(), entry["files"].end(),
                    [](const nlohmann::json& file) {
                        return !file.is_string() ||
                               file.get_ref<const std::string&>()
                                   .empty();
                    })) {
        throw std::runtime_error(
            "Model extension catalog entry '" + tag +
            "' files must be a non-empty array of non-empty strings");
    }
    // flm pull writes, and flm remove deletes, under
    // <model root>/<name>/<file>; none of them may leave that folder.
    detail::require_contained_name(entry, "name", tag);
    detail::require_contained_name(entry, "xclbin_folder", tag);
    detail::require_contained_files(entry["files"], tag);
    if (entry.contains("file_sources")) {
        if (!entry["file_sources"].is_object()) {
            throw std::runtime_error(
                "Model extension catalog entry '" + tag +
                "' file_sources must be an object");
        }
        for (const auto& [file, source] :
             entry["file_sources"].items()) {
            (void)source;
            if (!detail::is_contained_path(file, false)) {
                throw std::runtime_error(
                    "Model extension catalog entry '" + tag +
                    "' file_sources must name relative paths "
                    "inside the model folder");
            }
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
}  // namespace detail

/// \brief an entry and, when it has a preferred_source, the entry that makes
/// \note Checks that read an entry's selections apply to both: either can be
///       the one flm pull and flm run use.
inline std::vector<nlohmann::json> catalog_entry_sources(
    const nlohmann::json& entry) {
    std::vector<nlohmann::json> sources{entry};
    if (entry.contains("preferred_source")) {
        sources.push_back(detail::with_preferred_source(entry));
    }
    return sources;
}

inline std::string backend_for_model_type(const std::string& model_type) {
    const auto has_suffix = [&](const std::string& suffix) {
        return model_type.size() > suffix.size() &&
               model_type.compare(model_type.size() - suffix.size(),
                                  suffix.size(), suffix) == 0;
    };
    if (has_suffix(backend::kCoreFamilySuffix)) {
        return backend::kCoreBackendId;
    }
    if (has_suffix(backend::kOgaHipepFamilySuffix)) {
        return backend::kOgaHipepBackendId;
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
            detail::validate_catalog_entry(entry, tag);
            if (entry.contains("preferred_source")) {
                const auto& preferred = entry["preferred_source"];
                if (!preferred.is_object() ||
                    preferred.contains("preferred_source")) {
                    throw std::runtime_error(
                        "Model extension catalog entry '" + tag +
                        "' preferred_source must be an object without a "
                        "preferred_source of its own");
                }
                detail::validate_catalog_entry(
                    detail::with_preferred_source(entry),
                    tag + " preferred_source");
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
            for (const auto& source : catalog_entry_sources(entry)) {
                const std::string family =
                    source["details"]["family"].get<std::string>();
                if (std::find(frontends.begin(), frontends.end(), family) ==
                    frontends.end()) {
                    throw std::runtime_error(
                        "Model extension catalog entry '" + tag +
                        "' names unregistered frontend family '" +
                        family + "'");
                }
                const std::string selected_backend =
                    source.at(backend::kBackendKey).get<std::string>();
                const std::string provider =
                    source.at(backend::kBackendProviderKey).get<std::string>();
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
}

/// \brief refuse overlay entries that download into a folder another catalog
///        already uses
/// \param catalog the catalog the overlay merges into (built-in entries and
///        any overlay merged before this one)
/// \note flm remove deletes an entry's whole folder, so sharing one with
///       another catalog's model would delete that model too. Entries of the
///       same overlay may share a folder, as built-in entries do. Folder
///       names are compared case-insensitively, as Windows does.
inline void validate_model_catalog_overlay_folders(
    const ModelCatalogOverlay& overlay, const nlohmann::json& catalog) {
    const auto taken = detail::model_catalog_folders(catalog);
    const auto check = [&](const nlohmann::json& entry,
                           const std::string& tag) {
        if (!entry.contains("name")) return;
        const std::string name = entry["name"].get<std::string>();
        if (taken.count(detail::ascii_lower(name))) {
            throw std::runtime_error(
                "Model extension catalog entry '" + tag +
                "' uses model folder '" + name +
                "', which another catalog entry already uses");
        }
    };
    for (const auto& [model_type, sizes] :
         overlay.document.at("models").items()) {
        for (const auto& [size, entry] : sizes.items()) {
            const std::string tag = model_type + ":" + size;
            check(entry, tag);
            if (entry.contains("preferred_source")) {
                check(entry["preferred_source"], tag);
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
