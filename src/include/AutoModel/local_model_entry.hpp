/// \file local_model_entry.hpp
/// \brief A catalog entry for a model that already sits in a folder the user names
/// \note `flm serve --model-dir <folder> --family <family>` has no catalog entry
///       to look up: the folder is the model. This header reads the folder and
///       writes the entry the catalog would have held, so everything after it
///       (support filter, downloader status, loader) sees an ordinary entry.
/// \note What a folder holds depends on the package format, and each format is
///       described by its own header (local_model_oga.hpp, local_model_npu.hpp),
///       which registers a LocalPackageFormat here. A format is recognised by
///       its marker file; this header does the rest the same way for every
///       format: the file list check, the size, the tag and the chat template.
/// \note Header-only and free of NPU runtime headers, so it is unit tested the
///       same way model_list is.
#pragma once
#include "AutoModel/backend_contract.hpp"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace flm::catalog {

/// \brief the size part of a local model's tag; there is only ever one
inline constexpr const char* kLocalModelSize = "local";

/// \brief the largest context a local entry offers when --ctx-len is absent
/// \note The shipped entries use the same default. The package's own limit
///       (262144 for the Qwen3.8 27B) is far above what the KV cache of the
///       machine holds, so --ctx-len is how a user asks for more.
inline constexpr int kLocalModelDefaultContext = 32768;

/// \brief the prefill chunk the shipped entries use
inline constexpr int kLocalModelPrefillChunk = 4096;

/// \brief what a model folder becomes in the catalog
struct LocalModelEntry {
    std::string model_type;   ///< the part of the tag before ':'
    std::string size;         ///< the part of the tag after ':'
    std::string tag;          ///< model_type + ":" + size
    std::string backend;      ///< the backend the package runs on
    nlohmann::json entry;     ///< the catalog entry
};

/// \brief what a package format reads out of a folder
struct LocalPackageFacts {
    std::set<std::string> files;                            ///< the files the loader reads, relative to the folder
    int package_limit = kLocalModelDefaultContext;          ///< the package's own context limit
    bool vision = false;                                    ///< whether the package has a vision tower
    nlohmann::json entry_extra = nlohmann::json::object();  ///< keys the format adds to the entry
};

/// \brief one package format a model folder can hold
struct LocalPackageFormat {
    std::string name;        ///< how messages call the package, e.g. "OGA package"
    std::string marker;      ///< the file whose presence marks the format
    std::string backend;     ///< the backend that runs it
    std::string tag_suffix;  ///< appended to the folder's name to make the tag's model type
    /// reads the folder; throws std::runtime_error naming what is wrong with it
    /// \param folder the absolute model folder
    /// \param platform the NPU platform id this process runs on, "" if not known
    std::function<LocalPackageFacts(const std::filesystem::path& folder, const std::string& platform)> read;
};

/// \brief the formats this build reads, in the order their headers were loaded
inline std::vector<LocalPackageFormat>& local_package_formats() {
    static std::vector<LocalPackageFormat> formats;
    return formats;
}

/// \brief make a format known to build_local_model_entry
/// \return true, so a format header can register itself from an inline variable
inline bool register_local_package_format(LocalPackageFormat format) {
    local_package_formats().push_back(std::move(format));
    return true;
}

namespace local_detail {

inline std::string read_text(const std::filesystem::path& file) {
    std::ifstream stream(file, std::ios::binary);
    std::ostringstream text;
    text << stream.rdbuf();
    return text.str();
}

inline bool is_file(const std::filesystem::path& file) {
    std::error_code error;
    return std::filesystem::is_regular_file(file, error);
}

inline std::string lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

/// \brief a folder name as the family part of a tag
/// \note Tags are "family:size" and a "/" prefix is cut off, so the name keeps
///       only lower-case letters, digits, '.', '_' and '-'.
inline std::string tag_name_from(const std::string& folder_name) {
    std::string out;
    bool last_dash = false;
    for (unsigned char c : lower(folder_name)) {
        const bool keep = std::isalnum(c) || c == '.' || c == '_';
        if (keep) {
            out.push_back(static_cast<char>(c));
            last_dash = false;
        } else if (!last_dash && !out.empty()) {
            out.push_back('-');
            last_dash = true;
        }
    }
    while (!out.empty() && out.back() == '-') out.pop_back();
    return out.empty() ? std::string("model") : out;
}

inline std::string join(const std::vector<std::string>& items) {
    std::string joined;
    for (const auto& item : items) {
        if (!joined.empty()) joined += ", ";
        joined += item;
    }
    return joined;
}

inline nlohmann::json parse_file(const std::filesystem::path& file, const char* what) {
    try {
        return nlohmann::json::parse(read_text(file));
    } catch (const std::exception& problem) {
        throw std::runtime_error(std::string("cannot read ") + what + " " + file.string() + ": " +
                                 problem.what());
    }
}

}  // namespace local_detail

/// \brief read a model folder and describe it as a catalog entry
/// \param folder the model folder
/// \param family details.family: which frontend runs the package
/// \param platform the NPU platform id this process runs on (e.g. "aie2p");
///        a format that is bound to a generation needs it, the others ignore it
/// \param think_override true or false sets the entry's thinking flag; nullopt
///        reads it from the chat template
/// \param tools_override true or false sets the entry's tools flag; nullopt
///        reads it from the chat template
/// \return the tag parts, the backend and the entry; the entry's "name" is the
///         folder's absolute path, so the model path resolves to the folder itself
/// \throws std::runtime_error naming what is wrong with the folder
/// \note The entry says "local": true, which the downloader reads as "never
///       download, verify, or delete anything here". It has no file_sources,
///       so no file is hashed, and flm_min_version is 0.0.0, so a config.json
///       without flm_version counts as current.
inline LocalModelEntry build_local_model_entry(const std::filesystem::path& folder,
                                               const std::string& family,
                                               const std::string& platform,
                                               std::optional<bool> think_override = std::nullopt,
                                               std::optional<bool> tools_override = std::nullopt) {
    namespace fs = std::filesystem;
    using local_detail::is_file;

    if (family.empty()) throw std::runtime_error("--family is required with --model-dir");

    std::error_code error;
    fs::path absolute = fs::absolute(folder, error).lexically_normal();
    if (error || !fs::is_directory(absolute, error)) {
        throw std::runtime_error("model folder not found: " + folder.string());
    }

    // A trailing separator leaves filename() empty; the folder is the part before it.
    if (absolute.filename().empty()) absolute = absolute.parent_path();
    // The folder name becomes the tag.
    const std::string stem = local_detail::tag_name_from(absolute.filename().string());

    // The marker files say which format the folder holds.
    std::vector<const LocalPackageFormat*> found;
    std::vector<std::string> known;
    for (const auto& format : local_package_formats()) {
        known.push_back(format.marker + " (" + format.name + ")");
        if (is_file(absolute / format.marker)) found.push_back(&format);
    }
    std::sort(known.begin(), known.end());
    if (found.size() > 1) {
        std::vector<std::string> markers;
        for (const auto* format : found) markers.push_back(format->marker);
        std::sort(markers.begin(), markers.end());
        throw std::runtime_error("model folder " + absolute.string() + " holds more than one package (" +
                                 local_detail::join(markers) + "); keep one of them in the folder");
    }
    if (found.empty()) {
        throw std::runtime_error("not a model package: " + absolute.string() + " has none of: " +
                                 local_detail::join(known));
    }
    const LocalPackageFormat& format = *found.front();

    LocalPackageFacts facts = format.read(absolute, platform);

    std::vector<std::string> missing;
    std::uint64_t total = 0;
    for (const auto& name : facts.files) {
        std::error_code size_error;
        const auto bytes = fs::file_size(absolute / name, size_error);
        if (!is_file(absolute / name) || size_error) {
            missing.push_back(name);
        } else {
            total += bytes;
        }
    }
    if (!missing.empty()) {
        throw std::runtime_error("model folder " + absolute.string() + " is missing: " +
                                 local_detail::join(missing));
    }

    const int default_context = std::max(1, std::min(facts.package_limit, kLocalModelDefaultContext));

    // The chat template says whether the model reasons and takes tools; the
    // catalog entries say the same in two fields. An absent template says neither.
    // The template is a file of its own or a string in tokenizer_config.json.
    std::string chat_template;
    if (is_file(absolute / "chat_template.jinja")) {
        chat_template = local_detail::read_text(absolute / "chat_template.jinja");
    } else if (is_file(absolute / "tokenizer_config.json")) {
        try {
            const auto tokenizer_config = nlohmann::json::parse(local_detail::read_text(absolute / "tokenizer_config.json"));
            if (tokenizer_config.contains("chat_template") && tokenizer_config["chat_template"].is_string()) {
                chat_template = tokenizer_config["chat_template"].get<std::string>();
            }
        } catch (const std::exception&) {
            // an unreadable tokenizer_config.json is the tokenizer's to report; here it says nothing
        }
    }
    // A user override (--think / --tools) wins over what the template suggests; the
    // template is only a guess, and a comment that mentions "tools" would fool it.
    const bool thinks = think_override.value_or(
        chat_template.find("enable_thinking") != std::string::npos ||
        chat_template.find("<think>") != std::string::npos);
    const bool tools = tools_override.value_or(chat_template.find("tools") != std::string::npos);

    LocalModelEntry out;
    out.backend = format.backend;
    out.model_type = stem + format.tag_suffix;
    out.size = kLocalModelSize;
    out.tag = out.model_type + ":" + out.size;
    out.entry = {
        {"name", absolute.string()},
        {"url", "local"},
        {"size", total},
        {"default_context_length", default_context},
        {"max_prefill_len", kLocalModelPrefillChunk},
        {"details", {{"family", family},
                     {"think", thinks},
                     {"parameter_size", "local"},
                     {"quantization_level", "unknown"}}},
        {"flm_min_version", "0.0.0"},
        {"vlm", facts.vision},
        {"tools", tools},
        {"files", std::vector<std::string>(facts.files.begin(), facts.files.end())},
        {"local", true},
        {flm::backend::kBackendKey, out.backend},
    };
    for (const auto& [key, value] : facts.entry_extra.items()) out.entry[key] = value;
    return out;
}

}  // namespace flm::catalog
