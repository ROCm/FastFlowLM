/// \file local_model_entry.hpp
/// \brief A catalog entry for a model that already sits in a folder the user names
/// \note `flm serve --model-dir <folder> --family <family>` has no catalog entry
///       to look up: the folder is the model. This header reads the folder and
///       writes the entry the catalog would have held, so everything after it
///       (support filter, downloader status, loader) sees an ordinary entry.
/// \note Two package formats are recognised by their files: an OGA package
///       (genai_config.json, backend oga-hipep) and an NPU package (model.q4nx,
///       backend flm). An NPU package is built for one NPU generation, which
///       the folder does not say, so its entry names the platform the caller
///       runs on.
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
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
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

/// \brief the file that marks an NPU package
inline constexpr const char* kNpuWeightsFile = "model.q4nx";

/// \brief the file that marks an OGA package
inline constexpr const char* kOgaConfigFile = "genai_config.json";

/// \brief what a model folder becomes in the catalog
struct LocalModelEntry {
    std::string model_type;   ///< the part of the tag before ':'
    std::string size;         ///< the part of the tag after ':'
    std::string tag;          ///< model_type + ":" + size
    std::string backend;      ///< the backend the package runs on: "oga-hipep" or "flm"
    nlohmann::json entry;     ///< the catalog entry
};

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

/// \brief every string a genai_config.json names under "filename" or "config_filename"
/// \note These are the files the package loads: the decoder, the embedding and
///       vision graphs, and the processor config. Anything the config does not
///       name is not checked here.
inline void collect_named_files(const nlohmann::json& node, std::set<std::string>& out) {
    if (node.is_object()) {
        for (const auto& [key, value] : node.items()) {
            if ((key == "filename" || key == "config_filename") && value.is_string()) {
                out.insert(value.get<std::string>());
            } else {
                collect_named_files(value, out);
            }
        }
    } else if (node.is_array()) {
        for (const auto& value : node) collect_named_files(value, out);
    }
}

/// \brief whether a name is one relative path inside the model folder
inline bool stays_inside(const std::string& name) {
    if (name.empty()) return false;
    const std::filesystem::path relative = std::filesystem::path(name).lexically_normal();
    return !relative.has_root_path() && !relative.empty() && *relative.begin() != "..";
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
/// \param folder the model folder: an OGA package or an NPU package
/// \param family details.family: which frontend runs the package
/// \param platform the NPU platform id this process runs on (e.g. "aie2p");
///        required for an NPU package, unused for an OGA package
/// \return the tag parts, the backend and the entry; the entry's "name" is the
///         folder's absolute path, so the model path resolves to the folder itself
/// \throws std::runtime_error naming what is wrong with the folder
/// \note The entry says "local": true, which the downloader reads as "never
///       download, verify, or delete anything here". It has no file_sources,
///       so no file is hashed, and flm_min_version is 0.0.0, so a config.json
///       without flm_version counts as current.
inline LocalModelEntry build_local_model_entry(const std::filesystem::path& folder,
                                               const std::string& family,
                                               const std::string& platform) {
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

    const bool oga = is_file(absolute / kOgaConfigFile);
    const bool npu = is_file(absolute / kNpuWeightsFile);
    if (oga && npu) {
        throw std::runtime_error("model folder " + absolute.string() + " holds both an OGA package (" +
                                 kOgaConfigFile + ") and an NPU package (" + kNpuWeightsFile +
                                 "); keep one of them in the folder");
    }
    if (!oga && !npu) {
        throw std::runtime_error("not a model package: " + absolute.string() + " has neither " +
                                 kOgaConfigFile + " (OGA) nor " + kNpuWeightsFile + " (NPU)");
    }

    std::set<std::string> files;
    int package_limit = kLocalModelDefaultContext;
    bool vision = false;

    if (oga) {
        const nlohmann::json genai = local_detail::parse_file(absolute / kOgaConfigFile, "genai_config.json");

        // The files the package loads, plus the .data file next to each graph.
        files = {"genai_config.json", "config.json", "tokenizer.json"};
        std::set<std::string> named_files;
        local_detail::collect_named_files(genai, named_files);
        for (const auto& name : named_files) {
            if (!local_detail::stays_inside(name)) {
                throw std::runtime_error("genai_config.json names a file outside the model folder: " + name);
            }
            files.insert(name);
            if (name.size() > 5 && name.compare(name.size() - 5, 5, ".onnx") == 0 &&
                is_file(absolute / (name + ".data"))) {
                files.insert(name + ".data");
            }
        }
        for (const char* optional : {"tokenizer_config.json", "chat_template.jinja", "processor_config.json"}) {
            if (is_file(absolute / optional)) files.insert(optional);
        }

        const auto model_node = genai.contains("model") && genai["model"].is_object()
            ? genai["model"] : nlohmann::json::object();
        vision = model_node.contains("vision");
        if (genai.contains("search") && genai["search"].is_object() &&
            genai["search"].contains("max_length") && genai["search"]["max_length"].is_number_integer()) {
            package_limit = genai["search"]["max_length"].get<int>();
        } else if (model_node.contains("context_length") && model_node["context_length"].is_number_integer()) {
            package_limit = model_node["context_length"].get<int>();
        }
    } else {
        // The NPU package: the weights, the network config and the tokenizer are
        // what the loader reads; the vision tower is there only for a VLM.
        files = {"config.json", kNpuWeightsFile, "tokenizer.json"};
        for (const char* optional : {"tokenizer_config.json", "chat_template.jinja", "vision_weight.q4nx"}) {
            if (is_file(absolute / optional)) files.insert(optional);
        }
        vision = is_file(absolute / "vision_weight.q4nx");
        if (platform.empty()) {
            throw std::runtime_error("an NPU package needs the NPU platform it runs on");
        }
        if (is_file(absolute / "config.json")) {
            const auto config = local_detail::parse_file(absolute / "config.json", "config.json");
            if (config.contains("max_position_embeddings") &&
                config["max_position_embeddings"].is_number_integer()) {
                package_limit = config["max_position_embeddings"].get<int>();
            }
        }
    }

    std::vector<std::string> missing;
    std::uint64_t total = 0;
    for (const auto& name : files) {
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

    const int default_context = std::max(1, std::min(package_limit, kLocalModelDefaultContext));

    // The chat template says whether the model reasons and takes tools; the
    // catalog entries say the same in two fields. An absent template says neither.
    // The template is a file of its own or, as in the NPU packages, a string in
    // tokenizer_config.json.
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
    const bool thinks = chat_template.find("enable_thinking") != std::string::npos ||
                        chat_template.find("<think>") != std::string::npos;
    const bool tools = chat_template.find("tools") != std::string::npos;

    LocalModelEntry out;
    out.backend = oga ? flm::backend::kOgaHipepBackendId : flm::backend::kFlmBackendId;
    out.model_type = oga ? stem + flm::backend::kOgaHipepFamilySuffix : stem;
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
        {"vlm", vision},
        {"tools", tools},
        {"files", std::vector<std::string>(files.begin(), files.end())},
        {"local", true},
        {flm::backend::kBackendKey, out.backend},
    };
    if (npu) {
        // The kernels of an NPU package belong to one generation; an entry on
        // another one is refused by model_list, as a shipped entry would be.
        out.entry["supported_platforms"] = nlohmann::json::array({platform});
    }
    return out;
}

}  // namespace flm::catalog
